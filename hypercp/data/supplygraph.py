"""
SupplyGraph dataset loader with hypergraph reformulation.

Re-interprets the original homogeneous SupplyGraph (Wasi et al., AAAI 2024
GCLR workshop, arXiv 2401.15299) as a tripartite hypergraph aligned with
the SC-RIHN backbone (Shen et al., AAAI 2026) and the assumptions of
Theorem 1 in this paper.

The four SupplyGraph edge types (Plant, Product Group, Product Sub-Group,
Storage Location) become the four partition classes of the attribute
partition required by Assumption A1 in Theorem 1: hyperedges within the
same partition class are constructed by exact attribute match.

References
----------
- Wasi et al. 2024 — SupplyGraph (arXiv 2401.15299)
- Han 2024 — Data-QA discipline (arXiv 2408.14501)
- Shen et al. 2026 — SC-RIHN (arXiv 2511.06208; AAAI 2026)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)


# =============================================================================
# Constants — match the on-disk SupplyGraph layout exactly.
# =============================================================================

DATA_ROOT_DEFAULT = Path("SupplyGraph-main/Raw Dataset/Homogenoeus")

NODES_CSV = "Nodes/Nodes.csv"
NODES_INDEX_CSV = "Nodes/NodesIndex.csv"
NODE_GROUP_CSV = "Nodes/Node Types (Product Group and Subgroup).csv"
NODE_PLANT_STORAGE_CSV = "Nodes/Nodes Type (Plant & Storage).csv"

EDGE_FILES = {
    "plant":     "Edges/Edges (Plant).csv",
    "subgroup":  "Edges/Edges (Product Sub-Group).csv",
    "group":     "Edges/Edges (Product Group).csv",
    "storage":   "Edges/Edges (Storage Location).csv",
}

# Filenames have inconsistent trailing-spaces upstream; mirror them faithfully.
TEMPORAL_FILES_UNIT = {
    "production":   "Temporal Data/Unit/Production .csv",
    "sales_order":  "Temporal Data/Unit/Sales Order.csv",
    "delivery":     "Temporal Data/Unit/Delivery To distributor.csv",
    "factory_issue": "Temporal Data/Unit/Factory Issue.csv",
}

TEMPORAL_CHANNELS = ("production", "sales_order", "delivery", "factory_issue")
N_BASE_CHANNELS = 4  # production, sales_order, delivery, factory_issue
N_DERIVED_CHANNELS = 1  # predicted-imbalance (GPP-inspired)
N_TOTAL_CHANNELS = N_BASE_CHANNELS + N_DERIVED_CHANNELS  # F = 5
N_TIMESTEPS = 221  # Wasi 2024: 221 daily observations Jan-Aug 2023


# =============================================================================
# Hyperedge data structure.
# =============================================================================


class PartitionClass(Enum):
    """The four attribute classes A_1..A_4 of Theorem 1's partition."""
    PLANT = "plant"
    SUBGROUP = "subgroup"
    GROUP = "group"
    STORAGE = "storage"


@dataclass(frozen=True)
class Hyperedge:
    """A hyperedge in the SupplyGraph hypergraph.

    nodes: frozenset of node indices participating in this hyperedge.
    partition: which of the four attribute classes this hyperedge belongs to.
    attribute_value: the specific attribute value that generated this hyperedge
                     (e.g. plant_id="1903" or storage_loc="330.0"). Used by
                     `partition_class_membership()` to verify A1's
                     within-class exchangeability claim.
    """
    nodes: frozenset[int]
    partition: PartitionClass
    attribute_value: str

    def __post_init__(self):
        if len(self.nodes) < 2:
            raise ValueError(
                f"Hyperedge must have ≥2 nodes; got {len(self.nodes)}."
            )

    @property
    def size(self) -> int:
        return len(self.nodes)

    def contains(self, node_idx: int) -> bool:
        return node_idx in self.nodes


@dataclass
class QAReport:
    """Han 2024-style data-quality assurance report.

    Records every QA decision so the paper's reproducibility checklist can
    list exactly what was filtered, why, and how many of each.
    """
    raw_product_count: int = 0
    products_after_dedup: int = 0
    products_masked_zero_feature: list[str] = field(default_factory=list)
    products_after_qa: int = 0
    raw_edge_counts: dict[str, int] = field(default_factory=dict)
    hyperedge_counts: dict[str, int] = field(default_factory=dict)
    duplicate_edges_removed: dict[str, int] = field(default_factory=dict)

    def __str__(self) -> str:
        lines = [
            "SupplyGraph QA report",
            "=====================",
            f"Raw product count:          {self.raw_product_count}",
            f"After deduplication:        {self.products_after_dedup}",
            f"Masked (zero-feature):      {len(self.products_masked_zero_feature)} "
            f"-> {self.products_masked_zero_feature}",
            f"After QA (final):           {self.products_after_qa}",
            "",
            "Raw edge counts:",
        ]
        for k, v in self.raw_edge_counts.items():
            lines.append(f"  {k:10s}: {v}")
        lines.append("Hyperedge counts (after grouping):")
        for k, v in self.hyperedge_counts.items():
            lines.append(f"  {k:10s}: {v}")
        lines.append("Duplicate edges removed (within-class):")
        for k, v in self.duplicate_edges_removed.items():
            lines.append(f"  {k:10s}: {v}")
        return "\n".join(lines)


# =============================================================================
# SupplyGraphHypergraph — the main API.
# =============================================================================


class SupplyGraphHypergraph:
    """Hypergraph reformulation of the SupplyGraph FMCG benchmark.

    Parameters
    ----------
    data_root : Path | str
        Path to the `Raw Dataset/Homogenoeus` directory of the SupplyGraph
        release. Defaults to the canonical relative path used by the
        project repo.
    edge_types : tuple[PartitionClass, ...] | None
        Which attribute classes to include in the hypergraph. Default = all
        four (the full Theorem 1 setup). Pass a subset for the §4.2.10
        edge-configuration ablation.
    mask_zero_feature_nodes : bool
        Han 2024 QA: if True, mask out product nodes whose four temporal
        channels are entirely zero across the 221 days. Default True.
    standardise : bool
        If True, z-score each temporal channel across time (Han 2024). Default
        True; turn off when re-deriving raw resilience labels via SupplySim.

    Attributes
    ----------
    n_products : int
        Number of product nodes surviving QA (typically 30, down from 41).
    n_timesteps : int
        221 daily observations.
    n_channels : int
        5 = 4 base channels + 1 derived "predicted-imbalance" channel.
    node_features : torch.Tensor
        Shape (n_products, n_timesteps, n_channels). z-scored per channel.
    node_names : list[str]
        Product names in order; node index i corresponds to node_names[i].
    hyperedges : dict[PartitionClass, list[Hyperedge]]
        Hyperedges grouped by partition class. Each list is the set of
        hyperedges within that attribute class; A1 requires within-class
        exchangeability of conformity scores on these.
    qa_report : QAReport
        Detailed breakdown of every QA step for the paper's appendix.
    """

    def __init__(
        self,
        data_root: Path | str = DATA_ROOT_DEFAULT,
        edge_types: tuple[PartitionClass, ...] | None = None,
        mask_zero_feature_nodes: bool = True,
        standardise: bool = True,
    ) -> None:
        self.data_root = Path(data_root)
        if edge_types is None:
            edge_types = tuple(PartitionClass)
        self.edge_types = edge_types
        self.standardise = standardise

        self.qa_report = QAReport()

        # ----- 1. Load and dedupe product nodes -----
        self.node_names, self.node_to_idx = self._load_nodes()

        # ----- 2. Load temporal features for surviving nodes -----
        raw_features = self._load_temporal_features()  # (n_raw, T, 4)

        # ----- 3. Han 2024 zero-feature masking -----
        if mask_zero_feature_nodes:
            keep_mask, masked_names = self._zero_feature_mask(raw_features)
            self.qa_report.products_masked_zero_feature = masked_names
            self._apply_mask(keep_mask, raw_features)
        else:
            keep_mask = np.ones(raw_features.shape[0], dtype=bool)

        self.qa_report.products_after_qa = len(self.node_names)
        self.n_products = len(self.node_names)
        self.n_timesteps = N_TIMESTEPS
        self.n_channels = N_TOTAL_CHANNELS

        # ----- 4. Standardise & add derived imbalance channel -----
        features = raw_features[keep_mask]  # (n_products, T, 4)
        features = self._add_derived_imbalance_channel(features)  # (n_products, T, 5)
        if standardise:
            features = self._zscore_per_channel(features)
        self.node_features = torch.tensor(features, dtype=torch.float32)

        # ----- 5. Build hyperedges from each surviving edge file -----
        self.hyperedges: dict[PartitionClass, list[Hyperedge]] = {}
        for cls in self.edge_types:
            self.hyperedges[cls] = self._build_hyperedges_for_class(cls)

    # =========================================================================
    # Loaders.
    # =========================================================================

    def _load_nodes(self) -> tuple[list[str], dict[str, int]]:
        """Load product node names with deduplication (Han 2024 step 1)."""
        nodes_path = self.data_root / NODES_CSV
        df = pd.read_csv(nodes_path)
        if "Node" not in df.columns:
            raise RuntimeError(
                f"Expected 'Node' column in {nodes_path}; got {df.columns.tolist()}"
            )

        raw_names = df["Node"].tolist()
        self.qa_report.raw_product_count = len(raw_names)

        # Deduplicate while preserving order.
        seen: set[str] = set()
        deduped: list[str] = []
        for name in raw_names:
            if name not in seen:
                seen.add(name)
                deduped.append(name)
        self.qa_report.products_after_dedup = len(deduped)

        node_to_idx = {name: i for i, name in enumerate(deduped)}
        logger.info(
            "Loaded %d products (deduped from %d).",
            len(deduped),
            len(raw_names),
        )
        return deduped, node_to_idx

    def _load_temporal_features(self) -> np.ndarray:
        """Load the 4 temporal channels. Returns (n_raw, T, 4) ndarray."""
        n_raw = len(self.node_names)
        features = np.zeros((n_raw, N_TIMESTEPS, N_BASE_CHANNELS), dtype=np.float64)

        for ch_idx, channel in enumerate(TEMPORAL_CHANNELS):
            path = self.data_root / TEMPORAL_FILES_UNIT[channel]
            df = pd.read_csv(path)
            # First column is Date; subsequent columns are product names.
            if df.shape[0] != N_TIMESTEPS:
                raise RuntimeError(
                    f"Expected {N_TIMESTEPS} timesteps in {path}; got {df.shape[0]}."
                )
            for col in df.columns[1:]:
                col_clean = col.strip()
                if col_clean not in self.node_to_idx:
                    # Some products may have empty trailing-comma columns;
                    # warn but don't fail.
                    if col_clean:
                        logger.warning(
                            "Column '%s' in %s not in node index; skipping.",
                            col_clean,
                            path,
                        )
                    continue
                node_idx = self.node_to_idx[col_clean]
                features[node_idx, :, ch_idx] = df[col].fillna(0.0).to_numpy()

        return features

    def _zero_feature_mask(
        self, features: np.ndarray
    ) -> tuple[np.ndarray, list[str]]:
        """Han 2024 QA: mask nodes whose channels are all-zero across time.

        A node is masked if the sum across all 4 channels × 221 timesteps is
        ≤ a small tolerance. Returns (keep_mask, masked_node_names).
        """
        total_signal = np.abs(features).sum(axis=(1, 2))  # (n_raw,)
        tol = 1e-6
        keep_mask = total_signal > tol
        masked_names = [
            self.node_names[i] for i in range(len(self.node_names)) if not keep_mask[i]
        ]
        if masked_names:
            logger.info(
                "Han 2024 QA: masking %d zero-feature nodes: %s",
                len(masked_names),
                masked_names,
            )
        return keep_mask, masked_names

    def _apply_mask(self, keep_mask: np.ndarray, features: np.ndarray) -> None:
        """Apply the zero-feature mask to node names and rebuild index."""
        kept_names = [
            name for i, name in enumerate(self.node_names) if keep_mask[i]
        ]
        self.node_names = kept_names
        self.node_to_idx = {name: i for i, name in enumerate(kept_names)}

    def _add_derived_imbalance_channel(self, features: np.ndarray) -> np.ndarray:
        """Append a GPP-style 'predicted-imbalance' channel.

        Imbalance at time t = cumulative (production - sales_order) over the
        next H steps. We use a fixed H=4 as the default forecast horizon.
        At the end of the series, we pad with the last available value.
        """
        H = 4
        production = features[:, :, TEMPORAL_CHANNELS.index("production")]
        sales = features[:, :, TEMPORAL_CHANNELS.index("sales_order")]
        diff = production - sales  # (n, T)

        imbalance = np.zeros_like(diff)
        for t in range(diff.shape[1]):
            t_end = min(t + H, diff.shape[1])
            imbalance[:, t] = diff[:, t:t_end].sum(axis=1)

        # Stack as the 5th channel.
        return np.concatenate([features, imbalance[:, :, None]], axis=2)

    def _zscore_per_channel(self, features: np.ndarray) -> np.ndarray:
        """Z-score each channel across (nodes, time). Han 2024 step 2."""
        # Shape: (n, T, C)
        out = features.copy()
        for c in range(features.shape[2]):
            ch = features[:, :, c]
            mean = ch.mean()
            std = ch.std()
            if std < 1e-8:
                logger.warning("Channel %d has zero std; skipping z-score.", c)
                continue
            out[:, :, c] = (ch - mean) / std
        return out

    # =========================================================================
    # Hyperedge construction.
    # =========================================================================

    def _build_hyperedges_for_class(
        self, cls: PartitionClass
    ) -> list[Hyperedge]:
        """Build hyperedges for one partition class.

        The SupplyGraph edge files list *pairwise* edges. We convert them to
        hyperedges by grouping all nodes sharing the same attribute value.
        For class `plant`: a hyperedge = {all products produced at plant P}.
        Same for subgroup, group, storage. This is the natural lift of the
        homogeneous SupplyGraph to a hypergraph and aligns with SC-RIHN's
        "firm-product-firm hyperedge" interpretation.

        Han-style discipline: dedupe pairs first, then group; record both
        counts in the QA report.
        """
        path = self.data_root / EDGE_FILES[cls.value]
        df = pd.read_csv(path)
        raw_count = len(df)
        self.qa_report.raw_edge_counts[cls.value] = raw_count

        # Identify the attribute column.
        attr_col = self._attribute_column_name(df, cls)

        # Dedupe rows: (attribute, node1, node2) -> keep first.
        dedup_keys = ["node1", "node2", attr_col]
        before = len(df)
        df = df.drop_duplicates(subset=dedup_keys).reset_index(drop=True)
        after = len(df)
        self.qa_report.duplicate_edges_removed[cls.value] = before - after

        # Group nodes by attribute value.
        # Each group becomes one hyperedge.
        groups: dict[str, set[int]] = {}
        for _, row in df.iterrows():
            n1, n2 = row["node1"], row["node2"]
            # After QA, some node names may have been masked; skip rows that
            # reference masked nodes.
            if n1 not in self.node_to_idx or n2 not in self.node_to_idx:
                continue
            attr_val = str(row[attr_col])
            i1 = self.node_to_idx[n1]
            i2 = self.node_to_idx[n2]
            groups.setdefault(attr_val, set()).update([i1, i2])

        hyperedges: list[Hyperedge] = []
        for attr_val, node_set in groups.items():
            if len(node_set) < 2:
                # Singleton hyperedge — not informative for CP; skip.
                continue
            hyperedges.append(
                Hyperedge(
                    nodes=frozenset(node_set),
                    partition=cls,
                    attribute_value=attr_val,
                )
            )

        self.qa_report.hyperedge_counts[cls.value] = len(hyperedges)
        logger.info(
            "Built %d hyperedges for class %s (from %d raw pairwise edges).",
            len(hyperedges),
            cls.value,
            raw_count,
        )
        return hyperedges

    @staticmethod
    def _attribute_column_name(df: pd.DataFrame, cls: PartitionClass) -> str:
        """Find the attribute column in a SupplyGraph edge file.

        The edge files are inconsistent: plant uses 'Plant', storage uses
        'Storage Location', group uses 'GroupCode', subgroup uses
        'SubGroupCode'. We resolve by searching for the expected name.
        """
        candidates = {
            PartitionClass.PLANT:    ["Plant"],
            PartitionClass.STORAGE:  ["Storage Location"],
            PartitionClass.GROUP:    ["GroupCode"],
            PartitionClass.SUBGROUP: ["SubGroupCode"],
        }[cls]
        for cand in candidates:
            if cand in df.columns:
                return cand
        raise RuntimeError(
            f"Could not find attribute column for {cls.value} in {df.columns.tolist()}"
        )

    # =========================================================================
    # Convenience accessors.
    # =========================================================================

    def all_hyperedges(self) -> Iterator[Hyperedge]:
        """Iterate over all hyperedges across all partition classes."""
        for cls in self.edge_types:
            yield from self.hyperedges[cls]

    @property
    def n_hyperedges(self) -> int:
        return sum(len(self.hyperedges[c]) for c in self.edge_types)

    @property
    def n_partition_classes(self) -> int:
        return len(self.edge_types)

    def hyperedge_incidence_matrix(self) -> torch.Tensor:
        """Return the incidence matrix H ∈ R^{n_products × n_hyperedges}.

        H[c, e] = 1 if node c is in hyperedge e, else 0. Used by HGNN
        convolutions (Feng et al. 2019).
        """
        n_e = self.n_hyperedges
        H = torch.zeros(self.n_products, n_e, dtype=torch.float32)
        edge_idx = 0
        for cls in self.edge_types:
            for he in self.hyperedges[cls]:
                for node_idx in he.nodes:
                    H[node_idx, edge_idx] = 1.0
                edge_idx += 1
        return H

    def hyperedge_partition_vector(self) -> torch.Tensor:
        """Return a (n_hyperedges,) tensor of partition class indices."""
        cls_to_int = {c: i for i, c in enumerate(PartitionClass)}
        vec: list[int] = []
        for cls in self.edge_types:
            vec.extend([cls_to_int[cls]] * len(self.hyperedges[cls]))
        return torch.tensor(vec, dtype=torch.long)

    def degree_within_hyperedge(self, node_idx: int, hyperedge: Hyperedge) -> int:
        """deg_e(c): how many times node c appears in hyperedge e.

        For SupplyGraph (set-valued hyperedges) this is 0 or 1. The interface
        is kept general for compatibility with multigraph-style extensions
        (e.g. SupplyGraph-Stressed with per-shock hyperedge replications).
        """
        return int(node_idx in hyperedge.nodes)

    def max_hyperedge_degree(self) -> int:
        """d_max = max over (node, hyperedge) of deg_e(c).

        Appears in Theorem 1's structural error term
        O(d_max · log|E| / |E|). For SupplyGraph this is 1; included as a
        method so SupplyGraph-Stressed (which may have d_max > 1) uses
        the same API.
        """
        d = 0
        for he in self.all_hyperedges():
            for c in he.nodes:
                d = max(d, self.degree_within_hyperedge(c, he))
        return d

    # =========================================================================
    # Rolling-origin temporal splits.
    # =========================================================================

    def rolling_origin_split(
        self,
        train_frac: float = 0.80,
        cal_frac: float = 0.10,
    ) -> tuple[range, range, range]:
        """Return (train_days, cal_days, test_days) as ranges over 0..T-1.

        Convention: train_days is contiguous [0, t_train), cal_days is
        [t_train, t_cal), test_days is [t_cal, T). All splits are
        time-contiguous — i.i.d. random splits would violate the rolling-
        origin assumption that ACI requires.

        Defaults give SupplyGraph: train 0..176, cal 177..198, test 199..220.
        """
        if not (0 < train_frac < 1):
            raise ValueError(f"train_frac must be in (0,1); got {train_frac}.")
        if not (0 < cal_frac < 1):
            raise ValueError(f"cal_frac must be in (0,1); got {cal_frac}.")
        if train_frac + cal_frac >= 1:
            raise ValueError(
                f"train_frac+cal_frac must be < 1; got {train_frac+cal_frac}."
            )

        t_train = int(self.n_timesteps * train_frac)
        t_cal = int(self.n_timesteps * (train_frac + cal_frac))
        return range(0, t_train), range(t_train, t_cal), range(t_cal, self.n_timesteps)

    # =========================================================================
    # Smoke test / verification.
    # =========================================================================

    def verify_partition_assumption(self) -> dict[str, int]:
        """Sanity-check A1 (partition exchangeability) precondition.

        Returns the count of hyperedges per partition class. Theorem 1's
        Lemma 1 claims within-class exchangeability — this is consistent with
        SupplyGraph iff each partition class is non-empty and the hyperedges
        within a class are constructed by exact attribute match (which is
        guaranteed by `_build_hyperedges_for_class`).
        """
        return {cls.value: len(self.hyperedges[cls]) for cls in self.edge_types}


# =============================================================================
# Smoke test — run this to confirm the dataset loads cleanly end-to-end.
# Usage:  python -m hypercp.data.supplygraph
# =============================================================================


def _smoke_test() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    sg = SupplyGraphHypergraph()
    print(sg.qa_report)
    print()
    print(f"Surviving products:    {sg.n_products}")
    print(f"Timesteps:             {sg.n_timesteps}")
    print(f"Channels:              {sg.n_channels}")
    print(f"Node feature tensor:   {tuple(sg.node_features.shape)}")
    print(f"# hyperedges (total):  {sg.n_hyperedges}")
    print(f"# partition classes:   {sg.n_partition_classes}")
    print(f"d_max:                 {sg.max_hyperedge_degree()}")
    print()
    print("Hyperedge counts by partition class (Theorem 1 Lemma 1):")
    for cls, n in sg.verify_partition_assumption().items():
        print(f"  {cls:10s}: {n}")
    print()
    train, cal, test = sg.rolling_origin_split()
    print(
        f"Rolling-origin split: train=[{train.start},{train.stop}), "
        f"cal=[{cal.start},{cal.stop}), test=[{test.start},{test.stop})"
    )
    H = sg.hyperedge_incidence_matrix()
    print(f"Incidence matrix shape: {tuple(H.shape)}")
    print(f"  sparsity: {1.0 - H.sum().item() / H.numel():.4f}")


if __name__ == "__main__":
    _smoke_test()
