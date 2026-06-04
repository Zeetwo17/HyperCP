"""
M5 Forecasting Accuracy dataset loader with hyperedge reformulation.

Maps the M5 Walmart product hierarchy (item, dept, cat, store, state)
to a tripartite hypergraph with 4 partition classes:

    Class 0 -- (cat, state)   : 3 * 3 = 9  hyperedges, large per-edge
    Class 1 -- (dept, state)  : 7 * 3 = 21 hyperedges
    Class 2 -- (cat, store)   : 3 * 10 = 30 hyperedges
    Class 3 -- (dept, store)  : 7 * 10 = 70 hyperedges

Total ~130 hyperedges over 30490 SKUs. The partition is designed so each
class has |E_k| large enough that Theorem 1's structural term
O(log|E_k| / |E_k|) is small (<0.05 for the larger classes).

Reference
---------
M5 Forecasting Accuracy (Kaggle): https://kaggle.com/competitions/m5-forecasting-accuracy
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from .supplygraph import PartitionClass

logger = logging.getLogger(__name__)


DATA_ROOT_DEFAULT = Path("m5-forecasting")

# M5 has its own partition classes -- distinct from SupplyGraph's.
# We reuse `PartitionClass` for the API but the enum members are mapped via
# string keys at the level of the dataset.


@dataclass
class M5QAReport:
    """Han-style QA report for M5 loading."""
    raw_n_skus: int = 0
    n_skus_after_qa: int = 0
    n_zero_sales_skus_masked: int = 0
    n_timesteps: int = 0
    hyperedge_counts: dict[str, int] = field(default_factory=dict)

    def __str__(self) -> str:
        lines = [
            "M5 QA report",
            "============",
            f"Raw SKU count:       {self.raw_n_skus}",
            f"Zero-sales masked:   {self.n_zero_sales_skus_masked}",
            f"After QA (final):    {self.n_skus_after_qa}",
            f"Timesteps:           {self.n_timesteps}",
            "Hyperedge counts:",
        ]
        for k, v in self.hyperedge_counts.items():
            lines.append(f"  {k:18s}: {v}")
        return "\n".join(lines)


class M5Hypergraph:
    """M5 dataset reformulated as a hypergraph with 4-class partition.

    Parameters
    ----------
    data_root : Path | str
        Path to the M5 root directory (containing sales_train_evaluation.csv etc.).
    subsample_n : int | None
        If set, take only this many SKUs (random subset). Useful for smoke
        testing on CPU. Default None = full dataset.
    subsample_days : int | None
        If set, take only the last this-many days. Default None = all 1941 days.
    min_total_sales : float
        Han-style QA: drop SKUs whose total sales over the window are below
        this threshold. Default 1.0 (drops zero-sales SKUs only).
    standardise : bool
        z-score sales per SKU across time. Default True.
    seed : int
        RNG seed for subsampling.

    Attributes
    ----------
    n_skus : int
    n_timesteps : int
    n_channels : int  -- 1 for now (sales). Multi-channel extension possible.
    node_features : torch.Tensor   (n_skus, T, 1)
    hyperedges : dict[str, list[Hyperedge]]   keyed by partition-class name
    incidence : torch.Tensor       (n_skus, n_hyperedges_total)
    partition_vector : torch.Tensor  (n_hyperedges_total,) int class index
    qa_report : M5QAReport
    """

    PARTITION_KEYS = ("cat_state", "dept_state", "cat_store", "dept_store")

    def __init__(
        self,
        data_root: Path | str = DATA_ROOT_DEFAULT,
        subsample_n: Optional[int] = None,
        subsample_days: Optional[int] = None,
        min_total_sales: float = 1.0,
        standardise: bool = True,
        seed: int = 42,
    ) -> None:
        self.data_root = Path(data_root)
        self.seed = seed
        self.qa_report = M5QAReport()
        rng = np.random.default_rng(seed)

        # ----- 1. Load main sales table -----
        sales_path = self.data_root / "sales_train_evaluation.csv"
        if not sales_path.exists():
            raise FileNotFoundError(
                f"M5 sales file not found: {sales_path}. "
                f"Did you place the Kaggle M5 files under {self.data_root}?"
            )
        logger.info("Loading M5 sales from %s...", sales_path)
        df = pd.read_csv(sales_path)
        self.qa_report.raw_n_skus = len(df)
        logger.info("Loaded %d raw SKUs.", len(df))

        # ----- 2. Subsample SKUs (optional) -----
        if subsample_n is not None and subsample_n < len(df):
            keep_idx = rng.choice(len(df), size=subsample_n, replace=False)
            df = df.iloc[keep_idx].reset_index(drop=True)
            logger.info("Subsampled to %d SKUs (seed=%d).", subsample_n, seed)

        # ----- 3. Extract metadata columns -----
        meta_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
        meta = df[meta_cols].copy()
        sale_cols = [c for c in df.columns if c.startswith("d_")]

        # ----- 4. Subsample days (optional) -----
        if subsample_days is not None and subsample_days < len(sale_cols):
            # Take the LAST `subsample_days` (more recent).
            sale_cols = sale_cols[-subsample_days:]
            logger.info("Subsampled to last %d days.", subsample_days)

        sales = df[sale_cols].to_numpy(dtype=np.float32)  # (n_skus, T)
        self.qa_report.n_timesteps = sales.shape[1]

        # ----- 5. Han-style QA: drop zero/low-sales SKUs -----
        totals = sales.sum(axis=1)
        keep_mask = totals >= min_total_sales
        n_dropped = int((~keep_mask).sum())
        self.qa_report.n_zero_sales_skus_masked = n_dropped
        if n_dropped > 0:
            logger.info("Dropped %d zero/low-sales SKUs (Han QA).", n_dropped)
        sales = sales[keep_mask]
        meta = meta[keep_mask].reset_index(drop=True)
        self.qa_report.n_skus_after_qa = len(meta)

        # ----- 6. Standardise per-SKU across time -----
        if standardise:
            mean = sales.mean(axis=1, keepdims=True)
            std = sales.std(axis=1, keepdims=True)
            std = np.where(std < 1e-8, 1.0, std)
            sales = (sales - mean) / std

        self.n_skus = sales.shape[0]
        self.n_timesteps = sales.shape[1]
        self.n_channels = 1

        # (n_skus, T, 1) torch tensor for the encoder.
        self.node_features = torch.from_numpy(sales).unsqueeze(-1)

        # ----- 7. Build the 4-class hyperedge partition -----
        # Each partition class is a *composite-key* grouping of SKUs.
        self.meta = meta
        self.hyperedges: dict[str, list[set[int]]] = {}
        self._partition_class_attr_values: dict[str, list[tuple]] = {}

        compound_keys = {
            "cat_state":   ("cat_id", "state_id"),
            "dept_state":  ("dept_id", "state_id"),
            "cat_store":   ("cat_id", "store_id"),
            "dept_store":  ("dept_id", "store_id"),
        }

        for key_name, attrs in compound_keys.items():
            groups: dict[tuple, set[int]] = {}
            for sku_idx, row in meta.iterrows():
                attr_tuple = tuple(row[a] for a in attrs)
                groups.setdefault(attr_tuple, set()).add(int(sku_idx))
            # Filter out hyperedges with <2 members (CP requires |e| >= 2).
            edge_list = []
            attr_list = []
            for attr_tuple, members in groups.items():
                if len(members) >= 2:
                    edge_list.append(members)
                    attr_list.append(attr_tuple)
            self.hyperedges[key_name] = edge_list
            self._partition_class_attr_values[key_name] = attr_list
            self.qa_report.hyperedge_counts[key_name] = len(edge_list)

        logger.info("Hyperedge counts: %s", self.qa_report.hyperedge_counts)

    # =========================================================================
    # Tensor accessors.
    # =========================================================================

    @property
    def n_hyperedges(self) -> int:
        return sum(len(v) for v in self.hyperedges.values())

    def hyperedge_incidence_matrix(self) -> torch.Tensor:
        """(n_skus, n_hyperedges) {0,1} incidence."""
        n_e = self.n_hyperedges
        H = torch.zeros(self.n_skus, n_e, dtype=torch.float32)
        col = 0
        for key in self.PARTITION_KEYS:
            for members in self.hyperedges[key]:
                for sku in members:
                    H[sku, col] = 1.0
                col += 1
        return H

    def hyperedge_partition_vector(self) -> torch.Tensor:
        """(n_hyperedges,) int tensor of partition class indices."""
        vec: list[int] = []
        for class_idx, key in enumerate(self.PARTITION_KEYS):
            vec.extend([class_idx] * len(self.hyperedges[key]))
        return torch.tensor(vec, dtype=torch.long)

    def hyperedge_class_sizes(self) -> dict[str, int]:
        return {k: len(self.hyperedges[k]) for k in self.PARTITION_KEYS}

    def max_hyperedge_size(self) -> int:
        return max(len(e) for k in self.PARTITION_KEYS for e in self.hyperedges[k])

    def mean_hyperedge_size(self) -> float:
        sizes = [
            len(e) for k in self.PARTITION_KEYS for e in self.hyperedges[k]
        ]
        return float(np.mean(sizes)) if sizes else 0.0

    # =========================================================================
    # Splits.
    # =========================================================================

    def rolling_origin_split(
        self,
        train_frac: float = 0.80,
        cal_frac: float = 0.10,
    ) -> tuple[range, range, range]:
        """Rolling-origin temporal split (contiguous).

        Same convention as SupplyGraph: train is [0, t_train), cal is
        [t_train, t_cal), test is [t_cal, T). Test must be later than cal
        which must be later than train (rolling origin).
        """
        t_train = int(self.n_timesteps * train_frac)
        t_cal = int(self.n_timesteps * (train_frac + cal_frac))
        return (
            range(0, t_train),
            range(t_train, t_cal),
            range(t_cal, self.n_timesteps),
        )


# =============================================================================
# Smoke test.
# =============================================================================


def _smoke_test() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    # Smoke: 500 SKUs, last 120 days.
    m5 = M5Hypergraph(
        subsample_n=500,
        subsample_days=120,
        min_total_sales=1.0,
        standardise=True,
        seed=42,
    )
    print(m5.qa_report)
    print()
    print(f"Node feature tensor:  {tuple(m5.node_features.shape)}")
    print(f"Total hyperedges:     {m5.n_hyperedges}")
    print(f"Mean hyperedge size:  {m5.mean_hyperedge_size():.1f}")
    print(f"Max hyperedge size:   {m5.max_hyperedge_size()}")
    print()

    print("Per-class hyperedge counts and Theorem 1 structural-term scaling:")
    print(f"  {'class':>12s}  {'|E_k|':>6s}  {'log|E_k|/|E_k|':>16s}")
    for cls, n in m5.hyperedge_class_sizes().items():
        if n > 1:
            term = np.log(n) / n
        else:
            term = float('inf')
        print(f"  {cls:>12s}  {n:>6d}  {term:>16.4f}")

    H = m5.hyperedge_incidence_matrix()
    P = m5.hyperedge_partition_vector()
    print()
    print(f"Incidence shape:     {tuple(H.shape)}")
    print(f"Incidence sparsity:  {1.0 - H.mean().item():.4f}")
    print(f"Partition vector:    {tuple(P.shape)} (max class = {P.max().item()})")

    train, cal, test = m5.rolling_origin_split()
    print()
    print(f"Rolling-origin split: train=[{train.start},{train.stop}), "
          f"cal=[{cal.start},{cal.stop}), test=[{test.start},{test.stop})")


if __name__ == "__main__":
    _smoke_test()
