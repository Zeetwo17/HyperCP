"""
Hyperedge-level non-conformity scores.

This module implements Equation (3) of theorem.tex:

    s_e^tau  =  max_{c in e}  w_{c,e}  *  rho_tau(y_c - q_hat_{c,tau})

with the default inverse-degree weight  w_{c,e} = 1 / deg_e(c). For
set-valued hypergraphs (SupplyGraph), deg_e(c) = 1 for c in e so the
weight is the identity and the score reduces to an unweighted max over
hyperedge members.

The max aggregation is the contraction that makes Lemma 3 (hyperedge-to-
node coverage transfer) go through: if max_{c in e} rho_tau(...) <= Q,
then every node c in e has rho_tau(...) <= Q individually. Mean
aggregation, while mathematically valid as a score, shrinks the conformal
threshold by ~1/sqrt(|e|) and empirically collapses node-level marginal
coverage (Section 6.6 ablation (a)).

This score is the input to the ACI calibration loop (Phase 4) and the
empirical test in Gate 1 (Phase 6.1). Theorem 1 in the paper provides the
finite-sample marginal coverage guarantee on these scores under the
conditional partition exchangeability assumption (A1).

Two aggregation strategies are exposed (with `aggregation` config field):

1. `aggregation="max"`  -- default. Matches Eq. (3) of theorem.tex and
   the empirical pipeline in `scripts/run_gate1.py`. Theorem 1's
   coverage bound applies.

2. `aggregation="mean"` -- ablation variant. Reported in Section 6.6
   ablation (a) as a counterexample illustrating that the max aggregation
   is essential for the node-level guarantee.

Two weighting strategies are exposed (with `weighting` config field):

1. `weighting="inverse_degree"` -- default, w_{c,e} = 1/deg_e(c). For
   set-valued hyperedges this is identically 1 for c in e.

2. `weighting="attention"`      -- ablation variant, learned attention
   weights via encoder embeddings. Theorem 1's bound is conjectural for
   this variant (the constant in the structural error term is not bounded
   in closed form for arbitrary learnable weights); we test it
   empirically in Section 6.5.

Performance note: the max-aggregation path uses a masked broadcast +
torch.max reduction. Profiling shows < 10 ms for SupplyGraph's full
calibration window on CPU; GPU latency is negligible.

References
----------
- Romano, Patterson & Candes (NeurIPS 2019) -- CQR (asymmetric pinball)
- Theorem 1, theorem.tex Section 4
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


# =============================================================================
# Pinball residual helper.
# =============================================================================


def pinball_residual(
    y_true: torch.Tensor,
    q_pred: torch.Tensor,
    tau: torch.Tensor,
) -> torch.Tensor:
    """Asymmetric pinball residual rho_tau(y - q) = max(tau*(y-q), (tau-1)*(y-q)).

    Parameters
    ----------
    y_true : (..., H)   observed targets per (..., horizon)
    q_pred : (..., H, K) quantile predictions per (..., horizon, quantile)
    tau    : (K,)        quantile levels

    Returns
    -------
    rho    : (..., H, K)  per-quantile residual, non-negative.
    """
    if q_pred.shape[:-1] != y_true.shape:
        raise ValueError(
            f"shape mismatch: q_pred {tuple(q_pred.shape)} vs y_true {tuple(y_true.shape)}"
        )
    K = q_pred.size(-1)
    if tau.numel() != K:
        raise ValueError(
            f"tau has {tau.numel()} entries but q_pred last dim is {K}"
        )
    y_b = y_true.unsqueeze(-1)  # (..., H, 1)
    diff = y_b - q_pred  # (..., H, K)
    tau_b = tau.to(diff.device).view(*([1] * (diff.ndim - 1)), K)
    return torch.maximum(tau_b * diff, (tau_b - 1.0) * diff)


# =============================================================================
# CQR-asymmetric hyperedge score (the Gate-1 pipeline's actual scorer).
# =============================================================================


def hyperedge_cqr_score(
    q_pred: torch.Tensor,
    y_true: torch.Tensor,
    incidence: torch.Tensor,
    lo_idx: int = 0,
    hi_idx: int = -1,
    aggregation: str = "max",
    size_weight: str = "none",
) -> torch.Tensor:
    """Hyperedge-level CQR-asymmetric non-conformity score.

    For each (time t, hyperedge e, horizon h) computes the node-level
    CQR-asymmetric score
        s_c(t, h) = max(q_lo_c(t, h) - y_c(t, h), y_c(t, h) - q_hi_c(t, h))
    (Romano et al. NeurIPS 2019) and aggregates over the firms c in e via
    either max (default, Eq. 3 of theorem.tex) or weighted mean
    (Section 6.6 (a) ablation).

    This is the canonical score used by `scripts/run_gate1.py`. It maps
    directly onto a CQR prediction interval at the firm level:
        [q_lo_c - threshold, q_hi_c + threshold]
    with threshold = quantile of the calibration-set hyperedge scores.

    Parameters
    ----------
    q_pred : torch.Tensor, shape (..., n_firms, H, K)
        Quantile predictions. The lo and hi indices select the lower and
        upper conformal quantiles.
    y_true : torch.Tensor, shape (..., n_firms, H)
        Realised targets at the same (n_firms, H) layout.
    incidence : torch.Tensor, shape (n_firms, n_he)
        {0, 1} incidence matrix of the hypergraph.
    lo_idx, hi_idx : int
        Indices into the K-quantile axis of q_pred. Default lo_idx=0 (the
        lowest quantile) and hi_idx=-1 (the highest). For a 5-quantile
        head at (0.05, 0.10, 0.50, 0.90, 0.95) the defaults give a
        90%-target conformal interval.
    aggregation : str
        Either "max" (default, theorem.tex Eq. 3 -- worst-case-firm in the
        hyperedge) or "mean" (1/|e| weighted average, ablation in §6.6 (a)).
    size_weight : str
        Hyperedge-size-dependent weighting applied to each firm's CQR
        residual *before* the max aggregation. Discussed in §7 of the
        paper as a mitigation for the M5 width-premium issue.
        - "none"          : w_{c,e} = 1 (default; matches Eq. 3 verbatim).
        - "inverse_sqrt"  : w_{c,e} = 1/sqrt(|e|). The score for a
                            hyperedge of size |e| is reduced by a factor
                            sqrt(|e|), so large hyperedges contribute
                            less to the empirical conformal threshold.
                            Per-firm bound becomes
                            rho_c <= Q * sqrt(|e|) (looser per-firm
                            implication, but the *empirical* threshold
                            is smaller, often reducing PINAW).
        - "inverse"       : w_{c,e} = 1/|e|. More aggressive.
        These weight options only affect the "max" aggregation path; the
        "mean" aggregation uses the row-normalised W_norm as before.

    Returns
    -------
    s_e : torch.Tensor, shape (..., n_he, H)
        Hyperedge-level CQR-asymmetric scores.
    """
    if q_pred.shape[:-1] != y_true.shape:
        raise ValueError(
            f"q_pred shape {tuple(q_pred.shape)} incompatible with "
            f"y_true shape {tuple(y_true.shape)}; expected q_pred to be "
            f"y_true.shape + (K,)."
        )
    if incidence.ndim != 2:
        raise ValueError(
            f"incidence must be 2-D (n_firms, n_he); got {tuple(incidence.shape)}."
        )
    n_firms_inc, n_he = incidence.shape
    if y_true.size(-2) != n_firms_inc:
        raise ValueError(
            f"y_true last-but-one dim ({y_true.size(-2)}) must match "
            f"incidence n_firms ({n_firms_inc})."
        )

    K = q_pred.size(-1)
    if hi_idx < 0:
        hi_idx = K + hi_idx

    # Node-level CQR-asymmetric score.
    q_lo = q_pred[..., lo_idx]  # (..., n_firms, H)
    q_hi = q_pred[..., hi_idx]  # (..., n_firms, H)
    s_node = torch.maximum(q_lo - y_true, y_true - q_hi)  # (..., n_firms, H)

    # Aggregate over hyperedge members.
    if aggregation == "max":
        # Mask non-members with -inf then take max along the firm axis.
        # s_node: (..., n_firms, H).
        H_T = incidence.T  # (n_he, n_firms)
        n_he_local = H_T.size(0)
        n_firms_local = H_T.size(1)
        s_e = s_node.unsqueeze(-3)  # (..., 1, n_firms, H)
        mask_shape = [1] * (s_e.ndim - 3) + [n_he_local, n_firms_local, 1]
        non_member = (H_T == 0).view(*mask_shape)

        # Per-(edge, firm) size weights applied before the max.
        # For size_weight="none", w is identity and the path reduces to
        # the original max-aggregation Eq. 3 verbatim.
        if size_weight != "none":
            edge_sizes = H_T.sum(dim=1).clamp_min(1.0)  # (n_he,)
            if size_weight == "inverse_sqrt":
                w_per_edge = 1.0 / edge_sizes.sqrt()
            elif size_weight == "inverse":
                w_per_edge = 1.0 / edge_sizes
            else:
                raise ValueError(
                    f"Unknown size_weight: {size_weight!r}. "
                    f"Options: 'none', 'inverse_sqrt', 'inverse'."
                )
            # Broadcast (n_he,) to the (..., n_he, n_firms, H) shape.
            w_shape = [1] * (s_e.ndim - 3) + [n_he_local, 1, 1]
            w_view = w_per_edge.view(*w_shape).to(s_e.dtype)
            # Multiply rho by w_{c,e} = w_e (uniform within the hyperedge
            # for these size-weight modes).
            s_e = s_e * w_view

        s_e = s_e.masked_fill(non_member, float("-inf"))
        return s_e.max(dim=-2).values  # (..., n_he, H)

    elif aggregation == "mean":
        # Row-normalised incidence (1/|e| weights). Same einsum-based
        # weighted mean as `HyperedgeConformityScore` in aggregation="mean"
        # mode but operating on a single residual signal (no quantile axis).
        H_T = incidence.T.to(s_node.dtype)  # (n_he, n_firms)
        edge_sizes = H_T.sum(dim=1, keepdim=True).clamp_min(1.0)  # (n_he, 1)
        W_norm = H_T / edge_sizes  # (n_he, n_firms)
        return torch.einsum("ec,...ch->...eh", W_norm, s_node)

    else:
        raise ValueError(
            f"Unknown aggregation: {aggregation!r}. "
            f"Options: 'max' (default) or 'mean' (ablation)."
        )


# =============================================================================
# Hyperedge weight matrix W_norm[e, c].
# =============================================================================


def build_weight_matrix(
    incidence: torch.Tensor,
    weighting: str = "inverse_degree",
    z: Optional[torch.Tensor] = None,
    attention_vec: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Build the per-hyperedge per-firm weight matrix.

    Each row e holds the weights w_{c,e} normalised by |e|. The output
    satisfies row_sum_c W_norm[e, c] = 1 for every hyperedge e.

    Parameters
    ----------
    incidence : (n_firms, n_he) {0,1} matrix
    weighting : 'inverse_degree' (default) or 'attention' (v2 ablation)
    z : optional (n_firms, d) encoder embedding; required for 'attention'
    attention_vec : optional (d,) learnable parameter; required for 'attention'

    Returns
    -------
    W_norm : (n_he, n_firms), row-normalised
    """
    if incidence.ndim != 2:
        raise ValueError(f"incidence must be 2-D, got {tuple(incidence.shape)}")

    if weighting == "inverse_degree":
        # For SupplyGraph (set-valued hyperedges), deg_e(c) = 1 for c in e.
        # So w_{c,e} = 1, and W_norm[e, c] = 1/|e| if c in e else 0.
        H_T = incidence.T  # (n_he, n_firms)
        edge_sizes = H_T.sum(dim=1, keepdim=True).clamp_min(1.0)  # (n_he, 1)
        return H_T / edge_sizes

    elif weighting == "attention":
        if z is None or attention_vec is None:
            raise ValueError(
                "weighting='attention' requires z (firm embeddings) and "
                "attention_vec (learnable parameter)."
            )
        if z.size(0) != incidence.size(0):
            raise ValueError(
                f"z has {z.size(0)} firms but incidence has {incidence.size(0)}"
            )
        if z.size(1) != attention_vec.numel():
            raise ValueError(
                f"z embed dim {z.size(1)} != attention_vec dim {attention_vec.numel()}"
            )

        # Per-firm logit: <z_c, a>
        logits = z @ attention_vec  # (n_firms,)
        # For each hyperedge, take softmax over its members.
        H_T = incidence.T  # (n_he, n_firms)
        # Mask logits with very negative value outside the hyperedge so
        # they vanish in softmax.
        neg_inf = torch.tensor(-1e9, device=logits.device, dtype=logits.dtype)
        masked = torch.where(H_T > 0, logits.unsqueeze(0).expand_as(H_T), neg_inf)
        W_norm = torch.softmax(masked, dim=1)
        return W_norm

    else:
        raise ValueError(f"Unknown weighting='{weighting}'")


# =============================================================================
# Default hyperedge conformity score module (Eq. 3).
# =============================================================================


@dataclass
class ConformityConfig:
    """Configuration for the hyperedge conformity scorer.

    Attributes
    ----------
    aggregation : str
        How to combine the per-firm pinball residuals within each
        hyperedge into a single hyperedge-level score.
        - "max"  : worst-case-firm aggregation. Matches Eq. (3) in
                   theorem.tex and Theorem 1's Lemma 3 (hyperedge-to-node
                   coverage transfer). DEFAULT.
        - "mean" : weighted-mean aggregation. Ablation variant; reported
                   in Section 6.6 (a) as a counterexample where node-level
                   marginal coverage collapses.
    weighting : str
        Per-firm-per-hyperedge weighting strategy w_{c,e}.
        - "inverse_degree" : w_{c,e} = 1/deg_e(c). For set-valued
                             hyperedges (SupplyGraph) this is identically
                             1 for c in e. DEFAULT.
        - "attention"      : learned attention weights via encoder
                             embedding. Theorem 1's bound is conjectural
                             for this variant (Section 6.5 ablation).
    own_product_only : bool
        If True, extract the diagonal (firm c forecasts its own product
        c) from the (n_firms, n_products, ...) forecast tensor. This is
        the canonical SupplyGraph configuration. Set False to average
        across all products the firm handles (multi-product extension).
    """
    aggregation: str = "max"            # 'max' (default, Eq. 3) or 'mean' (ablation)
    weighting: str = "inverse_degree"   # 'inverse_degree' or 'attention'
    own_product_only: bool = True       # extract diagonal from forecast tensor


class HyperedgeConformityScore(nn.Module):
    """Equation (3) of theorem.tex.

    Computes per-hyperedge per-horizon per-quantile non-conformity scores
    from the quantile head's predictions and the realised targets.

    For SupplyGraph reformulation we treat n_firms = n_products and assume
    each firm produces only its own product (diagonal forecast). For
    multi-product extensions (M5), set `own_product_only=False` to average
    over all (firm, product) pairs the firm handles.
    """

    def __init__(self, cfg: ConformityConfig) -> None:
        super().__init__()
        self.cfg = cfg
        # Buffers for the precomputed weight matrix and incidence are set
        # via `set_hypergraph()`.
        self.register_buffer("W_norm", None, persistent=False)
        self.register_buffer("incidence", None, persistent=False)
        self.register_buffer("partition", None, persistent=False)
        self.attention_vec: Optional[nn.Parameter] = None

    # ---------------------------------------------------------------------
    # Hypergraph wiring.
    # ---------------------------------------------------------------------

    def set_hypergraph(
        self,
        incidence: torch.Tensor,
        partition: Optional[torch.Tensor] = None,
        embed_dim: Optional[int] = None,
    ) -> None:
        """Wire the hypergraph into the scorer.

        Parameters
        ----------
        incidence : (n_firms, n_he) {0,1} matrix
        partition : optional (n_he,) integer tensor of partition class indices,
                    used for per-class diagnostics and Theorem 1's per-class
                    statement.
        embed_dim : if weighting='attention', the encoder embedding dim; used
                    to allocate `attention_vec`.
        """
        if incidence.ndim != 2:
            raise ValueError(f"incidence must be 2-D, got {tuple(incidence.shape)}")
        self.incidence = incidence
        if partition is not None:
            if partition.numel() != incidence.size(1):
                raise ValueError(
                    f"partition has {partition.numel()} entries but "
                    f"incidence has {incidence.size(1)} hyperedges"
                )
            self.partition = partition

        if self.cfg.weighting == "inverse_degree":
            self.W_norm = build_weight_matrix(incidence, "inverse_degree")
        elif self.cfg.weighting == "attention":
            if embed_dim is None:
                raise ValueError("embed_dim required for attention weighting")
            self.attention_vec = nn.Parameter(
                torch.randn(embed_dim, device=incidence.device) * 0.01
            )
            # W_norm computed lazily in forward (depends on z).
        else:
            raise ValueError(f"Unknown weighting: {self.cfg.weighting}")

    # ---------------------------------------------------------------------
    # Forward.
    # ---------------------------------------------------------------------

    def forward(
        self,
        q_hat: torch.Tensor,
        y_true: torch.Tensor,
        tau: torch.Tensor,
        z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute hyperedge-level non-conformity scores.

        Parameters
        ----------
        q_hat  : (..., n_firms, n_products, H, K) quantile predictions
        y_true : (..., n_firms, n_products, H)    realised targets
        tau    : (K,) quantile levels
        z      : optional (..., n_firms, d) embeddings for 'attention' weighting

        Returns
        -------
        scores : (..., n_he, H, K) per-hyperedge per-horizon per-quantile scores
        """
        if self.incidence is None:
            raise RuntimeError(
                "HyperedgeConformityScore: call set_hypergraph(H) before forward()."
            )

        # Step 1: extract per-firm residuals.
        # If own_product_only, take diagonal: y_diag = y_true[..., c, c, h]
        if self.cfg.own_product_only:
            n_firms = q_hat.size(-4)
            # Build index for the diagonal.
            idx = torch.arange(n_firms, device=q_hat.device)
            # q_hat: (..., n_firms, n_products, H, K) -> select c == p
            # Use advanced indexing with arange on both firm and product axes.
            q_diag = q_hat[..., idx, idx, :, :]  # (..., n_firms, H, K)
            y_diag = y_true[..., idx, idx, :]    # (..., n_firms, H)
        else:
            # Average over the product axis (each firm handles multiple products).
            # Here we sum residuals across all products the firm handles.
            # For now we just collapse via mean; could mask by incidence in
            # extensions.
            q_diag = q_hat.mean(dim=-3)          # (..., n_firms, H, K)
            y_diag = y_true.mean(dim=-2)         # (..., n_firms, H)

        # Step 2: pinball residual rho_tau(y - q).
        rho = pinball_residual(y_diag, q_diag, tau)
        # rho shape: (..., n_firms, H, K)

        # Step 3: aggregate per-firm rho over hyperedge members.
        #
        # For "max" (default, Eq. 3): scores[e, h, k] = max over c in e of
        # w_{c,e} * rho[c, h, k]. For set-valued hyperedges with the
        # inverse-degree weighting (the SupplyGraph regime), w_{c,e} = 1
        # for c in e, so this is the unweighted max. We implement the
        # general weighted max by masking-then-reducing along the firm
        # axis.
        #
        # For "mean" (ablation in Section 6.6 (a)): scores[e, h, k] =
        # sum_c W_norm[e, c] * rho[c, h, k] with row-normalised W_norm so
        # the sum is a weighted average. This is the einsum path that
        # used to be the only behaviour; preserved here only as an
        # ablation comparator.
        incidence = self.incidence  # (n_firms, n_he)

        if self.cfg.aggregation == "max":
            # Per-firm weight: identity for set-valued hyperedges; the
            # generalisation to multigraph hyperedges multiplies rho by
            # 1/deg_e(c) per (firm, edge) cell *before* the max, which is
            # equivalent to scaling rho along the firm axis when
            # deg_e(c) is the same for every hyperedge the firm sits in
            # (true on SupplyGraph; we leave the per-edge case as a TODO
            # for multigraph extensions).
            if self.cfg.weighting == "inverse_degree":
                weighted_rho = rho  # identity weight for set-valued
            elif self.cfg.weighting == "attention":
                if z is None:
                    raise ValueError("attention weighting requires z")
                # For attention with max aggregation, we need per (firm,
                # edge) weights. Compute the per-edge softmax over members
                # and apply before the max. This is the v2 ablation; the
                # max-with-attention bound is conjectural.
                z_flat = z.reshape(-1, z.size(-1))
                z_use = z_flat[-incidence.size(0):]
                W_attn = build_weight_matrix(
                    incidence, "attention", z=z_use,
                    attention_vec=self.attention_vec,
                )  # (n_he, n_firms)
                # We will apply W_attn inside the masked-max below.
                weighted_rho = None  # handled below
            else:
                raise ValueError(self.cfg.weighting)

            # Masked max along the firm axis.
            # rho: (..., n_firms, H, K).
            # Build mask of shape broadcastable to (..., n_he, n_firms, H, K)
            # so non-members contribute -inf and don't win the max.
            H_T = incidence.T.to(rho.dtype)  # (n_he, n_firms)
            n_he = H_T.size(0)
            n_firms_inc = H_T.size(1)

            # Reshape rho to add an n_he axis for broadcasting against H_T.
            # rho_e : (..., 1, n_firms, H, K)
            rho_e = rho.unsqueeze(-4)
            # Mask shape that broadcasts: (n_he, n_firms, 1, 1).
            # We pad with leading 1s to match rho_e.ndim.
            mask_shape = [1] * (rho_e.ndim - 4) + [n_he, n_firms_inc, 1, 1]
            non_member = (H_T == 0).view(*mask_shape)

            if self.cfg.weighting == "attention":
                # Multiply rho by per-(edge,firm) weight before the max.
                w_shape = [1] * (rho_e.ndim - 4) + [n_he, n_firms_inc, 1, 1]
                w_view = W_attn.view(*w_shape)
                weighted_rho_e = rho_e * w_view
            else:
                weighted_rho_e = rho_e

            masked = weighted_rho_e.masked_fill(non_member, float("-inf"))
            scores = masked.max(dim=-3).values  # (..., n_he, H, K)

        elif self.cfg.aggregation == "mean":
            # Ablation: weighted-mean aggregation. Score behaviour is the
            # einsum path that used to be the only behaviour.
            if self.cfg.weighting == "inverse_degree":
                W_norm = self.W_norm  # (n_he, n_firms)
            elif self.cfg.weighting == "attention":
                if z is None:
                    raise ValueError("attention weighting requires z")
                z_flat = z.reshape(-1, z.size(-1))
                z_use = z_flat[-incidence.size(0):]
                W_norm = build_weight_matrix(
                    incidence, "attention", z=z_use,
                    attention_vec=self.attention_vec,
                )
            else:
                raise ValueError(self.cfg.weighting)
            scores = torch.einsum("ec,...chk->...ehk", W_norm, rho)

        else:
            raise ValueError(
                f"Unknown aggregation: {self.cfg.aggregation!r}. "
                f"Options: 'max' (default) or 'mean' (ablation)."
            )

        return scores

    # ---------------------------------------------------------------------
    # Per-class diagnostics (for Theorem 1's per-class statement).
    # ---------------------------------------------------------------------

    def per_class_score_stats(
        self,
        scores: torch.Tensor,
    ) -> dict[int, dict[str, torch.Tensor]]:
        """Compute mean and std of scores within each partition class.

        Useful for verifying that the score distribution doesn't differ
        drastically across classes (which would weaken A1's exchangeability
        assumption). Returns a dict keyed by partition-class index.
        """
        if self.partition is None:
            raise RuntimeError(
                "partition not set; pass `partition` to set_hypergraph()."
            )

        out: dict[int, dict[str, torch.Tensor]] = {}
        unique_classes = torch.unique(self.partition).tolist()
        # scores shape: (..., n_he, H, K)
        for k in unique_classes:
            mask = (self.partition == k)
            class_scores = scores[..., mask, :, :]  # (..., n_k, H, K)
            out[int(k)] = {
                "n_hyperedges": torch.tensor(int(mask.sum())),
                "mean": class_scores.mean(),
                "std":  class_scores.std(),
                "q90":  torch.quantile(class_scores.flatten(), 0.9),
            }
        return out


# =============================================================================
# Smoke test.
# =============================================================================


def _smoke_test() -> None:
    torch.manual_seed(0)

    # --- Tiny synthetic test ---
    n_firms = 10
    n_prod = 10
    n_he = 6
    H = 4
    K = 5
    tau = torch.tensor([0.05, 0.10, 0.50, 0.90, 0.95])

    # Random incidence (each hyperedge has ≥ 2 nodes).
    incidence = (torch.rand(n_firms, n_he) > 0.5).float()
    for e in range(n_he):
        if incidence[:, e].sum() < 2:
            incidence[:2, e] = 1.0

    # Random partition assignment.
    partition = torch.randint(0, 3, (n_he,))

    # Default config (aggregation="max", weighting="inverse_degree").
    cfg = ConformityConfig(weighting="inverse_degree", own_product_only=True)
    scorer = HyperedgeConformityScore(cfg)
    scorer.set_hypergraph(incidence, partition=partition)
    assert scorer.cfg.aggregation == "max", (
        "default ConformityConfig.aggregation should be 'max' (Eq. 3 of theorem.tex)."
    )
    print(f"[OK] default aggregation = {scorer.cfg.aggregation!r}.")

    # Verify W_norm row sums (mean-mode weight matrix).
    assert torch.allclose(scorer.W_norm.sum(dim=1), torch.ones(n_he)), \
        "W_norm rows should sum to 1."
    print(f"[OK] W_norm row-sums all 1 (n_he={n_he}, n_firms={n_firms}).")

    # Random q_hat, y_true.
    q_hat = torch.randn(n_firms, n_prod, H, K)
    # Ensure quantile monotonicity for sanity.
    q_hat, _ = torch.sort(q_hat, dim=-1)
    y_true = torch.randn(n_firms, n_prod, H)

    scores_max = scorer(q_hat, y_true, tau)
    print(f"Unbatched (max): scores shape = {tuple(scores_max.shape)}")
    assert scores_max.shape == (n_he, H, K)
    assert (scores_max >= 0).all(), "Pinball-based scores must be non-negative."
    print(f"[OK] max-aggregated scores non-negative "
          f"(min={scores_max.min().item():.3f}, max={scores_max.max().item():.3f}).")

    # --- Ablation: mean aggregation (Section 6.6 (a) counterexample) ---
    scorer_mean = HyperedgeConformityScore(
        ConformityConfig(aggregation="mean", weighting="inverse_degree",
                         own_product_only=True)
    )
    scorer_mean.set_hypergraph(incidence, partition=partition)
    scores_mean = scorer_mean(q_hat, y_true, tau)
    assert scores_mean.shape == scores_max.shape
    # Max aggregation must dominate the mean elementwise (over hyperedges
    # where both sides have at least one member, which is all of them by
    # construction). This is the contraction Lemma 3 relies on.
    assert (scores_max >= scores_mean - 1e-6).all(), (
        "Max aggregation should dominate mean aggregation elementwise."
    )
    diff = (scores_max - scores_mean).flatten()
    print(f"[OK] max >= mean elementwise; mean(max - mean) = {diff.mean().item():.4f}.")

    # Verify that max equals the manual reference implementation used by
    # scripts/run_gate1.py (so a future refactor that swaps the inline
    # max in run_gate1.py for the scorer produces identical numbers).
    rho_manual = pinball_residual(
        y_true[torch.arange(n_firms), torch.arange(n_firms)],  # (n_firms, H)
        q_hat[torch.arange(n_firms), torch.arange(n_firms)],   # (n_firms, H, K)
        tau,
    )  # (n_firms, H, K)
    scores_ref = torch.full((n_he, H, K), float("-inf"))
    for e in range(n_he):
        members = (incidence[:, e] > 0).nonzero(as_tuple=True)[0]
        if members.numel() == 0:
            continue
        scores_ref[e] = rho_manual[members].max(dim=0).values
    assert torch.allclose(scores_max, scores_ref, atol=1e-6), (
        "Scorer max output diverges from reference implementation."
    )
    print("[OK] max-aggregated scorer matches reference implementation "
          "(parity with scripts/run_gate1.py inline max).")
    # Use scores_max for downstream assertions.
    scores = scores_max

    # Batched forward (e.g., calibration window of T_cal days).
    T_cal = 22
    q_hat_b = torch.randn(T_cal, n_firms, n_prod, H, K)
    q_hat_b, _ = torch.sort(q_hat_b, dim=-1)
    y_true_b = torch.randn(T_cal, n_firms, n_prod, H)
    scores_b = scorer(q_hat_b, y_true_b, tau)
    print(f"Batched (T_cal={T_cal}): scores shape = {tuple(scores_b.shape)}")
    assert scores_b.shape == (T_cal, n_he, H, K)

    # Per-class diagnostics.
    print("\nPer-class score stats (Theorem 1 verification):")
    stats = scorer.per_class_score_stats(scores_b)
    for k, s in stats.items():
        print(f"  class {k}: n_he={int(s['n_hyperedges'])}, "
              f"mean={s['mean']:.3f}, std={s['std']:.3f}, "
              f"q90={s['q90']:.3f}")

    # --- hyperedge_cqr_score parity test ---
    # The free function `hyperedge_cqr_score` is what scripts/run_gate1.py
    # will call after the refactor. We verify it reproduces the existing
    # inline NumPy implementation in eval_aci_hyperedge bit-for-bit.
    print()
    print("Parity test: hyperedge_cqr_score vs run_gate1.py inline reference")
    T_cal = 7
    q_pred_b = torch.randn(T_cal, n_firms, H, K)
    q_pred_b, _ = torch.sort(q_pred_b, dim=-1)
    y_b = torch.randn(T_cal, n_firms, H)
    lo_idx_p = 0
    hi_idx_p = K - 1  # absolute index for the manual reference
    # Helper-based score.
    s_he_helper = hyperedge_cqr_score(
        q_pred_b, y_b, incidence, lo_idx=lo_idx_p, hi_idx=hi_idx_p,
        aggregation="max",
    )  # (T_cal, n_he, H)
    # Reference: replicate the inline numpy loop in
    # scripts/run_gate1.py:eval_aci_hyperedge verbatim.
    q_lo_np = q_pred_b[..., lo_idx_p].numpy()
    q_hi_np = q_pred_b[..., hi_idx_p].numpy()
    y_np = y_b.numpy()
    import numpy as _np
    s_cqr_node = _np.maximum(q_lo_np - y_np, y_np - q_hi_np)  # (T, n_firms, H)
    H_mat = incidence.numpy().astype(_np.float32)
    s_ref = _np.zeros((T_cal, n_he, H), dtype=_np.float32)
    for e in range(n_he):
        members = _np.where(H_mat[:, e] > 0)[0]
        if members.size == 0:
            continue
        s_ref[:, e, :] = s_cqr_node[:, members, :].max(axis=1)
    s_ref_t = torch.from_numpy(s_ref)
    assert torch.allclose(s_he_helper.to(torch.float32), s_ref_t, atol=1e-5), (
        "hyperedge_cqr_score(aggregation='max') diverges from the inline "
        "Gate-1 reference. Numbers from the existing sanity run will not "
        "match after the refactor."
    )
    print(f"[OK] hyperedge_cqr_score(max) matches inline Gate-1 reference "
          f"(shape={tuple(s_he_helper.shape)}, abs-diff={(s_he_helper-s_ref_t).abs().max().item():.2e}).")
    # Sanity: mean variant gives smaller scores per hyperedge.
    s_he_mean = hyperedge_cqr_score(
        q_pred_b, y_b, incidence, lo_idx=lo_idx_p, hi_idx=hi_idx_p,
        aggregation="mean",
    )
    assert s_he_mean.shape == s_he_helper.shape
    # CQR-asymmetric scores can be negative (e.g. when y is well inside
    # the interval), but max >= mean must still hold for each (t, e, h).
    assert (s_he_helper >= s_he_mean - 1e-5).all(), (
        "hyperedge_cqr_score(max) should dominate (mean) elementwise."
    )
    print(f"[OK] hyperedge_cqr_score(mean) <= (max) elementwise.")

    # --- Attention-weighted ablation smoke test ---
    cfg_attn = ConformityConfig(weighting="attention", own_product_only=True)
    scorer_attn = HyperedgeConformityScore(cfg_attn)
    d = 8
    scorer_attn.set_hypergraph(incidence, partition=partition, embed_dim=d)
    z = torch.randn(n_firms, d)
    scores_attn = scorer_attn(q_hat, y_true, tau, z=z)
    print(f"\n[OK] Attention variant produces scores shape {tuple(scores_attn.shape)}")
    print(f"  attention_vec params: {scorer_attn.attention_vec.numel()}")

    # --- End-to-end with real SupplyGraph and trained encoder ---
    print("\n" + "=" * 60)
    print("End-to-end with real SupplyGraph (encoder + heads + scorer):")
    print("=" * 60)
    from hypercp.data.supplygraph import SupplyGraphHypergraph
    from hypercp.models import (
        EncoderConfig, HyperCPEncoder,
        QuantileHeadConfig, QuantileHead,
    )

    sg = SupplyGraphHypergraph()
    incidence_real = sg.hyperedge_incidence_matrix()
    partition_real = sg.hyperedge_partition_vector()

    encoder = HyperCPEncoder(EncoderConfig(n_features=sg.n_channels))
    encoder.set_hypergraph(incidence_real)

    qhead = QuantileHead(QuantileHeadConfig(
        hidden_dim=64, n_products=sg.n_products, horizon=4
    ))

    scorer_real = HyperedgeConformityScore(
        ConformityConfig(weighting="inverse_degree", own_product_only=True)
    )
    scorer_real.set_hypergraph(incidence_real, partition=partition_real)

    # Calibration window: days 176..198 (22 days).
    train, cal, test = sg.rolling_origin_split()
    T_window = 5
    H_horizon = 4
    K_q = qhead.n_quantiles
    tau_real = qhead._quantiles

    # For each calibration day t, forecast horizon h = 1..H from rolling window
    # ending at t. Realised target is days t+1..t+H.
    cal_scores_list = []
    for t in cal:
        if t + H_horizon >= sg.n_timesteps:
            break
        feat = sg.node_features[:, t - T_window + 1:t + 1, :]
        with torch.no_grad():
            z_seq, z_final = encoder(feat)
            q = qhead(z_final)  # (n_firms, n_products, H, K)
        # Target: channel 1 (sales_order) at horizons 1..H.
        y = sg.node_features[:, t + 1:t + 1 + H_horizon, 1]  # (n_firms, H)
        y_full = y.unsqueeze(1).expand(sg.n_products, sg.n_products, H_horizon)
        s = scorer_real(q, y_full, tau_real)
        cal_scores_list.append(s)

    cal_scores = torch.stack(cal_scores_list, dim=0)
    print(f"Calibration scores tensor: {tuple(cal_scores.shape)}")
    print(f"  (T_cal_days, n_hyperedges, H, K) = expected layout for ACI.")

    # Per-class stats on the real calibration scores.
    print("\nReal SupplyGraph calibration-set per-class stats:")
    real_stats = scorer_real.per_class_score_stats(cal_scores)
    class_names = {0: "plant", 1: "subgroup", 2: "group", 3: "storage"}
    for k, s in real_stats.items():
        cn = class_names.get(k, str(k))
        print(f"  {cn:10s} (k={k}): n_he={int(s['n_hyperedges'])}, "
              f"mean={s['mean']:.3f}, std={s['std']:.3f}")
    print("[OK] End-to-end conformity-score pipeline works on real SupplyGraph.")


if __name__ == "__main__":
    _smoke_test()
