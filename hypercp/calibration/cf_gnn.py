"""
CF-GNN-style graph-smoothed conformal calibration on hypergraphs.

Huang, Jin, Candes, and Leskovec (NeurIPS 2023; arXiv:2305.14535) propose
a graph-aware conformal procedure for static pairwise GNNs that uses
the graph Laplacian to share calibration information across nodes. Their
exact 1-alpha guarantee relies on a transductive permutation-symmetry
argument between calibration and test nodes; this does not extend to
rolling-origin time series.

This module gives a faithful adapter of the CF-GNN *score* to our
hypergraph setting, run on the same calibration window as our other
baselines so that the comparison isolates the contribution of the
hyperedge unit (vs node-level scores with graph smoothing) and the
adaptation mechanism (split-CP vs ACI).

Adaptation
----------
1. Convert the hypergraph incidence H to a pairwise clique-expanded
   adjacency A (two nodes share an edge iff they co-occur in any
   hyperedge). Binarise (do not weight by hyperedge multiplicity --
   we report the unweighted variant as the headline CF-GNN-adapted
   baseline; the weighted variant is available via the
   `clique_weighting` flag).
2. Compute the symmetric normalised Laplacian L = I - D^(-1/2) A D^(-1/2).
3. Form the diffusion smoother S = (I + lambda * L)^(-1).
4. Compute node-level CQR-asymmetric scores s_c at the calibration set.
5. Smooth: s_tilde = S @ s.
6. Apply standard split-CP at level 1 - alpha on smoothed scores.

This preserves the CF-GNN spirit (graph structure participates in the
calibration via Laplacian-based score smoothing) without claiming the
transductive exact-coverage guarantee that does not apply to our
non-exchangeable rolling-origin setting.

The lambda parameter controls smoothing strength. lambda = 0 reduces
to standard split-CP (sanity check). Larger lambda pulls each node's
score toward its graph-neighbour average, which can either help (if
true residuals are smooth on the graph) or hurt (if they are not).

Public reference implementation of CF-GNN:
https://github.com/snap-stanford/conformalized-gnn

Reference
---------
Huang, K., Jin, Y., Candes, E., Leskovec, J. (2023). Uncertainty
Quantification over Graph with Conformalized Graph Neural Networks.
NeurIPS Spotlight. arXiv:2305.14535.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch import nn


@dataclass
class CFGNNHyperConfig:
    """Configuration for the CF-GNN-style graph-smoothed calibrator.

    Parameters
    ----------
    alpha_target : float
        Desired miscoverage level. 0.10 for 90% intervals.
    lambda_smooth : float
        Diffusion smoothing strength. lambda = 0 recovers standard
        split-CP; larger lambda pulls each node's score toward its
        clique-neighbour mean. Default 0.5 follows the recommended
        range from the CF-GNN reference repo.
    quantile_lo_idx, quantile_hi_idx : int
        Indices into the K-quantile axis of q_pred.
    clique_weighting : str
        How to weight the clique-expanded adjacency.
        - "unweighted": A_ij = 1 iff i and j share any hyperedge (default).
        - "co_occur":   A_ij = number of hyperedges containing both i and j.
        - "normalised": A_ij = co_occurrence(i,j) / sqrt(deg(i) * deg(j)).
    """
    alpha_target: float = 0.10
    lambda_smooth: float = 0.5
    quantile_lo_idx: int = 0
    quantile_hi_idx: int = -1
    clique_weighting: str = "unweighted"


class CFGNNSmoothedCalibrator(nn.Module):
    """CF-GNN-style graph-smoothed split-CP calibrator on hypergraphs.

    >>> cal = CFGNNSmoothedCalibrator(CFGNNHyperConfig(lambda_smooth=0.5))
    >>> cal.set_hypergraph(incidence)
    >>> cal.fit(q_cal, y_cal)
    >>> lower, upper = cal.predict_interval(q_test)
    """

    def __init__(self, cfg: CFGNNHyperConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.register_buffer("smoother", None, persistent=False)
        self.register_buffer("threshold", None, persistent=False)
        self._n_firms: int = 0

    # ---------------------------------------------------------------------
    # Hypergraph wiring.
    # ---------------------------------------------------------------------

    def set_hypergraph(self, incidence: torch.Tensor) -> None:
        """Precompute the diffusion smoother from the hypergraph incidence.

        Parameters
        ----------
        incidence : (n_firms, n_he) {0, 1} matrix.
        """
        if incidence.ndim != 2:
            raise ValueError(
                f"incidence must be 2-D (n_firms, n_he); got {tuple(incidence.shape)}."
            )
        H = incidence.cpu().numpy().astype(np.float64) if isinstance(incidence, torch.Tensor) \
            else np.asarray(incidence, dtype=np.float64)
        n_firms = H.shape[0]
        self._n_firms = n_firms

        # Clique-expanded adjacency.
        if self.cfg.clique_weighting == "unweighted":
            cooccur = H @ H.T
            A = (cooccur > 0).astype(np.float64)
        elif self.cfg.clique_weighting == "co_occur":
            A = H @ H.T
        elif self.cfg.clique_weighting == "normalised":
            cooccur = H @ H.T
            deg = cooccur.diagonal()
            denom = np.sqrt(np.outer(deg, deg)).clip(min=1e-12)
            A = cooccur / denom
        else:
            raise ValueError(f"Unknown clique_weighting: {self.cfg.clique_weighting!r}")

        np.fill_diagonal(A, 0.0)  # remove self-loops from clique expansion

        # Normalised Laplacian L = I - D^{-1/2} A D^{-1/2}.
        deg = A.sum(axis=1)
        D_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg.clip(min=1e-12)), 0.0)
        A_norm = (D_inv_sqrt[:, None] * A) * D_inv_sqrt[None, :]
        L = np.eye(n_firms) - A_norm

        # Diffusion smoother S = (I + lambda * L)^{-1}.
        S = np.linalg.solve(
            np.eye(n_firms) + self.cfg.lambda_smooth * L,
            np.eye(n_firms),
        )
        self.smoother = torch.tensor(S, dtype=torch.float32, device=incidence.device)

    # ---------------------------------------------------------------------
    # Calibration.
    # ---------------------------------------------------------------------

    def fit(
        self,
        q_pred_cal: torch.Tensor,
        y_true_cal: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> None:
        """Compute the conformal threshold from graph-smoothed scores.

        Parameters
        ----------
        q_pred_cal : (T_cal, n_firms, ..., K) quantile predictions.
        y_true_cal : (T_cal, n_firms, ...) realised targets.
        mask       : optional same-shape (without K) boolean mask.

        The smoothing is applied along the n_firms axis. For multi-step
        prediction, scores at each (timestep, horizon) are smoothed
        independently, then concatenated for the threshold computation.
        """
        if self.smoother is None:
            raise RuntimeError("CFGNNSmoothedCalibrator: call set_hypergraph(H) first.")
        if q_pred_cal.size(1) != self._n_firms:
            raise ValueError(
                f"q_pred_cal has {q_pred_cal.size(1)} firms but hypergraph has "
                f"{self._n_firms}."
            )

        K = q_pred_cal.size(-1)
        hi_idx = self.cfg.quantile_hi_idx if self.cfg.quantile_hi_idx >= 0 \
            else K + self.cfg.quantile_hi_idx
        lo_idx = self.cfg.quantile_lo_idx

        # CQR-asymmetric: s_c = max(q_lo - y, y - q_hi).
        q_lo = q_pred_cal[..., lo_idx]
        q_hi = q_pred_cal[..., hi_idx]
        s_raw = torch.maximum(q_lo - y_true_cal, y_true_cal - q_hi)
        # s_raw shape: (T_cal, n_firms, ...). Apply smoother along firm axis.
        s_smooth = self._apply_smoother(s_raw)

        # Flatten and apply mask.
        s_flat = s_smooth.reshape(-1)
        if mask is not None:
            m = mask.reshape(-1).bool()
            s_flat = s_flat[m]
        s_flat = s_flat[~torch.isnan(s_flat)]
        n = s_flat.numel()
        if n == 0:
            raise RuntimeError("All calibration scores are NaN after masking.")
        level = min(1.0, (1.0 - self.cfg.alpha_target) * (1.0 + 1.0 / n))
        self.threshold = torch.quantile(s_flat, level)

    def _apply_smoother(self, scores: torch.Tensor) -> torch.Tensor:
        """Apply the graph smoother along the firm axis (axis 1).

        scores: (T, n_firms, ...) -> same shape, smoothed.
        """
        S = self.smoother  # (n_firms, n_firms)
        # einsum: 'ij, tj... -> ti...' but '...' is variable -- use matmul reshape.
        orig_shape = scores.shape
        # Move firm axis to last position, smooth, move back.
        # scores: (T, n_firms, *trailing). Permute to (T, *trailing, n_firms).
        perm = (0,) + tuple(range(2, scores.ndim)) + (1,)
        x = scores.permute(*perm).contiguous()
        x_flat = x.reshape(-1, self._n_firms)  # (T * prod(trailing), n_firms)
        x_smooth = x_flat @ S.T  # smoothing acts on the rightmost axis
        x_back = x_smooth.reshape(*x.shape)
        inv_perm = list(range(scores.ndim))
        inv_perm[1] = scores.ndim - 1
        for k in range(2, scores.ndim):
            inv_perm[k] = k - 1
        return x_back.permute(*inv_perm).contiguous()

    # ---------------------------------------------------------------------
    # Prediction.
    # ---------------------------------------------------------------------

    def predict_interval(self, q_pred: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the CF-GNN-style smoothed conformal interval.

        Parameters
        ----------
        q_pred : (..., n_firms, ..., K)

        Returns
        -------
        (lower, upper) : interval bounds with the K axis dropped.
        """
        if self.threshold is None:
            raise RuntimeError("Call fit() before predict_interval().")
        K = q_pred.size(-1)
        hi = self.cfg.quantile_hi_idx if self.cfg.quantile_hi_idx >= 0 else K + self.cfg.quantile_hi_idx
        lower = q_pred[..., self.cfg.quantile_lo_idx] - self.threshold
        upper = q_pred[..., hi] + self.threshold
        return lower, upper

    def covered(self, q_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        lower, upper = self.predict_interval(q_pred)
        return (y_true >= lower) & (y_true <= upper)


# =============================================================================
# Smoke test.
# =============================================================================


def _smoke_test() -> None:
    torch.manual_seed(0)

    print("CF-GNN-style smoke test")
    print("=" * 60)
    n_firms = 10
    n_he = 6
    T = 50
    H_horizon = 4
    K = 5

    # Fake hypergraph + features.
    incidence = (torch.rand(n_firms, n_he) > 0.5).float()
    for e in range(n_he):
        if incidence[:, e].sum() < 2:
            incidence[:2, e] = 1.0

    quantiles = torch.tensor([0.05, 0.10, 0.50, 0.90, 0.95])
    sigma_pred = 0.8
    q_template = sigma_pred * torch.distributions.Normal(0, 1).icdf(quantiles)
    # Add a per-firm offset so smoothing has something to do.
    firm_offsets = torch.randn(1, n_firms, 1, 1) * 0.3
    q_cal = q_template.view(1, 1, 1, K).expand(T, n_firms, H_horizon, K).contiguous()
    q_cal = q_cal + firm_offsets
    y_cal = torch.randn(T, n_firms, H_horizon)

    # Lambda = 0 should match standard split-CP.
    cal0 = CFGNNSmoothedCalibrator(CFGNNHyperConfig(lambda_smooth=0.0))
    cal0.set_hypergraph(incidence)
    cal0.fit(q_cal, y_cal)
    thr0 = cal0.threshold.item()

    # Reference: standard split-CP.
    from .split_cp import SplitCP, SplitCPConfig
    spl = SplitCP(SplitCPConfig(alpha_target=0.10))
    spl.fit(q_cal.view(-1, K), y_cal.view(-1))
    thr_ref = spl.q_threshold.item()
    print(f"  lambda=0 threshold:    {thr0:.4f}")
    print(f"  Split-CP threshold:    {thr_ref:.4f}")
    print(f"  abs diff:              {abs(thr0 - thr_ref):.2e}")
    assert abs(thr0 - thr_ref) < 1e-4, (
        f"CF-GNN with lambda=0 should match split-CP exactly; got "
        f"{thr0:.6f} vs {thr_ref:.6f}"
    )
    print(f"  [OK] lambda=0 recovers standard split-CP.")

    # Lambda > 0 should give a different threshold (smoothing changes the score distribution).
    cal05 = CFGNNSmoothedCalibrator(CFGNNHyperConfig(lambda_smooth=0.5))
    cal05.set_hypergraph(incidence)
    cal05.fit(q_cal, y_cal)
    thr05 = cal05.threshold.item()
    print(f"  lambda=0.5 threshold:  {thr05:.4f}")

    # Coverage on iid-ish data should be near nominal in both cases.
    q_test = q_template.view(1, 1, 1, K).expand(20, n_firms, H_horizon, K).contiguous() + firm_offsets
    y_test = torch.randn(20, n_firms, H_horizon) + firm_offsets.squeeze(-1)
    lower, upper = cal05.predict_interval(q_test)
    picp = ((y_test >= lower) & (y_test <= upper)).float().mean().item()
    print(f"  lambda=0.5 PICP on i.i.d. test: {picp:.3f} (target 0.90)")
    assert abs(picp - 0.9) < 0.1, (
        f"CF-GNN smoothed should give roughly nominal coverage on iid data; "
        f"got {picp:.3f}"
    )
    print(f"  [OK] CF-GNN-style calibrator achieves roughly nominal coverage.")


if __name__ == "__main__":
    _smoke_test()
