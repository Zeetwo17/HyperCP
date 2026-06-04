"""
NCPNET-style non-exchangeable conformal prediction on hypergraph time
series.

Wang, Kang, Yan, Kulkarni, Zhou ("Non-exchangeable Conformal Prediction
for Temporal Graph Neural Networks," KDD 2025, DOI 10.1145/3711896.3737064)
build a CP procedure for temporal pairwise GNNs that uses a
diffusion-based non-conformity score capturing both topological and
temporal uncertainty, with finite-sample coverage-gap bounds in the
Barber-Candes-Ramdas-Tibshirani (Annals 2023) weighted-CP style.

This module gives a faithful adapter of the NCPNET *recipe* to our
hypergraph setting, on the same forecaster as our other comparators,
so the comparison isolates the contribution of the hyperedge unit and
the partition decomposition (vs node-level scores with graph diffusion
and online adaptation).

Adaptation
----------
1. Convert the hypergraph incidence H to a pairwise clique-expanded
   adjacency A (same as in `cf_gnn.py`).
2. Build the normalised Laplacian L and the heat-kernel-style diffusion
   smoother S_diffusion = (I + lambda * L)^(-1).
3. Compute node-level CQR-asymmetric scores s_c.
4. Smooth: s_tilde = S_diffusion @ s.
5. Feed the diffused scores to ACI (Gibbs & Candes 2021) for online
   alpha_t adaptation; the ACI step absorbs the time-series
   non-exchangeability that the original NCPNET handles via weighted
   CP.

The combination "diffusion-smoothed score + ACI" mirrors NCPNET's
"diffusion-aware score + non-exchangeable CP" but reuses our existing
ACI engine rather than NCPNET's TV-distance-bound weighted CP scheme.
We label the resulting baseline "NCPNET-style" to make the
simplification explicit.

Public reference implementation: https://github.com/ODYSSEYWT/NCPNET

Reference
---------
Wang, T., Kang, J., Yan, Y., Kulkarni, A., Zhou, D. (2025).
Non-exchangeable Conformal Prediction for Temporal Graph Neural
Networks. KDD '25. DOI 10.1145/3711896.3737064.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from torch import nn

from .aci import ACI, ACIConfig, cqr_score


@dataclass
class NCPNETHyperConfig:
    """Configuration for the NCPNET-style hypergraph calibrator.

    Parameters
    ----------
    alpha_target : float
        Desired miscoverage level. 0.10 for 90% intervals.
    lambda_smooth : float
        Diffusion smoothing strength. Default 0.5.
    aci_eta : float
        ACI step size for the online alpha update. Default 0.05 per
        Gibbs-Candes 2021.
    window : int
        ACI calibration-window size.
    quantile_lo_idx, quantile_hi_idx : int
        Indices into the K-quantile axis of q_pred.
    clique_weighting : str
        "unweighted" | "co_occur" | "normalised" (see cf_gnn.py).
    """
    alpha_target: float = 0.10
    lambda_smooth: float = 0.5
    aci_eta: float = 0.05
    window: int = 30
    quantile_lo_idx: int = 0
    quantile_hi_idx: int = -1
    clique_weighting: str = "unweighted"


class NCPNETStyleCalibrator(nn.Module):
    """NCPNET-style: graph-diffused CQR score + ACI.

    >>> cal = NCPNETStyleCalibrator(NCPNETHyperConfig(lambda_smooth=0.5))
    >>> cal.set_hypergraph(incidence)
    >>> cal.fit_initial(q_cal, y_cal)
    >>> for t in test_days:
    ...     lower, upper, covered, alpha_t = cal.step(q_t, y_t)
    """

    def __init__(self, cfg: NCPNETHyperConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.register_buffer("smoother", None, persistent=False)
        self._n_firms: int = 0
        self.aci: Optional[ACI] = None

    # ---------------------------------------------------------------------
    # Hypergraph wiring (identical to CF-GNN).
    # ---------------------------------------------------------------------

    def set_hypergraph(self, incidence: torch.Tensor) -> None:
        if incidence.ndim != 2:
            raise ValueError(
                f"incidence must be 2-D (n_firms, n_he); got {tuple(incidence.shape)}."
            )
        H = incidence.cpu().numpy().astype(np.float64) if isinstance(incidence, torch.Tensor) \
            else np.asarray(incidence, dtype=np.float64)
        n_firms = H.shape[0]
        self._n_firms = n_firms

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

        np.fill_diagonal(A, 0.0)
        deg = A.sum(axis=1)
        D_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg.clip(min=1e-12)), 0.0)
        A_norm = (D_inv_sqrt[:, None] * A) * D_inv_sqrt[None, :]
        L = np.eye(n_firms) - A_norm
        S = np.linalg.solve(
            np.eye(n_firms) + self.cfg.lambda_smooth * L,
            np.eye(n_firms),
        )
        self.smoother = torch.tensor(S, dtype=torch.float32, device=incidence.device)

        # Build the ACI engine on the diffused-score space.
        self.aci = ACI(ACIConfig(
            alpha_target=self.cfg.alpha_target,
            eta=self.cfg.aci_eta,
            window=self.cfg.window,
            window_mode="fixed",
            quantile_lo_idx=self.cfg.quantile_lo_idx,
            quantile_hi_idx=self.cfg.quantile_hi_idx,
        ))

    def _apply_smoother(self, scores: torch.Tensor) -> torch.Tensor:
        """Apply graph smoother along the firm axis (axis 1).

        scores: (T, n_firms, ...) -> same shape.
        """
        S = self.smoother
        orig_shape = scores.shape
        perm = (0,) + tuple(range(2, scores.ndim)) + (1,)
        x = scores.permute(*perm).contiguous()
        x_flat = x.reshape(-1, self._n_firms)
        x_smooth = x_flat @ S.T
        x_back = x_smooth.reshape(*x.shape)
        inv_perm = list(range(scores.ndim))
        inv_perm[1] = scores.ndim - 1
        for k in range(2, scores.ndim):
            inv_perm[k] = k - 1
        return x_back.permute(*inv_perm).contiguous()

    def _diffused_cqr_scores(
        self,
        q_pred: torch.Tensor,
        y_true: torch.Tensor,
    ) -> torch.Tensor:
        """Compute graph-diffused CQR scores for a (T, n_firms, ...) batch."""
        K = q_pred.size(-1)
        hi = self.cfg.quantile_hi_idx if self.cfg.quantile_hi_idx >= 0 else K + self.cfg.quantile_hi_idx
        q_lo = q_pred[..., self.cfg.quantile_lo_idx]
        q_hi = q_pred[..., hi]
        s_raw = torch.maximum(q_lo - y_true, y_true - q_hi)
        return self._apply_smoother(s_raw)

    # ---------------------------------------------------------------------
    # Calibration initialisation.
    # ---------------------------------------------------------------------

    def fit_initial(
        self,
        q_pred_cal: torch.Tensor,
        y_true_cal: torch.Tensor,
    ) -> None:
        """Seed the ACI window with graph-diffused CQR scores."""
        if self.smoother is None or self.aci is None:
            raise RuntimeError("Call set_hypergraph(H) before fit_initial.")
        if q_pred_cal.size(1) != self._n_firms:
            raise ValueError(
                f"q_pred_cal has {q_pred_cal.size(1)} firms but hypergraph has "
                f"{self._n_firms}."
            )

        s_diff = self._diffused_cqr_scores(q_pred_cal, y_true_cal)
        # s_diff shape: (T_cal, n_firms, ...). Seed ACI window per-timestep.
        T_cal = s_diff.size(0)
        start = max(0, T_cal - self.cfg.window)
        for t in range(start, T_cal):
            self.aci.state.score_window.append(s_diff[t])

    # ---------------------------------------------------------------------
    # One step.
    # ---------------------------------------------------------------------

    @torch.no_grad()
    def step(
        self,
        q_pred_t: torch.Tensor,
        y_true_t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict, observe, update.

        Parameters
        ----------
        q_pred_t : (n_firms, ..., K)
        y_true_t : (n_firms, ...)
        """
        if self.aci is None:
            raise RuntimeError("Call set_hypergraph(H) first.")
        threshold = self.aci._threshold()
        K = q_pred_t.size(-1)
        hi = self.cfg.quantile_hi_idx if self.cfg.quantile_hi_idx >= 0 else K + self.cfg.quantile_hi_idx
        lower = q_pred_t[..., self.cfg.quantile_lo_idx] - threshold
        upper = q_pred_t[..., hi] + threshold
        covered = (y_true_t >= lower) & (y_true_t <= upper)

        err_t = (~covered).float().mean()
        alpha_used = self.aci.state.alpha_t.detach().clone()
        new_alpha = self.aci.state.alpha_t + self.aci.cfg.eta * (
            self.aci.cfg.alpha_target - err_t
        )
        self.aci.state.alpha_t = self.aci._clamp_alpha(new_alpha)

        self.aci.state.coverage_history.append(float(covered.float().mean()))
        return lower, upper, covered, alpha_used

    def coverage_summary(self) -> dict:
        if self.aci is None or not self.aci.state.coverage_history:
            return {"n": 0}
        cov = torch.tensor(self.aci.state.coverage_history)
        return {
            "n": int(cov.numel()),
            "marginal_coverage": float(cov.mean()),
            "target_coverage": 1.0 - self.cfg.alpha_target,
            "final_alpha": float(self.aci.state.alpha_t.float().mean()),
            "lambda_smooth": self.cfg.lambda_smooth,
        }


# =============================================================================
# Smoke test.
# =============================================================================


def _smoke_test() -> None:
    torch.manual_seed(0)
    print("NCPNET-style smoke test")
    print("=" * 60)

    n_firms = 10
    n_he = 6
    T_cal = 200
    H_horizon = 1
    K = 5

    # Hypergraph with non-trivial structure.
    incidence = (torch.rand(n_firms, n_he) > 0.5).float()
    for e in range(n_he):
        if incidence[:, e].sum() < 2:
            incidence[:2, e] = 1.0

    quantiles = torch.tensor([0.05, 0.10, 0.50, 0.90, 0.95])
    sigma_pred = 0.8
    q_template = sigma_pred * torch.distributions.Normal(0, 1).icdf(quantiles)
    q_cal = q_template.view(1, 1, 1, K).expand(T_cal, n_firms, H_horizon, K).contiguous()
    y_cal = torch.randn(T_cal, n_firms, H_horizon)

    cal = NCPNETStyleCalibrator(NCPNETHyperConfig(
        alpha_target=0.10, lambda_smooth=0.5, aci_eta=0.05, window=200,
    ))
    cal.set_hypergraph(incidence)
    cal.fit_initial(q_cal, y_cal)

    # Run online.
    T_test = 500
    torch.manual_seed(1)
    for t in range(T_test):
        y_t = torch.randn(n_firms, H_horizon)
        q_t = q_template.view(1, 1, K).expand(n_firms, H_horizon, K).contiguous()
        cal.step(q_t, y_t)

    summ = cal.coverage_summary()
    print(f"  marginal coverage: {summ['marginal_coverage']:.4f}  (target {summ['target_coverage']:.2f})")
    print(f"  lambda_smooth   : {summ['lambda_smooth']}")
    print(f"  final alpha     : {summ['final_alpha']:.4f}")
    assert abs(summ["marginal_coverage"] - summ["target_coverage"]) < 0.08, (
        f"NCPNET-style should be within +/-0.08 of target on iid; got "
        f"{summ['marginal_coverage']:.4f}"
    )
    print("  [OK] NCPNET-style calibrator within +/-0.08 of nominal coverage.")


if __name__ == "__main__":
    _smoke_test()
