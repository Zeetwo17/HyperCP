"""
Split conformal prediction -- the invalid time-series baseline.

This is the canonical (Vovk-Shafer 2005) split-CP procedure with NO adaptation
to time-series non-exchangeability. It is included for two purposes:

1. As the baseline labelled "split CP" in §4.2.12 ablation (b). The paper's
   empirical claim is that split-CP suffers coverage drift on rolling-origin
   SupplyGraph time series whereas ACI does not.
2. As a sanity check on the conformity-score pipeline: split-CP coverage
   on i.i.d. data (e.g., randomly permuted SCR cross-sections) should be
   close to the nominal target.

We deliberately label this as "invalid for time series" -- the calibrated
threshold is fixed at fit time and never updated.

Reference
---------
Vovk, Gammerman & Shafer (2005). Algorithmic Learning in a Random World.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class SplitCPConfig:
    """Configuration for split conformal prediction (CQR variant)."""
    alpha_target: float = 0.10
    quantile_lo_idx: int = 0       # index into K quantiles for lower bound
    quantile_hi_idx: int = -1      # index for upper bound (default last)


class SplitCP(nn.Module):
    """Standard split-CP for quantile regression (Romano et al. 2019).

    Single fixed conformal threshold computed once from a calibration set.
    No update at test time.
    """

    def __init__(self, cfg: SplitCPConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.register_buffer("q_threshold", None, persistent=False)

    @property
    def lo_idx(self) -> int:
        return self.cfg.quantile_lo_idx

    @property
    def hi_idx(self) -> int:
        return self.cfg.quantile_hi_idx

    # ---------------------------------------------------------------------
    # Calibration.
    # ---------------------------------------------------------------------

    def fit(
        self,
        q_pred_cal: torch.Tensor,
        y_true_cal: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> None:
        """Compute the conformal threshold from a calibration set.

        Parameters
        ----------
        q_pred_cal : (..., K) quantile predictions on calibration data.
                     Leading dims can be anything; the function flattens.
        y_true_cal : (...,)    targets aligned with q_pred_cal[..., 0].
        mask       : optional (...,) mask of points to include.

        After this call, `q_threshold` holds the (1-α)(1+1/n)-empirical
        quantile of the CQR scores.
        """
        K = q_pred_cal.size(-1)
        if K < 2:
            raise ValueError(f"Need at least 2 quantiles; got K={K}.")
        hi = self.hi_idx if self.hi_idx >= 0 else K + self.hi_idx
        lo = self.lo_idx

        q_lo = q_pred_cal[..., lo]
        q_hi = q_pred_cal[..., hi]

        # CQR score: s_i = max(q_lo - y_i, y_i - q_hi)
        s = torch.maximum(q_lo - y_true_cal, y_true_cal - q_hi)
        if mask is not None:
            s = s[mask.bool()]
        s_flat = s.flatten()

        n = s_flat.numel()
        if n == 0:
            raise RuntimeError("Calibration set is empty after masking.")

        # Empirical (1-α)(1+1/n)-quantile.
        # Conservative rounding to ensure validity in finite samples.
        adjusted_level = min(1.0, (1.0 - self.cfg.alpha_target) * (1.0 + 1.0 / n))
        self.q_threshold = torch.quantile(s_flat, adjusted_level)

    # ---------------------------------------------------------------------
    # Prediction.
    # ---------------------------------------------------------------------

    def predict_interval(
        self,
        q_pred: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the conformal prediction interval [lower, upper].

        Parameters
        ----------
        q_pred : (..., K)

        Returns
        -------
        (lower, upper) : each (...,) without the K dimension.
        """
        if self.q_threshold is None:
            raise RuntimeError("Call fit() before predict_interval().")
        K = q_pred.size(-1)
        hi = self.hi_idx if self.hi_idx >= 0 else K + self.hi_idx
        lower = q_pred[..., self.lo_idx] - self.q_threshold
        upper = q_pred[..., hi] + self.q_threshold
        return lower, upper

    def covered(
        self,
        q_pred: torch.Tensor,
        y_true: torch.Tensor,
    ) -> torch.Tensor:
        """Boolean tensor: True where y_true is inside the prediction set."""
        lower, upper = self.predict_interval(q_pred)
        return (y_true >= lower) & (y_true <= upper)


if __name__ == "__main__":
    # Quick smoke test.
    torch.manual_seed(0)
    n_cal = 200
    K = 5
    # Synthetic: y ~ N(0, 1); q_pred is the 5%, 10%, 50%, 90%, 95% quantiles
    # of a N(0, 1.5) prediction -- slightly over-dispersed so split CP needs
    # to shrink the interval.
    cfg = SplitCPConfig(alpha_target=0.1)
    cp = SplitCP(cfg)

    y_cal = torch.randn(n_cal)
    levels = torch.tensor([0.05, 0.10, 0.50, 0.90, 0.95])
    # Predicted quantiles of a N(0, 1.5) distribution.
    q_pred_cal = 1.5 * torch.distributions.Normal(0, 1).icdf(levels)
    q_pred_cal = q_pred_cal.expand(n_cal, K)
    cp.fit(q_pred_cal, y_cal)
    print(f"SplitCP threshold = {cp.q_threshold.item():.3f}")

    # Test on fresh draws.
    n_test = 500
    y_test = torch.randn(n_test)
    q_pred_test = (1.5 * torch.distributions.Normal(0, 1).icdf(levels)).expand(n_test, K)
    covered = cp.covered(q_pred_test, y_test)
    print(f"Test coverage: {covered.float().mean().item():.3f} (target 0.900).")
    assert abs(covered.float().mean().item() - 0.9) < 0.05, \
        "Coverage should be close to nominal on i.i.d. data."
    print("[OK] SplitCP smoke test passed.")
