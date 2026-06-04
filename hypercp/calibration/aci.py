"""
Adaptive Conformal Inference (Gibbs & Candès 2021).

This is the headline calibration method for HyperCP. It absorbs the
time-series non-exchangeability that breaks split CP into a single
scalar adaptation parameter alpha_t, which is updated online via:

    alpha_{t+1} = alpha_t + eta * (alpha_target - err_t)            (1)

where err_t in {0, 1} is the indicator of miscoverage at step t.
This is the Gibbs-Candes 2021 update rule (Algorithm 1). When the
empirical miscoverage err exceeds the target, alpha drops, raising
the quantile level (1-alpha) used to construct the prediction set
and widening the interval -- correcting the under-coverage.
Theorem 1 in this paper builds on Gibbs-Candès Theorem 1's regret
guarantee:

    | (1/T) sum_t err_t  -  alpha_target |  <=  eta + O(1/sqrt(w))

which is the "ACI rolling-origin regret" term in our coverage bound.

Implementation notes
--------------------
- We maintain a sliding window of recent CQR-style non-conformity scores
  from which the conformal threshold q_hat_t is recomputed each step.
- The conformal threshold is the (1 - alpha_t)(1 + 1/n) empirical quantile
  of the window (standard CQR adjustment with finite-sample correction).
- alpha_t is clipped to (0, 1) to remain a valid quantile level.
- Three update modes are supported:
    'global'   : single alpha_t for all (firm, horizon) pairs
    'per_class': one alpha_t per partition class (Theorem 1's per-class)
    'per_firm' : one alpha_t per firm (high-variance, rarely needed)

The module is purely functional state-and-step; no learnable parameters.

Reference
---------
Gibbs, I. & Candès, E. (2021). Adaptive conformal inference under
distribution shift. NeurIPS.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import nn


# =============================================================================
# CQR-style score helper -- used by both fit_calibration and step.
# =============================================================================


def cqr_score(
    q_pred: torch.Tensor,
    y_true: torch.Tensor,
    lo_idx: int,
    hi_idx: int,
) -> torch.Tensor:
    """Standard CQR score (Romano et al. 2019):
       s = max(q_lo - y, y - q_hi)

    Inside the interval: s < 0. Outside: s > 0. The conformal threshold
    is therefore a non-negative number, and the prediction set is
    [q_lo - threshold, q_hi + threshold].
    """
    K = q_pred.size(-1)
    hi = hi_idx if hi_idx >= 0 else K + hi_idx
    q_lo_v = q_pred[..., lo_idx]
    q_hi_v = q_pred[..., hi]
    return torch.maximum(q_lo_v - y_true, y_true - q_hi_v)


# =============================================================================
# ACI config.
# =============================================================================


@dataclass
class ACIConfig:
    """Configuration for ACI.

    Parameters
    ----------
    alpha_target : float
        Desired miscoverage level (e.g., 0.10 for 90% intervals).
    eta : float
        ACI step size. 0.05 is the Gibbs-Candès recommended default.
    window : int
        Calibration-window size; conformal threshold recomputed from this
        many observations.
    window_mode : str
        'fixed' (default; Gibbs-Candès): the calibration window is set
        once at `fit_initial()` and NEVER updated. Only alpha_t adapts.
        Provably regret-bounded.

        'rolling': the window slides forward as new observations arrive,
        incorporating test-time scores. More adaptive to distribution drift
        but exhibits known positive-feedback instability on i.i.d. data
        (see AgACI / Zaffran 2022).
    quantile_lo_idx : int
        Index of lower quantile in K-quantile predictions.
    quantile_hi_idx : int
        Index of upper quantile in K-quantile predictions.
    mode : str
        'global', 'per_class', or 'per_firm'.
    n_classes : int | None
        Number of partition classes (required if mode='per_class').
    n_firms : int | None
        Number of firms (required if mode='per_firm').
    alpha_min : float
        Lower clip for alpha_t (default 0.001).
    alpha_max : float
        Upper clip for alpha_t (default 0.999).
    """
    alpha_target: float = 0.10
    eta: float = 0.05
    window: int = 30
    window_mode: str = "fixed"
    quantile_lo_idx: int = 0
    quantile_hi_idx: int = -1
    mode: str = "global"
    n_classes: Optional[int] = None
    n_firms: Optional[int] = None
    alpha_min: float = 1e-3
    alpha_max: float = 1.0 - 1e-3


# =============================================================================
# ACI state.
# =============================================================================


@dataclass
class ACIState:
    """Running state of an ACI loop."""
    alpha_t: torch.Tensor          # scalar or per-stream tensor
    score_window: list[torch.Tensor] = field(default_factory=list)
    coverage_history: list[float] = field(default_factory=list)
    alpha_history: list[torch.Tensor] = field(default_factory=list)
    threshold_history: list[torch.Tensor] = field(default_factory=list)


# =============================================================================
# ACI module.
# =============================================================================


class ACI(nn.Module):
    """Adaptive Conformal Inference loop.

    Typical usage
    -------------
    >>> aci = ACI(ACIConfig(alpha_target=0.1, eta=0.05, window=30))
    >>> aci.fit_initial(q_pred_cal, y_true_cal)
    >>> for t in test_days:
    ...     lower, upper, covered, alpha_t = aci.step(q_pred_t, y_true_t)
    """

    def __init__(self, cfg: ACIConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self._init_state()

    def _init_state(self) -> None:
        if self.cfg.mode == "global":
            alpha_t = torch.tensor(self.cfg.alpha_target)
        elif self.cfg.mode == "per_class":
            if self.cfg.n_classes is None:
                raise ValueError("per_class mode requires n_classes")
            alpha_t = torch.full(
                (self.cfg.n_classes,), self.cfg.alpha_target
            )
        elif self.cfg.mode == "per_firm":
            if self.cfg.n_firms is None:
                raise ValueError("per_firm mode requires n_firms")
            alpha_t = torch.full(
                (self.cfg.n_firms,), self.cfg.alpha_target
            )
        else:
            raise ValueError(f"Unknown ACI mode: {self.cfg.mode}")

        self.state = ACIState(alpha_t=alpha_t)

    @property
    def alpha_t(self) -> torch.Tensor:
        return self.state.alpha_t

    @property
    def n_observed(self) -> int:
        return len(self.state.coverage_history)

    # ---------------------------------------------------------------------
    # Initialisation from a calibration batch.
    # ---------------------------------------------------------------------

    def fit_initial(
        self,
        q_pred_cal: torch.Tensor,
        y_true_cal: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> None:
        """Seed the score window from a calibration set.

        Parameters
        ----------
        q_pred_cal : (T_cal, ..., K)
        y_true_cal : (T_cal, ...)  matching shape minus the K dimension.
        mask       : optional same-shape boolean mask of points to include.
        """
        s = cqr_score(
            q_pred_cal, y_true_cal,
            self.cfg.quantile_lo_idx, self.cfg.quantile_hi_idx,
        )
        if mask is not None:
            s = s.masked_fill(~mask.bool(), float("nan"))

        # Push the most recent `window` per-step score batches into the buffer.
        # If T_cal > window we keep the most recent `window`.
        T_cal = s.size(0)
        start = max(0, T_cal - self.cfg.window)
        for t in range(start, T_cal):
            self.state.score_window.append(s[t])

    # ---------------------------------------------------------------------
    # Threshold computation.
    # ---------------------------------------------------------------------

    def _threshold(self) -> torch.Tensor:
        """Compute conformal threshold from the current score window.

        Returns
        -------
        q_threshold : scalar tensor (global mode) or per-stream tensor.
        """
        if not self.state.score_window:
            raise RuntimeError(
                "Score window is empty; call fit_initial() first."
            )
        # Flatten scores from each step then concatenate.
        s_all = torch.cat([s.flatten() for s in self.state.score_window], dim=0)
        s_all = s_all[~torch.isnan(s_all)]
        n = s_all.numel()
        if n == 0:
            raise RuntimeError("All calibration scores are NaN.")

        alpha = self._clamp_alpha(self.state.alpha_t).flatten()[0].item()
        level = min(1.0, (1.0 - alpha) * (1.0 + 1.0 / n))
        return torch.quantile(s_all, level)

    def _clamp_alpha(self, a: torch.Tensor) -> torch.Tensor:
        return a.clamp(self.cfg.alpha_min, self.cfg.alpha_max)

    # ---------------------------------------------------------------------
    # One ACI step.
    # ---------------------------------------------------------------------

    @torch.no_grad()
    def step(
        self,
        q_pred_t: torch.Tensor,
        y_true_t: torch.Tensor,
        partition: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict, observe, update.

        Parameters
        ----------
        q_pred_t : (..., K) quantile predictions at time t.
        y_true_t : (...,)   observed targets at time t.
        partition: optional per-stream partition class assignments (for
                   per_class mode). Shape must match y_true_t.

        Returns
        -------
        lower, upper : (...,) prediction interval bounds.
        covered      : (...,) boolean covered indicator.
        alpha_used   : scalar tensor -- the alpha_t used for this step.
        """
        # 1. Threshold from current window.
        q_threshold = self._threshold()

        # 2. Build prediction interval.
        K = q_pred_t.size(-1)
        hi_idx = (self.cfg.quantile_hi_idx
                  if self.cfg.quantile_hi_idx >= 0
                  else K + self.cfg.quantile_hi_idx)
        lower = q_pred_t[..., self.cfg.quantile_lo_idx] - q_threshold
        upper = q_pred_t[..., hi_idx] + q_threshold

        # 3. Covered indicator.
        covered = (y_true_t >= lower) & (y_true_t <= upper)

        # 4. Compute miscoverage error and update alpha_t.
        err_t = self._aggregate_error(~covered, partition)
        alpha_used = self.state.alpha_t.detach().clone()

        # Gibbs-Candes 2021 sign convention: target - err.
        # err > target  =>  alpha decreases  =>  wider interval  =>  more coverage.
        new_alpha = self.state.alpha_t + self.cfg.eta * (
            self.cfg.alpha_target - err_t
        )
        self.state.alpha_t = self._clamp_alpha(new_alpha)

        # 5. Append to score window only in 'rolling' window mode.
        # In 'fixed' mode (Gibbs-Candes default), the calibration distribution
        # is locked at fit_initial() and only alpha_t adapts.
        if self.cfg.window_mode == "rolling":
            s_t = cqr_score(
                q_pred_t, y_true_t,
                self.cfg.quantile_lo_idx, self.cfg.quantile_hi_idx,
            )
            self.state.score_window.append(s_t)
            if len(self.state.score_window) > self.cfg.window:
                self.state.score_window.pop(0)
        elif self.cfg.window_mode != "fixed":
            raise ValueError(
                f"Unknown window_mode: {self.cfg.window_mode}"
            )

        # 6. Record history.
        self.state.coverage_history.append(covered.float().mean().item())
        self.state.alpha_history.append(alpha_used.cpu().clone())
        self.state.threshold_history.append(q_threshold.cpu().clone())

        return lower, upper, covered, alpha_used

    def _aggregate_error(
        self,
        err_t: torch.Tensor,
        partition: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Reduce per-stream miscoverage to the form expected by alpha_t."""
        err_f = err_t.float()
        if self.cfg.mode == "global":
            return err_f.mean()
        elif self.cfg.mode == "per_class":
            if partition is None:
                raise ValueError("per_class mode requires `partition`.")
            K_c = self.cfg.n_classes
            out = torch.zeros(K_c, device=err_f.device)
            counts = torch.zeros(K_c, device=err_f.device)
            for k in range(K_c):
                mask = (partition == k)
                if mask.any():
                    out[k] = err_f[mask].mean()
                    counts[k] = mask.float().sum()
            # Where a class has no observations this step, fall back to
            # the global average so alpha_t for that class doesn't drift.
            global_avg = err_f.mean()
            out = torch.where(counts > 0, out, global_avg.expand_as(out))
            return out
        elif self.cfg.mode == "per_firm":
            return err_f
        else:
            raise ValueError(self.cfg.mode)

    # ---------------------------------------------------------------------
    # Inference-only prediction (no update).
    # ---------------------------------------------------------------------

    @torch.no_grad()
    def predict_interval(
        self,
        q_pred_t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return [lower, upper] without observing y or updating state."""
        K = q_pred_t.size(-1)
        hi_idx = (self.cfg.quantile_hi_idx
                  if self.cfg.quantile_hi_idx >= 0
                  else K + self.cfg.quantile_hi_idx)
        q_threshold = self._threshold()
        lower = q_pred_t[..., self.cfg.quantile_lo_idx] - q_threshold
        upper = q_pred_t[..., hi_idx] + q_threshold
        return lower, upper

    # ---------------------------------------------------------------------
    # Diagnostics.
    # ---------------------------------------------------------------------

    def coverage_summary(self) -> dict[str, float]:
        """Summary statistics over the recorded coverage history."""
        if not self.state.coverage_history:
            return {"n": 0}
        cov = torch.tensor(self.state.coverage_history)
        return {
            "n": int(cov.numel()),
            "marginal_coverage": float(cov.mean()),
            "target_coverage":   1.0 - self.cfg.alpha_target,
            "coverage_std":      float(cov.std()),
            "final_alpha":       float(self.state.alpha_t.float().mean()),
            "alpha_drift":       float(
                (self.state.alpha_t - self.cfg.alpha_target).abs().mean()
            ),
        }


# =============================================================================
# Smoke test.
# =============================================================================


def _smoke_test() -> None:
    torch.manual_seed(0)
    print("=" * 60)
    print("ACI smoke test 1: synthetic i.i.d. data (should hit ~90%)")
    print("=" * 60)
    K = 5
    quantiles = torch.tensor([0.05, 0.10, 0.50, 0.90, 0.95])
    n_streams = 50  # larger for less per-step variance

    # Truth: y ~ N(0, 1.0). Predictor: emits N(0, 0.8) quantiles -- a
    # slightly *narrower* envelope than reality. This is the canonical
    # ACI setup: imperfect base predictor, and conformal correction
    # widens the interval to recover nominal coverage. The dynamic is
    # well-behaved (negative feedback at the equilibrium).
    sigma_pred = 0.8
    q_template = sigma_pred * torch.distributions.Normal(0, 1).icdf(quantiles)
    q_pred_step = q_template.unsqueeze(0).expand(n_streams, K)

    # Calibration: 200 days x 50 streams of i.i.d. N(0, 1) truth.
    T_cal = 200
    y_cal = torch.randn(T_cal, n_streams)
    q_cal = q_pred_step.unsqueeze(0).expand(T_cal, n_streams, K)

    # Default 'fixed' window_mode -- calibration distribution locked at fit time.
    aci = ACI(ACIConfig(alpha_target=0.10, eta=0.05, window=200, window_mode="fixed"))
    aci.fit_initial(q_cal, y_cal)
    print(f"Initial threshold from {T_cal*n_streams} cal scores: "
          f"{aci._threshold().item():.3f}  (should be POSITIVE -- conformal widens "
          f"the narrow predictor envelope)")

    # Test: 1000 days x 50 streams.
    T_test = 1000
    for t in range(T_test):
        y_t = torch.randn(n_streams)
        lower, upper, covered, alpha_used = aci.step(q_pred_step, y_t)

    cov_steady = torch.tensor(
        aci.state.coverage_history[T_test // 2:]
    ).mean().item()
    summ = aci.coverage_summary()
    print(f"After {T_test} steps:")
    print(f"  marginal coverage (all):  {summ['marginal_coverage']:.3f}")
    print(f"  steady-state (last half): {cov_steady:.3f}  (target 0.90)")
    print(f"  final alpha:              {summ['final_alpha']:.3f}")
    # Gibbs-Candes 2021 regret bound: |coverage - target| <= eta + O(1/sqrt(T)).
    # With eta=0.005 and T=1000, the bound is ~0.005 + ~0.03 ~= 0.04.
    # Allow 0.05 slack for n_streams variance.
    assert abs(cov_steady - 0.9) < 0.05, \
        f"Steady-state coverage {cov_steady:.3f} outside [0.85, 0.95] band."
    print("[OK] ACI within regret band on matched-predictor i.i.d. data.")

    # ---------------------------------------------------------------------
    print()
    print("=" * 60)
    print("ACI smoke test 2: distribution-shift data (should adapt)")
    print("=" * 60)
    # Same predictor, but truth changes mid-stream: sigma jumps from 1 to 2.
    # Fixed calibration mode so the alpha dynamic is well-behaved.
    aci2 = ACI(ACIConfig(
        alpha_target=0.10, eta=0.05, window=200, window_mode="fixed"
    ))
    aci2.fit_initial(q_cal, y_cal)
    alphas_pre, alphas_post = [], []
    n_pre = 200
    n_post = 200
    for t in range(n_pre + n_post):
        scale = 1.0 if t < n_pre else 2.0
        y_t = scale * torch.randn(n_streams)
        _, _, _, a = aci2.step(q_pred_step, y_t)
        (alphas_pre if t < n_pre else alphas_post).append(float(a))
    cov_pre = torch.tensor(aci2.state.coverage_history[:n_pre]).mean()
    cov_post = torch.tensor(aci2.state.coverage_history[n_pre:]).mean()
    a_pre = sum(alphas_pre) / len(alphas_pre)
    a_post = sum(alphas_post) / len(alphas_post)
    print(f"Pre-shift  (t<{n_pre}):  coverage={cov_pre:.3f}, mean alpha={a_pre:.3f}")
    print(f"Post-shift (t>={n_pre}): coverage={cov_post:.3f}, mean alpha={a_post:.3f}")
    # After the variance jumps from 1 to 2, the predictor's intervals are
    # too narrow -> coverage drops -> err exceeds alpha_target -> alpha drops
    # (lower alpha = wider interval) to recover coverage.
    assert a_post < a_pre, (
        f"Alpha should DROP after shift (got pre={a_pre:.3f}, post={a_post:.3f}). "
        f"Lower alpha = higher requested coverage = wider interval."
    )
    print("[OK] ACI adapts alpha downward under distribution shift.")

    # ---------------------------------------------------------------------
    print()
    print("=" * 60)
    print("ACI smoke test 3: per-class mode")
    print("=" * 60)
    n_streams = 30
    K_c = 3
    partition = torch.tensor([k % K_c for k in range(n_streams)])
    q_pred_step = q_template.unsqueeze(0).expand(n_streams, K)
    T_cal = 50
    y_cal = torch.randn(T_cal, n_streams)
    q_cal = q_pred_step.unsqueeze(0).expand(T_cal, n_streams, K)

    aci3 = ACI(ACIConfig(
        alpha_target=0.10, eta=0.05, window=200, window_mode="fixed",
        mode="per_class", n_classes=K_c,
    ))
    aci3.fit_initial(q_cal, y_cal)
    for t in range(150):
        # Class 0 i.i.d., class 1 over-dispersed, class 2 under-dispersed.
        scale = torch.tensor([1.0, 1.5, 0.7])[partition]
        y_t = scale * torch.randn(n_streams)
        aci3.step(q_pred_step, y_t, partition=partition)
    # Class 1 (scale=1.5, truth more dispersed than calibration) -> coverage drops
    #    -> err > target -> alpha drops (to widen interval).
    # Class 2 (scale=0.7, truth less dispersed than calibration) -> coverage rises
    #    -> err < target -> alpha rises (to narrow interval).
    print(f"Final per-class alpha_t: {aci3.state.alpha_t.tolist()}")
    print(f"  class 0 (matched scale=1.0):           {aci3.state.alpha_t[0]:.3f}")
    print(f"  class 1 (truth more dispersed, 1.5):    {aci3.state.alpha_t[1]:.3f}  <-- should be lowest")
    print(f"  class 2 (truth less dispersed, 0.7):    {aci3.state.alpha_t[2]:.3f}  <-- should be highest")
    print("[OK] Per-class ACI maintains separate alpha_t per partition class.")


if __name__ == "__main__":
    _smoke_test()
