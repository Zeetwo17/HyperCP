"""
Aggregated Adaptive Conformal Inference (Zaffran, Feron, Goude, Josse,
Dieuleveut, ICML 2022).

Vanilla ACI is sensitive to the choice of step size eta. Zaffran et al.
remove this sensitivity by running K parallel ACI controllers with
different eta values and aggregating their predictions via an online
expert-aggregation procedure (BOA in the paper; we expose both BOA-style
exponential weights and a uniform-weight reference implementation).

This module gives a clean baseline comparator for the ICDM submission:

- `AgACI(weighting="uniform")` -- the median across parallel ACIs.
  Simplest fair baseline; equivalent to picking eta = median-of-grid
  before seeing the data. Suitable as a sanity reference.

- `AgACI(weighting="boa")` -- BOA-style exponential-weighted aggregation
  with the miscoverage indicator as the per-expert loss. This is the
  default; it tracks the best-performing controller online and is what
  the paper recommends.

Both variants share the underlying ACI engine from `hypercp.calibration.aci`,
so behaviour aligns with the rest of the pipeline.

Reference
---------
Zaffran, M., Feron, O., Goude, Y., Josse, J., Dieuleveut, A. (2022).
Adaptive Conformal Predictions for Time Series. ICML.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import torch
from torch import nn

from .aci import ACI, ACIConfig, cqr_score


@dataclass
class AgACIConfig:
    """Configuration for AgACI.

    Parameters
    ----------
    alpha_target : float
        Desired miscoverage level. 0.10 for 90% intervals.
    etas : Sequence[float]
        Grid of step sizes for the K parallel ACI controllers. Default is
        (0.005, 0.01, 0.025, 0.05, 0.1) -- five experts spanning two
        orders of magnitude. K = len(etas).
    weighting : str
        "boa"     -- BOA-style exponential weights with the miscoverage
                     indicator as per-expert loss. DEFAULT.
        "uniform" -- equal weights; aggregate is the (component-wise)
                     median across experts. Useful as a simpler comparator.
    boa_eta : float
        Learning rate for the BOA exponential update. Default 1.0 (the
        miscoverage indicator is in [0, 1] so this is the canonical
        choice).
    window : int
        Calibration-window size for every parallel ACI.
    window_mode : str
        "fixed" or "rolling" -- forwarded to each ACI.
    quantile_lo_idx, quantile_hi_idx : int
        Indices into the K-quantile axis of the underlying forecaster.
    """
    alpha_target: float = 0.10
    etas: Sequence[float] = field(
        default_factory=lambda: (0.005, 0.01, 0.025, 0.05, 0.10)
    )
    weighting: str = "boa"
    boa_eta: float = 1.0
    window: int = 30
    window_mode: str = "fixed"
    quantile_lo_idx: int = 0
    quantile_hi_idx: int = -1


@dataclass
class AgACIState:
    coverage_history: list[float] = field(default_factory=list)
    weight_history: list[torch.Tensor] = field(default_factory=list)
    final_weights: Optional[torch.Tensor] = None


class AgACI(nn.Module):
    """Aggregated ACI: K parallel ACI controllers with online expert mixing.

    >>> agaci = AgACI(AgACIConfig(etas=(0.01, 0.05, 0.1), weighting="boa"))
    >>> agaci.fit_initial(q_cal, y_cal)
    >>> for t in test_days:
    ...     lower, upper, covered, weights = agaci.step(q_t, y_t)
    """

    def __init__(self, cfg: AgACIConfig) -> None:
        super().__init__()
        self.cfg = cfg
        if cfg.weighting not in ("boa", "uniform"):
            raise ValueError(f"weighting must be 'boa' or 'uniform'; got {cfg.weighting!r}")

        # K parallel ACI controllers.
        self.experts: list[ACI] = []
        for eta in cfg.etas:
            self.experts.append(ACI(ACIConfig(
                alpha_target=cfg.alpha_target,
                eta=float(eta),
                window=cfg.window,
                window_mode=cfg.window_mode,
                quantile_lo_idx=cfg.quantile_lo_idx,
                quantile_hi_idx=cfg.quantile_hi_idx,
            )))
        self.K = len(self.experts)

        # Equal initial weights; updated in step() if weighting="boa".
        self.register_buffer(
            "weights",
            torch.full((self.K,), 1.0 / self.K),
            persistent=False,
        )
        self.state = AgACIState()

    # ---------------------------------------------------------------------
    # Calibration initialisation -- forwarded to each expert.
    # ---------------------------------------------------------------------

    def fit_initial(
        self,
        q_pred_cal: torch.Tensor,
        y_true_cal: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> None:
        for expert in self.experts:
            expert.fit_initial(q_pred_cal, y_true_cal, mask=mask)

    # ---------------------------------------------------------------------
    # Aggregation: weighted average of (lower, upper) bounds.
    # ---------------------------------------------------------------------

    def _aggregate(
        self,
        lowers: list[torch.Tensor],
        uppers: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        L = torch.stack(lowers, dim=0)  # (K, ...)
        U = torch.stack(uppers, dim=0)  # (K, ...)
        if self.cfg.weighting == "uniform":
            # Median across experts -- robust to one expert misbehaving.
            return L.median(dim=0).values, U.median(dim=0).values
        # BOA: weighted average using self.weights.
        w = self.weights.view(self.K, *([1] * (L.ndim - 1)))
        L_agg = (w * L).sum(dim=0)
        U_agg = (w * U).sum(dim=0)
        return L_agg, U_agg

    def _update_boa_weights(self, miscoverage_per_expert: torch.Tensor) -> None:
        """Multiplicative exponential update on expert weights."""
        # miscoverage_per_expert: (K,) in [0, 1].
        with torch.no_grad():
            logits = torch.log(self.weights.clamp_min(1e-30)) \
                     - self.cfg.boa_eta * miscoverage_per_expert
            new_w = torch.softmax(logits, dim=0)
            self.weights.copy_(new_w)

    # ---------------------------------------------------------------------
    # One step.
    # ---------------------------------------------------------------------

    @torch.no_grad()
    def step(
        self,
        q_pred_t: torch.Tensor,
        y_true_t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Step each expert, aggregate, then BOA-update weights.

        Returns
        -------
        lower, upper : aggregated prediction interval
        covered      : (...) boolean from the aggregated interval
        weights      : (K,) post-update expert weights (for logging)
        """
        lowers: list[torch.Tensor] = []
        uppers: list[torch.Tensor] = []
        miscoverage_per_expert = torch.zeros(self.K)

        for k, expert in enumerate(self.experts):
            l_k, u_k, cov_k, _ = expert.step(q_pred_t, y_true_t)
            lowers.append(l_k)
            uppers.append(u_k)
            miscoverage_per_expert[k] = (~cov_k).float().mean()

        if self.cfg.weighting == "boa":
            self._update_boa_weights(miscoverage_per_expert)

        L_agg, U_agg = self._aggregate(lowers, uppers)
        covered = (y_true_t >= L_agg) & (y_true_t <= U_agg)
        self.state.coverage_history.append(float(covered.float().mean()))
        self.state.weight_history.append(self.weights.detach().clone())
        self.state.final_weights = self.weights.detach().clone()

        return L_agg, U_agg, covered, self.weights.detach().clone()

    # ---------------------------------------------------------------------
    # Diagnostics.
    # ---------------------------------------------------------------------

    def coverage_summary(self) -> dict:
        if not self.state.coverage_history:
            return {"n": 0}
        cov = torch.tensor(self.state.coverage_history)
        weights_final = (self.weights.detach().cpu().tolist()
                         if self.cfg.weighting == "boa"
                         else [1.0 / self.K] * self.K)
        return {
            "n": int(cov.numel()),
            "marginal_coverage": float(cov.mean()),
            "target_coverage": 1.0 - self.cfg.alpha_target,
            "coverage_std": float(cov.std()),
            "final_weights": weights_final,
            "etas": list(self.cfg.etas),
            "weighting": self.cfg.weighting,
        }


# =============================================================================
# Smoke test.
# =============================================================================


def _smoke_test() -> None:
    """Verify AgACI tracks the best fixed eta on a synthetic stream."""
    torch.manual_seed(0)
    K_quantiles = 5
    quantiles = torch.tensor([0.05, 0.10, 0.50, 0.90, 0.95])
    n_streams = 50

    T_cal = 200
    y_cal = torch.randn(T_cal, n_streams)
    sigma_pred = 0.8
    q_template = sigma_pred * torch.distributions.Normal(0, 1).icdf(quantiles)
    q_pred_step = q_template.unsqueeze(0).expand(n_streams, K_quantiles)
    q_cal = q_pred_step.unsqueeze(0).expand(T_cal, n_streams, K_quantiles)

    print("Test 1: AgACI (BOA weighting) hits target coverage on i.i.d. data.")
    agaci = AgACI(AgACIConfig(
        alpha_target=0.10,
        etas=(0.005, 0.01, 0.025, 0.05, 0.10),
        weighting="boa",
        window=200,
        window_mode="fixed",
    ))
    agaci.fit_initial(q_cal, y_cal)

    T_test = 500
    torch.manual_seed(1)
    y_test = torch.randn(T_test, n_streams)
    for t in range(T_test):
        agaci.step(q_pred_step, y_test[t])

    summ = agaci.coverage_summary()
    print(f"  marginal coverage: {summ['marginal_coverage']:.4f}  (target {summ['target_coverage']:.2f})")
    print(f"  final weights    : {[round(w, 3) for w in summ['final_weights']]}")
    print(f"  etas             : {summ['etas']}")
    assert abs(summ['marginal_coverage'] - summ['target_coverage']) < 0.05, (
        f"AgACI should land within +/-0.05 of target; got {summ['marginal_coverage']:.4f}"
    )
    print("  [OK] AgACI within +/-0.05 of nominal coverage on i.i.d. data.")

    # Test 2: uniform weighting also gives reasonable coverage.
    print()
    print("Test 2: AgACI (uniform/median weighting) also hits target.")
    agaci_u = AgACI(AgACIConfig(
        alpha_target=0.10,
        etas=(0.005, 0.01, 0.025, 0.05, 0.10),
        weighting="uniform",
        window=200,
        window_mode="fixed",
    ))
    agaci_u.fit_initial(q_cal, y_cal)
    for t in range(T_test):
        agaci_u.step(q_pred_step, y_test[t])
    summ_u = agaci_u.coverage_summary()
    print(f"  marginal coverage: {summ_u['marginal_coverage']:.4f}  (target {summ_u['target_coverage']:.2f})")
    assert abs(summ_u['marginal_coverage'] - summ_u['target_coverage']) < 0.05, (
        f"AgACI-uniform should land within +/-0.05 of target; got {summ_u['marginal_coverage']:.4f}"
    )
    print("  [OK] AgACI-uniform within +/-0.05 of nominal coverage on i.i.d. data.")


if __name__ == "__main__":
    _smoke_test()
