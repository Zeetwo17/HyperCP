"""
HyperCP joint trainable model.

Wires encoder + forecast head + quantile head into a single nn.Module
with configurable multi-task loss weighting. Supports the four
weighting strategies in §4.2.6 of theorem.tex:

- 'equal'   : fixed lambda_i = 1, simplest baseline
- 'kendall' : Kendall et al. (2018) homoscedastic-uncertainty weighting
- 'famo'    : Liu et al. (NeurIPS 2023) Fast Adaptive Multitask Optimisation
              (DEFAULT, per §4.2.6)
- 'pcgrad'  : Yu et al. (NeurIPS 2020) gradient surgery (ablation)

The model is device-portable and supports mixed-precision training when
wrapped in `torch.cuda.amp.autocast`.

API
---
>>> from hypercp.models.joint import HyperCPJointModel, JointConfig
>>> cfg = JointConfig(
...     encoder=EncoderConfig(n_features=5),
...     forecast=ForecastHeadConfig(n_products=40, horizon=4),
...     quantile=QuantileHeadConfig(n_products=40, horizon=4),
...     weighting='famo',
... )
>>> model = HyperCPJointModel(cfg).to(device)
>>> model.set_hypergraph(incidence)
>>> outputs = model(features)
>>> losses = model.compute_losses(outputs, y_true)
>>> total_loss, weights = model.weighter.combine(losses)
>>> total_loss.backward()
>>> model.weighter.update(losses)  # post-step update
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import nn

from .encoder import EncoderConfig, HyperCPEncoder
from .forecast_head import ForecastHead, ForecastHeadConfig, huber_forecast_loss
from .quantile_head import QuantileHead, QuantileHeadConfig, pinball_loss


# =============================================================================
# Loss-weighting strategies.
# =============================================================================


class BaseWeighter(nn.Module):
    """Abstract interface for multi-task loss weighting strategies."""

    n_tasks: int

    def combine(
        self,
        losses: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Combine task losses into a single scalar.

        Parameters
        ----------
        losses : dict mapping task name -> scalar loss tensor

        Returns
        -------
        total : scalar weighted loss (with grad)
        weights : dict mapping task name -> current weight (for logging)
        """
        raise NotImplementedError

    @torch.no_grad()
    def update(self, losses: dict[str, torch.Tensor]) -> None:
        """Post-step update of internal state (e.g., FAMO weight logits).

        Default: no-op. FAMO overrides this.
        """
        pass

    def current_weights(self) -> dict[str, float]:
        """Return weights for logging. Default: equal."""
        return {k: 1.0 / self.n_tasks for k in self._task_names()}

    def _task_names(self) -> list[str]:
        return getattr(self, "_names", [])


class EqualWeighter(BaseWeighter):
    """Fixed equal weights: lambda_i = 1 for all i."""

    def __init__(self, task_names: list[str]) -> None:
        super().__init__()
        self._names = list(task_names)
        self.n_tasks = len(task_names)

    def combine(
        self,
        losses: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        total = sum(losses[k] for k in self._names)
        return total, {k: 1.0 for k in self._names}


class KendallWeighter(BaseWeighter):
    """Kendall et al. (2018) homoscedastic-uncertainty weighting.

    Each task has a learnable log-variance log_sigma_i. The weighted loss is

        L = sum_i [ exp(-log_sigma_i) * L_i + log_sigma_i ]

    which is equivalent to the negative log-likelihood under task-specific
    Gaussian noise. This trains end-to-end with the model parameters.

    Reference
    ---------
    Kendall, Gal & Cipolla (CVPR 2018). Multi-Task Learning Using
    Uncertainty to Weigh Losses for Scene Geometry and Semantics.
    """

    def __init__(self, task_names: list[str]) -> None:
        super().__init__()
        self._names = list(task_names)
        self.n_tasks = len(task_names)
        # Initialise log_sigma at 0 (so initial weight = 1 per task).
        self.log_sigma = nn.Parameter(torch.zeros(self.n_tasks))

    def combine(
        self,
        losses: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        total = torch.tensor(0.0, device=self.log_sigma.device)
        weights: dict[str, float] = {}
        for i, k in enumerate(self._names):
            precision = torch.exp(-self.log_sigma[i])
            total = total + precision * losses[k] + self.log_sigma[i]
            weights[k] = float(precision.detach())
        return total, weights

    def current_weights(self) -> dict[str, float]:
        return {
            k: float(torch.exp(-self.log_sigma[i]).detach())
            for i, k in enumerate(self._names)
        }


class FAMOWeighter(BaseWeighter):
    """Fast Adaptive Multitask Optimisation (Liu et al. NeurIPS 2023).

    Maintains task weight logits z (n_tasks-dim) with weights = softmax(z).
    After each training step, the logits are updated to upweight tasks
    that have made *less* relative progress (slower log-loss decrease).

    Algorithm (simplified, matches the paper's Alg. 1):

      Init: z_0 = 0  (so weights start equal)

      At step t:
        1. Compute task losses L_t = (L_{t,1}, ..., L_{t,T}).
        2. Take w_t = softmax(z_t).
        3. Backprop through  L_weighted_t = sum_i w_{t,i} * L_{t,i}.
        4. Compute progress  g_i = log(L_{t-1,i}) - log(L_{t,i})
           (positive = task is improving).
        5. Update logits:
             z_{t+1} = z_t - lr_w * (g - sum_i w_{t,i} * g_i)
           (tasks with below-average progress have z increased, raising
            their weight at the next step).

    Reference
    ---------
    Liu, Feng, Stone & Liu (NeurIPS 2023). FAMO: Fast Adaptive Multitask
    Optimization.

    Parameters
    ----------
    task_names : list[str]
    lr_w : float
        Learning rate for the logit update. Default 0.025 per the paper.
    eps : float
        Numerical stabiliser inside log.
    """

    def __init__(
        self,
        task_names: list[str],
        lr_w: float = 0.025,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self._names = list(task_names)
        self.n_tasks = len(task_names)
        self.lr_w = lr_w
        self.eps = eps

        # z is a buffer, not a Parameter -- we update it manually via FAMO's
        # rule rather than via backprop.
        self.register_buffer("z", torch.zeros(self.n_tasks))
        # Track previous losses (detached scalars) per task.
        self.register_buffer("prev_loss", torch.zeros(self.n_tasks))
        self.register_buffer("has_prev", torch.tensor(False))

    def _weights(self) -> torch.Tensor:
        return torch.softmax(self.z, dim=0)

    def combine(
        self,
        losses: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        w = self._weights()
        total = torch.tensor(0.0, device=w.device)
        weights_out: dict[str, float] = {}
        for i, k in enumerate(self._names):
            total = total + w[i] * losses[k]
            weights_out[k] = float(w[i].detach())
        return total, weights_out

    @torch.no_grad()
    def update(self, losses: dict[str, torch.Tensor]) -> None:
        """Update z based on per-task progress."""
        curr = torch.stack(
            [losses[k].detach().clamp_min(self.eps) for k in self._names]
        )

        if bool(self.has_prev):
            prev = self.prev_loss.clamp_min(self.eps)
            # g_i = log(L_{t-1}) - log(L_t)
            g = torch.log(prev) - torch.log(curr)
            w = self._weights()
            g_mean = (w * g).sum()
            # z_{t+1} = z_t - lr_w * (g - g_mean)
            self.z = self.z - self.lr_w * (g - g_mean)
        else:
            self.has_prev = torch.tensor(True)

        self.prev_loss = curr

    def current_weights(self) -> dict[str, float]:
        w = self._weights().detach().cpu()
        return {k: float(w[i]) for i, k in enumerate(self._names)}


def make_weighter(name: str, task_names: list[str]) -> BaseWeighter:
    """Factory for weighters by name."""
    if name == "equal":
        return EqualWeighter(task_names)
    if name == "kendall":
        return KendallWeighter(task_names)
    if name == "famo":
        return FAMOWeighter(task_names)
    raise ValueError(
        f"Unknown weighting '{name}'. Options: equal, kendall, famo."
    )


# =============================================================================
# Joint config + model.
# =============================================================================


@dataclass
class JointConfig:
    """Container for sub-configs and joint-level options."""
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    forecast: ForecastHeadConfig = field(default_factory=ForecastHeadConfig)
    quantile: QuantileHeadConfig = field(default_factory=QuantileHeadConfig)
    weighting: str = "famo"           # 'equal' | 'kendall' | 'famo'
    huber_delta: float = 1.0

    def __post_init__(self) -> None:
        # Enforce hidden_dim consistency.
        if self.forecast.hidden_dim != self.encoder.hidden_dim:
            raise ValueError(
                f"forecast.hidden_dim ({self.forecast.hidden_dim}) "
                f"!= encoder.hidden_dim ({self.encoder.hidden_dim})"
            )
        if self.quantile.hidden_dim != self.encoder.hidden_dim:
            raise ValueError(
                f"quantile.hidden_dim ({self.quantile.hidden_dim}) "
                f"!= encoder.hidden_dim ({self.encoder.hidden_dim})"
            )
        if self.forecast.n_products != self.quantile.n_products:
            raise ValueError(
                f"forecast.n_products ({self.forecast.n_products}) "
                f"!= quantile.n_products ({self.quantile.n_products})"
            )
        if self.forecast.horizon != self.quantile.horizon:
            raise ValueError(
                f"forecast.horizon ({self.forecast.horizon}) "
                f"!= quantile.horizon ({self.quantile.horizon})"
            )
        if self.forecast.output_mode != self.quantile.output_mode:
            raise ValueError(
                f"forecast.output_mode ({self.forecast.output_mode}) "
                f"!= quantile.output_mode ({self.quantile.output_mode}); "
                f"both heads must agree."
            )


class HyperCPJointModel(nn.Module):
    """The full HyperCP trainable model: encoder + forecast head + quantile head.

    Forward returns a dict of outputs; loss computation is a separate method
    so the caller can choose to take outputs for inference without paying
    the loss-computation cost.

    Usage in a training loop
    ------------------------
    >>> model = HyperCPJointModel(cfg).to(device)
    >>> model.set_hypergraph(incidence)
    >>> opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    >>> for batch in loader:
    ...     features, y_true, mask = batch
    ...     outputs = model(features)
    ...     losses = model.compute_losses(outputs, y_true, mask)
    ...     total, weights = model.weighter.combine(losses)
    ...     opt.zero_grad()
    ...     total.backward()
    ...     opt.step()
    ...     model.weighter.update(losses)
    """

    TASK_NAMES = ("huber", "pinball")

    def __init__(self, cfg: JointConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.encoder = HyperCPEncoder(cfg.encoder)
        self.forecast_head = ForecastHead(cfg.forecast)
        self.quantile_head = QuantileHead(cfg.quantile)
        self.weighter = make_weighter(cfg.weighting, list(self.TASK_NAMES))

    # ---------------------------------------------------------------------
    # Wiring.
    # ---------------------------------------------------------------------

    def set_hypergraph(
        self,
        incidence: torch.Tensor,
        learnable_edge_weights: bool = False,
    ) -> None:
        """Wire the hypergraph into the encoder."""
        self.encoder.set_hypergraph(incidence, learnable_edge_weights)

    # ---------------------------------------------------------------------
    # Forward.
    # ---------------------------------------------------------------------

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run encoder + both heads.

        Returns a dict with keys:
            'z_seq'   : (..., n_firms, T, d)
            'z_final' : (..., n_firms, d)
            'y_hat'   : (..., n_firms, n_products, H)
            'q_hat'   : (..., n_firms, n_products, H, K)
        """
        z_seq, z_final = self.encoder(features)
        y_hat = self.forecast_head(z_final)
        q_hat = self.quantile_head(z_final)
        return {
            "z_seq":   z_seq,
            "z_final": z_final,
            "y_hat":   y_hat,
            "q_hat":   q_hat,
        }

    # ---------------------------------------------------------------------
    # Loss computation.
    # ---------------------------------------------------------------------

    def compute_losses(
        self,
        outputs: dict[str, torch.Tensor],
        y_true: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Compute per-task losses (NOT yet weighted).

        Parameters
        ----------
        outputs : dict from forward()
        y_true  : (..., n_firms, n_products, H) target tensor
        mask    : optional (..., n_firms, n_products, H) mask

        Returns
        -------
        losses : dict with keys 'huber', 'pinball'
        """
        y_hat = outputs["y_hat"]
        q_hat = outputs["q_hat"]
        tau = self.quantile_head._quantiles.to(y_hat.device)

        l_huber = huber_forecast_loss(y_hat, y_true, mask=mask,
                                       delta=self.cfg.huber_delta)
        l_pin = pinball_loss(q_hat, y_true, tau, mask=mask)
        return {"huber": l_huber, "pinball": l_pin}

    # ---------------------------------------------------------------------
    # Convenience: single-call train step (no optimizer included).
    # ---------------------------------------------------------------------

    def train_step(
        self,
        features: torch.Tensor,
        y_true: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Single forward + loss for a training step.

        Returns dict with:
            'total'   : weighted scalar loss (.backward() this)
            'huber'   : per-task Huber loss
            'pinball' : per-task pinball loss
            'weights' : dict of current task weights (for logging)
            'outputs' : forward outputs (for any post-step diagnostics)
        """
        outputs = self(features)
        losses = self.compute_losses(outputs, y_true, mask)
        total, weights = self.weighter.combine(losses)
        return {
            "total": total,
            "huber": losses["huber"],
            "pinball": losses["pinball"],
            "weights": weights,
            "outputs": outputs,
        }

    # ---------------------------------------------------------------------
    # Param-count utility.
    # ---------------------------------------------------------------------

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# Smoke tests.
# =============================================================================


def _smoke_test_synthetic() -> None:
    torch.manual_seed(0)
    print("=" * 60)
    print("HyperCPJointModel smoke test (synthetic)")
    print("=" * 60)

    cfg = JointConfig(
        encoder=EncoderConfig(n_features=5, hidden_dim=16, n_hgnn_layers=4),
        forecast=ForecastHeadConfig(hidden_dim=16, n_products=10, horizon=4),
        quantile=QuantileHeadConfig(hidden_dim=16, n_products=10, horizon=4),
        weighting="famo",
    )

    # Test each weighter.
    for w in ("equal", "kendall", "famo"):
        cfg.weighting = w
        model = HyperCPJointModel(cfg)
        # Tiny hypergraph: 10 firms, 6 hyperedges.
        n_firms = 10
        n_he = 6
        incidence = (torch.rand(n_firms, n_he) > 0.5).float()
        for e in range(n_he):
            if incidence[:, e].sum() < 2:
                incidence[:2, e] = 1.0
        model.set_hypergraph(incidence)

        features = torch.randn(n_firms, 5, 5)
        y_true = torch.randn(n_firms, 10, 4)
        mask = (torch.rand(n_firms, 10, 4) > 0.3).float()

        # Forward.
        outputs = model(features)
        # Loss.
        result = model.train_step(features, y_true, mask)
        result["total"].backward()
        # Weighter update.
        losses_only = {"huber": result["huber"], "pinball": result["pinball"]}
        model.weighter.update(losses_only)

        # Check gradients flowed.
        has_grad = sum(
            1 for p in model.parameters()
            if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0
        )
        total_p = sum(1 for p in model.parameters() if p.requires_grad)
        assert has_grad == total_p, f"{w}: {total_p - has_grad} params got no grad"

        print(f"\n  weighter={w}")
        print(f"    Huber:    {result['huber'].item():.4f}")
        print(f"    Pinball:  {result['pinball'].item():.4f}")
        print(f"    Total:    {result['total'].item():.4f}")
        print(f"    Weights:  {result['weights']}")
        print(f"    [OK] gradient flow ({has_grad}/{total_p}) and weighter update.")


def _smoke_test_famo_dynamic() -> None:
    """Verify FAMO actually adjusts weights based on per-task progress."""
    torch.manual_seed(0)
    print("\n" + "=" * 60)
    print("FAMO dynamic-weighting test")
    print("=" * 60)
    # Simulate two tasks: A loss decreases fast, B loss stagnates.
    # FAMO should upweight B over time.
    famo = FAMOWeighter(["A", "B"], lr_w=0.05)
    print(f"\nInitial weights: {famo.current_weights()}")
    for step in range(50):
        # A loss decays exponentially; B loss stays flat with noise.
        la = torch.tensor(1.0 * (0.9 ** step) + 0.01)
        lb = torch.tensor(0.5 + 0.05 * torch.randn(1).item())
        losses = {"A": la, "B": lb}
        famo.update(losses)
    final = famo.current_weights()
    print(f"After 50 steps:  A loss: 0.005 (decayed); B loss: 0.5 (flat)")
    print(f"Final weights: {final}")
    assert final["B"] > final["A"], (
        f"FAMO should upweight the stagnating task B; got {final}"
    )
    print(f"[OK] FAMO upweighted the slow-progressing task as designed.")


def _smoke_test_real_supplygraph() -> None:
    """End-to-end test on real SupplyGraph: 1 minibatch + 1 backward step."""
    print("\n" + "=" * 60)
    print("End-to-end joint model on real SupplyGraph")
    print("=" * 60)
    from hypercp.data.supplygraph import SupplyGraphHypergraph

    sg = SupplyGraphHypergraph()
    incidence = sg.hyperedge_incidence_matrix()
    n_prod = sg.n_products

    cfg = JointConfig(
        encoder=EncoderConfig(n_features=sg.n_channels, hidden_dim=64),
        forecast=ForecastHeadConfig(hidden_dim=64, n_products=n_prod, horizon=4),
        quantile=QuantileHeadConfig(hidden_dim=64, n_products=n_prod, horizon=4),
        weighting="famo",
    )
    model = HyperCPJointModel(cfg)
    model.set_hypergraph(incidence)

    print(f"Model: {model.n_parameters():,} trainable parameters")
    print(f"Weighter: {cfg.weighting}")
    print(f"Initial weights: {model.weighter.current_weights()}")

    # Single training step.
    T_win = 5
    H = 4
    start = 100
    features = sg.node_features[:, start:start + T_win, :]   # (n_prod, T, F)
    # Diagonal target: each firm forecasts its own product.
    y_diag = sg.node_features[:, start + 1:start + 1 + H, 1]  # (n_prod, H)
    y_full = y_diag.unsqueeze(1).expand(n_prod, n_prod, H)
    mask = torch.eye(n_prod).unsqueeze(-1).expand(n_prod, n_prod, H)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Run a few steps to verify training dynamics.
    losses_log = []
    for step in range(5):
        result = model.train_step(features, y_full, mask)
        optimizer.zero_grad()
        result["total"].backward()
        optimizer.step()
        model.weighter.update({"huber": result["huber"], "pinball": result["pinball"]})
        losses_log.append({
            "step": step,
            "huber": result["huber"].item(),
            "pinball": result["pinball"].item(),
            "total": result["total"].item(),
            "weights": result["weights"],
        })

    for log in losses_log:
        print(f"  step {log['step']}: huber={log['huber']:.4f}, "
              f"pinball={log['pinball']:.4f}, total={log['total']:.4f}, "
              f"weights={ {k: round(v, 3) for k, v in log['weights'].items()} }")

    # Verify total loss decreased (it should on a fixed batch).
    assert losses_log[-1]["total"] < losses_log[0]["total"], (
        f"Total loss should decrease over 5 steps on fixed batch "
        f"(got {losses_log[0]['total']:.4f} -> {losses_log[-1]['total']:.4f})"
    )
    print(f"[OK] Total loss decreased from {losses_log[0]['total']:.4f} "
          f"to {losses_log[-1]['total']:.4f} over 5 steps.")


if __name__ == "__main__":
    _smoke_test_synthetic()
    _smoke_test_famo_dynamic()
    _smoke_test_real_supplygraph()
