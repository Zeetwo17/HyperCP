"""
Quantile head: multi-quantile predictions per (firm, product, horizon, tau).

Implements Eq. (5) in theorem.tex (Section 4.1):
    q_hat_{c,p,h,tau} = MLP_B( [z_hat_c, pos(p), pos(h), pos(tau)] )

Trained with pinball loss summed over the quantile set T = {0.05, 0.10,
0.50, 0.90, 0.95}. Quantile outputs feed:
- Conformity-score computation (hyperedge-level CQR; §4.2 of theorem.tex)
- ACI-based prediction-set construction (§4.3)
- Empirical predictive distribution F_hat for the resilience functional
  (§4.4)

Quantile-monotonicity (q_tau_lo <= q_tau_hi for tau_lo < tau_hi) is NOT
enforced architecturally; reviewers will ask, so we provide a post-hoc
sorting helper. Sorting the predicted quantiles before computing pinball
loss is also a recognised stabilisation trick (Romano et al. 2019); we
ablate it in §6.5.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn


@dataclass
class QuantileHeadConfig:
    """Configuration for the quantile head.

    Output modes
    ------------
    output_mode : str
        - "cross_product" (default): (..., n_firms, n_products, H, K)
        - "diagonal":  (..., n_firms, H, K) -- firm c queries its own
          product embedding c. Required for M5-scale runs to fit memory.
    """
    hidden_dim: int = 64
    n_products: int = 40
    horizon: int = 4
    quantiles: tuple[float, ...] = (0.05, 0.10, 0.50, 0.90, 0.95)
    mlp_hidden_mult: int = 2
    dropout: float = 0.1
    activation: str = "gelu"
    sort_quantiles: bool = True   # post-hoc monotonicity enforcement
    output_mode: str = "cross_product"  # 'cross_product' or 'diagonal'


def _act(name: str) -> nn.Module:
    return {"gelu": nn.GELU(), "relu": nn.ReLU()}[name.lower()]


class QuantileHead(nn.Module):
    """Per (firm, product, horizon, tau) quantile head.

    The quantile level tau is encoded as a scalar feature (via a small
    embedding network) rather than a discrete embedding, so the head can
    in principle produce quantiles at any tau in (0, 1) — useful for the
    Conformal PID variant in §4.3 which adjusts tau on the fly.

    Architecture
    ------------
    z_c, pos(p), pos(h), tau_embed(tau) -> concat -> MLP_B -> R
    """

    def __init__(self, cfg: QuantileHeadConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.hidden_dim

        self.product_embed = nn.Embedding(cfg.n_products, d)
        self.horizon_embed = nn.Embedding(cfg.horizon, d)

        # tau is a continuous scalar -> small MLP to d-dim embedding.
        self.tau_proj = nn.Sequential(
            nn.Linear(1, d),
            _act(cfg.activation),
            nn.Linear(d, d),
        )

        hidden = d * cfg.mlp_hidden_mult
        self.mlp = nn.Sequential(
            nn.Linear(4 * d, hidden),
            _act(cfg.activation),
            nn.Dropout(cfg.dropout),
            nn.Linear(hidden, hidden),
            _act(cfg.activation),
            nn.Dropout(cfg.dropout),
            nn.Linear(hidden, 1),
        )

        # Register quantile tensor as buffer (moves with .to(device)).
        self.register_buffer(
            "_quantiles",
            torch.tensor(cfg.quantiles, dtype=torch.float32),
            persistent=False,
        )

    @property
    def n_quantiles(self) -> int:
        return len(self.cfg.quantiles)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Predict quantiles.

        Output shape depends on `cfg.output_mode`:
        - "cross_product": (..., n_firms, n_products, H, K)
        - "diagonal":      (..., n_firms, H, K)
        """
        d = self.cfg.hidden_dim
        n_prod = self.cfg.n_products
        H = self.cfg.horizon
        K = self.n_quantiles

        if z.size(-1) != d:
            raise ValueError(
                f"QuantileHead expected last dim {d}, got {z.size(-1)}"
            )

        device = z.device
        hor_emb = self.horizon_embed(torch.arange(H, device=device))        # (H, d)
        tau = self._quantiles.to(device).view(K, 1)                         # (K, 1)
        tau_emb = self.tau_proj(tau)                                        # (K, d)
        lead_shape = z.shape[:-1]                                           # (..., n_firms)

        if self.cfg.output_mode == "diagonal":
            n_firms_q = lead_shape[-1]
            if n_firms_q > n_prod:
                raise ValueError(
                    f"diagonal mode requires n_firms ({n_firms_q}) <= "
                    f"n_products ({n_prod})."
                )
            prod_emb_diag = self.product_embed(
                torch.arange(n_firms_q, device=device)
            )  # (n_firms, d)
            # Target shape: (..., n_firms, H, K, 4d).
            z_b = z.view(*lead_shape, 1, 1, d).expand(*lead_shape, H, K, d)
            prod_b = prod_emb_diag.view(
                *([1] * (len(lead_shape) - 1)), n_firms_q, 1, 1, d
            ).expand(*lead_shape, H, K, d)
            hor_b = hor_emb.view(
                *([1] * len(lead_shape)), H, 1, d
            ).expand(*lead_shape, H, K, d)
            tau_b = tau_emb.view(
                *([1] * len(lead_shape)), 1, K, d
            ).expand(*lead_shape, H, K, d)
            inp = torch.cat([z_b, prod_b, hor_b, tau_b], dim=-1)
            q_hat = self.mlp(inp).squeeze(-1)  # (..., n_firms, H, K)
            if self.cfg.sort_quantiles:
                q_hat, _ = torch.sort(q_hat, dim=-1)
            return q_hat

        # ---- cross_product mode (default) ----
        prod_emb = self.product_embed(torch.arange(n_prod, device=device))  # (n_prod, d)
        z_b = z.view(*lead_shape, 1, 1, 1, d).expand(
            *lead_shape, n_prod, H, K, d
        )
        prod_b = prod_emb.view(*([1] * len(lead_shape)), n_prod, 1, 1, d) \
                         .expand(*lead_shape, n_prod, H, K, d)
        hor_b = hor_emb.view(*([1] * len(lead_shape)), 1, H, 1, d) \
                       .expand(*lead_shape, n_prod, H, K, d)
        tau_b = tau_emb.view(*([1] * len(lead_shape)), 1, 1, K, d) \
                       .expand(*lead_shape, n_prod, H, K, d)
        inp = torch.cat([z_b, prod_b, hor_b, tau_b], dim=-1)
        q_hat = self.mlp(inp).squeeze(-1)  # (..., n_firms, n_prod, H, K)
        if self.cfg.sort_quantiles:
            q_hat, _ = torch.sort(q_hat, dim=-1)
        return q_hat


def pinball_loss(
    q_hat: torch.Tensor,
    y_true: torch.Tensor,
    quantiles: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pinball (asymmetric) quantile loss, summed over quantiles.

    L_pinball = mean over (firm, product, horizon) of
                sum_tau max( tau*(y - q_tau), (tau-1)*(y - q_tau) )

    Parameters
    ----------
    q_hat   : (..., n_firms, n_products, H, K)
    y_true  : (..., n_firms, n_products, H)  -- broadcast across K
    quantiles : (K,) tensor of tau values in (0,1)
    mask : optional (..., n_firms, n_products, H) mask
    """
    if q_hat.shape[:-1] != y_true.shape:
        raise ValueError(
            f"shape mismatch: q_hat {tuple(q_hat.shape)} vs y_true {tuple(y_true.shape)}"
        )
    K = q_hat.size(-1)
    if quantiles.numel() != K:
        raise ValueError(
            f"quantiles has {quantiles.numel()} entries; q_hat last dim is {K}"
        )

    y_b = y_true.unsqueeze(-1)              # (..., 1)
    diff = y_b - q_hat                      # (..., K)
    tau = quantiles.to(diff.device).view(*([1] * (diff.ndim - 1)), K)
    loss_per_q = torch.maximum(tau * diff, (tau - 1.0) * diff)  # (..., K)
    loss_per_pt = loss_per_q.sum(dim=-1)    # sum over quantiles -> (...,)

    if mask is not None:
        loss_per_pt = loss_per_pt * mask
        denom = mask.sum().clamp_min(1.0)
        return loss_per_pt.sum() / denom
    return loss_per_pt.mean()


# =============================================================================
# Smoke test.
# =============================================================================


def _smoke_test() -> None:
    torch.manual_seed(0)

    cfg = QuantileHeadConfig(
        hidden_dim=16,
        n_products=10,
        horizon=4,
        quantiles=(0.05, 0.10, 0.50, 0.90, 0.95),
    )
    head = QuantileHead(cfg)
    print(f"QuantileHead params: {sum(p.numel() for p in head.parameters()):,}")
    print(f"Quantiles: {cfg.quantiles} (K={head.n_quantiles})")

    # Unbatched.
    n_firms = 10
    z = torch.randn(n_firms, cfg.hidden_dim)
    q_hat = head(z)
    print(f"Unbatched: z {tuple(z.shape)} -> q_hat {tuple(q_hat.shape)}")
    assert q_hat.shape == (n_firms, cfg.n_products, cfg.horizon, head.n_quantiles)

    # Monotonicity check (sort_quantiles enabled by default).
    sorted_diffs = q_hat[..., 1:] - q_hat[..., :-1]
    assert (sorted_diffs >= -1e-6).all(), "Quantile crossing detected!"
    print("  [OK] quantiles are monotonically non-decreasing in tau.")

    # Batched.
    B = 4
    z_b = torch.randn(B, n_firms, cfg.hidden_dim)
    q_hat_b = head(z_b)
    print(f"Batched (B={B}): z {tuple(z_b.shape)} -> q_hat {tuple(q_hat_b.shape)}")
    assert q_hat_b.shape == (B, n_firms, cfg.n_products, cfg.horizon, head.n_quantiles)

    # Loss + gradient flow.
    y_true = torch.randn(n_firms, cfg.n_products, cfg.horizon)
    mask = (torch.rand(n_firms, cfg.n_products, cfg.horizon) > 0.3).float()
    quantiles_t = torch.tensor(cfg.quantiles)
    loss = pinball_loss(q_hat, y_true, quantiles_t, mask=mask)
    loss.backward()
    has_grad = sum(1 for p in head.parameters() if p.grad is not None
                   and p.grad.abs().sum() > 0)
    total = sum(1 for p in head.parameters() if p.requires_grad)
    print(f"Pinball loss = {loss.item():.4f}; gradient flow {has_grad}/{total} params.")
    assert has_grad == total
    print("  [OK] QuantileHead smoke test passed.")


if __name__ == "__main__":
    _smoke_test()
