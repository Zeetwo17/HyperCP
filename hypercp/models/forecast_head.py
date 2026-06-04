"""
Forecast head: point predictions per (firm, product, horizon).

Implements Eq. (4) in theorem.tex (Section 4.1):
    y_hat_{c,p,h} = MLP_A( [z_hat_c, pos(p), pos(h)] )

Trained with Huber loss (delta=1.0 after z-score normalisation), which is
robust to FMCG demand spikes that Han 2024 noted inflated squared-error
skewness.

Inputs (typical use)
--------------------
- z_seq from the encoder: (..., n_firms, T, d) or z_final: (..., n_firms, d)
  The forecast head can use either; default is z_final (post-TCN summary).
- target product indices (which products the firm produces; for SupplyGraph
  the product index = firm index since we treat each product as a firm).
- target horizon indices (h = 1..H).

Outputs
-------
- y_hat per (firm, product, horizon): shape (..., n_firms, n_products, H)
  where n_products is the number of product types the firm forecasts and
  H is the forecast horizon length.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ForecastHeadConfig:
    """Configuration for the forecast head.

    Output modes
    ------------
    output_mode : str
        - "cross_product" (default): compute y_hat for every
          (firm, product, horizon) tuple. Output shape
          (..., n_firms, n_products, horizon). Suitable for small
          benchmarks like SupplyGraph (n_firms = n_products = 40);
          memory grows as O(n_firms * n_products * H).
        - "diagonal":  assume firm c forecasts its own product c
          (i.e. n_firms == n_products and we only need the diagonal
          of the cross-product head). Output shape (..., n_firms,
          horizon). Memory is O(n_firms * H), enabling scaling to
          datasets like M5 (n_firms in the thousands).
    """
    hidden_dim: int = 64          # must match encoder.hidden_dim
    n_products: int = 40          # number of distinct products to predict
    horizon: int = 4              # H: 1, 2, ..., horizon (weeks ahead)
    mlp_hidden_mult: int = 2
    dropout: float = 0.1
    activation: str = "gelu"
    huber_delta: float = 1.0      # robust to spikes (Han 2024 motivation)
    output_mode: str = "cross_product"  # 'cross_product' or 'diagonal'


def _act(name: str) -> nn.Module:
    return {"gelu": nn.GELU(), "relu": nn.ReLU()}[name.lower()]


class ForecastHead(nn.Module):
    """Per (firm, product, horizon) point-forecast head.

    Architecture
    ------------
    z_c              -> (d,)
    pos_product(p)   -> (d,)   learnable embedding per product
    pos_horizon(h)   -> (d,)   learnable embedding per horizon step
    concat           -> (3d,)
    MLP_A            -> R (scalar forecast)

    For efficiency, the forward pass vectorises over (firms, products,
    horizons) rather than looping. The output is (..., n_firms, n_products,
    horizon).
    """

    def __init__(self, cfg: ForecastHeadConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.hidden_dim

        self.product_embed = nn.Embedding(cfg.n_products, d)
        self.horizon_embed = nn.Embedding(cfg.horizon, d)

        hidden = d * cfg.mlp_hidden_mult
        self.mlp = nn.Sequential(
            nn.Linear(3 * d, hidden),
            _act(cfg.activation),
            nn.Dropout(cfg.dropout),
            nn.Linear(hidden, hidden),
            _act(cfg.activation),
            nn.Dropout(cfg.dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Compute point forecasts.

        Output shape depends on `cfg.output_mode`:
        - "cross_product": (..., n_firms, n_products, horizon)
        - "diagonal":      (..., n_firms, horizon), with firm c queried
                           against its own product embedding c. Requires
                           n_firms = n_products at call time.

        Parameters
        ----------
        z : (..., n_firms, d)
            Per-firm embedding (typically z_final from the encoder).
        """
        d = self.cfg.hidden_dim
        n_prod = self.cfg.n_products
        H = self.cfg.horizon

        if z.size(-1) != d:
            raise ValueError(
                f"ForecastHead expected last dim {d}, got {z.size(-1)}"
            )

        device = z.device
        hor_idx = torch.arange(H, device=device)
        hor_emb = self.horizon_embed(hor_idx)         # (H, d)
        lead_shape = z.shape[:-1]                     # (..., n_firms)

        if self.cfg.output_mode == "diagonal":
            n_firms_q = lead_shape[-1]
            if n_firms_q > n_prod:
                raise ValueError(
                    f"diagonal mode requires n_firms ({n_firms_q}) <= "
                    f"n_products ({n_prod})."
                )
            # Each firm c queries its own product embedding c.
            prod_idx = torch.arange(n_firms_q, device=device)
            prod_emb_diag = self.product_embed(prod_idx)  # (n_firms, d)
            # Reshape for broadcasting over leading dims + horizon.
            # z: (..., n_firms, d) -> (..., n_firms, 1, d)
            z_b = z.unsqueeze(-2).expand(*lead_shape, H, d)
            prod_b = prod_emb_diag.view(
                *([1] * (len(lead_shape) - 1)), n_firms_q, 1, d
            ).expand(*lead_shape, H, d)
            hor_b = hor_emb.view(
                *([1] * len(lead_shape)), H, d
            ).expand(*lead_shape, H, d)
            inp = torch.cat([z_b, prod_b, hor_b], dim=-1)
            y_hat = self.mlp(inp).squeeze(-1)           # (..., n_firms, H)
            return y_hat

        # ---- cross_product mode (default) ----
        prod_idx = torch.arange(n_prod, device=device)
        prod_emb = self.product_embed(prod_idx)       # (n_prod, d)

        # Broadcast over firms, products, horizons.
        z_b = z.unsqueeze(-2).unsqueeze(-2)
        prod_b = prod_emb.view(*([1] * len(lead_shape)), n_prod, 1, d)
        prod_b = prod_b.expand(*lead_shape, n_prod, H, d)
        hor_b = hor_emb.view(*([1] * len(lead_shape)), 1, H, d)
        hor_b = hor_b.expand(*lead_shape, n_prod, H, d)
        z_expanded = z_b.expand(*lead_shape, n_prod, H, d)
        inp = torch.cat([z_expanded, prod_b, hor_b], dim=-1)
        y_hat = self.mlp(inp).squeeze(-1)
        return y_hat


def huber_forecast_loss(
    y_hat: torch.Tensor,
    y_true: torch.Tensor,
    mask: torch.Tensor | None = None,
    delta: float = 1.0,
) -> torch.Tensor:
    """Huber loss for the forecast head (robust to FMCG spikes).

    Parameters
    ----------
    y_hat   : (..., n_firms, n_products, H)  predicted point forecasts
    y_true  : (..., n_firms, n_products, H)  observed targets
    mask    : optional same-shape mask (1 = include, 0 = exclude). Used to
              mask out (firm, product) pairs that the firm doesn't handle
              (incidence-based) and to mask out empty horizons.
    delta   : Huber loss threshold; default 1.0.

    Returns
    -------
    Scalar loss = mean over included entries.
    """
    if y_hat.shape != y_true.shape:
        raise ValueError(
            f"shape mismatch: y_hat {tuple(y_hat.shape)} vs y_true {tuple(y_true.shape)}"
        )
    diff = y_hat - y_true
    abs_d = diff.abs()
    quad = torch.minimum(abs_d, torch.tensor(delta, device=diff.device))
    lin = abs_d - quad
    elem = 0.5 * quad.pow(2) + delta * lin
    if mask is not None:
        elem = elem * mask
        denom = mask.sum().clamp_min(1.0)
        return elem.sum() / denom
    return elem.mean()


# =============================================================================
# Smoke test.
# =============================================================================


def _smoke_test() -> None:
    torch.manual_seed(0)

    cfg = ForecastHeadConfig(hidden_dim=16, n_products=10, horizon=4)
    head = ForecastHead(cfg)
    print(f"ForecastHead params: {sum(p.numel() for p in head.parameters()):,}")

    # Unbatched: z of shape (n_firms, d).
    n_firms = 10
    z = torch.randn(n_firms, cfg.hidden_dim)
    y_hat = head(z)
    print(f"Unbatched: z {tuple(z.shape)} -> y_hat {tuple(y_hat.shape)}")
    assert y_hat.shape == (n_firms, cfg.n_products, cfg.horizon)

    # Batched: z of shape (B, n_firms, d).
    B = 4
    z_b = torch.randn(B, n_firms, cfg.hidden_dim)
    y_hat_b = head(z_b)
    print(f"Batched (B={B}): z {tuple(z_b.shape)} -> y_hat {tuple(y_hat_b.shape)}")
    assert y_hat_b.shape == (B, n_firms, cfg.n_products, cfg.horizon)

    # Loss + gradient flow.
    y_true = torch.randn_like(y_hat)
    mask = (torch.rand_like(y_hat) > 0.3).float()
    loss = huber_forecast_loss(y_hat, y_true, mask=mask, delta=cfg.huber_delta)
    loss.backward()
    has_grad = sum(1 for p in head.parameters() if p.grad is not None
                   and p.grad.abs().sum() > 0)
    total = sum(1 for p in head.parameters() if p.requires_grad)
    print(f"Loss = {loss.item():.4f}; gradient flow {has_grad}/{total} params.")
    assert has_grad == total
    print("  [OK] ForecastHead smoke test passed.")


if __name__ == "__main__":
    _smoke_test()
