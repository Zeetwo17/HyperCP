"""
TFT-style baseline.

Faithful but lightweight Temporal Fusion Transformer (Lim et al. 2021,
IJF). We strip the variable-selection / static-covariate machinery and
keep:

- a windowed embedding,
- a temporal multi-head self-attention block,
- a per-firm quantile head producing K quantile outputs.

Why a lightweight reimplementation:
The official `pytorch-forecasting` TFT (~25k LOC) requires a Pandas /
TimeSeriesDataSet pipeline and is non-trivial to swap into our windowed
loaders. The simplified variant gives a fair baseline at ~5k params per
firm and reproduces TFT's two defining features for ICDM:
1) attention-based temporal mixing,
2) quantile-regression loss (pinball).

Reference: Lim, Arik, Loeff, Pfister (IJF 2021), Temporal Fusion
Transformers for Interpretable Multi-horizon Time Series Forecasting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

logger = logging.getLogger(__name__)


@dataclass
class TFTConfig:
    hidden_dim: int = 64
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1
    quantiles: tuple = (0.05, 0.10, 0.50, 0.90, 0.95)
    horizon: int = 4
    window: int = 5


class _TemporalAttentionBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim),
        )
        self.ln2 = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, _ = self.attn(x, x, x, need_weights=False)
        x = self.ln1(x + self.drop(a))
        f = self.ff(x)
        return self.ln2(x + self.drop(f))


class _TFTNet(nn.Module):
    def __init__(self, n_features: int, cfg: TFTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Linear(n_features, cfg.hidden_dim)
        self.pos = nn.Parameter(torch.randn(cfg.window, cfg.hidden_dim) * 0.02)
        self.blocks = nn.ModuleList([
            _TemporalAttentionBlock(cfg.hidden_dim, cfg.n_heads, cfg.dropout)
            for _ in range(cfg.n_layers)
        ])
        self.head = nn.Linear(cfg.hidden_dim, cfg.horizon * len(cfg.quantiles))
        self.K = len(cfg.quantiles)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, F) -> q: (B, H, K)."""
        h = self.embed(x) + self.pos.unsqueeze(0)
        for b in self.blocks:
            h = b(h)
        last = h[:, -1, :]
        q = self.head(last)
        return q.view(-1, self.cfg.horizon, self.K)


class TFTBaseline:
    """Train one shared TFT for all firms (firm-agnostic temporal features).

    A separate forward pass is made per firm at inference time.
    """

    def __init__(self, n_features: int, cfg: TFTConfig | None = None,
                 device: str | torch.device = "cpu") -> None:
        self.cfg = cfg or TFTConfig()
        self.device = torch.device(device)
        self.net = _TFTNet(n_features, self.cfg).to(self.device)
        self.K = len(self.cfg.quantiles)
        self.quantiles = list(self.cfg.quantiles)

    def _pinball(self, q_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """q_pred: (B, H, K)  y_true: (B, H)."""
        tau = torch.tensor(self.quantiles, device=q_pred.device).view(1, 1, -1)
        diff = y_true.unsqueeze(-1) - q_pred
        losses = torch.maximum(tau * diff, (tau - 1.0) * diff)
        return losses.mean()

    def fit(self, features: np.ndarray, target_channel: int,
            train_range: range, epochs: int = 30,
            lr: float = 1e-3, batch_size: int = 64,
            seed: int = 0) -> None:
        """features: (n_firms, T, F)."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        x_all = torch.tensor(features, dtype=torch.float32, device=self.device)
        n_firms, T_full, _ = x_all.shape
        rng = np.random.default_rng(seed)

        valid = list(range(train_range.start,
                           train_range.stop - self.cfg.window - self.cfg.horizon))
        if not valid:
            raise RuntimeError("Train range too short for TFT.")

        opt = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=1e-4)
        for epoch in range(epochs):
            self.net.train()
            order = rng.permutation(len(valid))
            running = 0.0
            n_batches = 0
            for b_idx in range(0, len(order), batch_size):
                batch_idxs = order[b_idx:b_idx + batch_size]
                feats, tgts = [], []
                for j in batch_idxs:
                    s = valid[j]
                    for c in range(n_firms):
                        feats.append(x_all[c, s:s + self.cfg.window, :])
                        tgts.append(x_all[c, s + self.cfg.window:
                                          s + self.cfg.window + self.cfg.horizon,
                                          target_channel])
                if not feats:
                    continue
                X = torch.stack(feats, dim=0)
                Y = torch.stack(tgts, dim=0)
                q = self.net(X)
                loss = self._pinball(q, Y)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
                running += float(loss)
                n_batches += 1
            if (epoch + 1) % 10 == 0 and n_batches > 0:
                logger.info("  TFT epoch %d  loss=%.4f", epoch + 1,
                            running / n_batches)

    def predict(self, features: np.ndarray, target_channel: int,
                test_days: range) -> tuple[np.ndarray, np.ndarray]:
        """Returns (q_pred, y_true) with shape (T_test, n_firms, H, K) / (..., H)."""
        self.net.eval()
        x_all = torch.tensor(features, dtype=torch.float32, device=self.device)
        n_firms, T_full, _ = x_all.shape
        q_list, y_list = [], []
        with torch.no_grad():
            for t in test_days:
                if t - self.cfg.window + 1 < 0 or t + self.cfg.horizon >= T_full:
                    continue
                X = x_all[:, t - self.cfg.window + 1:t + 1, :]  # (n_firms, T, F)
                q = self.net(X)  # (n_firms, H, K)
                y = x_all[:, t + 1:t + 1 + self.cfg.horizon, target_channel]
                q_list.append(q.cpu().numpy())
                y_list.append(y.cpu().numpy())
        if not q_list:
            return (np.zeros((0, n_firms, self.cfg.horizon, self.K), dtype=np.float32),
                    np.zeros((0, n_firms, self.cfg.horizon), dtype=np.float32))
        return np.stack(q_list, axis=0), np.stack(y_list, axis=0)
