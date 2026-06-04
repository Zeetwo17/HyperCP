"""
CCF-HGNN baseline (SupplyGraph 2024 paper).

Wasi et al. (KDD-PSGAI 2024) proposed Cross-Channel-Fused Hypergraph
Convolutional Networks (CCF-HGNN) on SupplyGraph. We reimplement the
core idea — channel-wise feature fusion followed by a single hyperedge
convolution — and add a quantile head so it produces directly comparable
quantile predictions to HyperCP.

Differences vs HyperCP:
- single HGNN layer (vs 4 in HyperCP);
- no DeepSets permutation-invariant feature encoder;
- no FAMO multi-task balance (CCF-HGNN trains only the forecast loss);
- quantile predictions come from a small MLP head with pinball loss.

This is the closest published baseline on the same dataset and is the
right comparator for the §6 headline tables.

Reference: Wasi, Islam, Mahmud, Akram, Rahman (KDD-PSGAI 2024),
SupplyGraph: A Benchmark Dataset for Supply-Chain Planning.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

logger = logging.getLogger(__name__)


@dataclass
class CCFHGNNConfig:
    hidden_dim: int = 64
    n_hgnn_layers: int = 1
    horizon: int = 4
    window: int = 5
    quantiles: tuple = (0.05, 0.10, 0.50, 0.90, 0.95)
    dropout: float = 0.2


class _CCFFusion(nn.Module):
    """Cross-channel fusion: linear over channel dim per timestep."""

    def __init__(self, n_features: int, hidden_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(n_features, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, n_firms, T, F) -> (B, n_firms, hidden_dim)."""
        h = self.proj(x).mean(dim=2)  # pool over time
        return h


class _HypergraphConv(nn.Module):
    """One pass of hypergraph convolution.

    H_hat = sigma( D_v^-1 H W_e D_e^-1 H^T D_v^-1 X Theta )
    Here we keep W_e = I, D_e = degree(e), D_v = degree(v).
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.theta = nn.Linear(in_dim, out_dim, bias=True)
        self.act = nn.GELU()

    def forward(self, X: torch.Tensor, incidence: torch.Tensor) -> torch.Tensor:
        """X: (B, V, F) ; incidence: (V, E)."""
        Dv = incidence.sum(dim=1).clamp(min=1.0)  # (V,)
        De = incidence.sum(dim=0).clamp(min=1.0)  # (E,)
        Hn = incidence / Dv.unsqueeze(1)            # (V, E)
        # Edge messages: m_e = H^T X / De
        m_e = (incidence.t() @ X.transpose(0, 1).reshape(X.size(1), -1)) / De.unsqueeze(1)
        m_e = m_e.view(incidence.size(1), X.size(0), -1).transpose(0, 1)  # (B, E, F)
        # Node update: X' = H Hn m_e
        x_new = (incidence @ m_e.transpose(0, 1).reshape(incidence.size(1), -1)) \
            .view(incidence.size(0), X.size(0), -1).transpose(0, 1)  # (B, V, F)
        return self.act(self.theta(x_new + X))


class _CCFHGNNNet(nn.Module):
    def __init__(self, n_features: int, cfg: CCFHGNNConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.fuse = _CCFFusion(n_features, cfg.hidden_dim)
        self.layers = nn.ModuleList([
            _HypergraphConv(cfg.hidden_dim, cfg.hidden_dim)
            for _ in range(cfg.n_hgnn_layers)
        ])
        self.dropout = nn.Dropout(cfg.dropout)
        self.K = len(cfg.quantiles)
        self.head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.horizon * self.K),
        )

    def forward(self, x: torch.Tensor, incidence: torch.Tensor) -> torch.Tensor:
        """x: (B, n_firms, T, F) ; incidence: (n_firms, n_he).

        Returns q: (B, n_firms, H, K)
        """
        h = self.fuse(x)  # (B, n_firms, hidden_dim)
        for L in self.layers:
            h = L(h, incidence)
            h = self.dropout(h)
        q = self.head(h)                                     # (B, n_firms, H*K)
        return q.view(q.size(0), q.size(1), self.cfg.horizon, self.K)


class CCFHGNNBaseline:
    def __init__(self, n_features: int, cfg: CCFHGNNConfig | None = None,
                 device: str | torch.device = "cpu") -> None:
        self.cfg = cfg or CCFHGNNConfig()
        self.device = torch.device(device)
        self.net = _CCFHGNNNet(n_features, self.cfg).to(self.device)
        self.K = len(self.cfg.quantiles)
        self.quantiles = list(self.cfg.quantiles)

    def _pinball(self, q_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        tau = torch.tensor(self.quantiles, device=q_pred.device).view(1, 1, 1, -1)
        diff = y_true.unsqueeze(-1) - q_pred
        return torch.maximum(tau * diff, (tau - 1.0) * diff).mean()

    def fit(self, features: np.ndarray, incidence: np.ndarray, target_channel: int,
            train_range: range, epochs: int = 30, lr: float = 1e-3,
            batch_size: int = 16, seed: int = 0) -> None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        x_all = torch.tensor(features, dtype=torch.float32, device=self.device)
        H = torch.tensor(incidence, dtype=torch.float32, device=self.device)
        n_firms, T_full, _ = x_all.shape
        valid = list(range(train_range.start,
                           train_range.stop - self.cfg.window - self.cfg.horizon))
        if not valid:
            raise RuntimeError("train range too short for CCF-HGNN")
        opt = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=1e-4)
        rng = np.random.default_rng(seed)
        for epoch in range(epochs):
            self.net.train()
            order = rng.permutation(len(valid))
            running = 0.0
            n_batches = max(1, len(order) // batch_size)
            for b in range(n_batches):
                starts = [valid[i] for i in order[b * batch_size:(b + 1) * batch_size]]
                Xb = torch.stack([x_all[:, s:s + self.cfg.window, :] for s in starts],
                                 dim=0)
                Yb = torch.stack([x_all[:, s + self.cfg.window:
                                          s + self.cfg.window + self.cfg.horizon,
                                          target_channel] for s in starts], dim=0)
                q = self.net(Xb, H)
                loss = self._pinball(q, Yb)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
                running += float(loss)
            if (epoch + 1) % 10 == 0:
                logger.info("  CCF-HGNN epoch %d  loss=%.4f", epoch + 1,
                            running / n_batches)

    def predict(self, features: np.ndarray, incidence: np.ndarray,
                target_channel: int, test_days: range) -> tuple[np.ndarray, np.ndarray]:
        self.net.eval()
        x_all = torch.tensor(features, dtype=torch.float32, device=self.device)
        H = torch.tensor(incidence, dtype=torch.float32, device=self.device)
        n_firms, T_full, _ = x_all.shape
        q_list, y_list = [], []
        with torch.no_grad():
            for t in test_days:
                if t - self.cfg.window + 1 < 0 or t + self.cfg.horizon >= T_full:
                    continue
                X = x_all[:, t - self.cfg.window + 1:t + 1, :].unsqueeze(0)
                q = self.net(X, H)[0]  # (n_firms, H, K)
                y = x_all[:, t + 1:t + 1 + self.cfg.horizon, target_channel]
                q_list.append(q.cpu().numpy())
                y_list.append(y.cpu().numpy())
        if not q_list:
            return (np.zeros((0, n_firms, self.cfg.horizon, self.K), dtype=np.float32),
                    np.zeros((0, n_firms, self.cfg.horizon), dtype=np.float32))
        return np.stack(q_list, axis=0), np.stack(y_list, axis=0)
