"""
LightGBM-Quantile baseline.

Per-firm gradient-boosted quantile regression. For each (firm, horizon h,
quantile tau) we train a separate `LGBMRegressor` with
`objective='quantile', alpha=tau`. This is the standard SCM tabular
baseline and is reported in SupplyGraph (Wasi 2024).

Falls back gracefully if `lightgbm` is not installed — we use a simple
quantile linear regression so the pipeline still works on smoke runs.

Reference: Ke et al. (NeurIPS 2017), LightGBM: A Highly Efficient
Gradient Boosting Decision Tree.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

try:  # pragma: no cover -- optional dependency
    import lightgbm as lgb
    _HAS_LGB = True
except Exception:  # pragma: no cover
    lgb = None
    _HAS_LGB = False

logger = logging.getLogger(__name__)


@dataclass
class LGBMQuantileConfig:
    quantiles: tuple = (0.05, 0.10, 0.50, 0.90, 0.95)
    horizon: int = 4
    window: int = 5
    n_estimators: int = 100
    learning_rate: float = 0.05
    max_depth: int = 5
    num_leaves: int = 31


class _LinearQuantile:
    """Fallback per-quantile linear regressor by pinball-loss subgradient.

    Used only when lightgbm is unavailable. Trains by minimising mean
    pinball loss via Adam on a torch linear layer for ~200 epochs.
    """

    def __init__(self, n_features: int, quantile: float, lr: float = 1e-2):
        import torch
        from torch import nn
        self.quantile = quantile
        self.layer = nn.Linear(n_features, 1)
        self.lr = lr

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 200) -> None:
        import torch
        from torch import nn
        Xt = torch.from_numpy(X.astype(np.float32))
        yt = torch.from_numpy(y.astype(np.float32))
        opt = torch.optim.Adam(self.layer.parameters(), lr=self.lr)
        for _ in range(epochs):
            yhat = self.layer(Xt).squeeze(-1)
            diff = yt - yhat
            loss = torch.maximum(self.quantile * diff, (self.quantile - 1) * diff).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

    def predict(self, X: np.ndarray) -> np.ndarray:
        import torch
        with torch.no_grad():
            return self.layer(torch.from_numpy(X.astype(np.float32))).squeeze(-1).numpy()


class LightGBMQuantile:
    """Per-firm × per-horizon × per-quantile LightGBM forecaster.

    Hides the lightgbm / fallback choice behind a single interface.
    """

    def __init__(self, cfg: LGBMQuantileConfig | None = None) -> None:
        self.cfg = cfg or LGBMQuantileConfig()
        self.quantiles = list(self.cfg.quantiles)
        self.K = len(self.quantiles)
        # models[firm][h][k] = trained model
        self.models: list[list[list]] = []
        self.n_firms = 0
        self.n_features = 0
        if not _HAS_LGB:
            logger.warning("lightgbm not installed; falling back to linear quantile.")

    @staticmethod
    def _build_xy(features: np.ndarray, target_channel: int,
                  train_range: range, window: int, horizon: int):
        """Build (X_firm[c], y_firm[c, h]) sliding-window arrays."""
        n_firms = features.shape[0]
        valid = list(range(train_range.start, train_range.stop - window - horizon))
        n_samples = len(valid)
        n_features = window * features.shape[2]
        X_per_firm = np.zeros((n_firms, n_samples, n_features), dtype=np.float32)
        y_per_firm = np.zeros((n_firms, n_samples, horizon), dtype=np.float32)
        for i, s in enumerate(valid):
            for c in range(n_firms):
                X_per_firm[c, i, :] = features[c, s:s + window, :].reshape(-1)
                y_per_firm[c, i, :] = features[c, s + window:s + window + horizon,
                                                target_channel]
        return X_per_firm, y_per_firm

    def fit(self, features: np.ndarray, target_channel: int,
            train_range: range, seed: int = 0) -> None:
        """features: (n_firms, T, F) numpy."""
        self.n_firms = features.shape[0]
        self.n_features = self.cfg.window * features.shape[2]
        Xc, yc = self._build_xy(
            features, target_channel, train_range, self.cfg.window, self.cfg.horizon,
        )

        self.models = []
        for c in range(self.n_firms):
            firm_models = []
            for h in range(self.cfg.horizon):
                horizon_models = []
                for k, tau in enumerate(self.quantiles):
                    if _HAS_LGB:
                        m = lgb.LGBMRegressor(
                            objective="quantile",
                            alpha=tau,
                            n_estimators=self.cfg.n_estimators,
                            learning_rate=self.cfg.learning_rate,
                            max_depth=self.cfg.max_depth,
                            num_leaves=self.cfg.num_leaves,
                            random_state=seed,
                            verbosity=-1,
                        )
                        m.fit(Xc[c], yc[c, :, h])
                    else:
                        m = _LinearQuantile(self.n_features, float(tau))
                        m.fit(Xc[c], yc[c, :, h])
                    horizon_models.append(m)
                firm_models.append(horizon_models)
            self.models.append(firm_models)
            if (c + 1) % 10 == 0:
                logger.info("  LightGBM-Q fitted firm %d/%d", c + 1, self.n_firms)

    def predict(self, features: np.ndarray, target_channel: int,
                test_days: range) -> tuple[np.ndarray, np.ndarray]:
        """Returns (q_pred, y_true), shapes (T_test, n_firms, H, K) / (..., H)."""
        n_firms, T_full, n_chan = features.shape
        q_list, y_list = [], []
        for t in test_days:
            if t - self.cfg.window + 1 < 0 or t + self.cfg.horizon >= T_full:
                continue
            q_t = np.zeros((n_firms, self.cfg.horizon, self.K), dtype=np.float32)
            for c in range(n_firms):
                Xc = features[c, t - self.cfg.window + 1:t + 1, :].reshape(1, -1)
                for h in range(self.cfg.horizon):
                    for k in range(self.K):
                        q_t[c, h, k] = float(self.models[c][h][k].predict(Xc)[0])
            y_t = features[:, t + 1:t + 1 + self.cfg.horizon, target_channel]
            q_list.append(q_t)
            y_list.append(y_t)
        if not q_list:
            return (np.zeros((0, n_firms, self.cfg.horizon, self.K), dtype=np.float32),
                    np.zeros((0, n_firms, self.cfg.horizon), dtype=np.float32))
        return np.stack(q_list, axis=0), np.stack(y_list, axis=0)
