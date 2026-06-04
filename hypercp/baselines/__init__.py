"""Baseline forecasters for HyperCP comparison tables.

All baselines share a common interface:

    bl = BaselineXyz(quantiles=[...])
    bl.fit(history_train, history_cal)          # numpy arrays (n_firms, T)
    q_pred = bl.predict(history_test, horizon)  # (T_test, n_firms, H, K)

This makes them drop-in replacements for HyperCP's quantile head when
running the gate scripts in baseline mode.

Baselines
---------
- TFTBaseline           : simplified temporal-fusion-transformer-style net
- LightGBMQuantile      : per-firm gradient-boosted quantile regressors
- CCFHGNNBaseline       : SupplyGraph paper's CCF-HGNN reimplementation
"""

from .tft import TFTBaseline, TFTConfig
from .lgbm_quantile import LightGBMQuantile, LGBMQuantileConfig
from .ccf_hgnn import CCFHGNNBaseline, CCFHGNNConfig

__all__ = [
    "TFTBaseline",
    "TFTConfig",
    "LightGBMQuantile",
    "LGBMQuantileConfig",
    "CCFHGNNBaseline",
    "CCFHGNNConfig",
]
