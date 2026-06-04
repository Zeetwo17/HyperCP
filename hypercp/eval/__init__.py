"""Evaluation utilities for HyperCP.

Modules
-------
- metrics     : forecast / uncertainty / resilience metrics
- coverage    : conformal-coverage diagnostics
- stats       : statistical significance protocol (Friedman + Nemenyi, paired Wilcoxon)
"""

from .metrics import (
    wmape,
    mase,
    rmse,
    smape,
    pinball_loss_per_quantile,
    crps_quantile,
    picp,
    pinaw,
    ece,
    spearman_correlation,
)

__all__ = [
    "wmape",
    "mase",
    "rmse",
    "smape",
    "pinball_loss_per_quantile",
    "crps_quantile",
    "picp",
    "pinaw",
    "ece",
    "spearman_correlation",
]
