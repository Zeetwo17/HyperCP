"""Block bootstrap for time-series-aware uncertainty quantification.

SupplyGraph test period has only 23 days. Random-seed standard deviations
do not quantify sampling uncertainty of the temporal sequence (Reviewer #2).
This module implements a non-overlapping block bootstrap over temporal
test units, respecting dependence within forecast-horizon blocks.

Block length: default = forecast horizon H (4 days for SupplyGraph).
With 23 test days and block_size=4, this yields ~5 blocks per resample.

Reference:
    Kunsch, H.R. (1989). The Jackknife and the Bootstrap for General
    Stationary Observations. Annals of Statistics.
"""

import numpy as np
from hypercp.eval.metrics import picp, pinaw, _to_numpy

def block_bootstrap_ci(metric_fn, y_true, lower, upper, block_size=4, n_bootstrap=1000, confidence=0.95, seed=42):
    """
    Compute block bootstrap confidence interval for a given metric.
    y_true, lower, upper have shape (T, ...), where T is the time dimension.
    """
    y_true = _to_numpy(y_true)
    lower = _to_numpy(lower)
    upper = _to_numpy(upper)
    
    T = y_true.shape[0]
    rng = np.random.default_rng(seed)
    
    n_full_blocks = T // block_size
    remainder = T % block_size
    
    blocks = []
    for i in range(n_full_blocks):
        blocks.append(np.arange(i * block_size, (i + 1) * block_size))
    if remainder > 0:
        blocks.append(np.arange(n_full_blocks * block_size, T))
    
    n_blocks = len(blocks)
    
    metric_vals = []
    for _ in range(n_bootstrap):
        block_idxs = rng.choice(n_blocks, size=n_blocks, replace=True)
        time_idxs = np.concatenate([blocks[i] for i in block_idxs])
        time_idxs = time_idxs[:T]
        
        y_resampled = y_true[time_idxs]
        lower_resampled = lower[time_idxs]
        upper_resampled = upper[time_idxs]
        
        val = metric_fn(y_resampled, lower_resampled, upper_resampled)
        metric_vals.append(val)
        
    metric_vals = np.array(metric_vals)
    point_estimate = float(metric_fn(y_true, lower, upper))
    
    alpha = 1.0 - confidence
    lower_p = (alpha / 2) * 100
    upper_p = (1.0 - alpha / 2) * 100
    
    ci_lower = float(np.percentile(metric_vals, lower_p))
    ci_upper = float(np.percentile(metric_vals, upper_p))
    bootstrap_std = float(np.std(metric_vals))
    
    return {
        "point_estimate": point_estimate,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "bootstrap_std": bootstrap_std,
        "n_blocks": n_blocks,
        "block_size": block_size
    }

def bootstrap_picp_ci(y_true, lower, upper, block_size=4, n_bootstrap=1000, confidence=0.95, seed=42):
    """Wrapper that calls block_bootstrap_ci with picp as the metric."""
    return block_bootstrap_ci(picp, y_true, lower, upper, block_size, n_bootstrap, confidence, seed)

def bootstrap_pinaw_ci(y_true, lower, upper, block_size=4, n_bootstrap=1000, confidence=0.95, seed=42):
    """Wrapper that calls block_bootstrap_ci with pinaw as the metric."""
    def wrapped_pinaw(y, l, u):
        return pinaw(l, u, y_true=y)
    return block_bootstrap_ci(wrapped_pinaw, y_true, lower, upper, block_size, n_bootstrap, confidence, seed)

def _smoke_test():
    y = np.random.normal(10, 2, size=(23, 10, 4))
    l = y - 1
    u = y + 1
    
    res = bootstrap_picp_ci(y, l, u)
    print("PICP CI:", res)
    
    res2 = bootstrap_pinaw_ci(y, l, u)
    print("PINAW CI:", res2)

if __name__ == "__main__":
    _smoke_test()
