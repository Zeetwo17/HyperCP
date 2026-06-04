"""
HyperCP shared evaluation metrics.

Every gate runner and baseline comparator pulls metrics from this single
module so the paper's tables are guaranteed to use identical definitions.
All metrics accept numpy arrays or torch tensors; outputs are Python floats
to make them JSON-serialisable for run histories.

Reference conventions
---------------------
- Forecast metrics: Hyndman & Koehler (IJF 2006), Wasi et al. (SupplyGraph 2024)
- CRPS via quantile approximation: Gneiting & Raftery (JASA 2007)
- PICP / PINAW: Khosravi et al. (IEEE TNN 2011)
- ECE for quantile coverage: Naeini et al. (AAAI 2015), Kuleshov et al. (ICML 2018)
- Spearman: scipy.stats.spearmanr
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


# =============================================================================
# Type helpers -- accept numpy arrays or torch tensors, return floats.
# =============================================================================


def _to_numpy(x) -> np.ndarray:
    """Convert torch.Tensor / list / numpy -> numpy float array (detached)."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


# =============================================================================
# Forecast accuracy metrics.
# =============================================================================


def wmape(
    y_true,
    y_pred,
    mask=None,
) -> float:
    """Weighted Mean Absolute Percentage Error.

    WMAPE = sum |y - y_hat| / sum |y|

    Standard for retail / SCM forecasting (SupplyGraph convention). Robust
    to zeros in y_true unlike per-sample MAPE.
    """
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )
    if mask is not None:
        m = _to_numpy(mask).astype(bool)
        y_true = y_true[m]
        y_pred = y_pred[m]
    denom = np.abs(y_true).sum()
    if denom < 1e-12:
        return float("inf")
    return float(np.abs(y_true - y_pred).sum() / denom)


def mase(
    y_true,
    y_pred,
    y_train=None,
    seasonality: int = 1,
    mask=None,
) -> float:
    """Mean Absolute Scaled Error (Hyndman-Koehler 2006).

    MASE = MAE(y, y_hat) / MAE(y_train, naive_seasonal(y_train))

    Defaults to non-seasonal (seasonality=1). For weekly SupplyGraph,
    seasonality=7 may be more appropriate.

    If `y_train` is None, falls back to the in-sample naive baseline
    using y_true itself (less rigorous but useful for quick comparisons).
    """
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    if mask is not None:
        m = _to_numpy(mask).astype(bool)
        y_true = y_true[m]
        y_pred = y_pred[m]
    mae_test = np.abs(y_true - y_pred).mean()

    if y_train is None:
        y_train = y_true
    y_train = _to_numpy(y_train).flatten()
    if len(y_train) <= seasonality:
        return float("nan")
    naive = np.abs(y_train[seasonality:] - y_train[:-seasonality]).mean()
    if naive < 1e-12:
        return float("inf")
    return float(mae_test / naive)


def rmse(y_true, y_pred, mask=None) -> float:
    """Root mean squared error."""
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    if mask is not None:
        m = _to_numpy(mask).astype(bool)
        y_true = y_true[m]
        y_pred = y_pred[m]
    return float(np.sqrt(((y_true - y_pred) ** 2).mean()))


def smape(y_true, y_pred, mask=None) -> float:
    """Symmetric MAPE in [0, 200]. Common SCM benchmark metric.

    sMAPE = 100 * mean( 2 |y - y_hat| / (|y| + |y_hat|) )
    """
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    if mask is not None:
        m = _to_numpy(mask).astype(bool)
        y_true = y_true[m]
        y_pred = y_pred[m]
    denom = np.maximum(np.abs(y_true) + np.abs(y_pred), 1e-12)
    return float(200.0 * (np.abs(y_true - y_pred) / denom).mean())


# =============================================================================
# Uncertainty / quantile metrics.
# =============================================================================


def pinball_loss_per_quantile(
    y_true,
    q_pred,
    quantiles: Sequence[float],
    mask=None,
) -> dict:
    """Pinball (asymmetric) loss per quantile, plus the mean across quantiles.

    Parameters
    ----------
    y_true : (...,) array of observed targets
    q_pred : (..., K) array of K quantile predictions
    quantiles : (K,) levels in (0, 1)
    mask : optional (...,) bool mask

    Returns
    -------
    dict with keys 'mean' and 'per_quantile' (dict tau -> loss).
    """
    y_true = _to_numpy(y_true)
    q_pred = _to_numpy(q_pred)
    if q_pred.shape[:-1] != y_true.shape:
        raise ValueError(
            f"q_pred shape {q_pred.shape} does not broadcast to y_true {y_true.shape}"
        )
    quantiles = np.asarray(quantiles, dtype=np.float64)
    if quantiles.size != q_pred.shape[-1]:
        raise ValueError(
            f"quantiles has {quantiles.size} entries, q_pred last dim is {q_pred.shape[-1]}"
        )

    if mask is not None:
        m = _to_numpy(mask).astype(bool)
        y_true = y_true[m]
        q_pred = q_pred[m]

    y_b = y_true[..., None]
    diff = y_b - q_pred
    losses = np.maximum(quantiles * diff, (quantiles - 1.0) * diff)  # (..., K)
    per_q = losses.mean(axis=tuple(range(losses.ndim - 1)))  # (K,)
    return {
        "mean": float(per_q.mean()),
        "per_quantile": {float(t): float(p) for t, p in zip(quantiles, per_q)},
    }


def crps_quantile(
    y_true,
    q_pred,
    quantiles: Sequence[float],
    mask=None,
) -> float:
    """Continuous Ranked Probability Score, computed from a finite-quantile
    approximation. Lower is better.

    For predictive distribution F approximated by K quantiles q_{tau_k},
        CRPS(F, y) approx 2/K * sum_k pinball_{tau_k}(y, q_{tau_k}).

    This is the Gneiting-Raftery (JASA 2007) discrete approximation; exact
    in the limit of dense quantiles.
    """
    res = pinball_loss_per_quantile(y_true, q_pred, quantiles, mask=mask)
    K = len(quantiles)
    return float(2.0 * res["mean"] * 1.0)  # mean over K already factored in


def picp(
    y_true,
    lower,
    upper,
    mask=None,
) -> float:
    """Prediction Interval Coverage Probability.

    PICP = fraction of y_true that lie in [lower, upper].

    For nominal coverage 1 - alpha, PICP should be at least 1 - alpha
    asymptotically (under CP exchangeability assumptions).
    """
    y_true = _to_numpy(y_true)
    lower = _to_numpy(lower)
    upper = _to_numpy(upper)
    if mask is not None:
        m = _to_numpy(mask).astype(bool)
        y_true = y_true[m]
        lower = lower[m]
        upper = upper[m]
    covered = (y_true >= lower) & (y_true <= upper)
    return float(covered.mean())


def pinaw(
    lower,
    upper,
    y_true=None,
    mask=None,
) -> float:
    """Prediction Interval Normalised Average Width.

    PINAW = mean(upper - lower) / range(y_true)

    If y_true is not provided, uses 1.0 as the denominator so the metric
    reduces to the raw mean width.
    """
    lower = _to_numpy(lower)
    upper = _to_numpy(upper)
    if mask is not None:
        m = _to_numpy(mask).astype(bool)
        lower = lower[m]
        upper = upper[m]
    width = (upper - lower).mean()
    if y_true is None:
        return float(width)
    y_true = _to_numpy(y_true)
    if mask is not None:
        y_true = y_true[m]
    rng = y_true.max() - y_true.min()
    if rng < 1e-12:
        return float("inf")
    return float(width / rng)


def ece(
    y_true,
    q_pred,
    quantiles: Sequence[float],
    mask=None,
) -> float:
    """Expected Calibration Error (quantile version).

    For each predicted quantile tau, compute empirical coverage
    rho(tau) = P[y <= q_tau]. The ECE is the mean absolute gap
    |rho(tau) - tau| averaged across the quantile set.

    A perfectly calibrated predictor has ECE = 0.
    """
    y_true = _to_numpy(y_true)
    q_pred = _to_numpy(q_pred)
    if mask is not None:
        m = _to_numpy(mask).astype(bool)
        y_true = y_true[m]
        q_pred = q_pred[m]

    quantiles = np.asarray(quantiles, dtype=np.float64)
    gaps = []
    for k, tau in enumerate(quantiles):
        emp = float((y_true <= q_pred[..., k]).mean())
        gaps.append(abs(emp - float(tau)))
    return float(np.mean(gaps))


# =============================================================================
# Resilience correlation (Gate 2).
# =============================================================================


def spearman_correlation(x, y, mask=None) -> float:
    """Spearman rank correlation between two 1-D arrays.

    Used for Gate 2 (OR-grounding): correlate the functional resilience
    score with TTR, LPR, RL ground truths.
    """
    from scipy.stats import spearmanr

    x = _to_numpy(x).flatten()
    y = _to_numpy(y).flatten()
    if mask is not None:
        m = _to_numpy(mask).astype(bool).flatten()
        x = x[m]
        y = y[m]
    if x.size < 3 or y.size < 3:
        return float("nan")
    if np.isnan(x).any() or np.isnan(y).any():
        # Drop NaN rows pairwise.
        ok = ~(np.isnan(x) | np.isnan(y))
        x = x[ok]
        y = y[ok]
        if x.size < 3:
            return float("nan")
    if x.std() < 1e-12 or y.std() < 1e-12:
        return float("nan")
    return float(spearmanr(x, y).statistic)


# =============================================================================
# Smoke test.
# =============================================================================


def _smoke_test() -> None:
    rng = np.random.default_rng(0)

    # Forecasting metrics.
    y = rng.normal(loc=10.0, scale=2.0, size=200)
    y_hat = y + rng.normal(scale=0.5, size=200)
    print(f"WMAPE   = {wmape(y, y_hat):.4f}")
    print(f"MASE    = {mase(y, y_hat):.4f}")
    print(f"RMSE    = {rmse(y, y_hat):.4f}")
    print(f"sMAPE   = {smape(y, y_hat):.4f}")

    # Quantile metrics.
    quantiles = [0.05, 0.10, 0.50, 0.90, 0.95]
    # Synthetic quantile predictions: y_hat as median, +/- some width.
    q = np.stack([y_hat - 2, y_hat - 1, y_hat, y_hat + 1, y_hat + 2], axis=-1)
    res = pinball_loss_per_quantile(y, q, quantiles)
    print(f"Pinball mean  = {res['mean']:.4f}")
    print(f"Pinball per-q = {[round(v, 3) for v in res['per_quantile'].values()]}")
    print(f"CRPS    = {crps_quantile(y, q, quantiles):.4f}")
    print(f"PICP @ 90% = {picp(y, q[..., 0], q[..., 4]):.3f}")
    print(f"PINAW   = {pinaw(q[..., 0], q[..., 4], y):.4f}")
    print(f"ECE     = {ece(y, q, quantiles):.4f}")

    # Spearman.
    print(f"Spearman(y, y_hat) = {spearman_correlation(y, y_hat):.3f}")
    print(f"Spearman(y, rng)   = {spearman_correlation(y, rng.normal(size=200)):.3f}")

    print("[OK] metrics smoke test passed.")


if __name__ == "__main__":
    _smoke_test()
