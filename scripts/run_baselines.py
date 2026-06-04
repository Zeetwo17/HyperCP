"""
Run baseline forecasters on the standard SupplyGraph rolling-origin split.

Produces metrics in the same JSON shape as the gate runners so they can
slot directly into the §6 results tables. Three baselines:

  1. TFTBaseline       — simplified Temporal Fusion Transformer
  2. LightGBMQuantile  — per-firm gradient-boosted quantile regression
  3. CCFHGNNBaseline   — Wasi 2024 SupplyGraph paper backbone

Usage
-----
$ python scripts/run_baselines.py --device cuda:0 --n_seeds 3
$ python scripts/run_baselines.py --baselines tft ccf
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch

from hypercp.data.supplygraph import SupplyGraphHypergraph
from hypercp.eval.metrics import wmape, rmse, crps_quantile, picp, pinaw, ece
from hypercp.baselines import (
    TFTBaseline, TFTConfig,
    LightGBMQuantile, LGBMQuantileConfig,
    CCFHGNNBaseline, CCFHGNNConfig,
)

logger = logging.getLogger(__name__)

QUANTILES = (0.05, 0.10, 0.50, 0.90, 0.95)


@dataclass
class BaselinesConfig:
    device: str = "auto"
    n_seeds: int = 3
    window: int = 5
    horizon: int = 4
    train_epochs: int = 30
    batch_size: int = 16
    hidden_dim: int = 64
    target_alpha: float = 0.10
    baselines: tuple = ("tft", "lgbm", "ccf")
    output: str = "runs/baselines_results.json"


def metrics_from_predictions(q_pred: np.ndarray, y_true: np.ndarray,
                             quantiles=QUANTILES) -> dict:
    if q_pred.size == 0:
        return {"wmape": float("nan"), "rmse": float("nan"),
                "crps": float("nan"), "picp": float("nan"),
                "pinaw": float("nan"), "ece": float("nan")}
    qs = np.array(quantiles)
    mid_idx = int(np.argmin(np.abs(qs - 0.50)))
    lo_idx = int(np.argmin(np.abs(qs - 0.05)))
    hi_idx = int(np.argmin(np.abs(qs - 0.95)))
    y_hat = q_pred[..., mid_idx]
    lower = q_pred[..., lo_idx]
    upper = q_pred[..., hi_idx]
    return {
        "wmape": wmape(y_true, y_hat),
        "rmse":  rmse(y_true, y_hat),
        "crps":  crps_quantile(y_true, q_pred, quantiles),
        "picp":  picp(y_true, lower, upper),
        "pinaw": pinaw(lower, upper, y_true),
        "ece":   ece(y_true, q_pred, list(quantiles)),
    }


def run_tft(features_np: np.ndarray, incidence_np: np.ndarray,
            target_channel: int, splits: tuple, cfg: BaselinesConfig,
            device: torch.device, seed: int) -> dict:
    n_features = features_np.shape[2]
    tcfg = TFTConfig(hidden_dim=cfg.hidden_dim, horizon=cfg.horizon,
                     window=cfg.window, quantiles=QUANTILES)
    bl = TFTBaseline(n_features, tcfg, device=device)
    train_range, _, test_range = splits
    t0 = time.time()
    bl.fit(features_np, target_channel, train_range,
           epochs=cfg.train_epochs, batch_size=cfg.batch_size, seed=seed)
    fit_s = time.time() - t0
    q, y = bl.predict(features_np, target_channel, test_range)
    m = metrics_from_predictions(q, y)
    m["fit_time_s"] = fit_s
    m["n_test"] = int(y.size)
    return m


def run_lgbm(features_np: np.ndarray, incidence_np: np.ndarray,
             target_channel: int, splits: tuple, cfg: BaselinesConfig,
             device: torch.device, seed: int) -> dict:
    lcfg = LGBMQuantileConfig(quantiles=QUANTILES, horizon=cfg.horizon,
                              window=cfg.window)
    bl = LightGBMQuantile(lcfg)
    train_range, _, test_range = splits
    t0 = time.time()
    bl.fit(features_np, target_channel, train_range, seed=seed)
    fit_s = time.time() - t0
    q, y = bl.predict(features_np, target_channel, test_range)
    m = metrics_from_predictions(q, y)
    m["fit_time_s"] = fit_s
    m["n_test"] = int(y.size)
    return m


def run_ccf(features_np: np.ndarray, incidence_np: np.ndarray,
            target_channel: int, splits: tuple, cfg: BaselinesConfig,
            device: torch.device, seed: int) -> dict:
    n_features = features_np.shape[2]
    ccfg = CCFHGNNConfig(hidden_dim=cfg.hidden_dim, horizon=cfg.horizon,
                         window=cfg.window, quantiles=QUANTILES)
    bl = CCFHGNNBaseline(n_features, ccfg, device=device)
    train_range, _, test_range = splits
    t0 = time.time()
    bl.fit(features_np, incidence_np, target_channel, train_range,
           epochs=cfg.train_epochs, batch_size=cfg.batch_size, seed=seed)
    fit_s = time.time() - t0
    q, y = bl.predict(features_np, incidence_np, target_channel, test_range)
    m = metrics_from_predictions(q, y)
    m["fit_time_s"] = fit_s
    m["n_test"] = int(y.size)
    return m


RUNNERS = {"tft": run_tft, "lgbm": run_lgbm, "ccf": run_ccf}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n_seeds", type=int, default=3)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--train_epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--target_alpha", type=float, default=0.10)
    parser.add_argument("--baselines", nargs="+",
                        default=["tft", "lgbm", "ccf"],
                        choices=list(RUNNERS.keys()))
    parser.add_argument("--output", default="runs/baselines_results.json")
    args = parser.parse_args()
    cfg = BaselinesConfig(
        device=args.device, n_seeds=args.n_seeds,
        window=args.window, horizon=args.horizon,
        train_epochs=args.train_epochs, batch_size=args.batch_size,
        hidden_dim=args.hidden_dim, target_alpha=args.target_alpha,
        baselines=tuple(args.baselines), output=args.output,
    )

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    logger.info("Baselines config: %s", cfg)

    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if cfg.device == "auto" else torch.device(cfg.device))
    logger.info("Device: %s", device)

    sg = SupplyGraphHypergraph()
    features_np = sg.node_features.cpu().numpy()
    incidence_np = sg.hyperedge_incidence_matrix().cpu().numpy()
    splits = sg.rolling_origin_split()
    target_channel = 1

    per_seed: dict[str, list] = {b: [] for b in cfg.baselines}
    for seed in range(cfg.n_seeds):
        logger.info("=== Seed %d ===", seed)
        for bname in cfg.baselines:
            logger.info("  -> %s", bname)
            try:
                m = RUNNERS[bname](features_np, incidence_np, target_channel,
                                   splits, cfg, device, seed)
            except Exception as e:
                logger.exception("Baseline %s failed: %s", bname, e)
                m = {"error": str(e)}
            per_seed[bname].append({"seed": seed, **m})

    # Aggregate.
    metrics = ("wmape", "rmse", "crps", "picp", "pinaw", "ece")
    agg: dict[str, dict] = {}
    for bname in cfg.baselines:
        agg[bname] = {}
        for m in metrics:
            vals = [r[m] for r in per_seed[bname]
                    if m in r and isinstance(r[m], float)
                    and not np.isnan(r[m]) and np.isfinite(r[m])]
            if vals:
                agg[bname][m] = {"mean": float(np.mean(vals)),
                                 "std":  float(np.std(vals)),
                                 "n":    len(vals)}
            else:
                agg[bname][m] = {"mean": float("nan"),
                                 "std": float("nan"), "n": 0}

    summary = {
        "config": asdict(cfg),
        "per_seed": per_seed,
        "aggregate": agg,
    }
    out = Path(cfg.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, default=str))
    logger.info("Saved -> %s", out)

    # Print headline.
    print()
    print("=" * 80)
    print("Baselines Summary")
    print("=" * 80)
    print(f"  {'baseline':<10} {'WMAPE':<18} {'CRPS':<18} {'PICP':<14} {'ECE':<14}")
    for b in cfg.baselines:
        a = agg[b]
        print(f"  {b:<10} "
              f"{a['wmape']['mean']:.4f}+/-{a['wmape']['std']:.4f}    "
              f"{a['crps']['mean']:.4f}+/-{a['crps']['std']:.4f}    "
              f"{a['picp']['mean']:.3f}+/-{a['picp']['std']:.3f}  "
              f"{a['ece']['mean']:.4f}+/-{a['ece']['std']:.4f}")


if __name__ == "__main__":
    main()
