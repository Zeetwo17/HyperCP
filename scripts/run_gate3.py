"""
Gate 3 runner: Sample-efficiency curve.

We measure how forecast quality scales with the number of available
training examples — a critical property for the SCM domain where each
firm's history is short.

Procedure
---------
1. Fix the calibration and test windows from the standard rolling-origin
   split.
2. For each `train_frac` in {10%, 25%, 50%, 100%}, retain only the last
   `train_frac * |train|` timesteps of the training window (so the most
   recent history is always kept).
3. Train for `train_epochs` and evaluate WMAPE / CRPS / PICP on the test
   window.
4. Repeat for `n_seeds`; report mean +/- std at each fraction.

PASS criterion: WMAPE at 25% data is within 1.25x of WMAPE at 100%, i.e.
the model is sample-efficient — most of the value is captured by a
modest amount of history. Additionally CRPS must remain finite at 10%.

Usage
-----
$ python scripts/run_gate3.py --device cuda:0 --n_seeds 3
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

# Repo-root import shim.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch

from hypercp.data.supplygraph import SupplyGraphHypergraph
from hypercp.eval.metrics import wmape, rmse, crps_quantile, picp

from scripts.run_gate1 import Gate1Config, train_quick, collect_predictions

logger = logging.getLogger(__name__)


@dataclass
class Gate3Config:
    """Configuration for Gate 3 sample-efficiency."""
    dataset: str = "supplygraph"
    device: str = "auto"
    n_seeds: int = 3
    train_fracs: tuple = (0.10, 0.25, 0.50, 1.00)
    window: int = 5
    horizon: int = 4
    train_epochs: int = 30
    batch_size: int = 16
    hidden_dim: int = 64
    target_alpha: float = 0.10
    output: str = "runs/gate3_results.json"
    # M5-specific subsampling (ignored unless dataset == "m5").
    m5_subsample_n: int = 5000
    m5_subsample_days: int = 500
    m5_seed: int = 42


def truncate_train_window(train_range: range, frac: float) -> range:
    """Keep the *last* `frac * |train_range|` timesteps."""
    full_n = train_range.stop - train_range.start
    n_keep = max(2, int(round(frac * full_n)))
    new_start = train_range.stop - n_keep
    return range(new_start, train_range.stop)


def evaluate_test(model, data, cfg, device) -> dict:
    """Run prediction on test_range and compute metrics."""
    _, _, test_range = data["splits"]
    K = model.quantile_head.n_quantiles
    quantiles = list(model.quantile_head.cfg.quantiles)

    q_test, y_test = collect_predictions(
        model, data["features"], test_range, cfg.window, cfg.horizon,
        data["target_channel"], device, K,
    )
    if q_test.size == 0:
        return {"wmape": float("nan"), "rmse": float("nan"),
                "crps": float("nan"), "picp": float("nan")}

    qs = np.array(quantiles)
    mid_idx = int(np.argmin(np.abs(qs - 0.50)))
    lo_idx = int(np.argmin(np.abs(qs - 0.05)))
    hi_idx = int(np.argmin(np.abs(qs - 0.95)))

    y_hat = q_test[..., mid_idx]
    lower = q_test[..., lo_idx]
    upper = q_test[..., hi_idx]

    return {
        "wmape": wmape(y_test, y_hat),
        "rmse":  rmse(y_test, y_hat),
        "crps":  crps_quantile(y_test, q_test, quantiles),
        "picp":  picp(y_test, lower, upper),
        "n_test_samples": int(y_test.size),
    }


def run_one_seed(cfg: Gate3Config, seed: int, device: torch.device) -> dict:
    """Run all train_fracs for a single seed."""
    logger.info("=== Seed %d ===", seed)
    if cfg.dataset == "supplygraph":
        sg = SupplyGraphHypergraph()
        data = {
            "features": sg.node_features,
            "incidence": sg.hyperedge_incidence_matrix(),
            "partition": sg.hyperedge_partition_vector(),
            "splits": sg.rolling_origin_split(),
            "n_firms": sg.n_products,
            "n_features": sg.n_channels,
            "target_channel": 1,
        }
    elif cfg.dataset == "m5":
        from hypercp.data.m5 import M5Hypergraph
        m5 = M5Hypergraph(
            subsample_n=cfg.m5_subsample_n,
            subsample_days=cfg.m5_subsample_days,
            seed=cfg.m5_seed,
        )
        data = {
            "features": m5.node_features,
            "incidence": m5.hyperedge_incidence_matrix(),
            "partition": m5.hyperedge_partition_vector(),
            "splits": m5.rolling_origin_split(),
            "n_firms": m5.n_skus,
            "n_features": m5.n_channels,
            "target_channel": 0,
        }
        logger.info("  M5 loaded: %d SKUs, %d timesteps", m5.n_skus, m5.n_timesteps)
    else:
        raise ValueError(f"Dataset {cfg.dataset} not wired into Gate 3")
    train_range_full, cal_range, test_range = data["splits"]

    per_frac = {}
    for frac in cfg.train_fracs:
        truncated = truncate_train_window(train_range_full, frac)
        data_local = dict(data)
        data_local["splits"] = (truncated, cal_range, test_range)
        logger.info("  frac=%.2f -> train range %d..%d (n=%d)",
                    frac, truncated.start, truncated.stop,
                    truncated.stop - truncated.start)

        g1cfg = Gate1Config(
            dataset="supplygraph",
            device=cfg.device,
            n_seeds=1,
            window=cfg.window,
            horizon=cfg.horizon,
            train_epochs=cfg.train_epochs,
            batch_size=cfg.batch_size,
            hidden_dim=cfg.hidden_dim,
        )

        t0 = time.time()
        try:
            model = train_quick(g1cfg, data_local, device, seed)
        except RuntimeError as e:
            logger.warning("    train failed at frac=%.2f: %s", frac, e)
            per_frac[f"{frac:.2f}"] = {"wmape": float("nan"),
                                       "rmse": float("nan"),
                                       "crps": float("nan"),
                                       "picp": float("nan"),
                                       "train_time_s": 0.0,
                                       "error": str(e)}
            continue
        train_time = time.time() - t0

        m = evaluate_test(model, data_local, cfg, device)
        m["train_time_s"] = train_time
        m["n_train_steps"] = truncated.stop - truncated.start
        logger.info("    WMAPE=%.4f  RMSE=%.4f  CRPS=%.4f  PICP=%.3f  (%.1fs)",
                    m["wmape"], m["rmse"], m["crps"], m["picp"], train_time)
        per_frac[f"{frac:.2f}"] = m

    return {"seed": seed, "per_frac": per_frac}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n_seeds", type=int, default=3)
    parser.add_argument("--train_fracs", nargs="+", type=float,
                        default=[0.10, 0.25, 0.50, 1.00])
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--train_epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--target_alpha", type=float, default=0.10)
    parser.add_argument("--output", default="runs/gate3_results.json")
    parser.add_argument("--dataset", default="supplygraph",
                        choices=["supplygraph", "m5"])
    parser.add_argument("--m5_subsample_n", type=int, default=5000)
    parser.add_argument("--m5_subsample_days", type=int, default=500)
    parser.add_argument("--m5_seed", type=int, default=42)
    args = parser.parse_args()

    cfg = Gate3Config(
        dataset=args.dataset,
        device=args.device,
        n_seeds=args.n_seeds,
        train_fracs=tuple(args.train_fracs),
        window=args.window,
        horizon=args.horizon,
        train_epochs=args.train_epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        target_alpha=args.target_alpha,
        output=args.output,
        m5_subsample_n=args.m5_subsample_n,
        m5_subsample_days=args.m5_subsample_days,
        m5_seed=args.m5_seed,
    )

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    logger.info("Gate 3 config: %s", cfg)

    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if cfg.device == "auto" else torch.device(cfg.device))
    logger.info("Device: %s", device)

    all_results = []
    for seed in range(cfg.n_seeds):
        r = run_one_seed(cfg, seed, device)
        all_results.append(r)

    # Aggregate.
    metrics_to_agg = ("wmape", "rmse", "crps", "picp")
    agg = {}
    for frac in cfg.train_fracs:
        key = f"{frac:.2f}"
        agg[key] = {}
        for m in metrics_to_agg:
            vals = [r["per_frac"][key][m] for r in all_results
                    if key in r["per_frac"] and not np.isnan(r["per_frac"][key][m])]
            if vals:
                agg[key][m] = {"mean": float(np.mean(vals)),
                               "std":  float(np.std(vals)),
                               "n":    len(vals)}
            else:
                agg[key][m] = {"mean": float("nan"),
                               "std": float("nan"), "n": 0}

    summary = {
        "config": asdict(cfg),
        "per_seed": all_results,
        "aggregate": agg,
    }
    out = Path(cfg.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, default=str))
    logger.info("Saved -> %s", out)

    # Print headline table.
    print()
    print("=" * 72)
    print("Gate 3 Summary (sample-efficiency curve)")
    print("=" * 72)
    print(f"  {'frac':<6} {'WMAPE':<18} {'RMSE':<18} {'CRPS':<18} {'PICP':<10}")
    for frac in cfg.train_fracs:
        key = f"{frac:.2f}"
        a = agg[key]
        print(f"  {key:<6} "
              f"{a['wmape']['mean']:.4f}+/-{a['wmape']['std']:.4f}     "
              f"{a['rmse']['mean']:.4f}+/-{a['rmse']['std']:.4f}     "
              f"{a['crps']['mean']:.4f}+/-{a['crps']['std']:.4f}     "
              f"{a['picp']['mean']:.3f}+/-{a['picp']['std']:.3f}")

    # PASS criterion.
    wmape_100 = agg.get("1.00", {}).get("wmape", {}).get("mean", float("nan"))
    wmape_25 = agg.get("0.25", {}).get("wmape", {}).get("mean", float("nan"))
    crps_10 = agg.get("0.10", {}).get("crps", {}).get("mean", float("nan"))
    ratio = wmape_25 / wmape_100 if wmape_100 > 0 and not np.isnan(wmape_100) else float("nan")
    pass_efficiency = (not np.isnan(ratio)) and ratio < 1.25
    pass_low_data = (not np.isnan(crps_10)) and np.isfinite(crps_10)
    print()
    print(f"  WMAPE @25% / WMAPE @100% = {ratio:.3f}  (PASS if < 1.25): {pass_efficiency}")
    print(f"  CRPS @10% finite:                          {pass_low_data}")
    print(f"  Gate 3 PASS: {pass_efficiency and pass_low_data}")


if __name__ == "__main__":
    main()
