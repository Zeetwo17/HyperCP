"""
HyperCP-on-LightGBM-Q wrapper experiment.

Addresses the v8 review's top open critique: LightGBM-Q dominates point
accuracy (WMAPE 0.340 vs HyperCP's 0.508), but is uncalibrated
(PICP 0.827). Our hyperedge ACI is a *base-forecaster-agnostic*
conformal wrapper -- we can apply it on top of any quantile forecaster.
This script verifies the obvious: wrapping LightGBM-Q with hyperedge
ACI preserves LightGBM-Q's point accuracy by construction while
delivering hyperedge coverage validity. The result is a single new row
in the headline forecast-comparison table (Table V in the paper).

Usage
-----
$ python scripts/run_hypercp_on_lgbm.py --dataset supplygraph \\
    --device cuda:0 --n_seeds 3 \\
    --output runs/hypercp_on_lgbm_supplygraph.json
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
from hypercp.baselines import LightGBMQuantile, LGBMQuantileConfig
from hypercp.models import hyperedge_cqr_score
from hypercp.calibration import ACI, ACIConfig
from hypercp.eval.metrics import wmape, rmse, crps_quantile, picp, pinaw, ece

logger = logging.getLogger(__name__)


@dataclass
class HyperCPOnLGBMConfig:
    dataset: str = "supplygraph"
    device: str = "cuda:0"
    n_seeds: int = 3
    window: int = 5
    horizon: int = 4
    target_alpha: float = 0.10
    aci_eta: float = 0.05
    aci_window: int = 30
    n_estimators: int = 100
    learning_rate: float = 0.05
    output: str = "runs/hypercp_on_lgbm_results.json"


def run_one_seed(cfg: HyperCPOnLGBMConfig, seed: int) -> dict:
    logger.info("=== seed %d ===", seed)
    if cfg.dataset != "supplygraph":
        raise NotImplementedError(
            "Only SupplyGraph is wired into the LGBM headline comparison "
            "for now (Table V in the paper). M5-scale runs use the diagonal "
            "forecast head; LGBM on M5 is straightforward but out of scope "
            "for the headline comparison."
        )
    sg = SupplyGraphHypergraph()
    train_range, cal_range, test_range = sg.rolling_origin_split()
    features = sg.node_features.cpu().numpy()  # (n_firms, T, F)
    incidence = sg.hyperedge_incidence_matrix()
    target_channel = 1  # sales_order

    # ---- Train LightGBM-Q (per-firm, per-horizon, per-quantile) ----
    quantiles = (0.05, 0.10, 0.50, 0.90, 0.95)
    lgbm_cfg = LGBMQuantileConfig(
        quantiles=quantiles,
        horizon=cfg.horizon,
        window=cfg.window,
        n_estimators=cfg.n_estimators,
        learning_rate=cfg.learning_rate,
    )
    lgbm = LightGBMQuantile(cfg=lgbm_cfg)
    t0 = time.time()
    lgbm.fit(features, target_channel=target_channel,
             train_range=train_range, seed=seed)
    logger.info("  LGBM fit: %.1fs", time.time() - t0)

    # ---- Predict on calibration + test ----
    q_cal, y_cal = lgbm.predict(features, target_channel=target_channel,
                                 test_days=cal_range)
    q_test, y_test = lgbm.predict(features, target_channel=target_channel,
                                   test_days=test_range)
    logger.info("  q_cal=%s, q_test=%s", q_cal.shape, q_test.shape)

    K = q_cal.shape[-1]
    lo_idx, hi_idx = 0, K - 1

    # ---- Baseline LightGBM-Q metrics on test set ----
    q_lo_test = q_test[..., lo_idx]
    q_hi_test = q_test[..., hi_idx]
    point_pred = q_test[..., q_test.shape[-1] // 2]   # median as point
    res_lgbm = {
        "method": "lgbm_q_baseline",
        "wmape": wmape(y_test, point_pred),
        "rmse": rmse(y_test, point_pred),
        "crps": crps_quantile(y_test, q_test, quantiles),
        "picp": picp(y_test, q_lo_test, q_hi_test),
        "pinaw": pinaw(q_lo_test, q_hi_test, y_test),
        "ece": ece(y_test, q_test, quantiles),
    }
    logger.info(
        "  LGBM baseline: WMAPE=%.3f  CRPS=%.3f  PICP=%.3f  PINAW=%.3f",
        res_lgbm["wmape"], res_lgbm["crps"], res_lgbm["picp"], res_lgbm["pinaw"],
    )

    # ---- Wrap LightGBM-Q with hyperedge ACI (max aggregation) ----
    s_cqr_he = hyperedge_cqr_score(
        torch.from_numpy(q_cal).float(),
        torch.from_numpy(y_cal).float(),
        incidence,
        lo_idx=lo_idx, hi_idx=hi_idx,
        aggregation="max",
    ).cpu().numpy().astype(np.float32)

    aci = ACI(ACIConfig(
        alpha_target=cfg.target_alpha,
        eta=cfg.aci_eta,
        window=cfg.aci_window,
        window_mode="fixed",
    ))
    aci.state.score_window = [
        torch.from_numpy(s_cqr_he[t].flatten()).float()
        for t in range(s_cqr_he.shape[0])
    ]
    while len(aci.state.score_window) > cfg.aci_window:
        aci.state.score_window.pop(0)

    # Roll through test with online alpha adaptation.
    T_test = q_test.shape[0]
    lowers, uppers = [], []
    for t in range(T_test):
        threshold = aci._threshold()
        lower = q_test[t, :, :, lo_idx] - threshold.item()
        upper = q_test[t, :, :, hi_idx] + threshold.item()
        covered = (y_test[t] >= lower) & (y_test[t] <= upper)
        err = 1.0 - covered.mean()
        new_alpha = aci.state.alpha_t + aci.cfg.eta * (
            aci.cfg.alpha_target - err
        )
        aci.state.alpha_t = aci._clamp_alpha(new_alpha)
        lowers.append(lower)
        uppers.append(upper)
    lowers = np.stack(lowers, axis=0)
    uppers = np.stack(uppers, axis=0)

    res_wrapped = {
        "method": "hypercp_on_lgbm_q",
        "wmape": wmape(y_test, point_pred),       # SAME as LGBM by construction
        "rmse": rmse(y_test, point_pred),         # SAME as LGBM
        "crps": crps_quantile(y_test, q_test, quantiles),  # SAME (raw quantiles)
        "picp": picp(y_test, lowers, uppers),
        "pinaw": pinaw(lowers, uppers, y_test),
        "ece": ece(y_test, q_test, quantiles),
        "final_alpha": float(aci.state.alpha_t.flatten()[0]),
    }
    logger.info(
        "  HyperCP+LGBM: WMAPE=%.3f (unchanged)  PICP=%.3f  PINAW=%.3f  "
        "(vs LGBM baseline PICP=%.3f, PINAW=%.3f)",
        res_wrapped["wmape"], res_wrapped["picp"], res_wrapped["pinaw"],
        res_lgbm["picp"], res_lgbm["pinaw"],
    )
    return {
        "seed": seed,
        "lgbm_q_baseline": res_lgbm,
        "hypercp_on_lgbm_q": res_wrapped,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="supplygraph")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n_seeds", type=int, default=3)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--target_alpha", type=float, default=0.10)
    parser.add_argument("--aci_eta", type=float, default=0.05)
    parser.add_argument("--aci_window", type=int, default=30)
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--output", default="runs/hypercp_on_lgbm_results.json")
    args = parser.parse_args()
    cfg = HyperCPOnLGBMConfig(**vars(args))

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    logger.info("HyperCP-on-LGBM config: %s", cfg)

    per_seed = [run_one_seed(cfg, seed) for seed in range(cfg.n_seeds)]

    def agg(method: str, metric: str) -> dict:
        vals = [r[method][metric] for r in per_seed
                if not np.isnan(r[method].get(metric, np.nan))]
        if not vals:
            return {"mean": float("nan"), "std": float("nan"), "n": 0}
        return {"mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "n": len(vals)}

    methods = ("lgbm_q_baseline", "hypercp_on_lgbm_q")
    metrics = ("wmape", "rmse", "crps", "picp", "pinaw", "ece")
    summary = {
        "config": asdict(cfg),
        "per_seed": per_seed,
        "aggregate": {
            m: {k: agg(m, k) for k in metrics}
            for m in methods
        },
    }
    out = Path(cfg.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, default=str))
    logger.info("Saved -> %s", out)

    print()
    print("=" * 70)
    print("HyperCP-on-LGBM-Q Summary")
    print("=" * 70)
    for m in methods:
        s = summary["aggregate"][m]
        print(f"  {m:25s}  WMAPE {s['wmape']['mean']:.3f}+/-{s['wmape']['std']:.3f}  "
              f"CRPS {s['crps']['mean']:.3f}  "
              f"PICP {s['picp']['mean']:.3f}+/-{s['picp']['std']:.3f}  "
              f"PINAW {s['pinaw']['mean']:.3f}")


if __name__ == "__main__":
    main()
