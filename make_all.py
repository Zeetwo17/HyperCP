"""
make_all.py -- regenerate every results table in the paper from scratch.

Runs the Gate-1 coverage suite (SupplyGraph + M5), the sample-efficiency
curve, the forecasting-baseline comparison, and the HyperCP-on-LightGBM
wrapper experiment, writing one JSON per experiment under ``runs/``.
These JSONs are the exact sources for the results tables:

    runs/gate1_supplygraph.json  -> Table I + conditional-coverage tables
    runs/gate1_m5.json           -> Table III (M5 generalisation)
    runs/gate3.json              -> sample-efficiency table
    runs/baselines.json          -> Table IV (forecasting baselines)
    runs/hypercp_on_lgbm.json    -> Table V (base-forecaster-agnostic wrapper)

Usage
-----
    python make_all.py                 # full production configs (GPU recommended)
    python make_all.py --device cuda:0 # full, on a specific GPU
    python make_all.py --quick         # fast CPU smoke (tiny configs, for CI)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(cmd: list[str]) -> None:
    print("\n>>>", " ".join(cmd), flush=True)
    t0 = time.time()
    subprocess.run(cmd, check=True, cwd=str(ROOT))
    print(f"    done in {time.time() - t0:.1f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="auto",
                    help="torch device, e.g. 'cuda:0' or 'cpu' (default: auto)")
    ap.add_argument("--quick", action="store_true",
                    help="fast CPU smoke configuration for CI / sanity checks")
    args = ap.parse_args()

    (ROOT / "runs").mkdir(exist_ok=True)
    py = sys.executable
    dev = ["--device", args.device]

    if args.quick:
        sg = ["--n_seeds", "1", "--train_epochs", "5",
              "--batch_size", "4", "--hidden_dim", "16"]
        m5 = sg + ["--m5_subsample_n", "200", "--m5_subsample_days", "120"]
        few = ["--n_seeds", "1"]
    else:
        sg = ["--n_seeds", "5", "--train_epochs", "30",
              "--batch_size", "16", "--hidden_dim", "64"]
        m5 = ["--n_seeds", "3", "--train_epochs", "30",
              "--batch_size", "16", "--hidden_dim", "64",
              "--m5_subsample_n", "1000", "--m5_subsample_days", "500"]
        few = ["--n_seeds", "3"]

    # Table I (+ conditional coverage) -- SupplyGraph Gate 1, 9 CP methods.
    run([py, "scripts/run_gate1.py", "--dataset", "supplygraph", *dev, *sg,
         "--output", "runs/gate1_supplygraph.json"])

    # Table III -- M5 generalisation (diagonal-output head).
    run([py, "scripts/run_gate1.py", "--dataset", "m5", *dev, *m5,
         "--output", "runs/gate1_m5.json"])

    # Sample-efficiency curve.
    run([py, "scripts/run_gate3.py", *dev, *few,
         "--output", "runs/gate3.json"])

    # Table IV -- forecasting baselines (TFT, LightGBM-Q, SupplyGraph-HGCN).
    run([py, "scripts/run_baselines.py", *dev, *few,
         "--baselines", "tft", "lgbm", "ccf",
         "--output", "runs/baselines.json"])

    # Table V -- HyperCP wrapped on a LightGBM-Q base forecaster.
    run([py, "scripts/run_hypercp_on_lgbm.py", "--dataset", "supplygraph",
         *dev, *few, "--output", "runs/hypercp_on_lgbm.json"])

    print("\nAll experiments complete. JSON sources for every table are in runs/.")


if __name__ == "__main__":
    main()
