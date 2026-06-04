"""
Supplysim parameter / label-threshold tuning utility.

The SC-RIHN binary resilience label requires a label-balance tuning step
because the simulator's stability regime varies with (lead_time,
safety_factor, alpha_smooth, demand_noise, cap/demand_ratio). This script:

1. Generates N networks with the per-network parameter variation now
   baked into `generate_scr_network`.
2. For each, computes the SC-RIHN convergence ratio (max pairwise
   distance among final states divided by reference-state norm).
3. Reports the distribution of ratios and the label balance under a
   range of convergence thresholds.
4. Recommends the threshold that gives label balance closest to 50/50.

Run on CPU; takes ~30-60 seconds for N=100.

Usage
-----
$ python tune_supplysim.py --n_networks 100 --T_sim 200 --seed 42
"""

from __future__ import annotations

import argparse
import logging
import time

import numpy as np

from hypercp.data.supplysim import generate_scr_network
from replicate_sc_rihn import sc_rihn_binary_label

logger = logging.getLogger(__name__)


def diagnose(
    n_networks: int = 100,
    n_trajectories: int = 6,
    T_sim: int = 200,
    n_firms: int = 30,
    n_products: int = 20,
    density: float = 0.4,
    seed: int = 42,
) -> dict:
    """Generate networks, compute convergence ratios, scan thresholds."""
    rng = np.random.default_rng(seed)
    ratios = np.zeros(n_networks)
    ref_norms = np.zeros(n_networks)

    t_start = time.time()
    for i in range(n_networks):
        net = generate_scr_network(
            n_firms=n_firms,
            n_products=n_products,
            density=density,
            seed=int(rng.integers(0, 2**31)),
        )
        # Force a no-op convergence_tolerance (we just want the ratio).
        _, _, diag = sc_rihn_binary_label(
            net,
            n_trajectories=n_trajectories,
            T_sim=T_sim,
            convergence_tolerance=1e9,  # never label as 1; we use ratio directly
            rng=rng,
        )
        ratios[i] = diag["max_dist_over_ref"]
        ref_norms[i] = diag["ref_norm"]
        if (i + 1) % 20 == 0:
            elapsed = time.time() - t_start
            logger.info(
                "Processed %d/%d networks (%.1fs elapsed; "
                "ratio so far: median=%.4f p25=%.4f p75=%.4f)",
                i + 1, n_networks, elapsed,
                float(np.median(ratios[:i+1])),
                float(np.percentile(ratios[:i+1], 25)),
                float(np.percentile(ratios[:i+1], 75)),
            )

    elapsed = time.time() - t_start
    logger.info(
        "Generated %d networks in %.1fs. Ratio distribution:", n_networks, elapsed
    )
    logger.info("  min=%.4f  p10=%.4f  p25=%.4f  median=%.4f  p75=%.4f  p90=%.4f  max=%.4f",
                float(np.min(ratios)),
                float(np.percentile(ratios, 10)),
                float(np.percentile(ratios, 25)),
                float(np.median(ratios)),
                float(np.percentile(ratios, 75)),
                float(np.percentile(ratios, 90)),
                float(np.max(ratios)))

    # Scan thresholds.
    candidate_thresholds = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0]
    logger.info("Label balance at various convergence thresholds:")
    logger.info(f"  {'threshold':>12s}  {'#pos':>5s}  {'#neg':>5s}  {'balance':>8s}")
    best_threshold = None
    best_balance = -1.0
    best_dist_from_50 = 1.0
    for thr in candidate_thresholds:
        labels = (ratios < thr).astype(int)
        balance = float(labels.mean())
        n_pos = int(labels.sum())
        n_neg = n_networks - n_pos
        dist_from_50 = abs(balance - 0.5)
        logger.info(f"  {thr:>12.4f}  {n_pos:>5d}  {n_neg:>5d}  {balance:>8.3f}")
        if dist_from_50 < best_dist_from_50:
            best_dist_from_50 = dist_from_50
            best_threshold = thr
            best_balance = balance

    logger.info("")
    logger.info("Best threshold (closest to 50/50): %.4f  (balance %.3f)",
                best_threshold, best_balance)

    # Suggest the precise threshold that gives exactly 50/50: the median.
    median_thr = float(np.median(ratios))
    logger.info(
        "Median of ratios = %.4f -- using this as the threshold gives "
        "exactly 50/50 split by construction.",
        median_thr,
    )

    return {
        "ratios": ratios.tolist(),
        "ref_norms": ref_norms.tolist(),
        "best_threshold": best_threshold,
        "best_balance": best_balance,
        "median_threshold_for_50_50": median_thr,
        "elapsed_seconds": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_networks", type=int, default=100)
    parser.add_argument("--n_trajectories", type=int, default=6)
    parser.add_argument("--T_sim", type=int, default=200)
    parser.add_argument("--n_firms", type=int, default=30)
    parser.add_argument("--n_products", type=int, default=20)
    parser.add_argument("--density", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    diagnose(
        n_networks=args.n_networks,
        n_trajectories=args.n_trajectories,
        T_sim=args.T_sim,
        n_firms=args.n_firms,
        n_products=args.n_products,
        density=args.density,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
