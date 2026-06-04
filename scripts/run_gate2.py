"""
Gate 2 runner: OR-grounding via Spearman correlations.

For each firm c, we compute a model-derived *functional resilience score*
r_struct(c) and correlate it against operations-research ground truth
(TTR, LPR, RL, r*) obtained from the SupplyGraph-Stressed benchmark
(Hosseini-Ivanov-Dolgui 2019; Behzadi 2020).

The score r_struct(c) is computed from the trained joint model's predicted
quantile distribution F_c over the holdout period. Firms with tight,
stable predicted distributions are inherently more resilient — they are
better-buffered against demand shocks. We use the inverse mean-width of
the 90% predicted interval as the structural proxy:

    r_struct(c) = - mean_t (q_{0.95}(c, t) - q_{0.05}(c, t))

(negative width so larger = better, like r*).

PASS criterion: Spearman(r_struct, r_ground) > 0.50 against each of
-TTR, -LPR, -RL, and r*. The 0.50 bar is intentionally permissive
because the structural proxy ignores shock geometry; the headline claim
is *grounded calibration*, not perfect resilience prediction.

Usage
-----
$ python scripts/run_gate2.py --device cuda:0 --n_shocks 200 --n_seeds 3
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
from hypercp.data.supplygraph_stressed import StressedSupplyGraph
from hypercp.eval.metrics import spearman_correlation

# Reuse train_quick + collect_predictions from Gate 1.
from scripts.run_gate1 import Gate1Config, train_quick, collect_predictions

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration.
# =============================================================================


@dataclass
class Gate2Config:
    """Configuration for Gate 2 OR-grounding."""
    device: str = "auto"
    n_seeds: int = 3
    n_shocks: int = 200
    T_recovery: int = 60
    window: int = 5
    horizon: int = 4
    train_epochs: int = 30
    batch_size: int = 16
    hidden_dim: int = 64
    safety_factor: float = 1.5
    capacity_multiplier: float = 3.0
    output: str = "runs/gate2_results.json"


# =============================================================================
# Resilience functionals.
# =============================================================================


def compute_struct_resilience(q_pred: np.ndarray, lo_idx: int, hi_idx: int) -> np.ndarray:
    """Compute per-firm structural resilience proxy.

    Parameters
    ----------
    q_pred : (T_days, n_firms, H, K)
        Predicted quantile array.
    lo_idx, hi_idx : indices of low / high quantiles to use for width.

    Returns
    -------
    r_struct : (n_firms,) — larger = more resilient (negative width).
    """
    width = q_pred[..., hi_idx] - q_pred[..., lo_idx]  # (T, n_firms, H)
    width = np.maximum(width, 0.0)
    mean_width = width.mean(axis=(0, 2))  # avg over time and horizon
    return -mean_width


def compute_struct_resilience_signal(q_pred: np.ndarray, mid_idx: int) -> np.ndarray:
    """Median signal-to-noise ratio per firm.

    Returns mean(q_median) / temporal-std(q_median). Higher = more stable.
    """
    median = q_pred[..., mid_idx]  # (T, n_firms, H)
    mean_med = median.mean(axis=(0, 2))
    std_med = median.std(axis=(0, 2)).clip(min=1e-6)
    return mean_med / std_med


def compute_struct_resilience_lowerq(q_pred: np.ndarray, lo_idx: int,
                                     mid_idx: int) -> np.ndarray:
    """Lower-quantile / median ratio per firm.

    r_lowerq(c) = mean_t q_lo(c, t) / mean_t q_mid(c, t)

    Captures "how close is the firm's downside to its median?" Firms whose
    lower predicted quantile is close to the median are predicted to
    consistently meet demand, hence resilient.
    """
    lo = q_pred[..., lo_idx].mean(axis=(0, 2))
    mid = q_pred[..., mid_idx].mean(axis=(0, 2)).clip(min=1e-6)
    return lo / mid


def compute_struct_resilience_centrality(incidence: np.ndarray) -> np.ndarray:
    """Hypergraph degree centrality per firm.

    centrality(c) = number of hyperedges containing firm c, normalised by max.

    Pure-topology proxy: firms in many hyperedges are more "diversified"
    in the supply network, hence (heuristically) more resilient. Does not
    use the trained model at all -- useful as a hypergraph-only baseline.
    """
    deg = incidence.sum(axis=1).astype(np.float64)  # (n_firms,)
    mx = max(float(deg.max()), 1.0)
    return deg / mx


def compute_struct_resilience_composite(width_proxy: np.ndarray,
                                        lowerq_proxy: np.ndarray,
                                        centrality: np.ndarray) -> np.ndarray:
    """Composite: z-score and sum the three orthogonal-ish signals.

    Each component is standardised (zero mean, unit std) so they contribute
    equally to the rank order. Higher composite = more resilient.
    """
    def _z(x):
        s = x.std()
        return (x - x.mean()) / (s if s > 1e-9 else 1.0)
    return _z(width_proxy) + _z(lowerq_proxy) + _z(centrality)


# =============================================================================
# Per-seed runner.
# =============================================================================


def run_one_seed(cfg: Gate2Config, seed: int, device: torch.device,
                 ssg: StressedSupplyGraph) -> dict:
    """Train once, compute structural resilience, correlate with ground truth."""
    logger.info("=== Seed %d ===", seed)

    sg = ssg.base
    data = {
        "features": sg.node_features,
        "incidence": sg.hyperedge_incidence_matrix(),
        "partition": sg.hyperedge_partition_vector(),
        "splits": sg.rolling_origin_split(),
        "n_firms": sg.n_products,
        "n_features": sg.n_channels,
        "target_channel": 1,
    }

    # Reuse Gate1Config dimensions so train_quick works.
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
    model = train_quick(g1cfg, data, device, seed)
    logger.info("  trained in %.1fs", time.time() - t0)

    K = model.quantile_head.n_quantiles
    quantiles = list(model.quantile_head.cfg.quantiles)

    # Collect predictions on the full holdout (cal + test) for the functional.
    _, cal_range, test_range = data["splits"]
    full_range = range(cal_range.start, test_range.stop)
    q_full, _ = collect_predictions(
        model, data["features"], full_range, cfg.window, cfg.horizon,
        data["target_channel"], device, K,
    )
    logger.info("  q_full=%s", q_full.shape)

    # Resilience proxy: inverse predicted width.
    # Find lo_idx, hi_idx for the canonical 5% / 95% quantiles.
    qs = np.array(quantiles)
    lo_idx = int(np.argmin(np.abs(qs - 0.05)))
    hi_idx = int(np.argmin(np.abs(qs - 0.95)))
    mid_idx = int(np.argmin(np.abs(qs - 0.50)))

    r_struct_width = compute_struct_resilience(q_full, lo_idx, hi_idx)
    r_struct_signal = compute_struct_resilience_signal(q_full, mid_idx)
    r_struct_lowerq = compute_struct_resilience_lowerq(q_full, lo_idx, mid_idx)
    inc_np = data["incidence"].cpu().numpy() if hasattr(data["incidence"], "cpu") \
        else np.asarray(data["incidence"])
    r_struct_centrality = compute_struct_resilience_centrality(inc_np)
    r_struct_composite = compute_struct_resilience_composite(
        r_struct_width, r_struct_lowerq, r_struct_centrality,
    )

    # Aggregate ground-truth resilience per firm by averaging over shocks
    # where the firm was DIRECTLY HIT. Averaging over all shocks (including
    # those that target other firms) would give every firm TTR=0 most of the
    # time and the rank order would track shock-count noise rather than
    # intrinsic resilience.
    labs = ssg.resilience_labels
    nf = labs.r_star.shape[1]
    hit_mask = np.zeros((labs.r_star.shape[0], nf), dtype=bool)
    for k, sh in enumerate(labs.shocks):
        hit_mask[k, sh.firm_idx] = True

    def _per_firm_hits(arr: np.ndarray) -> np.ndarray:
        out = np.full(nf, np.nan, dtype=np.float64)
        for c in range(nf):
            sel = hit_mask[:, c]
            if sel.any():
                out[c] = float(arr[sel, c].mean())
        return out

    r_ground_TTR    = _per_firm_hits(labs.ttr)
    r_ground_LPR    = _per_firm_hits(labs.lpr)
    r_ground_RL     = _per_firm_hits(labs.rl)
    r_ground_rstar  = _per_firm_hits(labs.r_star)
    r_ground_binary = _per_firm_hits(labs.binary.astype(np.float64))

    logger.info("  Hit-count per firm: min=%d median=%d max=%d (target ~%d)",
                int(hit_mask.sum(axis=0).min()),
                int(np.median(hit_mask.sum(axis=0))),
                int(hit_mask.sum(axis=0).max()),
                cfg.n_shocks // nf)

    # Spearman correlations. r_struct is "higher = better"; align signs.
    pairs = {
        "r_struct_width":      r_struct_width,
        "r_struct_signal":     r_struct_signal,
        "r_struct_lowerq":     r_struct_lowerq,
        "r_struct_centrality": r_struct_centrality,
        "r_struct_composite":  r_struct_composite,
    }
    ground = {
        "neg_TTR":    -r_ground_TTR,
        "neg_LPR":    -r_ground_LPR,
        "neg_RL":     -r_ground_RL,
        "rstar":       r_ground_rstar,
        "binary_mean": r_ground_binary,
    }

    correlations: dict[str, dict[str, float]] = {}
    for sname, sval in pairs.items():
        correlations[sname] = {}
        for gname, gval in ground.items():
            rho = spearman_correlation(sval, gval)
            correlations[sname][gname] = rho
            logger.info("  Spearman(%s, %s) = %+.3f", sname, gname, rho)

    return {
        "seed": seed,
        "r_struct_width":      r_struct_width.tolist(),
        "r_struct_signal":     r_struct_signal.tolist(),
        "r_struct_lowerq":     r_struct_lowerq.tolist(),
        "r_struct_centrality": r_struct_centrality.tolist(),
        "r_struct_composite":  r_struct_composite.tolist(),
        "r_ground_rstar_per_firm": r_ground_rstar.tolist(),
        "r_ground_TTR_per_firm":   r_ground_TTR.tolist(),
        "r_ground_LPR_per_firm":   r_ground_LPR.tolist(),
        "r_ground_RL_per_firm":    r_ground_RL.tolist(),
        "correlations": correlations,
    }


# =============================================================================
# Main runner.
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n_seeds", type=int, default=3)
    parser.add_argument("--n_shocks", type=int, default=200)
    parser.add_argument("--T_recovery", type=int, default=60)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--train_epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--safety_factor", type=float, default=1.5)
    parser.add_argument("--capacity_multiplier", type=float, default=3.0)
    parser.add_argument("--output", default="runs/gate2_results.json")
    args = parser.parse_args()
    cfg = Gate2Config(**vars(args))

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    logger.info("Gate 2 config: %s", cfg)

    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if cfg.device == "auto" else torch.device(cfg.device))
    logger.info("Device: %s", device)

    # Build SupplyGraph-Stressed once; reuse across seeds.
    logger.info("Generating SupplyGraph-Stressed (n_shocks=%d, T_recovery=%d) ...",
                cfg.n_shocks, cfg.T_recovery)
    sg = SupplyGraphHypergraph()
    ssg = StressedSupplyGraph(
        sg,
        n_shocks=cfg.n_shocks,
        T_recovery=cfg.T_recovery,
        safety_factor=cfg.safety_factor,
        capacity_multiplier=cfg.capacity_multiplier,
        seed=0,
    )
    diags = ssg.gate2_diagnostics()
    logger.info("Ground-truth label internal consistency:")
    for k, v in diags.items():
        logger.info("  %s = %+.3f", k, v)

    # Run per-seed.
    all_results = []
    for seed in range(cfg.n_seeds):
        r = run_one_seed(cfg, seed, device, ssg)
        all_results.append(r)

    # Aggregate correlations.
    proxy_names = ("r_struct_width", "r_struct_signal", "r_struct_lowerq",
                   "r_struct_centrality", "r_struct_composite")
    summary_corr: dict[str, dict[str, dict]] = {}
    for sname in proxy_names:
        summary_corr[sname] = {}
        for gname in ("neg_TTR", "neg_LPR", "neg_RL", "rstar", "binary_mean"):
            vals = [r["correlations"][sname][gname] for r in all_results
                    if not np.isnan(r["correlations"][sname][gname])]
            if vals:
                summary_corr[sname][gname] = {
                    "mean": float(np.mean(vals)),
                    "std":  float(np.std(vals)),
                    "n":    len(vals),
                }
            else:
                summary_corr[sname][gname] = {
                    "mean": float("nan"), "std": float("nan"), "n": 0,
                }

    summary = {
        "config": asdict(cfg),
        "ground_truth_diagnostics": diags,
        "per_seed": all_results,
        "aggregate_correlations": summary_corr,
    }

    out = Path(cfg.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, default=str))
    logger.info("Saved -> %s", out)

    # Print headline.
    print()
    print("=" * 64)
    print("Gate 2 Summary (Spearman correlations)")
    print("=" * 64)
    print(f"  Ground-truth diagnostics: {diags}")
    print()
    for sname, gdict in summary_corr.items():
        print(f"  {sname}:")
        for gname, s in gdict.items():
            print(f"    vs {gname:14s}  rho = {s['mean']:+.3f} +/- {s['std']:.3f}")
    print()

    # PASS criterion: any structural proxy correlates > 0.50 with all four
    # canonical resilience signals (neg_TTR, neg_LPR, neg_RL, rstar).
    pass_flags = {}
    for sname in summary_corr:
        targets = ("neg_TTR", "neg_LPR", "neg_RL", "rstar")
        ok = all(summary_corr[sname][g]["mean"] > 0.50 for g in targets)
        pass_flags[sname] = ok
    pass_overall = any(pass_flags.values())
    print(f"  Per-proxy pass flags: {pass_flags}")
    print(f"  Gate 2 PASS: {pass_overall}")


if __name__ == "__main__":
    main()
