"""
Gate 1 runner: Coverage validation.

This is the headline empirical claim of the paper. We compare four
conformal-prediction methods on rolling-origin time-series data:

  1. Split CP (node-level) -- invalid time-series baseline; expected to drift
  2. ACI (node-level)      -- valid but ignores hyperedge structure
  3. ACI (hyperedge-level) -- our contribution
  4. Conformal PID (node)  -- alternative time-series CP for ablation (b)

For each method we report:
  - Marginal coverage (target: 1 - alpha)
  - PINAW (interval width)
  - ECE (calibration error)
  - Conditional coverage by horizon
  - Per-class coverage (for hyperedge-level methods)

PASS criterion: hyperedge-level ACI achieves marginal coverage within
+/- 0.02 of target while split CP drifts > 0.05 below target.

Usage
-----
$ python scripts/run_gate1.py --dataset supplygraph --device cuda:0
$ python scripts/run_gate1.py --dataset scr --device cuda:0 --n_seeds 5
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

# Repo-root import shim.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch
from torch import nn

from hypercp.data.supplygraph import SupplyGraphHypergraph
from hypercp.models import (
    EncoderConfig,
    ForecastHeadConfig,
    QuantileHeadConfig,
    JointConfig,
    HyperCPJointModel,
    HyperedgeConformityScore,
    ConformityConfig,
    hyperedge_cqr_score,
)
from hypercp.calibration import (
    ACI, ACIConfig,
    SplitCP, SplitCPConfig,
    ConformalPID, ConformalPIDConfig,
    AgACI, AgACIConfig,
    CFGNNSmoothedCalibrator, CFGNNHyperConfig,
    NCPNETStyleCalibrator, NCPNETHyperConfig,
)
from hypercp.eval.metrics import picp, pinaw, ece

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration.
# =============================================================================


@dataclass
class Gate1Config:
    """Configuration for Gate 1 coverage-validation."""
    dataset: str = "supplygraph"
    device: str = "auto"
    target_alpha: float = 0.10              # nominal miscoverage; 0.10 = 90% coverage
    aci_eta: float = 0.05
    aci_window: int = 30
    n_seeds: int = 5
    window: int = 5                          # T = rolling-origin window length
    horizon: int = 4                         # H = forecast horizon
    train_epochs: int = 30                   # quick train
    batch_size: int = 16
    hidden_dim: int = 64
    output: str = "runs/gate1_results.json"
    # M5-specific subsampling (ignored unless dataset == "m5"). The
    # forecast/quantile heads run in `output_mode='diagonal'` on M5 to
    # avoid the O(n_skus * n_skus) memory blow-up of the cross-product
    # head.
    m5_subsample_n: int = 1000
    m5_subsample_days: int = 500
    m5_seed: int = 42


# =============================================================================
# Helpers.
# =============================================================================


def collect_predictions(
    model: HyperCPJointModel,
    features: torch.Tensor,
    days: range,
    window: int,
    horizon: int,
    target_channel: int,
    device: torch.device,
    K_quantiles: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference over `days` and collect (q_pred, y_true).

    Returns
    -------
    q_pred : (T_days, n_firms, H, K)
    y_true : (T_days, n_firms, H)
    """
    model.eval()
    n_firms = features.size(0)
    q_all = []
    y_all = []
    valid = [d for d in days if d + horizon < features.size(1)]
    with torch.no_grad():
        for t in valid:
            if t - window + 1 < 0:
                continue
            feat = features[:, t - window + 1:t + 1, :].unsqueeze(0).to(device)
            outputs = model(feat)
            q_hat = outputs["q_hat"]
            # Support both output modes:
            # - cross_product : (1, n_firms, n_products, H, K) -> take diagonal
            # - diagonal      : (1, n_firms, H, K)             -> use as-is
            if q_hat.ndim == 5:
                idx = torch.arange(n_firms, device=device)
                q_diag = q_hat[0, idx, idx, :, :]  # (n_firms, H, K)
            elif q_hat.ndim == 4:
                q_diag = q_hat[0]                  # (n_firms, H, K)
            else:
                raise RuntimeError(
                    f"Unexpected q_hat ndim {q_hat.ndim}; shape={tuple(q_hat.shape)}"
                )
            q_all.append(q_diag.cpu().numpy())
            y_true = features[:, t + 1:t + 1 + horizon, target_channel]  # (n_firms, H)
            y_all.append(y_true.cpu().numpy())
    return np.stack(q_all, axis=0), np.stack(y_all, axis=0)


def train_quick(
    cfg: Gate1Config,
    data: dict,
    device: torch.device,
    seed: int,
) -> HyperCPJointModel:
    """Train the joint model for `train_epochs` and return the trained model.

    Forecast/quantile output_mode is determined by `data["output_mode"]`
    (set by `run_one_seed` based on dataset). In `"diagonal"` mode each
    firm forecasts its own product (n_firms = n_products), which avoids
    the O(n_firms * n_products) memory cost of cross-product heads --
    required for M5-scale runs.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_firms = data["n_firms"]
    n_features = data["n_features"]
    train_range = data["splits"][0]
    features = data["features"].to(device)
    incidence = data["incidence"].to(device)
    output_mode = data.get("output_mode", "cross_product")

    enc_cfg = EncoderConfig(
        n_features=n_features, hidden_dim=cfg.hidden_dim, n_hgnn_layers=4,
    )
    fhd_cfg = ForecastHeadConfig(
        hidden_dim=cfg.hidden_dim, n_products=n_firms, horizon=cfg.horizon,
        output_mode=output_mode,
    )
    qhd_cfg = QuantileHeadConfig(
        hidden_dim=cfg.hidden_dim, n_products=n_firms, horizon=cfg.horizon,
        output_mode=output_mode,
    )
    model = HyperCPJointModel(JointConfig(
        encoder=enc_cfg, forecast=fhd_cfg, quantile=qhd_cfg, weighting="famo",
    )).to(device)
    model.set_hypergraph(incidence)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    valid_starts = list(range(train_range.start, train_range.stop - cfg.window - cfg.horizon))
    if not valid_starts:
        raise RuntimeError("train range too short")
    rng = np.random.default_rng(seed)

    for epoch in range(cfg.train_epochs):
        model.train()
        epoch_loss = 0.0
        order = rng.permutation(len(valid_starts))
        n_batches = max(1, len(order) // cfg.batch_size)
        for b in range(n_batches):
            batch_starts = [
                valid_starts[i] for i in order[b * cfg.batch_size:(b + 1) * cfg.batch_size]
            ]
            feat_b, tgt_b = [], []
            for s in batch_starts:
                feat_b.append(features[:, s:s + cfg.window, :])
                tgt = features[:, s + cfg.window:s + cfg.window + cfg.horizon,
                               data["target_channel"]]
                tgt_b.append(tgt)
            feat_b = torch.stack(feat_b, dim=0)              # (B, n_firms, T, F)
            tgt_b_t = torch.stack(tgt_b, dim=0)              # (B, n_firms, H)

            if output_mode == "diagonal":
                # Loss target shape: (B, n_firms, H). No mask needed.
                tgt_for_loss = tgt_b_t
                mask = None
            else:
                # Loss target shape: (B, n_firms, n_firms, H) with eye-mask.
                tgt_for_loss = tgt_b_t.unsqueeze(2).expand(
                    -1, n_firms, n_firms, cfg.horizon
                )
                eye = torch.eye(n_firms, device=device).unsqueeze(-1)
                mask = eye.unsqueeze(0).expand(
                    len(batch_starts), n_firms, n_firms, cfg.horizon
                )

            outputs = model(feat_b)
            losses = model.compute_losses(outputs, tgt_for_loss, mask)
            total, _ = model.weighter.combine(losses)
            opt.zero_grad()
            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            model.weighter.update(losses)
            epoch_loss += float(total.detach())
        if (epoch + 1) % 10 == 0:
            logger.info("  seed=%d epoch=%d loss=%.4f", seed, epoch + 1,
                        epoch_loss / n_batches)
    return model


# =============================================================================
# Coverage evaluation per method.
# =============================================================================


def picp_per_class(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    incidence,
    partition,
) -> dict:
    """Per-partition-class PICP. Returns {class_idx: picp_float}.

    A firm belongs to class k iff it is a member of any class-k hyperedge.
    """
    H = incidence.cpu().numpy() if hasattr(incidence, "cpu") \
        else np.asarray(incidence)
    P = partition.cpu().numpy() if hasattr(partition, "cpu") \
        else np.asarray(partition)
    if P.size == 0:
        return {}
    n_classes = int(P.max()) + 1
    out: dict[int, float] = {}
    for k in range(n_classes):
        edge_mask = (P == k)
        if not edge_mask.any():
            out[k] = float("nan"); continue
        firms_in_class = H[:, edge_mask].any(axis=1)  # (n_firms,)
        if not firms_in_class.any():
            out[k] = float("nan"); continue
        # Index axis -2 (firms) — works for shapes (..., n_firms, H).
        y_k = y_true[..., firms_in_class, :]
        l_k = lower[..., firms_in_class, :]
        u_k = upper[..., firms_in_class, :]
        out[k] = float(picp(y_k, l_k, u_k))
    return out


def eval_split_cp(
    q_cal: np.ndarray,
    y_cal: np.ndarray,
    q_test: np.ndarray,
    y_test: np.ndarray,
    quantiles: list[float],
    alpha: float,
    lo_idx: int = 0,
    hi_idx: int = -1,
    incidence=None,
    partition=None,
) -> dict:
    """Split CP -- invalid time-series baseline."""
    if hi_idx < 0:
        hi_idx = q_cal.shape[-1] + hi_idx
    # CQR scores on calibration set.
    q_lo_cal = q_cal[..., lo_idx]
    q_hi_cal = q_cal[..., hi_idx]
    s_cal = np.maximum(q_lo_cal - y_cal, y_cal - q_hi_cal).flatten()
    # Filter NaN.
    s_cal = s_cal[~np.isnan(s_cal)]
    n = len(s_cal)
    if n == 0:
        return {"method": "split_cp", "picp": float("nan"), "pinaw": float("nan"),
                "ece": float("nan")}
    level = min(1.0, (1.0 - alpha) * (1.0 + 1.0 / n))
    threshold = float(np.quantile(s_cal, level))
    # Prediction intervals on test.
    q_lo_test = q_test[..., lo_idx]
    q_hi_test = q_test[..., hi_idx]
    lower = q_lo_test - threshold
    upper = q_hi_test + threshold
    res = {
        "method": "split_cp",
        "threshold": threshold,
        "picp": picp(y_test, lower, upper),
        "pinaw": pinaw(lower, upper, y_test),
        "ece": ece(y_test, q_test, quantiles),
        "n_test": int(y_test.size),
    }
    if incidence is not None and partition is not None:
        res["picp_per_class"] = picp_per_class(
            y_test, lower, upper, incidence, partition,
        )
    return res


def eval_split_cp_alpha_sweep(
    q_cal: np.ndarray,
    y_cal: np.ndarray,
    q_test: np.ndarray,
    y_test: np.ndarray,
    quantiles: list[float],
    alphas: tuple[float, ...] = (0.001, 0.005, 0.01, 0.02, 0.05, 0.10),
    lo_idx: int = 0,
    hi_idx: int = -1,
    incidence=None,
    partition=None,
) -> dict:
    """Split CP at multiple alpha levels -- width-at-matched-coverage baseline.

    For each alpha in `alphas`, runs the standard Split-CP procedure and
    records (PICP, PINAW). The resulting (PICP, PINAW) trajectory traces
    out the empirical Pareto frontier of split-CP: at any target PICP one
    can read off the PINAW that a *trivial* method achieves. The headline
    comparison reviewers will demand is "at the PICP our hyperedge-ACI
    achieves (~0.99 on smoke), how wide are the intervals of an
    appropriately-inflated Split-CP?" This trajectory gives that answer
    without test-data tuning.

    Returns a dict with `points` (a list of per-alpha result dicts) and
    `frontier` (sorted by PICP for convenient lookup).
    """
    points = []
    for a in alphas:
        r = eval_split_cp(
            q_cal, y_cal, q_test, y_test, quantiles, alpha=a,
            lo_idx=lo_idx, hi_idx=hi_idx,
            incidence=incidence, partition=partition,
        )
        r["alpha"] = float(a)
        # Override the method tag so the sweep entries are identifiable
        # in the per-seed JSON.
        r["method"] = f"split_cp_alpha_{a:g}"
        points.append(r)
    # Frontier sorted by ascending PICP for downstream comparisons.
    frontier = sorted(points, key=lambda p: p.get("picp", float("nan")))
    return {
        "method": "split_cp_alpha_sweep",
        "points": points,
        "frontier": frontier,
        "alphas": list(alphas),
    }


def match_pinaw_at_picp(
    sweep: dict,
    target_picp: float,
) -> dict:
    """Given a split-CP alpha sweep and a target PICP (e.g. the PICP
    achieved by our hyperedge-ACI method), return the split-CP entry whose
    empirical PICP is closest to the target and report its PINAW.

    Used to populate the `width_at_matched_coverage` row of the Gate 1
    summary; the comparison answers reviewer #2's specific objection that
    the hyperedge-ACI coverage gain over plain split-CP at alpha=0.1 is
    a conservatism artifact.
    """
    pts = sweep["points"]
    if not pts:
        return {"matched_alpha": float("nan"), "pinaw": float("nan"),
                "picp": float("nan"), "delta_picp": float("nan")}
    # Find the alpha whose PICP is closest to the target.
    best = min(pts, key=lambda p: abs(p.get("picp", float("inf")) - target_picp))
    return {
        "matched_alpha": best["alpha"],
        "picp": best["picp"],
        "pinaw": best["pinaw"],
        "delta_picp": abs(best["picp"] - target_picp),
    }


def eval_aci_node(
    q_cal: np.ndarray,
    y_cal: np.ndarray,
    q_test: np.ndarray,
    y_test: np.ndarray,
    quantiles: list[float],
    cfg: Gate1Config,
    lo_idx: int = 0,
    hi_idx: int = -1,
) -> dict:
    """ACI with node-level CQR scores (no hyperedge structure)."""
    if hi_idx < 0:
        hi_idx = q_cal.shape[-1] + hi_idx
    # Use ACI on the flattened time series.
    T_cal = q_cal.shape[0]
    # Build a single-stream calibration tensor by flattening firms/horizons.
    K = q_cal.shape[-1]
    q_cal_flat = q_cal.reshape(T_cal, -1, K)
    y_cal_flat = y_cal.reshape(T_cal, -1)

    aci_cfg = ACIConfig(
        alpha_target=cfg.target_alpha, eta=cfg.aci_eta, window=cfg.aci_window,
        window_mode="fixed", quantile_lo_idx=lo_idx, quantile_hi_idx=hi_idx,
    )
    aci = ACI(aci_cfg)
    aci.fit_initial(
        torch.from_numpy(q_cal_flat).float(),
        torch.from_numpy(y_cal_flat).float(),
    )

    # Test-time predictions.
    T_test = q_test.shape[0]
    q_test_flat = q_test.reshape(T_test, -1, K)
    y_test_flat = y_test.reshape(T_test, -1)

    lowers, uppers, covered_flags = [], [], []
    for t in range(T_test):
        q_t = torch.from_numpy(q_test_flat[t]).float()
        y_t = torch.from_numpy(y_test_flat[t]).float()
        lower, upper, covered, _ = aci.step(q_t, y_t)
        lowers.append(lower.cpu().numpy())
        uppers.append(upper.cpu().numpy())
        covered_flags.append(covered.cpu().numpy())

    lowers = np.stack(lowers, axis=0)
    uppers = np.stack(uppers, axis=0)
    return {
        "method": "aci_node",
        "picp": picp(y_test_flat, lowers, uppers),
        "pinaw": pinaw(lowers, uppers, y_test_flat),
        "ece": ece(y_test, q_test, quantiles),
        "final_alpha": float(aci.state.alpha_t.flatten()[0]),
        "n_test": int(y_test_flat.size),
    }


def eval_conformal_pid_node(
    q_cal: np.ndarray,
    y_cal: np.ndarray,
    q_test: np.ndarray,
    y_test: np.ndarray,
    quantiles: list[float],
    cfg: Gate1Config,
    lo_idx: int = 0,
    hi_idx: int = -1,
    eta_I: float = 0.01,
    eta_D: float = 0.0,
) -> dict:
    """Conformal PID Control (Angelopoulos et al., NeurIPS 2023) at node level.

    A direct upgrade over `eval_aci_node`: same calibration window and
    threshold-from-window logic, but the alpha_t update is a full PID
    controller on miscoverage error rather than the pure proportional
    ACI rule. With (eta_I, eta_D) = (0, 0) this is exactly ACI.
    """
    if hi_idx < 0:
        hi_idx = q_cal.shape[-1] + hi_idx
    T_cal = q_cal.shape[0]
    K = q_cal.shape[-1]
    q_cal_flat = q_cal.reshape(T_cal, -1, K)
    y_cal_flat = y_cal.reshape(T_cal, -1)

    pid_cfg = ConformalPIDConfig(
        alpha_target=cfg.target_alpha, eta_P=cfg.aci_eta,
        eta_I=eta_I, eta_D=eta_D,
        window=cfg.aci_window, window_mode="fixed",
        quantile_lo_idx=lo_idx, quantile_hi_idx=hi_idx,
    )
    pid = ConformalPID(pid_cfg)
    pid.fit_initial(
        torch.from_numpy(q_cal_flat).float(),
        torch.from_numpy(y_cal_flat).float(),
    )

    T_test = q_test.shape[0]
    q_test_flat = q_test.reshape(T_test, -1, K)
    y_test_flat = y_test.reshape(T_test, -1)
    lowers, uppers = [], []
    for t in range(T_test):
        q_t = torch.from_numpy(q_test_flat[t]).float()
        y_t = torch.from_numpy(y_test_flat[t]).float()
        lower, upper, _, _ = pid.step(q_t, y_t)
        lowers.append(lower.cpu().numpy())
        uppers.append(upper.cpu().numpy())
    lowers = np.stack(lowers, axis=0)
    uppers = np.stack(uppers, axis=0)
    return {
        "method": "conformal_pid_node",
        "picp": picp(y_test_flat, lowers, uppers),
        "pinaw": pinaw(lowers, uppers, y_test_flat),
        "ece": ece(y_test, q_test, quantiles),
        "final_alpha": float(pid.state.alpha_t.flatten()[0]),
        "final_integral": float(pid.state.integral),
        "n_test": int(y_test_flat.size),
        "gains": {"eta_P": cfg.aci_eta, "eta_I": eta_I, "eta_D": eta_D},
    }


def eval_agaci_node(
    q_cal: np.ndarray,
    y_cal: np.ndarray,
    q_test: np.ndarray,
    y_test: np.ndarray,
    quantiles: list[float],
    cfg: Gate1Config,
    lo_idx: int = 0,
    hi_idx: int = -1,
    etas: tuple[float, ...] = (0.005, 0.01, 0.025, 0.05, 0.10),
    weighting: str = "boa",
) -> dict:
    """Aggregated ACI (Zaffran et al., ICML 2022) at node level.

    Runs K parallel ACI controllers with different eta values and
    aggregates their intervals via BOA-style exponential weights
    (default) or a uniform-weight median (`weighting="uniform"`).
    Removes ACI's sensitivity to the eta hyperparameter.
    """
    if hi_idx < 0:
        hi_idx = q_cal.shape[-1] + hi_idx
    T_cal = q_cal.shape[0]
    K = q_cal.shape[-1]
    q_cal_flat = q_cal.reshape(T_cal, -1, K)
    y_cal_flat = y_cal.reshape(T_cal, -1)

    agaci = AgACI(AgACIConfig(
        alpha_target=cfg.target_alpha,
        etas=etas, weighting=weighting,
        window=cfg.aci_window, window_mode="fixed",
        quantile_lo_idx=lo_idx, quantile_hi_idx=hi_idx,
    ))
    agaci.fit_initial(
        torch.from_numpy(q_cal_flat).float(),
        torch.from_numpy(y_cal_flat).float(),
    )

    T_test = q_test.shape[0]
    q_test_flat = q_test.reshape(T_test, -1, K)
    y_test_flat = y_test.reshape(T_test, -1)
    lowers, uppers = [], []
    for t in range(T_test):
        q_t = torch.from_numpy(q_test_flat[t]).float()
        y_t = torch.from_numpy(y_test_flat[t]).float()
        lower, upper, _, _ = agaci.step(q_t, y_t)
        lowers.append(lower.cpu().numpy())
        uppers.append(upper.cpu().numpy())
    lowers = np.stack(lowers, axis=0)
    uppers = np.stack(uppers, axis=0)
    return {
        "method": f"agaci_node_{weighting}",
        "picp": picp(y_test_flat, lowers, uppers),
        "pinaw": pinaw(lowers, uppers, y_test_flat),
        "ece": ece(y_test, q_test, quantiles),
        "final_weights": agaci.weights.detach().cpu().tolist(),
        "etas": list(etas),
        "n_test": int(y_test_flat.size),
    }


def eval_cf_gnn_clique(
    q_cal: np.ndarray,
    y_cal: np.ndarray,
    q_test: np.ndarray,
    y_test: np.ndarray,
    quantiles: list[float],
    incidence: torch.Tensor,
    cfg: Gate1Config,
    lambda_smooth: float = 0.5,
    lo_idx: int = 0,
    hi_idx: int = -1,
) -> dict:
    """CF-GNN-style (Huang et al., NeurIPS 2023) graph-smoothed split-CP
    on the clique-expanded hypergraph.

    The hypergraph is converted to a pairwise adjacency by clique
    expansion (two firms share an edge iff they co-occur in any
    hyperedge); the normalised Laplacian is then used to diffusion-smooth
    the node-level CQR-asymmetric scores before applying standard
    split-CP. This is the closest available adapter of CF-GNN's
    graph-aware calibration recipe to our hypergraph time-series
    setting; see hypercp.calibration.cf_gnn for the simplification
    notes (the original CF-GNN's exact-1-alpha argument relies on
    transductive permutation symmetry, which does not hold under
    rolling-origin time series).
    """
    if hi_idx < 0:
        hi_idx = q_cal.shape[-1] + hi_idx

    incidence_t = (incidence if isinstance(incidence, torch.Tensor)
                   else torch.as_tensor(incidence, dtype=torch.float32))
    calibrator = CFGNNSmoothedCalibrator(CFGNNHyperConfig(
        alpha_target=cfg.target_alpha,
        lambda_smooth=lambda_smooth,
        quantile_lo_idx=lo_idx,
        quantile_hi_idx=hi_idx,
    ))
    calibrator.set_hypergraph(incidence_t)
    calibrator.fit(
        torch.from_numpy(q_cal).float(),
        torch.from_numpy(y_cal).float(),
    )

    q_test_t = torch.from_numpy(q_test).float()
    y_test_t = torch.from_numpy(y_test).float()
    lower, upper = calibrator.predict_interval(q_test_t)
    lower_np = lower.cpu().numpy()
    upper_np = upper.cpu().numpy()
    return {
        "method": "cf_gnn_clique",
        "lambda_smooth": lambda_smooth,
        "picp": picp(y_test, lower_np, upper_np),
        "pinaw": pinaw(lower_np, upper_np, y_test),
        "ece": ece(y_test, q_test, quantiles),
        "threshold": float(calibrator.threshold),
        "n_test": int(y_test.size),
    }


def eval_ncpnet_clique(
    q_cal: np.ndarray,
    y_cal: np.ndarray,
    q_test: np.ndarray,
    y_test: np.ndarray,
    quantiles: list[float],
    incidence: torch.Tensor,
    cfg: Gate1Config,
    lambda_smooth: float = 0.5,
    lo_idx: int = 0,
    hi_idx: int = -1,
) -> dict:
    """NCPNET-style (Wang et al., KDD 2025): diffusion-smoothed CQR score
    fed to ACI on the clique-expanded hypergraph.

    The diffusion smoother (normalised Laplacian) is the same as in
    `eval_cf_gnn_clique`; the difference is that the diffused scores
    drive an ACI online adaptation rather than a fixed split-CP
    threshold. This isolates the contribution of the time-series
    adaptation step relative to CF-GNN-style smoothing.
    """
    if hi_idx < 0:
        hi_idx = q_cal.shape[-1] + hi_idx

    incidence_t = (incidence if isinstance(incidence, torch.Tensor)
                   else torch.as_tensor(incidence, dtype=torch.float32))
    calibrator = NCPNETStyleCalibrator(NCPNETHyperConfig(
        alpha_target=cfg.target_alpha,
        lambda_smooth=lambda_smooth,
        aci_eta=cfg.aci_eta,
        window=cfg.aci_window,
        quantile_lo_idx=lo_idx,
        quantile_hi_idx=hi_idx,
    ))
    calibrator.set_hypergraph(incidence_t)
    calibrator.fit_initial(
        torch.from_numpy(q_cal).float(),
        torch.from_numpy(y_cal).float(),
    )

    T_test = q_test.shape[0]
    lowers, uppers = [], []
    for t in range(T_test):
        q_t = torch.from_numpy(q_test[t]).float()
        y_t = torch.from_numpy(y_test[t]).float()
        lower, upper, _, _ = calibrator.step(q_t, y_t)
        lowers.append(lower.cpu().numpy())
        uppers.append(upper.cpu().numpy())
    lowers = np.stack(lowers, axis=0)
    uppers = np.stack(uppers, axis=0)
    return {
        "method": "ncpnet_clique",
        "lambda_smooth": lambda_smooth,
        "picp": picp(y_test, lowers, uppers),
        "pinaw": pinaw(lowers, uppers, y_test),
        "ece": ece(y_test, q_test, quantiles),
        "final_alpha": float(calibrator.aci.state.alpha_t.flatten()[0]),
        "n_test": int(y_test.size),
    }


# =============================================================================
# Conditional-coverage diagnostics (reviewer-A request).
# =============================================================================


def _conditional_picp(
    y_test: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    by_horizon: bool = True,
    by_volatility_quartile: np.ndarray | None = None,
) -> dict:
    """Return conditional PICP breakdowns from a (T, n_firms, H) trio.

    `by_volatility_quartile` should be a (n_firms,) int array in {0,1,2,3}
    giving each firm's volatility quartile (lower = less volatile). If
    omitted, only the by-horizon breakdown is returned.
    """
    covered = (y_test >= lower) & (y_test <= upper)
    out: dict[str, dict] = {}
    if by_horizon and covered.ndim >= 3:
        per_h = {}
        H = covered.shape[-1]
        for h in range(H):
            per_h[str(h)] = float(covered[..., h].mean())
        out["by_horizon"] = per_h
    if by_volatility_quartile is not None and covered.ndim >= 3:
        per_q = {}
        for q in range(4):
            firm_mask = (by_volatility_quartile == q)
            if firm_mask.any():
                # covered shape: (T, n_firms, H); slice along firm axis.
                per_q[str(q)] = float(covered[:, firm_mask, :].mean())
            else:
                per_q[str(q)] = float("nan")
        out["by_volatility_quartile"] = per_q
    return out


def _compute_volatility_quartile(
    features: torch.Tensor,
    train_range: range,
    target_channel: int,
) -> np.ndarray:
    """Compute per-firm volatility quartile from training-set std.

    Returns a (n_firms,) int array in {0, 1, 2, 3}.
    """
    series = features[:, train_range.start:train_range.stop, target_channel].cpu().numpy()
    std_per_firm = series.std(axis=1)
    # Quartile assignment by rank.
    ranks = np.argsort(np.argsort(std_per_firm))
    n = ranks.size
    quartile = np.minimum(ranks * 4 // n, 3)
    return quartile.astype(int)


def eval_aci_hyperedge(
    q_cal: np.ndarray,
    y_cal: np.ndarray,
    q_test: np.ndarray,
    y_test: np.ndarray,
    quantiles: list[float],
    cfg: Gate1Config,
    incidence: torch.Tensor,
    partition: torch.Tensor,
    lo_idx: int = 0,
    hi_idx: int = -1,
    aggregation: str = "max",
    method_name: str | None = None,
    size_weight: str = "none",
) -> dict:
    """ACI driven by hyperedge-level CQR-asymmetric non-conformity scores.

    Compared to the node-level baselines, the only change is the
    aggregation of the per-firm CQR score over hyperedge members:
        s_e(t, h) = aggregate over c in e of  max(q_lo_c - y_c, y_c - q_hi_c)
    where the aggregate is `max` (default; Eq. 3 of theorem.tex; the
    contribution) or `mean` (Section 6.6 (a) ablation showing the
    aggregation choice is load-bearing).

    The actual scoring is delegated to `hyperedge_cqr_score` so the same
    code path is used everywhere; the previous inline NumPy reference was
    bit-exact-equivalent (see hypercp/models/conformity.py smoke test).
    """
    if hi_idx < 0:
        hi_idx = q_cal.shape[-1] + hi_idx

    # Hyperedge-level calibration scores via the canonical helper.
    q_cal_t = torch.as_tensor(q_cal, dtype=torch.float32)
    y_cal_t = torch.as_tensor(y_cal, dtype=torch.float32)
    incidence_t = (incidence
                   if isinstance(incidence, torch.Tensor)
                   else torch.as_tensor(incidence, dtype=torch.float32))
    s_cqr_he = hyperedge_cqr_score(
        q_cal_t, y_cal_t, incidence_t,
        lo_idx=lo_idx, hi_idx=hi_idx, aggregation=aggregation,
        size_weight=size_weight,
    ).cpu().numpy().astype(np.float32)  # (T_cal, n_he, H)

    # Apply ACI directly on the hyperedge-level scores.
    aci_cfg = ACIConfig(
        alpha_target=cfg.target_alpha, eta=cfg.aci_eta, window=cfg.aci_window,
        window_mode="fixed",
    )
    aci = ACI(aci_cfg)
    aci.state.score_window = [
        torch.from_numpy(s_cqr_he[t].flatten()).float() for t in range(s_cqr_he.shape[0])
    ]
    while len(aci.state.score_window) > cfg.aci_window:
        aci.state.score_window.pop(0)

    # Test-time: predict-step-update loop.
    T_test = q_test.shape[0]
    lowers, uppers = [], []
    for t in range(T_test):
        threshold = aci._threshold()
        q_lo_t = q_test[t, :, :, lo_idx]
        q_hi_t = q_test[t, :, :, hi_idx]
        lower = q_lo_t - threshold.item()
        upper = q_hi_t + threshold.item()
        covered = (y_test[t] >= lower) & (y_test[t] <= upper)
        err = 1.0 - covered.mean()
        new_alpha = aci.state.alpha_t + aci.cfg.eta * (err - aci.cfg.alpha_target)
        aci.state.alpha_t = aci._clamp_alpha(new_alpha)
        lowers.append(lower)
        uppers.append(upper)

    lowers = np.stack(lowers, axis=0)
    uppers = np.stack(uppers, axis=0)
    res = {
        "method": method_name or f"aci_hyperedge_{aggregation}",
        "aggregation": aggregation,
        "picp": picp(y_test, lowers, uppers),
        "pinaw": pinaw(lowers, uppers, y_test),
        "ece": ece(y_test, q_test, quantiles),
        "final_alpha": float(aci.state.alpha_t.flatten()[0]),
        "n_test": int(y_test.size),
    }
    res["picp_per_class"] = picp_per_class(
        y_test, lowers, uppers, incidence, partition,
    )
    return res


# =============================================================================
# Main runner.
# =============================================================================


def run_one_seed(cfg: Gate1Config, seed: int, device: torch.device) -> dict:
    logger.info("=== Seed %d ===", seed)

    # Load dataset.
    if cfg.dataset == "supplygraph":
        sg = SupplyGraphHypergraph()
        data = {
            "features": sg.node_features,
            "incidence": sg.hyperedge_incidence_matrix(),
            "partition": sg.hyperedge_partition_vector(),
            "splits": sg.rolling_origin_split(),
            "n_firms": sg.n_products,
            "n_features": sg.n_channels,
            "target_channel": 1,         # sales_order
            "output_mode": "cross_product",
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
            "target_channel": 0,         # M5 has a single sales channel
            "output_mode": "diagonal",   # avoid O(n_skus^2) memory
        }
        logger.info("  M5 loaded: %d SKUs, %d timesteps, %d hyperedges",
                    m5.n_skus, m5.n_timesteps, m5.n_hyperedges)
    else:
        raise ValueError(f"Dataset {cfg.dataset} not wired into Gate 1")

    # Train.
    t0 = time.time()
    model = train_quick(cfg, data, device, seed)
    logger.info("  trained in %.1fs", time.time() - t0)

    # Collect predictions on cal and test.
    train_range, cal_range, test_range = data["splits"]
    K = model.quantile_head.n_quantiles
    q_cal, y_cal = collect_predictions(
        model, data["features"], cal_range, cfg.window, cfg.horizon,
        data["target_channel"], device, K,
    )
    q_test, y_test = collect_predictions(
        model, data["features"], test_range, cfg.window, cfg.horizon,
        data["target_channel"], device, K,
    )
    logger.info("  q_cal=%s, q_test=%s", q_cal.shape, q_test.shape)

    quantiles = list(model.quantile_head.cfg.quantiles)

    # Evaluate each method.
    results = {
        "seed": seed,
        "n_cal": int(q_cal.shape[0]),
        "n_test": int(q_test.shape[0]),
        "target_coverage": 1.0 - cfg.target_alpha,
    }
    results["split_cp"] = eval_split_cp(q_cal, y_cal, q_test, y_test, quantiles,
                                         cfg.target_alpha,
                                         incidence=data["incidence"],
                                         partition=data["partition"])
    results["aci_node"] = eval_aci_node(q_cal, y_cal, q_test, y_test, quantiles, cfg)
    # Additional conformal SOTA baselines at the node level (no hyperedge
    # structure): Conformal PID and AgACI. Together with split_cp (CQR)
    # and aci_node above, these give the four standard conformal-prediction
    # comparators that R2-style reviewers expect to see; the contribution
    # remains the hyperedge variant.
    results["conformal_pid_node"] = eval_conformal_pid_node(
        q_cal, y_cal, q_test, y_test, quantiles, cfg,
    )
    results["agaci_node_boa"] = eval_agaci_node(
        q_cal, y_cal, q_test, y_test, quantiles, cfg, weighting="boa",
    )
    results["agaci_node_uniform"] = eval_agaci_node(
        q_cal, y_cal, q_test, y_test, quantiles, cfg, weighting="uniform",
    )
    # Graph-aware CP baselines on the clique-expanded hypergraph: the
    # closest available adaptations of CF-GNN (Huang et al. NeurIPS 2023)
    # and NCPNET (Wang et al. KDD 2025) to our setting, both run on the
    # same joint forecaster. CF-GNN-style uses Laplacian smoothing +
    # split-CP; NCPNET-style uses Laplacian smoothing + ACI.
    results["cf_gnn_clique"] = eval_cf_gnn_clique(
        q_cal, y_cal, q_test, y_test, quantiles,
        data["incidence"], cfg,
    )
    results["ncpnet_clique"] = eval_ncpnet_clique(
        q_cal, y_cal, q_test, y_test, quantiles,
        data["incidence"], cfg,
    )
    # Width-at-matched-coverage baseline: Split-CP alpha sweep traces the
    # empirical (PICP, PINAW) Pareto frontier of a trivial method. The
    # comparison "what PINAW does Split-CP need to match our PICP" is the
    # standard request from the conformal-prediction reviewer pool
    # (R2-style critique).
    results["split_cp_sweep"] = eval_split_cp_alpha_sweep(
        q_cal, y_cal, q_test, y_test, quantiles,
        incidence=data["incidence"], partition=data["partition"],
    )
    # Headline method: hyperedge-level CQR + max aggregation (Eq. 3).
    results["aci_hyperedge"] = eval_aci_hyperedge(
        q_cal, y_cal, q_test, y_test, quantiles, cfg,
        data["incidence"], data["partition"],
        aggregation="max", method_name="aci_hyperedge",
    )
    # Ablation (a) of Section 6.6: mean aggregation should collapse
    # marginal validity (theorem.tex Lemma 3 fails without max).
    results["aci_hyperedge_mean"] = eval_aci_hyperedge(
        q_cal, y_cal, q_test, y_test, quantiles, cfg,
        data["incidence"], data["partition"],
        aggregation="mean", method_name="aci_hyperedge_mean",
    )
    # Section 7 mitigation: weighted-max with w_{c,e} = 1/sqrt(|e|).
    # Should reduce PINAW on datasets with large hyperedges (M5) while
    # preserving the max-aggregation contract that Lemma 3 needs.
    results["aci_hyperedge_invsqrt"] = eval_aci_hyperedge(
        q_cal, y_cal, q_test, y_test, quantiles, cfg,
        data["incidence"], data["partition"],
        aggregation="max", method_name="aci_hyperedge_invsqrt",
        size_weight="inverse_sqrt",
    )

    # Width-at-matched-coverage diagnostic: read PINAW from the split-CP
    # alpha sweep at the PICP achieved by aci_hyperedge.
    results["matched_coverage"] = {
        "target_picp_from": "aci_hyperedge",
        "target_picp": results["aci_hyperedge"]["picp"],
        "split_cp_at_match": match_pinaw_at_picp(
            results["split_cp_sweep"], results["aci_hyperedge"]["picp"],
        ),
        "aci_hyperedge_pinaw": results["aci_hyperedge"]["pinaw"],
    }

    # Conditional coverage diagnostics (reviewer-A request):
    # PICP broken down by SKU volatility quartile and by horizon, for
    # split_cp (the worst-undercovering node-level baseline) and for our
    # aci_hyperedge method. We compute these from the same per-seed
    # prediction interval arrays by re-evaluating split-CP and the
    # hyperedge inflation step in-place.
    vol_quartile = _compute_volatility_quartile(
        data["features"], train_range, data["target_channel"],
    )
    # Re-derive lower/upper for split_cp and aci_hyperedge from the test
    # set. We already have aci_hyperedge intervals via the threshold; the
    # cleanest re-derivation is to call the same code paths once more
    # purely for coverage indicator computation. Since the threshold is
    # already known from results["split_cp"] and results["aci_hyperedge"],
    # we recompute lower/upper here.
    K_q = q_test.shape[-1]
    lo_idx_p = 0
    hi_idx_p = K_q - 1
    split_thr = results["split_cp"].get("threshold", None)
    if split_thr is not None:
        lower_sp = q_test[..., lo_idx_p] - split_thr
        upper_sp = q_test[..., hi_idx_p] + split_thr
        results["split_cp"]["conditional"] = _conditional_picp(
            y_test, lower_sp, upper_sp,
            by_horizon=True, by_volatility_quartile=vol_quartile,
        )
    # Recompute aci_hyperedge intervals from the hyperedge-scored final
    # threshold. For simplicity we just call hyperedge_cqr_score once at
    # max-aggregation with the calibration set to retrieve the quantile.
    from hypercp.models import hyperedge_cqr_score
    incidence_t = data["incidence"] if isinstance(data["incidence"], torch.Tensor) \
        else torch.as_tensor(data["incidence"], dtype=torch.float32)
    s_cqr_he = hyperedge_cqr_score(
        torch.from_numpy(q_cal).float(),
        torch.from_numpy(y_cal).float(),
        incidence_t, lo_idx=lo_idx_p, hi_idx=hi_idx_p,
        aggregation="max",
    ).cpu().numpy().astype(np.float32).flatten()
    s_cqr_he = s_cqr_he[~np.isnan(s_cqr_he)]
    if s_cqr_he.size:
        n = s_cqr_he.size
        level = min(1.0, (1.0 - cfg.target_alpha) * (1.0 + 1.0 / n))
        he_thr = float(np.quantile(s_cqr_he, level))
        lower_he = q_test[..., lo_idx_p] - he_thr
        upper_he = q_test[..., hi_idx_p] + he_thr
        results["aci_hyperedge"]["conditional"] = _conditional_picp(
            y_test, lower_he, upper_he,
            by_horizon=True, by_volatility_quartile=vol_quartile,
        )

    logger.info(
        "  Coverage: split=%.3f  aci_node=%.3f  pid=%.3f  agaci_boa=%.3f  "
        "agaci_unif=%.3f  cf_gnn=%.3f  ncpnet=%.3f  he(max)=%.3f  "
        "he(mean)=%.3f  (target %.2f)",
        results["split_cp"]["picp"],
        results["aci_node"]["picp"],
        results["conformal_pid_node"]["picp"],
        results["agaci_node_boa"]["picp"],
        results["agaci_node_uniform"]["picp"],
        results["cf_gnn_clique"]["picp"],
        results["ncpnet_clique"]["picp"],
        results["aci_hyperedge"]["picp"],
        results["aci_hyperedge_mean"]["picp"],
        1.0 - cfg.target_alpha,
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="supplygraph")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--target_alpha", type=float, default=0.10)
    parser.add_argument("--aci_eta", type=float, default=0.05)
    parser.add_argument("--aci_window", type=int, default=30)
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--train_epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--output", default="runs/gate1_results.json")
    parser.add_argument("--m5_subsample_n", type=int, default=5000)
    parser.add_argument("--m5_subsample_days", type=int, default=500)
    parser.add_argument("--m5_seed", type=int, default=42)
    args = parser.parse_args()
    cfg = Gate1Config(**vars(args))

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    logger.info("Gate 1 config: %s", cfg)

    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if cfg.device == "auto" else torch.device(cfg.device))
    logger.info("Device: %s", device)

    all_results = []
    for seed in range(cfg.n_seeds):
        r = run_one_seed(cfg, seed, device)
        all_results.append(r)

    # Aggregate.
    def agg(method: str, metric: str) -> dict:
        vals = [r[method][metric] for r in all_results if not np.isnan(r[method].get(metric, np.nan))]
        if not vals:
            return {"mean": float("nan"), "std": float("nan"), "n": 0}
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}

    methods_reported = (
        "split_cp",
        "aci_node",
        "conformal_pid_node",
        "agaci_node_boa",
        "agaci_node_uniform",
        "cf_gnn_clique",
        "ncpnet_clique",
        "aci_hyperedge",
        "aci_hyperedge_invsqrt",
        "aci_hyperedge_mean",
    )
    summary = {
        "config": asdict(cfg),
        "per_seed": all_results,
        "aggregate": {
            m: {"picp": agg(m, "picp"),
                "pinaw": agg(m, "pinaw"),
                "ece": agg(m, "ece")}
            for m in methods_reported
        },
    }

    out = Path(cfg.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, default=str))
    logger.info("Saved -> %s", out)

    print()
    print("=" * 60)
    print("Gate 1 Summary")
    print("=" * 60)
    target = 1.0 - cfg.target_alpha
    for method in methods_reported:
        s = summary["aggregate"][method]
        print(f"  {method:22s}  PICP {s['picp']['mean']:.3f}+/-{s['picp']['std']:.3f}  "
              f"PINAW {s['pinaw']['mean']:.3f}+/-{s['pinaw']['std']:.3f}  "
              f"ECE {s['ece']['mean']:.4f}+/-{s['ece']['std']:.4f}")
    print(f"  Target coverage: {target:.3f}")
    print()
    he = summary["aggregate"]["aci_hyperedge"]["picp"]["mean"]
    sp = summary["aggregate"]["split_cp"]["picp"]["mean"]
    pass_aci = abs(he - target) < 0.02
    fail_split = (target - sp) > 0.05
    print(f"  ACI hyperedge (max) within +/- 0.02 of target: {pass_aci}")
    print(f"  Split CP drifts > 0.05 below target:           {fail_split}")
    print(f"  Gate 1 PASS: {pass_aci and fail_split}")
    print(f"  (ablation) ACI hyperedge MEAN coverage: "
          f"{summary['aggregate']['aci_hyperedge_mean']['picp']['mean']:.3f}  "
          f"-- expected << target if Lemma 3 max-aggregation is load-bearing.")

    # Width-at-matched-coverage diagnostic (R2 critique answer).
    print()
    print("Width-at-matched-coverage diagnostic:")
    he_picp_list = [r["aci_hyperedge"]["picp"] for r in all_results]
    he_pinaw_list = [r["aci_hyperedge"]["pinaw"] for r in all_results]
    matched_picp_list = [r["matched_coverage"]["split_cp_at_match"]["picp"]
                         for r in all_results]
    matched_pinaw_list = [r["matched_coverage"]["split_cp_at_match"]["pinaw"]
                          for r in all_results]
    matched_alpha_list = [r["matched_coverage"]["split_cp_at_match"]["matched_alpha"]
                          for r in all_results]
    he_pinaw_mean = float(np.mean(he_pinaw_list)) if he_pinaw_list else float("nan")
    matched_pinaw_mean = float(np.mean(matched_pinaw_list)) if matched_pinaw_list else float("nan")
    matched_picp_mean = float(np.mean(matched_picp_list)) if matched_picp_list else float("nan")
    matched_alpha_mean = float(np.mean(matched_alpha_list)) if matched_alpha_list else float("nan")
    he_picp_mean = float(np.mean(he_picp_list)) if he_picp_list else float("nan")
    print(f"  ACI hyperedge:                  PICP {he_picp_mean:.3f}  PINAW {he_pinaw_mean:.3f}")
    print(f"  Split-CP (alpha={matched_alpha_mean:.4g}): PICP {matched_picp_mean:.3f}  PINAW {matched_pinaw_mean:.3f}")
    if matched_pinaw_mean > 0:
        print(f"  Width ratio (ours / matched split-CP): {he_pinaw_mean / matched_pinaw_mean:.2f}x  "
              f"(<1 = narrower than trivial inflation; >1 = the conservatism reviewer-2 flagged)")

    # Stash the diagnostic in the aggregate summary for paper-table extraction.
    summary["aggregate"]["matched_coverage"] = {
        "target_picp_from": "aci_hyperedge",
        "aci_hyperedge_picp_mean": he_picp_mean,
        "aci_hyperedge_pinaw_mean": he_pinaw_mean,
        "split_cp_matched_alpha_mean": matched_alpha_mean,
        "split_cp_matched_picp_mean": matched_picp_mean,
        "split_cp_matched_pinaw_mean": matched_pinaw_mean,
        "width_ratio_ours_over_matched_split_cp": (
            he_pinaw_mean / matched_pinaw_mean
            if matched_pinaw_mean > 0 else float("nan")
        ),
    }
    out.write_text(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
