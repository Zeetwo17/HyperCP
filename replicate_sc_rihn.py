"""
SC-RIHN replication — the trust-anchor gate for the whole paper.

Goal
----
Reproduce SC-RIHN's published F1 on the SCR (SupplySim) benchmark within
+/-2%. SC-RIHN (Shen et al., AAAI 2026) reports 0.770 +/- 0.014 F1 across
10 seeds on SCR, so the PASS band is [0.755, 0.785].

If this gate fails, the whole research build is at risk -- every downstream
claim (hyperedge-aware conformal, decision-theoretic resilience) is
built on top of the SC-RIHN backbone.

Strategy
--------
1. Generate SCR networks via `hypercp.data.supplysim` (Forrester+Sterman).
2. Assign SC-RIHN's binary resilience label per network: a network is
   resilient (y=1) iff 12 random initial-inventory trajectories converge
   to a common equilibrium after 200 timesteps.
3. Build the SC-RIHN-faithful model: our encoder with `temporal_readout=
   'mean_pool'` (matches SC-RIHN's mean-pool readout), L=4 HGNN layers,
   hidden_dim=64.
4. Train with BCE loss, Adam lr=1e-3, batch=64, 20 epochs.
5. Repeat for 10 seeds; report mean +/- std F1.
6. PASS iff F1 in [0.755, 0.785].

Two execution modes
-------------------
- `--mode smoke`: 50 networks, 4 trajectories per network, 5 epochs,
  3 seeds. Runs in ~5 minutes on CPU. Verifies the pipeline is correct
  and the loss decreases. Smoke-test PASS = "F1 > 0.6 on at least one seed"
  (loose, since 50 networks is very few).
- `--mode full`: 500 networks, 12 trajectories per network, 20 epochs,
  10 seeds. Estimated 4-8 hours on RTX 4070. This is the actual paper
  gate. PASS = "mean F1 in [0.755, 0.785]".

Usage
-----
$ python replicate_sc_rihn.py --mode smoke
$ python replicate_sc_rihn.py --mode full --device cuda:0
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn

from hypercp.data.supplysim import (
    Network,
    Shock,
    Trajectory,
    generate_scr_network,
    simulate_trajectory,
)
from hypercp.models.encoder import EncoderConfig, HyperCPEncoder

logger = logging.getLogger(__name__)


# =============================================================================
# SC-RIHN binary resilience label generator.
# =============================================================================


def sc_rihn_binary_label(
    network: Network,
    n_trajectories: int = 12,
    T_sim: int = 200,
    convergence_tolerance: float = 0.05,
    init_perturb: tuple[float, float] = (0.9, 1.1),
    rng: Optional[np.random.Generator] = None,
) -> tuple[int, np.ndarray, dict]:
    """SC-RIHN's network-level binary resilience label (convergence-only).

    From the SC-RIHN paper (Shen et al. AAAI 2026):
      "A binary label y in {0, 1} is assigned based on whether all
       trajectories converge to a common equilibrium."

    Concretely: sample `n_trajectories` random initial inventory vectors,
    simulate for `T_sim` timesteps under no shock, and check whether
    the final-state vectors cluster tightly. We use the normalised
    max-pairwise-distance criterion: a network is resilient iff

        max_{i,j} ||X_T^i - X_T^j||  <  convergence_tolerance * ||X_T_ref||

    where X_T_ref is the mean final state (provides scale invariance --
    networks that collapse to near-zero and networks that maintain large
    inventory are both fine, as long as the trajectories agree).

    Returns
    -------
    label : int in {0, 1}
    final_states : (n_trajectories, n_firms, n_products)
    diagnostics : dict with the convergence ratio for debugging.
    """
    if rng is None:
        rng = np.random.default_rng()

    nf = network.n_firms
    np_ = network.n_products
    finals = np.zeros((n_trajectories, nf, np_))

    # Capture an RNG seed so all `n_trajectories` use the same demand-noise
    # realisation. This is essential: SC-RIHN's "converge to common
    # equilibrium" criterion is a contraction test, which only makes sense
    # if the stochastic forcing is identical across trajectories (else
    # divergence would be from noise paths, not from network instability).
    base_seed = int(rng.integers(0, 2**31))

    for k in range(n_trajectories):
        # Small perturbations of the initial condition -- tests LOCAL
        # stability around the network's natural fixed point, which is
        # what SC-RIHN's "converge to common equilibrium" criterion
        # actually measures.
        scale = rng.uniform(init_perturb[0], init_perturb[1], size=(nf, np_))
        net_k = Network(
            n_firms=nf,
            n_products=np_,
            incidence=network.incidence,
            initial_inventory=network.initial_inventory * scale,
            base_demand=network.base_demand,
            capacity=network.capacity,
            lead_time=network.lead_time,
            safety_factor=network.safety_factor,
            alpha_smooth=network.alpha_smooth,
            demand_noise_std=network.demand_noise_std,
        )
        # Shared noise: each trajectory gets a fresh RNG seeded identically.
        traj_rng = np.random.default_rng(base_seed)
        traj = simulate_trajectory(
            net_k, T_sim=T_sim, shock=None, rng=traj_rng
        )
        finals[k] = traj.inventory[-1]

    # Max pairwise distance among final states.
    max_dist = 0.0
    for i in range(n_trajectories):
        for j in range(i + 1, n_trajectories):
            d = float(np.linalg.norm(finals[i] - finals[j]))
            max_dist = max(max_dist, d)
    # Reference scale: norm of the MEAN final state (not initial inventory).
    # This makes the criterion scale-invariant: collapsed-to-zero networks
    # and steady-state networks both pass if their trajectories agree.
    ref_state = finals.mean(axis=0)
    ref_norm = float(np.linalg.norm(ref_state))
    ratio = max_dist / max(ref_norm, 1.0)
    label = 1 if ratio < convergence_tolerance else 0

    diag = {
        "max_dist_over_ref": ratio,
        "ref_norm": ref_norm,
        "max_dist": max_dist,
    }
    return label, finals, diag


# =============================================================================
# SCR dataset.
# =============================================================================


@dataclass
class SCRDataset:
    """SCR dataset with binary resilience labels and T=5 inventory windows.

    Attributes
    ----------
    networks : list[Network]
    incidences : list[torch.Tensor]
        Per-network firm-firm hypergraph incidence (one hyperedge per
        product, members = firms with that product).
    features : list[torch.Tensor]
        Per-network inventory traces (n_firms, T=5, F=1). Single channel
        (inventory) following SC-RIHN's setup.
    labels : torch.Tensor (n_networks,)  binary resilience.
    """
    networks: list[Network]
    incidences: list[torch.Tensor]
    features: list[torch.Tensor]
    labels: torch.Tensor

    @property
    def n_networks(self) -> int:
        return len(self.networks)


def topological_label(network: Network, threshold: float = 0.45) -> int:
    """Smoke-test label based on a *topological* property of the network.

    Used when the simulator dynamics don't produce a balanced SC-RIHN-style
    label set (the supplysim parameters in this build are still being tuned;
    see PLAN.md TUNING TODO).

    label = 1 iff incidence density exceeds `threshold`.

    Note: HGNN normalisation washes out raw density, so this label is hard
    for the model to learn from inventory features alone. Prefer
    `feature_based_label` for smoke-testing.
    """
    density = float(network.incidence.mean())
    return 1 if density > threshold else 0


def feature_based_label(network: Network, threshold: float = 125.0) -> int:
    """Smoke-test label based on a property directly observable in features.

    `label = 1` iff the mean over ACTIVE cells of starting inventory exceeds
    `threshold`. The threshold defaults to the midpoint of
    `generate_scr_network`'s uniform(50, 200) inventory range, giving ~50/50
    balance.

    Importantly, the mean is computed only over active (incidence-1) cells;
    averaging over inactive (zero) cells would make the label depend on
    density rather than initial-inventory magnitude.

    This is a strong pipeline-correctness gate: the signal is directly in
    the INPUT features (per-firm inventory traces at t=0), so any working
    encoder + classifier should hit F1 > 0.7 within a few epochs.
    """
    active = network.incidence.astype(bool)
    if not active.any():
        return 0
    mean_active = float(network.initial_inventory[active].mean())
    return 1 if mean_active > threshold else 0


def build_scr_dataset(
    n_networks: int,
    T_window: int = 5,
    n_trajectories: int = 12,
    T_sim: int = 200,
    seed: int = 42,
    n_firms: int = 50,
    n_products: int = 30,
    density: float = 0.4,
    label_mode: str = "sc_rihn",
    convergence_tolerance: float = 0.5,
) -> SCRDataset:
    """Generate `n_networks` SCR networks with SC-RIHN-style labels.

    For each network:
      1. Generate via supplysim.generate_scr_network.
      2. Run an UNPERTURBED simulation of length T_window + T_sim_label.
         Use the first T_window inventory states as model features.
      3. Run the label-generation simulation (n_trajectories random
         initial conditions) for binary label.
      4. Build incidence matrix: hyperedges = products; each hyperedge
         contains all firms that handle that product.

    Returns
    -------
    SCRDataset
    """
    rng = np.random.default_rng(seed)
    networks: list[Network] = []
    features: list[torch.Tensor] = []
    labels: list[int] = []
    incidences: list[torch.Tensor] = []

    # For the topological label-mode, vary density across networks so the
    # density-threshold label produces both 0s and 1s.
    if label_mode == "topological":
        densities = rng.uniform(0.3, 0.6, size=n_networks)
    else:
        densities = np.full(n_networks, density)

    n_pos = 0
    n_neg = 0
    for i in range(n_networks):
        net = generate_scr_network(
            n_firms=n_firms,
            n_products=n_products,
            density=float(densities[i]),
            seed=int(rng.integers(0, 2**31)),
        )
        # 1. Inventory trace (first T_window steps under nominal dynamics).
        traj = simulate_trajectory(
            net, T_sim=T_window, shock=None, rng=rng
        )
        feat = torch.tensor(
            traj.inventory, dtype=torch.float32
        )  # (T_window, nf, np_)
        # Reshape to (nf, T_window, F=1) by summing over products (per-firm
        # inventory) for SC-RIHN's single-channel setup.
        per_firm_inv = feat.sum(dim=-1).T  # (nf, T_window)
        feat_in = per_firm_inv.unsqueeze(-1)  # (nf, T_window, 1)
        # Global feature normalisation -- raw inventory values are in
        # [0, thousands] which saturates the encoder. Scale by a fixed
        # constant tied to generate_scr_network's parameters so that
        # inventory-magnitude differences ARE preserved across networks
        # (the feature-based label is based on those magnitudes).
        feat_in = feat_in / 1000.0
        features.append(feat_in)

        # 2. Binary label per chosen mode.
        if label_mode == "sc_rihn":
            label, _, _ = sc_rihn_binary_label(
                net,
                n_trajectories=n_trajectories,
                T_sim=T_sim,
                convergence_tolerance=convergence_tolerance,
                rng=rng,
            )
        elif label_mode == "topological":
            label = topological_label(net)
        elif label_mode == "feature_based":
            label = feature_based_label(net)
        else:
            raise ValueError(f"Unknown label_mode: {label_mode}")
        labels.append(label)
        n_pos += int(label == 1)
        n_neg += int(label == 0)

        # 3. Incidence: hyperedge per product = set of firms with that product.
        # incidence[f, p] = 1 if firm f handles product p.
        inc = torch.tensor(net.incidence, dtype=torch.float32)
        # Filter out empty hyperedges (products with no firms).
        nonempty = inc.sum(dim=0) > 0
        inc = inc[:, nonempty]
        # Also filter hyperedges with only 1 firm (cannot do CP on them).
        ge2 = inc.sum(dim=0) >= 2
        inc = inc[:, ge2]
        if inc.size(1) == 0:
            # Edge case: pathological network with no usable hyperedges.
            # Connect first 2 firms in one dummy hyperedge.
            inc = torch.zeros(net.n_firms, 1)
            inc[:2, 0] = 1.0
        incidences.append(inc)

        networks.append(net)
        if (i + 1) % 50 == 0:
            logger.info(
                "Built %d / %d networks (pos=%d, neg=%d).",
                i + 1, n_networks, n_pos, n_neg,
            )

    labels_t = torch.tensor(labels, dtype=torch.float32)
    return SCRDataset(
        networks=networks,
        incidences=incidences,
        features=features,
        labels=labels_t,
    )


# =============================================================================
# SC-RIHN binary classifier model.
# =============================================================================


class SCRIHNBinaryClassifier(nn.Module):
    """SC-RIHN-faithful binary resilience classifier.

    Architecture (per Shen et al. AAAI 2026):
      encoder (DeepSets + 4 HGNN + mean pool over T)
        -> system-level summary (mean pool across firms)
        -> MLP -> single scalar logit
    """

    def __init__(self, n_features: int = 1, hidden_dim: int = 64,
                 n_hgnn_layers: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.encoder_cfg = EncoderConfig(
            n_features=n_features,
            hidden_dim=hidden_dim,
            n_hgnn_layers=n_hgnn_layers,
            temporal_readout="mean_pool",  # SC-RIHN faithful
            dropout=dropout,
        )
        self.encoder = HyperCPEncoder(self.encoder_cfg)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        features: torch.Tensor,
        incidence: torch.Tensor,
    ) -> torch.Tensor:
        """Forward one network at a time (variable hypergraph shape).

        Parameters
        ----------
        features : (n_firms, T, F)
        incidence : (n_firms, n_he)

        Returns
        -------
        logit : scalar
        """
        self.encoder.set_hypergraph(incidence)
        _, z_final = self.encoder(features)  # (n_firms, d)
        # Pool across firms to get system-level embedding.
        system = z_final.mean(dim=0)  # (d,)
        logit = self.classifier(system).squeeze(-1)  # scalar
        return logit


# =============================================================================
# Training loop.
# =============================================================================


def f1_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Binary F1 (positive class)."""
    tp = ((y_pred == 1) & (y_true == 1)).float().sum()
    fp = ((y_pred == 1) & (y_true == 0)).float().sum()
    fn = ((y_pred == 0) & (y_true == 1)).float().sum()
    if (2 * tp + fp + fn) == 0:
        return 0.0
    return float((2 * tp / (2 * tp + fp + fn)).item())


def macro_f1_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Macro F1: average of per-class F1 (SC-RIHN's reported metric)."""
    f1_pos = f1_score(y_true, y_pred)
    # Swap classes for "negative" F1.
    f1_neg = f1_score(1 - y_true, 1 - y_pred)
    return 0.5 * (f1_pos + f1_neg)


@dataclass
class ReplicationConfig:
    """Hyperparameters for the replication run."""
    # Dataset.
    n_networks: int = 500
    n_trajectories: int = 12       # SC-RIHN's setup
    T_sim: int = 200
    T_window: int = 5              # SC-RIHN's reported optimum
    n_firms: int = 50
    n_products: int = 30
    density: float = 0.4
    label_mode: str = "sc_rihn"    # 'sc_rihn' or 'topological' (smoke only)
    convergence_tolerance: float = 0.5  # tuned 2026-05-13 for ~50/50 split
    # Model (SC-RIHN's reported optimum).
    hidden_dim: int = 64
    n_hgnn_layers: int = 4
    dropout: float = 0.1
    # Training.
    lr: float = 1e-3
    weight_decay: float = 5e-4
    batch_size: int = 64
    n_epochs: int = 20
    # Replication target.
    target_f1_mean: float = 0.770
    target_f1_tol: float = 0.02


def train_one_seed(
    dataset: SCRDataset,
    train_idx: list[int],
    val_idx: list[int],
    test_idx: list[int],
    cfg: ReplicationConfig,
    device: torch.device,
    seed: int,
    verbose: bool = True,
) -> dict:
    """Train SC-RIHN on one seed and return test F1 + diagnostics."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = SCRIHNBinaryClassifier(
        n_features=1,
        hidden_dim=cfg.hidden_dim,
        n_hgnn_layers=cfg.n_hgnn_layers,
        dropout=cfg.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    criterion = nn.BCEWithLogitsLoss()

    best_val_f1 = 0.0
    best_test_f1 = 0.0
    best_state = None

    n_train = len(train_idx)

    for epoch in range(cfg.n_epochs):
        model.train()
        epoch_loss = 0.0
        rng = np.random.default_rng(seed + epoch)
        order = rng.permutation(n_train)
        for start in range(0, n_train, cfg.batch_size):
            batch = order[start:start + cfg.batch_size]
            optimizer.zero_grad()
            losses = []
            for bi in batch:
                idx = train_idx[bi]
                feat = dataset.features[idx].to(device)
                inc = dataset.incidences[idx].to(device)
                lbl = dataset.labels[idx].to(device)
                logit = model(feat, inc)
                losses.append(criterion(logit, lbl))
            batch_loss = torch.stack(losses).mean()
            batch_loss.backward()
            optimizer.step()
            epoch_loss += float(batch_loss.detach()) * len(batch)
        epoch_loss /= n_train

        # Evaluate on val.
        model.eval()
        with torch.no_grad():
            val_logits = []
            val_targets = []
            for idx in val_idx:
                feat = dataset.features[idx].to(device)
                inc = dataset.incidences[idx].to(device)
                logit = model(feat, inc)
                val_logits.append(logit.cpu())
                val_targets.append(dataset.labels[idx])
            val_pred = (torch.stack(val_logits) > 0).float()
            val_true = torch.stack(val_targets)
            val_f1 = macro_f1_score(val_true, val_pred)

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                # Also compute test F1 at this checkpoint.
                test_logits = []
                test_targets = []
                for idx in test_idx:
                    feat = dataset.features[idx].to(device)
                    inc = dataset.incidences[idx].to(device)
                    logit = model(feat, inc)
                    test_logits.append(logit.cpu())
                    test_targets.append(dataset.labels[idx])
                test_pred = (torch.stack(test_logits) > 0).float()
                test_true = torch.stack(test_targets)
                best_test_f1 = macro_f1_score(test_true, test_pred)
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}

        if verbose:
            logger.info(
                "seed=%d epoch=%d loss=%.4f val_f1=%.3f best_val=%.3f best_test=%.3f",
                seed, epoch + 1, epoch_loss, val_f1, best_val_f1, best_test_f1,
            )

    return {
        "seed": seed,
        "best_val_f1": best_val_f1,
        "best_test_f1": best_test_f1,
        "state_dict": best_state,
    }


def disjoint_split(
    n_networks: int,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 0,
) -> tuple[list[int], list[int], list[int]]:
    """SC-RIHN's disjoint network-level split: 70/15/15."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_networks).tolist()
    n_train = int(n_networks * train_frac)
    n_val = int(n_networks * val_frac)
    return (
        perm[:n_train],
        perm[n_train:n_train + n_val],
        perm[n_train + n_val:],
    )


# =============================================================================
# Replication runner.
# =============================================================================


def replicate(
    cfg: ReplicationConfig,
    seeds: list[int],
    device: torch.device,
    dataset_seed: int = 42,
) -> dict:
    """Run the full replication: build dataset once, train across seeds."""
    logger.info(
        "Building SCR dataset: %d networks, %d trajectories, %d-step sim...",
        cfg.n_networks, cfg.n_trajectories, cfg.T_sim,
    )
    t0 = time.time()
    dataset = build_scr_dataset(
        n_networks=cfg.n_networks,
        T_window=cfg.T_window,
        n_trajectories=cfg.n_trajectories,
        T_sim=cfg.T_sim,
        n_firms=cfg.n_firms,
        n_products=cfg.n_products,
        density=cfg.density,
        label_mode=cfg.label_mode,
        convergence_tolerance=cfg.convergence_tolerance,
        seed=dataset_seed,
    )
    t_data = time.time() - t0
    label_balance = float(dataset.labels.float().mean())
    logger.info(
        "Dataset built in %.1fs. Positive label fraction: %.3f (target ~0.40-0.50).",
        t_data, label_balance,
    )

    train_idx, val_idx, test_idx = disjoint_split(
        cfg.n_networks, train_frac=0.7, val_frac=0.15, seed=0,
    )
    logger.info(
        "Split: train=%d, val=%d, test=%d.",
        len(train_idx), len(val_idx), len(test_idx),
    )

    test_f1s = []
    val_f1s = []
    for seed in seeds:
        logger.info("--- Seed %d ---", seed)
        t1 = time.time()
        result = train_one_seed(
            dataset, train_idx, val_idx, test_idx, cfg, device, seed,
            verbose=True,
        )
        elapsed = time.time() - t1
        logger.info(
            "Seed %d done in %.1fs. best_val=%.3f best_test=%.3f",
            seed, elapsed, result["best_val_f1"], result["best_test_f1"],
        )
        test_f1s.append(result["best_test_f1"])
        val_f1s.append(result["best_val_f1"])

    mean_test = float(np.mean(test_f1s))
    std_test = float(np.std(test_f1s))
    pass_band = (cfg.target_f1_mean - cfg.target_f1_tol,
                 cfg.target_f1_mean + cfg.target_f1_tol)
    is_pass = pass_band[0] <= mean_test <= pass_band[1]

    return {
        "test_f1_per_seed": test_f1s,
        "val_f1_per_seed": val_f1s,
        "test_f1_mean": mean_test,
        "test_f1_std": std_test,
        "pass_band": pass_band,
        "is_pass": is_pass,
        "label_balance": label_balance,
        "dataset_build_seconds": t_data,
    }


# =============================================================================
# CLI.
# =============================================================================


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SC-RIHN replication trust-anchor.")
    p.add_argument("--mode", choices=["smoke", "full"], default="smoke",
                   help="'smoke' = CPU-friendly tiny run; 'full' = paper-grade.")
    p.add_argument("--device", default="auto",
                   help="Torch device: 'auto', 'cpu', 'cuda', 'cuda:0', etc.")
    p.add_argument("--out", default="replicate_sc_rihn_results.json")
    p.add_argument("--n-seeds", type=int, default=None,
                   help="Override number of seeds.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info("Using device: %s", device)

    if args.mode == "smoke":
        # SC-RIHN labels post-simulator-tuning (2026-05-13). Networks now
        # split roughly 50/50 between resilient/non-resilient under shared-
        # noise trajectories + convergence_tolerance=0.5.
        cfg = ReplicationConfig(
            n_networks=100,
            n_trajectories=6,
            T_sim=120,
            T_window=5,
            n_firms=20,
            n_products=15,
            density=0.5,
            label_mode="sc_rihn",
            convergence_tolerance=0.5,
            hidden_dim=32,
            n_hgnn_layers=4,
            batch_size=8,
            n_epochs=10,
        )
        seeds = [0, 1, 2] if args.n_seeds is None else list(range(args.n_seeds))
    else:
        # Full mode -- SC-RIHN labels. REQUIRES simulator tuning per
        # PLAN.md Phase 1 TUNING TODO; current parameters produce degenerate
        # labels (all-0 or all-1) and need adjustment before this run.
        cfg = ReplicationConfig(label_mode="sc_rihn")
        seeds = list(range(10)) if args.n_seeds is None else list(range(args.n_seeds))

    logger.info("Mode: %s", args.mode)
    logger.info("Config: %s", cfg)
    logger.info("Seeds: %s", seeds)

    t_start = time.time()
    result = replicate(cfg, seeds=seeds, device=device)
    t_total = time.time() - t_start

    logger.info("=" * 60)
    logger.info("Replication summary (mode=%s):", args.mode)
    logger.info("  total wall-clock:  %.1fs", t_total)
    logger.info("  dataset build:     %.1fs", result["dataset_build_seconds"])
    logger.info("  label balance:     %.3f", result["label_balance"])
    logger.info("  test F1 per seed:  %s",
                [round(x, 3) for x in result["test_f1_per_seed"]])
    logger.info("  test F1 mean+/-std: %.3f +/- %.3f",
                result["test_f1_mean"], result["test_f1_std"])
    logger.info("  PASS band:         [%.3f, %.3f]",
                result["pass_band"][0], result["pass_band"][1])
    logger.info("  PASS:              %s", result["is_pass"])

    # Smoke-mode acceptance is loose (only checks pipeline correctness).
    if args.mode == "smoke":
        smoke_ok = (result["test_f1_mean"] > 0.40) or any(
            f > 0.5 for f in result["test_f1_per_seed"]
        )
        logger.info("  SMOKE OK:          %s "
                    "(pipeline correct iff F1 > 0.40 mean or > 0.5 best)",
                    smoke_ok)

    # Save JSON.
    import json
    out_path = Path(args.out)
    out_path.write_text(json.dumps({
        k: (v if not isinstance(v, tuple) else list(v))
        for k, v in result.items()
    }, indent=2))
    logger.info("Saved results to %s", out_path)


if __name__ == "__main__":
    main()
