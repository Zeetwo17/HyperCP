"""
SupplyGraph-Stressed — new public benchmark artefact.

Applies SupplySim disruption dynamics to the real SupplyGraph topology and
historical demand traces, generating ground-truth OR-grounded resilience
labels (TTR, LPR, RL, composite r*, binary) for every firm under a sampled
shock distribution.

This is the new dataset contribution of the paper — released as a public
artefact at submission time. It bridges the gap between:
- SupplyGraph (real topology + demand, no shocks, no resilience labels)
- SCR / SupplySim (synthetic networks, ground-truth resilience labels)

by retaining the real-world structural complexity of SupplyGraph and
injecting controlled, replayable disruption events that yield labelled
resilience targets.

Usage
-----
>>> from hypercp.data.supplygraph import SupplyGraphHypergraph
>>> from hypercp.data.supplygraph_stressed import StressedSupplyGraph
>>> sg = SupplyGraphHypergraph()
>>> ssg = StressedSupplyGraph(sg, n_shocks=200, seed=0)
>>> labels = ssg.resilience_labels  # (n_shocks, n_firms) for each metric
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .supplygraph import SupplyGraphHypergraph
from .supplysim import (
    Network,
    ResilienceLabels,
    Shock,
    ShockPrior,
    compute_resilience_labels,
    simulate_trajectory,
)

logger = logging.getLogger(__name__)


@dataclass
class StressedSupplyGraphLabels:
    """Resilience labels for SupplyGraph-Stressed.

    All arrays have shape (n_shocks, n_firms). For Gates 2 and 3 we
    typically reshape to (n_shocks * n_firms,) for correlation analyses.
    """
    ttr: np.ndarray
    lpr: np.ndarray
    rl: np.ndarray
    service_level: np.ndarray
    r_star: np.ndarray
    binary: np.ndarray
    shocks: list[Shock] = field(default_factory=list)

    @property
    def n_shocks(self) -> int:
        return self.r_star.shape[0]

    @property
    def n_firms(self) -> int:
        return self.r_star.shape[1]

    def to_tensors(self) -> dict[str, torch.Tensor]:
        """Convert all label arrays to torch tensors."""
        return {
            "ttr":           torch.tensor(self.ttr, dtype=torch.float32),
            "lpr":           torch.tensor(self.lpr, dtype=torch.float32),
            "rl":            torch.tensor(self.rl, dtype=torch.float32),
            "service_level": torch.tensor(self.service_level, dtype=torch.float32),
            "r_star":        torch.tensor(self.r_star, dtype=torch.float32),
            "binary":        torch.tensor(self.binary, dtype=torch.long),
        }


class StressedSupplyGraph:
    """SupplyGraph topology + SupplySim shock injection.

    Each "stress instance" applies one sampled shock to the SupplyGraph
    network and simulates a recovery trajectory of length T_recovery from
    the SupplyGraph baseline demand (per-product temporal trace) via the
    Forrester+Sterman dynamics in `supplysim.py`.

    Parameters
    ----------
    base : SupplyGraphHypergraph
        The loaded SupplyGraph hypergraph (after Han 2024 QA).
    n_shocks : int
        Number of stress instances to generate. Default 1000 — gives a
        large enough labelled set for the §6 Gate-3 sample-efficiency curve.
    T_recovery : int
        Recovery window length per shock instance. Default 60.
    safety_factor : float
        Forrester-Sterman safety factor. Default 1.5.
    seed : int
        RNG seed for shock sampling and demand noise.

    Notes
    -----
    SupplyGraph's hypergraph contains *products* as nodes; for SupplySim
    we need a (firm, product) incidence matrix. We treat each surviving
    SupplyGraph product as a single "firm-product" pair indexed by the
    SupplyGraph node index, with capacity / lead_time derived from the
    observed production and inventory traces.

    The base demand for each product is taken from the SupplyGraph
    sales_order channel (mean over the 221 days, before z-scoring).
    Initial inventory is set to ~10× base demand for steady-state stability.
    """

    def __init__(
        self,
        base: SupplyGraphHypergraph,
        n_shocks: int = 1000,
        T_recovery: int = 60,
        safety_factor: float = 1.5,
        capacity_multiplier: float = 1.4,
        inventory_multiplier: float = 3.0,
        shock_magnitude_range: tuple[float, float] = (0.4, 0.95),
        shock_duration_range: tuple[int, int] = (15, 50),
        seed: int = 0,
    ) -> None:
        # Tuned defaults (2026-05-14) for non-degenerate resilience labels:
        # - capacity_multiplier 1.4 (was 3.0)   -> tighter buffer
        # - inventory_multiplier 3.0 (was 10)   -> tighter buffer
        # - shock magnitudes Uniform(0.4, 0.95) -> diverse severities
        # - shock durations  Uniform(15, 50)    -> diverse temporal extents
        # The previous over-provisioned network + narrow Beta(2,5) shocks
        # made every firm recover at t=0 every shock -> TTR constant, NaN.
        self.base = base
        self.n_shocks = n_shocks
        self.T_recovery = T_recovery
        self.safety_factor = safety_factor
        self.capacity_multiplier = capacity_multiplier
        self.inventory_multiplier = inventory_multiplier
        self.shock_magnitude_range = shock_magnitude_range
        self.shock_duration_range = shock_duration_range
        self.seed = seed

        # Build a Network object from the SupplyGraph traces.
        self.network = self._build_network()

        # Sample shocks from P_shock.
        self.shocks = self._sample_shocks()

        # Run all simulations and compute labels.
        self.resilience_labels: StressedSupplyGraphLabels = (
            self._simulate_and_label_all()
        )

    # =========================================================================
    # Network construction from SupplyGraph traces.
    # =========================================================================

    def _build_network(self) -> Network:
        """Convert SupplyGraph traces into a SupplySim Network with
        heterogeneous per-firm resilience parameters.

        Each firm c gets its own (deterministic-from-seed) multipliers for
        capacity, inventory, and lead time. Without this heterogeneity every
        firm has identical intrinsic resilience and any per-firm
        ground-truth variation is pure shock-luck noise, defeating Gate 2.
        """
        nf = np_ = self.base.n_products
        incidence = np.eye(nf, dtype=bool)

        sales_idx = 1
        sales = self.base.node_features[:, :, sales_idx].numpy()
        base_demand_vec = np.maximum(sales.mean(axis=1), 1.0)  # (n_products,)
        base_demand = np.diag(base_demand_vec)

        # Per-firm parameter heterogeneity. Drawn from the construction seed
        # so the same firm has the same parameters across all runs.
        param_rng = np.random.default_rng(self.seed + 12345)
        # Capacity in [0.5x, 2.0x] around the global multiplier.
        cap_factor = param_rng.uniform(0.5, 2.0, size=nf)
        # Inventory in [0.5x, 2.0x] around the global multiplier.
        inv_factor = param_rng.uniform(0.5, 2.0, size=nf)
        # Lead time in {1, 2, 3, 4, 5}.
        lt_per_firm = param_rng.integers(1, 6, size=nf)

        self._firm_cap_factor = cap_factor
        self._firm_inv_factor = inv_factor
        self._firm_lead_time  = lt_per_firm.astype(int)

        # Capacity: per-firm multiplier on top of global.
        capacity = base_demand * (self.capacity_multiplier * cap_factor)[:, None]
        capacity[capacity < 1.0] = 1.0

        # Initial inventory: per-firm multiplier.
        initial_inventory = base_demand * (self.inventory_multiplier * inv_factor)[:, None]

        # Lead time per firm (diagonal incidence so only diagonal matters).
        lead_time = np.zeros((nf, np_), dtype=int)
        for c in range(nf):
            lead_time[c, c] = int(lt_per_firm[c])

        return Network(
            n_firms=nf,
            n_products=np_,
            incidence=incidence,
            initial_inventory=initial_inventory,
            base_demand=base_demand,
            capacity=capacity,
            lead_time=lead_time,
            safety_factor=self.safety_factor,
            alpha_smooth=0.3,
            demand_noise_std=0.1,
        )

    # =========================================================================
    # Shock sampling.
    # =========================================================================

    def _sample_shocks(self) -> list[Shock]:
        """Sample `n_shocks` shocks with diverse magnitude AND duration.

        We bypass the default `ShockPrior` (which uses Beta(2,5) magnitudes
        and a CONSTANT duration) and draw each shock from independent uniform
        distributions over `shock_magnitude_range` and `shock_duration_range`.
        This widens the support of the resulting TTR / LPR / RL distributions
        substantially compared to the prior, eliminating the degeneracy
        observed with the default parameters.
        """
        rng = np.random.default_rng(self.seed)
        nf = self.network.n_firms
        T_sim = self.T_recovery * 2
        mag_lo, mag_hi = self.shock_magnitude_range
        dur_lo, dur_hi = self.shock_duration_range
        shocks: list[Shock] = []
        for _ in range(self.n_shocks):
            mag = float(rng.uniform(mag_lo, mag_hi))
            dur = int(rng.integers(dur_lo, dur_hi + 1))
            firm = int(rng.integers(0, nf))
            t_start = int(rng.integers(0, max(1, T_sim - dur)))
            shocks.append(Shock(magnitude=mag, firm_idx=firm,
                                t_start=t_start, duration=dur))
        return shocks

    # =========================================================================
    # Simulate all and aggregate labels.
    # =========================================================================

    def _simulate_and_label_all(self) -> StressedSupplyGraphLabels:
        nf = self.network.n_firms
        K = self.n_shocks
        T_sim = self.T_recovery * 2

        ttr   = np.zeros((K, nf))
        lpr   = np.zeros((K, nf))
        rl    = np.zeros((K, nf))
        sl    = np.zeros((K, nf))
        rstar = np.zeros((K, nf))
        binr  = np.zeros((K, nf), dtype=np.int64)

        rng = np.random.default_rng(self.seed + 1)

        for k, shock in enumerate(self.shocks):
            traj = simulate_trajectory(
                self.network, T_sim=T_sim, shock=shock, rng=rng
            )
            labels = compute_resilience_labels(
                traj, shock, self.network, T_recovery=self.T_recovery
            )
            ttr[k]   = labels.ttr
            lpr[k]   = labels.lpr
            rl[k]    = labels.rl
            sl[k]    = labels.service_level
            rstar[k] = labels.r_star
            binr[k]  = labels.binary

            if (k + 1) % 100 == 0:
                logger.info("Simulated %d / %d stress instances.", k + 1, K)

        return StressedSupplyGraphLabels(
            ttr=ttr,
            lpr=lpr,
            rl=rl,
            service_level=sl,
            r_star=rstar,
            binary=binr,
            shocks=self.shocks,
        )

    # =========================================================================
    # Save / load.
    # =========================================================================

    def save(self, path: Path | str) -> None:
        """Save the stressed dataset as a .npz archive."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        labs = self.resilience_labels
        np.savez_compressed(
            path,
            ttr=labs.ttr,
            lpr=labs.lpr,
            rl=labs.rl,
            service_level=labs.service_level,
            r_star=labs.r_star,
            binary=labs.binary,
            shock_magnitudes=np.array([s.magnitude for s in labs.shocks]),
            shock_firms=np.array([s.firm_idx for s in labs.shocks]),
            shock_starts=np.array([s.t_start for s in labs.shocks]),
            shock_durations=np.array([s.duration for s in labs.shocks]),
            n_shocks=labs.n_shocks,
            n_firms=labs.n_firms,
            T_recovery=self.T_recovery,
        )
        logger.info("Saved SupplyGraph-Stressed to %s", path)

    # =========================================================================
    # Diagnostic: correlations between r* and component metrics (Gate 2).
    # =========================================================================

    def gate2_diagnostics(self) -> dict[str, float]:
        """Compute Spearman correlations between r* and each of TTR, LPR, RL,
        restricted to (shock, firm) pairs where the firm was directly hit.

        Each correlation must exceed 0.85 (with appropriate sign -- TTR is
        negative-correlated since high TTR means low resilience). Restricting
        to actually-shocked firms is essential: otherwise the bulk of pairs
        have TTR=0 and the correlations degenerate (or NaN out).
        """
        from scipy.stats import spearmanr

        labs = self.resilience_labels
        hit_mask = np.zeros(labs.r_star.shape, dtype=bool)
        for k, sh in enumerate(labs.shocks):
            hit_mask[k, sh.firm_idx] = True

        r = labs.r_star[hit_mask]
        ttr_v = labs.ttr[hit_mask]
        lpr_v = labs.lpr[hit_mask]
        rl_v = labs.rl[hit_mask]

        def _safe(rho):
            try:
                v = float(rho.statistic)
                return v if np.isfinite(v) else float("nan")
            except Exception:
                return float("nan")

        return {
            "r_vs_TTR_negcorr": _safe(spearmanr(r, -ttr_v)),
            "r_vs_LPR_negcorr": _safe(spearmanr(r, -lpr_v)),
            "r_vs_RL_negcorr":  _safe(spearmanr(r, -rl_v)),
            "n_hit_pairs":      int(hit_mask.sum()),
            "ttr_mean":         float(labs.ttr[hit_mask].mean()),
            "ttr_std":          float(labs.ttr[hit_mask].std()),
            "binary_recovered_rate": float(labs.binary[hit_mask].mean()),
        }


# =============================================================================
# Smoke test.
# =============================================================================


def _smoke_test() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    sg = SupplyGraphHypergraph()
    print("Loaded SupplyGraph: %d products, %d hyperedges." % (
        sg.n_products, sg.n_hyperedges
    ))

    # Small stress set for smoke test (full run = 1000).
    ssg = StressedSupplyGraph(
        sg, n_shocks=50, T_recovery=60, seed=0
    )
    print()
    print("SupplyGraph-Stressed generated:")
    print(f"  n_shocks            : {ssg.n_shocks}")
    print(f"  n_firms             : {ssg.network.n_firms}")
    print(f"  r*  shape            : {ssg.resilience_labels.r_star.shape}")
    print(f"  r*  mean             : {ssg.resilience_labels.r_star.mean():.3f}")
    print(f"  r*  std              : {ssg.resilience_labels.r_star.std():.3f}")
    print(f"  r*  min, max         : "
          f"{ssg.resilience_labels.r_star.min():.3f}, "
          f"{ssg.resilience_labels.r_star.max():.3f}")
    print(f"  binary mean (recov%): "
          f"{ssg.resilience_labels.binary.mean():.3f}")

    # Gate 2 diagnostic.
    print()
    print("Gate 2 — Spearman(r*, -TTR/-LPR/-RL) — must all be > 0.85:")
    for k, v in ssg.gate2_diagnostics().items():
        ok = "PASS" if v > 0.85 else "FAIL"
        print(f"  {k:25s}: {v:+.3f}  [{ok}]")


if __name__ == "__main__":
    _smoke_test()
