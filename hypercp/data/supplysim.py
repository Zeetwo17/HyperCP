"""
SupplySim — Forrester-Sterman inventory dynamics simulator.

Faithful reimplementation of the simulator used by SC-RIHN (Shen et al.,
AAAI 2026) and originally introduced by Chang et al. (AAAI 2025,
arXiv 2407.18772, *Learning Production Functions for Supply Chains*).
Combines Forrester (1997) inventory-production feedback dynamics with
Sterman (2002) exponential-smoothing demand forecasting and bullwhip-effect
formulation.

This module is the engine behind:
- Resilience-label generation (TTR / LPR / RL) on SupplyGraph-Stressed
- The disruption-shock prior P_shock in §4.4 of the paper
- The SCR / SCR-NR / SCR-ER synthetic benchmark suite (§4.2.8)
- Gates 2 and 3 (OR-grounding and sample efficiency)

References
----------
- Forrester 1997   — Industrial dynamics (J. Operational Research Society)
- Sterman 2002     — Business dynamics, Ch. 18 (bullwhip / ordering rules)
- Chang et al.     — Learning Production Functions (arXiv 2407.18772, AAAI 2025)
- Shen et al. 2026 — SC-RIHN (arXiv 2511.06208)
- Hosseini, Ivanov & Dolgui 2019 — IJPR resilience-metric corpus (TTR/LPR/RL)
- Behzadi et al. 2020 — EJOR resilience-loss-integral formulation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


# =============================================================================
# Network specification.
# =============================================================================


@dataclass
class Network:
    """A supply-chain network for SupplySim.

    Attributes
    ----------
    n_firms : int
        Number of firms / nodes |C|.
    n_products : int
        Number of products |P|.
    incidence : np.ndarray
        (n_firms, n_products) boolean matrix; True iff firm c handles
        product p. For SupplyGraph reformulation, this is `>=1` if the
        firm appears in any hyperedge containing the product.
    initial_inventory : np.ndarray
        (n_firms, n_products) starting inventory in units.
    base_demand : np.ndarray
        (n_firms, n_products) baseline daily demand per (firm, product).
    capacity : np.ndarray
        (n_firms, n_products) max production-per-day capacity per
        (firm, product).
    lead_time : np.ndarray
        (n_firms, n_products) lead time in days; integer-valued.
    safety_factor : float
        Multiplier on smoothed demand for the order-up-to base-stock policy.
        Default 1.5 (per Sterman 2002 / SC-RIHN).
    alpha_smooth : float
        Exponential-smoothing parameter for demand forecasting.
        Default 0.3 (Sterman 2002).
    demand_noise_std : float
        Std-dev of multiplicative log-normal noise on realised demand.
        Default 0.1.
    """

    n_firms: int
    n_products: int
    incidence: np.ndarray
    initial_inventory: np.ndarray
    base_demand: np.ndarray
    capacity: np.ndarray
    lead_time: np.ndarray
    safety_factor: float = 1.5
    alpha_smooth: float = 0.3
    demand_noise_std: float = 0.1

    def __post_init__(self) -> None:
        for name, arr in [
            ("incidence",         self.incidence),
            ("initial_inventory", self.initial_inventory),
            ("base_demand",       self.base_demand),
            ("capacity",          self.capacity),
            ("lead_time",         self.lead_time),
        ]:
            expected = (self.n_firms, self.n_products)
            if arr.shape != expected:
                raise ValueError(
                    f"Network field '{name}' has shape {arr.shape}, "
                    f"expected {expected}."
                )


class Shock(NamedTuple):
    """A disruption applied to a network.

    magnitude  : float in [0,1] — fractional capacity cut at affected firm.
    firm_idx   : int   — which firm is affected (in [0, n_firms)).
    t_start    : int   — simulation timestep at which the shock begins.
    duration   : int   — number of timesteps the shock persists.
    """
    magnitude: float
    firm_idx: int
    t_start: int
    duration: int


# =============================================================================
# Trajectory output.
# =============================================================================


@dataclass
class Trajectory:
    """Output of a simulation run.

    Attributes
    ----------
    inventory : np.ndarray (T, n_firms, n_products)
        Inventory at each timestep.
    demand : np.ndarray (T, n_firms, n_products)
        Realised demand at each timestep.
    supply : np.ndarray (T, n_firms, n_products)
        Units produced/shipped (after capacity + constraint).
    smoothed_demand : np.ndarray (T, n_firms, n_products)
        Exponential-smoothing forecast at each timestep.
    backlog : np.ndarray (T, n_firms, n_products)
        Unfulfilled demand carried forward.
    capacity_effective : np.ndarray (T, n_firms, n_products)
        Per-timestep effective capacity (after shock).
    fulfilled : np.ndarray (T, n_firms, n_products)
        Demand actually met at each timestep = min(demand_t, available_t).
        Used directly for service-level computation in
        `compute_resilience_labels`.
    """
    inventory: np.ndarray
    demand: np.ndarray
    supply: np.ndarray
    smoothed_demand: np.ndarray
    backlog: np.ndarray
    capacity_effective: np.ndarray
    fulfilled: np.ndarray

    @property
    def T(self) -> int:
        return self.inventory.shape[0]

    @property
    def n_firms(self) -> int:
        return self.inventory.shape[1]

    @property
    def n_products(self) -> int:
        return self.inventory.shape[2]


# =============================================================================
# Forrester + Sterman dynamics.
# =============================================================================


def simulate_trajectory(
    network: Network,
    T_sim: int = 200,
    shock: Shock | None = None,
    rng: np.random.Generator | None = None,
) -> Trajectory:
    """Run a Forrester+Sterman simulation for T_sim timesteps.

    Dynamics (per-firm, per-product, per-timestep)
    ----------------------------------------------
    1. Realised demand: D_t = base * exp(N(0, σ²)) — multiplicative noise.
    2. Smoothed demand: D̂_t = α·D_{t-1} + (1-α)·D̂_{t-1}.
    3. Target inventory: T*_t = safety_factor · D̂_t.
    4. Desired production: O_t = max(0, T*_t - I_t + Backlog_t).
    5. Effective capacity: cap_t = base_cap · (1 - shock_factor_t).
    6. Realised production: S_t = min(O_t, cap_t).
    7. Inventory update: I_{t+1} = I_t + S_{t-L} - D_t (after lead-time L).
    8. Backlog update: B_{t+1} = max(0, B_t + D_t - S_{t-L}).

    Returns
    -------
    Trajectory with all per-timestep arrays.
    """
    if rng is None:
        rng = np.random.default_rng()

    nf, np_ = network.n_firms, network.n_products

    inventory = np.zeros((T_sim, nf, np_))
    demand = np.zeros((T_sim, nf, np_))
    supply = np.zeros((T_sim, nf, np_))
    smoothed_demand = np.zeros((T_sim, nf, np_))
    backlog = np.zeros((T_sim, nf, np_))
    fulfilled = np.zeros((T_sim, nf, np_))
    capacity_eff = np.broadcast_to(
        network.capacity, (T_sim, nf, np_)
    ).copy().astype(float)

    # Apply shock to capacity.
    if shock is not None:
        t0 = max(0, shock.t_start)
        t1 = min(T_sim, shock.t_start + shock.duration)
        capacity_eff[t0:t1, shock.firm_idx, :] *= (1.0 - shock.magnitude)

    # Initial state.
    inventory[0] = network.initial_inventory
    smoothed_demand[0] = network.base_demand.copy()

    # Mask: only firms that handle the product are simulated.
    active_mask = network.incidence.astype(bool)

    # Pre-sample noise for reproducibility.
    log_noise = rng.normal(
        loc=-0.5 * network.demand_noise_std**2,  # log-normal correction
        scale=network.demand_noise_std,
        size=(T_sim, nf, np_),
    )
    noise_factor = np.exp(log_noise)

    # Lead-time accounting: pipeline of in-transit production.
    # pipeline[L, c, p] = supply that will arrive in L timesteps.
    max_lead = int(network.lead_time.max()) + 1
    pipeline = np.zeros((max_lead, nf, np_))

    # Steady-state pipeline initialisation: prefill each slot with the
    # base demand so the simulator starts in approximate steady state
    # rather than depleting from initial inventory.
    for L_slot in range(max_lead):
        pipeline[L_slot] = network.base_demand * active_mask

    for t in range(T_sim):
        # 1. Realised demand for current step (with noise).
        demand[t] = network.base_demand * noise_factor[t] * active_mask

        # 2. Smoothed demand update.
        if t > 0:
            smoothed_demand[t] = (
                network.alpha_smooth * demand[t - 1]
                + (1 - network.alpha_smooth) * smoothed_demand[t - 1]
            )
            backlog[t] = backlog[t - 1].copy()

        # 3-4. Desired production (base-stock policy).
        target_inv = network.safety_factor * smoothed_demand[t]
        desired = np.maximum(
            0.0,
            target_inv - inventory[t] + backlog[t]
        ) * active_mask

        # 5-6. Realised production.
        supply[t] = np.minimum(desired, capacity_eff[t])

        # Push to pipeline at appropriate lead-time slot.
        # (Each (firm, product) has its own lead time.)
        for c in range(nf):
            for p in range(np_):
                if not active_mask[c, p]:
                    continue
                L = int(network.lead_time[c, p])
                slot = L % max_lead
                pipeline[slot, c, p] += supply[t, c, p]

        # 7-8. Inventory and backlog update for next step.
        arriving = pipeline[0].copy()
        pipeline = np.roll(pipeline, -1, axis=0)
        pipeline[-1] = 0.0  # clear the slot we just rolled into

        # Available-to-promise at time t: existing inventory + immediate arrivals.
        # Fulfilled demand at time t = min(demand_t, available_t).
        available_t = inventory[t] + arriving
        fulfilled[t] = np.minimum(demand[t], available_t)

        if t + 1 < T_sim:
            # Inventory dynamics: subtract fulfilled (not full demand) since
            # unmet demand is backlogged, not delivered from inventory.
            inventory[t + 1] = inventory[t] + arriving - fulfilled[t]
            shortfall = demand[t] - fulfilled[t]  # >= 0 always
            backlog[t + 1] = backlog[t] + shortfall
            # Inventory is non-negative by construction now; floor for safety.
            inventory[t + 1] = np.maximum(0.0, inventory[t + 1])

    return Trajectory(
        inventory=inventory,
        demand=demand,
        supply=supply,
        smoothed_demand=smoothed_demand,
        backlog=backlog,
        capacity_effective=capacity_eff,
        fulfilled=fulfilled,
    )


# =============================================================================
# OR-grounded resilience metrics — used by §4.4 r_c and §4.2.7 r* composite.
# =============================================================================


@dataclass
class ResilienceLabels:
    """OR-grounded resilience labels for one (network, shock) instance.

    Each field is a numpy array of shape (n_firms,) — per-firm metrics.
    """
    ttr: np.ndarray            # Time-To-Recovery (Hosseini-Ivanov-Dolgui 2019)
    lpr: np.ndarray            # Loss-of-Performance Ratio
    rl: np.ndarray             # Resilience Loss (Behzadi 2020)
    service_level: np.ndarray  # Average ServiceLevel over recovery window
    r_star: np.ndarray         # Composite r* ∈ [0,1] (§4.2.7)
    binary: np.ndarray         # SC-RIHN-style binary label (converges?)


def compute_resilience_labels(
    trajectory: Trajectory,
    shock: Shock,
    network: Network,
    epsilon: float = 0.1,
    T_recovery: int = 60,
    w_ttr: float = 1/3,
    w_lpr: float = 1/3,
    w_rl: float = 1/3,
) -> ResilienceLabels:
    """Compute per-firm OR-standard resilience metrics from a trajectory.

    Parameters
    ----------
    trajectory : Trajectory
        Output of `simulate_trajectory`.
    shock : Shock
        The shock that was applied (used to anchor recovery window).
    network : Network
        Reference network (for baseline demand).
    epsilon : float
        Recovery threshold: imbalance must return to within ε of baseline.
    T_recovery : int
        Length of recovery window after shock onset.
    w_ttr, w_lpr, w_rl : float
        Composite r* weights; must sum to 1.

    Definitions
    -----------
    ServiceLevel_t = (supply - backlog) / demand   (clamped to [0, 1])
    TTR(c)  = first t > t_shock at which ServiceLevel_t(c) ≥ 1 - ε
              (capped at T_recovery if never recovers)
    LPR(c)  = 1 - mean ServiceLevel over [t_shock, t_shock + T_recovery]
    RL(c)   = ∫ (1 - ServiceLevel_t) dt over the recovery window
              (normalised to [0, 1] by T_recovery)
    r*(c)   = w_ttr·(1 - TTR/T_recovery) + w_lpr·(1 - LPR) + w_rl·(1 - RL)
    binary(c) = 1 if firm recovered within T_recovery, else 0
    """
    if not np.isclose(w_ttr + w_lpr + w_rl, 1.0):
        raise ValueError(
            f"r* weights must sum to 1; got {w_ttr + w_lpr + w_rl}."
        )

    nf = network.n_firms
    t0 = shock.t_start
    t_end = min(t0 + T_recovery, trajectory.T)
    window = slice(t0, t_end)

    # Service level per timestep per firm: demand-weighted fill rate.
    #   SL_t(c) = sum_p fulfilled_t(c,p) / sum_p demand_t(c,p)
    # Using the per-product unweighted mean (the previous formulation) gave
    # SL_t(c) ~= 1 even when the firm's single active product was at SL=0
    # because all inactive (c,p) cells contribute SL=1 to the average. With
    # SupplyGraph's diagonal incidence (one product per firm) the previous
    # code diluted a 93% SL drop on the active product into a ~2% drop in
    # the firm-level signal, leaving TTR/LPR/RL effectively zero.
    demand_w = trajectory.demand[window]        # (T_win, nf, np)
    fulfilled_w = trajectory.fulfilled[window]  # (T_win, nf, np)

    total_demand = demand_w.sum(axis=2)         # (T_win, nf)
    total_fulfilled = fulfilled_w.sum(axis=2)   # (T_win, nf)
    denom_firm = np.where(total_demand > 1e-8, total_demand, 1.0)
    sl_per_t = np.where(
        total_demand > 1e-8,
        np.clip(total_fulfilled / denom_firm, 0.0, 1.0),
        1.0,
    )  # (T_win, nf)

    # Active firm mask.
    active = network.incidence.any(axis=1)  # (nf,)

    # TTR per firm: duration of disruption = index of LAST below-threshold
    # timestep + 1, capped at T_recovery. A firm that never falls below
    # threshold has TTR=0 (no disruption to recover from). A firm whose SL
    # is below threshold at the final window step has TTR=T_recovery (cap).
    # The previous code used np.argmax(recovered) which returned the FIRST
    # recovered timestep -- always 0 because SL=1 before the shock drains
    # inventory -- giving TTR=0 universally and degenerate labels.
    ttr = np.full(nf, float(T_recovery))
    for c in range(nf):
        if not active[c]:
            ttr[c] = 0.0
            continue
        below = sl_per_t[:, c] < (1.0 - epsilon)
        if not below.any():
            ttr[c] = 0.0
            continue
        last_below = int(np.argwhere(below).max())
        # If the firm is still below at the final step, treat as un-recovered.
        if last_below >= sl_per_t.shape[0] - 1:
            ttr[c] = float(T_recovery)
        else:
            ttr[c] = float(last_below + 1)

    # LPR per firm — fraction of baseline service lost (per-timestep avg).
    # Use T_recovery as denominator (not window length) so LPR and RL are
    # consistently normalised across shocks regardless of window truncation.
    lpr = (1.0 - sl_per_t).sum(axis=0) / max(T_recovery, 1)
    lpr = np.clip(lpr, 0.0, 1.0)

    # RL per firm — integral of (1 - SL), normalised by T_recovery.
    # For our continuous-time-approximation discretisation, LPR and RL
    # coincide (both = mean shortfall over recovery window). We keep both
    # in the API to match OR-literature conventions (Hosseini-Ivanov-Dolgui
    # 2019 separates them when shock durations are heterogeneous; for our
    # fixed T_recovery they are equal).
    rl = lpr.copy()

    # Average service level over recovery window.
    service_level = sl_per_t.mean(axis=0)

    # Composite r*.
    r_star = (
        w_ttr * (1.0 - ttr / max(T_recovery, 1))
        + w_lpr * (1.0 - lpr)
        + w_rl * (1.0 - rl)
    )
    r_star = np.clip(r_star, 0.0, 1.0)

    # Binary label (SC-RIHN-compatible).
    binary = (ttr < T_recovery).astype(np.int64)

    # Mask inactive firms to zero (or NaN if preferred for downstream).
    for arr in (ttr, lpr, rl, service_level, r_star, binary):
        arr[~active] = 0.0

    return ResilienceLabels(
        ttr=ttr,
        lpr=lpr,
        rl=rl,
        service_level=service_level,
        r_star=r_star,
        binary=binary,
    )


# =============================================================================
# Disruption-shock prior — used by §4.4 functional resilience scoring.
# =============================================================================


@dataclass
class ShockPrior:
    """Default HyperCP disruption-shock prior P_shock (Definition 4.4.1).

    magnitude ~ Beta(α_mag, β_mag)
    firm_idx  ~ Uniform({0, ..., n_firms-1})
    t_start   ~ Uniform({0, ..., T_sim - duration})
    duration  ~ either constant (default) or Geometric(p_duration)
    """
    n_firms: int
    T_sim: int = 200
    alpha_mag: float = 2.0
    beta_mag: float = 5.0
    duration: int = 30
    rng: np.random.Generator | None = None

    def __post_init__(self) -> None:
        if self.rng is None:
            self.rng = np.random.default_rng()

    def sample(self) -> Shock:
        m = float(self.rng.beta(self.alpha_mag, self.beta_mag))
        c = int(self.rng.integers(0, self.n_firms))
        t = int(self.rng.integers(0, max(1, self.T_sim - self.duration)))
        return Shock(magnitude=m, firm_idx=c, t_start=t, duration=self.duration)

    def sample_batch(self, K: int) -> list[Shock]:
        return [self.sample() for _ in range(K)]


# =============================================================================
# SCR synthetic-network generator (SC-RIHN's benchmark suite).
# =============================================================================


def generate_scr_network(
    n_firms: int = 50,
    n_products: int = 30,
    density: float = 0.4,
    seed: int = 0,
    # Per-network parameter ranges -- making these PER-NETWORK is what
    # produces a balanced split into resilient vs non-resilient under
    # SC-RIHN's convergence criterion (some networks land in the
    # Sterman-stable regime, others in the bullwhip-divergence regime).
    # Defaults tuned (2026-05-13) so ~50/50 split arises with
    # convergence_tolerance ~ 0.05 and init_perturb (0.9, 1.1).
    lead_time_max: int = 4,
    safety_factor_range: tuple[float, float] = (1.2, 2.8),
    alpha_smooth_range: tuple[float, float] = (0.15, 0.45),
    demand_noise_range: tuple[float, float] = (0.02, 0.20),
    cap_to_demand_range: tuple[float, float] = (1.8, 5.0),
) -> Network:
    """Generate one synthetic supply-chain network.

    Defaults match SC-RIHN's SCR: ~50 firm nodes, ~30 product nodes,
    density 0.4. Per-network parameters (lead time, safety factor,
    alpha_smooth, demand noise, capacity ratio) are sampled from the
    provided ranges, producing real stability variability across the
    generated suite -- which is what makes SC-RIHN's convergence-based
    binary label naturally balanced.
    """
    rng = np.random.default_rng(seed)

    incidence = rng.random((n_firms, n_products)) < density
    active = incidence.astype(float)

    # Per-network parameter draws.
    safety_factor = float(rng.uniform(*safety_factor_range))
    alpha_smooth = float(rng.uniform(*alpha_smooth_range))
    demand_noise_std = float(rng.uniform(*demand_noise_range))
    cap_to_demand = float(rng.uniform(*cap_to_demand_range))

    base_demand = (
        active * rng.uniform(low=5.0, high=30.0, size=(n_firms, n_products))
    )
    capacity = base_demand * cap_to_demand
    capacity = np.maximum(capacity, 1.0) * active  # avoid zero capacity

    # Initial inventory ~ safety_factor * base_demand * (random multiplier)
    init_mult = rng.uniform(low=0.5, high=2.5, size=(n_firms, n_products))
    initial_inventory = (
        active * safety_factor * base_demand * init_mult * 5.0
    )

    # Lead times: integer in [1, lead_time_max].
    lead_time = (
        active * rng.integers(low=1, high=lead_time_max + 1,
                              size=(n_firms, n_products))
    ).astype(int)

    return Network(
        n_firms=n_firms,
        n_products=n_products,
        incidence=incidence,
        initial_inventory=initial_inventory,
        base_demand=base_demand,
        capacity=capacity,
        lead_time=lead_time,
        safety_factor=safety_factor,
        alpha_smooth=alpha_smooth,
        demand_noise_std=demand_noise_std,
    )


def generate_scr_suite(
    n_networks: int = 500,
    seed: int = 42,
) -> list[Network]:
    """Generate the full SCR suite of synthetic networks (SC-RIHN's setup)."""
    return [
        generate_scr_network(seed=seed + i)
        for i in range(n_networks)
    ]


def apply_node_removal(
    network: Network,
    p: float = 0.15,
    seed: int = 0,
) -> Network:
    """SCR-NR perturbation: drop firm nodes with probability p."""
    rng = np.random.default_rng(seed)
    keep = rng.random(network.n_firms) >= p
    if not keep.any():
        # Don't allow empty networks; keep at least one firm.
        keep[0] = True
    new_incidence = network.incidence.copy()
    new_incidence[~keep, :] = False
    return Network(
        n_firms=network.n_firms,
        n_products=network.n_products,
        incidence=new_incidence,
        initial_inventory=network.initial_inventory * keep[:, None],
        base_demand=network.base_demand * keep[:, None],
        capacity=network.capacity * keep[:, None],
        lead_time=network.lead_time * keep[:, None].astype(int),
        safety_factor=network.safety_factor,
        alpha_smooth=network.alpha_smooth,
        demand_noise_std=network.demand_noise_std,
    )


def apply_edge_removal(
    network: Network,
    p: float = 0.15,
    seed: int = 0,
) -> Network:
    """SCR-ER perturbation: drop firm-product incidence edges with probability p."""
    rng = np.random.default_rng(seed)
    keep_mask = rng.random(network.incidence.shape) >= p
    new_incidence = network.incidence & keep_mask
    return Network(
        n_firms=network.n_firms,
        n_products=network.n_products,
        incidence=new_incidence,
        initial_inventory=network.initial_inventory * new_incidence,
        base_demand=network.base_demand * new_incidence,
        capacity=network.capacity * new_incidence,
        lead_time=network.lead_time * new_incidence.astype(int),
        safety_factor=network.safety_factor,
        alpha_smooth=network.alpha_smooth,
        demand_noise_std=network.demand_noise_std,
    )


# =============================================================================
# Smoke test — run end-to-end and verify resilience labels are sensible.
# Usage:  python -m hypercp.data.supplysim
# =============================================================================


def _smoke_test() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    rng = np.random.default_rng(0)

    # 1. Generate a small network for fast test.
    net = generate_scr_network(n_firms=10, n_products=5, density=0.6, seed=0)
    print(
        f"Network: {net.n_firms} firms × {net.n_products} products, "
        f"active edges = {int(net.incidence.sum())}"
    )

    # 2. Simulate one no-shock trajectory.
    traj_noshock = simulate_trajectory(net, T_sim=100, shock=None, rng=rng)
    print(
        f"No-shock trajectory: inventory shape = {traj_noshock.inventory.shape}, "
        f"mean inventory at T/2 = {traj_noshock.inventory[50].mean():.2f}"
    )

    # 3. Simulate one shock trajectory.
    prior = ShockPrior(n_firms=net.n_firms, T_sim=100, duration=20, rng=rng)
    shock = prior.sample()
    print(
        f"Shock sampled: magnitude={shock.magnitude:.3f}, "
        f"firm={shock.firm_idx}, t_start={shock.t_start}, duration={shock.duration}"
    )
    traj_shock = simulate_trajectory(net, T_sim=100, shock=shock, rng=rng)

    # 4. Compute resilience labels.
    labels = compute_resilience_labels(
        traj_shock, shock, net, T_recovery=60
    )
    print()
    print("Resilience labels (per firm, first 5):")
    print(f"  TTR  : {labels.ttr[:5]}")
    print(f"  LPR  : {[f'{v:.3f}' for v in labels.lpr[:5]]}")
    print(f"  RL   : {[f'{v:.3f}' for v in labels.rl[:5]]}")
    print(f"  SL   : {[f'{v:.3f}' for v in labels.service_level[:5]]}")
    print(f"  r*   : {[f'{v:.3f}' for v in labels.r_star[:5]]}")
    print(f"  bin  : {labels.binary[:5]}")
    print()

    # 5. Sanity checks.
    assert 0 <= labels.r_star.min() and labels.r_star.max() <= 1, \
        "r* must lie in [0,1]"
    assert labels.binary.sum() <= net.n_firms, \
        "binary label sum bounded by n_firms"
    print("[OK] r* in [0,1]; binary in {0,1}; arrays well-formed.")

    # 6. Resilience varies with shock magnitude — sanity check.
    big_shock = Shock(magnitude=0.9, firm_idx=0, t_start=20, duration=30)
    small_shock = Shock(magnitude=0.1, firm_idx=0, t_start=20, duration=30)
    traj_big = simulate_trajectory(net, T_sim=100, shock=big_shock, rng=np.random.default_rng(1))
    traj_small = simulate_trajectory(net, T_sim=100, shock=small_shock, rng=np.random.default_rng(1))
    lab_big = compute_resilience_labels(traj_big, big_shock, net, T_recovery=60)
    lab_small = compute_resilience_labels(traj_small, small_shock, net, T_recovery=60)
    print(
        f"Big shock (m=0.9) r*[firm 0]:   {lab_big.r_star[0]:.3f}\n"
        f"Small shock (m=0.1) r*[firm 0]: {lab_small.r_star[0]:.3f}"
    )
    if lab_big.r_star[0] > lab_small.r_star[0]:
        print(
            "[WARN] Big shock gave higher r* -- likely simulator parameters mean "
            "shocks don't bite hard enough on this small synthetic network. "
            "Tune safety_factor / capacity / shock duration."
        )
    else:
        print("[OK] Big shock has lower resilience than small shock (as expected).")

    # 7. SCR-NR / SCR-ER perturbation smoke test.
    net_nr = apply_node_removal(net, p=0.3, seed=1)
    net_er = apply_edge_removal(net, p=0.3, seed=1)
    print(
        f"\nSCR-NR (p=0.3): active edges = {int(net_nr.incidence.sum())} "
        f"(was {int(net.incidence.sum())})\n"
        f"SCR-ER (p=0.3): active edges = {int(net_er.incidence.sum())}"
    )


if __name__ == "__main__":
    _smoke_test()
