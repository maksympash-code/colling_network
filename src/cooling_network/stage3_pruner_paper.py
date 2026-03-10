from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

import src.cooling_network.config as cfg
from src.cooling_network.grid import CoolingNetwork
from src.cooling_network.types import CellType, ThermalResult, PruneStepResult, Index2D
from src.cooling_network.flow_solver import solve_flow
from src.cooling_network.thermal_solver import solve_thermal_one_source_one_channel
from src.cooling_network.pressure_opt import find_min_p_pump
from src.cooling_network.physics import ModelParams


def _cfg(name: str, default):
    return getattr(cfg, name, default)


P_INIT_LOW = _cfg("P_INIT_LOW", 1e3)
P_INIT_HIGH = _cfg("P_INIT_HIGH", 1e6)
SAFE_MARGIN = _cfg("SAFE_MARGIN", 0.2)
UNFAVORED_TTL = _cfg("UNFAVORED_TTL", 200)
TAU_RADIUS = _cfg("TAU_RADIUS", 4)
RNG_SEED = _cfg("RNG_SEED", 42)
MAX_REJECT_STREAK = _cfg("MAX_REJECT_STREAK", 60)
N_CANDIDATES = _cfg("N_CANDIDATES", 10)
SIGMA_HBS = _cfg("SIGMA_HBS", 2.5)
NR_RECOMPUTE_PRESSURE = _cfg("NR_RECOMPUTE_PRESSURE", 20)
MAX_TEMP_RISE_STAGE3 = _cfg("MAX_TEMP_RISE_STAGE3", 2.0)
MAX_GRAD_RISE_STAGE3 = _cfg("MAX_GRAD_RISE_STAGE3", 1.0)

class Unfavored:
    def __init__(self, ttl: int):
        self.ttl = ttl
        self.expire: Dict[Index2D, int] = {}

    def is_blocked(self, cell: Index2D, it: int) -> bool:
        exp = self.expire.get(cell)
        return exp is not None and it < exp

    def mark(self, cell: Index2D, it: int) -> None:
        self.expire[cell] = it + self.ttl

    def cleanup(self, it: int) -> None:
        dead = [c for c, exp in self.expire.items() if exp <= it]
        for c in dead:
            del self.expire[c]


def eval_network(
    net: CoolingNetwork,
    power_map: np.ndarray,
    P_pump: float,
    params: ModelParams,
) -> Tuple[ThermalResult, float]:
    flow = solve_flow(net, P_pump=P_pump, params=params)
    therm = solve_thermal_one_source_one_channel(net, power_map, flow, params)
    return ThermalResult(T=therm.T_source, max_T=therm.max_T, grad_T=therm.grad_T), flow.E_pump


def is_feasible(thermal: ThermalResult, Tmax: float, GradTmax: float) -> bool:
    return (thermal.max_T <= Tmax) and (thermal.grad_T <= GradTmax)


def compute_tau(net: CoolingNetwork, T: np.ndarray, radius: int) -> Dict[Index2D, float]:
    """
    Local weighted temperature average around each liquid cell.
    Lower tau -> colder region -> better pruning candidate.
    """
    n, m = net.C.shape
    tau: Dict[Index2D, float] = {}

    for (i, j) in np.argwhere(net.C == CellType.LIQUID):
        i = int(i)
        j = int(j)

        num = 0.0
        den = 0.0
        for di in range(-radius, radius + 1):
            for dj in range(-radius, radius + 1):
                ii = i + di
                jj = j + dj
                if ii < 0 or ii >= n or jj < 0 or jj >= m:
                    continue
                d = abs(di) + abs(dj)
                w = 1.0 / (d + 1.0)
                num += w * float(T[ii, jj])
                den += w

        tau[(i, j)] = num / den if den > 0 else float(T[i, j])

    return tau


def is_bad_prune_location(net: CoolingNetwork, cell: Index2D) -> bool:
    """
    Avoid pruning cells that are too close to ports / boundary.
    This makes Stage 3 much more stable.
    """
    i, j = cell
    n, m = net.C.shape

    # avoid direct boundary cells
    if i == 0 or i == n - 1 or j == 0 or j == m - 1:
        return True

    # avoid cells adjacent to inlet/outlet
    for ni, nj in net.neighbors4(i, j):
        if net.C[ni, nj] in (CellType.INLET, CellType.OUTLET):
            return True

    return False


def select_candidate_hbs(
    net: CoolingNetwork,
    tau: Dict[Index2D, float],
    unfavored: Unfavored,
    it: int,
    rng: np.random.Generator,
    n_candidates: int = N_CANDIDATES,
    sigma: float = SIGMA_HBS,
) -> Optional[Index2D]:
    """
    Select from the coldest candidate cells with a soft bias toward the coldest one.
    """
    items = []
    for cell, val in tau.items():
        i, j = cell
        if net.C[i, j] != CellType.LIQUID:
            continue
        if unfavored.is_blocked(cell, it):
            continue
        if is_bad_prune_location(net, cell):
            continue
        items.append((cell, val))

    if not items:
        return None

    items.sort(key=lambda x: x[1])
    items = items[: max(1, min(n_candidates, len(items)))]

    k = len(items)
    ranks = np.arange(k, dtype=np.float64)
    probs = np.exp(-(ranks * ranks) / (sigma * sigma))
    probs /= probs.sum()

    idx = int(rng.choice(k, p=probs))
    return items[idx][0]


@dataclass(frozen=True)
class RunResult:
    net: CoolingNetwork
    history: List[PruneStepResult]
    final_T: ThermalResult
    final_E: float
    final_P: float


def run_stage3_pruning(
    net: CoolingNetwork,
    power_map: np.ndarray,
    params: ModelParams,
    Tmax: float = cfg.T_MAX,
    GradTmax: float = cfg.GRAD_T_MAX,
    max_iters: int = 5000,
) -> RunResult:
    """
    Stable deterministic Stage 3:
      - prune low-tau liquid cells
      - keep only feasible designs
      - accept only if E_pump does not increase
      - periodically re-optimize P_pump
    """
    rng = np.random.default_rng(RNG_SEED)
    unfavored = Unfavored(UNFAVORED_TTL)

    history: List[PruneStepResult] = []

    # --- Initial minimal feasible pressure ---
    opt0 = find_min_p_pump(
        net=net,
        power_map=power_map,
        params=params,
        Tmax=Tmax,
        GradTmax=GradTmax,
        P_low=P_INIT_LOW,
        P_high=P_INIT_HIGH,
        iters=25,
    )

    P_pump = opt0.P_pump
    thermal, E_pump = eval_network(net, power_map, P_pump, params)

    # If even the starting design is infeasible, Stage 3 should not proceed.
    if not is_feasible(thermal, Tmax, GradTmax):
        return RunResult(
            net=net,
            history=history,
            final_T=thermal,
            final_E=E_pump,
            final_P=P_pump,
        )

    reject_streak = 0

    for it in range(max_iters):
        unfavored.cleanup(it)

        tau = compute_tau(net, thermal.T, radius=TAU_RADIUS)
        cand = select_candidate_hbs(net, tau, unfavored, it, rng)
        if cand is None:
            break

        old_net = net.clone()
        old_thermal = thermal
        old_E = E_pump
        old_P = P_pump

        # ---- tentative removal ----
        try:
            net.remove_cell(*cand)
        except Exception:
            reject_streak += 1
            if reject_streak >= MAX_REJECT_STREAK:
                break
            continue

        net.prune_irregular()

        if not net.has_inlet_to_outlet_path():
            net.C[:] = old_net.C
            unfavored.mark(cand, it)
            reject_streak += 1
            if reject_streak >= MAX_REJECT_STREAK:
                break
            continue

        # ---- recompute minimal feasible pressure on candidate design ----
        # We allow the new design to choose its own minimal feasible P_pump.
        opt_new = find_min_p_pump(
            net=net,
            power_map=power_map,
            params=params,
            Tmax=Tmax,
            GradTmax=GradTmax,
            P_low=P_INIT_LOW,
            P_high=P_INIT_HIGH,
            iters=25,
        )

        P_new = opt_new.P_pump
        thermal_new, E_new = eval_network(net, power_map, P_new, params)

        feasible_new = is_feasible(thermal_new, Tmax, GradTmax)

        accepted = False
        if feasible_new:
            temp_rise_ok = (thermal_new.max_T <= old_thermal.max_T + MAX_TEMP_RISE_STAGE3)
            grad_rise_ok = (thermal_new.grad_T <= old_thermal.grad_T + MAX_GRAD_RISE_STAGE3)

            if temp_rise_ok and grad_rise_ok:
                if E_new < old_E - 1e-12:
                    accepted = True
                elif abs(E_new - old_E) <= 1e-12:
                    if (
                            thermal_new.max_T <= old_thermal.max_T + 1e-6
                            and thermal_new.grad_T <= old_thermal.grad_T + 1e-6
                    ):
                        accepted = True

        if not accepted:
            net.C[:] = old_net.C
            thermal = old_thermal
            E_pump = old_E
            P_pump = old_P
            unfavored.mark(cand, it)
            reject_streak += 1
        else:
            thermal = thermal_new
            E_pump = E_new
            P_pump = P_new
            reject_streak = 0

            # optional periodic pressure refresh
            if (it + 1) % NR_RECOMPUTE_PRESSURE == 0:
                opt_ref = find_min_p_pump(
                    net=net,
                    power_map=power_map,
                    params=params,
                    Tmax=Tmax,
                    GradTmax=GradTmax,
                    P_low=P_INIT_LOW,
                    P_high=max(P_INIT_LOW, P_pump),
                    iters=20,
                )
                P_pump = opt_ref.P_pump
                thermal, E_pump = eval_network(net, power_map, P_pump, params)

        history.append(
            PruneStepResult(
                removed_cell=cand,
                accepted=accepted,
                thermal=thermal,
                E_pump=E_pump,
                P_pump=P_pump,
            )
        )

        if reject_streak >= MAX_REJECT_STREAK:
            break

        if not np.any(net.C == CellType.LIQUID):
            break

        # if we already have a comfortable thermal margin, pressure will be re-minimized anyway,
        # so no extra stochastic exploration is needed.

    return RunResult(
        net=net,
        history=history,
        final_T=thermal,
        final_E=E_pump,
        final_P=P_pump,
    )