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


def compute_tau(net: CoolingNetwork, T: np.ndarray, radius: int) -> Dict[Index2D, float]:
    n, m = net.C.shape
    tau: Dict[Index2D, float] = {}
    for (i, j) in np.argwhere(net.C == CellType.LIQUID):
        i = int(i); j = int(j)
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


def select_candidate_hbs(
    net: CoolingNetwork,
    tau: Dict[Index2D, float],
    unfavored: Unfavored,
    it: int,
    rng: np.random.Generator,
    n_candidates: int = cfg.N_CANDIDATES,
    sigma: float = cfg.SIGMA_HBS
) -> Optional[Index2D]:
    items = [(cell, val) for cell, val in tau.items()
             if net.C[cell[0], cell[1]] == CellType.LIQUID and not unfavored.is_blocked(cell, it)]
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


def acceptance_probability(old: ThermalResult, new: ThermalResult,
                           Tmax: float, GradTmax: float,
                           delta: float, lam: float) -> float:
    d_max = new.max_T - old.max_T
    d_grad = new.grad_T - old.grad_T

    if (d_max < 0.0) or (new.max_T <= Tmax - delta):
        pr1 = 1.0
    else:
        pr1 = float(np.exp(-abs(d_max) / lam))

    if (d_grad < 0.0) or (new.grad_T <= GradTmax - delta):
        pr2 = 1.0
    else:
        pr2 = float(np.exp(-abs(d_grad) / lam))

    return pr1 * pr2


def eval_network(net: CoolingNetwork, power_map: np.ndarray, P_pump: float, params: ModelParams) -> Tuple[ThermalResult, float]:
    flow = solve_flow(net, P_pump=P_pump, params=params)
    therm = solve_thermal_one_source_one_channel(net, power_map, flow, params)
    return ThermalResult(T=therm.T_source, max_T=therm.max_T, grad_T=therm.grad_T), flow.E_pump


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
    rng = np.random.default_rng(cfg.RNG_SEED)
    unfavored = Unfavored(cfg.UNFAVORED_TTL)

    # Rule 1: pick initial P_pump minimal meeting constraints (no margin)
    opt = find_min_p_pump(net, power_map, params, Tmax=Tmax, GradTmax=GradTmax,
                          P_low=cfg.P_INIT_LOW, P_high=cfg.P_INIT_HIGH, iters=25)
    P_pump = opt.P_pump
    thermal, E_pump = eval_network(net, power_map, P_pump, params)
    target_E = E_pump

    history: List[PruneStepResult] = []
    reject_streak = 0

    for it in range(max_iters):
        unfavored.cleanup(it)

        tau = compute_tau(net, thermal.T, radius=cfg.TAU_RADIUS)
        cand = select_candidate_hbs(net, tau, unfavored, it, rng)
        if cand is None:
            break

        old_net = net.clone()
        old_thermal = thermal
        old_E = E_pump
        old_P = P_pump

        # tentative removal
        try:
            net.remove_cell(*cand)
        except Exception:
            reject_streak += 1
            if reject_streak >= cfg.MAX_REJECT_STREAK:
                break
            continue

        net.prune_irregular()
        if not net.has_inlet_to_outlet_path():
            net.C[:] = old_net.C
            reject_streak += 1
            if reject_streak >= cfg.MAX_REJECT_STREAK:
                break
            continue

        # Rule 3 (paper-ish): keep same energy if constraints are tight.
        # We estimate P that maintains target_E via quadratic scaling.
        # Use a test pressure = old_P to measure new E_test, then scale.
        flow_test = solve_flow(net, P_pump=old_P, params=params)
        E_test = flow_test.E_pump
        if E_test <= 0.0:
            net.C[:] = old_net.C
            reject_streak += 1
            continue

        P_sameE = old_P * np.sqrt(max(target_E, 1e-30) / E_test)
        P_pump = float(P_sameE)

        # evaluate with this pressure
        thermal_new, E_new = eval_network(net, power_map, P_pump, params)

        pr = acceptance_probability(old_thermal, thermal_new, Tmax, GradTmax, cfg.DELTA_C, cfg.LAMBDA)
        accepted = bool(rng.random() < pr)

        if not accepted:
            # mark unfavored if big increase
            if (thermal_new.max_T - old_thermal.max_T > cfg.EPC_C) or (thermal_new.grad_T - old_thermal.grad_T > cfg.EPC_C):
                unfavored.mark(cand, it)
            # rollback
            net.C[:] = old_net.C
            thermal = old_thermal
            E_pump = old_E
            P_pump = old_P
            reject_streak += 1
        else:
            thermal = thermal_new
            E_pump = E_new
            reject_streak = 0

            # Rule 2 (paper-ish): if we have safe margin, reduce pressure (minimize)
            if (thermal.max_T <= Tmax - cfg.SAFE_MARGIN) and (thermal.grad_T <= GradTmax - cfg.SAFE_MARGIN):
                opt2 = find_min_p_pump(net, power_map, params, Tmax=Tmax, GradTmax=GradTmax,
                                       P_low=cfg.P_INIT_LOW, P_high=max(cfg.P_INIT_LOW, P_pump), iters=20)
                P_pump = opt2.P_pump
                thermal, E_pump = eval_network(net, power_map, P_pump, params)

            target_E = E_pump  # update target energy to current best (cooling energy minimization)

        history.append(
            PruneStepResult(
                removed_cell=cand,
                accepted=accepted,
                thermal=thermal,
                E_pump=E_pump,
                P_pump=P_pump,
            )
        )

        if reject_streak >= cfg.MAX_REJECT_STREAK:
            break

        if not np.any(net.C == CellType.LIQUID):
            break

    return RunResult(net=net, history=history, final_T=thermal, final_E=E_pump, final_P=P_pump)