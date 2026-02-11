from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.cooling_network.grid import CoolingNetwork
from src.cooling_network.types import CellType, PruneStepResult, ThermalResult, Index2D
import src.cooling_network.config as cfg


# -----------------------------
# Unfavored list (expiry-based)
# -----------------------------
class Unfavored:
    """
    Stores cells as 'blocked until iteration < expire_iter'.
    O(1) checks, no per-iteration decrement loops.
    """
    def __init__(self, ttl: int):
        self.ttl = ttl
        self._expire: Dict[Index2D, int] = {}

    def is_blocked(self, cell: Index2D, it: int) -> bool:
        exp = self._expire.get(cell)
        return exp is not None and it < exp

    def mark(self, cell: Index2D, it: int) -> None:
        self._expire[cell] = it + self.ttl

    def cleanup(self, it: int) -> None:
        # optional cleanup to keep dict small
        dead = [c for c, exp in self._expire.items() if exp <= it]
        for c in dead:
            del self._expire[c]


# -----------------------------
# τ priority computation
# -----------------------------
def compute_tau_for_liquid_cells(
    net: CoolingNetwork,
    T: np.ndarray,
    radius: int = cfg.TAU_RADIUS
) -> Dict[Index2D, float]:
    """
    τ(cell) = weighted average of surrounding temperatures (paper Fig.7-ish).
    Here we use local Manhattan neighborhood with weights 1/(d+1).
    """
    n, m = net.C.shape
    tau: Dict[Index2D, float] = {}

    liquid_positions = np.argwhere(net.C == CellType.LIQUID)

    for (i, j) in liquid_positions:
        num = 0.0
        den = 0.0

        # local neighborhood
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

        tau[(int(i), int(j))] = num / den if den > 0 else float(T[i, j])

    return tau


# -----------------------------
# HBS selection from n best τ
# -----------------------------
def select_candidate_hbs(
    tau_map: Dict[Index2D, float],
    net: CoolingNetwork,
    unfavored: Unfavored,
    it: int,
    rng: np.random.Generator,
    n_candidates: int = cfg.N_CANDIDATES,
    sigma: float = cfg.SIGMA_HBS,
) -> Optional[Index2D]:
    """
    Sort liquid cells by τ ascending, take first n, sample with bias exp(-r^2/sigma^2).
    Excludes blocked cells.
    """
    items = [(cell, val) for cell, val in tau_map.items()
             if (net.C[cell[0], cell[1]] == CellType.LIQUID) and (not unfavored.is_blocked(cell, it))]

    if not items:
        return None

    items.sort(key=lambda x: x[1])  # ascending τ
    items = items[: max(1, min(n_candidates, len(items)))]

    k = len(items)
    ranks = np.arange(k, dtype=np.float64)
    probs = np.exp(-(ranks * ranks) / (sigma * sigma))
    probs = probs / probs.sum()

    idx = int(rng.choice(k, p=probs))
    return items[idx][0]


# -----------------------------
# Acceptance probability (paper-like)
# -----------------------------
def acceptance_probability(
    old: ThermalResult,
    new: ThermalResult,
    t_max: float,
    grad_max: float,
    delta: float = cfg.DELTA_C,
    lam: float = cfg.LAMBDA
) -> float:
    """
    Paper:
      Pr1 = 1 if max(T) drops OR max(Tnew) <= Tmax - δ else exp(-|Δ|/λ)
      Pr2 = 1 if grad drops OR grad_new <= grad_max - δ else exp(-|Δ|/λ)
      Pr = Pr1 * Pr2
    """
    d_max = new.max_T - old.max_T
    d_grad = new.grad_T - old.grad_T

    if (d_max < 0.0) or (new.max_T <= t_max - delta):
        pr1 = 1.0
    else:
        pr1 = float(np.exp(-abs(d_max) / lam))

    if (d_grad < 0.0) or (new.grad_T <= grad_max - delta):
        pr2 = 1.0
    else:
        pr2 = float(np.exp(-abs(d_grad) / lam))

    return pr1 * pr2


# -----------------------------
# One pruning step (tentative -> simulate -> accept/reject)
# -----------------------------
def prune_step(
    net: CoolingNetwork,
    power_map: np.ndarray,
    thermal_fn,
    old_thermal: ThermalResult,
    unfavored: Unfavored,
    it: int,
    rng: np.random.Generator,
    t_max: float = cfg.T_MAX,
    grad_max: float = cfg.GRAD_T_MAX,
    eps_big: float = cfg.EPC_C,
) -> PruneStepResult:
    """
    1) compute τ
    2) select candidate by HBS
    3) tentative remove + prune_irregular + path-check
    4) simulate
    5) accept/reject probabilistically
    6) if rejected with big increase => add to unfavored
    """
    tau_map = compute_tau_for_liquid_cells(net, old_thermal.T, radius=cfg.TAU_RADIUS)
    cand = select_candidate_hbs(tau_map, net, unfavored, it, rng)

    if cand is None:
        return PruneStepResult(removed_cell=None, accepted=False, thermal=old_thermal)

    old_net = net.clone()

    ci, cj = cand
    try:
        net.remove_cell(ci, cj)
    except ValueError:
        return PruneStepResult(removed_cell=None, accepted=False, thermal=old_thermal)

    net.prune_irregular()

    if not net.has_inlet_to_outlet_path():
        # rollback
        net.C[:] = old_net.C
        return PruneStepResult(removed_cell=cand, accepted=False, thermal=old_thermal)

    new_thermal: ThermalResult = thermal_fn(net, power_map)

    pr = acceptance_probability(old_thermal, new_thermal, t_max=t_max, grad_max=grad_max)

    accepted = bool(rng.random() < pr)

    if not accepted:
        if (new_thermal.max_T - old_thermal.max_T > eps_big) or (new_thermal.grad_T - old_thermal.grad_T > eps_big):
            unfavored.mark(cand, it)

        net.C[:] = old_net.C
        return PruneStepResult(removed_cell=cand, accepted=False, thermal=old_thermal)

    # accepted
    return PruneStepResult(removed_cell=cand, accepted=True, thermal=new_thermal)


# -----------------------------
# Full loop (Stage 3 simplified)
# -----------------------------
@dataclass
class RunResult:
    history: List[PruneStepResult]
    final_net: CoolingNetwork


def run_pruning(
    net: CoolingNetwork,
    power_map: np.ndarray,
    thermal_fn,
    max_iters: int = 5000,
    t_max: float = cfg.T_MAX,
    grad_max: float = cfg.GRAD_T_MAX,
    seed: int = cfg.RNG_SEED,
) -> RunResult:
    rng = np.random.default_rng(seed)
    unfavored = Unfavored(ttl=cfg.UNFAVORED_TTL)

    thermal = thermal_fn(net, power_map)

    history: List[PruneStepResult] = []
    reject_streak = 0

    for it in range(max_iters):
        unfavored.cleanup(it)

        step = prune_step(
            net=net,
            power_map=power_map,
            thermal_fn=thermal_fn,
            old_thermal=thermal,
            unfavored=unfavored,
            it=it,
            rng=rng,
            t_max=t_max,
            grad_max=grad_max,
            eps_big=cfg.EPC_C,
        )
        history.append(step)

        if step.accepted:
            thermal = step.thermal
            reject_streak = 0
        else:
            reject_streak += 1

        if reject_streak >= cfg.MAX_REJECT_STREAK:
            break

        # also stop if no liquid cells left
        if not np.any(net.C == CellType.LIQUID):
            break

    return RunResult(history=history, final_net=net)
