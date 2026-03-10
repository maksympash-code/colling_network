from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np

from src.cooling_network.grid import CoolingNetwork
from src.cooling_network.types import CellType, ThermalResult
from src.cooling_network.stage2_straight_channels import (
    stage2_straight_channels,
    hotspots_mask,
)

Index2D = Tuple[int, int]
IOQ = Tuple[int, int, int, int]


@dataclass(frozen=True)
class Stage2FullResult:
    net: CoolingNetwork
    thermal_before: ThermalResult
    thermal_after_straight: ThermalResult
    thermal_after_barriers: Optional[ThermalResult]
    accepted_areas: int
    barrier_params: Optional[List[Tuple[int, int]]]


def hotspots_near_center_by_ratio(
    hot_mask: np.ndarray,
    center_frac: float = 0.25,
    ratio_thr: float = 0.35,
) -> bool:
    """
    If at least ratio_thr of hotspot cells lie in the central square,
    we enable corner barriers.
    """
    n, m = hot_mask.shape
    if n != m:
        return False

    total = int(np.sum(hot_mask))
    if total == 0:
        return False

    s0 = int(n * center_frac)
    s1 = int(n * (1.0 - center_frac))
    center = hot_mask[s0:s1, s0:s1]
    return (int(np.sum(center)) / total) >= ratio_thr


def paint_silicon_if_liquid(net: CoolingNetwork, cells: Iterable[Index2D]) -> None:
    """
    Convert LIQUID -> SILICON for given cells.
    Do not touch TSV / INLET / OUTLET.
    """
    C = net.C
    for i, j in cells:
        if not net.in_bounds(i, j):
            continue
        if C[i, j] in (CellType.TSV, CellType.INLET, CellType.OUTLET):
            continue
        if C[i, j] == CellType.LIQUID:
            C[i, j] = CellType.SILICON


def barrier_cells_for_corner(
    n: int,
    corner_id: int,
    length: int,
    theta_deg: int,
    thickness: int = 1,
) -> List[Index2D]:
    """
    Discretized corner barrier as a ray going into the chip.

    corner_id:
      0 = top-left
      1 = top-right
      2 = bottom-left
      3 = bottom-right
    """
    theta = np.deg2rad(theta_deg)
    di = float(np.sin(theta))
    dj = float(np.cos(theta))

    if corner_id == 0:      # TL
        si, sj = +1, +1
        ci, cj = 0, 0
    elif corner_id == 1:    # TR
        si, sj = +1, -1
        ci, cj = 0, n - 1
    elif corner_id == 2:    # BL
        si, sj = -1, +1
        ci, cj = n - 1, 0
    else:                   # BR
        si, sj = -1, -1
        ci, cj = n - 1, n - 1

    cells: List[Index2D] = []
    x = 0.0
    y = 0.0

    for _ in range(length):
        ii = int(round(ci + si * x))
        jj = int(round(cj + sj * y))
        cells.append((ii, jj))

        for k in range(1, thickness):
            cells.append((ii + k * (-sj), jj + k * si))
            cells.append((ii - k * (-sj), jj - k * si))

        x += di
        y += dj

    return cells


def apply_corner_barriers(
    net: CoolingNetwork,
    params: List[Tuple[int, int]],
    thickness: int = 1,
) -> None:
    """
    params: list of 4 tuples (length, theta_deg) for corners [TL, TR, BL, BR]
    """
    n, _ = net.C.shape
    for corner_id, (L, theta) in enumerate(params):
        cells = barrier_cells_for_corner(
            n=n,
            corner_id=corner_id,
            length=L,
            theta_deg=theta,
            thickness=thickness,
        )
        paint_silicon_if_liquid(net, cells)


def optimize_corner_barriers(
    base_net: CoolingNetwork,
    power_map: np.ndarray,
    thermal_fn,
    length_candidates: List[int],
    theta_candidates: List[int],
    thickness: int = 1,
) -> List[Tuple[int, int]]:
    """
    Choose best (L, theta) for each corner independently.
    Criterion: minimal max_T.
    """
    best_params: List[Tuple[int, int]] = []

    for corner_id in range(4):
        best = None
        best_max = float("inf")

        for L in length_candidates:
            for theta in theta_candidates:
                net = base_net.clone()

                cells = barrier_cells_for_corner(
                    n=net.C.shape[0],
                    corner_id=corner_id,
                    length=L,
                    theta_deg=theta,
                    thickness=thickness,
                )
                paint_silicon_if_liquid(net, cells)

                net.prune_irregular()
                if not net.has_inlet_to_outlet_path():
                    continue

                res = thermal_fn(net, power_map)
                if res.max_T < best_max:
                    best_max = res.max_T
                    best = (L, theta)

        if best is None:
            best = (
                length_candidates[len(length_candidates) // 2],
                theta_candidates[len(theta_candidates) // 2],
            )

        best_params.append(best)

    return best_params


def line_search_lengths(
    base_net: CoolingNetwork,
    power_map: np.ndarray,
    thermal_fn,
    per_corner_params: List[Tuple[int, int]],
    scales: List[float],
    thickness: int = 1,
) -> List[Tuple[int, int]]:
    """
    Scale all 4 barrier lengths together, keep angles fixed.
    """
    best = per_corner_params
    best_max = float("inf")

    for s in scales:
        params = []
        for (L, theta) in per_corner_params:
            Ls = max(1, int(round(L * s)))
            params.append((Ls, theta))

        net = base_net.clone()
        apply_corner_barriers(net, params=params, thickness=thickness)
        net.prune_irregular()

        if not net.has_inlet_to_outlet_path():
            continue

        res = thermal_fn(net, power_map)
        if res.max_T < best_max:
            best_max = res.max_T
            best = params

    return best


def stage2_full_patterning(
    net: CoolingNetwork,
    power_map: np.ndarray,
    thermal_fn,
    alpha_1: float,
    alpha_2: float,
    io_q: IOQ,
) -> Stage2FullResult:
    """
    Full Stage 2:
      1) straight-channel patterning
      2) if hotspots are central enough -> corner barriers
      3) accept barriers only if they improve max_T
    """
    s1 = stage2_straight_channels(
        net=net,
        power_map=power_map,
        thermal_fn=thermal_fn,
        alpha_1=alpha_1,
        io_q=io_q,
        pad=max(1, net.C.shape[0] // 12),
        stripe_candidates=(1, 2, 3),
        thickness_candidates=(1, 2),
    )

    current_net = s1.net
    current_thermal = s1.thermal_after

    hot2 = hotspots_mask(current_thermal.T, alpha_2)
    if not hotspots_near_center_by_ratio(hot2):
        return Stage2FullResult(
            net=current_net,
            thermal_before=s1.thermal_before,
            thermal_after_straight=s1.thermal_after,
            thermal_after_barriers=None,
            accepted_areas=s1.accepted_areas,
            barrier_params=None,
        )

    n = current_net.C.shape[0]
    length_candidates = sorted(set([
        max(1, n // 8),
        max(1, n // 6),
        max(1, n // 4),
    ]))
    theta_candidates = [25, 40, 55, 70]

    params0 = optimize_corner_barriers(
        base_net=current_net,
        power_map=power_map,
        thermal_fn=thermal_fn,
        length_candidates=length_candidates,
        theta_candidates=theta_candidates,
        thickness=1,
    )

    params1 = line_search_lengths(
        base_net=current_net,
        power_map=power_map,
        thermal_fn=thermal_fn,
        per_corner_params=params0,
        scales=[0.75, 1.0, 1.25],
        thickness=1,
    )

    cand = current_net.clone()
    apply_corner_barriers(cand, params=params1, thickness=1)
    cand.prune_irregular()

    if not cand.has_inlet_to_outlet_path():
        return Stage2FullResult(
            net=current_net,
            thermal_before=s1.thermal_before,
            thermal_after_straight=s1.thermal_after,
            thermal_after_barriers=None,
            accepted_areas=s1.accepted_areas,
            barrier_params=None,
        )

    therm_bar = thermal_fn(cand, power_map)

    better = (
        therm_bar.max_T < current_thermal.max_T - 1e-6
        or (
            abs(therm_bar.max_T - current_thermal.max_T) <= 1e-6
            and therm_bar.grad_T < current_thermal.grad_T
        )
    )

    if better:
        return Stage2FullResult(
            net=cand,
            thermal_before=s1.thermal_before,
            thermal_after_straight=s1.thermal_after,
            thermal_after_barriers=therm_bar,
            accepted_areas=s1.accepted_areas,
            barrier_params=params1,
        )

    return Stage2FullResult(
        net=current_net,
        thermal_before=s1.thermal_before,
        thermal_after_straight=s1.thermal_after,
        thermal_after_barriers=None,
        accepted_areas=s1.accepted_areas,
        barrier_params=None,
    )