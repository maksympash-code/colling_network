from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from src.cooling_network.grid import CoolingNetwork
from src.cooling_network.types import CellType, ThermalResult


Index2D = Tuple[int, int]
IOQ = Tuple[int, int, int, int]


# =========================================================
# Areas
# =========================================================

@dataclass(frozen=True)
class RectArea:
    r0: int
    r1: int   # exclusive
    c0: int
    c1: int   # exclusive

    def contains(self, i: int, j: int) -> bool:
        return self.r0 <= i < self.r1 and self.c0 <= j < self.c1

    @property
    def area(self) -> int:
        return (self.r1 - self.r0) * (self.c1 - self.c0)


# =========================================================
# Hotspots
# =========================================================

def hotspots_mask(T: np.ndarray, alpha: float) -> np.ndarray:
    mu = float(np.mean(T))
    sigma = float(np.std(T))
    thr = mu + alpha * sigma
    return T > thr


def clusters_from_mask(mask: np.ndarray, neighbors4_fn) -> List[List[Index2D]]:
    n, m = mask.shape
    vis = np.zeros((n, m), dtype=bool)
    comps: List[List[Index2D]] = []

    for i in range(n):
        for j in range(m):
            if not mask[i, j] or vis[i, j]:
                continue

            q = deque([(i, j)])
            vis[i, j] = True
            comp: List[Index2D] = []

            while q:
                x, y = q.popleft()
                comp.append((x, y))
                for nx, ny in neighbors4_fn(x, y):
                    if mask[nx, ny] and not vis[nx, ny]:
                        vis[nx, ny] = True
                        q.append((nx, ny))

            comps.append(comp)

    return comps


def choose_rect_area(comp: List[Index2D], pad: int, n: int) -> RectArea:
    rs = [i for i, _ in comp]
    cs = [j for _, j in comp]

    r0 = max(0, min(rs) - pad)
    r1 = min(n, max(rs) + pad + 1)
    c0 = max(0, min(cs) - pad)
    c1 = min(n, max(cs) + pad + 1)

    return RectArea(r0=r0, r1=r1, c0=c0, c1=c1)


# =========================================================
# Direction from IO
# =========================================================

def quadrant_of_point(n: int, i: int, j: int) -> int:
    mid = n // 2
    if i < mid and j < mid:
        return 0
    if i < mid and j >= mid:
        return 1
    if i >= mid and j < mid:
        return 2
    return 3


def flow_direction_from_ioq(n: int, area: RectArea, io_q: IOQ) -> str:
    ci = (area.r0 + area.r1) // 2
    cj = (area.c0 + area.c1) // 2
    qi = quadrant_of_point(n, ci, cj)
    mode = io_q[qi]
    return "horizontal" if mode == 0 else "vertical"


def opposite_direction(direction: str) -> str:
    return "vertical" if direction == "horizontal" else "horizontal"


# =========================================================
# Straight channels (НОВА логіка: відкриваємо канали, а не масово видаляємо)
# =========================================================

def apply_open_straight_channels(
    net: CoolingNetwork,
    area: RectArea,
    direction: str,
    stripe_step: int = 2,
    thickness: int = 1,
) -> None:
    """
    Відкриває straight channels у hot area:
    - SILICON -> LIQUID на stripe-лініях
    - existing LIQUID залишаємо як є
    - TSV/INLET/OUTLET не чіпаємо

    Це значно безпечніше для твоєї моделі, ніж масово LIQUID->SILICON.
    """
    C = net.C
    n, m = C.shape

    r0 = max(0, area.r0)
    r1 = min(n, area.r1)
    c0 = max(0, area.c0)
    c1 = min(m, area.c1)

    def paint_liquid(i: int, j: int):
        if not net.in_bounds(i, j):
            return

        # Never touch fixed boundary cells that define external walls
        if i == 0 or i == net.C.shape[0] - 1 or j == 0 or j == net.C.shape[1] - 1:
            return

        if C[i, j] in (CellType.TSV, CellType.INLET, CellType.OUTLET):
            return

        if C[i, j] == CellType.SILICON:
            C[i, j] = CellType.LIQUID

    if direction == "horizontal":
        rows = list(range(r0, r1, stripe_step))
        for i in rows:
            for t in range(thickness):
                ii = i + t
                if ii >= r1:
                    break
                for j in range(c0, c1):
                    paint_liquid(ii, j)
    else:
        cols = list(range(c0, c1, stripe_step))
        for j in cols:
            for t in range(thickness):
                jj = j + t
                if jj >= c1:
                    break
                for i in range(r0, r1):
                    paint_liquid(i, jj)


# =========================================================
# Result
# =========================================================

@dataclass(frozen=True)
class Stage2Result:
    net: CoolingNetwork
    thermal_before: ThermalResult
    thermal_after: ThermalResult
    n_hotspots: int
    n_clusters: int
    accepted_areas: int


# =========================================================
# Stage 2 main
# =========================================================

def stage2_straight_channels(
    net: CoolingNetwork,
    power_map: np.ndarray,
    thermal_fn,
    alpha_1: float,
    io_q: IOQ,
    pad: int = 1,
    stripe_candidates: Tuple[int, ...] = (1, 2),
    thickness_candidates: Tuple[int, ...] = (1,),
) -> Stage2Result:
    """
    Нова безпечна версія Stage 2:
    - будує candidate тільки у hotspot areas
    - пробує кілька варіантів (preferred / opposite direction)
    - ПРИЙМАЄ area patterning тільки якщо max_T реально зменшується
    """
    thermal_before: ThermalResult = thermal_fn(net, power_map)
    current_net = net.clone()
    current_thermal = thermal_before

    hot = hotspots_mask(thermal_before.T, alpha_1)
    comps = clusters_from_mask(hot, current_net.neighbors4)

    n_hotspots = int(np.sum(hot))
    accepted_areas = 0
    n = current_net.C.shape[0]

    # Сортуємо кластери: спочатку найгарячіші
    def comp_peak(comp: List[Index2D]) -> float:
        return max(float(thermal_before.T[i, j]) for i, j in comp)

    comps = sorted(comps, key=comp_peak, reverse=True)

    for comp in comps:
        area = choose_rect_area(comp, pad=pad, n=n)

        preferred = flow_direction_from_ioq(n, area, io_q)
        directions = [preferred, opposite_direction(preferred)]

        best_local_net = None
        best_local_thermal = current_thermal

        for direction in directions:
            for stripe in stripe_candidates:
                for thickness in thickness_candidates:
                    cand = current_net.clone()
                    apply_open_straight_channels(
                        cand,
                        area=area,
                        direction=direction,
                        stripe_step=stripe,
                        thickness=thickness,
                    )

                    if not cand.has_inlet_to_outlet_path():
                        continue

                    cand.prune_irregular()

                    if not cand.has_inlet_to_outlet_path():
                        continue

                    therm = thermal_fn(cand, power_map)

                    better = (
                        therm.max_T < best_local_thermal.max_T - 1e-6
                        or (
                            abs(therm.max_T - best_local_thermal.max_T) <= 1e-6
                            and therm.grad_T < best_local_thermal.grad_T
                        )
                    )

                    if better:
                        best_local_net = cand
                        best_local_thermal = therm

        # greedy accept only if this area really improves the current solution
        if best_local_net is not None and best_local_thermal.max_T < current_thermal.max_T - 1e-6:
            current_net = best_local_net
            current_thermal = best_local_thermal
            accepted_areas += 1

    return Stage2Result(
        net=current_net,
        thermal_before=thermal_before,
        thermal_after=current_thermal,
        n_hotspots=n_hotspots,
        n_clusters=len(comps),
        accepted_areas=accepted_areas,
    )