from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from src.cooling_network.grid import CoolingNetwork
from src.cooling_network.types import CellType, ThermalResult


Index2D = Tuple[int, int]


# -------------------------
# Hot area shapes
# -------------------------

@dataclass(frozen=True)
class RectArea:
    r0: int
    r1: int  # exclusive
    c0: int
    c1: int  # exclusive

    def contains(self, i: int, j: int) -> bool:
        return self.r0 <= i < self.r1 and self.c0 <= j < self.c1

    @property
    def area(self) -> int:
        return (self.r1 - self.r0) * (self.c1 - self.c0)


@dataclass(frozen=True)
class TriArea:
    """
    45-45-90 right isosceles triangle on grid.
    Defined by a corner and max Manhattan distance L:
      (i,j) is inside <=> manhattan((i,j), corner) <= L
    """
    corner: Index2D
    L: int

    def contains(self, i: int, j: int) -> bool:
        ci, cj = self.corner
        return abs(i - ci) + abs(j - cj) <= self.L

    @property
    def area(self) -> int:
        # number of lattice cells ~ (L+1)(L+2)/2
        return (self.L + 1) * (self.L + 2) // 2


@dataclass(frozen=True)
class HotArea:
    shape: str  # "rect" or "tri"
    rect: Optional[RectArea] = None
    tri: Optional[TriArea] = None

    @property
    def area(self) -> int:
        return self.rect.area if self.shape == "rect" else self.tri.area

    def contains(self, i: int, j: int) -> bool:
        return self.rect.contains(i, j) if self.shape == "rect" else self.tri.contains(i, j)

    def bounds(self) -> RectArea:
        """
        Return a bounding rectangle for iteration.
        For triangle we still iterate over a rectangle (the min/max of possible points).
        """
        if self.shape == "rect":
            return self.rect
        ci, cj = self.tri.corner
        L = self.tri.L
        # bounding box of manhattan ball
        return RectArea(ci - L, ci + L + 1, cj - L, cj + L + 1)


# -------------------------
# Hotspot detection
# -------------------------

def hotspots_mask(T: np.ndarray, alpha: float) -> np.ndarray:
    mu = float(np.mean(T))
    sigma = float(np.std(T))
    thr = mu + alpha * sigma
    return T > thr


# -------------------------
# Mapping hotspots to nearby liquid cells
# -------------------------

def map_hotspots_to_liquid(
    net: CoolingNetwork,
    hot: np.ndarray,
    radius: int = 2
) -> Set[Index2D]:
    """
    Paper: map hotspots on source layers to nearby liquid cells in channel layers.
    Simplification: we search within a radius and pick closest LIQUID (or active) cell.
    Returns a set of liquid cell coords.
    """
    n, m = net.C.shape
    mapped: Set[Index2D] = set()

    hot_positions = np.argwhere(hot)
    for (i, j) in hot_positions:
        i = int(i); j = int(j)

        # if it's already liquid, map to itself
        if net.C[i, j] == CellType.LIQUID:
            mapped.add((i, j))
            continue

        # search nearest liquid within manhattan radius
        best: Optional[Index2D] = None
        best_d = 10**9

        for di in range(-radius, radius + 1):
            for dj in range(-radius, radius + 1):
                ii, jj = i + di, j + dj
                if not (0 <= ii < n and 0 <= jj < m):
                    continue
                d = abs(di) + abs(dj)
                if d > radius:
                    continue
                if net.C[ii, jj] == CellType.LIQUID:
                    if d < best_d:
                        best_d = d
                        best = (ii, jj)

        if best is not None:
            mapped.add(best)

    return mapped


# -------------------------
# Clustering (connected components) of mapped liquid cells
# -------------------------

def clusters_from_set(net: CoolingNetwork, cells: Set[Index2D]) -> List[List[Index2D]]:
    """
    Connected components in 4-neighborhood restricted to 'cells'.
    """
    cells_set = set(cells)
    visited: Set[Index2D] = set()
    comps: List[List[Index2D]] = []

    for start in list(cells_set):
        if start in visited:
            continue
        q = deque([start])
        visited.add(start)
        comp: List[Index2D] = []

        while q:
            i, j = q.popleft()
            comp.append((i, j))
            for ni, nj in net.neighbors4(i, j):
                nxt = (ni, nj)
                if nxt in cells_set and nxt not in visited:
                    visited.add(nxt)
                    q.append(nxt)

        comps.append(comp)

    return comps


# -------------------------
# Hot area selection: rectangle vs 4 triangle orientations
# -------------------------

def choose_hot_area(comp: List[Index2D], pad: int, n: int) -> HotArea:
    rs = [i for i, _ in comp]
    cs = [j for _, j in comp]
    rmin, rmax = min(rs), max(rs)
    cmin, cmax = min(cs), max(cs)

    # padded rectangle
    r0 = max(0, rmin - pad)
    r1 = min(n, rmax + pad + 1)
    c0 = max(0, cmin - pad)
    c1 = min(n, cmax + pad + 1)
    rect = RectArea(r0, r1, c0, c1)

    # triangle candidates (corners at padded rectangle corners)
    corners = [
        (r0, c0),
        (r0, c1 - 1),
        (r1 - 1, c0),
        (r1 - 1, c1 - 1),
    ]

    best_tri: Optional[TriArea] = None
    best_area = 10**18

    for corner in corners:
        ci, cj = corner
        L = 0
        for (i, j) in comp:
            d = abs(i - ci) + abs(j - cj)
            if d > L:
                L = d
        tri = TriArea(corner=corner, L=L)
        if tri.area < best_area:
            best_area = tri.area
            best_tri = tri

    # choose minimal area shape
    if best_tri is not None and best_tri.area < rect.area:
        return HotArea(shape="tri", tri=best_tri)
    return HotArea(shape="rect", rect=rect)


# -------------------------
# Determine flow direction near an area (based on I/O)
# -------------------------

def quadrant_bounds(n: int, qi: int) -> RectArea:
    mid = n // 2
    if qi == 0:  # TL
        return RectArea(0, mid, 0, mid)
    if qi == 1:  # TR
        return RectArea(0, mid, mid, n)
    if qi == 2:  # BL
        return RectArea(mid, n, 0, mid)
    # BR
    return RectArea(mid, n, mid, n)


IOQ = Tuple[int, int, int, int]  # best_pat.q

def quadrant_of_point(n: int, i: int, j: int) -> int:
    mid = n // 2
    if i < mid and j < mid:
        return 0  # TL
    if i < mid and j >= mid:
        return 1  # TR
    if i >= mid and j < mid:
        return 2  # BL
    return 3      # BR

def flow_direction_from_ioq(n: int, area_bounds, io_q: IOQ) -> str:
    """
    area_bounds має поля r0,r1,c0,c1 (bounding rect).
    mode==0 => horizontal (left->right), mode==1 => vertical (top->bottom)
    """
    ci = (area_bounds.r0 + area_bounds.r1) // 2
    cj = (area_bounds.c0 + area_bounds.c1) // 2
    qi = quadrant_of_point(n, ci, cj)
    mode = io_q[qi]
    return "horizontal" if mode == 0 else "vertical"


# -------------------------
# Apply straight-channels inside hot area
# -------------------------

def apply_straight_channels(net: CoolingNetwork, area: HotArea, direction: str, stripe: int = 2) -> None:
    """
    Keep LIQUID stripes aligned with direction inside area; convert other LIQUID to SILICON.
    - direction "horizontal": keep every 'stripe'-th ROW inside area
    - direction "vertical": keep every 'stripe'-th COL inside area
    Never touch TSV/INLET/OUTLET.
    """
    C = net.C
    n, m = C.shape

    b = area.bounds()
    r0 = max(0, b.r0); r1 = min(n, b.r1)
    c0 = max(0, b.c0); c1 = min(m, b.c1)

    if direction == "horizontal":
        for i in range(r0, r1):
            keep_row = ((i - r0) % stripe == 0)
            for j in range(c0, c1):
                if not area.contains(i, j):
                    continue
                if C[i, j] in (CellType.TSV, CellType.INLET, CellType.OUTLET):
                    continue
                if C[i, j] == CellType.LIQUID and not keep_row:
                    C[i, j] = CellType.SILICON
    else:
        for j in range(c0, c1):
            keep_col = ((j - c0) % stripe == 0)
            for i in range(r0, r1):
                if not area.contains(i, j):
                    continue
                if C[i, j] in (CellType.TSV, CellType.INLET, CellType.OUTLET):
                    continue
                if C[i, j] == CellType.LIQUID and not keep_col:
                    C[i, j] = CellType.SILICON


# -------------------------
# Stage 2 driver (straight-channels only)
# -------------------------

@dataclass(frozen=True)
class Stage2Result:
    net: CoolingNetwork
    thermal_before: ThermalResult
    thermal_after: ThermalResult
    n_hotspots: int
    n_clusters: int


def stage2_straight_channels(
    net: CoolingNetwork,
    power_map: np.ndarray,
    thermal_fn,
    alpha_1: float,
    io_q: IOQ,
    map_radius: int = 2,
    pad: int = 1,
    stripe: int = 2
) -> Stage2Result:
    thermal_before: ThermalResult = thermal_fn(net, power_map)
    hot = hotspots_mask(thermal_before.T, alpha_1)
    n_hotspots = int(np.sum(hot))

    new_net = net.clone()

    mapped = map_hotspots_to_liquid(new_net, hot, radius=map_radius)
    comps = clusters_from_set(new_net, mapped)

    n = new_net.C.shape[0]
    for comp in comps:
        area = choose_hot_area(comp, pad=pad, n=n)
        direction = flow_direction_from_ioq(n, area.bounds(), io_q)
        apply_straight_channels(new_net, area, direction=direction, stripe=stripe)

    new_net.prune_irregular()

    # sanity: після patterning мережа має залишатись зв’язною
    if not new_net.has_inlet_to_outlet_path():
        # якщо так сталося, значить stripe/pad агресивні — краще повернути стару мережу
        new_net = net.clone()

    thermal_after: ThermalResult = thermal_fn(new_net, power_map)

    return Stage2Result(
        net=new_net,
        thermal_before=thermal_before,
        thermal_after=thermal_after,
        n_hotspots=n_hotspots,
        n_clusters=len(comps),
    )