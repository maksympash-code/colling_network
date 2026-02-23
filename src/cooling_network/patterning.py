from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from src.cooling_network.grid import CoolingNetwork
from src.cooling_network.types import CellType


Index2D = Tuple[int, int]


@dataclass(frozen=True)
class HotArea:
    r0: int
    r1: int  # exclusive
    c0: int
    c1: int


def find_hotspots(T: np.ndarray, alpha: float) -> np.ndarray:
    """
    Return boolean mask of hotspots: T > mean + alpha * std.
    """
    mu = float(np.mean(T))
    sigma = float(np.std(T))
    thr = mu + alpha * sigma
    return T > thr


def connected_components(mask: np.ndarray, neighbors4_fn) -> List[List[Index2D]]:
    """
    mask: bool array (True = included)
    Returns list of components, each component is list of (i,j).
    """
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


def bounding_box(cells: List[Index2D], pad: int = 1, n: int | None = None) -> HotArea:
    rs = [i for i, _ in cells]
    cs = [j for _, j in cells]
    r0 = min(rs) - pad
    r1 = max(rs) + pad + 1
    c0 = min(cs) - pad
    c1 = max(cs) + pad + 1

    if n is not None:
        r0 = max(0, r0); c0 = max(0, c0)
        r1 = min(n, r1); c1 = min(n, c1)

    return HotArea(r0=r0, r1=r1, c0=c0, c1=c1)


def apply_straight_channel_pattern(net: CoolingNetwork, area: HotArea, prefer: str = "auto") -> None:
    """
    Simplified straight-channel pattern:
    - choose vertical or horizontal stripes inside the hot area,
      keep stripes as LIQUID, convert other LIQUID to SILICON
    - never touch TSV/INLET/OUTLET.
    prefer: "vertical" | "horizontal" | "auto"
    """
    C = net.C
    r0, r1, c0, c1 = area.r0, area.r1, area.c0, area.c1

    h = r1 - r0
    w = c1 - c0

    if prefer == "auto":
        direction = "vertical" if w >= h else "horizontal"
    else:
        direction = prefer

    if direction == "vertical":
        for i in range(r0, r1):
            for j in range(c0, c1):
                if C[i, j] in (CellType.TSV, CellType.INLET, CellType.OUTLET):
                    continue
                if C[i, j] == CellType.LIQUID:
                    # keep stripes (columns)
                    if (j - c0) % 2 == 0:
                        continue
                    C[i, j] = CellType.SILICON
    else:
        for i in range(r0, r1):
            for j in range(c0, c1):
                if C[i, j] in (CellType.TSV, CellType.INLET, CellType.OUTLET):
                    continue
                if C[i, j] == CellType.LIQUID:
                    # keep stripes (rows)
                    if (i - r0) % 2 == 0:
                        continue
                    C[i, j] = CellType.SILICON


def hotspots_near_center(mask: np.ndarray, frac: float = 0.25) -> bool:
    """
    Heuristic: if a substantial portion of hotspots are inside the central square region,
    we consider 'center hotspots' => enable corner barriers.
    frac = size of central region side relative to N (0.25 => central N/4..3N/4).
    """
    n, m = mask.shape
    assert n == m
    s0 = int(n * frac)
    s1 = int(n * (1.0 - frac))
    center = mask[s0:s1, s0:s1]
    # use ratio threshold
    total = int(np.sum(mask))
    if total == 0:
        return False
    return (int(np.sum(center)) / total) >= 0.35


def add_corner_barriers(net: CoolingNetwork, length: int = 6, thickness: int = 1) -> None:
    """
    Simplified corner-barrier pattern:
    Add silicon 'walls' in four corners to discourage shortcut flow.
    We make diagonal-ish blocks: here simplified to axis-aligned L-shapes.

    length: how far the barrier extends from corner
    thickness: barrier thickness
    """
    C = net.C
    n, m = C.shape
    assert n == m

    def paint_silicon(r0, r1, c0, c1):
        for i in range(r0, r1):
            for j in range(c0, c1):
                if C[i, j] in (CellType.TSV, CellType.INLET, CellType.OUTLET):
                    continue
                if C[i, j] == CellType.LIQUID:
                    C[i, j] = CellType.SILICON

    L = length
    t = thickness

    paint_silicon(0, t, 0, L)      # top bar
    paint_silicon(0, L, 0, t)      # left bar

    paint_silicon(0, t, n - L, n)  # top bar
    paint_silicon(0, L, n - t, n)  # right bar

    paint_silicon(n - t, n, 0, L)  # bottom bar
    paint_silicon(n - L, n, 0, t)  # left bar

    paint_silicon(n - t, n, n - L, n)  # bottom bar
    paint_silicon(n - L, n, n - t, n)  # right bar


def stage2_patterning(
    net: CoolingNetwork,
    T: np.ndarray,
    alpha_1: float,
    alpha_2: float,
    pad: int = 1,
    barrier_length: int = 8,
) -> CoolingNetwork:
    """
    Implements Stage 2 in simplified form:
    1) detect hotspots with alpha_1
    2) apply straight-channel patterns in hot areas
    3) if hotspots are near center -> add corner barriers, re-detect with alpha_2, re-apply patterns
    Returns modified net (clone).
    """
    new_net = net.clone()
    n, m = new_net.C.shape
    assert n == m

    hot = find_hotspots(T, alpha_1)
    comps = connected_components(hot, new_net.neighbors4)

    for comp in comps:
        area = bounding_box(comp, pad=pad, n=n)
        apply_straight_channel_pattern(new_net, area, prefer="auto")

    new_net.prune_irregular()

    if hotspots_near_center(hot):
        add_corner_barriers(new_net, length=barrier_length, thickness=1)

        hot2 = find_hotspots(T, alpha_2)
        comps2 = connected_components(hot2, new_net.neighbors4)
        for comp in comps2:
            area = bounding_box(comp, pad=pad, n=n)
            apply_straight_channel_pattern(new_net, area, prefer="auto")

        new_net.prune_irregular()

    return new_net