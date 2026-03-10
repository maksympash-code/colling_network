from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from src.cooling_network.grid import CoolingNetwork
from src.cooling_network.types import CellType, ThermalResult


@dataclass(frozen=True)
class IOPattern:
    """
    Four quadrants: (top-left, top-right, bottom-left, bottom-right)
    Each value: 0 -> inlet on left, outlet on right
                1 -> inlet on top, outlet on bottom
    Це спрощення. Головне: 16 комбінацій і різні глобальні напрямки.
    """
    q: Tuple[int, int, int, int]


def generate_16_patterns() -> List[IOPattern]:
    patterns = []
    for mask in range(16):
        q = (
            (mask >> 0) & 1,
            (mask >> 1) & 1,
            (mask >> 2) & 1,
            (mask >> 3) & 1,
        )
        patterns.append(IOPattern(q=q))
    return patterns


def apply_io_pattern(net: CoolingNetwork, pattern: IOPattern) -> CoolingNetwork:
    """
    Apply I/O only on the OUTER boundary of the whole layer.
    Four quadrants are used only to choose which OUTER edge segment
    gets inlet/outlet.

    mode == 0 : horizontal flow
        TL -> inlet on LEFT(top half), outlet on TOP(left half)
        TR -> inlet on TOP(right half), outlet on RIGHT(top half)
        BL -> inlet on LEFT(bottom half), outlet on BOTTOM(left half)
        BR -> inlet on BOTTOM(right half), outlet on RIGHT(bottom half)

    mode == 1 : swapped orientation
        TL -> inlet on TOP(left half), outlet on LEFT(top half)
        TR -> inlet on RIGHT(top half), outlet on TOP(right half)
        BL -> inlet on BOTTOM(left half), outlet on LEFT(bottom half)
        BR -> inlet on RIGHT(bottom half), outlet on BOTTOM(right half)
    """
    new_net = net.clone()
    C = new_net.C
    n, m = C.shape
    assert n == m, "for simplicity assume square grid"

    # clear previous inlet/outlet tags
    C[(C == CellType.INLET) | (C == CellType.OUTLET)] = CellType.LIQUID

    mid = n // 2

    def mark_if_liquid(i: int, j: int, tag: CellType):
        if C[i, j] == CellType.LIQUID:
            C[i, j] = tag

    # segments on OUTER boundary only
    top_left = [(0, j) for j in range(0, mid)]
    top_right = [(0, j) for j in range(mid, n)]
    bottom_left = [(n - 1, j) for j in range(0, mid)]
    bottom_right = [(n - 1, j) for j in range(mid, n)]

    left_top = [(i, 0) for i in range(0, mid)]
    left_bottom = [(i, 0) for i in range(mid, n)]
    right_top = [(i, n - 1) for i in range(0, mid)]
    right_bottom = [(i, n - 1) for i in range(mid, n)]

    quadrant_edges = {
        0: (left_top, top_left),        # TL
        1: (top_right, right_top),      # TR
        2: (left_bottom, bottom_left),  # BL
        3: (bottom_right, right_bottom) # BR
    }

    for qi in range(4):
        edge_a, edge_b = quadrant_edges[qi]
        mode = pattern.q[qi]

        if mode == 0:
            inlet_segment = edge_a
            outlet_segment = edge_b
        else:
            inlet_segment = edge_b
            outlet_segment = edge_a

        for (i, j) in inlet_segment:
            mark_if_liquid(i, j, CellType.INLET)

        for (i, j) in outlet_segment:
            mark_if_liquid(i, j, CellType.OUTLET)

    return new_net


def select_best_io(
    base_net: CoolingNetwork,
    power_map: np.ndarray,
    thermal_fn
) -> Tuple[CoolingNetwork, IOPattern, ThermalResult]:
    """
    Try 16 I/O patterns, return the best by minimal max_T.
    """
    patterns = generate_16_patterns()

    best_net = None
    best_pat = None
    best_res = None
    best_max = float("inf")

    for pat in patterns:
        net = apply_io_pattern(base_net, pat)

        # sanity: must have connectivity, otherwise skip
        if not net.has_inlet_to_outlet_path():
            continue

        res: ThermalResult = thermal_fn(net, power_map)
        if res.max_T < best_max:
            best_max = res.max_T
            best_net = net
            best_pat = pat
            best_res = res

    if best_net is None:
        raise RuntimeError("No feasible IO pattern produced inlet->outlet path.")

    return best_net, best_pat, best_res