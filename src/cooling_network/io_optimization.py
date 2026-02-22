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
    Apply I/O to a *copy* of net.
    We assume only one channel layer for simplified model.

    We place continuous inlets/outlets at the boundary segments of each quadrant.
    """
    new_net = net.clone()
    C = new_net.C
    n, m = C.shape
    assert n == m, "for simplicity assume square grid"

    # Clear previous inlet/outlet tags (keep LIQUID/SILICON/TSV)
    C[(C == CellType.INLET) | (C == CellType.OUTLET)] = CellType.LIQUID

    mid = n // 2

    # Quadrant bounds: [0,mid) x [0,mid), etc.
    quads = [
        (0, mid, 0, mid),     # TL
        (0, mid, mid, n),     # TR
        (mid, n, 0, mid),     # BL
        (mid, n, mid, n),     # BR
    ]

    for qi, (r0, r1, c0, c1) in enumerate(quads):
        mode = pattern.q[qi]

        if mode == 0:
            # inlet on left edge of the quadrant boundary, outlet on right edge
            # Left boundary column = c0, right boundary column = c1-1
            for r in range(r0, r1):
                if C[r, c0] == CellType.LIQUID:
                    C[r, c0] = CellType.INLET
                if C[r, c1 - 1] == CellType.LIQUID:
                    C[r, c1 - 1] = CellType.OUTLET
        else:
            # inlet on top edge, outlet on bottom edge
            for c in range(c0, c1):
                if C[r0, c] == CellType.LIQUID:
                    C[r0, c] = CellType.INLET
                if C[r1 - 1, c] == CellType.LIQUID:
                    C[r1 - 1, c] = CellType.OUTLET

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