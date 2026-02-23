from typing import Iterable

def hotspots_near_center_by_ratio(hot_mask: np.ndarray, center_frac: float = 0.25, ratio_thr: float = 0.35) -> bool:
    """
    center_frac=0.25 -> central square [0.25N,0.75N]×[0.25N,0.75N]
    ratio_thr=0.35 -> якщо >=35% всіх hotspots лежить у центрі -> вмикаємо barriers.
    """
    n, m = hot_mask.shape
    assert n == m
    total = int(np.sum(hot_mask))
    if total == 0:
        return False

    s0 = int(n * center_frac)
    s1 = int(n * (1.0 - center_frac))
    center = hot_mask[s0:s1, s0:s1]
    return (int(np.sum(center)) / total) >= ratio_thr


def paint_silicon_if_liquid(net: CoolingNetwork, cells: Iterable[Index2D]) -> None:
    """
    Convert LIQUID -> SILICON for given cells; do not touch TSV/INLET/OUTLET.
    """
    C = net.C
    for i, j in cells:
        if not net.in_bounds(i, j):
            continue
        if C[i, j] in (CellType.TSV, CellType.INLET, CellType.OUTLET):
            continue
        if C[i, j] == CellType.LIQUID:
            C[i, j] = CellType.SILICON


def barrier_cells_for_corner(n: int, corner_id: int, length: int, theta_deg: int, thickness: int = 1) -> List[Index2D]:
    """
    Discretized corner barrier as a ray going into the chip with slope angle theta.
    angle is measured from horizontal axis toward vertical, in [0..90].
    corner_id:
      0 = top-left, 1 = top-right, 2 = bottom-left, 3 = bottom-right
    """
    theta = np.deg2rad(theta_deg)
    # direction components in "interior" (positive magnitude)
    di = float(np.sin(theta))
    dj = float(np.cos(theta))

    # sign depending on corner
    if corner_id == 0:      # TL: +i, +j
        si, sj = +1, +1
        ci, cj = 0, 0
    elif corner_id == 1:    # TR: +i, -j
        si, sj = +1, -1
        ci, cj = 0, n - 1
    elif corner_id == 2:    # BL: -i, +j
        si, sj = -1, +1
        ci, cj = n - 1, 0
    else:                   # BR: -i, -j
        si, sj = -1, -1
        ci, cj = n - 1, n - 1

    cells: List[Index2D] = []

    # accumulate along the ray
    x = 0.0
    y = 0.0
    for t in range(length):
        ii = int(round(ci + si * x))
        jj = int(round(cj + sj * y))
        cells.append((ii, jj))

        # thickness: paint a small perpendicular cross (cheap and effective)
        for k in range(1, thickness):
            # perpendicular approx: swap components
            cells.append((ii + k * (-sj), jj + k * (si)))
            cells.append((ii - k * (-sj), jj - k * (si)))

        x += di
        y += dj

    return cells


def apply_corner_barriers(net: CoolingNetwork, params: List[Tuple[int, int]], thickness: int = 1) -> None:
    """
    params: list of 4 tuples (length, theta_deg) for corners [TL, TR, BL, BR]
    """
    n, _ = net.C.shape
    for corner_id, (L, theta) in enumerate(params):
        cells = barrier_cells_for_corner(n, corner_id, length=L, theta_deg=theta, thickness=thickness)
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
    Returns best per-corner params [(L,theta)] for 4 corners.
    Step 1 (paper-like): choose best (L,theta) for each corner individually (min max_T).
    """
    best_params: List[Tuple[int, int]] = []

    for corner_id in range(4):
        best = None
        best_max = float("inf")

        for L in length_candidates:
            for theta in theta_candidates:
                net = base_net.clone()
                apply_corner_barriers(net, params=[
                    (0, 45), (0, 45), (0, 45), (0, 45)   # dummy, will override one corner
                ], thickness=thickness)

                # apply only current corner
                cells = barrier_cells_for_corner(net.C.shape[0], corner_id, L, theta, thickness)
                paint_silicon_if_liquid(net, cells)

                net.prune_irregular()
                if not net.has_inlet_to_outlet_path():
                    continue

                res = thermal_fn(net, power_map)
                if res.max_T < best_max:
                    best_max = res.max_T
                    best = (L, theta)

        # fallback if nothing feasible
        if best is None:
            best = (length_candidates[len(length_candidates)//2], theta_candidates[len(theta_candidates)//2])

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
    Paper-like simplification of line search: scale all 4 lengths together, keep thetas.
    Choose the scale that minimizes max_T.
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