import numpy as np


def make_power_map_from_areal_density(
    n: int,
    dx: float,
    dy: float,
    base_w_per_cm2: float = 10.0,     # базова щільність
    hot_w_per_cm2: float = 200.0      # hotspot щільність
) -> np.ndarray:
    # W/cm^2 -> W/m^2
    base = base_w_per_cm2 * 1e4
    hot = hot_w_per_cm2 * 1e4

    P = np.full((n, n), base * dx * dy, dtype=np.float64)
    P[n // 2, n // 2] = hot * dx * dy
    return P