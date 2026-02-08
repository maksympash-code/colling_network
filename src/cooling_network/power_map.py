import numpy as np


def make_power_map(
        n: int,
        hotspots = None,
        base: float = 1.0,
        hot: float = 50.0
) -> np.ndarray:
    P = np.full((n, n), fill_value=base, dtype=np.float64)
    if hotspots is None:
        hotspots = [(n // 2, n // 2)]

    for (i, j) in hotspots:
        P[i, j] += hot

    return P