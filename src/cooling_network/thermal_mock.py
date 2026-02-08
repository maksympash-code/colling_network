from collections import deque

import numpy as np

from src.cooling_network.grid import CoolingNetwork
from src.cooling_network.types import ThermalResult


def distance_to_active(net: CoolingNetwork) -> np.ndarray:
    n, m = net.C.shape
    dist = np.full((n,m), fill_value=10**9, dtype=np.int32)

    q = deque()
    for i in range(n):
        for j in range(m):
            if net.is_active(i, j):
                dist[i, j] = 0
                q.append((i, j))

    while q:
        i, j = q.popleft()
        d = dist[i, j]
        for ni, nj in net.neighbors4(i, j):
            if dist[ni, nj] > d + 1:
                dist[ni, nj] = d + 1
                q.append((ni, nj))

    return dist

def simulate_temperature(
        net: CoolingNetwork,
        power: np.ndarray,
        t_amb: float = 25.0,
        k_cool: float = 1.0,
        eps: float = 1e-9
) -> ThermalResult:
    dist = distance_to_active(net).astype(np.float64)

    denom = 1.0 + k_cool * dist
    T = t_amb + power / (denom + eps)

    max_T = float(np.max(T))
    grad_T = max_T - float(np.min(T))

    return ThermalResult(T=T, max_T=max_T, grad_T=grad_T)