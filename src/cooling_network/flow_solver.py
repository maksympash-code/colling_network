from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from src.cooling_network.grid import CoolingNetwork
from src.cooling_network.types import CellType
from src.cooling_network.physics import ModelParams

Index2D = Tuple[int, int]


@dataclass(frozen=True, slots=True)
class FlowResult:
    P: np.ndarray
    V_edge: Dict[Tuple[Index2D, Index2D], float]
    E_pump: float


def hydraulic_resistance(params: ModelParams) -> float:
    mu = params.fluid.mu
    Lc = params.geom.Lc
    Hc = params.geom.Hc
    Wc = params.geom.Wc
    R = 8.0 * mu * Lc * (Hc + Wc) ** 2 / ((Hc * Wc) ** 3)
    return float(R)


def reachable_active_mask(net: CoolingNetwork) -> np.ndarray:
    """
    Active cells reachable from at least one fixed-pressure boundary cell
    (INLET or OUTLET). This removes floating liquid islands from the flow solve.
    """
    C = net.C
    n, m = C.shape
    reachable = np.zeros((n, m), dtype=bool)

    q = deque()
    for i in range(n):
        for j in range(m):
            if C[i, j] in (CellType.INLET, CellType.OUTLET):
                reachable[i, j] = True
                q.append((i, j))

    while q:
        i, j = q.popleft()
        for ni, nj in net.neighbors4(i, j):
            if reachable[ni, nj]:
                continue
            if C[ni, nj] not in (CellType.LIQUID, CellType.INLET, CellType.OUTLET):
                continue
            reachable[ni, nj] = True
            q.append((ni, nj))

    return reachable


def solve_flow(net: CoolingNetwork, P_pump: float, params: ModelParams) -> FlowResult:
    """
    Simplified pressure solve:
      - Dirichlet: P = P_pump on INLET, P = 0 on OUTLET
      - Unknowns: reachable LIQUID cells only
      - Floating active islands are ignored in the flow solve
    """
    C = net.C
    n, m = C.shape
    R = hydraulic_resistance(params)
    g = 1.0 / R

    reachable = reachable_active_mask(net)

    idx = -np.ones((n, m), dtype=np.int32)
    liquid_cells: List[Index2D] = []
    k = 0
    for i in range(n):
        for j in range(m):
            if C[i, j] == CellType.LIQUID and reachable[i, j]:
                idx[i, j] = k
                liquid_cells.append((i, j))
                k += 1

    N = k

    P = np.full((n, m), np.nan, dtype=np.float64)
    P[(C == CellType.INLET) & reachable] = float(P_pump)
    P[(C == CellType.OUTLET) & reachable] = 0.0

    if N == 0:
        return FlowResult(P=P, V_edge={}, E_pump=0.0)

    rows = []
    cols = []
    data = []
    b = np.zeros(N, dtype=np.float64)

    def fixed_pressure(i: int, j: int) -> float | None:
        if not reachable[i, j]:
            return None
        if C[i, j] == CellType.INLET:
            return float(P_pump)
        if C[i, j] == CellType.OUTLET:
            return 0.0
        return None

    for (i, j) in liquid_cells:
        ii = idx[i, j]
        diag = 0.0

        for (ni, nj) in net.neighbors4(i, j):
            if not reachable[ni, nj]:
                continue
            if C[ni, nj] not in (CellType.LIQUID, CellType.INLET, CellType.OUTLET):
                continue

            diag += g
            fp = fixed_pressure(ni, nj)
            if fp is None:
                jj = idx[ni, nj]
                rows.append(ii); cols.append(jj); data.append(-g)
            else:
                b[ii] += g * fp

        rows.append(ii); cols.append(ii); data.append(diag)

    A = csr_matrix((data, (rows, cols)), shape=(N, N))

    # very small regularization, щоб не валитись на поганих демо-сітках
    if N > 0:
        A = A + 1e-12 * csr_matrix(np.eye(N))

    x = spsolve(A, b)

    for (i, j) in liquid_cells:
        P[i, j] = float(x[idx[i, j]])

    V_edge: Dict[Tuple[Index2D, Index2D], float] = {}
    E = 0.0
    seen_undirected = set()

    for i in range(n):
        for j in range(m):
            if not reachable[i, j]:
                continue
            if C[i, j] not in (CellType.LIQUID, CellType.INLET, CellType.OUTLET):
                continue

            Pi = P[i, j]
            if not np.isfinite(Pi):
                continue

            for (ni, nj) in net.neighbors4(i, j):
                if not reachable[ni, nj]:
                    continue
                if C[ni, nj] not in (CellType.LIQUID, CellType.INLET, CellType.OUTLET):
                    continue

                Pj = P[ni, nj]
                if not np.isfinite(Pj):
                    continue

                V = (Pi - Pj) / R
                V_edge[((i, j), (ni, nj))] = float(V)

                key = tuple(sorted(((i, j), (ni, nj))))
                if key not in seen_undirected:
                    dP = float(Pi - Pj)
                    E += (dP * dP) / R
                    seen_undirected.add(key)

    return FlowResult(P=P, V_edge=V_edge, E_pump=float(E))