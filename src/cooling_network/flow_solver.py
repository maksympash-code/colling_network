from __future__ import annotations

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
    # pressure at active cells (LIQUID/INLET/OUTLET), stored as array with NaN for non-active
    P: np.ndarray
    # volumetric flow per directed edge: ((i,j),(ni,nj)) -> V (m^3/s), sign = from cell -> neighbor
    V_edge: Dict[Tuple[Index2D, Index2D], float]
    # pumping energy (paper eq(3)) approximated as sum over undirected edges: ΔP * |V|
    E_pump: float


def hydraulic_resistance(params: ModelParams) -> float:
    """
    Paper eq(4):
    ΔP = (8 μ Lc (Hc+Wc)^2 / (Hc*Wc)^3) * V
    => R = coefficient
    """
    mu = params.fluid.mu
    Lc = params.geom.Lc
    Hc = params.geom.Hc
    Wc = params.geom.Wc
    R = 8.0 * mu * Lc * (Hc + Wc) ** 2 / ((Hc * Wc) ** 3)
    return float(R)


def solve_flow(net: CoolingNetwork, P_pump: float, params: ModelParams) -> FlowResult:
    """
    Simplified contest-like pressure solve:
    - Dirichlet: P = P_pump on INLET cells, P = 0 on OUTLET cells
    - Unknowns: LIQUID cells only (optional; inlet/outlet fixed)
    - Interior equations: sum_g (P_i - P_j) = 0  (Kirchhoff), g = 1/R
    """
    C = net.C
    n, m = C.shape
    R = hydraulic_resistance(params)
    g = 1.0 / R  # conductance

    # index mapping for LIQUID unknowns
    idx = -np.ones((n, m), dtype=np.int32)
    liquid_cells: List[Index2D] = []
    k = 0
    for i in range(n):
        for j in range(m):
            if C[i, j] == CellType.LIQUID:
                idx[i, j] = k
                liquid_cells.append((i, j))
                k += 1

    N = k
    if N == 0:
        # no liquid -> no flow
        P = np.full((n, m), np.nan, dtype=np.float64)
        return FlowResult(P=P, V_edge={}, E_pump=0.0)

    rows = []
    cols = []
    data = []
    b = np.zeros(N, dtype=np.float64)

    def fixed_pressure(i: int, j: int) -> float | None:
        if C[i, j] == CellType.INLET:
            return float(P_pump)
        if C[i, j] == CellType.OUTLET:
            return 0.0
        return None

    for (i, j) in liquid_cells:
        ii = idx[i, j]
        diag = 0.0

        for (ni, nj) in net.neighbors4(i, j):
            if C[ni, nj] not in (CellType.LIQUID, CellType.INLET, CellType.OUTLET):
                continue

            diag += g
            fp = fixed_pressure(ni, nj)
            if fp is None:
                # neighbor is LIQUID unknown
                jj = idx[ni, nj]
                rows.append(ii); cols.append(jj); data.append(-g)
            else:
                # neighbor fixed
                b[ii] += g * fp

        rows.append(ii); cols.append(ii); data.append(diag)

    A = csr_matrix((data, (rows, cols)), shape=(N, N))
    x = spsolve(A, b)  # LIQUID pressures

    # build full pressure field with NaN for non-active
    P = np.full((n, m), np.nan, dtype=np.float64)
    P[C == CellType.INLET] = float(P_pump)
    P[C == CellType.OUTLET] = 0.0
    for (i, j) in liquid_cells:
        P[i, j] = float(x[idx[i, j]])

    # compute directed edge flows V = (P_i - P_j)/R, for active neighbors
    V_edge: Dict[Tuple[Index2D, Index2D], float] = {}
    E = 0.0
    seen_undirected = set()

    for i in range(n):
        for j in range(m):
            if C[i, j] not in (CellType.LIQUID, CellType.INLET, CellType.OUTLET):
                continue
            Pi = P[i, j]
            for (ni, nj) in net.neighbors4(i, j):
                if C[ni, nj] not in (CellType.LIQUID, CellType.INLET, CellType.OUTLET):
                    continue
                Pj = P[ni, nj]
                V = (Pi - Pj) / R  # positive means i->neighbor
                V_edge[((i, j), (ni, nj))] = float(V)

                # energy per undirected edge: ΔP * |V| = (ΔP^2)/R
                key = tuple(sorted(((i, j), (ni, nj))))
                if key not in seen_undirected:
                    dP = float(Pi - Pj)
                    E += (dP * dP) / R
                    seen_undirected.add(key)

    return FlowResult(P=P, V_edge=V_edge, E_pump=float(E))