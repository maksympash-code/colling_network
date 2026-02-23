from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import inspect

import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import gmres, lgmres, spilu, LinearOperator
from scipy.sparse.linalg import splu  # direct fallback

from src.cooling_network.grid import CoolingNetwork
from src.cooling_network.types import CellType
from src.cooling_network.physics import ModelParams
from src.cooling_network.flow_solver import FlowResult

Index2D = Tuple[int, int]


@dataclass(frozen=True, slots=True)
class ThermalSolveResult:
    T_source: np.ndarray
    T_channel: np.ndarray
    max_T: float
    grad_T: float


def _gmres_solve(A, b, M, rtol: float, restart: int, maxiter: int):
    sig = inspect.signature(gmres).parameters
    if "rtol" in sig:
        return gmres(A, b, M=M, x0=None, rtol=rtol, atol=0.0, restart=restart, maxiter=maxiter)
    return gmres(A, b, M=M, x0=None, tol=rtol, restart=restart, maxiter=maxiter)


def _lgmres_solve(A, b, M, rtol: float, maxiter: int):
    sig = inspect.signature(lgmres).parameters
    if "rtol" in sig:
        return lgmres(A, b, M=M, x0=None, rtol=rtol, atol=0.0, maxiter=maxiter)
    return lgmres(A, b, M=M, x0=None, tol=rtol, maxiter=maxiter)


def solve_thermal_one_source_one_channel(
    net: CoolingNetwork,
    power_map: np.ndarray,
    flow: FlowResult,
    params: ModelParams,
    rtol: float = 1e-8,
    maxiter: int = 1500,
    restart: int = 200,
    ilu_drop_tol: float = 1e-6,
    ilu_fill_factor: float = 40.0,
    h_amb: float = 1e3,
    direct_fallback: bool = True,
) -> ThermalSolveResult:
    """
    2-layer steady-state thermal solve:
      - Source (solid) + Channel (fluid)
      - In-plane conduction on both layers
      - Source<->fluid convection for active fluid cells
      - Fluid advection via VCCS-like couplings (appendix-style)
      - Dirichlet: INLET = T_in, non-active fluid = T_amb
      - Ambient convection on source boundary (anchor)
    """
    C = net.C
    n, m = C.shape
    assert power_map.shape == (n, m)

    # index maps
    def sid(i: int, j: int) -> int:
        return i * m + j

    def fid(i: int, j: int) -> int:
        return n * m + i * m + j

    N = 2 * n * m

    # geometry
    dx = params.geom.Wc
    dy = params.geom.Lc
    dz = params.geom.Hc  # we reuse for thickness if specific thickness not provided

    # materials / BC
    k_s = params.solid.k
    k_f = params.fluid.k
    h_cv = params.conv.h
    Cv = params.fluid.rho_cp
    T_in = params.bc.T_in
    T_amb = params.bc.T_amb

    # Optional thickness (if you later add it to physics.py)
    t_s = getattr(params.solid, "thickness", dz)
    t_f = getattr(params.fluid, "thickness", dz)

    # areas (finite-volume conductances)
    A_contact = dx * dy
    A_x_s = dy * t_s  # face area between (i,j) and (i±1,j) in source
    A_y_s = dx * t_s
    A_x_f = dy * t_f
    A_y_f = dx * t_f

    # face areas for velocity -> volumetric velocity
    A_x_flow = dy * dz
    A_y_flow = dx * dz

    rows, cols, data = [], [], []
    b = np.zeros(N, dtype=np.float64)

    def add(r: int, c: int, v: float) -> None:
        rows.append(r); cols.append(c); data.append(float(v))

    # conductances
    gx_s = k_s * A_x_s / dx
    gy_s = k_s * A_y_s / dy
    gx_f = k_f * A_x_f / dx
    gy_f = k_f * A_y_f / dy

    gcv = h_cv * A_contact
    gamb = h_amb * A_contact

    # Signed edge flow: use both directions
    def V_signed(u: Index2D, v: Index2D) -> float:
        # if stored as u->v
        val = flow.V_edge.get((u, v), None)
        if val is not None:
            return float(val)
        # else maybe stored as v->u
        val2 = flow.V_edge.get((v, u), None)
        if val2 is not None:
            return -float(val2)
        return 0.0

    def v_out(i: int, j: int, ni: int, nj: int) -> float:
        V = V_signed((i, j), (ni, nj))
        return V / (A_x_flow if ni != i else A_y_flow)

    # ---------------- SOURCE layer assembly ----------------
    for i in range(n):
        for j in range(m):
            r = sid(i, j)
            diag = 0.0

            # x neighbors (NOTE SIGN!)
            if i - 1 >= 0:
                add(r, sid(i - 1, j), -gx_s)
                diag += gx_s
            if i + 1 < n:
                add(r, sid(i + 1, j), -gx_s)
                diag += gx_s

            # y neighbors
            if j - 1 >= 0:
                add(r, sid(i, j - 1), -gy_s)
                diag += gy_s
            if j + 1 < m:
                add(r, sid(i, j + 1), -gy_s)
                diag += gy_s

            # coupling to fluid (active only)
            if C[i, j] in (CellType.LIQUID, CellType.INLET, CellType.OUTLET):
                add(r, fid(i, j), -gcv)
                diag += gcv

            # ambient convection on boundary (anchor)
            if i == 0 or i == n - 1 or j == 0 or j == m - 1:
                diag += gamb
                b[r] += gamb * T_amb

            add(r, r, diag)

            # heat injection (assume power_map is W per cell)
            b[r] += float(power_map[i, j])

    # ---------------- CHANNEL layer assembly ----------------
    adv_coeff = 0.5 * Cv

    for i in range(n):
        for j in range(m):
            r = fid(i, j)

            # non-active fluid: Dirichlet to ambient
            if C[i, j] not in (CellType.LIQUID, CellType.INLET, CellType.OUTLET):
                add(r, r, 1.0)
                b[r] = float(T_amb)
                continue

            # inlet: Dirichlet to T_in
            if C[i, j] == CellType.INLET:
                add(r, r, 1.0)
                b[r] = float(T_in)
                continue

            diag = 0.0

            # diffusion in fluid (NOTE SIGN!)
            if i - 1 >= 0:
                add(r, fid(i - 1, j), -gx_f); diag += gx_f
            if i + 1 < n:
                add(r, fid(i + 1, j), -gx_f); diag += gx_f

            if j - 1 >= 0:
                add(r, fid(i, j - 1), -gy_f); diag += gy_f
            if j + 1 < m:
                add(r, fid(i, j + 1), -gy_f); diag += gy_f

            # coupling to source
            add(r, sid(i, j), -gcv)
            diag += gcv

            # advection (appendix-style): v*(T_neighbor - T_center)
            for (ni, nj) in net.neighbors4(i, j):
                if C[ni, nj] not in (CellType.LIQUID, CellType.INLET, CellType.OUTLET):
                    continue
                v = v_out(i, j, ni, nj)  # signed
                diag += adv_coeff * v
                add(r, fid(ni, nj), -adv_coeff * v)

            add(r, r, diag)
            b[r] = 0.0

    G = csr_matrix((data, (rows, cols)), shape=(N, N))

    # ---------------- Scaling ----------------
    diagG = G.diagonal()
    scale = 1.0 / np.maximum(np.sqrt(np.abs(diagG)), 1e-12)
    D = diags(scale, 0, shape=(N, N), format="csr")
    G2 = (D @ G).tocsr()
    b2 = scale * b

    # ---------------- ILU preconditioner ----------------
    M = None
    try:
        ilu = spilu(G2.tocsc(), drop_tol=ilu_drop_tol, fill_factor=ilu_fill_factor, permc_spec="COLAMD")
        M = LinearOperator((N, N), ilu.solve)
    except Exception:
        M = None

    # ---------------- Solve ----------------
    sol, info = _gmres_solve(G2, b2, M, rtol=rtol, restart=restart, maxiter=maxiter)
    if info != 0:
        sol, info2 = _lgmres_solve(G2, b2, M, rtol=rtol, maxiter=maxiter * 2)
        if info2 != 0:
            if not direct_fallback:
                raise RuntimeError(f"GMRES/LGMRES did not converge, info={info}, info2={info2}")
            lu = splu(G2.tocsc())
            sol = lu.solve(b2)

    T_source = sol[: n * m].reshape((n, m))
    T_channel = sol[n * m :].reshape((n, m))

    max_T = float(np.max(T_source))
    grad_T = float(np.max(T_source) - np.min(T_source))

    return ThermalSolveResult(
        T_source=T_source,
        T_channel=T_channel,
        max_T=max_T,
        grad_T=grad_T,
    )