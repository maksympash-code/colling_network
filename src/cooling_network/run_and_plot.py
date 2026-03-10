from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

import src.cooling_network.config as cfg
from src.cooling_network.types import CellType, ThermalResult
from src.cooling_network.grid import CoolingNetwork
from src.cooling_network.physics import (
    FluidGeom, FluidProps, SolidProps, ConvectionProps, BoundaryTemps, ModelParams
)
from src.cooling_network.flow_solver import solve_flow
from src.cooling_network.thermal_solver import solve_thermal_one_source_one_channel
from src.cooling_network.io_optimization import select_best_io
from src.cooling_network.stage2_straight_channels import stage2_straight_channels

# Optional full Stage 2
try:
    from src.cooling_network.stage2_corner_barriers import stage2_full_patterning
    HAVE_STAGE2_FULL = True
except Exception:
    HAVE_STAGE2_FULL = False

# Optional Stage 3
try:
    from src.cooling_network.stage3_pruner_paper import run_stage3_pruning
    HAVE_STAGE3 = True
except Exception:
    HAVE_STAGE3 = False


# =========================================================
# Demo params / baseline design / power map
# =========================================================

def make_demo_params() -> ModelParams:
    geom = FluidGeom(Wc=50e-6, Lc=50e-6, Hc=50e-6)
    fluid = FluidProps(mu=1e-3, k=0.6, rho_cp=4.2e6)
    solid = SolidProps(k=130.0)
    conv = ConvectionProps(h=2e4)
    bc = BoundaryTemps(T_in=25.0, T_amb=25.0)
    return ModelParams(geom=geom, fluid=fluid, solid=solid, conv=conv, bc=bc)


# def make_demo_net(n: int = 9) -> CoolingNetwork:
#     """
#     Symmetric initial design:
#     regular grid of solid pillars (TSV), liquid between them.
#     Inlet = whole left boundary, outlet = whole right boundary.
#     """
#     if n < 8:
#         raise ValueError("Use n >= 8 for a meaningful demo design.")
#
#     C = np.full((n, n), CellType.LIQUID, dtype=int)
#
#     # symmetric outer boundary ports
#     C[:, 0] = CellType.INLET
#     C[:, -1] = CellType.OUTLET
#
#     # regular symmetric pillar lattice inside the domain
#     for i in range(1, n - 1, 2):
#         for j in range(2, n - 2, 2):
#             C[i, j] = CellType.TSV
#
#     net = CoolingNetwork(C=C)
#
#     if not net.has_inlet_to_outlet_path():
#         raise RuntimeError("Initial regular pillar grid is not connected from inlet to outlet.")
#
#     return net

# def make_demo_net(n: int = 9) -> CoolingNetwork:
#     """
#     Symmetric initial design for Stage 2 demo:
#     - regular grid of solid pillars (TSV),
#     - liquid between them,
#     - plus regular solid ribs so that Stage 2 has room to improve the design.
#
#     Inlet = whole left boundary
#     Outlet = whole right boundary
#     """
#     if n < 8:
#         raise ValueError("Use n >= 8 for a meaningful demo design.")
#
#     # Start from mostly liquid
#     C = np.full((n, n), CellType.LIQUID, dtype=int)
#
#     # Symmetric boundary ports
#     C[:, 0] = CellType.INLET
#     C[:, -1] = CellType.OUTLET
#
#     # -----------------------------------------------------
#     # 1) Regular solid pillars (TSV) inside the chip
#     # -----------------------------------------------------
#     for i in range(1, n - 1, 2):
#         for j in range(2, n - 2, 2):
#             C[i, j] = CellType.TSV
#
#     # -----------------------------------------------------
#     # 2) Add weak regular solid ribs (SILICON),
#     #    but DO NOT block the full left->right connectivity
#     # -----------------------------------------------------
#     # Vertical silicon ribs every 4 columns, but with gaps
#     for j in range(3, n - 2, 4):
#         for i in range(1, n - 1):
#             # leave periodic gaps so flow is still connected
#             if i % 3 == 1:
#                 continue
#             if C[i, j] == CellType.LIQUID:
#                 C[i, j] = CellType.SILICON
#
#     # Horizontal silicon ribs every 4 rows, but with gaps
#     for i in range(3, n - 2, 4):
#         for j in range(1, n - 1):
#             # leave periodic gaps so flow is still connected
#             if j % 3 == 1:
#                 continue
#             if C[i, j] == CellType.LIQUID:
#                 C[i, j] = CellType.SILICON
#
#     # -----------------------------------------------------
#     # 3) Re-open a central cross of liquid channels
#     #    to guarantee a strong connected backbone
#     # -----------------------------------------------------
#     mid = n // 2
#
#     # central horizontal backbone
#     for j in range(1, n - 1):
#         if C[mid, j] != CellType.TSV:
#             C[mid, j] = CellType.LIQUID
#
#     # central vertical helper backbone
#     for i in range(1, n - 1):
#         if C[i, mid] != CellType.TSV:
#             C[i, mid] = CellType.LIQUID
#
#     # connect near boundaries
#     for i in range(n):
#         if C[i, 1] != CellType.TSV:
#             C[i, 1] = CellType.LIQUID
#         if C[i, n - 2] != CellType.TSV:
#             C[i, n - 2] = CellType.LIQUID
#
#     net = CoolingNetwork(C=C)
#
#     # clean useless irregular fragments if any appeared
#     net.prune_irregular()
#
#     if not net.has_inlet_to_outlet_path():
#         raise RuntimeError("Initial regular grid is not connected from inlet to outlet.")
#
#     return net

# def make_demo_net(n: int = 9, io_mode: str = "top-bottom") -> CoolingNetwork:
#     """
#     Regular initial design with configurable inlet/outlet placement.
#
#     io_mode:
#       - "left-right"  : inlet on left, outlet on right
#       - "top-bottom"  : inlet on top, outlet on bottom
#       - "top-right"   : inlet on top, outlet on right
#       - "split"       : symmetric segmented ports
#     """
#     if n < 8:
#         raise ValueError("Use n >= 8 for a meaningful demo design.")
#
#     C = np.full((n, n), CellType.LIQUID, dtype=int)
#
#     # -----------------------------------------------------
#     # 1) Regular solid pillars (TSV) inside the chip
#     # -----------------------------------------------------
#     for i in range(1, n - 1, 2):
#         for j in range(2, n - 2, 2):
#             C[i, j] = CellType.TSV
#
#     # -----------------------------------------------------
#     # 2) Regular silicon ribs, but with gaps
#     # -----------------------------------------------------
#     for j in range(3, n - 2, 4):
#         for i in range(1, n - 1):
#             if i % 3 == 1:
#                 continue
#             if C[i, j] == CellType.LIQUID:
#                 C[i, j] = CellType.SILICON
#
#     for i in range(3, n - 2, 4):
#         for j in range(1, n - 1):
#             if j % 3 == 1:
#                 continue
#             if C[i, j] == CellType.LIQUID:
#                 C[i, j] = CellType.SILICON
#
#     # -----------------------------------------------------
#     # 3) Re-open a central liquid backbone
#     # -----------------------------------------------------
#     mid = n // 2
#
#     for j in range(1, n - 1):
#         if C[mid, j] != CellType.TSV:
#             C[mid, j] = CellType.LIQUID
#
#     for i in range(1, n - 1):
#         if C[i, mid] != CellType.TSV:
#             C[i, mid] = CellType.LIQUID
#
#     # helper corridors near boundaries
#     for i in range(n):
#         if C[i, 1] != CellType.TSV:
#             C[i, 1] = CellType.LIQUID
#         if C[i, n - 2] != CellType.TSV:
#             C[i, n - 2] = CellType.LIQUID
#
#     for j in range(n):
#         if C[1, j] != CellType.TSV:
#             C[1, j] = CellType.LIQUID
#         if C[n - 2, j] != CellType.TSV:
#             C[n - 2, j] = CellType.LIQUID
#
#     # -----------------------------------------------------
#     # 4) Apply inlet / outlet layout
#     # -----------------------------------------------------
#     if io_mode == "left-right":
#         C[:, 0] = CellType.INLET
#         C[:, -1] = CellType.OUTLET
#
#     elif io_mode == "top-bottom":
#         C[0, :] = CellType.INLET
#         C[-1, :] = CellType.OUTLET
#
#     elif io_mode == "top-right":
#         C[0, :] = CellType.INLET
#         C[:, -1] = CellType.OUTLET
#
#     elif io_mode == "split":
#         # symmetric segmented ports
#         q1 = n // 3
#         q2 = n - q1
#
#         # top middle segment = inlet
#         C[0, q1:q2] = CellType.INLET
#
#         # bottom middle segment = outlet
#         C[-1, q1:q2] = CellType.OUTLET
#
#         # optional side assists
#         C[q1:q2, 0] = CellType.INLET
#         C[q1:q2, -1] = CellType.OUTLET
#
#     else:
#         raise ValueError(f"Unknown io_mode: {io_mode}")
#
#     net = CoolingNetwork(C=C)
#
#     net.prune_irregular()
#
#     if not net.has_inlet_to_outlet_path():
#         raise RuntimeError(f"Initial regular grid with io_mode='{io_mode}' is not connected.")
#
#     return net

# def make_demo_net(n: int = 9) -> CoolingNetwork:
#     """
#     Initial regular design exactly like in the shown 9x9 figure:
#     - left boundary = inlet
#     - right boundary = outlet
#     - regular TSV pillars inside
#     - predefined silicon islands (0) inside the channel field
#     """
#     if n != 9:
#         raise ValueError("This exact demo design is defined for n = 9.")
#
#     C = np.full((n, n), CellType.LIQUID, dtype=int)
#
#     # Boundary ports
#     C[:, 0] = CellType.INLET
#     C[:, -1] = CellType.OUTLET
#
#     # Regular TSV pillars
#     for i in (1, 3, 5, 7):
#         for j in (2, 4, 6):
#             C[i, j] = CellType.TSV
#
#     # Silicon cells (0) to match the shown pattern
#     silicon_cells = [
#         # upper block
#         (1, 3),
#         (2, 2), (2, 3), (2, 4),
#         (3, 3),
#         (3, 5),
#
#         # lower block
#         (5, 3),
#         (6, 2), (6, 3), (6, 4),
#         (7, 3),
#     ]
#
#     for i, j in silicon_cells:
#         if C[i, j] == CellType.LIQUID:
#             C[i, j] = CellType.SILICON
#
#     net = CoolingNetwork(C=C)
#
#     if not net.has_inlet_to_outlet_path():
#         raise RuntimeError("Initial regular grid is not connected from inlet to outlet.")
#
#     return net

def make_demo_net(n: int = 9) -> CoolingNetwork:
    """
    Regular initial design with diagonal-style I/O:
    - left boundary: only upper part is inlet, lower 5 cells are walls
    - right boundary: only lower part is outlet, upper 5 cells are walls
    - regular TSV pillars inside
    - regular liquid region inside
    """
    if n != 9:
        raise ValueError("This exact demo design is defined for n = 9.")

    C = np.full((n, n), CellType.LIQUID, dtype=int)

    # -----------------------------------------------------
    # 1) Regular TSV pillars inside
    # -----------------------------------------------------
    for i in (1, 3, 5, 7):
        for j in (2, 4, 6):
            C[i, j] = CellType.TSV

    # -----------------------------------------------------
    # 2) Boundary conditions:
    # left side: upper part inlet, lower part wall
    # right side: upper part wall, lower part outlet
    # -----------------------------------------------------
    # left boundary
    C[:, 0] = CellType.INLET
    for i in range(4, 9):   # lower 5 cells -> wall
        C[i, 0] = CellType.SILICON

    # right boundary
    C[:, -1] = CellType.OUTLET
    for i in range(0, 5):   # upper 5 cells -> wall
        C[i, -1] = CellType.SILICON

    net = CoolingNetwork(C=C)

    if not net.has_inlet_to_outlet_path():
        raise RuntimeError("Initial regular grid is not connected from inlet to outlet.")

    return net

def make_complex_power_map(
    n: int,
    dx: float,
    dy: float,
    base_w_per_cm2: float = 12.0,
    hotspots: Optional[Tuple[Tuple[int, int, float], ...]] = None,
    sigma_cells: Optional[float] = None,
    grad_strength: float = 0.35,
    noise: float = 0.04,
    seed: int = 42,
) -> np.ndarray:
    """
    Power per cell [W].
    SAME power map is reused for initial and optimized designs.
    Hotspots scale with n, so this works for 9x9, 10x10, 31x31, ...
    """
    rng = np.random.default_rng(seed)

    if hotspots is None:
        hotspots = (
            (max(1, n // 4), max(1, n // 4), 220.0),
            (min(n - 2, 3 * n // 4), min(n - 2, 3 * n // 4), 260.0),
            (max(1, n // 4), min(n - 2, 3 * n // 4), 180.0),
            (min(n - 2, 3 * n // 4), max(1, n // 3), 150.0),
        )

    if sigma_cells is None:
        sigma_cells = max(1.0, n / 10.0)

    # W/cm^2 -> W/m^2
    base = base_w_per_cm2 * 1e4
    P = np.full((n, n), base * dx * dy, dtype=np.float64)

    ii = np.arange(n)[:, None]
    jj = np.arange(n)[None, :]

    # mild global gradient
    diag = (ii + jj) / max(1, (2 * (n - 1)))
    P *= (1.0 + grad_strength * diag)

    # local hotspots
    for (ci, cj, peak_w_per_cm2) in hotspots:
        peak = peak_w_per_cm2 * 1e4
        g = np.exp(-((ii - ci) ** 2 + (jj - cj) ** 2) / (2.0 * sigma_cells**2))
        P += (peak * dx * dy) * g

    # small multiplicative noise
    if noise > 0:
        P *= (1.0 + noise * rng.standard_normal(size=P.shape))

    return np.maximum(P, 0.0)


# =========================================================
# Thermal wrappers
# =========================================================

def eval_network(
    net: CoolingNetwork,
    power_map: np.ndarray,
    params: ModelParams,
    P_pump: float,
) -> tuple[ThermalResult, float]:
    flow = solve_flow(net, P_pump=P_pump, params=params)
    therm = solve_thermal_one_source_one_channel(net, power_map, flow, params)
    return ThermalResult(T=therm.T_source, max_T=therm.max_T, grad_T=therm.grad_T), flow.E_pump


def make_stage12_thermal_fn(params: ModelParams, P_pump: float):
    def thermal_fn(net: CoolingNetwork, power_map: np.ndarray) -> ThermalResult:
        flow = solve_flow(net, P_pump=P_pump, params=params)
        therm = solve_thermal_one_source_one_channel(net, power_map, flow, params)
        return ThermalResult(T=therm.T_source, max_T=therm.max_T, grad_T=therm.grad_T)
    return thermal_fn


def scale_power_to_target(
    net: CoolingNetwork,
    power_map: np.ndarray,
    params: ModelParams,
    P_pump: float,
    target_max_T: float = 110.0,
) -> tuple[np.ndarray, float]:
    """
    Scale power map ONCE so that the chosen baseline design
    has max_T approximately equal to target_max_T.
    """
    therm, _ = eval_network(net, power_map, params, P_pump)
    cur = therm.max_T
    Tin = params.bc.T_in

    if cur <= Tin + 1e-9:
        return power_map * 10.0, 10.0

    scale = (target_max_T - Tin) / (cur - Tin)
    scale = max(0.2, min(scale, 100.0))
    return power_map * scale, scale


# =========================================================
# Plot helpers (no grid)
# =========================================================

def _clean_axes(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_celltype_grid(ax, C: np.ndarray, title: str, annotate: bool = True):
    A = C.astype(np.int32)
    im = ax.imshow(A, interpolation="nearest", cmap="tab20")
    ax.set_title(title)
    _clean_axes(ax)

    if annotate:
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                ax.text(j, i, str(int(A[i, j])), ha="center", va="center", fontsize=10, color="black")
    return im


def plot_scalar(ax, A: np.ndarray, title: str, fmt: str, cmap: str, cbar_label: str, annotate: bool = True):
    im = ax.imshow(A, interpolation="nearest", cmap=cmap)
    ax.set_title(title)
    _clean_axes(ax)

    if annotate:
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                ax.text(j, i, fmt.format(float(A[i, j])), ha="center", va="center", fontsize=8, color="black")

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)
    return im


# =========================================================
# Metrics / result
# =========================================================

def print_metrics(tag: str, net: CoolingNetwork, therm: ThermalResult, E_pump: float, P_pump: float):
    liquid = int(np.sum(net.C == CellType.LIQUID))
    silicon = int(np.sum(net.C == CellType.SILICON))
    pillars = int(np.sum(net.C == CellType.TSV))
    print(
        f"{tag}: "
        f"max_T={therm.max_T:.2f}C  "
        f"grad_T={therm.grad_T:.2f}C  "
        f"E_pump={E_pump:.4e}  "
        f"P_pump={P_pump:.2f}  "
        f"liquid={liquid}  silicon={silicon}  pillars={pillars}"
    )


@dataclass(frozen=True)
class PlotResult:
    net_raw: CoolingNetwork
    net_init: CoolingNetwork
    net_stage2: CoolingNetwork
    net_stage3: Optional[CoolingNetwork]

    power: np.ndarray

    therm_raw: ThermalResult
    therm_init: ThermalResult
    therm_stage2: ThermalResult
    therm_stage3: Optional[ThermalResult]

    E_raw: float
    E_init: float
    E_stage2: float
    E_stage3: Optional[float]

    P_demo: float
    P_stage3: Optional[float]

    power_scale: float
    stage2_accepted: bool
    baseline_label: str

def print_final_results_table(out: PlotResult) -> None:
    print("\n" + "=" * 74)
    print("FINAL RESULTS")
    print("=" * 74)
    print(f"{'Stage':<12} {'max_T':>10} {'grad_T':>10} {'E_pump':>16} {'P_pump':>12}")
    print("-" * 74)

    print(
        f"{'Initial':<12} "
        f"{out.therm_init.max_T:>10.2f} "
        f"{out.therm_init.grad_T:>10.2f} "
        f"{out.E_init:>16.4e} "
        f"{out.P_demo:>12.2f}"
    )

    print(
        f"{'Stage 2':<12} "
        f"{out.therm_stage2.max_T:>10.2f} "
        f"{out.therm_stage2.grad_T:>10.2f} "
        f"{out.E_stage2:>16.4e} "
        f"{out.P_demo:>12.2f}"
    )

    if out.therm_stage3 is not None and out.E_stage3 is not None and out.P_stage3 is not None:
        print(
            f"{'Stage 3':<12} "
            f"{out.therm_stage3.max_T:>10.2f} "
            f"{out.therm_stage3.grad_T:>10.2f} "
            f"{out.E_stage3:>16.4e} "
            f"{out.P_stage3:>12.2f}"
        )
    else:
        print(f"{'Stage 3':<12} {'-':>10} {'-':>10} {'-':>16} {'-':>12}")

    print("=" * 74)

    print("\nInterpretation:")
    print(" - Stage 2 -> thermal optimization (lower max_T is better)")
    print(" - Stage 3 -> pumping-energy optimization under thermal constraints")

# =========================================================
# Pipeline
# =========================================================

def run_pipeline_and_collect(
    n: int,
    seed: int,
    target_init_max_T: float = 110.0,
    run_stage3: bool = False,
    use_full_stage2: bool = True,
    use_best_io: bool = False,
) -> PlotResult:
    params = make_demo_params()
    P_demo = 1e4

    # 0) Raw symmetric regular pillar grid
    net_raw = make_demo_net(n)
    power = make_complex_power_map(
        n=n,
        dx=params.geom.Wc,
        dy=params.geom.Lc,
        seed=seed,
    )

    thermal_fn = make_stage12_thermal_fn(params, P_demo)

    # 1) Choose baseline
    if use_best_io:
        net_init_unscaled, best_pat, _ = select_best_io(net_raw, power, thermal_fn)
        io_q = best_pat.q
        baseline_label = "Initial design (regular grid + best IO)"
    else:
        net_init_unscaled = net_raw.clone()
        io_q = (0, 0, 0, 0)
        baseline_label = "Initial design (regular symmetric grid)"

    # 2) Scale the SAME power map once on the chosen baseline
    power, scale = scale_power_to_target(
        net=net_init_unscaled,
        power_map=power,
        params=params,
        P_pump=P_demo,
        target_max_T=target_init_max_T,
    )

    # 3) Rebuild final baseline with the SAME fixed power map
    if use_best_io:
        net_init, best_pat, _ = select_best_io(net_raw, power, thermal_fn)
        io_q = best_pat.q
    else:
        net_init = net_raw.clone()

    # 4) Baseline metrics
    therm_raw, E_raw = eval_network(net_raw, power, params, P_demo)
    therm_init, E_init = eval_network(net_init, power, params, P_demo)

    # 5) Stage 2 on top of the SAME power map
    if use_full_stage2 and HAVE_STAGE2_FULL:
        s2 = stage2_full_patterning(
            net=net_init,
            power_map=power,
            thermal_fn=thermal_fn,
            alpha_1=cfg.ALPHA_1,
            alpha_2=cfg.ALPHA_2,
            io_q=io_q,
        )
        net2_candidate = s2.net
    else:
        s2 = stage2_straight_channels(
            net=net_init,
            power_map=power,
            thermal_fn=thermal_fn,
            alpha_1=cfg.ALPHA_1,
            io_q=io_q,
            pad=2,
            stripe_candidates=(1, 2, 3),
            thickness_candidates=(1, 2),
        )
        net2_candidate = s2.net
        print(f"Stage2 accepted hotspot areas: {s2.accepted_areas}")

    therm_stage2_candidate, E_stage2_candidate = eval_network(net2_candidate, power, params, P_demo)

    # Safety: reject Stage 2 if it globally worsens max_T
    if therm_stage2_candidate.max_T <= therm_init.max_T + 1e-6:
        net_stage2 = net2_candidate
        therm_stage2 = therm_stage2_candidate
        E_stage2 = E_stage2_candidate
        stage2_accepted = True
    else:
        print("⚠️ Stage 2 rejected globally: it increased max_T.")
        net_stage2 = net_init
        therm_stage2 = therm_init
        E_stage2 = E_init
        stage2_accepted = False

    # 6) Optional Stage 3
    net_stage3 = None
    therm_stage3 = None
    E_stage3 = None
    P_stage3 = None

    if run_stage3 and HAVE_STAGE3:
        print("Running Stage 3 pruning...")

        # Для демонстрації Stage 3 має:
        # 1) стартувати гарантовано,
        # 2) не мати права занадто зіпсувати температуру відносно Stage 2.
        #
        # Тому беремо constraints не "сліпо з config",
        # а робимо їх не жорсткішими за поточний Stage 2,
        # з маленьким допустимим запасом.

        stage3_t_slack = 2.0  # дозволяємо Stage 3 підняти max_T не більше ніж на ~2°C
        stage3_g_slack = 1.0  # дозволяємо grad_T підняти не більше ніж на ~1°C

        Tmax_stage3 = max(cfg.T_MAX, therm_stage2.max_T + stage3_t_slack)
        GradTmax_stage3 = max(cfg.GRAD_T_MAX, therm_stage2.grad_T + stage3_g_slack)

        print(
            f"Stage 3 constraints: "
            f"Tmax <= {Tmax_stage3:.2f}, "
            f"GradTmax <= {GradTmax_stage3:.2f}"
        )

        s3 = run_stage3_pruning(
            net=net_stage2.clone(),
            power_map=power,
            params=params,
            Tmax=Tmax_stage3,
            GradTmax=GradTmax_stage3,
            max_iters=2000,
        )

        net_stage3 = s3.net
        therm_stage3 = s3.final_T
        E_stage3 = s3.final_E
        P_stage3 = s3.final_P

        print("Stage 3 finished.")
        print(
            f"Stage 3 final: "
            f"max_T={therm_stage3.max_T:.2f}, "
            f"grad_T={therm_stage3.grad_T:.2f}, "
            f"E_pump={E_stage3:.4e}, "
            f"P_pump={P_stage3:.2f}"
        )

    return PlotResult(
        net_raw=net_raw,
        net_init=net_init,
        net_stage2=net_stage2,
        net_stage3=net_stage3,
        power=power,
        therm_raw=therm_raw,
        therm_init=therm_init,
        therm_stage2=therm_stage2,
        therm_stage3=therm_stage3,
        E_raw=E_raw,
        E_init=E_init,
        E_stage2=E_stage2,
        E_stage3=E_stage3,
        P_demo=P_demo,
        P_stage3=P_stage3,
        power_scale=scale,
        stage2_accepted=stage2_accepted,
        baseline_label=baseline_label,
    )


# =========================================================
# Main
# =========================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=9)
    ap.add_argument("--seed", type=int, default=getattr(cfg, "RNG_SEED", 42))
    ap.add_argument("--out", type=str, default="out")
    ap.add_argument("--target", type=float, default=110.0, help="desired baseline max temperature")
    ap.add_argument("--stage3", action="store_true", help="also run Stage 3 (energy trade-off)")
    ap.add_argument("--full-stage2", action="store_true", help="use full Stage 2 if available")
    ap.add_argument("--best-io", action="store_true", help="use Stage 1 IO optimization")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    out = run_pipeline_and_collect(
        n=args.n,
        seed=args.seed,
        target_init_max_T=args.target,
        run_stage3=args.stage3,
        use_full_stage2=args.full_stage2,
        use_best_io=args.best_io,
    )

    annotate = args.n <= 15

    # Power map
    fig, ax = plt.subplots(figsize=(6, 6))
    plot_scalar(ax, out.power, "Power map (same for all designs)", "{:.3f}", "magma", "Power [W per cell]", annotate=annotate)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "power_map.png"), dpi=200)
    plt.close(fig)

    # Raw initial regular pillar grid
    fig, ax = plt.subplots(figsize=(6, 6))
    plot_celltype_grid(ax, out.net_raw.C, "Raw regular pillar grid", annotate=annotate)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "C_raw.png"), dpi=200)
    plt.close(fig)

    # Baseline
    fig, ax = plt.subplots(figsize=(7, 8))
    plot_celltype_grid(ax, out.net_init.C, out.baseline_label, annotate=annotate)

    legend_text = "Позначення: -1 — TSV/pillar, 0 — silicon, 1 — liquid, 2 — inlet, 3 — outlet"

    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.text(
        0.02, 0.015,
        legend_text,
        fontsize=11,
        ha="left",
        va="bottom",
        bbox=dict(facecolor="white", edgecolor="gray", alpha=0.9, boxstyle="round,pad=0.3")
    )

    fig.savefig(os.path.join(args.out, "C_init.png"), dpi=200)
    plt.close(fig)

    # Baseline temperature
    fig, ax = plt.subplots(figsize=(6, 6))
    plot_scalar(
        ax,
        out.therm_init.T,
        f"T init [°C], max={out.therm_init.max_T:.2f}",
        "{:.2f}",
        "viridis",
        "Temperature",
        annotate=annotate,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "T_init.png"), dpi=200)
    plt.close(fig)

    # Stage 2 design
    fig, ax = plt.subplots(figsize=(6, 6))
    plot_celltype_grid(ax, out.net_stage2.C, "Optimized design (after Stage 2)", annotate=annotate)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "C_stage2.png"), dpi=200)
    plt.close(fig)

    # Stage 2 temperature
    fig, ax = plt.subplots(figsize=(6, 6))
    plot_scalar(
        ax,
        out.therm_stage2.T,
        f"T after Stage 2 [°C], max={out.therm_stage2.max_T:.2f}",
        "{:.2f}",
        "viridis",
        "Temperature",
        annotate=annotate,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "T_stage2.png"), dpi=200)
    plt.close(fig)

    # Main comparison figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    plot_scalar(
        axes[0, 0],
        out.power,
        "Power map (fixed)",
        "{:.3f}",
        "magma",
        "Power [W/cell]",
        annotate=annotate,
    )
    plot_celltype_grid(axes[0, 1], out.net_init.C, out.baseline_label, annotate=annotate)
    plot_scalar(
        axes[0, 2],
        out.therm_init.T,
        f"Initial temperature\nmax={out.therm_init.max_T:.2f}°C",
        "{:.2f}",
        "viridis",
        "Temperature",
        annotate=annotate,
    )

    plot_scalar(
        axes[1, 0],
        out.power,
        "Same power map",
        "{:.3f}",
        "magma",
        "Power [W/cell]",
        annotate=annotate,
    )
    plot_celltype_grid(axes[1, 1], out.net_stage2.C, "Optimized design\n(after Stage 2)", annotate=annotate)
    plot_scalar(
        axes[1, 2],
        out.therm_stage2.T,
        f"Optimized temperature\nmax={out.therm_stage2.max_T:.2f}°C",
        "{:.2f}",
        "viridis",
        "Temperature",
        annotate=annotate,
    )

    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "comparison_init_vs_stage2.png"), dpi=200)
    plt.close(fig)

    # Optional Stage 3
    if out.net_stage3 is not None and out.therm_stage3 is not None:
        # 1) Final network matrix after Stage 3
        fig, ax = plt.subplots(figsize=(7, 8))
        plot_celltype_grid(ax, out.net_stage3.C, "Final network after Stage 3", annotate=annotate)

        legend_text = "Позначення: -1 — TSV/pillar, 0 — silicon, 1 — liquid, 2 — inlet, 3 — outlet"

        fig.tight_layout(rect=[0, 0.08, 1, 1])
        fig.text(
            0.02, 0.015,
            legend_text,
            fontsize=11,
            ha="left",
            va="bottom",
            bbox=dict(facecolor="white", edgecolor="gray", alpha=0.9, boxstyle="round,pad=0.3")
        )
        fig.savefig(os.path.join(args.out, "C_stage3.png"), dpi=200)
        plt.close(fig)

        # 2) Temperature matrix after Stage 3
        fig, ax = plt.subplots(figsize=(6, 6))
        plot_scalar(
            ax,
            out.therm_stage3.T,
            f"T after Stage 3 [°C], max={out.therm_stage3.max_T:.2f}",
            "{:.2f}",
            "viridis",
            "Temperature",
            annotate=annotate,
        )
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "T_stage3.png"), dpi=200)
        plt.close(fig)

        # 3) Combined side-by-side figure
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        plot_celltype_grid(axes[0], out.net_stage3.C, "Stage 3 design (final network)", annotate=annotate)
        plot_scalar(
            axes[1],
            out.therm_stage3.T,
            f"T after Stage 3\nmax={out.therm_stage3.max_T:.2f}°C",
            "{:.2f}",
            "viridis",
            "Temperature",
            annotate=annotate,
        )
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "stage3_tradeoff.png"), dpi=200)
        plt.close(fig)

    print_final_results_table(out)
    print("\n✅ Saved plots to:", os.path.abspath(args.out))
    print(" - power_map.png")
    print(" - C_raw.png")
    print(" - C_init.png")
    print(" - T_init.png")
    print(" - C_stage2.png")
    print(" - T_stage2.png")
    print(" - comparison_init_vs_stage2.png")
    if out.net_stage3 is not None:
        print(" - C_stage3.png")
        print(" - T_stage3.png")
        print(" - stage3_tradeoff.png")


if __name__ == "__main__":
    main()