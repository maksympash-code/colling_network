import numpy as np

import src.cooling_network.config as cfg
from src.cooling_network.types import CellType, ThermalResult
from src.cooling_network.grid import CoolingNetwork
from src.cooling_network.physics import FluidGeom, FluidProps, SolidProps, ConvectionProps, BoundaryTemps, ModelParams
from src.cooling_network.power_map import make_power_map_from_areal_density
from src.cooling_network.flow_solver import solve_flow
from src.cooling_network.thermal_solver import solve_thermal_one_source_one_channel
from src.cooling_network.io_optimization import select_best_io
from src.cooling_network.stage2_straight_channels import stage2_straight_channels
from src.cooling_network.stage3_pruner_paper import run_stage3_pruning


def make_pinfin_net(n: int) -> CoolingNetwork:
    C = np.full((n, n), CellType.LIQUID, dtype=int)
    return CoolingNetwork(C=C)


def make_params() -> ModelParams:
    geom = FluidGeom(Wc=50e-6, Lc=50e-6, Hc=50e-6)
    fluid = FluidProps(mu=1e-3, k=0.6, rho_cp=4.2e6)
    solid = SolidProps(k=130.0)
    conv = ConvectionProps(h=2e4)
    bc = BoundaryTemps(T_in=25.0, T_amb=25.0)
    return ModelParams(geom=geom, fluid=fluid, solid=solid, conv=conv, bc=bc)


def thermal_fn(net: CoolingNetwork, power_map: np.ndarray, P_pump: float, params: ModelParams) -> ThermalResult:
    flow = solve_flow(net, P_pump=P_pump, params=params)
    therm = solve_thermal_one_source_one_channel(net, power_map, flow, params)
    return ThermalResult(T=therm.T_source, max_T=therm.max_T, grad_T=therm.grad_T)


def thermal_fn_stage1(net: CoolingNetwork, power_map: np.ndarray) -> ThermalResult:
    params = make_params()
    # fixed pressure just for comparison between patterns
    P_pump = 2e4
    return thermal_fn(net, power_map, P_pump, params)


def main():
    n = 31
    params = make_params()

    base = make_pinfin_net(n)
    power = make_power_map_from_areal_density(n, params.geom.Wc, params.geom.Lc, base_w_per_cm2=10.0, hot_w_per_cm2=200.0)

    # Stage 1
    best_net, best_pat, best_res = select_best_io(base, power, thermal_fn_stage1)
    print("Stage1 best pattern:", best_pat.q, "maxT:", best_res.max_T, "grad:", best_res.grad_T)

    # Stage 2 (straight)
    s2 = stage2_straight_channels(best_net, power, thermal_fn_stage1, alpha_1=cfg.ALPHA_1, io_q=best_pat.q)
    print("Stage2:", s2.thermal_before.max_T, "->", s2.thermal_after.max_T)

    # Stage 3 (paper-like)
    res3 = run_stage3_pruning(s2.net, power, params, Tmax=cfg.T_MAX, GradTmax=cfg.GRAD_T_MAX, max_iters=2000)
    print("Stage3 final:", res3.final_T.max_T, res3.final_T.grad_T, "E:", res3.final_E, "P:", res3.final_P)
    print("Steps:", len(res3.history), "Accepted:", sum(1 for x in res3.history if x.accepted))


if __name__ == "__main__":
    main()