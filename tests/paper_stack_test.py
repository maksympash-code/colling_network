import numpy as np

import src.cooling_network.config as cfg
from src.cooling_network.grid import CoolingNetwork
from src.cooling_network.types import CellType
from src.cooling_network.physics import FluidGeom, FluidProps, SolidProps, ConvectionProps, BoundaryTemps, ModelParams
from src.cooling_network.flow_solver import solve_flow
from src.cooling_network.thermal_solver import solve_thermal_one_source_one_channel
from src.cooling_network.pressure_opt import find_min_p_pump
from src.cooling_network.power_map import make_power_map_from_areal_density


def make_simple_net(n: int = 31) -> CoolingNetwork:
    C = np.full((n, n), CellType.LIQUID, dtype=int)
    C[:, 0] = CellType.INLET
    C[:, -1] = CellType.OUTLET
    return CoolingNetwork(C=C)


def make_test_params() -> ModelParams:
    geom = FluidGeom(Wc=50e-6, Lc=50e-6, Hc=50e-6)
    fluid = FluidProps(mu=1e-3, k=0.6, rho_cp=4.2e6)
    solid = SolidProps(k=130.0)
    conv = ConvectionProps(h=2e4)
    bc = BoundaryTemps(T_in=25.0, T_amb=25.0)
    return ModelParams(geom=geom, fluid=fluid, solid=solid, conv=conv, bc=bc)


def main():
    n = 31
    net = make_simple_net(n)
    params = make_test_params()

    power = make_power_map_from_areal_density(
        n, dx=params.geom.Wc, dy=params.geom.Lc,
        base_w_per_cm2=10.0, hot_w_per_cm2=200.0
    )

    P1 = 1e4
    P2 = 2e4
    flow1 = solve_flow(net, P_pump=P1, params=params)
    flow2 = solve_flow(net, P_pump=P2, params=params)

    ratio = flow2.E_pump / flow1.E_pump
    print("Flow energy ratio (expect ~4):", ratio)
    assert 3.5 < ratio < 4.5

    therm1 = solve_thermal_one_source_one_channel(net, power, flow1, params)
    therm2 = solve_thermal_one_source_one_channel(net, power, flow2, params)

    print("Thermal max_T at P1:", therm1.max_T, "grad:", therm1.grad_T)
    print("Thermal max_T at P2:", therm2.max_T, "grad:", therm2.grad_T)

    inlet_coords = list(zip(*np.where(net.C == CellType.INLET)))
    if inlet_coords:
        i0, j0 = inlet_coords[0]
        assert abs(therm1.T_channel[i0, j0] - params.bc.T_in) < 1e-6

    # allow tiny numerical noise
    assert therm2.max_T <= therm1.max_T + 1e-3

    opt = find_min_p_pump(net, power, params, Tmax=cfg.T_MAX, GradTmax=cfg.GRAD_T_MAX, P_low=1e3, P_high=1e6, iters=25)
    print("Opt P_pump:", opt.P_pump, "E_pump:", opt.E_pump, "max_T:", opt.max_T, "grad:", opt.grad_T)

    assert opt.max_T <= cfg.T_MAX + 1e-3
    assert opt.grad_T <= cfg.GRAD_T_MAX + 1e-3

    print("✅ paper_stack_test passed")


if __name__ == "__main__":
    main()