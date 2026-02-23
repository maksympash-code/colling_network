from __future__ import annotations

from dataclasses import dataclass

from src.cooling_network.grid import CoolingNetwork
from src.cooling_network.flow_solver import solve_flow
from src.cooling_network.thermal_solver import solve_thermal_one_source_one_channel
from src.cooling_network.physics import ModelParams


@dataclass(frozen=True, slots=True)
class PressureOptResult:
    P_pump: float
    E_pump: float
    max_T: float
    grad_T: float


def find_min_p_pump(
    net: CoolingNetwork,
    power_map,
    params: ModelParams,
    Tmax: float,
    GradTmax: float,
    P_low: float = 1e3,
    P_high: float = 1e6,
    iters: int = 25,
) -> PressureOptResult:
    lo = float(P_low)
    hi = float(P_high)
    best = None

    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        flow = solve_flow(net, P_pump=mid, params=params)
        therm = solve_thermal_one_source_one_channel(net, power_map, flow, params)

        ok = (therm.max_T <= Tmax) and (therm.grad_T <= GradTmax)
        if ok:
            best = PressureOptResult(mid, flow.E_pump, therm.max_T, therm.grad_T)
            hi = mid
        else:
            lo = mid

    if best is None:
        flow = solve_flow(net, P_pump=hi, params=params)
        therm = solve_thermal_one_source_one_channel(net, power_map, flow, params)
        return PressureOptResult(hi, flow.E_pump, therm.max_T, therm.grad_T)

    return best