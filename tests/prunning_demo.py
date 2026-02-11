import numpy as np

from src.cooling_network.grid import CoolingNetwork
from src.cooling_network.power_map import make_power_map
from src.cooling_network.thermal_mock import simulate_temperature
from src.cooling_network.types import CellType
from src.cooling_network.prunner import run_pruning


def make_initial_net(n: int = 21) -> CoolingNetwork:
    C = np.full((n, n), CellType.LIQUID, dtype=int)

    C[:, 0] = CellType.INLET
    C[:, -1] = CellType.OUTLET

    return CoolingNetwork(C=C)


if __name__ == "__main__":
    n = 31
    net = make_initial_net(n)

    power = make_power_map(n, hotspots=[(n//2, n//2), (n//2 + 3, n//2 - 4)], base=1.0, hot=80.0)

    init = simulate_temperature(net, power)
    print("INIT:", init.max_T, init.grad_T, "liquid=", int(np.sum(net.C == CellType.LIQUID)))

    res = run_pruning(net, power, thermal_fn=simulate_temperature, max_iters=3000)

    final = simulate_temperature(res.final_net, power)
    print("FINAL:", final.max_T, final.grad_T, "liquid=", int(np.sum(res.final_net.C == CellType.LIQUID)))

    accepted = sum(1 for s in res.history if s.accepted)
    print("steps:", len(res.history), "accepted:", accepted)
