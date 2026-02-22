import numpy as np

from src.cooling_network.grid import CoolingNetwork
from src.cooling_network.types import CellType
from src.cooling_network.power_map import make_power_map
from src.cooling_network.thermal_mock import simulate_temperature
from src.cooling_network.io_optimization import select_best_io


def make_pinfin_net(n: int) -> CoolingNetwork:
    C = np.full((n, n), CellType.LIQUID, dtype=int)
    return CoolingNetwork(C=C)


n = 31
base = make_pinfin_net(n)
power = make_power_map(n, hotspots=[(n//2, n//2)], base=1.0, hot=80.0)

best_net, best_pat, best_res = select_best_io(base, power, simulate_temperature)

print("Best pattern:", best_pat)
print("Best max_T:", best_res.max_T, "grad_T:", best_res.grad_T)