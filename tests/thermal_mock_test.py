import numpy as np

from src.cooling_network.grid import CoolingNetwork
from src.cooling_network.power_map import make_power_map
from src.cooling_network.thermal_mock import simulate_temperature
from src.cooling_network.types import CellType

C = np.full((5, 5), CellType.SILICON, dtype=int)

C[2, 0] = CellType.INLET
C[2, 1] = CellType.LIQUID
C[2, 2] = CellType.LIQUID
C[2, 3] = CellType.OUTLET

net = CoolingNetwork(C=C)
power_map = make_power_map(5)

thermal_res = simulate_temperature(net, power_map)

net2 = CoolingNetwork(C=np.full((5,5), CellType.SILICON, dtype=int))
thermal2 = simulate_temperature(net2, power_map)



print(thermal_res.max_T, thermal_res.grad_T)
print(thermal2.max_T, thermal2.grad_T)