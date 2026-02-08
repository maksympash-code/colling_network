import numpy as np
from src.cooling_network.types import CellType
from src.cooling_network.grid import CoolingNetwork

C = np.full((5, 5), CellType.SILICON, dtype=int)

# Робимо коридор: INLET -> LIQUID -> LIQUID -> OUTLET
C[2, 0] = CellType.INLET
C[2, 1] = CellType.LIQUID
C[2, 2] = CellType.LIQUID
C[2, 3] = CellType.OUTLET

net = CoolingNetwork(C=C)

print(net.has_inlet_to_outlet_path())  # True

net.remove_cell(2, 1)

print(net.has_inlet_to_outlet_path())  # False
