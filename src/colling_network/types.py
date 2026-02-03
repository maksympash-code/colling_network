from __future__ import annotations

from enum import IntEnum
from typing import Optional

import numpy as np


class CellType(IntEnum):
    """
        Tags from the paper / ICCAD contest template:
    -1 = TSV (fixed)
    0 = silicon (solid)
    1 = liquid (channel)
    2 = inlet
    3 = outlet
    """
    TSV = -1
    SILICON = 0
    LIQUID = 1
    INLET = 2
    OUTLET = 3

Grid = np.ndarray
Index2D = Tuple[int, int]

@dataclass(frozen=True, slots=True)
class NetworkState:
    """
    Cooling network in a single channel layer (or one per layer if you later extend).
    grid stores CellType tags (TSV/SILICON/LIQUID/INLET/OUTLET).
    """
    grid: Grid
    p_pump: float


@dataclasses(frozen=True, slots=True)
class ThermalResult:
    """
    Output of thermal simulation.
    T: temperature field for a source layer (or aggregated field in simplified model).
    max_T: peak temperature
    grad_T: contest-defined 'gradient' = max(T) - min(T)
    """
    T: np.ndarray
    max_T: float
    grad_T: float



@dataclass(frozen=True, slots=True)
class PruneStepResult:
    """
    Output of one pruning attempt.
    removed_cell: which cell was tentatively removed (LIQUID -> SILICON), if any
    accepted: whether we actually keep this removal
    thermal: thermal result after this step (or after tentative removal)
    """
    removed_cell: Optional[Index2D]
    accepted: bool
    thermal: ThermalResult