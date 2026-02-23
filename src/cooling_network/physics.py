from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FluidGeom:
    # cell dimensions (paper: Δx=Wc, Δy=Lc, Δz=Hc)
    Wc: float  # x width  [m]
    Lc: float  # y length [m]
    Hc: float  # z height [m]


@dataclass(frozen=True, slots=True)
class FluidProps:
    mu: float      # dynamic viscosity [Pa*s]
    k: float       # thermal conductivity [W/(m*K)] (assume isotropic for now)
    rho_cp: float  # volumetric heat capacity [J/(m^3*K)]


@dataclass(frozen=True, slots=True)
class SolidProps:
    k: float  # thermal conductivity [W/(m*K)]


@dataclass(frozen=True, slots=True)
class ConvectionProps:
    h: float  # convection coefficient [W/(m^2*K)]


@dataclass(frozen=True, slots=True)
class BoundaryTemps:
    T_in: float   # coolant inlet temperature [°C or K consistent]
    T_amb: float  # ambient/reference temperature


@dataclass(frozen=True, slots=True)
class ModelParams:
    geom: FluidGeom
    fluid: FluidProps
    solid: SolidProps
    conv: ConvectionProps
    bc: BoundaryTemps