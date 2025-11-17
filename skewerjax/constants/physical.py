from __future__ import annotations
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class PhysicalConstants:
    # SI units unless stated otherwise
    c: float = 299792458                     # m / s
    c_A_THz: float = 2997924.58              # A · THz
    q_e: float = 1.602176634e-19             # C
    m_e: float = 9.10938356e-31              # kg
    m_p: float = 1.6726231e-27               # kg
    kB: float = 1.380649e-23                 # J / K
    kB_amu: float = 8314.46262102654         # amu m^2 / s^2 / K
    epsilon_0: float = 8.8541878188e-12      # C^2·kg^−1·m^−3·s^2

    # Handy unit conversions (SI base)
    ANGSTROM_2_m: float = 1e-10                  # m
    AMU_2_kg: float = 1.66053906660e-27           # kg

# single shared instance (immutable)
phys = PhysicalConstants()
