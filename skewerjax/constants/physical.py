from __future__ import annotations
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class PhysicalConstants:
    # SI units unless stated otherwise
    c: float = 299_792_458.0                 # m / s
    q_e: float = 1.602176634e-19             # C
    m_e: float = 9.10938356e-31              # kg
    m_p: float = 1.6726231e-27               # kg
    k_B: float = 1.380649e-23                # J / K
    epsilon_0: float = 8.8541878188e-12      # C^2·kg^−1·m^−3·s^2

    # Handy unit conversions (SI base)
    ANGSTROM: float = 1e-10                  # m
    AMU: float = 1.66053906660e-27           # kg
    CM: float = 1e-2                         # m

# single shared instance (immutable)
phys = PhysicalConstants()
