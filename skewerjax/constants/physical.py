from __future__ import annotations
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class PhysicalConstants:
    c_A_THz: float = 2997924.58              # Speed of light in Angstrom · THz
    kB_amu: float = 8314.46262102654         # Boltzman constant in amu m^2 / s^2 / K
    
# single shared instance (immutable)
phys = PhysicalConstants()
