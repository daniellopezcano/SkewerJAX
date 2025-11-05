"""
Spectral line constants used by SkewerJAX.
Extend this file as new transitions or ions are added.
All wavelengths are in Angstroms unless noted otherwise.
"""

from dataclasses import dataclass
from typing import Dict
from .physical import phys

@dataclass(frozen=True)
class Line:
    """Basic information about an absorption/emission line."""
    id: str
    species: str
    ion: int
    name: str
    lambda0_A: float      # rest wavelength [Å]
    f_osc: float          # oscillator strength
    A_s1: float           # Einstein A coefficient [s^-1]
    mass: float       # atomic mass [amu] (for Doppler widths)

    @property
    def lambda0_m(self) -> float:
        """Rest wavelength in meters."""
        return self.lambda0_A * phys.ANGSTROM



# --- Define the lines explicitly here ---
LINES: Dict[str, Line] = {
    "HI-LyA": Line(
        id="HI-LyA",
        species="H",
        ion=1,
        name="Lyman-α",
        lambda0_A=1215.67,
        f_osc=0.416,
        A_s1=6.265e8,
        mass=1.00784,
    ),
    "HI-LyB": Line(
        id="HI-LyB",
        species="H",
        ion=1,
        name="Lyman-β",
        lambda0_A=1025.72,
        f_osc=0.079,
        A_s1=1.64e8,
        mass=1.00784,
    ),
}

# convenient aliases
ALIASES = {
    "LyA": "HI-LyA",
    "LyB": "HI-LyB",
}

def get_line(name: str) -> Line:
    """Return a Line by canonical ID or alias (e.g., 'LyA')."""
    key = ALIASES.get(name, name)
    if key not in LINES:
        raise KeyError(f"Unknown line: {name}")
    return LINES[key]
