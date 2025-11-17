from dataclasses import dataclass
from typing import Dict
import csv
from importlib.resources import files

import skewerjax as sjx
from skewerjax.constants.physical import phys

# ----------------- dataclass -----------------

@dataclass(frozen=True)
class Line:
    id: str
    species: str
    ion: int
    name: str
    lambda0_A: float
    f_osc: float
    A21_Hz: float
    mass_amu: float
    
    @property
    def A21_THz(self) -> float:
        return self.A21_Hz * 1e-12

# ----------------- loading -------------------

_ROMAN = {
    "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5,
    "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10,
}


def _load_lines() -> Dict[str, Line]:
    path = files(__package__) / "lines_table.dat"
    lines: Dict[str, Line] = {}

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            ion = _ROMAN[r["ion"]]
            mass_amu = float(r["mass_amu"] or "nan")
            line = Line(
                id=r["id"],
                species=r["species"],
                ion=ion,
                name=r["name"],
                lambda0_A=float(r["lambda0_A"]),
                f_osc=float(r["f_osc"]),
                A21_Hz=float(r["A21_Hz"]),
                mass_amu=mass_amu,
            )
            lines[line.id] = line

    return lines


LINES: Dict[str, Line] = _load_lines()


def get_line(line_id: str) -> Line:
    try:
        return LINES[line_id]
    except KeyError:
        raise KeyError(f"Unknown line: {line_id}") from None
