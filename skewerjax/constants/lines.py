from dataclasses import dataclass
from typing import Dict
import csv
from importlib.resources import files
import skewerjax as sjx

# ---- dataclass ----
@dataclass(frozen=True)
class Line:
    id: str
    species: str
    ion: int               # keep int for API compatibility
    name: str
    lambda0_A: float
    f_osc: float
    A_s1: float
    mass: float

    @property
    def lambda0_m(self) -> float:
        return self.lambda0_A * sjx.constants.physical. ANGSTROM

# ---- minimal helpers ----
_ROMAN_MAP = {
    "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5,
    "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10,
}

def _roman_to_int(s: str) -> int:
    try:
        return _ROMAN_MAP[s]
    except KeyError:
        raise ValueError(f"Unknown Roman numeral for ion: {s!r}")

# ---- loader ----
def _load() -> Dict[str, Line]:
    """
    Load tab-separated 'lines_table.dat' packaged next to this module.
    Expected header (tab-separated):
    id  species ion name lambda_A f_osc A_s1 mass_amu
    """
    p = files(__package__) / "lines_table.dat"
    out: Dict[str, Line] = {}
    with p.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            # Convert types and map field names
            try:
                line = Line(
                    id=r["id"],
                    species=r["species"],
                    ion=_roman_to_int(r["ion"]),
                    name=r["name"],
                    lambda0_A=float(r["lambda_A"]),
                    f_osc=float(r["f_osc"]),
                    A_s1=float(r["A_s1"]),
                    mass=float(r["mass_amu"]) if r.get("mass_amu", "") != "" else float("nan"),
                )
            except Exception:
                # Keep minimal/robust: skip malformed rows silently
                continue
            out[line.id] = line
    return out

# Materialized registry
LINES: Dict[str, Line] = _load()

# Minimal aliases; extend later as needed
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
