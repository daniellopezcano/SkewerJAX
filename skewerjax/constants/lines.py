from dataclasses import dataclass
from typing import Dict
import re
from importlib.resources import files

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
    def A21_GHz(self) -> float:
        return self.A21_Hz * 1e-9

# ----------------- loading -------------------

_ROMAN = {
    "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5,
    "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10,
}

def _load_lines() -> Dict[str, Line]:
    path = files(__package__) / "lines_table.dat"
    lines: Dict[str, Line] = {}

    with path.open("r", encoding="utf-8") as f:
        # Skip header line
        next(f)
        
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('!'):
                continue
                
            parts = line.split()
            if len(parts) < 6:
                continue
            
            id_ = parts[0]
            species = parts[1]
            ion_roman = parts[2]
            lambda0_A = float(parts[-4])
            f_osc = float(parts[-3])
            A21_Hz = float(parts[-2])
            mass_amu = float(parts[-1]) if len(parts) > 5 else float('nan')
            
            # Handle ion Roman numerals with digits (e.g., "I2") and * suffix
            # Strip digits first, then * to get base Roman numeral
            ion_clean = re.sub(r'\d', '', ion_roman).replace('*', '')
            if ion_clean not in _ROMAN:
                print(f"Warning: Unknown ion state '{ion_roman}' for line {id_}")
                continue
                
            ion_int = _ROMAN[ion_clean]
            
            # Build name from remaining columns
            name_start_idx = 3
            name_end_idx = len(parts) - 4
            name = " ".join(parts[name_start_idx:name_end_idx]) if name_start_idx < name_end_idx else f"{species} {ion_roman}"
            
            line_obj = Line(
                id=id_,
                species=species,
                ion=ion_int,
                name=name,
                lambda0_A=lambda0_A,
                f_osc=f_osc,
                A21_Hz=A21_Hz,
                mass_amu=mass_amu,
            )
            lines[line_obj.id] = line_obj

    return lines

LINES: Dict[str, Line] = _load_lines()

def get_line(line_id: str) -> Line:
    try:
        return LINES[line_id]
    except KeyError:
        raise KeyError(f"Unknown line: {line_id}") from None