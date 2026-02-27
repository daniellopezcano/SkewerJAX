from dataclasses import dataclass
from typing import Dict, Sequence, List, Union
from jax import numpy as jnp
from importlib.resources import files
import re

# ----------------- dataclass -----------------

@dataclass(frozen=True)
class Line:
    id: str
    species: str
    name: str
    lambda0_A: float
    f_osc: float
    A21_Hz: float
    mass_amu: float
    
    @property
    def A21_GHz(self) -> float:
        return self.A21_Hz * 1e-9

# ----------------- loading -------------------

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
            lambda0_A = float(parts[-4])
            f_osc = float(parts[-3])
            A21_Hz = float(parts[-2])
            mass_amu = float(parts[-1]) if len(parts) > 5 else float('nan')
            
            # Build name from remaining columns
            name_start_idx = 2
            name_end_idx = len(parts) - 4
            name = " ".join(parts[name_start_idx:name_end_idx]) if name_start_idx < name_end_idx else f"{species}"
            
            line_obj = Line(
                id=id_,
                species=species,
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
    

# ----------------- LineBundle -----------------

class LineBundle:
    """A collection of absorption lines with convenient accessors."""
    
    def __init__(self, line_ids: Sequence[str]):
        """Create from line IDs like ['HI-LyA', 'HI-LyB', 'SiIII-1206']."""
        self.lines = [LINES[lid] for lid in line_ids]
        if len(self.lines) != len(line_ids):
            missing = set(line_ids) - set(LINES.keys())
            raise ValueError(f"Unknown line IDs: {missing}")
        
        self._species_map = self._build_species_map()
    
    def _build_species_map(self) -> Dict[str, List[Line]]:
        """Group lines by species (e.g., 'HI' → [LyA, LyB])."""
        mapping = {}
        for line in self.lines:
            mapping.setdefault(line.species, []).append(line)
        return mapping
    
    # ────────── Basic metadata ──────────
    @property
    def n_lines(self) -> int:
        return len(self.lines)
    
    @property
    def n_species(self) -> int:
        return len(self._species_map)
    
    @property
    def species_keys(self) -> List[str]:
        """Species identifiers like ['HI', 'SiIII']."""
        return list(self._species_map.keys())
    
    # ────────── JAX-friendly arrays ──────────
    @property
    def lambda0_A(self) -> jnp.ndarray:
        return jnp.array([line.lambda0_A for line in self.lines])
    
    @property
    def A21_GHz(self) -> jnp.ndarray:
        return jnp.array([line.A21_GHz for line in self.lines])
    
    @property
    def mass_amu(self) -> jnp.ndarray:
        return jnp.array([line.mass_amu for line in self.lines])
    
    @property
    def f_osc(self) -> jnp.ndarray:
        return jnp.array([line.f_osc for line in self.lines])
    
    # ────────── Species-based access ──────────
    def lines_for_species(self, species: str) -> List[Line]:
        """Get all lines for a species."""
        return self._species_map.get(species, [])
    
    def indices_for_species(self, species: str) -> List[int]:
        """Get line indices for slicing arrays like phi[indices, :, :]."""
        return [i for i, line in enumerate(self.lines) if line.species == species]



def _species_to_element(species: str) -> str:
    """
    Map a line-table `species` token (e.g. 'HI', 'HeII', 'SiIII', 'HI2', 'DI', 'CIV')
    to a plain element/isotope label (e.g. 'H', 'He', 'Si', 'H', 'D', 'C').

    Robust rule:
      1) Strip leading "null" if present (some of your IDs/names use it).
      2) Grab the first 1–2 letters as an element symbol:
         - first char must be a letter
         - optional second char is lowercase letter if present
      3) Special-case Deuterium 'D' if the token starts with 'D' (e.g. 'DI').
    """
    s = species.strip()
    s = re.sub(r"^null", "", s, flags=re.IGNORECASE)  # handle 'nullHI', etc.

    # Special case: Deuterium lines are 'DI', 'DII', etc -> element 'D'
    if s and s[0].upper() == "D":
        return "D"

    # Element symbol = first letter + optional lowercase second letter
    m = re.match(r"^([A-Z][a-z]?)", s)
    if not m:
        # Sometimes tables might use lowercase; try normalizing
        s2 = s[:1].upper() + s[1:]
        m = re.match(r"^([A-Z][a-z]?)", s2)
    if not m:
        raise ValueError(f"Cannot parse element symbol from species={species!r}")

    return m.group(1)


def get_mass_amu(elements: Sequence[str], *, return_dict: bool = True) -> Union[Dict[str, float], "jnp.ndarray"]:
    """
    Get atomic masses [amu] for requested plain elements like ['H','He','Si'].

    Scans your existing `LINES` dict (loaded from lines_table.dat), infers the
    plain element for each line via `_species_to_element(line.species)` and
    returns masses for the requested elements.

    Parameters
    ----------
    elements : sequence[str]
        Plain element symbols (case-insensitive): 'H','He','Si',...
    return_dict : bool
        If True -> dict {elem: mass_amu}. If False -> jnp.array in same order.
    """
    # Build element -> mass_amu by scanning LINES once
    elem_to_mass: Dict[str, float] = {}
    for line in LINES.values():
        try:
            elem = _species_to_element(line.species)
        except Exception:
            continue
        m = float(line.mass_amu)
        if not (m == m):  # NaN guard
            continue
        elem_to_mass.setdefault(elem, m)

    # Normalize requested elements
    def _norm_elem(e: str) -> str:
        e = e.strip()
        if not e:
            return e
        # normalize capitalization: 'si' -> 'Si', 'h' -> 'H'
        return e[0].upper() + (e[1:].lower() if len(e) > 1 else "")

    elems = [_norm_elem(e) for e in elements]

    out: Dict[str, float] = {}
    missing = [e for e in elems if e not in elem_to_mass]
    if missing:
        raise KeyError(
            f"Could not find mass_amu for elements: {missing}. "
            f"Available elements inferred from lines_table: {sorted(elem_to_mass.keys())}"
        )

    for e in elems:
        out[e] = elem_to_mass[e]

    if return_dict:
        return out
    return jnp.array([out[e] for e in elems])