from dataclasses import dataclass
from typing import Dict, Sequence, List
from jax import numpy as jnp
from importlib.resources import files

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
