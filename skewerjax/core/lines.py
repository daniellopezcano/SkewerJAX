"""Atomic transitions: the line table (``LINES``) and ``LineBundle``, the lines of one calculation."""
from __future__ import annotations

import re
from dataclasses import dataclass
from importlib.resources import files
from typing import Dict, List, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from skewerjax.core.constants import phys


@dataclass(frozen=True)
class Line:
    """One transition of ``lines_table.dat``: rest wavelength [A], oscillator strength, Einstein A [Hz], mass [amu]."""
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


def _load_lines() -> Dict[str, Line]:
    """Parse ``lines_table.dat`` (header line, then id, species, name..., lambda0_A, f_osc, A21_Hz, mass_amu)."""
    lines: Dict[str, Line] = {}
    with (files("skewerjax.core") / "lines_table.dat").open("r", encoding="utf-8") as f:
        next(f)
        for row in f:
            row = row.strip()
            if not row or row.startswith(("#", "!")):
                continue
            parts = row.split()
            if len(parts) < 6:
                continue
            name = " ".join(parts[2:len(parts) - 4]) if len(parts) > 6 else parts[1]
            lines[parts[0]] = Line(id=parts[0], species=parts[1], name=name, lambda0_A=float(parts[-4]),
                                   f_osc=float(parts[-3]), A21_Hz=float(parts[-2]), mass_amu=float(parts[-1]))
    return lines


LINES: Dict[str, Line] = _load_lines()


def get_line(line_id: str) -> Line:
    """The ``Line`` of ``line_id`` in ``lines_table.dat`` (KeyError if unknown)."""
    try:
        return LINES[line_id]
    except KeyError:
        raise KeyError(f"Unknown line: {line_id}") from None


@jax.tree_util.register_pytree_node_class
class LineBundle:
    """The transitions of one calculation, with per-line arrays and species bookkeeping.

    Parameters
    ----------
    line_ids : ids of ``lines_table.dat``, e.g. ``["HI-LyA", "HI-LyB", "SiIII-1206"]``.
    ref_line_id : reference line of the relative offsets delta_v_j = c ln(lambda_ref / lambda_j)
        (review notes, Sec. relative line-centred velocity units).

    Notes
    -----
    JAX pytree: the per-line arrays (``lambda0_A``, ``f_osc``, ``A21_GHz``, ``mass_amu``) are leaves; the
    ``Line`` records are static. The leaves are float64 NumPy arrays converted on access, so their dtype
    follows ``jax_enable_x64`` at the time of use.
    """

    def __init__(self, line_ids: Sequence[str], ref_line_id: str = "HI-LyA"):
        if isinstance(line_ids, str):
            raise TypeError(f"line_ids must be a sequence of line ids, e.g. [{line_ids!r}], not a string.")
        if len(line_ids) == 0:
            raise ValueError("a LineBundle needs at least one line.")
        missing = [lid for lid in line_ids if lid not in LINES]
        if missing:
            raise ValueError(f"Unknown line IDs: {missing}")
        if ref_line_id not in LINES:
            raise ValueError(f"Unknown reference line ID: {ref_line_id}")
        self.lines = [LINES[lid] for lid in line_ids]
        self.ref_line = LINES[ref_line_id]
        self._arrays = tuple(np.array([getattr(l, a) for l in self.lines], dtype=np.float64)
                             for a in ("lambda0_A", "f_osc", "A21_GHz", "mass_amu"))

    def tree_flatten(self):
        return self._arrays, (tuple(self.lines), self.ref_line)

    @classmethod
    def tree_unflatten(cls, aux, leaves):
        obj = object.__new__(cls)
        obj.lines, obj.ref_line = list(aux[0]), aux[1]
        obj._arrays = tuple(leaves)
        return obj

    # -- metadata
    @property
    def n_lines(self) -> int:
        return len(self.lines)

    @property
    def species_keys(self) -> List[str]:
        """Species in order of first appearance, e.g. ``["HI", "SiIII"]``."""
        return list(dict.fromkeys(l.species for l in self.lines))

    @property
    def n_species(self) -> int:
        return len(self.species_keys)

    @property
    def ref_line_id(self) -> str:
        return self.ref_line.id

    @property
    def ref_lambda0_A(self) -> float:
        return self.ref_line.lambda0_A

    # -- per-line arrays (N_lines,)
    @property
    def lambda0_A(self) -> jnp.ndarray:
        return jnp.asarray(self._arrays[0])

    @property
    def f_osc(self) -> jnp.ndarray:
        return jnp.asarray(self._arrays[1])

    @property
    def A21_GHz(self) -> jnp.ndarray:
        return jnp.asarray(self._arrays[2])

    @property
    def mass_amu(self) -> jnp.ndarray:
        return jnp.asarray(self._arrays[3])

    @property
    def delta_v(self) -> jnp.ndarray:
        """delta_v_j = c ln(lambda_ref / lambda_j) [km/s]."""
        return phys.c_kms * jnp.log(self.ref_line.lambda0_A / self.lambda0_A)

    # -- species bookkeeping
    def lines_for_species(self, species: str) -> List[Line]:
        """The lines of ``species``, in bundle order."""
        return [l for l in self.lines if l.species == species]

    def indices_for_species(self, species: str) -> List[int]:
        """Line indices of ``species`` (to slice arrays such as tau[indices])."""
        return [i for i, l in enumerate(self.lines) if l.species == species]

    def delta_v_for_line(self, line_id: str) -> float:
        """Delta v_j = c ln(lambda_ref / lambda_j) [km/s] of one line of the bundle."""
        for l in self.lines:
            if l.id == line_id:
                return float(phys.c_kms * jnp.log(self.ref_line.lambda0_A / l.lambda0_A))
        raise KeyError(f"Line {line_id!r} is not present in this LineBundle")

    def ref_index(self) -> Optional[int]:
        """Index of the reference line in the bundle, or None."""
        return next((i for i, l in enumerate(self.lines) if l.id == self.ref_line.id), None)


def _species_to_element(species: str) -> str:
    """Element of a line-table species token: 'HI' -> 'H', 'SiIII' -> 'Si', 'DI' -> 'D', 'nullHI' -> 'H'."""
    s = re.sub(r"^null", "", species.strip(), flags=re.IGNORECASE)
    if s and s[0].upper() == "D":
        return "D"
    m = re.match(r"^([A-Z][a-z]?)", s) or re.match(r"^([A-Z][a-z]?)", s[:1].upper() + s[1:])
    if not m:
        raise ValueError(f"Cannot parse element symbol from species={species!r}")
    return m.group(1)


def get_mass_amu(elements: Sequence[str]) -> Dict[str, float]:
    """Atomic masses [amu] of plain elements ('H', 'Si', ...; case-insensitive), from the line table: {element: mass}."""
    table: Dict[str, float] = {}
    for line in LINES.values():
        try:
            elem = _species_to_element(line.species)
        except ValueError:
            continue
        if line.mass_amu == line.mass_amu:                 # skip NaN
            table.setdefault(elem, float(line.mass_amu))
    elems = [e.strip()[:1].upper() + e.strip()[1:].lower() for e in elements]
    missing = [e for e in elems if e not in table]
    if missing:
        raise KeyError(f"Could not find mass_amu for elements: {missing}. Available: {sorted(table)}")
    return {e: table[e] for e in elems}
