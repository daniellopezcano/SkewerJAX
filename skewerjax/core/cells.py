"""Gas cells: the one representation of the gas, whatever its source (toy models or simulation outputs).

The gas is a set of **Voronoi cells**: every point of space belongs to the cell of its nearest generating point, and
inside a cell the gas is uniform. ``Cells`` holds

* the generating points [comoving Mpc] and the periodic box [cMpc] (or None), which the selection and the deposition
  need, and the cell volumes [cMpc^3], whose cube root V^(1/3) is the cell size (the selection radius, and the radius
  of the top-hat option);
* per absorbing species (the keys of ``LineBundle.species_keys``, e.g. "HI", "SiIII"), what the optical depth takes
  per cell (review notes, Sec. GasProperties):

  ========================  ========================================  ============
  ``number_density``        comoving cm^-3                            required
  ``temperature``           K                                         required
  ``velocity``              (M, 3) peculiar velocity, km/s            required
  ``turbulence``            b_turb, m/s                               optional (0)
  ``collisional``           A_coll, GHz                               optional (0)
  ========================  ========================================  ============

Nothing in it is specific to a source. The field sources of ``skewerjax.simulations`` fill it: the toy fields
(``toy_cells.grf_cells``, on the general builders ``cells_on_grid`` and ``cells_from_values`` here), and the snapshots
after their gas physics (``snapshot_cells`` for a region, ``SimulationSkewers.to_cells`` near skewers). The optical
depth is computed from it along skewers by one route: selection (``core.selection``), deposition into pieces
(``core.deposition``), pieces (``core.absorbers.cell_pieces``) and the sum over pieces (``core.optical_depth``).

A regular grid of cells (``cells_on_grid``) carries ``grid`` = its shape: its generators are the nodes x_j = j box / N_j
in C order, and its Voronoi cells are the cubes of side box / N centred on them, so the deposition can find the cell of
a position from the position itself (``deposition.grid_segments``).
"""
from __future__ import annotations

from typing import Callable, Optional, Union

import numpy as np
from jax import tree_util
from scipy.spatial import cKDTree

from skewerjax.core.selection import wrap_positions
from skewerjax.core.validation import check_values

REQUIRED = ("number_density", "temperature", "velocity")
OPTIONAL = ("turbulence", "collisional")
FIELDS = REQUIRED + OPTIONAL


@tree_util.register_pytree_node_class
class Cells:
    """Voronoi cells of the gas: generating points, volumes, the periodic box and, per species, the absorber fields.

    Parameters
    ----------
    positions : (M, 3) generating points [comoving Mpc] (wrapped into [0, box) when ``box`` is given).
    volumes : (M,) cell volumes [comoving Mpc^3].
    species : {species: {field: array}} with the fields of the module docstring; ``velocity`` is (M, 3), the others
        (M,). ``turbulence`` and ``collisional`` may be left out (zero).
    box : periodic box side [cMpc], or None for non-periodic data.
    ids : optional (M,) identifiers (e.g. the snapshot rows of the cells).
    grid : the shape (N_x, N_y, N_z) of a regular grid of cells in C order (``cells_on_grid``), else None.
    validate : check the values (default): finite positions, velocities and densities, volumes > 0, temperatures,
        turbulence and collisional terms >= 0, box > 0; ``ValueError`` names the field and the first bad cell. Negative
        densities are allowed (``simulations.grf_cells`` warns about them). ``take`` and ``select_species`` skip it.

    JAX pytree (positions, volumes, every field and the ids are leaves; the species and field names, ``box`` and
    ``grid`` are static).
    """

    def __init__(self, positions, volumes, species: dict, box: Optional[float] = None, ids=None, grid: Optional[tuple] = None,
                 validate: bool = True):
        self.positions = positions
        self.volumes = volumes
        self.species = {sp: dict(f) for sp, f in species.items()}
        self.box = None if box is None else float(box)
        self.ids = ids
        self.grid = None if grid is None else tuple(int(n) for n in grid)
        n = np.shape(positions)[0]
        if np.shape(positions) != (n, 3):
            raise ValueError(f"positions must be (M, 3), got {np.shape(positions)}.")
        if np.shape(volumes) != (n,):
            raise ValueError(f"volumes must be (M,), got {np.shape(volumes)}.")
        if self.grid is not None and int(np.prod(self.grid)) != n:
            raise ValueError(f"grid {self.grid} holds {int(np.prod(self.grid))} cells, positions hold {n}.")
        for sp, f in self.species.items():
            missing = [k for k in REQUIRED if k not in f]
            if missing:
                raise ValueError(f"species {sp!r}: missing fields {missing} (required: {REQUIRED}).")
            unknown = [k for k in f if k not in FIELDS]
            if unknown:
                raise ValueError(f"species {sp!r}: unknown fields {unknown} (allowed: {FIELDS}).")
            for k, v in f.items():
                want = (n, 3) if k == "velocity" else (n,)
                if np.shape(v) != want:
                    raise ValueError(f"species {sp!r}, field {k!r}: shape {np.shape(v)}, expected {want}.")
        if validate:
            self._validate()

    _MINIMUM = {"temperature": 0.0, "turbulence": 0.0, "collisional": 0.0}      # lower bounds of the fields (None: any)

    def _validate(self):
        """Finite values and physical ranges (class docstring); ``ValueError`` otherwise."""
        if self.box is not None and not (np.isfinite(self.box) and self.box > 0):
            raise ValueError(f"Cells: box must be a finite side > 0 [cMpc] or None, got {self.box}.")
        check_values("Cells: positions [cMpc]", self.positions)
        check_values("Cells: volumes [cMpc^3]", self.volumes, minimum=0.0, strict=True)
        for sp, f in self.species.items():
            for k, v in f.items():
                check_values(f"Cells: species {sp!r}, {k}", v, minimum=self._MINIMUM.get(k))

    # -- pytree protocol
    def tree_flatten(self):
        keys = tuple((sp, tuple(sorted(f))) for sp, f in self.species.items())
        leaves = (self.positions, self.volumes, tuple(self.species[sp][k] for sp, names in keys for k in names), self.ids)
        return leaves, (keys, self.box, self.grid)

    @classmethod
    def tree_unflatten(cls, aux, children):
        keys, box, grid = aux
        obj = object.__new__(cls)
        obj.positions, obj.volumes, vals, obj.ids = children
        it = iter(vals)
        obj.species = {sp: {k: next(it) for k in names} for sp, names in keys}
        obj.box, obj.grid = box, grid
        return obj

    # -- basic API
    def __len__(self) -> int:
        return int(np.shape(self.positions)[0])

    def __repr__(self):
        return (f"Cells({len(self)} cells, species {self.species_keys}, box={self.box}"
                + (f", grid={self.grid}" if self.grid is not None else "") + ")")

    @property
    def species_keys(self) -> tuple:
        return tuple(self.species)

    @property
    def size_cMpc(self):
        """Cube side V^(1/3) [cMpc]: the cell size of the selection radius and the top-hat radius h (the definition of
        ``simulations.GasCells.size_cMpc``, fake_spectra's h = (m / rho)^(1/3))."""
        return np.cbrt(np.asarray(self.volumes))

    def get(self, species: str, field: str) -> np.ndarray:
        """The per-cell values of ``field`` for ``species`` (zeros for an optional field that was left out)."""
        f = self.species[species]
        if field in f:
            return f[field]
        if field in OPTIONAL:
            return np.zeros(len(self))
        raise KeyError(f"unknown field {field!r} (allowed: {FIELDS}).")

    def take(self, idx) -> "Cells":
        """Subset (NumPy index). The subset is no longer a full grid (``grid`` = None)."""
        idx = np.asarray(idx)
        return Cells(np.asarray(self.positions)[idx], np.asarray(self.volumes)[idx],
                     {sp: {k: np.asarray(v)[idx] for k, v in f.items()} for sp, f in self.species.items()}, self.box,
                     None if self.ids is None else np.asarray(self.ids)[idx], validate=False)

    def select_species(self, species) -> "Cells":
        """The same cells with only the species ``species`` (a key or a sequence of keys)."""
        keys = [species] if isinstance(species, str) else list(species)
        missing = [sp for sp in keys if sp not in self.species]
        if missing:
            raise KeyError(f"the cells hold no species {missing} (they hold {self.species_keys}).")
        return Cells(self.positions, self.volumes, {sp: self.species[sp] for sp in keys}, self.box, self.ids, self.grid, validate=False)

    @staticmethod
    def merge_species(parts) -> "Cells":
        """One container with the species of every part (same generating points, volumes and box)."""
        parts = list(parts)
        p0 = parts[0]
        for p in parts[1:]:
            if len(p) != len(p0) or p.box != p0.box or not np.array_equal(np.asarray(p.positions), np.asarray(p0.positions)):
                raise ValueError("merge_species needs the same cells in every part.")
        species = {}
        for p in parts:
            species.update(p.species)
        return Cells(p0.positions, p0.volumes, species, p0.box, p0.ids, p0.grid, validate=False)

    def nbytes(self) -> int:
        """Host memory of the arrays [bytes]."""
        return int(sum(np.asarray(x).nbytes for x in tree_util.tree_leaves(self)))


# --------------------------------------------------------------------------- builders
Value = Union[float, np.ndarray, Callable, dict]


def _resolve(value, positions: np.ndarray, species: str, vector: bool) -> np.ndarray:
    """A per-cell array from a scalar (or (3,) vector), an array, a callable of the generating points (M, 3), or a dict
    {species: one of those}."""
    if isinstance(value, dict):
        value = value[species]
    n = positions.shape[0]
    if callable(value):
        value = value(positions)
    a = np.asarray(value, dtype=np.float64)
    want = (n, 3) if vector else (n,)
    if a.shape == want:
        return a
    if a.size == n * (3 if vector else 1):
        return a.reshape(want)
    if a.ndim == 0 or (vector and a.shape == (3,)):
        return np.broadcast_to(a, want).copy()
    raise ValueError(f"{species}: a per-cell value of shape {a.shape} cannot fill {want}.")


def cells_from_values(positions, box: Optional[float], number_density: dict, *, temperature: Value = 1e4, velocity: Value = 0.0,
                      turbulence: Optional[Value] = None, collisional: Optional[Value] = None, volumes=None, ids=None,
                      grid: Optional[tuple] = None) -> Cells:
    """``Cells`` from generating points and per-species values.

    Parameters
    ----------
    positions : (M, 3) generating points [cMpc].
    box : periodic box side [cMpc] or None.
    number_density : {species: (M,) comoving number density [cm^-3]} (the species of the container).
    temperature [K], velocity [km/s], turbulence [m/s], collisional [GHz] : a scalar (velocity: a scalar or a (3,)
        vector, the same for every cell), an (M,) array ((M, 3) for the velocity), a callable of the generating points
        (M, 3) returning one of those, or a dict {species: one of those}. turbulence / collisional None: left out (0).
    volumes : (M,) cell volumes [cMpc^3]; None: the box volume shared equally, box^3 / M (needs ``box``), or
        ``voronoi_volumes`` for the volume of every cell.
    """
    pos = np.asarray(positions, dtype=np.float64)
    if box is not None:
        pos = wrap_positions(pos, box)
    m = pos.shape[0]
    if volumes is None:
        if box is None:
            raise ValueError("volumes are needed when there is no periodic box.")
        volumes = np.full(m, float(box) ** 3 / m)
    species = {}
    for sp, n in number_density.items():
        f = {"number_density": _resolve(n, pos, sp, False), "temperature": _resolve(temperature, pos, sp, False),
             "velocity": _resolve(velocity, pos, sp, True)}
        if turbulence is not None:
            f["turbulence"] = _resolve(turbulence, pos, sp, False)
        if collisional is not None:
            f["collisional"] = _resolve(collisional, pos, sp, False)
        species[sp] = f
    return Cells(pos, np.asarray(volumes, dtype=np.float64), species, box, ids, grid)


def grid_points(box: float, shape) -> np.ndarray:
    """(N_x N_y N_z, 3) nodes x_j = j box / N_j of a periodic grid, in C order [cMpc]: the generating points of a grid
    of cells, the centres of its cubic Voronoi cells (and the positions of ``simulations.PowerLawGRF.x_grid``)."""
    axes = [np.arange(n) * (float(box) / n) for n in shape]
    return np.stack([g.ravel() for g in np.meshgrid(*axes, indexing="ij")], -1)


def cells_on_grid(box: float, number_density: dict, **kw) -> Cells:
    """``Cells`` of a periodic grid: one cell per node, with values given on the grid.

    The generating points are the nodes x_j = j box / N_j (``grid_points``), so each cell is the cube of side box / N_j
    centred on its node, and the gas is the grid values, piecewise constant. number_density : {species: (N_x, N_y, N_z)
    comoving density [cm^-3]}; the other values as in ``cells_from_values`` (arrays given on the grid (N_x, N_y, N_z)
    [, 3], or per cell in C order). The volumes are prod(box / N_j).
    """
    shapes = {np.shape(n) for n in number_density.values()}
    if len(shapes) != 1 or len(next(iter(shapes))) != 3:
        raise ValueError(f"number_density must hold (N_x, N_y, N_z) arrays of one shape, got {shapes}.")
    shape = next(iter(shapes))
    pos = grid_points(box, shape)
    flat = {sp: np.asarray(n, dtype=np.float64).reshape(-1) for sp, n in number_density.items()}
    vol = np.full(pos.shape[0], float(np.prod([float(box) / n for n in shape])))
    return cells_from_values(pos, box, flat, volumes=vol, grid=shape, **kw)


def voronoi_volumes(positions, box: float, *, samples_per_cell: int = 32, seed: int = 0, workers: int = -1) -> np.ndarray:
    """Monte Carlo estimate of the Voronoi volume of every generating point in a periodic box [cMpc^3].

    ``samples_per_cell`` x M uniform points are assigned to their nearest generator (periodic k-d tree); a cell's
    volume is its share of the points times box^3, with a floor of half a sample (a cell missed by every sample keeps a
    finite size). Unbiased otherwise; relative scatter 1 / sqrt(samples per cell), so the size V^(1/3) has a third of
    it. (For a grid of cells the volumes are exact: ``cells_on_grid``.)
    """
    pos = wrap_positions(positions, box)
    m = pos.shape[0]
    tree = cKDTree(pos, boxsize=box)
    rng = np.random.default_rng(seed)
    counts = np.zeros(m)
    n_total = int(samples_per_cell) * m
    chunk = 4_000_000
    for i0 in range(0, n_total, chunk):
        u = rng.uniform(0.0, box, (min(chunk, n_total - i0), 3))
        counts += np.bincount(tree.query(wrap_positions(u, box), k=1, workers=workers)[1], minlength=m)
    return np.maximum(counts, 0.5) / n_total * float(box) ** 3
