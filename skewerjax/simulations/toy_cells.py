"""Toy field source: the gas cells of a Gaussian random field (``grf_cells``) and their generating points.

A toy model enters the optical depth like a snapshot does: as ``core.Cells``. ``grf_cells`` turns one ``PowerLawGRF``
realisation delta into the densities n = n_bar (1 + delta) of a set of Voronoi cells, with temperatures and velocities
given as arrays or as functions of the generating points. The generating points are the grid nodes by default (the
cells are the grid cells), or any points: ``uniform_generators`` (a uniform random mesh) and ``density_weighted_generators``
(more points where the gas is denser, like a moving mesh).
"""
from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np

from skewerjax.core.cells import cells_from_values, cells_on_grid, voronoi_volumes
from skewerjax.simulations.grf import PowerLawGRF


def interpolate_periodic(field: jnp.ndarray, pos_xyz: jnp.ndarray, L_box) -> jnp.ndarray:
    """Periodic trilinear interpolation of a grid ``field`` (values at the nodes x_j = j L / N) at world positions.

    field   : (Nx, Ny, Nz) real
    pos_xyz : (M, 3) positions, periodic in [0, L_box)
    L_box   : float or (3,) array, box side(s) (anisotropic boxes allowed)
    returns : (M,) sampled values
    """
    Nx, Ny, Nz = field.shape
    N = jnp.array([Nx, Ny, Nz], dtype=pos_xyz.dtype)
    u = (jnp.mod(pos_xyz, L_box) / L_box) * N
    i0 = jnp.floor(u).astype(jnp.int32)
    f = u - i0.astype(u.dtype)

    ix0 = jnp.mod(i0[:, 0], Nx); ix1 = jnp.mod(ix0 + 1, Nx)
    iy0 = jnp.mod(i0[:, 1], Ny); iy1 = jnp.mod(iy0 + 1, Ny)
    iz0 = jnp.mod(i0[:, 2], Nz); iz1 = jnp.mod(iz0 + 1, Nz)
    fx, fy, fz = f[:, 0], f[:, 1], f[:, 2]

    c000 = field[ix0, iy0, iz0]; c100 = field[ix1, iy0, iz0]
    c010 = field[ix0, iy1, iz0]; c110 = field[ix1, iy1, iz0]
    c001 = field[ix0, iy0, iz1]; c101 = field[ix1, iy0, iz1]
    c011 = field[ix0, iy1, iz1]; c111 = field[ix1, iy1, iz1]

    c00 = c000 * (1 - fx) + c100 * fx
    c10 = c010 * (1 - fx) + c110 * fx
    c01 = c001 * (1 - fx) + c101 * fx
    c11 = c011 * (1 - fx) + c111 * fx
    c0 = c00 * (1 - fy) + c10 * fy
    c1 = c01 * (1 - fy) + c11 * fy
    return c0 * (1 - fz) + c1 * fz


# --------------------------------------------------------------------------- generating points
def _cubic_box(grf: PowerLawGRF) -> float:
    if grf.ndim != 3 or len(set(grf.L)) != 1:
        raise ValueError(f"the toy cells need a cubic 3D PowerLawGRF, got L = {grf.L}.")
    return float(grf.L[0])


def uniform_generators(grf: PowerLawGRF, n_points: int, *, seed: int = 0) -> np.ndarray:
    """(n_points, 3) generating points uniform in the periodic box of ``grf`` [cMpc] (``np.random.default_rng(seed)``):
    a random mesh with no relation to the field; the mean cell volume is L^3 / n_points."""
    box = _cubic_box(grf)
    return np.random.default_rng(seed).uniform(0.0, box, (int(n_points), 3))


def density_weighted_generators(delta, grf: PowerLawGRF, n_points: int, *, seed: int = 0) -> np.ndarray:
    """(n_points, 3) generating points whose number density follows the gas, as in a moving mesh [cMpc].

    Rule: the expected number density of points is (n_points / L^3) (1 + delta_j) inside the grid cell of node j (the
    cube of side L / N centred on it), i.e. proportional to the gas density, so the cells hold equal mass on average
    and are smaller where the gas is denser. Drawn exactly: a cell j with probability (1 + delta_j) / sum_i (1 + delta_i)
    (cells with 1 + delta <= 0 get none), then a uniform position inside it; ``np.random.default_rng(seed)``.
    delta : (N_x, N_y, N_z) the realisation of ``grf`` on its nodes.
    """
    box = _cubic_box(grf)
    d = np.asarray(delta, dtype=np.float64)
    if d.shape != grf.shape:
        raise ValueError(f"delta must have the grid shape {grf.shape}, got {d.shape}.")
    w = np.clip(1.0 + d.ravel(), 0.0, None)
    rng = np.random.default_rng(seed)
    j = rng.choice(w.size, size=int(n_points), p=w / w.sum())
    dx = np.asarray(grf.dx)
    node = np.stack(np.unravel_index(j, d.shape), -1) * dx
    return np.mod(node + rng.uniform(-0.5, 0.5, (int(n_points), 3)) * dx, box)


# --------------------------------------------------------------------------- toy gas cells
def grf_cells(delta, grf: PowerLawGRF, n_bar: dict, *, generators=None, temperature=1e4, velocity=0.0, turbulence=None,
              collisional=None, volumes: str = "voronoi", samples_per_cell: int = 32, seed: int = 0):
    """``core.Cells`` of the gas of one 3D ``PowerLawGRF`` realisation: n = n_bar (1 + delta) for every species.

    No transform, clipping or threshold: delta is the realisation as drawn (its variance is ``grf.variance()``). The
    amplitude must keep 1 + delta > 0; a warning is issued when the minimum over the cells is <= 0.

    Parameters
    ----------
    delta : (N_x, N_y, N_z) one realisation (``grf.sample(key)``), on the nodes x_j = j L / N_j.
    grf : the cubic 3D ``PowerLawGRF`` of ``delta`` (its box ``grf.L`` is the periodic box of the cells).
    n_bar : {species: mean comoving density [cm^-3]} (e.g. {"HI": 1e-12, "SiIII": 2e-14}).
    generators : None (default): the nodes of the grid, whose Voronoi cells are exactly the grid cells (cubes of side
        L / N centred on the nodes): the gas is the realisation itself, piecewise constant, an exact GRF at grid
        resolution (``cells.cells_on_grid``). Or (M, 3) generating points [same length unit as the box], e.g.
        ``uniform_generators`` or ``density_weighted_generators`` (a moving mesh): each takes delta interpolated
        trilinearly at its position (``interpolate_periodic``), so the gas is the realisation smoothed by the
        interpolation and by its own cells.
    temperature [K], velocity [km/s], turbulence [m/s], collisional [GHz] : per cell, as in ``cells.cells_from_values``:
        a scalar, an array (on the grid (N_x, N_y, N_z)[, 3] for the default generators, else (M,)[, 3]), a callable
        of the generating points (M, 3), or a dict {species: one of those}.
    volumes : generators given only: "voronoi" (default, ``cells.voronoi_volumes`` with ``samples_per_cell`` and
        ``seed``: each cell's own volume, so its size V^(1/3) is the simulation's definition) or "mean" (L^3 / M).
    """
    box = _cubic_box(grf)
    d = np.asarray(delta, dtype=np.float64)
    if d.shape != grf.shape:
        raise ValueError(f"delta must have the grid shape {grf.shape}, got {d.shape}.")
    if generators is None:
        one = 1.0 + d
        n = {sp: nb * one for sp, nb in n_bar.items()}
        cells = cells_on_grid(box, n, temperature=temperature, velocity=velocity, turbulence=turbulence, collisional=collisional)
    else:
        pos = np.mod(np.asarray(generators, dtype=np.float64), box)
        one = 1.0 + np.asarray(interpolate_periodic(jnp.asarray(d), jnp.asarray(pos), box))
        if volumes == "voronoi":
            vol = voronoi_volumes(pos, box, samples_per_cell=samples_per_cell, seed=seed)
        elif volumes == "mean":
            vol = np.full(pos.shape[0], box ** 3 / pos.shape[0])
        else:
            raise ValueError(f"volumes must be 'voronoi' or 'mean', got {volumes!r}.")
        n = {sp: nb * one for sp, nb in n_bar.items()}
        cells = cells_from_values(pos, box, n, temperature=temperature, velocity=velocity, turbulence=turbulence,
                                  collisional=collisional, volumes=vol)
    if float(one.min()) <= 0.0:
        warnings.warn(f"grf_cells: min(1 + delta) = {float(one.min()):.3f} <= 0, so the density is negative somewhere; "
                      "lower A_s.", stacklevel=2)
    return cells
