"""Toy field source: the gas cells of a Gaussian random field (``grf_cells``).

A toy model enters the optical depth like a snapshot does: as ``core.Cells``. ``grf_cells`` turns one ``PowerLawGRF``
realisation delta into the densities n = n_bar (1 + delta) of the cells of its own grid, with temperatures and
velocities given as arrays on the grid or as functions of the nodes. A regular grid is already a Voronoi tessellation:
the cell of node x_j is the cube of side L / N centred on it, so no other generating points are needed, and the gas is
the realisation itself, piecewise constant.
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np

from skewerjax.core.cells import cells_on_grid
from skewerjax.simulations.grf import PowerLawGRF


def interpolate_periodic(field: jnp.ndarray, pos_xyz: jnp.ndarray, L_box) -> jnp.ndarray:
    """Periodic trilinear interpolation of a grid ``field`` (values at the nodes x_j = j L / N) at world positions
    (e.g. a realisation read off the grid at arbitrary points; jit- and vmap-compatible).

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

    ix0 = jnp.mod(i0[:, 0], Nx)
    ix1 = jnp.mod(ix0 + 1, Nx)
    iy0 = jnp.mod(i0[:, 1], Ny)
    iy1 = jnp.mod(iy0 + 1, Ny)
    iz0 = jnp.mod(i0[:, 2], Nz)
    iz1 = jnp.mod(iz0 + 1, Nz)
    fx, fy, fz = f[:, 0], f[:, 1], f[:, 2]

    c000 = field[ix0, iy0, iz0]
    c100 = field[ix1, iy0, iz0]
    c010 = field[ix0, iy1, iz0]
    c110 = field[ix1, iy1, iz0]
    c001 = field[ix0, iy0, iz1]
    c101 = field[ix1, iy0, iz1]
    c011 = field[ix0, iy1, iz1]
    c111 = field[ix1, iy1, iz1]

    c00 = c000 * (1 - fx) + c100 * fx
    c10 = c010 * (1 - fx) + c110 * fx
    c01 = c001 * (1 - fx) + c101 * fx
    c11 = c011 * (1 - fx) + c111 * fx
    c0 = c00 * (1 - fy) + c10 * fy
    c1 = c01 * (1 - fy) + c11 * fy
    return c0 * (1 - fz) + c1 * fz


# --------------------------------------------------------------------------- toy gas cells
def _cubic_box(grf: PowerLawGRF) -> float:
    if grf.ndim != 3 or len(set(grf.L)) != 1:
        raise ValueError(f"the toy cells need a cubic 3D PowerLawGRF, got L = {grf.L}.")
    return float(grf.L[0])


def grf_cells(delta, grf: PowerLawGRF, n_bar: dict, *, temperature=1e4, velocity=0.0, turbulence=None, collisional=None):
    """``core.Cells`` of the gas of one 3D ``PowerLawGRF`` realisation on its grid: n = n_bar (1 + delta) per species.

    The cells are those of the grid (``cells.cells_on_grid``): generating points at the nodes x_j = j L / N_j, whose
    Voronoi cells are the cubes of side L / N centred on them, so the gas is the realisation itself, piecewise constant,
    an exact GRF at grid resolution. No transform, clipping or threshold: delta is the realisation as drawn (its
    variance is ``grf.variance()``). The amplitude must keep 1 + delta > 0; a warning is issued when its minimum is <= 0.

    Parameters
    ----------
    delta : (N_x, N_y, N_z) one realisation (``grf.sample(key)``), on the nodes x_j = j L / N_j.
    grf : the cubic 3D ``PowerLawGRF`` of ``delta`` (its box ``grf.L`` is the periodic box of the cells).
    n_bar : {species: mean comoving density [cm^-3]} (e.g. {"HI": 1e-12, "SiIII": 2e-14}).
    temperature [K], velocity [km/s], turbulence [m/s], collisional [GHz] : per cell, as in ``cells.cells_on_grid``:
        a scalar, an array on the grid (N_x, N_y, N_z)[, 3], a callable of the nodes (M, 3), or a dict {species: one of
        those}.
    """
    box = _cubic_box(grf)
    d = np.asarray(delta, dtype=np.float64)
    if d.shape != grf.shape:
        raise ValueError(f"delta must have the grid shape {grf.shape}, got {d.shape}.")
    one = 1.0 + d
    n = {sp: nb * one for sp, nb in n_bar.items()}
    cells = cells_on_grid(box, n, temperature=temperature, velocity=velocity, turbulence=turbulence, collisional=collisional)
    if float(one.min()) <= 0.0:
        warnings.warn(
            f"grf_cells: min(1 + delta) = {float(one.min()):.3f} <= 0, so the density is negative somewhere; lower A_s.",
            stacklevel=2,
        )
    return cells
