"""A regular lattice of straight skewers along y through the centres of the cell columns of a periodic N^3 grid of cell
size dx (notebook 09, Part A): the geometry only, as the points of the paths of a ``core.SkewerBundle``."""

from __future__ import annotations

import numpy as np


def lattice_columns(n_cells: int, stride) -> np.ndarray:
    """(N_sk, 2) node indices (i_x, i_z) of the skewer columns of an ``n_cells``^3 grid: every stride[0]-th column in x,
    every stride[1]-th in z, from column 0 (row-major, x outer: the order of ``summary_stats.on_lattice``).
    ceil(N / s_x) ceil(N / s_z) skewers; a periodic lattice (the 3D statistics) needs N to be a multiple of both strides."""
    n = int(n_cells)
    ix, iz = np.arange(0, n, stride[0]), np.arange(0, n, stride[1])
    return np.stack(np.meshgrid(ix, iz, indexing="ij"), -1).reshape(-1, 2)


def lattice_points(columns, dx: float, L: float, n_cells: int, points_per_cell: int) -> np.ndarray:
    """(N_sk, n_cells points_per_cell + 1, 3) points [cMpc] of skewers along y through the centres x = i_x dx,
    z = i_z dx of the cell columns ``columns`` ((N_sk, 2) node indices, ``lattice_columns``; the nodes of
    ``simulations.grf_cells``), from y = 0 to y = L, ``points_per_cell`` points per cell: the first and the last point are
    the same place of the periodic box, so ``velocity_period=True`` is one box. The points of a ``core.SkewerBundle``
    (``spatial_points``), with the cosmology and the frame passed to it."""
    cols = np.atleast_2d(columns)
    y = np.linspace(0.0, L, int(n_cells) * int(points_per_cell) + 1)
    pts = np.empty((cols.shape[0], y.size, 3))
    pts[..., 0] = cols[:, :1] * dx
    pts[..., 1] = y
    pts[..., 2] = cols[:, 1:] * dx
    return pts
