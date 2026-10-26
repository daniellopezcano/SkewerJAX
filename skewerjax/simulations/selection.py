"""Cell selection (layer 4): streaming k-d tree selection of the gas cells of a snapshot that can reach a set of paths.

The selection rule, its radii (``"voronoi"``: 2.5 V^(1/3) + ds/2; ``"tophat"``: V^(1/3) + ds/2) and its back ends
(``"cpu"``, ``"jztree"``, ``"auto"``) are ``core.selection``'s, the same for every source of cells; cells already in memory
(a toy model, ``snapshot_cells``) go through ``core.select_cells``. This module applies the same rule to a snapshot
streamed from disk, block by block: a periodic k-d tree is built on the *path points* (few: N_sk x N_pts) and each block
of cells streamed by the reader (layer 1) is queried against it. Memory is bounded by one block of cells (``max_rows``)
plus the tree. Skewer batches are independent, so multi-device runs can distribute them; each worker streams the same
files.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.spatial import cKDTree

from skewerjax.core.selection import (_nearest_distance_cpu, _nearest_distance_jztree, max_path_spacing, resolve_backend,
                                      selection_radius, wrap_positions)
from skewerjax.simulations.gas_cells import GasCells

GEOMETRY_DATASETS = ("Coordinates", "Masses", "Density")
PREFILTER_GRID = 64                       # cells per box side of the coarse occupancy grid of the paths


class _PathOccupancy:
    """The coarse periodic grid cells (``n`` per side, width h = box / n) that hold a path point, dilated by k cells
    (Chebyshev, periodic) on demand. A cell at distance <= r from a path point lies within k = floor(r / h) + 2 grid
    cells of it on every axis (+1 for the floor, +1 for rounding in x / h), so ``candidates(pos, r_max)`` keeps every
    cell that can be within r_max of a path: a conservative prefilter of the k-d tree query."""

    def __init__(self, pts: np.ndarray, box: float, n: int = PREFILTER_GRID):
        self.n, self.h = int(n), float(box) / int(n)
        occ = np.zeros((self.n,) * 3, dtype=bool)
        i = np.floor(pts / self.h).astype(np.int64) % self.n
        occ[i[:, 0], i[:, 1], i[:, 2]] = True
        self._dilated = {0: occ}

    def grid(self, k: int) -> Optional[np.ndarray]:
        """The occupancy dilated by k cells, or None when it covers the whole box."""
        if 2 * k + 1 >= self.n:
            return None
        if k not in self._dilated:
            prev = self.grid(k - 1)
            if prev is None:
                return None
            out = prev.copy()
            for ax in range(3):
                out = out | np.roll(out, 1, ax) | np.roll(out, -1, ax)
            self._dilated[k] = out
        return self._dilated[k]

    def candidates(self, pos: np.ndarray, r_max: float) -> Optional[np.ndarray]:
        """Mask of the positions (N, 3) in [0, box) that can be within ``r_max`` of a path; None: all of them."""
        g = self.grid(int(np.floor(r_max / self.h)) + 2)
        if g is None:
            return None
        flat = np.zeros(pos.shape[0], dtype=np.int32)
        for ax in range(3):                                             # flat index of the grid cell, axis by axis
            i = (pos[:, ax] * (1.0 / self.h)).astype(np.int32)         # positions >= 0: truncation is the floor
            i %= self.n
            flat *= self.n
            flat += i
        return g.ravel().take(flat)


def select_cells_near_paths(reader, paths, *, mode: str = "voronoi", max_rows: int = 4_000_000, workers: int = -1,
                            backend: str = "cpu") -> GasCells:
    """Stream the snapshot and keep the cells that can contribute to any of ``paths``.

    Parameters
    ----------
    reader : layer-1 reader (``iter_blocks``, ``read_rows``, ``to_gas_cells``, ``header``).
    paths : (N_sk, N_pts, 3) sampling points [cMpc], e.g. ``SkewerBundle.spatial_points``.
    mode : "voronoi" (default deposition) or "tophat" (fake_spectra's kernel); "voronoi" selects a superset of
        "tophat".
    max_rows : cells per streamed block (memory bound).
    workers : threads of the k-d tree queries (-1: all cores).
    backend : "cpu" (default, scipy cKDTree), "jztree" (CUDA GPU; ImportError / RuntimeError if jztree or a CUDA
        device is missing) or "auto" (``resolve_backend``). Every back end selects the same cells.

    Returns
    -------
    GasCells of the selected cells, physical units (``reader.to_gas_cells``), with ``ids`` =
    (file_index << 40) | row, so that every cell can be traced back to the snapshot.
    """
    from skewerjax.simulations.readers import GAS_DATASETS
    rest = [k for k in GAS_DATASETS if k not in GEOMETRY_DATASETS]
    hdr = reader.header
    box = hdr.box_cMpc
    ds = max_path_spacing(paths)
    pts = wrap_positions(np.asarray(paths, dtype=np.float64).reshape(-1, 3), box)
    tree = cKDTree(pts, boxsize=box)
    backend = resolve_backend(backend)
    occ = _PathOccupancy(pts, box)
    parts = []
    n_sel = 0
    for fi, r0, base in reader.iter_blocks(GEOMETRY_DATASETS, max_rows=max_rows):
        pos = reader.positions_cMpc(base["Coordinates"])
        size = np.cbrt(reader.volumes_cMpc3(base["Masses"], base["Density"]))
        rad = selection_radius(size, mode, ds)
        if backend == "jztree":
            d = _nearest_distance_jztree(pts, pos, box, tree, rad)
        else:                                                # only the cells near the paths reach the tree
            d = _nearest_distance_cpu(tree, pos, rad, workers, candidates=occ.candidates(pos, float(rad.max())))
        sel = np.nonzero(d <= rad)[0]
        if sel.size:
            raw = {k: base[k][sel] for k in GEOMETRY_DATASETS}
            lo, hi = r0 + int(sel[0]), r0 + int(sel[-1]) + 1     # one contiguous slab of the block (bounded by max_rows),
            slab = reader.read_rows(fi, rest, start=lo, stop=hi)  # indexed in memory: h5py's scattered reads are slower
            raw.update({k: v[r0 + sel - lo] for k, v in slab.items()})
            parts.append(reader.to_gas_cells(raw, ids=(np.int64(fi) << 40) | (r0 + sel).astype(np.int64)))
            n_sel += sel.size
    if n_sel == 0:
        raise RuntimeError("no gas cell is within reach of the paths.")
    return GasCells.concatenate(parts)
