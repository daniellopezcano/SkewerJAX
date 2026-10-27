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

import logging
import time

import numpy as np

from skewerjax.core.selection import _nearest_distance_cpu, _nearest_distance_jztree, path_tree, resolve_backend, selection_radius
from skewerjax.simulations.gas_cells import GasCells

GEOMETRY_DATASETS = ("Coordinates", "Masses", "Density")
PREFILTER_GRID = 64  # cells per box side of the coarse occupancy grid of the paths
logger = logging.getLogger(__name__)


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

    def grid(self, k: int) -> np.ndarray | None:
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

    def candidates(self, pos: np.ndarray, r_max: float, *, scale: float = 1.0, margin: int = 0) -> np.ndarray | None:
        """Mask of the positions (N, 3) in [0, box) that can be within ``r_max`` of a path; None: all of them.

        ``scale``: the positions are given in another unit, x = scale pos (e.g. code coordinates and their length in
        cMpc), so they need no conversion; ``margin``: extra grid cells of dilation (e.g. 1 for positions that may lie
        just outside [0, box) or carry a different rounding)."""
        g = self.grid(int(np.floor(r_max / self.h)) + 2 + int(margin))
        if g is None:
            return None
        q = (pos * (scale / self.h)).astype(np.int32)  # positions >= 0: truncation is the floor
        q %= self.n
        return g.ravel().take((q[:, 0] * self.n + q[:, 1]) * self.n + q[:, 2])


def read_selected(reader, file_index: int, row_offset: int, sel, base: dict, rest) -> GasCells:
    """``GasCells`` of the rows ``sel`` of one streamed block (``reader.iter_blocks``: file ``file_index``, first row
    ``row_offset``, datasets ``base`` already read), with the datasets ``rest`` read from one contiguous slab of the file
    (bounded by the block; indexed in memory, because h5py's scattered reads are slower), ids = (file_index << 40) | row."""
    lo, hi = row_offset + int(sel[0]), row_offset + int(sel[-1]) + 1
    slab = reader.read_rows(file_index, rest, start=lo, stop=hi)
    raw = {k: v[sel] for k, v in base.items()}
    raw.update({k: v[row_offset + sel - lo] for k, v in slab.items()})
    return reader.to_gas_cells(raw, ids=(np.int64(file_index) << 40) | (row_offset + sel).astype(np.int64))


def select_cells_near_paths(
    reader,
    paths,
    *,
    mode: str = "voronoi",
    max_rows: int = 4_000_000,
    workers: int = -1,
    backend: str = "cpu",
    return_info: bool = False,
) -> GasCells:
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

    return_info : also return dict(n_gas, n_blocks, n_candidates (cells left by the coarse prefilter), n_selected,
        ds_cMpc, backend, timings {read, prefilter, query, read_selected} [s], time_s).

    Returns
    -------
    GasCells of the selected cells, physical units (``reader.to_gas_cells``), with ``ids`` =
    (file_index << 40) | row, so that every cell can be traced back to the snapshot (and the info).
    """
    t0 = time.perf_counter()
    from skewerjax.simulations.readers import GAS_DATASETS

    rest = [k for k in GAS_DATASETS if k not in GEOMETRY_DATASETS]
    hdr = reader.header
    box = hdr.box_cMpc
    pts, tree, ds = path_tree(paths, box)
    backend = resolve_backend(backend)
    occ = _PathOccupancy(pts, box)
    parts = []
    n_sel = n_cand = n_blocks = 0
    tm = dict(read=0.0, prefilter=0.0, query=0.0, read_selected=0.0)
    t = time.perf_counter()
    for fi, r0, base in reader.iter_blocks(GEOMETRY_DATASETS, max_rows=max_rows):
        vol = reader.volumes_cMpc3(base["Masses"], base["Density"])
        n_rows = vol.shape[0]
        t1 = time.perf_counter()
        tm["read"] += t1 - t
        if backend == "jztree":
            pos, rad = reader.positions_cMpc(base["Coordinates"]), selection_radius(np.cbrt(vol), mode, ds)
            d = _nearest_distance_jztree(pts, pos, box, tree, rad)
            sel = np.nonzero(d <= rad)[0]
            n_blk = n_rows
            t2 = t1
        else:
            # The prefilter on the code coordinates (no conversion of the whole block; one more grid cell of dilation
            # for their rounding and for coordinates just outside the box) and the largest radius of the block (the
            # radius grows with the volume; 1e-9 of margin): only the cells that survive it are converted and queried.
            r_max = float(selection_radius(np.cbrt(vol.max()), mode, ds)) * (1.0 + 1e-9)
            cand = occ.candidates(base["Coordinates"], r_max, scale=hdr.code_length_cMpc, margin=1)
            idx = np.arange(n_rows) if cand is None else np.flatnonzero(cand)
            t2 = time.perf_counter()
            pos, rad = reader.positions_cMpc(base["Coordinates"][idx]), selection_radius(np.cbrt(vol[idx]), mode, ds)
            sel = idx[_nearest_distance_cpu(tree, pos, rad, workers) <= rad] if idx.size else idx
            n_blk = idx.size
        n_cand += n_blk
        t3 = time.perf_counter()
        tm["prefilter"] += t2 - t1
        tm["query"] += t3 - t2
        if sel.size:
            parts.append(read_selected(reader, fi, r0, sel, base, rest))
            n_sel += sel.size
        n_blocks += 1
        t = time.perf_counter()
        tm["read_selected"] += t - t3
        logger.debug(
            "select_cells_near_paths: file %d rows %d-%d: %d candidates, %d selected", fi, r0, r0 + n_rows, n_blk, sel.size
        )
    if n_sel == 0:
        raise RuntimeError(
            f"no gas cell is within reach of the paths: none of the {hdr.n_gas} cells of the snapshot lies within "
            f"its selection radius of a path point (box = {box:.4g} cMpc, path spacing ds = {ds:.4g} cMpc). Are the "
            "paths in comoving Mpc without h (SkewerJAX's unit; the snapshot's code unit is ckpc/h)?"
        )
    out = GasCells.concatenate(parts)
    info = dict(
        n_gas=hdr.n_gas,
        n_blocks=n_blocks,
        n_candidates=n_cand,
        n_selected=n_sel,
        ds_cMpc=ds,
        backend=backend,
        timings=tm,
        time_s=time.perf_counter() - t0,
    )
    logger.debug(
        "select_cells_near_paths: %d of %d cells selected (%d after the prefilter), %d blocks, %.2f s (%s)",
        n_sel,
        hdr.n_gas,
        n_cand,
        n_blocks,
        info["time_s"],
        ", ".join(f"{k} {v:.2f} s" for k, v in tm.items()),
    )
    return (out, info) if return_info else out
