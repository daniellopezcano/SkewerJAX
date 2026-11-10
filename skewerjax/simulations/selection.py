"""Cell selection (layer 4): streaming k-d tree selection of the gas cells of a snapshot that can reach a set of paths.

The selection rule and its radii (``"voronoi"``: 2.5 V^(1/3) + ds/2; ``"tophat"``: V^(1/3) + ds/2) are
``core.selection``'s, the same for every source of cells; cells already in memory
(a toy model, ``snapshot_cells``) go through ``core.select_cells``. This module applies the same rule to a snapshot
streamed from disk, block by block: a periodic k-d tree is built on the *path points* (few: N_sk x N_pts) and each block
of cells streamed by the reader (layer 1) is queried against it. Memory is bounded by one block of cells (``max_rows``)
plus the tree. Skewer batches are independent, so multi-device runs can distribute them; each worker streams the same
files.
"""

from __future__ import annotations

import logging
import os
import time

import numpy as np
from scipy.spatial import cKDTree

from skewerjax.core.selection import (
    VORONOI_FACTOR,
    _nearest_distance_cpu,
    path_tree,
    selection_radius,
)
from skewerjax.simulations.gas_cells import GasCells

GEOMETRY_DATASETS = ("Coordinates", "Masses", "Density")
PREFILTER_GRID = 256  # cells per box side of the grid of distance bounds of the paths (_PathDistanceGrid; 64 MB)
logger = logging.getLogger(__name__)


class _PathDistanceGrid:
    """Lower bounds of the distance to the nearest path point on a periodic grid of ``n``^3 cells (width h = box / n):
    for grid cell q with centre c_q, every point x of it has d(x) >= d(c_q) - h sqrt(3) / 2 (the triangle inequality,
    d(c_q) from one k-d tree query per centre). A gas cell is a candidate when the bound of its grid cell is within its
    own selection radius: a conservative prefilter of the k-d tree query (the cells it rejects are beyond their radius),
    with each cell's radius instead of the largest of its block (the voids' radius would make the dilation of a coarse
    occupancy grid cover the box as soon as there are tens of skewers)."""

    def __init__(self, tree: cKDTree, pts: np.ndarray, box: float, r_cap: float, n: int = PREFILTER_GRID, workers: int = -1):
        from scipy.ndimage import maximum_filter

        self.n, self.h = int(n), float(box) / int(n)
        slack = 0.5 * np.sqrt(3.0) * self.h + 1e-9 * box  # the half diagonal, and the rounding of x / h at the faces
        # The bounds are exact up to r_cap; a cell whose radius reaches beyond r_cap is always a candidate. Only the grid
        # cells within r_cap + slack of a path point are queried (the occupancy of the points, dilated periodically), with
        # that distance bound: a nearest-neighbour search far from every path point is the slow one.
        self.r_cap = float(r_cap)
        occ = np.zeros((self.n,) * 3, dtype=np.uint8)
        i = np.floor(pts / self.h).astype(np.int64) % self.n
        occ[i[:, 0], i[:, 1], i[:, 2]] = 1
        k = int(np.ceil((self.r_cap + slack) / self.h)) + 1
        near = np.flatnonzero(maximum_filter(occ, size=2 * k + 1, mode="wrap") if 2 * k + 1 < self.n else np.ones_like(occ))
        q = np.stack(np.unravel_index(near, occ.shape), axis=-1)
        d = tree.query((q + 0.5) * self.h, k=1, distance_upper_bound=self.r_cap + slack, workers=workers)[0]
        self.lower = np.full(occ.size, np.inf, dtype=np.float32)
        self.lower[near] = np.maximum(d - slack, 0.0)  # inf beyond r_cap

    def volume_thresholds(self, mode: str, ds: float, code_length: float) -> np.ndarray:
        """(n^3,) the smallest code volume V = m / rho a cell of each grid cell needs to be a candidate: its radius
        k V^(1/3) l + ds/2 (k: ``selection_radius``, l: the code length [cMpc]) must reach min(bound, r_cap), so
        V >= ((min(bound, r_cap) - ds/2) / (k l))^3 (0 when the bound is within ds/2). Lowered by 1e-9 (rounding)."""
        k = VORONOI_FACTOR if mode == "voronoi" else 1.0
        r = np.minimum(self.lower.astype(np.float64), self.r_cap) - 0.5 * ds
        return np.where(r > 0.0, (np.maximum(r, 0.0) / (k * code_length)) ** 3 * (1.0 - 1e-9), 0.0)

    def candidates(self, pos: np.ndarray, rad: np.ndarray, *, scale: float = 1.0) -> np.ndarray:
        """Mask of the positions (N, 3) (in [0, box) up to rounding; ``scale``: given in another unit, x = scale pos, e.g.
        code coordinates) whose nearest path point can be within ``rad`` (N,)."""
        q = (pos * (scale / self.h)).astype(np.int64)  # positions >= 0: truncation is the floor
        q %= self.n
        return (self.lower.take((q[:, 0] * self.n + q[:, 1]) * self.n + q[:, 2]) <= rad) | (rad > self.r_cap)


def _prefilter_kernel():
    """The fused prefilter (numba, no GIL, one block per thread): mask_i = m_i / rho_i >= V_thr[grid cell of x_i]."""
    import numba

    @numba.njit(nogil=True, cache=True)
    def run(coords, masses, density, inv_h, n, vthr, out):
        for i in range(coords.shape[0]):
            q0 = int(coords[i, 0] * inv_h) % n
            q1 = int(coords[i, 1] * inv_h) % n
            q2 = int(coords[i, 2] * inv_h) % n
            out[i] = np.float64(masses[i]) >= vthr[(q0 * n + q1) * n + q2] * np.float64(density[i])
        return out

    return run


_PREFILTER = {}


def _prefilter(coords, masses, density, inv_h: float, n: int, vthr) -> np.ndarray:
    """Mask of the candidates of one block (``_prefilter_kernel``, compiled once per process)."""
    run = _PREFILTER.setdefault("numba", _prefilter_kernel())
    return run(coords, masses, density, float(inv_h), int(n), vthr, np.empty(coords.shape[0], dtype=np.bool_))


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
    return_info: bool = False,
) -> GasCells:
    """Stream the snapshot and keep the cells that can contribute to any of ``paths``.

    Parameters
    ----------
    reader : layer-1 reader (``iter_blocks``, ``read_rows``, ``to_gas_cells``, ``header``; ``memmaps`` for the parallel
        reads).
    paths : (N_sk, N_pts, 3) sampling points [cMpc], e.g. ``SkewerBundle.spatial_points``.
    mode : "voronoi" (default deposition) or "tophat" (fake_spectra's kernel); "voronoi" selects a superset of
        "tophat".
    max_rows : cells per streamed block (memory bound: ~100 bytes per cell per thread).
    workers : threads (-1: all cores). The blocks of every file are processed in parallel, each by one thread (read
        through ``reader.memmaps``, prefilter, k-d tree query, read of the selected rows); a reader without memory maps,
        or a dataset that is not contiguous, falls back to h5py reads (serialised by h5py's lock).
    return_info : also return dict(n_gas, n_blocks, n_candidates (cells left by the prefilter), n_selected, ds_cMpc,
        threads, timings {read, prefilter, query, read_selected} [s] (each stage's share of the wall time: the
        stages of different blocks overlap), thread_seconds (the same, summed over the threads), time_s).

    Returns
    -------
    GasCells of the selected cells, physical units (``reader.to_gas_cells``), in the order of the snapshot (file, row),
    with ``ids`` = (file_index << 40) | row, so that every cell can be traced back to the snapshot (and the info).
    """
    t0 = time.perf_counter()
    from skewerjax.simulations.readers import GAS_DATASETS

    rest = [k for k in GAS_DATASETS if k not in GEOMETRY_DATASETS]
    hdr = reader.header
    box = hdr.box_cMpc
    pts, tree, ds = path_tree(paths, box)
    threads = (os.cpu_count() or 1) if workers == -1 else max(int(workers), 1)
    tm = dict(read=0.0, prefilter=0.0, query=0.0, read_selected=0.0)
    t = time.perf_counter()
    # r_cap: ten mean cell spacings, far beyond the radius of all but the largest void cells
    grid = _PathDistanceGrid(tree, pts, box, r_cap=10.0 * box / max(hdr.n_gas, 1) ** (1.0 / 3.0), workers=workers)
    grid.vthr = grid.volume_thresholds(mode, ds, hdr.code_length_cMpc)
    tm["prefilter"] += time.perf_counter() - t
    tasks = [(fi, r0, min(n, r0 + max_rows)) for fi in range(hdr.n_files) for n in [reader.n_rows(fi)] for r0 in range(0, n, max_rows)]
    job = lambda task: _select_block(reader, task, grid, tree, ds, mode, rest)
    if threads > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=threads) as pool:
            results = list(pool.map(job, tasks))
    else:
        results = [job(task) for task in tasks]
    parts = [r[0] for r in results if r[0] is not None]
    n_cand, n_blocks = sum(r[1] for r in results), len(tasks)
    for r in results:
        for k, v in r[2].items():
            tm[k] += v
    n_sel = sum(len(p) for p in parts)
    if n_sel == 0:
        raise RuntimeError(
            f"no gas cell is within reach of the paths: none of the {hdr.n_gas} cells of the snapshot lies within "
            f"its selection radius of a path point (box = {box:.4g} cMpc, path spacing ds = {ds:.4g} cMpc). Are the "
            "paths in comoving Mpc without h (SkewerJAX's unit; the snapshot's code unit is ckpc/h)?"
        )
    out = GasCells.concatenate(parts)
    wall = time.perf_counter() - t0
    total = sum(tm.values())
    info = dict(
        n_gas=hdr.n_gas,
        n_blocks=n_blocks,
        n_candidates=n_cand,
        n_selected=n_sel,
        ds_cMpc=ds,
        threads=threads,
        timings={k: wall * v / total if total > 0 else 0.0 for k, v in tm.items()},
        thread_seconds=dict(tm),
        time_s=wall,
    )
    logger.debug(
        "select_cells_near_paths: %d of %d cells selected (%d after the prefilter), %d blocks, %d threads, %.2f s (%s)",
        n_sel,
        hdr.n_gas,
        n_cand,
        n_blocks,
        info["threads"],
        wall,
        ", ".join(f"{k} {v:.2f} s" for k, v in info["timings"].items()),
    )
    return (out, info) if return_info else out


def _block(reader, fi: int, datasets, r0: int, r1: int, rows=None) -> dict:
    """Datasets of rows [r0, r1) of file ``fi`` (or of the absolute ``rows``): from the memory maps when the reader has
    them, else with h5py."""
    maps = reader.memmaps(fi, datasets) if hasattr(reader, "memmaps") else dict.fromkeys(datasets)
    out = {}
    slow = [k for k in datasets if maps[k] is None]
    for k in datasets:
        if maps[k] is not None:
            out[k] = np.array(maps[k][r0:r1]) if rows is None else maps[k][rows]  # copies: the pages are read here
    if slow:
        lo, hi = (r0, r1) if rows is None else (int(rows[0]), int(rows[-1]) + 1)
        slab = reader.read_rows(fi, slow, start=lo, stop=hi)
        for k, v in slab.items():
            out[k] = v if rows is None else v[rows - lo]
    return out


def _select_block(reader, task, grid: _PathDistanceGrid, tree, ds: float, mode: str, rest) -> tuple:
    """One block (file, rows [r0, r1)): (GasCells of its selected cells or None, number of candidates, timings)."""
    fi, r0, r1 = task
    hdr = reader.header
    tm = {}
    t = time.perf_counter()
    base = _block(reader, fi, GEOMETRY_DATASETS, r0, r1)  # into memory: the I/O of the block
    t1 = time.perf_counter()
    mask = _prefilter(base["Coordinates"], base["Masses"], base["Density"], hdr.code_length_cMpc / grid.h, grid.n, grid.vthr)
    idx = np.flatnonzero(mask)
    t2 = time.perf_counter()
    if idx.size:  # the exact rule on the candidates only
        rad = selection_radius(np.cbrt(reader.volumes_cMpc3(base["Masses"][idx], base["Density"][idx])), mode, ds)
        pos = reader.positions_cMpc(base["Coordinates"][idx])
        sel = idx[_nearest_distance_cpu(tree, pos, rad, 1) <= rad]
    else:
        sel = idx
    t3 = time.perf_counter()
    cells = None
    if sel.size:
        raw = {k: v[sel] for k, v in base.items()}
        raw.update(_block(reader, fi, rest, r0, r1, rows=r0 + sel))
        cells = reader.to_gas_cells(raw, ids=(np.int64(fi) << 40) | (r0 + sel).astype(np.int64))
    t4 = time.perf_counter()
    tm = dict(read=t1 - t, prefilter=t2 - t1, query=t3 - t2, read_selected=t4 - t3)
    logger.debug("select_cells_near_paths: file %d rows %d-%d: %d candidates, %d selected", fi, r0, r1, idx.size, sel.size)
    return cells, int(idx.size), tm
