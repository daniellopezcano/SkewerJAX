"""The streaming route: the optical depth of many straight skewers from one read of the snapshot, tile by tile
(``docs/performance.md``, section D1).

**One pass over the files.** The snapshot is read once, in file order, block by block on parallel threads. For each
block:

1. the geometry (``Coordinates``, ``Masses``, ``Density``: the generating points and the cell sizes h = V^(1/3)) is read
   whole, and the cells within reach R = 2.5 h of some skewer are found by index arithmetic on the skewers' plane
   (``core.crossings.line_pairs``: the skewers are bucketed, a cell at (x, z) looks only at the buckets of
   [x - R, x + R] x [z - R, z + R]; no tree);
2. the other datasets are read for those rows only, as one contiguous read when they touch most pages of the block, or
   as an indexed gather otherwise (``GATHER_FRACTION``);
3. the gas physics runs on those cells only, once per physics configuration (default, matched), and the block leaves
   a compact record per needed cell (position, size, per species n and T, peculiar velocity) and the list of the tiles
   that need it.

**Tiles.** The skewers are grouped into tiles of about ``tile_size`` neighbouring skewers. For each tile, the cells of
the tile are gathered from the compact store, their crossings with the tile's skewers built (``core.crossings``), turned
into contributions (``core.absorbers.crossing_contributions``) and summed by the windowed engine
(``core.tau_engine``) on the device. The host prepares tile k + 1 while the device computes tile k.

**Memory.** The peak is the compact store of the needed cells (48 bytes per cell, 16 more per physics configuration
for two species, 8 per (cell, tile) pair: 24 GB at 128^2 TNG50-2 skewers with both configurations), the blocks being
read (``max_rows`` per thread), the tiles in preparation and the result.
Nothing outlives the run.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from skewerjax.core import crossings as cx
from skewerjax.core.cells import Cells
from skewerjax.core.selection import VORONOI_FACTOR
from skewerjax.simulations import host_physics

logger = logging.getLogger(__name__)

GEOMETRY = ("Coordinates", "Masses", "Density")
REST = ("Velocities", "InternalEnergy", "ElectronAbundance", "NeutralHydrogenAbundance", "GFM_Metals", "StarFormationRate")
PHYSICS_THREADS = 8  # threads in the gas physics at once (``_block``)
_PHYSICS_SLOTS = threading.BoundedSemaphore(PHYSICS_THREADS)
GATHER_FRACTION = 0.5  # read a dataset of a block whole when the needed rows touch more than this fraction of its pages
PAGE = 4096


# --------------------------------------------------------------------------- reading
class _Layout:
    """Offsets, dtypes and shapes of the gas datasets of every file (contiguous, uncompressed: TNG's layout)."""

    def __init__(self, reader, names):
        import h5py

        self.files = [str(f) for f in reader.files]
        self.info = []
        for f in reader.files:
            with h5py.File(f, "r") as h:
                g = h[reader.part_type] if reader.part_type in h else None
                d = {}
                for k in names:
                    if g is None or k not in g:
                        d[k] = None
                        continue
                    ds = g[k]
                    off = ds.id.get_offset() if ds.chunks is None and ds.compression is None and ds.shape[0] > 0 else None
                    d[k] = (off, ds.dtype, ds.shape)
                self.info.append(d)

    def n_rows(self, fi: int) -> int:
        d = self.info[fi][GEOMETRY[0]]
        return 0 if d is None else int(d[2][0])

    def read(self, reader, fi: int, name: str, r0: int, r1: int, rows=None):
        """Rows [r0, r1) of a dataset (or the absolute ``rows`` of it): one contiguous read (``np.fromfile``), or an
        indexed gather through a memory map of the block's range when the rows touch few pages; h5py when the dataset
        is not contiguous."""
        d = self.info[fi][name]
        if d is None or d[0] is None:
            lo, hi = (r0, r1) if rows is None else (int(rows[0]), int(rows[-1]) + 1)
            x = reader.read_rows(fi, [name], start=lo, stop=hi)[name]
            return x if rows is None else x[rows - lo]
        off, dt, shape = d
        row_bytes = int(np.prod(shape[1:], dtype=np.int64)) * dt.itemsize
        if rows is None:
            n = r1 - r0
            x = np.fromfile(self.files[fi], dtype=dt, count=n * (row_bytes // dt.itemsize), offset=off + r0 * row_bytes)
            return x.reshape((n,) + tuple(shape[1:]))
        lo, hi = int(rows[0]), int(rows[-1]) + 1
        pages = np.unique((rows - lo) * row_bytes // PAGE)
        span_pages = (hi - lo) * row_bytes // PAGE + 1
        if pages.size > GATHER_FRACTION * span_pages:  # most pages: one contiguous read of the span
            x = np.fromfile(self.files[fi], dtype=dt, count=(hi - lo) * (row_bytes // dt.itemsize), offset=off + lo * row_bytes)
            return x.reshape((hi - lo,) + tuple(shape[1:]))[rows - lo]
        mm = np.memmap(self.files[fi], dtype=dt, mode="r", offset=off + lo * row_bytes, shape=(hi - lo,) + tuple(shape[1:]))
        out = np.array(mm[rows - lo])
        del mm
        return out


# --------------------------------------------------------------------------- the compact store
@dataclass
class _Chunk:
    pos: np.ndarray  # (m, 3) float64 [cMpc]
    size: np.ndarray  # (m,) float32 [cMpc]
    vel: np.ndarray  # (m, 3) float32 [km/s]
    fields: dict  # {config: {species: (n float32, T float32)}}
    pair_cell: np.ndarray  # (p,) int64 local cell index
    pair_tile: np.ndarray  # (p,) int32
    ids: np.ndarray  # (m,) int64 (file << 40 | row)
    n_read: int = 0
    timings: dict = field(default_factory=dict)


@dataclass
class StreamStore:
    """The needed cells of a run (host arrays, in file order): positions [cMpc], sizes V^(1/3) [cMpc], peculiar
    velocities [km/s], per physics configuration and species the comoving density n [cm^-3] and the temperature T [K],
    the snapshot ids; ``tile_start`` / ``tile_cells`` the cells of every tile (CSR)."""

    pos: np.ndarray
    size: np.ndarray
    vel: np.ndarray
    fields: dict
    ids: np.ndarray
    tile_start: np.ndarray
    tile_cells: np.ndarray
    box: float
    a: float

    @property
    def n_cells(self) -> int:
        return int(self.size.size)

    @property
    def nbytes(self) -> int:
        n = self.pos.nbytes + self.size.nbytes + self.vel.nbytes + self.ids.nbytes + self.tile_cells.nbytes
        return int(n + sum(a.nbytes + b.nbytes for f in self.fields.values() for a, b in f.values()))

    def cells(self, idx, config: str) -> Cells:
        """The cells ``idx`` as ``core.Cells`` with the species of one physics configuration."""
        sp = {
            s: dict(
                number_density=n[idx].astype(np.float64),
                temperature=T[idx].astype(np.float64),
                velocity=self.vel[idx].astype(np.float64),
            )
            for s, (n, T) in self.fields[config].items()
        }
        return Cells(self.pos[idx], self.size[idx].astype(np.float64) ** 3, sp, box=self.box, ids=self.ids[idx], validate=False)


# --------------------------------------------------------------------------- tiles
def tile_skewers(paths: cx.StraightPaths, tile_size: int) -> np.ndarray:
    """(N_sk,) the tile of every axis-aligned skewer: squares of the skewers' plane holding about ``tile_size`` skewers
    each (a G x G grid over the box, G = ceil(sqrt(N_sk / tile_size)))."""
    n = len(paths)
    G = max(1, int(np.ceil(np.sqrt(n / max(int(tile_size), 1)))))
    ax = paths.axis
    a0, a1 = [k for k in range(3) if k != ax]
    box = paths.box
    c0, c1 = paths.origin[:, a0], paths.origin[:, a1]
    if box:
        c0, c1 = np.mod(c0, box), np.mod(c1, box)
        lo0 = lo1 = 0.0
        w0 = w1 = box / G
    else:
        lo0, lo1 = c0.min(), c1.min()
        w0, w1 = max(np.ptp(c0), 1e-12) / G * (1 + 1e-9), max(np.ptp(c1), 1e-12) / G * (1 + 1e-9)
    t = np.clip(((c0 - lo0) // w0).astype(np.int64), 0, G - 1) * G + np.clip(((c1 - lo1) // w1).astype(np.int64), 0, G - 1)
    _, inv = np.unique(t, return_inverse=True)  # consecutive tile ids
    return np.ravel(inv).astype(np.int32)


# --------------------------------------------------------------------------- memory
MMAP_THRESHOLD_PASS = 1 << 20  # bytes: during the pass, arrays above this are mapped and unmapped when freed
MMAP_THRESHOLD_TILES = 32 << 20  # after it: glibc's own steady state (its dynamic threshold's maximum)


def release_memory(threshold: int | None = None) -> None:
    """Give freed memory back to the system (glibc; nothing elsewhere), and set the size above which arrays are mapped.

    The pass allocates and frees many arrays of a few MB on 32 threads. glibc's default raises its mmap threshold to the
    largest freed block (up to 32 MB), after which such arrays come from per-thread heaps that keep their freed pages:
    19 GB of a 128^2 TNG50-2 pass stayed resident after its 24 GB store was built (peak 48 GB). With the threshold fixed
    at ``MMAP_THRESHOLD_PASS`` during the pass, every array above 1 MB is unmapped when freed: peak 28 GB, the store
    alone afterwards, the pass no slower. The tiles then allocate many short-lived arrays of 1-30 MB per chunk, where
    mapping each one costs page faults (the matched engine 1.9x slower): ``stream_cells`` sets ``MMAP_THRESHOLD_TILES``
    when the pass ends. ``threshold`` None only trims (``malloc_trim``). The setting holds for the rest of the process."""
    import ctypes
    import sys

    if not sys.platform.startswith("linux"):
        return
    try:
        libc = ctypes.CDLL("libc.so.6")
        if threshold is not None:
            libc.mallopt(-3, int(threshold))  # M_MMAP_THRESHOLD
        libc.malloc_trim(0)
    except (OSError, AttributeError):  # not glibc
        pass


# --------------------------------------------------------------------------- the pass over the snapshot
def _block(reader, layout, fi, r0, r1, paths, buckets, tile_of, species, physics, need_vel3):
    """One block of one file: (``_Chunk`` of its needed cells, or None)."""
    from skewerjax.simulations.snapshot_cells import gas_to_cells

    H = reader.header
    tm = {}
    t0 = time.perf_counter()
    geo = {k: layout.read(reader, fi, k, r0, r1) for k in GEOMETRY}
    t1 = time.perf_counter()
    pos = reader.positions_cMpc(geo["Coordinates"])
    size = np.cbrt(reader.volumes_cMpc3(geo["Masses"], geo["Density"]))
    reach = VORONOI_FACTOR * size
    counts = buckets.count(pos, reach)
    need = np.flatnonzero(counts > 0)
    t2 = time.perf_counter()
    tm.update(read=t1 - t0, reach=t2 - t1)
    if not need.size:
        tm["read_rest"] = 0.0
        tm.update({f"physics:{name}": 0.0 for name in physics})
        return None, tm, 0
    # the tiles of every needed cell
    pc, ps = buckets.pairs(pos[need], reach[need])
    key = np.unique(pc.astype(np.int64) * (int(tile_of.max()) + 1) + tile_of[ps])
    pair_cell = key // (int(tile_of.max()) + 1)
    pair_tile = (key % (int(tile_of.max()) + 1)).astype(np.int32)
    t3 = time.perf_counter()
    rows = r0 + need
    raw = {k: geo[k][need] for k in GEOMETRY}
    for k in REST:
        raw[k] = layout.read(reader, fi, k, r0, r1, rows=rows)
    t4 = time.perf_counter()
    fields = {}
    fast = all(host_physics.supported(species, **opts) for opts in physics.values())
    if fast:  # the fused loops (``host_physics``): no GIL, the threads scale
        vel = (np.asarray(raw["Velocities"], np.float64) * np.sqrt(H.a)).astype(np.float32)  # as to_gas_cells
        ids = (np.int64(fi) << 40) | rows.astype(np.int64)
        for name, opts in physics.items():
            tp = time.perf_counter()
            fields[name] = host_physics.species_fields(raw, H, species, **opts)
            tm[f"physics:{name}"] = time.perf_counter() - tp
    else:
        with _PHYSICS_SLOTS:  # a few threads at a time in the NumPy gas physics (memory-bound, the GIL): the others keep reading
            t4 = time.perf_counter()
            gas = reader.to_gas_cells(raw, ids=(np.int64(fi) << 40) | rows.astype(np.int64))
            ids = np.asarray(gas.ids)
            vel = None
            conv = time.perf_counter() - t4
            for name, opts in physics.items():
                tp = time.perf_counter()
                c = gas_to_cells(gas, list(species), **opts)
                tm[f"physics:{name}"] = time.perf_counter() - tp + conv / len(physics)
                fields[name] = {
                    sp: (c.species[sp]["number_density"].astype(np.float32), c.species[sp]["temperature"].astype(np.float32))
                    for sp in species
                }
                if vel is None:
                    vel = c.species[species[0]]["velocity"].astype(np.float32)
    tm.update(tiles=t3 - t2, read_rest=t4 - t3)
    ch = _Chunk(pos[need], size[need].astype(np.float32), vel, fields, pair_cell, pair_tile, ids, timings=tm)
    return ch, tm, int(need.size)


class _Buckets:
    """The skewers bucketed on their plane, for the reach test of the cells (``core.crossings``' axis kernel)."""

    def __init__(self, paths: cx.StraightPaths):
        K = cx._kernels()
        self.K = K
        self.paths = paths
        ax = paths.axis
        self.ax = ax
        self.ax0, self.ax1 = [k for k in range(3) if k != ax]
        self.box = float(paths.box) if paths.box else 0.0
        c0, c1 = paths.origin[:, self.ax0].copy(), paths.origin[:, self.ax1].copy()
        if self.box:
            c0, c1 = np.mod(c0, self.box), np.mod(c1, self.box)
        (self.bstart, self.bcount, self.bitems, self.nb0, self.nb1, self.g0, self.g1, self.lo0, self.lo1) = cx._bucket_paths(
            c0, c1, self.box, len(paths), np.ptp(c0), np.ptp(c1)
        )
        self.c0, self.c1 = c0 - self.lo0, c1 - self.lo1
        self.sk_o = paths.origin[:, ax].astype(np.float64)
        self.sk_sign = paths.direction[:, ax].astype(np.float64)
        self.lens = paths.length.astype(np.float64)

    def _args(self):
        return (
            self.ax0,
            self.ax1,
            self.ax,
            self.box,
            self.bstart,
            self.bcount,
            self.bitems,
            self.c0,
            self.c1,
            self.sk_o,
            self.sk_sign,
            self.lens,
            self.nb0,
            self.nb1,
            self.g0,
            self.g1,
        )

    def _shift(self, pos):
        if self.lo0 == 0.0 and self.lo1 == 0.0:
            return pos
        p = pos.copy()
        p[:, self.ax0] -= self.lo0
        p[:, self.ax1] -= self.lo1
        return p

    def count(self, pos, reach):
        counts = np.zeros(pos.shape[0], np.int64)
        di, df = np.zeros(1, np.int64), np.zeros(1)
        self.K["axis_pairs"](
            self._shift(pos), np.ascontiguousarray(reach, np.float64), *self._args(), di, di, di, df, df, True, counts
        )
        return counts

    def pairs(self, pos, reach):
        p = self._shift(pos)
        R = np.ascontiguousarray(reach, np.float64)
        counts = self.count(pos, reach)
        offs = np.concatenate([[0], np.cumsum(counts)])
        n = int(offs[-1])
        cell, sk = np.empty(n, np.int64), np.empty(n, np.int64)
        u, r2 = np.empty(n), np.empty(n)
        self.K["axis_pairs"](p, R, *self._args(), offs[:-1].copy(), cell, sk, u, r2, False, np.empty(p.shape[0], np.int64))
        return cell, sk


def stream_cells(
    reader, paths: cx.StraightPaths, tile_of, species, physics: dict, *, max_rows: int = 2_000_000, threads: int | None = None
):
    """The pass over the snapshot (module docstring): (``StreamStore``, info). ``physics``: {name: ``gas_to_cells``
    options}; ``species`` the species keys of the lines."""
    if paths.axis is None:
        raise NotImplementedError("the streaming route takes axis-aligned skewers (any other straight paths: strategy='gather').")
    t0 = time.perf_counter()
    release_memory(MMAP_THRESHOLD_PASS)
    layout = _Layout(reader, GEOMETRY + REST)
    buckets = _Buckets(paths)
    tasks = [
        (fi, r0, min(layout.n_rows(fi), r0 + max_rows))
        for fi in range(len(layout.files))
        for r0 in range(0, layout.n_rows(fi), max_rows)
    ]
    n_thr = (os.cpu_count() or 1) if threads is None else max(1, int(threads))
    species = list(species)
    tile_of = np.asarray(tile_of, np.int32)

    def job(task):
        return _block(reader, layout, *task, paths, buckets, tile_of, species, physics, True)

    t1 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n_thr) as pool:
        results = list(pool.map(job, tasks))
    t2 = time.perf_counter()
    chunks = [r[0] for r in results if r[0] is not None]
    tm = {}
    for r in results:
        for k, v in r[1].items():
            tm[k] = tm.get(k, 0.0) + v
    del results  # the chunks are freed one by one as they are copied into the store
    sizes = np.array([c.size.size for c in chunks], np.int64)
    base = np.concatenate([[0], np.cumsum(sizes)])
    n_c = int(base[-1])
    psz = np.array([c.pair_cell.size for c in chunks], np.int64)
    pbase = np.concatenate([[0], np.cumsum(psz)])
    # the store, preallocated and filled by parallel copies (one chunk per task: numpy's copies release the GIL)
    pos = np.empty((n_c, 3), np.float64)
    size = np.empty(n_c, np.float32)
    vel = np.empty((n_c, 3), np.float32)
    ids = np.empty(n_c, np.int64)
    fields = {name: {sp: (np.empty(n_c, np.float32), np.empty(n_c, np.float32)) for sp in species} for name in physics}
    pc = np.empty(int(pbase[-1]), np.int64)
    pt = np.empty(int(pbase[-1]), np.int32)

    def put(i):
        c, a, b = chunks[i], base[i], base[i + 1]
        pos[a:b], size[a:b], vel[a:b], ids[a:b] = c.pos, c.size, c.vel, c.ids
        for name in physics:
            for sp in species:
                fields[name][sp][0][a:b], fields[name][sp][1][a:b] = c.fields[name][sp]
        pc[pbase[i] : pbase[i + 1]] = c.pair_cell + a
        pt[pbase[i] : pbase[i + 1]] = c.pair_tile
        chunks[i] = None

    with ThreadPoolExecutor(max_workers=n_thr) as pool:
        list(pool.map(put, range(len(chunks))))
    chunks.clear()
    n_tiles = int(tile_of.max()) + 1
    order, tile_start = cx.counting_order(pt, n_tiles)
    tile_cells = pc[order]
    t3 = time.perf_counter()
    store = StreamStore(
        pos, size, vel, fields, ids, tile_start, tile_cells, float(reader.header.box_cMpc), float(reader.header.a)
    )
    pc = pt = order = None  # freed (``put`` holds them in its closure: rebound, not deleted)
    # the blocks' and chunks' memory back to the system; the tiles' allocations from the heap
    release_memory(MMAP_THRESHOLD_TILES)
    wall = t2 - t1
    tot = sum(tm.values()) or 1.0
    info = dict(
        n_blocks=len(tasks),
        n_cells=int(reader.header.n_gas),
        n_needed=store.n_cells,
        fraction=store.n_cells / max(int(reader.header.n_gas), 1),
        threads=n_thr,
        thread_seconds=tm,
        timings={k: wall * v / tot for k, v in tm.items()},  # each stage's share of the pass's wall time
        setup_s=t1 - t0,
        pass_s=wall,
        assemble_s=t3 - t2,
        store_GB=store.nbytes / 1e9,
    )
    return store, info


# --------------------------------------------------------------------------- the optical depth
def stream_optical_depth(
    reader,
    paths: cx.StraightPaths,
    frame,
    v,
    line_bundle,
    *,
    physics: dict | None = None,
    tile_size: int = 1024,
    max_rows: int = 2_000_000,
    threads: int | None = None,
    profile: str = "integrated",
    detuning: str = "exact",
    voigt: str = "harris",
    wing_tolerance: float | None = None,
    a_eff=None,
    b_constants: str = "codata",
    hubble_flow: str = "exact",
    aH_ref=None,
    velocity_period=None,
    dtype=None,
    device=None,
    out=None,
    return_info: bool = False,
):
    """tau (N_lines, N_sk, N_v) of the axis-aligned skewers ``paths`` through the snapshot of ``reader`` (module
    docstring): one pass over the files, then the tiles.

    paths : ``core.crossings.StraightPaths`` (axis-aligned; e.g. ``paths_from_points`` of a ``SkewerBundle``'s points);
    frame : the frame of the skewers (``core.absorbers.bundle_frame`` of the bundle: a(s), a_dagger, photon directions);
    v : (N_v,) uniform velocity grid [km/s]; physics : the ``gas_to_cells`` options (default: SkewerJAX's);
    tile_size : skewers per tile; max_rows : cells per block of the pass; threads : of the pass (default all cores);
    the tau options (``profile``, ``detuning``, ``voigt``, ``wing_tolerance``, ``a_eff``, ``b_constants``,
    ``hubble_flow``, ``aH_ref``, ``velocity_period``) as ``core.compute_optical_depth``; device : of the engine;
    out : optional (N_lines, N_sk, N_v) host array to fill (of the working dtype: float32 by default).
    """

    t0 = time.perf_counter()
    physics = {"default": {}} if physics is None else physics
    if len(physics) != 1:
        raise ValueError("stream_optical_depth takes one physics configuration (stream_cells accepts several).")
    cfg = next(iter(physics))
    species = list(line_bundle.species_keys)
    tile_of = tile_skewers(paths, tile_size)
    store, pinfo = stream_cells(reader, paths, tile_of, species, physics, max_rows=max_rows, threads=threads)
    t1 = time.perf_counter()
    taus, tinfo = tiles_optical_depth(
        store,
        cfg,
        paths,
        tile_of,
        frame,
        v,
        line_bundle,
        profile=profile,
        detuning=detuning,
        voigt=voigt,
        wing_tolerance=wing_tolerance,
        a_eff=a_eff,
        b_constants=b_constants,
        hubble_flow=hubble_flow,
        aH_ref=aH_ref,
        velocity_period=velocity_period,
        dtype=dtype,
        device=device,
    )
    taus = taus["tau"]
    if out is not None:
        out[...] = taus
        taus = out
    info = dict(pass_info=pinfo, tiles=tinfo, timings=dict(pass_s=t1 - t0, tiles_s=time.perf_counter() - t1))
    return (taus, info) if return_info else taus


def tiles_optical_depth(
    store: StreamStore,
    config: str,
    paths,
    tile_of,
    frame,
    v,
    line_bundle,
    *,
    pieces: str = "crossings",
    chord_weight: float | None = None,
    profile="integrated",
    detuning="exact",
    voigt="harris",
    wing_tolerance=None,
    a_eff=None,
    b_constants="codata",
    hubble_flow="exact",
    aH_ref=None,
    velocity_period=None,
    tautail=None,
    pixel_average=False,
    dtype=None,
    device=None,
    engines: dict | None = None,
    pipeline: bool = True,
    prep_workers: int = 3,
):
    """tau of every tile from the compact store (module docstring), the host preparing tile k + 1 while the devices
    compute tile k.

    config : the physics configuration of the store to use; pieces : "crossings" (the Voronoi cell crossings, the
    default) or "chords" (fake_spectra's top-hat chords, ``core.crossings.straight_chords``, with ``chord_weight``);
    the tau options as ``core.compute_optical_depth`` (``tautail`` / ``pixel_average``: the per-cell schemes);
    engines : {name: list of devices} to compute every tile on several device sets from the same host work (e.g. the
    CPU and the GPU entries of a benchmark); default {"tau": [device]}; prep_workers : host threads preparing the next
    tiles while the devices compute (``pipeline``).

    Returns ({name: tau (N_lines, N_sk, N_v) host array of the working dtype (``dtype``, or the package's precision:
    float32 by default)}, info with the per-tile stage times)."""
    from skewerjax.config import real_dtype
    from skewerjax.core.absorbers import chord_contributions, crossing_contributions
    from skewerjax.core.optical_depth import _engine

    v = np.asarray(v, dtype=np.float64)
    n_l, n_sk, n_v = line_bundle.n_lines, len(paths), v.shape[-1]
    dt = real_dtype() if dtype is None else np.dtype(dtype)
    engines = {"tau": [device]} if engines is None else engines
    taus = {name: np.zeros((n_l, n_sk, n_v), dt) for name in engines}
    n_tiles = int(store.tile_start.size - 1)
    order = np.argsort(tile_of, kind="stable")
    bounds = np.concatenate([[0], np.cumsum(np.bincount(tile_of, minlength=n_tiles))])
    members = [order[bounds[k] : bounds[k + 1]] for k in range(n_tiles)]
    vv = np.broadcast_to(v, (n_l, n_v))
    if velocity_period is True:
        raise ValueError("velocity_period=True: give the period [km/s] (the skewers' span) to the streaming route.")
    P = None if velocity_period in (None, False) else float(velocity_period)

    def prepare(k):
        t = time.perf_counter()
        sk = members[k]
        idx = store.tile_cells[store.tile_start[k] : store.tile_start[k + 1]]
        sub = paths.take(sk)
        pos, size = store.pos[idx], store.size[idx].astype(np.float64)
        if pieces == "chords":
            geo = cx.straight_chords(sub, pos, size)
        else:
            geo = cx.straight_crossings(sub, pos, size)
        t1 = time.perf_counter()
        cells = store.cells(idx, config)
        fr = sub_frame(frame, sk)
        kw = dict(a_eff=a_eff, b_constants=b_constants, hubble_flow=hubble_flow, aH_ref=aH_ref)
        if pieces == "chords":
            contr = chord_contributions(
                geo, cells, line_bundle, fr, sub.direction, sub.length, sub.box, chord_weight=chord_weight, **kw
            )
        else:
            contr = crossing_contributions(geo, cells, line_bundle, fr, sub.direction, **kw)
        t2 = time.perf_counter()
        return dict(
            k=k,
            sk=sk,
            contr=contr,
            n_pieces=int(geo.cell.size),
            n_cells=int(idx.size),
            geometry_s=t1 - t,
            contributions_s=t2 - t1,
        )

    rows = []
    ahead = max(1, int(prep_workers)) if pipeline else 0
    pool = ThreadPoolExecutor(max_workers=ahead) if pipeline else None
    futs = {k: pool.submit(prepare, k) for k in range(min(ahead, n_tiles))} if pipeline else {}
    for k in range(n_tiles):
        t_wait = time.perf_counter()
        prep = futs.pop(k).result() if pipeline else prepare(k)
        t_wait = time.perf_counter() - t_wait
        if pipeline and k + ahead < n_tiles:
            futs[k + ahead] = pool.submit(prepare, k + ahead)
        sk = prep["sk"]
        row = dict(
            tile=k,
            n_skewers=int(sk.size),
            n_cells=prep["n_cells"],
            n_pieces=prep["n_pieces"],
            geometry_s=prep["geometry_s"],
            contributions_s=prep["contributions_s"],
            wait_s=t_wait,
            tau_s={},
        )
        for name, devs in engines.items():
            t = time.perf_counter()
            tau, eng = _engine(
                prep["contr"],
                vv,
                int(sk.size),
                line_bundle,
                None if P is None else np.full(sk.size, P),
                profile,
                detuning,
                voigt,
                wing_tolerance,
                dt,
                [d for d in devs if d is not None] or None,
                tautail,
                pixel_average,
            )
            taus[name][:, sk] = np.asarray(tau, dtype=dt)
            row["tau_s"][name] = time.perf_counter() - t
            row.setdefault("engine", {})[name] = eng
        rows.append(row)
    if pool is not None:
        pool.shutdown()
    return taus, dict(n_tiles=n_tiles, rows=rows)


def sub_frame(frame, sk):
    """The frame of the skewers ``sk`` of ``frame`` (row i: skewer sk[i]); a box-mode ``ExactFrame`` (one shared row of
    s and a) keeps its single row."""
    from skewerjax.core.absorbers import ExactFrame, SnapshotFrame

    sk = np.asarray(sk)
    if isinstance(frame, SnapshotFrame):
        pl = np.asarray(frame.path_length, dtype=np.float64)
        return SnapshotFrame(frame.a_snap, frame.velfac, pl[sk] if pl.ndim else np.full(sk.size, float(pl)))
    pick = lambda x: None if x is None else np.asarray(x)[sk]
    rows = (lambda x: x) if np.shape(frame.s_grid)[0] == 1 else (lambda x: np.asarray(x)[sk])
    return ExactFrame(
        rows(frame.s_grid),
        rows(frame.a_grid),
        a_dagger_value=frame.a_dagger_value,
        photon_sign=pick(frame.photon_sign),
        a_dagger_per_skewer=pick(frame.a_dagger_per_skewer),
    )


def box_skewers(cofm, axis: int, box: float, cosmology, a_ref: float, *, frame_spacing: float = 0.05):
    """The axis-aligned skewers through the points ``cofm`` (N_sk, 3) [cMpc] across the box once along ``axis`` (1, 2, 3),
    as ``SkewerBundle.axis_aligned`` defines them, without their sampling points: (``StraightPaths``, a box-mode
    ``ExactFrame`` with one shared row of s and a(s) every ``frame_spacing`` cMpc: a(s) is smooth, so the integrated
    profile needs no finer grid)."""
    from skewerjax.core import SkewerBundle
    from skewerjax.core.absorbers import ExactFrame

    cofm = np.atleast_2d(np.asarray(cofm, dtype=np.float64))
    one = SkewerBundle.axis_aligned(cofm[:1], axis, box, frame_spacing, cosmology, a_ref=a_ref)
    h = one.host
    pts = h["points"][0]
    ax = int(axis) - 1
    origin = cofm.copy()
    origin[:, ax] = pts[0, ax]
    direction = np.zeros_like(origin)
    direction[:, ax] = np.sign(pts[-1, ax] - pts[0, ax])
    L = float(abs(pts[-1, ax] - pts[0, ax]))
    paths = cx.StraightPaths(origin, direction, np.full(len(cofm), L), box, ax)
    s = np.abs(pts[:, ax] - pts[0, ax])
    sign = float(np.asarray(one.array_direction).ravel()[0])
    frame = ExactFrame(
        s[None, :], h["scale_factors"][0][:1], a_dagger_value=float(np.ravel(h["a0"])[0]), photon_sign=np.full(len(cofm), sign)
    )
    return paths, frame


# --------------------------------------------------------------------------- the routes
STRATEGIES = ("auto", "gather", "stream")


def choose_strategy(points) -> str:
    """The route ``snapshot_optical_depth`` takes with ``strategy="auto"``: "stream" for axis-aligned straight skewers
    (one pass over the snapshot with the reach test by index arithmetic; never slower than the gather route in the
    measurements of ``docs/performance.md``, from 16 skewers up), "gather" for every other path (curved, or straight
    in any direction: the selection by a k-d tree on the sampling points)."""
    pts = np.asarray(points, dtype=np.float64)
    if not np.all(cx.straight_paths_mask(pts)):
        return "gather"
    chord = pts[:, -1] - pts[:, 0]
    t = chord / np.linalg.norm(chord, axis=-1)[:, None]
    for k in range(3):
        if np.all(np.abs(np.abs(t[:, k]) - 1.0) < 1e-12):
            return "stream"
    return "gather"


def snapshot_optical_depth(
    reader,
    skewer_bundle,
    v,
    line_bundle,
    *,
    strategy: str = "auto",
    physics: dict | None = None,
    tile_size: int = 1024,
    threads: int | None = None,
    devices=None,
    return_info: bool = False,
    **tau_options,
):
    """tau (N_lines, N_sk, N_v) of the skewers of ``skewer_bundle`` through the snapshot of ``reader`` (the gas physics
    ``physics``: ``gas_to_cells`` options, default SkewerJAX's), by one of two routes:

    * "stream" (``stream_cells`` + ``tiles_optical_depth``): one pass over the files, the cells within reach of any
      skewer found by index arithmetic, the gas physics on those cells only, the skewers in tiles; axis-aligned straight
      skewers;
    * "gather" (``select_cells_near_paths`` + ``gas_to_cells`` + ``core.compute_optical_depth``): the selection by a
      k-d tree on the sampling points of the skewers, any paths (curved ones take the sampled dual-segment pieces);
    * "auto" (default): ``choose_strategy``.

    tau_options : those of ``core.compute_optical_depth`` (``velocity_period`` a number for the stream route; ``devices``
    the devices of the engine). Returns tau (and dict(strategy, timings, the route's info))."""
    from skewerjax.core import compute_optical_depth
    from skewerjax.core.absorbers import _bundle_host, bundle_frame
    from skewerjax.simulations.selection import select_cells_near_paths
    from skewerjax.simulations.snapshot_cells import gas_to_cells

    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}, got {strategy!r}.")
    t0 = time.perf_counter()
    pts = _bundle_host(skewer_bundle, 0)[0]
    route = choose_strategy(pts) if strategy == "auto" else strategy
    physics = {} if physics is None else dict(physics)
    if route == "stream":
        paths = cx.paths_from_points(pts, reader.header.box_cMpc)
        if paths.axis is None:
            raise ValueError("strategy='stream' takes axis-aligned skewers; use 'gather' (or 'auto').")
        frame = bundle_frame(skewer_bundle)
        tile_of = tile_skewers(paths, tile_size)
        store, pinfo = stream_cells(reader, paths, tile_of, list(line_bundle.species_keys), {"run": physics}, threads=threads)
        t1 = time.perf_counter()
        opts = dict(tau_options)
        vp = opts.pop("velocity_period", None)
        taus, tinfo = tiles_optical_depth(
            store,
            "run",
            paths,
            tile_of,
            frame,
            v,
            line_bundle,
            velocity_period=vp,
            engines={"tau": list(devices) if devices else [None]},
            **opts,
        )
        tau = taus["tau"]
        info = dict(
            strategy="stream", pass_info=pinfo, tiles=tinfo, timings=dict(pass_s=t1 - t0, tiles_s=time.perf_counter() - t1)
        )
    else:
        gas = select_cells_near_paths(reader, pts, mode="voronoi")
        cells = gas_to_cells(gas, list(line_bundle.species_keys), **physics)
        t1 = time.perf_counter()
        tau = np.asarray(compute_optical_depth(skewer_bundle, v, line_bundle, cells, devices=devices, **tau_options))
        info = dict(strategy="gather", n_cells=len(gas.positions), timings=dict(select_s=t1 - t0, tau_s=time.perf_counter() - t1))
    return (tau, info) if return_info else tau
