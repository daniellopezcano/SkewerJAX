"""How SkewerJAX and fake_spectra scale (notebook 14; ``scripts/benchmarks/benchmark_scaling.py``).

Four entries compute the same spectra (Lya and Si III 1206) on the same skewers and pixels, with the calls of notebook 13:

* ``fake_spectra`` (native ``Spectra``, 32 OpenMP threads, timed from outside: setup, I/O, gas physics, C engine);
* ``SkewerJAX default, CPU``, ``SkewerJAX matched, CPU`` (every option at fake_spectra's value) and
  ``SkewerJAX default, GPU`` (the tau sum on the GPU; the host stages are those of the default CPU entry).

The skewers run along y on a nested grid, x, z = i L / n (``nested_grid``), so that the **reference skewer**
(L/4, L/4) (notebook 12's) belongs to every grid with n a multiple of 4. SkewerJAX makes one pass over the snapshot for
all its entries (``simulations.stream.stream_cells``: the cells within reach of each tile of skewers, with both physics
configurations), then the optical depth tile by tile (``tiles_optical_depth``) for each entry and pixel grid
(``skewerjax_point``). The stages of the pass (I/O, selection, physics) are its wall time attributed by the shares of its
threads; the pieces and tau stages are the tiles' times.

Protocol (``run_point``): **cold** for every code (``evict_page_cache`` drops the snapshot from the page cache before each
code reads it; a snapshot larger than the RAM is cold anyway), fake_spectra first, then a SkewerJAX warm-up on TNG50-4
(the JIT compilation of the same kernels, timed as ``jit_s``, nothing of the target snapshot read), then the timed
SkewerJAX run; ``repeats`` runs of each code at small N, where the I/O varies (the fastest is kept, all are recorded).
Peak resident memory (total and anonymous: ``MemorySampler``) and the GPU's peak are recorded for every stage.

Blocks (``scaling_points``), every point one configuration (``point_config``), deduplicated across blocks:

* **A**, simulation resolution: TNG50-4, TNG50-3, TNG50-2 at n_max^2 skewers, 1 km/s pixels;
* **B**, number of skewers: TNG50-2, n = 4, 16, 32, 64, 128, 256 per side (16 to 65,536 skewers), 1 km/s;
* **C1**, pixel width: TNG50-2, n_max^2, pixels of 1, 2, 4, 8, 16 km/s, merged into the n_max^2 point of A and B
  (SkewerJAX's pass is shared by the five grids; fake_spectra runs once per grid);
* **C2**, the sampling of the midpoint option (``run_c2``): the earlier default sampled each skewer at 0.125 ... 4 km/s;
  the default integrates the profile over each crossing and has no sampling, so C2 is a precision study of the option
  against the default, not a timed point.

``scripts/benchmarks/benchmark_scaling.py`` runs every point in its own process and writes one record per point to
``benchmarks/results/scaling/`` (resumable), and C2 with ``--c2``.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from skewerjax.comparison.benchmark import MATCHED_CELLS, STAGES, StageTimer, machine_metadata

FS, CPU_DEF, CPU_MAT, GPU_DEF = "fake_spectra", "SkewerJAX default, CPU", "SkewerJAX matched, CPU", "SkewerJAX default, GPU"
ENTRIES = (FS, CPU_DEF, CPU_MAT, GPU_DEF)
SKEWERJAX_ENTRIES = (CPU_DEF, CPU_MAT, GPU_DEF)
LINE_IDS = ("HI-LyA", "SiIII-1206")
FS_LINES = (("H", 1, 1215), ("Si", 3, 1206))
SNAP, AXIS = 25, 2  # snapshot 25 (z = 3); the skewers along y (fake_spectra's axis 2)
REFERENCE = (0.25, 0.25)  # the reference skewer (x, z) / L: notebook 12's
FS_UPPER_BOUND = 256  # the largest n of the nested grids on which fake_spectra runs within 20 min on TNG50-2 (996 s; 512: > 20 min)
N_MAX = 128  # the grid of notebook 13 and of blocks A and C1 (16,384 skewers)
B_SIDES = (4, 16, 32, 64, 128, 256)  # block B: 16 to 65,536 skewers (256: fake_spectra's upper bound, ``FS_UPPER_BOUND``)
SIMULATIONS = ("TNG50-4", "TNG50-3", "TNG50-2")
PIXELS_KMS = (1.0, 2.0, 4.0, 8.0, 16.0)
DEFAULT_CELLS = dict(mass_unit="amu", T_constants="codata", x_HI_model="stored_sf", cloudy_density="hydrogen", cloudy_interpolation="linear")
DEFAULT_TAU = dict(
    velocity_period=False,
    voigt="harris",
    detuning="exact",
    a_eff=None,
    hubble_flow="exact",
    aH_ref=None,
    b_constants="codata",
    pixel_average=False,
    tautail=None,
    resolution_tolerance=0.25,
    piece_chunk=2048,
)


def matched_tau(v_period: float, a_snap: float, aH: float) -> dict:
    """The tau options of the matched mode (notebook 12's matched block; ``benchmark.MATCHED_TAU`` with the snapshot's
    values): one period of the box, fake_spectra's detuning, density conversion, Hubble flow, constants and per-cell
    schemes."""
    return dict(
        velocity_period=v_period,
        voigt="harris",
        detuning="linear",
        a_eff=a_snap,
        hubble_flow="linear",
        aH_ref=aH,
        b_constants="fake_spectra",
        pixel_average=True,
        tautail=1e-7,
        resolution_tolerance=None,
    )


# --------------------------------------------------------------------------- the configurations
def nested_grid(n: int, box: float) -> tuple:
    """(X, Z) [cMpc] of the n^2 skewers x, z = i box / n, i = 0 .. n - 1 (x outer): every grid holds the coarser ones."""
    g = np.arange(n) * box / n
    X, Z = np.meshgrid(g, g, indexing="ij")
    return X.ravel(), Z.ravel()


def reference_index(n: int) -> int:
    """The index of the reference skewer (``REFERENCE``) in ``nested_grid(n, .)`` (n a multiple of 4)."""
    if n % 4:
        raise ValueError(f"the reference skewer (L/4, L/4) is on the grid only for n a multiple of 4, got {n}.")
    i, k = int(round(REFERENCE[0] * n)), int(round(REFERENCE[1] * n))
    return i * n + k


def point_config(sim: str, n: int, dv_kms: float = 1.0, entries=ENTRIES, dv_list=None, repeats: int = 1, tile_size: int = 1024) -> dict:
    """One configuration: the snapshot, n per side, the pixel widths (``dv_list``: the grids sharing one pass), the
    entries run, the repeats and the tile size."""
    dvl = [float(dv_kms)] if dv_list is None else [float(x) for x in dv_list]
    c = dict(sim=sim, snap=SNAP, n=int(n), n_skewers=int(n) ** 2, dv_kms=float(dvl[0]), dv_list=dvl,
             entries=list(entries), repeats=int(repeats), tile_size=int(tile_size))
    c["id"] = f"{sim}_n{n}_dv{'-'.join(f'{x:g}' for x in dvl)}" + ("" if FS in entries else "_sjx")
    return c


def scaling_points(n_max: int = N_MAX, blocks=("A", "B", "C1")) -> list:
    """The configurations of the blocks, each once, with the blocks it belongs to (``blocks``):

    * **A**, simulation resolution: TNG50-4, TNG50-3, TNG50-2 at n_max^2 skewers, 1 km/s pixels;
    * **B**, number of skewers: TNG50-2, n = 4, 16, 32, 64, 128, 256 per side (16 to 65,536 skewers), 1 km/s; two runs of
      each code at n <= 16, where the I/O dominates and varies;
    * **C1**, pixel width: TNG50-2 at n_max^2 skewers, pixels of 1, 2, 4, 8, 16 km/s: SkewerJAX's pass and pieces shared
      by the five grids (only the optical depth depends on them), fake_spectra once per grid.

    Every entry (fake_spectra; SkewerJAX default CPU, default GPU, matched CPU) runs at every point; the TNG50-2 n_max^2
    point is shared by A, B and C1. The sampling study of the midpoint option (C2) is a separate precision run
    (``scripts/benchmarks/benchmark_scaling.py --c2``): the default does not sample."""
    pts = {}

    def add(block, c):
        pts.setdefault(c["id"], dict(c, blocks=[]))["blocks"].append(block)

    dv_c1 = PIXELS_KMS if "C1" in blocks else (1.0,)
    target = point_config("TNG50-2", n_max, dv_list=dv_c1)
    if "A" in blocks:
        for sim in SIMULATIONS:
            add("A", target if sim == "TNG50-2" else point_config(sim, n_max))
    if "B" in blocks:
        for n in B_SIDES:
            add("B", target if n == n_max else point_config("TNG50-2", n, repeats=2 if n <= 16 else 1))
    if "C1" in blocks:
        add("C1", target)
    return list(pts.values())


# --------------------------------------------------------------------------- measurement helpers
def cache_state(snapdir) -> float:
    """The fraction of the snapshot's files resident in the page cache (``fincore``; NaN without it)."""
    files = sorted(Path(snapdir).glob("*.hdf5"))
    if not files or shutil.which("fincore") is None:
        return float("nan")
    out = subprocess.run(["fincore", "-b", "-n", "-o", "RES,SIZE", *map(str, files)], capture_output=True, text=True).stdout
    res = size = 0
    for line in out.splitlines():
        p = line.split()
        if len(p) == 2:
            res, size = res + int(p[0]), size + int(p[1])
    return res / size if size else float("nan")


def cache_label(fraction: float) -> str:
    """"warm" when most of the snapshot is in the page cache, "cold" otherwise (a snapshot larger than the RAM is cold)."""
    return "warm" if fraction > 0.5 else "cold"


def rebin_average(F, edges_fine, edges_coarse) -> np.ndarray:
    """The average of a piecewise-constant F (..., N_fine) over each coarse pixel: every fine pixel weighted by its overlap
    with the coarse one (the pixel grids need not nest)."""
    F = np.asarray(F, dtype=np.float64)
    ef, ec = np.asarray(edges_fine, dtype=np.float64), np.asarray(edges_coarse, dtype=np.float64)
    lo = np.maximum(ef[None, :-1], ec[:-1, None])
    hi = np.minimum(ef[None, 1:], ec[1:, None])
    w = np.clip(hi - lo, 0.0, None)  # (N_coarse, N_fine)
    return (F @ w.T) / w.sum(axis=1)


def fit_slope(x, y) -> float:
    """The slope of log y against log x (least squares), NaN with fewer than two positive points."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    ok = (x > 0) & (y > 0) & np.isfinite(y)
    return float(np.polyfit(np.log(x[ok]), np.log(y[ok]), 1)[0]) if ok.sum() >= 2 else float("nan")


# --------------------------------------------------------------------------- one point
def run_fake_spectra(snap_base, X, Z, H, dv_kms: float) -> tuple:
    """fake_spectra on the skewers (X, Z) along y, pixels of ``dv_kms``: (tau (2, N_sk, nbins) of Lya and Si III, its
    stage times {stage: s}, nbins, dvbin). Timed from outside as in notebook 13 (``StageTimer``); its save file goes to a
    temporary folder removed afterwards (``reload_file=True``: it recomputes)."""
    from fake_spectra.spectra import Spectra

    cofm = np.column_stack([X, np.zeros_like(X), Z]) / H.code_length_cMpc  # [ckpc/h]
    axis = np.full(X.size, AXIS, dtype=np.int32)
    T = StageTimer()
    with tempfile.TemporaryDirectory(prefix="fs_scaling_") as work, contextlib.redirect_stdout(io.StringIO()):
        with T.stage("setup"):
            fs = Spectra(SNAP, str(snap_base), cofm, axis, res=dv_kms, savefile="t.hdf5", savedir=work, reload_file=True, quiet=True)
        with T.patched((fs.snapshot_set, "get_data", "io"), (fs, "_do_interpolation_work", "tau")):
            with T.stage("physics"):
                tau = np.stack([np.asarray(fs.get_tau(*line)) for line in FS_LINES])
    return tau, dict(T.t), int(fs.nbins), float(fs.dvbin)


def evict_page_cache(snapdir) -> int:
    """Drop the snapshot's files from the page cache (``posix_fadvise(DONTNEED)``: a hint to the kernel, nothing is
    written; pages mapped by a process stay): the cold protocol. Returns the number of files."""
    n = 0
    for p in sorted(Path(snapdir).glob("*.hdf5")):
        fd = os.open(p, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            n += 1
        finally:
            os.close(fd)
    return n


def rss_anon_GB() -> float:
    """Anonymous resident memory of this process [GB] (``RssAnon``: without file pages mapped from the snapshot)."""
    with open("/proc/self/status") as f:
        for ln in f:
            if ln.startswith("RssAnon:"):
                return int(ln.split()[1]) / 1e6
    return float("nan")


class MemorySampler:
    """Peak RSS and anonymous RSS [GB] of this process since ``reset``, sampled every ``dt`` seconds."""

    def __init__(self, dt: float = 0.05):
        import threading

        from skewerjax.comparison.benchmark import rss_GB

        self._rss = rss_GB
        self.dt, self._stop = dt, threading.Event()
        self.reset()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while not self._stop.wait(self.dt):
            self.rss = max(self.rss, self._rss())
            self.anon = max(self.anon, rss_anon_GB())

    def reset(self):
        self.rss, self.anon = self._rss(), rss_anon_GB()

    def read(self) -> dict:
        return dict(rss_GB=max(self.rss, self._rss()), anon_GB=max(self.anon, rss_anon_GB()))


def gpu_peak_GB(gpu) -> float | None:
    """The peak memory in use on the first GPU since the process started [GB] (``peak_bytes_in_use``), or None."""
    try:
        return gpu[0].memory_stats()["peak_bytes_in_use"] / 1e9 if gpu else None
    except Exception:
        return None


def mode_options(mode: str, nbins: int, dv: float, H) -> dict:
    """The tile options of an entry's mode: "default" (Voronoi crossings, the integrated profile, the exact frame) or
    "matched" (fake_spectra's chords and every option at its value, ``matched_tau``)."""
    if mode == "default":
        return dict(config="default", pieces="crossings", frame="exact", tau=dict(velocity_period=None))
    mt = matched_tau(nbins * dv, H.a, H.velfac_kms_per_cMpc)
    mt.pop("resolution_tolerance")
    return dict(config="matched", pieces="chords", frame="snapshot", tau=dict(mt, profile="midpoint", chord_weight=1 / 8))


def skewerjax_point(reader, cofm, grids: dict, lines, entries, *, cpu, gpu, tile_size: int = 1024, threads=None, memory=None, clock=None):
    """The SkewerJAX entries on the skewers ``cofm`` (n, 3) [cMpc] along y: one pass over the snapshot shared by every
    entry (``simulations.stream.stream_cells``, both physics configurations when the matched entry runs), then the
    tiles of every entry on every pixel grid of ``grids`` ({label: v}). Returns ({(entry, label): tau}, the record)."""
    from skewerjax.core.absorbers import SnapshotFrame
    from skewerjax.simulations import stream as st

    H = reader.header
    L = H.box_cMpc
    rec = dict(entries={}, tiles={})
    t0 = time.perf_counter()
    paths, frame = st.box_skewers(cofm, AXIS, L, H.cosmology(), H.a)
    tile_of = st.tile_skewers(paths, tile_size)
    snap = SnapshotFrame(H.a, H.velfac_kms_per_cMpc, path_length=np.full(len(paths), L))
    setup = time.perf_counter() - t0
    physics = {"default": DEFAULT_CELLS}
    if CPU_MAT in entries:
        physics["matched"] = MATCHED_CELLS
    if memory is not None:
        memory.reset()
    store, pinfo = st.stream_cells(reader, paths, tile_of, list(lines.species_keys), physics, threads=threads)
    rec["pass"] = {k: v for k, v in pinfo.items()}
    rec["pass_memory"] = None if memory is None else memory.read()
    tm = pinfo["timings"]  # wall shares of the pass's stages
    wall_pass = pinfo["pass_s"] + pinfo["setup_s"] + pinfo["assemble_s"]
    share = lambda keys: sum(tm.get(k, 0.0) for k in keys)
    io, sel = share(("read", "read_rest")), share(("reach", "tiles"))
    phys = {cfg: tm.get(f"physics:{cfg}", 0.0) for cfg in physics}
    other = wall_pass - io - sel - sum(phys.values())  # pass set-up and assembly: shared, with the I/O
    out = {}
    runs = [(e, lab) for lab in grids for e in (CPU_DEF, GPU_DEF, CPU_MAT) if e in entries and (e != CPU_MAT or lab == next(iter(grids)))]
    for e, lab in runs:
        v = grids[lab]
        nbins, dv = v.size, float(v[1] - v[0])
        mode = mode_options("matched" if e == CPU_MAT else "default", nbins, dv, H)
        devs = gpu if e == GPU_DEF else cpu
        if memory is not None:
            memory.reset()
        t = time.perf_counter()
        taus, tinfo = st.tiles_optical_depth(
            store, mode["config"], paths, tile_of, snap if mode["frame"] == "snapshot" else frame, v, lines,
            pieces=mode["pieces"], engines={"tau": devs}, **mode["tau"],
        )
        wall = time.perf_counter() - t
        rows = tinfo["rows"]
        pieces_s = sum(r["wait_s"] for r in rows)
        tau_s = sum(r["tau_s"]["tau"] for r in rows)
        cfg = mode["config"]
        stages = dict(setup=setup, io=io + other, selection=sel, physics=phys[cfg], pieces=pieces_s, tau=tau_s)
        rec["entries"][f"{e}|{lab}"] = dict(stages=stages, total=float(sum(stages.values())), tiles_wall=wall,
                                            memory=None if memory is None else memory.read(), gpu_peak_GB=gpu_peak_GB(gpu) if e == GPU_DEF else None)
        rec["tiles"][f"{e}|{lab}"] = [
            dict(n_skewers=r["n_skewers"], n_cells=r["n_cells"], n_pieces=r["n_pieces"], geometry_s=r["geometry_s"], contributions_s=r["contributions_s"],
                 wait_s=r["wait_s"], tau_s=r["tau_s"]["tau"], engine=r.get("engine", {}).get("tau"))
            for r in rows
        ]
        out[(e, lab)] = taus["tau"]
    rec["store_GB"] = store.nbytes / 1e9
    del store
    return out, rec


def run_point(c: dict, data_dir) -> dict:
    """Measure the configuration ``c`` (``point_config``) in this process, with the cold protocol (the snapshot is
    dropped from the page cache before each code reads it, ``evict_page_cache``):

    1. fake_spectra on every pixel grid (it compiles nothing; ``repeats`` runs at small N, where its I/O is noisy);
    2. the SkewerJAX warm-up: the same entries on min(N, tile size) TNG50-4 skewers (the JIT compilation of the same
       kernels, and numba's; nothing of the target snapshot is read), timed as ``jit_s``;
    3. the SkewerJAX entries: one pass over the snapshot and the tiles of every entry and grid (``skewerjax_point``),
       ``repeats`` times at small N (the fastest run is kept, every run is recorded).

    Needs JAX configured by the caller (``skewerjax.config.configure_devices``). Returns the record."""
    import jax

    from skewerjax import config
    from skewerjax.core import LineBundle
    from skewerjax.simulations import ArepoHDF5Reader

    jax.config.update("jax_default_device", jax.devices("cpu")[0])
    t_start = time.perf_counter()
    snap_base = Path(data_dir) / c["sim"]
    reader = ArepoHDF5Reader(snap_base, SNAP)
    H = reader.header
    L = H.box_cMpc
    X, Z = nested_grid(c["n"], L)
    cofm = np.column_stack([X, np.zeros_like(X), Z])
    grids = {}
    for dv in c["dv_list"]:
        nb, d = H.pixel_grid(dv)
        grids[f"{dv:g}"] = (np.arange(nb) + 0.5) * d
    lines = LineBundle(list(LINE_IDS), ref_line_id="HI-LyA")
    snapdir = Path(reader.files[0]).parent
    cpu, gpu = config.devices("cpu"), config.devices("gpu")
    entries = [e for e in c["entries"] if e != GPU_DEF or gpu]
    repeats = int(c.get("repeats", 1))
    tile = int(c.get("tile_size", 1024))
    mem = MemorySampler()
    rec = dict(
        id=c["id"], config=c, status="ok", machine=machine_metadata(), gpu=[d.device_kind for d in gpu], precision=config.get_precision(),
        cpu_devices=len(cpu), snapshot=dict(n_gas=int(H.n_gas), box_cMpc=float(L), n_files=int(H.n_files), redshift=float(H.redshift)),
        grids={k: dict(nbins=int(v.size), dv_kms=float(v[1] - v[0])) for k, v in grids.items()}, entries={}, runs=[], cache={},
        date=time.strftime("%Y-%m-%d %H:%M"),
    )
    iref = reference_index(c["n"]) if c["n"] % 4 == 0 else None
    flux, tau_fs = {}, {}
    if FS in entries:
        for lab, v in grids.items():
            runs = []
            for r in range(repeats):
                evict_page_cache(snapdir)
                rec["cache"][f"fs|{lab}|{r}"] = cache_state(snapdir)
                mem.reset()
                t0 = time.perf_counter()
                tau, st_, nb, dvb = run_fake_spectra(snap_base, X, Z, H, float(lab))
                stg = {k: float(st_.get(k, 0.0)) for k in STAGES}
                runs.append(dict(stages=stg, total=float(sum(stg.values())), wall=time.perf_counter() - t0, memory=mem.read()))
                if (nb, round(dvb, 9)) != (v.size, round(float(v[1] - v[0]), 9)):
                    raise RuntimeError(f"fake_spectra's pixels ({nb} of {dvb}) differ from SkewerJAX's ({v.size}).")
            tau_fs[lab] = tau
            rec["entries"][f"{FS}|{lab}"] = dict(runs[int(np.argmin([x["total"] for x in runs]))], repeats=runs)
            if iref is not None:
                flux[f"{FS}|{lab}"] = {ln: np.exp(-tau[j, iref]).tolist() for j, ln in enumerate(LINE_IDS)}
    sj = [e for e in entries if e != FS]
    agree = {}
    if sj:
        t0 = time.perf_counter()
        warm = ArepoHDF5Reader(Path(data_dir) / "TNG50-4", SNAP)
        nw = min(int(c["n"]), int(np.sqrt(tile)))
        Xw, Zw = nested_grid(nw, warm.header.box_cMpc)
        skewerjax_point(warm, np.column_stack([Xw, np.zeros_like(Xw), Zw]), grids, lines, sj, cpu=cpu, gpu=gpu, tile_size=tile)
        rec["jit_s"] = time.perf_counter() - t0
        best = None
        for r in range(repeats):
            evict_page_cache(snapdir)
            rec["cache"][f"SkewerJAX|{r}"] = cache_state(snapdir)
            out, srec = skewerjax_point(reader, cofm, grids, lines, sj, cpu=cpu, gpu=gpu, tile_size=tile, memory=mem)
            rec["runs"].append(srec)
            tot = sum(x["total"] for x in srec["entries"].values())
            if best is None or tot < best[0]:
                best = (tot, out, srec)
            else:
                del out
        _, out, srec = best
        rec["entries"].update(srec["entries"])
        rec["pass"] = srec["pass"]
        rec["store_GB"] = srec["store_GB"]
        for (e, lab), tau in out.items():
            F = np.exp(-tau.astype(np.float64))
            if iref is not None:
                flux[f"{e}|{lab}"] = {ln: F[j, iref].tolist() for j, ln in enumerate(LINE_IDS)}
            name = "matched" if e == CPU_MAT else "gpu" if e == GPU_DEF else "default"
            for j, ln in enumerate(LINE_IDS):
                if lab in tau_fs:
                    d = F[j] - np.exp(-tau_fs[lab][j])
                    agree[f"{name}_vs_fs|{lab}:{ln}"] = dict(rms_dF=float(np.sqrt(np.mean(d**2))), max_dF=float(np.abs(d).max()))
                if e == GPU_DEF and (CPU_DEF, lab) in out:
                    d = F[j] - np.exp(-out[(CPU_DEF, lab)][j].astype(np.float64))
                    agree[f"gpu_vs_cpu|{lab}:{ln}"] = dict(rms_dF=float(np.sqrt(np.mean(d**2))), max_dF=float(np.abs(d).max()))
        del out
    rec["agreement"] = agree
    rec["reference"] = dict(index=iref, flux=flux)
    rec["wall_s"] = time.perf_counter() - t_start
    rec["jax"] = jax.__version__
    return rec


# --------------------------------------------------------------------------- C2: the sampling of the midpoint option
C2_SPACINGS = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0)


def run_c2(data_dir, *, n: int = 8, sim: str = "TNG50-2") -> dict:
    """Block C2 (one process, the package's precision): the n^2 skewers of the nested grid through ``sim``, the dual-segment
    pieces with the midpoint profile (the earlier default) at every spacing of ``C2_SPACINGS`` against the default (cell
    crossings, the integrated profile, which has no sampling): rms and max |dF| per line, the pieces per skewer and the
    tau time per skewer of each. Run it in float64 for the precision, in float32 for the times."""
    import jax

    from skewerjax import config
    from skewerjax.core import LineBundle, SkewerBundle, cell_pieces, compute_optical_depth
    from skewerjax.simulations import ArepoHDF5Reader, gas_to_cells, select_cells_near_paths

    jax.config.update("jax_default_device", jax.devices("cpu")[0])
    reader = ArepoHDF5Reader(Path(data_dir) / sim, SNAP)
    H = reader.header
    L, AH = H.box_cMpc, H.velfac_kms_per_cMpc
    X, Z = nested_grid(n, L)
    cofm = np.column_stack([X, np.zeros_like(X), Z])
    nb, dv = H.pixel_grid(1.0)
    v = (np.arange(nb) + 0.5) * dv
    lines = LineBundle(list(LINE_IDS), ref_line_id="HI-LyA")
    fine = SkewerBundle.axis_aligned(cofm, AXIS, L, min(C2_SPACINGS) / AH, H.cosmology(), a_ref=H.a)
    cells = gas_to_cells(select_cells_near_paths(reader, fine.host["points"], mode="voronoi"), list(lines.species_keys), **DEFAULT_CELLS)
    coarse = SkewerBundle.axis_aligned(cofm, AXIS, L, 1.0 / AH, H.cosmology(), a_ref=H.a)

    def timed(fn):
        fn()  # compile
        t = time.perf_counter()
        out = np.asarray(fn())
        return out, time.perf_counter() - t

    ref, t_int = timed(lambda: compute_optical_depth(coarse, v, lines, cells, resolution_tolerance=None))
    rec = dict(sim=sim, n=n, precision=config.get_precision(), integrated=dict(tau_s_per_skewer=t_int / n**2), midpoint={})
    F0 = np.exp(-ref.astype(np.float64))
    for ds in C2_SPACINGS:
        b = SkewerBundle.axis_aligned(cofm, AXIS, L, ds / AH, H.cosmology(), a_ref=H.a)
        p = cell_pieces(cells, lines, skewer_bundle=b, deposition="voronoi_segments", a_data=H.a)
        tau, t = timed(lambda p=p: compute_optical_depth(p, v, lines, profile_integration="midpoint", resolution_tolerance=None))
        d = np.exp(-tau.astype(np.float64)) - F0
        rec["midpoint"][f"{ds:g}"] = dict(
            pieces_per_skewer=float((np.asarray(p.length_m) > 0).sum() / n**2),
            tau_s_per_skewer=t / n**2,
            **{f"{ln}:rms_dF": float(np.sqrt(np.mean(d[j] ** 2))) for j, ln in enumerate(LINE_IDS)},
            **{f"{ln}:max_dF": float(np.abs(d[j]).max()) for j, ln in enumerate(LINE_IDS)},
        )
    return rec


# --------------------------------------------------------------------------- records
def load_records(folder) -> dict:
    """{id: record} of the JSON records in ``folder``."""
    return {json.loads(p.read_text())["id"]: json.loads(p.read_text()) for p in sorted(Path(folder).glob("*.json"))}


def block_records(records: dict, block: str, n_max: int, blocks=("A", "B", "C1")) -> list:
    """The records of one block, in the order of its parameter (``scaling_points(n_max, blocks)``: the ids of the
    benchmark's points, the shared target included); missing points are left out."""
    recs = [records[c["id"]] for c in scaling_points(n_max, blocks) if block in c["blocks"] and c["id"] in records]
    return recs if block == "C1" else sorted(recs, key=lambda r: block_parameter(block, r))


def block_parameter(block: str, rec: dict, grid: str | None = None) -> float:
    """The parameter of a block at a record: the cells of the snapshot (A), the skewers (B), the pixel width (C1: of
    ``grid``)."""
    c = rec["config"]
    if block == "C1":
        return float(rec["grids"][grid]["dv_kms"])
    return {"A": rec["snapshot"]["n_gas"], "B": c["n_skewers"]}[block]


def entry_stages(rec: dict, grid: str) -> dict:
    """{entry: {stage: s}} of one pixel grid of a record (the matched entry runs on the first grid only)."""
    out = {}
    for e in ENTRIES:
        r = rec["entries"].get(f"{e}|{grid}")
        if r is not None:
            out[e] = {k: float(r["stages"].get(k, 0.0)) for k in STAGES}
    return out


def block_rows(records: dict, block: str, n_max: int) -> list:
    """One row per point of a block (C1: per pixel grid of the shared point), in the order of its parameter: dict(x,
    title, stages {entry: {stage: s}}, jit {entry: s}, flux {entry: {line: F}} of the reference skewer (or None), nbins,
    dv_kms, agreement {key: rms / max}, record)."""
    rows = []
    for rec in block_records(records, block, n_max):
        c = rec["config"]
        grids = list(rec["grids"]) if block == "C1" else [next(iter(rec["grids"]))]
        for g in grids:
            stg = entry_stages(rec, g)
            jit = {e: (0.0 if e == FS else float(rec.get("jit_s", 0.0))) for e in stg}
            fl = rec.get("reference", {}).get("flux", {})
            flux = {e: fl[f"{e}|{g}"] for e in ENTRIES if f"{e}|{g}" in fl} or None
            title = {"A": f"{c['sim']}: {int(rec['snapshot']['n_gas']):,} cells", "B": f"{c['n']} x {c['n']} = {c['n_skewers']:,} skewers",
                     "C1": f"pixels of {float(rec['grids'][g]['dv_kms']):.3g} km/s"}[block]
            agree = {k.split("|")[0] + ":" + k.split(":")[-1]: v for k, v in rec.get("agreement", {}).items() if k.split("|")[1].split(":")[0] == g}
            rows.append(dict(x=block_parameter(block, rec, g), title=title, stages=stg, jit=jit, flux=flux, nbins=int(rec["grids"][g]["nbins"]),
                             dv_kms=float(rec["grids"][g]["dv_kms"]), agreement=agree, record=rec))
    return rows


def write_record(rec: dict, folder) -> Path:
    """One record as ``<folder>/<id>.json``, compact (every float to 9 significant digits: exact for float32 values)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / f"{rec['id']}.json"
    p.write_text(json.dumps(_round(rec), separators=(",", ":")))  # compact: the fluxes are most of it
    return p


def _round(x):
    if isinstance(x, float):
        return float(f"{x:.9g}") if np.isfinite(x) else None
    if isinstance(x, dict):
        return {k: _round(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_round(v) for v in x]
    return x


# --------------------------------------------------------------------------- the launcher
def worker_env() -> dict:
    """The environment of a worker: 32 OpenMP threads for fake_spectra, the GPU next to the CPU without preallocation, and
    the CUDA libraries of the pip wheels on the library path."""
    import site

    root = str(Path(__file__).resolve().parents[2])
    env = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join([root] + [x for x in [os.environ.get("PYTHONPATH")] if x]),
        OMP_NUM_THREADS=str(os.cpu_count()),
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        JAX_PLATFORMS="cuda,cpu",
    )
    env.pop("XLA_FLAGS", None)
    libs = sorted(str(p) for s in site.getsitepackages() for p in Path(s).glob("nvidia/*/lib"))
    env["LD_LIBRARY_PATH"] = ":".join(libs + [env.get("LD_LIBRARY_PATH", "")])
    return env


def run_points(points: list, out, *, data_dir, log=print, timeout_s: float | None = None) -> list:
    """Measure every point of ``points`` without a record in ``out``, each in its own process (``main``), in order:
    finished points are skipped, so an interrupted run resumes. Returns the ids measured. The log of every point goes to
    ``out/logs/<id>.log``. The calling process must not hold the GPU: importing the package starts JAX, which on a GPU
    preallocates 75% of its memory unless ``JAX_PLATFORMS=cpu`` (or ``XLA_PYTHON_CLIENT_PREALLOCATE=false``) is set
    before the import, as ``scripts/benchmarks/benchmark_scaling.py`` does."""
    import sys

    out = Path(out)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    done = []
    for c in points:
        if (out / f"{c['id']}.json").exists():
            log(f"{c['id']}: recorded")
            continue
        cfg = {k: v for k, v in c.items() if k != "blocks"}
        cmd = [sys.executable, "-m", "skewerjax.comparison.scaling", json.dumps(cfg), str(out), "--data", str(data_dir)]
        t0 = time.perf_counter()
        log(f"{c['id']}: running ({', '.join(c.get('blocks', []))}) ...")
        with open(out / "logs" / f"{c['id']}.log", "w") as f:
            p = subprocess.run(cmd, env=worker_env(), stdout=f, stderr=subprocess.STDOUT, timeout=timeout_s)
        log(f"{c['id']}: {'done' if p.returncode == 0 else f'FAILED ({p.returncode})'} in {time.perf_counter() - t0:.0f} s")
        if p.returncode == 0:
            done.append(c["id"])
    return done


def main(argv=None):  # the worker of scripts/benchmarks/benchmark_scaling.py: one point, in its own process
    import argparse

    ap = argparse.ArgumentParser(description="One point of the scaling benchmark (a worker).")
    ap.add_argument("config", help="the configuration, JSON")
    ap.add_argument("out", help="the folder of the records")
    ap.add_argument("--data", required=True, help="the folder of the simulations (skewerjax.utils.paths.DATA_DIR)")
    a = ap.parse_args(argv)
    rec = run_point(json.loads(a.config), a.data)
    print(write_record(rec, a.out))


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count()))
    from skewerjax import config as _config

    _config.configure_devices(n_cpu_devices=2, platforms=os.environ.get("JAX_PLATFORMS", "cuda,cpu"))
    main()
