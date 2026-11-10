"""How SkewerJAX and fake_spectra scale (notebook 14; ``scripts/benchmarks/benchmark_scaling.py``).

Four entries compute the same spectra (Lya and Si III 1206) on the same skewers and pixels, with the calls of notebook 13:

* ``fake_spectra`` (native ``Spectra``, 32 OpenMP threads, timed from outside: setup, I/O, gas physics, C engine);
* ``SkewerJAX default, CPU``, ``SkewerJAX matched, CPU`` (every option at fake_spectra's value) and
  ``SkewerJAX default, GPU`` (the tau sum on the GPU; the host stages are those of the default CPU entry).

The skewers run along y on a nested grid, x, z = i L / n (``nested_grid``), so that the **reference skewer**
(L/4, L/4) (notebook 12's) belongs to every grid with n a multiple of 4. SkewerJAX processes the skewers in batches of at
most ``BATCH`` (``batches``: host memory and the GPU's 12 GB), each batch through the whole route; its stages add up over
the batches. The first batch runs once before the timed run (the warm-up: JIT compilation, numba, page cache), and the
JIT time of a tau stage is that warm-up minus the same batch in the timed run.

Blocks (``scaling_points``), every point one configuration (``point_config``), deduplicated across blocks:

* **A**, simulation resolution: TNG50-4, TNG50-3, TNG50-2 at n_max^2 skewers, 1 km/s pixels;
* **B**, number of skewers: TNG50-2, n = 4, 8, ..., n_max per side, 1 km/s;
* **C1**, pixel width: TNG50-2, n_max^2, pixels of 1, 2, 4, 8, 16 km/s (SkewerJAX at its default 0.25 km/s sampling);
* **C2**, sampling along the skewer: TNG50-2, n_max^2, 1 km/s, points every 0.125 ... 4 km/s, the SkewerJAX entries only
  (fake_spectra has no such parameter: its 1 km/s run of block A is the fixed reference).

``run_point`` measures one point in this process and returns its record (one timed run after the warm-up): the stage
times, the JIT time, the cache state of the snapshot (``cache_state``: the fraction of its files in the page cache before
each code reads it), the machine, the agreement over all skewers, the resolution check of the sampling, and the flux of
the reference skewer from every entry with the pixel grid. ``scripts/benchmarks/benchmark_scaling.py`` runs every point
in its own process and writes one record per point to ``benchmarks/results/scaling/`` (resumable).
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
import warnings
from pathlib import Path

import numpy as np

from skewerjax.comparison.benchmark import MATCHED_CELLS, STAGES, RunClock, StageTimer, machine_metadata

FS, CPU_DEF, CPU_MAT, GPU_DEF = "fake_spectra", "SkewerJAX default, CPU", "SkewerJAX matched, CPU", "SkewerJAX default, GPU"
ENTRIES = (FS, CPU_DEF, CPU_MAT, GPU_DEF)
SKEWERJAX_ENTRIES = (CPU_DEF, CPU_MAT, GPU_DEF)
LINE_IDS = ("HI-LyA", "SiIII-1206")
FS_LINES = (("H", 1, 1215), ("Si", 3, 1206))
SNAP, AXIS = 25, 2  # snapshot 25 (z = 3); the skewers along y (fake_spectra's axis 2)
REFERENCE = (0.25, 0.25)  # the reference skewer (x, z) / L: notebook 12's
BATCH = 1024  # skewers per batch at 0.25 km/s; fewer at a finer sampling (``batches``): the matched chords of 4096 TNG50-2 skewers exceed 125 GB
FS_UPPER_BOUND = 256  # the largest n of the nested grids on which fake_spectra runs within 20 min on TNG50-2 (996 s; 512: > 20 min)
N_MAX = 128  # the largest grid of notebooks 13 and 14: half of it per side (SkewerJAX's dense CPU tau at 256^2 takes ~6 h per run)
SIMULATIONS = ("TNG50-4", "TNG50-3", "TNG50-2")
PIXELS_KMS = (1.0, 2.0, 4.0, 8.0, 16.0)
SPACINGS_KMS = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0)
DEFAULT_SPACING_KMS = 0.25
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


def batches(n_skewers: int, spacing_kms: float = DEFAULT_SPACING_KMS) -> list:
    """Slices of at most ``BATCH`` skewers (proportionally fewer below 0.25 km/s: the pieces grow with the points)."""
    size = max(256, int(BATCH * min(1.0, spacing_kms / DEFAULT_SPACING_KMS)))
    return [slice(a, min(a + size, n_skewers)) for a in range(0, n_skewers, size)]


def point_config(sim: str, n: int, dv_kms: float = 1.0, spacing_kms: float = DEFAULT_SPACING_KMS, entries=ENTRIES) -> dict:
    """One configuration: the snapshot, n per side, the pixel width, the SkewerJAX sampling and the entries run."""
    c = dict(sim=sim, snap=SNAP, n=int(n), n_skewers=int(n) ** 2, dv_kms=float(dv_kms), spacing_kms=float(spacing_kms), entries=list(entries))
    c["id"] = f"{sim}_n{n}_dv{dv_kms:g}_ds{spacing_kms:g}" + ("" if FS in entries else "_sjx")
    return c


def scaling_points(n_max: int, blocks=("A", "B", "C1", "C2")) -> list:
    """The configurations of the blocks (module docstring), each once, with the blocks it belongs to (``blocks``)."""
    pts = {}

    def add(block, c):
        pts.setdefault(c["id"], dict(c, blocks=[]))["blocks"].append(block)

    if "A" in blocks:
        for sim in SIMULATIONS:
            add("A", point_config(sim, n_max))
    if "B" in blocks:
        n = 4
        while n <= n_max:
            add("B", point_config("TNG50-2", n))
            n *= 2
    if "C1" in blocks:
        for dv in PIXELS_KMS:
            add("C1", point_config("TNG50-2", n_max, dv_kms=dv))
    if "C2" in blocks:
        for ds in SPACINGS_KMS:
            entries = ENTRIES if ds == DEFAULT_SPACING_KMS else SKEWERJAX_ENTRIES
            add("C2", point_config("TNG50-2", n_max, spacing_kms=ds, entries=entries))
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


def flux_residuals(F, F_ref) -> dict:
    """rms and max |dF| of F against F_ref (any shape)."""
    d = np.asarray(F, dtype=np.float64) - np.asarray(F_ref, dtype=np.float64)
    return dict(rms_dF=float(np.sqrt(np.mean(d**2))), max_dF=float(np.max(np.abs(d))))


def fit_slope(x, y) -> float:
    """The slope of log y against log x (least squares), NaN with fewer than two positive points."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    ok = (x > 0) & (y > 0) & np.isfinite(y)
    return float(np.polyfit(np.log(x[ok]), np.log(y[ok]), 1)[0]) if ok.sum() >= 2 else float("nan")


class _Agreement:
    """Running rms and max of the flux differences over the batches."""

    def __init__(self):
        self.s2, self.n, self.mx = {}, {}, {}

    def add(self, key, F, F_ref):
        d = np.asarray(F, dtype=np.float64) - np.asarray(F_ref, dtype=np.float64)
        self.s2[key] = self.s2.get(key, 0.0) + float(np.sum(d**2))
        self.n[key] = self.n.get(key, 0) + d.size
        self.mx[key] = max(self.mx.get(key, 0.0), float(np.max(np.abs(d))))

    def result(self) -> dict:
        return {k: dict(rms_dF=float(np.sqrt(self.s2[k] / self.n[k])), max_dF=self.mx[k]) for k in self.s2}


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


def skewerjax_batch(reader, cofm, v, lines, entries, clock, run, cpu, gpu, spacing_kms: float):
    """The SkewerJAX entries on one batch of skewers (``cofm`` (n, 3) [cMpc], along y), the calls of notebook 13, each
    stage timed into ``clock`` (``RunClock``) for ``run``: returns {entry: tau (2, n, N_v)} and the resolution report of the
    default pieces. ``cpu``, ``gpu``: the ``devices`` of the tau sums."""
    from skewerjax.core import SkewerBundle, SnapshotFrame, cell_pieces, compute_optical_depth, resolution_report
    from skewerjax.simulations import gas_to_cells, select_cells_near_paths

    H = reader.header
    L, AH, a_snap = H.box_cMpc, H.velfac_kms_per_cMpc, H.a
    species = list(lines.species_keys)
    nbins, dv = v.size, float(v[1] - v[0])
    out = {}
    with clock.time(CPU_DEF, "setup", run):
        bundle = SkewerBundle.axis_aligned(cofm, AXIS, L, spacing_kms / AH, H.cosmology(), a_ref=a_snap)
    points = bundle.host["points"]
    gas, info = select_cells_near_paths(reader, points, mode="voronoi", max_rows=4_000_000, workers=-1, return_info=True)
    clock.record(CPU_DEF, "io", run, info["timings"]["read"] + info["timings"]["read_selected"])
    clock.record(CPU_DEF, "selection", run, info["timings"]["prefilter"] + info["timings"]["query"])
    with clock.time(CPU_DEF, "physics", run):
        cells = gas_to_cells(gas, species, **DEFAULT_CELLS)
    with clock.time(CPU_DEF, "pieces", run):
        pieces = cell_pieces(cells, lines, skewer_bundle=bundle, deposition="voronoi", a_data=a_snap)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with clock.time(CPU_DEF, "tau", run):
            out[CPU_DEF] = np.asarray(compute_optical_depth(pieces, v, lines, dtype=None, devices=cpu, **DEFAULT_TAU))
    rep = resolution_report(pieces, lines, DEFAULT_TAU["resolution_tolerance"])
    rep = dict(max_ratio=[float(x) for x in rep["max_ratio"]], column_fraction=[float(x) for x in rep["column_fraction"]],
               warned=any("too wide" in str(w.message) for w in caught))
    if GPU_DEF in entries and gpu:
        with clock.time(GPU_DEF, "tau", run):
            out[GPU_DEF] = np.asarray(compute_optical_depth(pieces, v, lines, dtype=None, devices=gpu, **DEFAULT_TAU))
    del pieces, cells
    if CPU_MAT in entries:
        with clock.time(CPU_MAT, "physics", run):
            cells_m = gas_to_cells(gas, species, **MATCHED_CELLS)
        frame = SnapshotFrame(a_snap, AH, path_length=np.full(len(cofm), L))
        with clock.time(CPU_MAT, "pieces", run):
            pieces_m = cell_pieces(cells_m, lines, points=points, frame=frame, deposition="tophat_chords", chord_weight=1 / 8, a_data=a_snap)
        with clock.time(CPU_MAT, "tau", run):
            out[CPU_MAT] = np.asarray(compute_optical_depth(pieces_m, v, lines, devices=cpu, **matched_tau(nbins * dv, a_snap, AH)))
    return out, rep


def run_point(c: dict, data_dir) -> dict:
    """Measure the configuration ``c`` (``point_config``) in this process: fake_spectra once (it compiles nothing), then
    the SkewerJAX entries: the warm-up on the first batch, and one timed run over every batch. Needs JAX configured by
    the caller (``skewerjax.config.configure_devices``, 32 OpenMP threads for fake_spectra). Returns the record."""
    import jax

    from skewerjax import config
    from skewerjax.core import LineBundle
    from skewerjax.simulations import ArepoHDF5Reader

    jax.config.update("jax_default_device", jax.devices("cpu")[0])  # host arrays on the host; the GPU only for its tau sum
    t_start = time.perf_counter()
    snap_base = Path(data_dir) / c["sim"]
    reader = ArepoHDF5Reader(snap_base, SNAP)
    H = reader.header
    L = H.box_cMpc
    X, Z = nested_grid(c["n"], L)
    cofm = np.column_stack([X, np.zeros_like(X), Z])
    iref = reference_index(c["n"])
    nbins, dv = H.pixel_grid(c["dv_kms"])
    v = (np.arange(nbins) + 0.5) * dv  # [km/s], fake_spectra's pixels over [0, v_max)
    lines = LineBundle(list(LINE_IDS), ref_line_id="HI-LyA")
    snapdir = Path(reader.files[0]).parent
    cpu, gpu = config.devices("cpu"), config.devices("gpu")
    entries = [e for e in c["entries"] if e != GPU_DEF or gpu]
    rec = dict(
        id=c["id"],
        config=c,
        status="ok",
        machine=machine_metadata(),
        gpu=[d.device_kind for d in gpu],
        precision=config.get_precision(),
        snapshot=dict(n_gas=int(H.n_gas), box_cMpc=float(L), n_files=int(H.n_files), redshift=float(H.redshift)),
        pixels=dict(nbins=int(nbins), dv_kms=float(dv)),
        reference=dict(index=iref, x_cMpc=float(X[iref]), z_cMpc=float(Z[iref]), flux={}),
        entries={},
        cache={},
        date=time.strftime("%Y-%m-%d %H:%M"),
    )
    tau_fs = None
    if FS in entries:
        rec["cache"][FS] = cache_state(snapdir)
        t0 = time.perf_counter()
        tau_fs, st, nb_fs, dv_fs = run_fake_spectra(snap_base, X, Z, H, c["dv_kms"])
        st = {s: float(st.get(s, 0.0)) for s in STAGES}
        rec["entries"][FS] = dict(stages=st, total=float(sum(st.values())), wall=time.perf_counter() - t0, jit_s=0.0)
        if (nb_fs, round(dv_fs, 9)) != (nbins, round(dv, 9)):
            raise RuntimeError(f"fake_spectra's pixels ({nb_fs} of {dv_fs}) differ from SkewerJAX's ({nbins} of {dv}).")
        rec["reference"]["flux"][FS] = {ln: np.exp(-tau_fs[j, iref]).tolist() for j, ln in enumerate(LINE_IDS)}
    sj = [e for e in entries if e != FS]
    if sj:
        clock = RunClock()
        parts = batches(c["n_skewers"], c["spacing_kms"])
        rec["cache"]["SkewerJAX"] = cache_state(snapdir)
        warm, _ = skewerjax_batch(reader, cofm[parts[0]], v, lines, sj, clock, 0, cpu, gpu, c["spacing_kms"])
        first = {e: {s: r[0] for s, r in st.items()} for e, st in clock.t.items()}
        del warm
        agree, res = _Agreement(), []
        batch0 = {}
        for b, sl in enumerate(parts):
            tau, rep = skewerjax_batch(reader, cofm[sl], v, lines, sj, clock, 1, cpu, gpu, c["spacing_kms"])
            res.append(rep)
            if b == 0:
                batch0 = {e: {s: r[1] for s, r in st.items()} for e, st in clock.t.items()}
            F = {e: np.exp(-t.astype(np.float64)) for e, t in tau.items()}
            if tau_fs is not None:
                Ffs = np.exp(-tau_fs[:, sl])
                for j, ln in enumerate(LINE_IDS):
                    agree.add(("default_vs_fs", ln), F[CPU_DEF][j], Ffs[j])
                    if CPU_MAT in F:
                        agree.add(("matched_vs_fs", ln), F[CPU_MAT][j], Ffs[j])
            if GPU_DEF in F:
                for j, ln in enumerate(LINE_IDS):
                    agree.add(("gpu_vs_cpu", ln), F[GPU_DEF][j], F[CPU_DEF][j])
            if sl.start <= iref < sl.stop:
                for e, Fe in F.items():
                    rec["reference"]["flux"][e] = {ln: Fe[j, iref - sl.start].tolist() for j, ln in enumerate(LINE_IDS)}
            del tau, F
        for e in sj:  # the host stages of the GPU entry are those of the default CPU entry; the shared ones of the matched
            for s in STAGES:
                if s not in clock.t.get(e, {}):
                    clock.share(CPU_DEF, s, e)
        med = clock.medians()  # one timed run: its value
        for e in sj:
            if e not in clock.t:
                continue
            st = {s: float(med[e].get(s, 0.0)) for s in STAGES}
            jit = {s: float(first[e][s] - batch0[e][s]) for s in ("pieces", "tau") if s in first.get(e, {})}
            rec["entries"][e] = dict(stages=st, total=float(sum(st.values())), jit_s=jit.get("tau", 0.0), jit=jit, warmup_batch=first.get(e, {}))
        rec["batches"] = dict(n=len(parts), size=parts[0].stop - parts[0].start)
        rec["resolution"] = dict(
            max_ratio=[max(r["max_ratio"][j] for r in res) for j in range(len(LINE_IDS))],
            column_fraction=[max(r["column_fraction"][j] for r in res) for j in range(len(LINE_IDS))],
            warned=any(r["warned"] for r in res),
        )
        rec["agreement"] = {f"{k[0]}:{k[1]}": v_ for k, v_ in agree.result().items()}
    rec["wall_s"] = time.perf_counter() - t_start
    rec["jax"] = jax.__version__
    return rec


# --------------------------------------------------------------------------- records
def load_records(folder) -> dict:
    """{id: record} of the JSON records in ``folder``."""
    return {json.loads(p.read_text())["id"]: json.loads(p.read_text()) for p in sorted(Path(folder).glob("*.json"))}


def block_records(records: dict, block: str, n_max: int) -> list:
    """The records of one block, in the order of its parameter (``scaling_points``); missing points are left out."""
    return [records[c["id"]] for c in scaling_points(n_max, (block,)) if c["id"] in records]


def block_parameter(block: str, rec: dict) -> float:
    """The parameter of a block at a record: the cells of the snapshot (A), the skewers (B), the pixel width (C1), the
    point spacing (C2)."""
    c = rec["config"]
    return {"A": rec["snapshot"]["n_gas"], "B": c["n_skewers"], "C1": c["dv_kms"], "C2": c["spacing_kms"]}[block]


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
    ``out/logs/<id>.log``."""
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

    _config.configure_devices(n_cpu_devices=1, platforms=os.environ.get("JAX_PLATFORMS", "cuda,cpu"))
    main()
