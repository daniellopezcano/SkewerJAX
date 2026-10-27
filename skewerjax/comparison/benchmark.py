"""Wall time and memory of SkewerJAX against fake_spectra on identical skewers (Lya and Si III 1206), with scaling in
the number of skewers, the device and the size of the simulation.

One configuration (code, snapshot, number of skewers, platform, threads, SkewerJAX modes) runs in its own process, so
that the peak memory is per configuration and the CPU set / thread count is fixed before any library starts its
thread pool:

    python skewerjax/comparison/benchmark.py worker --config '<json>' --out DIR

(run as a file, so that a fake_spectra worker does not import JAX through the ``skewerjax`` package). The launcher is
``scripts/benchmarks/benchmark_fake_spectra.py``; ``benchmark_sets`` defines its configurations.

**Protocol.** Each worker runs ``1 + repeats`` times. Run 0 is the *first run*: it includes SkewerJAX's JIT
compilation and reads the snapshot from a cold page cache; it is kept apart. The timed runs are the others (median and
min-max), with compilation excluded: ``compile_s`` = first-run mode time - median timed mode time. Every JAX result is
pulled to the host (``np.asarray``), which blocks until it is ready. A worker stops adding runs when the next one would
exceed ``budget_s`` (at least one timed run).

**Stages** (exclusive wall time, ``StageTimer``):

* ``setup``: the reader / fake_spectra's ``Spectra`` object;
* ``io``: raw HDF5 reads (SkewerJAX ``ArepoHDF5Reader.read_rows``, fake_spectra ``snapshot_set.get_data``);
* ``selection``: the cells near the skewers (SkewerJAX ``select_cells_near_paths``, streamed over the snapshot;
  fake_spectra ``particles_near_lines``);
* ``physics``: per-cell gas physics (ion fractions, temperatures; SkewerJAX ``SimulationSkewers.properties``; for
  fake_spectra, the rest of ``get_tau``);
* ``pieces``: the deposition of the cells on the skewers and the pieces the kernel sums (SkewerJAX: absorbers,
  ``absorber_pieces``, the matched mode's chords; fake_spectra does this inside its C engine, counted in ``tau``);
* ``tau``: the optical-depth sum (SkewerJAX ``compute_optical_depth`` and the matched mode's per-cell sums;
  fake_spectra's C engine ``_do_interpolation_work``).

SkewerJAX shares ``setup``, ``io`` and ``selection`` between its modes (one selection per run); each mode then runs
on a fresh ``SimulationSkewers`` holding the same cells (no cache shared between modes). Modes (``MODES``): ``matched``
(fake_spectra's scheme on its pixel grid), ``default``, ``float32``, ``crossings``.

**Memory.** Host: the peak RSS of each SkewerJAX mode (sampled every 20 ms, from the selection's cells upwards) and
of the whole process (``ru_maxrss``); the launcher also samples the RSS of the worker and stops it when it exceeds
``mem_cap_GB`` or when the machine's available memory falls below ``mem_floor_GB``. GPU: XLA's ``peak_bytes_in_use``
after each mode (cumulative over the process: modes run matched, default, float32, crossings), with
``XLA_PYTHON_CLIENT_PREALLOCATE=false``.

**Agreement.** SkewerJAX's matched mode against fake_spectra's tau on the first ``n_store`` skewers (max and rms
|dF|), and the mean flux of every mode against fake_spectra's on all skewers.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

if __name__ == "__main__" and Path(sys.path[0]).resolve() == Path(__file__).resolve().parent:
    sys.path.pop(0)  # run as a file: no shadowing by the sibling modules

import numpy as np  # noqa: E402

STAGES = ("setup", "io", "selection", "physics", "pieces", "tau")
SHARED_STAGES = ("setup", "io", "selection")  # once per SkewerJAX run, whatever the modes
SPECIES = {"HI-LyA": ("H", 1, 1215), "SiIII-1206": ("Si", 3, 1206)}
MODES = {
    "matched": "fake_spectra's scheme in SkewerJAX (FAKE_SPECTRA_EXTRACTION + FAKE_SPECTRA_INTEGRATION), its pixel grid",
    "default": "SkewerJAX's default: Voronoi pieces of 0.25 km/s dual segments, exact frame, float64",
    "float32": "the default with the float32 kernel",
    "crossings": "the default with deposition='voronoi_crossings' (one absorber per sub-piece of each cell crossing)",
}
CPU_SETS = {1: [0], 4: [0, 2, 4, 6], 8: list(range(0, 16, 2)), 16: list(range(16)), 32: list(range(32))}
"""Logical CPUs per thread count on the benchmark machine (i9-13900KF: CPUs 0-15 are the 8 P-cores with their
hyperthreads, 16-31 the E-cores): 4 and 8 use one thread per P-core."""
STOPPED = ("timeout", "memory_limit", "skipped")  # a series stops at the first of these


# --------------------------------------------------------------------------- skewers, timers, memory
def benchmark_skewers(n: int, box_code: float = 35000.0, seed: int = 2330):
    """``n`` straight skewers (cofm [code units], axis 1/2/3): the first m of any n are the same (nested sets)."""
    rng = np.random.default_rng(seed)
    cofm, axis = rng.uniform(0.0, box_code, size=(4096, 3)), rng.integers(1, 4, size=4096).astype(np.int32)
    return cofm[:n].copy(), axis[:n].copy()


class StageTimer:
    """Exclusive wall time per stage: entering a stage pauses the enclosing one."""

    def __init__(self):
        self.t = dict.fromkeys(STAGES, 0.0)
        self._stack = []

    @contextmanager
    def stage(self, name):
        now = time.perf_counter()
        if self._stack:
            self.t[self._stack[-1][0]] += now - self._stack[-1][1]
        self._stack.append([name, now])
        try:
            yield
        finally:
            name, t0 = self._stack.pop()
            now = time.perf_counter()
            self.t[name] += now - t0
            if self._stack:
                self._stack[-1][1] = now

    @contextmanager
    def patched(self, *targets):
        """Time every call of ``obj.attr`` as stage ``name`` for each ``(obj, attr, name)`` (an object or a module:
        an instance attribute shadows a method, a module attribute is replaced), restored on exit."""
        saved = []
        for obj, attr, name in targets:
            f, own = getattr(obj, attr), attr in vars(obj)

            def timed(*a, _f=f, _name=name, **k):
                with self.stage(_name):
                    return _f(*a, **k)

            saved.append((obj, attr, f, own))
            setattr(obj, attr, timed)
        try:
            yield self
        finally:
            for obj, attr, f, own in reversed(saved):
                setattr(obj, attr, f) if own else delattr(obj, attr)


def peak_rss_GB() -> float:
    """Peak resident memory of this process [GB] (Linux ``ru_maxrss`` is in kB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def rss_GB() -> float:
    """Current resident memory of this process [GB] (``/proc/self/statm``)."""
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1e9


class RssSampler:
    """Peak RSS of this process since the last ``reset``, sampled every ``dt`` seconds in a daemon thread."""

    def __init__(self, dt: float = 0.02):
        self.dt, self.peak, self._stop = dt, rss_GB(), threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while not self._stop.wait(self.dt):
            self.peak = max(self.peak, rss_GB())

    def reset(self) -> None:
        self.peak = rss_GB()

    def read(self) -> float:
        return max(self.peak, rss_GB())

    def stop(self) -> None:
        self._stop.set()


def machine_metadata() -> dict:
    """Hardware and software of this process (CPU set, versions, thread environment)."""
    import importlib.metadata as md

    cpu = next((ln.split(":", 1)[1].strip() for ln in open("/proc/cpuinfo") if ln.startswith("model name")), platform.processor())
    cores = len({ln.split(":", 1)[1].strip() for ln in open("/proc/cpuinfo") if ln.startswith("core id")}) or None
    mem = next((int(ln.split()[1]) / 1e6 for ln in open("/proc/meminfo") if ln.startswith("MemTotal")), None)
    vers = {}
    for p in ("jax", "jaxlib", "jax-cuda12-plugin", "numpy", "scipy", "h5py", "fake_spectra", "exojax"):
        try:
            vers[p] = md.version(p)
        except md.PackageNotFoundError:
            vers[p] = None
    gpu = None
    try:
        gpu = (
            subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=20,
            ).stdout.strip()
            or None
        )
    except (OSError, subprocess.SubprocessError):
        pass
    env = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "XLA_FLAGS",
        "JAX_PLATFORMS",
        "XLA_PYTHON_CLIENT_PREALLOCATE",
    )
    return dict(
        cpu=cpu,
        physical_cores=cores,
        logical_cpus=os.cpu_count(),
        cpu_set=sorted(os.sched_getaffinity(0)),
        ram_GB=mem,
        gpu=gpu,
        python=platform.python_version(),
        versions=vers,
        host=platform.node(),
        env={k: os.environ.get(k) for k in env},
    )


def snapshot_info(snap_base, snapnum: int) -> dict:
    """Gas cells, box, redshift, files and size on disk of one snapshot (from the header of its first file)."""
    import h5py

    d = Path(snap_base) / f"snapdir_{int(snapnum):03d}"
    files = sorted(d.glob(f"snap_{int(snapnum):03d}.*.hdf5"), key=lambda p: int(p.name.split(".")[1]))
    with h5py.File(files[0], "r") as f:
        h = f["Header"].attrs
        out = dict(
            sim=Path(snap_base).name,
            snapnum=int(snapnum),
            n_gas=int(h["NumPart_Total"][0]) + (int(h["NumPart_Total_HighWord"][0]) << 32),
            box_ckpc_h=float(h["BoxSize"]),
            redshift=float(h["Redshift"]),
            n_files=len(files),
        )
    out["size_GB"] = sum(p.stat().st_size for p in files) / 1e9
    return out


def config_id(c: dict) -> str:
    sim = Path(c["snap_base"]).name
    tail = "" if c["code"] == "fake_spectra" else f"_{c['platform']}_{'+'.join(c['modes'])}"
    return f"{c['code']}_{sim}_s{c['snapnum']}_n{c['n_skewers']}_t{c['threads']}{tail}"


def series_key(c: dict) -> tuple:
    """A configuration without its skewer count: the points of one scaling series."""
    return (c["code"], c["snap_base"], c["snapnum"], c["platform"], c["threads"], tuple(c.get("modes") or ()))


# --------------------------------------------------------------------------- the two workers
def _keep_running(reps: list, t_start: float, budget_s: float) -> bool:
    """One more run? Always one timed run after the first; then only while the next one fits in the budget."""
    return len(reps) < 2 or time.perf_counter() - t_start + reps[-1]["wall"] <= budget_s


def run_fake_spectra(c: dict, out: Path) -> dict:
    """fake_spectra: a new ``Spectra`` object per run, ``get_tau`` per species (the C engine is OpenMP)."""
    import shutil

    from fake_spectra import spectra

    cofm, axis = benchmark_skewers(c["n_skewers"])
    reps, taus, t_start = [], {}, time.perf_counter()
    work = out / "fs_work" / config_id(c)
    for r in range(1 + c["repeats"]):
        T = StageTimer()
        t0 = time.perf_counter()
        with T.stage("setup"):
            sp = spectra.Spectra(
                int(c["snapnum"]),
                str(c["snap_base"]),
                cofm.copy(),
                axis.copy(),
                res=c["res"],
                savefile="b.hdf5",
                savedir=str(work),
                reload_file=True,
                quiet=True,
            )
        with T.patched(
            (sp.snapshot_set, "get_data", "io"), (sp, "particles_near_lines", "selection"), (sp, "_do_interpolation_work", "tau")
        ):
            for name, (el, st, ll) in SPECIES.items():
                with T.stage("physics"):
                    taus[name] = np.asarray(sp.get_tau(el, st, ll))
        reps.append(dict(stages=dict(T.t), wall=time.perf_counter() - t0))
        nbins, dv = int(sp.nbins), float(sp.dvbin)
        del sp
        if not _keep_running(reps, t_start, c["budget_s"]):
            break
    shutil.rmtree(work, ignore_errors=True)
    np.savez(
        out / f"{config_id(c)}.npz",
        **{f"tau_{k}": v[: c["n_store"]] for k, v in taus.items()},
        **{f"meanF_{k}": np.exp(-v).mean(axis=1) for k, v in taus.items()},
    )
    return dict(runs=reps, peak_rss_GB=peak_rss_GB(), n_v=nbins, dv_kms=dv)


def _mode_tau(mode: str, sk, ion, lb, c: dict, T: StageTimer):
    """tau (N_sk, N_v) of one species in one SkewerJAX mode, timed by stage; also the number of absorbers with gas
    (all skewers; None for the matched mode, whose chords are built inside its tau)."""
    from skewerjax.comparison.effects import FAKE_SPECTRA_EXTRACTION, FAKE_SPECTRA_INTEGRATION
    from skewerjax.comparison.emulation import matched_tau
    from skewerjax.core import absorbers as ab_
    from skewerjax.simulations import ExtractionOptions

    if mode == "matched":
        with T.stage("physics"):
            sk.properties(ion, FAKE_SPECTRA_EXTRACTION)
        v, _ = sk.spectral_grid(c["res"])  # fake_spectra's periodic pixel grid
        with T.stage("tau"):  # chords and pieces are timed inside, as "pieces"
            return matched_tau(sk, ion, lb, v, FAKE_SPECTRA_EXTRACTION, FAKE_SPECTRA_INTEGRATION), None
    ext = ExtractionOptions(deposition="voronoi_crossings") if mode == "crossings" else ExtractionOptions()
    with T.stage("physics"):
        sk.properties(ion, ext)
    with T.stage("pieces"):
        fr = sk.frame("exact")
        v, _ = sk.spectral_grid(c["res"], fr)
        A = sk.absorbers(ion, ext, frame=fr)
    with T.stage("tau"):  # periodic, as fake_spectra
        tau = ab_.tau_absorbers(A, lb, v, fr, velocity_period=True, check=None, dtype="float32" if mode == "float32" else None)[0]
    return tau, int((np.asarray(A.n) > 0).sum())


def run_skewerjax(c: dict, out: Path) -> dict:
    """SkewerJAX: per run, one reader and one selection; then every mode of ``c["modes"]`` on the same cells."""
    import jax

    jax.config.update("jax_enable_x64", True)
    from skewerjax.comparison import emulation as em
    from skewerjax.core import LineBundle
    from skewerjax.core import absorbers as ab_
    from skewerjax.simulations import HI, ArepoHDF5Reader, Ion, SimulationSkewers

    ions = {"HI-LyA": HI, "SiIII-1206": Ion("Si", 3)}
    lbs = {k: LineBundle([k], ref_line_id=k) for k in SPECIES}
    cofm, axis = benchmark_skewers(c["n_skewers"])
    dev = jax.devices()[0]
    gpu = dev.platform == "gpu"
    sampler = RssSampler()
    reps, taus, info, t_start = [], {}, {}, time.perf_counter()
    for r in range(1 + c["repeats"]):
        T = StageTimer()
        t0 = time.perf_counter()
        with T.stage("setup"):
            rd = ArepoHDF5Reader(c["snap_base"], int(c["snapnum"]))
        with T.patched((rd, "read_rows", "io")), T.stage("selection"):
            cells = SimulationSkewers.axis_aligned(rd, cofm, axis, c["spacing_kms"]).cells
        rep = dict(shared={s: T.t[s] for s in SHARED_STAGES}, modes={})
        for m in c["modes"]:
            Tm = StageTimer()
            sampler.reset()
            tm = time.perf_counter()
            sk = SimulationSkewers.axis_aligned(rd, cofm, axis, c["spacing_kms"], cells=cells)  # fresh caches, same cells
            n_abs = {}
            with Tm.patched((rd, "read_rows", "io"), (ab_, "absorber_pieces", "pieces"), (em, "fake_chord_absorbers", "pieces")):
                for name, ion in ions.items():
                    taus[(m, name)], n_abs[name] = _mode_tau(m, sk, ion, lbs[name], c, Tm)
            rep["modes"][m] = dict(
                stages={s: Tm.t[s] for s in STAGES if s not in ("setup", "selection")}, wall=time.perf_counter() - tm
            )
            if r == 0:
                info[m] = dict(
                    n_absorbers=n_abs,
                    peak_rss_GB=sampler.read(),
                    gpu_peak_GB=(dev.memory_stats() or {}).get("peak_bytes_in_use", 0) / 1e9 if gpu else None,
                )
            else:
                info[m]["peak_rss_GB"] = max(info[m]["peak_rss_GB"], sampler.read())
        rep["wall"] = time.perf_counter() - t0
        rep["n_cells"] = len(cells)
        reps.append(rep)
        del cells, sk
        if not _keep_running(reps, t_start, c["budget_s"]):
            break
    sampler.stop()
    for m in c["modes"]:
        warm = [x["modes"][m]["wall"] for x in reps[1:]]
        info[m]["compile_s"] = reps[0]["modes"][m]["wall"] - float(np.median(warm)) if warm else None
    np.savez(
        out / f"{config_id(c)}.npz",
        **{f"tau_{m}_{k}": v[: c["n_store"]] for (m, k), v in taus.items()},
        **{f"meanF_{m}_{k}": np.exp(-v).mean(axis=1) for (m, k), v in taus.items()},
    )
    res = dict(runs=reps, modes=info, peak_rss_GB=peak_rss_GB(), devices=[str(d) for d in jax.devices()])
    if gpu:
        res["device_memory"] = {
            k: v for k, v in (dev.memory_stats() or {}).items() if k in ("peak_bytes_in_use", "bytes_in_use", "bytes_limit")
        }
    res["agreement"] = agreement(c, out, taus)
    return res


def agreement(c: dict, out: Path, taus: dict) -> Optional[dict]:
    """Against the fake_spectra result of the same snapshot and skewer count (None if it does not exist yet): the
    matched mode's max and rms |dF| on the stored skewers, and the mean flux of every mode on all skewers."""
    ref = out / f"{config_id(dict(c, code='fake_spectra', platform='cpu', threads=max(CPU_SETS), modes=None))}.npz"
    if not ref.exists():
        return None
    fs = np.load(ref)
    rows = {}
    for name in SPECIES:
        row = dict(meanF_fake_spectra=float(fs[f"meanF_{name}"].mean()))
        for m in c["modes"]:
            row[f"meanF_{m}"] = float(np.exp(-taus[(m, name)]).mean(axis=1).mean())
        if ("matched", name) in taus:
            k = min(c["n_store"], c["n_skewers"])
            dF = np.exp(-taus[("matched", name)][:k]) - np.exp(-fs[f"tau_{name}"][:k])
            row.update(n_skewers=k, max_dF=float(np.abs(dF).max()), rms_dF=float(np.sqrt(np.mean(dF**2))))
        rows[name] = row
    return rows


def worker(c: dict, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    res = (run_fake_spectra if c["code"] == "fake_spectra" else run_skewerjax)(c, out)
    rec = dict(
        id=config_id(c),
        config=c,
        status="ok",
        wall_s=time.perf_counter() - t0,
        metadata=machine_metadata(),
        snapshot=snapshot_info(c["snap_base"], c["snapnum"]),
        **res,
    )
    (out / f"{config_id(c)}.json").write_text(json.dumps(rec, indent=1))


# --------------------------------------------------------------------------- configurations
def make_config(
    code: str,
    snap_base,
    snapnum: int,
    n_skewers: int,
    *,
    platform: str = "cpu",
    threads: int = 32,
    modes=("matched", "default"),
    repeats: int = 3,
    budget_s: float = 1800.0,
    res: float = 1.0,
    spacing_kms: float = 0.25,
    n_store: int = 64,
    preset: str = "full",
    cap_s: Optional[float] = None,
    est_s: Optional[float] = None,
) -> dict:
    """One configuration. ``repeats``: timed runs after the first (fewer when ``budget_s`` [s] is reached);
    ``modes``: SkewerJAX's modes, in order (ignored for fake_spectra); ``res``: pixel width [km/s];
    ``spacing_kms``: SkewerJAX's point spacing [km/s]; ``n_store``: skewers whose tau is saved for the agreement;
    ``preset``: the preset it belongs to (``PRESETS``); ``cap_s``: its own wall-time cap [s] (default: the
    launcher's); ``est_s``: its estimated wall time [s] (the budget guard of ``run_suite``)."""
    if code == "fake_spectra":
        platform, modes = "cpu", None
    elif unknown := set(modes) - set(MODES):
        raise ValueError(f"unknown modes {unknown}; known: {list(MODES)}")
    return dict(
        code=code,
        snap_base=str(snap_base),
        snapnum=int(snapnum),
        n_skewers=int(n_skewers),
        platform=platform,
        threads=int(threads),
        modes=None if modes is None else list(modes),
        repeats=int(repeats),
        budget_s=float(budget_s),
        res=float(res),
        spacing_kms=float(spacing_kms),
        n_store=int(n_store),
        preset=preset,
        cap_s=None if cap_s is None else float(cap_s),
        est_s=None if est_s is None else float(est_s),
    )


SIMULATIONS = ("TNG50-4", "TNG50-3", "TNG50-2")
N_SKEWERS = (1, 4, 16, 64, 256, 1024)
PRESETS = {
    "full": "the whole study: N = 1 ... 1024 in TNG50-4, -3, -2, and 1-16 threads at N = 16 (~10 h or more; caps 1 h, 48 GB)",
    "quick": "a reduced study in <= 30 min of new runs: TNG50-3 and TNG50-2 at small N, 1-16 threads at N = 1, one or two "
    "timed runs, per-point caps (the TNG50-4 series of the full study is kept as it is)",
}
QUICK_BUDGET_S = 1800.0
# The quick preset's points, (sim, N, code, platform): estimated wall time and cap [s], timed runs. Estimates from the
# full study's records (TNG50-3, TNG50-4: setup + io + selection + the modes, per run) and the old suite's TNG50-2
# point (selection ~7.5x TNG50-3, ~2x the cells per skewer; fake_spectra's cold read of 349 GB ~250 s).
QUICK_SCALING = [
    ("TNG50-3", 1, "fake_spectra", "cpu", 90, 240, 2),
    ("TNG50-3", 1, "skewerjax", "cpu", 100, 300, 2),
    ("TNG50-3", 1, "skewerjax", "gpu", 120, 300, 3),
    ("TNG50-3", 4, "fake_spectra", "cpu", 110, 300, 3),
    ("TNG50-3", 4, "skewerjax", "cpu", 270, 600, 3),
    ("TNG50-3", 4, "skewerjax", "gpu", 280, 600, 3),
    ("TNG50-3", 16, "fake_spectra", "cpu", 110, 300, 3),
    ("TNG50-3", 16, "skewerjax", "cpu", 860, 1200, 3),
    ("TNG50-3", 16, "skewerjax", "gpu", 930, 1200, 3),
    ("TNG50-2", 1, "fake_spectra", "cpu", 420, 900, 1),
    ("TNG50-2", 1, "skewerjax", "cpu", 270, 600, 1),
    ("TNG50-2", 1, "skewerjax", "gpu", 270, 600, 1),
]
QUICK_THREADS = {1: (160, 400), 4: (45, 200), 8: (35, 200), 16: (30, 200)}  # SkewerJAX at N = 1: estimate, cap [s]


def benchmark_sets(name: str, data_dir, snapnum: int = 25, preset: str = "full") -> list:
    """The configurations of one set (fake_spectra before SkewerJAX at each point, for the agreement).

    ``preset="full"``:

    * ``scaling``: N = 1, 4, ..., 1024 skewers in TNG50-4, -3 and -2 (32 threads): fake_spectra, SkewerJAX on the
      CPU and on the GPU (modes matched + default, and float32 in TNG50-4); 3 timed runs (2 in TNG50-2); larger N
      stop at the limits;
    * ``threads``: TNG50-4, 16 skewers, 1 / 4 / 8 / 16 threads (32 is in ``scaling``); on one core SkewerJAX also
      runs float32 and crossings (the strict single-core comparison);
    * ``smoke``: TNG50-4, 1 and 4 skewers, one timed run (a quick check of the whole chain).

    ``preset="quick"`` (``QUICK_SCALING``, ``QUICK_THREADS``; each point with its own cap and estimate):

    * ``scaling``: TNG50-3 at N = 1, 4, 16 and TNG50-2 at N = 1 (fake_spectra, SkewerJAX CPU and GPU; points that
      the full study already measured are kept); the TNG50-4 series is the full study's;
    * ``threads``: TNG50-4, 1 skewer, 1 / 4 / 8 / 16 threads, one timed run;
    * ``smoke``: as in the full preset.
    """
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}: {list(PRESETS)}")
    sim = lambda s: Path(data_dir) / s
    if preset == "quick" and name == "scaling":
        return [
            make_config(code, sim(s), snapnum, n, platform=p, repeats=rep, preset="quick", cap_s=cap, est_s=est, budget_s=cap)
            for s, n, code, p, est, cap, rep in QUICK_SCALING
        ]
    if preset == "quick" and name == "threads":
        out = []
        for t, (est, cap) in QUICK_THREADS.items():
            modes = ("matched", "default", "float32", "crossings") if t == 1 else ("matched", "default")
            out += [
                make_config(
                    "fake_spectra",
                    sim("TNG50-4"),
                    snapnum,
                    1,
                    threads=t,
                    repeats=2,
                    preset="quick",
                    cap_s=120,
                    est_s=15,
                    budget_s=120,
                ),
                make_config(
                    "skewerjax",
                    sim("TNG50-4"),
                    snapnum,
                    1,
                    threads=t,
                    modes=modes,
                    repeats=1,
                    preset="quick",
                    cap_s=cap,
                    est_s=est,
                    budget_s=cap,
                ),
            ]
        return out
    if name == "scaling":
        out = []
        for s in SIMULATIONS:
            extra = ("float32",) if s == "TNG50-4" else ()
            rep = 2 if s == "TNG50-2" else 3  # each TNG50-2 run streams 349 GB
            for n in N_SKEWERS:
                out += [
                    make_config("fake_spectra", sim(s), snapnum, n, repeats=rep),
                    make_config("skewerjax", sim(s), snapnum, n, modes=("matched", "default") + extra, repeats=rep),
                    make_config(
                        "skewerjax", sim(s), snapnum, n, platform="gpu", modes=("matched", "default") + extra, repeats=rep
                    ),
                ]
        return out
    if name == "threads":
        out = []
        for t in (1, 4, 8, 16):
            modes = ("matched", "default", "float32", "crossings") if t == 1 else ("matched", "default")
            out += [
                make_config("fake_spectra", sim("TNG50-4"), snapnum, 16, threads=t),
                make_config("skewerjax", sim("TNG50-4"), snapnum, 16, threads=t, modes=modes, repeats=2),
            ]
        return out
    if name == "smoke":
        return [
            make_config(code, sim("TNG50-4"), snapnum, n, platform=p, repeats=1)
            for n in (1, 4)
            for code, p in (("fake_spectra", "cpu"), ("skewerjax", "cpu"), ("skewerjax", "gpu"))
        ]
    raise ValueError(f"unknown set {name!r}: scaling, threads, smoke")


# --------------------------------------------------------------------------- the launcher side
def worker_env(c: dict) -> dict:
    """Environment of a worker: the platform, fixed thread counts (OpenMP, BLAS; XLA by the CPU set, single-threaded
    on one core), no GPU preallocation, the CUDA libraries of the pip wheels."""
    root = str(Path(__file__).resolve().parents[2])
    t = str(c["threads"])
    env = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join([root] + [x for x in [os.environ.get("PYTHONPATH")] if x]),
        OMP_NUM_THREADS=t,
        OPENBLAS_NUM_THREADS=t,
        MKL_NUM_THREADS=t,
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        JAX_PLATFORMS="cuda" if c["platform"] == "gpu" else "cpu",
    )
    env.pop("XLA_FLAGS", None)
    if c["threads"] == 1:
        env["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
    if c["platform"] == "gpu":
        import site

        libs = sorted(str(p) for s in site.getsitepackages() for p in Path(s).glob("nvidia/*/lib"))
        env["LD_LIBRARY_PATH"] = ":".join(libs + [env.get("LD_LIBRARY_PATH", "")])
    return env


def gpu_available() -> bool:
    """Can a worker's environment start JAX on a GPU?"""
    env = worker_env(dict(threads=1, platform="gpu"))
    p = subprocess.run([sys.executable, "-c", "import jax; jax.devices('gpu')"], env=env, capture_output=True, timeout=120)
    return p.returncode == 0


def _record(c: dict, status: str, **kw) -> dict:
    return dict(id=config_id(c), config=c, status=status, **kw)


def run_config(c: dict, out, *, cap_s: float = 3600.0, mem_cap_GB: float = 48.0, mem_floor_GB: float = 16.0, log=print) -> dict:
    """One configuration in its own process (skipped if its result exists). The launcher samples the worker's RSS
    every 0.5 s and stops it (``memory_limit``) when it exceeds ``mem_cap_GB`` or when the machine's available
    memory falls below ``mem_floor_GB``; a run over ``cap_s`` is stopped as ``timeout``. A failed run is recorded as
    ``error`` (and run again next time)."""
    import psutil

    out = Path(out)
    f = out / f"{config_id(c)}.json"
    cap_s = c.get("cap_s") or cap_s
    if f.exists():
        rec = json.loads(f.read_text())
        if not _remeasure(rec, c):
            log(f"skip {f.name} ({rec.get('status')}, exists)")
            return rec
    (out / "logs").mkdir(parents=True, exist_ok=True)
    cpus = CPU_SETS.get(c["threads"], list(range(c["threads"])))
    cmd = [sys.executable, str(Path(__file__).resolve()), "worker", "--config", json.dumps(c), "--out", str(out)]
    log(f"run  {config_id(c)} ...")
    t0, peak, status = time.perf_counter(), 0.0, None
    with open(out / "logs" / f"{config_id(c)}.log", "w") as err:
        p = subprocess.Popen(cmd, env=worker_env(c), preexec_fn=lambda: os.sched_setaffinity(0, cpus), stdout=err, stderr=err)
        proc = psutil.Process(p.pid)
        while p.poll() is None:
            try:
                rss = sum(q.memory_info().rss for q in [proc] + proc.children(recursive=True)) / 1e9
            except psutil.Error:
                rss = 0.0
            peak = max(peak, rss)
            if rss > mem_cap_GB or psutil.virtual_memory().available / 1e9 < mem_floor_GB:
                status = "memory_limit"
            elif time.perf_counter() - t0 > cap_s:
                status = "timeout"
            if status:
                p.kill()
                p.wait()
                break
            time.sleep(0.5)
    wall = time.perf_counter() - t0
    if status:
        rec = _record(c, status, wall_s=wall, sampled_peak_rss_GB=peak, available_GB=psutil.virtual_memory().available / 1e9)
        f.write_text(json.dumps(rec, indent=1))
        log(f"     {status} after {wall:.0f} s (worker RSS {peak:.1f} GB)")
        return rec
    if p.returncode != 0 or not f.exists():
        tail = (out / "logs" / f"{config_id(c)}.log").read_text()[-4000:]
        rec = _record(c, "error", wall_s=wall, stderr=tail)
        f.write_text(json.dumps(rec, indent=1))
        log(f"     error ({tail.strip().splitlines()[-1] if tail.strip() else p.returncode})")
        return rec
    rec = json.loads(f.read_text())
    rec["sampled_peak_rss_GB"] = peak
    f.write_text(json.dumps(rec, indent=1))
    log(f"     done in {rec['wall_s']:.1f} s, peak RSS {rec['peak_rss_GB']:.2f} GB")
    return rec


def _remeasure(rec: dict, c: dict) -> bool:
    """Measure an existing record again? When it failed, or when the full preset asks for more than a record of the
    quick preset holds (more timed runs, or a point the quick preset's lower cap stopped)."""
    if rec.get("status") == "error":
        return True
    old = rec.get("config", {})
    if c.get("preset", "full") != "full" or old.get("preset", "full") == "full":
        return False
    return rec.get("status") != "ok" or old.get("repeats", 0) < c["repeats"]


def _predict(points: list, n: int) -> Optional[float]:
    """Linear extrapolation in N of the last two (N, y) points (None with fewer than two)."""
    if len(points) < 2:
        return None
    (n1, y1), (n2, y2) = points[-2:]
    return y2 + (y2 - y1) * (n - n2) / (n2 - n1)


def run_suite(
    configs,
    out,
    *,
    cap_s: float = 3600.0,
    mem_cap_GB: float = 48.0,
    mem_floor_GB: float = 16.0,
    total_budget_s: Optional[float] = None,
    log=print,
) -> list:
    """Every configuration in order; existing results are kept (``_remeasure``). Along a series (``series_key``, by
    increasing N), a point is skipped when a smaller N stopped at a limit, or when the extrapolation of the two
    previous points puts its first run plus one timed run over its cap (``c["cap_s"]``, else ``cap_s``), or its peak
    RSS over ``mem_cap_GB``. GPU configurations are skipped (``no_gpu``) when no GPU can be started. With
    ``total_budget_s``, a configuration is not started when the time spent so far plus its estimate (``c["est_s"]``)
    exceeds it (``budget``; not written, so a later launch measures it)."""
    out = Path(out)
    has_gpu = None
    done, recs = {}, []
    t_suite = time.perf_counter()
    for c in configs:
        key = series_key(c)
        prev = sorted(done.get(key, []), key=lambda r: r["config"]["n_skewers"])
        f = out / f"{config_id(c)}.json"
        exists = f.exists() and not _remeasure(json.loads(f.read_text()), c)
        rec = None
        spent = time.perf_counter() - t_suite
        if not exists and total_budget_s is not None and spent + (c.get("est_s") or 0.0) > total_budget_s:
            rec = _record(
                c,
                "budget",
                wall_s=0.0,
                reason=f"{spent:.0f} s spent + {c.get('est_s') or 0:.0f} s estimated > budget {total_budget_s:.0f} s",
            )
            log(f"skip {config_id(c)}: {rec['reason']}")
        if rec is None and c["platform"] == "gpu" and not exists:
            has_gpu = gpu_available() if has_gpu is None else has_gpu
            if not has_gpu:
                rec = _record(c, "no_gpu", wall_s=0.0, reason="no GPU available to JAX")
                log(f"skip {config_id(c)}: no GPU")
        if rec is None and not exists:
            cap_c = c.get("cap_s") or cap_s
            stopped = [r for r in prev if r["status"] in STOPPED and r["config"]["n_skewers"] < c["n_skewers"]]
            ok = [r for r in prev if r["status"] == "ok"]
            t_pts = [
                (
                    r["config"]["n_skewers"],
                    r["runs"][0]["wall"] + (r["runs"][1]["wall"] if len(r["runs"]) > 1 else r["runs"][0]["wall"]),
                )
                for r in ok
            ]
            m_pts = [(r["config"]["n_skewers"], r["peak_rss_GB"]) for r in ok]
            t_pred, m_pred = _predict(t_pts, c["n_skewers"]), _predict(m_pts, c["n_skewers"])
            reason = (
                f"N = {stopped[0]['config']['n_skewers']} stopped ({stopped[0]['status']})"
                if stopped
                else f"predicted {t_pred:.0f} s for the first + one timed run > cap {cap_c:.0f} s"
                if t_pred and t_pred > cap_c
                else f"predicted peak RSS {m_pred:.1f} GB > cap {mem_cap_GB:.0f} GB"
                if m_pred and m_pred > mem_cap_GB
                else None
            )
            if reason:
                rec = _record(c, "skipped", wall_s=0.0, reason=reason, predicted_s=t_pred, predicted_rss_GB=m_pred)
                f.write_text(json.dumps(rec, indent=1))
                log(f"skip {config_id(c)}: {reason}")
        if rec is None:
            rec = run_config(c, out, cap_s=cap_s, mem_cap_GB=mem_cap_GB, mem_floor_GB=mem_floor_GB, log=log)
        done.setdefault(key, []).append(rec)
        recs.append(rec)
    return recs


def load_results(out) -> list:
    """Every ``<id>.json`` record in ``out`` (any status)."""
    return [json.loads(p.read_text()) for p in sorted(Path(out).glob("*.json"))]


def collect(raw_dir, results_file, *, sets=(), launcher_wall_s: Optional[float] = None, launches=None) -> dict:
    """The records of ``raw_dir`` in one results file (what the performance notebook reads): records (each with its
    ``config["preset"]``), snapshots, machine, the benchmark's runtime (the sum of the configurations' wall times,
    whatever the resumptions) and per preset (``runtime_by_preset_s``), and the ``launches`` (preset, sets, wall time,
    date) kept from the previous results file plus this one."""
    recs = load_results(raw_dir)
    ok = [r for r in recs if r["status"] == "ok"]
    snaps = {r["snapshot"]["sim"]: r["snapshot"] for r in ok}
    by_preset = {}
    for r in recs:
        p = r["config"].get("preset", "full")
        by_preset[p] = by_preset.get(p, 0.0) + r.get("wall_s", 0.0)
    prev = json.loads(Path(results_file).read_text()).get("launches", []) if Path(results_file).exists() else []
    res = dict(
        created=time.strftime("%Y-%m-%d %H:%M"),
        sets=list(sets),
        machine=ok[0]["metadata"] if ok else None,
        snapshots=[snaps[s] for s in sorted(snaps, key=lambda s: snaps[s]["n_gas"])],
        runtime_s=sum(r.get("wall_s", 0.0) for r in recs),
        runtime_by_preset_s=by_preset,
        launcher_wall_s=launcher_wall_s,
        launches=prev + list(launches or []),
        records=recs,
    )
    Path(results_file).parent.mkdir(parents=True, exist_ok=True)
    Path(results_file).write_text(json.dumps(res, indent=1))
    return res


# --------------------------------------------------------------------------- tables of the results
def _stat(x) -> tuple:
    x = np.asarray(x, float)
    return float(np.median(x)), float(x.min()), float(x.max())


def result_rows(records) -> list:
    """One row per measured series point: (code, platform, mode) at (sim, N, threads), with the median, min and max
    over the timed runs of the total and of every stage [s], the first run, the compilation, memory and agreement.
    SkewerJAX's total of a mode = its shared stages + its own. Records that did not run give a row with ``status``."""
    rows = []
    for r in records:
        c = r["config"]
        base = dict(
            code=c["code"],
            platform=c["platform"],
            sim=Path(c["snap_base"]).name,
            n_skewers=c["n_skewers"],
            threads=c["threads"],
            status=r["status"],
            id=r["id"],
            preset=c.get("preset", "full"),
        )
        if r["status"] != "ok":
            for m in c.get("modes") or [None]:
                rows.append(dict(base, mode=m, reason=r.get("reason", ""), wall_s=r.get("wall_s", 0.0)))
            continue
        timed, first = r["runs"][1:] or r["runs"], r["runs"][0]
        snap = r["snapshot"]
        for m in c["modes"] or [None]:
            if m is None:
                st = {s: [x["stages"][s] for x in timed] for s in STAGES}
                tot, first_wall = [x["wall"] for x in timed], first["wall"]
                mem, extra = r["peak_rss_GB"], dict(compile_s=None, gpu_peak_GB=None, n_absorbers=None)
                agr = None
            else:
                st = {
                    s: [x["shared"][s] if s in SHARED_STAGES else x["modes"][m]["stages"].get(s, 0.0) for x in timed]
                    for s in STAGES
                }
                st["io"] = [a + x["modes"][m]["stages"].get("io", 0.0) for a, x in zip(st["io"], timed)]
                tot = [sum(x["shared"].values()) + x["modes"][m]["wall"] for x in timed]
                first_wall = sum(first["shared"].values()) + first["modes"][m]["wall"]
                info = r["modes"][m]
                mem, extra = (
                    info["peak_rss_GB"],
                    dict(compile_s=info["compile_s"], gpu_peak_GB=info["gpu_peak_GB"], n_absorbers=info["n_absorbers"]),
                )
                agr = r.get("agreement")
            rows.append(
                dict(
                    base,
                    mode=m,
                    n_timed=len(timed),
                    total=_stat(tot),
                    first=first_wall,
                    stages={s: _stat(v) for s, v in st.items()},
                    peak_rss_GB=mem,
                    process_rss_GB=r["peak_rss_GB"],
                    n_cells_selected=timed[0].get("n_cells"),
                    n_gas=snap["n_gas"],
                    snapshot=snap,
                    agreement=agr,
                    wall_s=r["wall_s"],
                    **extra,
                )
            )
    return rows


def select_rows(rows, **match) -> list:
    """The measured rows (``status == "ok"``) matching every ``match`` item."""
    return [r for r in rows if r["status"] == "ok" and all(r.get(k) == v for k, v in match.items())]


def value(rows, what: str = "total", **match) -> float:
    """The median of ``what`` (``total`` or a stage [s], ``first`` [s], ``compile_s`` [s], ``peak_rss_GB``,
    ``gpu_peak_GB``) of the one measured row matching ``match`` (NaN if there is none)."""
    rs = select_rows(rows, **match)
    if len(rs) != 1:
        return float("nan")
    r = rs[0]
    v = r["total"] if what == "total" else r["stages"][what] if what in STAGES else r.get(what)
    return float("nan") if v is None else float(v[0] if isinstance(v, (tuple, list)) else v)


def print_setup(res: dict) -> None:
    """Machine, software, thread settings, snapshots, runtime and the points that did not run, of a results file."""
    m = res["machine"] or {}
    print(
        f"CPU: {m.get('cpu')}, {m.get('physical_cores')} cores / {m.get('logical_cpus')} logical CPUs, {m.get('ram_GB', 0):.0f} GB RAM"
    )
    print(
        f"GPU: {m.get('gpu') or 'none'} | Python {m.get('python')} | "
        + ", ".join(f"{k} {v}" for k, v in (m.get("versions") or {}).items() if v)
    )
    cpus = lambda c: f"{c[0]}-{c[-1]}" if c == list(range(c[0], c[-1] + 1)) else ",".join(map(str, c))
    print(
        "threads: OMP / OpenBLAS / MKL = the configuration's count; process pinned to CPUs",
        "; ".join(f"{t} thr: {cpus(c)}" for t, c in CPU_SETS.items()),
    )
    print(
        "         XLA: the pinned CPU set (single-threaded Eigen on one core); fake_spectra: OpenMP; XLA_PYTHON_CLIENT_PREALLOCATE=false"
    )
    for s in res["snapshots"]:
        print(
            f"{s['sim']}: snapshot {s['snapnum']:03d} (z = {s['redshift']:.3f}), {s['n_gas']:,} gas cells, box {s['box_ckpc_h'] / 1e3:.0f} cMpc/h, "
            f"{s['n_files']} files, {s['size_GB']:.0f} GB"
        )
    recs = res["records"]
    n_ok = sum(r["status"] == "ok" for r in recs)
    print(
        f"{len(recs)} configurations, {n_ok} measured; benchmark runtime {res['runtime_s'] / 3600:.1f} h (sum over configurations), "
        f"results of {res['created']}"
    )
    for p, t in sorted((res.get("runtime_by_preset_s") or {}).items()):
        n_p = sum(r["config"].get("preset", "full") == p and r["status"] == "ok" for r in recs)
        print(f"  preset {p}: {n_p} measured configurations, {t / 60:.0f} min")
    for x in res.get("launches") or []:
        print(f"  launch {x['date']}: preset {x['preset']}, sets {','.join(x['sets'])}, {x['wall_s'] / 60:.1f} min")
    for r in recs:
        if r["status"] != "ok":
            print(
                f"  not measured: {r['id']}: {r['status']}"
                + (f" ({r['reason']})" if r.get("reason") else f" after {r.get('wall_s', 0):.0f} s")
            )


def print_rows(rows, **match) -> None:
    """Every measured row matching ``match``: median stages, total and first run [s], compilation [s], memory [GB],
    the number of timed runs, the spread of the total (max/min - 1) and the preset that measured it."""
    rs = sorted(
        select_rows(rows, **match),
        key=lambda r: (
            r["n_gas"],
            r["threads"],
            r["n_skewers"],
            r["code"] != "fake_spectra",
            r["platform"],
            list(MODES).index(r["mode"]) if r["mode"] else -1,
        ),
    )
    head = [
        "sim",
        "N",
        "thr",
        "code",
        "dev",
        "mode",
        "runs",
        *STAGES,
        "total",
        "spread",
        "first",
        "compile",
        "RSS",
        "GPU",
        "preset",
    ]
    out = [head]
    for r in rs:
        out.append(
            [
                r["sim"],
                r["n_skewers"],
                r["threads"],
                "fs" if r["code"] == "fake_spectra" else "sjx",
                r["platform"],
                r["mode"] or "",
                r["n_timed"],
                *[f"{r['stages'][s][0]:.3g}" for s in STAGES],
                f"{r['total'][0]:.4g}",
                f"{r['total'][2] / r['total'][1] - 1:.0%}",
                f"{r['first']:.4g}",
                "" if r["compile_s"] is None else f"{r['compile_s']:.3g}",
                f"{r['peak_rss_GB']:.2f}",
                "" if r["gpu_peak_GB"] is None else f"{r['gpu_peak_GB']:.2f}",
                r.get("preset", "full"),
            ]
        )
    w = [max(len(str(x)) for x in col) for col in zip(*out)]
    for row in out:
        print("  ".join(str(x).rjust(n) for x, n in zip(row, w)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=["worker"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    import logging
    import warnings

    logging.getLogger("jax._src.xla_bridge").setLevel(logging.CRITICAL)
    warnings.filterwarnings("ignore")
    worker(json.loads(a.config), Path(a.out))


if __name__ == "__main__":
    main()
