"""How much memory SkewerJAX and fake_spectra use, where and when (notebook 16; ``scripts/benchmarks/benchmark_memory.py``).

**Protocol.** One fresh process per (point, entry), with the configuration of notebook 14 (TNG50-2 snapshot 25 unless a
block varies it, Lya + Si III 1206, float32, 32 threads, the CPU as two JAX devices), the snapshot dropped from the page
cache before the measured run (``scaling.evict_page_cache``: cold). The entries:

* ``fake_spectra`` (native, timed and marked from outside: set-up, I/O, gas physics, the C engine), and
  ``fake_spectra, search only`` (the same run with ``tautail`` = 1e30: every particle stops after one pixel each way,
  so its C call is the search for the particles near each sightline plus a negligible engine);
* ``SkewerJAX default, CPU``, ``SkewerJAX default, GPU``, ``SkewerJAX matched, CPU``: the JIT warm-up of notebook 14 (the
  same kernels on TNG50-4, marked "warm-up"), then one pass over the snapshot with the entry's own physics configuration
  (``simulations.stream.stream_cells``) and its tiles (``tiles_optical_depth``).

**What is sampled** (``ProcessSampler``, a thread of the measured process, every 0.05 s): from ``/proc/self/status``
``RssAnon`` (anonymous memory: what the code allocates), ``RssFile`` (file-backed pages mapped by the process: the
memory-mapped reads, reclaimable), ``RssShmem`` and ``VmHWM`` (the peak resident set); ``MemAvailable`` of the system
(``/proc/meminfo``); the process's CPU time (``/proc/self/stat``: busy cores = its derivative, shown over 0.5 s); the
bytes it made the kernel read from the disk (``/proc/self/io`` ``read_bytes``); and, for a GPU entry, the process's
device memory from NVML (``nvidia-ml-py``: the CUDA context included). Stage markers are written by the code's hooks
(``stream_cells(marker=)``, patched fake_spectra methods); at each marker JAX's ``memory_stats()`` of the GPU
(``bytes_in_use``, ``peak_bytes_in_use``; ``XLA_PYTHON_CLIENT_PREALLOCATE=false``) is recorded too.

**Blocks** (``memory_points``): A, the simulation (TNG50-4, -3, -2 at 128^2 skewers, 1 km/s); B, the number of
skewers (TNG50-2, 16 to 65,536); C1, the pixel width (1 to 16 km/s at 128^2; fake_spectra, default CPU and GPU); D,
SkewerJAX's memory knobs at 128^2 TNG50-2 (``tile_size`` 256 / 4,096 for the default CPU and GPU entries; both physics
configurations in the pass of the default CPU entry; 8 and 16 pass threads). Records go to ``benchmarks/results/memory/``,
one JSON per (point, entry); finished points are skipped.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import numpy as np

from skewerjax.comparison.scaling import (
    AXIS,
    CPU_DEF,
    CPU_MAT,
    DEFAULT_CELLS,
    FS,
    GPU_DEF,
    LINE_IDS,
    SNAP,
    cache_state,
    evict_page_cache,
    mode_options,
    nested_grid,
    write_record,
)

FS_SEARCH = "fake_spectra, search only"
ENTRY_KEYS = {FS: "fs", FS_SEARCH: "fs_search", CPU_DEF: "cpu", GPU_DEF: "gpu", CPU_MAT: "matched"}
ENTRY_NAMES = {v: k for k, v in ENTRY_KEYS.items()}
STATUS_FIELDS = ("RssAnon", "RssFile", "RssShmem", "VmHWM", "VmRSS")
B_SIDES = (4, 16, 32, 64, 128, 256)  # block B: 16 to 65,536 skewers
TILE_SIZES = (256, 4096)  # block D: besides the default 1,024
PASS_THREADS = (8, 16)  # block D: besides the default (every core: 32)
MAX_SAMPLES = 6000  # a longer timeline is max-pooled to this many samples per field (the peaks are kept)
MACHINE_RAM_GB = 125.0  # this machine (i9-13900KF): 132 GB = 125 GiB
MACHINE_VRAM_GB = 12.0  # RTX 3060


# --------------------------------------------------------------------------- /proc
def read_status(pid="self") -> dict:
    """{field: bytes} of ``STATUS_FIELDS`` from ``/proc/<pid>/status``."""
    out = {}
    with open(f"/proc/{pid}/status") as f:
        for ln in f:
            k = ln.split(":", 1)[0]
            if k in STATUS_FIELDS:
                out[k] = int(ln.split()[1]) * 1024
    return out


def mem_available() -> int:
    """The system's available memory [bytes] (``MemAvailable`` of ``/proc/meminfo``)."""
    with open("/proc/meminfo") as f:
        for ln in f:
            if ln.startswith("MemAvailable:"):
                return int(ln.split()[1]) * 1024
    return -1


_TICK = os.sysconf("SC_CLK_TCK")


def cpu_seconds(pid="self") -> float:
    """The CPU time of a process, user + system, every thread [s] (``/proc/<pid>/stat``)."""
    with open(f"/proc/{pid}/stat") as f:
        parts = f.read().rsplit(")", 1)[1].split()
    return (int(parts[11]) + int(parts[12])) / _TICK


def io_read_bytes(pid="self") -> int:
    """The bytes the process made the kernel read from storage (``read_bytes`` of ``/proc/<pid>/io``), or -1."""
    try:
        with open(f"/proc/{pid}/io") as f:
            for ln in f:
                if ln.startswith("read_bytes:"):
                    return int(ln.split()[1])
    except OSError:
        pass
    return -1


class _NVML:
    """The device memory of this process on GPU 0 from NVML (the CUDA context included), or None without NVML."""

    def __init__(self):
        self.h = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self.nv = pynvml
            self.h = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.pid = os.getpid()
        except Exception:
            self.h = None

    def used(self) -> float:
        if self.h is None:
            return float("nan")
        try:
            for p in self.nv.nvmlDeviceGetComputeRunningProcesses(self.h):
                if p.pid == self.pid:
                    return float(p.usedGpuMemory or 0)
            return 0.0
        except Exception:
            return float("nan")


class ProcessSampler:
    """Samples this process every ``dt`` seconds in a daemon thread (module docstring): ``t`` [s since start],
    RssAnon, RssFile, RssShmem, VmHWM [bytes], MemAvailable [bytes], cpu [s, cumulative], read [bytes, cumulative], gpu
    [bytes, NVML; NaN without a GPU]. ``mark(name)`` records a stage marker (and JAX's GPU memory_stats when ``jax_gpu``
    devices are given)."""

    def __init__(self, dt: float = 0.05, gpu: bool = False, jax_gpu=None):
        self.dt = dt
        self.nvml = _NVML() if gpu else None
        self.jax_gpu = list(jax_gpu or [])
        self.t0 = time.perf_counter()
        self.cols = {k: [] for k in ("t", "anon", "file", "shmem", "hwm", "avail", "cpu", "read", "gpu")}
        self.markers, self.jax_stats = [], []
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._sample()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def now(self) -> float:
        return time.perf_counter() - self.t0

    def _sample(self):
        st = read_status()
        row = dict(
            t=self.now(),
            anon=st.get("RssAnon", -1),
            file=st.get("RssFile", -1),
            shmem=st.get("RssShmem", -1),
            hwm=st.get("VmHWM", -1),
            avail=mem_available(),
            cpu=cpu_seconds(),
            read=io_read_bytes(),
            gpu=self.nvml.used() if self.nvml is not None else float("nan"),
        )
        with self._lock:
            for k, v in row.items():
                self.cols[k].append(v)

    def _run(self):
        while not self._stop.wait(self.dt):
            self._sample()

    def mark(self, name: str):
        """A stage marker at this instant (and the JAX memory stats of the GPU, when given)."""
        t = self.now()
        self.markers.append((t, str(name)))
        for d in self.jax_gpu:
            try:
                ms = d.memory_stats()
                self.jax_stats.append(
                    dict(t=t, stage=str(name), bytes_in_use=ms.get("bytes_in_use"), peak_bytes_in_use=ms.get("peak_bytes_in_use"))
                )
            except Exception:
                pass

    def stop(self) -> dict:
        """Stop sampling; returns the timeline (``timeline``)."""
        self._stop.set()
        self._thread.join()
        self._sample()
        return self.timeline()

    def timeline(self, max_samples: int = MAX_SAMPLES) -> dict:
        """The samples as lists (bytes as GB, times in s), max-pooled to at most ``max_samples`` (peaks kept; the
        cumulative columns, cpu and read, keep their last value per bin)."""
        with self._lock:
            c = {k: np.asarray(v, dtype=np.float64) for k, v in self.cols.items()}
        n = c["t"].size
        if n > max_samples:
            edges = np.linspace(0, n, max_samples + 1).astype(np.int64)
            pool = lambda x, f: np.array([f(x[a:b]) for a, b in zip(edges[:-1], edges[1:])])
            c = {k: pool(v, (lambda y: y[-1]) if k in ("t", "cpu", "read") else np.nanmax) for k, v in c.items()}
        GB = 1e-9
        r = lambda x, d: [None if not np.isfinite(y) else round(float(y), d) for y in x]
        return dict(
            t=r(c["t"], 3),
            anon_GB=r(c["anon"] * GB, 4),
            file_GB=r(c["file"] * GB, 4),
            shmem_GB=r(c["shmem"] * GB, 4),
            hwm_GB=r(c["hwm"] * GB, 4),
            avail_GB=r(c["avail"] * GB, 3),
            cpu_s=r(c["cpu"], 2),
            read_GB=r(c["read"] * GB, 4),
            gpu_GB=r(c["gpu"] * GB, 4),
        )


def stage_peaks(timeline: dict, markers: list) -> dict:
    """{stage: {anon_GB, file_GB, gpu_GB, rss_GB: the peak within the stage; t0, t1}} of a sampled timeline, the stages
    between consecutive markers (the last one to the end)."""
    t = np.asarray(timeline["t"], dtype=float)
    out = {}
    marks = list(markers) + [(t[-1] + 1e-9, None)]
    for (t0, name), (t1, _) in zip(marks[:-1], marks[1:]):
        sel = (t >= t0) & (t <= t1)
        if not sel.any():
            sel = np.abs(t - t0) == np.abs(t - t0).min()

        def pk(key):
            x = np.asarray(timeline.get(key) or [np.nan] * t.size, dtype=float)[sel]
            return float(np.nanmax(x)) if np.any(np.isfinite(x)) else float("nan")

        rss = (
            np.asarray(timeline["anon_GB"], float)[sel]
            + np.asarray(timeline["file_GB"], float)[sel]
            + np.asarray(timeline["shmem_GB"], float)[sel]
        )
        prev = out.get(name)
        cur = dict(
            anon_GB=pk("anon_GB"),
            file_GB=pk("file_GB"),
            gpu_GB=pk("gpu_GB"),
            rss_GB=float(np.nanmax(rss)),
            t0=float(t0),
            t1=float(t1),
        )
        if prev is None:
            out[name] = cur
        else:  # a stage entered several times (fake_spectra's per-file stages): the peak over all, the total duration
            out[name] = {k: (max(prev[k], cur[k]) if k.endswith("GB") else prev[k]) for k in prev}
            out[name]["duration_s"] = prev.get("duration_s", prev["t1"] - prev["t0"]) + (t1 - t0)
    for v in out.values():
        v.setdefault("duration_s", v["t1"] - v["t0"])
    return out


def busy_cores(timeline: dict, window_s: float = 0.5) -> tuple:
    """(t [s], busy cores) of a timeline: the derivative of the process's CPU time over windows of ``window_s``."""
    t = np.asarray(timeline["t"], dtype=float)
    c = np.asarray(timeline["cpu_s"], dtype=float)
    edges = np.arange(t[0], t[-1] + window_s, window_s)
    ci = np.interp(edges, t, c)
    return 0.5 * (edges[1:] + edges[:-1]), np.diff(ci) / window_s


# --------------------------------------------------------------------------- the configurations
def point_id(
    sim: str, n: int, dv: float, entry: str, *, tile_size: int = 1024, physics: str = "own", threads: int | None = None
) -> str:
    knobs = (
        ("" if tile_size == 1024 else f"_tile{tile_size}")
        + ("" if physics == "own" else "_phys2")
        + ("" if threads is None else f"_thr{threads}")
    )
    return f"{sim}_n{n}_dv{dv:g}_{ENTRY_KEYS[entry]}{knobs}"


def memory_point(sim, n, dv, entry, *, tile_size=1024, physics="own", threads=None, blocks=()) -> dict:
    """One (point, entry): the configuration of one fresh process (``run_entry``)."""
    return dict(
        id=point_id(sim, n, dv, entry, tile_size=tile_size, physics=physics, threads=threads),
        sim=sim,
        snap=SNAP,
        n=int(n),
        n_skewers=int(n) ** 2,
        dv_kms=float(dv),
        entry=entry,
        tile_size=int(tile_size),
        physics=physics,
        threads=threads,
        blocks=list(blocks),
    )


def memory_points(blocks=("A", "B", "C1", "D"), n_max: int = 128) -> list:
    """Every (point, entry) of the blocks (module docstring), deduplicated (a point shared by blocks lists them all),
    in the order they run: B (small N first), A, C1, D."""
    pts = {}

    def add(p, b):
        if p["id"] in pts:
            pts[p["id"]]["blocks"].append(b)
        else:
            p["blocks"] = [b]
            pts[p["id"]] = p

    four = (FS, CPU_DEF, GPU_DEF, CPU_MAT)
    if "B" in blocks:
        for n in B_SIDES:
            for e in four:
                add(memory_point("TNG50-2", n, 1.0, e), "B")
        add(memory_point("TNG50-2", n_max, 1.0, FS_SEARCH), "B")
    if "A" in blocks:
        for sim in ("TNG50-4", "TNG50-3", "TNG50-2"):
            for e in four:
                add(memory_point(sim, n_max, 1.0, e), "A")
    if "C1" in blocks:
        for dv in (1.0, 2.0, 4.0, 8.0, 16.0):
            for e in (FS, CPU_DEF, GPU_DEF):
                add(memory_point("TNG50-2", n_max, dv, e), "C1")
    if "D" in blocks:
        for ts in TILE_SIZES:
            for e in (CPU_DEF, GPU_DEF):
                add(memory_point("TNG50-2", n_max, 1.0, e, tile_size=ts), "D")
        add(memory_point("TNG50-2", n_max, 1.0, CPU_DEF, physics="both"), "D")
        for th in PASS_THREADS:
            add(memory_point("TNG50-2", n_max, 1.0, CPU_DEF, threads=th), "D")
        add(memory_point("TNG50-2", n_max, 1.0, CPU_DEF), "D")
        add(memory_point("TNG50-2", n_max, 1.0, GPU_DEF), "D")
    return list(pts.values())


# --------------------------------------------------------------------------- one (point, entry)
def _fake_spectra(c, snap_base, X, Z, H, sampler, search_only: bool) -> dict:
    """fake_spectra on the skewers, its stages marked: set-up, I/O (``get_data``), the C engine
    (``_do_interpolation_work``), the rest of ``get_tau`` (gas physics). Returns the stage times and the run's facts."""
    import contextlib
    import io
    import tempfile

    from fake_spectra.spectra import Spectra

    from skewerjax.comparison.benchmark import StageTimer
    from skewerjax.comparison.scaling import FS_LINES

    cofm = np.column_stack([X, np.zeros_like(X), Z]) / H.code_length_cMpc
    axis = np.full(X.size, AXIS, dtype=np.int32)
    T = StageTimer()
    marked = []

    def mark_patch(obj, attr, name, back):
        f, own = getattr(obj, attr), attr in vars(obj)

        def w(*a, _f=f, **k):
            sampler.mark(name)
            try:
                with T.stage(name):
                    return _f(*a, **k)
            finally:
                sampler.mark(back)

        marked.append((obj, attr, f, own))
        setattr(obj, attr, w)

    with tempfile.TemporaryDirectory(prefix="fs_memory_") as work, contextlib.redirect_stdout(io.StringIO()):
        sampler.mark("setup")
        with T.stage("setup"):
            fs = Spectra(
                SNAP, str(snap_base), cofm, axis, res=c["dv_kms"], savefile="t.hdf5", savedir=work, reload_file=True, quiet=True
            )
        if search_only:
            fs.tautail = 1e30  # every particle stops after one pixel each way: the C call is the search
        mark_patch(fs.snapshot_set, "get_data", "io", "physics")
        mark_patch(fs, "_do_interpolation_work", "tau", "physics")
        sampler.mark("physics")
        try:
            with T.stage("physics"):
                taus = [np.asarray(fs.get_tau(*line)) for line in FS_LINES]
        finally:
            for obj, attr, f, own in reversed(marked):
                setattr(obj, attr, f) if own else delattr(obj, attr)
        sampler.mark("done")
    nb = int(fs.nbins)
    return dict(
        stages={k: float(v) for k, v in T.t.items()},
        nbins=nb,
        dvbin=float(fs.dvbin),
        tau_bytes=int(sum(t.nbytes for t in taus)),
        tau=taus,
    )


def _skewerjax(c, reader, data_dir, sampler, cpu, gpu) -> dict:
    """One SkewerJAX entry: the warm-up (marked), then the cold pass and tiles (module docstring)."""
    from skewerjax.comparison.benchmark import MATCHED_CELLS
    from skewerjax.comparison.scaling import skewerjax_point
    from skewerjax.core import LineBundle
    from skewerjax.core.absorbers import SnapshotFrame
    from skewerjax.simulations import ArepoHDF5Reader
    from skewerjax.simulations import stream as st

    H = reader.header
    L = H.box_cMpc
    e = c["entry"]
    lines = LineBundle(list(LINE_IDS), ref_line_id="HI-LyA")
    nb, d = H.pixel_grid(c["dv_kms"])
    v = (np.arange(nb) + 0.5) * d
    tile = int(c["tile_size"])
    sampler.mark("warm-up")
    t0 = time.perf_counter()
    warm = ArepoHDF5Reader(Path(data_dir) / "TNG50-4", SNAP)
    nw = min(int(c["n"]), int(np.sqrt(tile)))
    Xw, Zw = nested_grid(nw, warm.header.box_cMpc)
    skewerjax_point(
        warm, np.column_stack([Xw, np.zeros_like(Xw), Zw]), {f"{c['dv_kms']:g}": v}, lines, [e], cpu=cpu, gpu=gpu, tile_size=tile
    )
    jit_s = time.perf_counter() - t0
    snapdir = Path(reader.files[0]).parent
    evict_page_cache(snapdir)
    cache = cache_state(snapdir)
    sampler.mark("setup")
    t_run = time.perf_counter()
    X, Z = nested_grid(c["n"], L)
    cofm = np.column_stack([X, np.zeros_like(X), Z])
    paths, frame = st.box_skewers(cofm, AXIS, L, H.cosmology(), H.a)
    tile_of = st.tile_skewers(paths, tile)
    mode = mode_options("matched" if e == CPU_MAT else "default", nb, float(d), H)
    cells_opts = {"default": DEFAULT_CELLS, "matched": MATCHED_CELLS}
    physics = {mode["config"]: cells_opts[mode["config"]]}
    if c.get("physics") == "both":
        physics = dict(cells_opts)
    store, pinfo = st.stream_cells(
        reader, paths, tile_of, list(lines.species_keys), physics, threads=c.get("threads"), marker=sampler.mark
    )
    sampler.mark("tiles")
    snap = SnapshotFrame(H.a, H.velfac_kms_per_cMpc, path_length=np.full(len(paths), L))
    devs = gpu if e == GPU_DEF else cpu
    taus, tinfo = st.tiles_optical_depth(
        store,
        mode["config"],
        paths,
        tile_of,
        snap if mode["frame"] == "snapshot" else frame,
        v,
        lines,
        pieces=mode["pieces"],
        engines={"tau": devs},
        **mode["tau"],
    )
    sampler.mark("done")
    t_end = time.perf_counter()
    rows = tinfo["rows"]
    off = tinfo["t0"] - sampler.t0  # the tiles' clock -> the sampler's
    tiles = [
        dict(
            n_skewers=r["n_skewers"],
            n_cells=r["n_cells"],
            n_pieces=r["n_pieces"],
            n_contributions=r["n_contributions"],
            contributions_bytes=r["contributions_bytes"],
            geometry_s=r["geometry_s"],
            contributions_s=r["contributions_s"],
            wait_s=r["wait_s"],
            tau_s=r["tau_s"]["tau"],
            prep_start=r["prep_start"] + off,
            prep_end=r["prep_end"] + off,
            prep_thread=r["prep_thread"],
            tau_start=r["tau_start"]["tau"] + off,
            tau_end=r["tau_end"]["tau"] + off,
            engine=r.get("engine", {}).get("tau"),
        )
        for r in rows
    ]
    blocks = pinfo.pop("blocks")
    agg = dict(
        rows=int(sum(b["rows"] for b in blocks)),
        needed=int(sum(b["needed"] for b in blocks)),
        pairs=int(sum(b["pairs"] for b in blocks)),
        tests=float(sum(b["tests"] for b in blocks)),
        tests_h=float(sum(b["tests_h"] for b in blocks)),
        read_bytes=int(sum(x[3] for b in blocks for x in b["read"].values())),
    )
    per_block = dict(
        rows=[b["rows"] for b in blocks],
        needed=[b["needed"] for b in blocks],
        contiguous=[int(sum(x[2] for k, x in b["read"].items() if k not in st.GEOMETRY)) for b in blocks],
        pages=[int(sum(x[0] for k, x in b["read"].items() if k not in st.GEOMETRY)) for b in blocks],
        span=[int(sum(x[1] for k, x in b["read"].items() if k not in st.GEOMETRY)) for b in blocks],
    )
    pinfo["blocks"] = dict(total=agg, per_block=per_block)
    out = dict(
        jit_s=jit_s,
        cache=cache,
        pass_info=pinfo,
        tiles=tiles,
        nbins=int(nb),
        dvbin=float(d),
        tau_bytes=int(taus["tau"].nbytes),
        store_bytes=pinfo["store_bytes"],
        run_s=t_end - t_run,
        tau=taus["tau"],
    )
    del store
    return out


def run_entry(c: dict, data_dir) -> dict:
    """Measure one (point, entry) ``c`` (``memory_point``) in this process; returns the record (module docstring)."""
    import jax

    from skewerjax import config
    from skewerjax.comparison.benchmark import machine_metadata
    from skewerjax.comparison.scaling import FS_LINES  # noqa: F401 (fake_spectra's lines)
    from skewerjax.simulations import ArepoHDF5Reader

    jax.config.update("jax_default_device", jax.devices("cpu")[0])
    cpu = config.devices("cpu")
    gpu = config.devices("gpu") if c["entry"] == GPU_DEF else []
    if c["entry"] == GPU_DEF and not gpu:
        raise RuntimeError("the GPU entry needs a GPU (JAX_PLATFORMS=cuda,cpu).")
    sampler = ProcessSampler(gpu=c["entry"] == GPU_DEF, jax_gpu=gpu)
    sampler.mark("start")
    snap_base = Path(data_dir) / c["sim"]
    reader = ArepoHDF5Reader(snap_base, SNAP)
    H = reader.header
    rec = dict(
        id=c["id"],
        config=c,
        status="ok",
        machine=machine_metadata(),
        precision=config.get_precision(),
        cpu_devices=len(cpu),
        gpu=[d.device_kind for d in gpu],
        snapshot=dict(n_gas=int(H.n_gas), box_cMpc=float(H.box_cMpc), n_files=int(H.n_files)),
        date=time.strftime("%Y-%m-%d %H:%M"),
        jax=jax.__version__,
    )
    t0 = time.perf_counter()
    if c["entry"] in (FS, FS_SEARCH):
        snapdir = Path(reader.files[0]).parent
        evict_page_cache(snapdir)
        rec["cache"] = cache_state(snapdir)
        X, Z = nested_grid(c["n"], H.box_cMpc)
        res = _fake_spectra(c, snap_base, X, Z, H, sampler, c["entry"] == FS_SEARCH)
    else:
        res = _skewerjax(c, reader, data_dir, sampler, cpu, gpu)
        rec["cache"] = res.pop("cache")
    tau = res.pop("tau")
    rec.update(res)
    rec["wall_s"] = time.perf_counter() - t0
    tl = sampler.stop()
    rec["timeline"] = tl
    rec["markers"] = [[round(t, 4), n] for t, n in sampler.markers]
    rec["jax_gpu"] = sampler.jax_stats
    rec["peaks"] = stage_peaks(tl, sampler.markers)
    t = np.asarray(tl["t"], dtype=float)
    t_set = next(tm for tm, nm in sampler.markers if nm == "setup")  # the measured run: from "setup" on (after any warm-up)
    run = t >= t_set
    pk = lambda key, f=np.nanmax: (lambda x: None if not np.any(np.isfinite(x)) else float(f(x)))(
        np.asarray(tl[key], dtype=float)[run]
    )
    rec["peak"] = dict(
        anon_GB=pk("anon_GB"),
        file_GB=pk("file_GB"),
        shmem_GB=pk("shmem_GB"),
        gpu_GB=pk("gpu_GB"),
        min_avail_GB=pk("avail_GB", np.nanmin),
        hwm_GB=pk("hwm_GB"),  # VmHWM: the process's peak resident set since it started (the warm-up included)
        rss_GB=float(
            np.nanmax(
                (np.asarray(tl["anon_GB"], float) + np.asarray(tl["file_GB"], float) + np.asarray(tl["shmem_GB"], float))[run]
            )
        ),
    )
    rd = np.asarray(tl["read_GB"], dtype=float)
    rec["read_GB"] = float(rd[-1] - rd[run][0])  # bytes read from the disk by the measured run
    ref = (c["n"] // 4) * c["n"] + c["n"] // 4 if c["n"] % 4 == 0 else 0
    rec["reference_tau_max"] = [float(np.max(tau[j][ref])) for j in range(len(tau))]
    return rec


# --------------------------------------------------------------------------- the gap between the two codes (notebook 15)
def stage_spans(rec: dict, *, after: str | None = "setup") -> list:
    """[(t0, t1, stage)] of a record's markers from ``after`` on (the measured run), merged where a stage repeats."""
    marks = [tuple(m) for m in rec["markers"]]
    t_end = float(rec["timeline"]["t"][-1])
    if after is not None:
        t_a = next((t for t, n in marks if n == after), marks[0][0])
        marks = [(t, n) for t, n in marks if t >= t_a]
    out = []
    for (t0, n), (t1, _) in zip(marks, marks[1:] + [(t_end, None)]):
        if n in ("start", "done") or t1 <= t0:
            continue
        out.append((float(t0), float(t1), n))
    return out


def mean_busy_cores(rec: dict, stages, *, window_s: float = 0.5) -> float:
    """The mean busy cores of a record over the spans of the given stages (CPU time over wall time within them)."""
    t = np.asarray(rec["timeline"]["t"], dtype=float)
    c = np.asarray(rec["timeline"]["cpu_s"], dtype=float)
    cpu = wall = 0.0
    for t0, t1, n in stage_spans(rec):
        if n in stages:
            cpu += float(np.interp(t1, t, c) - np.interp(t0, t, c))
            wall += t1 - t0
    return cpu / wall if wall > 0 else float("nan")


def work_counts(fs: dict, sjx: dict, fs_search: dict | None = None) -> dict:
    """The work of each code (module docstring of notebook 15, section 6) from its records: bytes read (measured),
    cells whose gas physics is computed, (cell, skewer) distance tests of the search, chord points prepared on the
    host, profile evaluations. fake_spectra's search tests and walk come from SkewerJAX's pass and per-cell engine on
    the same cells (its reach h = V^(1/3), its walk estimated from each cell's profile), counted once per line (it
    repeats everything per line)."""
    n_lines = 2
    pi = sjx["pass_info"]
    tot = pi["blocks"]["total"]
    chords = [sum(t["n_contributions"][j] for t in sjx["tiles"]) for j in range(n_lines)]
    ev_sjx = [sum(t["engine"][j]["evaluations"] for t in sjx["tiles"]) for j in range(n_lines)]
    walk = [sum((t["engine"][j].get("walk_evaluations") or 0) for t in sjx["tiles"]) for j in range(n_lines)]
    return {
        "bytes read [GB]": (fs["read_GB"], sjx["read_GB"]),
        "cells whose physics is computed": (
            n_lines * pi["n_cells"],
            pi["n_needed"] * len([k for k in pi["store_bytes"] if k.startswith("fields:")]),
        ),
        "(cell, skewer) distance tests": (n_lines * tot["tests_h"], tot["tests"]),
        "chord points prepared on the host": (n_lines * 7 * chords[0], sum(chords)),
        "profile evaluations": (float(sum(walk)), float(sum(ev_sjx))),
    }


def gap_table(fs: dict, sjx: dict, fs_search: dict) -> list:
    """The gap between fake_spectra and matched SkewerJAX, stage by stage (notebook 15, section 6): one dict per stage
    (read, gas physics, search, chords, tau accumulation) with fs_s, sjx_s, ratio, the work of each (``work_counts``;
    the gas physics in cell-species: fake_spectra computes one species per line on every cell, SkewerJAX every species
    of the cells within reach), the mean busy cores of each, and the shares of the stage's gap (fs_s - sjx_s), split in
    proportion to the logarithms of the factors of the ratio of times:

    * **algorithmic**: the ratio of the work;
    * **hardware use**: the ratio of the busy cores, where both are measured on the stage alone (tau accumulation: the C
      call of fake_spectra, the tiles of SkewerJAX);
    * **implementation**: the rest (the same work done faster: for the read, the throughput of parallel block reads; in
      the pass, whose stages overlap on shared threads, also the use of the cores, which cannot be told apart there).

    fake_spectra's search is the C call of ``fs_search`` (``tautail`` = 1e30: the engine stops after one pixel; with the
    casts of the arrays to float64), its tau accumulation the rest of its C call; SkewerJAX's pass stages are the wall
    shares of the pass's thread times, its chords the tiles' host preparation not hidden behind the engine."""
    W = work_counts(fs, sjx, fs_search)
    st, pt = fs["stages"], sjx["pass_info"]["timings"]
    search_fs = fs_search["stages"]["tau"]
    tau_fs = max(st["tau"] - search_fs, 0.0)
    cfgs = [k.split(":", 1)[1] for k in sjx["pass_info"]["store_bytes"] if k.startswith("fields:")]
    n_species = len(sjx["tiles"][0]["n_contributions"]) if sjx["tiles"] else 2
    tiles = sjx["tiles"]
    phys_fs = W["cells whose physics is computed"][0]  # one species per line, every cell
    phys_sjx = sjx["pass_info"]["n_needed"] * len(cfgs) * n_species
    rows = [
        dict(
            stage="read", fs_s=st["io"], sjx_s=pt.get("read", 0) + pt.get("read_rest", 0), work=W["bytes read [GB]"], cores=None
        ),
        dict(
            stage="gas physics",
            fs_s=st["physics"],
            sjx_s=sum(pt.get(f"physics:{c}", 0.0) for c in cfgs),
            work=(phys_fs, phys_sjx),
            cores=None,
        ),
        dict(
            stage="search",
            fs_s=search_fs,
            sjx_s=pt.get("reach", 0) + pt.get("tiles", 0),
            work=W["(cell, skewer) distance tests"],
            cores=None,
        ),
        dict(
            stage="chords",
            fs_s=float("nan"),
            sjx_s=sum(t["wait_s"] for t in tiles),
            work=W["chord points prepared on the host"],
            cores=None,
        ),
        dict(
            stage="tau accumulation",
            fs_s=tau_fs,
            sjx_s=sum(t["tau_s"] for t in tiles),
            work=W["profile evaluations"],
            cores=(mean_busy_cores(fs, ("tau",)), mean_busy_cores(sjx, ("tiles",))),
        ),
    ]
    stage_cores = {
        "read": ("io", "pass"),
        "gas physics": ("physics", "pass"),
        "search": ("tau", "pass"),
        "chords": ("tau", "tiles"),
    }
    for r in rows:
        r["work_fs"], r["work_sjx"] = r.pop("work")
        if r["cores"] is None:
            a, b = stage_cores[r["stage"]]
            r["cores_fs"] = mean_busy_cores(fs_search if r["stage"] == "search" else fs, (a,))
            r["cores_sjx"] = mean_busy_cores(sjx, (b,))
            separable = False
        else:
            r["cores_fs"], r["cores_sjx"] = r["cores"]
            separable = True
        del r["cores"]
        r["ratio"] = r["fs_s"] / r["sjx_s"] if r["sjx_s"] > 0 else float("nan")
        lr = np.log(r["ratio"]) if np.isfinite(r["ratio"]) and r["ratio"] > 0 else float("nan")
        la = np.log(r["work_fs"] / r["work_sjx"]) if r["work_fs"] > 0 and r["work_sjx"] > 0 else 0.0
        lh = np.log(r["cores_sjx"] / r["cores_fs"]) if separable and r["cores_fs"] > 0 and r["cores_sjx"] > 0 else 0.0
        li = lr - la - lh
        gap = r["fs_s"] - r["sjx_s"]
        share = lambda x: gap * x / lr if np.isfinite(lr) and lr != 0 else (0.0 if gap == 0 else float("nan"))
        r["shares"] = {
            "algorithmic": share(la),
            "hardware use": share(lh) if separable else float("nan"),
            "implementation": share(li),
        }
        r["separable"] = separable
    return rows


# --------------------------------------------------------------------------- accounting (notebook 16)
FS_BYTES_PER_CELL = 96  # fake_spectra's arrays of one file at its C call, per cell (float32 originals + float64 copies)


def _at(rec: dict, key: str, t: float) -> float:
    tt = np.asarray(rec["timeline"]["t"], dtype=float)
    return float(np.asarray(rec["timeline"][key], dtype=float)[min(int(np.searchsorted(tt, t)), tt.size - 1)])


def peak_components(rec: dict) -> dict:
    """The measured peak of anonymous host memory (and of GPU memory) of one record split into what the code knows
    (read-only): dict(stage (where the peak is), host {component: GB}, gpu {component: GB} or None). SkewerJAX: the
    baseline (Python, JAX, the compiled kernels: the anonymous memory when the measured run starts), then at a peak in the
    pass the needed cells held by its blocks (about the store) and the blocks in flight, at a peak in the tiles the store
    (``StreamStore.nbytes_components``), the spectra and the contributions of the tiles in flight (the current one and
    ``prep_workers`` = 3 ahead); fake_spectra: the baseline, one file's arrays (``FS_BYTES_PER_CELL`` x its cells) and its
    tau arrays (H I's cached result, the result and one file's part: 1.5 x the two lines'). The tiles' part is an estimate
    from the code: three tiles in preparation, each with its candidate pairs (64 bytes per pair), the float64 copy of its
    cells (~136 bytes per cell) and its contributions. The rest is "unaccounted"
    (allocator retention, Python objects, temporaries). GPU (NVML): the CUDA context (the first sample) and the device
    buffers (the rest)."""
    GB = 1e-9
    spans = stage_spans(rec)
    pk = rec["peaks"]
    stage = max((n for _, _, n in spans if n in pk), key=lambda n: pk[n]["anon_GB"] if pk[n]["anon_GB"] is not None else -1)
    peak = rec["peak"]["anon_GB"]
    base = _at(rec, "anon_GB", spans[0][0]) if spans else 0.0
    host = {"baseline": base}
    if "pass_info" in rec:
        store = sum(rec["store_bytes"].values()) * GB
        if stage in ("pass", "assemble"):
            host["store"] = store  # the needed cells, held by the blocks' chunks (and copied into the store at assembly)
            th = rec["pass_info"]["threads"]
            host["pass blocks"] = (
                th * 2_000_000 * 140 * GB
            )  # 2 M rows x (geometry read, converted positions and sizes, the rest) per thread
        else:
            host["store"] = store
            host["spectra"] = rec["tau_bytes"] * GB
            # the tiles in preparation (``tiles_optical_depth``: 3 host threads): each holds its candidate pairs (cell,
            # skewer, u, r^2 and their sorted copy: 64 bytes per pair), the float64 copy of its cells (``StreamStore.cells``
            # and the positions: ~136 bytes per cell) and its contributions; the engine's tile besides
            nt = len(rec["tiles"])
            pairs = rec["pass_info"]["blocks"]["total"]["pairs"] / max(nt, 1)
            cells = max((t["n_cells"] for t in rec["tiles"]), default=0)
            contr = max((t["contributions_bytes"] for t in rec["tiles"]), default=0)
            host["tiles"] = (min(3, nt) * (64 * pairs + 136 * cells + contr) + contr) * GB
    else:
        n_file = rec["snapshot"]["n_gas"] / rec["snapshot"]["n_files"]
        host["per-file arrays"] = FS_BYTES_PER_CELL * n_file * GB
        host["tau arrays"] = 1.5 * rec["tau_bytes"] * GB
    known = sum(host.values())
    if known > peak:  # the estimates above the measurement: scale the estimated parts (never the measured or exact ones)
        est = [k for k in host if k in ("pass blocks", "tiles", "per-file arrays")]
        rest = sum(v for k, v in host.items() if k not in est)
        f = max(peak - rest, 0.0) / max(sum(host[k] for k in est), 1e-30)
        for k in est:
            host[k] *= f
        known = sum(host.values())
    host["unaccounted"] = max(peak - known, 0.0)
    gpu = None
    g = np.asarray(rec["timeline"].get("gpu_GB") or [], dtype=float)
    if g.size and np.any(np.isfinite(g)):
        ctx = float(g[np.isfinite(g)][0])
        gpu = {"CUDA context": ctx, "device buffers": max(float(np.nanmax(g)) - ctx, 0.0)}
    return dict(stage=stage, host=host, gpu=gpu)


def fit_linear(x, y) -> tuple:
    """(a, b) of y = a + b x by least squares on the relative residuals (points weighted by 1 / y)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y) & (y > 0)
    A = np.stack([np.ones(ok.sum()), x[ok]], 1) / y[ok][:, None]
    a, b = np.linalg.lstsq(A, np.ones(ok.sum()), rcond=None)[0]
    return float(a), float(b)


# --------------------------------------------------------------------------- records and the launcher
def load_memory_records(folder) -> dict:
    """{id: record} of the JSON records in ``folder``."""
    out = {}
    for p in sorted(Path(folder).glob("*.json")):
        r = json.loads(p.read_text())
        out[r["id"]] = r
    return out


def worker_env(entry: str) -> dict:
    """The environment of a worker: the scaling benchmark's (32 OpenMP threads, the pip CUDA libraries, no
    preallocation), with JAX on the CPU only unless the entry is the GPU one (a CPU entry holds no CUDA context)."""
    from skewerjax.comparison.scaling import worker_env as base

    env = base()
    env["JAX_PLATFORMS"] = "cuda,cpu" if entry == GPU_DEF else "cpu"
    return env


def run_points(points: list, out, *, data_dir, log=print, timeout_s: float | None = None) -> list:
    """Measure every (point, entry) of ``points`` without a record in ``out``, each in its own process, in order
    (resumable); logs in ``out/logs/<id>.log``. The calling process must not hold the GPU (``JAX_PLATFORMS=cpu`` before
    importing the package, as ``scripts/benchmarks/benchmark_memory.py`` does)."""
    import subprocess
    import sys

    out = Path(out)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    done = []
    for c in points:
        if (out / f"{c['id']}.json").exists():
            log(f"{c['id']}: recorded")
            continue
        cmd = [sys.executable, "-m", "skewerjax.comparison.memory", json.dumps(c), str(out), "--data", str(data_dir)]
        t0 = time.perf_counter()
        log(f"{c['id']}: running ({', '.join(c.get('blocks', []))}) ...")
        with open(out / "logs" / f"{c['id']}.log", "w") as f:
            p = subprocess.run(cmd, env=worker_env(c["entry"]), stdout=f, stderr=subprocess.STDOUT, timeout=timeout_s)
        log(f"{c['id']}: {'done' if p.returncode == 0 else f'FAILED ({p.returncode})'} in {time.perf_counter() - t0:.0f} s")
        if p.returncode == 0:
            done.append(c["id"])
    return done


def main(argv=None):  # the worker: one (point, entry), in its own process
    import argparse

    ap = argparse.ArgumentParser(description="One (point, entry) of the memory benchmark (a worker).")
    ap.add_argument("config", help="the configuration, JSON")
    ap.add_argument("out", help="the folder of the records")
    ap.add_argument("--data", required=True, help="the folder of the simulations (skewerjax.utils.paths.DATA_DIR)")
    a = ap.parse_args(argv)
    rec = run_entry(json.loads(a.config), a.data)
    print(write_record(rec, a.out))


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count()))
    from skewerjax import config as _config

    _config.configure_devices(n_cpu_devices=2, platforms=os.environ.get("JAX_PLATFORMS", "cpu"))
    main()
