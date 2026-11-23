"""One production run: its configuration (``RunConfig``), its folder and record (``run.json``), its state, and its
execution in a child process (``execute``; ``run_production.py``; README.md).

**The output folder** (self-describing: readable with h5py and json alone, ``STORE.md``)::

    <out>/
        store/                     the campaign's skewer store (``skewer_store``): one dataset per configuration (the hash of
                                   its resolved options), each holding nested lattices:
                                   store/<sim>/snap_<NNN>/axis_<a>/<hash>/{config.json, skewers/batch_*.h5, runs/run_*.json}
        <sweep>/index.json         the sweep's runs, their states, configurations and datasets
        <sweep>/sweep.log          the launcher's log of the sweep
        <sweep>/<run>/run.json     the record: the full configuration, the status, the dataset and lattice it reads (``dataset``),
                                   the sampling and the pixels as computed, the package version and git commit, the snapshot
                                   and its cosmology, the hardware, the timings and peak memory, the estimate, the manifest
                                   of the batch files that hold its lattice
        <sweep>/<run>/run.log      the run's output (the child process's stdout and stderr)

**Runs share datasets.** A run is an n x n lattice of one dataset. Runs whose options resolve to the same dataset (the
same hash) share it: a run identical to a finished one, or a coarser lattice of a finer one (the lattices are nested,
``skewer_store``), computes nothing; a finer lattice extends the dataset, computing only the skewers it does not hold
(``skewer_extraction.extract``). The fiducial run of the resolution campaign is computed once and serves the four sweeps.

**States** (``run_state``): "absent" (no run.json); "running" (status running and its process alive on this host);
"interrupted" (status running, launched or interrupted, its process gone: killed, power cut); "failed" (an error);
"damaged" (status done, but a batch file of the manifest is missing or of another size); "done" (status done, every file
of the manifest present with its size). Only "done" is skipped by a sweep; "interrupted", "failed" and "damaged" are
redone: the run's record is reset, and the extraction resumes from the batches already in the store (each batch is
written atomically, so only the tiles in progress are lost). ``--force`` deletes the run's dataset (every run sharing it
loses its skewers) and recomputes it.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
import resource
import shutil
import socket
import subprocess
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

SCHEMA = "skewerjax-production-run/1"
RUN_FILE = "run.json"
LOG_FILE = "run.log"
STORE_DIR = "store"  # the campaign's skewer store, <out>/store
LINE_IDS = ("HI-LyA", "SiIII-1206")
NOT_IN_KEY = ("name", "threads", "memory_fraction")  # fields that do not change what is computed


@dataclass
class RunConfig:
    """Everything that defines one run. Defaults: the benchmark setup (TNG50-2, snapshot 25, Lya + Si III 1206, float32,
    1 km/s pixels, skewers along y, the default preset: exact cell crossings, integrated profile, snapshot frame).

    sim, sim_path, snap : the simulation's name (its folder under ``skewerjax.utils.paths.DATA_DIR`` unless ``sim_path``)
        and snapshot; lines : the line ids (the first the reference); n : skewers per transverse axis, a power of two (the
        n x n lattice, x_i = i L / n, nested in every finer lattice); axis : the line of sight (1, 2, 3 = x, y, z);
    dv_kms or n_pix : the pixel grid (target width [km/s], n_pix = int(span / dv), dv = span / n_pix: the grid tiles the
        velocity period exactly; or the number of pixels over the period);
    points_cMpc or points_kms : the spacing of the sampling points along a skewer [cMpc], or [km/s] of Hubble flow (one of
        them; ``skewer_extraction``'s docstring): the dual segments of ``options={"pieces": "segments"}`` and the a(s)
        table of ``options={"frame": "exact"}``; no effect on the crossings in the snapshot frame;
    mode, options : the preset (``skewer_extraction.MODES``: "default", "matched") and overrides of its options
        (e.g. {"profile": "midpoint", "pieces": "segments"}, {"physics.x_HI_model": "stored"});
    precision : "float32" or "float64" (tau stored as float32); device : "cpu" or "gpu" (where the tau sum runs);
    tile : skewers per tile (one HDF5 batch each); threads : the pass's threads (None: every core);
    n_ref : the finest lattice of the dataset; memory_fraction : of the available memory one pass may use."""

    name: str = "run"
    sim: str = "TNG50-2"
    sim_path: str | None = None
    snap: int = 25
    lines: tuple = LINE_IDS
    n: int = 64
    axis: int = 2
    dv_kms: float | None = 1.0
    n_pix: int | None = None
    points_kms: float | None = None
    points_cMpc: float | None = None
    mode: str = "default"
    options: dict = field(default_factory=dict)
    precision: str = "float32"
    device: str = "cpu"
    tile: int = 1024
    threads: int | None = None
    n_ref: int = 4096
    memory_fraction: float = 0.6

    def __post_init__(self):
        self.lines = tuple(str(x) for x in self.lines)
        self.options = dict(self.options or {})

    # ---- (de)serialisation
    def to_dict(self) -> dict:
        d = asdict(self)
        d["lines"] = list(self.lines)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> RunConfig:
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown run fields {sorted(unknown)} (known: {sorted(known)}).")
        return cls(**d)

    def replace(self, **kw) -> RunConfig:
        d = self.to_dict()
        d.update(kw)
        return RunConfig.from_dict(d)

    def key(self) -> str:
        """The identity of what the run computes (``NOT_IN_KEY`` left out): SHA-256 of its canonical JSON, 16 digits."""
        d = {k: v for k, v in self.to_dict().items() if k not in NOT_IN_KEY}
        d["sim_path"] = str(self.resolved_sim_path())
        return hashlib.sha256(json.dumps(d, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]

    def resolved_sim_path(self) -> Path:
        if self.sim_path:
            return Path(self.sim_path).expanduser()
        from skewerjax.utils.paths import DATA_DIR

        return Path(DATA_DIR) / self.sim

    # ---- checks (no snapshot read)
    def validate(self) -> RunConfig:
        """A ValueError naming the first inconsistent field."""
        import skewer_extraction as se
        import skewer_store as ss

        ss.check_n(int(self.n), int(self.n_ref))
        if int(self.axis) not in (1, 2, 3):
            raise ValueError(f"axis must be 1, 2 or 3 (x, y, z), got {self.axis!r}.")
        if (self.dv_kms is None) == (self.n_pix is None):
            raise ValueError("give the pixel grid by dv_kms or by n_pix (one of them).")
        if self.dv_kms is not None and not float(self.dv_kms) > 0:
            raise ValueError(f"dv_kms must be > 0, got {self.dv_kms}.")
        if self.precision not in ("float32", "float64"):
            raise ValueError(f"precision must be float32 or float64, got {self.precision!r}.")
        if self.device not in ("cpu", "gpu"):
            raise ValueError(f"device must be cpu or gpu, got {self.device!r}.")
        if int(self.tile) < 1:
            raise ValueError(f"tile must be >= 1, got {self.tile}.")
        if not self.lines:
            raise ValueError("at least one line is needed.")
        o = se.mode_options(self.mode, self.options)
        if self.points_kms is not None and self.points_cMpc is not None:
            raise ValueError("give the sampling spacing by points_cMpc or by points_kms, not both.")
        if o["pieces"] == "segments" and self.points_kms is None and self.points_cMpc is None:
            raise ValueError("options pieces='segments' needs points_cMpc or points_kms (the spacing of the sampling points).")
        for name in ("points_kms", "points_cMpc"):
            x = getattr(self, name)
            if x is not None and not float(x) > 0:
                raise ValueError(f"{name} must be > 0, got {x}.")
        return self

    def spacing_text(self) -> str | None:
        """The sampling spacing as given ("0.05 cMpc", "0.25 km/s"), or None."""
        if self.points_cMpc is not None:
            return f"{self.points_cMpc:g} cMpc"
        return None if self.points_kms is None else f"{self.points_kms:g} km/s"

    def route(self) -> str:
        """The route: "sampled" (``pieces="segments"``: the gather route, ``compute_skewers_sampled``) or "streaming"."""
        import skewer_extraction as se

        return "sampled" if se.mode_options(self.mode, self.options)["pieces"] == "segments" else "streaming"

    def points_effect(self) -> str:
        """What ``points_kms`` controls in this configuration (README, the ``points`` check)."""
        import skewer_extraction as se

        o = se.mode_options(self.mode, self.options)
        sp = self.spacing_text()
        if sp is None:
            if o["pieces"] == "segments":
                return "missing"
            if o["frame"] == "exact":
                return "the a(s) table of the exact frame at stream.box_skewers' default spacing (0.05 cMpc)"
            return "no sampling points (the pieces need none)"
        parts = []
        if o["pieces"] == "segments":
            parts.append(
                f"the pieces: the dual segments of points every {sp} (the cells crossed, cut at the midpoints between "
                f"points; {o['profile']} profile)"
            )
        if o["frame"] == "exact":
            parts.append(f"the a(s) table of the exact frame every {sp}")
        if not parts:
            return "no effect (crossings in the snapshot frame: no sampling points, no a(s))"
        if o["pieces"] != "segments":
            parts.append("not the pieces (the exact cell crossings need no sampling points)")
        return "; ".join(parts)

    def summary(self) -> str:
        grid = f"{self.dv_kms:g} km/s" if self.dv_kms is not None else f"{self.n_pix} px"
        opts = ",".join(f"{k}={v}" for k, v in sorted(self.options.items()))
        pts = f", dx {self.spacing_text()}" if self.spacing_text() else ""
        return (
            f"{self.sim} snap {self.snap}, {self.n}^2 along {'xyz'[int(self.axis) - 1]}, {grid}{pts}, "
            f"{self.mode}{'+' + opts if opts else ''}, {'+'.join(self.lines)}, {self.precision}, {self.device}"
        )


# --------------------------------------------------------------------------- the folder and its record
def now() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def read_record(run_dir) -> dict | None:
    p = Path(run_dir) / RUN_FILE
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return {"status": "unreadable"}


def write_record(run_dir, rec: dict) -> None:
    """Write ``run.json`` atomically (a temporary file, fsync, rename)."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    tmp = run_dir / f".{RUN_FILE}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        f.write(json.dumps(rec, indent=1, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, run_dir / RUN_FILE)


def store_root(run_dir) -> Path:
    """The campaign's store of a run folder ``<out>/<sweep>/<run>``: ``<out>/store``."""
    return Path(run_dir).resolve().parents[1] / STORE_DIR


def manifest(run_dir, dataset=None, n: int | None = None) -> dict:
    """{path relative to the run folder: size} of the batch files that hold the run's n x n lattice in its dataset (the
    outputs it reads; ``SkewerDataset.sources``). With no ``dataset``, the one named by the run's record."""
    run_dir = Path(run_dir)
    if dataset is None:
        import skewer_store as ss

        rec = read_record(run_dir) or {}
        d = rec.get("dataset")
        if not d:
            return {}
        try:
            dataset = ss.SkewerDataset.open(run_dir / d["root"], d["sim"], d["snap"], d["axis"], d["hash"])
        except (FileNotFoundError, ValueError, OSError):
            return {}
        n = int(d["n_per_axis"]) if n is None else n
    files = sorted({f for row in dataset.sources(int(n)) for f in row["files"]})
    out = {}
    for name in files:
        f = dataset.path / "skewers" / name
        if f.exists():
            out[os.path.relpath(f, run_dir)] = f.stat().st_size
    return out


def _pid_alive(pid) -> bool:
    try:
        os.kill(int(pid), 0)
    except (ProcessLookupError, ValueError, TypeError):
        return False
    except PermissionError:
        return True
    return True


def run_state(run_dir) -> str:
    """The state of a run folder (module docstring)."""
    rec = read_record(run_dir)
    if rec is None:
        return "absent"
    st = rec.get("status")
    if st == "done":
        files = rec.get("outputs", {}).get("files", {})
        ok = bool(files) and all(
            (Path(run_dir) / k).is_file() and (Path(run_dir) / k).stat().st_size == v for k, v in files.items()
        )
        return "done" if ok else "damaged"
    if st in ("running", "launched"):
        same_host = rec.get("host") in (None, socket.gethostname())
        return "running" if same_host and rec.get("pid") and _pid_alive(rec["pid"]) else "interrupted"
    if st == "failed":
        return "failed"
    return "interrupted"


def reset(run_dir, out) -> None:
    """Clear a run folder before the run is redone: its record and log (only a folder inside ``out`` holding a run
    record, or empty). The skewers are in the campaign's store, untouched: the extraction resumes from them."""
    run_dir, out = Path(run_dir).resolve(), Path(out).resolve()
    if out not in run_dir.parents:
        raise RuntimeError(f"refusing to clear {run_dir}: not inside the output folder {out}.")
    if run_dir.exists():
        if not (run_dir / RUN_FILE).exists() and any(run_dir.iterdir()):
            raise RuntimeError(f"refusing to clear {run_dir}: it holds files but no {RUN_FILE}.")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)


def remove_dataset(path, out) -> int:
    """Delete one dataset of the campaign's store (``--force``): only a folder inside ``<out>/store`` holding a
    config.json. Returns the bytes freed."""
    path, root = Path(path).resolve(), (Path(out) / STORE_DIR).resolve()
    if root not in path.parents or not (path / "config.json").is_file():
        raise RuntimeError(f"refusing to delete {path}: not a dataset of the store {root}.")
    freed = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    shutil.rmtree(path)
    return freed


def runs_on_dataset(out, dataset_hash: str) -> list:
    """The run folders under ``out`` whose record reads the dataset ``dataset_hash``."""
    hits = []
    for p in sorted(Path(out).glob(f"*/*/{RUN_FILE}")):
        rec = read_record(p.parent) or {}
        if (rec.get("dataset") or {}).get("hash") == dataset_hash:
            hits.append(p.parent)
    return hits


def peak_rss_GB() -> float:
    """The peak resident memory of this process so far [GB] (``ru_maxrss``)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / 1e9


# --------------------------------------------------------------------------- the environment of a run
def git_info() -> dict:
    """The commit of the repository and whether its tree has changes (read only)."""
    import skewer_extraction as se

    return se.git_commit()


def hardware() -> dict:
    """The machine: CPU model, cores, RAM, GPU (``nvidia-smi``, if present)."""
    cpu = platform.processor()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    ram = None
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                ram = float(line.split()[1]) * 1024 / 1e9
    except OSError:
        pass
    gpu = None
    if shutil.which("nvidia-smi"):
        try:
            gpu = (
                subprocess.run(
                    ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                ).stdout.strip()
                or None
            )
        except (OSError, subprocess.SubprocessError):
            gpu = None
    return dict(
        host=socket.gethostname(),
        cpu=cpu,
        logical_cpus=os.cpu_count(),
        ram_GB=ram,
        gpu=gpu,
        python=platform.python_version(),
        platform=platform.platform(),
    )


# --------------------------------------------------------------------------- the execution (in the child process)
def _log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def sampling_record(cfg: RunConfig, opts: dict, u: float) -> dict:
    """The sampling and pixels of a run as computed (run.json "sampling"): the spacing in both units (u = aH [km/s per cMpc]
    of the snapshot: 1 km/s = 1 / u cMpc), what it controls, the pixel grid's nominal and actual widths."""
    ds = opts["tau"].get("frame_spacing_cMpc")
    if opts["pieces"]["pieces"] == "segments":
        ds = opts["pieces"].get("points_cMpc", opts["pieces"]["points_kms"] / u)
    px = opts["pixels"]
    return dict(
        points_cMpc=ds,
        points_kms=None if ds is None else ds * u,
        given=cfg.spacing_text(),
        u_kms_per_cMpc=u,
        cMpc_per_kms=1.0 / u,
        effect=cfg.points_effect(),
        route=cfg.route(),
        pieces=opts["pieces"]["pieces"],
        profile=opts["tau"]["profile"],
        frame=opts["tau"]["frame"],
        dv_nominal_kms=px["spacing_kms"],
        dv_kms=px["dv_kms"],
        n_pix=px["n_pix"],
        period_kms=px["period_kms"],
    )


def execute(run_dir) -> int:
    """Compute the run of ``run_dir/run.json`` (written by the launcher, status "launched") into the campaign's store
    (``store_root``), in this process: JAX's platforms and the precision are set before SkewerJAX is imported. The lattice
    of the run is extracted into its dataset (only the skewers the dataset does not hold). Updates the record (status
    running, then done / failed / interrupted). Returns the exit code."""
    run_dir = Path(run_dir).resolve()
    rec = read_record(run_dir)
    cfg = RunConfig.from_dict(rec["config"])
    rec.update(status="running", pid=os.getpid(), host=socket.gethostname(), started=now())
    write_record(run_dir, rec)
    t0 = time.perf_counter()
    try:
        if cfg.device == "gpu":
            os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        from skewerjax import config as sjconfig

        sjconfig.configure_devices(n_cpu_devices=2, platforms="cuda,cpu" if cfg.device == "gpu" else "cpu")
        sjconfig.set_precision(cfg.precision)
        import skewer_extraction as se
        import skewer_store as ss

        import skewerjax
        from skewerjax.simulations import TNG_COSMOLOGY, ArepoHDF5Reader

        devices = sjconfig.devices(cfg.device)
        if not devices:
            raise RuntimeError(f"no {cfg.device} device visible to JAX")
        reader = ArepoHDF5Reader(cfg.resolved_sim_path(), cfg.snap)
        H = reader.header
        opts = se.dataset_options(
            H,
            mode=cfg.mode,
            overrides=cfg.options,
            spacing_kms=cfg.dv_kms,
            n_pix=cfg.n_pix,
            line_ids=cfg.lines,
            precision=cfg.precision,
            n_ref=cfg.n_ref,
            points_kms=cfg.points_kms,
            points_cMpc=cfg.points_cMpc,
        )
        dcfg = se.dataset_config(reader, cfg.sim, cfg.snap, cfg.axis, opts, preset=cfg.mode)
        root = store_root(run_dir)
        ds = ss.SkewerDataset.create(root, dcfg)
        stored_before = int(ds.stored(ss.lattice_indices(cfg.n, ds.n_ref)).sum())
        snap = dict(dcfg["snapshot"])
        cosmo = {k: getattr(H, k) for k in ("h", "omega_m", "omega_lambda", "omega_b") if hasattr(H, k)}
        if cfg.sim.upper().startswith("TNG"):
            cosmo = dict(TNG_COSMOLOGY, **cosmo)
        rec.update(
            dataset=dict(
                root=os.path.relpath(root, run_dir),
                path=os.path.relpath(ds.path, run_dir),
                hash=ds.hash,
                sim=cfg.sim,
                snap=cfg.snap,
                axis=cfg.axis,
                n_per_axis=cfg.n,
                line_ids=list(cfg.lines),
                stored_before=stored_before,
            ),
            options=opts,
            snapshot=snap,
            cosmology=cosmo,
            sampling=sampling_record(cfg, opts, float(H.velfac_kms_per_cMpc)),
            points_effect=cfg.points_effect(),
            software=dict(skewerjax=skewerjax.__version__, versions=se.versions(), git=git_info()),
            hardware=hardware(),
            devices=[str(d) for d in devices],
            memory=dict(available_at_start_GB=se.memory_available() / 1e9, fraction=cfg.memory_fraction),
        )
        write_record(run_dir, rec)
        _log(f"run {rec['sweep']}/{rec['name']}: {cfg.summary()}")
        _log(f"sampling: {cfg.points_effect()}; dataset {ds.hash}: {stored_before} of {cfg.n}^2 skewers stored")
        er = se.extract(
            ds,
            reader,
            [cfg.n],
            tile_size=cfg.tile,
            devices=devices,
            device_name=cfg.device,
            threads=cfg.threads,
            memory_budget=cfg.memory_fraction * se.memory_available(),
            log=_log,
            argv=["run_production.py", "--child", str(run_dir)],
        )
        if not ds.complete(cfg.n):
            raise RuntimeError(f"the {cfg.n} x {cfg.n} lattice is not complete after the extraction")
        passes = [p for lat in er["lattices"] for p in lat["passes"]]
        files = manifest(run_dir, ds, cfg.n)
        wall = time.perf_counter() - t0
        rec.update(
            status="done",
            finished=now(),
            wall_s=wall,
            timings=dict(
                wall_s=wall,
                extract_s=er.get("wall_s"),
                pass_s=sum(p.get("pass_s", 0.0) for p in passes),
                tiles_s=sum(p.get("tiles_s", 0.0) for p in passes),
                n_passes=len(passes),
                n_batches=len(er["batches"]),
                computed=int(er.get("computed", 0)),
                reused=cfg.n * cfg.n - int(er.get("computed", 0)),
                extract_record=os.path.relpath(ds.path / "runs" / f"run_{er['run_id']}.json", run_dir),
            ),
            memory=dict(rec["memory"], peak_rss_GB=peak_rss_GB(), skewers_per_pass=[p.get("n_skewers") for p in passes]),
            outputs=dict(
                files=files,
                bytes=sum(files.values()),
                n_skewers=cfg.n * cfg.n,
                n_pix=int(opts["pixels"]["n_pix"]),
            ),
        )
        write_record(run_dir, rec)
        _log(
            f"run done in {wall:.1f} s: {rec['timings']['computed']} skewers computed, {rec['timings']['reused']} reused; "
            f"the lattice in {len(files)} batch files ({rec['outputs']['bytes'] / 1e6:.1f} MB); peak RSS "
            f"{rec['memory']['peak_rss_GB']:.1f} GB"
        )
        return 0
    except BaseException as e:
        rec.update(
            status="interrupted" if isinstance(e, KeyboardInterrupt) else "failed",
            error=f"{type(e).__name__}: {e}",
            finished=now(),
            wall_s=time.perf_counter() - t0,
        )
        rec.setdefault("memory", {})["peak_rss_GB"] = peak_rss_GB()
        write_record(run_dir, rec)
        _log(f"run {rec['status']}: {rec['error']}")
        if isinstance(e, KeyboardInterrupt):
            return 130
        import traceback

        traceback.print_exc()
        return 1
