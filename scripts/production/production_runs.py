"""One production run: its configuration (``RunConfig``), its folder and record (``run.json``), its state, the sharing
of an identical finished run, and its execution in a child process (``execute``; ``run_production.py``; README.md).

**A run folder** (self-describing: readable with h5py and json alone, ``STORE.md``)::

    <out>/<sweep>/<run>/
        run.json     the record: the full configuration, the status, the package version and git commit, the snapshot and
                     its cosmology, the hardware, the timings, the estimate, the manifest of the files written
        run.log      the run's output (the child process's stdout and stderr)
        store/       a skewer store (``skewer_store``) holding the run's one dataset:
                     store/<sim>/snap_<NNN>/axis_<a>/<hash>/{config.json, skewers/batch_*.h5, runs/run_*.json}

**States** (``run_state``): "absent" (no run.json); "running" (status running and its process alive on this host);
"interrupted" (status running, launched or interrupted, its process gone: killed, power cut); "failed" (an error);
"damaged" (status done, but a file of the manifest is missing or of another size); "done" (status done, every file of the
manifest present with its size). Only "done" is skipped by a sweep; "interrupted", "failed" and "damaged" are redone from
scratch (the folder emptied first), "done" only with ``--force``.

**Sharing.** Two runs with the same ``RunConfig.key()`` (every field that changes what is computed: not the name, the
threads or the memory fraction) compute the same skewers. A finished run is reused by hard links of its files (no extra
disk; a copy across file systems), recorded as ``shared_from``.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

SCHEMA = "skewerjax-production-run/1"
RUN_FILE = "run.json"
LOG_FILE = "run.log"
STORE_DIR = "store"
LINE_IDS = ("HI-LyA", "SiIII-1206")
NOT_IN_KEY = ("name", "threads", "memory_fraction")  # fields that do not change what is computed


@dataclass
class RunConfig:
    """Everything that defines one run. Defaults: the benchmark setup (TNG50-2, snapshot 25, Lya + Si III 1206, float32,
    1 km/s pixels, skewers along y, the default preset: exact cell crossings, integrated profile, snapshot frame).

    sim, sim_path, snap : the simulation's name (its folder under ``skewerjax.utils.paths.DATA_DIR`` unless ``sim_path``)
        and snapshot; lines : the line ids (the first the reference); n : skewers per transverse axis, a power of two (the
        n x n lattice, x_i = i L / n, nested in every finer lattice); axis : the line of sight (1, 2, 3 = x, y, z);
    dv_kms or n_pix : the pixel grid (target width [km/s], n_pix = int(span / dv); or the number of pixels over the
        period); points_kms : the spacing of the sampling points [km/s] (``skewer_extraction``'s docstring: the dual
        segments of ``options={"pieces": "segments"}``, the a(s) table of ``options={"frame": "exact"}``; no effect on
        the default crossings in the snapshot frame);
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
        if o["pieces"] == "segments" and self.points_kms is None:
            raise ValueError("options pieces='segments' needs points_kms (the spacing of the sampling points).")
        if self.points_kms is not None and not float(self.points_kms) > 0:
            raise ValueError(f"points_kms must be > 0, got {self.points_kms}.")
        return self

    def points_effect(self) -> str:
        """What ``points_kms`` controls in this configuration (README, the ``points`` check)."""
        import skewer_extraction as se

        o = se.mode_options(self.mode, self.options)
        if self.points_kms is None:
            return "no sampling points (the pieces need none)" if o["pieces"] != "segments" else "missing"
        parts = []
        if o["pieces"] == "segments":
            parts.append(f"the dual segments of points every {self.points_kms:g} km/s ({o['profile']} profile)")
        if o["frame"] == "exact":
            parts.append(f"the a(s) table every {self.points_kms:g} km/s (exact frame)")
        return "; ".join(parts) if parts else "no effect (crossings in the snapshot frame: no sampling points, no a(s))"

    def summary(self) -> str:
        grid = f"{self.dv_kms:g} km/s" if self.dv_kms is not None else f"{self.n_pix} px"
        opts = ",".join(f"{k}={v}" for k, v in sorted(self.options.items()))
        pts = f", points {self.points_kms:g} km/s" if self.points_kms is not None else ""
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


def manifest(run_dir) -> dict:
    """{relative path: size} of every file of the run's store (the outputs)."""
    root = Path(run_dir) / STORE_DIR
    if not root.is_dir():
        return {}
    return {str(p.relative_to(run_dir)): p.stat().st_size for p in sorted(root.rglob("*")) if p.is_file() and p.name != ".lock"}


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
        have = manifest(run_dir)
        ok = bool(files) and all(have.get(k) == v for k, v in files.items())
        return "done" if ok else "damaged"
    if st in ("running", "launched"):
        same_host = rec.get("host") in (None, socket.gethostname())
        return "running" if same_host and rec.get("pid") and _pid_alive(rec["pid"]) else "interrupted"
    if st == "failed":
        return "failed"
    return "interrupted"


def reset(run_dir, out) -> None:
    """Empty a run folder before it is redone (only a folder inside ``out`` holding a run record, or empty)."""
    run_dir, out = Path(run_dir).resolve(), Path(out).resolve()
    if out not in run_dir.parents:
        raise RuntimeError(f"refusing to empty {run_dir}: not inside the output folder {out}.")
    if run_dir.exists():
        if not (run_dir / RUN_FILE).exists() and any(run_dir.iterdir()):
            raise RuntimeError(f"refusing to empty {run_dir}: it holds files but no {RUN_FILE}.")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)


def find_shareable(out, key: str, exclude=None) -> Path | None:
    """A finished run under ``out`` (any sweep) with the same key, or None."""
    for p in sorted(Path(out).glob(f"*/*/{RUN_FILE}")):
        d = p.parent
        if exclude is not None and d.resolve() == Path(exclude).resolve():
            continue
        rec = read_record(d)
        if rec and rec.get("key") == key and run_state(d) == "done":
            return d
    return None


def share(src, dst, rec_template: dict) -> dict:
    """Reuse the finished run ``src`` in the (empty) folder ``dst``: hard links of its store's files (copies across file
    systems), its record with this run's name, sweep and ``shared_from``. Returns the record written."""
    src, dst = Path(src), Path(dst)
    for rel in read_record(src)["outputs"]["files"]:
        a, b = src / rel, dst / rel
        b.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(a, b)
        except OSError:
            shutil.copy2(a, b)
    rec = dict(read_record(src))
    rec.update(
        name=rec_template["name"],
        sweep=rec_template["sweep"],
        config=rec_template["config"],
        shared_from=str(src),
        shared_at=now(),
        estimate=rec_template.get("estimate"),
    )
    rec["outputs"] = dict(rec["outputs"], files=manifest(dst))
    write_record(dst, rec)
    return rec


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


def execute(run_dir) -> int:
    """Compute the run of ``run_dir/run.json`` (written by the launcher, status "launched") into ``run_dir/store``, in this
    process: JAX's platforms and the precision are set before SkewerJAX is imported. Updates the record (status running,
    then done / failed / interrupted). Returns the exit code."""
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
        )
        dcfg = se.dataset_config(reader, cfg.sim, cfg.snap, cfg.axis, opts, preset=cfg.mode)
        ds = ss.SkewerDataset.create(run_dir / STORE_DIR, dcfg)
        snap = dict(dcfg["snapshot"])
        cosmo = {k: getattr(H, k) for k in ("h", "omega_m", "omega_lambda", "omega_b") if hasattr(H, k)}
        if cfg.sim.upper().startswith("TNG"):
            cosmo = dict(TNG_COSMOLOGY, **cosmo)
        rec.update(
            dataset=dict(
                root=STORE_DIR,
                path=str(ds.path.relative_to(run_dir)),
                hash=ds.hash,
                sim=cfg.sim,
                snap=cfg.snap,
                axis=cfg.axis,
                n_per_axis=cfg.n,
                line_ids=list(cfg.lines),
            ),
            options=opts,
            snapshot=snap,
            cosmology=cosmo,
            points_effect=cfg.points_effect(),
            software=dict(skewerjax=skewerjax.__version__, versions=se.versions(), git=git_info()),
            hardware=hardware(),
            devices=[str(d) for d in devices],
        )
        write_record(run_dir, rec)
        _log(f"run {rec['sweep']}/{rec['name']}: {cfg.summary()}; points: {cfg.points_effect()}")
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
        rec.update(
            status="done",
            finished=now(),
            wall_s=time.perf_counter() - t0,
            timings=dict(
                wall_s=time.perf_counter() - t0,
                extract_s=er.get("wall_s"),
                pass_s=sum(p.get("pass_s", 0.0) for p in passes),
                tiles_s=sum(p.get("tiles_s", 0.0) for p in passes),
                n_passes=len(passes),
                n_batches=len(er["batches"]),
                extract_record=str((ds.path / "runs" / f"run_{er['run_id']}.json").relative_to(run_dir)),
            ),
            outputs=dict(
                files=manifest(run_dir),
                bytes=sum(manifest(run_dir).values()),
                n_skewers=cfg.n * cfg.n,
                n_pix=int(opts["pixels"]["n_pix"]),
            ),
        )
        write_record(run_dir, rec)
        _log(f"run done in {rec['wall_s']:.1f} s: {rec['outputs']['bytes'] / 1e6:.1f} MB in {len(rec['outputs']['files'])} files")
        return 0
    except BaseException as e:
        rec.update(
            status="interrupted" if isinstance(e, KeyboardInterrupt) else "failed",
            error=f"{type(e).__name__}: {e}",
            finished=now(),
            wall_s=time.perf_counter() - t0,
        )
        write_record(run_dir, rec)
        _log(f"run {rec['status']}: {rec['error']}")
        if isinstance(e, KeyboardInterrupt):
            return 130
        import traceback

        traceback.print_exc()
        return 1
