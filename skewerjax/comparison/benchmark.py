"""Wall time and memory of SkewerJAX against fake_spectra on identical skewers (Lya and Si III 1206).

One configuration (code, snapshot, number of skewers, platform, threads, precision) runs in its own process, so that the
peak RSS is per configuration and the CPU set / thread count is fixed before any library starts its thread pool:

    python skewerjax/comparison/benchmark.py worker --config '<json>' --out DIR

(run as a file, so that a fake_spectra worker does not import JAX through the ``skewerjax`` package).

``run_suite`` launches the workers (skipping configurations whose result exists, with a wall-time cap each) and
``load_results`` reads them back. Each worker writes ``<id>.json`` (stages, repeats, memory, metadata, agreement) and
``<id>.npz`` (tau of the first skewers, mean flux of all).

Stages (exclusive wall time, ``StageTimer``): ``setup`` (reader / ``Spectra`` object), ``io`` (raw HDF5 reads: SkewerJAX
``ArepoHDF5Reader.read_rows``, fake_spectra ``snapshot_set.get_data``), ``selection`` (cells near the skewers),
``physics`` (per-cell ion densities and temperatures, deposition; the rest of fake_spectra's ``get_tau``), ``tau`` (the
optical-depth integral: SkewerJAX ``tau_absorbers``, fake_spectra's C engine ``_do_interpolation_work``). SkewerJAX's
first tau includes the JIT compilation: ``compile_s`` = first tau - median warm tau.

Agreement: SkewerJAX's matched mode (``FAKE_SPECTRA_EXTRACTION`` / ``FAKE_SPECTRA_INTEGRATION``, fake_spectra's pixel
grid) against fake_spectra's tau on the first ``n_agree`` skewers, and the mean flux of the default mode against
fake_spectra's on all skewers. SkewerJAX is timed in its default mode (exact frame), its production path.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

if __name__ == "__main__" and Path(sys.path[0]).resolve() == Path(__file__).resolve().parent:
    sys.path.pop(0)                                    # run as a file: no shadowing by the sibling modules

import numpy as np                                     # noqa: E402

STAGES = ("setup", "io", "selection", "physics", "tau")
SPECIES = {"HI-LyA": ("H", 1, 1215), "SiIII-1206": ("Si", 3, 1206)}
CPU_SETS = {1: [0], 4: [0, 2, 4, 6], 8: list(range(0, 16, 2)), 16: list(range(16)), 32: list(range(32))}
"""Logical CPUs per thread count on the benchmark machine (i9-13900KF: CPUs 0-15 are the 8 P-cores with their
hyperthreads, 16-31 the E-cores): 4 and 8 use one thread per P-core."""


# --------------------------------------------------------------------------- skewers and timers
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

    def wrap(self, obj, method: str, name: str):
        """Time every call of ``obj.method`` (an instance attribute shadows the method) as stage ``name``."""
        f = getattr(obj, method)

        def timed(*a, **k):
            with self.stage(name):
                return f(*a, **k)
        setattr(obj, method, timed)


def peak_rss_GB() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def machine_metadata() -> dict:
    """Hardware and software of this process (CPU set, versions, environment)."""
    import importlib.metadata as md
    cpu = next((ln.split(":", 1)[1].strip() for ln in open("/proc/cpuinfo") if ln.startswith("model name")), platform.processor())
    mem = next((int(ln.split()[1]) / 1e6 for ln in open("/proc/meminfo") if ln.startswith("MemTotal")), None)
    vers = {}
    for p in ("jax", "jaxlib", "numpy", "scipy", "h5py", "fake_spectra", "exojax"):
        try:
            vers[p] = md.version(p)
        except md.PackageNotFoundError:
            vers[p] = None
    gpu = None
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        pass
    return dict(cpu=cpu, logical_cpus=os.cpu_count(), cpu_set=sorted(os.sched_getaffinity(0)), ram_GB=mem, gpu=gpu,
                python=platform.python_version(), versions=vers, host=platform.node(),
                env={k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "XLA_FLAGS", "JAX_PLATFORMS", "XLA_PYTHON_CLIENT_PREALLOCATE")})


def config_id(c: dict) -> str:
    sim = Path(c["snap_base"]).name
    tail = "" if c["code"] == "fake_spectra" else f"_{c['platform']}_{c.get('dtype') or 'float64'}"
    return f"{c['code']}_{sim}_s{c['snapnum']}_n{c['n_skewers']}_t{c['threads']}{tail}"


# --------------------------------------------------------------------------- the two workers
def run_fake_spectra(c: dict, out: Path) -> dict:
    """fake_spectra: a new ``Spectra`` object per repeat, ``get_tau`` per species (the C engine is OpenMP)."""
    import shutil
    from fake_spectra import spectra
    cofm, axis = benchmark_skewers(c["n_skewers"])
    reps, taus, t_start = [], {}, time.perf_counter()
    work = out / "fs_work" / config_id(c)
    for r in range(c["repeats"]):
        T = StageTimer()
        t0 = time.perf_counter()
        with T.stage("setup"):
            sp = spectra.Spectra(int(c["snapnum"]), str(c["snap_base"]), cofm.copy(), axis.copy(), res=c["res"], savefile="b.hdf5",
                                 savedir=str(work), reload_file=True, quiet=True)
        T.wrap(sp.snapshot_set, "get_data", "io")
        T.wrap(sp, "particles_near_lines", "selection")
        T.wrap(sp, "_do_interpolation_work", "tau")
        for name, (el, st, ll) in SPECIES.items():
            with T.stage("physics"):
                taus[name] = np.asarray(sp.get_tau(el, st, ll))
        reps.append(dict(stages=dict(T.t), total=time.perf_counter() - t0))
        nbins, dv = int(sp.nbins), float(sp.dvbin)
        del sp
        if time.perf_counter() - t_start + reps[-1]["total"] > c["budget_s"]:
            break
    shutil.rmtree(work, ignore_errors=True)
    np.savez(out / f"{config_id(c)}.npz", **{f"tau_{k}": v[:c["n_save"]] for k, v in taus.items()},
             **{f"meanF_{k}": np.exp(-v).mean(axis=1) for k, v in taus.items()})
    return dict(repeats=reps, peak_rss_GB=peak_rss_GB(), nbins=nbins, dv_kms=dv)


def run_skewerjax(c: dict, out: Path) -> dict:
    """SkewerJAX default mode: a new ``SimulationSkewers`` per repeat; Lya and Si III share the selection."""
    import jax
    jax.config.update("jax_enable_x64", True)
    from skewerjax.core import LineBundle
    from skewerjax.simulations import HI, ArepoHDF5Reader, Ion, SimulationSkewers
    from skewerjax.simulations.absorbers import tau_absorbers
    ions = {"HI-LyA": HI, "SiIII-1206": Ion("Si", 3)}
    lbs = {k: LineBundle([k], ref_line_id=k) for k in SPECIES}
    cofm, axis = benchmark_skewers(c["n_skewers"])
    reps, taus, t_start = [], {}, time.perf_counter()
    for r in range(c["repeats"]):
        T = StageTimer()
        t0 = time.perf_counter()
        with T.stage("setup"):
            rd = ArepoHDF5Reader(c["snap_base"], int(c["snapnum"]))
        T.wrap(rd, "read_rows", "io")
        with T.stage("selection"):
            sk = SimulationSkewers.axis_aligned(rd, cofm, axis, c["spacing_kms"])
        for name, ion in ions.items():
            with T.stage("physics"):
                A = sk.absorbers(ion)
                fr = sk.frame("exact")
                v, dv = sk.spectral_grid(c["res"], fr)
            with T.stage("tau"):
                taus[name] = tau_absorbers(A, lbs[name], v, fr, check=None, dtype=c.get("dtype"))[0]
        reps.append(dict(stages=dict(T.t), total=time.perf_counter() - t0, n_cells=len(sk.cells), n_v=int(v.size),
                         n_absorbers=int(A.s.shape[1])))
        if time.perf_counter() - t_start + reps[-1]["total"] > c["budget_s"] and r >= 1:
            break
    res = dict(repeats=reps, peak_rss_GB=peak_rss_GB(), devices=[str(d) for d in jax.devices()], dv_kms=float(dv))
    warm = [x["stages"]["tau"] for x in reps[1:]]
    res["compile_s"] = reps[0]["stages"]["tau"] - float(np.median(warm)) if warm else None
    if jax.devices()[0].platform == "gpu":
        res["device_memory"] = {k: v for k, v in (jax.devices()[0].memory_stats() or {}).items()
                                if k in ("peak_bytes_in_use", "bytes_in_use", "bytes_limit")}
    np.savez(out / f"{config_id(c)}.npz", **{f"tau_{k}": v[:c["n_save"]] for k, v in taus.items()},
             **{f"meanF_{k}": np.exp(-v).mean(axis=1) for k, v in taus.items()})
    res["agreement"] = agreement(c, out, sk, ions, lbs, taus)
    return res


def agreement(c: dict, out: Path, sk, ions: dict, lbs: dict, taus: dict) -> Optional[dict]:
    """Matched mode vs fake_spectra (first ``n_agree`` skewers) and default vs fake_spectra mean flux (all skewers),
    against the fake_spectra result of the same snapshot and skewer count (None if it does not exist yet)."""
    from skewerjax.simulations import SimulationSkewers
    from skewerjax.comparison.effects import FAKE_SPECTRA_EXTRACTION, FAKE_SPECTRA_INTEGRATION
    from skewerjax.comparison.emulation import matched_tau
    ref = out / f"{config_id(dict(c, code='fake_spectra', threads=max(CPU_SETS)))}.npz"
    if not ref.exists():
        return None
    fs = np.load(ref)
    k = min(c["n_agree"], c["n_skewers"])
    sub = SimulationSkewers(sk.reader, sk.points[:k], cells=sk.cells)
    v, dv = sub.spectral_grid(c["res"])
    rows = {}
    for name, ion in ions.items():
        F_m = np.exp(-matched_tau(sub, ion, lbs[name], v, FAKE_SPECTRA_EXTRACTION, FAKE_SPECTRA_INTEGRATION))
        F_f = np.exp(-fs[f"tau_{name}"][:k])
        dF = F_m - F_f
        rows[name] = dict(n_skewers=k, max_dF=float(np.abs(dF).max()), rms_dF=float(np.sqrt(np.mean(dF ** 2))),
                          meanF_fake_spectra=float(fs[f"meanF_{name}"].mean()),
                          meanF_skewerjax_default=float(np.exp(-taus[name]).mean(axis=1).mean()))
    return rows


def worker(c: dict, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    res = (run_fake_spectra if c["code"] == "fake_spectra" else run_skewerjax)(c, out)
    rec = dict(id=config_id(c), config=c, status="ok", wall_s=time.perf_counter() - t0, metadata=machine_metadata(), **res)
    (out / f"{config_id(c)}.json").write_text(json.dumps(rec, indent=1))


# --------------------------------------------------------------------------- the suite
def make_config(code: str, snap_base, snapnum: int, n_skewers: int, *, platform: str = "cpu", threads: int = 32,
                dtype: Optional[str] = None, repeats: int = 3, budget_s: float = 1800.0, res: float = 1.0,
                spacing_kms: float = 0.25, n_save: int = 8, n_agree: int = 4) -> dict:
    return dict(code=code, snap_base=str(snap_base), snapnum=int(snapnum), n_skewers=int(n_skewers), platform=platform,
                threads=int(threads), dtype=dtype, repeats=repeats, budget_s=budget_s, res=res, spacing_kms=spacing_kms,
                n_save=n_save, n_agree=n_agree)


def worker_env(c: dict) -> dict:
    """Environment of a worker: the platform, OpenMP threads, no GPU preallocation, CUDA libraries of the pip wheels."""
    root = str(Path(__file__).resolve().parents[2])
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([root] + [x for x in [os.environ.get("PYTHONPATH")] if x]),
               OMP_NUM_THREADS=str(c["threads"]), XLA_PYTHON_CLIENT_PREALLOCATE="false",
               JAX_PLATFORMS="cuda" if c["platform"] == "gpu" else "cpu")
    if c["platform"] == "gpu":
        import site
        libs = sorted(str(p) for s in site.getsitepackages() for p in Path(s).glob("nvidia/*/lib"))
        env["LD_LIBRARY_PATH"] = ":".join(libs + [env.get("LD_LIBRARY_PATH", "")])
    return env


def run_config(c: dict, out, *, cap_s: float = 1800.0, log=print) -> dict:
    """One configuration in its own process (skipped if its result exists); a run over ``cap_s`` is recorded as timeout."""
    out = Path(out)
    f = out / f"{config_id(c)}.json"
    if f.exists():
        log(f"skip {f.name} (exists)")
        return json.loads(f.read_text())
    out.mkdir(parents=True, exist_ok=True)
    cpus = CPU_SETS.get(c["threads"], list(range(c["threads"])))
    cmd = [sys.executable, str(Path(__file__).resolve()), "worker", "--config", json.dumps(c), "--out", str(out)]
    log(f"run  {config_id(c)} ...")
    t0 = time.perf_counter()
    try:
        p = subprocess.run(cmd, env=worker_env(c), preexec_fn=lambda: os.sched_setaffinity(0, cpus), capture_output=True,
                           text=True, timeout=cap_s)
        if p.returncode != 0:
            rec = dict(id=config_id(c), config=c, status="error", wall_s=time.perf_counter() - t0, stderr=p.stderr[-4000:])
            f.with_suffix(".error.json").write_text(json.dumps(rec, indent=1))
            log(f"     error ({p.stderr.strip().splitlines()[-1] if p.stderr.strip() else p.returncode})")
            return rec
    except subprocess.TimeoutExpired:
        rec = dict(id=config_id(c), config=c, status="timeout", wall_s=cap_s)
        f.write_text(json.dumps(rec, indent=1))
        log(f"     timeout after {cap_s:.0f} s")
        return rec
    rec = json.loads(f.read_text())
    log(f"     done in {rec['wall_s']:.1f} s, peak RSS {rec.get('peak_rss_GB', float('nan')):.2f} GB")
    return rec


def run_suite(configs, out, *, cap_s: float = 1800.0, log=print) -> list:
    """Every configuration in order (fake_spectra before SkewerJAX for the agreement); existing results are kept."""
    return [run_config(c, out, cap_s=cap_s, log=log) for c in configs]


def load_results(out) -> list:
    """Every ``<id>.json`` result in ``out`` (errors excluded)."""
    return [json.loads(p.read_text()) for p in sorted(Path(out).glob("*.json")) if not p.name.endswith(".error.json")]


def summary(rec: dict) -> dict:
    """Median and spread (min, max) of the total and of every stage over the repeats of one result."""
    reps = rec.get("repeats") or []
    if not reps:
        return {}
    tot = np.array([r["total"] for r in reps])
    out = dict(total=(float(np.median(tot)), float(tot.min()), float(tot.max())), n_repeats=len(reps))
    for s in STAGES:
        x = np.array([r["stages"][s] for r in reps])
        out[s] = (float(np.median(x)), float(x.min()), float(x.max()))
    if rec["config"]["code"] == "skewerjax" and len(reps) > 1:
        x = np.array([r["stages"]["tau"] for r in reps[1:]])
        out["tau_warm"] = (float(np.median(x)), float(x.min()), float(x.max()))
    return out


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
