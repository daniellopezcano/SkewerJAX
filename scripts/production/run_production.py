#!/usr/bin/env python
"""Launch SkewerJAX production runs: one run, or a named sweep of runs, resumably (README.md in this directory).

    python scripts/production/run_production.py --sweep all --dry-run             # what would run: disk, time, memory
    python scripts/production/run_production.py --sweep all                       # the campaign (resumable: rerun to resume)
    python scripts/production/run_production.py --sweep pixels                    # one sweep
    python scripts/production/run_production.py --run my_run --sim TNG50-3 --n 128 --dv-kms 2   # one run
    python scripts/production/run_production.py --status --sweep all              # the state of every run
    python scripts/production/run_production.py --list                            # the sweeps and their runs

One process computes one run at a time, each in its own child process (its log ``<run>/run.log``); the launcher writes
the sweep's ``index.json`` and ``sweep.log``. Every run is a lattice of a dataset of the campaign's store ``<out>/store``:
runs on the same dataset share it (a finished or coarser lattice computes nothing, a finer one extends it). Before each
run the available memory is checked against the run's estimated peak: the run waits while not even one tile fits, and
otherwise uses as many passes as the memory needs. A failed run is recorded and the next one starts; the failures are
listed at the end. Output: ``--out`` (default ``$SKEWERJAX_PRODUCTION`` or ``~/skewerjax_production``), never inside the
repository or a simulation's folder. Exit codes: 0 done (or dry run), 1 a run failed, 2 bad arguments, 3 refused (disk
space, overwrite, another launcher), 130 interrupted.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
os.environ.setdefault("JAX_PLATFORMS", "cpu")  # the launcher only plans: JAX (started by exojax at import) stays on the CPU

import production_estimates as pe  # noqa: E402
import production_runs as pr  # noqa: E402
import production_sweeps as ps  # noqa: E402

DISK_MARGIN = 1.10  # the estimate of the disk is multiplied by this before the check
MEMORY_MARGIN = 1.2  # a run starts when MemAvailable >= MEMORY_MARGIN x its smallest pass (one tile) + MEMORY_RESERVE
MEMORY_RESERVE = 4e9
MEMORY_POLL_S = 30.0
REUSE_S = 15.0  # a run whose lattice is stored: the child's start (JAX, the snapshot's header) and the index of the dataset


# --------------------------------------------------------------------------- arguments
def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(__doc__.splitlines()[2:9]),
    )
    what = ap.add_mutually_exclusive_group()
    what.add_argument("--sweep", help=f"a named sweep: {', '.join(sorted(ps.SWEEPS))}, or all ({', '.join(ps.ALL_ORDER)})")
    what.add_argument("--run", metavar="NAME", help="one run, named NAME (in <out>/single/NAME), configured by the options below")
    what.add_argument("--list", action="store_true", help="list the sweeps and their runs, and exit")
    what.add_argument("--child", metavar="RUN_DIR", help=argparse.SUPPRESS)  # internal: compute one run in this process
    ap.add_argument("--out", help="the output folder (default $SKEWERJAX_PRODUCTION or ~/skewerjax_production)")
    ap.add_argument(
        "--dry-run", action="store_true", help="print every run with its estimated disk, time and memory; write nothing"
    )
    ap.add_argument("--status", action="store_true", help="print the state of every run of --sweep / --run and exit")
    ap.add_argument(
        "--force",
        action="store_true",
        help="recompute finished runs too: their datasets are deleted first (every run sharing one loses its skewers)",
    )
    ap.add_argument("--only", nargs="+", metavar="RUN", help="only these runs of the sweep")
    ap.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="FIELD=VALUE",
        help="override a field of every run of the sweep (e.g. device=gpu, tile=512); JSON values",
    )
    ap.add_argument("--reserve-gb", type=float, default=10.0, help="free space [GB] to keep on the output disk")
    g = ap.add_argument_group("one run (--run): the fields of production_runs.RunConfig; defaults = the benchmark setup")
    d = pr.RunConfig()
    g.add_argument("--sim", default=d.sim, help="TNG50-2, TNG50-3, TNG50-4, ... (a folder of $SKEWERJAX_DATA)")
    g.add_argument("--sim-path", help="the simulation's folder (with snapdir_<NNN>/), if not <DATA_DIR>/<sim>")
    g.add_argument("--snap", type=int, default=d.snap)
    g.add_argument("--lines", nargs="+", default=list(d.lines), help="line ids, the first the reference")
    g.add_argument("--n", type=int, default=d.n, help="skewers per transverse axis (a power of two): the n x n lattice")
    g.add_argument("--axis", type=int, choices=(1, 2, 3), default=d.axis, help="the line of sight: 1, 2, 3 = x, y, z")
    g.add_argument("--dv-kms", type=float, default=d.dv_kms, help="the target pixel width [km/s]")
    g.add_argument("--n-pix", type=int, help="the number of pixels over the period (instead of --dv-kms)")
    g.add_argument("--points-cMpc", type=float, help="the spacing of the sampling points [cMpc] (segments, exact frame's a(s))")
    g.add_argument("--points-kms", type=float, help="the same in km/s of Hubble flow (instead of --points-cMpc)")
    g.add_argument("--mode", choices=("default", "matched"), default=d.mode, help="the preset of the options")
    g.add_argument(
        "--opt",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override an option of the preset (e.g. pieces=segments, profile=midpoint, frame=exact)",
    )
    g.add_argument("--precision", choices=("float32", "float64"), default=d.precision)
    g.add_argument("--device", choices=("cpu", "gpu"), default=d.device)
    g.add_argument("--tile", type=int, default=d.tile, help="skewers per tile (one batch file each)")
    g.add_argument("--threads", type=int, help="threads of the pass (default: every core)")
    return ap


def _json_value(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _pairs(items, what: str) -> dict:
    out = {}
    for x in items:
        if "=" not in x:
            raise SystemExit(f"{what} expects KEY=VALUE, got {x!r}")
        k, v = x.split("=", 1)
        out[k.strip()] = _json_value(v.strip())
    return out


def _git_ignored(path: Path) -> bool:
    """Whether git ignores ``path`` inside the repository (``git check-ignore``, read only; False if git is absent)."""
    try:
        return subprocess.run(["git", "-C", str(REPO), "check-ignore", "-q", str(path)], timeout=10).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def resolve_out(out) -> Path:
    """The output folder: refused inside the simulations' folder (read only) and inside the repository, unless git ignores
    it there (e.g. a scratch folder of ``claude_agents/``): outputs never enter the repository's tracked tree."""
    from skewerjax.utils.paths import DATA_DIR, PRODUCTION_DIR

    p = Path(out).expanduser().resolve() if out else Path(PRODUCTION_DIR).resolve()
    data = Path(DATA_DIR).resolve()
    if p == data or data in p.parents:
        raise SystemExit(f"--out {p} is inside the simulations' folder {data} (read only): choose another folder.")
    if (p == REPO or REPO in p.parents) and not _git_ignored(p):
        raise SystemExit(f"--out {p} is inside the repository and not git-ignored: choose another folder.")
    return p


def plan_runs(a) -> list:
    """[(sweep, [RunConfig])] for the arguments."""
    if a.run:
        n_pix = a.n_pix
        cfg = pr.RunConfig(
            name=a.run,
            sim=a.sim,
            sim_path=a.sim_path,
            snap=a.snap,
            lines=tuple(a.lines),
            n=a.n,
            axis=a.axis,
            dv_kms=None if n_pix else a.dv_kms,
            n_pix=n_pix,
            points_kms=a.points_kms,
            points_cMpc=a.points_cMpc,
            mode=a.mode,
            options=_pairs(a.opt, "--opt"),
            precision=a.precision,
            device=a.device,
            tile=a.tile,
            threads=a.threads,
        ).validate()
        return [("single", [cfg])]
    names = ps.ALL_ORDER if a.sweep == "all" else (a.sweep,)
    overrides = _pairs(a.set, "--set")
    out = []
    for s in names:
        runs = ps.sweep_runs(s, overrides)
        if a.only:
            runs = [r for r in runs if r.name in a.only]
        out.append((s, runs))
    if a.only and not any(r for _, r in out):
        raise SystemExit(f"--only {a.only}: no such run in {list(names)}")
    return out


# --------------------------------------------------------------------------- estimates
_HEADERS: dict = {}


def _header(cfg: pr.RunConfig):
    """The snapshot's header (cached), or None if it cannot be read (the estimate then uses the records' grid)."""
    key = (str(cfg.resolved_sim_path()), cfg.snap)
    if key not in _HEADERS:
        try:
            from skewerjax.simulations import ArepoHDF5Reader

            _HEADERS[key] = ArepoHDF5Reader(cfg.resolved_sim_path(), cfg.snap).header
        except Exception:  # noqa: BLE001 (any unreadable snapshot: the estimate falls back)
            _HEADERS[key] = None
    return _HEADERS[key]


def _options(cfg: pr.RunConfig, H) -> dict:
    import skewer_extraction as se

    return se.dataset_options(
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


_DATASETS: dict = {}


def stored_in_lattice(out: Path, cfg: pr.RunConfig, chash: str) -> int:
    """How many skewers of the run's n x n lattice its dataset holds already (0 if the dataset does not exist)."""
    import skewer_store as ss

    path = ss.dataset_path(out / pr.STORE_DIR, cfg.sim, cfg.snap, cfg.axis, chash)
    if not (path / "config.json").exists():
        return 0
    ds = ss.SkewerDataset.open(out / pr.STORE_DIR, cfg.sim, cfg.snap, cfg.axis, chash)
    return int(ds.stored(ss.lattice_indices(cfg.n, ds.n_ref)).sum())


def estimate(cfg: pr.RunConfig, out: Path | None = None, planned: dict | None = None) -> dict:
    """The cost of one run (``production_estimates``): dict(dataset (sim, snap, axis, hash), n_new (the skewers its
    dataset does not hold, nor a run planned before it in ``planned``: {dataset: n}), n_pix, dv_kms, disk_bytes, wall_s,
    peak_GB, n_passes, skewers_per_pass, min_pass_GB (one tile), route, basis)."""
    import skewer_extraction as se
    import skewer_store as ss

    H = _header(cfg)
    n_sk, n_lines = cfg.n * cfg.n, len(cfg.lines)
    route = cfg.route()
    if H is not None:
        opts = _options(cfg, H)
        n_pix, dv, n_gas, how = int(opts["pixels"]["n_pix"]), float(opts["pixels"]["dv_kms"]), int(H.n_gas), "header"
        chash, L, u = ss.config_hash(opts), float(H.box_cMpc), float(H.velfac_kms_per_cMpc)
    else:
        n_pix = pe.n_pix_estimate(cfg.sim, cfg.dv_kms, cfg.n_pix)
        dv, n_gas, how = pe.NBINS_1KMS.get(cfg.sim, 3961) / n_pix, pe.N_GAS.get(cfg.sim, pe.N_GAS["TNG50-2"]), "no header"
        chash, L, u = None, pe.BOX_CMPC, pe.U_KMS_PER_CMPC
    did = (cfg.sim, cfg.snap, cfg.axis, chash)
    stored = stored_in_lattice(out, cfg, chash) if (out is not None and chash) else 0
    covered = max(stored, min((planned or {}).get(did, 0), cfg.n) ** 2)
    n_new = n_sk - covered
    ds = cfg.points_cMpc if cfg.points_cMpc is not None else (None if cfg.points_kms is None else cfg.points_kms / u)
    pts = int(round(L / ds)) if (route == "sampled" and ds) else None
    budget = cfg.memory_fraction * se.memory_available()
    work = max(n_new, 1)
    per = se.pass_size(work, n_gas, n_lines, n_pix, cfg.precision, budget, points_per_skewer=pts, n_lattice=cfg.n)
    n_passes = -(-n_new // per) if n_new else 0
    peak = (
        se.pass_memory(min(per, work), n_gas, n_lines, n_pix, cfg.precision, points_per_skewer=pts, n_lattice=cfg.n)
        if n_new
        else 0.0
    )
    min_pass = se.pass_memory(min(cfg.tile, work), n_gas, n_lines, n_pix, cfg.precision, points_per_skewer=pts, n_lattice=cfg.n)
    if not n_new:
        t = dict(total_s=REUSE_S, pass_s=0.0, basis="the lattice is stored (or computed earlier in the plan): nothing to compute")
    elif route == "sampled":
        cells = se.sampled_cells(min(per, work), n_gas, pts, cfg.n)
        t = pe.sampled_time(cfg.sim, min(per, work), pts, cells, precision=cfg.precision)
        t["total_s"] += (n_new - min(per, work)) * (t["tau_s"] / min(per, work))  # the pieces of every pass
    else:
        t = pe.streaming_time(cfg.sim, n_new, dv, mode=cfg.mode, device=cfg.device, precision=cfg.precision)
    wall = t["total_s"] + max(n_passes - 1, 0) * t["pass_s"]
    return dict(
        dataset=list(did),
        n_new=int(n_new),
        n_stored=int(stored),
        n_pix=n_pix,
        dv_kms=dv,
        n_pix_from=how,
        points_per_skewer=pts,
        disk_bytes=pe.disk_bytes(n_new, n_lines, n_pix, cfg.tile) if n_new else 0.0,
        wall_s=wall,
        wall_s_raw=wall,
        peak_GB=peak / 1e9,
        min_pass_GB=min_pass / 1e9,
        n_passes=n_passes,
        skewers_per_pass=per,
        route=route,
        basis=t["basis"],
    )


CALIBRATION_MIN_SKEWERS = 1024  # runs that computed fewer skewers (the smoke runs) are dominated by fixed costs


def calibration(out: Path, route: str) -> tuple:
    """(factor, n): the median of measured / estimated wall time over the finished runs under ``out`` of ``route`` that
    computed at least ``CALIBRATION_MIN_SKEWERS`` skewers (1.0, 0 if none). Applied to the later estimates of the route."""
    ratios = []
    for p in sorted(out.glob(f"*/*/{pr.RUN_FILE}")):
        rec = pr.read_record(p.parent) or {}
        e, t = rec.get("estimate") or {}, rec.get("timings") or {}
        big = (t.get("computed") or 0) >= CALIBRATION_MIN_SKEWERS
        if rec.get("status") == "done" and e.get("route") == route and big and e.get("wall_s_raw"):
            ratios.append(float(rec["wall_s"]) / float(e["wall_s_raw"]))
    if not ratios:
        return 1.0, 0
    ratios.sort()
    m = len(ratios)
    return (ratios[m // 2] if m % 2 else 0.5 * (ratios[m // 2 - 1] + ratios[m // 2])), m


def _fmt_s(x: float) -> str:
    if x != x:
        return "?"
    return f"{x:.0f} s" if x < 120 else (f"{x / 60:.1f} min" if x < 7200 else f"{x / 3600:.1f} h")


def _fmt_b(x: float) -> str:
    return f"{x / 1e9:.2f} GB" if x >= 1e8 else f"{x / 1e6:.1f} MB"


def _free_bytes(out: Path) -> float:
    p = next(x for x in [out, *out.parents] if x.exists())
    return float(shutil.disk_usage(p).free)


# --------------------------------------------------------------------------- the sweep
def _mem_total() -> float:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return float(line.split()[1]) * 1024
    except OSError:
        pass
    return float("nan")


class Launcher:
    def __init__(self, out: Path, a):
        self.out, self.a = out, a
        self.child = None
        self.log_files = []
        self.stop = False
        self.failures = []  # (sweep, run, state, log) of the runs that did not finish

    def log(self, msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        for f in self.log_files:
            f.write(line + "\n")
            f.flush()

    def _signal(self, signum, frame):
        self.stop = True
        if self.child is not None and self.child.poll() is None:
            self.child.send_signal(signal.SIGTERM)  # the child records "interrupted" and stops

    def rows(self, sweep: str, runs: list, planned: dict | None = None) -> list:
        """The plan of a sweep: per run its folder, state and estimate. ``planned`` ({dataset: n}) carries the lattices
        that runs before it will compute (in this sweep and, for the dry run of "all", the sweeps before): a run they
        cover computes nothing. The estimates are calibrated by the finished runs of the same route (``calibration``)."""
        planned = {} if planned is None else planned
        rows = []
        for cfg in runs:
            d = self.out / sweep / cfg.name
            st = pr.run_state(d)
            todo = st != "done" or self.a.force
            e = estimate(cfg, self.out, planned if not self.a.force else {})
            if todo and self.a.force:
                e["n_new"] = cfg.n * cfg.n  # the dataset is deleted first
            f, m = calibration(self.out, e["route"])
            if m and e["n_new"]:
                e["wall_s"] = e["wall_s_raw"] * f
                e["calibration"] = dict(factor=f, runs=m)
            if todo:
                did = tuple(e["dataset"])
                planned[did] = max(planned.get(did, 0), cfg.n)
            rows.append(dict(cfg=cfg, dir=d, state=st, est=e, todo=todo))
        return rows

    @staticmethod
    def action(row) -> str:
        if not row["todo"]:
            return "skip"
        e = row["est"]
        if not e["n_new"]:
            return "reuse"
        return "extend" if e["n_stored"] or e["n_new"] < row["cfg"].n ** 2 else "compute"

    def print_plan(self, sweep: str, rows: list) -> None:
        _, varied, desc = ps.SWEEPS.get(sweep, (None, "-", "one run"))
        self.log(f"sweep {sweep}: {desc} (varied: {varied}); {len(rows)} run(s) in {self.out / sweep}")
        self.log(
            f"  {'run':<14} {'state':<11} {'action':<7} {'new sk.':>8} {'disk':>9} {'time':>9} {'peak':>8} {'passes':>6}  "
            f"{'dataset':<10} configuration"
        )
        for r in rows:
            e = r["est"]
            self.log(
                f"  {r['cfg'].name:<14} {r['state']:<11} {self.action(r):<7} {e['n_new']:>8} {_fmt_b(e['disk_bytes']):>9} "
                f"{_fmt_s(e['wall_s']):>9} {e['peak_GB']:>6.1f} G {e['n_passes']:>6}  {str(e['dataset'][3]):<10} "
                f"{r['cfg'].summary()}"
            )
        comp = [r for r in rows if r["todo"] and r["est"]["n_new"]]
        self.log(
            f"  to compute: {len(comp)} run(s), {sum(r['est']['n_new'] for r in comp)} skewers, "
            f"{_fmt_b(sum(r['est']['disk_bytes'] for r in comp))}, {_fmt_s(sum(r['est']['wall_s'] for r in rows if r['todo']))}, "
            f"peak {max([r['est']['peak_GB'] for r in comp] or [0]):.1f} GB; reuse: "
            f"{sum(1 for r in rows if self.action(r) == 'reuse')}; finished: {sum(1 for r in rows if not r['todo'])}"
        )
        for basis in dict.fromkeys(r["est"]["basis"] for r in rows if r["todo"]):
            self.log(f"  time basis: {basis}")
        for r in rows:
            c = r["est"].get("calibration")
            if c:
                self.log(
                    f"  {r['cfg'].name}: estimate x {c['factor']:.2f} (the median measured / estimated of {c['runs']} run(s))"
                )
                break

    def index(self, sweep: str, rows: list) -> None:
        _, varied, desc = ps.SWEEPS.get(sweep, (None, "-", "single runs"))
        runs = []
        for r in rows:
            rec = pr.read_record(r["dir"]) or {}
            runs.append(
                dict(
                    name=r["cfg"].name,
                    path=r["cfg"].name,
                    state=pr.run_state(r["dir"]),
                    config=r["cfg"].to_dict(),
                    estimate=r["est"],
                    dataset=rec.get("dataset"),
                    sampling=rec.get("sampling"),
                    wall_s=rec.get("wall_s"),
                    computed=(rec.get("timings") or {}).get("computed"),
                    peak_rss_GB=(rec.get("memory") or {}).get("peak_rss_GB"),
                    finished=rec.get("finished"),
                )
            )
        idx = dict(
            schema="skewerjax-production-sweep/2",
            sweep=sweep,
            description=desc,
            varied=varied,
            fiducial=ps.FIDUCIAL.to_dict(),
            store=os.path.relpath(self.out / pr.STORE_DIR, self.out / sweep),
            updated=pr.now(),
            runs=runs,
        )
        p = self.out / sweep / "index.json"
        tmp = p.with_name(".index.json.tmp")
        tmp.write_text(json.dumps(idx, indent=1, default=str) + "\n")
        os.replace(tmp, p)

    def wait_for_memory(self, r) -> bool:
        """Wait until the smallest pass of the run (one tile) fits the available memory with the margin; then log how
        many passes the memory allows (the child sizes its passes from MemAvailable as it starts). False if stopped."""
        import skewer_extraction as se

        e, cfg = r["est"], r["cfg"]
        need = MEMORY_MARGIN * e["min_pass_GB"] * 1e9 + MEMORY_RESERVE
        t0, said = time.time(), -1.0
        while True:
            avail = se.memory_available()
            if avail >= need:
                break
            if self.stop:
                return False
            if time.time() - said > 600:
                self.log(
                    f"{cfg.name}: waiting for memory: {avail / 1e9:.1f} GB available, {need / 1e9:.1f} GB needed for one tile "
                    f"(waiting {_fmt_s(time.time() - t0)})"
                )
                said = time.time()
            time.sleep(MEMORY_POLL_S)
        n_lines, n_pix = len(cfg.lines), e["n_pix"]
        H = _header(cfg)
        n_gas = int(H.n_gas) if H is not None else pe.N_GAS.get(cfg.sim, pe.N_GAS["TNG50-2"])
        pts = e["points_per_skewer"]
        per = se.pass_size(
            e["n_new"], n_gas, n_lines, n_pix, cfg.precision, cfg.memory_fraction * avail, points_per_skewer=pts, n_lattice=cfg.n
        )
        k = -(-e["n_new"] // per)
        peak = se.pass_memory(min(per, e["n_new"]), n_gas, n_lines, n_pix, cfg.precision, points_per_skewer=pts, n_lattice=cfg.n)
        more = f" (planned {e['n_passes']}: more passes for the memory available)" if k > e["n_passes"] else ""
        self.log(
            f"{cfg.name}: memory {avail / 1e9:.1f} GB available: {k} pass(es) of <= {per} skewers, estimated peak "
            f"{peak / 1e9:.1f} GB{more}"
        )
        return True

    def run_sweep(self, sweep: str, runs: list) -> int:
        folder = self.out / sweep
        folder.mkdir(parents=True, exist_ok=True)
        lock = open(folder / ".lock", "a+")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.log(f"REFUSED: another launcher is running the sweep {sweep} ({folder / '.lock'} is held).")
            return 3
        logf = open(folder / "sweep.log", "a")
        self.log_files.append(logf)
        status = 0
        try:
            rows = self.rows(sweep, runs)
            self.print_plan(sweep, rows)
            self.index(sweep, rows)
            for r in rows:
                if self.stop:
                    break
                cfg, d = r["cfg"], r["dir"]
                if not r["todo"]:
                    self.log(f"{cfg.name}: finished, skipped (--force to recompute)")
                    continue
                if r["state"] == "running":
                    self.log(f"{cfg.name}: REFUSED: running in process {pr.read_record(d).get('pid')} (another launcher?)")
                    status = 3
                    continue
                # the estimate again, now that the runs before it have finished (what they stored, the calibration)
                r.update(self.rows(sweep, [cfg])[0])
                if self.a.force:
                    self.force_dataset(r)
                e = r["est"]
                need = DISK_MARGIN * e["disk_bytes"] + self.a.reserve_gb * 1e9
                free = _free_bytes(self.out)
                if need > free:
                    self.log(f"{cfg.name}: REFUSED: it needs {_fmt_b(need)} with the reserve, {_fmt_b(free)} free on {self.out}.")
                    status = 3
                    break
                if r["state"] != "absent":
                    self.log(f"{cfg.name}: {r['state']} -> redone (the stored batches of its dataset are kept)")
                pr.reset(d, self.out)
                if e["n_new"] and not self.wait_for_memory(r):
                    break
                template = dict(
                    schema=pr.SCHEMA,
                    name=cfg.name,
                    sweep=sweep,
                    config=cfg.to_dict(),
                    key=cfg.key(),
                    estimate=e,
                    created=pr.now(),
                )
                code = self.launch(d, template, self.action(r))
                if code != 0:
                    st = 130 if (code == 130 or self.stop) else 1
                    self.log(f"{cfg.name}: {'interrupted' if st == 130 else 'FAILED'} (exit {code}; {d / pr.LOG_FILE})")
                    if st == 130:
                        status = 130
                        break
                    status = 1
                    self.failures.append((sweep, cfg.name, pr.run_state(d), str(d / pr.LOG_FILE)))
                self.index(sweep, rows)
            self.index(sweep, rows)
            states = [pr.run_state(r["dir"]) for r in rows]
            self.log(
                f"sweep {sweep}: {states.count('done')}/{len(rows)} done"
                + (f"; states {dict(zip([r['cfg'].name for r in rows], states))}" if states.count("done") < len(rows) else "")
            )
        finally:
            self.log_files.remove(logf)
            logf.close()
            fcntl.flock(lock, fcntl.LOCK_UN)
            lock.close()
        return status

    def force_dataset(self, r) -> None:
        """--force: delete the run's dataset (inside <out>/store) so that it is recomputed; log the runs that read it."""
        import skewer_store as ss

        sim, snap, axis, chash = r["est"]["dataset"]
        path = ss.dataset_path(self.out / pr.STORE_DIR, sim, snap, axis, chash)
        if (path / "config.json").exists():
            others = [str(x.relative_to(self.out)) for x in pr.runs_on_dataset(self.out, chash)]
            freed = pr.remove_dataset(path, self.out)
            self.log(
                f"{r['cfg'].name}: --force: dataset {chash} deleted ({_fmt_b(freed)}); the runs that read it: {others or 'none'}"
            )

    def launch(self, d: Path, template: dict, action: str = "compute") -> int:
        """Compute one run in a child process (``--child``), its output to ``run.log``; the signals reach it."""
        rec = dict(template, status="launched", host=os.uname().nodename, launched=pr.now())
        pr.write_record(d, rec)
        cfg, e = template["config"], template["estimate"]
        self.log(
            f"{template['name']}: {action} ({pr.RunConfig.from_dict(cfg).summary()}); {e['n_new']} new skewers, estimate "
            f"{_fmt_s(e['wall_s'])}, log {d / pr.LOG_FILE}"
        )
        env = dict(os.environ)
        env.pop("JAX_PLATFORMS", None)  # the child sets its own platforms (configure_devices)
        t0 = time.perf_counter()
        with open(d / pr.LOG_FILE, "a") as logf:
            self.child = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), "--child", str(d)],
                stdout=logf,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=str(HERE),
            )
            now_rec = pr.read_record(d) or rec
            if now_rec.get("status") == "launched":  # the child writes "running" with its own pid as it starts
                pr.write_record(d, dict(now_rec, pid=self.child.pid))
            code = self.child.wait()
            self.child = None
        done = pr.run_state(d) == "done"  # the record decides (a signal may reach a child that has just finished)
        fin = pr.read_record(d) or {}
        extra = ""
        if done:
            t = fin.get("timings") or {}
            extra = (
                f"; {t.get('computed', 0)} computed, {t.get('reused', 0)} reused; peak RSS "
                f"{(fin.get('memory') or {}).get('peak_rss_GB', float('nan')):.1f} GB (estimated {e['peak_GB']:.1f})"
            )
        self.log(
            f"{template['name']}: {'done' if done else 'not done'} in {_fmt_s(time.perf_counter() - t0)} (estimated "
            f"{_fmt_s(e['wall_s'])}; exit {code}){extra}"
        )
        if done:
            return 0
        return 130 if code in (130, -signal.SIGTERM, -signal.SIGINT, -signal.SIGHUP) else (code or 1)


# --------------------------------------------------------------------------- main
def _interrupt(signum, frame):
    raise KeyboardInterrupt(f"signal {signal.Signals(signum).name}")


def main(argv=None) -> int:
    a = _parser().parse_args(argv)
    if a.child:  # a kill (SIGTERM) or a closed terminal (SIGHUP) stops like Ctrl-C: the record says "interrupted"
        signal.signal(signal.SIGTERM, _interrupt)
        signal.signal(signal.SIGHUP, _interrupt)
        return pr.execute(a.child)
    if a.list:
        for name, (make, varied, desc) in ps.SWEEPS.items():
            print(f"{name}: {desc} (varied: {varied})")
            for r in make():
                print(f"    {r.name:<14} {r.summary()}")
        print(f"all: {', '.join(ps.ALL_ORDER)} in this order (runs on the same dataset share it)")
        return 0
    if not (a.sweep or a.run):
        _parser().error("give --sweep NAME, --run NAME or --list")
    try:
        plan = plan_runs(a)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    out = resolve_out(a.out)
    L = Launcher(out, a)
    if a.status:
        for sweep, runs in plan:
            for cfg in runs:
                d = out / sweep / cfg.name
                rec = pr.read_record(d) or {}
                t = rec.get("timings") or {}
                wall = f"{_fmt_s(rec['wall_s']):>9}" if rec.get("wall_s") else " " * 9
                comp = f"{t.get('computed', ''):>7}" if t else " " * 7
                print(
                    f"{sweep + '/' + cfg.name:<26} {pr.run_state(d):<12} {wall} {comp}  "
                    f"{rec.get('finished') or rec.get('started') or rec.get('launched') or ''}  {d}"
                )
        return 0
    if a.dry_run:
        planned, allrows = {}, []
        for sweep, runs in plan:
            rows = L.rows(sweep, runs, planned)
            L.print_plan(sweep, rows)
            allrows += rows
        comp = [r for r in allrows if r["todo"] and r["est"]["n_new"]]
        import skewer_extraction as se

        L.log(
            f"dry run: {len(comp)} run(s) to compute, {sum(1 for r in allrows if L.action(r) == 'reuse')} reusing a dataset; "
            f"disk {_fmt_b(sum(r['est']['disk_bytes'] for r in comp))} ({_fmt_b(_free_bytes(out))} free on {out}), "
            f"time {_fmt_s(sum(r['est']['wall_s'] for r in allrows if r['todo']))}, peak {max([0] + [r['est']['peak_GB'] for r in comp]):.1f} GB "
            f"(RAM: {se.memory_available() / 1e9:.0f} GB available of {_mem_total() / 1e9:.0f} GB); nothing written"
        )
        return 0
    if a.run:
        d = out / "single" / a.run
        if pr.run_state(d) == "done" and not a.force:
            L.log(f"REFUSED: the run {d} is finished; --force recomputes it (its dataset is deleted first).")
            return 3
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, L._signal)
    status = 0
    for sweep, runs in plan:
        if L.stop:
            break
        s = L.run_sweep(sweep, runs)
        status = s if s and not status else status
        if s in (3, 130):
            break
    if L.failures:
        L.log(f"FAILED runs ({len(L.failures)}): " + "; ".join(f"{s}/{n} ({st}, {lg})" for s, n, st, lg in L.failures))
    elif not L.stop and len(plan) > 1:
        L.log("every run of " + ", ".join(s for s, _ in plan) + " is done" if status == 0 else f"exit {status}")
    return 130 if L.stop else status


if __name__ == "__main__":
    sys.exit(main())
