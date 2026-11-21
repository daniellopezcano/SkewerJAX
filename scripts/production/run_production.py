#!/usr/bin/env python
"""Launch SkewerJAX production runs: one run, or a named sweep of runs, resumably (README.md in this directory).

    python scripts/production/run_production.py --sweep pixels --dry-run          # what would run: disk, time, memory
    python scripts/production/run_production.py --sweep pixels                    # run it (resumable: rerun to resume)
    python scripts/production/run_production.py --sweep all                       # the four checks, sharing identical runs
    python scripts/production/run_production.py --run my_run --sim TNG50-3 --n 128 --dv-kms 2   # one run
    python scripts/production/run_production.py --status --sweep pixels           # the state of every run
    python scripts/production/run_production.py --list                            # the sweeps and their runs

Every run is computed in its own child process (its log ``<run>/run.log``), one after another; the launcher writes the
sweep's ``index.json`` and ``sweep.log``. Output: ``--out`` (default ``$SKEWERJAX_PRODUCTION`` or
``~/skewerjax_production``), never inside the repository or a simulation's folder. Exit codes: 0 done (or dry run),
1 a run failed, 2 bad arguments, 3 refused (disk space, overwrite, another launcher), 130 interrupted.
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
    ap.add_argument("--force", action="store_true", help="recompute finished runs too (their folders are emptied first)")
    ap.add_argument("--only", nargs="+", metavar="RUN", help="only these runs of the sweep")
    ap.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="FIELD=VALUE",
        help="override a field of every run of the sweep (e.g. device=gpu, tile=512); JSON values",
    )
    ap.add_argument("--reserve-gb", type=float, default=10.0, help="free space [GB] to keep on the output disk")
    ap.add_argument("--no-share", action="store_true", help="recompute runs identical to a finished run instead of linking it")
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
    g.add_argument("--points-kms", type=float, help="the spacing of the sampling points [km/s] (segments, exact frame)")
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


def estimate(cfg: pr.RunConfig) -> dict:
    """dict(n_pix, disk_bytes, wall_s, peak_GB, n_passes, basis) of one run (``production_estimates``)."""
    import skewer_extraction as se

    H = _header(cfg)
    if H is not None:
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
        n_pix, dv, n_gas, how = int(opts["pixels"]["n_pix"]), float(opts["pixels"]["dv_kms"]), int(H.n_gas), "header"
    else:
        n_pix = pe.n_pix_estimate(cfg.sim, cfg.dv_kms, cfg.n_pix)
        dv, n_gas, how = pe.NBINS_1KMS.get(cfg.sim, 3961) / n_pix, pe.N_GAS.get(cfg.sim, pe.N_GAS["TNG50-2"]), "no header"
    n_sk, n_lines = cfg.n * cfg.n, len(cfg.lines)
    o = se.mode_options(cfg.mode, cfg.options)
    budget = cfg.memory_fraction * se.memory_available()
    per = se.pass_size(n_sk, n_gas, n_lines, n_pix, cfg.precision, budget)
    n_passes = -(-n_sk // per)
    peak = se.pass_memory(min(per, n_sk), n_gas, n_lines, n_pix, cfg.precision)
    if o["pieces"] == "segments":
        t = pe.sampled_time(n_sk, float(cfg.points_kms), precision=cfg.precision)
    else:
        t = pe.streaming_time(cfg.sim, n_sk, dv, mode=cfg.mode, device=cfg.device, precision=cfg.precision)
    wall = t["total_s"] + (n_passes - 1) * t["pass_s"]
    return dict(
        n_pix=n_pix,
        n_pix_from=how,
        disk_bytes=pe.disk_bytes(n_sk, n_lines, n_pix, cfg.tile),
        wall_s=wall,
        peak_GB=peak / 1e9,
        n_passes=n_passes,
        skewers_per_pass=per,
        basis=t["basis"],
    )


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
class Launcher:
    def __init__(self, out: Path, a):
        self.out, self.a = out, a
        self.child = None
        self.log_files = []
        self.stop = False

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

    def rows(self, sweep: str, runs: list, planned: set | None = None) -> list:
        """The plan of a sweep: per run its folder, state, the finished run it would link (``share``; with ``planned``,
        the keys computed earlier in the same plan count too: the dry run of "all") and its estimate."""
        rows = []
        for cfg in runs:
            d = self.out / sweep / cfg.name
            st = pr.run_state(d)
            src = None
            if (st != "done" or self.a.force) and not self.a.no_share and not self.a.force:
                src = pr.find_shareable(self.out, cfg.key(), exclude=d)
                if src is None and planned is not None and cfg.key() in planned:
                    src = "(a run computed earlier in this plan)"
            if planned is not None and (st != "done" or self.a.force) and src is None:
                planned.add(cfg.key())
            rows.append(dict(cfg=cfg, dir=d, state=st, share=src, est=estimate(cfg)))
        return rows

    def todo(self, row) -> bool:
        return row["state"] != "done" or self.a.force

    def print_plan(self, sweep: str, rows: list) -> None:
        _, varied, desc = ps.SWEEPS.get(sweep, (None, "-", "one run"))
        self.log(f"sweep {sweep}: {desc} (varied: {varied}); {len(rows)} run(s) in {self.out / sweep}")
        self.log(f"  {'run':<20} {'state':<12} {'action':<10} {'disk':>9} {'time':>9} {'peak':>8} {'passes':>6}  configuration")
        for r in rows:
            e = r["est"]
            act = "skip" if not self.todo(r) else ("link" if r["share"] else "compute")
            self.log(
                f"  {r['cfg'].name:<20} {r['state']:<12} {act:<10} {_fmt_b(e['disk_bytes']):>9} {_fmt_s(e['wall_s']):>9} "
                f"{e['peak_GB']:>6.1f} G {e['n_passes']:>6}  {r['cfg'].summary()}"
            )
        comp = [r for r in rows if self.todo(r) and not r["share"]]
        self.log(
            f"  to compute: {len(comp)} run(s), {_fmt_b(sum(r['est']['disk_bytes'] for r in comp))}, "
            f"{_fmt_s(sum(r['est']['wall_s'] for r in comp))}, peak {max([r['est']['peak_GB'] for r in comp] or [0]):.1f} GB; "
            f"to link: {sum(1 for r in rows if self.todo(r) and r['share'])}; finished: {sum(1 for r in rows if not self.todo(r))}"
        )
        for basis in dict.fromkeys(r["est"]["basis"] for r in rows):
            self.log(f"  time basis: {basis}")

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
                    key=r["cfg"].key(),
                    config=r["cfg"].to_dict(),
                    estimate=r["est"],
                    dataset=rec.get("dataset"),
                    wall_s=rec.get("wall_s"),
                    shared_from=rec.get("shared_from"),
                    finished=rec.get("finished"),
                )
            )
        idx = dict(
            schema="skewerjax-production-sweep/1",
            sweep=sweep,
            description=desc,
            varied=varied,
            baseline=ps.BASELINE.to_dict(),
            updated=pr.now(),
            runs=runs,
        )
        p = self.out / sweep / "index.json"
        tmp = p.with_name(".index.json.tmp")
        tmp.write_text(json.dumps(idx, indent=1, default=str) + "\n")
        os.replace(tmp, p)

    def run_sweep(self, sweep: str, rows: list) -> int:
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
            self.print_plan(sweep, rows)
            self.index(sweep, rows)
            for r in rows:
                if self.stop:
                    break
                cfg, d = r["cfg"], r["dir"]
                if not self.todo(r):
                    self.log(f"{cfg.name}: finished, skipped (--force to recompute)")
                    continue
                if r["state"] == "running":
                    self.log(f"{cfg.name}: REFUSED: running in process {pr.read_record(d).get('pid')} (another launcher?)")
                    status = 3
                    continue
                need = DISK_MARGIN * (0 if r["share"] else r["est"]["disk_bytes"]) + self.a.reserve_gb * 1e9
                free = _free_bytes(self.out)
                if need > free:
                    self.log(f"{cfg.name}: REFUSED: it needs {_fmt_b(need)} with the reserve, {_fmt_b(free)} free on {self.out}.")
                    status = 3
                    break
                if r["state"] != "absent":
                    self.log(f"{cfg.name}: {r['state']} -> redone from scratch")
                pr.reset(d, self.out)
                template = dict(
                    schema=pr.SCHEMA,
                    name=cfg.name,
                    sweep=sweep,
                    config=cfg.to_dict(),
                    key=cfg.key(),
                    estimate=r["est"],
                    created=pr.now(),
                )
                src = pr.find_shareable(self.out, cfg.key(), exclude=d) if (r["share"] and not self.a.force) else None
                if src is not None:
                    pr.share(src, d, template)
                    self.log(f"{cfg.name}: identical to the finished run {src}: linked (no computation)")
                else:
                    code = self.launch(d, template)
                    if code != 0:
                        status = 130 if (code == 130 or self.stop) else 1
                        self.log(f"{cfg.name}: {'interrupted' if status == 130 else 'FAILED'} (exit {code}; {d / pr.LOG_FILE})")
                        if status == 130:
                            break
                self.index(sweep, rows)
            self.index(sweep, rows)
            states = [pr.run_state(r["dir"]) for r in rows]
            self.log(
                f"sweep {sweep}: {states.count('done')}/{len(rows)} done"
                + (f"; states {states}" if states.count("done") < len(rows) else "")
            )
        finally:
            self.log_files.remove(logf)
            logf.close()
            fcntl.flock(lock, fcntl.LOCK_UN)
            lock.close()
        return status

    def launch(self, d: Path, template: dict) -> int:
        """Compute one run in a child process (``--child``), its output to ``run.log``; the signals reach it."""
        rec = dict(template, status="launched", host=os.uname().nodename, launched=pr.now())
        pr.write_record(d, rec)
        cfg = template["config"]
        self.log(
            f"{template['name']}: computing ({pr.RunConfig.from_dict(cfg).summary()}); estimate "
            f"{_fmt_s(template['estimate']['wall_s'])}, log {d / pr.LOG_FILE}"
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
        self.log(f"{template['name']}: {'done' if done else 'not done'} in {_fmt_s(time.perf_counter() - t0)} (exit {code})")
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
                print(f"    {r.name:<20} {r.summary()}")
        print(f"all: {', '.join(ps.ALL_ORDER)} in this order (identical runs linked)")
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
                print(f"{sweep}/{cfg.name:<20} {pr.run_state(d):<12} {rec.get('finished') or rec.get('started') or ''}  {d}")
        return 0
    if a.dry_run:
        total_d = total_t = 0.0
        peak = 0.0
        planned = set()
        for sweep, runs in plan:
            rows = L.rows(sweep, runs, planned)
            L.print_plan(sweep, rows)
            comp = [r for r in rows if L.todo(r) and not r["share"]]
            total_d += sum(r["est"]["disk_bytes"] for r in comp)
            total_t += sum(r["est"]["wall_s"] for r in comp)
            peak = max([peak] + [r["est"]["peak_GB"] for r in comp])
        free = _free_bytes(out)
        L.log(
            f"dry run: to compute {_fmt_b(total_d)} ({_fmt_b(free)} free on {out}), {_fmt_s(total_t)}, peak {peak:.1f} GB "
            "(identical runs linked); nothing written"
        )
        return 0
    if a.run:
        d = out / "single" / a.run
        if pr.run_state(d) == "done" and not a.force:
            L.log(f"REFUSED: the run {d} is finished; --force recomputes it (its folder is emptied first).")
            return 3
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, L._signal)
    status = 0
    for sweep, runs in plan:
        if L.stop:
            break
        rows = L.rows(sweep, runs)
        s = L.run_sweep(sweep, rows)
        status = s if s and not status else status
        if s in (3, 130):
            break
    return 130 if L.stop else status


if __name__ == "__main__":
    sys.exit(main())
