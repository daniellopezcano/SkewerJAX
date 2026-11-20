#!/usr/bin/env python
"""Compute skewers through a TNG snapshot and store them (the skewer store: ``docs/skewer_store.md``).

The skewers of nested N x N lattices (x_i = i L / N on the two transverse axes, N a power of two) along one axis of the
box, by notebook 11's pipeline on the streaming route (one pass over the snapshot for the skewers not stored yet, then
tiles of ``--batch`` skewers, each tile written as one batch as soon as it is computed). The lattices of
``--n-per-axis`` run in increasing order; a skewer already stored (a coarser lattice, an earlier run) is never
recomputed, so rerunning the same command resumes an interrupted run.

    python scripts/production/extract_skewers.py --sim TNG50-2 --snap 25 --axis 2 --n-per-axis 2 4 8 16 32 64 128 \\
        --spacing-kms 1.0 --out /path/to/skewer_store
    python scripts/production/extract_skewers.py --config my_run.yaml --dry-run

Every option (the command line overrides the ``--config`` file, YAML or JSON, whose keys are the long option names with
``_``, e.g. ``n_per_axis: [2, 4]``, ``set: {voigt: ExoJAX}``): see ``--help``. The heavy logic is
``skewerjax.simulations.skewer_extraction`` and ``skewer_store``. Inputs: the snapshot (read only). Outputs: ``--out``
only (never inside the repository nor inside the simulation's folder): ``<out>/<sim>/snap_<NNN>/axis_<a>/<hash>/``.
"""

import argparse
import json
import os
import shutil
import signal
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SIMS = ("TNG50-2", "TNG50-3", "TNG50-4")


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument(
        "--config", help="a YAML or JSON file of these options (keys: the long names with _); the command line overrides it"
    )
    ap.add_argument(
        "--sim", help=f"the simulation's name, the dataset's folder ({', '.join(SIMS)}, or any other name with --sim-path)"
    )
    ap.add_argument("--sim-path", help="the folder of the simulation (with snapdir_<NNN>/); default <DATA_DIR>/<sim>")
    ap.add_argument("--snap", type=int, default=25, help="the snapshot number")
    ap.add_argument(
        "--axis",
        type=int,
        nargs="+",
        choices=(1, 2, 3),
        default=[2],
        help="the line of sight (1, 2, 3 = x, y, z); several run in turn",
    )
    ap.add_argument("--n-per-axis", type=int, nargs="+", help="the lattices N x N (powers of two), run in increasing order")
    ap.add_argument("--spacing-kms", type=float, default=1.0, help="the target pixel width [km/s] (N_pix = int(span / spacing))")
    ap.add_argument("--n-pix", type=int, help="the number of pixels over the period instead of --spacing-kms")
    ap.add_argument(
        "--lines", nargs="+", default=["HI-LyA", "SiIII-1206"], help="the line ids (core.lines; the first the reference)"
    )
    ap.add_argument(
        "--mode", choices=("default", "matched"), default="default", help="the preset of every option (skewer_extraction.MODES)"
    )
    ap.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override one option of the preset (e.g. voigt=ExoJAX, physics.x_HI_model=stored)",
    )
    ap.add_argument(
        "--precision",
        choices=("float32", "float64"),
        default="float32",
        help="the precision of the computation (tau is stored as float32)",
    )
    ap.add_argument(
        "--n-ref", type=int, default=4096, help="the finest lattice of the dataset (a power of two; part of its identity)"
    )
    ap.add_argument(
        "--device", choices=("cpu", "gpu"), default="cpu", help="where the tau sum runs (the pass and the pieces run on the host)"
    )
    ap.add_argument("--batch", type=int, default=1024, help="skewers per batch: a tile of the streaming route, one HDF5 file")
    ap.add_argument(
        "--skewers-per-pass", type=int, help="skewers per pass over the snapshot (default: the most that fit --memory-fraction)"
    )
    ap.add_argument("--memory-fraction", type=float, default=0.6, help="the fraction of the available memory one pass may use")
    ap.add_argument("--threads", type=int, help="threads of the pass over the snapshot (default: every core)")
    ap.add_argument("--out", help="the root of the store (outside the repository and the simulation's folder)")
    ap.add_argument(
        "--dry-run", action="store_true", help="print the dataset, what is stored and what would be computed; compute nothing"
    )
    return ap


def parse_args(argv=None) -> argparse.Namespace:
    ap = _parser()
    a, _ = ap.parse_known_args(argv)
    if a.config:
        text = Path(a.config).read_text()
        if a.config.endswith((".yaml", ".yml")):
            import yaml

            vals = yaml.safe_load(text) or {}
        else:
            vals = json.loads(text)
        vals = {k.replace("-", "_"): v for k, v in vals.items()}
        if isinstance(vals.get("set"), dict):
            vals["set"] = [f"{k}={json.dumps(v)}" for k, v in vals["set"].items()]
        for k in ("axis", "n_per_axis", "lines"):
            if k in vals and not isinstance(vals[k], list):
                vals[k] = [vals[k]]
        unknown = set(vals) - {x.dest for x in ap._actions}
        if unknown:
            ap.error(f"unknown keys in {a.config}: {sorted(unknown)}")
        ap.set_defaults(**vals)
    a = ap.parse_args(argv)
    for k in ("sim", "n_per_axis", "out"):
        if getattr(a, k) is None:
            ap.error(f"--{k.replace('_', '-')} is required (on the command line or in --config)")
    for n in [*a.n_per_axis, a.n_ref]:
        if n < 1 or n & (n - 1):
            ap.error(f"every N (--n-per-axis, --n-ref) must be a power of two (the lattices nest by doubling), got {n}")
    if max(a.n_per_axis) > a.n_ref:
        ap.error(f"--n-per-axis {max(a.n_per_axis)} exceeds --n-ref {a.n_ref}")
    if a.n_pix is not None and "--spacing-kms" not in (argv or sys.argv):
        a.spacing_kms = None  # --n-pix given: the grid by its number of pixels
    out = Path(a.out).expanduser().resolve()
    if out == REPO or REPO in out.parents:
        ap.error(f"--out {out} is inside the repository: choose another root.")
    a.out = out
    return a


def _log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def _interrupt(signum, frame):
    raise KeyboardInterrupt(f"signal {signal.Signals(signum).name}")


def main(argv=None) -> int:
    a = parse_args(argv)
    # a kill (SIGTERM) or a closed terminal (SIGHUP) stops like Ctrl-C: the run record says "interrupted"; the batches
    # written stay, the tile in progress is lost, and the same command resumes
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, _interrupt)
    # JAX's platforms before the first import of the package (exojax starts JAX at import: on a GPU it would preallocate
    # 75 % of its memory): the CPU as two devices (the engine overlaps two kernels), the GPU without preallocation
    if a.device == "gpu":
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    from skewerjax import config

    config.configure_devices(n_cpu_devices=2, platforms="cuda,cpu" if a.device == "gpu" else "cpu")
    config.set_precision(a.precision)
    from skewerjax.simulations import ArepoHDF5Reader
    from skewerjax.simulations import skewer_extraction as se
    from skewerjax.simulations import skewer_store as ss
    from skewerjax.utils.paths import DATA_DIR

    devices = config.devices(a.device)
    if not devices:
        _log(f"no {a.device} device visible to JAX")
        return 2
    sim_path = Path(a.sim_path) if a.sim_path else Path(DATA_DIR) / a.sim
    sim_dir = sim_path.expanduser().resolve()
    if a.out == sim_dir or sim_dir in a.out.parents:
        _log(f"--out {a.out} is inside the simulation's folder {sim_dir} (read only): choose another root.")
        return 2
    reader = ArepoHDF5Reader(sim_path, a.snap)
    H = reader.header
    overrides = dict(se.parse_override(x) for x in a.set)
    opts = se.dataset_options(
        H,
        mode=a.mode,
        overrides=overrides,
        spacing_kms=a.spacing_kms,
        n_pix=a.n_pix,
        line_ids=a.lines,
        precision=a.precision,
        n_ref=a.n_ref,
    )
    budget = a.memory_fraction * se.memory_available()
    n_lines, n_pix = len(a.lines), opts["pixels"]["n_pix"]
    _log(
        f"{a.sim} snapshot {a.snap} ({sim_path}): z = {H.redshift:.4f}, L = {H.box_cMpc:.3f} cMpc, {H.n_gas:,} gas cells; "
        f"{n_pix} pixels of {opts['pixels']['dv_kms']:.4f} km/s over {opts['pixels']['period_kms']:.2f} km/s; lines {a.lines}; "
        f"mode {a.mode}{' + ' + json.dumps(overrides) if overrides else ''}; {a.precision}; device {a.device} {[str(d) for d in devices]}"
    )
    status = 0
    for axis in a.axis:
        cfg = se.dataset_config(reader, a.sim, a.snap, axis, opts, preset=a.mode)
        p = se.plan(a.out, cfg, a.n_per_axis)
        new_bytes = p["new_total"] * n_lines * n_pix * 4
        free = shutil.disk_usage(next(x for x in [a.out, *a.out.parents] if x.exists())).free
        per = a.skewers_per_pass or se.pass_size(max(p["new_total"], 1), H.n_gas, n_lines, n_pix, a.precision, budget)
        peak = se.pass_memory(min(per, max(p["new_total"], 1)), H.n_gas, n_lines, n_pix, a.precision)
        _log(f"axis {axis}: dataset {p['path']} ({'exists' if p['exists'] else 'new'})")
        for r in p["rows"]:
            _log(f"  N = {r['n']:5d}: {r['skewers']:10,d} skewers, {r['stored']:10,d} stored, {r['new']:10,d} to compute")
        _log(
            f"  to compute: {p['new_total']:,} skewers, {new_bytes / 1e9:.2f} GB on disk ({free / 1e9:.0f} GB free); batches of "
            f"{a.batch} skewers; passes of <= {per:,} skewers (estimated peak {peak / 1e9:.1f} GB; budget {budget / 1e9:.0f} GB = "
            f"{a.memory_fraction:g} x {se.memory_available() / 1e9:.0f} GB available)"
        )
        if new_bytes > 0.95 * free:
            _log(f"  STOP: the run needs {new_bytes / 1e9:.1f} GB, more than the disk's {free / 1e9:.1f} GB free.")
            status = 3
            continue
        if peak > se.memory_available():
            _log(f"  STOP: one pass of {per} skewers is estimated at {peak / 1e9:.1f} GB, more than the memory available.")
            status = 3
            continue
        if a.dry_run:
            _log("  config.json: " + json.dumps(dict(cfg, hash=p["hash"]), indent=1).replace("\n", "\n  "))
            continue
        ds = ss.SkewerDataset.create(a.out, cfg)
        se.extract(
            ds,
            reader,
            a.n_per_axis,
            tile_size=a.batch,
            devices=devices,
            device_name=a.device,
            threads=a.threads,
            skewers_per_pass=a.skewers_per_pass,
            memory_budget=budget,
            log=_log,
            argv=sys.argv if argv is None else ["extract_skewers.py", *argv],
        )
    return status


if __name__ == "__main__":
    sys.exit(main())
