#!/usr/bin/env python
"""SkewerJAX against fake_spectra: wall time per stage, JIT compilation, host and GPU memory, and flux agreement on
identical skewers (Lya and Si III 1206), scaling with the number of skewers, the device and the simulation size. The
logic and the protocol are ``skewerjax.comparison.benchmark``.

    python scripts/benchmarks/benchmark_fake_spectra.py [--preset quick|full] [--sets scaling,threads] [--raw DIR]
                                                        [--results FILE] [--cap 3600] [--budget S] [--mem-cap 48]
                                                        [--mem-floor 16] [--list]

Presets (``benchmark.PRESETS``):

* ``quick`` (the default): a reduced study in <= 30 min of new runs on this machine. TNG50-3 at N = 1, 4, 16 and
  TNG50-2 at N = 1 skewers (fake_spectra, SkewerJAX CPU and GPU, one or two timed runs), and 1 / 4 / 8 / 16 threads
  on TNG50-4 at N = 1; the TNG50-4 series of the full study is kept as it is. Each point has its own cap and
  estimate (``QUICK_SCALING``, ``QUICK_THREADS``), and no point starts once the time spent plus its estimate would
  exceed ``--budget`` (1800 s); a point left out that way is measured by the next launch.
* ``full``: the whole study, for later: N = 1 ... 1024 in TNG50-4, -3, -2 (3 timed runs, 2 in TNG50-2), and 1-16
  threads at N = 16; caps ``--cap`` 1 h, ``--mem-cap`` 48 GB; ~10 h or more. It re-measures the quick preset's points
  that hold fewer timed runs.

Sets (``benchmark_sets``): ``scaling``, ``threads``, ``smoke`` (TNG50-4, 1 and 4 skewers, one timed run). Every
configuration runs in its own process; results that exist are kept, so an interrupted run resumes (delete a record to
measure it again). Limits: the cap per configuration, ``--mem-cap`` GB of worker RSS, ``--mem-floor`` GB of memory left
on the machine; larger skewer counts of a series that hit a limit (or are predicted to) are skipped and recorded as
such. Every record keeps its preset (``config["preset"]``); notebook 10 says which point comes from which.

Inputs: ``skewerjax.utils.paths.DATA_DIR`` (``$SKEWERJAX_DATA``) / TNG50-4, -3, -2, snapshot 25 (z = 3), read only.
Outputs: one record per configuration (``.json``, ``.npz`` of tau, a log) in ``--raw`` (default
``BENCHMARK_RAW_DIR / "fake_spectra"`` = ``benchmarks/raw/fake_spectra``, git-ignored), and the summary of all records,
``--results`` (default ``BENCHMARK_RECORDS_DIR / "fake_spectra_benchmark.json"`` =
``benchmarks/results/fake_spectra_benchmark.json``, committed), which notebook ``10_comparison_fake_spectra_performance``
reads.
"""

import argparse
import importlib.util
import sys
import time
from pathlib import Path


def _module(rel: str, name: str):
    """One module of the package alone (NumPy only), not the skewerjax package: this launcher does not start JAX."""
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[2] / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


B = _module("skewerjax/comparison/benchmark.py", "skewerjax_benchmark")
PATHS = _module("skewerjax/utils/paths.py", "skewerjax_paths")
RAW = PATHS.BENCHMARK_RAW_DIR / "fake_spectra"
RESULTS = PATHS.BENCHMARK_RECORDS_DIR / "fake_spectra_benchmark.json"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--preset", default="quick", choices=sorted(B.PRESETS), help="quick (<= 30 min) or full (the whole study)")
    ap.add_argument("--sets", default="scaling,threads", help="comma-separated: scaling, threads, smoke")
    ap.add_argument("--raw", default=str(RAW), help="folder of the per-configuration records")
    ap.add_argument("--results", default=str(RESULTS), help="the results file the notebook reads")
    ap.add_argument("--data", default=str(PATHS.DATA_DIR), help="folder with TNG50-4/, TNG50-3/, TNG50-2/")
    ap.add_argument("--cap", type=float, default=3600.0, help="wall-time cap per configuration without its own [s]")
    ap.add_argument(
        "--budget",
        type=float,
        default=None,
        help="no configuration starts beyond this total [s] (default: 1800 for quick, none for full)",
    )
    ap.add_argument("--mem-cap", type=float, default=48.0, help="worker RSS cap [GB]")
    ap.add_argument("--mem-floor", type=float, default=16.0, help="stop a worker when the machine has less available [GB]")
    ap.add_argument("--list", action="store_true", help="print the configurations and exit")
    a = ap.parse_args(argv)
    sets = a.sets.split(",")
    cs = [c for s in sets for c in B.benchmark_sets(s, a.data, preset=a.preset)]
    budget = a.budget if a.budget is not None else (B.QUICK_BUDGET_S if a.preset == "quick" else None)
    if a.list:
        print(
            "\n".join(
                f"{B.config_id(c)}  (cap {c.get('cap_s') or a.cap:.0f} s"
                + (f", estimate {c['est_s']:.0f} s)" if c.get("est_s") else ")")
                for c in cs
            )
        )
        return
    t0 = time.perf_counter()
    print(f"preset {a.preset}, sets {a.sets}, budget {budget or 'none'} s, {time.strftime('%Y-%m-%d %H:%M')}", flush=True)
    B.run_suite(
        cs,
        a.raw,
        cap_s=a.cap,
        mem_cap_GB=a.mem_cap,
        mem_floor_GB=a.mem_floor,
        total_budget_s=budget,
        log=lambda m: print(m, flush=True),
    )
    wall = time.perf_counter() - t0
    res = B.collect(
        a.raw,
        a.results,
        sets=sets,
        launcher_wall_s=wall,
        launches=[dict(preset=a.preset, sets=sets, wall_s=wall, date=time.strftime("%Y-%m-%d %H:%M"))],
    )
    print(
        f"{len(res['records'])} records -> {a.results}; this launch {time.perf_counter() - t0:.0f} s, "
        f"all configurations {res['runtime_s'] / 3600:.2f} h",
        flush=True,
    )


if __name__ == "__main__":
    sys.exit(main())
