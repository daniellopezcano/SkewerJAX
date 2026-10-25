#!/usr/bin/env python
"""SkewerJAX against fake_spectra: wall time per stage, JIT compilation, peak memory and flux agreement on identical
skewers (Lya and Si III 1206), one process per configuration. The logic is ``skewerjax.comparison.benchmark``.

    python scripts/benchmarks/benchmark_vs_fake_spectra.py --sets quick [--results DIR] [--cap 1800] [--list]

Snapshots are read from ``$SKEWERJAX_DATA`` (default ``/home/STORAGE``). Sets: ``quick`` (TNG50-4: skewer count 1-64 on CPU and GPU, float32, thread counts), ``skewers`` (TNG50-4, 256 skewers),
``resolution`` (16 skewers in TNG50-3 and TNG50-2). Results that exist are skipped, so an interrupted run resumes.

Raw results (one ``.json`` record and one ``.npz`` of spectra per configuration, ~15 MB for all sets) go to ``RESULTS``:
``$SKEWERJAX_BENCHMARK_RAW``, by default ``claude_agents/assets/benchmarks`` (outside git). The ``.json`` records that
notebook 09 reads are copied to ``notebooks/benchmark_records/suite``; rerun this script to rebuild the raw files.
"""
import argparse
import importlib.util
import os
import sys
from pathlib import Path

# the module alone (NumPy only), not the skewerjax package: this launcher process does not start JAX
_f = Path(__file__).resolve().parents[2] / "skewerjax/comparison/benchmark.py"
_spec = importlib.util.spec_from_file_location("skewerjax_benchmark", _f)
B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B)
make_config, run_suite, config_id = B.make_config, B.run_suite, B.config_id

STORAGE, SNAP = Path(os.environ.get("SKEWERJAX_DATA", "/home/STORAGE")), 25      # folder with TNG50-4/, TNG50-3/, TNG50-2/
RESULTS = Path(os.environ.get("SKEWERJAX_BENCHMARK_RAW", Path(__file__).resolve().parents[2] / "claude_agents/assets/benchmarks"))


def configs(name: str) -> list:
    sim = lambda s: STORAGE / s
    both = lambda s, n, **kw: [make_config("fake_spectra", sim(s), SNAP, n, **kw), make_config("skewerjax", sim(s), SNAP, n, **kw),
                               make_config("skewerjax", sim(s), SNAP, n, platform="gpu", **kw)]
    if name == "quick":
        out = [c for n in (1, 4, 16, 64) for c in both("TNG50-4", n)]
        out += [make_config("skewerjax", sim("TNG50-4"), SNAP, n, dtype="float32", platform=p) for n in (16, 64) for p in ("cpu", "gpu")]
        out += [make_config(code, sim("TNG50-4"), SNAP, 16, threads=t) for t in (1, 4, 8, 16) for code in ("fake_spectra", "skewerjax")]
        return out
    if name == "skewers":
        return both("TNG50-4", 256, repeats=2)
    if name == "resolution":
        return both("TNG50-3", 16) + both("TNG50-2", 16, repeats=2)
    raise ValueError(name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sets", default="quick", help="comma-separated: quick, skewers, resolution")
    ap.add_argument("--results", default=str(RESULTS))
    ap.add_argument("--cap", type=float, default=1800.0, help="wall-time cap per configuration [s]")
    ap.add_argument("--list", action="store_true", help="print the configurations and exit")
    a = ap.parse_args()
    cs = [c for s in a.sets.split(",") for c in configs(s)]
    if a.list:
        print("\n".join(config_id(c) for c in cs))
        sys.exit(0)
    run_suite(cs, a.results, cap_s=a.cap, log=lambda m: print(m, flush=True))
