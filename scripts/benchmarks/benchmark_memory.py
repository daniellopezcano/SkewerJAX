#!/usr/bin/env python
"""The memory of SkewerJAX and fake_spectra (notebook 16): how much each code and mode uses, where (host RAM: anonymous and
file-backed; GPU VRAM), in which stage and how it scales. Blocks A (simulation), B (number of skewers, 16 to 65,536), C1
(pixel width) and D (SkewerJAX's memory knobs: tile size, physics configurations, pass threads), each (point, entry) in
its own fresh process, cold. The logic and the protocol are ``skewerjax.comparison.memory``.

    python scripts/benchmarks/benchmark_memory.py [--blocks A,B,C1,D] [--only ID ...] [--out DIR] [--list]

A (point, entry) with a record in ``--out`` is skipped, so an interrupted run resumes (delete a record to measure it
again). Inputs: ``skewerjax.utils.paths.DATA_DIR`` (``$SKEWERJAX_DATA``) / TNG50-4, -3, -2, snapshot 25, read only.
Outputs: ``--out`` (default ``benchmarks/results/memory``): ``<id>.json`` per (point, entry) (the sampled timeline of
anonymous, file-backed and GPU memory, the CPU time and the bytes read, the stage markers, the peaks per stage, the
stage times, the pass's and tiles' facts), and ``logs/<id>.log``.
"""

import argparse
import os
import sys

# The launcher computes nothing, but importing the package starts JAX: it stays on the CPU and never holds the GPU (the
# GPU entry's worker sets its own platforms, ``memory.worker_env``).
os.environ["JAX_PLATFORMS"] = "cpu"

from skewerjax.comparison import memory  # noqa: E402
from skewerjax.utils.paths import BENCHMARK_RECORDS_DIR, DATA_DIR  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--blocks", default="A,B,C1,D", help="comma-separated: A, B, C1, D")
    ap.add_argument("--only", nargs="*", default=None, help="only these (point, entry) ids")
    ap.add_argument("--out", default=str(BENCHMARK_RECORDS_DIR / "memory"))
    ap.add_argument("--list", action="store_true", help="list the (point, entry) ids and exit")
    a = ap.parse_args(argv)
    points = memory.memory_points(tuple(a.blocks.split(",")))
    if a.only:
        points = [c for c in points if c["id"] in a.only]
    if a.list:
        for c in points:
            print(f"{c['id']:40s} blocks {','.join(c['blocks']):10s} {c['n_skewers']:6d} skewers, {c['entry']}")
        return 0
    memory.run_points(points, a.out, data_dir=DATA_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
