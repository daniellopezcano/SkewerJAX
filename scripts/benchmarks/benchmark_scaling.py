#!/usr/bin/env python
"""The scaling of SkewerJAX against fake_spectra (notebook 14): blocks A (simulation resolution), B (number of skewers),
C1 (pixel width), C2 (sampling along the skewer), every point with the four entries of notebook 13 (fake_spectra; SkewerJAX
default CPU, matched CPU, default GPU) and one record per point. The logic and the protocol are
``skewerjax.comparison.scaling``.

    python scripts/benchmarks/benchmark_scaling.py [--nmax N] [--blocks A,B,C1,C2] [--only ID ...] [--out DIR] [--list]

Every point runs in its own process; a point with a record in ``--out`` is skipped, so an interrupted run resumes
(delete a record to measure it again). Inputs: ``skewerjax.utils.paths.DATA_DIR`` (``$SKEWERJAX_DATA``) / TNG50-4, -3,
-2, snapshot 25, read only. Outputs: ``--out`` (default ``benchmarks/results/scaling``): ``<id>.json`` per point (the
stage times, the JIT time, the cache state, the machine, the agreement over all skewers and the flux of the reference
skewer from every entry) and ``logs/<id>.log``.
"""

import argparse
import sys

from skewerjax.comparison import scaling
from skewerjax.utils.paths import BENCHMARK_RECORDS_DIR, DATA_DIR


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--nmax", type=int, default=scaling.N_MAX, help="skewers per side of the largest grid (N_max)")
    ap.add_argument("--blocks", default="A,B,C1,C2", help="comma-separated: A, B, C1, C2")
    ap.add_argument("--only", nargs="*", default=None, help="only these point ids")
    ap.add_argument("--out", default=str(BENCHMARK_RECORDS_DIR / "scaling"))
    ap.add_argument("--list", action="store_true", help="list the points and exit")
    a = ap.parse_args(argv)
    points = scaling.scaling_points(a.nmax, tuple(a.blocks.split(",")))
    if a.only:
        points = [c for c in points if c["id"] in a.only]
    if a.list:
        for c in points:
            print(f"{c['id']:32s} blocks {','.join(c['blocks']):10s} {c['n_skewers']:6d} skewers, entries {len(c['entries'])}")
        return 0
    scaling.run_points(points, a.out, data_dir=DATA_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
