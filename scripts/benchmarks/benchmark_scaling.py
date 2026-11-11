#!/usr/bin/env python
"""The scaling of SkewerJAX against fake_spectra (notebook 14): blocks A (simulation resolution), B (number of skewers,
16 to 65,536) and C1 (pixel width, merged into the TNG50-2 128^2 point), every point with the four entries of notebook 13
(fake_spectra; SkewerJAX default CPU, matched CPU, default GPU), cold, one record per point; and C2 (``--c2``), the
precision of the midpoint option's sampling against the default. The logic and the protocol are
``skewerjax.comparison.scaling``.

    python scripts/benchmarks/benchmark_scaling.py [--nmax N] [--blocks A,B,C1] [--only ID ...] [--out DIR] [--list]
    python scripts/benchmarks/benchmark_scaling.py --c2 [--c2-n 8] [--out DIR]

Every point runs in its own process; a point with a record in ``--out`` is skipped, so an interrupted run resumes
(delete a record to measure it again). The cold protocol drops the snapshot from the page cache before each code reads
it (``posix_fadvise``: nothing is written). Inputs: ``skewerjax.utils.paths.DATA_DIR`` (``$SKEWERJAX_DATA``) /
TNG50-4, -3, -2, snapshot 25, read only. Outputs: ``--out`` (default ``benchmarks/results/scaling``): ``<id>.json`` per
point (the stage times, the JIT time, the cache state, the peak memory, the machine, the agreement over all skewers and
the flux of the reference skewer from every entry), ``logs/<id>.log``, and ``C2.json`` (float64) with ``--c2``.
"""

import argparse
import json
import os
import subprocess
import sys

# The launcher computes nothing, but importing the package starts JAX (exojax runs JAX code at import), and JAX on a GPU
# preallocates 75% of its memory: held for the whole benchmark, that leaves the workers' GPU entry ~1.4 GB of a 12 GB
# card (out of memory at 256^2). The launcher stays on the CPU; every worker sets its own platforms (``worker_env``).
os.environ["JAX_PLATFORMS"] = "cpu"

from skewerjax.comparison import scaling  # noqa: E402
from skewerjax.utils.paths import BENCHMARK_RECORDS_DIR, DATA_DIR  # noqa: E402

_C2_WORKER = """
import json, sys
from skewerjax import config
config.configure_devices(n_cpu_devices=2, platforms="cpu")
config.set_precision("float64")
from skewerjax.comparison import scaling
rec = scaling.run_c2(sys.argv[1], n=int(sys.argv[2]))
print(scaling.write_record(dict(rec, id="C2"), sys.argv[3]))
"""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--nmax", type=int, default=scaling.N_MAX, help="skewers per side of the grid of blocks A and C1 (N_max)")
    ap.add_argument("--blocks", default="A,B,C1", help="comma-separated: A, B, C1")
    ap.add_argument("--only", nargs="*", default=None, help="only these point ids")
    ap.add_argument("--out", default=str(BENCHMARK_RECORDS_DIR / "scaling"))
    ap.add_argument("--list", action="store_true", help="list the points and exit")
    ap.add_argument("--c2", action="store_true", help="run block C2 (the midpoint option's sampling, float64) instead")
    ap.add_argument("--c2-n", type=int, default=8, help="skewers per side of block C2")
    a = ap.parse_args(argv)
    if a.c2:
        cmd = [sys.executable, "-c", _C2_WORKER, str(DATA_DIR), str(a.c2_n), a.out]
        return subprocess.run(cmd, env=scaling.worker_env()).returncode
    points = scaling.scaling_points(a.nmax, tuple(a.blocks.split(",")))
    if a.only:
        points = [c for c in points if c["id"] in a.only]
    if a.list:
        for c in points:
            print(
                f"{c['id']:32s} blocks {','.join(c['blocks']):10s} {c['n_skewers']:6d} skewers, entries {len(c['entries'])}, "
                f"grids {json.dumps(c['dv_list'])}, repeats {c['repeats']}"
            )
        return 0
    scaling.run_points(points, a.out, data_dir=DATA_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
