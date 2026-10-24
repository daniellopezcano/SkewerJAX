#!/usr/bin/env python
"""A strict single-CPU comparison: SkewerJAX's integration modes on one core, on the inputs of the fake_spectra
benchmark (``benchmark_vs_fake_spectra.py``: TNG50-4 snapshot 25, the first 16 skewers of ``benchmark_skewers``,
Lya and Si III 1206 on 1 km/s pixels).

    python scripts/benchmarks/single_cpu_modes.py --results DIR [--snap-base PATH] [--n 16] [--repeats 2]

Every mode runs in its own process pinned to CPU 0, with one OpenMP thread and XLA's single-threaded CPU backend
(``--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1``), the setting of fake_spectra's one-thread run
(``fake_spectra_TNG50-4_s25_n16_t1.json`` of the suite). Modes:

* ``default``: SkewerJAX's default (Voronoi pieces of 0.25 km/s dual segments, exact frame, float64);
* ``crossings``: the Voronoi-adapted mode (``deposition="voronoi_crossings"``: one absorber per sub-piece of each
  cell crossing, W <= 0.25 b);
* ``matched``: fake_spectra's scheme reproduced in SkewerJAX (``comparison.matched_tau`` with the fake_spectra
  options: top hats, 7-point chords, snapshot frame, linear detuning, tautail), the same algorithm as fake_spectra;
* ``float32``: the default with the opt-in float32 kernel.

Stages (exclusive wall time): io, selection, physics (ion properties + deposition + absorbers), tau. Results
(``single_cpu_<mode>.json``) that exist are skipped, so an interrupted run resumes.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

MODES = ("default", "crossings", "matched", "float32")
ROOT = Path(__file__).resolve().parents[2]


def worker(mode: str, snap_base: str, n: int, repeats: int, out: Path):
    import importlib.util
    import numpy as np
    import jax
    jax.config.update("jax_enable_x64", True)
    spec = importlib.util.spec_from_file_location("skewerjax_benchmark", ROOT / "skewerjax/comparison/benchmark.py")
    B = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(B)
    from skewerjax.core import LineBundle
    from skewerjax.simulations import HI, ArepoHDF5Reader, ExtractionOptions, Ion, SimulationSkewers
    from skewerjax.simulations.absorbers import tau_absorbers
    from skewerjax.comparison.effects import FAKE_SPECTRA_EXTRACTION, FAKE_SPECTRA_INTEGRATION
    from skewerjax.comparison.emulation import matched_tau
    ions = {"HI-LyA": HI, "SiIII-1206": Ion("Si", 3)}
    lbs = {k: LineBundle([k], ref_line_id=k) for k in ions}
    cofm, axis = B.benchmark_skewers(n)
    reps = []
    for r in range(repeats):
        T = B.StageTimer()
        t0 = time.perf_counter()
        with T.stage("setup"):
            rd = ArepoHDF5Reader(snap_base, 25)
        T.wrap(rd, "read_rows", "io")
        with T.stage("selection"):
            sk = SimulationSkewers.axis_aligned(rd, cofm, axis, 0.25)
        n_abs = {}
        for name, ion in ions.items():
            if mode == "matched":
                with T.stage("physics"):
                    sk.properties(ion, FAKE_SPECTRA_EXTRACTION)
                    v, dv = sk.spectral_grid(1.0)
                with T.stage("tau"):
                    tau = matched_tau(sk, ion, lbs[name], v, FAKE_SPECTRA_EXTRACTION, FAKE_SPECTRA_INTEGRATION)
                continue
            ext = ExtractionOptions(deposition="voronoi_crossings") if mode == "crossings" else ExtractionOptions()
            with T.stage("physics"):
                fr = sk.frame("exact")
                v, dv = sk.spectral_grid(1.0, fr)
                A = sk.absorbers(ion, ext, frame=fr)
                n_abs[name] = int((A.n > 0).sum())
            with T.stage("tau"):
                tau = tau_absorbers(A, lbs[name], v, fr, velocity_period=True, check=None,      # periodic, as fake_spectra
                                    dtype="float32" if mode == "float32" else None)[0]
        reps.append(dict(stages=dict(T.t), total=time.perf_counter() - t0, n_cells=len(sk.cells), n_v=int(v.size), n_absorbers=n_abs,
                         meanF={name: None for name in ions}))
    rec = dict(id=f"single_cpu_{mode}", mode=mode, n_skewers=n, snap_base=snap_base, repeats=reps, peak_rss_GB=B.peak_rss_GB(),
               metadata=B.machine_metadata(), cpu_affinity=sorted(os.sched_getaffinity(0)))
    (out / f"single_cpu_{mode}.json").write_text(json.dumps(rec, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True)
    ap.add_argument("--snap-base", default=os.path.join(os.environ.get("SKEWERJAX_DATA", "/home/STORAGE"), "TNG50-4"))
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--worker", default=None)
    a = ap.parse_args()
    out = Path(a.results)
    out.mkdir(parents=True, exist_ok=True)
    if a.worker:
        return worker(a.worker, a.snap_base, a.n, a.repeats, out)
    env = dict(os.environ, OMP_NUM_THREADS="1", JAX_PLATFORMS="cpu", XLA_PYTHON_CLIENT_PREALLOCATE="false",
               XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
               PYTHONPATH=os.pathsep.join([str(ROOT)] + [x for x in [os.environ.get("PYTHONPATH")] if x]))
    for mode in a.modes.split(","):
        if (out / f"single_cpu_{mode}.json").exists():
            print(f"{mode}: exists, skipped", flush=True)
            continue
        t0 = time.perf_counter()
        p = subprocess.run([sys.executable, __file__, "--results", str(out), "--snap-base", a.snap_base, "--n", str(a.n),
                            "--repeats", str(a.repeats), "--worker", mode], env=env, preexec_fn=lambda: os.sched_setaffinity(0, {0}),
                           capture_output=True, text=True)
        print(f"{mode}: exit {p.returncode} in {time.perf_counter() - t0:.0f} s", flush=True)
        if p.returncode:
            print(p.stderr[-3000:], flush=True)


if __name__ == "__main__":
    main()
