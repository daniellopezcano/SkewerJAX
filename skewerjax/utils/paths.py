"""Where SkewerJAX looks for data that is not part of the package: one place, each path overridable by an environment
variable.

* ``DATA_DIR``: the folder that holds the simulation snapshots (``TNG50-4/``, and ``TNG50-3/``, ``TNG50-2/`` for the
  benchmarks), each with its ``snapdir_XXX``; ``$SKEWERJAX_DATA``, default ``/home/STORAGE``. Read only.
* ``BENCHMARK_RAW_DIR``: the per-configuration records of ``scripts/benchmarks/benchmark_fake_spectra.py`` (in
  ``fake_spectra/``; rebuilt by that script, git-ignored); ``$SKEWERJAX_BENCHMARK_RAW``, default ``benchmarks/raw`` in the
  repository.
* ``CACHE_DIR``: results that are slow to recompute (e.g. the toy-box statistics of notebook 08); rebuilt by the code
  that writes them, git-ignored; ``$SKEWERJAX_CACHE``, default ``cache`` in the repository.
* ``BENCHMARK_RECORDS_DIR``: the benchmark summary read by notebook 10 (``fake_spectra_benchmark.json``: every record,
  the machine, the snapshots and the launches; written by the same script, committed); ``$SKEWERJAX_BENCHMARK_RESULTS``,
  default ``benchmarks/results`` in the repository.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("SKEWERJAX_DATA", "/home/STORAGE"))
BENCHMARK_RAW_DIR = Path(os.environ.get("SKEWERJAX_BENCHMARK_RAW", REPO_ROOT / "benchmarks" / "raw"))
CACHE_DIR = Path(os.environ.get("SKEWERJAX_CACHE", REPO_ROOT / "cache"))
BENCHMARK_RECORDS_DIR = Path(os.environ.get("SKEWERJAX_BENCHMARK_RESULTS", REPO_ROOT / "benchmarks" / "results"))


def snapshot_dir(simulation: str = "TNG50-4") -> Path:
    """The folder of one simulation's snapshots (``DATA_DIR / simulation``)."""
    return DATA_DIR / simulation
