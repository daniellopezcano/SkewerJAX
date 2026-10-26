"""Where SkewerJAX looks for data that is not part of the repository: one place, each path overridable by an
environment variable.

* ``DATA_DIR``: the folder that holds the simulation snapshots (``TNG50-4/``, and ``TNG50-3/``, ``TNG50-2/`` for the
  benchmarks), each with its ``snapdir_XXX``; ``$SKEWERJAX_DATA``, default ``/home/STORAGE``. Read only.
* ``BENCHMARK_RAW_DIR``: the per-configuration records of ``scripts/benchmarks/benchmark_fake_spectra.py`` (in
  ``fake_spectra/``; rebuilt by that script, git-ignored); ``$SKEWERJAX_BENCHMARK_RAW``, default
  ``benchmark_results/raw`` in the repository.
* ``BENCHMARK_RECORDS_DIR``: the benchmark results file read by notebook 10 (``fake_spectra_benchmark.json``, written by
  the same script, git-ignored); ``$SKEWERJAX_BENCHMARK_RESULTS``, default ``notebooks/benchmark_records`` in the
  repository.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("SKEWERJAX_DATA", "/home/STORAGE"))
BENCHMARK_RAW_DIR = Path(os.environ.get("SKEWERJAX_BENCHMARK_RAW", REPO_ROOT / "benchmark_results" / "raw"))
BENCHMARK_RECORDS_DIR = Path(os.environ.get("SKEWERJAX_BENCHMARK_RESULTS", REPO_ROOT / "notebooks" / "benchmark_records"))


def snapshot_dir(simulation: str = "TNG50-4") -> Path:
    """The folder of one simulation's snapshots (``DATA_DIR / simulation``)."""
    return DATA_DIR / simulation
