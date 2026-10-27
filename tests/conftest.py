"""Shared setup of the test suite (``tests/README.md``).

* JAX on the CPU (``JAX_PLATFORMS=cpu`` unless set: ``JAX_PLATFORMS=cuda,cpu`` for the ``gpu`` tests) and float64
  (``jax_enable_x64``, enabled *before* SkewerJAX is imported, as every validated tolerance is a float64 one);
* the markers of ``pyproject.toml`` (``slow``, ``tng``, ``gpu``, ``fake_spectra``): the tests that need what this machine
  lacks are skipped with the reason (``MARKER_REQUIREMENTS``); ``slow`` tests are deselected by the default
  ``-m "not slow"`` of ``pyproject.toml``.
"""

import importlib.util
import logging
import os
from functools import cache

os.environ.setdefault("JAX_PLATFORMS", "cpu")
logging.getLogger("jax._src.xla_bridge").setLevel(logging.CRITICAL)
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import pytest  # noqa: E402

from skewerjax.utils.paths import snapshot_dir  # noqa: E402

TNG_SNAPSHOTS = (25, 33)  # the TNG50-4 snapshots the tests read ($SKEWERJAX_DATA/TNG50-4)


@cache
def tng_available() -> bool:
    return all((snapshot_dir("TNG50-4") / f"snapdir_{s:03d}").is_dir() for s in TNG_SNAPSHOTS)


@cache
def gpu_available() -> bool:
    try:
        return any(d.platform == "gpu" for d in jax.devices())
    except RuntimeError:
        return False


@cache
def fake_spectra_available() -> bool:
    return importlib.util.find_spec("fake_spectra") is not None


MARKER_REQUIREMENTS = {
    "tng": (tng_available, f"needs the TNG50-4 snapshots {TNG_SNAPSHOTS} in {snapshot_dir('TNG50-4')} ($SKEWERJAX_DATA)"),
    "gpu": (gpu_available, "needs a GPU visible to JAX (run with JAX_PLATFORMS=cuda,cpu)"),
    "fake_spectra": (fake_spectra_available, "needs fake_spectra importable (pip install -e path/to/fake_spectra)"),
}


def skip_reason(marker_names) -> str:
    """Why a test with these markers cannot run here ('' if it can): the first missing requirement."""
    for name in marker_names:
        if name in MARKER_REQUIREMENTS and not MARKER_REQUIREMENTS[name][0]():
            return MARKER_REQUIREMENTS[name][1]
    return ""


def pytest_collection_modifyitems(config, items):
    for item in items:
        reason = skip_reason([m.name for m in item.iter_markers()])
        if reason:
            item.add_marker(pytest.mark.skip(reason=reason))
