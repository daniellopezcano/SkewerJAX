"""SkewerJAX: absorption along skewers (tau, F = exp(-tau)) for many transitions, and their summary statistics, in JAX.

The core and two realms on it, with dependencies in one direction only (each realm imports ``core`` at most and
neither imports the other; ``tests/test_import_graph.py`` checks it):

* ``skewerjax.core`` (A): lines, cosmology, the paths (``SkewerBundle``, lattices of skewers through a box:
  ``lattice_columns``, ``lattice_points``, ``hubble_velocity``), the gas cells (``Cells``), the deposition, line
  profiles and the tau integral; self-contained.
* ``skewerjax.simulations`` (B): the gas: toy fields (``PowerLawGRF``, ``ToyBoxes``) and hydrodynamical (or N-body)
  snapshots, turned into the inputs of the core.
* ``skewerjax.summary_stats``: the statistics, from arrays and geometry only: the overflux and its 1D and 3D statistics
  (P1D, xi1D, P3D, xi3D, the (par, perp) plane and the wedges of mu), term by term and order by order; P and xi of 3D
  fields and of the density along skewers, against an expectation P(k).
* ``skewerjax.comparison`` (C): the benchmark against fake_spectra (uses the blocks above).

Notebooks compose the realms. The subpackages are imported lazily (PEP 562): ``import skewerjax.summary_stats`` loads
``core`` at most; ``skewerjax.simulations`` and ``from skewerjax import simulations`` load that subpackage on access.

``skewerjax.utils.plotting`` holds the shared figure style.

Debugging: ``core.check_skewers`` (independent checks of the pieces and of tau for a set of skewers), the
``return_info=True`` option of the entry points (``select_cells``, ``simulations.select_cells_near_paths``,
``cell_pieces``, ``compute_optical_depth``) and the package's log messages (``logging.getLogger("skewerjax")``,
silent by default; ``core.enable_logging()`` shows them).

Precision and devices: ``skewerjax.config`` (docs/precision.md). float32 by default, ``config.set_precision("float64")``
for validation (it turns ``jax_enable_x64`` on); the device kernels follow the policy, the host geometry, the cosmology
a(s), the Hubble velocities of the line centres and the per-cell gas physics are float64 whatever it is.
``config.configure_devices`` (before JAX starts) and ``config.devices("cpu" | "gpu")`` give the ``devices=`` of
``core.compute_optical_depth``.
"""

import importlib as _importlib
import logging as _logging

_SUBPACKAGES = ("core", "summary_stats", "simulations", "comparison", "utils", "config")

__version__ = "0.0.1"
_logging.getLogger("skewerjax").addHandler(_logging.NullHandler())  # silent unless configured (core.enable_logging)

__all__ = ["__version__", *_SUBPACKAGES]


def __getattr__(name: str):
    """Import a subpackage on first access (PEP 562): ``skewerjax.core``, ``skewerjax.simulations``, ..."""
    if name in _SUBPACKAGES:
        module = _importlib.import_module(f"skewerjax.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module 'skewerjax' has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_SUBPACKAGES))
