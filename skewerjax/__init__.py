"""SkewerJAX: absorption along skewers (tau, F = exp(-tau)) for many transitions, and their summary statistics, in JAX.

Three blocks, with dependencies in one direction only (comparison -> simulations -> core):

* ``skewerjax.core`` (A): lines, cosmology, skewers, the gas cells (``Cells``), the deposition, line profiles, the tau integral, the overflux
  and the summary statistics; self-contained.
* ``skewerjax.simulations`` (B): reading hydrodynamical (or N-body) snapshots into the inputs of the core.
* ``skewerjax.comparison`` (C): fake_spectra's scheme and results, compared with SkewerJAX's.

``skewerjax.utils.plotting`` holds the shared figure style.

Debugging: ``core.check_skewers`` (independent checks of the pieces and of tau for a set of skewers), the
``return_info=True`` option of the entry points (``select_cells``, ``simulations.select_cells_near_paths``,
``cell_pieces``, ``compute_optical_depth``) and the package's log messages (``logging.getLogger("skewerjax")``,
silent by default; ``core.enable_logging()`` shows them).

Precision: SkewerJAX never changes JAX's configuration. Arrays follow ``jax_enable_x64`` as set by the user before
the first computation (``jax.config.update("jax_enable_x64", True)``, as the notebooks and scripts do; float32
otherwise). ``simulations.PowerLawGRF`` warns when x64 is off; ``voigt_convolution`` works in float64 when x64 is on; the
simulation and comparison blocks hold host arrays in float64 NumPy.
"""
__version__ = "0.0.1"

import logging as _logging

_logging.getLogger("skewerjax").addHandler(_logging.NullHandler())     # silent unless configured (core.enable_logging)

from skewerjax import core, simulations, comparison, utils

__all__ = ["__version__", "core", "simulations", "comparison", "utils"]
