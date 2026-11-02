"""The paths: skewers through a periodic box, their geometry and their frame (imports ``skewerjax.core`` only).

* ``lattice``: a regular lattice of straight skewers along y through the centres of the cell columns of an N^3 grid:
  the columns (``lattice_columns``) and the points of the paths (``lattice_points``), to build a ``core.SkewerBundle``
  with any cosmology and frame;
* ``frame``: the Hubble velocity of the line centres along a bundle (``hubble_velocity``).

The bundle itself is ``core.SkewerBundle`` (points, photon direction, a(x)); the gas comes from ``skewerjax.simulations``
and the statistics from ``skewerjax.summary_stats``, which import nothing from here: notebooks compose the three.
"""

from skewerjax.skewers.frame import hubble_velocity
from skewerjax.skewers.lattice import lattice_columns, lattice_points

__all__ = ["lattice_columns", "lattice_points", "hubble_velocity"]
