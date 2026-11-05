"""The summary statistics of the flux: the overflux of each line and of their product, and its 1D and 3D two-point
statistics, total, term by term and order by order; and the statistics of 3D fields and of the density along skewers.
From arrays, geometry and ``core.Cells`` only: imports ``skewerjax.core`` at most, nothing from ``skewerjax.simulations``
(``tests/test_import_graph.py``); nothing in ``core`` imports it.

* ``overflux``: ``DeltaFieldBuilder`` (delta_j = T_j / T_bar_j - 1 with T = e^-tau the transmission and T_bar_j its
  mean, ``mean`` = "redshift" (default, T_bar(z)), "global", "box" or "skewer", the subset products delta_A, the total
  and its exact decomposition), ``overflux_terms``, ``mean_transmissions``, ``ensemble_mean_transmissions`` (an ensemble
  of realisations), ``overflux_check``, ``check_shared_grid``, ``redshift_groups``, ``MEAN_MODES``, ``equivalent_width_kms``;
* ``orders``: the order-by-order decomposition (the order-0 term ``ZERO_ORDER_KEY``, ``subset_orders``, ``order_sum``);
* ``stats_1d``: P1D and xi1D of the overflux by term and order (``OverfluxStatistics1D``, ``decompose_1d``,
  ``ensemble_1d``, ``p1d_from_tau``);
* ``lattice``: the statistics of an ensemble of periodic skewers from arrays and geometry only: P1D and xi1D by one 1D
  FFT per skewer (``skewer_p1d``, ``skewer_xi1d``), and the 3D statistics on the lattice of skewers (``lattice_grid``,
  ``lattice_limits``, ``on_lattice``, ``measure_lattice``), and the cell values at the points of the skewers (``on_points``);
* ``field_stats``: P and xi of the density of gas cells (``core.Cells``, any source: toy grids and snapshots alike)
  over an ensemble of realisations, through a mesh, against an optional expectation P(k) (``box_grid``, ``grid_limits``,
  ``grid_power``, ``field_statistics``, ``cell_weights``, ``xi_chi2``, ``xi_subset_offsets``, ``bao_height``, ``xi_corner``);
* ``mesh``: points on a periodic mesh in 1D, 2D, 3D (Sefusatti et al. 2016): mass assignment ``paint`` (NGP, CIC, TSC,
  PCS; chunked), its window ``mas_window``, the interlaced and deconvolved Fourier field ``fourier_mesh``, the alias sum
  ``aliased_power``, the transfer of fields given at the nodes ``node_transfer``; ``mas`` / ``interlace`` only on the mesh
  path of the cells (``field_statistics``, ``GridStatistics3D.measure_points``); the skewer samples stay on their own
  lattice and refuse both (``mesh.lattice_only``: ``measure_lattice``, ``skewer_density_statistics``, ``skewer_p1d``, ...);
* ``density``: the density along a lattice of skewers through the same 3D estimator, and its exact expectation on that
  lattice from P(k) (``skewer_density_statistics``, ``lattice_density_theory``);
* ``stats_3d``: P3D and xi3D of the overflux on a grid of skewers, by term and order (``OverfluxStatistics3D``);
* ``grid3d``: P and xi of a periodic 3D field on its grid, isotropic, in the (par, perp) plane and in wedges of mu, with
  exact expectations and covariance (``GridStatistics3D``, ``MU_WEDGES``);
* ``theory``: the mean of xi over the box, the term of the k = 0 mode (``integral_constraint``), and the positions of the
  BAO maxima in k (``bao_maxima``), and the Fourier imprint of the separation of two lines (``pair_imprint``);
* ``fourier``, ``binning``: the shared conventions (lag axes, the Wiener-Khinchin pair) and binning (bin indices, jitted
  bin means, bin edges (``bao_k_edges``, ``native_edges``, ``full_coverage``), the mean over realisations
  ``ensemble_mean``);
* ``contaminants``: the P1D metal-contamination models (McDonald et al.; Ma et al.) and their fits.
"""

from skewerjax.summary_stats import (
    binning,
    contaminants,
    density,
    field_stats,
    fourier,
    grid3d,
    lattice,
    mesh,
    orders,
    overflux,
    stats_1d,
    stats_3d,
    theory,
)
from skewerjax.summary_stats.binning import (
    X_BIN_MODES,
    bao_k_edges,
    bin_centres,
    binned_mean,
    ensemble_mean,
    full_coverage,
    hybrid_k_edges,
    native_edges,
    occupied_edges,
)
from skewerjax.summary_stats.contaminants import fit_ma_model_profiled, fit_mcdonald_a, ma_factor, mcdonald_factor
from skewerjax.summary_stats.density import lattice_density_theory, skewer_density_statistics
from skewerjax.summary_stats.field_stats import (
    bao_height,
    box_grid,
    cell_weights,
    field_statistics,
    grid_limits,
    grid_power,
    xi_chi2,
    xi_corner,
    xi_subset_offsets,
)
from skewerjax.summary_stats.fourier import lag_axis, xi3d_from_power, xi_from_power
from skewerjax.summary_stats.grid3d import GRID_KEYS, MU_WEDGES, GridStatistics3D
from skewerjax.summary_stats.lattice import (
    lattice_grid,
    lattice_limits,
    measure_lattice,
    on_lattice,
    on_points,
    skewer_p1d,
    skewer_xi1d,
)
from skewerjax.summary_stats.mesh import (
    MAS_SCHEMES,
    aliased_power,
    fourier_mesh,
    mas_window,
    node_transfer,
    paint,
)
from skewerjax.summary_stats.orders import ZERO_ORDER_KEY, order_sum
from skewerjax.summary_stats.overflux import (
    MEAN_MODES,
    DeltaFieldBuilder,
    check_shared_grid,
    ensemble_mean_transmissions,
    equivalent_width_kms,
    mean_transmissions,
    overflux_check,
    overflux_terms,
    redshift_groups,
)
from skewerjax.summary_stats.stats_1d import OverfluxStatistics1D, decompose_1d, ensemble_1d, p1d_from_tau
from skewerjax.summary_stats.stats_3d import OverfluxStatistics3D
from skewerjax.summary_stats.theory import bao_maxima, integral_constraint, pair_imprint, sphere_area_in_cube

__all__ = [
    "DeltaFieldBuilder",
    "overflux_terms",
    "mean_transmissions",
    "ensemble_mean_transmissions",
    "overflux_check",
    "MEAN_MODES",
    "check_shared_grid",
    "redshift_groups",
    "equivalent_width_kms",
    "OverfluxStatistics1D",
    "OverfluxStatistics3D",
    "p1d_from_tau",
    "decompose_1d",
    "ensemble_1d",
    "skewer_p1d",
    "skewer_xi1d",
    "lattice_grid",
    "lattice_limits",
    "on_lattice",
    "measure_lattice",
    "on_points",
    "box_grid",
    "grid_limits",
    "grid_power",
    "field_statistics",
    "cell_weights",
    "paint",
    "mas_window",
    "fourier_mesh",
    "node_transfer",
    "aliased_power",
    "MAS_SCHEMES",
    "xi_chi2",
    "xi_subset_offsets",
    "xi_corner",
    "integral_constraint",
    "sphere_area_in_cube",
    "bao_maxima",
    "pair_imprint",
    "bao_height",
    "skewer_density_statistics",
    "lattice_density_theory",
    "ensemble_mean",
    "GridStatistics3D",
    "GRID_KEYS",
    "MU_WEDGES",
    "hybrid_k_edges",
    "bao_k_edges",
    "bin_centres",
    "X_BIN_MODES",
    "native_edges",
    "full_coverage",
    "occupied_edges",
    "ZERO_ORDER_KEY",
    "binned_mean",
    "lag_axis",
    "order_sum",
    "xi_from_power",
    "xi3d_from_power",
    "mcdonald_factor",
    "ma_factor",
    "fit_mcdonald_a",
    "fit_ma_model_profiled",
]
