"""The summary statistics of the flux: the overflux of each line and of their product, and its 1D and 3D two-point
statistics, total, term by term and order by order (imports ``skewerjax.core`` at most; nothing in ``core`` imports it).

* ``overflux``: ``DeltaFieldBuilder`` (delta_j = F_j / <F_j> - 1, the subset products delta_A, the total and its exact
  decomposition), ``overflux_terms``, ``equivalent_width_kms``;
* ``orders``: the order-by-order decomposition (the order-0 term ``ZERO_ORDER_KEY``, ``subset_orders``, ``order_sum``);
* ``stats_1d``: P1D and xi1D of the overflux (``OverfluxStatistics1D``, ``ensemble_1d``, ``p1d_from_tau``);
* ``stats_3d``: P3D and xi3D of the overflux on a grid of skewers, by term and order (``OverfluxStatistics3D``);
* ``grid3d``: P and xi of a periodic 3D field on its grid, isotropic, in the (par, perp) plane and in wedges of mu, with
  exact expectations and covariance (``GridStatistics3D``, ``MU_WEDGES``);
* ``fourier``, ``binning``: the shared conventions (lag axes, the Wiener-Khinchin pair) and binning (bin indices, jitted
  bin means, bin edges, the mean over realisations ``ensemble_mean``);
* ``contaminants``: the P1D metal-contamination models (McDonald et al.; Ma et al.) and their fits.
"""

from skewerjax.summary_stats import binning, contaminants, fourier, grid3d, orders, overflux, stats_1d, stats_3d
from skewerjax.summary_stats.binning import binned_mean, ensemble_mean, hybrid_k_edges, occupied_edges
from skewerjax.summary_stats.contaminants import fit_ma_model_profiled, fit_mcdonald_a, ma_factor, mcdonald_factor
from skewerjax.summary_stats.fourier import lag_axis, xi3d_from_power, xi_from_power
from skewerjax.summary_stats.grid3d import GRID_KEYS, MU_WEDGES, GridStatistics3D
from skewerjax.summary_stats.orders import ZERO_ORDER_KEY, order_sum
from skewerjax.summary_stats.overflux import DeltaFieldBuilder, equivalent_width_kms, overflux_terms
from skewerjax.summary_stats.stats_1d import OverfluxStatistics1D, ensemble_1d, p1d_from_tau
from skewerjax.summary_stats.stats_3d import OverfluxStatistics3D

__all__ = [
    "DeltaFieldBuilder",
    "overflux_terms",
    "equivalent_width_kms",
    "OverfluxStatistics1D",
    "OverfluxStatistics3D",
    "p1d_from_tau",
    "ensemble_1d",
    "ensemble_mean",
    "GridStatistics3D",
    "GRID_KEYS",
    "MU_WEDGES",
    "hybrid_k_edges",
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
