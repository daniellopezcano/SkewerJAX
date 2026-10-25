"""Block A, the core backbone: lines, cosmology, skewers, gas cells, the route cells -> pieces, line profiles, tau, F and
statistics.

Self-contained (imports nothing from ``skewerjax.simulations`` or ``skewerjax.comparison``). The gas of every source
(toy models here, simulations in ``skewerjax.simulations``) is one container, ``Cells`` (Voronoi cells); one route takes
it to tau: ``select_cells`` -> deposition (``voronoi_segments`` / ``grid_segments`` / ``tophat_segments``) ->
``cell_pieces`` (``Pieces``) -> ``compute_optical_depth``. Every numerical function of the tau sum is written in
``jax.numpy`` and works under ``jit`` and ``vmap``; the containers (``Cosmology``, ``LineBundle``, ``SkewerBundle``,
``Cells``, ``Pieces``, ``PowerLawGRF``) are pytrees; the geometry (selection, deposition) runs on the host.
"""
from skewerjax.core.constants import PhysicalConstants, phys
from skewerjax.core.lines import LINES, Line, LineBundle, get_line, get_mass_amu
from skewerjax.core.cosmology import Cosmology
from skewerjax.core.skewers import (SkewerBundle, axis_aligned_skewer_points, grid_skewer_points, line_elements, path_geometry,
                                    plane_points)
from skewerjax.core.cells import Cells, cells_from_values, cells_on_grid, grid_points, voronoi_volumes
from skewerjax.core.toy_fields import PowerLawGRF, grf_cells, interpolate_periodic
from skewerjax.core.selection import select_cells
from skewerjax.core.deposition import (CellCrossings, SegmentComponents, cell_crossings, grid_segments, segment_fields, tophat_segments,
                                       voronoi_segments)
from skewerjax.core.voigt import gaussian, log10_voigt, lorentzian, voigt_convolution, voigt_exojax, voigt_scipy
from skewerjax.core.spectral import (frequency_offset_GHz, line_centre_velocity_kms, line_wavelength_grid_A, local_wavelength_A,
                                     local_wavelength_from_observed_A, velocities_in_line_frames)
from skewerjax.core.optical_depth import (Pieces, compute_optical_depth, line_profiles, optical_depth_from_pieces, optical_depth_kernel,
                                          optical_depth_sharded, resolution_report, voigt_line_profiles, voigt_width_kms,
                                          voigt_widths_GHz, wrap_to_ref_line)
from skewerjax.core.absorbers import (Absorbers, ExactFrame, SnapshotFrame, absorber_pieces, absorbers_from_components,
                                      absorbers_from_crossings, bundle_frame, cell_pieces, column_per_pixel, deposit, sampling_check,
                                      scale_factors_along_paths, tau_absorbers, tau_by_path_bins)
from skewerjax.core.overflux import DeltaFieldBuilder, equivalent_width_kms, overflux_terms
from skewerjax.core import analytic, stats

__all__ = [n for n in dir() if not n.startswith("_")]
