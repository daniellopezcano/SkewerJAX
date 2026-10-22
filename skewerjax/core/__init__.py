"""Block A, the core backbone: lines, cosmology, skewers, field samplers, line profiles, tau, F and statistics.

Self-contained (imports nothing from ``skewerjax.simulations`` or ``skewerjax.comparison``). Every numerical function
is written in ``jax.numpy`` and works under ``jit`` and ``vmap``; the containers (``Cosmology``, ``LineBundle``,
``SkewerBundle``, ``Fields`` and its samplers, ``LoSSamples``, ``PowerLawGRF``) are pytrees.
"""
from skewerjax.core.constants import PhysicalConstants, phys
from skewerjax.core.lines import LINES, Line, LineBundle, get_line, get_mass_amu
from skewerjax.core.cosmology import Cosmology
from skewerjax.core.skewers import (SkewerBundle, axis_aligned_skewer_points, grid_skewer_points, line_elements, path_geometry,
                                    plane_points)
from skewerjax.core.fields import Fields, SampledField, ScalarField, VectorField
from skewerjax.core.toy_fields import Constant, GaussianDensity, LogRamp, PowerLawGRF, VelocityRamp
from skewerjax.core.voigt import gaussian, log10_voigt, lorentzian, voigt_convolution, voigt_exojax, voigt_scipy
from skewerjax.core.spectral import (frequency_offset_GHz, line_centre_velocity_kms, line_wavelength_grid_A, local_wavelength_A,
                                     local_wavelength_from_observed_A, velocities_in_line_frames)
from skewerjax.core.optical_depth import (LoSSamples, compute_optical_depth, line_profiles, los_samples, optical_depth_from_samples, optical_depth_sharded,
                                          optical_depth_kernel, resolution_report, voigt_line_profiles, voigt_width_kms,
                                          voigt_widths_GHz, wrap_to_ref_line)
from skewerjax.core.overflux import DeltaFieldBuilder, equivalent_width_kms, overflux_terms
from skewerjax.core import analytic, stats

__all__ = [n for n in dir() if not n.startswith("_")]
