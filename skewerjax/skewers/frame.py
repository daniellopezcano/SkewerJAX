"""The frame of a bundle of skewers: the Hubble velocity of the line centres along its paths."""

from __future__ import annotations

import numpy as np

from skewerjax.core import SkewerBundle, phys


def hubble_velocity(bundle: SkewerBundle) -> np.ndarray:
    """v_c = c ln(a_dagger / a) [km/s] of the points of the first skewer (the line centre without peculiar velocity,
    ``core.spectral.line_centre_velocity_kms``), a_dagger = the bundle's a0."""
    a = np.asarray(bundle.scale_factors[0, 0], dtype=np.float64)
    return phys.c_kms * np.log(float(np.ravel(bundle.a0)[0]) / a)
