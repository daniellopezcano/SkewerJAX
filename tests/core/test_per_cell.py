"""The windowed per-cell kernel (``core.per_cell``): the windows and their escalation change the cost, never the result.

fake_spectra's walk of each cell (``tautail``) and its pixel average are checked against fake_spectra's rules in
``test_fake_spectra_options.py``; here the same calls are made with windows far too small (every cell escalates through
several levels) and with one window holding the whole grid, on many skewers at once, and must agree to rounding.
"""

import numpy as np

import skewerjax.core.per_cell as pc
from skewerjax.core import compute_optical_depth

from helpers import ramp_pieces, tol


def _tau(pieces, v, lines, levels, **kw):
    saved = pc.WINDOW_LEVELS
    pc.WINDOW_LEVELS = levels
    try:
        return np.asarray(compute_optical_depth(pieces, v, lines, resolution_tolerance=None, **kw), dtype=np.float64)
    finally:
        pc.WINDOW_LEVELS = saved


def _setup():
    pieces, v, lines = ramp_pieces(n_skewers=5)
    cell = np.broadcast_to(np.arange(pieces.shape[-1]) // 3, pieces.scale_factor.shape)  # three pieces per cell
    import dataclasses

    import jax.numpy as jnp

    return dataclasses.replace(pieces, cell=jnp.asarray(cell)), v, lines


def test_windows_escalate_to_the_full_walk():
    pieces, v, lines = _setup()
    P = float(v.size * (v[1] - v[0]))
    for tautail in (1e-7, 1e-3):
        for kw in (dict(), dict(velocity_period=P), dict(pixel_average=True, velocity_period=P)):
            small = _tau(pieces, v, lines, (1, 2, 4, 8), tautail=tautail, **kw)  # every cell escalates
            whole = _tau(pieces, v, lines, (10**6,), tautail=tautail, **kw)  # one window: the whole walk
            scale = np.abs(whole).max()
            assert np.max(np.abs(small - whole)) <= tol(1e-12, 1e-5) * scale, (tautail, kw)
            assert np.any(small > 0)


def test_pixel_average_without_the_walk_covers_the_grid():
    pieces, v, lines = _setup()
    a = _tau(pieces, v, lines, (1,), pixel_average=True)
    b = _tau(pieces, v, lines, (10**6,), pixel_average=True)
    np.testing.assert_array_equal(a, b)  # without tautail the windows do not apply
