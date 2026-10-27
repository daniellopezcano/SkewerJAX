"""The expected spectra of a GRF along skewers (``utils.tutorial``, used by notebook 08): the mean P1D does not depend on
the subset of skewers, its scatter grows for fewer, the windows of the cells and of linear interpolation lower it, the
expected P3D on the skewer grid and the binned input power."""

import numpy as np

from skewerjax.simulations import PowerLawGRF
from skewerjax.utils import tutorial as tu


def test_grf_expectations():
    g = PowerLawGRF(A_s=1.0, k_0=1.0, n_s=-1.0, L=(10.0,) * 3, N=(16,) * 3)
    k = 2 * np.pi * np.arange(1, 8) / 10.0
    m1, s1 = tu.p1d_expected(g, k, return_sigma=True)
    m2, s2 = tu.p1d_expected(g, k, return_sigma=True, step=2)
    np.testing.assert_allclose(m1, m2, rtol=1e-12)  # the mean does not depend on the subset
    assert np.all(s2 >= s1)
    assert np.all(tu.p1d_expected(g, k, window="linear") <= tu.p1d_expected(g, k, window="cells"))
    assert np.all(tu.p1d_expected(g, k, window="cells") <= m1)
    E = tu.p3d_expected_on_skewer_grid(g, 2, np.array([0.0, 0.6]))
    assert E.shape == (8, 8, 2) and np.all(E >= 0)
    edges = np.geomspace(0.5, 4.0, 6)
    assert tu.binned_input_power(g, edges).size == 5
