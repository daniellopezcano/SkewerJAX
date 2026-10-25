"""The toy set-ups of the tutorial notebooks: the periodic pieces demo (the two effects of velocity_period, the
nearest-image limitation) and the GRF expectations."""
import numpy as np

from skewerjax.core import PowerLawGRF
from skewerjax.utils import tutorial as tu


def test_periodic_pieces_demo():
    d = tu.periodic_pieces_demo(n_images=20)
    v, P = d["v"], d["P"]
    i = int(np.argmax(d["v_c"] - P))                                               # (i): line centre beyond P
    assert d["v_c"][i] > P and abs(v[np.argmax(d["tau_wrap"][i])] - (d["v_c"][i] - P)) < 2.0
    assert d["tau_nowrap"][i].max() < 1e-2 * d["tau_wrap"][i].max()
    j = d["strong"]                                                                # (ii): the wing re-enters at v ~ 0
    assert d["v_c"][j] < P and d["tau_wrap"][j][v < 50].max() > 1e3 * d["tau_nowrap"][j][v < 50].max()
    for k in range(len(d["s"])):                                                   # pieces far from the ends: unchanged
        if k not in (i, j):
            np.testing.assert_allclose(d["tau_wrap"][k][d["tau_wrap"][k] > 1e-3], d["tau_nowrap"][k][d["tau_wrap"][k] > 1e-3], rtol=1e-6)
    miss = d["tau_strong_exact"] - d["tau_strong_nearest"]                        # the nearest image misses the far images
    assert miss.min() > -1e-9 and abs(v[np.argmax(miss)] - np.mod(d["v_c"][j] - 0.5 * P, P)) < 5.0


def test_grf_expectations():
    g = PowerLawGRF(A_s=1.0, k_0=1.0, n_s=-1.0, L=(10.0,) * 3, N=(16,) * 3)
    k = 2 * np.pi * np.arange(1, 8) / 10.0
    m1, s1 = tu.p1d_expected(g, k, return_sigma=True)
    m2, s2 = tu.p1d_expected(g, k, return_sigma=True, step=2)
    np.testing.assert_allclose(m1, m2, rtol=1e-12)                                 # the mean does not depend on the subset
    assert np.all(s2 >= s1)
    assert np.all(tu.p1d_expected(g, k, window="linear") <= tu.p1d_expected(g, k, window="cells"))
    assert np.all(tu.p1d_expected(g, k, window="cells") <= m1)
    E = tu.p3d_expected_on_skewer_grid(g, 2, np.array([0.0, 0.6]))
    assert E.shape == (8, 8, 2) and np.all(E >= 0)
    edges = np.geomspace(0.5, 4.0, 6)
    assert tu.binned_input_power(g, edges).size == 5
