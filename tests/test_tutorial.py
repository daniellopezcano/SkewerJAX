"""The toy set-ups of the tutorial notebooks: the sum converges to the integral at second order and conserves the column,
the linear detuning follows -(v - v_c)/c in the wings, the periodicity options agree, and the GRF expectations."""
import numpy as np

from skewerjax.core import PowerLawGRF, phys
from skewerjax.utils import tutorial as tu


def test_two_elements_sum_vs_integral():
    r = tu.two_elements(pieces=(4, 8, 16), n_fine=4001)
    err = {K: np.abs(t - r["tau_ref"]).max() / r["tau_ref"].max() for K, t in r["tau_sum"].items()}
    assert 3.5 < err[8] / err[16] < 4.5 and 3.5 < err[4] / err[8] < 4.5          # second order in the piece length
    for t in r["tau_sum"].values():
        assert abs(t.sum() / r["tau_ref"].sum() - 1) < 1e-6                        # the column is exact
    np.testing.assert_allclose(r["tau_sum_parts"][16].sum(0), r["tau_sum"][16], rtol=1e-12)   # additive
    assert abs(r["v"][np.argmax(r["tau_ref_parts"][0])] - r["v"][np.argmax(r["tau_ref_parts"][1])]) < 1.0


def test_linear_detuning_wings():
    d = tu.linear_detuning(log_N=(14.0,), n_v=4001)
    rel = d["tau_linear"][0] / d["tau_exact"][0] - 1
    for dv in (-2000.0, 1000.0, 2000.0):
        assert abs(np.interp(dv, d["v"], rel) / (-dv / phys.c_kms) - 1) < 0.02


def test_periodicity_options():
    p = tu.periodic_box_demo()
    t = p["tau"]
    assert t["wrapped"].sum() > t["none"].sum()
    assert np.abs(t["wrapped"] - t["buffer"]).max() < 1e-2 * t["wrapped"].max()


def test_grf_expectations():
    g = PowerLawGRF(A_s=1.0, k_0=1.0, n_s=-1.0, L=(10.0,) * 3, N=(16,) * 3)
    k = 2 * np.pi * np.arange(1, 8) / 10.0
    m1, s1 = tu.p1d_expected(g, k, return_sigma=True)
    m2, s2 = tu.p1d_expected(g, k, return_sigma=True, step=2)
    np.testing.assert_allclose(m1, m2, rtol=1e-12)                                 # the mean does not depend on the subset
    assert np.all(s2 >= s1)
    assert np.all(tu.p1d_expected(g, k, interpolated=True) <= m1)
    E = tu.p3d_expected_on_skewer_grid(g, 2, np.array([0.0, 0.6]))
    assert E.shape == (8, 8, 2) and np.all(E >= 0)
    edges = np.geomspace(0.5, 4.0, 6)
    assert tu.binned_input_power(g, edges).size == 5
