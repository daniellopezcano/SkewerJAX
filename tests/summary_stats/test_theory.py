"""``summary_stats.theory``: the integral constraint against a brute-force 3D quadrature and closed forms, finite for the red
toy spectrum; the positions of the BAO maxima from j_0 and from the toy's own bump; the Fourier imprint of a pair of lines."""

import numpy as np
import pytest
from scipy.special import erf, spherical_jn

from skewerjax.simulations import ToyBoxes
from skewerjax.summary_stats import bao_maxima, integral_constraint, pair_imprint, sphere_area_in_cube


def _brute(xi, L, n=256):
    x = (np.arange(n) + 0.5) / n * L - L / 2
    r = np.sqrt(x[:, None, None] ** 2 + x[None, :, None] ** 2 + x[None, None, :] ** 2)
    return float(np.mean(xi(r)))


def test_area_of_the_sphere_inside_the_cube():
    L = 2.0
    rng = np.random.default_rng(0)
    u = rng.normal(size=(1_000_000, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    for r in (0.5, 1.0, 1.2, np.sqrt(2.0), 1.5, 1.65, 1.72):
        frac = np.mean(np.all(np.abs(u * r) <= 1.0, axis=1))
        got = float(sphere_area_in_cube(np.array([r]), L)[0]) / (4 * np.pi * r**2)
        assert got == pytest.approx(frac, abs=2e-3), r
    assert float(sphere_area_in_cube(np.array([1.8]), L)[0]) == 0.0
    a, b = sphere_area_in_cube(np.array([np.sqrt(2.0) * (1 - 1e-9), np.sqrt(2.0) * (1 + 1e-9)]), L)
    assert a == pytest.approx(b, rel=1e-6), "continuous across the first overlap of the caps"
    r = np.linspace(0, np.sqrt(3.0) * L / 2, 400001)
    assert np.trapz(sphere_area_in_cube(r, L), r) == pytest.approx(L**3, rel=1e-6), "the area integrates to the volume"


def test_integral_constraint_against_closed_forms_and_a_3d_quadrature():
    L = 150.0
    assert integral_constraint(lambda r: np.ones_like(r), L) == pytest.approx(1.0, rel=1e-10)
    for s in (20.0, 100.0):
        exact = (2 * np.pi * s**2) ** 1.5 / L**3 * erf(L / (2 * np.sqrt(2) * s)) ** 3
        assert integral_constraint(lambda r: np.exp(-(r**2) / (2 * s**2)), L) == pytest.approx(exact, rel=1e-10)
    xi = lambda r: r**-1.6  # noqa: E731
    assert integral_constraint(xi, L) == pytest.approx(_brute(xi, L), rel=2e-3), "the brute-force 3D quadrature"


def test_integral_constraint_of_the_red_toy_spectrum_is_finite():
    b = ToyBoxes(box_cMpc=150.0, n_s=-0.4, n_cells=128, n_real=1, bao_ratio=0.003, sigma_bao=1.0, r_bao=40.0)
    g = b.grf()
    xi = lambda r: np.asarray(g.correlation(r), dtype=float)  # noqa: E731
    bar = integral_constraint(xi, b.box_cMpc)
    assert np.isfinite(bar) and bar > 0
    assert bar == pytest.approx(_brute(xi, b.box_cMpc, 256), rel=5e-3)
    assert bar == pytest.approx(integral_constraint(xi, b.box_cMpc, n_panels=8192, nodes=12), rel=1e-5), "converged (narrow bump)"


def test_bao_maxima():
    r_d = 147.3
    k = bao_maxima(0.5, r_bao=r_d)
    np.testing.assert_allclose(k[:3] * r_d, [7.725, 14.066, 20.371], atol=1e-3)
    assert np.all(spherical_jn(0, k * r_d) > 0) and np.all(np.abs(spherical_jn(1, k * r_d)) < 1e-9)
    assert k[-1] <= 0.5 and np.all(np.diff(k) > 0)
    np.testing.assert_allclose(bao_maxima(0.5, r_bao=r_d, kms_per_cMpc=80.0), k / 80.0)
    assert bao_maxima(0.5, r_bao=r_d, k_min=0.12)[0] > 0.12
    b = ToyBoxes(box_cMpc=150.0, n_s=-0.4, n_cells=128, n_real=1, bao_ratio=0.003, sigma_bao=1.0, r_bao=40.0)
    g = b.grf()
    km = bao_maxima(np.pi / b.dx, window=g.bao_window, r_bao=b.r_bao)
    W = np.asarray(g.bao_window(km), dtype=float)
    h = 1e-3
    assert np.all(W > 0) and np.all(np.asarray(g.bao_window(km + h)) < W) and np.all(np.asarray(g.bao_window(km - h)) < W)
    assert np.mean(np.diff(km)) == pytest.approx(2 * np.pi / b.r_bao, rel=0.02), "one maximum per BAO period"
    np.testing.assert_allclose(km, bao_maxima(np.pi / b.dx, window=g.bao_window), atol=2e-5)
    with pytest.raises(ValueError, match="window"):
        bao_maxima(1.0)


def test_pair_imprint_along_the_line_of_sight():
    """P1D: the cross term goes as cos(k Delta), maxima at k = 2 pi n / Delta."""
    d = 2271.0
    k = pair_imprint(d, 0.03)
    np.testing.assert_allclose(k, 2 * np.pi * np.arange(1, k.size + 1) / d)
    assert k[-1] <= 0.03 < k[-1] + 2 * np.pi / d, "every maximum within the range"
    np.testing.assert_allclose(np.cos(k * d), 1.0)
    np.testing.assert_allclose(pair_imprint(d, 0.03, n_harmonics=2), k[:2])
    assert pair_imprint(d, 0.03, k_min=k[2]).tolist() == k[3:].tolist(), "k_min excluded"
    assert pair_imprint(d, 2 * np.pi / d).size == 1 and pair_imprint(d, 0.99 * 2 * np.pi / d).size == 0
    assert pair_imprint(0.0, 1.0).size == 0, "no separation, no imprint"
    assert pair_imprint(d, 0.03, n_harmonics=1000).size == k.size


def test_pair_imprint_folds_with_the_period():
    """On the modes k = 2 pi m / P of a periodic box a separation and its nearest image are the same: cos(k Delta) =
    cos(k Delta_fold), so the imprint is that of the folded separation."""
    P = 3961.5
    d = 2271.0
    fold = P - d
    assert fold < P / 2 < d
    np.testing.assert_allclose(pair_imprint(d, 0.03, period=P), pair_imprint(fold, 0.03))
    np.testing.assert_allclose(pair_imprint(d + 2 * P, 0.03, period=P), pair_imprint(fold, 0.03))
    modes = 2 * np.pi * np.arange(1, 64) / P
    np.testing.assert_allclose(np.cos(modes * d), np.cos(modes * fold), atol=1e-9)
    assert pair_imprint(P, 1.0, period=P).size == 0, "a separation of a whole period: no imprint"
    np.testing.assert_allclose(pair_imprint(32.0, 1.0, kind="iso", period=51.67), pair_imprint(51.67 - 32.0, 1.0, kind="iso"))


def test_pair_imprint_in_the_isotropic_spectrum():
    """P(|k|): the average of cos(k mu Delta) over the directions is j_0(k Delta), and its maxima are those of ``bao_maxima``
    for a shell of radius Delta; compared with the direction average computed by brute force."""
    d = 32.0
    k = pair_imprint(d, 1.3, kind="iso")
    np.testing.assert_allclose(k * d, [7.725, 14.066, 20.371, 26.666, 32.956, 39.244], atol=2e-3)
    np.testing.assert_allclose(k, bao_maxima(1.3, r_bao=d))
    mu = (np.arange(4000) + 0.5) / 4000
    average = lambda q: np.mean(np.cos(q * d * mu))  # noqa: E731
    np.testing.assert_allclose([average(q) for q in k], spherical_jn(0, k * d), atol=1e-6)
    for q in k:
        assert average(q) > average(q - 1e-3) and average(q) > average(q + 1e-3), "a maximum of the direction average"
        assert average(q) > 0
    np.testing.assert_allclose(pair_imprint(d, 1.3, kind="iso", n_harmonics=3), k[:3])
    with pytest.raises(ValueError, match="kind"):
        pair_imprint(d, 1.3, kind="3d")
    with pytest.raises(ValueError, match="n_harmonics"):
        pair_imprint(d, 1.3, n_harmonics=0)
