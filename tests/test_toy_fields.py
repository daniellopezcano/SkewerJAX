"""Toy fields (notebook 04): ``PowerLawGRF`` (the two grid modes and their checks, reproducible zero-mean draws whose
periodogram averages to P(k), the optional Kaiser / Fingers-of-God / BAO terms), ``grf_cells`` (n = n_bar (1 + delta)
with no transform on the grid cells: generators at the nodes, exact volumes, one realisation shared by several species,
per-cell T and v, the positivity warning, the input checks), ``interpolate_periodic``, and the expected spectra of a GRF
along skewers (``utils.tutorial``, used by notebook 08). The statistics of many draws
(P(k), xi(r), wedges): ``scripts/validation/check_summary_statistics.py``, notebook 08 §4."""
import warnings

import jax
import numpy as np
import pytest

from skewerjax.simulations import PowerLawGRF, grf_cells, interpolate_periodic
from skewerjax.utils import tutorial as tu


def test_grid_modes_and_checks():
    a = PowerLawGRF(1.0, 1.0, -1.0, L=(20.0, 10.0), N=(16, 8))
    b = PowerLawGRF(1.0, 1.0, -1.0, k_min=(2 * np.pi / 20.0, 2 * np.pi / 10.0), N_k=(8, 4))        # mode (b)
    c = PowerLawGRF(1.0, 1.0, -1.0, k_min=2 * np.pi / 20.0, k_max=8 * 2 * np.pi / 20.0)           # scalars broadcast
    assert a.shape == b.shape == (16, 8) and np.allclose(a.L, b.L, rtol=1e-12) and c.shape == (16,)
    assert a.dx == (1.25, 1.25) and a.cell_volume == 1.25 ** 2 and a.volume == 200.0 and a.ndim == 2
    np.testing.assert_allclose(a.k_max, (np.pi / 1.25, np.pi / 1.25), rtol=1e-12)               # Nyquist
    for kw in (dict(L=10.0), dict(L=10.0, N=8, N_k=4), dict(k_min=1.0, k_max=4.5), dict(k_min=1.0, k_max=4.0, N_k=3),
               dict(L=(1.0, 2.0, 3.0, 4.0), N=4), dict(L=10.0, N=1)):
        with pytest.raises(ValueError):
            PowerLawGRF(1.0, 1.0, -1.0, **kw)
    with pytest.raises(ValueError):
        PowerLawGRF(-1.0, 1.0, -1.0, L=10.0, N=8)                                               # A_s < 0


def test_draws_reproducible_zero_mean_and_unbiased():
    g = PowerLawGRF(A_s=2.0, k_0=1.0, n_s=-1.0, L=50.0, N=64)
    k1, k2 = jax.random.PRNGKey(1), jax.random.PRNGKey(2)
    d1 = np.asarray(g.sample(k1))
    assert np.array_equal(d1, np.asarray(g.sample(k1))) and not np.allclose(d1, np.asarray(g.sample(k2)))
    assert abs(d1.mean()) < 1e-14                                                              # the k = 0 mode is zero
    d, dk = jax.vmap(lambda k: g.sample(k, return_modes=True))(jax.random.split(k1, 800))
    np.testing.assert_allclose(np.asarray(dk)[0], g.cell_volume * np.fft.rfft(np.asarray(d)[0]), rtol=1e-10, atol=1e-12)
    P_hat = (np.abs(np.asarray(dk)) ** 2 / g.volume).mean(0)[1:]                               # <|delta_k|^2> / V = P(k)
    P = np.asarray(g.power_grid())[1:]
    assert np.max(np.abs(P_hat / P - 1)) < 0.3 and abs(np.mean(P_hat / P) - 1) < 0.03          # 800 draws: 5 % per mode
    assert abs(np.asarray(d).var(axis=1).mean() / float(g.variance()) - 1) < 0.1   # dominated by the k_min modes: ~4 %


def test_grid_cells_reproduce_n_bar_one_plus_delta():
    grf = PowerLawGRF(A_s=0.02, k_0=2.0, n_s=-1.25, L=(20.0,) * 3, N=(16,) * 3)
    delta = np.asarray(grf.sample(jax.random.PRNGKey(4)))
    c = grf_cells(delta, grf, {"HI": 3e-12, "SiIII": 0.5}, temperature=lambda x: 1e4 + x[:, 0], velocity=(1.0, 2.0, 3.0))
    assert c.grid == grf.shape and len(c) == 16 ** 3 and c.box == 20.0
    n = np.asarray(c.species["HI"]["number_density"]).reshape(grf.shape)
    np.testing.assert_allclose(n, 3e-12 * (1.0 + delta), rtol=1e-12)
    np.testing.assert_allclose(n.mean(), 3e-12, rtol=1e-12)                 # the k = 0 mode of delta is zero
    pos = np.asarray(c.positions).reshape(grf.shape + (3,))
    np.testing.assert_allclose(pos[:, 0, 0, 0], grf.x_grid()[0])           # generators = nodes, in C order
    np.testing.assert_allclose(pos[3, 5, 7], [3 * 1.25, 5 * 1.25, 7 * 1.25])
    np.testing.assert_allclose(c.volumes, 1.25 ** 3, rtol=1e-15)            # the cubes of the grid
    np.testing.assert_allclose(c.species["SiIII"]["temperature"], 1e4 + np.asarray(c.positions)[:, 0])
    np.testing.assert_allclose(c.species["SiIII"]["velocity"][7], [1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        grf_cells(delta[:8], grf, {"HI": 1.0})                              # not the grid shape
    with pytest.raises(ValueError):
        grf_cells(np.zeros((16, 8, 8)), PowerLawGRF(0.02, 2.0, -1.25, L=(20.0, 10.0, 10.0), N=(16, 8, 8)), {"HI": 1.0})


def test_shared_realisation_and_negative_warning():
    grf = PowerLawGRF(A_s=4.0, k_0=2.0, n_s=-1.25, L=(20.0,) * 3, N=(16,) * 3)   # sigma ~ 1: negative somewhere
    delta = np.asarray(grf.sample(jax.random.PRNGKey(0)))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        c = grf_cells(delta, grf, {"HI": 1.0, "SiIII": 2.0})
    assert any("min(1 + delta)" in str(x.message) for x in w)
    np.testing.assert_allclose(c.species["SiIII"]["number_density"] - 2.0, 2.0 * (c.species["HI"]["number_density"] - 1.0),
                               rtol=1e-12, atol=1e-12)


def test_interpolate_periodic():
    """Exact at the nodes, linear between them, periodic across the faces, anisotropic boxes."""
    rng = np.random.default_rng(0)
    L, N = np.array([8.0, 4.0, 6.0]), (8, 4, 6)
    f = rng.normal(size=N)
    idx = np.stack(np.meshgrid(*[np.arange(n) for n in N], indexing="ij"), -1).reshape(-1, 3)
    nodes = idx * L / N
    np.testing.assert_allclose(np.asarray(interpolate_periodic(f, nodes, L)), f.reshape(-1), rtol=1e-12)
    np.testing.assert_allclose(np.asarray(interpolate_periodic(f, nodes + L * [2, -1, 3], L)), f.reshape(-1), rtol=1e-12)
    half = np.array([[7.5, 0.0, 0.0]])                                      # between the last and the first x node
    np.testing.assert_allclose(np.asarray(interpolate_periodic(f, half, L)), 0.5 * (f[7, 0, 0] + f[0, 0, 0]), rtol=1e-12)
    lin = np.add.outer(np.add.outer(np.arange(8.0), 2 * np.arange(4.0)), 3 * np.arange(6.0))   # linear inside the box
    p = rng.uniform(0, 1, (50, 3)) * (L - L / N)                            # away from the periodic seam
    np.testing.assert_allclose(np.asarray(interpolate_periodic(lin, p, L)), (p * np.array(N) / L) @ [1.0, 2.0, 3.0], rtol=1e-12)


def test_grf_optional_terms():
    """Kaiser, Fingers of God and BAO terms: exact transforms, positivity check, mu along los_axis, theory helpers."""
    from scipy import integrate, special
    A, r0, sg = 0.02, 10.0, 1.5
    xi = lambda r: A * (np.exp(-0.5 * ((r - r0) / sg) ** 2) + np.exp(-0.5 * ((r + r0) / sg) ** 2))
    kern = {1: lambda r, k: 2 * xi(r) * np.cos(k * r), 2: lambda r, k: 2 * np.pi * r * xi(r) * special.j0(k * r),
            3: lambda r, k: 4 * np.pi * r ** 2 * xi(r) * np.sinc(k * r / np.pi)}
    for d, N in ((1, 64), (2, (32,) * 2), (3, (8,) * 3)):
        g = PowerLawGRF(1e3, 1.0, -0.5, L=(200.0,) * d if d > 1 else 200.0, N=N, A_bao=A, r_bao=r0, sigma_bao=sg)
        for k in (0.05, 0.7, 2.0):
            ref = integrate.quad(kern[d], 0, r0 + 30 * sg, args=(k,), limit=2000, points=[r0])[0]
            assert abs(float(g.bao_power(k)) - ref) < 1e-6 * abs(4 * np.pi * A * r0 ** 2 * sg), (d, k)
    try:
        PowerLawGRF(1e-3, 1.0, -0.5, L=200.0, N=512, A_bao=1.0, r_bao=20.0, sigma_bao=2.0)
        raise AssertionError("P < 0 must raise")
    except ValueError as e:
        assert "P(k) < 0" in str(e)
    g = PowerLawGRF(2.0, 1.0, -1.0, L=(30.0,) * 3, N=(12,) * 3, beta=0.5, sigma_fog=2.0, los_axis=1)
    np.testing.assert_allclose(float(g.power(0.7, 0.6)), 2.0 * 0.7 ** -2 * (1 + 0.5 * 0.36) ** 2 / (1 + (0.7 * 0.6 * 2.0) ** 2), rtol=1e-14)
    kv, km = g.k_grid(real=True)
    np.testing.assert_allclose(np.asarray(g.power_at(*kv)), np.asarray(g.power_grid()), rtol=1e-14)   # mu along axis 1
    mu = np.linspace(0, 1, 20001)
    np.testing.assert_allclose(float(g.power_monopole(0.7)), np.trapz(np.asarray(g.power(0.7, mu)), mu), rtol=1e-8)
    iso = PowerLawGRF(2.0, 1.0, -1.0, L=(30.0,) * 3, N=(12,) * 3)
    np.testing.assert_array_equal(np.asarray(iso.power_grid()), np.asarray(iso.power(iso.k_grid()[1])))   # terms off: unchanged
    leaves, aux = jax.tree_util.tree_flatten(g)
    assert repr(jax.tree_util.tree_unflatten(aux, leaves)) == repr(g)


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
