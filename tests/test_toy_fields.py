"""grf_cells: n = n_bar (1 + delta) with no transform (grid generators reproduce the realisation, zero-mean delta,
positivity check, one realisation shared by several species); the uniform and density-weighted generating points."""
import warnings

import jax
import numpy as np

from skewerjax.simulations import PowerLawGRF, density_weighted_generators, grf_cells, uniform_generators


def test_grid_cells_reproduce_n_bar_one_plus_delta():
    grf = PowerLawGRF(A_s=0.02, k_0=2.0, n_s=-1.25, L=(20.0,) * 3, N=(16,) * 3)
    delta = np.asarray(grf.sample(jax.random.PRNGKey(4)))
    c = grf_cells(delta, grf, {"HI": 3e-12})
    n = np.asarray(c.species["HI"]["number_density"]).reshape(grf.shape)
    np.testing.assert_allclose(n, 3e-12 * (1.0 + delta), rtol=1e-12)
    np.testing.assert_allclose(n.mean(), 3e-12, rtol=1e-12)                 # the k = 0 mode of delta is zero
    np.testing.assert_allclose(np.asarray(c.positions).reshape(grf.shape + (3,))[:, 0, 0, 0], grf.x_grid()[0])   # nodes = generators


def test_shared_realisation_and_negative_warning():
    grf = PowerLawGRF(A_s=4.0, k_0=2.0, n_s=-1.25, L=(20.0,) * 3, N=(16,) * 3)   # sigma ~ 1: negative somewhere
    delta = np.asarray(grf.sample(jax.random.PRNGKey(0)))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        c = grf_cells(delta, grf, {"HI": 1.0, "SiIII": 2.0})
    assert any("min(1 + delta)" in str(x.message) for x in w)
    np.testing.assert_allclose(c.species["SiIII"]["number_density"] - 2.0, 2.0 * (c.species["HI"]["number_density"] - 1.0),
                               rtol=1e-12, atol=1e-12)


def test_generators_reproducible_and_density_weighted():
    grf = PowerLawGRF(A_s=0.04, k_0=2.0, n_s=-1.25, L=(20.0,) * 3, N=(8,) * 3)
    delta = np.asarray(grf.sample(jax.random.PRNGKey(2)))
    u = uniform_generators(grf, 4000, seed=3)
    np.testing.assert_array_equal(u, uniform_generators(grf, 4000, seed=3))
    np.testing.assert_array_equal(u, np.random.default_rng(3).uniform(0.0, 20.0, (4000, 3)))
    M = 400_000
    g = density_weighted_generators(delta, grf, M, seed=5)
    np.testing.assert_array_equal(g, density_weighted_generators(delta, grf, M, seed=5))
    assert g.shape == (M, 3) and g.min() >= 0.0 and g.max() < 20.0
    # counts per grid cell (the cube centred on each node): Poisson with mean M (1 + delta_j) / N_cells
    j = np.mod(np.floor(g / grf.dx[0] + 0.5).astype(int), 8)
    counts = np.bincount(np.ravel_multi_index(j.T, (8,) * 3), minlength=512)
    expected = M * (1.0 + delta.ravel()) / 512
    pull = (counts - expected) / np.sqrt(expected)
    assert abs(pull.mean()) < 0.2 and 0.8 < pull.std() < 1.2
    slope = np.polyfit(delta.ravel(), counts / (M / 512) - 1.0, 1)[0]
    assert abs(slope - 1.0) < 0.05, slope                                   # point density proportional to 1 + delta


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
