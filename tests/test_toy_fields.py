"""grf_cells: n = n_bar (1 + delta) with no transform (grid generators reproduce the realisation, zero-mean delta,
positivity check, one realisation shared by several species)."""
import warnings

import jax
import numpy as np

from skewerjax.core import PowerLawGRF, grf_cells


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
