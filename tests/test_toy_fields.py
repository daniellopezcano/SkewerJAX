"""GRFDensity: n = n_bar (1 + delta) with no transform (nodes reproduce the realisation, zero-mean delta, positivity check)."""
import warnings

import jax
import jax.numpy as jnp
import numpy as np

from skewerjax.core import GRFDensity, PowerLawGRF


def _grid_events(grf):
    axes = grf.x_grid()
    X, Y, Z = np.meshgrid(*axes, indexing="ij")
    return jnp.asarray(np.stack([np.ones(X.size), X.ravel(), Y.ravel(), Z.ravel()], -1))


def test_nodes_reproduce_n_bar_one_plus_delta():
    grf = PowerLawGRF(A_s=0.02, k_0=2.0, n_s=-1.25, L=(20.0,) * 3, N=(16,) * 3)
    key = jax.random.PRNGKey(4)
    d = GRFDensity(key, grf, n_bar=3e-12)
    n = np.asarray(d(_grid_events(grf))).reshape(grf.shape)
    delta = np.asarray(grf.sample(key))
    np.testing.assert_allclose(n, 3e-12 * (1.0 + delta), rtol=1e-12)
    np.testing.assert_allclose(n.mean(), 3e-12, rtol=1e-12)                 # the k = 0 mode of delta is zero
    assert float(d.min_one_plus_delta) == 1.0 + delta.min() > 0.0


def test_shared_realisation_and_negative_warning():
    grf = PowerLawGRF(A_s=4.0, k_0=2.0, n_s=-1.25, L=(20.0,) * 3, N=(16,) * 3)   # sigma ~ 1: negative somewhere
    key = jax.random.PRNGKey(0)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        a = GRFDensity(key, grf, n_bar=1.0)
    assert any("min(1 + delta)" in str(x.message) for x in w)
    b = GRFDensity(None, grf, n_bar=2.0, delta_grid=a.delta_grid)
    ev = _grid_events(grf)
    np.testing.assert_allclose(np.asarray(b(ev)) - 2.0, 2.0 * (np.asarray(a(ev)) - 1.0), rtol=1e-12, atol=1e-12)
