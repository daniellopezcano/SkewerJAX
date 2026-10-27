"""a(x) along a photon path (``core.Cosmology.solve_a_of_x``): the diffrax solution against the closed forms of
radiation, matter and Lambda domination, forward and backward, at the default and at tight tolerances; a realistic
background is monotonic and bracketed; densities must sum to 1 (review notes, Eq. dadx_general, Fig. a_of_x;
formerly ``scripts/validation/check_cosmology.py``, same tolerances)."""

import jax.numpy as jnp
import numpy as np
import pytest

from skewerjax.core import Cosmology, phys

H = phys.H0_over_c  # [1/Mpc]
PURE = Cosmology(
    H0=jnp.full(3, phys.H0),
    Omega_r_0=jnp.array([1.0, 0.0, 0.0]),
    Omega_m_0=jnp.array([0.0, 1.0, 0.0]),
    Omega_k_0=jnp.zeros(3),
    Omega_lambda_0=jnp.array([0.0, 0.0, 1.0]),
)
EXACT = {
    "radiation": lambda x: 1.0 + H * x,
    "matter": lambda x: (1.0 + 0.5 * H * x) ** 2,
    "Lambda": lambda x: 1.0 / (1.0 - H * x),
}
X_FORWARD = np.linspace(0.0, 0.8 / H, 1000)  # towards the future (a grows)
X_BACKWARD = np.linspace(0.0, -0.8 / H, 1000)  # towards the past


@pytest.mark.parametrize("tolerances", ["default", "tight"])
def test_pure_backgrounds_match_the_closed_forms(tolerances):
    kw, tol = ({}, 2e-3) if tolerances == "default" else (dict(rtol=1e-10, atol=1e-12), 1e-8)
    for x in (X_FORWARD, X_BACKWARD):
        a = np.asarray(PURE.solve_a_of_x(jnp.ones(3), jnp.asarray(x), **kw))
        for i, name in enumerate(EXACT):
            assert np.max(np.abs(a[i] / EXACT[name](x) - 1)) <= tol, (name, x[-1], tolerances)


def test_planck_background_is_monotonic_and_bracketed():
    planck = Cosmology(H0=phys.H0, Omega_r_0=9.1436e-5, Omega_m_0=0.3089, Omega_k_0=0.0, Omega_lambda_0=0.6911)
    a = np.asarray(planck.solve_a_of_x(jnp.ones(1), jnp.asarray(X_BACKWARD)))[0]
    assert np.sum(np.diff(a) > 0) == 0 and np.sum(a < EXACT["matter"](X_BACKWARD) - 1e-6) == 0


def test_densities_must_sum_to_one():
    with pytest.raises(ValueError):
        Cosmology(H0=phys.H0, Omega_r_0=0.0, Omega_m_0=0.5, Omega_k_0=0.0, Omega_lambda_0=0.4)
