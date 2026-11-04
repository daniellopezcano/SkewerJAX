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


def test_sound_horizon_at_the_drag_epoch():
    """``sound_horizon_drag``: the fit of Aubourg et al. 2015 (eq. 16), r_d = 55.154 exp[-72.3 (omega_nu + 0.0006)^2] /
    (omega_cb^0.25351 omega_b^0.12807) Mpc; for the cosmology of IllustrisTNG (Planck 2015) r_d = 147.30 Mpc; vectorised,
    and smaller for more matter, more baryons or massive neutrinos."""
    from skewerjax.core import sound_horizon_drag
    from skewerjax.simulations import TNG_COSMOLOGY as C

    h2 = C["h"] ** 2
    wb, wcb = C["Omega_b"] * h2, C["Omega_m"] * h2
    rd = float(sound_horizon_drag(wb, wcb))
    assert rd == pytest.approx(55.154 * np.exp(-72.3 * 0.0006**2) / (wcb**0.25351 * wb**0.12807), rel=1e-14)
    assert rd == pytest.approx(147.30, abs=0.01)
    r = np.asarray(sound_horizon_drag(np.array([wb, wb, 1.1 * wb]), np.array([wcb, 1.1 * wcb, wcb])))
    assert r[1] < r[0] and r[2] < r[0] and r[0] == pytest.approx(rd)
    assert float(sound_horizon_drag(wb, wcb, 0.0006)) < rd, "massive neutrinos lower r_d"
