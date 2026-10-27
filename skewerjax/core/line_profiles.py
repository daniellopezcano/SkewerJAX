"""Line profiles: the widths and the normalised profile phi(nu) of every transition for gas elements (review notes,
Eq. line_profile), on the Voigt functions of ``core.voigt``.

* ``voigt_widths_GHz``: the Gaussian sigma (thermal + turbulent Doppler width) and the Lorentzian gamma (natural +
  collisional) of every (line, element);
* ``voigt_width_kms``: a Voigt width as an equivalent Doppler parameter (the resolution check);
* ``voigt_line_profiles``: phi = V(nu; sigma, gamma) [ns], unit area in nu (Voigt modes ``PROFILE_MODES``: ExoJAX, Harris);
* ``line_profiles``: phi_ij(v) of every line for gas elements given by arrays (temperature, velocity along the photon,
  scale factor), on the line-centred velocity axis (``core.spectral.frequency_offset_GHz``).

``core.optical_depth`` sums these profiles over the pieces of the path.

``skewerjax.core`` re-exports the four functions, so the package attribute ``skewerjax.core.line_profiles`` is the
function ``line_profiles``, not this module: import from the module by its full path
(``from skewerjax.core.line_profiles import voigt_widths_GHz``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from skewerjax.core.constants import phys
from skewerjax.core.spectral import frequency_offset_GHz
from skewerjax.core.voigt import log10_voigt, voigt_harris


def voigt_widths_GHz(temperature, turbulence, collisional, lambda0_A, A21_GHz, mass_amu):
    """Gaussian sigma and Lorentzian gamma [GHz] per (line, point).

    b_th = sqrt(2 k_B T / m) [m/s], b = b_th + b_turb (a linear sum, as implemented so far; the notes add them in
    quadrature), sigma = b / (sqrt(2) lambda), gamma = (A_21 + A_coll) / (4 pi).
    temperature [K], turbulence [m/s], collisional [GHz] : (N_lines, N_points); the per-line arrays (N_lines,).
    """
    gamma = (A21_GHz[:, None] + collisional) / (4.0 * jnp.pi)
    v_th_ms = jnp.sqrt(2.0 * phys.kB * temperature / mass_amu[:, None])
    sigma = (v_th_ms + turbulence) * 10.0 / (jnp.sqrt(2.0) * lambda0_A[:, None])
    return sigma, gamma


def voigt_width_kms(sigma_GHz, gamma_GHz, lambda0_A):
    """Width of a Voigt profile as an equivalent Doppler parameter [km/s].

    f_G = 2 sqrt(ln 2) sqrt(2) sigma lambda, f_L = 2 gamma lambda, f_V = 0.5346 f_L + sqrt(0.2166 f_L^2 + f_G^2)
    (better than 0.02 %); returns f_V / (2 sqrt(ln 2)): the thermal b for a Doppler line, finite for T -> 0.
    sigma_GHz, gamma_GHz : (N_lines, ...); lambda0_A : (N_lines,) [A].
    """
    lam = jnp.reshape(jnp.asarray(lambda0_A), (-1,) + (1,) * (jnp.ndim(sigma_GHz) - 1))
    to_kms = lam * 1e-4  # GHz x A -> km/s
    f_G = 2.0 * jnp.sqrt(jnp.log(2.0)) * jnp.sqrt(2.0) * sigma_GHz * to_kms
    f_L = 2.0 * gamma_GHz * to_kms
    f_V = 0.5346 * f_L + jnp.sqrt(0.2166 * f_L**2 + f_G**2)
    return f_V / (2.0 * jnp.sqrt(jnp.log(2.0)))


PROFILE_MODES = ("ExoJAX", "harris")  # the Voigt modes of the line profiles and of tau


def check_profile_mode(voigt: str) -> str:
    """The Voigt mode of the line profiles (case-insensitive), or ``ValueError`` naming ``PROFILE_MODES``. The FFT
    convolution and SciPy's ``wofz`` are standalone functions of ``core.voigt`` (``voigt_convolution``,
    ``voigt_scipy``): the convolution needs an increasing uniform grid, which the frequency offsets of tau are not, and
    ``wofz`` cannot run inside the jitted kernel."""
    m = str(voigt).lower()
    if m not in tuple(x.lower() for x in PROFILE_MODES):
        raise ValueError(
            f"voigt must be one of {PROFILE_MODES} (case-insensitive), got {voigt!r}; the FFT convolution and SciPy's "
            "wofz are standalone functions of core.voigt (voigt_convolution, voigt_scipy)."
        )
    return m


def voigt_line_profiles(nu_GHz: ArrayLike, sigma_GHz: ArrayLike, gamma_GHz: ArrayLike, voigt: str = "ExoJAX") -> jnp.ndarray:
    """phi = V(nu; sigma, gamma) [ns] (Eq. line_profile), unit area in nu. nu (N, N_v); sigma, gamma (N,).

    ``voigt``: a mode of ``PROFILE_MODES``: "ExoJAX" (default, vmapped over the N elements) or "harris" (the first-order
    expansion in the damping parameter, ``voigt_harris``, on the whole (N, N_v) batch, so that its exact fallback is
    skipped when every element has a <= ``HARRIS_A_MAX``). Any other mode raises ``ValueError``.
    """
    if check_profile_mode(voigt) == "harris":
        return voigt_harris(jnp.asarray(nu_GHz), jnp.asarray(sigma_GHz)[:, None], jnp.asarray(gamma_GHz)[:, None])
    log10_phi = jax.vmap(lambda x, s, g: log10_voigt(x, s, g, mode=voigt), in_axes=(0, 0, 0))(nu_GHz, sigma_GHz, gamma_GHz)
    return 10.0**log10_phi


def line_profiles(
    velocities_kms: ArrayLike,
    line_bundle,
    *,
    temperature,
    los_velocity=0.0,
    scale_factor=1.0,
    a_dagger=1.0,
    turbulence=0.0,
    collisional=0.0,
    voigt: str = "ExoJAX",
    velocity_period: float | None = None,
    detuning: str = "exact",
) -> jnp.ndarray:
    """phi_ij(v) [ns] of every line j for gas elements i, on the line-centred velocities (N_v,) [km/s].

    The elements are given by arrays broadcastable to (N,): temperature [K], los_velocity [km/s, along the photon],
    scale_factor a, turbulence [m/s], collisional [GHz]; a_dagger is the reference scale factor of v_dagger. Every line
    takes the same elements (e.g. the cells crossed by a path). ``velocity_period`` and ``detuning`` as in
    ``frequency_offset_GHz``. Returns (N_lines, N, N_v).
    """
    velocities_kms = jnp.asarray(velocities_kms)
    T, u, a, tb, co = jnp.broadcast_arrays(
        *(jnp.atleast_1d(jnp.asarray(x, dtype=float)) for x in (temperature, los_velocity, scale_factor, turbulence, collisional))
    )
    n_lines, n = int(line_bundle.n_lines), T.shape[0]
    rep = lambda x: jnp.broadcast_to(x[None], (n_lines, n))
    lam = jnp.asarray(line_bundle.lambda0_A)
    sigma, gamma = voigt_widths_GHz(
        rep(T), rep(tb), rep(co), lam, jnp.asarray(line_bundle.A21_GHz), jnp.asarray(line_bundle.mass_amu)
    )
    nu = frequency_offset_GHz(
        velocities_kms, lam, jnp.asarray(a_dagger, dtype=float), a, rep(u), velocity_period=velocity_period, detuning=detuning
    )
    phi = voigt_line_profiles(nu.reshape(n_lines * n, -1), sigma.reshape(-1), gamma.reshape(-1), voigt)
    return phi.reshape(n_lines, n, -1)
