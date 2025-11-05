# skewerjax/optical_depth/line_profiles.py
from __future__ import annotations

from typing import Optional

import jax.numpy as jnp
from jax import jit
# from jax.scipy.special import wofz

from skewerjax.constants.physical import phys


# ----------------------------- #
# Primitive, normalized profiles
# ----------------------------- #

@jit
def gaussian(x: jnp.ndarray, sigma: jnp.ndarray, mu: jnp.ndarray = 0.0) -> jnp.ndarray:
    """
    Normalized Gaussian in x:
        G(x; mu, sigma) = 1/(sqrt(2π) σ) * exp(-(x - mu)^2 / (2 σ^2))

    Args
    ----
    x     : variable (any shape, broadcasts)
    sigma : standard deviation (>0), same units as x
    mu    : center, same units as x

    Returns
    -------
    G(x) with ∫ G dx = 1
    """
    sigma = jnp.asarray(sigma)
    z = (x - mu) / sigma
    return jnp.exp(-0.5 * z * z) / (jnp.sqrt(2.0 * jnp.pi) * sigma)


@jit
def lorentzian(x: jnp.ndarray, gamma: jnp.ndarray, mu: jnp.ndarray = 0.0) -> jnp.ndarray:
    """
    Normalized Lorentzian (Cauchy) in x with HWHM=gamma:
        L(x; mu, gamma) = (1/π) * γ / [ (x - mu)^2 + γ^2 ]

    Args
    ----
    x     : variable (any shape, broadcasts)
    gamma : half-width at half-maximum (>0), same units as x
    mu    : center, same units as x

    Returns
    -------
    L(x) with ∫ L dx = 1
    """
    gamma = jnp.asarray(gamma)
    return (1.0 / jnp.pi) * (gamma / ((x - mu) ** 2 + gamma ** 2))


# ------------------------------------------------ #
# Physical, frequency-domain line profiles (φ_ν(ν))
# ------------------------------------------------ #

@jit
def natural_profile(nu: jnp.ndarray, nu0: jnp.ndarray, Gamma: jnp.ndarray) -> jnp.ndarray:
    """
    Natural (radiative) broadening profile in frequency units (Hz):
        φ_L(ν) = (1/π) * γ / [ (ν - ν0)^2 + γ^2 ],  with  γ = Γ / (4π)

    Args
    ----
    nu    : frequency array [Hz]
    nu0   : line center frequency [Hz]
    Gamma : Einstein A total damping constant [s^-1]

    Returns
    -------
    φ_L(ν) normalized so ∫ φ_L dν = 1
    """
    gamma = jnp.asarray(Gamma) / (4.0 * jnp.pi)  # HWHM in Hz
    return lorentzian(nu, gamma, mu=nu0)


@jit
def doppler_b_from_T(T: jnp.ndarray, mass_amu: jnp.ndarray) -> jnp.ndarray:
    """
    Doppler parameter b = sqrt(2 k_B T / m) [m/s], with m = mass_amu * AMU.
    """
    m = jnp.asarray(mass_amu) * phys.AMU
    return jnp.sqrt(2.0 * phys.k_B * jnp.asarray(T) / m)


@jit
def thermal_profile(
    nu: jnp.ndarray,
    nu0: jnp.ndarray,
    *,
    b_ms: Optional[jnp.ndarray] = None,
    T_K: Optional[jnp.ndarray] = None,
    mass_amu: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """
    Thermal (Doppler) profile in frequency units (Hz), normalized:
        φ_G(ν) = 1 / (Δν_D √π) * exp(-[(ν - ν0)/Δν_D]^2)
      where Δν_D = (b / c) * ν0 and b = sqrt(2 k_B T / m).

    Provide either:
      - b_ms (the Doppler parameter b in m/s), or
      - T_K and mass_amu.

    Args
    ----
    nu, nu0 : frequency and line center [Hz]
    b_ms    : Doppler parameter [m/s]
    T_K     : temperature [K]
    mass_amu: absorber mass [amu]

    Returns
    -------
    φ_G(ν) with ∫ φ_G dν = 1
    """
    if b_ms is None:
        if (T_K is None) or (mass_amu is None):
            raise ValueError("Provide either b_ms, or T_K and mass_amu.")
        b_ms = doppler_b_from_T(jnp.asarray(T_K), jnp.asarray(mass_amu))

    # Doppler width in Hz
    delta_nu_D = (jnp.asarray(b_ms) / phys.c) * jnp.asarray(nu0)
    u = (nu - nu0) / delta_nu_D
    return jnp.exp(-(u * u)) / (jnp.sqrt(jnp.pi) * delta_nu_D)


@jit
def voigt_profile(
    nu: jnp.ndarray,
    nu0: jnp.ndarray,
    Gamma: jnp.ndarray,
    *,
    b_ms: Optional[jnp.ndarray] = None,
    T_K: Optional[jnp.ndarray] = None,
    mass_amu: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """
    Voigt profile (convolution of Lorentzian+Gaussian) in frequency units (Hz), normalized:
        φ_V(ν) = 1 / (Δν_D √π) * Re[ wofz( u + i a ) ]
      with u = (ν - ν0)/Δν_D and a = Γ / (4π Δν_D).

    Args
    ----
    nu, nu0 : frequency and line center [Hz]
    Gamma   : Einstein A total damping constant [s^-1]
    b_ms    : Doppler parameter [m/s], or infer from T_K & mass_amu
    T_K     : temperature [K]
    mass_amu: absorber mass [amu]

    Returns
    -------
    φ_V(ν) with ∫ φ_V dν = 1
    """
    if b_ms is None:
        if (T_K is None) or (mass_amu is None):
            raise ValueError("Provide either b_ms, or T_K and mass_amu.")
        b_ms = doppler_b_from_T(jnp.asarray(T_K), jnp.asarray(mass_amu))

    delta_nu_D = (jnp.asarray(b_ms) / phys.c) * jnp.asarray(nu0)
    u = (nu - nu0) / delta_nu_D
    a = jnp.asarray(Gamma) / (4.0 * jnp.pi * delta_nu_D)

    z = u + 1j * a  # complex argument
    H = jnp.real(wofz(z))  # Voigt function H(a,u)
    return H / (jnp.sqrt(jnp.pi) * delta_nu_D)
