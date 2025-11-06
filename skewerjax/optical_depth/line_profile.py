# skewerjax/optical_depth/line_profiles.py
from __future__ import annotations
from typing import Optional

import jax.numpy as jnp
from jax import jit

import numpy as np
from scipy.special import wofz

from skewerjax.constants.physical import phys

def Voigt_np(x: np.ndarray, sigma: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """
    Voigt function via Faddeeva W(z):
      V(x; σ, γ) = Re[W(z)] / (√(2π) σ), with z = (x + iγ) / (√2 σ).

    Broadcasting behavior:
      - x: shape (N,) or broadcastable
      - sigma, gamma: scalars, shape (N,), or shape (P,) for a parameter grid
      - If x is (N,) and sigma/gamma are (P,), result has shape (P, N).
    """
    from scipy.special import wofz

    x = np.asarray(x)
    sigma = np.asarray(sigma, dtype=x.dtype)
    gamma = np.asarray(gamma, dtype=x.dtype)

    # Avoid σ = 0 → division by zero; keeps correct limiting behavior numerically.
    sigma = np.maximum(sigma, np.finfo(x.dtype).tiny)

    # Try standard broadcasting first
    try:
        x_b, sigma_b, gamma_b = np.broadcast_arrays(x, sigma, gamma)
    except ValueError:
        # Common case: x=(N,), sigma=(P,), gamma=(P,) → produce (P,N)
        if x.ndim == 1 and sigma.ndim == 1 and gamma.ndim == 1:
            x_b = x[None, :]           # (1, N)
            sigma_b = sigma[:, None]   # (P, 1)
            gamma_b = gamma[:, None]   # (P, 1)
        else:
            raise

    z = (x_b + 1j * gamma_b) / (np.sqrt(2.0) * sigma_b)
    return np.real(wofz(z)) / (np.sqrt(2.0 * np.pi) * sigma_b)

@jit
def Lorentzian(x: jnp.ndarray, gamma: jnp.ndarray) -> jnp.ndarray:
    """
    Lorentz function L(x; γ) = (1/π) * γ / (x^2 + γ^2), normalized: ∫ L dx = 1.
    Safe for γ→0 via a tiny floor to avoid 0/0 at x=0.
    """
    # Avoid the 0/0 at (x=0, gamma=0)
    gsafe = jnp.maximum(gamma, jnp.finfo(x.dtype).tiny)
    return (1.0 / jnp.pi) * (gsafe / (x * x + gsafe * gsafe))

@jit
def Gaussian(x: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
    """
    Gauss function G(x; σ) = (1/√(2π)σ) * exp[-x^2/(2σ^2)], normalized: ∫ G dx = 1.
    """
    # Avoid σ = 0 → division by zero
    sigma = jnp.maximum(sigma, jnp.finfo(x.dtype).tiny)
    z = x / sigma
    return jnp.exp(-0.5 * z * z) / (jnp.sqrt(2.0 * jnp.pi) * sigma)

@jit
def Voigt(x: jnp.ndarray, sigma: jnp.ndarray, gamma: jnp.ndarray) -> jnp.ndarray:
    """
    V(x; σ, γ) ≈ (G * L)(x) via *linear* convolution implemented with FFT.

    Broadcasting rules:
      - x: shape (N,)
      - sigma, gamma: scalars OR arrays broadcastable to shape (P,)
      - Result shape: (P, N) for parameter vectors; (N,) for scalars.

    Notes:
      * Uses zero-padding to length 2N-1 to realize linear conv; then 'same' crop.
      * Multiplies by dx so that ∫V dx ≈ 1 if the domain is sufficiently wide.
    """
    # grid stats (assume uniform)
    N = x.shape[-1]
    dx = (x[-1] - x[0]) / (N - 1)

    # Broadcast parameters to a common batch shape S (possibly empty)
    # We then add a trailing axis to match x.
    sigma_b = jnp.expand_dims(sigma, -1)  # (..., 1)
    gamma_b = jnp.expand_dims(gamma, -1)  # (..., 1)

    # Broadcast x to (..., N) implicitly via arithmetic
    # Build profiles with shape (..., N)
    g = Gaussian(x, sigma_b)      # (..., N)
    l = Lorentzian(x, gamma_b)    # (..., N)

    # --- Linear convolution via FFT along the last axis ---
    n_lin = 2 * N - 1
    # next power-of-two for speed (static given N, OK for JIT)
    n_fft = 1 << (n_lin - 1).bit_length()

    pad_specs = [(0, 0)] * g.ndim
    pad_specs[-1] = (0, n_fft - N)

    Gf = jnp.fft.rfft(jnp.pad(g, pad_specs))
    Lf = jnp.fft.rfft(jnp.pad(l, pad_specs))
    Vf = Gf * Lf
    v_full_padded = jnp.fft.irfft(Vf, n_fft)  # (..., n_fft)

    # Take the first n_lin samples (linear conv), then crop to 'same' length N
    v_full = v_full_padded[..., :n_lin]       # (..., 2N-1)
    start = (n_lin - N) // 2
    v_same = v_full[..., start:start + N]     # (..., N)

    # Riemann-sum factor to approximate the integral
    return v_same * dx

@jit
def _K_TG(x: jnp.ndarray) -> jnp.ndarray:
    """
    K(x) from Tepper-García (2006), eq. (24), implemented stably.
    Uses a small-x series to avoid cancellation/div-by-zero.
    """
    x = jnp.asarray(x)
    t = x * x
    eps = jnp.finfo(x.dtype).eps

    # general form (with safe denominators)
    t_safe = jnp.maximum(t, eps)
    term1 = (4.0 * t + 3.0) * (t + 1.0) * jnp.exp(-t)
    term2 = ((2.0 * t + 3.0) / t_safe) * jnp.sinh(t)
    k_general = (term1 - term2) / (2.0 * t_safe)

    # small-x series: K(x) = 1 - (7/6) x^2 + O(x^4)
    k_series = 1.0 - (7.0 / 6.0) * t

    return jnp.where(t < 1e-6, k_series, k_general)

@jit
def Voigt_TG(a: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    """Tepper-García first-order approximation to the Voigt–Hjerting function H(a,x) ≈ e^{-x^2}[1 - a/(2√π) K(x)]."""
    a = jnp.asarray(a); x = jnp.asarray(x)
    return jnp.exp(-x * x) * (1.0 - a * _K_TG(x) / (2.0 * jnp.sqrt(jnp.pi)))

@jit
def Voigt_TG_profile(x: jnp.ndarray, sigma: jnp.ndarray, gamma: jnp.ndarray) -> jnp.ndarray:
    """
    Convenience wrapper that maps (x, σ, γ) → V_TG(x; σ, γ).

    TG gives H(a, u) with u = x/(√2 σ) and a = γ/(√2 σ); convert back via:
      V_TG(x; σ, γ) = H(a, u) / (√(2π) σ)

    Supports the same broadcasting and returns (P, N) for vector σ, γ.
    """
    sqrt2 = jnp.sqrt(2.0)

    x = jnp.asarray(x)
    sigma = jnp.asarray(sigma)
    gamma = jnp.asarray(gamma)

    sigma = jnp.maximum(sigma, jnp.finfo(x.dtype).tiny)

    # Shape to (..., N)
    u = x / (sqrt2 * jnp.expand_dims(sigma, -1))     # (..., N)
    a = gamma / (sqrt2 * sigma)                      # (...,)

    # Your TG core expects H(a, u) with broadcasting
    H = Voigt_TG(jnp.expand_dims(a, -1), u)          # (..., N)

    return H / (jnp.sqrt(2.0 * jnp.pi) * jnp.expand_dims(sigma, -1))


# ---------------------------------------------------------------------
# Helpers for physical widths in frequency space (Hz)
#   - γ_j(x) = [A_j + 2/τ_coll(x)] / (4π)
#   - σ_j(x) = (ν_j / (√2 c)) * b_eff(x)
#     with b_eff^2 = 2kT/m + b_turb^2
# ---------------------------------------------------------------------

@jit
def gamma_from_rates(A_s1: jnp.ndarray, tau_coll_s: Optional[jnp.ndarray]) -> jnp.ndarray:
    """
    Compute Lorentzian HWHM γ [Hz] from radiative+collisional rates.
    If tau_coll_s is None, uses pure radiative damping.
    """
    A_s1 = jnp.asarray(A_s1)
    if tau_coll_s is None:
        return A_s1 / (4.0 * jnp.pi)
    tau_coll_s = jnp.asarray(tau_coll_s)
    return (A_s1 + 2.0 / tau_coll_s) / (4.0 * jnp.pi)


@jit
def sigma_from_T_b(
    nu0_hz: jnp.ndarray,
    *,
    T_K: Optional[jnp.ndarray] = None,
    mass_amu: Optional[jnp.ndarray] = None,
    b_ms: Optional[jnp.ndarray] = None,        # effective Doppler parameter (includes turbulence) [m/s]
    b_turb_ms: Optional[jnp.ndarray] = None,    # microturbulent velocity [m/s]
) -> jnp.ndarray:
    """
    Return Gaussian σ in frequency units [Hz] for the line center ν0.

    Provide either:
      - b_ms (effective Doppler parameter, already includes turbulence), or
      - T_K and mass_amu (to compute thermal b_th), with optional b_turb_ms.
    """
    nu0_hz = jnp.asarray(nu0_hz)

    if b_ms is None:
        if (T_K is None) or (mass_amu is None):
            raise ValueError("Provide either b_ms, or T_K and mass_amu.")
        m = jnp.asarray(mass_amu) * phys.AMU
        b_th = jnp.sqrt(2.0 * phys.k_B * jnp.asarray(T_K) / m)  # [m/s]
        if b_turb_ms is None:
            b_eff = b_th
        else:
            b_eff = jnp.sqrt(b_th * b_th + jnp.asarray(b_turb_ms) ** 2)
    else:
        b_eff = jnp.asarray(b_ms)

    # σ_ν = (ν0 / (√2 c)) * b_eff
    return (nu0_hz / (jnp.sqrt(2.0) * phys.c)) * b_eff


# ---------------------------------------------------------------------
# Physical profiles in λ per your notes:
#   x_phys(λ; ν_j) = (c/λ) - ν_j   [Hz]
#   φ_j^(N)(λ; x)  = L( x_phys(λ;ν_j) ; γ_j(x) )
#   φ_j^(T)(λ; x)  = G( x_phys(λ;ν_j) ; σ_j(x) )
#   φ_j^(V)(λ; x)  = V( x_phys(λ;ν_j) ; σ_j(x), γ_j(x) )
# ---------------------------------------------------------------------

@jit
def x_phys_from_lambda(lambda_m: jnp.ndarray, nu0_hz: jnp.ndarray) -> jnp.ndarray:
    """x_phys(λ; ν0) = c/λ - ν0  [Hz].  Broadcasts over λ and ν0."""
    lambda_m = jnp.asarray(lambda_m)
    nu0_hz = jnp.asarray(nu0_hz)
    return phys.c / lambda_m - nu0_hz


@jit
def phi_N_lambda(
    lambda_m: jnp.ndarray,
    nu0_hz: jnp.ndarray,
    *,
    A_s1: jnp.ndarray,
    tau_coll_s: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Natural/collisional broadening φ_j^(N)(λ; x) = L( c/λ - ν0 ; γ )."""
    x = x_phys_from_lambda(lambda_m, nu0_hz)
    gamma = gamma_from_rates(A_s1, tau_coll_s)
    return Lorentzian(x, gamma)


@jit
def phi_T_lambda(
    lambda_m: jnp.ndarray,
    nu0_hz: jnp.ndarray,
    *,
    T_K: Optional[jnp.ndarray] = None,
    mass_amu: Optional[jnp.ndarray] = None,
    b_ms: Optional[jnp.ndarray] = None,
    b_turb_ms: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Thermal/turbulent Doppler φ_j^(T)(λ; x) = G( c/λ - ν0 ; σ )."""
    x = x_phys_from_lambda(lambda_m, nu0_hz)
    sigma = sigma_from_T_b(
        nu0_hz, T_K=T_K, mass_amu=mass_amu, b_ms=b_ms, b_turb_ms=b_turb_ms
    )
    return Gaussian(x, sigma)


@jit
def phi_V_lambda(
    lambda_m: jnp.ndarray,
    nu0_hz: jnp.ndarray,
    *,
    A_s1: jnp.ndarray,
    tau_coll_s: Optional[jnp.ndarray] = None,
    T_K: Optional[jnp.ndarray] = None,
    mass_amu: Optional[jnp.ndarray] = None,
    b_ms: Optional[jnp.ndarray] = None,
    b_turb_ms: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Voigt φ_j^(V)(λ; x) = V( c/λ - ν0 ; σ(·), γ(·) )."""
    x = x_phys_from_lambda(lambda_m, nu0_hz)
    gamma = gamma_from_rates(A_s1, tau_coll_s)
    sigma = sigma_from_T_b(
        nu0_hz, T_K=T_K, mass_amu=mass_amu, b_ms=b_ms, b_turb_ms=b_turb_ms
    )
    return Voigt(x, sigma, gamma)




























