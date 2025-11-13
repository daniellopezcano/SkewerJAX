# skewerjax/optical_depth/line_profiles.py
from __future__ import annotations
from typing import Optional

import jax
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


def Lorentzian(x: jnp.ndarray, gamma: jnp.ndarray) -> jnp.ndarray:
    """L(x;γ) = (1/π) * γ / (x² + γ²), normalized."""
    g = jnp.maximum(gamma, jnp.finfo(x.dtype).tiny)
    return (1.0 / jnp.pi) * g / (x * x + g * g)


def Gaussian(x: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
    """G(x;σ) = (1/√(2π)σ) * exp[-x²/(2σ²)], normalized."""
    s = jnp.maximum(sigma, jnp.finfo(x.dtype).tiny)
    z = x / s
    return jnp.exp(-0.5 * z * z) / (jnp.sqrt(2.0 * jnp.pi) * s)

def _next_pow2(n: int) -> int:
    return 1 << (int(n) - 1).bit_length()

def Voigt_JAX_convolution(
    x: jnp.ndarray,
    sigma: jnp.ndarray,
    gamma: jnp.ndarray,
    *,
    oversample: int = 1,            # ≥1; e.g., 4–8 for higher fidelity
    work_dtype = jnp.float64        # internal compute dtype
) -> jnp.ndarray:
    """
    Voigt(x; σ, γ) via discrete convolution at internally increased precision.

    Strategy:
      - Build a finer grid with 'oversample'× more points.
      - Evaluate Gaussian & Lorentzian on the fine grid.
      - Convolve on the fine grid (direct or FFT), scale by fine dx (unit area).
      - Downsample back to the original grid by striding.

    Notes:
      - Assumes x is uniformly spaced.
      - Returns an array with the same shape and dtype as x.
    """
    # --- original grid info ---
    n = x.shape[0]
    x0, x1 = x[0], x[-1]
    dx = (x1 - x0) / (n - 1)

    # --- build oversampled grid (matching endpoints exactly) ---
    factor = int(oversample)
    n_hi = (n - 1) * factor + 1
    x_hi = jnp.linspace(x0, x1, n_hi, dtype=work_dtype)
    dx_hi = dx / factor

    # --- evaluate kernels on fine grid in higher precision ---
    g_hi = Gaussian(x_hi, sigma).astype(work_dtype)
    l_hi = Lorentzian(x_hi, gamma).astype(work_dtype)

    # Linear convolution via FFT with zero-padding to avoid circular wrap
    n_full = n_hi + n_hi - 1
    n_fft = _next_pow2(n_full)
    G = jnp.fft.rfft(g_hi, n_fft)
    L = jnp.fft.rfft(l_hi, n_fft)
    full = jnp.fft.irfft(G * L, n_fft)[:n_full]
    # Extract 'same' segment (centered) to match jnp.convolve(..., mode='same')
    start = (n_hi - 1) // 2
    conv_hi = full[start:start + n_hi]

    # --- preserve area in discrete sum ---
    conv_hi = conv_hi * dx_hi

    # --- downsample back to original grid positions (exact index alignment) ---
    conv_lo = conv_hi[::factor]

    # Safety: in case of any rounding mismatch, slice to n
    conv_lo = conv_lo[:n]

    # Cast back to input dtype
    return conv_lo.astype(x.dtype)
























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




























