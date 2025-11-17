from __future__ import annotations
from typing import Literal, Union, Any

import jax
import jax.numpy as jnp
from jax import vmap

from skewerjax.constants.physical import phys

import numpy as np
from scipy.special import wofz

from exojax.opacity.lpf.lpf import voigtone

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

def Voigt_exojax(
    x: jnp.ndarray,
    sigma: jnp.ndarray,
    gamma: jnp.ndarray,
) -> jnp.ndarray:
    """
    Voigt(x; σ, γ) using ExoJAX `voigtone`: https://secondearths.sakura.ne.jp/exojax/tutorials/voigt_function.html?highlight=voigt
    Assumes:
      - x: 1D grid, shape (N,)
      - sigma, gamma: scalars (or 0-D arrays)
    """
    tiny = jnp.finfo(x.dtype).tiny
    sigma = jnp.maximum(sigma, tiny)
    gamma = jnp.maximum(gamma, tiny)
    # voigtone is scalar in all args → vmap over x only
    return jax.vmap(voigtone, in_axes=(0, None, None))(x, sigma, gamma)


def Voigt(
    x: Union[jnp.ndarray, np.ndarray],
    sigma: Union[jnp.ndarray, np.ndarray],
    gamma: Union[jnp.ndarray, np.ndarray],
    *,
    mode: Literal["ExoJAX", "convolution", "scipy_wofz"] = "ExoJAX",
    **kwargs: Any,
):
    """
    Unified Voigt profile wrapper.
    -----
    - For autodiff and JIT, use modes "ExoJAX" or "convolution".
    - The "scipy_wofz" mode must not be used inside jax.jit / grad / vmap
      with tracer arguments; it is purely a host-side evaluation.
    """

    mode_norm = mode.lower()

    # --- JAX / ExoJAX implementation ---
    if mode_norm == "exojax":
        return Voigt_exojax(x, sigma, gamma)

    # --- JAX convolution implementation ---
    if mode_norm == "convolution":
        return Voigt_JAX_convolution(x, sigma, gamma, **kwargs)

    # --- SciPy / NumPy implementation (NO autodiff / jit) ---
    if mode_norm == "scipy_wofz":
        x_np = np.asarray(x)
        sigma_np = np.asarray(sigma)
        gamma_np = np.asarray(gamma)
        return Voigt_np(x_np, sigma_np, gamma_np)

    raise ValueError(
        f"Unknown Voigt mode '{mode}'. "
        "Supported modes: 'ExoJAX', 'convolution', 'scipy_wofz'."
    )

def compute_sigma_gamma(
    T_K: jnp.ndarray,                 # (N_sp, N_sk, N_ppsk) — UNITS: [K]   — gas temperature
    b_turb_m_per_s: jnp.ndarray,      # (N_sp, N_sk, N_ppsk) — UNITS: [m/s] — microturbulent velocity
    tau_coll_ps: jnp.ndarray,         # (N_sp, N_sk, N_ppsk) — UNITS: [ps]  — collisional dephasing time
    lambda0_A: jnp.ndarray,           # (N_sp,)              — UNITS: [Å]   — rest-frame wavelength of each line
    A21_THz: jnp.ndarray,             # (N_sp,)              — UNITS: [THz] — Einstein A21 coefficient
    mass_amu: jnp.ndarray,            # (N_sp,)              — UNITS: [amu] — absorber mass in atomic mass units
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute Gaussian (σ_j) and Lorentzian (γ_j) widths for each line/species.

    Returns
    -------
    sigma_THz : array, shape (N_sp, N_sk, N_ppsk), UNITS: [THz]
    gamma_THz : array, shape (N_sp, N_sk, N_ppsk), UNITS: [THz]
    """

    # Broadcast line parameters along (skewer, point) dimensions
    A21_THz = A21_THz[:, None, None]              # (N_sp, 1, 1) [A21_THz]
    lambda0_A = lambda0_A[:, None, None]          # (N_sp, 1, 1) [m]
    mass_amu = mass_amu[:, None, None]            # (N_sp, 1, 1) [kg]
    
    gamma_THz = (A21_THz + 2.0 / tau_coll_ps) / (4.0 * jnp.pi)  # UNITS: [THz]
    v_th_m_per_s = jnp.sqrt(2.0 * phys.kB_amu * T_K / mass_amu) # UNITS: [m/s]
    v_eff_A_THz = (v_th_m_per_s + b_turb_m_per_s)*1e-2          # UNITS: [Å THz]
    sigma_THz = v_eff_A_THz / (2.0 * lambda0_A)                 # UNITS: [THz]
    
    return sigma_THz, gamma_THz


def line_profiles(
    lambdas: jnp.ndarray,        # (N_lambda,)          — UNITS: [Angs] — wavelength range
    lambda0_A: jnp.ndarray,      # (N_sp,)              — UNITS: [Angs] — rest-frame wavelength of each line
    A21_THz: jnp.ndarray,        # (N_sp,)              — UNITS: [THz]  — Einstein A21 coefficient
    mass_amu: jnp.ndarray,       # (N_sp,)              — UNITS: [amu]  — absorber mass in atomic mass units
    T_K: jnp.ndarray,            # (N_sp, N_sk, N_ppsk) — UNITS: [K]    — gas temperature
    b_turb_m_per_s: jnp.ndarray, # (N_sp, N_sk, N_ppsk) — UNITS: [m/s]  — microturbulent velocity
    tau_coll_ps: jnp.ndarray,    # (N_sp, N_sk, N_ppsk) — UNITS: [ps]   — collisional dephasing time
    mode: str = "ExoJAX",        # or "convolution"
):
    """
    Returns line profiles:
        phi: shape (N_sp, N_sk, N_ppsk, N_lambda) — UNITS: []
    """

    # --- σ_j(x) and γ_j(x) on full (sp, sk, ppsk) grid ---
    sigma_THz, gamma_THz = compute_sigma_gamma(T_K, b_turb_m_per_s, tau_coll_ps, lambda0_A, A21_THz, mass_amu)

    # --- x_j(λ) for each species and lambda ---
    nu_THz      = phys.c_A_THz / lambdas[None, :]   # shape: (1, N_lambda)    — UNITS: THz
    nu0_THz     = phys.c_A_THz / lambda0_A[:, None] # shape: (N_sp, 1)        — UNITS: THz
    nu_grid_THz = nu_THz - nu0_THz                  # shape: (N_sp, N_lambda) — UNITS: THz. x_j(λ) for each line species j. Argument to Voigt for each line

    def voigt_one_species(x_j, sigma_j, gamma_j):
        # sigma_j, gamma_j: (N_sk, N_ppsk)

        def voigt_cell(s, g):
            return Voigt(x_j, s, g, mode=mode)  # (N_lambda,)

        # vmap over ppsk inside skewer
        voigt_over_pp = vmap(voigt_cell, in_axes=(0, 0))          # (N_ppsk, N_lambda)

        # vmap over skewer
        voigt_over_sk = vmap(voigt_over_pp, in_axes=(0, 0))       # (N_sk, N_ppsk, N_lambda)

        return voigt_over_sk(sigma_j, gamma_j)

    # vmap over species j
    phi = vmap(voigt_one_species, in_axes=(0, 0, 0))(nu_grid_THz, sigma_THz, gamma_THz) # shape: (N_sp, N_sk, N_ppsk, N_lambda)

    return phi
