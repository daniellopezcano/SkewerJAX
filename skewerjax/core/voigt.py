"""The Voigt function V(x; sigma, gamma) = (G * L)(x) (notes ``sec_flux.tex``, Eq. line_profile), three implementations."""
from __future__ import annotations

from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
from exojax.opacity.lpf.lpf import voigtone
from scipy.special import wofz

VOIGT_MODES = ("ExoJAX", "convolution", "scipy_wofz")


def gaussian(x: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
    """G(x; sigma) = exp(-x^2 / 2 sigma^2) / (sqrt(2 pi) sigma), unit area."""
    s = jnp.maximum(sigma, jnp.finfo(x.dtype).tiny)
    z = x / s
    return jnp.exp(-0.5 * z * z) / (jnp.sqrt(2.0 * jnp.pi) * s)


def lorentzian(x: jnp.ndarray, gamma: jnp.ndarray) -> jnp.ndarray:
    """L(x; gamma) = gamma / (pi (x^2 + gamma^2)), unit area."""
    g = jnp.maximum(gamma, jnp.finfo(x.dtype).tiny)
    return (1.0 / jnp.pi) * g / (x * x + g * g)


def voigt_scipy(x: np.ndarray, sigma: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """Re[W(z)] / (sqrt(2 pi) sigma), z = (x + i gamma) / (sqrt(2) sigma), with ``scipy.special.wofz`` (NumPy only).

    The accuracy reference. x (N,) with sigma, gamma scalars or (N,); x (N,) with sigma, gamma (P,) gives (P, N).
    """
    sigma = np.maximum(sigma, np.finfo(x.dtype).tiny)
    try:
        x_b, sigma_b, gamma_b = np.broadcast_arrays(x, sigma, gamma)
    except ValueError:
        if x.ndim == 1 and sigma.ndim == 1 and gamma.ndim == 1:
            x_b, sigma_b, gamma_b = x[None, :], sigma[:, None], gamma[:, None]
        else:
            raise
    z = (x_b + 1j * gamma_b) / (np.sqrt(2.0) * sigma_b)
    return np.real(wofz(z)) / (np.sqrt(2.0 * np.pi) * sigma_b)


def _next_pow2(n: int) -> int:
    return 1 << (int(n) - 1).bit_length()


def voigt_convolution(x: jnp.ndarray, sigma: jnp.ndarray, gamma: jnp.ndarray, *, oversample: int = 1,
                      work_dtype=jnp.float64) -> jnp.ndarray:
    """G * L by a zero-padded FFT convolution on a uniform grid x (N,), refined ``oversample`` times.

    The kernels are evaluated in ``work_dtype`` (float64 needs ``jax_enable_x64``), convolved, scaled by the fine
    spacing (unit area) and sampled back at x. Returns (N,) in the dtype of x.
    """
    n = x.shape[0]
    x0, x1 = x[0], x[-1]
    dx = (x1 - x0) / (n - 1)
    factor = int(oversample)
    n_hi = (n - 1) * factor + 1
    x_hi = jnp.linspace(x0, x1, n_hi, dtype=work_dtype)
    g_hi = gaussian(x_hi, sigma).astype(work_dtype)
    l_hi = lorentzian(x_hi, gamma).astype(work_dtype)
    n_full = 2 * n_hi - 1
    n_fft = _next_pow2(n_full)
    full = jnp.fft.irfft(jnp.fft.rfft(g_hi, n_fft) * jnp.fft.rfft(l_hi, n_fft), n_fft)[:n_full]
    start = (n_hi - 1) // 2                                        # the 'same' part of the linear convolution
    conv_hi = full[start:start + n_hi] * (dx / factor)
    return conv_hi[::factor][:n].astype(x.dtype)


def voigt_exojax(x: jnp.ndarray, sigma: jnp.ndarray, gamma: jnp.ndarray) -> jnp.ndarray:
    """V(x) with ExoJAX's ``voigtone`` (Humlicek-type Faddeeva approximation), vmapped over x (N,); sigma, gamma scalars."""
    tiny = jnp.finfo(x.dtype).tiny
    return jax.vmap(voigtone, in_axes=(0, None, None))(x, jnp.maximum(sigma, tiny), jnp.maximum(gamma, tiny))


def log10_voigt(x, sigma, gamma, *, mode: Literal["ExoJAX", "convolution", "scipy_wofz"] = "ExoJAX", **kwargs: Any):
    """log10 V(x; sigma, gamma) with one of ``VOIGT_MODES`` (case-insensitive).

    "ExoJAX" and "convolution" are JAX (jit / vmap / grad); "scipy_wofz" is host-side NumPy and fails on tracers.
    ``kwargs`` go to ``voigt_convolution``.
    """
    m = mode.lower()
    if m == "exojax":
        V = voigt_exojax(x, sigma, gamma)
    elif m == "convolution":
        V = voigt_convolution(x, sigma, gamma, **kwargs)
    elif m == "scipy_wofz":
        V = voigt_scipy(np.asarray(x), np.asarray(sigma), np.asarray(gamma))
    else:
        raise ValueError(f"Unknown Voigt mode '{mode}'. Supported modes: {', '.join(repr(v) for v in VOIGT_MODES)}.")
    return jnp.log10(V)
