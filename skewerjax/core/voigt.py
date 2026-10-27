"""The Voigt function V(x; sigma, gamma) = (G * L)(x) (review notes, Eq. line_profile), four implementations."""

from __future__ import annotations

from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
from exojax.opacity.lpf.lpf import voigtone
from scipy.special import dawsn, wofz

VOIGT_MODES = ("ExoJAX", "convolution", "harris", "scipy_wofz")


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


def voigt_convolution(
    x: jnp.ndarray, sigma: jnp.ndarray, gamma: jnp.ndarray, *, oversample: int = 1, work_dtype=jnp.float64
) -> jnp.ndarray:
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
    start = (n_hi - 1) // 2  # the 'same' part of the linear convolution
    conv_hi = full[start : start + n_hi] * (dx / factor)
    return conv_hi[::factor][:n].astype(x.dtype)


def voigt_exojax(x: jnp.ndarray, sigma: jnp.ndarray, gamma: jnp.ndarray) -> jnp.ndarray:
    """V(x) with ExoJAX's ``voigtone`` (Humlicek-type Faddeeva approximation), vmapped over x (N,); sigma, gamma scalars."""
    tiny = jnp.finfo(x.dtype).tiny
    return jax.vmap(voigtone, in_axes=(0, None, None))(x, jnp.maximum(sigma, tiny), jnp.maximum(gamma, tiny))


# First-order expansion of H(a, u) = Re w(u + i a) in the damping parameter a (Harris 1948, ApJ 108, 112), the scheme
# of fake_spectra (fastvoigt.h, commit 2aebe44; singleabs.h ``profile``):
#     H(a, u) = exp(-u^2) + a H1(u) + O(a^2),   H1(u) = 2 u Im w(u) - 2 / sqrt(pi) = (2 / sqrt(pi)) (2 u Daw(u) - 1),
# H1 tabulated on [0, U_MAX] (N_TAB intervals) and interpolated linearly; past U_MAX the Gaussian has underflowed and
# H1 is its asymptotic series. For a > HARRIS_A_MAX the expansion is not used (fake_spectra calls Faddeeva::w there;
# here ExoJAX's ``voigtone``). The Gaussian is evaluated exactly (fake_spectra tabulates it with a Taylor correction,
# good to 1e-8).
HARRIS_U_MAX, HARRIS_N_TAB, HARRIS_A_MAX = 6.0, 1024, 1e-3
_HARRIS_U = np.arange(HARRIS_N_TAB + 2) * (HARRIS_U_MAX / HARRIS_N_TAB)
_HARRIS_H1 = (2.0 / np.sqrt(np.pi)) * (2.0 * _HARRIS_U * dawsn(_HARRIS_U) - 1.0)


def harris_H(u: jnp.ndarray, a: jnp.ndarray) -> jnp.ndarray:
    """H(a, u) to first order in a (no fallback): exp(-u^2) + a H1(u) for |u| < 6, the asymptotic
    a / (sqrt(pi) u^2) (1 + 3/(2u^2) + 15/(4u^4) + 105/(8u^6) + 945/(16u^8)) beyond. u, a broadcastable (dimensionless)."""
    au = jnp.abs(u)
    t = jnp.minimum(au, HARRIS_U_MAX) * (HARRIS_N_TAB / HARRIS_U_MAX)  # the table index, as in fastvoigt.h
    i = jnp.minimum(jnp.floor(t), HARRIS_N_TAB).astype(jnp.int32)
    tab = jnp.asarray(_HARRIS_H1, dtype=au.dtype)
    h1 = tab[i] + (t - i) * (tab[i + 1] - tab[i])  # linear interpolation, no search
    inner = jnp.exp(-au * au) + a * h1
    u2 = 1.0 / jnp.maximum(au, HARRIS_U_MAX) ** 2  # safe in the inner region (not used there)
    wing = a / jnp.sqrt(jnp.pi) * u2 * (1.0 + u2 * (1.5 + u2 * (3.75 + u2 * (13.125 + 59.0625 * u2))))
    return jnp.where(au >= HARRIS_U_MAX, wing, inner)


def voigt_harris(x: jnp.ndarray, sigma: jnp.ndarray, gamma: jnp.ndarray, *, a_max: float | None = HARRIS_A_MAX) -> jnp.ndarray:
    """V(x; sigma, gamma) = H(a, u) / (sqrt(2 pi) sigma), u = x / (sqrt(2) sigma), a = gamma / (sqrt(2) sigma), with H to
    first order in a (``harris_H``), as fake_spectra evaluates it.

    x, sigma, gamma broadcastable (e.g. x (N, N_v) with sigma, gamma (N, 1)). Elements with a > ``a_max`` take the exact
    evaluation (``voigt_exojax``); one ``lax.cond`` on the whole batch skips it when every a <= a_max (under an outer
    ``vmap`` both branches run). ``a_max=None``: the expansion everywhere. jit / vmap / grad (piecewise linear in |u|
    through the table).
    """
    x = jnp.asarray(x)
    tiny = jnp.finfo(x.dtype).tiny
    s = jnp.maximum(jnp.asarray(sigma, dtype=x.dtype), tiny)
    g = jnp.maximum(jnp.asarray(gamma, dtype=x.dtype), 0.0)
    a = g / (jnp.sqrt(2.0) * s)
    fast = harris_H(x / (jnp.sqrt(2.0) * s), a) / (jnp.sqrt(2.0 * jnp.pi) * s)
    if a_max is None:
        return fast
    xb, sb, gb, ab = jnp.broadcast_arrays(x, s, jnp.maximum(g, tiny), a)

    def mixed(_):
        exact = jax.vmap(voigtone)(xb.ravel(), sb.ravel(), gb.ravel()).reshape(xb.shape)
        return jnp.where(ab <= a_max, jnp.broadcast_to(fast, xb.shape), exact)

    return jax.lax.cond(jnp.all(ab <= a_max), lambda _: jnp.broadcast_to(fast, xb.shape), mixed, None)


def log10_voigt(x, sigma, gamma, *, mode: Literal["ExoJAX", "convolution", "harris", "scipy_wofz"] = "ExoJAX", **kwargs: Any):
    """log10 V(x; sigma, gamma) with one of ``VOIGT_MODES`` (case-insensitive).

    "ExoJAX", "convolution" and "harris" are JAX (jit / vmap / grad); "scipy_wofz" is host-side NumPy and fails on
    tracers. ``kwargs`` go to ``voigt_convolution`` or ``voigt_harris``.
    """
    m = mode.lower()
    if m == "exojax":
        V = voigt_exojax(x, sigma, gamma)
    elif m == "convolution":
        V = voigt_convolution(x, sigma, gamma, **kwargs)
    elif m == "harris":
        V = voigt_harris(x, sigma, gamma, **kwargs)
    elif m == "scipy_wofz":
        V = voigt_scipy(np.asarray(x), np.asarray(sigma), np.asarray(gamma))
    else:
        raise ValueError(f"Unknown Voigt mode '{mode}'. Supported modes: {', '.join(repr(v) for v in VOIGT_MODES)}.")
    return jnp.log10(V)
