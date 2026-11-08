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
    x: jnp.ndarray, sigma: jnp.ndarray, gamma: jnp.ndarray, *, oversample: int = 1, work_dtype=None
) -> jnp.ndarray:
    """G * L by a zero-padded FFT convolution on a uniform grid x (N,), refined ``oversample`` times.

    The kernels are evaluated in ``work_dtype``, convolved, scaled by the fine spacing (unit area) and sampled back at x.
    None: float64 when ``jax_enable_x64`` is on, float32 otherwise. This standalone reference keeps the best precision
    available whatever the package's policy: in float32 the FFT's round-off (~1e-7 of the peak) dominates the far wings.
    Returns (N,) in the dtype of x.
    """
    if work_dtype is None:
        work_dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
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
    s, g = jnp.maximum(sigma, tiny).astype(x.dtype), jnp.maximum(gamma, tiny).astype(x.dtype)
    # ExoJAX's constants are weakly typed only in part: with x64 on, float32 inputs come back float64 (cast back)
    return jax.vmap(voigtone, in_axes=(0, None, None))(x, s, g).astype(x.dtype)


# First-order expansion of H(a, u) = Re w(u + i a) in the damping parameter a (Harris 1948, ApJ 108, 112), the scheme
# of fake_spectra (fastvoigt.h, commit 2aebe44; singleabs.h ``profile``):
#     H(a, u) = exp(-u^2) + a H1(u) + O(a^2),   H1(u) = 2 u Im w(u) - 2 / sqrt(pi) = (2 / sqrt(pi)) (2 u Daw(u) - 1),
# H1 on [0, U_MAX] from a Chebyshev series of degree HARRIS_DEGREE (max error 1.3e-8; fake_spectra tabulates it on 1024
# intervals and interpolates linearly, max error 1.9e-5): a polynomial, so the evaluation vectorises (a table lookup is
# a gather, which XLA's CPU back end does not vectorise). Past U_MAX the Gaussian has underflowed and H1 is its
# asymptotic series. For a > HARRIS_A_MAX the expansion is not used (fake_spectra calls Faddeeva::w there; here
# ExoJAX's ``voigtone``). The Gaussian is evaluated exactly (fake_spectra tabulates it with a Taylor correction, good to
# 1e-8).
HARRIS_U_MAX, HARRIS_N_TAB, HARRIS_A_MAX, HARRIS_DEGREE = 6.0, 1024, 1e-3, 28
_HARRIS_U = np.arange(HARRIS_N_TAB + 2) * (HARRIS_U_MAX / HARRIS_N_TAB)
_HARRIS_H1 = (2.0 / np.sqrt(np.pi)) * (2.0 * _HARRIS_U * dawsn(_HARRIS_U) - 1.0)  # fake_spectra's table (tests)


def _harris_h1_chebyshev(degree: int = HARRIS_DEGREE) -> np.ndarray:
    """Chebyshev coefficients of H1(u) in x = 2 u / U_MAX - 1 on [0, U_MAX] (least squares on a fine grid)."""
    u = np.linspace(0.0, HARRIS_U_MAX, 20001)
    h1 = (2.0 / np.sqrt(np.pi)) * (2.0 * u * dawsn(u) - 1.0)
    return np.polynomial.chebyshev.chebfit(2.0 * u / HARRIS_U_MAX - 1.0, h1, degree)


_HARRIS_CHEB = _harris_h1_chebyshev()


def _chebval(x, c):
    """Clenshaw's recurrence for sum_k c_k T_k(x) (c a NumPy array: unrolled into the trace)."""
    b1 = b2 = jnp.zeros_like(x)
    for ck in c[:0:-1]:
        b1, b2 = 2.0 * x * b1 - b2 + ck, b1
    return x * b1 - b2 + c[0]


def harris_H(u: jnp.ndarray, a: jnp.ndarray) -> jnp.ndarray:
    """H(a, u) to first order in a (no fallback): exp(-u^2) + a H1(u) for |u| < 6, the asymptotic
    a / (sqrt(pi) u^2) (1 + 3/(2u^2) + 15/(4u^4) + 105/(8u^6) + 945/(16u^8)) beyond. u, a broadcastable (dimensionless)."""
    au = jnp.abs(u)
    x = jnp.minimum(au, HARRIS_U_MAX) * (2.0 / HARRIS_U_MAX) - 1.0
    h1 = _chebval(x, _HARRIS_CHEB)
    inner = jnp.exp(-au * au) + a * h1
    u2 = 1.0 / jnp.maximum(au, HARRIS_U_MAX) ** 2  # safe in the inner region (not used there)
    wing = a / jnp.sqrt(jnp.pi) * u2 * (1.0 + u2 * (1.5 + u2 * (3.75 + u2 * (13.125 + 59.0625 * u2))))
    return jnp.where(au >= HARRIS_U_MAX, wing, inner)


# Fourth order in a, for HARRIS_A_MAX < a <= HARRIS4_A_MAX (fake_spectra calls the exact Faddeeva::w there; ExoJAX's
# ``voigtone`` costs ~25 times more): H = sum_{n <= 4} a^n H_n(u) with H_n = Re[i^n w^(n)(u) / n!] (Harris 1948),
#     H0 = exp(-u^2), H2 = (1 - 2u^2) exp(-u^2), H4 = (1/2 - 2u^2 + 2u^4/3) exp(-u^2),
#     H1 = (2/sqrt(pi)) (2 u Daw(u) - 1), H3 = -(2/sqrt(pi)) (2(1 - u^2)/3 - 2u (1 - 2u^2/3) Daw(u)),
# H1 and H3 from Chebyshev series on [0, HARRIS4_U_MAX] (errors 1e-9, 1e-8); the series is accurate to 5e-8 at a = 0.03,
# 5e-7 at a = 0.05. Beyond HARRIS4_U_MAX the asymptotic expansion of w(u + i a) to third order in a,
#     H = (a / sqrt(pi)) (W1(u) - a^2 W3(u)),  W1 = sum_k c_k (2k + 1) / u^(2k+2),  W3 = sum_k c_k C(2k+3, 3) / u^(2k+4),
# c_k = (2k - 1)!! / 2^k (six terms: 3e-8 at u = 8). Checked against SciPy's wofz in tests/core/test_voigt.py.
HARRIS4_U_MAX, HARRIS4_A_MAX = 8.0, 0.05


def _harris4_chebyshev(n: int, degree: int) -> np.ndarray:
    """Chebyshev coefficients of H_n (n = 1, 3) in x = 2 u / U - 1 on [0, HARRIS4_U_MAX] (least squares, from the
    closed forms with SciPy's Dawson function, in float64)."""
    u = np.linspace(0.0, HARRIS4_U_MAX, 40001)
    F = dawsn(u)
    if n == 1:
        h = (2.0 / np.sqrt(np.pi)) * (2.0 * u * F - 1.0)
    else:
        h = -(2.0 / np.sqrt(np.pi)) * ((2.0 / 3.0) * (1.0 - u * u) - 2.0 * u * (1.0 - (2.0 / 3.0) * u * u) * F)
    return np.polynomial.chebyshev.chebfit(2.0 * u / HARRIS4_U_MAX - 1.0, h, degree)


_HARRIS4_CHEB = {1: _harris4_chebyshev(1, 36), 3: _harris4_chebyshev(3, 36)}
_W1 = (1.0, 1.5, 3.75, 13.125, 59.0625, 324.84375)  # c_k (2k + 1)
_W3 = (1.0, 5.0, 26.25, 157.5, 1082.8125, 8445.9375)  # c_k C(2k + 3, 3)


def _horner(q, coeffs):
    out = jnp.full_like(q, coeffs[-1])
    for c in coeffs[-2::-1]:
        out = out * q + c
    return out


def harris_H4(u: jnp.ndarray, a: jnp.ndarray) -> jnp.ndarray:
    """H(a, u) to fourth order in a (accurate to 5e-7 for a <= 0.05; module comments): the series below
    ``HARRIS4_U_MAX``, the asymptotic wing beyond. u, a broadcastable (dimensionless)."""
    au = jnp.abs(u)
    x = jnp.minimum(au, HARRIS4_U_MAX) * (2.0 / HARRIS4_U_MAX) - 1.0
    u2 = au * au
    a2 = a * a
    core = jnp.exp(-u2) * (1.0 + a2 * ((1.0 - 2.0 * u2) + a2 * (0.5 - 2.0 * u2 + (2.0 / 3.0) * u2 * u2)))
    core = core + a * (_chebval(x, _HARRIS4_CHEB[1]) + a2 * _chebval(x, _HARRIS4_CHEB[3]))
    q = 1.0 / jnp.maximum(u2, HARRIS4_U_MAX**2)  # safe in the inner region (not used there)
    wing = (a / jnp.sqrt(jnp.pi)) * q * (_horner(q, _W1) - a2 * q * _horner(q, _W3))
    return jnp.where(au >= HARRIS4_U_MAX, wing, core)


def voigt_harris(x: jnp.ndarray, sigma: jnp.ndarray, gamma: jnp.ndarray, *, a_max: float | None = HARRIS_A_MAX) -> jnp.ndarray:
    """V(x; sigma, gamma) = H(a, u) / (sqrt(2 pi) sigma), u = x / (sqrt(2) sigma), a = gamma / (sqrt(2) sigma), with H
    from Harris' (1948) expansion in a: to first order (``harris_H``, as fake_spectra evaluates it) for a <= ``a_max``.

    x, sigma, gamma broadcastable (e.g. x (N, N_v) with sigma, gamma (N, 1)). Elements with a > ``a_max`` (1e-3, where
    fake_spectra switches to the exact Faddeeva function) take the fourth order (``harris_H4``, accurate to 5e-7) up to
    ``HARRIS4_A_MAX`` = 0.05, and the exact evaluation (``voigt_exojax``) beyond. ``lax.cond`` on the damping parameters
    (of the shape of sigma, e.g. (N, 1)) picks the cheapest branch that covers the whole batch: under an outer ``vmap``
    every branch runs. ``a_max=None``: the first order everywhere. jit / vmap / grad (polynomials in |u|).
    """
    x = jnp.asarray(x)
    tiny = jnp.finfo(x.dtype).tiny
    s = jnp.maximum(jnp.asarray(sigma, dtype=x.dtype), tiny)
    g = jnp.maximum(jnp.asarray(gamma, dtype=x.dtype), 0.0)
    a = g / (jnp.sqrt(2.0) * s)  # in the shape of sigma and gamma, e.g. (N, 1): never broadcast over x for the test
    shape = jnp.broadcast_shapes(x.shape, s.shape, g.shape)

    u = lambda: x * (1.0 / (jnp.sqrt(2.0) * s))
    norm = lambda h: jnp.broadcast_to(h * (1.0 / (jnp.sqrt(2.0 * jnp.pi) * s)), shape)

    def first(_):  # fake_spectra's expansion
        return norm(harris_H(u(), a))

    if a_max is None:
        return first(None)

    def fourth(_):  # every element below HARRIS4_A_MAX: the first order up to a_max, the fourth beyond
        return norm(jnp.where(a <= a_max, harris_H(u(), a), harris_H4(u(), a)))

    def mixed(_):  # some element above HARRIS4_A_MAX: the exact function there
        xb, sb, gb = jnp.broadcast_arrays(x, s, jnp.maximum(g, tiny))
        exact = jax.vmap(voigtone)(xb.ravel(), sb.ravel(), gb.ravel()).reshape(shape).astype(x.dtype)
        return jnp.where(jnp.broadcast_to(a, shape) <= HARRIS4_A_MAX, fourth(None), exact)

    rest = lambda _: jax.lax.cond(jnp.all(a <= HARRIS4_A_MAX), fourth, mixed, None)
    return jax.lax.cond(jnp.all(a <= a_max), first, rest, None)


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
