"""The Voigt function integrated over a cell crossing: the cumulative profile Phi(a, u) and the mean of H over an
interval (``docs/performance.md``, section D2).

Inside one Voronoi crossing the density, temperature and peculiar velocity are constant and the line centre moves
linearly with the path (the Hubble flow), so the crossing absorbs as the *mean* of the profile over the detunings of its
two ends, u_lo and u_hi (Doppler units, u = nu / (sqrt(2) sigma)):

    <H>(u_lo, u_hi) = [Phi(u_hi) - Phi(u_lo)] / (u_hi - u_lo),    Phi(u) = int_0^u H(a, t) dt,

with H(a, u) = Re w(u + i a) the Voigt function of ``core.voigt`` (unit area sqrt(pi)). Harris' (1948) expansion
H = sum_n a^n H_n(u), H_n = Re[i^n w^(n)(u) / n!], integrates term by term in closed form. With w(u) = exp(-u^2) +
(2 i / sqrt(pi)) F(u) on the real axis (F Dawson's function) and G_n = i^n w^(n) / n!, int H_n du = -Im G_(n-1) / n:

    Phi_0 = (sqrt(pi) / 2) erf(u),             Phi_1 = -(2 / sqrt(pi)) F(u),         Phi_2 = u exp(-u^2),
    Phi_3 = -(2 / (3 sqrt(pi))) g_3(u),  g_3 = (1 - 2u^2) F(u) + u,                   Phi_4 = (u / 2 - u^3 / 3) exp(-u^2).

The first order (a <= ``HARRIS_A_MAX`` = 1e-3) keeps Phi_0 + a Phi_1, the fourth order (a <= ``HARRIS4_A_MAX`` = 0.05)
every term, as ``core.voigt.voigt_harris`` does for H. F and g_3 come from Chebyshev series on [0, ``U_CHEB``] (fitted
at import in float64 with SciPy's ``dawsn``) and their asymptotic series beyond.

**Numerics.** The difference of Phi loses precision when the interval is narrow (Phi ~ 1, the difference ~ H du) and in
the far tails (erf -> 1). Two evaluations therefore cover the two regimes:

* ``harris_mean_wide`` (half width h = (u_hi - u_lo) / 2 >= ``NARROW_HALF_WIDTH``): the erf part as a difference of
  ``erfc`` of |u| (exact cancellation-free forms on each side of 0), the Dawson parts as plain differences;
* ``harris_mean_narrow`` (h < ``NARROW_HALF_WIDTH``): the Taylor series of the mean about the midpoint,
  <H> = H + (h^2 / 6) H'' + (h^4 / 120) H'''' + (h^6 / 5040) H^(6) + O(h^8), every derivative in closed form (Hermite
  polynomials for the Gaussian term, Chebyshev series for the Dawson terms): below 1e-7 of the peak for h < 0.2.
  The switch at h = 0.2 bounds the float32 cancellation of the wide form to ~1e-6 of the mean.

Both are jit / vmap / grad compatible (polynomials, exp, erfc). The damping parameters above ``HARRIS4_A_MAX`` (very
cold metal lines) are integrated numerically by ``core.tau_engine`` (sub-pieces with Gauss-Legendre nodes on the exact
profile).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax.scipy.special import erfc
from scipy.special import dawsn

from skewerjax.core.voigt import HARRIS4_A_MAX, HARRIS_A_MAX, _chebval, harris_H, harris_H4

U_CHEB = 8.0  # Chebyshev domain [0, U_CHEB] of F, g_3 and H_1''; asymptotic series beyond
CHEB_DEGREE = 40
NARROW_HALF_WIDTH = 0.2  # half width h of the interval in u below which the Taylor form is used
SQRT_PI = float(np.sqrt(np.pi))
__all__ = [
    "HARRIS_A_MAX",
    "HARRIS4_A_MAX",
    "NARROW_HALF_WIDTH",
    "dawson",
    "harris_cumulative",
    "harris_mean",
    "harris_mean_narrow",
    "harris_mean_wide",
]


def _fit(fn, degree: int = CHEB_DEGREE) -> np.ndarray:
    """Chebyshev coefficients of fn(u) in x = 2 u / U_CHEB - 1 on [0, U_CHEB] (least squares, float64)."""
    u = np.linspace(0.0, U_CHEB, 40001)
    return np.polynomial.chebyshev.chebfit(2.0 * u / U_CHEB - 1.0, fn(u), degree)


def _g3(u):
    return (1.0 - 2.0 * u * u) * dawsn(u) + u


def _h1pp(u):
    """H_1''(u) = (2 / sqrt(pi)) (4 - 4u^2 - 12 u F + 8 u^3 F) (from H_1 = -(2 / sqrt(pi)) F', F' = 1 - 2 u F)."""
    F = dawsn(u)
    return (2.0 / SQRT_PI) * (4.0 - 4.0 * u * u - 12.0 * u * F + 8.0 * u**3 * F)


def _h1pppp(u):
    """H_1'''' = -(2 / sqrt(pi)) F^(5), the Dawson derivatives from F^(n+1) = -2u F^(n) - 2n F^(n-1), F' = 1 - 2uF."""
    d = [dawsn(u)]
    d.append(1.0 - 2.0 * u * d[0])
    for n in range(1, 5):
        d.append(-2.0 * u * d[n] - 2.0 * n * d[n - 1])
    return -(2.0 / SQRT_PI) * d[5]


def _h3pp(u):
    """H_3'' from H_3 = (2 / sqrt(pi)) (p F + q), p = 2u - 4u^3 / 3, q = (2/3)(u^2 - 1) (``core.voigt``), F' = 1 - 2uF."""
    F = dawsn(u)
    F1 = 1.0 - 2.0 * u * F
    F2 = -2.0 * F - 2.0 * u * F1
    p, p1, p2 = 2.0 * u - 4.0 * u**3 / 3.0, 2.0 - 4.0 * u * u, -8.0 * u
    return (2.0 / SQRT_PI) * (p2 * F + 2.0 * p1 * F1 + p * F2 + 4.0 / 3.0)


# Degrees from the accuracy each term needs (max errors on [0, 8]): F 7e-9 (it enters as a (F(u_hi) - F(u_lo)) / 2h);
# the others enter multiplied by a h^2 / 6 <= 3e-4 (H_1''), a^3 (g_3, H_3''), a h^4 / 120 <= 7e-7 (H_1''''): every
# term stays below 1e-8 of the peak
_CHEB_F = _fit(dawsn, 32)
_CHEB_G3 = _fit(_g3, 24)
_CHEB_H1PP = _fit(lambda u: SQRT_PI * _h1pp(u), 28)  # sqrt(pi) H_1'', as its asymptotic series
_CHEB_H3PP = _fit(lambda u: SQRT_PI * _h3pp(u), 24)
_CHEB_H1PPPP = _fit(lambda u: SQRT_PI * _h1pppp(u), 28)
# asymptotic series beyond U_CHEB (relative errors ~1e-8 at u = 8):
# F = (1 / 2u) sum_k (2k - 1)!! / (2u^2)^k ; g_3 = sum_k (c_k - 2 c_(k+1)) / u^(2k+1), c_k = (2k - 1)!! / 2^(k+1);
# H_1'' = (1 / sqrt(pi)) d^2/du^2 sum_k c'_k / u^(2k+2), c'_k = (1, 1.5, 3.75, 13.125, 59.0625, 324.84375)
_F_ASYM = (0.5, 0.25, 0.375, 0.9375, 3.28125, 14.765625, 81.2109375)  # F = sum_k a_k / u^(2k+1)
_G3_ASYM = (0.0, -0.5, -1.5, -5.625, -26.25, -147.65625, -965.390625)  # g_3 = sum_k d_k / u^(2k+1)
_H1PP_ASYM = (6.0, 30.0, 157.5, 945.0, 6496.875, 50675.625)  # sqrt(pi) H_1'' = sum_k e_k / u^(2k+4)
# H_3 = -(1 / sqrt(pi)) W_3, W_3 = sum_k w_k / u^(2k+4) (core.voigt._W3): sqrt(pi) H_3'' = -sum_k w_k (2k+4)(2k+5) / u^(2k+6)
_H3PP_ASYM = (-20.0, -210.0, -1890.0, -17325.0, -168918.75, -1773646.875)
# sqrt(pi) H_1'''' = sum_k c'_k (2k+2)(2k+3)(2k+4)(2k+5) / u^(2k+6)
_H1PPPP_ASYM = (120.0, 1260.0, 11340.0, 103950.0, 1013512.5, 10641881.25)


def _odd_series(au, coeffs, first_power: int):
    """sum_k coeffs[k] / au^(first_power + 2k) (au > 0), by Horner in 1 / au^2."""
    q = 1.0 / (au * au)
    out = jnp.full_like(au, coeffs[-1])
    for c in coeffs[-2::-1]:
        out = out * q + c
    return out / au**first_power


def _cheb_or_asym(u, cheb, asym, first_power: int, odd: bool):
    """A function of u from its Chebyshev series on |u| <= U_CHEB and its asymptotic series beyond; ``odd``: f(-u) =
    -f(u) (else even)."""
    au = jnp.abs(u)
    x = jnp.minimum(au, U_CHEB) * (2.0 / U_CHEB) - 1.0
    inner = _chebval(x, cheb)
    outer = _odd_series(jnp.maximum(au, U_CHEB), asym, first_power)
    f = jnp.where(au <= U_CHEB, inner, outer)
    return jnp.sign(u) * f if odd else f


def dawson(u):
    """Dawson's function F(u) = exp(-u^2) int_0^u exp(t^2) dt (odd; max error ~1e-9 in float64)."""
    return _cheb_or_asym(u, _CHEB_F, _F_ASYM, 1, odd=True)


def _g3_of(u):
    return _cheb_or_asym(u, _CHEB_G3, _G3_ASYM, 1, odd=True)


def _h1pp_of(u):
    return _cheb_or_asym(u, _CHEB_H1PP, _H1PP_ASYM, 4, odd=False) * (1.0 / SQRT_PI)


def _h3pp_of(u):
    return _cheb_or_asym(u, _CHEB_H3PP, _H3PP_ASYM, 6, odd=False) * (1.0 / SQRT_PI)


def _h1pppp_of(u):
    return _cheb_or_asym(u, _CHEB_H1PPPP, _H1PPPP_ASYM, 6, odd=False) * (1.0 / SQRT_PI)


def _dawson_terms(u, a, fourth: bool):
    """Phi - Phi_0: the Dawson-function terms of the cumulative profile (first or fourth order in a)."""
    out = a * (-(2.0 / SQRT_PI)) * dawson(u)
    if fourth:
        e = jnp.exp(-u * u)
        a2 = a * a
        out = out + a2 * (u * e) + a2 * a * (-(2.0 / (3.0 * SQRT_PI))) * _g3_of(u) + a2 * a2 * ((0.5 * u - u**3 / 3.0) * e)
    return out


def harris_cumulative(u, a, *, fourth: bool = False):
    """Phi(a, u) = int_0^u H(a, t) dt (Harris' expansion to first, or with ``fourth``, fourth order in a); odd in u,
    Phi(+inf) = sqrt(pi) / 2 + O(a). u, a broadcastable (dimensionless)."""
    return 0.5 * SQRT_PI * jnp.sign(u) * (1.0 - _erfc_abs(u)) + _dawson_terms(u, a, fourth)


def _erfc_abs(u):
    return erfc(jnp.abs(u))


def _erf_difference(u_lo, u_hi):
    """erf(u_hi) - erf(u_lo) (u_hi >= u_lo) without cancellation: from erfc(|u|) on each side of 0."""
    e_lo, e_hi = _erfc_abs(u_lo), _erfc_abs(u_hi)
    same_pos = u_lo >= 0
    same_neg = u_hi <= 0
    return jnp.where(same_pos, e_lo - e_hi, jnp.where(same_neg, e_hi - e_lo, 2.0 - e_lo - e_hi))


def harris_mean_wide(u_lo, u_hi, a, *, fourth: bool = False):
    """<H> over [u_lo, u_hi] (u_hi > u_lo) from differences of the cumulative profile (intervals with half width >=
    ``NARROW_HALF_WIDTH``; module docstring)."""
    d = 0.5 * SQRT_PI * _erf_difference(u_lo, u_hi) + _dawson_terms(u_hi, a, fourth) - _dawson_terms(u_lo, a, fourth)
    return d / (u_hi - u_lo)


def harris_mean_narrow(u, h, a, *, fourth: bool = False):
    """<H> over [u - h, u + h] from its Taylor series about u (h < ``NARROW_HALF_WIDTH``; module docstring):
    H + (h^2 / 6) H'' + (h^4 / 120) H'''' + (h^6 / 5040) H^(6). The Gaussian term's derivatives come from the Hermite
    polynomials, (d/du)^n exp(-u^2) = (-1)^n He_n(u) exp(-u^2); H'' and H'''' follow the order of H in a (the fourth
    order adds the a^2, a^3 terms of H'' and the a^2 term of H'''', with H_2 = -(1/2) (exp(-u^2))'')."""
    H = harris_H4(u, a) if fourth else harris_H(u, a)
    u2 = u * u
    e = jnp.exp(-u2)
    He4 = (16.0 * u2 - 48.0) * u2 + 12.0
    He6 = ((64.0 * u2 - 480.0) * u2 + 720.0) * u2 - 120.0
    d2 = (4.0 * u2 - 2.0) * e + a * _h1pp_of(u)
    d4 = He4 * e + a * _h1pppp_of(u)
    if fourth:
        d2 = d2 + a * a * (-0.5 * He4 * e + a * _h3pp_of(u))
        d4 = d4 + a * a * (-0.5 * He6 * e)
    h2 = h * h
    return H + h2 * (d2 * (1.0 / 6.0) + h2 * (d4 * (1.0 / 120.0) + h2 * (He6 * e) * (1.0 / 5040.0)))


def harris_mean(u_lo, u_hi, a, *, fourth: bool = False):
    """<H>(a) over [u_lo, u_hi] (any order of the ends; equal ends: H(u)), choosing the narrow or the wide form per
    element (both are evaluated: for tests and figures; ``core.tau_engine`` calls one of them per batch)."""
    lo, hi = jnp.minimum(u_lo, u_hi), jnp.maximum(u_lo, u_hi)
    h = 0.5 * (hi - lo)
    mid = 0.5 * (hi + lo)
    wide = h >= NARROW_HALF_WIDTH
    safe_hi = jnp.where(wide, hi, lo + 2.0 * NARROW_HALF_WIDTH)  # finite in the unused branch
    return jnp.where(
        wide,
        harris_mean_wide(lo, safe_hi, a, fourth=fourth),
        harris_mean_narrow(mid, jnp.minimum(h, NARROW_HALF_WIDTH), a, fourth=fourth),
    )


def gaussian_moment_wide(u_lo, u_hi):
    """<(u - u_mid) exp(-u^2)> over [u_lo, u_hi]: the first moment of the Gaussian term about the midpoint (the tilt of
    a crossing's amplitude, ``core.tau_engine``); [(e^(-u_lo^2) - e^(-u_hi^2)) / 2 - u_mid (sqrt(pi) / 2) (erf(u_hi) -
    erf(u_lo))] / (u_hi - u_lo)."""
    mid = 0.5 * (u_lo + u_hi)
    m = 0.5 * (jnp.exp(-u_lo * u_lo) - jnp.exp(-u_hi * u_hi)) - mid * (0.5 * SQRT_PI) * _erf_difference(u_lo, u_hi)
    return m / (u_hi - u_lo)


def gaussian_moment_narrow(u, h):
    """<(u' - u) exp(-u'^2)> over [u - h, u + h] to leading order: (h^2 / 3) d/du exp(-u^2)."""
    return (h * h / 3.0) * (-2.0 * u * jnp.exp(-u * u))
