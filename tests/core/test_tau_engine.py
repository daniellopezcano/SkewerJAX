"""The integrated profile (``core.voigt_integral``), the windowed engine (``core.tau_engine``) and the cell crossings
(``core.crossings``): against numerical quadrature of Harris' expansion and of SciPy's ``wofz``, against a dense
evaluation of every pixel, and against the dual-segment route they replace."""

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import wofz

from skewerjax.core import crossings as cx
from skewerjax.core import tau_engine as te
from skewerjax.core import voigt_integral as vi
from skewerjax.core.constants import phys
from skewerjax.core.lines import LineBundle
from skewerjax.core.voigt import harris_H, harris_H4

from helpers import float64_only, tol


def _H(u, a):
    return np.real(wofz(u + 1j * a))


@float64_only
def test_integrated_profile_against_quadrature():
    """<H> over an interval = the quadrature of Harris' expansion (exactly the function integrated: 1e-9) and of wofz
    (Harris' own truncation: 1e-6 at a = 1e-3, 2e-7 at a = 0.05)."""
    rng = np.random.default_rng(1)
    for a, fourth, tol_wofz in ((1e-4, False, 2e-8), (1e-3, False, 1.1e-6), (1e-2, True, 5e-8), (0.05, True, 3e-7)):
        Hh = (lambda t: float(harris_H4(jnp.asarray(t), a))) if fourth else (lambda t: float(harris_H(jnp.asarray(t), a)))
        for _ in range(40):
            c, h = rng.uniform(-12, 12), 10 ** rng.uniform(-3, 1.2)
            got = float(vi.harris_mean(jnp.asarray(c - h), jnp.asarray(c + h), jnp.asarray(a), fourth=fourth))
            pts = [0.0] if c - h < 0 < c + h else None
            ref_h = quad(Hh, c - h, c + h, epsabs=1e-14, epsrel=1e-12, limit=400, points=pts)[0] / (2 * h)
            ref_w = quad(lambda t: _H(t, a), c - h, c + h, epsabs=1e-14, epsrel=1e-12, limit=400, points=pts)[0] / (2 * h)
            assert abs(got - ref_h) < 2e-8, (a, c, h)
            assert abs(got - ref_w) < tol_wofz, (a, c, h)


@float64_only
def test_cumulative_profile_is_the_integral():
    u = np.linspace(-20, 20, 4001)
    for a, fourth in ((3e-4, False), (0.03, True)):
        Phi = np.asarray(vi.harris_cumulative(jnp.asarray(u), a, fourth=fourth))
        dPhi = np.gradient(Phi, u)
        H = np.asarray((harris_H4 if fourth else harris_H)(jnp.asarray(u), a))
        assert np.max(np.abs(dPhi - H)) < 2e-4  # the finite difference
        assert abs(Phi[-1] - (np.sqrt(np.pi) / 2 - a * (2 / np.sqrt(np.pi)) * vi.dawson(20.0))) < 1e-9


def _synthetic(lines, j, n_sk, n_v, rng, n_per=150):
    lam, A21, mass = float(lines.lambda0_A[j]), float(lines.A21_GHz[j]), float(lines.mass_amu[j])
    n = n_sk * n_per
    T = 10 ** rng.uniform(3.6, 5.5, n)
    sig, gam = te.line_widths(T, np.zeros(n), np.zeros(n), lam, A21, mass)
    tau0 = 10 ** rng.uniform(-4, 1.5, n)
    tau0[rng.choice(n, 3, replace=False)] = 1e5  # damped absorbers: the whole grid
    c = te.LineContributions(
        np.repeat(np.arange(n_sk), n_per),
        rng.uniform(-100, n_v + 100, n),
        np.abs(rng.normal(2, 4, n)),
        tau0 * np.sqrt(2 * np.pi) * sig,
        sig,
        gam,
        None,
        rng.uniform(-1e-5, 1e-5, n),
    )
    return c, lam


def _dense(c, lam, v, n_sk, period=None):
    """Every contribution on every pixel (the reference of the window / far-wing partition)."""
    kappa = 1e4 / lam
    out = np.zeros((n_sk, v.size))
    delta = v[None, :] - c.vc[:, None]
    if period is not None:
        delta = np.mod(delta + 0.5 * period, period) - 0.5 * period
    s2 = np.sqrt(2) * c.sigma[:, None]
    x = -delta / phys.c_kms
    em1 = np.expm1(x)
    nu = kappa * phys.c_kms * em1
    half0 = kappa * c.width[:, None] / (2 * s2)
    h = half0 * (1 + em1)
    u = nu / s2 + h * half0 * s2 / (2 * kappa * phys.c_kms)
    a = c.gamma[:, None] / s2
    M = np.where(
        a > 1e-3,
        np.asarray(vi.harris_mean(jnp.asarray(u - h), jnp.asarray(u + h), jnp.asarray(a), fourth=True)),
        np.asarray(vi.harris_mean(jnp.asarray(u - h), jnp.asarray(u + h), jnp.asarray(a))),
    )
    hh = np.maximum(h, 1e-30)
    M1 = np.where(
        h >= vi.NARROW_HALF_WIDTH,
        np.asarray(vi.gaussian_moment_wide(jnp.asarray(u - hh), jnp.asarray(u + hh))),
        np.asarray(vi.gaussian_moment_narrow(jnp.asarray(u), jnp.asarray(h))),
    )
    g = c.tilt[:, None] - 1 / phys.c_kms
    M = M + g * s2 / (kappa * (1 + em1)) * M1
    np.add.at(out, c.row, c.amp[:, None] / (np.sqrt(np.pi) * s2) * M)
    return out


@float64_only
@pytest.mark.parametrize("period", [None, "grid", 777.7])
def test_windows_and_far_wings_equal_the_dense_sum(period, monkeypatch):
    """The window / far-wing partition (FFT of the Lorentz series) changes nothing but rounding and the wing bound
    (``wing_tolerance`` 1e-9 per contribution at its window's edge)."""
    monkeypatch.setattr(te, "SMALL_PROBLEM", 0)  # the windows even on a small problem
    lines = LineBundle(["HI-LyA", "SiIII-1206"])
    rng = np.random.default_rng(3)
    n_sk, n_v, dv = 4, 1200, 1.0
    v = (np.arange(n_v) + 0.5) * dv
    P = None if period is None else (n_v * dv if period == "grid" else period)
    for j in range(2):
        c, lam = _synthetic(lines, j, n_sk, n_v, rng)
        tau, info = te.windowed_line_tau(
            c, v, n_sk, lam, period=None if P is None else np.full(n_sk, P), dtype=np.float64, return_info=True
        )
        ref = _dense(c, lam, v, n_sk, P)
        assert np.max(np.abs(np.asarray(tau) - ref)) < 2e-7, (j, period)
        if period != 777.7:
            assert info["levels"] and info["D_x"] is not None  # windows were used


def test_wing_levels_bound_the_omitted_series():
    """The window of a contribution grows with its peak (the far term's omitted terms scale with tau_0)."""
    sig = np.full(3, 10.0)
    gam = np.full(3, 0.05)
    amp = np.array([1e-3, 1.0, 1e3]) * np.sqrt(2 * np.pi) * sig
    lev = te.wing_levels(amp, sig, gam, np.zeros(3), np.zeros(3), 8.2, 1.0, 1e-9)
    assert lev[0] <= lev[1] <= lev[2] and lev[0] < lev[2]


def test_engine_precisions_agree(monkeypatch):
    monkeypatch.setattr(te, "SMALL_PROBLEM", 0)
    lines = LineBundle(["HI-LyA"])
    rng = np.random.default_rng(5)
    v = (np.arange(800) + 0.5) * 1.0
    c, lam = _synthetic(lines, 0, 3, 800, rng, n_per=60)
    t = np.asarray(te.windowed_line_tau(c, v, 3, lam))
    ref = _dense(c, lam, v, 3)
    F, Fr = np.exp(-t), np.exp(-ref)
    assert np.max(np.abs(F - Fr)) < tol(1e-7, 2e-6)  # the wing tolerance (1e-9 per contribution) summed


def test_crossings_tile_the_path_and_are_the_nearest_cells():
    """The envelope's crossings tile [0, L], and the midpoint of every crossing is nearest to its cell's generator (a
    brute-force check over every generator, periodic box)."""
    rng = np.random.default_rng(7)
    box = 10.0
    pos = rng.uniform(0, box, (4000, 3))
    size = np.full(4000, (box**3 / 4000) ** (1 / 3))
    origin = np.array([[1.0, 0.0, 2.0], [7.5, 0.0, 9.9], [0.05, 0.0, 5.0]])
    paths = cx.StraightPaths(origin, np.tile([0.0, 1.0, 0.0], (3, 1)), np.full(3, box), box, 1)
    cr = cx.straight_crossings(paths, pos, size)
    for k in range(3):
        sl = slice(cr.start[k], cr.start[k + 1])
        lo, hi = cr.s_lo[sl], cr.s_hi[sl]
        assert lo[0] == 0.0 and hi[-1] == pytest.approx(box) and np.allclose(lo[1:], hi[:-1])
        mid = 0.5 * (lo + hi)
        x = origin[k][None, :] + mid[:, None] * np.array([0.0, 1.0, 0.0])
        d = x[:, None, :] - pos[None, :, :]
        d -= box * np.round(d / box)
        assert np.array_equal(np.argmin((d**2).sum(-1), axis=1), cr.cell[sl])


def test_general_directions_match_the_axis_kernel():
    """The brute-force candidates of a straight path in any direction give the axis kernel's crossings when the path
    is along an axis."""
    rng = np.random.default_rng(8)
    box = 8.0
    pos = rng.uniform(0, box, (3000, 3))
    size = np.full(3000, (box**3 / 3000) ** (1 / 3))
    p_axis = cx.StraightPaths(np.array([[2.0, 0.0, 3.0]]), np.array([[0.0, 1.0, 0.0]]), np.array([box]), box, 1)
    p_gen = cx.StraightPaths(p_axis.origin, p_axis.direction, p_axis.length, box, None)
    a = cx.straight_crossings(p_axis, pos, size)
    b = cx.straight_crossings(p_gen, pos, size)
    assert np.array_equal(a.cell, b.cell) and np.allclose(a.s_lo, b.s_lo)


def test_chords_are_fake_spectras_spheres():
    """Every sphere of radius h touching the line once, with z = sqrt(h^2 - r^2) (a brute-force check)."""
    rng = np.random.default_rng(9)
    box = 6.0
    pos = rng.uniform(0, box, (2000, 3))
    h = rng.uniform(0.1, 0.4, 2000)
    paths = cx.StraightPaths(np.array([[3.0, 0.0, 3.0]]), np.array([[0.0, 1.0, 0.0]]), np.array([box]), box, 1)
    ch = cx.straight_chords(paths, pos, h)
    d = pos - np.array([3.0, 3.0, 3.0])
    d -= box * np.round(d / box)
    r2 = d[:, 0] ** 2 + d[:, 2] ** 2
    touch = np.flatnonzero(r2 < h * h)
    assert np.array_equal(np.sort(ch.cell), touch)
    o = np.argsort(ch.cell)
    np.testing.assert_allclose(ch.z[o], np.sqrt(h[touch] ** 2 - r2[touch]))
    np.testing.assert_allclose(ch.u[o], d[touch, 1] + 3.0)
