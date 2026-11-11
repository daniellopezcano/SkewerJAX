"""The windowed optical-depth engine: every contribution on a window of pixels around its line centre, the far Lorentz
wings by one FFT convolution per skewer (``docs/performance.md``, section D3).

A **contribution** is one piece of the path (a cell crossing, or a sampled piece) seen by one line: its line centre
v_c at the midpoint, the velocity extent W of the Hubble flow across it (0 for a point), its amplitude A = C f n l a^-2
and its widths sigma, gamma. It adds to the optical depth (``core.optical_depth``)

    tau_i(v) = A_i <phi>_i(v),   <phi>_i = mean of the Voigt profile over the line centres of the piece,

``core.voigt_integral`` (the integrated profile; ``profile="midpoint"``: the profile at v_c). Away from the line centre
the profile is the Lorentz wing of the Voigt function, in the exact detuning nu = -kappa Delta', Delta' =
c (1 - exp(-Delta / c)), kappa = 1 / lambda (Harris' asymptotic series, ``core.voigt``),

    A phi(nu) = w / Delta'^2 [1 + (3/2) b^2 / Delta'^2 + O(b^4 / Delta'^4)],    w = A gamma / (pi kappa^2),  b = sqrt(2) sigma / kappa,

which is the same function of Delta for every contribution of a line up to the weights w and w b^2. So the engine splits
the pixels of each contribution exactly in two:

* **the window** |k - c| <= D (c the pixel nearest to v_c): the exact (integrated) profile, minus the far term below
  where the far kernel is not zero (the ring D_x < |k - c| <= D), so that the two parts never overlap;
* **the far wing**, every pixel: the moments of each contribution deposited at its pixel c, m_0 = w, m_1 = w <t>,
  m_2 = w <t^2>, m_3 = (3/2) w b^2 (<t^n> over the piece's line centres relative to the pixel centre: <t> = delta,
  <t^2> = delta^2 + W^2 / 12), convolved (FFT) with the kernels K, -K', K'' / 2 and 1 / Delta'^4 of the pixel offsets,
  K = 1 / Delta'^2, set to zero for |k - c| <= D_x (the smallest window of the line).

**The window bound** (``wing_tolerance`` epsilon). The far term omits the next terms of the series; at the first pixel
outside the window, at a distance Delta_e >= (D + 1) dv - |delta| - W / 2 from every line centre of the piece, its
relative error is at most 3.75 (b / Delta_e)^4 + 15 (b s / Delta_e^2)^2 + 4 (s / Delta_e)^3 + 5 (s / Delta_e)^4
(s = |delta| + W / 2), and it omits the Gaussian core, at most tau_0 exp(-(Delta_e / b)^2) (tau_0 = A / (sqrt(2 pi)
sigma), the peak). D is the smallest level of ``LEVELS`` that brings w / Delta_e^2 x (relative error) + tau_0
exp(-(Delta_e / b)^2) below epsilon; beyond it the error falls as Delta^-6. A contribution that needs half the grid or
more (a damped absorber) takes every pixel of the grid with the exact profile, and no far term ("full").

Pixel grid: uniform. Periodicity ``velocity_period``: a period equal to the grid's span (N dv, the periodic spectrum of
a box) wraps the window indices and makes the far convolution circular. Any other period takes every pixel of the grid
for every contribution, with the nearest image of each detuning (exact, at the cost of the dense sum: the boundary of
the nearest image, half a period from each line centre, does not fall on the pixel offsets of a shared kernel).

**Per-cell schemes** (fake_spectra's ``tautail`` early stop and ``pixel_average``, ``per_cell_optical_depth``): the
pieces of one cell form one absorber, evaluated on windows escalated until fake_spectra's walk stops inside (the
full-grid result), with no far term (the early stop cuts the wings by definition).

Every kernel is jitted with static shapes (a window level, a chunk of contributions padded to a power of two); the host
groups the contributions by level and kind, the device evaluates and scatter-adds them into the (N_sk N_v) optical
depth.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from skewerjax.core.constants import phys
from skewerjax.core.voigt import HARRIS4_A_MAX, HARRIS_A_MAX, harris_H, harris_H4
from skewerjax.core.voigt_integral import (
    NARROW_HALF_WIDTH,
    gaussian_moment_narrow,
    gaussian_moment_wide,
    harris_mean_narrow,
    harris_mean_wide,
)

logger = logging.getLogger(__name__)

LEVELS = (4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)  # window half-widths D [pixels]
WING_TOLERANCE = 1e-9  # default epsilon: the optical depth a contribution may misplace at its window edge
SMALL_PROBLEM = 2e7  # contributions x pixels below which every contribution takes the whole grid (``windowed_line_tau``)
EVALUATIONS_PER_CALL = 1 << 21  # window elements per kernel call (memory ~ a few x 8 MB in float32)
PROFILE_INTEGRATIONS = ("integrated", "midpoint")
KINDS = ("point", "narrow", "wide")
GL3 = (np.array([-np.sqrt(0.6), 0.0, np.sqrt(0.6)]) / 2.0, np.array([5.0, 8.0, 5.0]) / 18.0)  # nodes on [-1/2, 1/2], weights


@dataclass
class LineContributions:
    """The contributions of one line (host arrays, (n,)): ``row`` the skewer, ``vc`` the line centre at the piece's
    midpoint [km/s] on the axis of the grid (float64), ``width`` the extent W of the line centre across the piece [km/s]
    (>= 0; 0 for a point), ``amp`` = C f n l a^-2 [GHz] (tau = amp x profile [ns]), ``sigma``, ``gamma`` [GHz] (Gaussian
    and Lorentzian widths), ``group`` (optional) the cell of every piece (the per-cell schemes), ``tilt`` (optional) the
    gradient of the amplitude along the piece, d ln(amp) / d v_c [s/km] (the a(s)^-2 of the comoving density; None: 0),
    ``spacing`` (optional; fake_spectra's chords) every contribution is the 7 points of a chord, at vc + (k - 4) spacing
    [km/s], k = 1..7, each with amp / 7 (the per-cell schemes generate them on the device)."""

    row: np.ndarray
    vc: np.ndarray
    width: np.ndarray
    amp: np.ndarray
    sigma: np.ndarray
    gamma: np.ndarray
    group: np.ndarray | None = None
    tilt: np.ndarray | None = None
    spacing: np.ndarray | None = None

    def __len__(self) -> int:
        return int(np.size(self.row))

    def take(self, idx) -> LineContributions:
        f = lambda x: None if x is None else np.asarray(x)[idx]
        return LineContributions(
            f(self.row),
            f(self.vc),
            f(self.width),
            f(self.amp),
            f(self.sigma),
            f(self.gamma),
            f(self.group),
            f(self.tilt),
            f(self.spacing),
        )

    @staticmethod
    def concatenate(parts) -> LineContributions:
        parts = list(parts)

        def cat(k):
            if all(getattr(p, k) is None for p in parts):
                return None
            fill = lambda p: getattr(p, k) if getattr(p, k) is not None else np.zeros(len(p), np.int64 if k == "group" else float)
            return np.concatenate([fill(p) for p in parts])

        return LineContributions(*(cat(k) for k in ("row", "vc", "width", "amp", "sigma", "gamma", "group", "tilt", "spacing")))


def line_widths(T, turbulence, collisional, lambda0_A: float, A21_GHz: float, mass_amu: float):
    """(sigma, gamma) [GHz] of one line for gas elements (float64 NumPy; ``line_profiles.voigt_widths_GHz``)."""
    gamma = (A21_GHz + np.asarray(collisional, dtype=np.float64)) / (4.0 * np.pi)
    v_th = np.sqrt(2.0 * phys.kB * np.asarray(T, dtype=np.float64) / mass_amu)
    sigma = (v_th + np.asarray(turbulence, dtype=np.float64)) * 10.0 / (np.sqrt(2.0) * lambda0_A)
    return np.maximum(sigma, 1e-300), np.broadcast_to(gamma, sigma.shape)


# --------------------------------------------------------------------------- kernels (device)
def _exact_H(u, a):
    """H(a, u) = sqrt(pi) V(u; 1/sqrt(2), a) with ExoJAX's Faddeeva approximation (``core.voigt.voigt_exojax``),
    elementwise for u, a broadcastable."""
    from exojax.opacity.lpf.lpf import voigtone

    ub, ab = jnp.broadcast_arrays(u, a)
    s = jnp.asarray(1.0 / np.sqrt(2.0), ub.dtype)
    V = jax.vmap(lambda x, g: voigtone(x, s, g))(ub.ravel(), jnp.maximum(ab.ravel(), jnp.finfo(ub.dtype).tiny))
    return (np.sqrt(np.pi) * V.reshape(ub.shape)).astype(u.dtype)


def _profile_mean(kind: str, order: int, u, h, a):
    """<H> of a batch of one ``kind`` and Voigt ``order`` (1, 4: Harris; 0: exact): the value at u (point), or the
    mean over [u - h, u + h] (narrow: Taylor; wide: the cumulative profile)."""
    if kind == "point":
        return _exact_H(u, a) if order == 0 else (harris_H(u, a) if order == 1 else harris_H4(u, a))
    if kind == "narrow":
        return harris_mean_narrow(u, h, a, fourth=order == 4)
    return harris_mean_wide(u - h, u + h, a, fourth=order == 4)


def _expm1_small(x):
    """exp(x) - 1 for |x| < 1e-2 (the detuning over c: |Delta| / c < 4e-3 on any velocity grid of a box): four terms of
    the series, error < x^5 / 120 < 1e-12, cheaper than the transcendental."""
    return x * (1.0 + x * (0.5 + x * (1.0 / 6.0 + x * (1.0 / 24.0))))


def far_kernels(delta_kms, detuning: str):
    """The far-wing kernels at velocity offsets ``delta_kms`` [km/s] (Delta = v - v_c): (K, -K', K'' / 2, K_4) with
    K = 1 / Delta'^2, K_4 = 1 / Delta'^4, Delta' = c (1 - exp(-Delta / c)) ("exact") or Delta ("linear")."""
    d = delta_kms
    if detuning == "exact":
        x = -d / phys.c_kms
        e = jnp.exp(x)  # dDelta' / dDelta
        dp = -phys.c_kms * jnp.expm1(x)
    else:
        e = jnp.ones_like(d)
        dp = d
    inv = 1.0 / jnp.where(dp == 0, 1.0, dp)
    K = inv * inv
    K1 = -2.0 * K * inv * e
    K2 = 6.0 * K * K * e * e
    if detuning == "exact":
        K2 = K2 + 2.0 * K * inv * e / phys.c_kms
    return K, -K1, 0.5 * K2, K * K


def _window_body(d, half0, amp, sigma, a, tilt, centre, P, m, dv, kappa, D_x, *, D, kind, order, detuning, mode, n_v, ring):
    """(tau (C, W), pixel (C, W)) of C contributions on their windows (``_window_kernel``)."""
    C = d.shape[0]
    if mode == "full":
        j = jnp.arange(n_v)[None, :] - centre[:, None]
    else:
        j = jnp.broadcast_to(jnp.arange(-D, D + 1)[None, :], (C, 2 * D + 1))
    jf = j.astype(d.dtype) * dv
    delta = jf - d[:, None]  # v_pixel - v_c [km/s], formed from small numbers
    if P is not None:
        Pc = P[:, None]
        delta = jnp.mod(delta + 0.5 * Pc, Pc) - 0.5 * Pc
    s2 = np.sqrt(2.0) * sigma[:, None]
    if detuning == "exact":
        x = -delta * (1.0 / phys.c_kms)
        em1 = _expm1_small(x)
        nu = (kappa * phys.c_kms) * em1
        h = half0[:, None] * (1.0 + em1)
        # the ends of the piece map to u_lo, u_hi by the exponential: their centre lies above u(v_c) by h W / (4 c)
        u = nu / s2 + h * half0[:, None] * (s2 / (2.0 * kappa * phys.c_kms))
    else:
        nu = -kappa * delta
        h = jnp.broadcast_to(half0[:, None], delta.shape)
        u = nu / s2
    M = _profile_mean(kind, order, u, h, a[:, None])
    if kind != "point":  # the weight of the line centres varies linearly along the piece: + tilt (dt/du) <(u - u_mid) H>
        g = tilt[:, None] - (1.0 / phys.c_kms if detuning == "exact" else 0.0)
        dt_du = s2 / kappa * (1.0 / (1.0 + em1) if detuning == "exact" else 1.0)
        M1 = gaussian_moment_narrow(u, h) if kind == "narrow" else gaussian_moment_wide(u - h, u + h)
        M = M + g * dt_du * M1
    tau = (amp[:, None] / (np.sqrt(np.pi) * s2)) * M
    if ring:  # the far term the FFT adds on D_x < |j| <= D, removed here: the two parts do not overlap
        k0, k1, k2, k3 = far_kernels(jf, detuning)
        far = m[0][:, None] * k0 + m[1][:, None] * k1 + m[2][:, None] * k2 + m[3][:, None] * k3
        tau = tau - jnp.where(jnp.abs(j) > D_x, far, 0.0)
    pix = centre[:, None] + j if mode != "full" else jnp.broadcast_to(jnp.arange(n_v)[None, :], j.shape)
    if mode == "circular":
        pix = jnp.mod(pix, n_v)
    else:
        keep = (pix >= 0) & (pix < n_v)
        pix = jnp.clip(pix, 0, n_v - 1)
        tau = jnp.where(keep, tau, 0.0)
    return tau, pix


@partial(jax.jit, static_argnames=("D", "kind", "order", "detuning", "mode", "n_v", "ring"), donate_argnums=(0,))
def _window_kernel(
    out, d, half0, amp, sigma, a, tilt, centre, row, P, m, dv, kappa, D_x, *, D, kind, order, detuning, mode, n_v, ring
):
    """Adds C contributions to ``out`` (n_sk n_v,) on windows of 2D + 1 pixels (the device's scatter-add: GPUs).
    ``mode`` "circular": pixel indices wrapped; "linear": the pixels off the grid dropped; "full": every pixel of the
    grid, the detuning wrapped by ``P``. d (C,) [km/s]: v_c minus the velocity of the centre pixel; half0 (C,): the
    half extent h of the piece in u at Delta = 0; amp, sigma [GHz] (C,); a (C,) damping parameters; tilt (C,)
    d ln(amp) / d v_c [s/km] (the exact detuning adds its Jacobian, -1 / c); centre, row (C,) int; P (C,) periods
    [km/s] or None ("full" only); m (4, C) the far moments, subtracted on the ring D_x < |j| <= D when ``ring``;
    dv, kappa, D_x scalars (traced)."""
    tau, pix = _window_body(
        d,
        half0,
        amp,
        sigma,
        a,
        tilt,
        centre,
        P,
        m,
        dv,
        kappa,
        D_x,
        D=D,
        kind=kind,
        order=order,
        detuning=detuning,
        mode=mode,
        n_v=n_v,
        ring=ring,
    )
    return out.at[(row[:, None] * n_v + pix).ravel()].add(tau.ravel())


@partial(jax.jit, static_argnames=("D", "kind", "order", "detuning", "mode", "n_v", "ring"))
def _window_values(d, half0, amp, sigma, a, tilt, centre, P, m, dv, kappa, D_x, *, D, kind, order, detuning, mode, n_v, ring):
    """The values (C, W) of ``_window_kernel``'s contributions, without the scatter (on the CPU the windows are added
    on the host, ``_accumulate``: XLA's scatter on the CPU is serial)."""
    return _window_body(
        d,
        half0,
        amp,
        sigma,
        a,
        tilt,
        centre,
        P,
        m,
        dv,
        kappa,
        D_x,
        D=D,
        kind=kind,
        order=order,
        detuning=detuning,
        mode=mode,
        n_v=n_v,
        ring=ring,
    )[0]


_ACC = {}


def _accumulator():
    """numba: out[row_i, pix(c_i + j)] += vals[i, j], parallel over the skewers of a chunk sorted by row (no two threads
    write one row). mode 0 circular (wrapped), 1 linear (off the grid dropped), 2 full (j is the pixel)."""
    if _ACC:
        return _ACC["f"]
    import numba

    @numba.njit(nogil=True, parallel=True)
    def acc(out, vals, row, centre, rstart, D, mode):
        n_v = out.shape[1]
        W = vals.shape[1]
        for g in numba.prange(rstart.size - 1):
            for i in range(rstart[g], rstart[g + 1]):
                r = row[i]
                if mode == 2:
                    for j in range(W):
                        out[r, j] += vals[i, j]
                    continue
                c0 = centre[i] - D
                for j in range(W):
                    k = c0 + j
                    if mode == 0:
                        k %= n_v
                    elif k < 0 or k >= n_v:
                        continue
                    out[r, k] += vals[i, j]

    _ACC["f"] = acc
    return acc


@partial(jax.jit, static_argnames=("n_dom",), donate_argnums=(0,))
def _deposit_far(dep, row, cpix, m, *, n_dom):
    """Adds the far moments m (4, C) at (row, cpix) of dep (4, n_sk n_dom)."""
    return dep.at[:, row * n_dom + cpix].add(m)


@partial(jax.jit, static_argnames=("n_sk", "n_dom", "n_fft", "n_v", "offset"))
def _far_convolve(dep, kern_hat, *, n_sk, n_dom, n_fft, n_v, offset):
    """sum_m dep_m * kernel_m for every skewer: dep (4, n_sk n_dom), kern_hat (4, n_fft // 2 + 1) the kernels' FFTs;
    returns the (n_sk, n_v) outputs from ``offset``."""
    X = jnp.fft.rfft(dep.reshape(4, n_sk, n_dom), n=n_fft, axis=-1)
    y = jnp.fft.irfft((X * kern_hat[:, None, :]).sum(0), n=n_fft, axis=-1)
    return jax.lax.slice_in_dim(y, offset, offset + n_v, axis=-1)


# --------------------------------------------------------------------------- host side
def _dtype(dtype):
    """The working dtype: ``dtype``, or the package's precision policy (``skewerjax.config.real_dtype``)."""
    from skewerjax.config import real_dtype

    return real_dtype() if dtype is None else np.dtype(dtype)


def _pow2(n: int) -> int:
    return 1 << max(int(n) - 1, 0).bit_length()


def _fast_len(n: int) -> int:
    """The smallest 2^a 3^b 5^c >= n (a fast FFT length)."""
    best = _pow2(n)
    p5 = 1
    while p5 < 2 * n:
        p35 = p5
        while p35 < 2 * n:
            x = p35
            while x < n:
                x *= 2
            best = min(best, x)
            p35 *= 3
        p5 *= 5
    return int(best)


def wing_levels(amp, sigma, gamma, width, delta, kappa: float, dv: float, tolerance: float, levels=LEVELS) -> np.ndarray:
    """(n,) the window level of every contribution (an index of ``levels``; len(levels) when none suffices): the
    smallest half width D that brings the far-term error at the first pixel outside the window below ``tolerance``
    (module docstring)."""
    b = np.sqrt(2.0) * sigma / kappa  # Doppler width [km/s]
    w = amp * gamma / (np.pi * kappa * kappa)  # far-wing amplitude [(km/s)^2]
    tau0 = amp / (np.sqrt(2.0 * np.pi) * sigma)
    s_ = np.abs(delta) + 0.5 * width
    lev = np.full(np.shape(amp), len(levels), np.int64)
    todo = np.arange(np.size(amp))
    for i, D in enumerate(levels):
        ss, bb = s_[todo], b[todo]
        De = (D + 1) * dv - ss
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            De_ = np.where(De > 0, De * (1.0 - np.minimum(De, 0.5 * phys.c_kms) / phys.c_kms), np.nan)  # Delta' <= Delta
            rb, rs = bb / De_, ss / De_
            rel = 3.75 * rb**4 + 15.0 * (rb * rs) ** 2 + 4.0 * rs**3 + 5.0 * rs**4
            err = w[todo] / De_**2 * rel + tau0[todo] * np.exp(-np.minimum((De_ / bb) ** 2, 700.0))
        good = err <= tolerance
        lev[todo[good]] = i
        todo = todo[~good]
        if not todo.size:
            break
    return lev


def _grid(v):
    v = np.asarray(v, dtype=np.float64)
    n_v = v.shape[-1]
    if not is_uniform(v):
        raise ValueError("the windowed engine needs a uniform, increasing velocity grid of at least 2 pixels.")
    return float(v[0]), float(v[1] - v[0]), n_v


def is_uniform(v) -> bool:
    """True when every row of ``v`` (N_v,) or (N_lines, N_v) is a uniform increasing grid of >= 2 pixels."""
    v = np.atleast_2d(np.asarray(v, dtype=np.float64))
    if v.shape[-1] < 2:
        return False
    d = np.diff(v, axis=-1)
    return bool(np.all(d > 0) and np.allclose(d, d[:, :1], rtol=1e-9, atol=0.0))


def _gl_expand(c: LineContributions, sel: np.ndarray, kappa: float) -> LineContributions:
    """The contributions ``sel`` as point sub-pieces: each piece cut in q equal parts of half the profile's width in u
    (q = ceil(2 h0 / (0.5 max(1, a))), h0 its half extent in u, at most 4096), each integrated by 3-point Gauss-Legendre
    on the exact profile."""
    sub = c.take(sel)
    h0 = kappa * sub.width / (2.0 * np.sqrt(2.0) * sub.sigma)
    a = sub.gamma / (np.sqrt(2.0) * sub.sigma)
    # sub-pieces of half the profile's width in u: the Doppler core (1), or the Lorentz core (a) when wider
    q = np.clip(np.nan_to_num(np.ceil(2.0 * h0 / (0.5 * np.maximum(1.0, a))), nan=1.0, posinf=4096.0), 1, 4096).astype(np.int64)
    rep = np.repeat(np.arange(sel.size), q * 3)
    k = np.arange(rep.size) - np.repeat(np.cumsum(q * 3) - q * 3, q * 3)
    part, node = k // 3, k % 3
    qq = q[rep]
    out = sub.take(rep)
    t = ((part + 0.5) / qq - 0.5 + GL3[0][node] / qq) * out.width  # the node's line centre offset [km/s]
    out.vc = out.vc + t
    out.amp = out.amp * GL3[1][node] / qq * (1.0 + (0.0 if out.tilt is None else out.tilt) * t)
    out.width = np.zeros_like(out.width)
    return out


class _Dispatcher:
    """Runs window kernels over chunks of contributions: a fixed chunk shape per window (``chunk_size``; the last chunk
    padded with empty contributions, so that each window compiles once), the chunks dealt in turn to ``devices``.
    On GPUs every device scatter-adds into its own copy of the optical depth; on the CPU (``host``) the kernels return
    their windows and the host adds them (``_accumulate``, parallel over skewers), a few chunks behind the kernels so
    that both run at once."""

    def __init__(self, n_rows: int, n_v: int, dtype, devices, host: bool):
        self.dt = np.dtype(dtype)
        self.devices = list(devices) if devices else [None]
        self.put = [(lambda x: jnp.asarray(x)) if d is None else (lambda x, d=d: jax.device_put(x, d)) for d in self.devices]
        self.host = host
        self.n_rows, self.n_v = n_rows, n_v
        if host:
            self.acc = np.zeros((n_rows, n_v), np.float64)
            self.pending = []
        else:
            self.out = [p(np.zeros(n_rows * n_v, self.dt)) for p in self.put]
        self.calls = {}
        self._next = 0
        self._consts = {}

    def _const(self, i, key, x):
        k = (i, key, float(x))
        if k not in self._consts:
            self._consts[k] = self.put[i](np.asarray(x, self.dt))
        return self._consts[k]

    def _drain(self, keep: int):
        acc = _accumulator()
        while len(self.pending) > keep:
            vals, row, cen, D, mode = self.pending.pop(0)
            v = np.asarray(vals)
            rstart = np.flatnonzero(np.diff(np.concatenate([[-1], row, [-2]]))).astype(np.int64)
            acc(self.acc, v, row, cen, rstart, D, mode)

    def run(self, arrays: dict, width: int, D_x: float, static: dict, tag: str, dv: float, kappa: float):
        n = arrays["d"].size
        if self.host:  # sorted by skewer: the host's accumulation runs in parallel over skewers
            o = np.argsort(arrays["row"], kind="stable")
            arrays = {k: (None if x is None else (x[:, o] if np.ndim(x) == 2 else x[o])) for k, x in arrays.items()}
        C = chunk_size(width)
        names = ("d", "half0", "amp", "sigma", "a", "tilt", "centre", "row", "P")
        mode_id = {"circular": 0, "linear": 1, "full": 2}[static["mode"]]
        for c0 in range(0, n, C):
            m_ = min(C, n - c0)
            i = self._next
            self._next = (self._next + 1) % len(self.devices)
            put = self.put[i]

            def take(name):
                x = arrays.get(name)
                if x is None:
                    return None
                arr = x[c0 : c0 + m_]
                if m_ < C:  # padding: copies of the last contribution with no amplitude
                    arr = np.concatenate([arr, np.repeat(arr[-1:], C - m_, axis=0)])
                    if name == "amp":
                        arr[m_:] = 0.0
                return put(np.ascontiguousarray(arr, dtype=np.int32 if name in ("centre", "row") else self.dt))

            vals = {k: take(k) for k in names}
            m = arrays.get("m")
            if m is not None:
                mm = np.zeros((4, C), self.dt)
                mm[:, :m_] = m[:, c0 : c0 + m_]
                m = put(mm)
            consts = (self._const(i, "dv", dv), self._const(i, "kappa", kappa), self._const(i, "Dx", D_x))
            if self.host:
                v = _window_values(
                    vals["d"],
                    vals["half0"],
                    vals["amp"],
                    vals["sigma"],
                    vals["a"],
                    vals["tilt"],
                    vals["centre"],
                    vals["P"],
                    m,
                    *consts,
                    **static,
                )
                self.pending.append(
                    (
                        v,
                        arrays["row"][c0 : c0 + m_].astype(np.int64),
                        arrays["centre"][c0 : c0 + m_].astype(np.int64),
                        static["D"],
                        mode_id,
                    )
                )
                self._drain(2 * len(self.devices))
            else:
                self.out[i] = _window_kernel(
                    self.out[i],
                    vals["d"],
                    vals["half0"],
                    vals["amp"],
                    vals["sigma"],
                    vals["a"],
                    vals["tilt"],
                    vals["centre"],
                    vals["row"],
                    vals["P"],
                    m,
                    *consts,
                    **static,
                )
        self.calls[tag] = self.calls.get(tag, 0) + -(-n // C)

    def total(self):
        """The optical depth (n_rows, n_v): on the host (float64) or on the first device."""
        if self.host:
            self._drain(0)
            return self.acc
        out = self.out[0]
        for o in self.out[1:]:
            out = out + (o if self.devices[0] is None else jax.device_put(o, self.devices[0]))
        return out.reshape(self.n_rows, self.n_v)


def chunk_size(width: int) -> int:
    """Contributions per kernel call for windows of ``width`` pixels (a power of two, about ``EVALUATIONS_PER_CALL``
    evaluations)."""
    return max(_pow2(max(EVALUATIONS_PER_CALL // max(int(width), 1), 1)), 16)


def windowed_line_tau(
    c: LineContributions,
    v,
    n_sk: int,
    lambda0_A: float,
    *,
    period=None,
    profile: str = "integrated",
    detuning: str = "exact",
    voigt: str = "harris",
    wing_tolerance: float = WING_TOLERANCE,
    dtype=None,
    device=None,
    devices=None,
    return_info: bool = False,
):
    """tau (n_sk, N_v) of one line from its contributions (module docstring) on the uniform grid ``v`` (N_v,) [km/s].

    period : None (no wrap) or (n_sk,) periods [km/s] (``velocity_period``).
    profile : "integrated" (default: the mean of the profile over each piece's line centres) or "midpoint" (at v_c).
    voigt : "harris" (default: first order for a <= 1e-3, fourth to 0.05, the exact function beyond) or "ExoJAX" (the
        exact function everywhere; integrated pieces through Gauss-Legendre sub-pieces).
    wing_tolerance : epsilon of the window bound (module docstring).
    device : the device of the kernels (None: JAX's default); devices : a list of devices instead, over which the
        chunks of contributions are dealt (e.g. several CPU devices, ``skewerjax.config.configure_devices``).
    Returns the (n_sk, N_v) device array, and with ``return_info`` a dict (levels {D: count}, n_full, D_x, kinds, calls,
    time_s)."""
    t0 = time.perf_counter()
    if profile not in PROFILE_INTEGRATIONS:
        raise ValueError(f"profile must be one of {PROFILE_INTEGRATIONS}, got {profile!r}.")
    v0, dv, n_v = _grid(v)
    kappa = 1e4 / float(lambda0_A)  # GHz per km/s
    dt = _dtype(dtype)
    devs = list(devices) if devices else [device]
    put = (lambda x: jnp.asarray(x)) if devs[0] is None else (lambda x: jax.device_put(x, devs[0]))
    info = dict(n=len(c), levels={}, n_full=0, D_x=None, kinds={}, calls={}, wing_tolerance=wing_tolerance, time_s=0.0)
    if len(c) == 0:
        z = put(np.zeros((n_sk, n_v), dt))
        return (z, info) if return_info else z
    exact_all = str(voigt).lower() == "exojax"
    width = np.asarray(c.width, dtype=np.float64) if profile == "integrated" else np.zeros(len(c))
    tilt = np.zeros(len(c)) if c.tilt is None else np.asarray(c.tilt, dtype=np.float64)
    c = LineContributions(c.row, c.vc, width, c.amp, c.sigma, c.gamma, c.group, tilt)
    a = c.gamma / (np.sqrt(2.0) * c.sigma)
    expand = (width > 0) & (exact_all | (a > HARRIS4_A_MAX))  # the exact profile: Gauss-Legendre sub-pieces
    if expand.any():
        c = LineContributions.concatenate([c.take(np.flatnonzero(~expand)), _gl_expand(c, np.flatnonzero(expand), kappa)])
        a = c.gamma / (np.sqrt(2.0) * c.sigma)
    P_sk = None if period is None else np.broadcast_to(np.asarray(period, dtype=np.float64), (n_sk,))
    circular = P_sk is not None and bool(np.allclose(P_sk, n_v * dv, rtol=1e-9, atol=0.0))
    general = P_sk is not None and not circular
    rel = (c.vc - v0) / dv
    if circular:
        rel = np.mod(rel, n_v)
    cpix = np.floor(rel + 0.5).astype(np.int64)
    delta = (rel - cpix) * dv  # v_c minus the centre pixel's velocity [km/s] (float64)
    if circular:
        cpix = np.mod(cpix, n_v)
    half0 = kappa * c.width / (2.0 * np.sqrt(2.0) * c.sigma)
    kind = np.where(c.width <= 0, 0, np.where(half0 < NARROW_HALF_WIDTH, 1, 2))
    # one Harris order per line (the fourth order is valid at every a <= 0.05: one kernel per window), the exact
    # function for a > 0.05 (points only: the integrated ones were expanded above)
    line_order = 4 if np.any((a > HARRIS_A_MAX) & (a <= HARRIS4_A_MAX)) else 1
    order = np.where(exact_all | (a > HARRIS4_A_MAX), 0, line_order)
    lev = wing_levels(c.amp, c.sigma, c.gamma, c.width, delta, kappa, dv, wing_tolerance)
    Ds = np.array(LEVELS + (n_v,))
    D_of = Ds[np.minimum(lev, len(LEVELS))]
    max_D = (n_v // 2 - 1) if P_sk is not None else (n_v - 1)
    margin = far_margin(n_v)
    off_grid = (not circular) & ((cpix < -margin) | (cpix > n_v - 1 + margin))  # beyond the far term's domain
    full = (lev >= len(LEVELS)) | (D_of > max_D) | general | off_grid
    if len(c) * n_v <= SMALL_PROBLEM:  # a small problem: every contribution on the whole grid (exact; few kernels to compile)
        full[:] = True
    win = np.flatnonzero(~full)
    D_x = int(D_of[win].min()) if win.size else None
    b = np.sqrt(2.0) * c.sigma / kappa
    w = c.amp * c.gamma / (np.pi * kappa * kappa)
    m = np.stack([w, w * delta, w * (delta**2 + c.width**2 / 12.0), 1.5 * w * b * b])  # (4, n), about the centre pixel
    n_rows = _pow2(n_sk)  # the output is padded to a power of two of skewers: the same shapes for every tile
    host = all((d.platform == "cpu") if d is not None else jax.default_backend() == "cpu" for d in devs)
    disp = _Dispatcher(n_rows, n_v, dt, devs, host)
    base = dict(detuning=detuning, n_v=n_v)
    if win.size:
        key = (lev[win] * 4 + kind[win]) * 8 + order[win]
        o = np.argsort(key, kind="stable")
        cut = np.flatnonzero(np.diff(key[o])) + 1
        for grp in np.split(o, cut):
            sel = win[grp]
            k0 = int(key[grp[0]])
            L, kd, od = k0 // 32, (k0 // 8) % 4, k0 % 8
            D = int(LEVELS[L])
            arrays = dict(
                d=delta[sel],
                half0=half0[sel],
                amp=c.amp[sel],
                sigma=c.sigma[sel],
                a=a[sel],
                tilt=c.tilt[sel],
                centre=cpix[sel],
                row=c.row[sel],
                m=m[:, sel],
            )
            static = dict(base, D=D, kind=KINDS[kd], order=int(od), mode="circular" if circular else "linear", ring=True)
            disp.run(arrays, 2 * D + 1, D_x, static, f"D{D}:{KINDS[kd]}:{od}", dv, kappa)
            info["levels"][D] = info["levels"].get(D, 0) + int(sel.size)
    fsel = np.flatnonzero(full)
    if fsel.size:
        uk, inv = np.unique(np.stack([kind[fsel], order[fsel]], 1), axis=0, return_inverse=True)
        inv = np.ravel(inv)
        for g, (kd, od) in enumerate(uk):
            sel = fsel[inv == g]
            arrays = dict(
                d=delta[sel],
                half0=half0[sel],
                amp=c.amp[sel],
                sigma=c.sigma[sel],
                a=a[sel],
                tilt=c.tilt[sel],
                centre=cpix[sel],
                row=c.row[sel],
                P=None if P_sk is None else P_sk[c.row[sel]],
                m=None,
            )
            static = dict(base, D=0, kind=KINDS[kd], order=int(od), mode="full", ring=False)
            disp.run(arrays, n_v, 0.0, static, f"full:{KINDS[kd]}:{od}", dv, kappa)
    far = _far_term(c.row[win], cpix[win], m[:, win], n_rows, n_v, dv, D_x, detuning, circular, dt, put) if win.size else None
    tau = disp.total()
    if host:
        if far is not None:
            tau += np.asarray(far, dtype=np.float64)
        tau = put(tau[:n_sk].astype(dt))
    else:
        tau = (tau if far is None else tau + far)[:n_sk]
    info.update(n_full=int(full.sum()), D_x=D_x, calls=disp.calls, time_s=time.perf_counter() - t0)
    info["kinds"] = {KINDS[k]: int((kind == k).sum()) for k in range(3)}
    return (tau, info) if return_info else tau


def far_margin(n_v: int) -> int:
    """Pixels beyond each end of a non-periodic grid where the far term takes line centres (others: the full grid)."""
    return max(n_v // 2, 64)


FAR_CHUNK = 1 << 18  # contributions per deposit call (fixed shape)


def _far_term(row, cpix, m, n_rows, n_v, dv, D_x, detuning, circular, dt, put):
    """(n_rows, n_v) the far wings of the windowed contributions (module docstring): their moments deposited at their
    centre pixels and convolved with the far kernels (zero within D_x pixels): circular for a period of the grid's
    span, linear without a period (zero-padded, over a fixed domain of ``far_margin`` pixels beyond each end: the same
    shapes for every call)."""
    if circular:
        cp = cpix
        n_dom = n_fft = n_v
        jk = np.arange(n_v)
        jk = np.where(jk >= (n_v + 1) // 2, jk - n_v, jk)  # the nearest image of every offset
        offset = 0
    else:
        M = far_margin(n_v)
        lo, hi = -M, n_v - 1 + M
        n_dom = hi - lo + 1
        cp = cpix - lo
        n_fft = _fast_len(2 * (hi - lo) + n_v + 1)
        jk = np.arange(-hi, n_v - lo)  # the offsets k - c, stored from index 0: y[k] = z[k + hi - lo]
        offset = hi - lo
    ks = far_kernels(jnp.asarray(jk.astype(np.float64) * dv, dtype=dt), detuning)
    keep = jnp.asarray(np.abs(jk) > D_x)
    kern = jnp.stack([jnp.where(keep, k, 0.0) for k in ks])
    if not circular:
        kern = jnp.zeros((4, n_fft), dt).at[:, : jk.size].set(kern)
    kern_hat = jnp.fft.rfft(put(np.asarray(kern)), n=n_fft, axis=-1)
    dep = put(np.zeros((4, n_rows * n_dom), dt))
    for c0 in range(0, cp.size, FAR_CHUNK):
        sl = slice(c0, c0 + FAR_CHUNK)
        k = cp[sl].size
        rr = np.zeros(FAR_CHUNK, np.int32)
        cc = np.zeros(FAR_CHUNK, np.int32)
        mm = np.zeros((4, FAR_CHUNK), dt)
        rr[:k], cc[:k], mm[:, :k] = row[sl], cp[sl], m[:, sl]
        dep = _deposit_far(dep, put(rr), put(cc), put(mm), n_dom=n_dom)
    return _far_convolve(dep, kern_hat, n_sk=n_rows, n_dom=n_dom, n_fft=n_fft, n_v=n_v, offset=offset)


# --------------------------------------------------------------------------- per-cell schemes (fake_spectra's)
WINDOW_LEVELS = (4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 8192, 32768)  # half widths [pixels] of the per-cell windows
CORE_U = 10.0  # the core of a cell's window: this many Doppler widths beyond its extreme line centres
CORE_LEVELS = (32, 64, 128, 256)  # half widths [pixels] of the cores of wide per-cell windows (one compilation each)


def pixel_rule(n_points: int, dv: float):
    """fake_spectra's sub-pixel trapezoid (``singleabs.h:tau_kern_outer`` L133-L155): (offsets from the pixel centre
    [km/s], weights), a single point at the centre when ``n_points`` = 1."""
    if n_points == 1:
        return np.zeros(1), np.ones(1)
    w = np.ones(n_points) / (n_points - 1)
    w[[0, -1]] *= 0.5
    return (np.arange(n_points) / (n_points - 1) - 0.5) * dv, w


def _cell_values(delta, pk, kappa, order, detuning, integrate):
    """(C, M) optical depth of every cell at M velocities: the sum over its K pieces (``pk`` (C, K, 6): d, amp, sigma,
    gamma, half0, tilt; padded with amp = 0), ``delta`` (C, K, M) = v - v_c,p the detuning of every piece [km/s].
    Points (``integrate`` False): the profile at the line centre, Harris' expansion of ``order`` (1, 4; 0: the exact
    function); else the mean over the piece's line centres (``voigt_integral.harris_mean``) with its tilt."""
    amp, sigma, gamma, half0, tilt = (pk[..., i][:, :, None] for i in range(1, 6))
    x = -delta * (1.0 / phys.c_kms)
    em1 = _expm1_small(x) if detuning == "exact" else None
    nu = ((kappa * phys.c_kms) * em1) if detuning == "exact" else (-kappa * delta)
    s2 = np.sqrt(2.0) * sigma
    a = gamma / s2
    if not integrate:
        u = nu / s2
        H = _exact_H(u, a) if order == 0 else (harris_H(u, a) if order == 1 else harris_H4(u, a))
        return ((amp / (np.sqrt(np.pi) * s2)) * H).sum(axis=1)
    from skewerjax.core.voigt_integral import harris_mean

    e = (1.0 + em1) if detuning == "exact" else 1.0
    h = half0 * e
    u = nu / s2 + (h * half0 * (s2 / (2.0 * kappa * phys.c_kms)) if detuning == "exact" else 0.0)
    M = jnp.where(a <= HARRIS_A_MAX, harris_mean(u - h, u + h, a), harris_mean(u - h, u + h, a, fourth=True))
    g = tilt - (1.0 / phys.c_kms if detuning == "exact" else 0.0)
    hh = jnp.maximum(h, 1e-30)
    M1 = jnp.where(h >= NARROW_HALF_WIDTH, gaussian_moment_wide(u - hh, u + hh), gaussian_moment_narrow(u, h))
    M = M + g * (s2 / kappa / e) * M1
    return ((amp / (np.sqrt(np.pi) * s2)) * M).sum(axis=1)


@partial(
    jax.jit,
    static_argnames=("order", "detuning", "n_v", "w_back", "w_fwd", "half", "periodic", "walk", "integrate", "jc", "chord"),
    donate_argnums=(0,),
)
def _cell_kernel(
    out,
    pk,
    ik,
    P,
    consts,
    off,
    wts,
    mom,
    *,
    order,
    detuning,
    n_v,
    w_back,
    w_fwd,
    half,
    periodic,
    walk,
    integrate,
    jc=0,
    chord=False,
):
    """One chunk of C cells of K pieces each: ``pk`` (C, K, 6) the pieces (d [km/s]: the line centre relative to the
    velocity of the cell's centre pixel; amp, sigma, gamma [GHz]; half0; tilt), ``ik`` (C, 2) the centre pixel and the
    skewer of every cell, ``P`` (C,) the periods or None, ``consts`` (dv, kappa, tautail). Returns (the windows (C, W),
    or with ``out`` (n_sk n_v,) their scatter-add into it (GPUs); the flags of the cells whose walk reached the window's
    edge above tautail: left out, done again with a wider window). Cell c covers the pixels centre - w_back ... centre +
    w_fwd - 1 (wrapped modulo n_v; ``walk`` False: the whole grid from pixel 0), each at its sub-pixel velocities
    (``off``, weights ``wts``). With ``jc`` > 0 only the core |j| < jc is evaluated piece by piece; beyond, where every
    piece of the cell is in its Lorentz wing, the cell's wing comes from its moments ``mom`` (C, 4) (the far series of
    ``windowed_line_tau``: one evaluation instead of K, at the pixel centre)."""
    dv, kappa, tautail = consts[0], consts[1], consts[2]
    if chord:  # (C, 1, 6) chords: their 7 points (d + (k - 4) spacing, amp / 7), spacing in the half0 slot
        k = (jnp.arange(7, dtype=pk.dtype) - 3.0)[None, :]
        base = pk[:, 0, :]
        pk = jnp.broadcast_to(base[:, None, :], (base.shape[0], 7, 6))
        pk = pk.at[:, :, 0].set(base[:, 0:1] + k * base[:, 4:5]).at[:, :, 1].set(base[:, 1:2] / 7.0).at[:, :, 4].set(0.0)
    centre, row = ik[:, 0], ik[:, 1]
    n_cells = centre.shape[0]
    offs = jnp.arange(-w_back, w_fwd)
    cell_pix = centre[:, None] + offs[None, :]
    pix = jnp.mod(cell_pix, n_v)
    n_pts = wts.shape[0]
    step = (offs[None, :] if periodic else pix - centre[:, None]).astype(off.dtype) * dv
    step = jnp.broadcast_to(step, (n_cells, offs.size))
    delta = (step[:, :, None] + off[None, None, :]).reshape(n_cells, 1, -1) - pk[..., 0][:, :, None]  # (C, K, W n_pts)
    if periodic:  # the nearest image
        Pc = P[:, None, None]
        delta = jnp.where(delta >= 0.5 * Pc, delta - Pc, jnp.where(delta < -0.5 * Pc, delta + Pc, delta))
    if jc and jc < min(w_back, w_fwd):  # the core piece by piece, the wings from the moments
        core = slice(w_back - jc, w_back + jc)
        d_core = delta.reshape(n_cells, pk.shape[1], offs.size, n_pts)[:, :, core].reshape(n_cells, pk.shape[1], -1)
        vals = _cell_values(d_core, pk, kappa, order, detuning, integrate)
        core_tau = (vals.reshape(n_cells, 2 * jc, n_pts) * wts).sum(-1)
        jf = (offs.astype(off.dtype) * dv)[None, :]
        k0, k1, k2, k3 = far_kernels(jf, detuning)
        wing = mom[:, 0:1] * k0 + mom[:, 1:2] * k1 + mom[:, 2:3] * k2 + mom[:, 3:4] * k3  # (C, W)
        cell_tau = jnp.concatenate([wing[:, : w_back - jc], core_tau, wing[:, w_back + jc :]], axis=1)
    else:
        vals = _cell_values(delta, pk, kappa, order, detuning, integrate)
        cell_tau = (vals.reshape(n_cells, offs.size, n_pts) * wts).sum(-1)  # (C, W)
    if walk:  # fake_spectra keeps every pixel up to and including the first one below tautail, each way
        o = offs[None, :]
        below = cell_tau < tautail
        ff = jnp.min(jnp.where(below & (o >= 0), o, w_fwd), axis=1)  # the first pixel below, ahead (w_fwd: none)
        lb = jnp.max(jnp.where(below & (o < 0), o, -w_back - 1), axis=1)  # the first pixel below, behind (-w_back - 1: none)
        flag = ((w_fwd < half) & (ff == w_fwd)) | ((w_back < half) & (lb == -w_back - 1))
        keep = jnp.where(o >= 0, o <= ff[:, None], o >= lb[:, None]) & (~flag)[:, None]
        if not periodic:
            keep = keep & (cell_pix >= 0) & (cell_pix < n_v)
        vals = jnp.where(keep, cell_tau, 0.0)
    else:
        flag = jnp.zeros(n_cells, bool)
        vals = cell_tau if periodic else jnp.where((cell_pix >= 0) & (cell_pix < n_v), cell_tau, 0.0)
    if out is None:  # the CPU: the host adds the windows (``_accumulator``)
        return vals, flag
    return out.at[(row[:, None] * n_v + pix).ravel()].add(vals.ravel()), flag


def _window_estimate(tau0, u_wing, spread_kms, b_kms, dv, tautail) -> np.ndarray:
    """(N_cells,) half width [pixels] that holds each cell's walk, from its peak tau0, its damping wing (u_wing: tau0 a
    / sqrt(pi) per tautail) and the spread of its pieces' line centres (an estimate: a cell whose walk goes further is
    flagged and done again)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        u_core = np.sqrt(np.maximum(np.log(np.maximum(tau0, 1e-300) / tautail), 0.0))
        u_stop = np.maximum(u_core, np.sqrt(np.maximum(u_wing, 0.0)))
    return np.ceil(np.minimum(np.nan_to_num((u_stop * b_kms + spread_kms) / dv, nan=1e9, posinf=1e9), 1e9)).astype(np.int64) + 2


def per_cell_line_tau(
    c: LineContributions,
    v,
    n_sk: int,
    lambda0_A: float,
    *,
    period=None,
    profile: str = "integrated",
    detuning: str = "exact",
    voigt: str = "harris",
    tautail: float | None = None,
    pixel_average: bool = False,
    dtype=None,
    devices=None,
    levels=None,
):
    """tau (n_sk, N_v) of one line with fake_spectra's per-cell schemes (module docstring): the pieces of one cell
    (``c.group``, per skewer) form one absorber, centred on the pixel of its column-weighted line centre; ``tautail``:
    its early stop (``absorption.cpp`` L253-L281) on windows escalated until the walk stops inside (the full-grid
    result); ``pixel_average``: the trapezoid over each pixel for cells of Doppler width b < 2 dv (``singleabs.h``
    L133-L155). Uniform grid; ``period`` (n_sk,) must be the grid's span with ``tautail``."""
    v0, dv, n_v = _grid(v)
    levels = np.asarray(WINDOW_LEVELS if levels is None else levels)
    kappa = 1e4 / float(lambda0_A)
    dt = _dtype(dtype)
    devs = list(devices) if devices else [None]
    puts = [(lambda x: jnp.asarray(x)) if d is None else (lambda x, d=d: jax.device_put(x, d)) for d in devs]
    n_rows = _pow2(n_sk)
    host = all((d.platform == "cpu") if d is not None else jax.default_backend() == "cpu" for d in devs)
    outs = [np.zeros((n_rows, n_v), np.float64)] if host else [p(np.zeros(n_rows * n_v, dt)) for p in puts]
    P_sk = None if period is None else np.broadcast_to(np.asarray(period, dtype=np.float64), (n_sk,))
    periodic = P_sk is not None
    if tautail is not None and periodic and not np.allclose(P_sk, n_v * dv, rtol=1e-6, atol=0.0):
        raise ValueError(
            f"tautail walks the pixels modulo the grid: velocity_period {P_sk} must be its span {n_v * dv} (N_v dv)."
        )
    half = n_v // 2 if periodic else n_v
    live = np.flatnonzero(c.amp != 0)
    if not live.size:
        return puts[0](np.zeros((n_sk, n_v), dt))
    c = c.take(live)
    chord = c.spacing is not None
    integrate = profile == "integrated" and bool(np.any(c.width > 0)) and not chord
    width = c.width if integrate else np.zeros(len(c))
    tilt = np.zeros(len(c)) if c.tilt is None else c.tilt
    key = np.arange(len(c)) if c.group is None else np.asarray(c.group, dtype=np.int64)
    row_p = np.asarray(c.row, dtype=np.int64)
    vc = np.asarray(c.vc, dtype=np.float64)
    # the cells: runs of pieces of one (skewer, cell); sorted only when a cell's pieces are not consecutive
    if chord:  # every contribution is one cell (one chord per skewer): no grouping
        order = np.arange(len(c))
    else:
        brk = np.flatnonzero((row_p[1:] != row_p[:-1]) | (key[1:] != key[:-1])) + 1
        starts = np.concatenate([[0], brk])
        run_key = row_p[starts] * (int(key.max()) + 1) + key[starts]
        if np.unique(run_key).size != starts.size or (not periodic):
            order = np.lexsort((vc, key, row_p))
        else:
            order = np.arange(len(c))
    kk, kc, vc = row_p[order], key[order], vc[order]
    if chord:
        new = np.ones(max(kk.size - 1, 0), bool)
    else:
        new = (kc[1:] != kc[:-1]) | (kk[1:] != kk[:-1])
        if not periodic:  # a cell crossed at both ends of the path is two absorbers
            new |= np.diff(vc) > 0.5 * n_v * dv
    first = np.concatenate([[0], np.flatnonzero(new) + 1])
    sizes = np.diff(np.concatenate([first, [kk.size]]))
    owner = np.repeat(np.arange(first.size), sizes)
    row = kk[first]
    if periodic:  # the pieces of a cell on one image: the nearest to its first piece
        Pp = P_sk[kk]
        ref = vc[first][owner]
        vc = ref + np.mod(vc - ref + 0.5 * Pp, Pp) - 0.5 * Pp
    amp_p = c.amp[order]
    col = np.abs(amp_p)
    wcol = np.bincount(owner, col, first.size)
    vc_cell = np.bincount(owner, vc * col, first.size) / np.maximum(wcol, 1e-300)
    centre = np.floor((vc_cell - v0 + 0.5 * dv) / dv).astype(np.int64)
    centre = centre if periodic else np.clip(centre, 0, n_v - 1)
    sig_p, gam_p = c.sigma[order], c.gamma[order]
    sig_f = sig_p[first]
    b_f = np.sqrt(2.0) * sig_f / kappa  # Doppler width [km/s]
    a_f = gam_p[first] / (np.sqrt(2.0) * sig_f)
    pts = np.ones(first.size, np.int64)
    if pixel_average:
        pts = np.where(dv < 0.5 * b_f, 1, 2 * np.ceil(dv / b_f).astype(np.int64) + 1)
    if tautail is not None:
        tau0 = np.bincount(owner, amp_p, first.size) / (np.sqrt(2.0 * np.pi) * sig_f)
        u_wing = tau0 * a_f / (np.sqrt(np.pi) * tautail)
        spread = np.maximum.reduceat(np.abs(vc - vc_cell[owner]) + 0.5 * width[order], first)
        if chord:
            spread = spread + 3.0 * np.abs(np.asarray(c.spacing, dtype=np.float64)[order])
        est = _window_estimate(tau0, u_wing, spread, b_f, dv, tautail)
        level = np.minimum(np.searchsorted(levels, est), levels.size - 1)
    else:
        level = np.full(first.size, -1)
    v_centre = v0 + centre.astype(np.float64) * dv
    half_or_spacing = (
        np.asarray(c.spacing, dtype=np.float64)[order] if chord else kappa * width[order] / (2.0 * np.sqrt(2.0) * sig_p)
    )
    pk_all = np.stack([vc - v_centre[owner], amp_p, sig_p, gam_p, half_or_spacing, tilt[order]], axis=-1)
    d0 = vc - v0  # relative to the first pixel (no walk: the whole grid from pixel 0)
    # the core of every cell, where its pieces are evaluated one by one: CORE_U Doppler widths beyond its extreme line
    # centres; beyond, its Lorentz wing from its moments about the centre pixel (``_cell_kernel``)
    t_p = pk_all[:, 0]
    w_p = amp_p * gam_p / (np.pi * kappa * kappa)
    b_p = np.sqrt(2.0) * sig_p / kappa
    spread_c = np.maximum.reduceat(np.abs(t_p) + 0.5 * width[order] + (3.0 * np.abs(pk_all[:, 4]) if chord else 0.0), first)
    jc_need = np.ceil((CORE_U * np.maximum.reduceat(b_p, first) + spread_c) / dv).astype(np.int64) + 2
    t2 = t_p**2 + width[order] ** 2 / 12.0 + (4.0 * pk_all[:, 4] ** 2 if chord else 0.0)  # chords: <(k - 4)^2> = 4 spacings^2
    mom_all = np.stack(
        [
            np.bincount(owner, w_p, first.size),
            np.bincount(owner, w_p * t_p, first.size),
            np.bincount(owner, w_p * t2, first.size),
            np.bincount(owner, 1.5 * w_p * b_p**2, first.size),
        ],
        axis=-1,
    )
    jc_level = np.searchsorted(np.asarray(CORE_LEVELS), jc_need)  # len(CORE_LEVELS): no split
    if integrate or str(voigt).lower() == "exojax" or detuning not in ("exact", "linear"):
        jc_level[:] = len(CORE_LEVELS)
    K_of = (1 << np.ceil(np.log2(np.maximum(sizes, 1))).astype(np.int64)).astype(np.int64)
    K_of = np.where(sizes == 7, 7, K_of)  # fake_spectra's chords: 7 points each, unpadded
    if chord:
        K_of = np.ones(first.size, np.int64)  # one row of (centre, spacing): the kernel generates the 7 points
    if str(voigt).lower() == "exojax":
        vorder = np.zeros(first.size, np.int64)
    else:
        vorder = np.where(a_f <= HARRIS_A_MAX, 1, np.where(a_f <= HARRIS4_A_MAX, 4, 0))
    state = dict(next=0)
    for n_points in np.unique(pts):
        off, wts = pixel_rule(int(n_points), dv)
        for K in np.unique(K_of):
            for od in np.unique(vorder):
                todo = np.flatnonzero((pts == n_points) & (K_of == K) & (vorder == od))
                lev = level[todo].copy()
                while todo.size:
                    nxt_lev = lev.copy()
                    for L in np.unique(lev):
                        sel = lev == L
                        cells_L = todo[sel]
                        w = half if L >= levels.size - 1 else min(int(levels[L]), half)
                        wb, wf, walk = (0, n_v, False) if L < 0 else (w, w, True)
                        flags = np.zeros(cells_L.size, bool)
                        # the split core / wings for wide windows (by core width); narrow windows piece by piece
                        jl = np.where(walk and wb >= 256, jc_level[cells_L], len(CORE_LEVELS))
                        for J in np.unique(jl):
                            jc = 0 if J >= len(CORE_LEVELS) or CORE_LEVELS[J] >= wb else int(CORE_LEVELS[J])
                            sub = jl == J
                            flags[sub] = _run_cells(
                                cells_L[sub],
                                first,
                                sizes,
                                centre,
                                row,
                                P_sk,
                                pk_all,
                                d0,
                                outs,
                                puts,
                                state,
                                dt,
                                off,
                                wts,
                                dv,
                                kappa,
                                tautail,
                                mom_all,
                                K=int(K),
                                order=int(od),
                                detuning=detuning,
                                n_v=n_v,
                                w_back=wb,
                                w_fwd=wf,
                                half=half,
                                periodic=periodic,
                                walk=walk,
                                integrate=integrate,
                                jc=jc,
                                chord=chord,
                            )
                        nxt_lev[np.flatnonzero(sel)[flags]] = L + 1
                        nxt_lev[np.flatnonzero(sel)[~flags]] = -2
                    keep = nxt_lev != -2
                    todo, lev = todo[keep], nxt_lev[keep]
    if host:
        return puts[0](outs[0][:n_sk].astype(dt))
    total = outs[0]
    for o in outs[1:]:
        total = total + (o if devs[0] is None else jax.device_put(o, devs[0]))
    return total.reshape(n_rows, n_v)[:n_sk]


def _run_cells(
    cells_L,
    first,
    sizes,
    centre,
    row,
    P_sk,
    pk_all,
    d0,
    outs,
    puts,
    state,
    dt,
    off,
    wts,
    dv,
    kappa,
    tautail,
    mom_all,
    *,
    K,
    order,
    detuning,
    n_v,
    w_back,
    w_fwd,
    half,
    periodic,
    walk,
    integrate,
    jc=0,
    chord=False,
):
    """The cells ``cells_L`` (K pieces at most, one Voigt order) at one window, in chunks of C cells (padded; one
    compilation per window, K, C and order), sorted by skewer within each chunk, dealt in turn to the devices; returns
    the flags of the walks that reached the window's edge."""
    width = w_back + w_fwd
    host = isinstance(outs[0], np.ndarray)
    Kw = 7 if chord else K
    work = (2 * jc * int(off.size) * Kw + width) if jc else width * int(off.size) * Kw
    C = _pow2(max(EVALUATIONS_PER_CALL // work, 1))
    flags = np.zeros(cells_L.size, bool)
    slot = np.arange(K)[None, :]
    consts_cache = {}
    pending = []
    mode = (0 if periodic else 1) if walk else 2
    # chunks that span many skewers (the cells dealt round-robin over the skewers, by rank within their skewer): the host
    # accumulates a chunk in parallel over its skewers; within a chunk the cells are sorted by skewer (below)
    r_L = row[cells_L]
    o_r = np.argsort(r_L, kind="stable")
    rs = r_L[o_r]
    rank = np.arange(rs.size) - np.searchsorted(rs, rs, side="left")
    o_all = o_r[np.lexsort((rs, rank))]
    for c0 in range(0, cells_L.size, C):
        part = o_all[c0 : c0 + C]
        part = part[np.argsort(row[cells_L[part]], kind="stable")]
        g = cells_L[part]
        m = g.size
        idx = first[g][:, None] + slot
        live = slot < (np.ones(m, np.int64) if chord else sizes[g])[:, None]
        idx = np.where(live, idx, first[g][:, None])
        pk = np.zeros((C, K, 6), dt)
        pk[:m] = pk_all[idx]
        pk[:m, :, 1] *= live
        if not walk:
            pk[:m, :, 0] = d0[idx]
        if m < C:
            pk[m:] = pk[m - 1]
            pk[m:, :, 1] = 0.0
        cen = centre[g] if walk else np.zeros(m, np.int64)
        ik = np.zeros((C, 2), np.int32)
        ik[:m, 0], ik[:m, 1] = cen, row[g]
        i = state["next"]
        state["next"] = (i + 1) % len(puts)
        put = puts[i]
        if i not in consts_cache:
            consts_cache[i] = (
                put(np.array([dv, kappa, 0.0 if tautail is None else tautail], dt)),
                put(np.asarray(off, dt)),
                put(np.asarray(wts, dt)),
            )
        cst, offd, wtsd = consts_cache[i]
        Pc = None
        if periodic:
            Pp = np.zeros(C, dt)
            Pp[:m] = P_sk[row[g]]
            Pp[m:] = Pp[m - 1] if m else 1.0
            Pc = put(Pp)
        mom = np.zeros((C, 4), dt)
        mom[:m] = mom_all[g]
        res, f = _cell_kernel(
            None if host else outs[i],
            put(pk),
            put(ik),
            Pc,
            cst,
            offd,
            wtsd,
            put(mom),
            order=order,
            detuning=detuning,
            n_v=n_v,
            w_back=w_back,
            w_fwd=w_fwd,
            half=half,
            periodic=periodic,
            walk=walk,
            integrate=integrate,
            jc=jc,
            chord=chord,
        )
        if host:
            pending.append((f, part, m, res, row[g].astype(np.int64), cen.astype(np.int64)))
            while len(pending) > 2:
                _drain_cells(pending.pop(0), outs[0], flags, w_back, mode)
        else:
            outs[i] = res
            pending.append((f, part, m, None, None, None))
    for item in pending:
        if item[3] is None:
            flags[item[1]] = np.asarray(item[0])[: item[2]]
        else:
            _drain_cells(item, outs[0], flags, w_back, mode)
    return flags


def _drain_cells(item, acc_out, flags, w_back, mode):
    """Adds one chunk of per-cell windows to the host's optical depth (flagged cells are zero) and records the flags."""
    f, part, m, vals, rows, cen = item
    flags[part] = np.asarray(f)[:m]
    v = np.asarray(vals)[:m]
    rstart = np.flatnonzero(np.diff(np.concatenate([[-1], rows, [-2]]))).astype(np.int64)
    _accumulator()(acc_out, v, rows, cen, rstart, w_back, mode)


# --------------------------------------------------------------------------- from Pieces
def contributions_from_pieces(
    pieces,
    j: int,
    line_bundle,
    *,
    a_eff=None,
    b2_ratio: float = 1.0,
    hubble_flow: str = "exact",
    aH_ref=None,
    profile: str = "integrated",
) -> LineContributions:
    """The contributions of line ``j`` from ``core.Pieces`` (host float64; the live pieces only): v_c = V_H + the
    peculiar part (``spectral.line_centre_velocity_kms``), W = ``width_kms`` (|(aH)_ref| l with ``hubble_flow="linear"``),
    amp = C f n l a^-2 (``a_eff``), the widths with ``b2_ratio`` x T, the tilt from ``Pieces.tilt_a2`` (dropped with
    ``a_eff``) and ``Pieces.tilt_flow`` (dropped with ``hubble_flow="linear"``), the group ``Pieces.cell``."""
    from skewerjax.core.optical_depth import _hubble_rates

    f = lambda x: None if x is None else np.asarray(x, dtype=np.float64)
    n = f(pieces.number_density[j])
    ln = f(pieces.length_m)
    kk, pp = np.nonzero((n != 0.0) & (ln > 0.0))
    a = f(pieces.scale_factor)[kk, pp]
    los = f(pieces.los_velocity[j])[kk, pp]
    beta = los / phys.c_kms
    half_log = phys.c_kms * 0.5 * (np.log1p(-beta) - np.log1p(beta))
    if hubble_flow == "linear":
        aH = np.asarray(_hubble_rates(pieces, aH_ref), dtype=np.float64)
        h0 = np.zeros(aH.shape) if pieces.vH_first is None else f(pieces.vH_first)
        hub = h0[kk] + aH[kk] * f(pieces.path_cMpc)[kk, pp]
        W = np.abs(aH[kk]) * ln[kk, pp] / phys.Mpc_to_m
    else:
        hub = (
            f(pieces.hubble_kms)[kk, pp]
            if pieces.hubble_kms is not None
            else phys.c_kms * np.log(f(pieces.a_dagger)[kk if np.ndim(pieces.a_dagger) == 1 else (kk, pp)] / a)
        )
        W = f(pieces.width_kms[j])[kk, pp]
    a2 = a**-2 if a_eff is None else np.full(a.shape, float(a_eff) ** -2)
    amp = (phys.cross_section_prefactor * 1e2) * float(np.asarray(line_bundle.f_osc)[j]) * n[kk, pp] * ln[kk, pp] * a2
    sig, gam = line_widths(
        f(pieces.temperature[j])[kk, pp] * b2_ratio,
        f(pieces.turbulence[j])[kk, pp],
        f(pieces.collisional[j])[kk, pp],
        float(np.asarray(line_bundle.lambda0_A)[j]),
        float(np.asarray(line_bundle.A21_GHz)[j]),
        float(np.asarray(line_bundle.mass_amu)[j]),
    )
    tilt = np.zeros(kk.size)
    if getattr(pieces, "tilt_flow", None) is not None and hubble_flow != "linear":  # a linear flow has no curvature
        tilt = tilt + f(pieces.tilt_flow)[kk, pp]
    if a_eff is None and getattr(pieces, "tilt_a2", None) is not None:
        tilt = tilt + f(pieces.tilt_a2)[kk, pp]
    group = None if pieces.cell is None else np.asarray(pieces.cell)[kk, pp]
    return LineContributions(
        kk, hub + half_log, W if profile == "integrated" else np.zeros(kk.size), amp, sig, np.ascontiguousarray(gam), group, tilt
    )


def warm_up(
    line_bundle,
    v,
    *,
    n_sk: int = 1024,
    devices=None,
    dtype=None,
    detuning: str = "exact",
    profile: str = "integrated",
    periodic: bool = False,
    max_level: int | None = None,
    threads: int = 8,
) -> float:
    """Compile the window kernels a run is likely to call (every level up to ``max_level`` (default: what the grid
    allows), the kinds of ``profile``, the Voigt orders of the lines), on parallel threads; returns the wall time [s].
    The kernels then run without compiling (JAX's in-memory cache of this process; nothing is stored on disk)."""
    from concurrent.futures import ThreadPoolExecutor

    from skewerjax.config import real_dtype

    t0 = time.perf_counter()
    dt = real_dtype() if dtype is None else np.dtype(dtype)
    vv = np.atleast_2d(np.asarray(v, dtype=np.float64))
    n_v = vv.shape[-1]
    devs = list(devices) if devices else [None]
    top = (n_v // 2 - 1) if periodic else (n_v - 1)
    top = top if max_level is None else min(top, int(max_level))
    kinds = ("point",) if profile == "midpoint" else ("narrow", "wide")
    orders = {1, 4}
    jobs = []
    for D in LEVELS:
        if D > top:
            break
        for kind in kinds:
            for order in orders:
                for d in devs:
                    jobs.append((D, kind, order, d))

    def one(job):
        D, kind, order, d = job
        C = chunk_size(2 * D + 1)
        put = (lambda x: jnp.asarray(x)) if d is None else (lambda x: jax.device_put(x, d))
        f = lambda: put(np.zeros(C, dt))
        i = lambda: put(np.zeros(C, np.int32))
        out = put(np.zeros(_pow2(n_sk) * n_v, dt))
        sig = put(np.ones(C, dt))
        _window_kernel(
            out,
            f(),
            f(),
            f(),
            sig,
            f(),
            f(),
            i(),
            i(),
            None,
            put(np.zeros((4, C), dt)),
            put(np.asarray(1.0, dt)),
            put(np.asarray(1.0, dt)),
            put(np.asarray(0.0, dt)),
            D=D,
            kind=kind,
            order=order,
            detuning=detuning,
            mode="circular" if periodic else "linear",
            n_v=n_v,
            ring=True,
        ).block_until_ready()

    with ThreadPoolExecutor(max_workers=max(1, threads)) as pool:
        list(pool.map(one, jobs))
    return time.perf_counter() - t0
