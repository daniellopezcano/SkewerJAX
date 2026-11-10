"""fake_spectra's per-cell schemes of the optical depth, windowed and batched over the skewers.

Two options of ``core.compute_optical_depth`` evaluate the sum per **cell** instead of per piece, as fake_spectra's C
engine does (``absorption.cpp:add_tau_particle`` L215-L282): the pieces of one cell form one absorber, whose line centre
is the column-weighted mean of its pieces'.

* ``pixel_average``: a cell of Doppler width b < 2 dv is averaged over each pixel by the trapezoid on 2 ceil(dv / b) + 1
  points (``singleabs.h:tau_kern_outer`` L133-L155; ``pixel_rule``), the others take the profile at the pixel centre;
* ``tautail``: the early stop (L253-L281): walking out from the pixel of the cell's line centre, forward and backward
  (at most half the grid each way when periodic), the cell adds a pixel up to and including the first one that
  receives less than ``tautail`` from it.

**Windows.** The walk makes each cell's support finite, so a cell is evaluated only on a window of
pixels around its centre pixel, not on the whole grid: the window is sized from the cell's peak optical depth and its
damping wing (``_window_estimate``), rounded up to a level of ``WINDOW_LEVELS``. A cell whose walk reaches the edge of
its window while still above ``tautail`` is flagged and evaluated again with the next level, until its walk stops inside
the window or the window holds the whole walk (half the grid). The result is therefore the walk on the full grid,
whatever the estimate (which only sets the cost). Without ``tautail`` every cell covers the whole grid (no window).

The cells of every skewer of one line are processed together (one jitted kernel call per chunk of cells of one window
level and one number of sub-pixel points), so the number of calls does not grow with the number of skewers. The host
work (grouping the pieces by cell, the line centres, the windows) is vectorised NumPy in float64.
"""

from __future__ import annotations

import logging
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from skewerjax.core.constants import phys
from skewerjax.core.line_profiles import voigt_line_profiles, voigt_widths_GHz

logger = logging.getLogger(__name__)
WINDOW_LEVELS = (32, 128, 512, 2048, 8192, 32768)  # half-widths [pixels] of the windows, each 4x the previous
EVALUATIONS_PER_CALL = 1 << 22  # profile evaluations per kernel call (memory ~ 4 x 4M x 4 bytes in float32)


def pixel_rule(n_points: int, dv: float):
    """fake_spectra's sub-pixel trapezoid (``singleabs.h:tau_kern_outer`` L133-L155): (offsets from the pixel centre
    [km/s], weights), a single point at the centre when ``n_points`` = 1."""
    if n_points == 1:
        return np.zeros(1), np.ones(1)
    w = np.ones(n_points) / (n_points - 1)
    w[[0, -1]] *= 0.5
    return (np.arange(n_points) / (n_points - 1) - 0.5) * dv, w


def _pow2(n: int) -> int:
    return 1 << max(int(n) - 1, 0).bit_length()


def _cell_values(delta, lam, A21, mass, f_osc, a, T, tb, co, n, length, a_eff, voigt, detuning):
    """(C, M) optical depth of every cell c at M velocities: the sum over its K pieces (the per-piece arrays (C, K),
    padded with n = 0) of the terms of ``core.optical_depth.optical_depth_kernel`` (one line: lam, A21, mass, f_osc
    scalars; per piece: a, T, turbulence, collisional, comoving density, length [m]), ``delta`` (C, K, M) = v - v_c,p
    the detuning of every piece at those velocities [km/s] (its nearest periodic image)."""
    C, K = T.shape
    one = lambda x: jnp.reshape(x, (1,))
    sigma, gamma = voigt_widths_GHz(T.reshape(1, -1), tb.reshape(1, -1), co.reshape(1, -1), one(lam), one(A21), one(mass))
    lam, f_osc = jnp.reshape(lam, ()), jnp.reshape(f_osc, ())
    x = -delta / phys.c_kms
    factor = jnp.expm1(x) if detuning == "exact" else x
    nu = (factor * ((phys.c * 10.0) / lam)).reshape(C * K, -1)
    phi = voigt_line_profiles(nu, sigma[0], gamma[0], voigt).reshape(C, K, -1)
    a2 = a**-2 if a_eff is None else jnp.broadcast_to(jnp.asarray(a_eff, a.dtype) ** -2, a.shape)
    amp = (phys.cross_section_prefactor * 1e2) * f_osc * (n * (length * a2))  # (C, K)
    return (amp[:, :, None] * phi).sum(axis=1)  # a fused reduction (a batched dot of K ~ 8 is slow on the CPU)


@partial(jax.jit, static_argnames=("voigt", "detuning", "n_v", "n_out", "w_back", "w_fwd", "half", "periodic", "walk"))
def _window_kernel(
    dv,
    off,
    wts,
    lam,
    A21,
    mass,
    f_osc,
    d,
    a,
    T,
    tb,
    co,
    n,
    length,
    P,
    a_eff,
    centre,
    row,
    tautail,
    *,
    voigt,
    detuning,
    n_v,
    n_out,
    w_back,
    w_fwd,
    half,
    periodic,
    walk,
):
    """One chunk of C cells of at most K pieces each (the per-piece arrays (C, K)): (out (n_out,), flag (C,)). Cell c
    covers the pixels centre_c - w_back ... centre_c + w_fwd - 1 (indices wrapped modulo n_v; with ``walk=False`` the
    whole grid, w_back = 0, centre = 0), each at its sub-pixel velocities (``off``, weights ``wts``). ``d`` (C, K): the
    line centre of every piece relative to the velocity of its cell's centre pixel [km/s] (float64 on the host), so the
    detuning (pixel - centre) dv + off - d is formed from small numbers: no gather, and no rounding of the ~1e3 km/s
    velocities in float32. With ``walk``, fake_spectra's early stop. ``flag``: the walk reached the edge of the window
    above ``tautail`` (the cell must be done again with a wider one; its contribution is left out). ``out`` holds the sum
    at row_c n_v + pixel (the one scatter)."""
    n_cells = centre.shape[0]
    offs = jnp.arange(-w_back, w_fwd)
    cell_pix = centre[:, None] + offs[None, :]  # (C, W) unwrapped
    pix = jnp.mod(cell_pix, n_v)
    n_pts = wts.shape[0]
    # the pixel's velocity minus the centre pixel's: offs dv (periodic: the nearest image of the window offset); without a
    # period the walk reads the pixels of the other end at their own velocities (pix - centre), as fake_spectra's indices do
    step = (offs[None, :] if periodic else pix - centre[:, None]).astype(off.dtype) * dv
    delta = (step[:, :, None] + off[None, None, :]).reshape(step.shape[0], 1, -1) - d[:, :, None]  # (C, K, W n_pts)
    if periodic:  # the nearest image, as the mod of the dense kernel: |delta| exceeds P/2 by at most the spread of d
        Pc = P[:, None, None]
        delta = jnp.where(delta >= 0.5 * Pc, delta - Pc, jnp.where(delta < -0.5 * Pc, delta + Pc, delta))
    vals = _cell_values(delta, lam, A21, mass, f_osc, a, T, tb, co, n, length, a_eff, voigt, detuning)
    cell_tau = (vals.reshape(n_cells, offs.size, n_pts) * wts).sum(-1)  # (C, W)
    if walk:
        fwd, bwd = cell_tau[:, w_back:], cell_tau[:, :w_back][:, ::-1]  # steps 0, 1, ... and -1, -2, ...
        ok_f, ok_b = fwd >= tautail, bwd >= tautail
        one = jnp.ones((n_cells, 1), cell_tau.dtype)
        keep_f = jnp.concatenate([one, jnp.cumprod(ok_f[:, :-1], axis=1).astype(cell_tau.dtype)], axis=1)
        keep_b = jnp.concatenate([one, jnp.cumprod(ok_b[:, :-1], axis=1).astype(cell_tau.dtype)], axis=1)
        flag = ((w_fwd < half) & jnp.all(ok_f, axis=1)) | ((w_back < half) & jnp.all(ok_b, axis=1))
        keep = jnp.concatenate([keep_b[:, ::-1], keep_f], axis=1) * (~flag)[:, None]
    else:
        flag = jnp.zeros(n_cells, bool)
        keep = jnp.ones_like(cell_tau)
    inside = jnp.ones_like(keep) if periodic else ((cell_pix >= 0) & (cell_pix < n_v)).astype(keep.dtype)
    contrib = cell_tau * keep * inside
    out = jnp.zeros(n_out, cell_tau.dtype).at[(row[:, None] * n_v + pix).ravel()].add(contrib.ravel())
    return out, flag


def _window_estimate(tau0, u_wing, spread_kms, to_kms_sigma, dv, tautail) -> np.ndarray:
    """(N_cells,) half-width [pixels] that holds each cell's walk, from its peak tau0 = sum_p amp_p / (sqrt(2 pi) sigma_p),
    its damping parameter (u_wing: tau0 a / sqrt(pi) per tautail) and the spread of its pieces' line centres: the
    Doppler core exp(-u^2) and the Lorentz wing a / (sqrt(pi) u^2) of H(a, u) fall below ``tautail`` at u_stop, i.e. at
    sqrt(2) sigma u_stop in velocity. An estimate: a cell whose walk goes further is flagged and done again."""
    with np.errstate(divide="ignore", invalid="ignore"):
        u_core = np.sqrt(np.maximum(np.log(np.maximum(tau0, 1e-300) / tautail), 0.0))
        u_stop = np.maximum(u_core, np.sqrt(np.maximum(u_wing, 0.0)))
    return np.ceil((u_stop * to_kms_sigma + spread_kms) / dv).astype(np.int64) + 2


def per_cell_optical_depth(
    pieces,
    v,
    line_bundle,
    P,
    *,
    voigt,
    detuning,
    a_eff,
    hubble_flow,
    aH_ref,
    dtype,
    tautail,
    pixel_average,
    chunk=None,
    devices=None,
) -> jnp.ndarray:
    """tau (N_lines, N_sk, N_v) with fake_spectra's per-cell scheme (module docstring): ``pixel_average`` and / or
    ``tautail``. ``v`` (N_lines, N_v) the uniform grid of every line [km/s] (float64); ``P`` (N_sk,) the periods or None;
    ``dtype`` the kernel precision; ``devices``: None (the default device) or a list, over which the chunks of cells are
    dispatched in turn (asynchronously; the same result). See ``core.compute_optical_depth``; ``chunk`` is unused (the
    chunks follow ``EVALUATIONS_PER_CALL``)."""
    from skewerjax.core.optical_depth import _hubble_rates

    n_l, n_sk, n_p = pieces.shape
    vv = np.broadcast_to(np.asarray(v, dtype=np.float64), (n_l, np.shape(v)[-1]))
    n_v = vv.shape[1]
    dv = float(vv[0, 1] - vv[0, 0]) if n_v > 1 else 0.0
    if n_v < 2 or not np.allclose(np.diff(vv, axis=1), dv, rtol=1e-9, atol=0.0):
        raise ValueError("pixel_average and tautail need a uniform velocity grid of at least 2 pixels.")
    periodic = P is not None
    P_sk = None if P is None else np.asarray(P, dtype=np.float64)
    if tautail is not None:
        if pieces.cell is None:
            raise ValueError("tautail needs Pieces.cell, the cell of every piece (core.absorbers.cell_pieces fills it).")
        if periodic and not np.allclose(P_sk, n_v * dv, rtol=1e-6, atol=0.0):
            raise ValueError(
                f"tautail walks the pixels modulo the grid: velocity_period {P_sk} must be its span {n_v * dv} (N_v dv)."
            )
    half = n_v // 2 if periodic else n_v  # fake_spectra stops half a box away: beyond it the indices have wrapped
    dt = np.dtype(dtype)
    host = lambda f: None if getattr(pieces, f) is None else np.asarray(getattr(pieces, f), dtype=np.float64)
    a_d, a, ln, cells = host("a_dagger"), host("scale_factor"), host("length_m"), pieces.cell
    cells = None if cells is None else np.asarray(cells)
    if hubble_flow == "linear":
        aH = np.asarray(_hubble_rates(pieces, aH_ref), dtype=np.float64)
        h0 = np.zeros(n_sk) if pieces.vH_first is None else host("vH_first")
        hub = h0[:, None] + aH[:, None] * host("path_cMpc")
    else:
        hub = host("hubble_kms")  # None: c ln(a_dagger / a) of the rounded scale factors (Pieces built by hand)
    fields = {f: host(f) for f in ("temperature", "turbulence", "collisional", "los_velocity", "number_density")}
    lam, A21, mass, f_osc = (
        np.asarray(x, dtype=np.float64)
        for x in (line_bundle.lambda0_A, line_bundle.A21_GHz, line_bundle.mass_amu, line_bundle.f_osc)
    )
    out = np.zeros((n_l, n_sk * n_v))
    for j in range(n_l):
        T, tb, co, los, n = (fields[f][j] for f in ("temperature", "turbulence", "collisional", "los_velocity", "number_density"))
        kk, pp = np.nonzero((n != 0.0) & (ln > 0.0))
        if not kk.size:
            continue
        beta = los[kk, pp] / phys.c_kms
        half_log = 0.5 * (np.log1p(-beta) - np.log1p(beta))
        vc = phys.c_kms * (np.log(a_d[kk, pp] / a[kk, pp]) + half_log) if hub is None else hub[kk, pp] + phys.c_kms * half_log
        col = np.abs(n[kk, pp] * ln[kk, pp])
        key = pp if cells is None else cells[kk, pp]
        order = np.lexsort((vc, key, kk))
        kk, pp, vc, col, key = kk[order], pp[order], vc[order], col[order], key[order]
        new = (key[1:] != key[:-1]) | (kk[1:] != kk[:-1])
        if not periodic:  # a cell crossed at both ends of the path is two absorbers
            new |= np.diff(vc) > 0.5 * n_v * dv
        first = np.concatenate([[0], np.nonzero(new)[0] + 1])
        sizes = np.diff(np.concatenate([first, [kk.size]]))
        owner = np.repeat(np.arange(first.size), sizes)
        row = kk[first]  # the skewer of every cell
        if periodic:  # the pieces of a cell on one image: the nearest one to its first piece
            Pp = P_sk[kk]
            ref = vc[first][owner]
            vc = ref + np.mod(vc - ref + 0.5 * Pp, Pp) - 0.5 * Pp
        wcol = np.bincount(owner, col, first.size)
        vc_cell = np.bincount(owner, vc * col, first.size) / np.maximum(wcol, 1e-300)
        centre = np.floor((vc_cell - vv[j, 0] + 0.5 * dv) / dv).astype(np.int64)
        centre = centre if periodic else np.clip(centre, 0, n_v - 1)
        T_f, tb_f = T[kk[first], pp[first]], tb[kk[first], pp[first]]
        pts = np.ones(first.size, np.int64)
        if pixel_average:  # b = b_th + b_turb, as voigt_widths_GHz
            b = (np.sqrt(2.0 * phys.kB * T_f / mass[j]) + tb_f) / 1e3
            pts = np.where(dv < 0.5 * b, 1, 2 * np.ceil(dv / b).astype(np.int64) + 1)
        if tautail is not None:  # the window of every cell, as a level of WINDOW_LEVELS
            sig = (np.sqrt(2.0 * phys.kB * T_f / mass[j]) + tb_f) * 10.0 / (np.sqrt(2.0) * lam[j])  # sigma [GHz]
            gam = (A21[j] + co[kk[first], pp[first]]) / (4.0 * np.pi)
            amp = (
                (phys.cross_section_prefactor * 1e2)
                * f_osc[j]
                * wcol
                * (a[kk[first], pp[first]] ** -2 if a_eff is None else a_eff**-2)
            )
            tau0 = amp / (np.sqrt(2.0 * np.pi) * np.maximum(sig, 1e-300))
            u_wing = tau0 * (gam / (np.sqrt(2.0) * np.maximum(sig, 1e-300))) / (np.sqrt(np.pi) * tautail)
            spread = np.maximum.reduceat(np.abs(vc - vc_cell[owner]), first)
            est = _window_estimate(tau0, u_wing, spread, np.sqrt(2.0) * sig * lam[j] * 1e-4, dv, tautail)
            level = np.minimum(np.searchsorted(np.asarray(WINDOW_LEVELS), est), len(WINDOW_LEVELS) - 1)
        else:
            level = np.full(first.size, -1)  # the whole grid
        # the per-piece inputs of the kernel, in its precision
        cast = lambda x: jnp.asarray(np.ascontiguousarray(x), dt)
        # the line centre of every piece relative to the velocity of its cell's centre pixel (unwrapped), in float64
        v_centre = vv[j, 0] + centre.astype(np.float64) * dv if periodic else vv[j, np.clip(centre, 0, n_v - 1)]
        piece = dict(
            d=vc - v_centre[owner],
            d0=vc - vv[j, 0],
            a=a[kk, pp],
            T=T[kk, pp],
            tb=tb[kk, pp],
            co=co[kk, pp],
            n=n[kk, pp],
            length=ln[kk, pp],
        )
        P_cell = None if P_sk is None else P_sk[row]  # the period of every cell (its skewer's)
        consts = [cast(x[j]) for x in (lam, A21, mass, f_osc)]
        for n_points in np.unique(pts):
            off, wts = pixel_rule(int(n_points), dv)
            vsub = (cast(dv), cast(off))  # the uniform grid: (pixel - centre) dv + off
            todo = np.nonzero(pts == n_points)[0]
            lev = level[todo]
            while todo.size:
                for L in np.unique(lev):
                    cells_L = todo[lev == L]
                    t_lev = time.perf_counter()
                    w = half if L >= len(WINDOW_LEVELS) - 1 else min(WINDOW_LEVELS[L], half)  # the last level: the whole walk
                    wb, wf, walk = (0, n_v, False) if L < 0 else (w, w, True)
                    flags = _run_level(
                        cells_L,
                        first,
                        sizes,
                        centre,
                        row,
                        P_cell,
                        piece,
                        consts,
                        vsub,
                        cast(wts),
                        a_eff,
                        tautail,
                        out[j],
                        voigt=voigt,
                        detuning=detuning,
                        n_v=n_v,
                        n_out=n_sk * n_v,
                        w_back=wb,
                        w_fwd=wf,
                        half=half,
                        periodic=periodic,
                        walk=walk,
                        n_points=int(n_points),
                        cast=cast,
                        devices=devices,
                    )
                    lev[np.isin(todo, cells_L[flags])] = L + 1
                    lev[np.isin(todo, cells_L[~flags])] = -2  # done
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            "per_cell: line %d, %d sub-pixel points, window %s: %d cells, %d pieces, %.3g evaluations, %d flagged, %.3f s",
                            j,
                            n_points,
                            "full" if L < 0 else wb,
                            cells_L.size,
                            int(sizes[cells_L].sum()),
                            float(sizes[cells_L].sum()) * (wb + wf) * int(n_points),
                            int(flags.sum()),
                            time.perf_counter() - t_lev,
                        )
                keep = lev != -2
                todo, lev = todo[keep], lev[keep]
    return jnp.asarray(out.reshape(n_l, n_sk, n_v), dtype=dt)


def _run_level(
    cells_L,
    first,
    sizes,
    centre,
    row,
    P_cell,
    piece,
    consts,
    vsub,
    wts,
    a_eff,
    tautail,
    out_j,
    *,
    voigt,
    detuning,
    n_v,
    n_out,
    w_back,
    w_fwd,
    half,
    periodic,
    walk,
    n_points,
    cast,
    devices=None,
):
    """The cells ``cells_L`` at one window (w_back, w_fwd): grouped by their number of pieces (K, a power of two: the
    per-piece arrays of a chunk are (C, K), padded with empty pieces), in chunks of C cells (a power of two sized by
    ``EVALUATIONS_PER_CALL``; padded, so that each (window, n_points, K) compiles once per C), dispatched in turn to
    ``devices`` (asynchronously) and gathered at the end. Adds their sum to ``out_j`` (float64, on the host) and returns
    the (len(cells_L),) flags of the walks that reached the window's edge."""
    width = w_back + w_fwd
    devs = [None] if not devices else list(devices)
    shared = {}  # the arrays every chunk takes, once per device
    pending = []
    flags = np.zeros(cells_L.size, bool)
    K_of = np.array([_pow2(int(m)) for m in sizes[cells_L]])
    T_cell = piece["T"][first[cells_L]]
    for K in np.unique(K_of):
        sel = np.nonzero(K_of == K)[0]
        # by temperature: chunks of similar damping parameters, so that voigt="harris" takes one branch per chunk (one cold
        # piece, a > 1e-3, would make a whole chunk evaluate two expansions)
        sel = sel[np.argsort(-T_cell[sel], kind="stable")]
        C = _pow2(max(EVALUATIONS_PER_CALL // (width * int(n_points) * int(K)), 1))
        for c0 in range(0, sel.size, C):
            part = sel[c0 : c0 + C]
            g = cells_L[part]
            m = g.size
            slot = np.arange(K)[None, :]
            idx = first[g][:, None] + slot  # (m, K) the pieces of every cell, padded
            live = slot < sizes[g][:, None]
            idx = np.where(live, idx, first[g][:, None])
            # padded cells repeat the last one (finite values everywhere) with no density: they add nothing
            pad = lambda x, fill=None: np.concatenate([x, np.repeat(x[-1:], C - m, axis=0)]) if C > m else x
            dev = devs[len(pending) % len(devs)]
            put = (lambda x: x) if dev is None else (lambda x: None if x is None else jax.device_put(x, dev))
            if dev not in shared:
                shared[dev] = [put(x) for x in (*vsub, wts, *consts)] + [
                    None if a_eff is None else put(cast(a_eff)),
                    None if tautail is None else put(cast(tautail)),
                ]
            dv_, off, ws, c0_, c1_, c2_, c3_, ae, tt = shared[dev]
            take = lambda x: None if x is None else put(cast(pad(x[idx])))
            dens = np.where(live, piece["n"][idx], 0.0)
            cen = centre[g] if walk else np.zeros(m, np.int64)
            dd = piece["d" if walk else "d0"][idx]  # relative to the centre pixel (walk) or to the first one (whole grid)
            o, f = _window_kernel(
                dv_,
                off,
                ws,
                c0_,
                c1_,
                c2_,
                c3_,
                put(cast(pad(dd))),
                take(piece["a"]),
                take(piece["T"]),
                take(piece["tb"]),
                take(piece["co"]),
                put(cast(np.concatenate([dens, np.zeros((C - m, int(K)))]) if C > m else dens)),
                take(piece["length"]),
                None if P_cell is None else put(cast(pad(P_cell[g]))),
                ae,
                put(jnp.asarray(pad(cen))),
                put(jnp.asarray(pad(row[g]))),
                tt,
                voigt=voigt,
                detuning=detuning,
                n_v=n_v,
                n_out=n_out,
                w_back=w_back,
                w_fwd=w_fwd,
                half=half,
                periodic=periodic,
                walk=walk,
            )
            pending.append((o, f, part, m))
    for o, f, part, m in pending:
        out_j += np.asarray(o, dtype=np.float64)
        flags[part] = np.asarray(f)[:m]
    return flags

