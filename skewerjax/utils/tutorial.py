"""Small toy set-ups of the tutorial notebooks (NumPy + the core kernels), kept here so the notebooks stay readable.

* ``two_elements``: two gas elements on opposite sides of a reference point, with opposite peculiar velocities that put
  their line centres at the same v_dagger; tau from a finely resolved quadrature of the integral (review notes,
  Eq. tau_straight_los; scipy's Faddeeva function, independent of the package's Voigt) and from the package's sum
  (``core.optical_depth_kernel``) with K pieces per element.
* ``linear_detuning``: one absorber, tau with the exact and the linear detuning (``core.spectral.frequency_offset_GHz``).
* ``periodic_box_demo``: one absorber near the far edge of a periodic box, integrated with each periodicity option.
* ``p1d_expected``, ``p3d_expected_on_skewer_grid``: the expected spectra of a GRF sampled along skewers (discrete
  transverse projection or aliasing, the linear-interpolation window).
"""
from __future__ import annotations

from typing import Optional, Sequence

import jax.numpy as jnp
import numpy as np
from scipy.special import wofz

from skewerjax.core.constants import phys
from skewerjax.core.cosmology import Cosmology
from skewerjax.core.lines import LineBundle
from skewerjax.core.optical_depth import optical_depth_kernel, voigt_widths_GHz
from skewerjax.core.skewers import SkewerBundle


def _line_arrays(line_bundle: LineBundle):
    return tuple(jnp.asarray(x) for x in (line_bundle.lambda0_A, line_bundle.A21_GHz, line_bundle.mass_amu, line_bundle.f_osc))


def kernel_tau(line_bundle: LineBundle, v_kms, a_dagger, a, u_away_kms, T, n_com, w_cMpc, *, detuning: str = "exact",
               velocity_period: Optional[float] = None, per_point: bool = False) -> np.ndarray:
    """The package's sum (``optical_depth_kernel``) for points of one skewer given as plain arrays.

    a, u_away_kms (peculiar velocity along the path, positive away from the observer), T [K], n_com [comoving cm^-3],
    w_cMpc (line elements) : (N_p,). The photon travels towards the observer, so its velocity along the photon is -u.
    Returns (N_lines, N_v), or (N_lines, N_p, N_v) with ``per_point``.
    """
    lam, A21, mass, f = _line_arrays(line_bundle)
    nl = lam.shape[0]
    rows = lambda x: jnp.broadcast_to(jnp.asarray(x, dtype=float)[None], (nl, np.size(x)))
    z = rows(np.zeros(np.size(a)))
    return np.asarray(optical_depth_kernel(jnp.asarray(v_kms, dtype=float), lam, A21, mass, f, jnp.asarray(a_dagger, dtype=float),
                                           jnp.asarray(a, dtype=float), rows(-np.asarray(u_away_kms)), rows(T), z, z, rows(n_com),
                                           jnp.asarray(w_cMpc, dtype=float) * phys.Mpc_to_m, detuning=detuning,
                                           velocity_period=velocity_period, per_point=per_point))


def scale_factor_of_x(x_max_cMpc: float = 20.0, a_obs: float = 0.25, cosmology: Optional[Cosmology] = None, n: int = 4001):
    """a(x) along a straight path that starts at the observer's end (x = 0, a = a_obs) [cMpc]: (x grid, a grid)."""
    cosmology = cosmology or Cosmology(phys.H0, 0.0, 0.31, 0.0, 0.69)
    x = np.linspace(0.0, x_max_cMpc, n)
    sb = SkewerBundle(np.stack([x, 0 * x, 0 * x], -1)[None], a0=a_obs, x0=0.0, ICs_defined_at_index=0,
                      ICs_at_emission_or_absorption="absorption", cosmologies=cosmology)
    return x, np.asarray(sb.scale_factors[0, 0])


def two_elements(x_ref: float = 10.0, d: float = 1.5, length: float = 0.3, n_com: Sequence[float] = (2e-11, 1e-11),
                 T: Sequence[float] = (1.0e4, 6.0e4), line_id: str = "HI-LyA", a_obs: float = 0.25, pieces=(1, 4, 32),
                 n_v: int = 1201, half_width_kms: float = 150.0, n_fine: int = 20001) -> dict:
    """Elements A (x_ref - d, nearer the observer) and B (x_ref + d) of uniform n_com, T over ``length`` [cMpc], with
    peculiar velocities u_A = -u_B = c tanh(Delta / c), Delta = [v_H(x_B) - v_H(x_A)] / 2, so that both line centres
    v_c = v_H(x) + c artanh(u / c) fall at the same v_dagger (v_H = c ln(a_dagger / a(x))).

    Returns a dict with the velocity grid ``v``, the reference ``tau_ref`` (N_v,) and its per-element parts
    ``tau_ref_parts`` (2, N_v) (Simpson quadrature on ``n_fine`` points per element, scipy's wofz), the package's sum
    ``tau_sum[K]`` (N_v,) and ``tau_sum_parts[K]`` (2, N_v) for every number of pieces K per element, and the set-up.
    """
    from scipy.integrate import simpson
    lb = LineBundle([line_id], ref_line_id=line_id)
    xg, ag = scale_factor_of_x(x_ref + d + length + 1.0, a_obs)
    a_of = lambda x: np.exp(np.interp(x, xg, np.log(ag)))
    vH = lambda x: phys.c_kms * np.log(a_obs / a_of(x))
    xc = np.array([x_ref - d, x_ref + d])
    Delta = 0.5 * (vH(xc[1]) - vH(xc[0]))
    u = np.array([1.0, -1.0]) * phys.c_kms * np.tanh(Delta / phys.c_kms)
    v_star = vH(xc[0]) + Delta
    v = v_star + np.linspace(-half_width_kms, half_width_kms, n_v)
    lam, A21, mass, f = (np.asarray(x) for x in _line_arrays(lb))
    sig, gam = (np.asarray(x)[0, 0] for x in voigt_widths_GHz(jnp.asarray([[T[0]]]), jnp.zeros((1, 1)), jnp.zeros((1, 1)),
                                                             jnp.asarray(lam), jnp.asarray(A21), jnp.asarray(mass)))
    parts = []
    for e in range(2):
        xs = np.linspace(xc[e] - 0.5 * length, xc[e] + 0.5 * length, n_fine)
        a = a_of(xs)
        sig_e = float(np.asarray(voigt_widths_GHz(jnp.asarray([[T[e]]]), jnp.zeros((1, 1)), jnp.zeros((1, 1)), jnp.asarray(lam),
                                                  jnp.asarray(A21), jnp.asarray(mass))[0])[0, 0])
        beta = -u[e] / phys.c_kms                                           # along the photon
        v_c = phys.c_kms * (np.log(a_obs / a) + 0.5 * (np.log1p(-beta) - np.log1p(beta)))
        nu = (phys.c * 10.0 / lam[0]) * np.expm1(-(v[None, :] - v_c[:, None]) / phys.c_kms)      # exact detuning [GHz]
        phi = np.real(wofz((nu + 1j * gam) / (np.sqrt(2.0) * sig_e))) / (np.sqrt(2.0 * np.pi) * sig_e)
        integrand = phys.cross_section_prefactor * f[0] * n_com[e] * a[:, None] ** -2 * 1e2 * phi
        parts.append(simpson(integrand, x=xs * phys.Mpc_to_m, axis=0))
    tau_ref_parts = np.stack(parts)
    sums, sum_parts = {}, {}
    for K in pieces:
        edges = np.linspace(-0.5, 0.5, K + 1) * length
        mids = 0.5 * (edges[1:] + edges[:-1])
        x_p = np.concatenate([xc[0] + mids, xc[1] + mids])
        tp = kernel_tau(lb, v, a_obs, a_of(x_p), np.repeat(u, K), np.repeat(T, K), np.repeat(n_com, K), np.full(2 * K, length / K),
                        per_point=True)[0]
        sum_parts[K] = np.stack([tp[:K].sum(0), tp[K:].sum(0)])
        sums[K] = sum_parts[K].sum(0)
    return dict(v=v, v_star=v_star, x_c=xc, u=u, T=np.asarray(T), n_com=np.asarray(n_com), length=length, a=a_of(xc),
                hubble_width_kms=np.abs(np.gradient(vH(xg), xg)).mean() * length, b_kms=np.sqrt(2 * phys.kB * np.asarray(T) / mass[0]) / 1e3,
                x_grid=xg, v_hubble=vH(xg), tau_ref=tau_ref_parts.sum(0), tau_ref_parts=tau_ref_parts, tau_sum=sums, tau_sum_parts=sum_parts)


def linear_detuning(log_N: Sequence[float] = (14.0, 17.0, 20.5), T: float = 1.5e4, line_id: str = "HI-LyA", a: float = 0.25,
                    half_width_kms: float = 2000.0, n_v: int = 8001) -> dict:
    """One absorber at rest at v_dagger = 0 with column N [cm^-2]: tau with the exact and the linear detuning.

    Returns v (N_v,), tau_exact and tau_linear (N_col, N_v), the columns, and b [km/s].
    """
    lb = LineBundle([line_id], ref_line_id=line_id)
    v = np.linspace(-half_width_kms, half_width_kms, n_v)
    w = 1e-3                                                                 # a 1 ckpc element; n_com sets the column
    out = {}
    for mode in ("exact", "linear"):
        rows = []
        for lN in log_N:
            n_com = 10 ** lN / (a ** -2 * w * phys.Mpc_to_m * 1e2)
            rows.append(kernel_tau(lb, v, a, [a], [0.0], [T], [n_com], [w], detuning=mode)[0])
        out[mode] = np.stack(rows)
    mass = float(np.asarray(lb.mass_amu)[0])
    return dict(v=v, tau_exact=out["exact"], tau_linear=out["linear"], log_N=np.asarray(log_N), b_kms=np.sqrt(2 * phys.kB * T / mass) / 1e3)


def periodic_box_demo(L_cMpc: float = 20.0, x_abs: float = 19.75, log_N: float = 14.0, T: float = 4.0e4, a_obs: float = 0.25,
                      buffer_cMpc: float = 4.0, line_id: str = "HI-LyA", dv_kms: float = 1.0) -> dict:
    """One absorber near the far edge of a periodic box of side L, integrated with each periodicity option.

    * ``none``: the path [0, L], a grid [0, P) (P = v_H(L)), detunings not wrapped: the absorption past P is lost;
    * ``wrapped``: the same path, detunings wrapped to their minimum image modulo P (``velocity_period=P``);
    * ``buffer``: the path extended by ``buffer_cMpc`` on both sides with the periodic copies of the absorber
      (non-periodic integral on a longer path), then read on [0, P).
    Also: the absorber's whole (non-periodic) profile ``tau_full`` on the extended grid, and for the line-frame shift an
    absorber in the middle of the box seen in Lya and Si III 1206 (``two_lines_own``, periodic) and both on the Lya frame
    (``two_lines_ref``, ``core.wrap_to_ref_line``). Returns the grid v (N_v,), the tau (N_v,), P, the absorber's line
    centre, the buffer's extended grid and these extras.
    """
    lb = LineBundle([line_id], ref_line_id=line_id)
    xg, ag = scale_factor_of_x(L_cMpc + buffer_cMpc + 1.0, a_obs)
    xg_neg, ag_neg = scale_factor_of_x(buffer_cMpc + 1.0, a_obs)            # towards the observer: mirror (a > a_obs)
    a_of = lambda x: np.where(x >= 0, np.exp(np.interp(x, xg, np.log(ag))), np.exp(2 * np.log(a_obs) - np.interp(-x, xg_neg, np.log(ag_neg))))
    P = float(phys.c_kms * np.log(a_obs / a_of(L_cMpc)))
    nb = int(round(P / dv_kms))
    v = (np.arange(nb) + 0.5) * (P / nb)
    w = 1e-3
    n_com = 10 ** log_N / (a_obs ** -2 * w * phys.Mpc_to_m * 1e2)
    one = lambda x, vv, period=None: kernel_tau(lb, vv, a_obs, a_of(np.atleast_1d(x)), [0.0], [T], [n_com], [w], velocity_period=period)[0]
    tau = dict(none=one(x_abs, v), wrapped=one(x_abs, v, P))
    copies = [x for x in (x_abs - L_cMpc, x_abs, x_abs + L_cMpc) if -buffer_cMpc <= x <= L_cMpc + buffer_cMpc]
    tau["buffer"] = sum(one(x, v) for x in copies)
    v_ext = np.arange(-buffer_cMpc / L_cMpc * P, P * (1 + buffer_cMpc / L_cMpc), P / nb)
    tau_ext = sum(one(x, v_ext) for x in copies)
    # one axis for two lines: an absorber in the middle of the box, Lya and Si III, periodic, then wrap_to_ref_line
    from skewerjax.core.optical_depth import wrap_to_ref_line
    two = LineBundle([line_id, "SiIII-1206"], ref_line_id=line_id)
    a_mid = a_of(np.atleast_1d(0.5 * L_cMpc))
    own = np.stack([kernel_tau(LineBundle([lid], ref_line_id=lid), v, a_obs, a_mid, [0.0], [T], [n_com * f], [w], velocity_period=P)[0]
                    for lid, f in ((line_id, 1.0), ("SiIII-1206", 1.0 / 30))])                 # Si III 30 x less dense
    ref = np.asarray(wrap_to_ref_line(own[:, None], v, two))[:, 0]
    return dict(v=v, tau=tau, period_kms=P, v_c=float(phys.c_kms * np.log(a_obs / a_of(x_abs))), v_ext=v_ext, tau_ext=tau_ext,
                v_full=v_ext, tau_full=one(x_abs, v_ext), copies=np.asarray(copies), L=L_cMpc, x_abs=x_abs,
                v_of_x=lambda x: float(phys.c_kms * np.log(a_obs / a_of(np.asarray(x)))),
                two_lines_own=own, two_lines_ref=ref, delta_v=float(two.delta_v[1]))


# --------------------------------------------------------------------------- expectations for the GRF study of notebook 07
def linear_interp_window(k, dx):
    """|T(k)|^2 / dx^2 = sinc^4(k dx / 2) of linear interpolation between nodes dx apart (k [1/length])."""
    return np.sinc(np.asarray(k, dtype=float) * dx / (2 * np.pi)) ** 4


def fold(k, dx):
    """k folded into the band [-pi/dx, pi/dx) of a grid of spacing dx (the periodic spectrum of node samples)."""
    kn = np.pi / dx
    return np.mod(np.asarray(k, dtype=float) + kn, 2 * kn) - kn


def p1d_expected(grf, k_par, axis: int = 1, interpolated: bool = False, return_sigma: bool = False, step: int = 1):
    """Expected P1D of a 3D ``PowerLawGRF`` delta along grid lines parallel to ``axis`` [length], averaged over the lines
    at every ``step``-th grid column (``step=1``: all of them).

    The mean is the discrete projection over the grid's transverse modes (``plotting.grf_p1d_along_axis``) at the nodes;
    with ``interpolated`` the lines are sampled between the nodes by linear interpolation, so the node spectrum,
    periodic in k_par with period 2 pi / dx, is multiplied by the window sinc^4(k dx / 2).
    ``return_sigma``: also the standard deviation of that average for one realisation. Every 3D mode is an independent
    complex Gaussian; the lines of a coarse transverse grid see the sum of the step^2 fine modes that alias onto each
    coarse mode j, so sigma = sqrt(sum_j (sum_aliases P)^2) / L_perp^2 (the mean does not depend on ``step``)."""
    tr = [i for i in range(3) if i != axis]
    N = [grf.shape[i] for i in tr]
    ka, kb = (2 * np.pi * np.fft.fftfreq(n, d=grf.dx[i]) for n, i in zip(N, tr))
    dx = grf.dx[axis]
    k = np.asarray(k_par, dtype=float)
    kp = np.abs(fold(k, dx))
    P = np.asarray(grf.power(np.sqrt(kp[:, None, None] ** 2 + ka[None, :, None] ** 2 + kb[None, None, :] ** 2)))
    area = grf.L[tr[0]] * grf.L[tr[1]]
    mean = P.sum((1, 2)) / area
    Nc = [n // step for n in N]
    Pc = P.reshape(-1, step, Nc[0], N[1]).sum(1).reshape(-1, Nc[0], step, Nc[1]).sum(2)
    sig = np.sqrt((Pc ** 2).sum((1, 2))) / area
    W = linear_interp_window(k, dx) if interpolated else 1.0
    return (mean * W, sig * W) if return_sigma else mean * W


def p3d_expected_on_skewer_grid(grf, step: int, k_par, axis: int = 1, interpolated: bool = True):
    """Expected P3D (Nx, Ny, N_kpar) of delta on skewers along ``axis`` placed at every ``step``-th grid column, in the
    layout of ``OverfluxStatistics3D`` (transverse fftfreq of the coarse grid, k_par >= 0 [1/length]).

    Sampling the transverse plane every ``step`` nodes aliases the fine-grid modes onto the coarse ones: P_coarse(j) =
    sum over the fine modes i = j (mod N / step). Along the line of sight: the node spectrum folded into the fine grid's
    band and, with ``interpolated``, the linear-interpolation window."""
    tr = [i for i in range(3) if i != axis]
    N = [grf.shape[i] for i in tr]
    ka, kb = (2 * np.pi * np.fft.fftfreq(n, d=grf.dx[i]) for n, i in zip(N, tr))
    dpar = grf.dx[axis]
    kp = np.abs(fold(np.asarray(k_par, dtype=float), dpar))
    P = np.asarray(grf.power(np.sqrt(ka[:, None, None] ** 2 + kb[None, :, None] ** 2 + kp[None, None, :] ** 2)))
    Nc = [n // step for n in N]
    Pc = P.reshape(step, Nc[0], N[1], -1).sum(0)                       # fold axis a: fine index i = j + m Nc (fftfreq order)
    Pc = Pc.reshape(Nc[0], step, Nc[1], -1).sum(1)
    if interpolated:
        Pc = Pc * linear_interp_window(np.asarray(k_par, dtype=float), dpar)[None, None, :]
    return Pc


def binned_input_power(grf, edges):
    """The input P(|k|) of a ``PowerLawGRF`` averaged over the grid's modes in each bin of |k| (the same weights as
    ``plotting.binned_power``): the exact expectation of a binned measurement, free of the curvature of P inside a bin."""
    _, kmod = grf.k_grid(real=True)
    kmod = np.asarray(kmod)
    n_last = grf.shape[-1]
    w1 = np.full(n_last // 2 + 1, 2.0)
    w1[0] = 1.0
    if n_last % 2 == 0:
        w1[-1] = 1.0
    w = np.broadcast_to(w1, kmod.shape).ravel()
    k = kmod.ravel()
    P = np.asarray(grf.power(k))
    j = np.digitize(k, edges) - 1
    ok = (j >= 0) & (j < len(edges) - 1) & (k > 0)
    W = np.bincount(j[ok], weights=w[ok], minlength=len(edges) - 1)
    Pb = np.bincount(j[ok], weights=(w * P)[ok], minlength=len(edges) - 1) / np.maximum(W, 1e-300)
    return Pb[W > 0]
