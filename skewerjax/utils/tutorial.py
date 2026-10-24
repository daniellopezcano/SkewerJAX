"""Small toy set-ups of the tutorial notebooks (NumPy + the core), kept here so the notebooks stay readable.

* ``toy_pieces``: ``core.Pieces`` of one skewer from plain arrays (centroid a, length, T, peculiar velocity, density).
* ``periodic_pieces_demo``: a few pieces along a box, one with its line centre beyond the period (effect i) and a
  damped absorber at the edge (effect ii), integrated without wrap and with ``velocity_period=True`` through
  ``core.compute_optical_depth``; and the exact periodic sum over images against the nearest image.
* ``p1d_expected``, ``p3d_expected_on_skewer_grid``: the expected spectra of a GRF sampled along skewers (discrete
  transverse projection or aliasing, the linear-interpolation window).
"""
from __future__ import annotations

from typing import Optional, Sequence

import jax.numpy as jnp
import numpy as np

from skewerjax.core.constants import phys
from skewerjax.core.cosmology import Cosmology
from skewerjax.core.lines import LineBundle
from skewerjax.core.optical_depth import Pieces, compute_optical_depth
from skewerjax.core.skewers import SkewerBundle


def toy_pieces(line_bundle: LineBundle, a_dagger: float, a, length_cMpc, T, u_away_kms, n_com, velocity_span_kms: float) -> Pieces:
    """``core.Pieces`` of one skewer from plain (N_p,) arrays: a at the centroids, lengths [cMpc], T [K], the peculiar
    velocity along the path [km/s, positive away from the observer] and the comoving density [cm^-3].

    The photon travels towards the observer, so its velocity along the photon is -u. The widths are 0 (no check).
    """
    nl = line_bundle.n_lines
    rows = lambda x: jnp.broadcast_to(jnp.asarray(x, dtype=float)[None, None], (nl, 1, np.size(x)))
    one = lambda x: jnp.asarray(x, dtype=float)[None]
    z = np.zeros(np.size(a))
    return Pieces(a_dagger=one(np.full(np.size(a), a_dagger)), scale_factor=one(a), length_m=one(np.asarray(length_cMpc) * phys.Mpc_to_m),
                  temperature=rows(T), turbulence=rows(z), collisional=rows(z), los_velocity=rows(-np.asarray(u_away_kms)),
                  number_density=rows(n_com), width_kms=rows(z), velocity_span_kms=jnp.asarray([float(velocity_span_kms)]))


def scale_factor_of_x(x_max_cMpc: float = 20.0, a_obs: float = 0.25, cosmology: Optional[Cosmology] = None, n: int = 4001):
    """a(x) along a straight path that starts at the observer's end (x = 0, a = a_obs) [cMpc]: (x grid, a grid)."""
    cosmology = cosmology or Cosmology(phys.H0, 0.0, 0.31, 0.0, 0.69)
    x = np.linspace(0.0, x_max_cMpc, n)
    sb = SkewerBundle(np.stack([x, 0 * x, 0 * x], -1)[None], a0=a_obs, x0=0.0, ICs_defined_at_index=0,
                      ICs_at_emission_or_absorption="absorption", cosmologies=cosmology)
    return x, np.asarray(sb.scale_factors[0, 0])


PERIODIC_DEMO_PIECES = (            # s [cMpc], length [cMpc], log10 N [cm^-2], T [K], u [km/s, away from the observer]
    (3.0, 0.3, 13.8, 1.5e4, 40.0),
    (8.0, 0.4, 14.2, 2.0e4, -60.0),
    (12.5, 0.3, 13.6, 1.0e4, 20.0),
    (19.2, 0.3, 14.0, 2.0e4, 120.0),     # effect (i): its line centre falls beyond the period
    (19.6, 0.2, 19.3, 1.0e4, 0.0),       # effect (ii): a damped absorber at the edge, at rest
)


def periodic_pieces_demo(L_cMpc: float = 20.0, a_obs: float = 0.25, pieces: Sequence = PERIODIC_DEMO_PIECES,
                         line_id: str = "HI-LyA", dv_kms: float = 1.0, n_images: int = 200) -> dict:
    """A few pieces along a box of side L (observer at s = 0, a(0) = a_obs), each with its own colour in the figure.

    The spectrum is computed on one period [0, P), P = c ln(a(0) / a(L)) (the velocity span of the path), with
    ``core.compute_optical_depth(..., per_piece=True)``: ``velocity_period=False`` (no wrap) and ``True`` (each detuning
    replaced by its nearest image modulo P). Also: every piece's whole profile on an extended axis (where its
    absorption is before any wrap), and for the damped piece the exact periodic sum over |n| <= ``n_images`` images
    against the nearest image (the limitation of keeping one image).
    """
    lb = LineBundle([line_id], ref_line_id=line_id)
    xg, ag = scale_factor_of_x(L_cMpc + 1.0, a_obs)
    a_of = lambda x: np.exp(np.interp(np.asarray(x, dtype=float), xg, np.log(ag)))
    P = float(phys.c_kms * np.log(a_obs / a_of(L_cMpc)))
    s, ell, logN, T, u = (np.array([p[i] for p in pieces], dtype=float) for i in range(5))
    a = a_of(s)
    n_com = 10 ** logN / (ell * phys.Mpc_to_m * 1e2) * a ** 2           # sum_p n a^-2 l = N (physical column)
    pc = toy_pieces(lb, a_obs, a, ell, T, u, n_com, P)
    nb = int(round(P / dv_kms))
    v = (np.arange(nb) + 0.5) * (P / nb)
    tau = {k: np.asarray(compute_optical_depth(pc, v, lb, velocity_period=k, per_piece=True, resolution_tolerance=None))[0, 0]
           for k in (False, True)}
    v_ext = np.arange(-0.35 * P, 1.35 * P, P / nb)
    tau_ext = np.asarray(compute_optical_depth(pc, v_ext, lb, per_piece=True, resolution_tolerance=None))[0, 0]
    v_H = phys.c_kms * np.log(a_obs / a)
    beta = -u / phys.c_kms
    v_c = v_H + 0.5 * phys.c_kms * (np.log1p(-beta) - np.log1p(beta))
    # the exact periodic sum of the damped piece: sum_n tau_1(v + n P), against the nearest image
    j = int(np.argmax(logN))
    one = toy_pieces(lb, a_obs, a[j:j + 1], ell[j:j + 1], T[j:j + 1], u[j:j + 1], n_com[j:j + 1], P)
    exact = sum(np.asarray(compute_optical_depth(one, v + n * P, lb, resolution_tolerance=None))[0, 0]
                for n in range(-n_images, n_images + 1))
    return dict(v=v, P=P, L=L_cMpc, s=s, length=ell, logN=logN, T=T, u=u, v_H=v_H, v_c=v_c, tau_nowrap=tau[False],
                tau_wrap=tau[True], v_ext=v_ext, tau_ext=tau_ext, strong=j, tau_strong_exact=exact,
                tau_strong_nearest=tau[True][j], v_of_s=lambda x: phys.c_kms * np.log(a_obs / a_of(x)))


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
