"""Two-point statistics of an ensemble of periodic skewers, from arrays and geometry only (notebook 10, Part D):
no box or set-up class enters, only the fields along the skewers, the pixel width, the lattice spacing and the bins.

**1D, along each skewer** (review notes, P1D and xi1D sections; the conventions of ``stats_1d``): N pixels of width dv
[km/s, or cMpc for a comoving axis], one 1D FFT per skewer, averaged over the skewers,

    P1D(k_m)    = < (dv / N) |rfft(delta)_m|^2 >_skewers,                       k_m = 2 pi m / (N dv), m = 0 .. N / 2,
    xi1D(j dv)  = < (1 / N) sum_v delta(v) delta(v + j dv) >_skewers = irfft(< |rfft(delta)|^2 >) / N,  j = 0 .. N / 2,

an exact discrete Fourier pair (``skewer_p1d``, ``skewer_xi1d``). Nothing is derived from the 3D statistics.

**3D, on the lattice of skewers** (``lattice_grid``, ``on_lattice``, ``measure_lattice``): the skewers along axis 1 at
the nodes of a regular (n_x, n_z) lattice of spacing d_perp, every pixel one lattice point along the line of sight;
``GridStatistics3D`` on exactly that lattice (no regridding), in comoving units: a pixel spans L / n_los, so the box is
(n_x d_perp_x, L, n_z d_perp_z). Bins: |k| ``k_edges`` (also the k_perp columns of the (k_par, k_perp) plane, whose rows
are the lattice's own k_par = 2 pi m / L), |r| and the mu wedges ``r_edges``, r_perp ``rperp_edges``, the r_par rows the
lattice's own separations j L / n_los; every bin beyond the full-coverage limits (the smallest Nyquist frequency, half
the box) cut (``binning.full_coverage``); ``lattice_limits`` the spacings and those limits; ``on_points`` the values of
a field uniform in each cell at the points of the skewers (p per cell).

The two are related on the lattice by P1D(k_par) = (1 / (L_x L_z)) sum_{k_perp} P(k_perp, k_par) (Parseval across the
line of sight), the discrete form of P1D(k_par) = int d^2k_perp / (2 pi)^2 P3D.
"""

from __future__ import annotations

import numpy as np

from skewerjax.summary_stats.binning import full_coverage, native_edges
from skewerjax.summary_stats.grid3d import GridStatistics3D
from skewerjax.summary_stats.mesh import check_grid_native


def skewer_p1d(delta, dv: float):
    """P1D along each skewer, averaged over the skewers (module docstring), every mode including k = 0.

    delta : (..., N_sk, N) the fields along the skewers (e.g. (N_fields, N_sk, N)); dv : the pixel width.
    Returns (k (N // 2 + 1,) [1 / unit of dv], P1D (..., N // 2 + 1) [unit of dv])."""
    d = np.asarray(delta)
    n = d.shape[-1]
    return 2 * np.pi * np.fft.rfftfreq(n, dv), (dv / n) * np.mean(np.abs(np.fft.rfft(d, axis=-1)) ** 2, axis=-2)


def skewer_xi1d(delta, dv: float):
    """xi1D along each skewer, averaged over the skewers (module docstring), at every lag j dv, j = 0 .. N / 2 (a
    periodic skewer: the lags beyond half the period repeat those below).

    delta : (..., N_sk, N); dv : the pixel width. Returns (lags (N // 2 + 1,) [unit of dv], xi1D (..., N // 2 + 1))."""
    d = np.asarray(delta)
    n = d.shape[-1]
    xi = np.fft.irfft(np.mean(np.abs(np.fft.rfft(d, axis=-1)) ** 2, axis=-2), n=n, axis=-1) / n
    return dv * np.arange(n // 2 + 1), xi[..., : n // 2 + 1]


def lattice_grid(n_x: int, n_z: int, n_los: int, L: float, d_perp, *, k_edges, r_edges=None, rperp_edges=None, log=None):
    """``GridStatistics3D`` on the lattice of skewers (module docstring): (n_x, n_los, n_z) points, the line of sight
    along axis 1 with n_los points over the period L [cMpc] (a pixel dv spans dv / u = L / n_los), transverse spacing
    ``d_perp`` (scalar or (d_x, d_z)) [cMpc].

    k_edges : the |k| bins [1/cMpc] (required: the bins are explicit), cut at the smallest Nyquist frequency k_N =
    min(pi n_los / L, pi / d_x, pi / d_z); r_edges : |r| and the mu wedges (default width the coarser spacing, max(d_perp,
    L / n_los), up to L / 2); rperp_edges : the r_perp columns (default width the smaller transverse spacing, from 0: the
    first column holds the pairs on the same skewer). ``log``: reports the bins cut by the full-coverage limits."""
    L = float(L)
    d_perp = np.broadcast_to(np.asarray(d_perp, dtype=float), (2,))
    n_los = int(n_los)
    k_nyq = float(min(np.pi * n_los / L, *(np.pi / d_perp)))
    half = 0.5 * L
    dr = max(float(d_perp.max()), L / n_los)
    k = full_coverage(k_edges, k_nyq, "|k| bins", log)
    r = np.arange(0.0, half + 0.5 * dr, dr) if r_edges is None else np.asarray(r_edges, dtype=float)
    rq = (
        np.arange(0.0, half + 0.5 * d_perp.min(), float(d_perp.min()))
        if rperp_edges is None
        else np.asarray(rperp_edges, dtype=float)
    )
    r, rq = full_coverage(r, half, "|r| bins", log), full_coverage(rq, half, "r_perp bins", log)
    return GridStatistics3D(
        (int(n_x), n_los, int(n_z)),
        (int(n_x) * d_perp[0], L, int(n_z) * d_perp[1]),
        los_axis=1,
        k_edges=k,
        k_mu_edges=k,
        kpar_edges=native_edges(2 * np.pi / L, n_los // 2, start=1),
        kperp_edges=k,
        r_edges=r,
        r_mu_edges=r,
        rpar_edges=native_edges(L / n_los, n_los // 2),
        rperp_edges=rq,
    )


def lattice_limits(L: float, d_perp, n_los: int, period: float) -> dict:
    """The lattice of skewers of ``lattice_grid`` and its full-coverage limits, from its geometry: the period L [cMpc]
    and ``period`` [km/s] of the line of sight (u = period / L), n_los pixels, the transverse spacing ``d_perp`` (scalar
    or (d_x, d_z)) [cMpc].

    Native units: d_perp [cMpc] per axis, k_x, k_z from k_fund = 2 pi / L to k_nyq_perp = pi / d_perp [1/cMpc]; along
    the line of sight the pixel dv = period / n_los [km/s], k_par from k_par_min_skm = 2 pi / period to k_par_max_skm =
    pi / dv [s/km], r_par from dv to period / 2 [km/s] (period_kms, kms_per_cMpc = u). Comoving units (the isotropic
    quantities): k_nyq_par = pi u / dv = pi n_los / L, k_nyq = the smallest Nyquist frequency, dr_iso = max(d_perp,
    dv / u) (the coarser spacing), r_max = L / 2, r_perp_min = the smaller transverse spacing; and the full-coverage
    limits of the isotropic bins: k_min = k_fund, k_max = k_nyq, r_min = dr_iso, r_max."""
    L, period = float(L), float(period)
    d_perp = np.broadcast_to(np.asarray(d_perp, dtype=float), (2,)).copy()
    dv = period / int(n_los)
    k_par, k_perp = np.pi * int(n_los) / L, np.pi / d_perp
    k_nyq, dr = float(min(k_par, *k_perp)), float(max(d_perp.max(), L / int(n_los)))
    return dict(
        d_perp=d_perp,
        k_nyq_par=float(k_par),
        k_nyq_perp=k_perp,
        k_nyq=k_nyq,
        k_fund=2 * np.pi / L,
        dv_kms=dv,
        period_kms=period,
        kms_per_cMpc=period / L,
        k_par_min_skm=2 * np.pi / period,
        k_par_max_skm=np.pi / dv,
        dr_iso=dr,
        r_max=0.5 * L,
        r_perp_min=float(d_perp.min()),
        k_min=2 * np.pi / L,
        k_max=k_nyq,
        r_min=dr,
    )


def on_lattice(delta, n_x: int, n_z: int) -> np.ndarray:
    """The fields of the skewers (..., N_sk, n_los), the skewers x-major (``core.lattice_columns``: i_x outer, i_z inner),
    on the lattice (..., n_x, n_los, n_z), the line of sight along the middle axis."""
    d = np.asarray(delta)
    return d.reshape(d.shape[:-2] + (int(n_x), int(n_z), d.shape[-1])).swapaxes(-1, -2)


def measure_lattice(delta, grid: GridStatistics3D, *, mas=None) -> dict:
    """The 3D statistics (``GridStatistics3D.measure``: the ``GRID_KEYS``) of one field along the skewers of a lattice,
    delta (N_sk, n_los) x-major, on ``grid`` (``lattice_grid``).

    mas : None (the only value): the samples of the skewers are the nodes of the lattice, so the field is already on
    the mesh (lattice-native) and its FFT is exact, with no window and no alias images of an assignment (Sefusatti et
    al. 2016, §2.3); a scheme raises."""
    check_grid_native(mas, "measure_lattice")
    return grid.measure(on_lattice(delta, grid.shape[0], grid.shape[2]))


def on_points(columns, points_per_cell: int) -> np.ndarray:
    """The values at the points of the skewers of cell values ``columns`` (..., N, ...) along axis 1 (y): (..., N p, ...)
    at y_m = m dx / p, m = 0 .. N p - 1 (p = ``points_per_cell``; the end point y = L, the same place of the periodic
    box as y = 0, dropped, so the lattice is exactly periodic). The field is uniform in each cell, so a point takes the
    value of its cell, that of the nearest node j = floor(m / p + 1/2) mod N (a point on a cell boundary goes to the
    upper cell): every cell holds exactly p points."""
    p = int(points_per_cell)
    return np.roll(np.repeat(np.asarray(columns), p, axis=1), -(p // 2), axis=1)
