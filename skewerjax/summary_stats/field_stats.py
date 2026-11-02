"""P and xi of periodic 3D fields on their own grid, over an ensemble of realisations, against an exact expectation
(notebook 08, Part A), from arrays and geometry only: no box or model class enters, only the fields, the grid (its
shape, side L [cMpc] and bins) and, optionally, the expected spectrum as a callable P(k).

* ``box_grid``, ``grid_limits``: ``GridStatistics3D`` on an N^3 grid of side L with explicit bins, every bin inside the
  full-coverage limits (k_min = 2 pi / L, k_max = pi / dx, r_min = dx, r_max = L / 2);
* ``grid_power``: an isotropic spectrum P(k) on the rfft modes of a grid (what ``GridStatistics3D.theory`` and
  ``xi_iso_covariance`` take);
* ``field_statistics``: the statistics of every field of an iterable (each one measured and discarded), their mean over
  the ensemble, the expectation and the exact Gaussian covariance of the isotropic xi bins;
* ``xi_chi2``, ``xi_subset_offsets``: the chi^2 and the offset of the mean xi from its expectation;
* ``bao_height``: the height of the BAO peak of a curve r^2 xi(r) above its sidebands.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable

import jax.numpy as jnp
import numpy as np

from skewerjax.summary_stats.binning import ensemble_mean, full_coverage, native_edges
from skewerjax.summary_stats.grid3d import GRID_KEYS, GridStatistics3D


def grid_limits(n_cells: int, L: float) -> dict:
    """The full-coverage limits of an ``n_cells``^3 grid of side ``L`` [cMpc], in [1/cMpc] and [cMpc]: k_min = 2 pi / L
    (the fundamental mode) and k_max = pi / dx (the Nyquist frequency of each axis: inside the sphere |k| <= pi / dx every
    direction is available, beyond it only the corners of the cube of modes); r_min = dx and r_max = L / 2 (|r| <= L / 2
    holds every direction of the minimum-image separations, beyond it only the corners)."""
    L = float(L)
    dx = L / int(n_cells)
    return dict(k_min=2 * np.pi / L, k_max=np.pi / dx, r_min=dx, r_max=0.5 * L)


def box_grid(n_cells: int, L: float, *, k_edges, r_edges, los_axis: int = 1, log=print) -> GridStatistics3D:
    """``GridStatistics3D`` on an ``n_cells``^3 periodic grid of side ``L`` [cMpc], line of sight ``los_axis``, with the
    bins [1/cMpc], [cMpc]:

    * |k| ``k_edges``, also the k_perp columns of the (k_par, k_perp) map; the map's rows are the grid's own
      k_par = 2 pi m / L, m = 1 .. N / 2, one row per mode centred on it;
    * |r| ``r_edges``, also the |r| bins of the mu wedges and the r_perp columns of the (r_par, r_perp) map; the map's
      rows are the grid's own r_par = j dx, j = 0 .. N / 2.

    Every |k|, k_perp, |r| and r_perp bin that extends beyond the full-coverage limits (``grid_limits``: pi / dx, L / 2)
    is cut (``binning.full_coverage``, reported by ``log``)."""
    n = int(n_cells)
    lim = grid_limits(n, L)
    k = full_coverage(np.asarray(k_edges, dtype=float), lim["k_max"], "|k| bins", log)
    r = full_coverage(np.asarray(r_edges, dtype=float), lim["r_max"], "|r| bins", log)
    half = n // 2
    return GridStatistics3D(
        (n,) * 3,
        L,
        los_axis=los_axis,
        k_edges=k,
        kpar_edges=native_edges(lim["k_min"], half, start=1),
        kperp_edges=k,
        r_edges=r,
        r_mu_edges=r,
        rpar_edges=native_edges(lim["r_min"], half),
        rperp_edges=r,
    )


def grid_power(power: Callable, shape, L):
    """P(|k|) [cMpc^3] on the rfft modes of a grid of ``shape`` and side(s) ``L`` [cMpc] (the layout of
    ``GridStatistics3D.theory`` and ``xi_iso_covariance``): ``power`` a callable of |k| [1/cMpc] (an isotropic
    expectation, e.g. ``PowerLawGRF.power``; P(0) is the caller's, the estimators drop the k = 0 mode). |k| is built as
    ``PowerLawGRF.k_grid`` builds it, so the result equals ``PowerLawGRF.power_grid`` bit for bit."""
    shape = tuple(int(n) for n in shape)
    Ls = np.broadcast_to(np.asarray(L, dtype=float), (len(shape),))
    dx = tuple(float(l) / n for l, n in zip(Ls, shape))
    axes = [2.0 * jnp.pi * jnp.fft.fftfreq(n, d=d) for n, d in zip(shape, dx)]
    axes[-1] = 2.0 * jnp.pi * jnp.fft.rfftfreq(shape[-1], d=dx[-1])
    k_vec = jnp.meshgrid(*axes, indexing="ij", sparse=True)
    return power(jnp.sqrt(sum(k**2 for k in k_vec)))


def field_statistics(
    fields: Iterable, grid: GridStatistics3D, *, power: Callable | None = None, power_nw: Callable | None = None, log=print
) -> dict:
    """P and xi of every field of ``fields`` (an iterable of arrays of ``grid.shape``, e.g. the overdensities of the
    realisations, generated one at a time: ``(boxes.delta(i) for i in idx)``) on ``grid`` (``box_grid``), streamed: each
    field is measured and discarded. Nothing is stored.

    power : callable P(k) [cMpc^3] of |k| [1/cMpc], optional: the expected spectrum (``grid_power``), for the expectation
        ``theory`` {key: array} of every statistic on the grid (``GridStatistics3D.theory``) and the exact Gaussian
        covariance of the isotropic xi bins of one field, ``cov_xi_iso`` (``GridStatistics3D.xi_iso_covariance``).
    power_nw : callable, optional: a second expectation, ``theory_nw`` (e.g. the power law without the BAO bump, so
        ``theory - theory_nw`` is the bump on the grid).

    Returns ``ensemble_mean`` over the fields of every ``GRID_KEYS`` statistic (mean, ``_err``: the error of the mean,
    std / sqrt(n) with ddof = 1, of each bin alone; ``_each``), ``theory``, ``theory_nw`` and ``cov_xi_iso`` (with
    ``power`` / ``power_nw``), the bin ``axes`` (``GridStatistics3D.axes``), ``min_one_plus_delta`` (n,) (1 + min of
    every field), ``limits`` (``grid_limits``, a cubic grid), ``time_s`` (the measurements) and ``time_cov_s`` (the
    covariance)."""
    gs = grid
    t0 = time.perf_counter()
    recs, mins = [], []
    for n, d in enumerate(fields):
        mins.append(float(1.0 + np.min(d)))
        recs.append(gs.measure(d))
        if log and (n + 1) % 16 == 0:
            log(f"field statistics: {n + 1} fields, {time.perf_counter() - t0:.0f} s")
    t_meas = time.perf_counter() - t0
    res = ensemble_mean(recs, GRID_KEYS)
    P = None if power is None else grid_power(power, gs.shape, gs.L)
    if P is not None:
        res["theory"] = gs.theory(P)
    if power_nw is not None:
        res["theory_nw"] = gs.theory(grid_power(power_nw, gs.shape, gs.L))
    t0 = time.perf_counter()
    if P is not None:
        res["cov_xi_iso"] = gs.xi_iso_covariance(P)
    cubic = len(set(gs.shape)) == 1 and np.all(gs.L == gs.L[0])
    res.update(
        axes=gs.axes(),
        min_one_plus_delta=np.array(mins),
        limits=grid_limits(gs.shape[0], gs.L[0]) if cubic else None,
        time_s=t_meas,
        time_cov_s=time.perf_counter() - t0,
    )
    return res


def _xi_iso_offset(res):
    """(r, mean - expectation, expectation, each field, covariance) of the isotropic xi bins with r > 0."""
    r, th = res["axes"]["r_iso"], res["theory"]["xi_iso"]
    ok = np.isfinite(th) & np.isfinite(res["xi_iso"]) & (r > 0)
    C = res["cov_xi_iso"][np.ix_(ok, ok)]
    return r[ok], (res["xi_iso"] - th)[ok], th[ok], res["xi_iso_each"][:, ok], C


def xi_chi2(res, r_range=None) -> dict:
    """chi^2 of the mean isotropic xi of a ``field_statistics`` result (with ``power``) against its expectation, with the
    exact covariance of the mean (``cov_xi_iso`` / n) and with its diagonal only, over the bins with r in ``r_range``
    [cMpc] (default: every bin with r > 0). Returns dict(chi2_full, chi2_diag, dof (the number of bins), p_full (the
    probability of a larger chi2), mode1_share (the share of the variance of one field in the first eigenvector of the
    covariance), offset_top3_share (the share of the offset |mean - expectation|^2 along the 3 eigenvectors of largest
    variance), pulls_top3 (the offset along those 3 eigenvectors in units of their standard deviation for the mean),
    chi2_each (n,) (each field against the covariance of one field))."""
    from scipy.stats import chi2 as chi2_dist

    r, d, th, each, C = _xi_iso_offset(res)
    sel = np.ones(r.size, bool) if r_range is None else (r >= r_range[0]) & (r <= r_range[1])
    d, each, C = d[sel], each[:, sel], C[np.ix_(sel, sel)]
    n = each.shape[0]
    w, U = np.linalg.eigh(C)  # ascending: the last eigenvectors have the largest variance
    proj = U.T @ d
    chi2_full = float(np.sum(proj**2 / (w / n)))
    return dict(
        chi2_full=chi2_full,
        chi2_diag=float(np.sum(d**2 / (np.diag(C) / n))),
        dof=int(d.size),
        p_full=float(chi2_dist.sf(chi2_full, d.size)),
        mode1_share=float(w[-1] / w.sum()),
        offset_top3_share=float(np.sum(proj[-3:] ** 2) / np.sum(proj**2)),
        pulls_top3=(proj / np.sqrt(w / n))[::-1][:3],
        chi2_each=np.array([float((x - th[sel]) @ np.linalg.solve(C, x - th[sel])) for x in each]),
    )


def xi_subset_offsets(res, sizes=(4, 8, 16), r_range=None) -> list:
    """The offset of the mean r^2 xi(|r|) of a ``field_statistics`` result (with ``power``) from its expectation,
    averaged over the bins with r in ``r_range`` [cMpc] (default every bin with r > 0), for disjoint subsets of ``size``
    consecutive fields, and the standard deviation of that offset for a mean of ``size`` fields from the exact
    covariance, sqrt(w^T C w / size) with w = r^2 / n_bins. Returns [dict(size, offsets (n_subsets,), sigma)] [cMpc^2]."""
    r, _, th, each, C = _xi_iso_offset(res)
    sel = np.ones(r.size, bool) if r_range is None else (r >= r_range[0]) & (r <= r_range[1])
    w = np.where(sel, r**2, 0.0) / sel.sum()
    sig1 = float(np.sqrt(w @ C @ w))
    out = []
    for m in sizes:
        if m <= each.shape[0]:
            offs = [float(w @ (each[j : j + m].mean(0) - th)) for j in range(0, each.shape[0] - m + 1, m)]
            out.append(dict(size=int(m), offsets=np.array(offs), sigma=sig1 / np.sqrt(m)))
    return out


def bao_height(r, r2xi, r_bao: float, sigma_bao: float, err=None) -> dict:
    """The BAO peak of r^2 xi(r): its value in the bin nearest r_bao and its height above the straight line through the
    bins nearest r_bao -+ 4 sigma_bao (the sidebands), with the error of that height from the errors ``err`` of the
    three bins (independent errors assumed). r, r_bao, sigma_bao [cMpc]. Returns dict(r, value, height, height_err (NaN
    without ``err``))."""
    r, y = np.asarray(r, dtype=float), np.asarray(r2xi, dtype=float)
    j, lo, hi = (int(np.argmin(np.abs(r - x))) for x in (r_bao, r_bao - 4 * sigma_bao, r_bao + 4 * sigma_bao))
    if hi == lo:  # the bins are too coarse to hold sidebands
        return dict(r=float(r[j]), value=float(y[j]), height=np.nan, height_err=np.nan)
    t = (r[j] - r[lo]) / (r[hi] - r[lo])
    h = y[j] - (y[lo] + t * (y[hi] - y[lo]))
    he = np.nan if err is None else float(np.sqrt(err[j] ** 2 + ((1 - t) * err[lo]) ** 2 + (t * err[hi]) ** 2))
    return dict(r=float(r[j]), value=float(y[j]), height=float(h), height_err=he)
