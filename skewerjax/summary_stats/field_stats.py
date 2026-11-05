"""P and xi of the density of gas cells in a periodic box, over an ensemble of realisations, against an optional
expectation (notebooks 08 and 09): the gas of every source is ``core.Cells`` (generating points, volumes, number
densities per species), deposited on a mesh (mass assignment, interlacing, the window divided; Sefusatti et al. 2016);
no box or model class enters, only the cells, the grid (its shape, side L [cMpc] and bins) and, optionally, the expected
spectrum as a callable P(k).

* ``box_grid``, ``grid_limits``: ``GridStatistics3D`` on an N^3 mesh of side L with explicit bins, every bin inside the
  full-coverage limits (k_min = 2 pi / L, k_max = pi / dx, r_min = dx, r_max = L / 2);
* ``grid_power``: an isotropic spectrum P(k) on the rfft modes of a grid (what ``GridStatistics3D.theory`` and
  ``xi_iso_covariance`` take);
* ``field_statistics``: the statistics of the density of each species of every ``Cells`` of an iterable (each one
  measured and discarded) through the mesh, their mean over the ensemble, the expectation and the exact Gaussian
  covariance of the isotropic xi bins; ``cell_weights``: the deposit of every cell, n_X V over its mean;
* ``xi_chi2``, ``xi_subset_offsets``: the chi^2 and the offset of the mean xi from its expectation;
* ``bao_height``: the height of the BAO peak of a curve r^2 xi(r) above its sidebands;
* ``xi_corner``: the corner r_par ~ L / 2, r_perp ~ 0 of the (r_perp, r_par) map, its offset and its error.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import jax.numpy as jnp
import numpy as np

from skewerjax.core.cells import Cells
from skewerjax.summary_stats.binning import ensemble_mean, full_coverage, native_edges
from skewerjax.summary_stats.grid3d import GRID_KEYS, GridStatistics3D
from skewerjax.summary_stats.mesh import PAINT_CHUNK, TRUSTED_K_NYQ, TRUSTED_R_H


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


def _as_cells_iter(cells):
    """An iterable of ``core.Cells`` (one Cells: a list of one); anything else raises (the fields enter as cells)."""
    if isinstance(cells, Cells):
        return [cells]
    if isinstance(cells, (np.ndarray, jnp.ndarray)):
        raise TypeError(
            "field_statistics takes core.Cells (generating points, volumes, number densities), one or an iterable of them, "
            "not arrays: a toy field enters through simulations.grf_cells (or ToyBoxes.cells), a snapshot through gas_to_cells."
        )
    return cells


def cell_weights(cells: Cells, species: str) -> np.ndarray:
    """The number of particles of ``species`` in every cell, n_X V (comoving cm^-3 x cMpc^3: a constant factor apart),
    normalised by its mean over the cells: (M,) values whose mesh holds 1 + delta of the species
    (``GridStatistics3D.measure_points`` divides by the mean number of points per node)."""
    if species not in cells.species:
        raise ValueError(f"species {species!r} not in the cells (species {cells.species_keys}).")
    w = np.asarray(cells.species[species]["number_density"], dtype=np.float64) * np.asarray(cells.volumes, dtype=np.float64)
    m = w.mean()
    if not m > 0:
        raise ValueError(f"species {species!r}: the mean number per cell is {m}, not > 0.")
    return w / m


def field_statistics(
    cells,
    grid: GridStatistics3D,
    *,
    species,
    mas: str = "pcs",
    interlace: bool = True,
    power: Callable | None = None,
    power_nw: Callable | None = None,
    chunk: int | None = PAINT_CHUNK,
    log=print,
) -> dict:
    """P and xi of the density of every species of ``species`` in the gas cells ``cells`` (one ``core.Cells`` or an
    iterable of them, one per realisation, streamed: each one is measured and discarded), through a mesh of ``grid``'s
    shape (``box_grid``; S16). The one path of every source of gas (toy grids through ``grf_cells``, snapshots through
    ``gas_to_cells``), with no branch on the source:

    1. every cell deposits its number of particles n_X V at its generating point (``cell_weights``) on the mesh with
       the scheme ``mas`` (default "pcs"), interlaced (``interlace``, default True): ``mesh.fourier_mesh``, periodic;
    2. the mesh is normalised by its mean (1 + delta; the k = 0 mode is dropped), the window divided;
    3. P and xi binned as ``GridStatistics3D`` (isotropic, (par, perp), wedges of mu).

    species : a key of the cells' species (e.g. "HI") or a sequence of them (the result is then {species: result}).
    power : callable P(k) [cMpc^3] of |k| [1/cMpc], optional: the spectrum of the field (e.g. ``PowerLawGRF.power``),
        on the mesh modes (``grid_power``) and binned like the measurement (``GridStatistics3D.theory``): the expectation
        ``theory`` and the exact Gaussian covariance of the isotropic xi bins ``cov_xi_iso``. Through the mesh the
        measurement approaches it at k << k_Nyq: for fields given at the nodes of the mesh (a toy grid of the mesh's
        shape) the exact expectation is ``mesh.node_transfer`` x P, a bias ~3e-4 at k_Nyq / 2 for interlaced PCS.
    power_nw : callable, optional: a second expectation, ``theory_nw`` (e.g. the power law without the BAO bump).
    chunk : cells painted at a time (memory, ``mesh.paint``).

    Returns (per species) ``ensemble_mean`` over the realisations of every ``GRID_KEYS`` statistic (mean, ``_err``: the
    error of the mean, std / sqrt(n) with ddof = 1, of each bin alone; ``_each``), ``theory``, ``theory_nw`` and
    ``cov_xi_iso`` (with ``power`` / ``power_nw``), the bin ``axes``, ``min_one_plus_delta`` (n,) (the smallest n_X / n_bar
    of the cells of every realisation, n_bar the volume-weighted mean), ``limits`` (``grid_limits``), ``mas``,
    ``interlace``, ``n_mesh``, ``k_nyq`` (pi / H), ``trusted`` (the trusted range of the mesh path: k_max = k_Nyq / 2,
    r_min = 6 H; ``mesh.TRUSTED_K_NYQ``, ``TRUSTED_R_H``), ``n_cells`` (the cells of the last realisation), ``time_s`` and
    ``time_cov_s``."""
    gs = grid
    names = [species] if isinstance(species, str) else list(species)
    t0 = time.perf_counter()
    recs, mins, n_cells = {sp: [] for sp in names}, {sp: [] for sp in names}, 0
    for n, c in enumerate(_as_cells_iter(cells)):
        if c.box is not None and not np.allclose(c.box, gs.L):
            raise ValueError(f"the cells' box ({c.box} cMpc) is not the grid's ({gs.L} cMpc).")
        x = np.asarray(c.positions, dtype=np.float64)
        V = np.asarray(c.volumes, dtype=np.float64)
        n_cells = len(c)
        for sp in names:
            w = cell_weights(c, sp)
            n_sp = np.asarray(c.species[sp]["number_density"], dtype=np.float64)
            mins[sp].append(float(n_sp.min() / (np.sum(n_sp * V) / V.sum())))
            recs[sp].append(gs.measure_points(x, w, mas=mas, interlace=interlace, chunk=chunk))
        if log and (n + 1) % 16 == 0:
            log(f"field statistics: {n + 1} realisations, {time.perf_counter() - t0:.0f} s")
    t_meas = time.perf_counter() - t0
    on_modes = lambda f: grid_power(f, gs.shape, gs.L)  # noqa: E731
    out = {}
    for sp in names:
        res = _ensemble(recs[sp], mins[sp], gs, power, power_nw, on_modes, t_meas)
        k_nyq = float(np.min(np.pi / gs.dx))
        trusted = dict(k_max=TRUSTED_K_NYQ * k_nyq, r_min=TRUSTED_R_H * float(np.max(gs.dx)))
        res.update(mas=mas, interlace=bool(interlace), n_mesh=gs.shape, k_nyq=k_nyq, trusted=trusted, n_cells=n_cells)
        out[sp] = res
    return out[names[0]] if isinstance(species, str) else out


def _ensemble(recs, mins, gs, power, power_nw, on_modes, t_meas) -> dict:
    """The ensemble mean of the measurements of one field, the expectations (``on_modes``: a callable P(k) -> P on the
    rfft modes), the covariance of the isotropic xi and the axes (``field_statistics``)."""
    res = ensemble_mean(recs, GRID_KEYS)
    P = None if power is None else on_modes(power)
    if P is not None:
        res["theory"] = gs.theory(P)
    if power_nw is not None:
        res["theory_nw"] = gs.theory(on_modes(power_nw))
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


def xi_corner(
    res, *, grid: GridStatistics3D | None = None, power: Callable | None = None, r_perp_max=None, n_rows=3, n_alt=16
) -> dict:
    """The corner of the (r_perp, r_par) map of a ``field_statistics`` result: the last ``n_rows`` rows (r_par up to
    L / 2) and the columns with r_perp < ``r_perp_max`` [cMpc] (default L / 15). Near L / 2 on the line of sight r^2 is at
    its largest and the noise of a lag is largest (2 r = 0 modulo L, ``GridStatistics3D.xi_map_sigma``), so a fluctuation
    of the few largest modes shows here first (notebooks 08 and 09).

    Returns dict(rows (r_par), columns (r_perp), value (the mean of r^2 xi over the cells [cMpc^2]), expected (the same for
    ``res["theory"]``, NaN without), offset (value - expected, or value), each (n,) (the offset of every realisation), error
    (its error of the mean, from their scatter), sigma (with ``grid`` and ``power``, a callable P(k): the exact Gaussian error
    of the mean of the n realisations; NaN without), alternating (the mean over the last ``n_alt`` rows of (-1)^j r^2 xi in
    the column r_perp = 0, j the row, minus the same for the expectation: a (-1)^j pattern along the line of sight, e.g. of
    the kernel of the mass assignment) and alternating_err)."""
    a = res["axes"]
    rpar, rperp, each = a["rpar"], a["rperp"], np.asarray(res["xi_map_each"], dtype=float)
    L = 2.0 * float(rpar[-1])
    r_perp_max = L / 15.0 if r_perp_max is None else float(r_perp_max)
    rows, cols = np.arange(rpar.size - int(n_rows), rpar.size), np.flatnonzero(rperp < r_perp_max)
    r2 = np.nan_to_num(a["rpar_map"]) ** 2 + np.nan_to_num(a["rperp_map"]) ** 2
    sel = np.ix_(rows, cols)
    has_th = "theory" in res
    th = res["theory"]["xi_map"] if has_th else np.zeros_like(r2)
    mean = lambda x: np.nanmean(r2[sel] * x[..., sel[0], sel[1]], axis=(-2, -1))  # noqa: E731
    per_real = mean(each - th)
    n = per_real.size
    sigma = np.nan
    if grid is not None and power is not None:
        weights = np.zeros_like(r2)
        weights[sel] = r2[sel] / (rows.size * cols.size)
        sigma = grid.xi_map_sigma(grid_power(power, grid.shape, grid.L), weights) / np.sqrt(n)
    last = np.arange(rpar.size - int(n_alt), rpar.size)
    sign = (-1.0) ** last
    alt = np.mean(sign * r2[last, 0] * (each[:, last, 0] - th[last, 0]), axis=1)
    err = lambda x: float(x.std(ddof=1) / np.sqrt(x.size)) if x.size > 1 else np.nan  # noqa: E731
    return dict(
        rows=rpar[rows],
        columns=rperp[cols],
        value=float(mean(np.asarray(res["xi_map"]))),
        expected=float(mean(np.asarray(th))) if has_th else np.nan,
        offset=float(per_real.mean()),
        each=per_real,
        error=err(per_real),
        sigma=float(sigma),
        alternating=float(alt.mean()),
        alternating_err=err(alt),
    )
