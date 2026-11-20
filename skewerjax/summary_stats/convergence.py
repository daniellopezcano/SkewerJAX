"""The statistics of the overflux on nested lattices of skewers and their convergence with the number of skewers
(notebook 18; the lattices of ``simulations.skewer_store``).

``flux_statistics`` is notebook 11's chain for one lattice, from the optical depth of an ensemble (the lines of sight
x, y, z): T = e^-tau, its mean T_bar (``ensemble_mean_transmissions``), the overflux delta = T / T_bar - 1 of every line
and of the total, then P1D, xi1D (``skewer_p1d``, ``skewer_xi1d``) and the 3D statistics on the lattice
(``measure_lattice``), each per line of sight, and their mean over the ensemble with its error (``ensemble_mean``).

``lattice_convergence`` compares the statistics of every lattice N with those of the largest one, N_max: the ratio X_N / X_max
on the bins of each statistic (P1D on its native modes k > 0, or in the opt-in bins ``k_edges_1d``, P and xi in their wedges of mu on the |k| and |r| bins of
the lattice; a lattice of fewer skewers covers fewer |k| bins, up to its Nyquist frequency pi N / L, and no |r| bin
without pairs), and, per statistic, the largest |ratio - 1| over its bins against N, next to the relative error of the
N_max estimate (the error of the mean over the lines of sight over |X_max|, its median over the same bins): a lattice
has converged where its deviation is below that error. xi changes sign: its ratio is kept only where |xi_max| exceeds
``xi_sigma`` times its error.
"""

import time

import numpy as np

from skewerjax.summary_stats.binning import bin_index, ensemble_mean
from skewerjax.summary_stats.grid3d import GRID_KEYS
from skewerjax.summary_stats.lattice import measure_lattice, skewer_p1d, skewer_xi1d
from skewerjax.summary_stats.overflux import ensemble_mean_transmissions, overflux_check


def flux_statistics(
    tau, dv: float, grid, *, line_ids, mean: str = "redshift", a_pix=None, keep_delta: bool = False, log=None
) -> dict:
    """Notebook 11's statistics of the overflux of one lattice (module docstring).

    tau : (n_ax, n_lines, N_sk, N_pix) the optical depth of every line of sight (the ensemble), the skewers a-major
    (``core.lattice_columns``); dv : the pixel [km/s]; grid : ``lattice_grid`` of the lattice (the 3D statistics);
    line_ids : the lines (the fields: every line and "total"); mean, a_pix : ``ensemble_mean_transmissions``'s mean and
    the scale factor of every pixel (``mean="redshift"``); log : None or a callable, given one line after each step
    (the transmissions and overfluxes, then per line of sight P1D and xi1D, and the 3D statistics of each field) with
    its time: ``flux_statistics_steps`` is their number (for a progress bar).

    Returns dict(fields, T_bar (n_ax, n_lines + 1, N_sk, N_pix) (a view), check (``overflux_check``), p1d (``ensemble_mean``
    of "p1d" (n_fields, N_pix // 2 + 1), with "k" [s/km]), xi1d (with "lags" [km/s]), flux3d {field: the mean of the
    ``GRID_KEYS``}, axes3d (``grid.axes()``), and delta (n_ax, n_fields, N_sk, N_pix) with ``keep_delta``."""
    tau = np.asarray(tau)
    fields = (*line_ids, "total")
    clock = [time.perf_counter()]

    def step(msg):
        if log is not None:
            t = time.perf_counter()
            log(f"{msg} ({t - clock[0]:.1f} s)")
            clock[0] = t

    T = np.exp(-tau)
    T_bar = ensemble_mean_transmissions(T, mean=mean, grid=a_pix)
    delta = np.concatenate([T, T.prod(axis=1, keepdims=True)], axis=1) / T_bar - 1
    del T
    check = overflux_check(delta, T_bar, mean=mean, grid=a_pix)
    step(f"transmissions and overfluxes of {delta.shape[0]} x {delta.shape[2]:,} skewers")
    p1d, xi1d, f3 = [], [], []
    for i in range(delta.shape[0]):
        k, p = skewer_p1d(delta[i], dv)
        lags, x = skewer_xi1d(delta[i], dv)
        p1d.append(dict(p1d=p))
        xi1d.append(dict(xi1d=x))
        step(f"line of sight {i + 1}/{delta.shape[0]}: P1D and xi1D")
        r = {}
        for j, f in enumerate(fields):
            r[f] = measure_lattice(delta[i, j], grid)
            step(f"line of sight {i + 1}/{delta.shape[0]}: 3D statistics of {f}")
        f3.append(r)
    out = dict(
        fields=fields,
        T_bar=T_bar,
        check=check,
        p1d=dict(ensemble_mean(p1d, ("p1d",)), k=k),
        xi1d=dict(ensemble_mean(xi1d, ("xi1d",)), lags=lags),
        flux3d={f: ensemble_mean([r[f] for r in f3], GRID_KEYS) for f in fields},
        axes3d=grid.axes(),
    )
    if keep_delta:
        out["delta"] = delta
    return out


def flux_statistics_steps(n_axes: int, n_lines: int) -> int:
    """The number of lines ``flux_statistics`` gives its ``log``: 1 + n_axes (1 + n_lines + 1)."""
    return 1 + int(n_axes) * (int(n_lines) + 2)


def binned_p1d(p1d: dict, k_edges=None) -> dict:
    """P1D of ``flux_statistics`` in the bins ``k_edges`` [s/km]: the mean over the modes of each bin, per line of sight,
    then over the ensemble: dict(k (the mean k of each bin), p1d, p1d_err (n_fields, n_bins), n_modes); the empty bins
    dropped (k = 0, the mean, is outside every bin from k_f). ``k_edges`` None (default): the native modes k_m = 2 pi m
    / v_per, m = 1 .. N_pix / 2, one per bin (no averaging; k = 0 left out)."""
    k = np.asarray(p1d["k"], dtype=float)
    if k_edges is None:  # one bin per native mode: edges half way between the modes, from half the first
        kp = k[k > 0]
        edges = np.concatenate([[0.5 * kp[0]], 0.5 * (kp[1:] + kp[:-1]), [1.5 * kp[-1] - 0.5 * kp[-2]]])
    else:
        edges = np.asarray(k_edges, dtype=float)
    j = bin_index(k, edges)
    nb = edges.size - 1
    ok = j >= 0
    cnt = np.bincount(j[ok], minlength=nb).astype(float)
    each = np.asarray(p1d["p1d_each"])[..., ok]  # (n_ax, n_fields, n_modes)
    b = np.stack([[np.bincount(j[ok], weights=f, minlength=nb) for f in e] for e in each]) / np.maximum(cnt, 1)
    n = b.shape[0]
    keep = cnt > 0
    err = b.std(0, ddof=1) / np.sqrt(n) if n > 1 else np.full(b.shape[1:], np.nan)
    kc = np.bincount(j[ok], weights=k[ok], minlength=nb) / np.maximum(cnt, 1)
    return dict(k=kc[keep], p1d=b.mean(0)[:, keep], p1d_err=err[:, keep], n_modes=cnt[keep])


def _pad(x, n: int) -> np.ndarray:
    """``x`` (..., m) padded with NaN to (..., n) (a lattice of fewer skewers has fewer |k| bins: a prefix of the edges)."""
    x = np.asarray(x, dtype=float)
    if x.shape[-1] >= n:
        return x[..., :n]
    return np.concatenate([x, np.full(x.shape[:-1] + (n - x.shape[-1],), np.nan)], axis=-1)


def _deviation(ratio) -> float:
    r = np.asarray(ratio, dtype=float)
    r = r[np.isfinite(r)]
    return float(np.max(np.abs(r - 1))) if r.size else np.nan


def _median_rel_err(val, err, mask=None) -> float:
    v, e = np.asarray(val, dtype=float), np.asarray(err, dtype=float)
    m = np.isfinite(v) & np.isfinite(e) & (v != 0)
    if mask is not None:
        m &= mask
    return float(np.median(np.abs(e[m] / v[m]))) if m.any() else np.nan


def lattice_convergence(stats: dict, *, k_edges_1d=None, p1d_k_max=None, fields_3d=("total",), xi_sigma: float = 2.0) -> dict:
    """The statistics of every lattice against the largest (module docstring).

    stats : {N: ``flux_statistics`` of the N x N lattice}, every lattice on the same |k| and |r| edges (those of the
    largest, cut by ``lattice_grid`` at each lattice's Nyquist frequency) and the same mu wedges; k_edges_1d : None
    (default: P1D on its native modes k > 0, ``binned_p1d``) or the P1D bins [s/km] (opt-in); p1d_k_max : None (every bin) or the largest k [s/km] of the P1D summary (e.g. the range of a
    measurement: the high-k power of a few skewers is dominated by their rare strong absorbers); fields_3d : the fields
    of the 3D statistics compared (default the total).

    Returns dict(n (the N, increasing), n_max, fields, p1d: dict(k, ref, ref_err, ratio {N: (n_fields, n_bins)}),
    P_mu / xi_mu {field: dict(x (the bin centres of the reference, (n_wedges, n_bins)), ref, ref_err, ratio {N: ...},
    mask (xi: |xi_max| > xi_sigma error))}, and summary {name: dict(n, deviation (max |ratio - 1| per N), ref_err (the
    median relative error of the N_max estimate))} with the names "P1D <field>", "P(k, mu) <field>", "xi(r, mu) <field>"."""
    ns = sorted(int(n) for n in stats)
    top = ns[-1]
    ref = stats[top]
    fields = ref["fields"]
    b_ref = binned_p1d(ref["p1d"], k_edges_1d)
    out = dict(n=ns, n_max=top, fields=fields, summary={}, p1d_k_max=None if p1d_k_max is None else float(p1d_k_max))
    ratios = {}
    for n in ns:
        b = binned_p1d(stats[n]["p1d"], k_edges_1d)
        if b["p1d"].shape != b_ref["p1d"].shape:
            raise ValueError("every lattice must have the same pixels (the P1D bins differ).")
        ratios[n] = b["p1d"] / b_ref["p1d"]
    out["p1d"] = dict(k=b_ref["k"], ref=b_ref["p1d"], ref_err=b_ref["p1d_err"], ratio=ratios)
    kin = np.ones(b_ref["k"].size, bool) if p1d_k_max is None else b_ref["k"] <= float(p1d_k_max)
    for j, f in enumerate(fields):
        out["summary"][f"P1D {f}"] = dict(
            n=ns,
            deviation=[_deviation(ratios[n][j][kin]) for n in ns],
            ref_err=_median_rel_err(b_ref["p1d"][j][kin], b_ref["p1d_err"][j][kin]),
        )
    ax_ref = ref["axes3d"]
    for q, xkey in (("P_mu", "k_mu"), ("xi_mu", "r_mu")):
        out[q] = {}
        for f in fields_3d:
            R = np.asarray(ref["flux3d"][f][q], dtype=float)
            E = np.asarray(ref["flux3d"][f][q + "_err"], dtype=float)
            mask = np.isfinite(R) & (np.abs(R) > xi_sigma * E) if q == "xi_mu" else np.isfinite(R)
            rq = {}
            for n in ns:
                X = _pad(stats[n]["flux3d"][f][q], R.shape[-1])
                rq[n] = np.where(mask, X / np.where(R != 0, R, np.nan), np.nan)
            out[q][f] = dict(x=np.asarray(ax_ref[xkey], dtype=float), ref=R, ref_err=E, ratio=rq, mask=mask)
            name = ("P(k, mu) " if q == "P_mu" else "xi(r, mu) ") + f
            out["summary"][name] = dict(n=ns, deviation=[_deviation(rq[n]) for n in ns], ref_err=_median_rel_err(R, E, mask))
    return out
