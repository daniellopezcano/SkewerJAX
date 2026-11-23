"""The statistics of the overflux on lattices of skewers and their convergence: with the number of skewers on nested
lattices (``flux_statistics``, ``lattice_convergence``), and between the runs of a resolution sweep on one line of sight
(``run_statistics``, ``compare_runs``, ``k_max_table``; notebooks 18-21, the run folders of ``scripts/production``).

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

**Runs of a resolution sweep** (one line of sight, one lattice per run; notebooks 18-21). ``run_statistics`` gives, for
one run, the mean transmission, P1D on its native modes and P(k, mu) of the chosen fields (every line and the total) with
their delete-one-block jackknife samples: the n x n lattice is cut into B x B squares of skewers (``block_labels``), and
each sample leaves one square out (the mean and P1D over the other skewers; P(k, mu) of the field with that square set to
zero, divided by the fraction of the skewers kept). ``compare_runs`` takes the relative difference X_run / X_ref - 1 per
bin, its error from the jackknife samples of the ratio (the runs of a sweep share the skewer positions and the blocks, so
the sample variance they share cancels in the ratio), and ``k_max_table`` the largest k up to which every bin differs by
less than a tolerance. ``rebin_pixels`` averages a finer run's transmission over the pixels of a coarser one.

**The decomposition of a run** (``run_statistics(terms_grid=)``; notebooks 18-21, Parts D and E): P1D and P(k, mu) of the
total split exactly into the pairs of subset fields of ``decomposition`` (F_{} = -C / (1 + C), F_A = delta_A / (1 + C);
``docs/decomposition.md``), grouped by order |A| + |B| and, within order 2, by pair (the autos, the crosses, the C
term), each with the jackknife samples of the total, on the wedges of a ``LatticeWedges`` (e.g. the 16 wedges of
notebook 11). Streamed (``_streamed_terms``) so that 512^2 lattices fit: the transforms along the line of sight first,
then across the lattice one slab of k_par at a time, binned in float64. ``compare_runs`` compares the decompositions of
two runs, with ``normalised_difference`` for the quantities that change sign.
"""

import os
import time

import numpy as np

from skewerjax.summary_stats.binning import bin_index, ensemble_mean
from skewerjax.summary_stats.grid3d import GRID_KEYS
from skewerjax.summary_stats.lattice import measure_lattice, on_lattice, skewer_p1d, skewer_xi1d
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


# --------------------------------------------------------------------------- the runs of a resolution sweep
def block_labels(columns, n: int, n_blocks: int) -> np.ndarray:
    """The jackknife block of every skewer of an n x n lattice: columns (N_sk, 2) its (i_a, i_b) on the lattice
    (``StoredSkewers.columns``); the B x B squares of side n / B skewers, label (i_a // s) B + i_b // s, s = n / B."""
    n, b = int(n), int(n_blocks)
    if b < 2 or n % b:
        raise ValueError(f"n_blocks must divide n = {n} and be >= 2, got {n_blocks}.")
    c = np.asarray(columns, dtype=np.int64).reshape(-1, 2)
    s = n // b
    return (c[:, 0] // s) * b + c[:, 1] // s


def jackknife_error(samples) -> np.ndarray:
    """The delete-one jackknife error of an estimate from its samples (B, ...): sqrt((B - 1) / B sum (x_i - x_mean)^2)."""
    x = np.asarray(samples, dtype=float)
    b = x.shape[0]
    return np.sqrt((b - 1) / b * np.sum((x - x.mean(axis=0)) ** 2, axis=0))


def lattice_power(delta, grid, keys=("P_mu",)) -> dict:
    """The binned P of one field along the skewers of a lattice, delta (N_sk, N_pix), on ``grid`` (``lattice_grid``): only
    ``keys`` of the ``GRID_KEYS`` of P (default the mu wedges), as ``measure_lattice`` without xi (one rfftn)."""
    import jax.numpy as jnp

    d = jnp.asarray(on_lattice(delta, grid.shape[0], grid.shape[2]))
    f2 = (jnp.abs(jnp.fft.rfftn(d)) ** 2).at[0, 0, 0].set(0.0)
    return grid._bin((grid.dV / grid.N) * f2, None, keys=tuple(keys))


class LatticeWedges:
    """The mu wedges of P on the lattice of n x n skewers of n_los pixels over the period L [cMpc]: the "P_mu" of
    ``lattice_grid`` alone, with its bins (the |k| edges cut at the smallest Nyquist frequency, ``full_coverage``; the
    wedges in mu = k_par / |k|, half-open, the last closed; the rfft modes weighted 2, 1 on the k_z = 0 and Nyquist planes;
    k = 0 excluded) and its bin centres ``k_mu`` (the mode-weighted mean |k|), without the other bins of
    ``GridStatistics3D``: its separations alone (float64 over the n^2 n_los lags) took 41 GB on 256^2 TNG50-2 skewers of
    3949 pixels, ~160 GB on 512^2. Here one int32 bin index per rfft mode (2.1 GB on 512^2), built and used slab by slab
    (``power``), or none (``store_index=False``): the bins of each slab of k_par computed when it is used
    (``slab_index``: the streamed decomposition of ``run_statistics``, its transform along the line of sight first). Both
    layouts hold every mode of the full grid once, with the same |k| and mu to the last bit.

    Attributes: ``k_mu_edges``, ``pk_mu_edges``, ``k_mu`` (n_mu, n_k), ``n_modes`` (n_mu, n_k) (the summed weights)."""

    def __init__(self, n: int, n_los: int, L: float, *, k_edges, mu_edges=None, store_index: bool = True, log=None):
        from skewerjax.summary_stats.binning import full_coverage
        from skewerjax.summary_stats.grid3d import MU_WEDGES, _mu_edges, _rfft_weights

        self.n, self.n_los, self.L = int(n), int(n_los), float(L)
        d_perp = self.L / self.n
        k_nyq = float(min(np.pi * self.n_los / self.L, np.pi / d_perp))
        self.k_mu_edges = full_coverage(k_edges, k_nyq, "|k| bins", log)
        self.pk_mu_edges = _mu_edges(MU_WEDGES if mu_edges is None else mu_edges, "pk_mu_edges")
        n_mu, n_k = self.pk_mu_edges.size - 1, self.k_mu_edges.size - 1
        self.shape = (n_mu, n_k)
        self._kx = 2 * np.pi * np.fft.fftfreq(self.n, d_perp)  # lattice axis 0 (i_x), and axis 2 (i_z) in full
        self._kl = 2 * np.pi * np.fft.fftfreq(self.n_los, self.L / self.n_los)  # the line of sight
        self._kz = 2 * np.pi * np.fft.rfftfreq(self.n, d_perp)
        self._w = _rfft_weights(self.n)  # (n // 2 + 1,): the full-grid modes of each rfft column
        self._w_los = _rfft_weights(self.n_los)  # (n_los // 2 + 1,): the same along the line of sight (slab_index)
        self._norm = d_perp * d_perp * (self.L / self.n_los) / (self.n * self.n * self.n_los)  # dV / N
        w_sum, k_sum = np.zeros(n_mu * n_k + 1), np.zeros(n_mu * n_k + 1)
        if store_index:
            self._idx = np.empty((self.n, self.n_los, self._kz.size), np.int32)
            k_lz = np.sqrt(self._kl[:, None] ** 2 + self._kz[None, :] ** 2)  # without k_x
            kpar = np.broadcast_to(np.abs(self._kl)[:, None], k_lz.shape)
            wts = np.broadcast_to(self._w, k_lz.shape).ravel()
            for i in range(self.n):  # one slab of k_x at a time: no array of the full grid but the index
                idx, kmod = self._cell_index(self._kx[i] ** 2, k_lz, kpar)
                self._idx[i] = idx
                w_sum += np.bincount(idx.ravel() + 1, weights=wts, minlength=w_sum.size)
                k_sum += np.bincount(idx.ravel() + 1, weights=wts * kmod.ravel(), minlength=k_sum.size)
        else:
            self._idx = None
            step = max(1, int(2**24 // (self.n * self.n)))
            for m0 in range(0, self.n_los // 2 + 1, step):
                idx, wm, kmod = self.slab_index(m0, min(m0 + step, self.n_los // 2 + 1), with_k=True)
                wts = np.broadcast_to(wm[:, None, None], idx.shape).ravel()
                w_sum += np.bincount(idx.ravel() + 1, weights=wts, minlength=w_sum.size)
                k_sum += np.bincount(idx.ravel() + 1, weights=wts * kmod.ravel(), minlength=k_sum.size)
        self.n_modes = w_sum[1:].reshape(self.shape)
        with np.errstate(invalid="ignore", divide="ignore"):
            self.k_mu = np.where(self.n_modes > 0, k_sum[1:].reshape(self.shape) / self.n_modes, np.nan)

    def _cell_index(self, kx2, k_lz, kpar):
        """The flat bin (mu wedge, |k| bin) of every mode, -1 outside or at k = 0, and its |k|: |k| = sqrt(k_x^2 +
        k_lz^2), k_lz = sqrt(k_par^2 + k_z^2) (one formula for both layouts: the same bins to the last bit)."""
        from skewerjax.summary_stats.grid3d import _flat2

        kmod = np.sqrt(kx2 + k_lz**2)
        live = kmod > 0
        mu = np.where(live, kpar / np.where(live, kmod, 1.0), 0.0)
        n_k = self.shape[1]
        idx = np.where(live, _flat2(bin_index(mu, self.pk_mu_edges), bin_index(kmod, self.k_mu_edges), n_k), -1)
        return idx.astype(np.int32), kmod

    def slab_index(self, m0: int, m1: int, with_k: bool = False):
        """The bins of the modes k_par = 2 pi m / L, m0 <= m < m1 (m <= n_los // 2: the rfft along the line of sight),
        in the layout of a transform along the line of sight first, then across the lattice: (m, k_x, k_z), k_x and k_z
        over the full lattice (``np.fft.fft2`` of the skewers x-major, ``on_lattice``). Returns idx (m1 - m0, n, n) int32
        (the flat bin, -1 outside or at k = 0), wm (m1 - m0,) the full-grid modes each column stands for (2, and 1 at
        k_par = 0 and the Nyquist plane), and |k| with ``with_k``."""
        m = np.arange(int(m0), int(m1))
        kl = np.abs(self._kl[m])  # (ns,): the rfft frequencies are those of the fft at m >= 0
        kz = self._kx  # the full k_z axis: (-k)^2 = k^2 exactly
        k_lz = np.sqrt(kl[:, None, None] ** 2 + kz[None, None, :] ** 2)  # (ns, 1, n)
        idx, kmod = self._cell_index(self._kx[None, :, None] ** 2, k_lz, np.broadcast_to(kl[:, None, None], (m.size, 1, 1)))
        out = (idx, self._w_los[m].astype(float))
        return out + (kmod,) if with_k else out

    def power(self, delta) -> np.ndarray:
        """(n_mu, n_k) the binned P [cMpc^3] of one field along the skewers, delta (N_sk, n_los) x-major (as
        ``lattice_power``: one rfftn in the package's precision; the bins summed in float64); NaN in empty bins."""
        import jax.numpy as jnp

        if self._idx is None:
            raise ValueError("this LatticeWedges has no index (store_index=False): use it through run_statistics(terms_grid=).")
        f = jnp.fft.rfftn(jnp.asarray(on_lattice(delta, self.n, self.n)))
        f2 = np.asarray(jnp.abs(f) ** 2)
        del f
        acc = np.zeros(self.n_modes.size + 1)
        wts = np.broadcast_to(self._w, f2.shape[1:]).ravel()
        for i in range(self.n):
            acc += np.bincount(self._idx[i].ravel() + 1, weights=wts * f2[i].ravel(), minlength=acc.size)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(self.n_modes > 0, self._norm * acc[1:].reshape(self.shape) / self.n_modes, np.nan)


def _wedge_power(delta, grid) -> np.ndarray:
    """The mu wedges of P of one field on ``grid``: a ``LatticeWedges`` or a ``lattice_grid``."""
    if isinstance(grid, LatticeWedges):
        return grid.power(delta)
    return np.asarray(lattice_power(delta, grid)["P_mu"], dtype=float)


def rebin_pixels(F, n_coarse: int) -> np.ndarray:
    """The average of F (..., N_fine), piecewise constant on N_fine pixels over one period, over each of ``n_coarse``
    pixels of the same period: every fine pixel weighted by its overlap with the coarse one (the grids need not nest; a
    fine pixel overlaps at most two coarse ones). The transmission of a coarser pixel grid seen by an instrument that
    averages the flux over each pixel."""
    from scipy import sparse

    F = np.asarray(F)
    n_f, n_c = F.shape[-1], int(n_coarse)
    if not 1 <= n_c <= n_f:
        raise ValueError(f"n_coarse must be in [1, {n_f}], got {n_coarse}.")
    j = np.arange(n_f)
    lo, hi = j * n_c, (j + 1) * n_c  # the fine pixel j in units of 1 / (n_f n_c) of the period (a coarse pixel: n_f)
    c0 = lo // n_f  # the coarse pixel of its left edge
    split = np.minimum((c0 + 1) * n_f, hi)  # the right edge of that coarse pixel, or of the fine one
    rows = np.concatenate([j, j])
    cols = np.concatenate([c0, np.minimum(c0 + 1, n_c - 1)])
    w = np.concatenate([split - lo, hi - split]) / n_f  # overlap / the coarse pixel's width
    W = sparse.csr_matrix((w, (rows, cols)), shape=(n_f, n_c))
    flat = F.reshape(-1, n_f)
    out = np.empty((flat.shape[0], n_c))
    step = max(1, int(2e8 // n_f))  # rows per chunk (~1.6 GB of float64)
    for a in range(0, flat.shape[0], step):
        out[a : a + step] = (W.T @ flat[a : a + step].astype(np.float64).T).T
    return out.reshape(F.shape[:-1] + (n_c,))


# --------------------------------------------------------------------------- the decomposition of a run, streamed
_KERNELS = {}


def _pair_kernel():
    """The numba kernel of the streamed decomposition (compiled on first use, no disk cache): for every sample s and
    pair p = (a, b) of fields, the float64 bin sums of w Re(F_a F_b^*), times 2 for a != b (the symmetric sum P_ab + P_ba
    of ``decomposition``), over the modes of a slab (w: the full-grid modes each stands for); parallel over the samples."""
    if "pairs" in _KERNELS:
        return _KERNELS["pairs"]
    import numba

    @numba.njit(parallel=True)
    def pair_bins(F, pa, pb, idx, wm, n_per_m, n_bins):  # F (n_samples, n_fields, n_modes) complex
        n_s, _, n_m = F.shape
        n_p = pa.size
        acc = np.zeros((n_s, n_p, n_bins))
        for s in numba.prange(n_s):
            for i in range(n_m):
                j = idx[i]
                if j < 0:
                    continue
                w = wm[i // n_per_m]
                for p in range(n_p):
                    a = F[s, pa[p], i]
                    b = F[s, pb[p], i]
                    v = np.float64(a.real) * np.float64(b.real) + np.float64(a.imag) * np.float64(b.imag)
                    acc[s, p, j] += (w if pa[p] == pb[p] else 2.0 * w) * v
        return acc

    _KERNELS["pairs"] = pair_bins
    return pair_bins


def _fft_across(x, keep):
    """The 2D transform across the lattice (the last two axes, (i_x, i_z)) of the slab x (n_fields, n_m, n, n) times each
    sample's mask ``keep`` (n_samples, n, n): (n_samples, n_fields, n_m, n, n), in the package's precision (JAX)."""
    import jax
    import jax.numpy as jnp

    if "fft2" not in _KERNELS:
        _KERNELS["fft2"] = jax.jit(lambda a, k: jnp.fft.fft2(a[None] * k[:, None, None], axes=(-2, -1)))
    return _KERNELS["fft2"](x, keep)


def _streamed_terms(
    T, T_bar, labels, dv: float, line_ids, grid, check_grid=None, *, decompose: bool = True, memory: float = 2**30, log=None
) -> dict:
    """The decomposition of one run (``run_statistics(terms_grid=)``): P1D and the mu wedges of P of every pair of subset
    fields of ``decomposition`` and of the total, with their delete-one-block jackknife samples (module docstring of
    ``decomposition``: F_{} = -C / (1 + C), F_A = delta_A / (1 + C), the pair (A, B) its symmetric sum; the orders and the
    order-2 terms are sums of pairs).

    T (n_lines + 1, N_sk, N_pix) the transmissions (the total last), T_bar their means (shared by the skewers: a row per
    field), labels the jackknife block of every skewer, grid the ``LatticeWedges`` of the wedges, check_grid optional
    (the run's own ``LatticeWedges``: the total also in its bins, the check against ``LatticeWedges.power``); decompose
    False: the total alone; memory [bytes]: the size of a chunk of skewers and of a slab of k_par.

    Streamed, one run's fields at a time: per chunk of skewers, the fields and their transform along the line of sight in
    float64 (SciPy): P1D of every pair (the block sums of Re X_a X_b^*, as ``skewer_p1d``), and the transforms kept in the
    package's precision as (N_pix // 2 + 1, N_sk); then per slab of k_par, the 2D transform across the lattice of each
    field and of its 16 (B) jackknife copies (the block set to zero), the pair products binned in float64 (numba) with
    the bins of ``LatticeWedges.slab_index``; the slab freed. The field F_{} of a C that is the same at every pixel is
    the constant -C / (1 + C): its transform is exactly zero at k_par != 0, and it is skipped there."""
    import jax.numpy as jnp
    import scipy.fft

    from skewerjax.summary_stats.decomposition import subset_terms

    clock = [time.perf_counter()]

    def step(msg):
        if log is not None:
            t = time.perf_counter()
            log(f"{msg} ({t - clock[0]:.1f} s)")
            clock[0] = t

    n_lines = T.shape[0] - 1
    n_sk, n_pix = T.shape[1], T.shape[2]
    n = grid.n
    if n * n != n_sk or grid.n_los != n_pix:
        raise ValueError(f"terms_grid is a {n}^2 lattice of {grid.n_los} pixels, the run {n_sk} skewers of {n_pix}.")
    if check_grid is not None and (check_grid.n != n or check_grid.n_los != n_pix):
        check_grid = None
    Tb = np.asarray(T_bar[:, 0, :], dtype=np.float64)  # a row per field: the mean is shared by the skewers
    C = Tb[-1] / np.prod(Tb[:-1], axis=0) - 1.0  # the correlator of the mean mode (decomposition, Sec. 1)
    winv = 1.0 / (1.0 + C)
    keys, members, orders = subset_terms(line_ids, "+")
    n_t = len(keys)
    tot = n_t  # the field index of the total
    pairs = [(a, b) for a in range(n_t) for b in range(a, n_t)] if decompose else []
    pairs.append((tot, tot))
    n_p = len(pairs)
    M = n_pix // 2 + 1
    real = np.dtype(jnp.zeros(()).dtype)
    cdt = np.complex64 if real == np.float32 else np.complex128
    workers = max(1, min(16, (os.cpu_count() or 2) // 2))
    if np.all(C == C[0]):  # the transform of the constant F_{}: n_pix (-C / (1 + C)) at k_par = 0, exactly zero elsewhere
        X0 = np.zeros(M, np.complex128)
        X0[0] = -C[0] / (1.0 + C[0]) * n_pix
    else:
        X0 = scipy.fft.rfft(-C * winv)
    fields = (list(range(1, n_t)) if decompose else []) + [tot]  # the stored transforms; F_{} is the row X0
    store = {f: np.empty((M, n_sk), cdt) for f in fields}
    nb = int(labels.max()) + 1
    cnt = np.bincount(labels, minlength=nb).astype(float)
    p1d = np.zeros((nb, n_p, M))
    ch = max(1, int(memory // (16 * M * (len(fields) + 1))))
    for s0 in range(0, n_sk, ch):
        s1 = min(n_sk, s0 + ch)
        X = {0: X0[None]}
        if decompose:
            d = T[:n_lines, s0:s1] / Tb[:n_lines, None, :] - 1.0  # the overflux of every line (float64)
            for t in range(1, n_t):
                f = d[members[t][0]] * winv
                for j in members[t][1:]:
                    f = f * d[j]
                X[t] = scipy.fft.rfft(f, axis=-1, workers=workers)
            del d, f
        X[tot] = scipy.fft.rfft(T[-1, s0:s1] / Tb[-1] - 1.0, axis=-1, workers=workers)
        for t in fields:
            store[t][:, s0:s1] = X[t].T
        one = (labels[s0:s1][None, :] == np.arange(nb)[:, None]).astype(np.float64)  # (B, chunk)
        for p, (a, b) in enumerate(pairs):
            xa, xb = X[a], X[b]
            v = xa.real * xb.real + xa.imag * xb.imag
            v = np.broadcast_to(v if a == b else 2.0 * v, (s1 - s0, M))
            p1d[:, p] += one @ v
        del X
    step(f"the transforms along the line of sight and P1D of {n_p} pairs")

    S = nb + 1  # the samples: every skewer, then each block left out
    keep = np.ones((S, n_sk), real)
    for b in range(nb):
        keep[b + 1, labels == b] = 0.0
    frac = keep.mean(axis=1)
    keep_l = jnp.asarray(keep.reshape(S, n, n))
    nbins = grid.shape[0] * grid.shape[1]
    acc = np.zeros((S, n_p, nbins))
    acc_chk = np.zeros(check_grid.shape[0] * check_grid.shape[1]) if check_grid is not None else None
    kernel = _pair_kernel()
    per_m = S * (len(fields) + 1) * n * n * np.dtype(cdt).itemsize * 3  # the slab, its transforms and their copy
    ns = max(1, int(memory // per_m))
    for m0 in range(0, M, ns):
        m1 = min(M, m0 + ns)
        here = ([0] if decompose and np.any(X0[m0:m1]) else []) + fields
        loc = {f: i for i, f in enumerate(here)}
        xs = np.stack(
            [np.broadcast_to(X0[m0:m1, None], (m1 - m0, n_sk)).astype(cdt) if f == 0 else store[f][m0:m1] for f in here]
        ).reshape(len(here), m1 - m0, n, n)
        Fk = np.asarray(_fft_across(jnp.asarray(xs), keep_l)).reshape(S, len(here), -1)
        del xs
        idx, wm = grid.slab_index(m0, m1)
        use = [p for p, (a, b) in enumerate(pairs) if a in loc and b in loc]
        pa = np.array([loc[pairs[p][0]] for p in use], np.int64)
        pb = np.array([loc[pairs[p][1]] for p in use], np.int64)
        acc[:, use] += kernel(Fk, pa, pb, idx.ravel(), wm, n * n, nbins)
        if acc_chk is not None:
            idx_c, wm_c = check_grid.slab_index(m0, m1)
            it = np.array([loc[tot]], np.int64)
            acc_chk += kernel(Fk[:1], it, it, idx_c.ravel(), wm_c, n * n, acc_chk.size)[0, 0]
        del Fk
    step(f"P(k, mu) of {n_p} pairs and their {nb} jackknife samples in slabs of {ns} k_par")

    with np.errstate(invalid="ignore", divide="ignore"):
        P = grid._norm * acc.reshape(S, n_p, *grid.shape) / np.where(grid.n_modes > 0, grid.n_modes, np.nan)
        P[1:] /= frac[1:, None, None, None]
    pn = dv / n_pix  # skewer_p1d's normalisation
    out = dict(
        keys=keys,
        pairs=pairs,
        pair_orders=[orders[a] + orders[b] if a < n_t else -1 for a, b in pairs],
        p1d=pn * p1d.sum(0) / cnt.sum(),
        p1d_jk=pn * (p1d.sum(0) - p1d) / (cnt.sum() - cnt)[:, None, None],
        P_mu=P[0],
        P_mu_jk=P[1:],
    )
    if acc_chk is not None:
        with np.errstate(invalid="ignore", divide="ignore"):
            out["P_mu_check"] = (
                check_grid._norm
                * acc_chk.reshape(check_grid.shape)
                / np.where(check_grid.n_modes > 0, check_grid.n_modes, np.nan)
            )
    return out


def _term_quantities(raw: dict, line_ids, n_lines: int) -> dict:
    """The quantities of the decomposition from the pairs of ``_streamed_terms``: "total" (the total overflux, measured
    directly), "order n" (n = 0 .. 2L, the sum of the pairs of order n) and every pair of order 2, named "A|B" (the keys
    of ``decomposition.subset_terms``: a line, "𝒞" for the empty subset, "l1+l2" for a product): the autos "l|l", the
    crosses "l1|l2" and "𝒞|l1+l2". Returns dict(names, p1d (n_q, n_k), p1d_jk, P_mu (n_q, n_mu, n_k), P_mu_jk, sum_rule)."""
    pairs, po, keys = raw["pairs"], np.asarray(raw["pair_orders"]), raw["keys"]
    rows = {"total": [len(pairs) - 1]}
    if len(pairs) > 1:
        for o in range(2 * n_lines + 1):
            rows[f"order {o}"] = [int(i) for i in np.flatnonzero(po == o)]
        for i, (a, b) in enumerate(pairs[:-1]):
            if po[i] == 2:
                rows[f"{keys[a]}|{keys[b]}"] = [i]
    names = tuple(rows)
    out = dict(names=names)
    for key in ("p1d", "p1d_jk", "P_mu", "P_mu_jk"):
        x = np.asarray(raw[key])
        ax = 0 if key in ("p1d", "P_mu") else 1
        out[key] = np.stack([np.take(x, r, axis=ax).sum(axis=ax) for r in rows.values()], axis=ax)
    if len(pairs) > 1:
        orders = [names.index(f"order {o}") for o in range(2 * n_lines + 1)]
        out["sum_rule"] = {}
        for key in ("p1d", "P_mu"):
            tot, s = out[key][0], out[key][orders].sum(0)
            out["sum_rule"][key] = float(np.nanmax(np.abs(s - tot)) / np.nanmax(np.abs(tot)))
    return out


def run_statistics(
    tau=None,
    dv: float | None = None,
    *,
    line_ids,
    labels,
    grid=None,
    fields_3d=("total",),
    transmissions=None,
    mean: str = "redshift",
    a_pix=None,
    terms_grid=None,
    decompose: bool = True,
    memory: float = 2**30,
    log=None,
) -> dict:
    """The statistics of one run of a sweep with their jackknife samples (module docstring).

    tau : (n_lines, N_sk, N_pix) the optical depth of the run (T = e^-tau, the total the product), or ``transmissions``
    (n_lines + 1, N_sk, N_pix) the transmission of every line and of the total directly (e.g. ``rebin_pixels`` of a finer
    run); dv : the pixel [km/s]; line_ids : the lines; labels : (N_sk,) the jackknife block of every skewer
    (``block_labels``); grid : ``LatticeWedges`` of the run's lattice (the wedges alone: the memory of 512^2 lattices),
    or its ``lattice_grid`` (None: no 3D statistics); fields_3d : the fields of
    P(k, mu); mean, a_pix : the mean transmission (``ensemble_mean_transmissions``, as ``flux_statistics``); log : None
    or a callable given one line per step with its time.

    Returns dict(fields, n_blocks, mean_flux (n_fields,) and mean_flux_jk (B, n_fields): the mean transmission; k (N_pix // 2
    + 1,) [s/km], p1d (n_fields, n_k) and p1d_jk (B, n_fields, n_k): P1D [km/s] on the native modes (k = 0 included);
    P_mu, P_mu_jk {field: (n_wedges, n_bins), (B, n_wedges, n_bins)} [cMpc^3], k_mu (the bin centres), pk_mu_edges).

    **The decomposition** (``terms_grid``: a ``LatticeWedges`` of the run's lattice, e.g. the 16 wedges of notebook 11,
    ``store_index=False``): P1D and P(k, mu) on its wedges of the total overflux, of every order n = |A| + |B| = 0 .. 2L
    and of every pair of order 2 (the autos, the crosses and the 𝒞 term; ``summary_stats.decomposition``,
    ``docs/decomposition.md``), each with its jackknife samples, in ``out["terms"]``: dict(names ("total", "order 0" ..,
    "A|B"), p1d (n_q, n_k) [km/s] and p1d_jk (B, n_q, n_k) on the modes ``k``, P_mu (n_q, n_mu, n_bins) [cMpc^3] and
    P_mu_jk, k_mu, pk_mu_edges, sum_rule {"p1d", "P_mu": max |sum of the orders - total| / max |total|}, check
    {"p1d": the total here against ``p1d``'s, "P_mu": against ``P_mu["total"]``, binned again on ``grid`` (a
    ``LatticeWedges``), max |difference| / max |total|}). The total is measured directly (T / T_bar - 1), not summed.
    Streamed (``_streamed_terms``): the transforms along the line of sight first (P1D in float64, as here), then the 2D
    transform across the lattice of one slab of k_par at a time, for the field and its B jackknife copies; memory about
    (2^L + 1) N_sk (N_pix / 2 + 1) complex numbers of the package's precision plus ``memory`` [bytes] per slab (16.6 GB
    + 1 GB for two lines on 512^2 skewers of 3949 pixels in float32). It needs the mean shared by the skewers (not
    ``mean="skewer"``) and T_total = prod_j T_j (``tau``; with ``transmissions`` the sum rule fails otherwise, e.g.
    after ``rebin_pixels``: use ``decompose=False``, the total alone)."""
    clock = [time.perf_counter()]

    def step(msg):
        if log is not None:
            t = time.perf_counter()
            log(f"{msg} ({t - clock[0]:.1f} s)")
            clock[0] = t

    fields = (*line_ids, "total")
    if transmissions is None:
        T = np.exp(-np.asarray(tau))
        T = np.concatenate([T, T.prod(axis=0, keepdims=True)])
    else:
        T = np.asarray(transmissions)
    if T.shape[0] != len(fields):
        raise ValueError(f"{len(fields)} fields ({fields}) expected, got {T.shape[0]}.")
    labels = np.asarray(labels)
    nb = int(labels.max()) + 1
    cnt = np.bincount(labels, minlength=nb).astype(float)
    T_bar = ensemble_mean_transmissions(T[None, :-1], mean=mean, grid=a_pix)[0]  # (n_fields, N_sk, N_pix), float64
    step(f"transmissions of {T.shape[1]:,} skewers x {T.shape[2]} pixels")
    mf_sum = np.stack(
        [np.bincount(labels, weights=T[j].mean(axis=1, dtype=np.float64), minlength=nb) for j in range(len(fields))], 1
    )
    out = dict(
        fields=fields,
        n_blocks=nb,
        mean_flux=mf_sum.sum(0) / cnt.sum(),
        mean_flux_jk=(mf_sum.sum(0) - mf_sum) / (cnt.sum() - cnt)[:, None],
    )
    p1d_sum, P_mu, P_mu_jk = [], {}, {}
    for j, f in enumerate(fields):
        delta = T[j] / T_bar[j] - 1.0
        sums = []
        for b in range(nb):
            k, p = skewer_p1d(delta[labels == b], dv)
            sums.append(p * cnt[b])
        p1d_sum.append(np.stack(sums))
        step(f"P1D of {f}")
        if grid is not None and f in fields_3d:
            P_mu[f] = _wedge_power(delta, grid)
            jk = []
            for b in range(nb):
                keep = labels != b
                jk.append(_wedge_power(delta * keep[:, None], grid) / keep.mean())
            P_mu_jk[f] = np.stack(jk)
            step(f"P(k, mu) of {f} and its {nb} jackknife samples")
        del delta
    ps = np.stack(p1d_sum, 1)  # (B, n_fields, n_k)
    out.update(
        k=k, p1d=ps.sum(0) / cnt.sum(), p1d_jk=(ps.sum(0) - ps) / (cnt.sum() - cnt)[:, None, None], P_mu=P_mu, P_mu_jk=P_mu_jk
    )
    if isinstance(grid, LatticeWedges):
        out.update(k_mu=grid.k_mu, pk_mu_edges=grid.pk_mu_edges)
    elif grid is not None:
        ax = grid.axes()
        out.update(k_mu=np.asarray(ax["k_mu"], dtype=float), pk_mu_edges=np.asarray(ax["pk_mu_edges"], dtype=float))
    if terms_grid is not None:
        if mean == "skewer":
            raise ValueError("the decomposition needs a mean shared by the skewers (not mean='skewer').")
        check = grid if isinstance(grid, LatticeWedges) and "total" in P_mu else None
        raw = _streamed_terms(T, T_bar, labels, dv, line_ids, terms_grid, check, decompose=decompose, memory=memory, log=log)
        terms = _term_quantities(raw, line_ids, len(line_ids))
        terms.update(k=out["k"], k_mu=terms_grid.k_mu, pk_mu_edges=terms_grid.pk_mu_edges, pairs=raw["pairs"], keys=raw["keys"])
        ref1 = out["p1d"][-1]
        terms["check"] = {"p1d": float(np.max(np.abs(terms["p1d"][0] - ref1)) / np.max(np.abs(ref1)))}
        if "P_mu_check" in raw:
            ref3 = P_mu["total"]
            terms["check"]["P_mu"] = float(np.nanmax(np.abs(raw["P_mu_check"] - ref3)) / np.nanmax(np.abs(ref3)))
        out["terms"] = terms
    return out


def _ratio(run, ref, run_jk, ref_jk):
    """X_run / X_ref - 1 and its jackknife error (the samples of the ratio, block by block)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.asarray(run, dtype=float) / np.asarray(ref, dtype=float) - 1.0
        rel_jk = np.asarray(run_jk, dtype=float) / np.asarray(ref_jk, dtype=float) - 1.0
    return rel, jackknife_error(rel_jk)


def _bin_modes(k, p, p_jk, edges):
    """P1D on its modes (..., n_k) averaged over the modes of each bin of ``edges`` (empty bins dropped): (k, p, p_jk)."""
    j = bin_index(k, np.asarray(edges, dtype=float))
    nb = len(edges) - 1
    ok = j >= 0
    cnt = np.bincount(j[ok], minlength=nb).astype(float)
    keep = cnt > 0

    def b(x):
        x = np.asarray(x)[..., ok]
        flat = x.reshape(-1, x.shape[-1])
        out = np.stack([np.bincount(j[ok], weights=r, minlength=nb) for r in flat]) / np.maximum(cnt, 1)
        return out.reshape(x.shape[:-1] + (nb,))[..., keep]

    return b(k), b(p), b(p_jk)


def normalised_difference(run, ref, ref_err, *, floor: float = 0.01, half_width: int = 2, sign_changing=None):
    """The difference of a run from the reference, normalised for display, panel by panel (the last axis: the bins of a
    panel, leading axes: the panels, e.g. the mu wedges): (run - ref) / D and the reference's error ref_err / D.

    For a quantity that keeps its sign, D = |ref| (the relative difference). For one that changes sign (``sign_changing``;
    default: the finite reference has both signs anywhere), D = |ref| where |ref| > ``floor`` x max |ref| of the panel,
    and elsewhere the local envelope of the reference, the running maximum of |ref| over the 2 ``half_width`` + 1
    neighbouring bins (near a zero of the reference a relative difference diverges; the envelope keeps the scale of
    the oscillation). Bins without a value (NaN) stay NaN and do not enter the envelope.

    Returns (diff, band, envelope (bool: the bins normalised by the envelope), sign_changing)."""
    from scipy.ndimage import maximum_filter1d

    run, ref, err = (np.asarray(x, dtype=float) for x in (run, ref, ref_err))
    fin = np.isfinite(ref)
    if sign_changing is None:
        sign_changing = bool(np.any(ref[fin] > 0) and np.any(ref[fin] < 0))
    a = np.where(fin, np.abs(ref), 0.0)
    if sign_changing:
        top = a.max(axis=-1, keepdims=True)
        env = maximum_filter1d(a, size=2 * int(half_width) + 1, axis=-1, mode="nearest")
        use_env = fin & (a <= floor * top)
        D = np.where(use_env, env, a)
    else:
        use_env = np.zeros(ref.shape, bool)
        D = a
    with np.errstate(divide="ignore", invalid="ignore"):
        D = np.where(fin & (D > 0), D, np.nan)
        return (run - ref) / D, err / D, use_env, sign_changing


def _compare_terms(run: dict, ref: dict, k_edges_1d=None) -> dict:
    """``compare_runs`` of the decomposition (``run_statistics(terms_grid=)``) of both runs: every quantity they share
    (``ref["terms"]["names"]``), P1D on the shared modes k > 0 (or ``k_edges_1d``) and P(k, mu) on the reference's wedges."""
    a, b = run["terms"], ref["terms"]
    names = [q for q in b["names"] if q in a["names"]]
    m = min(a["k"].size, b["k"].size)
    s = slice(1, m)
    out = dict(names=tuple(names), pk_mu_edges=b["pk_mu_edges"], p1d={}, P_mu={})
    for q in names:
        i, j = a["names"].index(q), b["names"].index(q)
        k, x, xj, y, yj = b["k"][s], a["p1d"][i, s], a["p1d_jk"][:, i, s], b["p1d"][j, s], b["p1d_jk"][:, j, s]
        if k_edges_1d is not None:
            k2, x, xj = _bin_modes(k, x, xj, k_edges_1d)
            _, y, yj = _bin_modes(k, y, yj, k_edges_1d)
            k = k2
        rel, err = _ratio(x, y, xj, yj)
        e_ref = jackknife_error(yj)
        diff, band, env, sc = normalised_difference(x, y, e_ref)
        out["p1d"][q] = dict(
            k=k, ref=y, run=x, ref_err=e_ref, rel=rel, err=err, diff=diff, band=band, envelope=env, sign_changing=sc
        )
        nbin = b["P_mu"].shape[-1]
        X, Xj = _pad(a["P_mu"][i], nbin), _pad(a["P_mu_jk"][:, i], nbin)
        Y, Yj = b["P_mu"][j], b["P_mu_jk"][:, j]
        rel, err = _ratio(X, Y, Xj, Yj)
        e_ref = jackknife_error(Yj)
        diff, band, env, sc = normalised_difference(X, Y, e_ref)
        out["P_mu"][q] = dict(
            k=b["k_mu"], ref=Y, run=X, ref_err=e_ref, rel=rel, err=err, diff=diff, band=band, envelope=env, sign_changing=sc
        )
    return out


def compare_runs(run: dict, ref: dict, *, k_edges_1d=None) -> dict:
    """A run against the reference of its sweep (``run_statistics`` of both, the same jackknife blocks): the relative
    difference X_run / X_ref - 1 and its jackknife error for the mean transmission (n_fields,), P1D on the native modes
    k > 0 the two runs share (the same period: k_m = 2 pi m / v_per, m up to the coarser Nyquist frequency; or averaged
    in the bins ``k_edges_1d`` [s/km] for display, every jackknife sample alike) and P(k, mu) of every field of both on the
    reference's bins (a run of fewer skewers covers fewer |k| bins: NaN beyond).

    Returns dict(fields, mean_flux: dict(rel, err), p1d: dict(k, rel, err, ref, run), P_mu {field: dict(k (the
    reference's bin centres), rel, err, ref, run)}), and with the decomposition of both runs (``run_statistics(
    terms_grid=)``) terms: dict(names, pk_mu_edges, p1d {quantity: dict(k, ref, run, ref_err (the jackknife error of
    the reference), rel, err, diff, band, envelope, sign_changing)}, P_mu {quantity: the same on the wedges}): diff and
    band the difference and the reference's error normalised by ``normalised_difference`` (relative, or by the local
    envelope near the zeros of a quantity that changes sign)."""
    if run["n_blocks"] != ref["n_blocks"]:
        raise ValueError("the run and the reference need the same jackknife blocks.")
    out = dict(fields=ref["fields"])
    rel, err = _ratio(run["mean_flux"], ref["mean_flux"], run["mean_flux_jk"], ref["mean_flux_jk"])
    out["mean_flux"] = dict(rel=rel, err=err, ref=ref["mean_flux"], run=run["mean_flux"])
    m = min(run["k"].size, ref["k"].size)
    if not np.allclose(run["k"][:m], ref["k"][:m], rtol=1e-6):
        raise ValueError("the runs have different periods: their P1D modes differ.")
    s = slice(1, m)  # k > 0
    k, a, aj, b, bj = ref["k"][s], run["p1d"][:, s], run["p1d_jk"][:, :, s], ref["p1d"][:, s], ref["p1d_jk"][:, :, s]
    if k_edges_1d is not None:
        k2, a, aj = _bin_modes(k, a, aj, k_edges_1d)
        _, b, bj = _bin_modes(k, b, bj, k_edges_1d)
        k = k2
    rel, err = _ratio(a, b, aj, bj)
    out["p1d"] = dict(k=k, rel=rel, err=err, ref=b, run=a, ref_err=jackknife_error(bj))
    out["P_mu"] = {}
    out["pk_mu_edges"] = ref.get("pk_mu_edges")
    for f in ref["P_mu"]:
        if f not in run["P_mu"]:
            continue
        nbin = ref["P_mu"][f].shape[-1]
        X, Xj = _pad(run["P_mu"][f], nbin), _pad(run["P_mu_jk"][f], nbin)
        rel, err = _ratio(X, ref["P_mu"][f], Xj, ref["P_mu_jk"][f])
        out["P_mu"][f] = dict(
            k=ref["k_mu"], rel=rel, err=err, ref=ref["P_mu"][f], run=X, ref_err=jackknife_error(ref["P_mu_jk"][f])
        )
    if "terms" in run and "terms" in ref:
        out["terms"] = _compare_terms(run, ref, k_edges_1d)
    return out


def format_terms_table(comparisons: dict, quantities, statistic: str = "p1d", *, k_max=None, names=None) -> str:
    """The decomposition of every run against the reference as text (``compare_runs``' ``terms``): one line per run and
    quantity, over the bins up to ``k_max`` (P1D [s/km], or |k| [1/cMpc] in every wedge of P(k, mu); None: every bin the
    run covers): the median and the largest |diff| [%] (``normalised_difference``), the median error of the reference
    [%] (the band of the figures) and the share of the bins outside that band. statistic : "p1d" or "P_mu"; names :
    {quantity: label}."""
    head = f"{'run':<14} {'quantity':<28} {'median |d| [%]':>14} {'max |d| [%]':>12} {'ref err [%]':>11} {'outside':>8}"
    lines = [head]
    for run, c in comparisons.items():
        for q in quantities:
            t = c["terms"][statistic].get(q)
            if t is None:
                continue
            k = np.asarray(t["k"], dtype=float)
            d, b = 100 * np.asarray(t["diff"], dtype=float), 100 * np.asarray(t["band"], dtype=float)
            m = np.isfinite(d) & np.isfinite(b) & np.isfinite(k) & ((k <= k_max) if k_max is not None else True)
            if not m.any():
                continue
            ad = np.abs(d[m])
            lines.append(
                f"{run:<14} {(names or {}).get(q, q):<28} {np.median(ad):>14.3g} {ad.max():>12.3g} {np.median(b[m]):>11.3g}"
                f" {100 * np.mean(ad > b[m]):>7.0f}%"
            )
    return "\n".join(lines)


def k_max_converged(k, rel, tol: float) -> float:
    """The largest k up to which every bin has |rel| < tol (the bins ordered by k; a bin with no modes in the reference,
    NaN k, is not a bin; a bin the run does not cover, NaN rel, ends the range): the k of the last such bin, 0 if the
    first fails, +inf if every bin passes (the range is then the run's whole coverage)."""
    k, r = np.asarray(k, dtype=float), np.asarray(rel, dtype=float)
    ok = np.isfinite(k)
    k, r = k[ok], r[ok]
    if not k.size:
        return np.nan
    order = np.argsort(k)
    k, r = k[order], r[order]
    bad = ~(np.abs(r) < tol)  # NaN fails too
    if not bad.any():
        return np.inf
    i = int(np.argmax(bad))
    return float(k[i - 1]) if i else 0.0


def k_max_table(comparisons: dict, tols=(0.001, 0.005, 0.01), *, p1d_fields=None, field_3d: str = "total") -> list:
    """The largest k of every statistic converged to each tolerance (``k_max_converged``), per run: rows dict(run,
    statistic ("P1D <field>" [s/km], "P(k, mu) <field> mu <lo>-<hi>" [1/cMpc]), unit, k_max {tol: k}, k_first, k_last (the
    first and last bins the run covers), err (the median jackknife error of the relative difference over the bins up to k_last)).
    comparisons : {run name: ``compare_runs``}; p1d_fields : the P1D fields (default every field)."""
    rows = []
    for name, c in comparisons.items():
        fields = list(c["fields"])
        for f in p1d_fields or fields:
            j = fields.index(f)
            k, rel, err = c["p1d"]["k"], c["p1d"]["rel"][j], c["p1d"]["err"][j]
            cov = np.isfinite(rel)
            rows.append(
                dict(
                    run=name,
                    statistic=f"P1D {f}",
                    unit="s/km",
                    k_max={t: k_max_converged(k, rel, t) for t in tols},
                    k_first=float(k[cov].min()) if cov.any() else np.nan,
                    k_last=float(k[cov].max()) if cov.any() else np.nan,
                    err=float(np.nanmedian(err)) if np.isfinite(err).any() else np.nan,
                )
            )
        P = c["P_mu"].get(field_3d)
        if P is None:
            continue
        edges = c.get("pk_mu_edges")
        for w in range(P["rel"].shape[0]):
            k, rel, err = P["k"][w], P["rel"][w], P["err"][w]
            cov = np.isfinite(rel) & np.isfinite(k)
            mu = f" mu {edges[w]:.2f}-{edges[w + 1]:.2f}" if edges is not None else f" wedge {w}"
            rows.append(
                dict(
                    run=name,
                    statistic=f"P(k, mu) {field_3d}{mu}",
                    unit="1/cMpc",
                    k_max={t: k_max_converged(k, rel, t) for t in tols},
                    k_first=float(k[cov].min()) if cov.any() else np.nan,
                    k_last=float(k[cov].max()) if cov.any() else np.nan,
                    err=float(np.nanmedian(err[cov])) if cov.any() else np.nan,
                )
            )
    return rows


def format_k_max_table(rows, tols=(0.001, 0.005, 0.01)) -> str:
    """The rows of ``k_max_table`` as text: one line per run and statistic, k_max at each tolerance ("all" when every bin
    up to the run's last one passes, then ">= k_last"), the last bin and the median jackknife error [%]."""
    head = f"{'run':<14} {'statistic':<30} " + " ".join(f"{'k_max(' + format(100 * t, 'g') + '%)':>14}" for t in tols)
    head += f" {'last bin':>9} {'jk err [%]':>10}  unit"
    lines = [head]
    for r in rows:
        cells = []
        for t in tols:
            v = r["k_max"][t]
            txt = (
                "-" if not np.isfinite(r["k_last"]) else (">= " + format(r["k_last"], ".3g")) if np.isinf(v) else format(v, ".3g")
            )
            cells.append(f"{txt:>14}")
        lines.append(
            f"{r['run']:<14} {r['statistic']:<30} "
            + " ".join(cells)
            + f" {r['k_last']:>9.3g} {100 * r['err']:>10.2g}  {r['unit']}"
        )
    return "\n".join(lines)
