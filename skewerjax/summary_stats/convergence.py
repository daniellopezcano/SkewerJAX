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
"""

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
    3949 pixels, ~160 GB on 512^2. Here one int32 bin index per rfft mode (2.1 GB on 512^2), built and used slab by slab.

    Attributes: ``k_mu_edges``, ``pk_mu_edges``, ``k_mu`` (n_mu, n_k), ``n_modes`` (n_mu, n_k) (the summed weights)."""

    def __init__(self, n: int, n_los: int, L: float, *, k_edges, mu_edges=None, log=None):
        from skewerjax.summary_stats.binning import full_coverage
        from skewerjax.summary_stats.grid3d import MU_WEDGES, _flat2, _mu_edges, _rfft_weights

        self.n, self.n_los, self.L = int(n), int(n_los), float(L)
        d_perp = self.L / self.n
        k_nyq = float(min(np.pi * self.n_los / self.L, np.pi / d_perp))
        self.k_mu_edges = full_coverage(k_edges, k_nyq, "|k| bins", log)
        self.pk_mu_edges = _mu_edges(MU_WEDGES if mu_edges is None else mu_edges, "pk_mu_edges")
        n_mu, n_k = self.pk_mu_edges.size - 1, self.k_mu_edges.size - 1
        self.shape = (n_mu, n_k)
        kx = 2 * np.pi * np.fft.fftfreq(self.n, d_perp)
        kl = 2 * np.pi * np.fft.fftfreq(self.n_los, self.L / self.n_los)
        kz = 2 * np.pi * np.fft.rfftfreq(self.n, d_perp)
        self._w = _rfft_weights(self.n)  # (n // 2 + 1,): the full-grid modes of each rfft column
        self._norm = d_perp * d_perp * (self.L / self.n_los) / (self.n * self.n * self.n_los)  # dV / N
        self._idx = np.empty((self.n, self.n_los, kz.size), np.int32)
        w_sum, k_sum = np.zeros(n_mu * n_k + 1), np.zeros(n_mu * n_k + 1)
        k_lz = np.sqrt(kl[:, None] ** 2 + kz[None, :] ** 2)  # without k_x
        kpar = np.broadcast_to(np.abs(kl)[:, None], k_lz.shape)
        wts = np.broadcast_to(self._w, k_lz.shape).ravel()
        for i in range(self.n):  # one slab of k_x at a time: no array of the full grid but the index
            kmod = np.sqrt(kx[i] ** 2 + k_lz**2)
            live = kmod > 0
            mu = np.where(live, kpar / np.where(live, kmod, 1.0), 0.0)
            idx = np.where(live, _flat2(bin_index(mu, self.pk_mu_edges), bin_index(kmod, self.k_mu_edges), n_k), -1)
            self._idx[i] = idx
            w_sum += np.bincount(idx.ravel() + 1, weights=wts, minlength=w_sum.size)
            k_sum += np.bincount(idx.ravel() + 1, weights=wts * kmod.ravel(), minlength=k_sum.size)
        self.n_modes = w_sum[1:].reshape(self.shape)
        with np.errstate(invalid="ignore", divide="ignore"):
            self.k_mu = np.where(self.n_modes > 0, k_sum[1:].reshape(self.shape) / self.n_modes, np.nan)

    def power(self, delta) -> np.ndarray:
        """(n_mu, n_k) the binned P [cMpc^3] of one field along the skewers, delta (N_sk, n_los) x-major (as
        ``lattice_power``: one rfftn in the package's precision; the bins summed in float64); NaN in empty bins."""
        import jax.numpy as jnp

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
    P_mu, P_mu_jk {field: (n_wedges, n_bins), (B, n_wedges, n_bins)} [cMpc^3], k_mu (the bin centres), pk_mu_edges)."""
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


def compare_runs(run: dict, ref: dict, *, k_edges_1d=None) -> dict:
    """A run against the reference of its sweep (``run_statistics`` of both, the same jackknife blocks): the relative
    difference X_run / X_ref - 1 and its jackknife error for the mean transmission (n_fields,), P1D on the native modes
    k > 0 the two runs share (the same period: k_m = 2 pi m / v_per, m up to the coarser Nyquist frequency; or averaged
    in the bins ``k_edges_1d`` [s/km] for display, every jackknife sample alike) and P(k, mu) of every field of both on the
    reference's bins (a run of fewer skewers covers fewer |k| bins: NaN beyond).

    Returns dict(fields, mean_flux: dict(rel, err), p1d: dict(k, rel, err, ref, run), P_mu {field: dict(k (the
    reference's bin centres), rel, err, ref, run)})."""
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
    return out


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
