"""Binning of the two-point statistics: bin indices, jitted weighted bin means, bin edges (``hybrid_k_edges``,
``set_k_edges``, ``native_edges``, ``occupied_edges``, ``full_coverage``), where a bin is drawn (``bin_centres``), and the
mean over independent realisations."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np


def bin_index(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Bin of each x for increasing ``edges``: [lo, hi), last edge inclusive; -1 outside.

    Values within 1e-12 * max|edges| of an edge are assigned as if exactly on it, so grid
    points that sit on an edge (e.g. r = j d) do not flip bins with 1-ulp rounding.
    """
    edges = np.asarray(edges, dtype=float)
    x = np.asarray(x, dtype=float)
    tol = 1e-12 * np.max(np.abs(edges))
    idx = np.digitize(x + tol, edges) - 1
    idx = np.where(np.abs(x - edges[-1]) <= tol, len(edges) - 2, idx)
    return np.where((idx >= 0) & (idx < len(edges) - 1), idx, -1)


_BIN_CHUNK = 4096  # values per partial sum in binned_mean (bounds float32 accumulation error)
_BIN_BLOCK = 2**24  # the most partial-sum entries (chunks x bins x trailing) binned_mean holds


@partial(jax.jit, static_argnames=("n_bins",))
def binned_mean(values, idx, weights, n_bins: int):
    """Weighted mean of ``values`` per bin with a fixed number of bins (segment_sum).

    values, weights : (M, ...) arrays of the same shape; idx : (M,) ints, -1 = dropped.
    Returns (mean, total weight) with shapes (n_bins, ...); empty bins give NaN.

    The sums are accumulated per chunk of values and the partial sums are then added, so the
    rounding error does not grow with M. A single sequential segment_sum in float32 loses
    ~eps * M relative accuracy: with ~5e7 modes in one bin (512^3 grids) the bin centre and the
    binned power were biased by several per cent. The chunks hold ``_BIN_CHUNK`` values, or more
    when chunks x bins would exceed ``_BIN_BLOCK`` (memory and time are O(M + chunks x bins)):
    then the values of one bin within one chunk are about M / _BIN_BLOCK on average (16 for
    the 2.6e8 separations of the lattice of 256^2 TNG50 skewers and its 2.5e5 bins of the
    (r_par, r_perp) map), and the few bins of |k|, |r| and the wedges keep chunks of _BIN_CHUNK.
    """
    M = values.shape[0]
    per = max(1, n_bins * int(np.prod(values.shape[1:], dtype=np.int64)))
    n_chunks = max(1, min(-(-M // _BIN_CHUNK), _BIN_BLOCK // per))
    chunk = -(-M // n_chunks)
    pad = n_chunks * chunk - M
    ok = idx >= 0
    seg = jnp.where(ok, idx, 0)
    w = jnp.where(ok.reshape((-1,) + (1,) * (values.ndim - 1)), weights, 0.0)

    def chunks(a):  # (M, ...) -> (n_chunks, chunk, ...), zero padded
        a = jnp.pad(a, [(0, pad)] + [(0, 0)] * (a.ndim - 1))
        return a.reshape((n_chunks, chunk) + a.shape[1:])

    partial_sum = jax.vmap(lambda a, s: jax.ops.segment_sum(a, s, num_segments=n_bins))
    seg_c = chunks(seg)  # padded entries have weight 0
    W = jnp.sum(partial_sum(chunks(w), seg_c), axis=0)
    S = jnp.sum(partial_sum(chunks(w * values), seg_c), axis=0)
    return jnp.where(W > 0, S / jnp.where(W > 0, W, 1.0), jnp.nan), W


X_BIN_MODES = ("weighted", "linear", "log")  # where a figure draws a bin: ``bin_centres``


def bin_centres(edges, mode: str = "linear", weighted=None) -> np.ndarray:
    """The x at which a bin of ``edges`` is drawn: "linear" the arithmetic midpoint (e_i + e_{i+1}) / 2, "log" the
    geometric mean sqrt(e_i e_{i+1}) (the midpoint where the lower edge is <= 0, which has no log centre), "weighted" the
    centres ``weighted`` given by the measurement (e.g. the mode- or separation-weighted mean |k| or |r| of
    ``GridStatistics3D``, the mean of the modes or lags of a 1D bin; returned as they are, any shape). Only the drawn
    position: the values of a bin do not depend on it."""
    if mode not in X_BIN_MODES:
        raise ValueError(f"x_bin must be one of {X_BIN_MODES}, got {mode!r}.")
    if mode == "weighted":
        if weighted is None:
            raise ValueError('x_bin="weighted" needs the weighted centres of the measurement.')
        return np.asarray(weighted, dtype=float)
    e = np.asarray(edges, dtype=float)
    lo, hi = e[:-1], e[1:]
    mid = 0.5 * (lo + hi)
    if mode == "linear":
        return mid
    return np.where(lo > 0, np.sqrt(np.clip(lo, 1e-300, None) * hi), mid)


def ensemble_mean(records, keys) -> dict:
    """Mean over independent realisations of the arrays ``keys`` of a list of records (dicts of arrays).

    Returns, per key: ``<key>`` (the mean), ``<key>_err`` (the error of the mean, std / sqrt(n) with ddof = 1; NaN for one
    realisation) and ``<key>_each`` (n, ...) the values of every realisation; and n_real.
    """
    records = list(records)
    n = len(records)
    out = {"n_real": n}
    for key in keys:
        each = np.stack([np.asarray(r[key]) for r in records])
        out[key + "_each"] = each
        out[key] = each.mean(0)
        out[key + "_err"] = each.std(0, ddof=1) / np.sqrt(n) if n > 1 else np.full(each.shape[1:], np.nan)
    return out


def hybrid_k_edges(k_fund, k_max, n_lin: int = 12, n_log: int = 18):
    """k bin edges: one mode per bin up to (n_lin + 1/2) k_fund (edges at (j + 1/2) k_fund), then n_log log bins."""
    lin = (np.arange(n_lin + 1) + 0.5) * k_fund
    lin = lin[lin < k_max]
    return np.concatenate([lin[:-1], np.geomspace(lin[-1], k_max, n_log + 1)])


def occupied_edges(values, edges):
    """The bin edges with every empty bin merged into the next one, so that each bin holds at least one of ``values``
    (a log bin narrower than the spacing of the discrete modes along an axis would be empty: a blank row or column)."""
    v = np.unique(np.asarray(values, dtype=float))
    out = [edges[0]]
    for hi in edges[1:]:
        if np.any((v >= out[-1]) & (v < hi)):
            out.append(hi)
    out[-1] = edges[-1]  # trailing empty bins join the last occupied one
    return np.asarray(out)


def set_k_edges(
    k_f: float,
    *,
    k_1: float,
    k_2: float,
    k_max: float,
    dk_factor: float = 0.25,
    n_log: int = 12,
    info: bool = False,
):
    """|k| bin edges in three ranges set by hand [same units as the inputs]:

    * one bin per mode up to k_1 (edges at (j + 1/2) k_f; k_1 rounded down to such an edge, e.g. 3.5 k_f);
    * bins of dk = ``dk_factor`` k_f (default k_f / 4) from k_1 to k_2, the last one ending exactly at k_2 (it absorbs
      the remainder: between 0.6 and 1.4 dk);
    * ``n_log`` logarithmic bins from k_2 to k_max.

    When k_2 >= k_max, steps of dk up to k_max (no log bins); when k_2 <= k_1, no steps; when k_max <= k_1, one mode per
    bin up to k_max. On a 1D axis (modes at multiples of k_f only) a step dk < k_f holds at most one mode. ``info``: also
    return dict(k_1, k_2, dk, n_mode, n_step, n_log) (the edges actually used and the numbers of bins in the three
    ranges)."""
    if not dk_factor > 0:
        raise ValueError(f"dk_factor must be > 0, got {dk_factor}.")
    dk = float(dk_factor) * k_f
    k_1 = (np.floor(k_1 / k_f - 0.5 + 1e-9) + 0.5) * k_f  # 3.5 k_f stays 3.5 k_f despite rounding
    if k_max <= k_1:
        lin = np.arange(0.5, k_max / k_f, 1.0) * k_f
        e = np.append(lin, k_max) if k_max - lin[-1] > 0.1 * k_f else lin
        k_1 = k_2 = float(e[-1])
        parts = (e, np.empty(0), np.empty(0))
    else:
        mode = np.arange(0.5, k_1 / k_f + 0.25, 1.0) * k_f
        top = min(float(k_2), float(k_max))
        if top - k_1 < 0.6 * dk:  # no room for a step: the log bins start at k_1
            q, k_2 = np.empty(0), float(k_1)
        else:
            q = np.append(np.arange(k_1 + dk, top - 0.4 * dk, dk), top)
            k_2 = top
        log_edges = np.geomspace(k_2, k_max, int(n_log) + 1)[1:] if k_2 < k_max else np.empty(0)
        parts = (mode, q, log_edges)
    e = np.concatenate(parts)
    if not info:
        return e
    n = dict(n_mode=parts[0].size - 1, n_step=parts[1].size, n_log=parts[2].size)
    return e, dict(k_1=float(k_1), k_2=float(k_2), dk=dk, **n)


def _axis_offsets(n: int):
    """The minimum-image offsets |o| = 0 .. n // 2 of a periodic axis of n nodes and the number of signed offsets of each
    (1 for 0 and, n even, for n / 2; 2 otherwise): the separation vectors of ``GridStatistics3D``."""
    o = np.arange(n // 2 + 1)
    m = np.full(o.size, 2, dtype=np.int64)
    m[0] = 1
    if n % 2 == 0:
        m[-1] = 1
    return o, m


def _nodes(L: float, spacing: float, name: str) -> int:
    """The number of nodes n = L / spacing of a periodic axis, checked to be a whole number."""
    n = int(round(float(L) / float(spacing)))
    if n < 1 or abs(n * float(spacing) - float(L)) > 1e-6 * float(L):
        raise ValueError(f"{name} = {spacing} must divide L = {L} (L / {name} = {float(L) / float(spacing):.6g}).")
    return n


def _separations(L: float, d_perp: float, d_los: float | None):
    """The distinct separations of a periodic cubic lattice of side L, transverse spacing d_perp and line-of-sight spacing
    d_los (None: the transverse plane only, r_perp), and the number of separation vectors (minimum image) at each:
    (r (M,) increasing [units of L], count (M,) ints). Exact: r^2 / L^2 = (i^2 + j^2) / n_perp^2 + m^2 / n_los^2 is
    compared as the integer (i^2 + j^2) n_los^2 + m^2 n_perp^2, so equal separations are one value (e.g. 2 d_los = d_perp)."""
    n_p = _nodes(L, d_perp, "d_perp")
    o, m = _axis_offsets(n_p)
    key = (o[:, None] ** 2 + o[None, :] ** 2).ravel().astype(np.int64)
    cnt = (m[:, None] * m[None, :]).ravel()
    if d_los is None:
        scale = float(L) / n_p
    else:
        n_l = _nodes(L, d_los, "d_los")
        ol, ml = _axis_offsets(n_l)
        key = (key[:, None] * n_l**2 + (ol.astype(np.int64) ** 2 * n_p**2)[None, :]).ravel()
        cnt = (cnt[:, None] * ml[None, :]).ravel()
        scale = float(L) / (n_p * n_l)
    uk, inv = np.unique(key, return_inverse=True)
    return scale * np.sqrt(uk.astype(np.float64)), np.bincount(inv.ravel(), weights=cnt).astype(np.int64)


def _lattice_r_edges(r, cnt, spacing: float, *, r_1, r_2, r_max, dr_factor, n_log, merge_empty, info, perp: bool):
    """The three ranges of ``set_r_edges`` and ``set_rperp_edges`` on the distinct separations r (with their counts)."""
    if not dr_factor > 0:
        raise ValueError(f"dr_factor must be > 0, got {dr_factor}.")
    r_max = float(r_max)
    dr = float(dr_factor) * float(spacing)
    v = r if perp else r[1:]  # |r|: the zero lag (the variance) belongs to no bin
    mid = 0.5 * (v[:-1] + v[1:])  # one bin per distinct separation: edges halfway between consecutive ones
    lo = 0.0 if perp else 0.5 * float(v[0])
    n_sep = max(1, int(np.searchsorted(mid, float(r_1) * (1 + 1e-9), side="right")))  # r_1 rounded down to an edge
    sep = np.concatenate([[lo], mid[:n_sep]])
    sep = sep[sep < r_max * (1 - 1e-12)]
    if sep[-1] < float(r_1) and n_sep == mid.size or float(r_1) >= r_max:  # one bin per separation up to r_max
        sep = np.append(sep, r_max)
    r_1 = float(sep[-1])
    top = min(float(r_2), r_max)
    if top - r_1 < 0.6 * dr:  # no room for a step
        q, r_2 = np.empty(0), r_1
    else:
        q = np.append(np.arange(r_1 + dr, top - 0.4 * dr, dr), top)
        r_2 = top
    if r_2 < r_max * (1 - 1e-9):
        if int(n_log) < 1:
            raise ValueError(
                f"r_2 = {r_2:.6g} < r_max = {r_max:.6g} needs n_log >= 1 log bins to r_max (or r_2 >= r_max: steps to r_max)."
            )
        lg = np.geomspace(r_2, r_max, int(n_log) + 1)[1:]
    else:
        lg = np.empty(0)
    e = np.concatenate([sep, q, lg])
    n_parts = dict(n_sep=sep.size - 1, n_step=q.size, n_log=lg.size)

    def occupancy(edges):
        idx = bin_index(r, edges)
        ok = idx >= 0
        nb = edges.size - 1
        n = np.bincount(idx[ok], weights=cnt[ok], minlength=nb).astype(np.int64)
        nd = np.bincount(idx[ok], minlength=nb).astype(np.int64)
        rs = np.bincount(idx[ok], weights=(r * cnt)[ok], minlength=nb)
        return n, nd, np.where(n > 0, rs / np.where(n > 0, n, 1), np.nan)

    n_lags, n_distinct, r_mean = occupancy(e)
    n_merged = 0
    if merge_empty and np.any(n_lags == 0):
        keep = [0]
        for b in range(e.size - 1):  # an empty bin joins the next one; trailing empty bins join the last occupied one
            if n_lags[b] > 0:
                keep.append(b + 1)
        keep[-1] = e.size - 1
        n_merged = int(e.size - len(keep))
        e = e[np.unique(keep)]
        n_lags, n_distinct, r_mean = occupancy(e)
    if not info:
        return e
    return e, dict(
        r_1=r_1,
        r_2=float(r_2),
        r_max=r_max,
        dr=dr,
        spacing=float(spacing),
        **n_parts,
        n_merged=n_merged,
        n_empty=int(np.sum(n_lags == 0)),
        n_lags=n_lags,
        n_distinct=n_distinct,
        r_mean=r_mean,
    )


def set_r_edges(
    d_perp: float,
    d_los: float,
    *,
    L: float,
    r_1: float,
    r_2: float,
    r_max: float | None = None,
    dr_factor: float = 1.0,
    n_log: int = 0,
    merge_empty: bool = True,
    info: bool = False,
):
    """|r| bin edges on a periodic cubic lattice of side ``L`` [cMpc], transverse spacing ``d_perp`` and line-of-sight
    spacing ``d_los`` (a lattice of skewers: L / N_SK and a pixel dv / u, or the points dx / p; a grid: dx and dx), the
    counterpart of ``set_k_edges`` in three ranges set by hand [cMpc]:

    * one bin per distinct separation up to r_1: the lattice has few distinct |r| at small r (sqrt(i^2 + j^2) d_perp
      combined with m d_los), so every bin holds one, with edges halfway between consecutive ones (r_1 rounded down to
      such an edge; at least the first separation). The zero lag r = 0 (the variance) belongs to no bin: the first edge
      is half the smallest separation, min(d_perp, d_los);
    * bins of dr = ``dr_factor`` x the spacing, max(d_perp, d_los) (the spacing in every direction), from r_1 to r_2, the
      last one ending exactly at r_2 (it absorbs the remainder, between 0.6 and 1.4 dr); dr_factor < 1 is allowed where
      many distinct separations exist;
    * ``n_log`` logarithmic bins from r_2 to r_max (default L / 2, the full-coverage limit; r_2 < r_max needs n_log >= 1).

    When r_2 >= r_max, steps up to r_max (no log bins); when r_2 <= r_1, no steps. ``merge_empty`` (default): a bin that
    holds no separation (a step finer than the spacing between distinct separations) joins the next one, so no bin is
    empty; ``merge_empty=False`` keeps them (``info["n_empty"]`` counts them). The counts follow ``bin_index``'s rule (an
    edge belongs to the upper bin) and the minimum-image separation vectors of ``GridStatistics3D``, so ``n_lags`` equals
    the grid's ``n_lags["xi_iso"]``. ``info``: also return dict(r_1, r_2, r_max, dr, spacing (the edges actually used),
    n_sep, n_step, n_log (bins per range), n_merged, n_empty, n_lags (separation vectors per bin, per lattice point: the
    number of pairs is N n_lags / 2), n_distinct (distinct separations per bin), r_mean (the separation-weighted mean |r|
    of each bin, the grid's r_iso))."""
    r, cnt = _separations(L, d_perp, d_los)
    return _lattice_r_edges(
        r,
        cnt,
        max(float(d_perp), float(d_los)),
        r_1=r_1,
        r_2=r_2,
        r_max=0.5 * float(L) if r_max is None else r_max,
        dr_factor=dr_factor,
        n_log=n_log,
        merge_empty=merge_empty,
        info=info,
        perp=False,
    )


def set_rperp_edges(
    d_perp: float,
    *,
    L: float,
    r_1: float,
    r_2: float,
    r_max: float | None = None,
    dr_factor: float = 1.0,
    n_log: int = 0,
    merge_empty: bool = True,
    info: bool = False,
):
    """r_perp bin edges (the columns of the (r_perp, r_par) map) of a periodic lattice of side ``L`` [cMpc] and transverse
    spacing ``d_perp``: ``set_r_edges`` across the line of sight alone, r_perp = sqrt(i^2 + j^2) d_perp. The first bin,
    [0, d_perp / 2), holds r_perp = 0 only (the pairs on the same skewer, or the same column of a grid); then one bin per
    distinct r_perp up to r_1, steps of dr = ``dr_factor`` d_perp to r_2 and ``n_log`` log bins to r_max (default L / 2).
    ``info["n_lags"]``: the transverse separation vectors per bin (the skewers at that r_perp from a skewer; times the
    n_los line-of-sight offsets, the grid's counts of the map summed over its rows); the other entries as ``set_r_edges``."""
    r, cnt = _separations(L, d_perp, None)
    return _lattice_r_edges(
        r,
        cnt,
        float(d_perp),
        r_1=r_1,
        r_2=r_2,
        r_max=0.5 * float(L) if r_max is None else r_max,
        dr_factor=dr_factor,
        n_log=n_log,
        merge_empty=merge_empty,
        info=info,
        perp=True,
    )


def native_edges(spacing: float, n_max: int, start: int = 0) -> np.ndarray:
    """The edges of one bin per grid value j spacing, j = ``start`` .. ``n_max``, each centred on its value:
    (j -+ 1/2) spacing (the rows of the maps on a grid's own modes or separations)."""
    return (np.arange(int(start), int(n_max) + 2) - 0.5) * float(spacing)


def full_coverage(edges, hi: float, name: str = "bins", log=print) -> np.ndarray:
    """``edges`` without the bins that extend beyond ``hi``, the end of the range where every direction is available
    (|k| <= the smallest Nyquist frequency, |r| <= half the box); ``log`` reports a cut in one line."""
    e = np.asarray(edges, dtype=float)
    keep = e <= hi * (1 + 1e-9)
    if keep.all():
        return e
    if log:
        log(f"{name}: {int((~keep).sum())} edge(s) beyond the full-coverage limit {hi:.4g} cut (last edge {e[-1]:.4g})")
    return e[keep]
