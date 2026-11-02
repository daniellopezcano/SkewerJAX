"""Binning of the two-point statistics: bin indices, jitted weighted bin means, bin edges (``hybrid_k_edges``,
``bao_k_edges``, ``native_edges``, ``occupied_edges``, ``full_coverage``), where a bin is drawn (``bin_centres``), and the
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


@partial(jax.jit, static_argnames=("n_bins",))
def binned_mean(values, idx, weights, n_bins: int):
    """Weighted mean of ``values`` per bin with a fixed number of bins (segment_sum).

    values, weights : (M, ...) arrays of the same shape; idx : (M,) ints, -1 = dropped.
    Returns (mean, total weight) with shapes (n_bins, ...); empty bins give NaN.

    The sums are accumulated per chunk of ``_BIN_CHUNK`` values and the partial sums are then
    added, so the rounding error does not grow with M. A single sequential segment_sum in
    float32 loses ~eps * M relative accuracy: with ~5e7 modes in one bin (512^3 grids) the
    bin centre and the binned power were biased by several per cent.
    """
    M = values.shape[0]
    chunk = min(M, _BIN_CHUNK)
    n_chunks = -(-M // chunk)
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


def bao_k_edges(
    k_f: float,
    k_max: float,
    k_w: float,
    *,
    k_1: float | None = None,
    dk_factor: float = 0.25,
    n_log: int = 12,
    info: bool = False,
):
    """|k| bin edges that resolve BAO wiggles of period ~2 k_f [same units as the inputs], in three ranges:

    * one bin per mode up to k_1 (edges at (j + 1/2) k_f; default k_1 = 3.5 k_f, rounded to such an edge);
    * bins of dk = ``dk_factor`` k_f (default k_f / 4) from k_1 to k_2, the end of the wiggles: k_w (e.g. 3 / sigma_bao)
      rounded to the nearest multiple of dk above k_1, so that every such bin is a full step;
    * ``n_log`` logarithmic bins from k_2 to k_max: the steps end and the log bins start at the same k_2.

    When k_2 >= k_max, steps of dk up to k_max; when k_w <= k_1, no steps (k_2 = k_1); when k_max <= k_1, one mode per
    bin up to k_max. On a 1D axis (modes at multiples of k_f only) a step dk < k_f holds at most one mode, so the occupied
    bins are one per mode up to k_2. ``info``: also return dict(k_1, k_2, k_w, dk, n_mode, n_step, n_log) (the numbers of
    bins in the three ranges)."""
    if not dk_factor > 0:
        raise ValueError(f"dk_factor must be > 0, got {dk_factor}.")
    dk = float(dk_factor) * k_f
    k_1 = 3.5 * k_f if k_1 is None else (np.floor(k_1 / k_f - 0.5) + 0.5) * k_f
    if k_max <= k_1:
        lin = np.arange(0.5, k_max / k_f, 1.0) * k_f
        e = np.append(lin, k_max) if k_max - lin[-1] > 0.1 * k_f else lin
        k_1 = k_2 = float(e[-1])
        parts = (e, np.empty(0), np.empty(0))
    else:
        mode = np.arange(0.5, k_1 / k_f + 0.25, 1.0) * k_f
        k_2 = k_1 + max(0, int(np.round((k_w - k_1) / dk))) * dk
        if k_2 >= k_max:
            q = np.arange(k_1 + dk, k_max - 0.4 * dk, dk)
            parts = (mode, np.append(q, k_max), np.empty(0))
            k_2 = float(k_max)
        else:
            q = np.arange(k_1 + dk, k_2 + 0.4 * dk, dk)
            log_edges = np.geomspace(k_2, k_max, int(n_log) + 1)[1:]
            parts = (mode, q, log_edges)
    e = np.concatenate(parts)
    if not info:
        return e
    n = dict(n_mode=parts[0].size - 1, n_step=parts[1].size, n_log=parts[2].size)
    return e, dict(k_1=float(k_1), k_2=float(k_2), k_w=float(k_w), dk=dk, **n)


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
