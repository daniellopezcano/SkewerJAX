"""Binning of the two-point statistics: bin indices, jitted weighted bin means, bin edges, and the mean over
independent realisations."""

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
