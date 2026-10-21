"""Shared JAX kernels of the two-point statistics (FFT spectra, Wiener-Khinchin, binning, exact decomposition).

Conventions (notes ``subsec_P1D.tex`` Eqs. P1D_delta_tilde_sample, P1D_sample_definition,
P1D_ensemble_translation_invariant, xi1D_from_P1D_ensemble; generalised to d FFT axes)
-----------------------------------------------------------------------------------
Grid: N = prod(N_i) cells of size dV = prod(d_i) per realisation, V = N dV; forward FFT
sign e^{-i k.x} (``jnp.fft``), angular wavenumbers k = 2 pi fftfreq(N_i, d_i).

    P_ab(k)  = (dV / N) F_a(k) F_b(k)^*,       F = rfftn(field)   (P_ab(-k) = P_ab(k)^*)
    xi_ab(r) = (1/N) sum_x a(x) b(x + r)        (periodic estimator)
             = irfftn( P_ab(k)^* ) / dV         (exact discrete Fourier pair)

so that P_ab(k) = dV sum_r xi_ab(r) e^{+i k.r}, the notes' sign (P = int xi e^{+i w Delta}).
Cross spectra are complex; P_ab + P_ba = 2 Re P_ab is the notes' ``sym`` combination.

Decomposition (notes ``sec_flux.tex`` Eqs. total_overflux_expansion_general,
overflux_denominator_expansion_general, Table overflux_subset_pairings_L2)
------------------------------------------------------------------------------
With ``core.overflux.DeltaFieldBuilder`` normalising every sample s by its own spectral mean,

    delta_tot,s = sum_A w_s delta_A,s - c_s,   w_s = 1/(1+C_s),  c_s = C_s/(1+C_s),

exactly, where A runs over all non-empty line subsets and C_s = sum_{|A|>=2} <delta_A,s>.
The constant -c_s is the order-0 term (key ``ZERO_ORDER_KEY``). The term (A, B) has order
|A| + |B|, and the sum of all weighted terms reproduces the total P and xi exactly, at
every k (including k = 0) and every lag.
"""
from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from itertools import combinations
from typing import Callable, Hashable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

ZERO_ORDER_KEY = "𝒞"


def subset_orders(overflux, subset_keys: Sequence[str]) -> dict:
    """Order |A| (number of lines) of every subset key of a ``DeltaFieldBuilder``.

    Keys are regenerated from ``overflux.labels`` and ``overflux.key_joiner``; if those are
    missing, ``len(key)`` is used (single-character labels, as in the plotting tools).
    """
    labels = getattr(overflux, "labels", None)
    joiner = getattr(overflux, "key_joiner", "")
    if labels:
        orders = {joiner.join(m): r for r in range(1, len(labels) + 1)
                  for m in combinations(labels, r)}
        if all(k in orders for k in subset_keys):
            return {k: orders[k] for k in subset_keys}
    return {k: len(k) for k in subset_keys}


def lag_axis(n: int, d: float) -> np.ndarray:
    """Lags of an fftshift-ed correlation function: (j - n//2) d, j = 0..n-1."""
    return (np.arange(n) - n // 2) * d


def rfft_multiplicity(n: int) -> np.ndarray:
    """Full-grid modes represented by each rfft column: 2, except k = 0 and Nyquist (1)."""
    m = np.full(n // 2 + 1, 2.0)
    m[0] = 1.0
    if n % 2 == 0:
        m[-1] = 1.0
    return m


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


_BIN_CHUNK = 4096   # values per partial sum in binned_mean (bounds float32 accumulation error)


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

    def chunks(a):                       # (M, ...) -> (n_chunks, chunk, ...), zero padded
        a = jnp.pad(a, [(0, pad)] + [(0, 0)] * (a.ndim - 1))
        return a.reshape((n_chunks, chunk) + a.shape[1:])

    partial_sum = jax.vmap(lambda a, s: jax.ops.segment_sum(a, s, num_segments=n_bins))
    seg_c = chunks(seg)                  # padded entries have weight 0
    W = jnp.sum(partial_sum(chunks(w), seg_c), axis=0)
    S = jnp.sum(partial_sum(chunks(w * values), seg_c), axis=0)
    return jnp.where(W > 0, S / jnp.where(W > 0, W, 1.0), jnp.nan), W


class LazyPairs(Mapping):
    """Read-only mapping {(a, b): array} evaluated on access (optionally cached)."""

    def __init__(self, keys: Sequence[Hashable], fn: Callable, cache: bool = False):
        self._keys = list(keys)
        self._set = set(self._keys)
        self._fn = fn
        self._cache = {} if cache else None

    def __getitem__(self, key):
        if key not in self._set:
            raise KeyError(key)
        if self._cache is None:
            return self._fn(key)
        if key not in self._cache:
            self._cache[key] = self._fn(key)
        return self._cache[key]

    def __iter__(self):
        return iter(self._keys)

    def __len__(self):
        return len(self._keys)

    def __repr__(self):
        return f"LazyPairs({len(self._keys)} pairs)"


def order_sum(pair_matrix, term_orders: Sequence[int]):
    """Sum a (n, n, ...) pair matrix into total orders |A|+|B| -> (2 max_order + 1, ...)."""
    o = np.asarray(term_orders)
    pair_ord = (o[:, None] + o[None, :]).ravel()
    n_ord = 2 * int(o.max()) + 1
    flat = pair_matrix.reshape((-1,) + pair_matrix.shape[2:])
    return jax.ops.segment_sum(flat, jnp.asarray(pair_ord), num_segments=n_ord)


# ---------------------------------------------------------------------------
# 1D kernel: many skewers, FFT along the last axis, chunked over skewers
# ---------------------------------------------------------------------------
@partial(jax.jit, static_argnames=("chunk",))
def p1d_kernel(total, terms, w, c, pref, chunk: int):
    """Sample-averaged spectra of all subset terms along the LoS.

    Parameters
    ----------
    total : (B, N) real; terms : (n, B, N) real (raw subset fields delta_A);
    w, c : (B,) per-sample weights 1/(1+C_s) and offsets C_s/(1+C_s); pref = dv / N.

    Returns
    -------
    P_raw : (n, n, K) complex, mean_s pref F_a F_b^* of the raw terms
    P_w   : (n+1, n+1, K) complex, same for the weighted terms, index 0 = order-0 term -c_s
    P_tot : (B, K) real, per-sample pref |F_tot|^2
    """
    B, N = total.shape
    n = terms.shape[0]
    K = N // 2 + 1
    n_chunks = -(-B // chunk)
    pad = n_chunks * chunk - B
    total = jnp.pad(total, ((0, pad), (0, 0))).reshape(n_chunks, chunk, N)
    terms = jnp.pad(terms, ((0, 0), (0, pad), (0, 0))).reshape(n, n_chunks, chunk, N)
    terms = jnp.moveaxis(terms, 1, 0)
    w = jnp.pad(w, (0, pad)).reshape(n_chunks, chunk)
    c = jnp.pad(c, (0, pad)).reshape(n_chunks, chunk)
    cdt = jnp.result_type(total.dtype, jnp.complex64)

    def body(carry, xs):
        s_raw, s_w = carry
        t, tm, ww, cc = xs
        f_tot = jnp.fft.rfft(t, axis=-1)
        f_raw = jnp.fft.rfft(tm, axis=-1)                                    # (n, chunk, K)
        f_c = jnp.zeros((1, chunk, K), cdt).at[0, :, 0].set(-cc * N)         # rfft of -c_s
        f_w = jnp.concatenate([f_c, ww[None, :, None] * f_raw], axis=0)      # (n+1, chunk, K)
        s_raw = s_raw + jnp.einsum("asf,bsf->abf", f_raw, f_raw.conj())
        s_w = s_w + jnp.einsum("asf,bsf->abf", f_w, f_w.conj())
        return (s_raw, s_w), pref * jnp.abs(f_tot) ** 2

    init = (jnp.zeros((n, n, K), cdt), jnp.zeros((n + 1, n + 1, K), cdt))
    (s_raw, s_w), p_tot = jax.lax.scan(body, init, (total, terms, w, c))
    return pref * s_raw / B, pref * s_w / B, p_tot.reshape(-1, K)[:B]


@partial(jax.jit, static_argnames=("n",))
def xi_from_power(P, n: int, d: float):
    """xi = irfft(P^*, n) / d along the last axis (raw FFT lag order)."""
    return jnp.fft.irfft(jnp.conj(P), n=n, axis=-1) / d


# ---------------------------------------------------------------------------
# 3D kernel: one (Nx, Ny, Nz) field per term, rfftn over all three axes
# ---------------------------------------------------------------------------
@jax.jit
def p3d_kernel(total, terms, w, c):
    """rfftn of the total, raw terms, weighted terms and the order-0 term.

    total : (Nx, Ny, Nz); terms : (n, Nx, Ny, Nz); w, c : (Nx, Ny) per-skewer weights.
    Returns F_tot (Nx, Ny, K), F_raw (n, ...), F_w (n+1, ...) with index 0 = order-0 term.
    """
    rf = lambda x: jnp.fft.rfftn(x, axes=(-3, -2, -1))
    f_tot = rf(total)
    f_raw = rf(terms)
    f_c = rf(jnp.broadcast_to(-c[:, :, None], total.shape))
    f_w = jnp.concatenate([f_c[None], rf(w[None, :, :, None] * terms)], axis=0)
    return f_tot, f_raw, f_w


@partial(jax.jit, static_argnames=("n_orders",))
def p3d_order_spectra(f_w, order_ids, pref, n_orders: int):
    """Weighted spectra summed per total order: P_n = pref sum_{m+m'=n} G_m G_m'^*,
    with G_m = sum_{|A|=m} F_w,A (G_0 = order-0 term). Returns (2 n_orders - 1, ...) complex."""
    G = jax.ops.segment_sum(f_w, order_ids, num_segments=n_orders)
    out = []
    for n_tot in range(2 * n_orders - 1):
        acc = jnp.zeros_like(G[0])
        for m in range(max(0, n_tot - n_orders + 1), min(n_tot, n_orders - 1) + 1):
            acc = acc + G[m] * jnp.conj(G[n_tot - m])
        out.append(pref * acc)
    return jnp.stack(out)


@partial(jax.jit, static_argnames=("shape",))
def xi3d_from_power(P, shape: tuple, dV: float):
    """xi(r) = fftshift(irfftn(P^*)) / dV on the full (Nx, Ny, Nz) lag grid, centred."""
    return jnp.fft.fftshift(jnp.fft.irfftn(jnp.conj(P), s=shape, axes=(-3, -2, -1)),
                            axes=(-3, -2, -1)) / dV
