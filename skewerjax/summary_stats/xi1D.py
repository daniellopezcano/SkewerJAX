from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import logging

import jax
import jax.numpy as jnp

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PairBinning1D:
    """
    Container for geometry-dependent precomputations used by the 1D pair-binned 2PCF.

    Attributes
    ----------
    pair_i, pair_j : jnp.ndarray, shape (n_chunks, chunk_size), dtype int32
        Indices of point pairs (i, j) defining all unique pairs (upper triangle).
        Stored chunked to allow JIT compilation without dynamic slicing.

    bin_idx : jnp.ndarray, shape (n_chunks, chunk_size), dtype int32
        Separation-bin index for each pair, always in [0, n_bins-1].
        For invalid/padded pairs, bin_idx is forced to 0 but masked out by `mask`.
        This makes jnp.bincount(..., length=n_bins) safe under JAX.

    mask : jnp.ndarray, shape (n_chunks, chunk_size), dtype bool
        True for valid pairs that fall within the binning range [r_min, r_max),
        False for out-of-range pairs and padding entries.

    denom_w : jnp.ndarray, shape (n_chunks, chunk_size), float
        Pair weights w_ij = w_i * w_j (or 1 if no weights provided). Used in both numerator
        and denominator.

    den_per_bin : jnp.ndarray, shape (n_bins,), float
        Denominator per bin: sum_{pairs in bin k} w_ij. Independent of `y` and reused.

    counts_per_bin : jnp.ndarray, shape (n_bins,), int32
        Number of valid pairs per bin (useful for QA/debugging, and for understanding where
        estimates become noisy due to low pair counts).

    r_centers : jnp.ndarray, shape (n_bins,), float
        Centers of separation bins.

    n_bins : int
        Number of separation bins.

    chunk_size : int
        Number of pairs processed per chunk.

    n_x : int
        Number of x positions used to build this precomputation (for validation against y).
    """
    pair_i: jnp.ndarray
    pair_j: jnp.ndarray
    bin_idx: jnp.ndarray
    mask: jnp.ndarray
    denom_w: jnp.ndarray
    den_per_bin: jnp.ndarray
    counts_per_bin: jnp.ndarray
    r_centers: jnp.ndarray
    n_bins: int
    chunk_size: int
    n_x: int


def _make_linear_bins(
    x: jnp.ndarray,
    n_bins: int,
    r_min: float = 0.0,
    r_max: Optional[float] = None,
    *,
    periodic_L: Optional[float] = None,
) -> jnp.ndarray:
    """
    Build linear separation-bin edges for r = |x_i - x_j|.

    Parameters
    ----------
    x : jnp.ndarray, shape (N_x,), float
        1D positions where the field is sampled.
    n_bins : int
        Number of separation bins.
    r_min : float
        Minimum separation considered (left edge of first bin).
    r_max : float or None
        Maximum separation considered (right edge of last bin). If None:
          - periodic_L is not None -> r_max = L/2 (maximum distinct separation on a circle)
          - else                  -> r_max = max(x) - min(x)
    periodic_L : float or None
        If not None, indicates periodic domain length L (affects default r_max only here).

    Returns
    -------
    bins : jnp.ndarray, shape (n_bins+1,), float
        Bin edges such that bin k is [bins[k], bins[k+1]).
    """
    x = jnp.asarray(x)
    x_dtype = x.dtype

    r_min_v = jnp.asarray(r_min, dtype=x_dtype)

    if r_max is None:
        if periodic_L is not None:
            r_max_v = 0.5 * jnp.asarray(periodic_L, dtype=x_dtype)
        else:
            r_max_v = jnp.max(x) - jnp.min(x)
    else:
        r_max_v = jnp.asarray(r_max, dtype=x_dtype)

    return jnp.linspace(r_min_v, r_max_v, n_bins + 1)


def _min_image_separation_1d(dx: jnp.ndarray, L: jnp.ndarray) -> jnp.ndarray:
    """
    Robust minimum-image separation on a 1D periodic domain of length L.

    This version remains correct even if x is not pre-wrapped to [0, L) and/or spans
    more than one period.
    """
    # Map to (-L/2, L/2] then take abs
    return jnp.abs(jnp.mod(dx + 0.5 * L, L) - 0.5 * L)


def precompute_pair_binning_1d(
    x: jnp.ndarray,
    *,
    bins: Optional[jnp.ndarray] = None,
    n_bins: int = 64,
    r_min: float = 0.0,
    r_max: Optional[float] = None,
    periodic_L: Optional[float] = None,
    include_self: bool = False,
    weights: Optional[jnp.ndarray] = None,
    chunk_size: int = 1 << 16,
    weight_dtype: Optional[jnp.dtype] = None,
) -> PairBinning1D:
    """
    Precompute geometry-dependent pair/bin information for a 1D pair-binned 2PCF.

    Objective (conceptual)
    ----------------------
    For a (possibly irregular) set of sample positions x_i, we will later estimate:
        xi(r_k) = average of yy(x_i) * yy(x_j) over all pairs with |x_i-x_j| in bin k
    where yy is a per-sample preprocessed field (mean-subtracted and/or standardized).

    This function prepares everything that depends only on x (and binning choices), so that
    computing xi for many realizations y is fast and JIT-safe.

    Key points
    ----------
    - `bin_idx` is made *always safe* for `jnp.bincount(..., length=n_bins)`:
        invalid/out-of-range pairs are mapped to bin 0, and excluded via `mask`.
      This avoids potential out-of-range index errors even when invalid pairs have zero weight.
    - If `periodic_L` is provided, separations use a robust minimum-image convention that
      remains correct even if x is not pre-wrapped into [0, L).
    - Arrays are padded to a multiple of `chunk_size` and reshaped into
      (n_chunks, chunk_size) to avoid dynamic slicing inside JIT-compiled loops.
    - The denominator per bin is precomputed once and reused.

    Parameters
    ----------
    x : jnp.ndarray, shape (N_x,), float
        Sample positions (not necessarily uniform).
    bins : jnp.ndarray, shape (n_bins+1,), float, optional
        Explicit bin edges. If provided, overrides (n_bins, r_min, r_max) for binning.
    n_bins : int
        Number of bins (used only if `bins` is None).
    r_min : float
        Minimum separation (used only if `bins` is None).
    r_max : float or None
        Maximum separation (used only if `bins` is None).
    periodic_L : float or None
        If provided, separations use minimum-image convention on a domain of length L.
        If r_max is None, default r_max = L/2.
    include_self : bool
        If True, include i==j pairs (separation 0). This adds yy_i^2 contributions to the
        smallest-separation bin (often not desired; default False uses i<j only).
    weights : jnp.ndarray, shape (N_x,), float, optional
        Per-point weights w_i. Pair weights become w_ij = w_i*w_j and are used in both
        numerator and denominator.
    chunk_size : int
        Number of pairs stored per chunk. Larger values reduce Python overhead but increase
        memory. Chunking avoids dynamic slicing errors under JAX tracing/JIT.
    weight_dtype : jnp.dtype or None
        Optional dtype used for pair weights and denominators. If None, a reasonable dtype
        is inferred from x/weights (at least float32).

    Returns
    -------
    pb : PairBinning1D
        Precomputed pair list, bin assignments (safe), masks, weights, denominator-per-bin,
        pair counts per bin, and bin centers. Intended to be fed to
        `two_point_corr_1d_from_precomp(pb, y, ...)`.
    """
    x = jnp.asarray(x)
    if x.ndim != 1:
        raise ValueError(f"x must be 1D, got shape {x.shape}")
    N = int(x.shape[0])

    if logger.isEnabledFor(logging.INFO):
        logger.info(
            "precompute_pair_binning_1d: N=%d, n_bins=%d, include_self=%s, periodic_L=%s, chunk_size=%d",
            N, n_bins, include_self, str(periodic_L), int(chunk_size),
        )

    # Bin edges and centers
    if bins is None:
        bins = _make_linear_bins(x, n_bins, r_min=r_min, r_max=r_max, periodic_L=periodic_L)
    else:
        bins = jnp.asarray(bins)
        if bins.ndim != 1 or bins.shape[0] < 2:
            raise ValueError(f"bins must be 1D with length >= 2, got shape {bins.shape}")
        n_bins = int(bins.shape[0] - 1)
    r_centers = 0.5 * (bins[:-1] + bins[1:])

    # Unique pairs: upper triangle to avoid double-counting
    if include_self:
        ii, jj = jnp.triu_indices(N, k=0)
    else:
        ii, jj = jnp.triu_indices(N, k=1)

    # Separations r_ij
    dx = x[jj] - x[ii]
    if periodic_L is not None:
        L = jnp.asarray(periodic_L, dtype=x.dtype)
        dx = _min_image_separation_1d(dx, L)
    else:
        dx = jnp.abs(dx)

    # Map each separation to a bin index; mask out out-of-range separations
    b = jnp.searchsorted(bins, dx, side="right") - 1  # may be -1 or n_bins for out-of-range
    m = (b >= 0) & (b < n_bins)

    # IMPORTANT: make bin indices always safe for bincount
    b_safe = jnp.where(m, b, 0).astype(jnp.int32)

    # Pad pair arrays to multiple of chunk_size; reshape to (n_chunks, chunk_size)
    n_pairs = int(ii.shape[0])
    chunk_size = int(chunk_size)
    n_chunks = (n_pairs + chunk_size - 1) // chunk_size
    n_pad = n_chunks * chunk_size
    pad_len = n_pad - n_pairs

    if logger.isEnabledFor(logging.INFO):
        logger.info(
            "Pairs: n_pairs=%d, n_chunks=%d, n_pad=%d (pad_len=%d)",
            n_pairs, n_chunks, n_pad, pad_len
        )

    def _pad(arr, pad_value):
        return jnp.pad(arr, (0, pad_len), mode="constant", constant_values=pad_value)

    ii_p = _pad(ii, 0).astype(jnp.int32)
    jj_p = _pad(jj, 0).astype(jnp.int32)
    b_p  = _pad(b_safe, 0).astype(jnp.int32)
    m_p  = _pad(m, False)

    # Pair weights w_ij
    if weights is None:
        w_pt = None
    else:
        w_pt = jnp.asarray(weights)
        if w_pt.shape != (N,):
            raise ValueError(f"weights must have shape {(N,)}, got {w_pt.shape}")

    if weight_dtype is None:
        # Ensure at least float32 accumulation for denominators by default
        if w_pt is None:
            weight_dtype = jnp.result_type(x.dtype, jnp.float32)
        else:
            weight_dtype = jnp.result_type(x.dtype, w_pt.dtype, jnp.float32)

    if w_pt is None:
        denom_w = jnp.ones((n_pad,), dtype=weight_dtype)
    else:
        denom_w = (w_pt[ii_p] * w_pt[jj_p]).astype(weight_dtype)

    # Chunked views (static shapes -> JIT-safe indexing in the compute step)
    ii_2d = ii_p.reshape(n_chunks, chunk_size)
    jj_2d = jj_p.reshape(n_chunks, chunk_size)
    b_2d  = b_p.reshape(n_chunks, chunk_size)
    m_2d  = m_p.reshape(n_chunks, chunk_size)
    w_2d  = denom_w.reshape(n_chunks, chunk_size)

    # Denominator per bin: sum of weights of valid pairs in that bin (independent of y)
    den_per_bin = jnp.bincount(
        b_p,
        weights=jnp.where(m_p, denom_w, jnp.asarray(0.0, dtype=weight_dtype)),
        length=n_bins,
    ).astype(weight_dtype)

    # Pair counts per bin (debug/QA)
    counts_per_bin = jnp.bincount(
        b_p,
        weights=m_p.astype(jnp.int32),
        length=n_bins,
    ).astype(jnp.int32)

    return PairBinning1D(
        pair_i=ii_2d,
        pair_j=jj_2d,
        bin_idx=b_2d,
        mask=m_2d,
        denom_w=w_2d,
        den_per_bin=den_per_bin,
        counts_per_bin=counts_per_bin,
        r_centers=r_centers,
        n_bins=n_bins,
        chunk_size=chunk_size,
        n_x=N,
    )


def two_point_corr_1d_from_precomp(
    pb: PairBinning1D,
    y: jnp.ndarray,
    *,
    center: bool = True,
    standardize: bool = False,
    eps: float = 1e-12,
    acc_dtype: Optional[jnp.dtype] = None,
) -> jnp.ndarray:
    """
    Compute the 1D two-point correlation function xi(r) using precomputed pairs/bins.

    Objective (conceptual)
    ----------------------
    Given precomputed bins and pair lists (from x), compute for each sample realization y[s]:
      1) preprocess the field along x: yy = y - mean(y) (optional)
      2) optionally normalize: yy /= std(yy)
      3) bin-average the pair products yy[i]*yy[j] as a function of separation r

    Estimator
    ---------
    For each bin k:
        num[k] = sum_{(i,j) in bin k} w_ij * yy[i] * yy[j]
        den[k] = sum_{(i,j) in bin k} w_ij         (= pb.den_per_bin)
        xi[k]  = num[k] / den[k]

    Notes
    -----
    - pb.bin_idx is guaranteed to be in-range [0, n_bins-1], so jnp.bincount is safe.
      Invalid/padded pairs are excluded with pb.mask.
    - acc_dtype controls accumulation precision/speed (defaults to a reasonable dtype
      inferred from y and the precomputed weights).

    Parameters
    ----------
    pb : PairBinning1D
        Output of `precompute_pair_binning_1d(...)` containing pair indices, bin indices,
        masks, pair weights, denominators, and r bin centers.
    y : jnp.ndarray, shape (N_samples, N_x), float
        Field values for multiple realizations, sampled at the same x used to build `pb`.
        Each row y[s] is one realization.
    center : bool
        If True, subtract the per-sample mean over x before correlating:
            yy = y - mean(y)
        This computes correlations of fluctuations rather than raw y.
    standardize : bool
        If True, divide by the per-sample std over x:
            yy = yy / (std(yy) + eps)
        This yields a correlation-coefficient-like quantity.
    eps : float
        Small value to avoid division by zero when standardize=True.
    acc_dtype : jnp.dtype or None
        Accumulation dtype for yy, numerator, and the division by the denominator.
        If None, inferred from (y.dtype, pb.denom_w.dtype) and at least float32.

    Returns
    -------
    xi : jnp.ndarray, shape (N_samples, pb.n_bins), float
        Estimated 2PCF for each sample realization, evaluated at the bin centers pb.r_centers.
        Bins with zero valid pairs return NaN.
    """
    y = jnp.asarray(y)
    if y.ndim != 2:
        raise ValueError(f"y must be 2D (N_samples, N_x), got shape {y.shape}")
    if int(y.shape[1]) != pb.n_x:
        raise ValueError(f"y.shape[1] must match pb.n_x={pb.n_x}, got {int(y.shape[1])}")

    n_bins = pb.n_bins
    den = pb.den_per_bin

    if acc_dtype is None:
        acc_dtype = jnp.result_type(y.dtype, pb.denom_w.dtype, jnp.float32)

    den_acc = den.astype(acc_dtype)

    if logger.isEnabledFor(logging.INFO):
        logger.info(
            "two_point_corr_1d_from_precomp: N_samples=%d, N_x=%d, n_bins=%d, n_chunks=%d, acc_dtype=%s",
            int(y.shape[0]), int(y.shape[1]), n_bins, int(pb.pair_i.shape[0]), str(acc_dtype),
        )

    def corr_one(y1: jnp.ndarray) -> jnp.ndarray:
        # Preprocess one realization y1 -> yy
        yy = y1.astype(acc_dtype)
        if center:
            yy = yy - jnp.mean(yy)
        if standardize:
            yy = yy / (jnp.std(yy) + jnp.asarray(eps, dtype=acc_dtype))

        # Numerator accumulation: static loop over chunks
        def body(k, num):
            ii = pb.pair_i[k]   # (chunk_size,)
            jj = pb.pair_j[k]
            b  = pb.bin_idx[k]  # safe in [0, n_bins-1]
            m  = pb.mask[k]
            w  = pb.denom_w[k].astype(acc_dtype)

            # Pair contribution: w_ij * yy[i] * yy[j]
            prod = w * yy[ii] * yy[jj]
            prod = jnp.where(m, prod, jnp.asarray(0.0, dtype=acc_dtype))  # drop invalid/padded pairs

            # Bin contributions by separation bin index
            num = num + jnp.bincount(b, weights=prod, length=n_bins).astype(acc_dtype)
            return num

        num0 = jnp.zeros((n_bins,), dtype=acc_dtype)
        num = jax.lax.fori_loop(0, pb.pair_i.shape[0], body, num0)

        # Normalize per bin
        xi = jnp.where(den_acc > 0, num / den_acc, jnp.nan)
        return xi

    return jax.vmap(corr_one)(y)


def two_point_corr_1d(
    x: jnp.ndarray,
    y: jnp.ndarray,
    *,
    bins: Optional[jnp.ndarray] = None,
    n_bins: int = 64,
    r_min: float = 0.0,
    r_max: Optional[float] = None,
    periodic_L: Optional[float] = None,
    include_self: bool = False,
    weights: Optional[jnp.ndarray] = None,
    center: bool = True,
    standardize: bool = False,
    chunk_size: int = 1 << 16,
    eps: float = 1e-12,
    weight_dtype: Optional[jnp.dtype] = None,
    acc_dtype: Optional[jnp.dtype] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Convenience wrapper that computes xi(r) directly from (x, y).

    Objective (conceptual)
    ----------------------
    This is the "one-shot" interface:
      - build the pair/bin mapping from x
      - compute xi(r) for each sample in y

    If you will compute xi for many different y arrays sharing the same x/bins,
    prefer the two-step API:
      pb = precompute_pair_binning_1d(x, ...)
      xi = two_point_corr_1d_from_precomp(pb, y, ...)

    Parameters
    ----------
    x : jnp.ndarray, shape (N_x,), float
        Sample positions (not necessarily uniform).
    y : jnp.ndarray, shape (N_samples, N_x), float
        Field values at positions x for multiple realizations (rows).
    bins : jnp.ndarray, shape (n_bins+1,), float, optional
        Explicit separation-bin edges. If provided, overrides (n_bins, r_min, r_max).
    n_bins : int
        Number of bins (used only if bins is None).
    r_min : float
        Minimum separation (used only if bins is None).
    r_max : float or None
        Maximum separation (used only if bins is None).
    periodic_L : float or None
        If provided, separations are computed with minimum-image convention (robust to x not being wrapped).
    include_self : bool
        If True, include i==j pairs (separation 0). Default False uses i<j only.
    weights : jnp.ndarray, shape (N_x,), float, optional
        Per-point weights w_i used to build pair weights w_ij = w_i*w_j.
    center : bool
        If True, subtract per-sample mean over x before correlating.
    standardize : bool
        If True, divide per-sample by std over x (correlation-coefficient-like).
    chunk_size : int
        Pair chunk size for JIT-safe accumulation.
    eps : float
        Stabilizer for standardize=True.
    weight_dtype : jnp.dtype or None
        Optional dtype used for pair weights and denominators. If None, inferred.
    acc_dtype : jnp.dtype or None
        Accumulation dtype for the correlation computation. If None, inferred.

    Returns
    -------
    r_centers : jnp.ndarray, shape (n_bins,), float
        Separation bin centers (same units as x).
    xi : jnp.ndarray, shape (N_samples, n_bins), float
        Estimated 2PCF for each sample realization, evaluated at r_centers.
    """
    pb = precompute_pair_binning_1d(
        x,
        bins=bins,
        n_bins=n_bins,
        r_min=r_min,
        r_max=r_max,
        periodic_L=periodic_L,
        include_self=include_self,
        weights=weights,
        chunk_size=chunk_size,
        weight_dtype=weight_dtype,
    )
    xi = two_point_corr_1d_from_precomp(
        pb,
        y,
        center=center,
        standardize=standardize,
        eps=eps,
        acc_dtype=acc_dtype,
    )
    return pb.r_centers, xi