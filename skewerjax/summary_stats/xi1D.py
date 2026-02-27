from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class PairBinning1D:
    # chunked (n_chunks, chunk_size)
    pair_i: jnp.ndarray       # int32
    pair_j: jnp.ndarray       # int32
    bin_idx: jnp.ndarray      # int32 in [0, n_bins-1]
    mask: jnp.ndarray         # bool
    denom_w: jnp.ndarray      # float
    den_per_bin: jnp.ndarray  # (n_bins,) float
    r_centers: jnp.ndarray    # (n_bins,) float
    n_bins: int
    chunk_size: int


def _make_linear_bins(
    x: jnp.ndarray,
    n_bins: int,
    r_min: float = 0.0,
    r_max: Optional[float] = None,
    *,
    periodic_L: Optional[float] = None,
) -> jnp.ndarray:
    if r_max is None:
        if periodic_L is not None:
            r_max = 0.5 * float(periodic_L)
        else:
            r_max = float(jnp.max(x) - jnp.min(x))
    return jnp.linspace(r_min, r_max, n_bins + 1)


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
) -> PairBinning1D:
    """
    Precompute (pair -> bin) mapping for 1D pair-binned 2PCF.
    Designed so the *computation* can be safely jitted (no dynamic slicing).
    """
    x = jnp.asarray(x)
    if x.ndim != 1:
        raise ValueError(f"x must be 1D, got shape {x.shape}")
    N = int(x.shape[0])

    if bins is None:
        bins = _make_linear_bins(x, n_bins, r_min=r_min, r_max=r_max, periodic_L=periodic_L)
    else:
        bins = jnp.asarray(bins)
        n_bins = int(bins.shape[0] - 1)

    r_centers = 0.5 * (bins[:-1] + bins[1:])

    # all upper-triangular pairs
    if include_self:
        ii, jj = jnp.triu_indices(N, k=0)
    else:
        ii, jj = jnp.triu_indices(N, k=1)

    # separations
    dx = jnp.abs(x[jj] - x[ii])
    if periodic_L is not None:
        L = jnp.asarray(periodic_L, dtype=dx.dtype)
        dx = jnp.minimum(dx, L - dx)

    # bin assignment
    b = jnp.searchsorted(bins, dx, side="right") - 1  # candidate in [-1..n_bins]
    m = (b >= 0) & (b < n_bins)

    # pad to multiple of chunk_size and reshape to (n_chunks, chunk_size)
    n_pairs = int(ii.shape[0])
    chunk_size = int(chunk_size)
    n_chunks = (n_pairs + chunk_size - 1) // chunk_size
    n_pad = n_chunks * chunk_size
    pad_len = n_pad - n_pairs

    def _pad(arr, pad_value):
        return jnp.pad(arr, (0, pad_len), mode="constant", constant_values=pad_value)

    ii_p = _pad(ii, 0).astype(jnp.int32)
    jj_p = _pad(jj, 0).astype(jnp.int32)
    b_p  = _pad(b, 0).astype(jnp.int32)
    m_p  = _pad(m, False)

    if weights is None:
        denom_w = jnp.ones((n_pad,), dtype=x.dtype)
    else:
        w = jnp.asarray(weights)
        if w.shape != (N,):
            raise ValueError(f"weights must have shape {(N,)}, got {w.shape}")
        denom_w = (w[ii_p] * w[jj_p]).astype(x.dtype)

    # reshape to (n_chunks, chunk_size) to avoid dynamic slice errors inside jit
    ii_2d = ii_p.reshape(n_chunks, chunk_size)
    jj_2d = jj_p.reshape(n_chunks, chunk_size)
    b_2d  = b_p.reshape(n_chunks, chunk_size)
    m_2d  = m_p.reshape(n_chunks, chunk_size)
    w_2d  = denom_w.reshape(n_chunks, chunk_size)

    # precompute denominator per bin (independent of y)
    den_per_bin = jnp.bincount(
        b_p,
        weights=jnp.where(m_p, denom_w, 0.0),
        length=n_bins,
    )

    return PairBinning1D(
        pair_i=ii_2d,
        pair_j=jj_2d,
        bin_idx=b_2d,
        mask=m_2d,
        denom_w=w_2d,
        den_per_bin=den_per_bin,
        r_centers=r_centers,
        n_bins=n_bins,
        chunk_size=chunk_size,
    )


def two_point_corr_1d_from_precomp(
    pb: PairBinning1D,
    y: jnp.ndarray,
    *,
    center: bool = True,
    standardize: bool = False,
    eps: float = 1e-12,
) -> jnp.ndarray:
    """
    Compute xi using a precomputed PairBinning1D.
    Output shape: (N_samples, n_bins)
    """
    y = jnp.asarray(y)
    if y.ndim != 2:
        raise ValueError(f"y must be 2D (N_samples, N_x), got shape {y.shape}")

    n_bins = pb.n_bins
    den = pb.den_per_bin

    def corr_one(y1: jnp.ndarray) -> jnp.ndarray:
        yy = y1
        if center:
            yy = yy - jnp.mean(yy)
        if standardize:
            yy = yy / (jnp.std(yy) + eps)

        def body(k, num):
            ii = pb.pair_i[k]       # (chunk_size,)
            jj = pb.pair_j[k]
            b  = pb.bin_idx[k]
            m  = pb.mask[k]
            w  = pb.denom_w[k]

            prod = w * yy[ii] * yy[jj]
            prod = jnp.where(m, prod, 0.0)

            num = num + jnp.bincount(b, weights=prod, length=n_bins)
            return num

        num0 = jnp.zeros((n_bins,), dtype=y.dtype)
        num = jax.lax.fori_loop(0, pb.pair_i.shape[0], body, num0)

        xi = jnp.where(den > 0, num / den, jnp.nan)
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
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Convenience wrapper: precompute bins/pairs then compute xi.
    Returns (r_centers, xi) with xi shape (N_samples, n_bins).
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
    )
    xi = two_point_corr_1d_from_precomp(pb, y, center=center, standardize=standardize, eps=eps)
    return pb.r_centers, xi