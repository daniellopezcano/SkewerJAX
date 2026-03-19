from __future__ import annotations
from typing import Optional, Tuple, Literal, Dict, Any

import jax
import jax.numpy as jnp
import jax.scipy.signal as jsig


def _infer_uniform_dx(x: jnp.ndarray, rtol: float = 1e-5, atol: float = 1e-8) -> float:
    x = jnp.asarray(x)
    if x.ndim != 1 or x.size < 2:
        raise ValueError(f"x must be 1D with >=2 points, got {x.shape}")
    dxs = jnp.diff(x)
    dx0 = dxs[0]
    if not bool(jnp.all(jnp.isclose(dxs, dx0, rtol=rtol, atol=atol))):
        raise ValueError(
            "Welch/FFT spectral estimation assumes uniform sampling in x. "
            "Resample/interpolate to a uniform grid first if needed."
        )
    return float(dx0)


def precompute_welch_1d(
    x: jnp.ndarray,
    *,
    nperseg: Optional[int] = None,
    noverlap: Optional[int] = None,
    nfft: Optional[int] = None,
    window: str = "hann",
    detrend: Literal["constant", "linear"] = "constant",
    scaling: Literal["density", "spectrum"] = "density",
    average: Literal["mean", "median"] = "mean",
    return_onesided: bool = True,
    as_angular: bool = True,   # True -> k=2πf, False -> f
    rtol: float = 1e-2,
    atol: float = 1e-2,
) -> Dict[str, Any]:
    """
    Returns a small dict of Welch settings derived from the common x-grid.
    """
    x = jnp.asarray(x)
    N = int(x.size)
    dx = _infer_uniform_dx(x, rtol=rtol, atol=atol)
    fs = 1.0 / dx

    nperseg = min(256, N) if nperseg is None else int(nperseg)
    noverlap = (nperseg // 2) if noverlap is None else int(noverlap)
    nfft = nperseg if nfft is None else int(nfft)

    if nperseg < 2:
        raise ValueError("nperseg must be >= 2")
    if not (0 <= noverlap < nperseg):
        raise ValueError("noverlap must satisfy 0 <= noverlap < nperseg")
    if nfft < nperseg:
        raise ValueError("nfft must be >= nperseg")

    return dict(
        fs=float(fs),
        dx=float(dx),
        N=N,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        window=window,
        detrend=str(detrend),
        scaling=str(scaling),
        average=str(average),
        return_onesided=bool(return_onesided),
        as_angular=bool(as_angular),
    )


def psd_auto_from_precomp(
    pc: Dict[str, Any],
    y: jnp.ndarray,  # (N_realizations, N_x)
    *,
    return_mean_over_realizations: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Auto PSD via Welch: S_xx(f) for each realization (and optionally mean across realizations).
    """
    y = jnp.asarray(y)
    if y.ndim != 2 or int(y.shape[1]) != int(pc["N"]):
        raise ValueError(f"y must have shape (N_realizations, {pc['N']}), got {y.shape}")

    f, Pxx = jsig.welch(
        y,
        fs=pc["fs"],
        window=pc["window"],
        nperseg=pc["nperseg"],
        noverlap=pc["noverlap"],
        nfft=pc["nfft"],
        detrend=pc["detrend"],
        return_onesided=pc["return_onesided"],
        scaling=pc["scaling"],
        axis=-1,
        average=pc["average"],
    )
    xfreq = (2.0 * jnp.pi * f) if pc["as_angular"] else f
    if return_mean_over_realizations:
        return xfreq, jnp.mean(Pxx, axis=0)
    return xfreq, Pxx


def csd_cross_from_precomp(
    pc: Dict[str, Any],
    y1: jnp.ndarray,
    y2: jnp.ndarray,
    *,
    symmetrize: bool = False,
    symmetrize_norm: Literal["sum", "mean"] = "sum",
    return_mean_over_realizations: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Cross spectrum via Welch CSD: S_xy(f), complex in general.
    If symmetrize=True:
      'sum'  -> S_xy + S_yx = 2*Re(S_xy)
      'mean' -> Re(S_xy)
    """
    y1 = jnp.asarray(y1)
    y2 = jnp.asarray(y2)
    if y1.shape != y2.shape or y1.ndim != 2 or int(y1.shape[1]) != int(pc["N"]):
        raise ValueError(f"y1,y2 must have same shape (N_realizations, {pc['N']}), got {y1.shape}, {y2.shape}")

    f, Pxy = jsig.csd(
        y1, y2,
        fs=pc["fs"],
        window=pc["window"],
        nperseg=pc["nperseg"],
        noverlap=pc["noverlap"],
        nfft=pc["nfft"],
        detrend=pc["detrend"],
        return_onesided=pc["return_onesided"],
        scaling=pc["scaling"],
        axis=-1,
        average=pc["average"],
    )
    xfreq = (2.0 * jnp.pi * f) if pc["as_angular"] else f

    out = Pxy
    if symmetrize:
        out = 2.0 * jnp.real(Pxy)
        if symmetrize_norm == "mean":
            out = 0.5 * out

    if return_mean_over_realizations:
        return xfreq, jnp.mean(out, axis=0)
    return xfreq, out


def coherence_from_psd_csd(Pxx: jnp.ndarray, Pyy: jnp.ndarray, Pxy: jnp.ndarray, *, eps: float = 1e-30) -> jnp.ndarray:
    """Magnitude-squared coherence: |Pxy|^2 / (Pxx*Pyy)."""
    return (jnp.abs(Pxy) ** 2) / (Pxx * Pyy + eps)