"""Input validation shared by the entry points: informative errors for non-finite or out-of-range values.

Every check is eager and cheap (one reduction per array, on the device for JAX arrays); the details (how many values,
the first bad index) are only computed when a check fails. Traced values (inside ``jit`` / ``vmap``) are not checked.
"""
from __future__ import annotations

import warnings
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np


def is_traced(*xs) -> bool:
    """True if any of ``xs`` is a JAX tracer (inside ``jit`` / ``vmap`` / ``grad``): values cannot be checked."""
    return any(isinstance(x, jax.core.Tracer) for x in xs)


def _first_bad(a: np.ndarray, bad: np.ndarray) -> str:
    if a.ndim == 0:
        return f"got {a.item()!r}"
    idx = tuple(int(i) for i in np.argwhere(bad)[0])
    return f"{int(bad.sum())} of {a.size}, the first at index {idx if len(idx) > 1 else idx[0]} = {a[idx]!r}"


def check_values(name: str, x, *, finite: bool = True, minimum: Optional[float] = None, strict: bool = False) -> None:
    """Raise ``ValueError`` if ``x`` holds NaN / inf (``finite``) or values below ``minimum`` (``strict``: <= minimum).

    ``name`` names the input in the message (e.g. "Cells: species 'HI', temperature [K]"). Traced arrays are skipped.
    """
    if x is None or is_traced(x):
        return
    on_device = isinstance(x, jax.Array)
    xp = jnp if on_device else np
    a = x if on_device else np.asarray(x)
    if a.size == 0 or a.dtype.kind not in "fiu":
        return
    with jax.ensure_compile_time_eval():          # concrete values inside a jit trace: evaluate, do not stage
        bad_finite = finite and a.dtype.kind == "f" and not bool(xp.all(xp.isfinite(a)))
        bad_low = minimum is not None and bool(xp.any((a <= minimum) if strict else (a < minimum)))
    if bad_finite:
        h = np.asarray(a)
        raise ValueError(f"{name}: non-finite values (NaN or inf): {_first_bad(h, ~np.isfinite(h))}.")
    if bad_low:
        h = np.asarray(a)
        raise ValueError(f"{name} must be {'>' if strict else '>='} {minimum:g}: {_first_bad(h, (h <= minimum) if strict else (h < minimum))}.")


def check_shape(name: str, x, shape: tuple) -> None:
    """Raise ``ValueError`` if ``np.shape(x) != shape``."""
    if tuple(np.shape(x)) != tuple(shape):
        raise ValueError(f"{name} must have shape {tuple(shape)}, got {tuple(np.shape(x))}.")


def check_positive_int(name: str, x) -> None:
    """Raise ``ValueError`` unless ``x`` is None or an integer >= 1 (chunk sizes and the like)."""
    if x is not None and (int(x) != x or int(x) < 1):
        raise ValueError(f"{name} must be a positive integer (or None), got {x!r}.")


_WARNED = set()


def warn_once(key: str, message: str, stacklevel: int = 3) -> None:
    """``warnings.warn(message)`` the first time ``key`` is seen in this process."""
    if key not in _WARNED:
        _WARNED.add(key)
        warnings.warn(message, stacklevel=stacklevel)


def warn_if_float32(where: str) -> None:
    """Warn once when ``jax_enable_x64`` is off: SkewerJAX follows the flag (it never sets it), and its tolerances, the
    column sums and tau are validated in float64 (float32 tau: max |dF| ~1e-3 on TNG50-4 skewers)."""
    if not jax.config.jax_enable_x64:
        warn_once("x64", f"{where}: jax_enable_x64 is off, so the computation runs in float32 (max |dF| ~1e-3 on "
                         "TNG50-4 skewers); call jax.config.update('jax_enable_x64', True) before any computation for float64.")
