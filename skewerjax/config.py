"""The package's precision policy and its parallel set-up (CPU threads and host devices, GPU devices).

**Precision.** One policy for the whole package, ``"float32"`` by default, ``"float64"`` an explicit opt-in for
validation:

    import skewerjax
    skewerjax.config.set_precision("float64")      # or the environment variable SKEWERJAX_PRECISION=float64
    with skewerjax.config.precision("float64"): ...  # for a block only

The policy is the dtype of the device kernels (the tau sum, the line profiles, the statistics of the arrays they
return): ``real_dtype()``. Two things never follow it, because float32 would lose accuracy there and they are cheap:

* the **host geometry and cosmology** run in float64 NumPy / SciPy whatever the policy: path coordinates (a cumulative
  sum over ~10^4 points), the scale factors a(s) and the Hubble velocities V_H = c ln(a_dagger / a) of the line centres
  (a difference of nearly equal numbers times c: in float32 it would err by ~0.02 km/s, 1 % of a Si III width),
  selection and deposition;
* JAX's own ``jax_enable_x64`` flag: float64 needs it, so ``set_precision("float64")`` turns it on; ``"float32"`` leaves
  it as it is and casts what the package sends to the device (with x64 on, as in the tests, inputs are float64 and
  are cast at the entry points).

**Parallelism.** ``configure_devices(n_cpu_devices=..., cpu_threads=...)`` sets XLA's host platform before JAX starts
(``--xla_force_host_platform_device_count``, intra-op threads): it must run before the first ``import jax`` of the
process. ``devices("cpu" | "gpu" | None)`` lists the devices that the ``devices=`` option of
``core.compute_optical_depth`` splits the skewers over, the same API on both.
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager

import numpy as np

PRECISIONS = ("float32", "float64")
_ENV = "SKEWERJAX_PRECISION"
_state = {"precision": None}


def _check(p: str) -> str:
    p = str(p).lower()
    if p not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}, got {p!r}.")
    return p


def get_precision() -> str:
    """The current policy: ``set_precision``'s value, else ``$SKEWERJAX_PRECISION``, else ``"float32"``."""
    if _state["precision"] is None:
        _state["precision"] = _check(os.environ.get(_ENV, "float32"))
        if _state["precision"] == "float64":
            _enable_x64()
    return _state["precision"]


def _enable_x64() -> None:
    import jax

    if not jax.config.jax_enable_x64:
        jax.config.update("jax_enable_x64", True)


def set_precision(precision: str) -> None:
    """Set the package's precision, ``"float32"`` (default) or ``"float64"`` (turns ``jax_enable_x64`` on). Set it
    before the computations: arrays already on the device keep their dtype, and jitted functions retrace for the new
    one."""
    p = _check(precision)
    if p == "float64":
        _enable_x64()
    _state["precision"] = p


@contextmanager
def precision(p: str):
    """``with precision("float64"): ...``: the policy for a block, restored on exit (``jax_enable_x64`` stays on once
    turned on)."""
    old = get_precision()
    set_precision(p)
    try:
        yield
    finally:
        set_precision(old)


def real_dtype() -> np.dtype:
    """The floating dtype of the device kernels under the current policy (``np.float32`` or ``np.float64``)."""
    if get_precision() == "float64":
        import jax

        if not jax.config.jax_enable_x64:  # turned off by the user after set_precision("float64")
            raise RuntimeError('precision "float64" needs jax_enable_x64: call skewerjax.config.set_precision("float64") again.')
        return np.dtype(np.float64)
    return np.dtype(np.float32)


def eps() -> float:
    """The machine epsilon of ``real_dtype()`` (1.19e-7 in float32, 2.22e-16 in float64): the scale of the rounding
    tolerances of the tests and of the float32 audit (docs/precision.md)."""
    return float(np.finfo(real_dtype()).eps)


def device_array(x, dtype=None):
    """``jnp.asarray(x)`` with floating values in ``dtype`` (default ``real_dtype()``); integers and booleans unchanged."""
    import jax.numpy as jnp

    a = jnp.asarray(x)
    if jnp.issubdtype(a.dtype, jnp.floating):
        return a.astype(real_dtype() if dtype is None else dtype)
    return a


# --------------------------------------------------------------------------- parallel set-up
def configure_devices(n_cpu_devices: int | None = None, cpu_threads: int | None = None, *, platforms: str | None = None) -> dict:
    """Configure XLA's host platform **before JAX is imported** (an error otherwise: the flags are read once).

    n_cpu_devices : expose the CPU as this many JAX devices (``--xla_force_host_platform_device_count``), so that
        ``core.compute_optical_depth(devices=jax.devices("cpu"))`` runs one shard of the skewers per device; default
        ``os.cpu_count()``. Each device is an independent XLA client thread pool: sharding over them is how the tau sum
        uses every core (one device runs its elementwise loops on a few threads only).
    cpu_threads : the threads of XLA's CPU intra-op pool (``--xla_cpu_multi_thread_eigen``, ``intra_op_parallelism_threads``);
        default: left to XLA (all the cores).
    platforms : ``JAX_PLATFORMS`` (e.g. "cpu", or "cuda,cpu" for a GPU next to the host devices); default: unchanged.

    Returns the flags set. Sets ``XLA_FLAGS`` and, if given, ``JAX_PLATFORMS`` in ``os.environ``.
    """
    if "jax" in sys.modules:
        raise RuntimeError("configure_devices must run before the first import of jax (the XLA flags are read once).")
    n = os.cpu_count() if n_cpu_devices is None else int(n_cpu_devices)
    flags = [f for f in os.environ.get("XLA_FLAGS", "").split() if not f.startswith("--xla_force_host_platform_device_count")]
    flags.append(f"--xla_force_host_platform_device_count={n}")
    if cpu_threads is not None:
        flags = [f for f in flags if "intra_op_parallelism_threads" not in f]
        flags.append(f"--xla_cpu_multi_thread_eigen={'true' if int(cpu_threads) > 1 else 'false'}")
        flags.append(f"intra_op_parallelism_threads={int(cpu_threads)}")
    os.environ["XLA_FLAGS"] = " ".join(flags)
    if platforms is not None:
        os.environ["JAX_PLATFORMS"] = platforms
    return dict(XLA_FLAGS=os.environ["XLA_FLAGS"], JAX_PLATFORMS=os.environ.get("JAX_PLATFORMS"))


def devices(kind: str | None = None) -> list:
    """The devices of one kind: ``"cpu"`` (the host devices of ``configure_devices``), ``"gpu"`` (empty without one) or
    None (JAX's default backend). The value of the ``devices=`` option of ``core.compute_optical_depth``."""
    import jax

    if kind is None:
        return list(jax.devices())
    try:
        return list(jax.devices(kind))
    except RuntimeError:  # no such backend
        return []
