"""The precision policy and the device helpers (``skewerjax.config``; docs/precision.md)."""

import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np

import skewerjax.config as cfg
from skewerjax.core import LineBundle, cell_pieces, compute_optical_depth

from helpers import raises, uniform_skewer


def test_policy_and_context():
    old = cfg.get_precision()
    try:
        cfg.set_precision("float32")
        assert cfg.real_dtype() == np.float32 and cfg.eps() == np.finfo(np.float32).eps
        assert cfg.device_array(np.ones(3)).dtype == jnp.float32 and cfg.device_array(np.arange(3)).dtype.kind == "i"
        with cfg.precision("float64"):
            assert cfg.real_dtype() == np.float64 and jax.config.jax_enable_x64
            assert cfg.device_array(np.ones(3)).dtype == jnp.float64
        assert cfg.get_precision() == "float32"
        raises(ValueError, "precision must be one of", cfg.set_precision, "float16")
    finally:
        cfg.set_precision(old)


def test_tau_follows_the_policy_and_the_host_geometry_does_not():
    """tau is returned in the policy's dtype, the pieces' Hubble velocities and the bundle's host copies are float64, and
    float32 agrees with float64 within the float32 floor of the velocity coordinate (docs/precision.md)."""
    lya = LineBundle(["HI-LyA"])
    v = np.linspace(0.0, 120.0, 241)
    out = {}
    for p in ("float32", "float64"):
        with cfg.precision(p):
            sb, cells, _, _ = uniform_skewer()
            assert sb.host["scale_factors"].dtype == np.float64 and sb.scale_factors.dtype == cfg.real_dtype()
            pieces = cell_pieces(cells, lya, skewer_bundle=sb)
            assert pieces.hubble_kms is not None and pieces.temperature.dtype == cfg.real_dtype()
            tau = compute_optical_depth(pieces, v, lya, resolution_tolerance=None)
            assert tau.dtype == cfg.real_dtype()
            out[p] = np.asarray(tau, dtype=np.float64)
    scale = out["float64"].max()
    assert np.max(np.abs(out["float32"] - out["float64"])) <= 1e-4 * scale


def test_configure_devices_before_jax():
    """``configure_devices`` must run before JAX is imported (here it already is) and sets the host device count."""
    raises(RuntimeError, "before the first import of jax", cfg.configure_devices, 4)
    code = (
        "import skewerjax.config as c; c.configure_devices(n_cpu_devices=3, platforms='cpu'); "
        "import jax; print(len(c.devices('cpu')), len(c.devices('gpu')))"
    )
    env = {k: v for k, v in os.environ.items() if k not in ("XLA_FLAGS", "JAX_PLATFORMS")}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, env=env).stdout.split()
    assert out == ["3", "0"]
