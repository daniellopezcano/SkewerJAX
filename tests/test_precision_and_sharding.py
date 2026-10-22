"""The opt-in float32 kernel and the sharded optical depth (``core.optical_depth_from_samples(dtype=)``,
``core.optical_depth_sharded``) against the default path, on toy fields."""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

from skewerjax.core import (Constant, Fields, LineBundle, LogRamp, ScalarField, SkewerBundle, VectorField, VelocityRamp,
                            los_samples, optical_depth_from_samples, optical_depth_sharded, plane_points)

L = 1e-3


def toy(n_skewers: int = 7):
    sp = lambda: {"temperature": ScalarField(LogRamp(1e4, 1e6, L, 3)), "velocity": VectorField(VelocityRamp(-50.0, 80.0, L)),
                  "number_density": ScalarField(LogRamp(1e-9, 1e-6, L, 2)), "turbulence": ScalarField(Constant(0.0)),
                  "collisional": ScalarField(Constant(0.1))}
    lb = LineBundle(["HI-LyA", "SiIII-1206"])
    pts = plane_points(L, "x", L / 2, n_skewers, 64)
    sb = SkewerBundle(spatial_points=pts, manual_scale_factors=jnp.full((1,) + pts.shape[:2], 0.9))
    return los_samples(sb, Fields({"HI": sp(), "SiIII": sp()}), lb, a_dagger=0.88), jnp.linspace(-400, 400, 300), lb


def test_float32_is_opt_in_and_close():
    smp, v, lb = toy()
    t64 = np.asarray(optical_depth_from_samples(smp, v, lb))
    t32 = np.asarray(optical_depth_from_samples(smp, v, lb, dtype="float32"))
    assert t32.dtype == t64.dtype == np.float64                     # cast back to the input precision
    assert np.max(np.abs(np.exp(-t32) - np.exp(-t64))) < 1e-4


def test_sharded_one_device_is_identical():
    smp, v, lb = toy()
    assert np.array_equal(np.asarray(optical_depth_sharded(smp, v, lb)), np.asarray(optical_depth_from_samples(smp, v, lb)))


SCRIPT = """
import os; os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"; os.environ["JAX_PLATFORMS"] = "cpu"
import sys; sys.path.insert(0, {here!r})
import numpy as np, jax; jax.config.update("jax_enable_x64", True)
from test_precision_and_sharding import toy
from skewerjax.core import optical_depth_sharded, optical_depth_from_samples
assert len(jax.devices()) == 4
smp, v, lb = toy(7)                                                  # 7 skewers on 4 devices: padded
assert np.array_equal(np.asarray(optical_depth_sharded(smp, v, lb)), np.asarray(optical_depth_from_samples(smp, v, lb)))
"""


def test_sharded_four_cpu_devices_is_identical():
    """In a separate process: the device count is fixed when JAX starts."""
    here = str(Path(__file__).resolve().parent)
    p = subprocess.run([sys.executable, "-c", SCRIPT.format(here=here)], capture_output=True, text=True,
                       env={k: v for k, v in os.environ.items() if k != "XLA_FLAGS"})
    assert p.returncode == 0, p.stderr[-2000:]


def test_jit_of_the_samples_path():
    smp, v, lb = toy(3)
    f = jax.jit(lambda s, vv: optical_depth_from_samples(s, vv, lb))
    assert np.allclose(np.asarray(f(smp, v)), np.asarray(optical_depth_from_samples(smp, v, lb)), rtol=1e-12, atol=0)
