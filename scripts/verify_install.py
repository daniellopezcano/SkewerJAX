#!/usr/bin/env python
"""SkewerJAX environment smoke tests (safe on CPU-only or GPU hosts).

Reports the environment, the JAX devices (with a jitted matmul), the scientific stack (NumPy, SciPy, h5py,
matplotlib, iminuit, diffrax, ExoJAX), the optional fake_spectra, and imports the three SkewerJAX blocks and runs
one tiny optical depth. Forces JAX to the CPU for this process if no NVIDIA GPU is found (quiet CUDA logs).
Exit code 1 only if a core component (jax, skewerjax) is missing.
"""
from __future__ import annotations

import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time

RESULTS = []


def header(t: str):
    print(f"\n{'=' * len(t)}\n{t}\n{'=' * len(t)}")


def report(ok: bool, msg: str):
    RESULTS.append(ok)
    print(f"[{'OK' if ok else 'FAIL'}] {msg}")


def try_import(name: str):
    try:
        return __import__(name, fromlist=["_"]), ""
    except Exception as e:                                    # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"


def nvidia_gpus():
    """GPU names from nvidia-smi ([] if absent), and the driver version."""
    if shutil.which("nvidia-smi") is None:
        return None, []
    q = lambda f: subprocess.run(["nvidia-smi", f"--query-gpu={f}", "--format=csv,noheader"], capture_output=True, text=True,
                                 check=True).stdout.split("\n")
    try:
        return q("driver_version")[0].strip(), [n.strip() for n in q("name") if n.strip()]
    except Exception:                                         # noqa: BLE001
        return "unknown", []


def section(name: str, test):
    """Import ``name``, print its version and run ``test(module) -> (ok, message)``."""
    header(name)
    mod, err = try_import(name)
    if mod is None:
        report(False, f"import {name} -> {err}")
        return None
    print(f"{name:11s}: {getattr(mod, '__version__', 'unknown')}")
    try:
        report(*test(mod))
    except Exception as e:                                    # noqa: BLE001
        report(False, f"{name} test -> {type(e).__name__}: {e}")
    return mod


def test_jax(jax):
    import jax.numpy as jnp
    print(f"Backend    : {jax.default_backend()} | devices {jax.devices()}")
    plugins = [p for p in ("jax_plugins.xla_cuda13", "jax_plugins.xla_cuda12") if try_import(p)[0] is not None]
    print(f"Plugins    : {', '.join(plugins) or 'none'}")
    try:
        gpus = jax.devices("gpu")
    except Exception:                                         # noqa: BLE001
        gpus = []
    dev = gpus[0] if gpus else jax.devices()[0]
    if plugins and not gpus:
        print("Info: CUDA plugin wheels are installed but no CUDA GPU is usable; set JAX_PLATFORMS=cpu to silence the logs.")
    n = 1024 if dev.platform == "gpu" else 512
    a = jax.device_put(jax.random.normal(jax.random.key(0), (n, n), dtype=jnp.float32), dev)
    mm = jax.jit(lambda x, y: x @ y)
    mm(a, a).block_until_ready()
    t = time.time()
    c = mm(a, a).block_until_ready()
    return True, f"jitted matmul {c.shape} on {dev.platform.upper()} in {(time.time() - t) * 1e3:.1f} ms"


def test_h5py(h5py):
    import numpy as np
    fd, path = tempfile.mkstemp(prefix="skewerjax_", suffix=".h5")
    os.close(fd)
    with h5py.File(path, "w") as f:
        f["d"] = np.arange(5, dtype=np.int32)
    with h5py.File(path, "r") as f:
        ok = bool((f["d"][...] == np.arange(5)).all())
    os.remove(path)
    return ok, "HDF5 write/read round trip"


def test_iminuit(iminuit):
    import numpy as np
    x = np.linspace(-1, 1, 50)
    y = 2.5 * x - 0.4 + np.random.default_rng(0).normal(0, 0.05, x.size)
    m = iminuit.Minuit(lambda a, b: np.sum(((y - a * x - b) / 0.05) ** 2), a=1.0, b=0.0)
    m.errordef = iminuit.Minuit.LEAST_SQUARES
    m.migrad()
    return abs(m.values["a"] - 2.5) < 0.1 and abs(m.values["b"] + 0.4) < 0.1, f"linear fit a = {m.values['a']:.3f}, b = {m.values['b']:.3f}"


def test_skewerjax(sjx):
    import jax.numpy as jnp
    from skewerjax.core import Constant, Fields, LineBundle, LogRamp, ScalarField, SkewerBundle, VectorField, VelocityRamp, compute_optical_depth
    import skewerjax.simulations, skewerjax.comparison                       # noqa: F401, E401
    pts = jnp.stack([jnp.linspace(0.0, 1e-3, 16), jnp.full(16, 5e-4), jnp.full(16, 5e-4)], -1)[None]
    sb = SkewerBundle(spatial_points=pts, manual_scale_factors=jnp.ones((1, 1, 16)))
    sp = {"temperature": ScalarField(LogRamp(1e4, 1e4)), "velocity": VectorField(VelocityRamp(0.0, 0.0)),
          "number_density": ScalarField(LogRamp(1e-7, 1e-7)), "turbulence": ScalarField(Constant(0.0)),
          "collisional": ScalarField(Constant(0.0))}
    tau = compute_optical_depth(sb, jnp.linspace(-200, 200, 64), LineBundle(["HI-LyA"]), Fields({"HI": sp}), a_dagger=1.0)
    return bool(jnp.all(jnp.isfinite(tau)) and float(tau.max()) > 0), f"blocks core / simulations / comparison import; tau {tuple(tau.shape)}"


def main() -> int:
    driver, gpus = nvidia_gpus()
    if "JAX_PLATFORMS" not in os.environ and not gpus:
        os.environ["JAX_PLATFORMS"] = "cpu"
    header("Environment")
    print(f"Python     : {platform.python_version()} | {platform.platform()}\nExecutable : {sys.executable}")
    print(f"JAX_PLATFORMS={os.getenv('JAX_PLATFORMS')!r} | XLA_PYTHON_CLIENT_PREALLOCATE={os.getenv('XLA_PYTHON_CLIENT_PREALLOCATE')!r} "
          f"| XLA_PYTHON_CLIENT_MEM_FRACTION={os.getenv('XLA_PYTHON_CLIENT_MEM_FRACTION')!r}")
    print(f"NVIDIA     : driver={driver} | GPUs={gpus or 'none'}" if driver else "NVIDIA     : nvidia-smi not found (CPU-only is fine)")
    jax = section("jax", test_jax)
    section("numpy", lambda np: (bool(np.isfinite(np.sin(np.arange(10.0))).all()), "np.sin finite"))
    section("scipy", lambda sp: (lambda v: (abs(v - 2) < 1e-8, f"quad sin(0..pi) = {v:.12f}"))(
        __import__("scipy.integrate", fromlist=["quad"]).quad(math.sin, 0.0, math.pi)[0]))
    section("h5py", test_h5py)
    section("matplotlib", lambda mpl: (mpl.use("Agg") or True, "headless backend 'Agg'"))
    section("iminuit", test_iminuit)
    section("diffrax", lambda d: (True, "import ok"))
    section("exojax", lambda e: (True, "import ok (ExoJAX Voigt, the default line profile)"))
    section("fake_spectra", lambda f: (True, "import ok (optional: live comparisons, skewerjax.comparison.driver)"))
    sjx = section("skewerjax", test_skewerjax)
    header("Summary")
    print(f"Checks passed: {sum(RESULTS)}/{len(RESULTS)}")
    if all(RESULTS):
        print("SkewerJAX environment looks good")
        return 0
    print("Some optional components failed; the core can still work.")
    return 1 if jax is None or sjx is None else 0


if __name__ == "__main__":
    raise SystemExit(main())
