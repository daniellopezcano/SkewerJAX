#!/usr/bin/env python
"""
verify_stack.py — SkewerJAX environment smoke tests
This script is safe on CPU-only or GPU machines. It auto-forces CPU for this
process if no NVIDIA GPU is detected, to avoid noisy CUDA plugin logs.
"""
from __future__ import annotations

import os
import sys
import time
import math
import tempfile
import platform
import shutil
import subprocess
from dataclasses import dataclass

# ----------------------------- pretty printing --------------------------------
def header(t: str) -> None:
    print("\n" + "=" * len(t) + f"\n{t}\n" + "=" * len(t))

# We'll keep counters for a summary at the end
class Counter:
    ok = 0
    fail = 0

def report(ok: bool, msg: str) -> None:
    tag = "[OK]" if ok else "[FAIL]"
    if ok: Counter.ok += 1
    else:  Counter.fail += 1
    print(f"{tag} {msg}")

def try_import(name: str):
    try:
        mod = __import__(name)
        return True, mod, ""
    except Exception as e:
        return False, None, f"{type(e).__name__}: {e}"

# ----------------------------- helpers ----------------------------------------
@dataclass
class NvidiaInfo:
    present: bool
    driver: str | None
    gpus: list[str]

def detect_nvidia() -> NvidiaInfo:
    """Check for nvidia-smi and parse driver / GPUs if available."""
    if shutil.which("nvidia-smi") is None:
        return NvidiaInfo(False, None, [])
    try:
        # Driver
        drv = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, check=True
        ).stdout.strip().splitlines()
        driver = drv[0].strip() if drv else "unknown"
        # GPU names
        names = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, check=True
        ).stdout.strip().splitlines()
        gpus = [n.strip() for n in names if n.strip()]
        return NvidiaInfo(True, driver, gpus)
    except Exception:
        # nvidia-smi exists but something's off (permissions/X not started/etc.)
        return NvidiaInfo(True, None, [])

def maybe_force_cpu_for_this_process(nv: NvidiaInfo) -> bool:
    """
    To keep logs clean on CPU-only hosts where CUDA wheels might be installed,
    force JAX to CPU for THIS process if we detect no NVIDIA GPUs and the user
    hasn't already chosen platforms.
    """
    if "JAX_PLATFORMS" in os.environ:
        return False
    if not nv.present or not nv.gpus:
        os.environ["JAX_PLATFORMS"] = "cpu"
        return True
    return False

# -------------------------------- main ----------------------------------------
def main() -> int:
    # Detect NVIDIA & optionally force CPU for this process
    nv = detect_nvidia()
    forced_cpu = maybe_force_cpu_for_this_process(nv)

    # ----- Environment -----
    header("Environment")
    print(f"Python     : {platform.python_version()}")
    print(f"Platform   : {platform.platform()}")
    print(f"Executable : {sys.executable}")
    print(f"JAX_PLATFORMS={os.getenv('JAX_PLATFORMS')!r} "
          f"| XLA_PYTHON_CLIENT_PREALLOCATE={os.getenv('XLA_PYTHON_CLIENT_PREALLOCATE')!r} "
          f"| XLA_PYTHON_CLIENT_MEM_FRACTION={os.getenv('XLA_PYTHON_CLIENT_MEM_FRACTION')!r}")
    if nv.present:
        print(f"NVIDIA     : driver={nv.driver or 'unknown'} | GPUs={nv.gpus or 'none'}")
    else:
        print("NVIDIA     : nvidia-smi not found (CPU-only host is okay)")
    if forced_cpu:
        print("Note       : Auto-set JAX_PLATFORMS='cpu' for this process (no NVIDIA GPU detected).")

    # ----- SkewerJAX -----
    header("SkewerJAX package")
    ok, mod, err = try_import("skewerjax")
    if not ok:
        report(False, f"import skewerjax -> {err}")
        print("Tip: from repo root, run `pip install -e .[dev]` inside your env.")
    else:
        ver = getattr(mod, "__version__", "unknown")
        report(True, f"SkewerJAX imported (version {ver})")

    # ----- JAX & devices -----
    header("JAX")
    ok, jax, err = try_import("jax")
    if not ok:
        report(False, f"import jax -> {err}")
        print("Tip: CPU-only: `pip install -U jax`; "
              "Linux+NVIDIA: `pip install -U 'jax[cuda13]'` or `'jax[cuda12]'`.")
        backend = "unknown"
        devices = []
    else:
        import jax.numpy as jnp  # noqa: F401
        print(f"JAX        : {jax.__version__}")
        try:
            backend = getattr(jax, "default_backend", lambda: "?")()
        except Exception:
            backend = "?"
        try:
            devices = list(jax.devices())
        except Exception:
            devices = []
        print(f"Backend    : {backend}")
        print(f"All devices: {devices}")

        # Plugin wheels presence (informational)
        cuda13_present, _, _ = try_import("jax_plugins.xla_cuda13")
        cuda12_present, _, _ = try_import("jax_plugins.xla_cuda12")
        msg = []
        if cuda13_present: msg.append("CUDA13 plugin")
        if cuda12_present: msg.append("CUDA12 plugin")
        report(True, f"JAX plugins detected: {', '.join(msg) if msg else 'none'}")

        # Choose best available device (prefer GPU, else CPU)
        chosen = None
        gpu_devs = []
        try:
            gpu_devs = jax.devices("gpu")  # may raise if backend not registered
        except Exception:
            gpu_devs = []
        if gpu_devs:
            chosen = gpu_devs[0]
        elif devices:
            chosen = devices[0]

        if chosen is None:
            report(False, "No JAX devices visible")
        else:
            report(True, f"Selected device: {chosen}")
            # Tiny JIT sanity test on chosen device
            try:
                import jax.numpy as jnp
                import numpy as _np

                N = 1024 if getattr(chosen, "platform", "") == "gpu" else 512
                key = jax.random.key(0)
                a = jax.random.normal(key, (N, N), dtype=jnp.float32)
                b = jax.random.normal(key, (N, N), dtype=jnp.float32)
                # Put on chosen device to avoid host<->device copies
                a = jax.device_put(a, device=chosen)
                b = jax.device_put(b, device=chosen)

                @jax.jit
                def mm(x, y): return x @ y

                _ = mm(a, b).block_until_ready()  # compile + warmup
                t0 = time.time()
                c = mm(a, b).block_until_ready()
                dt_ms = (time.time() - t0) * 1e3
                
                dev_attr = getattr(c, "device", None)
                dev = dev_attr() if callable(dev_attr) else dev_attr  # may be None on some builds
                plat = getattr((dev or chosen), "platform", "unknown").upper()
                report(True, f"JIT matmul ran on {plat} in {dt_ms:.1f} ms (shape {c.shape})")

            except Exception as e:
                report(False, f"JAX JIT test -> {type(e).__name__}: {e}")

            # Friendly note if CUDA plugin is present but no GPU found
            if (cuda12_present or cuda13_present) and not gpu_devs:
                print("Info: CUDA plugin wheels are installed but no CUDA-capable GPU is available.")
                print("      This is fine on CPU-only hosts. To silence CUDA logs in general, you can:")
                print("      - Uninstall CUDA wheels: pip uninstall -y jax-cuda13-plugin jax-cuda13-pjrt jax-cuda12-plugin jax-cuda12-pjrt")
                print("      - Or set JAX_PLATFORMS=cpu when launching notebooks.")

    # ----- NumPy -----
    header("NumPy")
    ok, np, err = try_import("numpy")
    if not ok:
        report(False, f"import numpy -> {err}")
    else:
        print(f"NumPy      : {np.__version__}")
        try:
            x = np.arange(10, dtype=np.float64)
            y = np.sin(x)
            report(np.isfinite(y).all(), "np.sin computed finite values")
        except Exception as e:
            report(False, f"NumPy ops -> {type(e).__name__}: {e}")

    # ----- SciPy (optional but common) -----
    header("SciPy")
    ok, scipy, err = try_import("scipy")
    if not ok:
        report(False, f"import scipy -> {err}")
    else:
        print(f"SciPy      : {scipy.__version__}")
        try:
            from scipy import integrate
            val, err_est = integrate.quad(lambda t: math.sin(t), 0.0, math.pi)
            report(abs(val - 2.0) < 1e-8, f"integrate.quad sin(0..π)≈2 => {val:.12f}")
        except Exception as e:
            report(False, f"SciPy test -> {type(e).__name__}: {e}")

    # ----- h5py (optional) -----
    header("h5py")
    ok, h5py, err = try_import("h5py")
    if not ok:
        report(False, f"import h5py -> {err}")
    else:
        print(f"h5py       : {h5py.__version__}")
        try:
            import numpy as _np
            fd, path = tempfile.mkstemp(prefix="skewerjax_", suffix=".h5")
            os.close(fd)
            with h5py.File(path, "w") as f:
                f["dset"] = _np.arange(5, dtype=_np.int32)
            with h5py.File(path, "r") as f:
                data = f["dset"][...]
            os.remove(path)
            report((data == _np.arange(5, dtype=_np.int32)).all(), "HDF5 write/read round-trip")
        except Exception as e:
            report(False, f"h5py test -> {type(e).__name__}: {e}")

    # ----- matplotlib (optional) -----
    header("matplotlib")
    ok, mpl, err = try_import("matplotlib")
    if not ok:
        report(False, f"import matplotlib -> {err}")
    else:
        print(f"matplotlib : {mpl.__version__}")
        try:
            mpl.use("Agg")
            import matplotlib.pyplot as plt  # noqa: F401
            report(True, "Backend 'Agg' usable (headless plotting ok)")
        except Exception as e:
            report(False, f"matplotlib test -> {type(e).__name__}: {e}")

    # ----- iminuit (optional) -----
    header("iminuit")
    ok, _iminuit, err = try_import("iminuit")
    if not ok:
        report(False, f"import iminuit -> {err}")
    else:
        from iminuit import Minuit
        print(f"iminuit    : {_iminuit.__version__}")
        try:
            import numpy as _np
            rng = _np.random.default_rng(0)
            x = _np.linspace(-1, 1, 50)
            m_true, c_true, sigma = 2.5, -0.4, 0.05
            y = m_true * x + c_true + rng.normal(0, sigma, size=x.size)

            def chi2(m, c):
                y_model = m * x + c
                return _np.sum(((y - y_model) / sigma) ** 2)

            m = Minuit(chi2, m=1.0, c=0.0)
            m.errordef = Minuit.LEAST_SQUARES
            m.migrad(); m.hesse()
            m_fit, c_fit = m.values["m"], m.values["c"]
            report(abs(m_fit - m_true) < 0.1 and abs(c_fit - c_true) < 0.1,
                   f"Linear fit recovered m≈{m_fit:.3f}, c≈{c_fit:.3f}")
        except Exception as e:
            report(False, f"iminuit test -> {type(e).__name__}: {e}")

    # ----- diffrax (optional) -----
    header("diffrax (optional)")
    ok, diffrax, err = try_import("diffrax")
    if not ok:
        report(False, f"import diffrax -> {err}")
    else:
        print(f"diffrax    : {getattr(diffrax, '__version__', 'unknown')}")
        report(True, "diffrax import ok")

    # ----- Summary -----
    header("Summary")
    total = Counter.ok + Counter.fail
    print(f"Checks passed: {Counter.ok}/{total} "
          f"(failures: {Counter.fail})")
    if Counter.fail == 0:
        print("SkewerJAX environment looks good ✅")
        return 0
    else:
        print("Some optional components failed. Core CPU/GPU functionality can still work.")
        # Non-zero only if core components missing (jax or skewerjax)
        core_missing = False
        # Quick re-check for core bits
        core_missing |= not try_import("skewerjax")[0]
        core_missing |= not try_import("jax")[0]
        return 1 if core_missing else 0

if __name__ == "__main__":
    raise SystemExit(main())
