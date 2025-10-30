#!/usr/bin/env python
"""
verify_stack.py — SkewerJAX environment smoke tests
"""
from __future__ import annotations
import os, sys, time, math, tempfile, platform
from dataclasses import dataclass

def header(t): print("\n" + "="*len(t) + f"\n{t}\n" + "="*len(t))

def status(ok: bool, msg: str):
    tag = "[OK]" if ok else "[FAIL]"
    print(f"{tag} {msg}")

def try_import(name: str):
    try:
        mod = __import__(name)
        return True, mod, ""
    except Exception as e:
        return False, None, f"{type(e).__name__}: {e}"

def main():
    header("Environment")
    print(f"Python    : {platform.python_version()}")
    print(f"Platform  : {platform.platform()}")
    print(f"Executable: {sys.executable}")
    print(f"CUDA env  : XLA_PYTHON_CLIENT_PREALLOCATE={os.getenv('XLA_PYTHON_CLIENT_PREALLOCATE')} "
          f"| XLA_PYTHON_CLIENT_MEM_FRACTION={os.getenv('XLA_PYTHON_CLIENT_MEM_FRACTION')}")

    # JAX
    header("JAX")
    ok, jax, err = try_import("jax")
    if not ok:
        status(False, f"import jax -> {err}")
        print("Tip: pip install -U jax  (or jax[cuda13]/jax[cuda12] on Linux+NVIDIA)")
    else:
        import jax.numpy as jnp
        print(f"JAX       : {jax.__version__}")
        try:
            backend = getattr(jax, "default_backend", lambda: "?")()
        except Exception:
            backend = "?"
        print(f"Backend   : {backend}")
        print(f"Devices   : {jax.devices()}")
        try:
            key = jax.random.key(0)
            a = jax.random.normal(key, (512, 512), dtype=jnp.float32)
            b = jax.random.normal(key, (512, 512), dtype=jnp.float32)
            @jax.jit
            def mm(x, y): return x @ y
            _ = mm(a, b).block_until_ready()  # compile+warmup
            t0 = time.time()
            c = mm(a, b).block_until_ready()
            dt = (time.time() - t0) * 1e3
            dev = getattr(c, "device", None)
            status(True, f"JIT matmul on {dev.platform.upper() if dev else 'unknown'} in {dt:.1f} ms")
        except Exception as e:
            status(False, f"JAX JIT test -> {type(e).__name__}: {e}")

    # NumPy
    header("NumPy")
    ok, np, err = try_import("numpy")
    if not ok:
        status(False, f"import numpy -> {err}")
    else:
        print(f"NumPy     : {np.__version__}")
        try:
            x = np.arange(10, dtype=np.float64)
            y = np.sin(x)
            status(np.isfinite(y).all(), "np.sin computed finite values")
        except Exception as e:
            status(False, f"NumPy ops -> {type(e).__name__}: {e}")

    # SciPy
    header("SciPy")
    ok, scipy, err = try_import("scipy")
    if not ok:
        status(False, f"import scipy -> {err}")
    else:
        print(f"SciPy     : {scipy.__version__}")
        try:
            from scipy import integrate
            val, err_est = integrate.quad(lambda t: math.sin(t), 0.0, math.pi)
            status(abs(val - 2.0) < 1e-8, f"integrate.quad sin(0..π)≈2 => {val:.12f}")
        except Exception as e:
            status(False, f"SciPy test -> {type(e).__name__}: {e}")

    # h5py
    header("h5py")
    ok, h5py, err = try_import("h5py")
    if not ok:
        status(False, f"import h5py -> {err}")
    else:
        print(f"h5py      : {h5py.__version__}")
        try:
            import numpy as _np
            fd, path = tempfile.mkstemp(prefix="skewerjax_", suffix=".h5")
            os.close(fd)
            with h5py.File(path, "w") as f:
                f["dset"] = _np.arange(5, dtype=_np.int32)
            with h5py.File(path, "r") as f:
                data = f["dset"][...]
            os.remove(path)
            status((data == _np.arange(5, dtype=_np.int32)).all(), "HDF5 write/read round-trip")
        except Exception as e:
            status(False, f"h5py test -> {type(e).__name__}: {e}")

    # matplotlib
    header("matplotlib")
    ok, mpl, err = try_import("matplotlib")
    if not ok:
        status(False, f"import matplotlib -> {err}")
    else:
        print(f"matplotlib: {mpl.__version__}")
        try:
            mpl.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as _np
            fig = plt.figure()
            plt.plot(_np.linspace(0, 1, 5), _np.linspace(0, 1, 5)**2)
            plt.close(fig)
            status(os.path.exists(out), f"Saved plot: ./{out}")
        except Exception as e:
            status(False, f"matplotlib test -> {type(e).__name__}: {e}")

    # iminuit
    header("iminuit")
    ok, _iminuit, err = try_import("iminuit")
    if not ok:
        status(False, f"import iminuit -> {err}")
    else:
        from iminuit import Minuit
        print(f"iminuit   : {_iminuit.__version__}")
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
            status(abs(m_fit - m_true) < 0.1 and abs(c_fit - c_true) < 0.1,
                   f"Linear fit recovered m≈{m_fit:.3f}, c≈{c_fit:.3f}")
        except Exception as e:
            status(False, f"iminuit test -> {type(e).__name__}: {e}")

    # diffrax (optional)
    header("diffrax (optional)")
    ok, diffrax, err = try_import("diffrax")
    if not ok:
        status(False, f"import diffrax -> {err}")
        print("Tip: pip install diffrax  (ensure compatibility with your JAX version)")
    else:
        print(f"diffrax   : {getattr(diffrax, '__version__', 'unknown')}")

    print("\nAll checks done.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())