#!/usr/bin/env python
"""Install check: the dependencies import, JAX runs a jitted kernel on every platform it sees, and the package and its
subpackages import. Exit status 1 if a required component fails; the optional ones (h5py, fake_spectra, a GPU) are
reported only.

    python scripts/checks/check_install.py
"""

import importlib
import sys
import time

REQUIRED = ("jax", "numpy", "scipy", "matplotlib", "diffrax", "exojax", "numba", "skewerjax")
OPTIONAL = ("h5py", "fake_spectra", "psutil")


def main() -> int:
    failed = []
    for name in REQUIRED + OPTIONAL:
        try:
            mod = importlib.import_module(name)
            print(f"[OK]   {name} {getattr(mod, '__version__', '')}")
        except Exception as e:  # noqa: BLE001
            print(f"[{'FAIL' if name in REQUIRED else 'skip'}] {name}: {type(e).__name__}: {e}")
            if name in REQUIRED:
                failed.append(name)
    if failed:
        return 1
    import jax
    import jax.numpy as jnp

    for sub in ("core", "simulations", "summary_stats", "comparison", "utils.plotting", "config"):
        importlib.import_module(f"skewerjax.{sub}")
    print("[OK]   skewerjax subpackages: core, simulations, summary_stats, comparison, utils.plotting, config")
    for platform in ("cpu", "gpu"):
        try:
            dev = jax.devices(platform)[0]
        except RuntimeError:
            print(f"[skip] no {platform.upper()} device visible to JAX")
            continue
        x = jax.device_put(jnp.ones((256, 256), jnp.float32), dev)
        f = jax.jit(lambda a: (a @ a).sum())
        f(x).block_until_ready()
        t = time.perf_counter()
        ok = float(f(x).block_until_ready()) == 256.0**3
        print(f"[{'OK' if ok else 'FAIL'}]   jitted matmul on {dev} in {(time.perf_counter() - t) * 1e3:.2f} ms")
        if not ok:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
