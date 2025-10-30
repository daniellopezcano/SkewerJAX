#!/usr/bin/env python
"""
SkewerJAX install verifier:
- Prints Python, NumPy, JAX, jaxlib versions
- Shows available JAX devices
- If a CUDA GPU is present, runs a tiny JIT on the GPU
- Otherwise, runs on CPU
Exits 0 on success, 1 on error.
"""
import sys, platform, subprocess
from importlib import metadata as ilmd

def v(pkg):
    try:
        return ilmd.version(pkg)
    except ilmd.PackageNotFoundError:
        return None

def print_kv(k, v):
    print(f"{k:<26} {v}")

def main():
    ok = True

    print("=== Environment ===")
    print_kv("Python", sys.version.split()[0])
    print_kv("Platform", platform.platform())

    # Core packages
    npv = v("numpy")
    jaxv = v("jax")
    jlv  = v("jaxlib")
    print("\n=== Packages ===")
    print_kv("NumPy", npv or "<missing>")
    print_kv("JAX", jaxv or "<missing>")
    print_kv("jaxlib", jlv or "<missing>")

    # CUDA extras (printed if present)
    print_kv("jax-cuda13-plugin", v("jax-cuda13-plugin") or "-")
    print_kv("jax-cuda12-plugin", v("jax-cuda12-plugin") or "-")
    print_kv("jax-cuda12-pjrt",  v("jax-cuda12-pjrt")  or "-")
    print_kv("nvidia-cudnn-cu12", v("nvidia-cudnn-cu12") or "-")
    print_kv("nvidia-nccl-cu12",  v("nvidia-nccl-cu12")  or "-")

    # Import JAX and query devices
    try:
        import jax, jax.numpy as jnp  # type: ignore
    except Exception as e:
        ok = False
        print("\n[ERROR] Failed to import JAX:", e)
        sys.exit(1)

    print("\n=== JAX Runtime ===")
    print_kv("JAX default backend", getattr(jax, "default_backend", lambda: "?")())
    devs = jax.devices()
    print_kv("Devices found", len(devs))
    for i, d in enumerate(devs):
        # device.platform is typically "cpu", "cuda", or "gpu"
        print(f"  - [{i}] {d.platform} {getattr(d, 'device_kind', getattr(d, 'kind', '?'))}")

    # Prefer CUDA if present (supports both 'cuda' and legacy 'gpu')
    cuda_devs = []
    for name in ("cuda","gpu"):
        try:
            cuda_devs = jax.devices(name)
            if cuda_devs:
                break
        except Exception:
            pass

    print("\n=== JIT Sanity Test ===")
    try:
        if cuda_devs:
            target = cuda_devs[0]
            print_kv("Target", f"CUDA on {getattr(target, 'device_kind', getattr(target, 'kind','GPU'))}")
            a = jax.device_put(jnp.ones((1024,1024), dtype=jnp.float32), device=target)
            b = jax.device_put(jnp.ones((1024,1024), dtype=jnp.float32), device=target)

            @jax.jit
            def f(x, y):
                return x @ y + 1.0

            out = f(a, b).block_until_ready()
            print_kv("Result shape", str(out.shape))
            print("[OK] JIT matmul ran on GPU (CUDA).")
        else:
            print_kv("Target", "CPU")
            a = jnp.ones((1024,1024), dtype=jnp.float32)
            b = jnp.ones((1024,1024), dtype=jnp.float32)

            @jax.jit
            def f(x, y):
                return x @ y + 1.0

            out = f(a, b).block_until_ready()
            print_kv("Result shape", str(out.shape))
            print("[OK] JIT matmul ran on CPU.")
    except Exception as e:
        ok = False
        print("[ERROR] JIT test failed:", e)

    # Optional: print nvidia-smi for visibility if available
    try:
        smi = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=False)
        if smi.returncode == 0 and smi.stdout.strip():
            print("\n=== nvidia-smi (devices) ===")
            print(smi.stdout.strip())
    except Exception:
        pass

    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
