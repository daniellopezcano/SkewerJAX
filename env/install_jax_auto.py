#!/usr/bin/env python
import os, shutil, subprocess, sys, platform

def pip_install(*args):
    cmd = [sys.executable, "-m", "pip", "install", "-U"] + list(args)
    print("[CMD]", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)

def nvidia_driver_major():
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
        ).strip().splitlines()[0]
        major = int(out.split(".")[0])
        return major
    except Exception:
        return None

def main():
    system = platform.system().lower()
    if system != "linux":
        print("[INFO] Non-Linux platform -> installing CPU JAX.")
        pip_install("jax")
        return

    drv_major = nvidia_driver_major()
    if drv_major is None:
        print("[INFO] No NVIDIA GPU detected -> installing CPU JAX.")
        pip_install("jax")
        return

    print(f"[INFO] NVIDIA driver major version: {drv_major}")
    # Per JAX docs: >=580 for CUDA 13; >=525 for CUDA 12
    if drv_major >= 580:
        print("[INFO] Installing JAX (CUDA 13 wheels).")
        pip_install('jax[cuda13]')
    elif drv_major >= 525:
        print("[INFO] Installing JAX (CUDA 12 wheels).")
        pip_install('jax[cuda12]')
    else:
        print("[WARN] Driver too old for CUDA 12/13 wheels -> installing CPU JAX.")
        pip_install("jax")

if __name__ == "__main__":
    main()
