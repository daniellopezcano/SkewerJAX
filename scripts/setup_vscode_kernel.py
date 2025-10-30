#!/usr/bin/env python
"""
Create/patch a Jupyter kernelspec named 'skewerjax-gpu' so notebooks:
- prefer CUDA (fallback CPU),
- avoid huge VRAM preallocation,
- and see the CUDA libs installed by pip (cuSPARSE/cuDNN/NCCL).
Run from the activated 'skewerjax' env.
"""
import json, os, site, glob, subprocess, sys
from jupyter_client.kernelspec import KernelSpecManager

NAME = "skewerjax-gpu"
DISPLAY = "skewerjax (GPU)"

ksm = KernelSpecManager()

def ensure_kernel():
    try:
        return ksm.get_kernel_spec(NAME)
    except Exception:
        subprocess.check_call([sys.executable, "-m", "ipykernel", "install", "--user",
                               "--name", NAME, "--display-name", DISPLAY])
        return ksm.get_kernel_spec(NAME)

spec = ensure_kernel()
cfg_path = os.path.join(spec.resource_dir, "kernel.json")

with open(cfg_path) as f:
    j = json.load(f)

# Collect pip-provided CUDA lib dirs inside THIS env
paths = set()
for sp in site.getsitepackages() + [site.getusersitepackages()]:
    for d in glob.glob(os.path.join(sp, "nvidia", "*", "lib")):
        if os.path.isdir(d):
            paths.add(d)
ld = ":".join(sorted(paths))

env = j.setdefault("env", {})
env["JAX_PLATFORMS"] = "cuda,cpu"
env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.50"
if ld:
    # Prepend pip CUDA libs; keep any existing LD_LIBRARY_PATH after them
    env["LD_LIBRARY_PATH"] = ld + (":" + env.get("LD_LIBRARY_PATH",""))

with open(cfg_path, "w") as f:
    json.dump(j, f, indent=2)

print("Patched:", cfg_path)
print("LD_LIBRARY_PATH first entry:", env.get("LD_LIBRARY_PATH","").split(":")[0] if env.get("LD_LIBRARY_PATH") else "<unset>")
