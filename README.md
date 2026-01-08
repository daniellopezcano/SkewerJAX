# SkewerJAX

Minimal bootstrapped repository for **SkewerJAX** — a JAX-based pipeline to synthesize Ly-α
and metal-line skewers and compute survey-facing statistics (e.g., P1D).

---

## Installation (CUDA 12 • JAX 0.6.2 • NumPy < 2)

This repo targets a **known-good GPU stack**:

- `jax==0.6.2`, `jaxlib==0.6.2`, `jax-cuda12-plugin==0.6.2`
- `NumPy < 2` (required by `exojax==2.2.2`)
- CUDA 12 runtime libs shipped via pip wheels (no system CUDA toolkit needed)
- NVIDIA **driver ≥ 525** (check with `nvidia-smi`)

> **Do not mix** `conda-forge` JAX with the pip CUDA12 plugin. We install JAX **only** via pip.

### 1) Create and activate a conda environment

```bash
# Add conda-forge with strict priority (safe to re-run)
conda config --add channels conda-forge || true
conda config --set channel_priority strict

# Create the environment with the GPU-agnostic scientific stack
conda create -y -n skewerjax \
  python=3.11 "numpy<2" "scipy>=1.11" "h5py>=3.10" "matplotlib>=3.7" \
  "ipympl>=0.9" "ipywidgets>=8.1" "iminuit>=2.27" "ipdb>=0.13" "ipykernel>=6.29" pip

# Activate it
conda activate skewerjax

### 2) Install the pinned CUDA12 JAX stack + libraries (pip)
python -m pip install -U pip wheel

python -m pip install --no-cache-dir -U \
  "jax==0.6.2" "jaxlib==0.6.2" "jax-cuda12-plugin==0.6.2" \
  "nvidia-cuda-runtime-cu12>=12.1" "nvidia-cublas-cu12>=12.1" \
  "nvidia-cusparse-cu12>=12.1" "nvidia-cufft-cu12>=11.0" \
  "nvidia-cusolver-cu12>=11.4" "nvidia-cudnn-cu12==9.12.0.46" \
  "nvidia-nccl-cu12>=2.18"

### 3) Install exojax and diffrax (GPU-compatible with this stack)
python -m pip install --no-cache-dir -U "exojax==2.2.2" "diffrax>=0.6,<0.8"

### 4) Install SkewerJAX (editable)

From the repository root:

pip install -e .

### 5) Make CUDA12 wheel libraries visible (required once per shell)

JAX’s CUDA plugin must see the NVIDIA CUDA12 wheel libraries. Export them to LD_LIBRARY_PATH:

export LD_LIBRARY_PATH="$(
python - <<'PY'
import site, glob, os
sp = site.getsitepackages()[0]
dirs = []
for pkg in ("nvidia/cusparse","nvidia/cublas","nvidia/cudnn",
            "nvidia/cufft","nvidia/cusolver","nvidia/cuda_runtime",
            "nvidia/nccl","nvidia/nvjitlink"):
    dirs += glob.glob(os.path.join(sp, pkg, "lib*"))
print(":".join(dirs))
PY
):${LD_LIBRARY_PATH:-}"

### Make it persistent (optional): create env-activation hooks so this is automatic:

mkdir -p "$CONDA_PREFIX/etc/conda/activate.d" "$CONDA_PREFIX/etc/conda/deactivate.d"
python - <<'PY'
import site, glob, os
prefix = os.environ["CONDA_PREFIX"]; sp = site.getsitepackages()[0]
dirs=[]; 
for p in ("nvidia/cusparse","nvidia/cublas","nvidia/cudnn","nvidia/cufft",
          "nvidia/cusolver","nvidia/cuda_runtime","nvidia/nccl","nvidia/nvjitlink"):
    dirs += glob.glob(os.path.join(sp,p,"lib*"))
LD=":".join(dirs)
open(os.path.join(prefix,"etc/conda/activate.d","skewerjax_cuda_libs.sh"),"w").write(
  "#!/usr/bin/env bash\nexport _SKEWERJAX_OLD_LD_LIBRARY_PATH=\"${LD_LIBRARY_PATH:-}\"\nexport LD_LIBRARY_PATH=\"%s:$LD_LIBRARY_PATH\"\n" % LD)
open(os.path.join(prefix,"etc/conda/deactivate.d","skewerjax_cuda_libs.sh"),"w").write(
  "#!/usr/bin/env bash\nexport LD_LIBRARY_PATH=\"${_SKEWERJAX_OLD_LD_LIBRARY_PATH}\"\nunset _SKEWERJAX_OLD_LD_LIBRARY_PATH\n")
PY
# Re-activate to apply:
conda deactivate && conda activate skewerjax

### 6) (Optional) Register a Jupyter kernel
python -m ipykernel install --user --name skewerjax --display-name "SkewerJAX (CUDA)"

### 7) Verify GPU is detected
python - <<'PY'
import jax, importlib, numpy as np
print("JAX:", jax.__version__)
print("Devices:", jax.devices())
assert any(d.platform=="gpu" for d in jax.devices()), "No GPU detected by JAX."
print("exojax:", importlib.import_module("exojax").__version__)
print("diffrax:", importlib.import_module("diffrax").__version__)
print("✅ SkewerJAX GPU stack OK")
PY



---

## Layout

# Under developement
- README.md
- LICENSE
- MANIFEST.in
- pyproject.toml
- .gitignore
...



---

## License
This project is released under the MIT License (see `LICENSE`).
