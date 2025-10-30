# SkewerJAX

Minimal bootstrapped repository for **SkewerJAX** — a JAX-based pipeline to synthesize Ly-α
and metal-line skewers and compute survey-facing statistics (e.g., P1D).

---

## Installation

You have two paths. The recommended one creates a fresh conda env and auto-installs **GPU JAX if supported** (else CPU). The second installs into your already-activated env.

### Option A — One command (Conda/Mamba; auto-detects GPU)
```bash
# from the repo root
bash env/create.sh            # creates env (Python 3.11), installs package, installs JAX GPU/CPU
conda activate skewerjax
```

What it does:
- Creates/updates the `skewerjax` conda env from `env/environment.yml` (**Python 3.11**).
- `pip install -e .[dev]` (editable install + dev tools).
- Runs `env/install_jax_auto.py`:
  - On Linux with a recent NVIDIA driver → installs `jax[cuda12]` or `jax[cuda13]` automatically.
  - Otherwise → installs CPU JAX.

### Option B — Use your current virtual environment (pip)
```bash
# activate your venv/conda env first
python -m pip install --upgrade pip
pip install -e .[dev]

# auto-pick the right JAX wheel (GPU if supported, else CPU)
python env/install_jax_auto.py
```

#### Manual override (optional)
If you *know* what you want:
```bash
# choose exactly one:
pip install -e .[cuda13]
pip install -e .[cuda12]
pip install -e .[cpu]
```

> Notes
> - NVIDIA GPU wheels are Linux-only. On macOS or Windows (without WSL+CUDA), you’ll get CPU JAX.
> - If you expected GPU but see CPU, check `nvidia-smi` and your driver version, or see **Notebook setup** below.

---

## Notebook setup (VS Code / Jupyter)

To ensure notebooks find the **pip-provided CUDA libraries** and avoid huge VRAM preallocation, create a dedicated kernel:

```bash
# inside the activated 'skewerjax' env
python -m pip install -U ipykernel
python scripts/setup_vscode_kernel.py
```

This creates a kernel named **`skewerjax (GPU)`** with:
- `JAX_PLATFORMS=cuda,cpu` (prefer CUDA, fall back to CPU),
- `XLA_PYTHON_CLIENT_PREALLOCATE=false` and `XLA_PYTHON_CLIENT_MEM_FRACTION=0.50` (friendlier GPU memory behavior),
- `LD_LIBRARY_PATH` pointing to the CUDA libs bundled with the JAX wheels (so cuSPARSE/cuDNN/NCCL are found).

**In VS Code:** open your notebook → kernel picker (top-right) → select **`skewerjax (GPU)`**.

*(CLI convenience)* If you run Python/Jupyter from a shell and want the same behavior, use:
```bash
scripts/with_jax_gpu.sh python -c "import jax; print(jax.devices())"
scripts/with_jax_gpu.sh jupyter lab
```

---

## Quick check

```bash
python - <<'PY'
import skewerjax as m, jax
print("SkewerJAX:", getattr(m, "__version__", "unknown"))
print("JAX:", jax.__version__)
print("Devices:", jax.devices())
if not any(d.platform in ("gpu","cuda") for d in jax.devices()):
    print("NOTE: running on CPU. In notebooks, pick the 'skewerjax (GPU)' kernel.")
PY
```

In a notebook (first cells):
```python
import os
print("JAX_PLATFORMS =", os.environ.get("JAX_PLATFORMS"))
print("XLA_PYTHON_CLIENT_PREALLOCATE =", os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"))
print("XLA_PYTHON_CLIENT_MEM_FRACTION =", os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION"))

import jax, jax.numpy as jnp
print("JAX:", jax.__version__)
print("Devices:", jax.devices())

dev = (jax.devices("cuda") or jax.devices("cpu"))[0]
@jax.jit
def mm(x, y): return x @ y + 1.0
a = jax.device_put(jnp.ones((1024,1024), jnp.float32), device=dev)
b = jax.device_put(jnp.ones((1024,1024), jnp.float32), device=dev)
out = mm(a, b).block_until_ready()
out.shape, float(out.mean())
```

---

## Layout
- `skewerjax/`: Python package (kept minimal on purpose)
  - `__init__.py`
  - `config/__init__.py`
  - `optical_depth/__init__.py`
- `configs/`: user-editable YAMLs (examples included)
- `notebooks/`: exploration (empty placeholder)
- `scripts/`: small ops helpers
  - `verify_install.py` — versions/devices + tiny JIT test
  - `with_jax_gpu.sh` — run commands with CUDA-first + friendly memory settings
  - `setup_vscode_kernel.py` — creates the **skewerjax (GPU)** kernelspec

> Intentionally omitted for now: tests, examples, data, benchmarks, docs.
> We’ll add them when features need them to keep things modular.

---

## License
This project is released under the MIT License (see `LICENSE`).

