# SkewerJAX

Minimal bootstrapped repository for **SkewerJAX** — a JAX-based pipeline to synthesize Ly-α
and metal-line skewers and compute survey-facing statistics (e.g., P1D).

---

## Installation (simple)

You can use either **conda/mamba** or **pip**. In both paths, we install the package normally and then install **JAX** with GPU support when available.

### Option A — Conda/Mamba
```bash
# from the repo root
bash env/create.sh         # creates/updates env & installs JAX (GPU on Linux+NVIDIA, else CPU)
conda activate skewerjax
```

### Option B — Pip (your current venv)
```bash
python -m pip install --upgrade pip
pip install -e .[dev]

# JAX install:
# Linux + NVIDIA (CUDA 13 preferred; fallback to CUDA 12)
if command -v nvidia-smi >/dev/null 2>&1 && [ "$(uname -s)" = "Linux" ]; then
  pip install -U "jax[cuda13]" || pip install -U "jax[cuda12]"
else
  # macOS (CPU/Metal) or Windows → regular JAX
  pip install -U jax
fi
```

> Notes: CUDA GPU wheels are **Linux-only**; macOS/Windows will use CPU by default. Apple Silicon users may get Metal support via the regular `jax` package depending on local setup.

---

## Jupyter kernel (manual step)

Register a kernel after installing (environment activated):
```bash
python -m ipykernel install --user --name skewerjax --display-name "SkewerJAX (CUDA)"
```

---

## Quick check
```bash
python - <<'PY'
import jax, skewerjax as m
print("SkewerJAX:", getattr(m, "__version__", "unknown"))
print("JAX:", jax.__version__)
print("Devices:", jax.devices())
PY
```

---

## CUDA drivers (Linux/NVIDIA) — heads‑up

For **GPU JAX** you need both the correct **JAX wheels** *and* a new enough **NVIDIA driver**:
- **`jax[cuda13]`** expects **driver ≥ 580.xx**.
- **`jax[cuda12]`** expects **driver ≥ 525.xx**.

### Debian 12 quick path (what we use)
Install the official NVIDIA driver from the CUDA repo (gives you 580+):
```bash
# add NVIDIA’s repo key and refresh
wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update

# install the driver meta-package (brings nvidia-smi)
sudo apt install -y cuda-drivers

# reboot, then verify
sudo reboot
# after reboot:
nvidia-smi   # should show Driver Version 580.xx or newer
```
If you prefer **Debian backports** instead of the NVIDIA repo, you’ll typically get a **535.xx** driver on bookworm; in that case install **`jax[cuda12]`**.

> **Secure Boot:** If enabled, the kernel may block unsigned DKMS modules. NVIDIA/DKMS can sign with a local MOK; you might be prompted on reboot to enroll it. If the driver doesn’t load, check `mokutil --sb-state` and enroll/disable as needed.

### Troubleshooting
- **`nvidia-smi` missing** → driver not installed. Install via the NVIDIA repo (`cuda-drivers`) or Debian backports.
- **JAX sees GPU but cuDNN errors** → driver/runtime mismatch. Either upgrade to R580+ and use `jax[cuda13]`, or install `jax[cuda12]` with R525+.
- **Still CPU only** → confirm CUDA path with a minimal test:
  ```bash
  python - <<'PY'
  import jax; print(jax.devices())
  PY
  ```

---

## Layout
- `skewerjax/`: Python package (kept minimal on purpose)
  - `__init__.py`
  - `config/__init__.py`
  - `optical_depth/__init__.py`
- `configs/`: user-editable YAMLs (examples included)
- `notebooks/`: exploration (empty placeholder)

> Intentionally omitted (for now): tests, examples, data, benchmarks, docs.

---

## License
This project is released under the MIT License (see `LICENSE`).
