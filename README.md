# SkewerJAX

Minimal bootstrapped repository for **SkewerJAX** — a JAX-based pipeline to synthesize Ly-α
and metal-line skewers and compute survey-facing statistics (e.g., P1D).

---

## Installation (simple)

```bash
# from the repo root
bash env/create.sh         # creates/updates env & installs JAX (GPU on Linux+NVIDIA, else CPU)
conda activate skewerjax
python scripts/verify_install.py
```

---

## Jupyter kernel (manual step)

Register a kernel after installing (environment activated):
```bash
python -m ipykernel install --user --name skewerjax --display-name "SkewerJAX (CUDA)"
```

---

## CUDA drivers (Linux/NVIDIA) — heads‑up

For **GPU JAX** you need both the correct **JAX wheels** *and* a new enough **NVIDIA driver**:
- **`jax[cuda13]`** expects **driver ≥ 580.xx**.
- **`jax[cuda12]`** expects **driver ≥ 525.xx**.

### Debian 12 quick path
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

# Under developement
- README.md
- LICENSE
- MANIFEST.in
- pyproject.toml
- .gitignore
- configs/
  - global_setup.yaml
- notebooks/
  - tests.ipynb
- skewerjax/
  - __init__.py
  - constants/
    - __init__.py
    - lines.py
    - physical.py
  - optical_depth/
    - __init__.py
  - config/
    - __init__.py
- env/
  - create.sh
  - environment.yml
- scripts/
  - verify_install.py



---

## License
This project is released under the MIT License (see `LICENSE`).
