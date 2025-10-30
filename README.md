# SkewerJAX

Minimal bootstrapped repository for **SkewerJAX** — a JAX-based pipeline to synthesize Ly-α
and metal-line skewers and compute survey-facing statistics (e.g., P1D).

---

## Installation

You’ve got two simple paths. The first one (recommended) creates a fresh Conda/Mamba env and
auto-installs **GPU JAX if supported** (else CPU). The second installs into whatever virtual
environment you already activated.

### Option A — One command (Conda/Mamba, auto-detects GPU)
```bash
# from the repo root
bash env/create.sh            # creates/updates env, installs package, picks JAX GPU/CPU
conda activate skewerjax
```

What it does:
- Creates/updates the `skewerjax` Conda env from `env/environment.yml`.
- `pip install -e .[dev]` (editable install + dev tools).
- Runs `env/install_jax_auto.py`:
  - If Linux + NVIDIA driver is recent → installs `jax[cuda13]` (or `jax[cuda12]`).
  - Otherwise → installs CPU `jax`.

### Option B — Use your current virtual environment (pip)
```bash
# activate your venv/conda env first
python -m pip install --upgrade pip
pip install -e .[dev]

# auto-pick the right JAX wheel (GPU if supported, else CPU)
python env/install_jax_auto.py
```

#### Manual override (optional)
If you *know* what you want, install an extra directly:
```bash
# choose exactly one:
pip install -e .[cuda13]
pip install -e .[cuda12]
pip install -e .[cpu]
```

> Notes:
> - NVIDIA GPU wheels are Linux-only. On macOS or Windows without WSL+CUDA, the script will install CPU JAX.
> - If GPU isn’t detected but you expected it, check `nvidia-smi` and your driver versions.

---

## Quick check
```bash
python - <<'PY'
import skewerjax as m, jax
print("SkewerJAX:", getattr(m, "__version__", "unknown"))
print("JAX:", jax.__version__)
print("Devices:", jax.devices())
if not any(d.platform in ("gpu","cuda") for d in jax.devices()):
    print("NOTE: running on CPU.")
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

> Intentionally omitted for now: tests, examples, data, scripts, benchmarks, docs.
> We’ll add them when the first features need them to keep things modular.

---

## License
This project is released under the MIT License (see `LICENSE`).

