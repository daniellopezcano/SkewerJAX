#!/usr/bin/env bash
set -euo pipefail

ENV_NAME=${1:-skewerjax}

# 1) Create or update the conda env
if conda env list | grep -qE "^\s*${ENV_NAME}\s"; then
  echo "[INFO] Updating existing env '${ENV_NAME}'..."
  conda env update -n "${ENV_NAME}" -f env/environment.yml --prune
else
  echo "[INFO] Creating env '${ENV_NAME}'..."
  conda env create -n "${ENV_NAME}" -f env/environment.yml
fi

# 2) Install SkewerJAX (editable) + dev tools
conda run -n "${ENV_NAME}" python -m pip install --upgrade pip
conda run -n "${ENV_NAME}" python -m pip install -e .[dev]

# 3) Auto-install the right JAX (CUDA13 > CUDA12 > CPU), Linux-only for GPU wheels
conda run -n "${ENV_NAME}" python env/install_jax_auto.py

# 4) Sanity check
conda run -n "${ENV_NAME}" python - <<'PY'
import jax
print("JAX:", jax.__version__)
print("Devices:", jax.devices())
if not any(d.platform in ("gpu","cuda") for d in jax.devices()):
    print("NOTE: running on CPU. If you expected GPU, check NVIDIA driver versions.")
PY

echo
echo "Done. Activate with:"
echo "  conda activate ${ENV_NAME}"
