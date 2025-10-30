#!/usr/bin/env bash
set -euo pipefail

# Usage: bash env/create.sh [ENV_NAME]
ENV_NAME=${1:-skewerjax}

# Resolve repo root (this script lives in env/)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "[INFO] Repo root: ${REPO_ROOT}"

# ---------------------------------------------------------------------------
# 0) Conda config: conda-forge + strict priority (ok to be no-op)
# ---------------------------------------------------------------------------
conda config --add channels conda-forge || true
conda config --set channel_priority strict

# ---------------------------------------------------------------------------
# 1) Create/update the environment from environment.yml
# ---------------------------------------------------------------------------
if conda env list | grep -qE "^\s*${ENV_NAME}\s"; then
  echo "[INFO] Updating existing env '${ENV_NAME}'..."
  conda env update -n "${ENV_NAME}" -f "${REPO_ROOT}/env/environment.yml" --prune
else
  echo "[INFO] Creating env '${ENV_NAME}'..."
  conda env create -n "${ENV_NAME}" -f "${REPO_ROOT}/env/environment.yml"
fi

# Keep pip current, ensure local package is installed (editable)
conda run -n "${ENV_NAME}" python -m pip install --upgrade pip
conda run -n "${ENV_NAME}" python -m pip install -e "${REPO_ROOT}[dev]"

# ---------------------------------------------------------------------------
# 2) JAX install (GPU on Linux+NVIDIA, else CPU) — official extras
# ---------------------------------------------------------------------------
if [[ "$(uname -s)" == "Linux" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  echo "[INFO] Installing JAX (CUDA, preferring CUDA13)…"
  conda run -n "${ENV_NAME}" python -m pip install -U "jax[cuda13]" ||   conda run -n "${ENV_NAME}" python -m pip install -U "jax[cuda12]"
else
  echo "[INFO] Installing JAX (CPU or platform default)…"
  conda run -n "${ENV_NAME}" python -m pip install -U jax
fi

# Install diffrax
echo "[INFO] Installing diffrax…"
conda run -n "${ENV_NAME}" python -m pip install -U diffrax

# ---------------------------------------------------------------------------
# 3) Sanity check
# ---------------------------------------------------------------------------
echo "[INFO] Sanity check (JAX versions & devices)…"
conda run -n "${ENV_NAME}" python - <<'PY'
import jax
print("jax =", jax.__version__)
print("devices:", jax.devices())
PY

echo
echo "Done. Next steps:"
echo "  - conda activate ${ENV_NAME}"
echo "  - Register a Jupyter kernel (see README):"
echo "      python -m ipykernel install --user --name skewerjax --display-name 'SkewerJAX (CUDA)'"
