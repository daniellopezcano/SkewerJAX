#!/usr/bin/env bash
# Run a command with JAX preferring CUDA, avoiding library-path conflicts and
# disabling big upfront GPU memory grabs.
set -euo pipefail

# Prefer CUDA, allow CPU fallback
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda,cpu}"

# Avoid huge VRAM grabs; cap usage to 50% by default
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.50}"

# Ensure pip-provided CUDA libs are visible first
PYTHON_SITE_PATHS="$(
python - <<'PY'
import site, glob, os
paths=set()
for base in site.getsitepackages() + [site.getusersitepackages()]:
    for d in glob.glob(os.path.join(base, "nvidia", "*", "lib")):
        if os.path.isdir(d): paths.add(d)
print(":".join(sorted(paths)))
PY
)"
export LD_LIBRARY_PATH="${PYTHON_SITE_PATHS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

exec "$@"
