"""Shared setup of the test suite: CPU, float64 (the tolerances are float64 ones), quiet JAX plugin discovery."""
import logging
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
logging.getLogger("jax._src.xla_bridge").setLevel(logging.CRITICAL)
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
REPO = Path(__file__).resolve().parents[1]
