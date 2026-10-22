"""The toy-input validation scripts (``scripts/validation``, ``--quick``), each in its own process: the suite reuses
their checks instead of duplicating them. The scripts that need TNG50-4 or fake_spectra run with SKEWERJAX_SLOW=1."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

VALIDATION = Path(__file__).resolve().parents[1] / "scripts" / "validation"
FAST = ["check_cosmology.py", "check_voigt.py", "check_line_frames.py", "check_optical_depth_analytic.py",
        "check_summary_statistics.py", "check_jax.py"]
SLOW = ["check_tng_fields.py", "check_tng_optical_depth.py", "check_fake_spectra.py"]


def run_check(script: str):
    p = subprocess.run([sys.executable, str(VALIDATION / script), "--quick"], capture_output=True, text=True,
                       env=dict(os.environ, JAX_PLATFORMS="cpu"))
    assert p.returncode == 0, f"{script} failed:\n" + "\n".join(l for l in p.stdout.splitlines() if "FAIL" in l) + p.stderr[-2000:]


@pytest.mark.parametrize("script", FAST)
def test_validation_script_quick(script):
    run_check(script)


@pytest.mark.skipif(os.environ.get("SKEWERJAX_SLOW") != "1" or not (Path(os.environ.get("SKEWERJAX_DATA", "/home/STORAGE")) / "TNG50-4").exists(),
                    reason="needs TNG50-4 (and fake_spectra); set SKEWERJAX_SLOW=1")
@pytest.mark.parametrize("script", SLOW)
def test_validation_script_simulation(script):
    run_check(script)
