#!/usr/bin/env python
"""Regenerate the reference arrays of the regression tests (``tests/regression/data/*.npz``) with the current code.

    python scripts/dev/update_regression_references.py [route_toy modules_toy modules_tng]    (default: all of them)

Run it **only for an intended change of the results**, and record the reason in ``CHANGELOG.md``: the regression tests
exist to catch the unintended ones. It runs on the CPU with float64 enabled before SkewerJAX is imported (as the tests
do: import-time constants follow the x64 flag); ``modules_tng`` needs the TNG50-4 snapshot ($SKEWERJAX_DATA).
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import numpy as np  # noqa: E402

REGRESSION = Path(__file__).resolve().parents[2] / "tests" / "regression"
sys.path.insert(0, str(REGRESSION))
import outputs  # noqa: E402


def main(names):
    for name in names or list(outputs.REFERENCES):
        if name not in outputs.REFERENCES:
            raise SystemExit(f"unknown reference {name!r}; choose from {sorted(outputs.REFERENCES)}")
        arrays = getattr(outputs, outputs.REFERENCES[name])()
        path = REGRESSION / "data" / f"{name}.npz"
        np.savez_compressed(path, **arrays)
        print(f"wrote {path} ({len(arrays)} arrays, {path.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main(sys.argv[1:])
