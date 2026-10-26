"""Shared command line, PASS/FAIL bookkeeping and figure saving of the validation scripts."""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import logging                                                               # noqa: E402

logging.getLogger("jax._src.xla_bridge").setLevel(logging.CRITICAL)
import jax                                                                   # noqa: E402

from skewerjax.utils.paths import snapshot_dir                              # noqa: E402

TNG = (str(snapshot_dir("TNG50-4")), 25)                # $SKEWERJAX_DATA/TNG50-4 snapshot 25 (skewerjax.utils.paths)


class Checks:
    """``check(name, value, tol)`` records PASS if value <= tol (``expect=True``: PASS if value > tol) and prints it."""

    def __init__(self, title: str, description: str = "", quick_help: str = "fewer / smaller cases"):
        ap = argparse.ArgumentParser(description=description or title)
        ap.add_argument("--quick", action="store_true", help=quick_help)
        ap.add_argument("--outdir", default=None, help="save diagnostic figures here")
        self.args = ap.parse_args()
        jax.config.update("jax_enable_x64", True)            # the tolerances are float64 ones
        self.title, self.rows, self.t0 = title, [], time.perf_counter()
        self.quick = self.args.quick
        self.outdir = None if self.args.outdir is None else Path(self.args.outdir)
        print(f"== {title} ({'quick' if self.quick else 'full'}, x64 {jax.config.jax_enable_x64})")

    def check(self, name: str, value, tol: float, expect: bool = False, detail: str = ""):
        value = float(value)
        ok = value > tol if expect else value <= tol
        self.rows.append((name, ok))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {value:.3e} ({'>' if expect else '<='} {tol:g}) {detail}".rstrip())
        return ok

    def savefig(self, fig, name: str):
        if self.outdir is not None:
            self.outdir.mkdir(parents=True, exist_ok=True)
            fig.savefig(self.outdir / f"{name}.png", bbox_inches="tight")
        import matplotlib.pyplot as plt
        plt.close(fig)

    def done(self):
        n_ok = sum(ok for _, ok in self.rows)
        print(f"== {self.title}: {n_ok}/{len(self.rows)} PASS in {time.perf_counter() - self.t0:.0f} s")
        sys.exit(0 if n_ok == len(self.rows) else 1)


def rel(a, b):
    """max |a - b| / max |b| (relative to the scale of b)."""
    import numpy as np
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.max(np.abs(a - b)) / max(float(np.max(np.abs(b))), 1e-300))


def figure_backend():
    import matplotlib
    matplotlib.use("Agg")
    from skewerjax.utils.plotting import use_style
    use_style()
