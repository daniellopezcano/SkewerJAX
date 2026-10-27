#!/usr/bin/env python
"""The optical depth of uniform gas against its closed forms, as a figure: static natural and thermal lines, and
radiation domination (natural lines unresolved by the points; thermal lines). The cases are those of the test suite
(``tests/core/analytic_cases.py``; the checks themselves: ``tests/core/test_optical_depth.py``).

    python scripts/figures/optical_depth_vs_analytic.py [--out optical_depth_vs_analytic.png]
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests" / "core"))
from skewerjax.utils.plotting import MODE_STYLES as MS  # noqa: E402
from skewerjax.utils.plotting import add_panel_letters, use_style

import analytic_cases as ac  # noqa: E402


def main(out: str):
    use_style()
    cases = {
        "static, T = 0 (natural)": ac.static_natural(),
        "static, A_21 = 0 (thermal)": ac.static_thermal(),
        "radiation domination, T = 0 (natural, unresolved comb)": ac.radiation_natural()[:3],
        "radiation domination, A_21 = 0 (thermal, 336 points)": ac.radiation_thermal(336),
    }
    fig, axs = plt.subplots(
        2, len(cases), figsize=(22, 8), sharex="col", gridspec_kw={"height_ratios": [2, 1]}, layout="constrained"
    )
    for k, (name, (v, tau, ref)) in enumerate(cases.items()):
        F, Fa = np.exp(-np.asarray(tau)), np.exp(-np.asarray(ref))
        for s in range(F.shape[1]):
            (line,) = axs[0, k].plot(v, F[0, s], **MS["default"])
            axs[0, k].plot(v, Fa[0, s], color=line.get_color(), **MS["reference"])
            axs[1, k].plot(v, F[0, s] - Fa[0, s], color=line.get_color())
        axs[0, k].set_title(name, fontsize=15)
        axs[1, k].set_xlabel(r"$v$ [km/s]")
    axs[0, 0].set_ylabel(r"$F$ (Ly$\alpha$)")
    axs[1, 0].set_ylabel(r"$F - F_{\rm analytic}$")
    add_panel_letters(axs)
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="optical_depth_vs_analytic.png")
    main(ap.parse_args().out)
