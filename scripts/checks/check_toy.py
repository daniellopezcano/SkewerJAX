#!/usr/bin/env python
"""End-to-end check on a tiny toy box (a few seconds on a CPU): a Gaussian random field on 30^3 grid cells, 4 x 4
straight skewers, Lya and Si III, tau periodic over the box, F and P1D. Checks: finite tau, 0 < <F> < 1, the absorption
on one period equals the column of the pieces (``window_fraction`` = 1), the column of the pieces equals the exact
column of the grid cells along each skewer, and a finite, positive P1D. Exit status 1 on any failure.

    python scripts/checks/check_toy.py
"""

import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax  # noqa: E402
import numpy as np  # noqa: E402

from skewerjax.core import (  # noqa: E402
    Cosmology,
    LineBundle,
    SkewerBundle,
    cell_pieces,
    compute_optical_depth,
    grid_skewer_points,
    phys,
)
from skewerjax.simulations import PowerLawGRF, grf_cells  # noqa: E402
from skewerjax.summary_stats import DeltaFieldBuilder, OverfluxStatistics1D  # noqa: E402


def main() -> int:
    L, z, n_grid, n_sk, n_pts, n_pix = 20.0, 2.2, 30, 4, 512, 1024  # skewers between the cell faces
    lines = LineBundle(["HI-LyA", "SiIII-1206"], ref_line_id="HI-LyA")
    cosmo = Cosmology(H0=phys.H0, Omega_r_0=0.0, Omega_m_0=1.0, Omega_k_0=0.0, Omega_lambda_0=0.0)
    grf = PowerLawGRF(A_s=0.04, k_0=2.0, n_s=-1.25, L=(L,) * 3, N=(n_grid,) * 3)
    delta = np.maximum(np.asarray(grf.sample(jax.random.PRNGKey(0))), -0.9)
    n_bar = {"HI": 1e-12, "SiIII": 2e-14}
    cells = grf_cells(delta, grf, n_bar, temperature=1.5e4)
    points = np.asarray(grid_skewer_points(L, n_sk, n_pts, direction="y"), dtype=np.float64).reshape(n_sk**2, n_pts, 3)
    bundle = SkewerBundle(spatial_points=points, cosmologies=cosmo, a0=1 / (1 + z), x0=L)
    a_dagger = 1 / (1 + z)
    v_max = float(phys.c_kms * np.log(a_dagger / float(bundle.scale_factors[0, 0, 0])))
    v = (np.arange(n_pix) + 0.5) * v_max / n_pix
    pieces = cell_pieces(cells, lines, skewer_bundle=bundle, a_dagger=a_dagger)
    tau, info = compute_optical_depth(pieces, v, lines, velocity_period=True, resolution_tolerance=None, return_info=True)
    tau = np.asarray(tau, dtype=np.float64)
    F = np.exp(-tau)
    checks = {
        "tau finite and >= 0": bool(np.all(np.isfinite(tau)) and tau.min() >= 0),
        "0 < <F> < 1 for every line": bool(np.all((F.mean((1, 2)) > 0) & (F.mean((1, 2)) < 1))),
        "window_fraction = 1 on one period": bool(np.allclose(info["window_fraction"], 1.0, rtol=1e-3)),
    }
    # the column of the pieces against the exact column of the grid cells crossed (each skewer crosses one row of cells
    # along y, from 0 to L: sum_j n_j dx)
    col = (np.asarray(pieces.number_density, dtype=np.float64)[0] * np.asarray(pieces.length_m, dtype=np.float64)).sum(-1)
    col /= phys.Mpc_to_cm / 100.0  # [cMpc cm^-3]
    ix = np.floor(points[:, 0, [0, 2]] / (L / n_grid)).astype(int)
    col_exact = n_bar["HI"] * (1 + delta)[ix[:, 0], :, ix[:, 1]].sum(-1) * (L / n_grid)
    checks["column of the pieces = column of the cells"] = bool(np.allclose(col, col_exact, rtol=1e-4))
    st = OverfluxStatistics1D(DeltaFieldBuilder(tau, labels=["Lya", "SiIII"], key_joiner="+", mean="global").build(), float(v[1] - v[0]))
    p1d = np.asarray(st.compute_p1d().Pk_total)[1:].real
    checks["P1D finite and positive"] = bool(np.all(np.isfinite(p1d)) and p1d.mean() > 0)
    for k, ok in checks.items():
        print(f"[{'OK' if ok else 'FAIL'}] {k}")
    print(f"<F>: Lya {F[0].mean():.3f}, Si III {F[1].mean():.4f}; {pieces.shape[-1]} pieces per skewer; window fraction "
          f"{info['window_fraction'].min(1)} to {info['window_fraction'].max(1)}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
