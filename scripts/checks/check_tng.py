#!/usr/bin/env python
"""End-to-end check on TNG50-4 snapshot 25 (about 10 s; skipped, exit status 0, without the data in $SKEWERJAX_DATA):
one skewer along y through the box, the default route (``select_cells_near_paths``, ``gas_to_cells``, ``cell_pieces``,
``compute_optical_depth``), Lya and Si III on 1 km/s pixels, periodic over the box. Checks: cells selected, finite tau,
a mean Lya flux between 0.3 and 0.99 (z = 3), and the absorption on one period equal to the column of the pieces
(``window_fraction`` = 1 to 1e-3). Exit status 1 on any failure.

    python scripts/checks/check_tng.py
"""

import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np  # noqa: E402

from skewerjax.utils.paths import DATA_DIR  # noqa: E402


def main() -> int:
    if not (DATA_DIR / "TNG50-4" / "snapdir_025").is_dir():
        print(f"[skip] no TNG50-4 snapshot 25 in {DATA_DIR} ($SKEWERJAX_DATA)")
        return 0
    from skewerjax.core import LineBundle, SkewerBundle, cell_pieces, compute_optical_depth
    from skewerjax.simulations import ArepoHDF5Reader, gas_to_cells, select_cells_near_paths

    reader = ArepoHDF5Reader(DATA_DIR / "TNG50-4", 25)
    H = reader.header
    L, AH = H.box_cMpc, H.velfac_kms_per_cMpc
    lines = LineBundle(["HI-LyA", "SiIII-1206"], ref_line_id="HI-LyA")
    bundle = SkewerBundle.axis_aligned(np.array([[L / 4, 0.0, L / 4]]), 2, L, 0.25 / AH, H.cosmology(), a_ref=H.a)
    gas = select_cells_near_paths(reader, bundle.host["points"])
    cells = gas_to_cells(gas, list(lines.species_keys))
    pieces = cell_pieces(cells, lines, skewer_bundle=bundle, a_data=H.a)
    nbins, dv = H.pixel_grid(1.0)
    v = (np.arange(nbins) + 0.5) * dv
    tau, info = compute_optical_depth(pieces, v, lines, velocity_period=True, return_info=True)
    tau = np.asarray(tau, dtype=np.float64)
    meanF = np.exp(-tau[0]).mean()
    checks = {
        f"{len(gas):,} cells selected": len(gas) > 0,
        "tau finite and >= 0": bool(np.all(np.isfinite(tau)) and tau.min() >= 0),
        f"0.3 < <F_Lya> = {meanF:.3f} < 0.99": bool(0.3 < meanF < 0.99),
        "window_fraction = 1 on one period": bool(np.allclose(info["window_fraction"], 1.0, rtol=1e-3)),
    }
    for k, ok in checks.items():
        print(f"[{'OK' if ok else 'FAIL'}] {k}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
