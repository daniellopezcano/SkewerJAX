"""Projected maps of the gas in a slab of a snapshot, streamed in blocks (for figures of the gas around skewers). NumPy."""
from __future__ import annotations

import numpy as np

from skewerjax.simulations import gas_physics as gp


def slab_projection(reader, normal_axis: int, centre_cMpc: float, thickness_cMpc: float, n_pix: int = 512, *,
                    temperature: bool = True, max_rows: int = 4_000_000) -> dict:
    """Gas surface density and mass-weighted temperature of the slab |x_normal - centre| < thickness / 2 (periodic).

    Cells are binned at their centres (no spreading over their volume): a display map, adequate when the pixel is not
    much smaller than the cells it shows. ``normal_axis`` is 1/2/3 = x/y/z, as the skewer axes.

    Returns
    -------
    dict with ``sigma`` (n_pix, n_pix) [Msun / ckpc^2] and, if ``temperature``, ``T_mw`` [K] (NaN in empty pixels),
    indexed [i, j] = (first, second in-plane axis); ``axes`` (the two in-plane axes, 1-based); ``extent`` [cMpc] for
    ``imshow(m.T, origin="lower", extent=extent)``; ``n_cells`` in the slab.
    """
    if normal_axis not in (1, 2, 3):
        raise ValueError(f"normal_axis must be 1, 2 or 3 (x, y, z), got {normal_axis}.")
    H = reader.header
    box = H.box_cMpc
    ax_n = normal_axis - 1
    ax_a, ax_b = [i for i in range(3) if i != ax_n]
    keys = ["Coordinates", "Masses"] + (["InternalEnergy", "ElectronAbundance"] if temperature else [])
    m_sum = np.zeros((n_pix, n_pix))
    mT_sum = np.zeros((n_pix, n_pix)) if temperature else None
    n_cells = 0
    edges = np.linspace(0.0, box, n_pix + 1)
    for _, _, raw in reader.iter_blocks(keys, max_rows=max_rows):
        x = reader.positions_cMpc(raw["Coordinates"])
        d = x[:, ax_n] - centre_cMpc
        d -= box * np.round(d / box)
        sel = np.abs(d) < 0.5 * thickness_cMpc
        if not np.any(sel):
            continue
        n_cells += int(sel.sum())
        m = np.asarray(raw["Masses"], dtype=np.float64)[sel] * 1e10 / H.h                  # [Msun]
        h2, _, _ = np.histogram2d(x[sel, ax_a], x[sel, ax_b], bins=(edges, edges), weights=m)
        m_sum += h2
        if temperature:
            T = np.asarray(gp.temperature(np.asarray(raw["InternalEnergy"], dtype=np.float64)[sel] * H.unit_velocity_cms ** 2,
                                          np.asarray(raw["ElectronAbundance"], dtype=np.float64)[sel]))
            h2, _, _ = np.histogram2d(x[sel, ax_a], x[sel, ax_b], bins=(edges, edges), weights=m * T)
            mT_sum += h2
    pix_ckpc2 = (box / n_pix * 1e3) ** 2
    out = dict(sigma=m_sum / pix_ckpc2, axes=(ax_a + 1, ax_b + 1), extent=(0.0, box, 0.0, box), n_cells=n_cells)
    if temperature:
        with np.errstate(invalid="ignore", divide="ignore"):
            out["T_mw"] = np.where(m_sum > 0, mT_sum / m_sum, np.nan)
    return out
