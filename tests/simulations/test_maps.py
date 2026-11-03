"""Maps of the gas for figures (``simulations.maps``): ``voronoi_slice``, the section of the Voronoi cells by a plane,
against the nearest generating point found by brute force over every cell and its periodic images (notebook 10, C1).
``slab_projection`` reads a snapshot: notebook 00_quickstart_step_by_step."""

import numpy as np
import pytest

from skewerjax.core.cells import cells_from_values
from skewerjax.simulations import voronoi_slice


def test_voronoi_slice_is_the_nearest_cell():
    rng = np.random.default_rng(3)
    L, M, n_pix = 10.0, 3000, 24
    pos = rng.uniform(0.0, L, (M, 3))
    cells = cells_from_values(pos, L, {"HI": rng.lognormal(size=M), "SiIII": rng.lognormal(size=M)})
    for axis, centre in ((3, 0.0), (1, 7.3), (2, 9.95)):
        out = voronoi_slice(cells, "SiIII", axis, centre, n_pix=n_pix)
        a, b = out["axes"]
        assert out["n"].shape == (n_pix, n_pix) and out["extent"] == (0.0, L, 0.0, L) and 0 < out["n_cells"] <= M
        c = (np.arange(n_pix) + 0.5) * L / n_pix
        q = np.zeros((n_pix * n_pix, 3))
        q[:, a - 1], q[:, b - 1], q[:, axis - 1] = np.repeat(c, n_pix), np.tile(c, n_pix), centre
        d = q[:, None, :] - np.asarray(cells.positions)[None]
        d -= L * np.round(d / L)  # the nearest periodic image
        owner = np.argmin((d**2).sum(-1), axis=1)
        want = np.asarray(cells.species["SiIII"]["number_density"])[owner].reshape(n_pix, n_pix)
        assert np.array_equal(out["n"], want), axis
    with pytest.raises(ValueError, match="normal_axis"):
        voronoi_slice(cells, "HI", 0, 1.0)
