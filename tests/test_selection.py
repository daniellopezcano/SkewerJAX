"""Cell selection (``core.selection``, the rule of every source of cells): the selection radii, the completeness of
``select_cells`` (every cell that owns a point of a curved path through the periodic faces is kept, for Voronoi and grid
cells; "tophat" is a subset of "voronoi"), the CPU distance (periodic minimum image), and the back ends ("auto" falls
back to the CPU without jztree, "jztree" fails clearly). The streamed version on a snapshot
(``simulations.select_cells_near_paths``) against ``select_cells``: ``test_snapshot.py``."""
import numpy as np
import pytest
from scipy.spatial import cKDTree

from skewerjax.core import cells_from_values, cells_on_grid, select_cells, voronoi_volumes
from skewerjax.core import selection as sel

BOX = 10.0


def _curved_path(n=300):
    """(1, n, 3) an arc that crosses the x and y faces (unwrapped)."""
    t = np.linspace(0.0, 1.0, n)
    return np.stack([1.0 + 1.7 * BOX * t, 9.0 + 0.6 * BOX * t + 1.5 * np.sin(3 * t), 5.0 + 2.0 * np.cos(4 * t)], -1)[None]


def _owners(gen, path):
    """Indices of the generators nearest to dense samples of the polyline (periodic): the cells it crosses."""
    p = path[0]
    fine = (p[:-1, None, :] + np.linspace(0, 1, 9)[None, :, None] * np.diff(p, axis=0)[:, None, :]).reshape(-1, 3)
    return np.unique(cKDTree(gen, boxsize=BOX).query(sel.wrap_positions(fine, BOX))[1])


def test_radii():
    size = np.array([0.1, 1.0])
    np.testing.assert_allclose(sel.selection_radius(size, "tophat", 0.2), size + 0.1)
    np.testing.assert_allclose(sel.selection_radius(size, "voronoi", 0.2), sel.VORONOI_FACTOR * size + 0.1)
    with pytest.raises(ValueError):
        sel.selection_radius(size, "sph", 0.2)
    assert np.isclose(sel.max_path_spacing(np.array([[[0, 0, 0], [0, 0, 1.0], [0, 3.0, 1.0]]])), 3.0)
    with pytest.raises(ValueError):
        sel.max_path_spacing(np.zeros((5, 3)))


def test_select_cells_is_complete():
    rng = np.random.default_rng(1)
    gen = rng.uniform(0.0, BOX, (4000, 3))
    gen[:1500, 2] = 4.0 + rng.uniform(0.0, 2.0, 1500)                       # uneven: a dense slab around the path
    c = cells_from_values(gen, BOX, {"HI": np.arange(4000.0)}, volumes=voronoi_volumes(gen, BOX, samples_per_cell=64),
                          ids=np.arange(4000))
    path = _curved_path()
    near = select_cells(c, path)
    own = _owners(c.positions, path)
    assert np.isin(own, near.ids).all() and len(near) < len(c) / 4         # complete, and still a small subset
    np.testing.assert_array_equal(near.species["HI"]["number_density"], near.ids)   # the fields follow the cells
    top = select_cells(c, path, mode="tophat")
    assert np.isin(top.ids, near.ids).all() and len(top) < len(near)
    shifted = select_cells(c, path + BOX * np.array([3, -1, 2]))           # any periodic image of the path
    assert np.array_equal(np.sort(shifted.ids), np.sort(near.ids))
    grid = cells_on_grid(BOX, {"HI": np.ones((10, 10, 10))}, ids=np.arange(1000))
    g_near = select_cells(grid, path)
    assert np.isin(_owners(grid.positions, path), g_near.ids).all() and g_near.grid is None
    far = cells_from_values(gen + 100.0, None, {"HI": 1.0}, volumes=np.full(len(gen), 1e-3))
    with pytest.raises(RuntimeError, match="no gas cell"):
        select_cells(far, path)                                             # no box: nothing within reach


def test_cpu_distance_is_the_periodic_minimum_image():
    rng = np.random.default_rng(3)
    pts = sel.wrap_positions(_curved_path()[0], BOX)
    cells = rng.uniform(0.0, BOX, (2000, 3))
    rad = np.full(cells.shape[0], 0.8)
    d = sel._nearest_distance_cpu(cKDTree(pts, boxsize=BOX), cells, rad, workers=1)
    diff = cells[:, None, :] - pts[None, :, :]
    diff -= BOX * np.round(diff / BOX)
    brute = np.sqrt((diff ** 2).sum(-1)).min(1)
    inside = brute <= rad
    np.testing.assert_allclose(d[inside], brute[inside], rtol=0, atol=1e-12)
    assert np.all(np.isinf(d[~inside]) | (d[~inside] > rad[~inside]))
    assert np.all(sel.wrap_positions(np.array([-1e-17, BOX, 2.5 * BOX]), BOX) < BOX)


def test_backends():
    assert sel.resolve_backend("auto") == ("jztree" if sel.jztree_available() else "cpu")
    assert sel.resolve_backend("cpu") == "cpu"
    with pytest.raises(ValueError):
        sel.resolve_backend("gpu")
    if not sel.jztree_available():
        with pytest.raises(RuntimeError, match="jztree"):
            sel.resolve_backend("jztree")
