"""Voronoi deposition: the JAX backend against the NumPy default, and the Voronoi-adapted cell crossings (toy mesh)."""
import numpy as np

from skewerjax.core import Cells
from skewerjax.core import absorbers as ab_
from skewerjax.core import deposition as dep

BOX = 10.0


def _toy(seed=0, n_gen=3000):
    """Random generators in a periodic box (uneven: a dense slab) and two paths, one crossing the faces."""
    rng = np.random.default_rng(seed)
    g = rng.uniform(0.0, BOX, (n_gen, 3))
    g[: n_gen // 3, 2] = 4.5 + rng.uniform(0.0, 1.0, n_gen // 3)
    t = np.linspace(0.0, 1.0, 400)
    straight = np.stack([t * BOX, np.full_like(t, 3.3), np.full_like(t, 5.1)], -1)
    oblique = np.stack([1.0 + 1.7 * BOX * t, 9.0 + 0.6 * BOX * t, 4.0 + 0.4 * BOX * t], -1)   # crosses x, y faces
    size = np.full(n_gen, (BOX ** 3 / n_gen) ** (1 / 3))
    return np.stack([straight, oblique]), g, size


def test_jax_backend_is_identical_to_numpy():
    pts, g, size = _toy()
    a = dep.voronoi_segments(pts, g, BOX, gas_size=size)
    b = dep.voronoi_segments(pts, g, BOX, gas_size=size, backend="jax", chunk=32, k_chunk=64)
    assert a.cell.shape == b.cell.shape
    assert np.array_equal(a.cell, b.cell)
    np.testing.assert_allclose(b.length, a.length, rtol=0, atol=1e-12)
    np.testing.assert_allclose(b.offset, a.offset, rtol=0, atol=1e-12)
    assert b.diagnostics["k_chunk"] >= 64                # doubled until the candidate sets were complete


def test_cell_crossings_conserve_the_column():
    pts, g, size = _toy(1)
    seg = dep.voronoi_segments(pts, g, BOX, gas_size=size)
    cr = dep.cell_crossings(seg, pts)
    assert cr.diagnostics["max_gap"] < 1e-9               # straight segments: the pieces tile the path
    n = np.random.default_rng(2).lognormal(size=len(g))
    col_seg = seg.column_lengths(n).sum(-1)
    col_cr = (np.where(cr.cell >= 0, n[np.maximum(cr.cell, 0)], 0.0) * cr.length).sum(-1)
    np.testing.assert_allclose(col_cr, col_seg, rtol=1e-12)
    assert np.all(cr.n_crossings <= (seg.cell >= 0).sum((-1, -2)))
    # consecutive crossings hold different cells
    for k in range(cr.cell.shape[0]):
        c = cr.cell[k][cr.cell[k] >= 0]
        assert np.all(c[1:] != c[:-1])


def test_split_crossings_keep_the_column():
    pts, g, size = _toy(3)
    seg = dep.voronoi_segments(pts, g, BOX, gas_size=size)
    cr = dep.cell_crossings(seg, pts)
    rng = np.random.default_rng(4)
    n, T, vel = rng.lognormal(size=len(g)), rng.uniform(1e3, 1e5, len(g)), rng.normal(0, 30, (len(g), 3))
    s = np.linalg.norm(np.diff(pts, axis=1), axis=-1).cumsum(-1)
    s = np.concatenate([np.zeros((2, 1)), s], -1)
    frame = ab_.SnapshotFrame(a_snap=0.25, velfac=300.0, path_length=s[:, -1])
    cols = []
    for tol in (None, 0.25, 0.05):
        cells = Cells(g, size ** 3, {"SiIII": {"number_density": n, "temperature": T, "velocity": vel}}, box=BOX)
        A = ab_.absorbers_from_crossings(cr, pts, cells, "SiIII", 28.0855, a_data=0.25, frame=frame, tolerance=tol)
        cols.append((A.n * A.length).sum(-1))
        if tol is not None:
            assert A.meta["max_split"] > 1
    np.testing.assert_allclose(cols[1], cols[0], rtol=1e-12)
    np.testing.assert_allclose(cols[2], cols[0], rtol=1e-12)
