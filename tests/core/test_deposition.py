"""Voronoi deposition (toy mesh): the exact lower envelope of straight paths against a fine sampling, the detection of
straight paths, straight and curved paths together, the JAX backend of the sampled construction against the NumPy one,
and the Voronoi-adapted cell crossings."""

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
    oblique = np.stack([1.0 + 1.7 * BOX * t, 9.0 + 0.6 * BOX * t, 4.0 + 0.4 * BOX * t], -1)  # crosses x, y faces
    size = np.full(n_gen, (BOX**3 / n_gen) ** (1 / 3))
    return np.stack([straight, oblique]), g, size


def test_jax_backend_is_identical_to_numpy():
    pts, g, size = _toy()
    a = dep.voronoi_segments(pts, g, BOX, gas_size=size, method="sampled")
    b = dep.voronoi_segments(pts, g, BOX, gas_size=size, method="sampled", backend="jax", chunk=32, k_chunk=64)
    assert a.cell.shape == b.cell.shape
    assert np.array_equal(a.cell, b.cell)
    np.testing.assert_allclose(b.length, a.length, rtol=0, atol=1e-12)
    np.testing.assert_allclose(b.offset, a.offset, rtol=0, atol=1e-12)
    assert b.diagnostics["k_chunk"] >= 64  # doubled until the candidate sets were complete


def _same_pieces(a, b, atol=1e-12):
    """The same cells in every dual segment, with the same lengths and offsets (to ``atol`` cMpc)."""
    for i in np.ndindex(a.cell.shape[:2]):
        da = {c: (ln, o) for c, ln, o in zip(a.cell[i], a.length[i], a.offset[i]) if c >= 0}
        db = {c: (ln, o) for c, ln, o in zip(b.cell[i], b.length[i], b.offset[i]) if c >= 0}
        assert da.keys() == db.keys(), i
        for c in da:
            assert abs(da[c][0] - db[c][0]) <= atol and abs(da[c][1] - db[c][1]) <= atol, (i, c)


def test_envelope_is_the_limit_of_the_sampling():
    """Along straight paths (one along x, one oblique through the periodic faces, one unevenly spaced) the envelope gives
    the pieces of the sampled construction at a sub-interval far below the smallest cell, and conserves the length."""
    pts, g, size = _toy(5)
    rng = np.random.default_rng(6)
    s = np.sort(np.concatenate([np.linspace(0.0, 0.8 * BOX, 300), rng.uniform(0.0, 0.8 * BOX, 100)]))
    uneven = np.array([7.0, 1.0, 3.0]) + s[:, None] * np.array([0.0, 0.6, 0.8])
    pts = np.concatenate([pts, uneven[None, :400]])
    env = dep.voronoi_segments(pts, g, BOX, gas_size=size, method="envelope")
    fine = dep.voronoi_segments(pts, g, BOX, gas_size=size, method="sampled", n_sub=1024)
    assert env.diagnostics["n_envelope"] == 3 and env.diagnostics["candidates_per_path"] > 0
    _same_pieces(env, fine)
    L = np.linalg.norm(pts[:, -1] - pts[:, 0], axis=-1)
    np.testing.assert_allclose(env.length.sum((-1, -2)), L, rtol=1e-13)


def test_straight_paths_are_detected_and_mixed_with_curved_ones():
    pts, g, size = _toy(7)
    t = np.linspace(0.0, 1.0, 400)
    curved = np.stack([5.0 + 3.0 * np.cos(2.0 * t), 5.0 + 3.0 * np.sin(2.0 * t), 2.0 + 6.0 * t], -1)
    f32 = pts[1].astype(np.float32).astype(np.float64)  # an oblique path rounded to float32: still straight
    assert list(dep.straight_paths(np.stack([pts[0], pts[1], curved, pts[0][::-1], f32]))) == [True, True, False, True, True]
    zigzag = pts[0].copy()
    zigzag[[10, 11]] = zigzag[[11, 10]]  # not in order along the chord
    assert not dep.straight_paths(zigzag[None])[0]
    both = np.stack([pts[0], curved, pts[1]])
    auto = dep.voronoi_segments(both, g, BOX, gas_size=size)
    assert auto.diagnostics["n_envelope"] == 2 and auto.diagnostics["n_sampled"] == 1
    env = dep.voronoi_segments(both[[0, 2]], g, BOX, gas_size=size, method="envelope")
    smp = dep.voronoi_segments(both[[1]], g, BOX, gas_size=size, method="sampled")
    k = auto.cell.shape[-1]
    pad = lambda x: x if x.cell.shape[-1] == k else dep.SegmentComponents(*(np.concatenate([a, np.full(a.shape[:-1] + (k - a.shape[-1],), f, a.dtype)], -1) for a, f in ((x.cell, -1), (x.length, 0.0), (x.offset, 0.0))), 1.0, "voronoi", {})
    _same_pieces(dep.SegmentComponents(auto.cell[[0, 2]], auto.length[[0, 2]], auto.offset[[0, 2]], 1.0, "voronoi", {}), pad(env), atol=0)
    _same_pieces(dep.SegmentComponents(auto.cell[[1]], auto.length[[1]], auto.offset[[1]], 1.0, "voronoi", {}), pad(smp), atol=0)
    try:
        dep.voronoi_segments(both, g, BOX, gas_size=size, method="envelope")
    except ValueError as e:
        assert "straight" in str(e)
    else:
        raise AssertionError("method='envelope' accepted a curved path")


def test_cell_crossings_conserve_the_column():
    pts, g, size = _toy(1)
    seg = dep.voronoi_segments(pts, g, BOX, gas_size=size)
    cr = dep.cell_crossings(seg, pts)
    assert cr.diagnostics["max_gap"] < 1e-9  # straight segments: the pieces tile the path
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
        cells = Cells(g, size**3, {"SiIII": {"number_density": n, "temperature": T, "velocity": vel}}, box=BOX)
        A = ab_.absorbers_from_crossings(cr, pts, cells, "SiIII", 28.0855, a_data=0.25, frame=frame, tolerance=tol)
        cols.append((A.n * A.length).sum(-1))
        if tol is not None:
            assert A.meta["max_split"] > 1
    np.testing.assert_allclose(cols[1], cols[0], rtol=1e-12)
    np.testing.assert_allclose(cols[2], cols[0], rtol=1e-12)
