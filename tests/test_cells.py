"""The gas-cell container and the one route cells -> pieces -> tau (toy inputs, CPU).

* ``grid_segments`` (the regular-grid fast path) gives the pieces of the generic ``voronoi_segments`` on the same
  cells, for straight and bent paths that cross the periodic faces;
* ``Cells`` checks its inputs, is a pytree, and ``take`` / ``get`` behave;
* ``cells_on_grid`` / ``grf_cells``: n = n_bar (1 + delta) exactly at the generators, interpolated for random ones;
* ``cell_pieces``: the column of uniform cells is exact, several species share the pieces, and
  ``compute_optical_depth(bundle, v, lines, cells)`` is the sum over those pieces; the top-hat option runs on toy cells.
"""
import numpy as np
import jax
import jax.numpy as jnp
import pytest

from skewerjax.core import (Cells, Cosmology, LineBundle, SkewerBundle, cell_pieces, cells_from_values, cells_on_grid, compute_optical_depth,
                            grid_points, grid_segments, phys, voronoi_segments, voronoi_volumes)
from skewerjax.simulations import PowerLawGRF, grf_cells

BOX = 4.0
SHAPE = (8, 6, 5)


def _paths(rng, n_sk=6, n=40):
    """Straight and bent polylines through the box, crossing its faces (unwrapped), at generic positions."""
    out = []
    for k in range(n_sk):
        a = rng.uniform(0, BOX, 3)
        d1, d2 = rng.normal(size=3), rng.normal(size=3)
        d1 /= np.linalg.norm(d1); d2 /= np.linalg.norm(d2)
        t = np.sort(rng.uniform(0, 1.6 * BOX, n))
        bend = 0.5 * t.max()
        pts = a + np.where(t[:, None] < bend, t[:, None] * d1, bend * d1 + (t[:, None] - bend) * (d2 if k % 2 else d1))
        out.append(pts)
    return np.stack(out)


def _pieces_by_point(seg, k, i):
    m = seg.cell[k, i] >= 0
    order = np.lexsort((seg.offset[k, i][m], seg.cell[k, i][m]))
    return seg.cell[k, i][m][order], seg.length[k, i][m][order], seg.offset[k, i][m][order]


def test_grid_fast_path_gives_the_voronoi_pieces():
    rng = np.random.default_rng(7)
    pts = _paths(rng)
    gen = grid_points(BOX, SHAPE)
    size = np.full(gen.shape[0], np.cbrt(np.prod([BOX / n for n in SHAPE])))
    a = voronoi_segments(pts, gen, BOX, gas_size=size, n_sub=64)
    b = grid_segments(pts, SHAPE, BOX)
    for k in range(pts.shape[0]):
        for i in range(pts.shape[1]):
            ca, la, oa = _pieces_by_point(a, k, i)
            cb, lb, ob = _pieces_by_point(b, k, i)
            assert np.array_equal(ca, cb), (k, i, ca, cb)
            np.testing.assert_allclose(la, lb, rtol=0, atol=1e-12)
            np.testing.assert_allclose(oa, ob, rtol=0, atol=1e-12)
    # the pieces tile every dual segment
    w = np.asarray(jnp.asarray(b.length).sum(-1))
    from skewerjax.core import line_elements
    np.testing.assert_allclose(w, np.asarray(line_elements(jnp.asarray(pts))), rtol=1e-12)


def test_cells_container():
    n = 5
    pos = np.random.default_rng(0).uniform(0, BOX, (n, 3))
    sp = {"HI": {"number_density": np.ones(n), "temperature": np.full(n, 1e4), "velocity": np.zeros((n, 3))}}
    c = Cells(pos, np.ones(n), sp, box=BOX)
    assert len(c) == n and c.species_keys == ("HI",)
    np.testing.assert_array_equal(c.get("HI", "turbulence"), np.zeros(n))
    leaves, tree = jax.tree_util.tree_flatten(c)
    c2 = jax.tree_util.tree_unflatten(tree, leaves)
    assert c2.box == BOX and np.array_equal(c2.species["HI"]["temperature"], c.species["HI"]["temperature"])
    sub = c.take([0, 2])
    assert len(sub) == 2 and sub.grid is None
    with pytest.raises(ValueError):
        Cells(pos, np.ones(n), {"HI": {"number_density": np.ones(n), "temperature": np.ones(n)}})        # no velocity
    with pytest.raises(ValueError):
        Cells(pos, np.ones(n), {"HI": {**sp["HI"], "velocity": np.zeros(n)}})                              # velocity (M,)
    with pytest.raises(ValueError):
        Cells(pos, np.ones(n), {"HI": {**sp["HI"], "pressure": np.ones(n)}})                               # unknown field


def test_grid_and_random_generators():
    grf = PowerLawGRF(A_s=0.01, k_0=2.0, n_s=-1.25, L=(BOX,) * 3, N=(16,) * 3)
    delta = np.asarray(grf.sample(jax.random.PRNGKey(3)))
    c = grf_cells(delta, grf, {"HI": 2.0, "SiIII": 0.5}, temperature=lambda x: 1e4 + 0 * x[:, 0], velocity=(1.0, 2.0, 3.0))
    assert c.grid == (16, 16, 16)
    np.testing.assert_allclose(c.species["HI"]["number_density"], 2.0 * (1.0 + delta.ravel()), rtol=1e-15)
    np.testing.assert_allclose(c.species["SiIII"]["velocity"][7], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(c.volumes, (BOX / 16) ** 3)
    # random generators at the nodes themselves recover the grid values; their Voronoi volumes add up to the box
    gen = grid_points(BOX, grf.shape)[::7]
    r = grf_cells(delta, grf, {"HI": 2.0}, generators=gen, samples_per_cell=8)
    np.testing.assert_allclose(r.species["HI"]["number_density"], 2.0 * (1.0 + delta.ravel()[::7]), rtol=1e-12)
    np.testing.assert_allclose(r.volumes.sum(), BOX ** 3, rtol=0.05)
    v = voronoi_volumes(grid_points(BOX, (4, 4, 4)), BOX, samples_per_cell=256)
    np.testing.assert_allclose(v, (BOX / 4) ** 3, rtol=0.2)


def _bundle(n_sk=3, n=200):
    x = np.linspace(0.0, BOX, n)
    pts = np.stack([np.stack([x, np.full_like(x, 0.3 + 1.1 * k), np.full_like(x, 1.7)], -1) for k in range(n_sk)])
    eds = Cosmology(H0=phys.H0, Omega_r_0=0.0, Omega_m_0=1.0, Omega_k_0=0.0, Omega_lambda_0=0.0)
    return SkewerBundle(spatial_points=jnp.asarray(pts), cosmologies=eds, a0=0.25, x0=BOX)


def test_uniform_cells_column_and_species():
    sb = _bundle()
    n0 = 1e-12
    c = cells_on_grid(BOX, {"HI": np.full(SHAPE, n0), "SiIII": np.full(SHAPE, n0 / 10)}, temperature=2e4)
    lines = LineBundle(["HI-LyA", "SiIII-1206"], ref_line_id="HI-LyA")
    p = cell_pieces(c, lines, skewer_bundle=sb)
    col = np.asarray(p.number_density * p.length_m[None]).sum(-1) / phys.Mpc_to_m          # [cm^-3 cMpc]
    np.testing.assert_allclose(col[0], n0 * BOX, rtol=1e-12)
    np.testing.assert_allclose(col[1], n0 / 10 * BOX, rtol=1e-12)
    # the same pieces with the generic Voronoi deposition (no fast path), and tau through the bundle entry point
    q = cell_pieces(c, lines, skewer_bundle=sb, grid_fast_path=False)
    np.testing.assert_allclose(np.sort(np.asarray(q.length_m), -1), np.sort(np.asarray(p.length_m), -1), rtol=1e-12)
    v = np.linspace(-100.0, 700.0, 400)
    t_b = compute_optical_depth(sb, v, lines, c, resolution_tolerance=None)
    t_p = compute_optical_depth(p, v, lines, resolution_tolerance=None)
    np.testing.assert_allclose(np.asarray(t_b), np.asarray(t_p), rtol=0, atol=1e-14 * float(np.max(t_p)))


def test_tophat_on_toy_cells_conserves_the_column():
    sb = _bundle(n_sk=1, n=400)
    rng = np.random.default_rng(1)
    gen = rng.uniform(0, BOX, (3000, 3))
    c = cells_from_values(gen, BOX, {"HI": np.full(gen.shape[0], 1e-12)}, temperature=1e4, volumes=voronoi_volumes(gen, BOX))
    lya = LineBundle(["HI-LyA"], ref_line_id="HI-LyA")
    pv = cell_pieces(c, lya, skewer_bundle=sb)
    pt = cell_pieces(c, lya, skewer_bundle=sb, deposition="tophat")
    col = lambda p: float(np.asarray(p.number_density[0] * p.length_m[None, 0]).sum())
    assert abs(col(pv) / (1e-12 * BOX * phys.Mpc_to_m) - 1) < 1e-10           # Voronoi: exact for uniform gas
    assert abs(col(pt) / col(pv) - 1) < 0.3                                    # top hat: the same on average, not exactly
