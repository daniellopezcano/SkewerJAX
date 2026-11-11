"""The streaming route (``simulations.stream``): the tiles partition the skewers; every tile holds every cell within reach
of its skewers (the crossings of every skewer equal those from the gather route's selection); the two routes of
``snapshot_optical_depth`` give the same tau; the store holds what its layout says and the pass gives its memory back."""

import numpy as np
import pytest

from skewerjax.comparison import scaling as S
from skewerjax.core import LineBundle, SkewerBundle
from skewerjax.core import crossings as cx
from skewerjax.simulations import ArepoHDF5Reader, snapshot_optical_depth, stream
from skewerjax.simulations.selection import select_cells_near_paths
from skewerjax.utils.paths import snapshot_dir

from helpers import tol

LINES = LineBundle(["HI-LyA", "SiIII-1206"], ref_line_id="HI-LyA")


def test_tiles_partition_the_skewers():
    """Every skewer in one tile; the tiles are squares of the skewers' plane of about ``tile_size`` skewers."""
    box = 10.0
    X, Z = S.nested_grid(16, box)
    paths = cx.StraightPaths(np.column_stack([X, 0 * X, Z]), np.tile([0.0, 1.0, 0.0], (X.size, 1)), np.full(X.size, box), box, 1)
    tile = stream.tile_skewers(paths, 16)
    assert tile.shape == (256,) and set(np.unique(tile)) == set(range(16))
    assert np.all(np.bincount(tile) == 16)
    for k in range(16):  # a square: the skewers of a tile span a quarter of the box in x and in z
        sel = tile == k
        assert np.ptp(X[sel]) < box / 4 and np.ptp(Z[sel]) < box / 4
    one = stream.tile_skewers(paths, 10**6)
    assert np.all(one == 0)


def _setup(n=6):
    reader = ArepoHDF5Reader(snapshot_dir("TNG50-4"), 25)
    H = reader.header
    X, Z = S.nested_grid(n, H.box_cMpc)
    cofm = np.column_stack([X, np.zeros_like(X), Z])
    return reader, H, cofm


@pytest.mark.tng
def test_every_tile_holds_every_cell_within_reach():
    """The crossings of every skewer from its tile's cells are those from the gather route's selection (the k-d tree on
    the sampling points): the same cells, the same boundaries."""
    reader, H, cofm = _setup()
    paths, _ = stream.box_skewers(cofm, S.AXIS, H.box_cMpc, H.cosmology(), H.a)
    tile_of = stream.tile_skewers(paths, 9)  # four tiles of nine skewers
    store, info = stream.stream_cells(reader, paths, tile_of, list(LINES.species_keys), {"default": S.DEFAULT_CELLS})
    assert info["n_needed"] == store.n_cells and int(tile_of.max()) == 3
    b = SkewerBundle.axis_aligned(cofm, S.AXIS, H.box_cMpc, 0.25 / H.velfac_kms_per_cMpc, H.cosmology(), a_ref=H.a)
    gas = select_cells_near_paths(reader, np.asarray(b.host["points"]), mode="voronoi")
    ref = cx.straight_crossings(paths, np.asarray(gas.positions, np.float64), np.cbrt(np.asarray(gas.volumes, np.float64)))
    for k in range(4):
        sk = np.flatnonzero(tile_of == k)
        idx = store.tile_cells[store.tile_start[k] : store.tile_start[k + 1]]
        got = cx.straight_crossings(paths.take(sk), store.pos[idx], store.size[idx].astype(np.float64))
        for i, s in enumerate(sk):
            a = slice(got.start[i], got.start[i + 1])
            r = slice(ref.start[s], ref.start[s + 1])
            assert np.array_equal(store.ids[idx][got.cell[a]], np.asarray(gas.ids)[ref.cell[r]]), (k, s)
            np.testing.assert_allclose(got.s_lo[a], ref.s_lo[r], rtol=0, atol=1e-6)  # float32 sizes in the store


@pytest.mark.tng
def test_the_two_routes_agree():
    """``strategy="stream"`` and ``"gather"`` of ``snapshot_optical_depth``: the same tau (the store keeps n, T and the
    velocities in float32), and ``"auto"`` takes the stream for axis-aligned skewers."""
    reader, H, cofm = _setup(4)
    nb, dv = H.pixel_grid(1.0)
    v = (np.arange(nb) + 0.5) * dv
    b = SkewerBundle.axis_aligned(cofm, S.AXIS, H.box_cMpc, 0.25 / H.velfac_kms_per_cMpc, H.cosmology(), a_ref=H.a)
    t_s, i_s = snapshot_optical_depth(reader, b, v, LINES, strategy="auto", physics=S.DEFAULT_CELLS, return_info=True)
    t_g = snapshot_optical_depth(reader, b, v, LINES, strategy="gather", physics=S.DEFAULT_CELLS)
    assert i_s["strategy"] == "stream"
    F_s, F_g = np.exp(-np.asarray(t_s, np.float64)), np.exp(-np.asarray(t_g, np.float64))
    assert np.max(np.abs(F_s - F_g)) < tol(2e-6, 2e-5)


@pytest.mark.tng
def test_store_layout_and_memory():
    """The store's bytes are its layout (48 bytes per cell, 16 per configuration for two species, 8 per (cell, tile)
    pair), and the memory of the pass's blocks goes back to the system (the anonymous memory after the pass is the
    store's, within the interpreter's and JAX's own)."""
    reader, H, cofm = _setup(8)
    paths, _ = stream.box_skewers(cofm, S.AXIS, H.box_cMpc, H.cosmology(), H.a)
    tile_of = stream.tile_skewers(paths, 16)
    before = S.rss_anon_GB()
    store, info = stream.stream_cells(reader, paths, tile_of, list(LINES.species_keys), {"a": S.DEFAULT_CELLS, "b": S.DEFAULT_CELLS})
    n, pairs = store.n_cells, store.tile_cells.size
    assert store.nbytes == n * (24 + 4 + 12 + 8) + 2 * n * 16 + 8 * pairs
    assert S.rss_anon_GB() - before < store.nbytes / 1e9 + 1.5
