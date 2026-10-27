"""The pieces of the optical-depth sum from cells (``core.cell_pieces``): they tile the path, at their centroids (ln a
linear in s between the points), and the sum converges at second order in the piece length with an exact column at
every spacing. The sum itself (periodicity, frames, precision, closed forms): ``test_optical_depth.py``."""

import jax.numpy as jnp
import numpy as np

from skewerjax.core import LineBundle, SkewerBundle, cell_pieces, cells_on_grid, compute_optical_depth, phys

from helpers import uniform_skewer as uniform

LYA = LineBundle(["HI-LyA"], ref_line_id="HI-LyA")


def test_pieces_tile_the_path_and_centroids():
    sb, f, pos, a = uniform()
    p = cell_pieces(f, LYA, skewer_bundle=sb, a_dagger=0.302)
    order = np.argsort(np.asarray(p.scale_factor[0]))
    L = np.asarray(p.length_m[0])[order] / phys.Mpc_to_m
    assert np.isclose(L.sum(), pos[0, -1, 0] - pos[0, 0, 0], rtol=1e-12), "the pieces tile the path"
    x = pos[0, :, 0]
    # the scale factor at the centroid: ln a linear in s between the points
    s_c = np.concatenate([[0.0], np.cumsum(L)])[:-1] + 0.5 * L
    assert np.allclose(np.asarray(p.scale_factor[0])[order], np.exp(np.interp(s_c, x - x[0], np.log(a[0, 0]))), rtol=1e-12)
    p1 = p
    d = np.diff(pos[0, :, 0])
    assert np.isclose(
        float(np.asarray(p1.scale_factor[0]).min()),
        np.exp(np.log(a[0, 0, 0]) + 0.25 * np.log(a[0, 0, 1] / a[0, 0, 0])),
        rtol=1e-12,
    ), "the first piece sits at d_0 / 4"
    assert np.isclose(float(p1.velocity_span_kms[0]), phys.c_kms * np.log(0.302 / 0.30), rtol=1e-12)
    assert d.min() < d.max()


def test_finer_sampling_converges_and_keeps_the_column():
    """Cells of a coarse grid sampled with more and more points: the pieces are the cell crossings cut at the dual
    segments, so the sum converges at second order in the spacing and the column is exact at every spacing."""
    L, n0 = 2.0, 1e-12
    grid = np.full((8, 8, 8), n0) * (1.0 + 0.5 * np.sin(2 * np.pi * np.arange(8) / 8))[:, None, None]
    cells = cells_on_grid(L, {"HI": grid}, temperature=2e3)
    v = jnp.linspace(-100.0, 2100.0, 4401)
    taus, cols = {}, {}
    for n_pt in (161, 321, 641, 5121):
        pos = np.zeros((1, n_pt, 3))
        pos[0, :, 0] = np.linspace(0.0, L * (1 - 1e-9), n_pt)
        pos[0, :, 1:] = 0.25
        a = np.exp(np.linspace(np.log(0.30), np.log(0.302), n_pt))[None, None]
        sb = SkewerBundle(spatial_points=jnp.asarray(pos), manual_scale_factors=jnp.asarray(a), a0=0.302, ICs_defined_at_index=-1)
        p = cell_pieces(cells, LYA, skewer_bundle=sb)
        cols[n_pt] = float(np.sum(np.asarray(p.number_density[0]) * np.asarray(p.length_m)))
        taus[n_pt] = np.asarray(compute_optical_depth(p, v, LYA, resolution_tolerance=None))[0, 0]
    e = [np.abs(taus[n] - taus[5121]).max() for n in (161, 321, 641)]
    assert e[0] > e[1] > e[2] and e[1] / e[2] > 3.0, f"second-order convergence, got {e}"
    assert np.allclose(list(cols.values()), cols[5121], rtol=1e-12), "the column is exact at any spacing"
