"""The piece-based optical depth from cells (``core.cell_pieces``, ``core.compute_optical_depth``): the pieces tile the
path at their centroids, the sum converges at second order in the piece length with an exact column, the
``velocity_period`` option (False / True / a number) and its two effects, and the reference line frame."""
import numpy as np
import jax.numpy as jnp
import pytest

from skewerjax.core import (LineBundle, Pieces, SkewerBundle, cell_pieces, cells_from_values, cells_on_grid, compute_optical_depth,
                            phys)

LYA = LineBundle(["HI-LyA"], ref_line_id="HI-LyA")


def uniform(n_pt=201, L=2.0, n=1e-12, T=1e4):
    """A straight skewer of manual a(x) (log-linear) through uniform gas."""
    pos = np.zeros((1, n_pt, 3)); pos[0, :, 0] = np.linspace(0.0, L, n_pt) ** 1.3 / L ** 0.3      # uneven spacing
    a = np.exp(np.linspace(np.log(0.30), np.log(0.302), n_pt))[None, None]
    sb = SkewerBundle(spatial_points=jnp.asarray(pos), manual_scale_factors=jnp.asarray(a), a0=0.302, ICs_defined_at_index=-1)
    # cells at the points: the Voronoi cell of a point of the straight skewer is its dual segment
    cells = cells_from_values(pos[0], None, {"HI": np.full(n_pt, n)}, temperature=T, volumes=np.full(n_pt, (L / n_pt) ** 3))
    return sb, cells, pos, a


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
    assert np.isclose(float(np.asarray(p1.scale_factor[0]).min()), np.exp(np.log(a[0, 0, 0]) + 0.25 * np.log(a[0, 0, 1] / a[0, 0, 0])),
                      rtol=1e-12), "the first piece sits at d_0 / 4"
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
        pos = np.zeros((1, n_pt, 3)); pos[0, :, 0] = np.linspace(0.0, L * (1 - 1e-9), n_pt); pos[0, :, 1:] = 0.25
        a = np.exp(np.linspace(np.log(0.30), np.log(0.302), n_pt))[None, None]
        sb = SkewerBundle(spatial_points=jnp.asarray(pos), manual_scale_factors=jnp.asarray(a), a0=0.302, ICs_defined_at_index=-1)
        p = cell_pieces(cells, LYA, skewer_bundle=sb)
        cols[n_pt] = float(np.sum(np.asarray(p.number_density[0]) * np.asarray(p.length_m)))
        taus[n_pt] = np.asarray(compute_optical_depth(p, v, LYA, resolution_tolerance=None))[0, 0]
    e = [np.abs(taus[n] - taus[5121]).max() for n in (161, 321, 641)]
    assert e[0] > e[1] > e[2] and e[1] / e[2] > 3.0, f"second-order convergence, got {e}"
    assert np.allclose(list(cols.values()), cols[5121], rtol=1e-12), "the column is exact at any spacing"


def test_velocity_period_options():
    sb, f, _, _ = uniform()
    p = cell_pieces(f, LYA, skewer_bundle=sb, a_dagger=0.302)
    P = float(p.velocity_span_kms[0])
    v = (np.arange(800) + 0.5) * P / 800
    t_no = np.asarray(compute_optical_depth(p, v, LYA))
    t_T = np.asarray(compute_optical_depth(p, v, LYA, velocity_period=True))
    t_P = np.asarray(compute_optical_depth(p, v, LYA, velocity_period=P))
    assert np.array_equal(t_T, t_P), "True = the skewer's own span"
    assert np.array_equal(t_no, np.asarray(compute_optical_depth(p, v, LYA, velocity_period=False)))
    assert t_T[0, 0, 0] > t_no[0, 0, 0] and t_T[0, 0, -1] > t_no[0, 0, -1], "the wings re-enter at both ends"
    with pytest.raises(ValueError):
        compute_optical_depth(p, v, LYA, velocity_period=-1.0)


def one_piece(v_c, P, log_N=17.0):
    """One static piece whose line centre is v_c (a = a_dagger, the velocity sets v_c)."""
    a, L_m = 0.3, 1e-3 * phys.Mpc_to_m
    n_com = 10 ** log_N / (1e-3 * 3.085678e24 * a) * a ** 3
    u = -phys.c_kms * np.tanh(v_c / phys.c_kms)
    one = lambda x: jnp.full((1, 1, 1), x)
    return Pieces(a_dagger=jnp.full((1, 1), a), scale_factor=jnp.full((1, 1), a), length_m=jnp.full((1, 1), L_m),
                  temperature=one(1e4), turbulence=one(0.0), collisional=one(0.0), los_velocity=one(u), number_density=one(n_com),
                  width_kms=one(0.0), velocity_span_kms=jnp.full(1, P))


def test_the_two_periodic_effects():
    P = 1000.0
    v = (np.arange(2000) + 0.5) * P / 2000
    # (i) a line centre outside the period absorbs at v_c - P
    t = np.asarray(compute_optical_depth(one_piece(P + 40.0, P), v, LYA, velocity_period=True))[0, 0]
    assert abs(v[np.argmax(t)] - 40.0) < 1.0
    assert np.asarray(compute_optical_depth(one_piece(P + 40.0, P), v, LYA))[0, 0].max() < 1e-3 * t.max(), "no wrap: off the grid"
    # (ii) a line centre inside, at the edge: its wing re-enters at small v
    t2 = np.asarray(compute_optical_depth(one_piece(P - 30.0, P), v, LYA, velocity_period=True))[0, 0]
    t2n = np.asarray(compute_optical_depth(one_piece(P - 30.0, P), v, LYA))[0, 0]
    assert t2[v < 20].max() > 1e3 * t2n[v < 20].max()


def test_reference_frame_is_the_shifted_evaluation():
    two = LineBundle(["HI-LyA", "SiIII-1206"], ref_line_id="HI-LyA")
    sb, f1, _, _ = uniform()
    f = f1.merge_species([f1, cells_from_values(f1.positions, None, {"SiIII": f1.species["HI"]["number_density"]}, temperature=1e4,
                                                volumes=f1.volumes)])
    v = jnp.linspace(-3000.0, 1000.0, 801)
    t_ref = np.asarray(compute_optical_depth(sb, v, two, f, a_dagger=0.302, line_frame="reference", resolution_tolerance=None))
    t_si = np.asarray(compute_optical_depth(sb, v + float(two.delta_v[1]), two, f, a_dagger=0.302, resolution_tolerance=None))[1]
    t_ly = np.asarray(compute_optical_depth(sb, v, two, f, a_dagger=0.302, resolution_tolerance=None))[0]
    assert np.allclose(t_ref[1], t_si, rtol=1e-12, atol=0), "line j on the reference axis = line j at v + delta_v_j"
    assert np.array_equal(t_ref[0], t_ly), "the reference line is unchanged"
