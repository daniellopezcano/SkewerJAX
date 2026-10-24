"""The piece-based optical depth (``core.pieces_from_fields``, ``core.compute_optical_depth``): dual-segment pieces at
their centroids, ``subdivide``, the ``velocity_period`` option (False / True / a number) and its two effects, and the
reference line frame."""
import numpy as np
import jax.numpy as jnp
import pytest

from skewerjax.core import (Constant, Fields, LineBundle, Pieces, SampledField, ScalarField, SkewerBundle, VectorField,
                            compute_optical_depth, phys, pieces_from_fields)

LYA = LineBundle(["HI-LyA"], ref_line_id="HI-LyA")


def uniform(n_pt=201, L=2.0, n=1e-12, T=1e4):
    """A straight skewer of manual a(x) (log-linear) through uniform gas."""
    pos = np.zeros((1, n_pt, 3)); pos[0, :, 0] = np.linspace(0.0, L, n_pt) ** 1.3 / L ** 0.3      # uneven spacing
    a = np.exp(np.linspace(np.log(0.30), np.log(0.302), n_pt))[None, None]
    sb = SkewerBundle(spatial_points=jnp.asarray(pos), manual_scale_factors=jnp.asarray(a), a0=0.302, ICs_defined_at_index=-1)
    sp = {"temperature": ScalarField(Constant(T)), "turbulence": ScalarField(Constant(0.0)), "collisional": ScalarField(Constant(0.0)),
          "number_density": ScalarField(Constant(n)), "velocity": VectorField(SampledField(np.zeros((1, n_pt, 3))))}
    return sb, Fields({"HI": sp}), pos, a


def test_dual_segments_tile_the_path_and_centroids():
    sb, f, pos, a = uniform()
    for q in (1, 3):
        p = pieces_from_fields(sb, f, LYA, a_dagger=0.302, subdivide=q)
        L = np.asarray(p.length_m[0]) / phys.Mpc_to_m
        assert np.isclose(L.sum(), pos[0, -1, 0] - pos[0, 0, 0], rtol=1e-12), "the pieces tile the path"
        x = pos[0, :, 0]
        # the scale factor at the centroid: ln a linear in s between the points
        s_c = np.concatenate([[0.0], np.cumsum(L)])[:-1] + 0.5 * L
        assert np.allclose(np.asarray(p.scale_factor[0]), np.exp(np.interp(s_c, x - x[0], np.log(a[0, 0]))), rtol=1e-12)
    p1 = pieces_from_fields(sb, f, LYA, a_dagger=0.302)
    d = np.diff(pos[0, :, 0])
    assert np.isclose(float(p1.scale_factor[0, 0]), np.exp(np.log(a[0, 0, 0]) + 0.25 * np.log(a[0, 0, 1] / a[0, 0, 0])), rtol=1e-12), \
        "the first piece sits at d_0 / 4"
    assert np.isclose(float(p1.velocity_span_kms[0]), phys.c_kms * np.log(0.302 / 0.30), rtol=1e-12)
    assert d.min() < d.max()


def test_subdivide_converges_and_keeps_the_column():
    sb, f, _, _ = uniform(n_pt=41, T=2e3)
    v = jnp.linspace(-100.0, 2100.0, 4401)                          # the whole velocity span of the skewer
    taus = {q: np.asarray(compute_optical_depth(sb, v, LYA, f, a_dagger=0.302, subdivide=q, resolution_tolerance=None))[0, 0]
            for q in (1, 2, 4, 64)}
    e = [np.abs(taus[q] - taus[64]).max() for q in (1, 2, 4)]
    assert e[0] > e[1] > e[2] and e[1] / e[2] > 3.0, f"second-order convergence, got {e}"
    cols = [taus[q].sum() for q in (1, 2, 4, 64)]
    assert np.allclose(cols, cols[-1], rtol=2e-3), "the column is the same for any subdivision"


def test_velocity_period_options():
    sb, f, _, _ = uniform()
    p = pieces_from_fields(sb, f, LYA, a_dagger=0.302)
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
    sp = f1["HI"]
    f = Fields({"HI": sp, "SiIII": sp})
    v = jnp.linspace(-3000.0, 1000.0, 801)
    t_ref = np.asarray(compute_optical_depth(sb, v, two, f, a_dagger=0.302, line_frame="reference", resolution_tolerance=None))
    t_si = np.asarray(compute_optical_depth(sb, v + float(two.delta_v[1]), two, f, a_dagger=0.302, resolution_tolerance=None))[1]
    t_ly = np.asarray(compute_optical_depth(sb, v, two, f, a_dagger=0.302, resolution_tolerance=None))[0]
    assert np.allclose(t_ref[1], t_si, rtol=1e-12, atol=0), "line j on the reference axis = line j at v + delta_v_j"
    assert np.array_equal(t_ref[0], t_ly), "the reference line is unchanged"
