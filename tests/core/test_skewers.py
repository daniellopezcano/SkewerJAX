"""Skewers (notebook 03): the geometry of curved paths that cross the periodic faces (unwrapped points), the scale
factors a(x) of a ``SkewerBundle`` against the Einstein-de Sitter closed form for every IC convention, the photon
directions, the input checks, and the point layouts (``axis_aligned_skewer_points``, ``plane_points``,
``grid_skewer_points``). jit / vmap of the bundle and its kernels: ``tests/core/test_jax_transformations.py``."""

import jax.numpy as jnp
import numpy as np
import pytest

from skewerjax.core import (
    Cosmology,
    SkewerBundle,
    axis_aligned_skewer_points,
    grid_skewer_points,
    line_elements,
    path_geometry,
    phys,
    plane_points,
)

BOX = 20.0
EDS = Cosmology(H0=phys.H0, Omega_r_0=0.0, Omega_m_0=1.0, Omega_k_0=0.0, Omega_lambda_0=0.0)


def helix(n=400, turns=1.5, r=3.0, length=45.0):
    """(1, n, 3) a helix along (1, 0.6, 0.4) that leaves the box [0, 20)^3 through two faces, unwrapped; exact length."""
    t = np.linspace(0.0, 1.0, n)
    axis = np.array([1.0, 0.6, 0.4]) / np.linalg.norm([1.0, 0.6, 0.4])
    e1 = np.cross(axis, [0.0, 0.0, 1.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    ph = 2 * np.pi * turns * t
    pts = np.array([12.0, 15.0, 9.0]) + length * t[:, None] * axis + r * (np.cos(ph)[:, None] * e1 + np.sin(ph)[:, None] * e2)
    return pts[None], np.hypot(length, 2 * np.pi * turns * r)


def test_curved_path_geometry_across_the_faces():
    pts, L_exact = helix()
    assert pts.max() > BOX  # leaves the box: the points stay unwrapped
    g = path_geometry(jnp.asarray(pts))
    s, w, t = np.asarray(g["s"])[0], np.asarray(g["line_elements"])[0], np.asarray(g["tangents"])[0]
    np.testing.assert_allclose(w.sum(), s[-1], rtol=1e-13)  # the dual segments tile the path
    assert abs(s[-1] / L_exact - 1) < 1e-4  # polyline -> arc length at second order
    np.testing.assert_allclose(np.linalg.norm(t, axis=-1), 1.0, rtol=1e-13)
    lo, hi = np.asarray(g["mid_lo"])[0], np.asarray(g["mid_hi"])[0]
    half = np.linalg.norm(pts[0] - lo, axis=-1) + np.linalg.norm(hi - pts[0], axis=-1)
    np.testing.assert_allclose(half, w, rtol=1e-12)  # w_i = |x_i - m_i-| + |m_i+ - x_i|
    np.testing.assert_allclose(lo[0], pts[0, 0])
    np.testing.assert_allclose(hi[-1], pts[0, -1])  # = x at the ends
    g2 = path_geometry(jnp.asarray(pts + BOX * np.array([1, -2, 0])))  # a periodic image: the same geometry
    np.testing.assert_allclose(np.asarray(g2["s"])[0], s, rtol=1e-12, atol=1e-12)
    sb = SkewerBundle(spatial_points=jnp.asarray(pts), a0=0.25, cosmologies=EDS)
    np.testing.assert_allclose(np.asarray(sb.line_elements), np.asarray(line_elements(jnp.asarray(pts))), rtol=1e-15)
    np.testing.assert_allclose(np.asarray(sb.path_lengths)[0], s, rtol=1e-15)
    np.testing.assert_allclose(np.asarray(sb.segment_midpoints[1])[0], hi, rtol=1e-15)


def test_scale_factors_match_einstein_de_sitter():
    """EdS: da/dx = (H0 / c) a^(1/2), so sqrt(a) = sqrt(a0) -/+ (H0 / 2c) d at path distance d from the IC point
    (towards the past for absorption, the future for emission), for each IC convention."""
    for ic, kind in ((-1, "absorption"), (0, "absorption"), (0, "emission")):
        _check_eds(ic, kind)


def _check_eds(ic, kind):
    pts, _ = helix(n=200)
    a0 = 0.25
    sb = SkewerBundle(
        spatial_points=jnp.asarray(pts), a0=a0, ICs_defined_at_index=ic, ICs_at_emission_or_absorption=kind, cosmologies=EDS
    )
    s = np.asarray(sb.path_lengths)[0]
    d = s[-1] - s if ic == -1 else s
    sign = -1.0 if kind == "absorption" else 1.0
    exact = (np.sqrt(a0) + sign * 0.5 * phys.H0_over_c * d) ** 2
    a = np.asarray(sb.scale_factors)[0, 0]
    np.testing.assert_allclose(a, exact, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(sb.quadri_vectors)[0, 0, :, 0], a, rtol=0)
    np.testing.assert_allclose(np.asarray(sb.quadri_vectors)[0, 0, :, 1:], pts[0], rtol=0)
    # the photon travels towards the observer (absorption: away from the IC point's past), along the path or against it
    t = np.asarray(path_geometry(jnp.asarray(pts))["tangents"])[0]
    cos = np.sum(np.asarray(sb.los_directions)[0] * t, -1)
    toward_end = (ic == -1) == (kind == "absorption")
    assert np.all(cos > 0.99) if toward_end else np.all(cos < -0.99)


def test_bundle_inputs():
    pts, _ = helix(n=50)
    a = jnp.full((1, 1, 50), 0.3)
    sb = SkewerBundle(spatial_points=jnp.asarray(pts), manual_scale_factors=a)
    assert np.array_equal(np.asarray(sb.scale_factors), np.asarray(a)) and sb.N_sk == 1 and sb.N_ppsk == 50
    with pytest.raises(ValueError):
        SkewerBundle(spatial_points=jnp.asarray(pts))  # no cosmology, no a(x)
    with pytest.raises(ValueError):
        SkewerBundle(spatial_points=jnp.asarray(pts[0]), cosmologies=EDS)  # (N, 3): no skewer axis
    with pytest.raises(ValueError):
        SkewerBundle(spatial_points=jnp.asarray(pts), ICs_defined_at_index=1, cosmologies=EDS)
    with pytest.raises(ValueError):
        SkewerBundle(spatial_points=jnp.asarray(pts), los_directions=2 * jnp.ones((1, 50, 3)), cosmologies=EDS)


def test_point_layouts():
    cofm = np.array([[3.0, 4.0, 5.0], [6.0, 7.0, 8.0], [1.0, 2.0, 9.0]])
    p = axis_aligned_skewer_points(cofm, [1, 2, 3], BOX, 0.3)
    n = int(round(BOX / 0.3))
    assert p.shape == (3, n + 1, 3)
    for k, ax in enumerate((0, 1, 2)):
        np.testing.assert_allclose(np.diff(p[k, :, ax]), BOX / n, rtol=1e-12)  # spacing that divides the box
        np.testing.assert_allclose(np.mod(p[k, -1] - p[k, 0], BOX) % BOX, 0.0, atol=1e-12)  # first = last, periodically
        other = [i for i in range(3) if i != ax]
        assert np.all(p[k, :, other] == cofm[k, other][:, None])
    q = np.asarray(plane_points(BOX, "y", 5.0, 4, 6))
    assert q.shape == (4, 6, 3) and np.all(q[..., 1] == 5.0) and q[..., 2].max() == BOX
    gpts = np.asarray(grid_skewer_points(BOX, 4, 9, direction="x"))
    assert gpts.shape == (4, 4, 9, 3)
    np.testing.assert_allclose(np.unique(gpts[..., 1]), (np.arange(4) + 0.5) * BOX / 4, rtol=1e-6)  # cell centres
    np.testing.assert_allclose(gpts[0, 0, :, 0], np.linspace(0, BOX, 9), rtol=1e-6)
