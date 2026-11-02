"""The lattice of skewers through a periodic box (``skewerjax.skewers``): the columns (every stride-th, through the cell
centres), the points of the paths and the bundle built from them (the exact frame, z at y = 0), the Hubble velocity of
the line centres, and the 3D grid of the statistics on exactly that lattice in comoving units (``summary_stats.lattice_grid``,
``lattice_limits``: a cube of side L, the line of sight converted with u = period / L, transverse spacing s dx)."""

import jax.numpy as jnp
import numpy as np
import pytest

from skewerjax.core import SkewerBundle, phys, velocity_span
from skewerjax.simulations import ToyBoxes
from skewerjax.skewers import hubble_velocity, lattice_columns, lattice_points
from skewerjax.summary_stats import bao_k_edges, lattice_grid, lattice_limits

BOXES = ToyBoxes(box_cMpc=8.0, n_cells=16, n_real=3, bao_ratio=1e-3)
STRIDE, P, N_PIX = (4, 4), 80, 64


def _bundle(cols, p=P, boxes=BOXES):
    pts = lattice_points(cols, boxes.dx, boxes.box_cMpc, boxes.n_cells, p)
    return SkewerBundle(
        spatial_points=jnp.asarray(pts), cosmologies=boxes.cosmology(), a0=1 / (1 + boxes.z), x0=0.0, ICs_defined_at_index=0
    )


def test_columns_points_and_frame():
    cols = lattice_columns(BOXES.n_cells, STRIDE)
    assert cols.shape == (16, 2) and np.array_equal(cols[:4], [[0, 0], [0, 4], [0, 8], [0, 12]]), "x-major"
    assert lattice_columns(BOXES.n_cells, (3, 4)).shape == (6 * 4, 2), "ceil(N / s) per axis"
    pts = lattice_points(cols, BOXES.dx, BOXES.box_cMpc, BOXES.n_cells, P)
    assert pts.shape == (16, BOXES.n_cells * P + 1, 3)
    np.testing.assert_array_equal(pts[:, 0, 0], cols[:, 0] * BOXES.dx)  # the nodes of grf_cells: the cell centres
    np.testing.assert_array_equal(pts[:, 0, 2], cols[:, 1] * BOXES.dx)
    assert pts[0, 0, 1] == 0.0 and pts[0, -1, 1] == BOXES.box_cMpc, "from y = 0 to y = L: the same place of the box"
    sb = _bundle(cols)
    a = np.asarray(sb.scale_factors[0, 0], dtype=np.float64)
    assert a[0] == pytest.approx(1 / (1 + BOXES.z), rel=1e-12) and np.all(np.diff(a) < 0), "z at y = 0, a decreasing"
    vH = hubble_velocity(sb)
    np.testing.assert_allclose(vH, phys.c_kms * np.log(a[0] / a), rtol=1e-14)
    assert vH[-1] == pytest.approx(float(velocity_span(sb, "exact")[0]), rel=1e-12), "one box = the period"
    y = np.asarray(sb.spatial_points)[0, :, 1]
    aH_start = (vH[1] - vH[0]) / (y[1] - y[0])
    assert aH_start == pytest.approx(BOXES.aH, rel=1e-6) and (vH[-1] - vH[-2]) / (y[-1] - y[-2]) > aH_start


def test_the_3d_grid_of_the_lattice():
    """The 3D statistics on the lattice of skewers in comoving units: a cube of side L, a pixel dv spanning dv / u, the
    (k_par, k_perp) rows the lattice's own modes, the (r_par, r_perp) rows one per pixel separation, every bin inside
    the full-coverage limits (the smallest Nyquist frequency, half the box)."""
    L = BOXES.box_cMpc
    period = float(velocity_span(_bundle([[0, 0]]), "exact")[0])
    u, dv = period / L, period / N_PIX
    d_perp = np.array(STRIDE, dtype=float) * BOXES.dx
    k_f = 2 * np.pi / L
    k_nyq = min(np.pi / dv * u, *(np.pi / d_perp))  # the sphere where every direction is available
    gs = lattice_grid(4, 4, N_PIX, L, d_perp, k_edges=bao_k_edges(k_f, k_nyq, 3.0 / BOXES.sigma_bao))
    assert gs.shape == (4, N_PIX, 4) and np.allclose(gs.L, L) and gs.los_axis == 1
    assert gs.dx[1] == pytest.approx(dv / u), "a pixel spans dv / u cMpc"
    lim = lattice_limits(L, d_perp, N_PIX, period)
    assert lim["k_nyq"] == pytest.approx(k_nyq) and gs.k_edges[-1] == pytest.approx(k_nyq) and np.nanmax(gs.k_iso) <= k_nyq
    np.testing.assert_allclose(gs.kpar, k_f * np.arange(1, N_PIX // 2 + 1))  # the map: one row per mode k_par = 2 pi m / L
    half = L / 2  # xi of a periodic field: nothing beyond half the box
    for e in (gs.r_edges, gs.r_mu_edges, gs.rperp_edges):
        assert e[-1] <= half * (1 + 1e-9)
    # the (r_par, r_perp) map: one row per pixel separation j dv, j = 0 .. n_pix / 2, centred on it; r_perp columns of
    # the transverse spacing, the first holding the pairs on the same skewer (r_perp = 0)
    np.testing.assert_allclose(gs.rpar, np.arange(N_PIX // 2 + 1) * L / N_PIX, atol=1e-12)
    np.testing.assert_allclose(np.diff(gs.rperp_edges), d_perp.min())
    assert gs.rperp[0] == 0.0 and np.diff(gs.r_edges)[0] == pytest.approx(max(d_perp.max(), dv / u))
    assert lim["r_max"] == pytest.approx(half) and lim["r_perp_min"] == pytest.approx(d_perp.min())
    assert lim["k_par_max_skm"] == pytest.approx(np.pi / dv) and lim["k_par_min_skm"] == pytest.approx(2 * np.pi / period)
    np.testing.assert_allclose(lim["k_nyq_perp"], np.pi / d_perp)
    assert lim["dr_iso"] == pytest.approx(np.diff(gs.r_edges)[0])
    # explicit bins: used as given, every bin beyond the full-coverage limits cut (and reported)
    logs = []
    e = dict(
        k_edges=np.linspace(0.5 * k_f, 2 * k_nyq, 9),
        r_edges=np.arange(0.0, 0.75 * L + 2.0, 2.0),
        rperp_edges=np.arange(0.0, half + 2, 1.0),
    )
    ge = lattice_grid(4, 4, N_PIX, L, d_perp, **e, log=logs.append)
    assert ge.k_edges[-1] <= k_nyq and ge.r_edges[-1] <= half and ge.rperp_edges[-1] <= half and len(logs) == 3
    np.testing.assert_array_equal(ge.r_edges, e["r_edges"][e["r_edges"] <= half])
    assert lattice_grid(4, 4, BOXES.n_cells, L, d_perp, k_edges=e["k_edges"]).shape == (4, BOXES.n_cells, 4)
