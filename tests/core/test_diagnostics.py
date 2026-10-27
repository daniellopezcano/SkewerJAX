"""The debugging tools: ``check_skewers`` (passes on a sound route, and finds a selection that misses cells),
``return_info`` of the entry points (consistent numbers, no change of the results) and the log messages."""

import io
import logging

import jax.numpy as jnp
import numpy as np

from skewerjax.core import (
    Cosmology,
    LineBundle,
    SkewerBundle,
    cell_pieces,
    cells_from_values,
    cells_on_grid,
    check_skewers,
    compute_optical_depth,
    enable_logging,
    phys,
    select_cells,
    voronoi_volumes,
)
from skewerjax.core import selection as sel

BOX = 10.0
COSMO = Cosmology(H0=phys.H0, Omega_r_0=0.0, Omega_m_0=0.3, Omega_k_0=0.0, Omega_lambda_0=0.7)
LINES = LineBundle(["HI-LyA", "SiIII-1206"])


def _toy():
    rng = np.random.default_rng(7)
    gen = rng.uniform(0.0, BOX, (2000, 3))
    n = 1e-10 * rng.lognormal(0.0, 1.0, 2000)
    return cells_from_values(
        gen,
        BOX,
        {"HI": n, "SiIII": 1e-4 * n},
        temperature=10 ** rng.uniform(4.0, 5.0, 2000),
        velocity=rng.normal(0.0, 50.0, (2000, 3)),
        volumes=voronoi_volumes(gen, BOX, samples_per_cell=32),
    )


def _bundle(n=2000):
    t = np.linspace(0.0, 1.0, n)[:, None]
    pts = np.stack(
        [
            np.concatenate([1.3 * BOX * t, 4.0 + 0 * t, 6.0 + 0.2 * BOX * t], -1),
            np.concatenate([2.0 + 0.4 * BOX * t, BOX * t, 1.0 + 0.3 * np.sin(4 * t)], -1),
        ]
    )
    return SkewerBundle(spatial_points=jnp.asarray(pts), a0=0.25, ICs_defined_at_index=0, cosmologies=COSMO)


def test_check_skewers_passes_on_a_sound_route():
    cells, b = _toy(), _bundle()
    for dep in ("voronoi", "voronoi_crossings"):
        rep = check_skewers(cells, b, LINES, deposition=dep)
        for name in ("coverage", "column", "window"):
            assert rep.checks[name]["ok"], (dep, name, rep.checks[name]["summary"])
        assert rep.checks["column"]["missing_cells"].size == 0
        assert np.all(np.abs(rep.checks["window"]["window_fraction"] - 1.0) < 1e-3)
        assert "check_skewers" in str(rep) and "coverage" in str(rep)
    top = check_skewers(cells, b, LINES, deposition="tophat", window=False)
    assert top.checks["column"]["ok"] is None and top.checks["coverage"]["ok"] is None and top.checks["window"]["ok"] is None
    grid = cells_on_grid(
        BOX,
        {"HI": np.random.default_rng(2).lognormal(0, 1, (12, 12, 12)) * 1e-10, "SiIII": np.full((12, 12, 12), 1e-14)},
        temperature=2e4,
    )
    g = check_skewers(grid, b, LINES, window=False)
    assert g.checks["coverage"]["ok"] and g.checks["column"]["ok"]


def test_check_skewers_finds_missing_cells():
    """A selection radius too small for the Voronoi cells (a bug the route cannot see) leaves cells out of the pieces:
    the column check fails and names them."""
    cells, b = _toy(), _bundle()
    k = sel.VORONOI_FACTOR
    try:
        sel.VORONOI_FACTOR = 0.15
        rep = check_skewers(cells, b, LINES, window=False)
    finally:
        sel.VORONOI_FACTOR = k
    col = rep.checks["column"]
    assert not col["ok"] and not rep.ok and col["missing_cells"].size > 0 and col["missing_length_cMpc"] > 0
    assert "PROBLEMS FOUND" in str(rep)


def test_return_info_is_consistent_and_changes_nothing():
    cells, b = _toy(), _bundle()
    pts = np.asarray(b.spatial_points)
    near, info = select_cells(cells, pts, return_info=True)
    assert info["n_selected"] == len(near) and info["n_cells"] == len(cells)
    np.testing.assert_array_equal(near.positions, cells.take(info["indices"]).positions)
    p, pinfo = cell_pieces(cells, LINES, skewer_bundle=b, return_info=True)
    p0, parts, pinfo2 = cell_pieces(cells, LINES, skewer_bundle=b, return_parts=True, return_info=True)
    for f in p.__dataclass_fields__:
        np.testing.assert_array_equal(
            np.asarray(getattr(p, f)), np.asarray(getattr(cell_pieces(cells, LINES, skewer_bundle=b), f))
        )
    np.testing.assert_allclose(pinfo["piece_length_cMpc"], pinfo["path_length_cMpc"], rtol=1e-12)
    A = parts["absorbers"]["HI"]
    np.testing.assert_allclose(pinfo["column_cm2"]["HI"], (A.n * A.length).sum(-1) * phys.Mpc_to_cm, rtol=1e-14)
    assert set(pinfo["timings"]) == {"deposit", "absorbers", "pieces"} and pinfo["selection"]["n_selected"] == len(parts["cells"])
    P = float(np.min(p.velocity_span_kms))
    v = (np.arange(4000) + 0.5) * P / 4000  # one period of the shorter skewer
    tau, tinfo = compute_optical_depth(p, v, LINES, velocity_period=P, return_info=True)
    np.testing.assert_array_equal(np.asarray(tau), np.asarray(compute_optical_depth(p, v, LINES, velocity_period=P)))
    assert np.all(np.abs(tinfo["window_fraction"] - 1.0) < 1e-3)  # one period holds every profile
    np.testing.assert_allclose(tinfo["tau_max"], np.asarray(tau).max(-1))
    tau_b, binfo = compute_optical_depth(b, v, LINES, cells, return_info=True)
    assert "pieces_info" in binfo and binfo["shape"] == tuple(p.shape)
    np.testing.assert_array_equal(np.asarray(tau_b), np.asarray(compute_optical_depth(b, v, LINES, cells)))


def test_logging():
    cells, b = _toy(), _bundle(400)
    stream = io.StringIO()
    log = enable_logging(logging.DEBUG, stream=stream)
    try:
        compute_optical_depth(b, np.arange(0.0, 1200.0, 2.0), LINES, cells, resolution_tolerance=None)
    finally:
        log.setLevel(logging.WARNING)
        for h in [h for h in log.handlers if getattr(h, "_skewerjax", False)]:
            log.removeHandler(h)
    out = stream.getvalue()
    for name in ("select_cells:", "deposit:", "cell_pieces:", "compute_optical_depth:"):
        assert name in out, (name, out)
    quiet = io.StringIO()  # off by default: nothing is emitted
    handler = logging.StreamHandler(quiet)
    logging.getLogger("skewerjax").addHandler(handler)
    try:
        select_cells(cells, np.asarray(b.spatial_points))
    finally:
        logging.getLogger("skewerjax").removeHandler(handler)
    assert quiet.getvalue() == ""
