"""The input checks of the entry points (``core.validation``; cross-cutting: cells, skewers, cosmology, lines, selection,
pieces, tau): informative errors for bad inputs (shapes, NaN / inf, out-of-range values, empty
selections, missing species or lines, inconsistent scale factors), the periodic edge cases (paths through the faces any
number of times, along a face, exactly at the box edge, coordinates just outside the box) and the precision checks."""

import warnings

import jax
import jax.numpy as jnp
import numpy as np

from skewerjax.core import (
    Cells,
    Cosmology,
    LineBundle,
    Pieces,
    SkewerBundle,
    cell_pieces,
    cells_from_values,
    cells_on_grid,
    compute_optical_depth,
    phys,
    select_cells,
    validation,
    voronoi_volumes,
)

from helpers import raises as _raises

BOX = 8.0
COSMO = Cosmology(H0=phys.H0, Omega_r_0=0.0, Omega_m_0=0.3, Omega_k_0=0.0, Omega_lambda_0=0.7)
LYA = LineBundle(["HI-LyA"])


def _toy(n=1500, seed=3):
    rng = np.random.default_rng(seed)
    gen = rng.uniform(0.0, BOX, (n, 3))
    return cells_from_values(
        gen,
        BOX,
        {"HI": 1e-10 * rng.lognormal(0, 1, n)},
        temperature=10 ** rng.uniform(3.8, 5, n),
        velocity=rng.normal(0, 40, (n, 3)),
        volumes=voronoi_volumes(gen, BOX, samples_per_cell=16),
        ids=np.arange(n),
    )


def _bundle(points, a0=0.25):
    return SkewerBundle(spatial_points=jnp.asarray(points), a0=a0, ICs_defined_at_index=0, cosmologies=COSMO)


def _line(n=600, origin=(0.3, 4.1, 2.7), direction=(1.0, 0.0, 0.0), length=BOX):
    t = np.linspace(0.0, 1.0, n)[:, None]
    return (np.asarray(origin) + length * t * np.asarray(direction) / np.linalg.norm(direction))[None]


def test_cells_values():
    n = 10
    pos, vol = np.random.default_rng(0).uniform(0, BOX, (n, 3)), np.full(n, 1.0)
    f = lambda **kw: {"HI": {"number_density": np.ones(n), "temperature": np.full(n, 1e4), "velocity": np.zeros((n, 3)), **kw}}
    Cells(pos, vol, f(), BOX)  # valid
    bad = pos.copy()
    bad[3, 1] = np.nan
    e = _raises(ValueError, "positions", Cells, bad, vol, f(), BOX)
    assert "index (3, 1)" in str(e)  # the first bad entry is named
    _raises(ValueError, "volumes", Cells, pos, np.zeros(n), f(), BOX)
    _raises(ValueError, "temperature", Cells, pos, vol, f(temperature=-np.ones(n)), BOX)
    _raises(ValueError, "number_density", Cells, pos, vol, f(number_density=np.full(n, np.inf)), BOX)
    _raises(ValueError, "velocity", Cells, pos, vol, f(velocity=np.full((n, 3), np.nan)), BOX)
    _raises(ValueError, "turbulence", Cells, pos, vol, f(turbulence=-np.ones(n)), BOX)
    _raises(ValueError, "box", Cells, pos, vol, f(), -1.0)
    _raises(ValueError, "shape", Cells, pos, vol, f(temperature=np.ones(n + 1)), BOX)
    Cells(pos, vol, f(number_density=-np.ones(n)), BOX)  # negative densities: allowed (grf_cells warns)
    c = Cells(pos, vol, f(), BOX)
    _raises(KeyError, "SiIII", c.select_species, "SiIII")


def test_bundle_cosmology_and_lines():
    pts = _line()
    _raises(ValueError, "at least 2 points", SkewerBundle, spatial_points=pts[:, :1], cosmologies=COSMO)
    bad = pts.copy()
    bad[0, 5, 2] = np.inf
    _raises(ValueError, "spatial_points", _bundle, bad)
    _raises(ValueError, "a0", _bundle, pts, a0=0.0)
    _raises(ValueError, "manual_scale_factors", SkewerBundle, spatial_points=pts, manual_scale_factors=np.ones((1, 1, 7)))
    a = np.full((1,) + pts.shape[:2], 0.3)
    a[0, 0, -1] = 0.0
    _raises(ValueError, "scale factor must stay finite", SkewerBundle, spatial_points=pts, manual_scale_factors=a)
    _raises(ValueError, "H0", Cosmology, H0=0.0, Omega_r_0=0.0, Omega_m_0=0.3, Omega_k_0=0.0, Omega_lambda_0=0.7)
    _raises(ValueError, "Omega_m_0", Cosmology, H0=phys.H0, Omega_r_0=0.0, Omega_m_0=np.nan, Omega_k_0=0.0, Omega_lambda_0=0.7)
    _raises(ValueError, "at least one line", LineBundle, [])
    _raises(TypeError, "not a string", LineBundle, "HI-LyA")
    _raises(ValueError, "Unknown line", LineBundle, ["HI-NotALine"])


def test_selection_errors():
    c = _toy()
    _raises(ValueError, "paths", select_cells, c, np.full((1, 5, 3), np.nan))
    _raises(ValueError, "no cells", select_cells, c.take(np.zeros(0, dtype=int)), _line())
    far = cells_from_values(c.positions + 100.0, None, {"HI": 1.0}, volumes=np.full(len(c), 1e-3))
    e = _raises(RuntimeError, "no gas cell", select_cells, far, _line())
    assert "selection radius" in str(e) and "same units" in str(e)  # says why: the closest cell and its radius


def test_route_errors():
    c, b = _toy(), _bundle(_line())
    v = np.arange(0.0, 900.0, 2.0)
    _raises(ValueError, "the cells hold no ['SiIII']", cell_pieces, c, LineBundle(["SiIII-1206"]), skewer_bundle=b)
    _raises(ValueError, "deposition", cell_pieces, c, LYA, skewer_bundle=b, deposition="sph")
    p, parts = cell_pieces(c, LYA, skewer_bundle=b, return_parts=True)
    _raises(ValueError, "segments", cell_pieces, c, LYA, points=_line(n=50), frame=parts["frame"], segments=parts["segments"])
    _raises(ValueError, "velocities_kms", compute_optical_depth, p, np.array([0.0, np.nan]), LYA)
    _raises(ValueError, "velocities_kms", compute_optical_depth, p, np.zeros((3, 10)), LYA)
    _raises(ValueError, "line_frame", compute_optical_depth, p, v, LYA, line_frame="global")
    _raises(ValueError, "voigt", compute_optical_depth, p, v, LYA, voigt="scipy_wofz")
    _raises(ValueError, "detuning", compute_optical_depth, p, v, LYA, detuning="quadratic")
    _raises(ValueError, "piece_chunk", compute_optical_depth, p, v, LYA, piece_chunk=0)
    _raises(ValueError, "velocity_period", compute_optical_depth, p, v, LYA, velocity_period=-5.0)
    _raises(ValueError, "velocity_period", compute_optical_depth, p, v, LYA, velocity_period=np.nan)
    _raises(ValueError, "the pieces hold 1 lines", compute_optical_depth, p, v, LineBundle(["HI-LyA", "HI-LyB"]))
    bad = Pieces(
        **{**{f: getattr(p, f) for f in p.__dataclass_fields__}, "number_density": p.number_density.at[0, 0, 7].set(jnp.nan)}
    )
    e = _raises(ValueError, "Pieces.number_density", compute_optical_depth, bad, v, LYA)
    assert "(0, 0, 7)" in str(e)
    neg = Pieces(**{**{f: getattr(p, f) for f in p.__dataclass_fields__}, "scale_factor": -p.scale_factor})
    _raises(ValueError, "Pieces.scale_factor", compute_optical_depth, neg, v, LYA)


def test_checks_are_skipped_under_tracing():
    """The eager checks never break jit / grad: traced values are not inspected."""
    p = cell_pieces(_toy(), LYA, skewer_bundle=_bundle(_line()))
    v = jnp.arange(0.0, 900.0, 3.0)
    tau = jax.jit(lambda q: compute_optical_depth(q, v, LYA, resolution_tolerance=None, velocity_period=500.0))(p)
    ref = compute_optical_depth(p, v, LYA, resolution_tolerance=None, velocity_period=500.0)
    np.testing.assert_allclose(np.asarray(tau), np.asarray(ref), rtol=1e-12, atol=0)
    g = jax.grad(
        lambda s: compute_optical_depth(
            Pieces(**{**{f: getattr(p, f) for f in p.__dataclass_fields__}, "number_density": s * p.number_density}),
            v,
            LYA,
            resolution_tolerance=None,
        ).sum()
    )(1.0)
    assert np.isfinite(float(g)) and float(g) > 0


def test_periodic_paths():
    """The same gas along any periodic image of a path, through the faces several times, along a face and from the box
    edge: the pieces tile the path, and images give the same pieces (to round-off) and tau."""
    c = _toy()
    v = np.arange(0.0, 2500.0, 2.0)
    base = _line(n=900, origin=(0.3, 4.1, 2.7), direction=(1.0, 0.31, 0.17), length=2.6 * BOX)  # 2.6 boxes, oblique
    p0 = cell_pieces(c, LYA, skewer_bundle=_bundle(base))
    for shift in (np.array([BOX, 0, 0]), np.array([-3 * BOX, 2 * BOX, BOX])):
        p1 = cell_pieces(c, LYA, skewer_bundle=_bundle(base + shift))
        np.testing.assert_allclose(
            np.asarray(p1.length_m), np.asarray(p0.length_m), rtol=0, atol=1e-9 * float(np.max(p0.length_m))
        )
        np.testing.assert_array_equal(np.asarray(p1.number_density), np.asarray(p0.number_density))
        t0, t1 = (np.asarray(compute_optical_depth(q, v, LYA, resolution_tolerance=None)) for q in (p0, p1))
        assert np.max(np.abs(t1 - t0)) <= 1e-9 * np.max(t0)
    L = float(np.linalg.norm(np.diff(base[0], axis=0), axis=-1).sum())
    assert abs(float(np.sum(p0.length_m)) / phys.Mpc_to_m / L - 1.0) < 1e-12
    grid = cells_on_grid(BOX, {"HI": np.random.default_rng(1).lognormal(0, 1, (8, 8, 8)) * 1e-10})
    for origin, direction in (((0.0, 0.0, 0.0), (1, 0, 0)), ((0.0, 2.0, 0.0), (0, 0, 1)), ((BOX, BOX, BOX), (-1, -1, 0.5))):
        path = _line(n=257, origin=origin, direction=direction, length=1.5 * BOX)  # along a face / an edge of the box
        pg = cell_pieces(grid, LYA, skewer_bundle=_bundle(path))
        ps = cell_pieces(grid, LYA, skewer_bundle=_bundle(path), grid_fast_path=False)
        for q in (pg, ps):  # the fast path and the general one
            assert abs(float(np.sum(q.length_m)) / phys.Mpc_to_m / (1.5 * BOX) - 1.0) < 1e-12
        col = lambda q: float(np.sum(np.asarray(q.number_density) * np.asarray(q.length_m)))
        assert abs(col(pg) / col(ps) - 1.0) < 1e-12


def test_precision_warning():
    """compute_optical_depth warns once when jax_enable_x64 is off (the package never sets it)."""
    validation._WARNED.discard("x64")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        validation.warn_if_float32("here")  # x64 on: silent
        assert not w
        jax.config.update("jax_enable_x64", False)
        try:
            validation.warn_if_float32("here")
            validation.warn_if_float32("here")
        finally:
            jax.config.update("jax_enable_x64", True)
    assert len(w) == 1 and "jax_enable_x64 is off" in str(w[0].message)
    p = cell_pieces(_toy(), LYA, skewer_bundle=_bundle(_line()))
    assert compute_optical_depth(p, np.arange(0.0, 900.0, 2.0), LYA, resolution_tolerance=None).dtype == jnp.float64


def test_resolution_warning():
    """Pieces too wide in velocity for the line widths: a warning, not a silent error."""
    c = _toy()
    coarse = _bundle(_line(n=12))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        compute_optical_depth(coarse, np.arange(0.0, 900.0, 2.0), LYA, c)
    assert any("too wide in velocity" in str(x.message) for x in w)


def test_paths_leaving_non_periodic_gas():
    """Without a periodic box, a path that runs beyond the gas would take the nearest cell's gas silently: the Voronoi
    deposition warns (``max_distance_over_size``); inside the gas it stays quiet."""
    from skewerjax.core import deposition as dep

    rng = np.random.default_rng(4)
    gen = rng.uniform(0.0, BOX, (3000, 3))
    size = np.full(3000, (BOX**3 / 3000) ** (1 / 3))
    inside = _line(n=200, origin=(1.0, 4.0, 4.0), length=6.0)
    beyond = _line(n=200, origin=(1.0, 4.0, 4.0), length=3 * BOX)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        seg = dep.voronoi_segments(inside, gen, None, gas_size=size)
        assert not w and seg.diagnostics["max_distance_over_size"] < 2.5
        dep.voronoi_segments(beyond, gen, None, gas_size=size)
    assert any("leave the (non-periodic) gas" in str(x.message) for x in w)
