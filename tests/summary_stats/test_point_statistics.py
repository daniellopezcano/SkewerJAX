"""The one path of the field statistics: the gas of every source (``core.Cells``: toy grids through ``grf_cells``,
snapshots through ``gas_to_cells``) is deposited on a mesh (``field_stats.field_statistics``: n_X V at the generating
points, ``mesh.fourier_mesh``: assignment, interlacing, the window divided; Sefusatti et al. 2016) and measured by
``GridStatistics3D.measure_points``; the lattice path of the skewer samples (``GridStatistics3D.measure``,
``measure_lattice``, ``skewer_density_statistics``) takes the exact FFT and refuses a scheme.

* the defaults (interlaced PCS) and the guards (arrays are not cells; a scheme on the lattice path raises);
* the toy cells and Voronoi-like cells go through the same calls (no branch on the source; ``measure`` never called);
* the mesh path converges to the exact grid FFT (the validation reference, kept here only) at k << k_Nyq, and equals
  it times ``mesh.node_transfer`` on every mode for a toy grid of the mesh's shape;
* mass conservation of the deposit; redshift-space cells (``core.redshift_space_cells``);
* a plane wave and the shot noise; fields at points against the alias-sum expectation (``mesh.aliased_power``);
* the corner r_par = L / 2 of the map (notebook 08): the bookkeeping of its row, the exact sigma of its cells, the
  interlacing phase at the Nyquist mode and the (-1)^j term of the kernel (``xi_corner``, ``xi_map_sigma``)."""

import inspect
from unittest.mock import patch

import numpy as np
import pytest

from skewerjax.core import Cells, cells_from_values, redshift_space_cells
from skewerjax.simulations import ToyBoxes
from skewerjax.summary_stats import (
    GridStatistics3D,
    aliased_power,
    box_grid,
    cell_weights,
    decompose_1d,
    field_statistics,
    fourier_mesh,
    grid_limits,
    grid_power,
    lattice_grid,
    measure_lattice,
    node_transfer,
    on_lattice,
    on_points,
    paint,
    skewer_density_statistics,
    skewer_p1d,
    skewer_xi1d,
    xi_corner,
)
from skewerjax.summary_stats import grid3d as G3
from skewerjax.summary_stats import mesh as M

from helpers import raises

BOXES = ToyBoxes(box_cMpc=64.0, n_cells=64, n_real=4, r_bao=20.0, sigma_bao=2.0, bao_ratio=0.01)
SMALL = ToyBoxes(box_cMpc=32.0, n_cells=32, n_real=3, r_bao=10.0, sigma_bao=2.0, bao_ratio=0.01)


def _grid(n=16, L=64.0):
    lim = grid_limits(n, L)
    return box_grid(
        n, L, k_edges=np.linspace(0.5 * lim["k_min"], lim["k_max"], 9), r_edges=np.arange(0, L / 2 + 0.1, 4.0), log=None
    )


def _voronoi_like(n=4000, L=32.0, seed=3):
    """Cells at random generating points with random volumes (a stand-in for a snapshot's cells): two species."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, L, (n, 3))
    vol = rng.uniform(0.5, 1.5, n) * L**3 / n
    nh = rng.lognormal(0.0, 0.5, n)
    v = rng.normal(0, 100.0, (n, 3))
    return cells_from_values(x, L, {"HI": nh, "SiIII": 0.1 * nh}, temperature=1e4, velocity=v, volumes=vol)


def test_the_mesh_path_defaults_to_interlaced_pcs():
    for fn in (paint, fourier_mesh, field_statistics, GridStatistics3D.measure_points, node_transfer):
        par = inspect.signature(fn).parameters
        assert par["mas"].default == "pcs", fn.__name__
        assert "interlace" not in par or par["interlace"].default is True, fn.__name__


def test_the_skewer_statistics_refuse_mass_assignment():
    """The statistics of skewer samples take neither mas nor interlace, whatever the value, and say why."""
    gs = _grid()
    d = np.random.default_rng(0).normal(size=gs.shape)
    lat = lattice_grid(4, 4, 32, 64.0, 16.0, k_edges=np.linspace(0.05, 0.2, 4))
    cols = [{"HI": np.ones((4, 16, 4)), "SiIII": np.ones((4, 16, 4))}]
    tau = np.full((2, 3, 16), 0.1)
    calls = (
        (gs.measure, (d,)),
        (measure_lattice, (d.transpose(0, 2, 1).reshape(-1, 16), gs)),
        (skewer_density_statistics, (cols, 2, lat, 64.0)),
        (skewer_p1d, (tau, 1.0)),
        (skewer_xi1d, (tau, 1.0)),
        (decompose_1d, (tau, 1.0)),
    )
    for fn, args in calls:
        for option in (
            dict(mas="tsc"),
            dict(mas=None),
            dict(interlace=True),
            dict(interlace=False),
            dict(mas="pcs", interlace=True),
        ):
            raises(ValueError, "no mass assignment", fn, *args, **option)
        par = inspect.signature(fn).parameters
        assert "mas" not in par and "interlace" not in par, fn.__name__


def test_the_skewer_path_never_calls_the_mesh():
    """Every statistic of notebooks 10 and 11 on skewer samples, with the mesh replaced by a spy: nothing is deposited. The
    positive control shows that the spy does catch the mesh path. The line-of-sight sizes: n_cells points_per_cell points (the
    density, the periodic end point dropped) and N_pix pixels (the overflux); across, the skewer spacing."""
    N, p, s, n_pix, L = 16, 4, 2, 32, 64.0
    n_t, d_perp = N // s, s * L / N
    rng = np.random.default_rng(5)
    n_hi = np.exp(0.3 * rng.normal(size=(n_t, N, n_t)))
    cols = [{"HI": n_hi, "SiIII": 0.1 * n_hi}]
    k_edges = np.linspace(2 * np.pi / L, np.pi / d_perp, 5)
    g_dens, g_flux = (lattice_grid(n_t, n_t, n, L, d_perp, k_edges=k_edges) for n in (N * p, n_pix))
    tau = rng.uniform(0.01, 1.0, (2, n_t * n_t, n_pix))
    delta = np.exp(-tau) / np.exp(-tau).mean(axis=-1, keepdims=True) - 1.0

    calls = []

    def spy(*args, **kwargs):
        calls.append(args)
        raise AssertionError("the mesh was called")

    names = ("paint", "fourier_mesh", "_paint", "_paint_chunked", "_fourier_mesh", "_finish", "mas_window")
    with patch.multiple(M, **{n: spy for n in names}), patch.object(G3, "fourier_mesh", spy):
        dens = skewer_density_statistics(cols, p, g_dens, L)
        flux = measure_lattice(delta[0], g_flux)
        skewer_p1d(delta, 1.0)
        skewer_xi1d(delta, 1.0)
        decompose_1d(tau, 1.0)
        assert not calls, "the skewer path never calls the mesh"
        raises(AssertionError, "the mesh was called", g_flux.measure_points, np.zeros((4, 3)), np.ones(4))
        assert calls, "the control: the spy catches the mesh path"

    assert on_points(n_hi, p).shape == (n_t, N * p, n_t), "n_cells x points_per_cell points, the end point dropped"
    assert g_dens.shape == (n_t, N * p, n_t) and g_flux.shape == (n_t, n_pix, n_t)
    for grid, n_los in ((g_dens, N * p), (g_flux, n_pix)):
        np.testing.assert_allclose(np.asarray(grid.dx), [d_perp, L / n_los, d_perp], err_msg="the spacings of the lattice")
        rows = grid.axes()["rpar"]
        assert rows.size == n_los // 2 + 1, "one row per separation"
        np.testing.assert_allclose(np.diff(rows), L / n_los)
    assert dens["density_p1d"]["k"].size == N * p // 2 + 1
    # the samples themselves, with no window: the line-of-sight column of xi is the plain pair average
    d = on_lattice(delta[0], n_t, n_t)
    d = d - d.mean()
    for j in (1, 5, n_pix // 2):
        assert flux["xi_map"][j, 0] == pytest.approx(np.mean(d * np.roll(d, -j, axis=1)), rel=1e-10), j
    assert flux["xi_iso"].shape == g_flux.axes()["r_iso"].shape


def test_the_fields_enter_as_cells():
    gs = _grid()
    d = np.random.default_rng(0).normal(size=gs.shape)
    raises(ValueError, "mas must be one of", gs.measure_points, np.zeros((5, 3)), np.ones(5), mas="sph")
    raises(ValueError, "(N, 3)", gs.measure_points, np.zeros((5, 2)), np.ones(5))
    raises(TypeError, "core.Cells", field_statistics, d, gs, species="HI")
    raises(ValueError, "not in the cells", field_statistics, SMALL.cells(0), _grid(16, 32.0), species="CIV", log=None)
    raises(ValueError, "box", field_statistics, SMALL.cells(0), _grid(16, 64.0), species="HI", log=None)
    assert not hasattr(M, "fourier_grid"), "the exact grid FFT is a test reference only"


def test_toy_and_simulation_cells_take_the_same_calls():
    """One path, no branch on the source: the same functions, with the same options, for the toy grid cells and for
    Voronoi-like cells; the grid-native estimator is never called."""
    calls = []
    real_mp, real_fm, real_m = G3.GridStatistics3D.measure_points, G3.fourier_mesh, G3.GridStatistics3D.measure

    def spy_mp(self, x, v, **kw):
        calls.append(("measure_points", tuple(sorted(kw.items()))))
        return real_mp(self, x, v, **kw)

    def spy_fm(x, w, n, L, **kw):
        calls.append(("fourier_mesh", tuple(sorted(kw.items()))))
        return real_fm(x, w, n, L, **kw)

    def spy_m(self, *a, **kw):
        calls.append(("measure",))
        return real_m(self, *a, **kw)

    G3.GridStatistics3D.measure_points, G3.fourier_mesh, G3.GridStatistics3D.measure = spy_mp, spy_fm, spy_m
    try:
        gs = _grid(16, 32.0)
        toy = field_statistics(SMALL.cells(0), gs, species=["HI", "SiIII"], log=None)
        toy_calls, calls[:] = list(calls), []
        sim = field_statistics(_voronoi_like(), gs, species=["HI", "SiIII"], log=None)
    finally:
        G3.GridStatistics3D.measure_points, G3.fourier_mesh, G3.GridStatistics3D.measure = real_mp, real_fm, real_m
    assert toy_calls == calls and len(calls) == 4 and ("measure",) not in calls
    assert set(toy) == set(sim) == {"HI", "SiIII"} and toy["HI"]["mas"] == "pcs" and toy["HI"]["interlace"] is True
    np.testing.assert_allclose(toy["HI"]["P_iso"], toy["SiIII"]["P_iso"], rtol=1e-13)  # n_SiIII = r n_HI: one delta


def test_the_mesh_path_converges_to_the_exact_grid_fft():
    """Toy cells at the nodes of a grid of the mesh's shape: on every mode, |F_mesh|^2 = node_transfer x |F_grid|^2
    (the exact FFT of the grid, the validation reference); binned, |P_mesh / P_grid - 1| < 2e-3 below k_Nyq / 2 and
    < 3e-2 below k_Nyq (interlaced PCS), larger without interlacing; the field statistics carry the same."""
    b, gs = SMALL, _grid(32, 32.0)
    T = np.asarray(node_transfer(32, 32.0, mas="pcs", interlace=True))
    c, d = b.cells(0), b.delta(0)
    F, _ = fourier_mesh(np.asarray(c.positions), cell_weights(c, "HI"), 32, 32.0)
    G = np.fft.rfftn(d / (1 + d.mean()))  # the mesh holds n / mean(n) = (1 + delta) / (1 + mean(delta))
    live = np.ones(G.shape, bool)
    live[0, 0, 0] = False
    np.testing.assert_allclose(np.abs(np.asarray(F)[live]) ** 2, T[live] * np.abs(G[live]) ** 2, rtol=1e-9)
    exact = gs.measure(d / (1 + d.mean()))  # reference (the lattice estimator on the grid's own nodes)
    mesh = field_statistics(c, gs, species="HI", log=None)
    k = mesh["axes"]["k_iso"]
    r = mesh["P_iso"] / exact["P_iso"] - 1
    knyq = np.pi / b.dx
    assert np.all(np.abs(r[k < knyq / 2]) < 2e-3) and np.all(np.abs(r[k < knyq]) < 3e-2), r
    assert np.max(np.abs(r[k < knyq / 2])) > 1e-7, "a bias, not an identity: the mesh differs from the grid"
    plain = gs.measure_points(np.asarray(c.positions), cell_weights(c, "HI"), interlace=False)
    assert np.nanmax(np.abs(plain["P_iso"] / exact["P_iso"] - 1)[k < knyq]) > np.nanmax(np.abs(r[k < knyq])), "interlacing helps"
    th = gs.theory(grid_power(b.grf().power, gs.shape, gs.L))  # the expectation the statistics carry
    res = field_statistics((b.cells(i) for i in range(b.n_real)), gs, species="HI", power=b.grf().power, log=None)
    np.testing.assert_array_equal(res["theory"]["P_iso"], th["P_iso"])
    assert res["n_real"] == 3 and res["n_mesh"] == (32,) * 3 and res["k_nyq"] == pytest.approx(knyq)


def test_mass_conservation_of_the_deposit():
    """The mesh holds every particle: sum(paint(n V)) = sum(n V), for the toy cells and for Voronoi-like cells, with
    every scheme, chunked or not; the deposit of each cell is n V over its mean."""
    for c in (SMALL.cells(1), _voronoi_like()):
        nV = np.asarray(c.species["HI"]["number_density"]) * np.asarray(c.volumes)
        w = cell_weights(c, "HI")
        assert w.mean() == pytest.approx(1.0, rel=1e-14) and np.allclose(w * nV.mean(), nV, rtol=1e-14)
        for mas in M.MAS_SCHEMES:
            for chunk in (None, 997):
                m = np.asarray(paint(np.asarray(c.positions), nV, 16, c.box, mas, chunk=chunk))
                assert abs(m.sum() / nV.sum() - 1) < 1e-13, (mas, chunk)


def test_redshift_space_cells():
    c = _voronoi_like(500)
    aH = 100.0
    for ax in range(3):
        s = redshift_space_cells(c, ax, aH)
        want = np.mod(np.asarray(c.positions)[:, ax] + np.asarray(c.species["HI"]["velocity"])[:, ax] / aH, c.box)
        np.testing.assert_allclose(np.asarray(s.positions)[:, ax], want, rtol=0, atol=1e-12)
        others = [j for j in range(3) if j != ax]
        np.testing.assert_array_equal(np.asarray(s.positions)[:, others], np.asarray(c.positions)[:, others])
        assert s.volumes is c.volumes and isinstance(s, Cells) and s.grid is None
    zero = redshift_space_cells(SMALL.cells(0), 1, aH)  # zero velocities: the same points
    np.testing.assert_array_equal(np.asarray(zero.positions), np.asarray(SMALL.cells(0).positions))
    raises(ValueError, "axis", redshift_space_cells, c, 3, aH)


def test_a_plane_wave_through_the_mesh_and_the_shot_noise():
    """A plane wave sampled by jittered random points, measured through the mesh, equals its measurement on the grid's
    nodes (1 %, the sampling noise) in the bins that hold it; ``shot_noise`` lowers every P bin by exactly its value."""
    gs, rng = _grid(), np.random.default_rng(1)
    q, n, L = 4, 16, 64.0
    g = np.arange(n * q) * L / (n * q)
    X = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    X = X + rng.uniform(0, L / (n * q), X.shape)  # one random point in every sub-cell (jittered)
    k0 = 2 * np.pi * np.array([2, 3, 1]) / L
    on_nodes = np.cos(np.stack(np.meshgrid(*[np.arange(n) * L / n] * 3, indexing="ij"), -1) @ k0 + 0.2)
    ref = gs.measure(on_nodes)
    j = int(np.nanargmax(ref["P_iso"]))
    for mas in ("cic", "tsc", "pcs"):
        got = gs.measure_points(X, np.cos(X @ k0 + 0.2), mas=mas, interlace=True)
        assert got["P_iso"][j] == pytest.approx(ref["P_iso"][j], rel=1e-2), mas
        np.testing.assert_allclose(got["xi_iso"], ref["xi_iso"], atol=5e-3)  # xi = cos(k0.r) / 2 averaged
        less = gs.measure_points(X, np.cos(X @ k0 + 0.2), mas=mas, interlace=True, shot_noise=0.75)
        for key in ("P_iso", "P_map", "P_mu"):
            ok = np.isfinite(got[key])
            np.testing.assert_allclose(less[key][ok], got[key][ok] - 0.75, rtol=0, atol=1e-9 * np.nanmax(np.abs(got[key])))


def test_points_against_the_aliased_expectation():
    """Fields of a toy GRF known at the 64^3 nodes of their grid, measured through a 16^3 mesh: with every scheme and
    with or without interlacing, P(|k|) follows the alias-sum expectation within its errors (|pull| < 3.5 in every
    bin for TSC and PCS, below k_Nyq / 2 for NGP and CIC), while the same fields decimated to the 16^3 nodes (no
    assignment) carry the aliased power."""
    b, g = BOXES, BOXES.grf()
    gs = _grid()
    x1 = np.arange(b.n_cells) * b.dx
    X = np.stack(np.meshgrid(x1, x1, x1, indexing="ij"), -1).reshape(-1, 3)
    th = {}
    for mas in ("ngp", "cic", "tsc", "pcs"):
        for interlace in (False, True):
            P = np.stack(
                [gs.measure_points(X, b.delta(i).ravel(), mas=mas, interlace=interlace)["P_iso"] for i in range(b.n_real)]
            )
            m, e = P.mean(0), P.std(0, ddof=1) / np.sqrt(b.n_real)
            th = gs.theory(aliased_power(g.power, gs.shape, gs.L, mas=mas, interlace=interlace))["P_iso"]
            pull = (m - th) / e
            # The points sit at 4 fixed offsets in every node's cell: their kernel is W aliased at 4 k_s, W(k) +
            # W(k -+ 4 k_s) + ..., a few % above W near k_Nyq for the kinked NGP and CIC (|W(k_Nyq - 4 k_s) / W(k_Nyq)|
            # = 1/7 per order), < 0.5 % for TSC and PCS: NGP and CIC are checked below k_Nyq / 2 only
            low = gs.k_iso < (np.inf if mas in ("tsc", "pcs") else 0.5 * np.pi / gs.dx[0])
            assert np.all(np.abs(pull[low]) < 3.5), (mas, interlace, pull)
    dec = np.stack([gs.measure(b.delta(i)[::4, ::4, ::4])["P_iso"] for i in range(b.n_real)])
    th = gs.theory(grid_power(g.power, gs.shape, gs.L))["P_iso"]
    assert np.max(np.abs(dec.mean(0) - th) / (dec.std(0, ddof=1) / np.sqrt(b.n_real))) > 5, "decimation aliases"


def test_the_row_at_half_the_box_is_counted_once():
    """r_par = L / 2 (the separations +- N / 2 are one): one row of half the lags, in the measurement and the expectation
    alike, the pair count and the direct sum over the modes agreeing with the inverse transform (notebook 08, the corner)."""
    n, L = 16, 64.0
    grid = _grid(n, L)
    d = np.random.default_rng(1).normal(size=(n, n, n))
    d -= d.mean()
    F2 = np.abs(np.fft.rfftn(d)) ** 2
    F2[0, 0, 0] = 0.0
    xi = np.fft.irfftn(F2, s=(n, n, n)) / n**3
    assert xi[0, n // 2, 0] == pytest.approx(np.mean(d * np.roll(d, -n // 2, axis=1)), rel=1e-12), (
        "the lag (0, N / 2, 0): the pairs"
    )
    mult = np.broadcast_to(G3._rfft_weights(n), F2.shape)
    my = (-1.0) ** np.arange(n)
    direct = np.sum(mult * F2 * my[None, :, None]) / n**6
    assert xi[0, n // 2, 0] == pytest.approx(direct, rel=1e-12), "(1 / V) sum_k P(k) (-1)^m_y"
    idx = np.asarray(grid._idx_r["map"])
    a = box_grid(n, L, k_edges=grid.k_edges, r_edges=grid.r_edges, log=None)
    rows = np.bincount(idx[idx >= 0] // (a.rperp_edges.size - 1), minlength=a.rpar_edges.size - 1)
    assert rows[-1] * 2 == rows[-2] and a.rpar[-1] == pytest.approx(L / 2), "the last row holds half the lags of the others"
    assert np.asarray(grid.axes()["n_lags_xi_map"])[-1, 0] * 2 == np.asarray(grid.axes()["n_lags_xi_map"])[-2, 0]


def test_the_exact_sigma_of_the_map_cells():
    """``xi_map_sigma`` against the covariance of the isotropic bins and a Monte Carlo; a lag at L / 2 has twice the
    variance of a generic lag (2 r = 0 modulo L)."""
    n, L = 16, 64.0
    k_edges = np.linspace(2 * np.pi / L * 0.5, np.pi * n / L, 9)
    grid = box_grid(n, L, k_edges=k_edges, r_edges=np.arange(0, L / 2 + L / n, L / n), log=None)
    power = lambda k: 500.0 * np.maximum(k, 1e-3) ** -1.4  # noqa: E731
    P = np.asarray(grid_power(power, grid.shape, grid.L)).copy()
    P[0, 0, 0] = 0.0
    nr, nc = grid.rpar_edges.size - 1, grid.rperp_edges.size - 1

    def cell(jr, ic):
        w = np.zeros((nr, nc))
        w[jr, ic] = 1.0
        return grid.xi_map_sigma(P, w)

    assert cell(nr - 1, 0) ** 2 == pytest.approx(2.0 * np.sum(np.broadcast_to(G3._rfft_weights(n), P.shape) * P**2) / grid.V**2)
    rng = np.random.default_rng(5)
    samples = []
    for _ in range(600):
        f = np.fft.rfftn(rng.normal(size=grid.shape)) * np.sqrt(P / grid.dV)
        samples.append(np.fft.irfftn(np.abs(f) ** 2, s=grid.shape) * grid.dV / grid.N / grid.dV)
    mc = np.array(samples)
    idx = np.asarray(grid._idx_r["map"]).reshape(grid.shape)
    for jr, ic in ((nr - 1, 0), (nr - 3, 2), (nr // 2, 1)):
        vals = mc[:, idx == jr * nc + ic].mean(axis=1)
        assert vals.std(ddof=1) == pytest.approx(cell(jr, ic), rel=0.12), (jr, ic)
    with pytest.raises(ValueError, match="shape of the map"):
        grid.xi_map_sigma(P, np.zeros(3))


def test_the_corner_of_the_map_is_read_the_same_for_toy_and_simulation():
    """``xi_corner``: the offset against the expectation, its error and its exact sigma; through the mesh at the nodes the
    (-1)^j term of the column r_perp = 0 is the one of ``node_transfer``; without an expectation the value itself."""
    box = SMALL
    grid = box_grid(box.n_cells, box.box_cMpc, k_edges=np.linspace(0.3, 2.9, 9), r_edges=np.arange(0, 16.5, 1.0), log=None)
    g = box.grf()
    res = field_statistics((box.cells(i) for i in range(3)), grid, species="HI", power=g.power, log=None)
    c = xi_corner(res, grid=grid, power=g.power)
    assert c["offset"] == pytest.approx(c["value"] - c["expected"], rel=1e-9) and c["each"].shape == (3,)
    assert c["offset"] == pytest.approx(c["each"].mean()) and c["sigma"] > 0 and c["error"] > 0
    assert c["rows"][-1] == pytest.approx(16.0) and np.all(c["columns"] < 32.0 / 15.0)
    no_th = {k: v for k, v in res.items() if k != "theory"}
    d = xi_corner(no_th)
    assert np.isnan(d["expected"]) and np.isnan(d["sigma"]) and d["offset"] == pytest.approx(d["value"])


def test_interlacing_at_the_nyquist_mode():
    """numpy keeps only k = -pi / H, where the phase of the interlaced mesh is -i (S16 eq. 36 with the odd image +pi / H).
    A wave cos(pi x / H + phi) sampled by points off the nodes gives the amplitude of the k = -pi / H component, a / 2
    e^{-i phi}, whatever phi; the mesh alone sees only cos(phi). On the nodes the shifted PCS mesh vanishes there (K = 0),
    and the mode is the unshifted one over two."""
    n, L = 32, 32.0
    H = L / n
    rng = np.random.default_rng(3)
    x = rng.uniform(0, L, 2_000_000)
    k_ny = -np.pi / H
    a = 0.1
    for phi in (0.0, 0.7, np.pi / 2):
        w = 1.0 + a * np.cos(k_ny * x + phi)
        F, _ = fourier_mesh(x, w, n, L, mas="pcs", interlace=True)
        got = complex(F[n // 2]) / x.size
        assert abs(got - 0.5 * a * np.exp(-1j * phi)) < 3e-3, phi
        F0, _ = fourier_mesh(x, w, n, L, mas="pcs", interlace=False)
        assert abs(abs(complex(F0[n // 2]) / x.size) - a * abs(np.cos(phi))) < 3e-3, phi
    nodes = np.arange(n) * H
    F1 = np.fft.rfft(np.asarray(paint(nodes, (-1.0) ** np.arange(n), n, L, mas="pcs", shift=0.5)))
    assert abs(F1[n // 2]) < 1e-12 * n, "the PCS kernel on half nodes vanishes at the Nyquist mode"


def test_the_kernel_of_the_nodes_alternates_and_zeroing_the_nyquist_planes_does_not_help():
    """Through the mesh at the nodes, xi - xi_exact = (1/V) sum_k (T - 1) P cos(k.r) alternates as (-1)^j along r_perp = 0
    (the kink of T across k_y = pi / H), small next to the expectation; removing the Nyquist planes adds a far larger
    (-1)^j term to the expectation itself (the sum of P over the plane), so it is not a remedy."""
    n, L = 64, 150.0
    box = ToyBoxes(box_cMpc=L, n_s=-0.4, n_cells=n, n_real=1, r_bao=40.0, sigma_bao=1.0, bao_ratio=0.001)
    P = np.asarray(grid_power(box.grf().power, (n, n, n), L)).copy()
    P[0, 0, 0] = 0.0
    T = np.asarray(node_transfer(n, L, mas="pcs", interlace=True))
    dV = (L / n) ** 3
    col = lambda Q: np.fft.irfftn(Q, s=(n, n, n))[0, : n // 2 + 1, 0] / dV  # noqa: E731
    d = col((T - 1) * P)
    tail = d[n // 2 - 8 :]
    assert np.all(np.sign(tail) == (-1.0) ** np.arange(n // 2 - 8, n // 2 + 1) * np.sign(tail[-1])), "(-1)^j"
    Z = P.copy()
    Z[:, n // 2, :] = 0.0
    z = col(Z) - col(P)
    assert np.abs(z[-8:]).min() > 5 * np.abs(tail).max(), "the plane removed: a larger alternating term"
