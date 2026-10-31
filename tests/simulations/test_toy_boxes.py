"""The toy boxes of notebook 08 (``simulations.toy_boxes``) on a small grid: realisations regenerated from their seeds,
the gas (n_SiIII = r n_HI, T = T0 (1 + delta)^(gamma - 1), zero or linear-theory velocities, no clipping), the skewer
set-up and its derived quantities, one realisation along its skewers (tau, the resolution check, the 1D statistics by
order, the 3D statistics and P1D of the overfluxes in comoving units), the run over the realisations in memory with the
global mean of the overflux (<delta> = 0 over the ensemble, C = <delta_1 delta_2>, the exact decomposition; or the mean
at each pixel), the 3D estimator on the density sampled at the points of the skewers against its exact expectation (the
BAO peak included), the continuous xi(r), and the chi^2 of xi with its exact covariance."""

from dataclasses import replace

import numpy as np
import pytest

from skewerjax.simulations import (
    SkewerRun,
    ToyBoxes,
    bao_height,
    field_statistics,
    lattice_density_check,
    lattice_density_theory,
    run_skewers,
    skewer_box,
    xi_chi2,
    xi_subset_offsets,
)
from skewerjax.simulations.toy_boxes import OVERFLUX_FIELDS
from skewerjax.summary_stats import GRID_KEYS, ensemble_mean

BOXES = ToyBoxes(box_cMpc=8.0, n_cells=16, n_real=3, bao_ratio=1e-3)  # below R_max of this grid
SETUP = SkewerRun(stride=(4, 4), points_per_cell=80, n_pix=64, batch=8)  # 16 skewers; cells of 40 km/s: W = 0.5 km/s


def test_realisations_and_gas():
    b = BOXES
    assert np.array_equal(b.seeds, b.base_seed + np.arange(b.n_real))
    np.testing.assert_array_equal(b.delta(1), b.delta(1))  # regenerated, identical
    assert not np.allclose(b.delta(0), b.delta(1))
    assert np.isclose(float(b.grf().variance()), b.sigma_delta**2, rtol=1e-12)
    assert not b.grf().anisotropic and b.grf().A_bao == pytest.approx(b.bao_ratio * b.A_s)
    f = b.fields(0)
    np.testing.assert_allclose(f["n"]["HI"], b.n_bar_HI * (1 + f["delta"]), rtol=1e-14)
    np.testing.assert_allclose(f["n"]["SiIII"], b.siiii_to_hi * f["n"]["HI"], rtol=1e-14)
    np.testing.assert_allclose(f["T"], b.T0 * (1 + f["delta"]) ** b.gamma_minus_1, rtol=1e-14)
    assert b.velocity == "zero" and f["v"].shape == f["delta"].shape + (3,) and not np.any(f["v"]), "zero velocities by default"
    with pytest.raises(ValueError, match="velocity"):
        ToyBoxes(box_cMpc=8.0, velocity="kaiser", bao_ratio=1e-3)
    # linear theory: div v = -a H f delta, mode by mode (except on the Nyquist planes, where k_i v_i is a sine of the grid)
    lin = replace(b, velocity="linear")
    f = lin.fields(0)
    np.testing.assert_array_equal(f["delta"], b.fields(0)["delta"])  # the same density
    g = b.grf()
    k_vec, _ = g.k_grid(real=True)
    div = sum(1j * np.asarray(kv) * np.fft.rfftn(f["v"][..., a]) for a, kv in enumerate(k_vec))
    k_nyq = np.pi / b.dx
    inner = ~np.any([np.isclose(np.abs(np.broadcast_to(kv, div.shape)), k_nyq) for kv in k_vec], axis=0)
    inner.ravel()[0] = False
    np.testing.assert_allclose(div[inner], -b.aHf * np.fft.rfftn(f["delta"])[inner], rtol=1e-8, atol=1e-8)
    assert b.aH == pytest.approx(70.0 * np.sqrt(0.31 * 64 + 0.69) / 4) and b.aHf == b.aH * b.f_growth  # notebook 04, z = 3
    cells = b.cells(0, f)
    assert len(cells) == b.n_cells**3 and set(cells.species_keys) == {"HI", "SiIII"}
    with pytest.raises(ValueError, match="1 \\+ delta"):  # no clipping: a negative density is an error
        ToyBoxes(box_cMpc=8.0, n_cells=16, sigma_delta=0.8, bao_ratio=1e-3).fields(0)


def test_skewer_setup():
    """``SkewerRun``: the columns (every stride-th, through the cell centres), the derived quantities, and the 3D grid in
    comoving units (a cube of side L: the line of sight converted with u = period / L, transverse spacing s dx)."""
    cols = BOXES.columns(SETUP.stride)
    sb = BOXES.skewer_bundle(cols, SETUP.points_per_cell)
    pts = np.asarray(sb.spatial_points)
    np.testing.assert_array_equal(pts[:, 0, 0], cols[:, 0] * BOXES.dx)  # the nodes of grf_cells: the cell centres
    np.testing.assert_array_equal(pts[:, 0, 2], cols[:, 1] * BOXES.dx)
    d = SETUP.derived(BOXES)
    assert d["per_box"] == len(cols) == 16 and d["total"] == 16 * BOXES.n_real
    assert d["points_per_skewer"] == pts.shape[1] == BOXES.n_cells * SETUP.points_per_cell + 1
    assert d["aH_start"] < d["kms_per_cMpc"] < d["aH_end"] and d["aH_start"] == pytest.approx(BOXES.aHf, rel=1e-6)
    np.testing.assert_allclose(d["d_perp_kms"], np.array(SETUP.stride) * BOXES.dx * d["period_kms"] / BOXES.box_cMpc)
    gs = SETUP.grid3d(BOXES)
    assert gs.shape == (4, SETUP.n_pix, 4) and np.allclose(gs.L, BOXES.box_cMpc) and gs.los_axis == 1
    u = d["kms_per_cMpc"]
    assert gs.dx[1] == pytest.approx(d["dv_kms"] / u), "a pixel spans dv / u cMpc"
    lim = SETUP.limits(BOXES)
    k_nyq = min(np.pi / d["dv_kms"] * u, *(np.pi / d["d_perp_cMpc"]))  # the sphere where every direction is available
    assert lim["k_nyq"] == pytest.approx(k_nyq) and gs.k_edges[-1] == pytest.approx(k_nyq) and np.nanmax(gs.k_iso) <= k_nyq
    k_f = 2 * np.pi / BOXES.box_cMpc
    np.testing.assert_allclose(gs.kpar, k_f * np.arange(1, SETUP.n_pix // 2 + 1))  # the map: one row per mode k_par = 2 pi m / L
    half = BOXES.box_cMpc / 2  # xi of a periodic field: nothing beyond half the box
    for e in (gs.r_edges, gs.r_mu_edges, gs.rperp_edges):
        assert e[-1] <= half * (1 + 1e-9)
    # the (r_par, r_perp) map on the lattice: one row per pixel separation j dv, j = 0 .. n_pix / 2, centred on it; r_perp
    # columns of the transverse spacing, the first holding the pairs on the same skewer (r_perp = 0)
    np.testing.assert_allclose(gs.rpar, np.arange(SETUP.n_pix // 2 + 1) * BOXES.box_cMpc / SETUP.n_pix, atol=1e-12)
    np.testing.assert_allclose(np.diff(gs.rperp_edges), d["d_perp_cMpc"].min()) and gs.rperp[0] == 0.0
    assert gs.rperp[0] == 0.0 and np.diff(gs.r_edges)[0] == pytest.approx(max(d["d_perp_cMpc"].max(), d["dv_kms"] / u))
    assert lim["r_max"] == pytest.approx(half) and lim["r_perp_min"] == pytest.approx(d["d_perp_cMpc"].min())
    assert lim["k_par_max_skm"] == pytest.approx(np.pi / d["dv_kms"]) and lim["k_par_min_skm"] == pytest.approx(
        2 * np.pi / d["period_kms"]
    )
    np.testing.assert_allclose(lim["k_nyq_perp"], np.pi / d["d_perp_cMpc"]) and lim["dr_iso"] == pytest.approx(
        np.diff(gs.r_edges)[0]
    )
    # explicit bins: used as given, every bin beyond the full-coverage limits cut (and reported)
    logs = []
    e = dict(
        k_edges=np.linspace(0.5 * k_f, 2 * k_nyq, 9),
        r_edges=np.arange(0.0, 0.75 * BOXES.box_cMpc + 2.0, 2.0),
        rperp_edges=np.arange(0.0, half + 2, 1.0),
    )
    ge = SETUP.grid3d(BOXES, edges=e, log=logs.append)
    assert ge.k_edges[-1] <= k_nyq and ge.r_edges[-1] <= half and ge.rperp_edges[-1] <= half and len(logs) == 3
    np.testing.assert_array_equal(ge.r_edges, e["r_edges"][e["r_edges"] <= half])
    assert SETUP.grid3d(BOXES, n_los=BOXES.n_cells).shape == (4, BOXES.n_cells, 4)
    assert not SkewerRun(stride=(3, 4)).periodic(BOXES)
    with pytest.raises(ValueError, match="multiple of both strides"):
        SkewerRun(stride=(3, 4)).grid3d(BOXES)


def test_one_box_along_its_skewers():
    r = skewer_box(BOXES, 0, SETUP)
    n_sk = BOXES.columns(SETUP.stride).shape[0]
    assert r["tau"].shape == (2, n_sk, SETUP.n_pix)
    assert r["warnings"] == [] and np.all(r["resolution"][..., 1] < 1e-3), "the resolution check passes"
    assert 0 < r["mean_F_total"] < r["mean_F"].min() < 1
    s = r["flux"]
    assert np.max(np.abs(s["P_order"].sum(0) - s["P_total"])) < 1e-10 * np.max(np.abs(s["P_total"]))
    assert r["period"] == pytest.approx(SETUP.derived(BOXES)["period_kms"], rel=1e-12)
    # 3D: the overflux of each line and of the total against the mean of the box's skewers (its own ensemble), one
    # number per line (mean="global"); P1D obeys Parseval
    assert set(r["flux3d"]) == set(OVERFLUX_FIELDS) and r["p1d"].shape == (3, SETUP.n_pix // 2 + 1)
    F = np.exp(-r["tau"])
    delta = F.prod(0) / F.prod(0).mean() - 1
    np.testing.assert_allclose(r["delta_sum"][2], delta.sum(0), atol=1e-12)
    assert abs(delta.mean()) < 1e-14 and r["decomposition_error"] < 1e-14
    w = np.r_[1.0, np.full(SETUP.n_pix // 2 - 1, 2.0), 1.0]  # rfft modes -> full grid
    assert np.sum(w * r["p1d"][2]) / (SETUP.n_pix * r["dv"]) == pytest.approx(np.mean(delta**2), rel=1e-10)
    xi = r["flux3d"]["total"]["xi_iso"]
    assert np.isfinite(xi[0]) and xi[0] <= np.mean(delta**2) * (1 + 1e-12)  # bin 0 holds lag 0 and its neighbours


def test_run_in_memory_and_field_statistics():
    """``run_skewers`` and ``field_statistics`` return their results in memory (nothing is stored), recomputed at every
    call: identical results for the same configuration."""
    logs = []
    run = run_skewers(BOXES, SETUP, realisations=[0, 1, 2], log=logs.append)
    assert len(logs) == 3 and set(run["skewers"]) == {0} and run["skewers"][0]["tau"].shape[1] == 16
    assert run["flux"]["n_real"] == 3 and run["flux3d"]["total"]["n_real"] == 3
    assert np.all(np.isfinite(run["p1d"]["p1d_err"][:, 1:])) and run["p1d"]["k"].size == SETUP.n_pix // 2 + 1
    assert run["config"]["boxes"]["velocity"] == "zero" and list(run["realisations"]) == [0, 1, 2]
    again = run_skewers(BOXES, SETUP, realisations=[0, 1, 2], keep=(), log=None)
    np.testing.assert_array_equal(again["flux3d"]["total"]["xi_iso_each"], run["flux3d"]["total"]["xi_iso_each"])
    assert again["skewers"] == {}
    np.testing.assert_array_equal(again["density3d"]["total"]["xi_iso_each"], run["density3d"]["total"]["xi_iso_each"])
    res = field_statistics(BOXES, log=None)
    assert res["n_real"] == 3 and np.all(res["min_one_plus_delta"] > 0)
    nr = res["axes"]["r_iso"].size
    assert res["cov_xi_iso"].shape == (nr, nr) and set(res["theory_nw"]) == set(res["theory"])
    c = xi_chi2(res)
    assert c["dof"] == np.sum(res["axes"]["r_iso"] > 0) and 0 < c["p_full"] < 1 and c["chi2_each"].shape == (3,)
    offs = xi_subset_offsets(res, sizes=(1, 3))
    assert [o["size"] for o in offs] == [1, 3] and offs[0]["offsets"].size == 3
    assert offs[1]["offsets"][0] == pytest.approx(offs[0]["offsets"].mean(), rel=1e-10)
    assert offs[1]["sigma"] == pytest.approx(offs[0]["sigma"] / np.sqrt(3))
    np.testing.assert_array_equal(field_statistics(BOXES, log=None)["P_iso_each"], res["P_iso_each"])  # recomputed, identical
    for k, v in BOXES.bin_edges().items():
        np.testing.assert_array_equal(res["axes"][k], v)
    edges = np.geomspace(0.5, 6.0, 9)
    from skewerjax.summary_stats import GridStatistics3D

    other = field_statistics(BOXES, grid=GridStatistics3D((16,) * 3, 8.0, los_axis=1, k_edges=edges), log=None)
    np.testing.assert_array_equal(other["axes"]["k_edges"], edges)


def test_ensemble_normalisation_of_the_overflux():
    """``run_skewers``: every overflux is normalised by the mean transmission of the ensemble, one number per line and for
    the total over every pixel of every skewer of every realisation (``mean="global"``), so <delta> = 0 holds over the
    ensemble (not at each pixel: the drift of the mean along the skewers stays, shared by every skewer; nor along a
    skewer), C = T_bar / prod_j T_bar_j - 1 = <delta_1 delta_2>, and the decomposition delta_tot = [sum_A delta_A - C] /
    (1 + C) is exact; with ``mean="per_pixel"``, <delta> = 0 at every pixel."""
    run = run_skewers(BOXES, SETUP, realisations=[0, 1, 2], keep=(0, 1, 2), log=None)
    T = np.stack([np.exp(-run["skewers"][i]["tau"]) for i in range(3)], 1)  # (2, 3 boxes, N_sk, n_pix)
    T_bar = np.concatenate([T.mean(axis=(1, 2, 3)), [T.prod(0).mean()]])
    np.testing.assert_allclose(run["mean_transmission"], np.broadcast_to(T_bar[:, None], (3, SETUP.n_pix)), rtol=1e-13)
    np.testing.assert_allclose(run["mean_transmission_per_pixel"][0], T[0].mean(axis=(0, 1)), rtol=1e-13)
    assert np.max(np.abs(run["mean_delta_global"])) < 1e-14, "<delta> = 0 over the ensemble, for each line and the total"
    delta_tot = T.prod(0) / T_bar[-1] - 1
    assert abs(delta_tot.mean()) < 1e-14 and np.max(np.abs(delta_tot.mean(axis=2))) > 1e-3, "the mean along a skewer is not 0"
    np.testing.assert_allclose(run["mean_delta"][2], delta_tot.mean(axis=(0, 1)), atol=1e-14)
    assert np.max(np.abs(run["mean_delta"])) > 1e-4, "the drift along the skewers stays, shared by every skewer"
    d1, d2 = T[0] / T_bar[0] - 1, T[1] / T_bar[1] - 1
    assert run["C"] == pytest.approx(T_bar[2] / (T_bar[0] * T_bar[1]) - 1, rel=1e-12)
    assert abs(run["C"] - np.mean(d1 * d2)) < 1e-14 and run["C_error"] < 1e-14, "C = <delta_1 delta_2>, the global mean"
    assert run["decomposition_error"] < 1e-14
    e = run["flux"]
    for kind in ("P", "xi"):
        sc = np.max(np.abs(e[f"{kind}_total"]))
        assert np.max(np.abs(e[f"{kind}_order"].sum(0) - e[f"{kind}_total"])) < 1e-12 * sc
        assert np.max(np.abs(e[f"{kind}_order"][1])) < 1e-12 * sc, "order 1 vanishes: the normalisation is ensemble-wide"
    pp = run_skewers(BOXES, SETUP, realisations=[0, 1, 2], keep=(), mean="per_pixel", log=None)
    np.testing.assert_allclose(pp["mean_transmission"], run["mean_transmission_per_pixel"], rtol=1e-14)
    assert np.max(np.abs(pp["mean_delta"])) < 1e-14 and pp["C_error"] < 1e-14 and pp["decomposition_error"] < 1e-14


def test_lattice_estimator_recovers_the_density():
    """The 3D estimator of the overflux (``SkewerRun.grid3d``, comoving units) applied to the density overdensity sampled
    on the lattice of skewers recovers its exact expectation, ``lattice_density_theory`` (xi_grid at the lattice
    separations, the aliased P; at the points of the skewers, xi_grid interpolated linearly between the cells), BAO peak
    included; that expectation is the aliasing sum of the input spectrum; and ``run_skewers`` measures the density at the
    points of the skewers with the same estimator, the three overdensities (H I, Si III, their sum) identical (notebook
    08, Parts E and F)."""
    from skewerjax.simulations import DENSITY_FIELDS, on_points
    from skewerjax.simulations.toy_boxes import _lattice_power

    boxes = ToyBoxes(box_cMpc=64.0, n_cells=32, n_real=48, r_bao=20.0, sigma_bao=2.0, bao_ratio=0.01)
    setup = SkewerRun(stride=(2, 2), points_per_cell=4, n_pix=32)
    gs = setup.grid3d(boxes, n_los=boxes.n_cells)
    e = ensemble_mean([gs.measure(boxes.delta(i)[::2, :, ::2]) for i in range(boxes.n_real)], GRID_KEYS)
    th, nw = lattice_density_theory(boxes, setup), lattice_density_theory(boxes, setup, bao=False)
    for key in GRID_KEYS:
        ok = np.isfinite(e[key]) & np.isfinite(th[key]) & (e[key + "_err"] > 0)
        assert np.mean(((e[key] - th[key])[ok] / e[key + "_err"][ok]) ** 2) < 2.0, key
    r = gs.r_iso
    near = np.abs(r - boxes.r_bao) <= 2 * boxes.sigma_bao  # the BAO peak: the measurement follows it, not its absence
    chi = lambda m: float(np.sum(((e["xi_iso"] - m["xi_iso"])[near] / e["xi_iso_err"][near]) ** 2))
    assert chi(th) < 0.5 * chi(nw) and np.all((th["xi_iso"] - nw["xi_iso"])[near] > 0)
    # at the points of the skewers (p per cell): the expectation, exactly, of the piecewise-constant columns
    p = setup.points_per_cell
    gp = setup.grid3d(boxes, n_los=boxes.n_cells * p)
    ep = ensemble_mean([gp.measure(on_points(boxes.delta(i)[::2, :, ::2], p)) for i in range(16)], GRID_KEYS)
    thp, P1 = lattice_density_theory(boxes, setup, points_per_cell=p, p1d=True)
    for key in GRID_KEYS:
        ok = np.isfinite(ep[key]) & np.isfinite(thp[key]) & (ep[key + "_err"] > 0)
        assert np.mean(((ep[key] - thp[key])[ok] / ep[key + "_err"][ok]) ** 2) < 2.0, key
    xi_pts = np.fft.irfftn(_lattice_power(boxes, setup, points_per_cell=p), s=gp.shape) / gp.dV
    xi_cells = np.fft.irfftn(_lattice_power(boxes, setup), s=gs.shape) / gs.dV
    np.testing.assert_allclose(xi_pts[:, ::p, :], xi_cells, rtol=1e-10, atol=1e-14)  # on the cells: the cell lattice
    d0 = on_points(boxes.delta(0)[::2, :, ::2], p)
    assert d0.shape == (16, boxes.n_cells * p, 16) and np.array_equal(d0[:, ::p, :][:, 1:], boxes.delta(0)[::2, 1:, ::2])
    assert np.array_equal(d0[:, p // 2 : p // 2 + p, :], np.repeat(boxes.delta(0)[::2, 1:2, ::2], p, axis=1)), "p points per cell"
    assert P1.shape == (boxes.n_cells * p // 2 + 1,) and np.all(P1[1:] > -1e-12 * P1.max())
    # the piecewise-constant columns have no power at the harmonics of the cell frequency, k_par = q 2 pi / dx
    assert np.all(np.abs(P1[boxes.n_cells :: boxes.n_cells]) < 1e-12 * P1.max()) and np.all(P1[1 : boxes.n_cells] > 0)
    # the expectation on the lattice = the input spectrum folded over the aliases of the transverse lattice: lattice
    # mode a in x holds the grid modes i = a (mod N / s_x), and likewise in z
    small, st = ToyBoxes(box_cMpc=8.0, n_cells=8, bao_ratio=1e-3), SkewerRun(stride=(2, 4))
    g = small.grf()
    P = np.asarray(g.power_grid(real=False))  # (8, 8, 8), fftn layout
    alias = P.reshape(2, 4, 8, 4, 2).sum(axis=(0, 3))  # (4, 8, 2): x folded twice, z four times
    gl = st.grid3d(small, n_los=small.n_cells)
    xi_lat = np.fft.irfftn(np.asarray(g.power_grid()), s=(8, 8, 8))[::2, :, ::4] / g.cell_volume
    np.testing.assert_allclose(np.fft.rfftn(xi_lat).real * gl.dV, alias, rtol=1e-10, atol=1e-12 * alias.max())
    np.testing.assert_allclose(lattice_density_theory(small, st)["P_map"], gl.theory(alias)["P_map"], rtol=1e-10)
    # run_skewers: the density at the points of the skewers against its global mean, through the same estimator
    run = run_skewers(BOXES, SETUP, realisations=[0, 1], keep=(), log=None)
    gs16 = SETUP.grid3d(BOXES, n_los=BOXES.n_cells * SETUP.points_per_cell)
    f = BOXES.fields(1)
    n_hi = on_points(f["n"]["HI"][::4, :, ::4], SETUP.points_per_cell)
    m = BOXES.n_bar_HI * run["density_mean"]["HI"]
    np.testing.assert_allclose(run["density3d"]["HI"]["xi_map_each"][1], gs16.measure(n_hi / m - 1)["xi_map"], rtol=1e-10)
    assert set(run["density3d"]) == set(DENSITY_FIELDS) and set(run["density_theory"]) == set(GRID_KEYS)
    assert np.all(run["density_identity"] < 1e-13), "n_SiIII = r n_HI: the three overdensities are identical"
    for kind in ("P", "xi"):
        e1 = run["density_flux"]
        assert np.max(np.abs(e1[f"{kind}_order"].sum(0) - e1[f"{kind}_total"])) < 1e-12 * np.max(np.abs(e1[f"{kind}_total"]))
    pk = run["density_p1d"]
    assert pk["p1d"].shape == (3, gs16.shape[1] // 2 + 1) and pk["theory"].shape == pk["k"].shape
    chk = lattice_density_check(run, BOXES, SETUP)
    assert chk["n_real"] == 2 and chk["xi_dof"] > 0 and np.isfinite(chk["xi_chi2"]) and chk["P_dof"] > 0
    x = np.linspace(0.0, 50.0, 501)  # bao_height: a bump of height 2 on a straight line
    b = bao_height(x, 1.0 + 0.1 * x + 2.0 * np.exp(-0.5 * ((x - 25.0) / 1.0) ** 2), 25.0, 1.0, err=np.full(x.size, 0.1))
    assert b["height"] == pytest.approx(2.0, abs=1e-3) and b["r"] == 25.0 and 0.1 < b["height_err"] < 0.15


def test_continuous_correlation():
    """``PowerLawGRF.correlation``, the closed form of the continuous xi(r) (xi_pl = A_s k_0^(1 - n_s) Gamma(1 + n_s)
    sin(pi (1 + n_s) / 2) / (2 pi^2 r^(n_s + 2)) and the bump with its mirror), against the numerical Hankel transform
    of the full spectrum P_pl + A_bao W (``correlation_hankel``) across -2 < n_s < 0 (n_s = -1 the limit pi / 2);
    outside that range, one line and the transform over the grid's k range."""
    from dataclasses import replace

    from scipy.special import gamma

    b = ToyBoxes(box_cMpc=260.0, n_cells=64, bao_ratio=1e-4, sigma_bao=5.0, r_bao=100.0)  # below R_max for every n_s here
    g = b.grf()
    r = np.linspace(0.5, 260.0, 300)
    A, k0, ns = g.A_s, g.k_0, g.n_s
    c = np.pi / 2 if ns == -1 else gamma(1 + ns) * np.sin(np.pi * (1 + ns) / 2)  # the limit pi / 2 at n_s = -1
    xi_pl = A * k0 ** (1 - ns) * c / (2 * np.pi**2 * r ** (ns + 2))
    Ab, r0, sg = g.A_bao, g.r_bao, g.sigma_bao
    bump = lambda x: np.exp(-(x**2) / (2 * sg**2))
    np.testing.assert_allclose(g.correlation(r), xi_pl + Ab * (bump(r - r0) + bump(r + r0)), rtol=1e-12)
    for n_s in (-1.9, -1.25, -1.0, -0.5, -0.05):
        gn = replace(b, n_s=n_s).grf()
        np.testing.assert_allclose(gn.correlation(r), gn.correlation_hankel(r, k_tail=400), rtol=1e-7)
    with pytest.raises(ValueError, match="r > 0"):
        g.correlation([0.0, 1.0])
    logs = []
    g_out = replace(b, n_s=0.5, bao_ratio=1e-5).grf()
    xi = g_out.correlation(r, log=logs.append)
    assert len(logs) == 1 and "-2 < n_s < 0" in logs[0] and np.all(np.isfinite(xi))
    np.testing.assert_allclose(xi, g_out.correlation_hankel(r, k_range=(2 * np.pi / 260.0, np.sqrt(3) * np.pi / b.dx)))
    with pytest.raises(ValueError, match="converges"):
        g_out.correlation_hankel(r)


def test_bins_of_the_fields():
    """``ToyBoxes.k_edges`` (``bao_k_edges``): one bin per mode up to k_1, quarter bins of the fundamental mode up to k_2
    (the same scale where the log bins start), n_log log bins to k_max; ``limits``; ``bin_edges``: the map rows on the
    grid's own modes and separations, every bin beyond the full-coverage limits cut; ``field_statistics`` with them."""
    from skewerjax.simulations.toy_boxes import bao_k_edges

    k_f = 1.0
    e, inf = bao_k_edges(k_f, 100.0, 20.1, info=True)
    np.testing.assert_allclose(e[:4], [0.5, 1.5, 2.5, 3.5]) and inf["k_1"] == 3.5
    assert inf["k_2"] == 20.0 and inf["n_mode"] == 3 and inf["n_quarter"] == (20.0 - 3.5) / 0.25 and inf["n_log"] == 12
    q = e[3 : 4 + inf["n_quarter"]]
    np.testing.assert_allclose(np.diff(q), 0.25) and q[-1] == 20.0  # the quarter bins end where the log bins start
    np.testing.assert_allclose(e[-13:], np.geomspace(20.0, 100.0, 13))
    e2, inf2 = bao_k_edges(k_f, 100.0, 20.1, k_1=6.0, n_log=5, info=True)
    assert inf2["k_1"] == 5.5 and inf2["n_mode"] == 5 and inf2["n_log"] == 5 and e2[-1] == 100.0
    e3, inf3 = bao_k_edges(k_f, 10.0, 20.0, info=True)  # the wiggles run past k_max: quarter bins to k_max, no log bins
    assert inf3["n_log"] == 0 and e3[-1] == 10.0 and inf3["k_2"] == 10.0
    b = BOXES
    lim = b.limits()
    assert lim == dict(k_min=2 * np.pi / b.box_cMpc, k_max=np.pi / b.dx, r_min=b.dx, r_max=b.box_cMpc / 2)
    np.testing.assert_allclose(b.k_edges(), bao_k_edges(lim["k_min"], lim["k_max"], 3.0 / b.sigma_bao))
    np.testing.assert_allclose(b.r_edges(), np.arange(0, b.box_cMpc / 2 + b.dx, b.dx), atol=1e-12)
    logs = []
    be = b.bin_edges(np.linspace(0.1, 2 * lim["k_max"], 11), np.arange(0, b.box_cMpc, b.dx), log=logs.append)
    assert len(logs) == 2 and be["k_edges"][-1] <= lim["k_max"] and be["r_edges"][-1] <= lim["r_max"] * (1 + 1e-12)
    res = field_statistics(b, k_edges=b.k_edges(), r_edges=b.r_edges(), log=None)
    a = res["axes"]
    np.testing.assert_allclose(a["kpar"], lim["k_min"] * np.arange(1, b.n_cells // 2 + 1))  # one row per mode
    np.testing.assert_allclose(a["rpar"], b.dx * np.arange(b.n_cells // 2 + 1), atol=1e-12)  # one row per separation
    assert a["rperp"][0] == 0.0 and np.all(a["n_lags_xi_map"][:, 0] > 0), "the first r_perp column: r_perp = 0 only"
    np.testing.assert_array_equal(a["kperp_edges"], a["k_edges"]) and res["limits"] == lim
    assert np.all(np.isfinite(res["cov_xi_iso"][1:, 1:])) and res["axes"]["r_iso"][0] == 0.0
