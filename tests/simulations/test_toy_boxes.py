"""The toy boxes of notebook 08 (``simulations.toy_boxes``) on a small grid: realisations regenerated from their seeds,
the gas (n_SiIII = r n_HI, T = T0 (1 + delta)^(gamma - 1), linear-theory velocities, no clipping), the skewer set-up and
its derived quantities, one realisation along its skewers (tau, the resolution check, the 1D flux statistics by order, the
3D statistics and P1D of the overfluxes in velocity units), the resumable, cached run, and the chi^2 of xi with its exact
covariance."""

import numpy as np
import pytest

from skewerjax.simulations import (
    SkewerRun,
    ToyBoxes,
    field_statistics,
    load_skewer_run,
    run_skewers,
    skewer_box,
    xi_chi2,
    xi_subset_offsets,
)
from skewerjax.simulations.toy_boxes import OVERFLUX_FIELDS

BOXES = ToyBoxes(box_cMpc=8.0, n_cells=16, n_real=3)
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
    # linear theory: div v = -a H f delta, mode by mode (except on the Nyquist planes, where k_i v_i is a sine of the grid)
    g = b.grf()
    k_vec, _ = g.k_grid(real=True)
    div = sum(1j * np.asarray(kv) * np.fft.rfftn(f["v"][..., a]) for a, kv in enumerate(k_vec))
    k_nyq = np.pi / b.dx
    inner = ~np.any([np.isclose(np.abs(np.broadcast_to(kv, div.shape)), k_nyq) for kv in k_vec], axis=0)
    inner.ravel()[0] = False
    np.testing.assert_allclose(div[inner], -b.aHf * np.fft.rfftn(f["delta"])[inner], rtol=1e-8, atol=1e-8)
    assert b.aHf == pytest.approx(70.0 * np.sqrt(0.31 * 64 + 0.69) / 4)  # notebook 04, z = 3
    cells = b.cells(0, f)
    assert len(cells) == b.n_cells**3 and set(cells.species_keys) == {"HI", "SiIII"}
    with pytest.raises(ValueError, match="1 \\+ delta"):  # no clipping: a negative density is an error
        ToyBoxes(box_cMpc=8.0, n_cells=16, sigma_delta=0.8).fields(0)


def test_skewer_setup():
    """``SkewerRun``: the columns (every stride-th, through the cell centres), the derived quantities, and the 3D grid in
    velocity units (a cube of side the period, transverse spacing s dx u)."""
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
    gs = SETUP.grid3d(BOXES, d["period_kms"])
    assert gs.shape == (4, SETUP.n_pix, 4) and np.allclose(gs.L, d["period_kms"]) and gs.los_axis == 1
    assert gs.k_edges[-1] == pytest.approx(np.pi / d["d_perp_kms"][0])  # the transverse Nyquist frequency
    assert not SkewerRun(stride=(3, 4)).periodic(BOXES)
    with pytest.raises(ValueError, match="multiple of both strides"):
        SkewerRun(stride=(3, 4)).grid3d(BOXES, d["period_kms"])


def test_one_box_along_its_skewers():
    r = skewer_box(BOXES, 0, SETUP)
    n_sk = BOXES.columns(SETUP.stride).shape[0]
    assert r["tau"].shape == (2, n_sk, SETUP.n_pix)
    assert r["warnings"] == [] and np.all(r["resolution"][..., 1] < 1e-3), "the resolution check passes"
    assert 0 < r["mean_F_total"] < r["mean_F"].min() < 1
    s = r["flux"]
    assert np.max(np.abs(s["P_order"].sum(0) - s["P_total"])) < 1e-10 * np.max(np.abs(s["P_total"]))
    assert r["period"] == pytest.approx(SETUP.derived(BOXES)["period_kms"], rel=1e-12)
    # 3D: the overflux of each line and of the total against the mean of the box; P1D obeys Parseval
    assert set(r["flux3d"]) == set(OVERFLUX_FIELDS) and r["p1d"].shape == (3, SETUP.n_pix // 2 + 1)
    F = np.exp(-r["tau"])
    delta = F.prod(0) / F.prod(0).mean() - 1
    w = np.r_[1.0, np.full(SETUP.n_pix // 2 - 1, 2.0), 1.0]  # rfft modes -> full grid
    assert np.sum(w * r["p1d"][2]) / (SETUP.n_pix * r["dv"]) == pytest.approx(np.mean(delta**2), rel=1e-10)
    xi = r["flux3d"]["total"]["xi_iso"]
    assert np.isfinite(xi[0]) and xi[0] <= np.mean(delta**2) * (1 + 1e-12)  # bin 0 holds lag 0 and its neighbours


def test_run_is_resumable_and_cached():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        _resumable_and_cached(tmp)


def _resumable_and_cached(tmp_path):
    logs = []
    folder = run_skewers(BOXES, tmp_path, SETUP, realisations=[0, 1], log=logs.append)
    assert len(logs) == 2 and (folder / "skewers_000.npz").exists() and not (folder / "skewers_001.npz").exists()
    mtime = (folder / "box_000.npz").stat().st_mtime
    (folder / "box_002.lock").write_text("999999999")  # a lock left by a process that no longer runs
    run_skewers(BOXES, tmp_path, SETUP, log=logs.append)
    assert len(logs) == 3 and (folder / "box_000.npz").stat().st_mtime == mtime  # only realisation 2 was computed
    assert not list(folder.glob("*.lock"))
    run = load_skewer_run(folder)
    assert run["flux"]["n_real"] == 3 and run["flux3d"]["total"]["n_real"] == 3
    assert np.all(np.isfinite(run["p1d"]["p1d_err"][:, 1:])) and run["p1d"]["k"].size == SETUP.n_pix // 2 + 1
    assert set(run["skewers"]) == {0} and run["skewers"][0]["tau"].shape[1] == 16
    # overwrite: the folder is emptied and the requested realisations recomputed
    run_skewers(BOXES, tmp_path, SETUP, realisations=[1], overwrite=True, log=logs.append)
    assert len(logs) == 4 and sorted(p.name for p in folder.glob("*.npz")) == ["box_001.npz"]
    res = field_statistics(BOXES, tmp_path, log=None)
    assert res["n_real"] == 3 and np.all(res["min_one_plus_delta"] > 0)
    nr = res["axes"]["r_iso"].size
    assert res["cov_xi_iso"].shape == (nr, nr) and set(res["theory_nw"]) == set(res["theory"])
    c = xi_chi2(res)
    assert c["dof"] == np.sum(res["axes"]["r_iso"] > 0) and 0 < c["p_full"] < 1 and c["chi2_each"].shape == (3,)
    offs = xi_subset_offsets(res, sizes=(1, 3))
    assert [o["size"] for o in offs] == [1, 3] and offs[0]["offsets"].size == 3
    assert offs[1]["offsets"][0] == pytest.approx(offs[0]["offsets"].mean(), rel=1e-10)
    assert offs[1]["sigma"] == pytest.approx(offs[0]["sigma"] / np.sqrt(3))
    again = field_statistics(BOXES, tmp_path, log=None)  # read back from the cache
    np.testing.assert_array_equal(again["P_iso_each"], res["P_iso_each"])
    assert again["time_s"] == res["time_s"]
    for k, v in BOXES.bin_edges().items():
        np.testing.assert_array_equal(res["axes"][k], v)
    edges = np.geomspace(0.5, 6.0, 9)  # other |k| bins: the cache does not hold them, so they are computed
    from skewerjax.core.stats import GridStatistics3D

    other = field_statistics(BOXES, tmp_path, grid=GridStatistics3D((16,) * 3, 8.0, los_axis=1, k_edges=edges), log=None)
    np.testing.assert_array_equal(other["axes"]["k_edges"], edges)
    np.testing.assert_array_equal(field_statistics(BOXES, tmp_path, log=None)["axes"]["k_edges"], res["axes"]["k_edges"])


def test_cache_key():
    """The cache key changes with every field of the configuration (and the skewer folder with the set-up)."""
    from dataclasses import fields, replace

    tags = {BOXES.tag()}
    for f in fields(ToyBoxes):
        v = getattr(BOXES, f.name)
        tags.add(replace(BOXES, **{f.name: v + 1 if isinstance(v, int) else v * 1.01}).tag())
    assert len(tags) == len(fields(ToyBoxes)) + 1
    folders = {
        SETUP.folder(BOXES, "c"),
        SkewerRun((8, 4), 80, 64).folder(BOXES, "c"),
        SkewerRun((8, 8), 81, 64).folder(BOXES, "c"),
    }
    folders |= {SkewerRun((8, 8), 80, 65).folder(BOXES, "c"), SETUP.folder(ToyBoxes(box_cMpc=8.0, n_cells=16, n_real=4), "c")}
    assert len(folders) == 5
