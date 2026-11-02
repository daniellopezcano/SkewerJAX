"""P and xi of periodic 3D fields on their grid (``summary_stats.field_stats``), from arrays and an expectation P(k):
``grid_power`` = ``PowerLawGRF.power_grid`` bit for bit; ``grid_limits``, ``box_grid`` (the map rows on the grid's own
modes and separations, every bin beyond the full-coverage limits cut) and the |k| bins of ``bao_k_edges``;
``field_statistics`` over an iterable of fields (streamed, recomputed identically), with the expectation, the
expectation without the BAO bump and the exact covariance of xi; ``xi_chi2``, ``xi_subset_offsets``; ``bao_height``."""

import numpy as np
import pytest

from skewerjax.simulations import ToyBoxes
from skewerjax.summary_stats import (
    GridStatistics3D,
    bao_height,
    bao_k_edges,
    box_grid,
    field_statistics,
    grid_limits,
    grid_power,
    xi_chi2,
    xi_subset_offsets,
)

BOXES = ToyBoxes(box_cMpc=8.0, n_cells=16, n_real=3, bao_ratio=1e-3)  # below R_max of this grid


def _stats(b, k_edges, r_edges, **kw):
    g = b.grf()
    gs = box_grid(b.n_cells, b.box_cMpc, k_edges=k_edges, r_edges=r_edges, log=kw.pop("log", None))
    fields = (b.delta(i) for i in range(b.n_real))
    return field_statistics(fields, gs, power=g.power, power_nw=lambda k: g.power(k, bao=False), log=None), gs


def test_grid_power_is_the_spectrum_on_the_modes():
    for b in (BOXES, ToyBoxes(box_cMpc=150, n_s=-0.4, n_cells=32, bao_ratio=0.003, sigma_bao=1.0, r_bao=40)):
        g = b.grf()
        for bao in (True, False):
            P = grid_power(lambda k: g.power(k, bao=bao), (b.n_cells,) * 3, b.box_cMpc)
            assert np.array_equal(np.asarray(P), np.asarray(g.power_grid(bao=bao))), "bit for bit"


def test_bins_of_the_fields():
    """``bao_k_edges``: one bin per mode up to k_1, steps of dk_factor k_f (default 1/4) up to k_2 (where the log bins
    start), n_log log bins to k_max; ``grid_limits``; ``box_grid``: the map rows on the grid's own modes and separations,
    every bin beyond the full-coverage limits cut; ``field_statistics`` with them."""
    k_f = 1.0
    e, inf = bao_k_edges(k_f, 100.0, 20.1, info=True)
    np.testing.assert_allclose(e[:4], [0.5, 1.5, 2.5, 3.5]) and inf["k_1"] == 3.5
    assert inf["k_2"] == 20.0 and inf["n_mode"] == 3 and inf["n_step"] == (20.0 - 3.5) / 0.25 and inf["n_log"] == 12
    assert inf["dk"] == 0.25
    q = e[3 : 4 + inf["n_step"]]
    np.testing.assert_allclose(np.diff(q), 0.25) and q[-1] == 20.0  # the quarter bins end where the log bins start
    np.testing.assert_allclose(e[-13:], np.geomspace(20.0, 100.0, 13))
    e2, inf2 = bao_k_edges(k_f, 100.0, 20.1, k_1=6.0, n_log=5, info=True)
    assert inf2["k_1"] == 5.5 and inf2["n_mode"] == 5 and inf2["n_log"] == 5 and e2[-1] == 100.0
    e3, inf3 = bao_k_edges(k_f, 10.0, 20.0, info=True)  # the wiggles run past k_max: quarter bins to k_max, no log bins
    assert inf3["n_log"] == 0 and e3[-1] == 10.0 and inf3["k_2"] == 10.0
    np.testing.assert_array_equal(bao_k_edges(k_f, 100.0, 20.1, dk_factor=0.25), e)
    e4, inf4 = bao_k_edges(k_f, 100.0, 20.1, dk_factor=0.5, info=True)
    assert inf4["k_2"] == 20.0 and inf4["n_step"] == (20.0 - 3.5) / 0.5 and inf4["dk"] == 0.5
    np.testing.assert_allclose(np.diff(e4[3 : 4 + inf4["n_step"]]), 0.5)
    with pytest.raises(ValueError, match="dk_factor"):
        bao_k_edges(k_f, 100.0, 20.1, dk_factor=0.0)
    b = BOXES
    lim = grid_limits(b.n_cells, b.box_cMpc)
    assert lim == dict(k_min=2 * np.pi / b.box_cMpc, k_max=np.pi / b.dx, r_min=b.dx, r_max=b.box_cMpc / 2)
    logs = []
    gs = box_grid(
        b.n_cells,
        b.box_cMpc,
        k_edges=np.linspace(0.1, 2 * lim["k_max"], 11),
        r_edges=np.arange(0, b.box_cMpc, b.dx),
        log=logs.append,
    )
    assert len(logs) == 2 and gs.k_edges[-1] <= lim["k_max"] and gs.r_edges[-1] <= lim["r_max"] * (1 + 1e-12)
    k_edges = bao_k_edges(lim["k_min"], lim["k_max"], 3.0 / b.sigma_bao)
    r_edges = np.arange(0, b.box_cMpc / 2 + b.dx, b.dx)
    res, _ = _stats(b, k_edges, r_edges)
    a = res["axes"]
    np.testing.assert_allclose(a["kpar"], lim["k_min"] * np.arange(1, b.n_cells // 2 + 1))  # one row per mode
    np.testing.assert_allclose(a["rpar"], b.dx * np.arange(b.n_cells // 2 + 1), atol=1e-12)  # one row per separation
    assert a["rperp"][0] == 0.0 and np.all(a["n_lags_xi_map"][:, 0] > 0), "the first r_perp column: r_perp = 0 only"
    np.testing.assert_array_equal(a["kperp_edges"], a["k_edges"]) and res["limits"] == lim
    assert np.all(np.isfinite(res["cov_xi_iso"][1:, 1:])) and res["axes"]["r_iso"][0] == 0.0


def test_field_statistics_and_the_chi2_of_xi():
    """``field_statistics`` streams the fields of an iterable (nothing stored; recomputed identically), with the
    expectation, the expectation without the bump, the covariance of xi and the limits; ``xi_chi2`` and
    ``xi_subset_offsets`` from its result; without ``power``, the measurement alone."""
    b = BOXES
    lim = grid_limits(b.n_cells, b.box_cMpc)
    k_edges, r_edges = bao_k_edges(lim["k_min"], lim["k_max"], 3.0), np.arange(0, b.box_cMpc / 2 + b.dx, b.dx)
    res, gs = _stats(b, k_edges, r_edges)
    assert res["n_real"] == 3 and np.all(res["min_one_plus_delta"] > 0) and res["limits"] == lim
    np.testing.assert_allclose(res["min_one_plus_delta"], [1 + b.delta(i).min() for i in range(3)], rtol=0)
    nr = res["axes"]["r_iso"].size
    assert res["cov_xi_iso"].shape == (nr, nr) and set(res["theory_nw"]) == set(res["theory"])
    c = xi_chi2(res)
    assert c["dof"] == np.sum(res["axes"]["r_iso"] > 0) and 0 < c["p_full"] < 1 and c["chi2_each"].shape == (3,)
    offs = xi_subset_offsets(res, sizes=(1, 3))
    assert [o["size"] for o in offs] == [1, 3] and offs[0]["offsets"].size == 3
    assert offs[1]["offsets"][0] == pytest.approx(offs[0]["offsets"].mean(), rel=1e-10)
    assert offs[1]["sigma"] == pytest.approx(offs[0]["sigma"] / np.sqrt(3))
    again, _ = _stats(b, k_edges, r_edges)
    np.testing.assert_array_equal(again["P_iso_each"], res["P_iso_each"])  # recomputed, identical
    bare = field_statistics([b.delta(0), b.delta(1)], gs, log=None)
    assert bare["n_real"] == 2 and "theory" not in bare and "cov_xi_iso" not in bare
    np.testing.assert_array_equal(bare["P_iso_each"], res["P_iso_each"][:2])
    edges = np.geomspace(0.5, 6.0, 9)
    other = field_statistics(
        (b.delta(i) for i in range(3)), GridStatistics3D((16,) * 3, 8.0, los_axis=1, k_edges=edges), log=None
    )
    np.testing.assert_array_equal(other["axes"]["k_edges"], edges)


def test_bao_height():
    x = np.linspace(0.0, 50.0, 501)  # a bump of height 2 on a straight line
    b = bao_height(x, 1.0 + 0.1 * x + 2.0 * np.exp(-0.5 * ((x - 25.0) / 1.0) ** 2), 25.0, 1.0, err=np.full(x.size, 0.1))
    assert b["height"] == pytest.approx(2.0, abs=1e-3) and b["r"] == 25.0 and 0.1 < b["height_err"] < 0.15
    assert np.isnan(bao_height(x, x, 25.0, 1.0)["height_err"]), "no errors given"
    assert np.isnan(bao_height(np.array([0.0, 50.0]), np.array([1.0, 1.0]), 10.0, 0.5)["height"]), "no sidebands"
