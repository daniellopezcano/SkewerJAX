"""The r bins of the 3D statistics (``binning.set_r_edges``, ``set_rperp_edges``) and the full-coverage limits of every
figure that shades them (``grid3d.coverage_limits``, ``GridStatistics3D.limits``): the occupancy equals the counts of
``GridStatistics3D`` on the same lattice, the first r_perp bin is r_perp = 0 alone, the zero lag is in no |r| bin, the
edges reach r_max, no bin is empty; ``lattice_limits`` and ``grid_limits`` give the limits of the grid that bins the
statistics, with no bin centre below r_min. Run in both precisions (``SKEWERJAX_PRECISION``)."""

import numpy as np
import pytest

from skewerjax.summary_stats import (
    box_grid,
    coverage_limits,
    grid_limits,
    lattice_grid,
    lattice_limits,
    set_k_edges,
    set_r_edges,
    set_rperp_edges,
)

L, N_SK, N_PIX = 32.0, 8, 16  # a small lattice of skewers: d_perp = 4, the line of sight 2 cMpc (d_los < d_perp)
D_PERP, D_LOS = L / N_SK, L / N_PIX


def _lattice(r_edges, rperp_edges, n_los=N_PIX):
    k_f = 2 * np.pi / L
    k_nyq = min(np.pi / D_PERP, np.pi * n_los / L)
    k = set_k_edges(k_f, k_1=1.5 * k_f, k_2=k_nyq, k_max=k_nyq, dk_factor=1.0)
    return lattice_grid(N_SK, N_SK, n_los, L, D_PERP, k_edges=k, r_edges=r_edges, rperp_edges=rperp_edges)


def test_the_occupancy_is_the_grid_counts():
    """``info["n_lags"]`` (the separation vectors per bin) equals ``GridStatistics3D``'s counts on the same lattice, and
    ``info["r_mean"]`` its separation-weighted centres; the r_perp occupancy times n_los is the map summed over its rows."""
    e, inf = set_r_edges(D_PERP, D_LOS, L=L, r_1=2 * D_PERP, r_2=L / 2, dr_factor=1.0, info=True)
    q, iq = set_rperp_edges(D_PERP, L=L, r_1=2 * D_PERP, r_2=L / 2, dr_factor=0.5, info=True)
    g = _lattice(e, q)
    np.testing.assert_array_equal(np.asarray(g.n_lags["xi_iso"]), inf["n_lags"])
    np.testing.assert_allclose(g.r_iso, inf["r_mean"], rtol=1e-5)
    np.testing.assert_array_equal(np.asarray(g.n_lags["xi_map"]).sum(axis=0), iq["n_lags"] * N_PIX)
    assert inf["n_lags"].sum() == np.sum(np.asarray(g.n_lags["xi_iso"])) and np.all(inf["n_lags"] > 0)
    # the first range: one distinct separation per bin, edges halfway between them (d_los = 2, d_perp = 4: 2, 4, 4.47, 5.66, 6, ...)
    assert np.all(inf["n_distinct"][: inf["n_sep"]] == 1) and inf["n_sep"] >= 4
    np.testing.assert_allclose(g.r_iso[:3], [2.0, 4.0, np.sqrt(20.0)], rtol=1e-6)
    # a cubic grid (d_perp = d_los): box_grid counts alike
    gb = box_grid(
        16, 16.0, k_edges=np.array([0.3, 1.0, 3.0]), r_edges=(eb := set_r_edges(1.0, 1.0, L=16.0, r_1=3.0, r_2=8.0)), log=None
    )
    np.testing.assert_array_equal(
        np.asarray(gb.n_lags["xi_iso"]), set_r_edges(1.0, 1.0, L=16.0, r_1=3.0, r_2=8.0, info=True)[1]["n_lags"]
    )
    assert eb[0] == 0.5 and gb.r_iso[0] == 1.0  # the first bin: r = dx alone (6 vectors)


def test_the_first_bins():
    """r_perp: the first bin, [0, d_perp / 2), holds r_perp = 0 alone (the same skewer); |r|: the zero lag (the variance)
    is in no bin, the first bin holds the smallest separation alone."""
    q = set_rperp_edges(D_PERP, L=L, r_1=3 * D_PERP, r_2=L / 2)
    e = set_r_edges(D_PERP, D_LOS, L=L, r_1=2 * D_PERP, r_2=L / 2)
    assert q[0] == 0.0 and q[1] == 0.5 * D_PERP and e[0] == 0.5 * D_LOS
    g = _lattice(e, q)
    assert g.rperp[0] == 0.0 and np.all(np.asarray(g.rperp_map)[:, 0] == 0.0)
    np.testing.assert_array_equal(np.asarray(g.n_lags["xi_map"])[:, 0].sum(), N_PIX)  # one skewer: its own N_PIX offsets
    assert np.asarray(g.n_lags["xi_iso"])[0] == 2 and np.isclose(g.r_iso[0], D_LOS)  # +- d_los, no zero lag
    lim = g.limits()
    assert lim["r_min"] == D_LOS and g.r_iso[0] >= lim["r_min"] * (1 - 1e-6)  # no point below the shaded r_min


@pytest.mark.parametrize("n_log", [0, 4])
def test_the_edges_reach_r_max(n_log):
    """Steps of dr_factor x the spacing to r_2 (the last one ending exactly there), then n_log log bins to r_max (default
    L / 2); r_2 >= r_max: steps to r_max; r_2 < r_max needs n_log >= 1."""
    r_2 = 0.5 * L if n_log == 0 else 8.0
    e, inf = set_r_edges(D_PERP, D_LOS, L=L, r_1=D_PERP, r_2=r_2, dr_factor=0.5, n_log=n_log, info=True)
    assert e[-1] == 0.5 * L and inf["r_max"] == 0.5 * L and np.all(np.diff(e) > 0)
    assert inf["r_2"] == r_2 and r_2 in e and inf["n_log"] == n_log
    q, iq = set_rperp_edges(D_PERP, L=L, r_1=D_PERP, r_2=r_2, r_max=12.0, n_log=n_log, info=True)
    assert q[-1] == min(12.0, 0.5 * L) and np.all(iq["n_lags"] > 0)
    if n_log:
        np.testing.assert_allclose(e[-n_log - 1 :], np.geomspace(r_2, 0.5 * L, n_log + 1))
    with pytest.raises(ValueError, match="n_log"):
        set_r_edges(D_PERP, D_LOS, L=L, r_1=D_PERP, r_2=8.0, n_log=0)
    with pytest.raises(ValueError, match="divide"):
        set_r_edges(3.0, D_LOS, L=L, r_1=D_PERP, r_2=L / 2)


def test_no_empty_bins():
    """Steps finer than the spacing between the distinct separations leave empty bins: merged into the next one by
    default (``n_merged``), kept and counted with ``merge_empty=False``."""
    kw = dict(L=L, r_1=0.0, r_2=L / 2, dr_factor=0.1)
    e, inf = set_rperp_edges(D_PERP, info=True, **kw)
    k, ink = set_rperp_edges(D_PERP, merge_empty=False, info=True, **kw)
    assert inf["n_empty"] == 0 and np.all(inf["n_lags"] > 0) and inf["n_merged"] > 0
    assert ink["n_empty"] == inf["n_merged"] and k.size - e.size == inf["n_merged"] and e[-1] == k[-1] == L / 2
    assert inf["n_lags"].sum() == ink["n_lags"].sum()


def test_coverage_limits():
    """One rule for every figure that shades: k_min = 2 pi / L, k_max = the smallest Nyquist frequency, r_min = the
    smallest spacing, r_max = L / 2; ``lattice_limits`` and ``grid_limits`` equal the ``limits()`` of the grids they
    describe."""
    keys = ("k_min", "k_max", "r_min", "r_max")
    g = _lattice(None, None)
    ll = lattice_limits(L, D_PERP, N_PIX, 1000.0)
    assert {k: ll[k] for k in keys} == g.limits() == coverage_limits(L, (D_PERP, D_LOS, D_PERP))
    assert ll["r_min"] == D_LOS and ll["k_max"] == np.pi / D_PERP and ll["dr_iso"] == D_PERP
    gb = box_grid(16, 16.0, k_edges=np.array([0.3, 1.0, 3.0]), r_edges=np.arange(0, 8.5, 1.0), log=None)
    assert grid_limits(16, 16.0) == gb.limits() == dict(k_min=2 * np.pi / 16.0, k_max=np.pi, r_min=1.0, r_max=8.0)
