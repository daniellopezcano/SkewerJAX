"""The density on a lattice of skewers (``summary_stats.density``): the 3D estimator of the overflux (``lattice_grid``,
comoving units) applied to the overdensity sampled on the lattice recovers its exact expectation,
``lattice_density_theory`` (xi_grid at the lattice separations, the aliased P; at the points of the skewers, xi_grid
interpolated linearly between the cells), BAO peak included; that expectation is the aliasing sum of the input
spectrum; ``on_points`` puts p points in every cell; and ``skewer_density_statistics`` (two species and their sum,
against their global mean) measures the same field through the same estimator, against the same expectation scaled by
``density_scale`` (notebook 10, Part B)."""

import numpy as np
import pytest

from skewerjax.simulations import ToyBoxes
from skewerjax.summary_stats import (
    GRID_KEYS,
    bao_k_edges,
    ensemble_mean,
    lattice_density_theory,
    lattice_grid,
    on_points,
    skewer_density_statistics,
)
from skewerjax.summary_stats.density import _lattice_power

BOXES = ToyBoxes(box_cMpc=64.0, n_cells=32, n_real=48, r_bao=20.0, sigma_bao=2.0, bao_ratio=0.01)
STRIDE, P = (2, 2), 4


def _grid(b, stride, n_los):
    """``lattice_grid`` of the lattice with ``stride`` and n_los points along y, |k| up to its smallest Nyquist frequency."""
    L = float(b.box_cMpc)
    d_perp = np.array(stride, dtype=float) * b.dx
    k_edges = bao_k_edges(2 * np.pi / L, float(min(np.pi * n_los / L, *(np.pi / d_perp))), 3.0 / b.sigma_bao)
    return lattice_grid(b.n_cells // stride[0], b.n_cells // stride[1], n_los, L, d_perp, k_edges=k_edges)


def _mean_pull2(e, th, key):
    ok = np.isfinite(e[key]) & np.isfinite(th[key]) & (e[key + "_err"] > 0)
    return np.mean(((e[key] - th[key])[ok] / e[key + "_err"][ok]) ** 2)


def test_lattice_estimator_recovers_the_density():
    b, g = BOXES, BOXES.grf()
    nw = lambda k: g.power(k, bao=False)  # noqa: E731 (the power law alone)
    gs = _grid(b, STRIDE, b.n_cells)
    e = ensemble_mean([gs.measure(b.delta(i)[::2, :, ::2]) for i in range(b.n_real)], GRID_KEYS)
    th, th_nw = lattice_density_theory(g.power, gs, b.n_cells, STRIDE), lattice_density_theory(nw, gs, b.n_cells, STRIDE)
    for key in GRID_KEYS:
        assert _mean_pull2(e, th, key) < 2.0, key
    r = gs.r_iso
    near = np.abs(r - b.r_bao) <= 2 * b.sigma_bao  # the BAO peak: the measurement follows it, not its absence
    chi = lambda m: float(np.sum(((e["xi_iso"] - m["xi_iso"])[near] / e["xi_iso_err"][near]) ** 2))  # noqa: E731
    assert chi(th) < 0.5 * chi(th_nw) and np.all((th["xi_iso"] - th_nw["xi_iso"])[near] > 0)
    # at the points of the skewers (p per cell): the expectation, exactly, of the piecewise-constant columns
    gp = _grid(b, STRIDE, b.n_cells * P)
    ep = ensemble_mean([gp.measure(on_points(b.delta(i)[::2, :, ::2], P)) for i in range(16)], GRID_KEYS)
    thp, P1 = lattice_density_theory(g.power, gp, b.n_cells, STRIDE, points_per_cell=P, p1d=True)
    for key in GRID_KEYS:
        assert _mean_pull2(ep, thp, key) < 2.0, key
    xi_pts = np.fft.irfftn(_lattice_power(g.power, b.n_cells, b.box_cMpc, STRIDE, P), s=gp.shape) / gp.dV
    xi_cells = np.fft.irfftn(_lattice_power(g.power, b.n_cells, b.box_cMpc, STRIDE), s=gs.shape) / gs.dV
    np.testing.assert_allclose(xi_pts[:, ::P, :], xi_cells, rtol=1e-10, atol=1e-14)  # on the cells: the cell lattice
    d0 = on_points(b.delta(0)[::2, :, ::2], P)
    assert d0.shape == (16, b.n_cells * P, 16) and np.array_equal(d0[:, ::P, :][:, 1:], b.delta(0)[::2, 1:, ::2])
    assert np.array_equal(d0[:, P // 2 : P // 2 + P, :], np.repeat(b.delta(0)[::2, 1:2, ::2], P, axis=1)), "p points per cell"
    assert P1.shape == (b.n_cells * P // 2 + 1,) and np.all(P1[1:] > -1e-12 * P1.max())
    # the piecewise-constant columns have no power at the harmonics of the cell frequency, k_par = q 2 pi / dx
    assert np.all(np.abs(P1[b.n_cells :: b.n_cells]) < 1e-12 * P1.max()) and np.all(P1[1 : b.n_cells] > 0)
    with pytest.raises(ValueError, match="lattice"):
        lattice_density_theory(g.power, gs, b.n_cells, STRIDE, points_per_cell=P)  # gs is the lattice of the cells


def test_aliasing_sum():
    """The expectation on the lattice = the input spectrum folded over the aliases of the transverse lattice: lattice
    mode a in x holds the grid modes i = a (mod N / s_x), and likewise in z."""
    small, st = ToyBoxes(box_cMpc=8.0, n_cells=8, bao_ratio=1e-3), (2, 4)
    g = small.grf()
    Pg = np.asarray(g.power_grid(real=False))  # (8, 8, 8), fftn layout
    alias = Pg.reshape(2, 4, 8, 4, 2).sum(axis=(0, 3))  # (4, 8, 2): x folded twice, z four times
    gl = _grid(small, st, small.n_cells)
    xi_lat = np.fft.irfftn(np.asarray(g.power_grid()), s=(8, 8, 8))[::2, :, ::4] / g.cell_volume
    np.testing.assert_allclose(np.fft.rfftn(xi_lat).real * gl.dV, alias, rtol=1e-10, atol=1e-12 * alias.max())
    np.testing.assert_allclose(
        lattice_density_theory(g.power, gl, small.n_cells, st)["P_map"], gl.theory(alias)["P_map"], rtol=1e-10
    )


def test_skewer_density_statistics_against_the_lattice_expectation():
    """``skewer_density_statistics`` on the cells the lattice crosses (three species, n_X = r_X n_HI): the four
    overdensities identical, the 3D statistics = ``measure`` of n / <n> - 1 at the points, the exact decomposition of the
    1D statistics, and the estimator against its exact expectation on the lattice of the points
    (``lattice_density_theory`` with ``scale = density_scale``: the global-mean normalisation), P and xi, BAO included."""
    b, g = BOXES, BOXES.grf()
    n_real = 16
    gp = _grid(b, STRIDE, b.n_cells * P)
    fields = [b.fields(i) for i in range(n_real)]
    cols = [{sp: n[::2, :, ::2] for sp, n in f["n"].items()} for f in fields]
    ds = skewer_density_statistics(cols, P, gp, b.box_cMpc, n_bar=b.n_bar)
    assert list(ds["density3d"]) == ["HI", "SiIII", "SiII", "total"] and ds["density_axes3d"]["k_edges"].size == gp.k_edges.size
    assert np.all(ds["density_identity"] < 1e-13), "n_X = r_X n_HI: the overdensities are identical"
    m = b.n_bar_HI * ds["density_mean"]["HI"]
    np.testing.assert_allclose(ds["density_means"]["HI"], m, rtol=1e-14)
    one = gp.measure(on_points(cols[1]["HI"], P) / m - 1)["xi_map"]
    np.testing.assert_allclose(ds["density3d"]["HI"]["xi_map_each"][1], one, rtol=1e-10)
    for kind in ("P", "xi"):
        e1 = ds["density_flux"]
        assert np.max(np.abs(e1[f"{kind}_order"].sum(0) - e1[f"{kind}_total"])) < 1e-12 * np.max(np.abs(e1[f"{kind}_total"]))
    pk = ds["density_p1d"]
    assert pk["p1d"].shape == (4, gp.shape[1] // 2 + 1) and pk["k"].shape == (gp.shape[1] // 2 + 1,)
    assert ds["density_scale"] == pytest.approx((1 / ds["density_mean"]["total"]) ** 2, rel=1e-14)
    th = lattice_density_theory(g.power, gp, b.n_cells, STRIDE, points_per_cell=P, scale=ds["density_scale"])
    for key in GRID_KEYS:
        assert _mean_pull2(ds["density3d"]["total"], th, key) < 2.0, key
    one = skewer_density_statistics([{"HI": c["HI"]} for c in cols[:2]], P, gp, b.box_cMpc)  # any number of species
    assert list(one["density3d"]) == ["HI", "total"] and one["density_identity"][1] < 1e-13
    with pytest.raises(ValueError, match="at least one species"):
        skewer_density_statistics([{}], P, gp, b.box_cMpc)
