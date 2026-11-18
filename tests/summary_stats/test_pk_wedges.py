"""The mu wedges of P (``GridStatistics3D``, ``box_grid``, ``lattice_grid``, ``field_statistics``,
``lattice_density_theory``): the wedges of P and of xi take separate edges (``pk_mu_edges``, ``xi_mu_edges``); a mode
belongs to exactly one wedge (half-open bins, mu = 1 in the last), so the mode counts of the wedges add up to those of
|k|; the expectation is P(k, mu) averaged over exactly the modes of each (wedge, |k|) cell, the Kaiser term included;
an isotropic toy field gives equal wedges and a Kaiser one recovers (1 + beta mu^2)^2 between them."""

import jax
import numpy as np
import pytest

from skewerjax.simulations import PowerLawGRF, ToyBoxes, grf_cells
from skewerjax.summary_stats import (
    MU_WEDGES,
    GridStatistics3D,
    box_grid,
    field_statistics,
    grid_limits,
    grid_power,
    lattice_density_theory,
    lattice_grid,
)
from skewerjax.summary_stats.binning import bin_index

from helpers import raises, tol

N, L = 32, 64.0
PK_EDGES = (0.0, 0.3, 0.6, 1.0)  # mu = 0.6 = 3 / 5 is the mu of the modes (4, 3, 0) k_f: on an inner edge
XI_EDGES = (0.0, 0.5, 1.0)


def _full_modes(n, side, los_axis=1):
    """|k| and mu = |k_los| / |k| of every mode of the full FFT grid (each counted once), k = 0 dropped."""
    f = 2 * np.pi * np.fft.fftfreq(n, side / n)
    kv = np.meshgrid(f, f, f, indexing="ij")
    k = np.sqrt(sum(x**2 for x in kv)).ravel()
    mu = np.abs(kv[los_axis]).ravel() / np.where(k > 0, k, 1.0)
    return k[k > 0], mu[k > 0], [x.ravel()[k > 0] for x in kv]


def _grid(**kw):
    lim = grid_limits(N, L)
    k_edges = np.linspace(0.5 * lim["k_min"], lim["k_max"], 13)
    return box_grid(N, L, k_edges=k_edges, r_edges=np.arange(0, L / 2 + 4.0, 4.0), log=None, **kw)


def test_the_wedges_of_p_and_xi_take_separate_edges():
    """``pk_mu_edges`` bins the wedges of P, ``xi_mu_edges`` those of xi; each output depends on its own edges only (bit
    for bit against a grid with both edges equal), and the axes hold "mu_edges" (xi, as before) and "pk_mu_edges"."""
    gs = _grid(xi_mu_edges=XI_EDGES, pk_mu_edges=PK_EDGES)
    a = gs.axes()
    np.testing.assert_array_equal(a["mu_edges"], XI_EDGES)
    np.testing.assert_array_equal(a["pk_mu_edges"], PK_EDGES)
    np.testing.assert_array_equal(a["k_mu_edges"], a["k_edges"])  # the |k| bins of the wedges: those of P(|k|)
    assert a["k_mu"].shape == a["n_modes_P_mu"].shape == (len(PK_EDGES) - 1, a["k_edges"].size - 1)
    assert a["r_mu"].shape == (len(XI_EDGES) - 1, a["r_mu_edges"].size - 1)
    d = np.asarray(PowerLawGRF(1.0, 0.1, -1.0, L=(L,) * 3, N=(N,) * 3, los_axis=1).sample(jax.random.PRNGKey(0)))
    m = gs.measure(d)
    assert m["P_mu"].shape[0] == len(PK_EDGES) - 1 and m["xi_mu"].shape[0] == len(XI_EDGES) - 1
    only_xi, only_pk = _grid(xi_mu_edges=XI_EDGES, pk_mu_edges=XI_EDGES), _grid(xi_mu_edges=PK_EDGES, pk_mu_edges=PK_EDGES)
    np.testing.assert_array_equal(m["xi_mu"], only_xi.measure(d)["xi_mu"])
    np.testing.assert_array_equal(m["P_mu"], only_pk.measure(d)["P_mu"])
    for key in ("P_iso", "P_map", "xi_iso", "xi_map"):  # the other binnings do not see the wedges
        np.testing.assert_array_equal(m[key], _grid().measure(d)[key])
    default = _grid()  # the defaults: both MU_WEDGES, as the single edges were
    np.testing.assert_array_equal(default.axes()["mu_edges"], MU_WEDGES)
    np.testing.assert_array_equal(default.axes()["pk_mu_edges"], MU_WEDGES)
    lat = lattice_grid(4, 4, 32, L, L / 4, k_edges=np.linspace(0.05, 0.3, 6), xi_mu_edges=XI_EDGES, pk_mu_edges=PK_EDGES)
    np.testing.assert_array_equal(lat.axes()["pk_mu_edges"], PK_EDGES)
    np.testing.assert_array_equal(lat.axes()["mu_edges"], XI_EDGES)
    for bad in ((0.0,), (0.0, 0.5, 0.5, 1.0), (-0.1, 1.0), (0.0, 1.2)):
        raises(ValueError, "pk_mu_edges", GridStatistics3D, (8, 8, 8), 8.0, pk_mu_edges=bad)


def test_every_mode_lies_in_one_wedge():
    """The edge rule: half-open bins [mu_i, mu_i+1), the last closed. The counts of every (wedge, |k|) cell equal a direct
    count over the full FFT grid (modes on the inner edge mu = 0.6 in the upper wedge, mu = 1 in the last, mu = 0 in the
    first), and the counts of the wedges add up to those of |k|, on the mesh and on a lattice of skewers."""
    gs = _grid(pk_mu_edges=PK_EDGES)
    a = gs.axes()
    k, mu, _ = _full_modes(N, L)
    assert np.any(np.isclose(mu, 0.6, rtol=0, atol=1e-12)), "modes on the inner edge exist"
    iw, ik = bin_index(mu, np.asarray(PK_EDGES)), bin_index(k, a["k_mu_edges"])
    ok = (iw >= 0) & (ik >= 0)
    direct = np.zeros(a["n_modes_P_mu"].shape)
    np.add.at(direct, (iw[ok], ik[ok]), 1.0)
    np.testing.assert_array_equal(np.asarray(a["n_modes_P_mu"]), direct)
    on_edge = np.isclose(mu, 0.6, rtol=0, atol=1e-12)
    assert np.all(iw[on_edge] == 2) and np.all(iw[np.isclose(mu, 1.0)] == 2) and np.all(iw[mu == 0] == 0)
    np.testing.assert_array_equal(np.sum(a["n_modes_P_mu"], axis=0), a["n_modes_P_iso"])
    lat = lattice_grid(4, 4, 64, L, L / 4, k_edges=np.linspace(0.05, 0.35, 7), pk_mu_edges=PK_EDGES)
    la = lat.axes()
    np.testing.assert_array_equal(np.sum(la["n_modes_P_mu"], axis=0), la["n_modes_P_iso"])


def _kaiser_case(beta, n_real=8):
    """``field_statistics`` of ``n_real`` toy fields of a power law with the Kaiser factor (1 + beta mu^2)^2 along y
    (none for beta None), as cells through the mesh, against the exact expectation on the modes (``power_mu``)."""
    g1 = PowerLawGRF(1.0, 0.1, -1.0, L=(L,) * 3, N=(N,) * 3, beta=beta, los_axis=1)
    g = PowerLawGRF(
        0.01 / float(g1.variance()), 0.1, -1.0, L=(L,) * 3, N=(N,) * 3, beta=beta, los_axis=1
    )  # sigma = 0.1: 1 + delta > 0
    gs = _grid(pk_mu_edges=PK_EDGES)
    cells = (grf_cells(g.sample(jax.random.PRNGKey(i)), g, {"HI": 1.0}) for i in range(n_real))
    power = g.power if beta is not None else (lambda k, mu: g.power(k))
    return g, gs, field_statistics(cells, gs, species="HI", power=power, power_mu=True, log=None)


def test_the_expectation_is_exact_on_the_modes_of_each_cell():
    """The expectation of P_mu is P(k, mu) of the Kaiser model averaged over exactly the modes of every (wedge, |k|) cell
    (a direct mean over the full FFT grid); ``grid_power(los_axis=)`` is ``PowerLawGRF.power_grid`` bit for bit; with
    ``power_mu`` an isotropic callable gives the isotropic expectation, on the mesh and on the lattice of skewers."""
    g = PowerLawGRF(2.0, 0.1, -1.0, L=(L,) * 3, N=(N,) * 3, beta=0.8, sigma_fog=3.0, los_axis=1)
    P = grid_power(g.power, (N,) * 3, L, los_axis=1)
    assert np.array_equal(np.asarray(P), np.asarray(g.power_grid())), "bit for bit"
    gs = _grid(pk_mu_edges=PK_EDGES)
    th = gs.theory(P)["P_mu"]
    k, mu, kv = _full_modes(N, L)
    Pk = np.asarray(g.power_at(*kv), dtype=np.float64)
    iw, ik = bin_index(mu, np.asarray(PK_EDGES)), bin_index(k, gs.k_mu_edges)
    ok = (iw >= 0) & (ik >= 0)
    s, c = np.zeros(th.shape), np.zeros(th.shape)
    np.add.at(s, (iw[ok], ik[ok]), Pk[ok])
    np.add.at(c, (iw[ok], ik[ok]), 1.0)
    np.testing.assert_allclose(th, np.where(c > 0, s / np.where(c > 0, c, 1), np.nan), rtol=tol(1e-12, 2e-6), equal_nan=True)
    iso = PowerLawGRF(2.0, 0.1, -1.0, L=(L,) * 3, N=(N,) * 3)
    a = grid_power(iso.power, (N,) * 3, L)
    b = grid_power(lambda kk, m: iso.power(kk), (N,) * 3, L, los_axis=1)
    assert np.array_equal(np.asarray(a), np.asarray(b))
    box = ToyBoxes(box_cMpc=32.0, n_cells=16, n_real=1, r_bao=10.0, sigma_bao=2.0, bao_ratio=0.01)
    lat = lattice_grid(8, 8, 16, 32.0, 4.0, k_edges=np.linspace(0.1, 0.78, 5), pk_mu_edges=PK_EDGES)
    gb = box.grf()
    one = lattice_density_theory(gb.power, lat, 16, (2, 2))
    two = lattice_density_theory(lambda kk, m: gb.power(kk), lat, 16, (2, 2), power_mu=True)
    for key in one:
        np.testing.assert_array_equal(one[key], two[key])


@pytest.mark.parametrize("beta", [None, 1.0])
def test_the_wedges_recover_the_kaiser_factor(beta):
    """Through the mesh (k < k_Nyq / 2, the trusted range): every wedge follows its exact expectation (the mean squared
    diagonal pull over every cell < 2.5, no pull beyond 4). Divided by the isotropic expectation on the same modes (the power law alone), each wedge
    gives the Kaiser factor of its modes: 1 in every wedge for an isotropic field (equal wedges), and for a Kaiser one
    the wedge average of (1 + beta mu^2)^2, within 5 % of the continuous average in the expectation and within 10 % of
    the expectation in the measurement."""
    g, gs, res = _kaiser_case(beta)
    a = res["axes"]
    sel = np.all(np.isfinite(res["P_mu"]), axis=0) & np.all(a["k_mu"] < 0.5 * res["k_nyq"], axis=0)
    sel &= np.all(a["n_modes_P_mu"] >= 20, axis=0)
    assert sel.sum() >= 4
    pull = (res["P_mu"] - res["theory"]["P_mu"])[:, sel] / res["P_mu_err"][:, sel]
    # 8 realisations: each pull follows Student's t with 7 dof, <pull^2> = 7 / 5; the joint mean over every cell, and no outlier
    assert np.mean(pull**2) < 2.5 and np.max(np.abs(pull)) < 4.0, pull
    iso = gs.theory(grid_power(PowerLawGRF(g.A_s, g.k_0, g.n_s, L=(L,) * 3, N=(N,) * 3).power, gs.shape, gs.L))["P_mu"][:, sel]
    K = lambda m: (1.0 + (beta or 0.0) * m**2) ** 2  # noqa: E731
    expect = np.array([np.mean(K(np.linspace(lo, hi, 20001))) for lo, hi in zip(PK_EDGES[:-1], PK_EDGES[1:])])
    f_th = res["theory"]["P_mu"][:, sel].sum(1) / iso.sum(1)
    f_meas = res["P_mu"][:, sel].sum(1) / iso.sum(1)
    np.testing.assert_allclose(f_th, expect, rtol=0.05)
    np.testing.assert_allclose(f_meas, f_th, rtol=0.1)
    if beta is None:
        np.testing.assert_allclose(f_th, 1.0, rtol=tol(1e-12, 1e-6))
    else:
        assert f_th[-1] / f_th[0] > 2.0, "the Kaiser factor separates the wedges"
