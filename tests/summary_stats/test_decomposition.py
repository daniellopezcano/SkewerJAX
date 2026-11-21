"""The general decomposition of the total overflux statistics (``summary_stats.decomposition``): the sum rule for every
reduction, the commuting diagram term by term, the recovery of ``decompose_1d``, the symmetries of the pairs, one line,
the 𝒞 terms, the memory guard and the ensemble. Toy lattices, in the run's precision (``helpers.tol``)."""

from itertools import combinations

import numpy as np
import pytest

from skewerjax.config import real_dtype
from skewerjax.summary_stats import (
    GRID_KEYS,
    DeltaFieldBuilder,
    OverfluxStatistics1D,
    decompose,
    decompose_fields,
    ensemble_decomposition,
    lattice_grid,
    measure_lattice,
    notes_orders,
    skewer_p1d,
)
from skewerjax.summary_stats import decomposition as D

from helpers import raises, rel, tol

NX, NZ, NL, LBOX, DV = 6, 6, 24, 60.0, 2.5  # the lattice of skewers: 6 x 6 skewers of 24 pixels; dv [km/s]
ALL = GRID_KEYS + ("P1D", "xi1D")


def _grid():
    return lattice_grid(NX, NZ, NL, LBOX, LBOX / NX, k_edges=np.linspace(2 * np.pi / LBOX, np.pi * NX / LBOX, 4))


def _tau(n_lines, seed=0):
    """Correlated optical depths (N_lines, NX NZ, NL), x-major skewers, each line a shifted, scaled copy plus noise."""
    rng = np.random.default_rng(seed)
    g = rng.normal(size=(NX, NZ, NL))
    return np.stack(
        [
            (0.6 / (j + 1)) * np.exp(0.4 * (np.roll(g, 2 * j, axis=-1) + 0.4 * rng.normal(size=g.shape))).reshape(NX * NZ, NL)
            for j in range(n_lines)
        ]
    )


def _built(tau, mean="redshift", Tb=None):
    return DeltaFieldBuilder(tau, dtype=real_dtype(), mean=mean, mean_transmission=Tb).build()


def _fields(ov):
    """delta_lines and C of a built overflux."""
    return np.stack([np.asarray(ov.delta_terms[k]) for k in ov.labels]), ov.C_per_pixel


@pytest.mark.parametrize("mean", ["global", "redshift"])
@pytest.mark.parametrize("n_lines", [1, 2, 3, 4])
def test_sum_rule_every_reduction(n_lines, mean):
    """sum_{A,B} S[F_A, F_B] = S[delta_tot, delta_tot] for every reduction (P and xi: isotropic, maps, wedges, the axis
    planes, P1D, xi1D), the total by the existing estimators (``measure_lattice``), by order and, for L <= 3, by pair;
    the pairs regroup into the same orders, 𝒞 terms and split of order 2."""
    grid, ov = _grid(), _built(_tau(n_lines), mean)
    d, C = _fields(ov)
    r = decompose_fields(d, C, grid, DV, total=ov.total)
    assert r["P1D_order"].shape == (2 * n_lines + 1, NL // 2 + 1) and r["xi1D_order"].shape == (2 * n_lines + 1, NL)
    for k in GRID_KEYS:
        assert r[f"{k}_order"].shape[1:] == r[f"{k}_total"].shape
    assert set(r["sum_rule"]) == set(ALL)
    assert max(r["sum_rule"].values()) < tol(1e-12, 5e-6), r["sum_rule"]
    ref = measure_lattice(np.asarray(ov.total), grid)
    for k in GRID_KEYS:
        np.testing.assert_array_equal(r[f"{k}_total"], ref[k], err_msg=f"{k}: the total is the existing estimator")
    if n_lines <= 3:
        b = decompose_fields(d, C, grid, DV, total=ov.total, group="both")
        assert b["pairs"].shape == (2 ** (n_lines - 1) * (2**n_lines + 1), 2)
        assert max(b["sum_rule"].values()) < tol(1e-12, 5e-6), b["sum_rule"]
        for k in ALL:
            for g in ("order", "C", "o2"):
                sc = max(np.nanmax(np.abs(r[f"{k}_{g}"])), 1e-30)  # a normal floor: a denormal may be flushed to zero
                assert np.nanmax(np.abs(b[f"{k}_{g}"] - r[f"{k}_{g}"])) <= tol(1e-12, 5e-6) * sc, (k, g)
        np.testing.assert_allclose(b["pair_orders"], [len(a.replace("𝒞", "")) + len(c.replace("𝒞", "")) for a, c in b["pairs"]])


def test_commuting_diagram_term_by_term():
    """Every pair term: P1D = L_perp^-2 sum_{k_perp} P3D (on the full fftn grid) = the mean over the skewers of its
    per-sightline cross spectrum, and xi1D = the r_perp = 0 slice of its xi3D (notes, Eqs. Pk3D_P1D_from_P3D,
    xi3D_sample_zero_lag, Pk3D_diagram), from fields built here; and in the record itself, the r_perp = 0 column of the
    binned (r_par, r_perp) map + P(k = 0) / V (the box mean the 3D statistics drop) = xi1D."""
    grid, ov = _grid(), _built(_tau(2))
    d, C = _fields(ov)
    r = decompose_fields(d, C, grid, DV, group="pair", reductions=("P1D", "xi1D", "xi_map"))
    assert r["diagram_xi1D"] < tol(1e-12, 1e-6)
    assert grid.rperp_edges[1] <= LBOX / NX and r["xi_map_pair"].shape[1] == NL // 2 + 1, "column 0: r_perp = 0 alone"
    keys, members, _ = D.subset_terms(ov.labels)
    C = np.broadcast_to(np.asarray(C, dtype=float), d[0].shape)
    F = [-C / (1 + C)] + [np.prod(d[list(m)].astype(float), axis=0) / (1 + C) for m in members[1:]]
    lat = lambda f: f.reshape(NX, NZ, NL).swapaxes(-1, -2)
    dV, N, V, u = grid.dV, grid.N, grid.V, DV / (LBOX / NL)
    t = tol(1e-10, 1e-5)
    sP, sx = np.abs(r["P1D_pair"]).max(), np.abs(r["xi1D_pair"]).max()  # the scale of the terms (CS_1 is zero to rounding)
    err = lambda x, y, sc: float(np.max(np.abs(np.asarray(x) - np.asarray(y))) / sc)
    for i, (a, b) in enumerate(r["pairs"]):
        fa, fb = F[keys.index(a)], F[keys.index(b)]
        sym = 2.0 if a != b else 1.0
        p1 = sym * (DV / NL) * np.mean((np.fft.rfft(fa) * np.conj(np.fft.rfft(fb))).real, axis=0)  # per sightline
        assert err(r["P1D_pair"][i], p1, sP) < t, (a, b)
        p3 = sym * (dV / N) * (np.fft.fftn(lat(fa)) * np.conj(np.fft.fftn(lat(fb)))).real  # the full grid
        kperp_sum = p3.sum(axis=(0, 2))[: NL // 2 + 1] / (LBOX * LBOX) * u
        assert err(r["P1D_pair"][i], kperp_sum, sP) < t, (a, b)
        xi3 = np.fft.ifftn(p3).real * N / (N * dV)  # (1/N) sum_x a(x) b(x + r) summed both ways, k = 0 kept
        assert err(r["xi1D_pair"][i], np.fft.fftshift(xi3[0, :, 0]), sx) < t, (a, b)
        col = r["xi_map_pair"][i][:, 0] + sym * (dV / N) * fa.sum() * fb.sum() / V
        x1 = r["xi1D_pair"][i][(NL // 2 + np.arange(NL // 2 + 1)) % NL]  # lags j dv, j = 0 .. NL / 2
        assert err(col, x1, sx) < t, (a, b)


@pytest.mark.parametrize("mean", ["redshift", "global", "skewer"])
@pytest.mark.parametrize("n_lines", [2, 3])
def test_recovers_decompose_1d(n_lines, mean):
    """The 1D orders and totals of the general decomposition equal those of the dedicated 1D path (``decompose_1d``'s
    ``OverfluxStatistics1D.summary``: per-skewer FFTs) to rounding, k = 0 included."""
    tau = _tau(n_lines, seed=3)
    old = OverfluxStatistics1D(_built(tau, mean), DV).summary()
    new = decompose(tau, DV, grid=_grid(), mean=mean, dtype=real_dtype(), reductions=("P1D", "xi1D"))
    for a, b in (("P_order", "P1D_order"), ("xi_order", "xi1D_order"), ("P_total", "P1D_total"), ("xi_total", "xi1D_total")):
        assert old[a].shape == new[b].shape, (a, b)
        assert rel(new[b], old[a]) < tol(1e-12, 2e-6), (a, rel(new[b], old[a]))
    np.testing.assert_allclose(new["k_par"], old["k"])
    np.testing.assert_allclose(new["lags"], old["lags"])
    if real_dtype() == np.float64:  # the same as decompose_1d itself (dtype float)
        from skewerjax.summary_stats import decompose_1d

        o1 = decompose_1d(tau, DV, mean=mean)
        assert rel(new["P1D_order"], o1["P_order"]) < 1e-12 and rel(new["xi1D_order"], o1["xi_order"]) < 1e-12


def test_symmetries_and_the_phase_of_an_offset():
    """Pk3D_cross_symmetry, xi3D_cross_symmetry: the autos and every symmetric sum are even in the lag; a second line that
    is the first shifted by Delta_0 = s dv gives P_12 = e^{+i k_par Delta_0} P_11 (1D: Re and Im; 3D: every cell of the
    (k_par, k_perp) map, whose rows share |k_par|) and xi_12 - xi_21 = xi_11(Delta - Delta_0) - xi_11(Delta + Delta_0)."""
    grid, s = _grid(), 5
    d1 = np.asarray(_built(_tau(1)).delta_terms["1"], dtype=real_dtype())
    d = np.stack([d1, np.roll(d1, s, axis=-1)])  # delta_2(v) = delta_1(v - s dv)
    r = decompose_fields(d, 0.0, grid, DV, group="pair", reductions=("P1D", "xi1D", "P_map"))
    pairs = [tuple(p) for p in r["pairs"]]
    i11, i22, i12 = pairs.index(("1", "1")), pairs.index(("2", "2")), pairs.index(("1", "2"))
    t = tol(1e-12, 2e-6)
    for i in range(len(pairs)):  # every stored (symmetric) term is even in the lag
        x = r["xi1D_pair"][i]
        assert np.max(np.abs(x[1:] - x[1:][::-1])) <= t * max(np.abs(x).max(), 1e-30), pairs[i]
    k = r["k_par"]
    P11 = r["P1D_pair"][i11]
    assert rel(r["P1D_pair"][i22], P11) < t, "a shifted copy has the same auto spectrum"
    assert rel(r["P1D_pair"][i12], 2 * np.cos(k * s * DV) * P11) < t
    ia = [tuple(p) for p in r["anti_pairs"]].index(("1", "2"))
    assert rel(r["P1D_imag"][ia], np.sin(k * s * DV) * P11) < t
    x11 = r["xi1D_pair"][i11]
    assert rel(r["xi1D_anti"][ia], np.roll(x11, s) - np.roll(x11, -s)) < t
    m = np.round(grid.kpar * LBOX / (2 * np.pi)).astype(int)  # the rows of the map: k_par = 2 pi m / L
    np.testing.assert_allclose(
        r["P_map_pair"][i12], 2 * np.cos(2 * np.pi * m * s / NL)[:, None] * r["P_map_pair"][i11], rtol=t, atol=t * np.nanmax(P11)
    )


def test_one_line():
    """L = 1: 𝒞 = 0, so F_{} = 0, the orders are 0, 1, 2 (only order 2 non-zero) and the total is the line's own
    statistic, in 3D (``measure_lattice``) and in 1D (``skewer_p1d``)."""
    grid, ov = _grid(), _built(_tau(1))
    d, C = _fields(ov)
    assert np.all(np.asarray(C) == 0.0)
    r = decompose_fields(d, C, grid, DV, total=ov.total)
    assert r["orders"].tolist() == [0, 1, 2] and r["C_rows"].tolist() == ["CC", "CS1"]
    for k in ALL:
        assert np.nanmax(np.abs(r[f"{k}_order"][:2])) == 0.0, k
    ref = measure_lattice(d[0], grid)
    for k in GRID_KEYS:
        assert rel(np.nan_to_num(r[f"{k}_order"][2]), np.nan_to_num(ref[k])) < tol(1e-12, 2e-6), k
    _, p1 = skewer_p1d(d[0], DV)
    assert rel(r["P1D_order"][2], p1) < tol(1e-12, 2e-6)


def test_the_C_terms():
    """The 𝒞 terms (notes, Eqs. xi1D_general_L_step1, xi1D_general_L_decomposition). With the means of this realisation
    ("redshift"), CS_1 = 0 (sum_{x_perp} delta_j = 0 at every pixel) and sum_{n>=2} CS_n = -2 CC, so the notes' collected
    order 0 is -CC; with the means of a larger ensemble CS_1 is not zero (it vanishes only on average). ``notes_orders``
    regroups the orders with the same sum."""
    grid, tau = _grid(), _tau(3, seed=7)
    r = decompose(tau, DV, grid=grid, dtype=real_dtype())
    for k in ("P1D", "xi1D", "xi_map"):
        Cs = r[f"{k}_C"]
        sc = np.nanmax(np.abs(Cs[0]))
        assert np.nanmax(np.abs(Cs[1])) < tol(1e-10, 2e-5) * sc, k
        assert np.nanmax(np.abs(Cs[2:].sum(0) + 2 * Cs[0])) < tol(1e-10, 2e-5) * sc, k
        no = notes_orders(r, k)
        assert np.nanmax(np.abs(no[1])) < tol(1e-10, 2e-5) * sc
        assert np.nanmax(np.abs(no[0] + Cs[0])) < tol(1e-10, 2e-5) * sc, "the collected order 0: -CC"
        np.testing.assert_allclose(np.nansum(no, 0), np.nansum(r[f"{k}_order"], 0), rtol=1e-6, atol=1e-6 * sc)
    other = np.exp(-_tau(3, seed=8))  # the means of a larger ensemble: these skewers and others
    T = np.concatenate([np.exp(-tau), other], axis=1)
    Tb = np.concatenate([T, T.prod(0)[None]]).mean(axis=1)
    r2 = decompose(tau, DV, Tb, grid=grid, dtype=real_dtype())
    assert np.abs(r2["xi1D_C"][1]).max() > 1e-3 * np.abs(r2["xi1D_C"][0]).max(), "CS_1 is not zero for one realisation"
    assert max(r2["sum_rule"].values()) < tol(1e-12, 5e-6)


def test_memory_guard_and_errors(monkeypatch):
    """The estimate grows as 2^L; a decomposition beyond the available memory raises ``MemoryError``; bad inputs raise."""
    grid = _grid()
    m = [D.decomposition_memory(grid, n) for n in (1, 2, 3, 4)]
    assert all(b > a for a, b in zip(m, m[1:]))
    tau = _tau(2)
    monkeypatch.setattr(D, "_available_memory", lambda: 1000)
    raises(MemoryError, "GiB available", decompose, tau, DV, grid=grid)
    monkeypatch.setattr(D, "_available_memory", lambda: None)
    ov = _built(tau)
    d, C = _fields(ov)
    raises(ValueError, "group must be", decompose_fields, d, C, grid, DV, group="orders")
    raises(ValueError, "unknown reductions", decompose_fields, d, C, grid, DV, reductions=("P3D",))
    raises(ValueError, "does not hold", decompose_fields, d[:, :30], C, grid, DV)
    raises(ValueError, "no mass assignment", decompose_fields, d, C, grid, DV, mas="cic")


def test_ensemble_and_selected_reductions():
    """``ensemble_decomposition`` averages the records (mean, error, every realisation); ``reductions`` limits the work to
    the keys asked; ``GridStatistics3D._bin(keys=...)`` gives the same arrays as the full binning."""
    grid = _grid()
    recs = [decompose(_tau(2, seed=s), DV, grid=grid, dtype=real_dtype(), reductions=("P_iso", "P1D")) for s in (1, 2, 3)]
    assert {k for k in recs[0] if k.endswith("_order")} == {"P_iso_order", "P1D_order"}
    e = ensemble_decomposition(recs)
    assert e["n_real"] == 3 and e["P1D_order_each"].shape == (3,) + recs[0]["P1D_order"].shape
    np.testing.assert_allclose(e["P1D_order"], np.mean([r["P1D_order"] for r in recs], axis=0), rtol=1e-6)
    assert np.all(np.isfinite(e["P1D_order_err"])) and e["sum_rule"]["P1D"] == max(r["sum_rule"]["P1D"] for r in recs)
    d = np.random.default_rng(0).normal(size=grid.shape)
    full = grid.measure(d)
    import jax.numpy as jnp

    f2 = (jnp.abs(jnp.fft.rfftn(jnp.asarray(d))) ** 2).at[0, 0, 0].set(0.0)
    part = grid._bin((grid.dV / grid.N) * f2, None, keys=("P_mu", "P_kpar0"))
    assert set(part) == {"P_mu", "P_kpar0"}
    for k in part:
        np.testing.assert_array_equal(part[k], full[k])


def test_subset_terms_order():
    """The terms follow ``DeltaFieldBuilder.subset_keys`` after the 𝒞 term."""
    keys, members, orders = D.subset_terms(["a", "b", "c"], "+")
    ov = _built(_tau(3))
    assert keys[1:] == ["+".join(x) for r in (1, 2, 3) for x in combinations("abc", r)]
    assert [k.replace("+", "") for k in keys[1:]] == [
        k.replace("1", "a").replace("2", "b").replace("3", "c") for k in ov.subset_keys
    ]
    assert orders == [0, 1, 1, 1, 2, 2, 2, 3] and members[0] == ()


@pytest.mark.parametrize("n_lines", [2, 3])
def test_the_wedge_sum_rule(n_lines):
    """The mu wedges of P and xi (the edges of ``lattice_grid``, separate for P and xi): the sum of the orders, and of the
    pairs, equals the total in every wedge and every bin (``wedge_sum_rule``, one value per wedge, the largest over the
    realisations of an ensemble); the total is the wedge of ``measure_lattice``; the wedges hold the separate edges."""
    mu_p, mu_x = np.sin(np.linspace(0.0, np.pi / 2, 7)), (0.0, 0.5, 0.8, 0.95, 1.0)
    grid = lattice_grid(
        NX,
        NZ,
        NL,
        LBOX,
        LBOX / NX,
        k_edges=np.linspace(2 * np.pi / LBOX, np.pi * NX / LBOX, 4),
        pk_mu_edges=mu_p,
        xi_mu_edges=mu_x,
    )
    recs = [
        decompose(_tau(n_lines, seed=s), DV, grid=grid, dtype=real_dtype(), group="both", reductions=D.WEDGE_REDUCTIONS)
        for s in (4, 5)
    ]
    for r, s in zip(recs, (4, 5)):
        ov = _built(_tau(n_lines, seed=s))
        assert r["P_mu_order"].shape[1] == len(mu_p) - 1 and r["xi_mu_order"].shape[1] == len(mu_x) - 1
        ref = measure_lattice(np.asarray(ov.total), grid)
        for R in D.WEDGE_REDUCTIONS:
            np.testing.assert_array_equal(r[f"{R}_total"], ref[R])
            for g in ("order", "pair"):
                w = D.wedge_sum_rule(r, R, g)
                assert w.shape == (r[f"{R}_total"].shape[0],) and np.all(w < tol(1e-12, 5e-6)), (R, g, w)
                diff = np.abs(r[f"{R}_{g}"].sum(0) - r[f"{R}_total"])  # by hand, wedge by wedge
                sc = np.maximum(np.nanmax(np.abs(r[f"{R}_total"]), -1), np.nanmax(np.abs(r[f"{R}_{g}"]), (0, -1)))
                np.testing.assert_allclose(w, np.nanmax(diff, -1) / sc, rtol=1e-12, atol=0)
    e = ensemble_decomposition(recs)
    for R in D.WEDGE_REDUCTIONS:
        np.testing.assert_array_equal(D.wedge_sum_rule(e, R), np.maximum(*(D.wedge_sum_rule(r, R) for r in recs)))
    raises(ValueError, "reduction must be", D.wedge_sum_rule, recs[0], "P_iso")
    raises(ValueError, "has no", D.wedge_sum_rule, {"P_mu_order": recs[0]["P_mu_order"]}, "P_mu")
