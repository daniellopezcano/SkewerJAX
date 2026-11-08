"""Summary statistics: P1D / xi1D / P3D recovery of Gaussian random fields, the exact decomposition by order and line
pair, the shift signature of a second line, the linear-flux McDonald form and the McDonald / Ma contamination fits
(review notes, P1D, xi1D and Sec. Overflux; formerly ``scripts/validation/check_summary_statistics.py``, same
tolerances; 100 skewers and a 16^3 grid by default, 500 and 32^3 with the ``slow`` marker)."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from skewerjax.simulations import PowerLawGRF
from skewerjax.summary_stats import (
    DeltaFieldBuilder,
    OverfluxStatistics1D,
    OverfluxStatistics3D,
    fit_ma_model_profiled,
    fit_mcdonald_a,
    mcdonald_factor,
    xi_from_power,
)
from skewerjax.utils.plotting import hybrid_k_edges, pk_sigma

from helpers import float64_only


def statistics_checks(n_s_, n3):
    """Every check of the module docstring for ``n_s_`` GRF skewers and a ``n3``^3 3D grid: [(name, value, tolerance, expect)]."""
    checks = []

    def check(name, value, tol, expect=False, detail=""):
        checks.append((name, float(value), tol, expect))

    Npix, L = 1000, 100.0
    dv = L / Npix
    shift = Npix // 10  # second line at dv12 = L / 10
    k_0, n_s = 2 * np.pi / 10.0, 0.5
    A_s = 1.0 / float(PowerLawGRF(1.0, k_0, n_s, L=L, N=Npix).variance())  # sigma_delta = 1
    grf = PowerLawGRF(A_s, k_0, n_s, L=L, N=Npix)
    keys = jax.random.split(jax.random.PRNGKey(137), n_s_)
    delta = jax.vmap(grf.sample)(keys)
    single = lambda d: SimpleNamespace(subset_keys=["a"], delta_terms={"a": d}, C_per_pixel=jnp.zeros((1, d.shape[1])), total=d)
    edges = hybrid_k_edges(2 * np.pi / L, grf.k_max[0])

    def pulls(binned, expect):
        P_b, _, _, n_b = (np.asarray(x) for x in binned)
        s = pk_sigma(expect, n_b)
        ok = np.isfinite(P_b) & (n_b > 0) & np.isfinite(s)
        return (P_b[ok] - np.asarray(expect)[ok]) / s[ok]

    # GRF level
    st_g = OverfluxStatistics1D(single(delta), dv).compute_xi1d()
    p = pulls(st_g.bin_p1d(edges), st_g.bin_p1d(edges, Pk=grf.power(st_g.k))[0])
    check(
        f"GRF, {n_s_} skewers: binned P1D = input (chi2 / dof)", np.mean(p**2), 2.0, detail=f"max |pull| {np.max(np.abs(p)):.1f}"
    )
    xi_in = np.fft.fftshift(np.asarray(xi_from_power(grf.power(st_g.k), Npix, dv)))
    xs = np.asarray(st_g.xi_total_per_sample)
    xi_err = xs.std(0, ddof=1) / np.sqrt(xs.shape[0])
    near = np.abs(np.asarray(st_g.lags)) <= 0.2 * L
    check(
        "GRF: xi1D = discrete inverse transform of the input (fraction of lags beyond 2 sigma, |dv| < L/5)",
        np.mean(np.abs(np.asarray(st_g.xi_total) - xi_in)[near] > 2 * xi_err[near]),
        0.25,
    )
    # composite field: two lines, the second shifted by dv12
    tau = jnp.stack([jnp.exp(delta), jnp.exp(jnp.roll(delta, shift, axis=-1))])
    st = OverfluxStatistics1D(DeltaFieldBuilder(tau).build(), dv).compute_p1d().compute_xi1d()
    sc = float(np.max(np.abs(np.asarray(st.Pk_total))))
    check(
        "decomposition, P1D: sum of the orders = total (max |diff| / max |P|)",
        np.max(np.abs(sum(np.asarray(v) for v in st.Pk_total_by_order.values()) - np.asarray(st.Pk_total))) / sc,
        1e-10,
    )
    check(
        "decomposition, xi1D: sum of the orders = total (max |diff| / max |xi|)",
        np.max(np.abs(sum(np.asarray(v) for v in st.xi_total_by_order.values()) - np.asarray(st.xi_total)))
        / float(np.max(np.abs(np.asarray(st.xi_total)))),
        1e-10,
    )
    k = np.asarray(st.k)
    P11, P12 = np.asarray(st.Pk_pair_complex[("1", "1")]), np.asarray(st.Pk_pair_complex[("1", "2")])
    check(
        "shift signature: P_12 = P_11 exp(+i k dv12) (max |diff| / max |P_11|)",
        np.max(np.abs(P12 - P11 * np.exp(1j * k * shift * dv))) / np.max(np.abs(P11)),
        1e-10,
    )
    xi12 = np.asarray(st.xi_pair[("1", "2")])
    check("shift signature: xi_12 peaks at +dv12 [km/s]", abs(np.argmax(xi12) * dv - shift * dv), dv)
    # linear flux regime (sigma_delta = 0.01): P_tot -> tau0^2 (1 + a^2 + 2 a cos k dv12) P_in (McDonald form)
    a_lin = 0.5
    g_lin = PowerLawGRF(1e-4 * A_s, k_0, n_s, L=L, N=Npix)
    d_lin = jax.vmap(g_lin.sample)(keys)
    st_l = OverfluxStatistics1D(
        DeltaFieldBuilder(jnp.stack([jnp.exp(d_lin), a_lin * jnp.exp(jnp.roll(d_lin, shift, -1))])).build(), dv
    ).compute_p1d()
    k_l = np.asarray(st_l.k)
    P_exp = mcdonald_factor(k_l, a_lin, delta_v=shift * dv) * np.asarray(g_lin.power(k_l))
    e1 = hybrid_k_edges(2 * np.pi / L, grf.k_max[0], n_lin=Npix // 2, n_log=1)
    p = pulls(st_l.bin_p1d(e1), st_l.bin_p1d(e1, Pk=P_exp)[0])
    check("linear flux, two lines: P_tot = tau0^2 [1 + a^2 + 2a cos(k dv)] P_in (chi2 / dof)", np.mean(p**2), 2.0)
    fit = fit_mcdonald_a(
        k_l[1:], np.asarray(st_l.Pk_total)[1:], np.asarray(st_l.Pk_pair[("1", "1")])[1:], delta_v=shift * dv, k_range=(0.0, 1.0)
    )
    check("McDonald fit recovers the input a (relative error)", abs(fit["a"] / a_lin - 1), 0.02)
    fit_ma = fit_ma_model_profiled(
        k_l[1:],
        np.asarray(st_l.Pk_total)[1:],
        np.asarray(st_l.Pk_pair[("1", "1")])[1:],
        delta_v=shift * dv,
        k_range=(0.0, 1.0),
        run_global=False,
    )
    check("Ma et al. chi2 <= McDonald chi2 (nested model; ratio - 1)", fit_ma["chi2"] / fit["chi2"] - 1, 1e-9)
    # 3D GRF: P3D(|k|) recovery on an isotropic grid

    g3 = PowerLawGRF(1.0, 1.0, -1.0, L=(20.0,) * 3, N=(n3,) * 3)
    d3 = g3.sample(jax.random.PRNGKey(3))
    st3 = OverfluxStatistics3D(single(d3.reshape(n3 * n3, n3)), 20.0 / n3, 20.0 / n3).compute_p3d()
    b3 = st3.bin_p3d_kmod(n_bins=10)
    p = pulls(
        b3,
        st3.bin_p3d_kmod(
            k_edges=b3[2], Pk3d_input=g3.power(jnp.sqrt(st3.k_perp_2d[:, :, None] ** 2 + st3.k_par[None, None, :] ** 2))
        )[0],
    )
    check(f"GRF 3D ({n3}^3): binned P3D(|k|) = input (chi2 / dof)", np.mean(p**2), 2.0)
    return checks


def _assert_all(checks):
    bad = [(n, v, t) for n, v, t, expect in checks if not (v > t if expect else v <= t)]
    assert not bad, bad


def test_statistics():
    _assert_all(statistics_checks(100, 16))


@pytest.mark.slow
@float64_only
def test_statistics_full():
    _assert_all(statistics_checks(500, 32))


# ---------------------------------------------------------------------------------------------- 3D fields on their grid
def _brute_xi_iso(d, L, edges):
    """xi(|r|) by brute force: (1/N) sum_x d(x) d(x + r) for every minimum-image lag vector r (np.roll), averaged in bins."""
    n = d.shape[0]
    dx = L / n
    off = lambda j: dx * (j if j <= n // 2 else j - n)
    sums, cnt = np.zeros(len(edges) - 1), np.zeros(len(edges) - 1)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                r = np.sqrt(off(i) ** 2 + off(j) ** 2 + off(k) ** 2)
                b = np.searchsorted(edges, r, side="right") - 1
                if 0 <= b < len(edges) - 1:
                    sums[b] += np.mean(d * np.roll(d, (-i, -j, -k), axis=(0, 1, 2)))
                    cnt[b] += 1
    return sums / np.maximum(cnt, 1)


@float64_only
def test_grid_statistics_conventions():
    """``GridStatistics3D``: xi is the brute-force periodic correlation (the FFT pair and the minimum-image binning), the
    rfft half-space weights give the mean over every full-grid mode of a bin, and the expectation of xi at r = 0 is the
    variance of the field."""
    from skewerjax.summary_stats import GridStatistics3D

    n, L = 6, 6.0
    g = PowerLawGRF(A_s=1.0, k_0=1.0, n_s=-1.0, L=(L,) * 3, N=(n,) * 3)
    d = np.asarray(g.sample(jax.random.PRNGKey(4)))
    r_edges = np.array([0.0, 0.5, 1.5, 2.2, 3.1])
    gs = GridStatistics3D(d.shape, L, los_axis=1, r_edges=r_edges)
    m = gs.measure(d)
    np.testing.assert_allclose(m["xi_iso"], _brute_xi_iso(d, L, r_edges), rtol=1e-10, atol=1e-14)
    # P(|k|) from the full fftn: every mode of the bin with weight 1
    kk = 2 * np.pi * np.fft.fftfreq(n, L / n)
    kmod = np.sqrt(sum(k**2 for k in np.meshgrid(kk, kk, kk, indexing="ij")))
    P_full = np.abs(np.fft.fftn(d)) ** 2 * (L / n) ** 3 / n**3
    from skewerjax.summary_stats.binning import bin_index

    j = bin_index(kmod.ravel(), gs.k_edges).reshape(kmod.shape)
    ok = (kmod > 0) & (j >= 0)
    nb = gs.k_edges.size - 1
    ref = np.bincount(j[ok], weights=P_full[ok], minlength=nb) / np.maximum(np.bincount(j[ok], minlength=nb), 1)
    good = np.isfinite(m["P_iso"])
    np.testing.assert_allclose(m["P_iso"][good], ref[good], rtol=1e-10)
    th = gs.theory(g.power_grid())
    assert np.isclose(th["xi_iso"][0], float(g.variance()), rtol=1e-10), "xi_in(0) = sigma^2 (the only lag of bin 0)"


@pytest.mark.parametrize("n_real", [400])
def test_grid_statistics_recover_an_isotropic_field(n_real):
    """An isotropic GRF (power law + BAO, as the toy boxes of notebook 08, Part A): the mean over many realisations of P and xi,
    isotropic, in the (par, perp) plane and in wedges of mu, agrees with ``theory`` within the error of the mean."""
    from skewerjax.summary_stats import GRID_KEYS, GridStatistics3D, ensemble_mean

    g = PowerLawGRF(A_s=0.02, k_0=2.0, n_s=-1.25, L=(12.0,) * 3, N=(16,) * 3, A_bao=3e-4, r_bao=4.0, sigma_bao=0.8, los_axis=1)
    gs = GridStatistics3D(g.shape, g.L, los_axis=1)
    th = gs.theory(g.power_grid())
    keys = jax.random.split(jax.random.PRNGKey(11), n_real)
    e = ensemble_mean([gs.measure(g.sample(k)) for k in keys], GRID_KEYS)
    for key in GRID_KEYS:
        ok = np.isfinite(e[key]) & (e[key + "_err"] > 0)
        pull = (e[key][ok] - th[key][ok]) / e[key + "_err"][ok]
        assert np.mean(pull**2) < 2.0, (key, np.mean(pull**2))


def test_grid_xi_covariance():
    """``GridStatistics3D.xi_iso_covariance``: equal to the brute-force sum 2 sum_k P(k)^2 c_a(k) c_b(k) / V^2 (c_a the
    mean of cos(k.r) over the separations of bin a) on a small anisotropic grid, and to the scatter of many draws."""
    from skewerjax.summary_stats import GridStatistics3D
    from skewerjax.summary_stats.binning import bin_index

    n, L = (6, 8, 5), (6.0, 8.0, 7.0)
    g = PowerLawGRF(1.0, 1.0, -1.0, L=L, N=n, A_bao=0.02, r_bao=2.0, sigma_bao=0.6, los_axis=1)
    gs = GridStatistics3D(n, L, los_axis=1, r_edges=np.arange(0.0, 4.6, 0.9))
    C = gs.xi_iso_covariance(g.power_grid())
    dx = np.array(L) / np.array(n)
    K = np.stack(np.meshgrid(*[2 * np.pi * np.fft.fftfreq(m, d) for m, d in zip(n, dx)], indexing="ij"), -1).reshape(-1, 3)
    off = [d * np.where(np.arange(m) <= m // 2, np.arange(m), np.arange(m) - m) for m, d in zip(n, dx)]
    R = np.stack(np.meshgrid(*off, indexing="ij"), -1).reshape(-1, 3)
    idx = bin_index(np.linalg.norm(R, axis=1), gs.r_edges)
    cos = np.cos(R @ K.T)
    M = np.stack([cos[idx == b].mean(0) for b in range(gs.r_edges.size - 1)]) / np.prod(L)
    P = np.asarray(g.power_at(K[:, 0], K[:, 1], K[:, 2]))
    np.testing.assert_allclose(C, 2 * (M * P**2) @ M.T, rtol=1e-10, atol=1e-14 * np.abs(C).max())
    xs = np.stack([gs.measure(d)["xi_iso"] for d in jax.vmap(g.sample)(jax.random.split(jax.random.PRNGKey(0), 4000))])
    np.testing.assert_allclose(np.diag(np.cov(xs.T)), np.diag(C), rtol=0.08)


def test_grid_xi_multipoles():
    """``GridStatistics3D(multipoles=True)``: the Legendre fit returns xi_0, xi_2, xi_4 of xi = sum_l c_l L_l(mu) exactly
    in every |r| bin that spans enough values of mu (NaN in the others), on a lattice coarse across the line of sight."""
    from skewerjax.summary_stats import GridStatistics3D

    gs = GridStatistics3D((4, 32, 4), (32.0, 32.0, 32.0), los_axis=1, r_edges=np.arange(0.0, 17.0, 2.0), multipoles=True)
    off = [d * np.where(np.arange(n) <= n // 2, np.arange(n), np.arange(n) - n) for n, d in zip(gs.shape, gs.dx)]
    R = np.stack(np.meshgrid(*off, indexing="ij"), -1).reshape(-1, 3)
    r = np.linalg.norm(R, axis=1)
    mu = np.abs(R[:, 1]) / np.where(r > 0, r, 1.0)
    c = np.array([2.0, 3.0, -0.5])
    xi = c[0] + c[1] * 0.5 * (3 * mu**2 - 1) + c[2] * (35 * mu**4 - 30 * mu**2 + 3) / 8
    ell = gs._bin(np.zeros(gs._rfft_shape), xi)["xi_ell"]
    ok = np.all(np.isfinite(ell), axis=0)
    assert ok.sum() >= 3 and not ok[0], "the first bins hold line-of-sight separations only: no fit"
    np.testing.assert_allclose(ell[:, ok], np.broadcast_to(c[:, None], (3, ok.sum())), rtol=1e-9, atol=1e-9)


@float64_only
def test_summary_and_ensemble():
    """``OverfluxStatistics1D.summary``: the terms and the orders add up to the total; ``ensemble_1d`` of identical
    records has no scatter."""
    from skewerjax.summary_stats import ensemble_1d

    grf = PowerLawGRF(0.1, 1.0, -0.5, L=200.0, N=256)
    d = jax.vmap(grf.sample)(jax.random.split(jax.random.PRNGKey(2), 20))
    f = jnp.stack([1.0 + 0.3 * d, 0.2 * (1.0 + 0.3 * jnp.roll(d, 30, axis=-1))])
    assert np.all(np.asarray(f) > 0)
    a = DeltaFieldBuilder(-jnp.log(f), labels=["1", "2"], mean="global").build()
    np.testing.assert_allclose(  # every line against its mean over every pixel of every sample (mean="global")
        np.asarray(a.delta_terms["12"]),
        np.asarray(f[0] / f[0].mean() - 1) * np.asarray(f[1] / f[1].mean() - 1),
        rtol=1e-10,
        atol=1e-14,
    )
    s = OverfluxStatistics1D(a, 200.0 / 256).summary()
    for kind in ("P", "xi"):
        sc = np.max(np.abs(s[f"{kind}_total"]))
        assert np.max(np.abs(s[f"{kind}_order"].sum(0) - s[f"{kind}_total"])) < 1e-10 * sc
        assert np.max(np.abs(s[f"{kind}_terms"].sum(0) - s[f"{kind}_total"])) < 1e-10 * sc
    e = ensemble_1d([s, s])
    assert e["n_real"] == 2 and np.allclose(e["P_total"], s["P_total"]) and np.all(e["xi_order_err"] == 0)


# the axes of the set of pixels of each mean mode, on (samples, pixels)
MEAN_AXES = {"redshift": 0, "global": (0, 1), "box": (0, 1), "skewer": 1}


@pytest.mark.parametrize("mean", list(MEAN_AXES))
def test_ensemble_normalisation(mean):
    """``DeltaFieldBuilder`` in every ``MEAN_MODES`` mode, on synthetic transmissions with a mean that grows along v:
    delta_j = T_j / T_bar_j - 1 with T_bar over the set of the mode (the samples at each pixel, "redshift"; every pixel
    of every sample, "global" = "box" on one array; the pixels of each sample, "skewer"), from the samples or
    ``mean_transmission`` (an ensemble split over several arrays), so <delta> = 0 over that set; "redshift" equals the
    per-pixel ensemble mean on a shared grid, and mismatched grids raise; the total uses T_bar = <prod_j T_j> != prod_j
    T_bar_j; C = T_bar / prod_j T_bar_j - 1 = <delta_1 delta_2> over the same set; the decomposition delta_tot =
    [sum_A delta_A - C] / (1 + C) is exact at every pixel, and the order-1 terms of P1D and xi1D vanish."""
    from skewerjax.summary_stats import MEAN_MODES, check_shared_grid, mean_transmissions

    assert set(MEAN_MODES) == set(MEAN_AXES)
    rng = np.random.default_rng(5)
    N_s, N_v = 40, 64
    trend = np.linspace(0.0, 0.3, N_v)  # a mean optical depth that grows along v
    tau = np.stack([trend + 0.4 * rng.gamma(2.0, 0.5, (N_s, N_v)), 0.5 * trend + 0.2 * rng.gamma(2.0, 0.5, (N_s, N_v))])
    tau[1] = 0.7 * tau[0] + 0.3 * tau[1]  # correlated lines
    tau += rng.normal(0.0, 0.2, (1, N_s, 1)) ** 2  # a different mean per sample
    T = np.exp(-tau)
    axes = MEAN_AXES[mean]
    a_pix = 0.25 * np.exp(-np.arange(N_v) * 1e-3)  # the scale factor of the pixels, shared by every sample
    ov = DeltaFieldBuilder(tau, labels=["1", "2"], mean=mean, grid=np.tile(a_pix, (N_s, 1))).build()
    T_bar = np.asarray(mean_transmissions(jnp.asarray(T), mean))
    np.testing.assert_allclose(np.asarray(ov.mean_transmission), T_bar, rtol=1e-14)
    known = np.broadcast_to(np.mean(T[0] * T[1], axis=axes, keepdims=True), (N_s, N_v))  # the known mean of the total
    np.testing.assert_allclose(np.broadcast_to(T_bar[-1], (N_s, N_v)), known, rtol=1e-14)
    if mean == "redshift":  # the per-pixel ensemble mean
        np.testing.assert_allclose(T_bar[:2], T.mean(axis=1), rtol=1e-14)
    assert np.max(np.abs(T_bar[-1] / (T_bar[0] * T_bar[1]) - 1)) > 1e-3, "<T_1 T_2> != <T_1><T_2>"
    for d in (ov.delta_terms["1"], ov.delta_terms["2"], ov.total):
        assert np.max(np.abs(np.asarray(d).mean(axes))) < 1e-15, "<delta> = 0 over the set of the mode"
    shared = np.max(np.abs(np.asarray(ov.total).mean(0)))  # the pattern shared by the samples
    assert (shared < 1e-15) if mean == "redshift" else (shared > 1e-2), "only 'redshift' removes the drift along v"
    along = np.max(np.abs(np.asarray(ov.total).mean(1)))  # the mean along one sample
    assert (along < 1e-15) if mean == "skewer" else (along > 1e-2), "only 'skewer' removes the k_par = 0 mode"
    C = np.asarray(ov.C_per_pixel)
    assert C.shape == ((N_s, N_v) if mean == "skewer" else (1, N_v))
    d12 = np.asarray(ov.delta_terms["12"])
    np.testing.assert_allclose(
        np.broadcast_to(C, (N_s, N_v)), np.broadcast_to(d12.mean(axes, keepdims=True), (N_s, N_v)), rtol=1e-12, atol=1e-16
    )
    recon = (sum(np.asarray(v) for v in ov.delta_terms.values()) - C) / (1 + C)
    assert np.max(np.abs(recon - np.asarray(ov.total))) < 1e-15
    st = OverfluxStatistics1D(ov, 2.0).compute_xi1d()
    for tot, orders in ((st.Pk_total, st.Pk_total_by_order), (st.xi_total, st.xi_total_by_order)):
        sc = float(np.max(np.abs(np.asarray(tot))))
        assert np.max(np.abs(sum(np.asarray(v) for v in orders.values()) - np.asarray(tot))) < 1e-13 * sc
        assert np.max(np.abs(np.asarray(orders[1]))) < 1e-13 * sc, "order 1 vanishes"
    st3 = OverfluxStatistics3D(ov, 2.0, 1.0, grid_shape=(5, 8)).compute_p3d()  # the samples as a 5 x 8 lattice
    sc = float(np.max(np.asarray(st3.Pk3d_total)))
    assert np.max(np.abs(np.asarray(st3.Pk3d_total_from_terms) - np.asarray(st3.Pk3d_total))) < 1e-12 * sc
    assert np.max(np.abs(np.asarray(st3.Pk3d_total_by_order[1]))) < 1e-12 * sc, "order 1 vanishes in 3D"
    # an ensemble split in two: the same overflux with the mean of the whole ensemble
    halves = [
        DeltaFieldBuilder(
            tau[:, h], labels=["1", "2"], mean=mean, mean_transmission=T_bar[:, h] if mean == "skewer" else T_bar
        ).build()
        for h in (slice(0, 25), slice(25, None))
    ]
    np.testing.assert_allclose(
        np.concatenate([np.asarray(h.total) for h in halves]), np.asarray(ov.total), rtol=1e-13, atol=1e-15
    )
    with pytest.raises(ValueError, match="mean_transmission"):
        DeltaFieldBuilder(tau, mean_transmission=T_bar[:2])
    with pytest.raises(ValueError, match="mean must be"):
        DeltaFieldBuilder(tau, mean="per_pixel")  # the old name, removed
    bad = np.tile(a_pix, (N_s, 1))
    bad[3] *= 1.0 + 1e-6  # one sample on another redshift grid
    with pytest.raises(ValueError, match="shared"):
        DeltaFieldBuilder(tau, mean="redshift", grid=bad)
    with pytest.raises(ValueError, match="shared"):
        mean_transmissions(jnp.asarray(T), "redshift", grid=bad)
    np.testing.assert_array_equal(check_shared_grid(bad[:3] * (1 + 1e-14)), bad[0] * (1 + 1e-14))  # rounding passes


def test_grid_statistics_on_an_anisotropic_lattice():
    """``GridStatistics3D`` on a lattice of skewers (a coarse transverse spacing, a fine one along the line of sight,
    sides of different lengths): xi in the isotropic, (r_par, r_perp) and mu bins equals a direct pair sum over the
    points' coordinates with minimum-image separations, without the k = 0 mode (the field minus its mean); P and xi are
    one Fourier pair (Parseval: xi(0) = sum_{k != 0} P / V)."""
    from skewerjax.summary_stats import GridStatistics3D
    from skewerjax.summary_stats.binning import bin_index

    shape, L = (4, 12, 3), np.array([8.0, 9.0, 7.5])
    rng = np.random.default_rng(11)
    d = 0.3 + rng.normal(size=shape)  # a non-zero mean: left out like the k = 0 mode
    edges = np.array([0.0, 1.0, 2.1, 3.3, 4.6])
    gs = GridStatistics3D(
        shape, L, los_axis=1, r_edges=edges, r_mu_edges=edges, rpar_edges=edges, rperp_edges=np.array([0.0, 1.5, 3.0, 4.6])
    )
    m = gs.measure(d)
    dx = L / np.asarray(shape)
    pos = np.stack(np.meshgrid(*[np.arange(n) * h for n, h in zip(shape, dx)], indexing="ij"), -1).reshape(-1, 3)
    val = (d - d.mean()).ravel()
    sep = pos[None, :, :] - pos[:, None, :]
    sep = (sep + L / 2) % L - L / 2  # minimum image (a separation of L / 2 is its own image)
    prod = (val[:, None] * val[None, :]).ravel()
    r = np.sqrt((sep**2).sum(-1)).ravel()
    rpar, rperp = np.abs(sep[..., 1]).ravel(), np.hypot(sep[..., 0], sep[..., 2]).ravel()
    mu = np.where(r > 0, rpar / np.where(r > 0, r, 1.0), 0.0)

    def pair_mean(*idx_shape):
        idx = np.ravel_multi_index([np.where(i >= 0, i, 0) for i, _ in idx_shape], [n for _, n in idx_shape])
        ok = np.all([i >= 0 for i, _ in idx_shape], axis=0)
        n = int(np.prod([n for _, n in idx_shape]))
        c = np.bincount(idx[ok], minlength=n)
        return (np.bincount(idx[ok], prod[ok], minlength=n) / np.where(c > 0, c, np.nan)).reshape([n for _, n in idx_shape])

    nb = edges.size - 1
    np.testing.assert_allclose(m["xi_iso"], pair_mean((bin_index(r, edges), nb)), rtol=1e-12)
    np.testing.assert_allclose(
        m["xi_map"], pair_mean((bin_index(rpar, edges), nb), (bin_index(rperp, gs.rperp_edges), 3)), rtol=1e-12
    )
    imu = np.where(r > 0, bin_index(mu, gs.mu_edges), -1)
    np.testing.assert_allclose(
        m["xi_mu"], pair_mean((imu, gs.mu_edges.size - 1), (bin_index(r, edges), nb)), rtol=1e-12, equal_nan=True
    )
    f2 = np.abs(np.fft.fftn(d)) ** 2
    f2[0, 0, 0] = 0.0
    assert np.mean(val**2) == pytest.approx((gs.dV / gs.N * f2).sum() / gs.V, rel=1e-12), "xi(0) = sum_k P / V, k != 0"


def test_skewer_and_lattice_statistics():
    """``summary_stats.lattice``: P1D and xi1D by one 1D FFT per skewer, an exact Fourier pair (xi1D(0) the variance,
    Parseval); on the lattice, P1D(k_par) = (1 / (L_x L_z)) sum_{k_perp} P3D(k_perp, k_par) exactly (the discrete form of
    int d^2k_perp / (2 pi)^2 P3D), and ``measure_lattice`` = ``GridStatistics3D.measure`` of the x-major skewers."""
    from skewerjax.summary_stats import lattice_grid, measure_lattice, on_lattice, skewer_p1d, skewer_xi1d
    from skewerjax.summary_stats.grid3d import _rfft_weights

    rng = np.random.default_rng(3)
    n_x, n_z, n, dv = 4, 6, 32, 2.5
    d = rng.normal(size=(n_x * n_z, n))
    k, P = skewer_p1d(d, dv)
    lags, xi = skewer_xi1d(d, dv)
    np.testing.assert_allclose(k, 2 * np.pi * np.fft.rfftfreq(n, dv)) and np.testing.assert_allclose(
        lags, dv * np.arange(n // 2 + 1)
    )
    np.testing.assert_allclose(xi[0], np.mean(d**2), rtol=1e-12)  # the variance
    np.testing.assert_allclose(xi[3], np.mean(d * np.roll(d, -3, axis=1)), rtol=1e-12)  # one lag, every skewer
    np.testing.assert_allclose(np.sum(_rfft_weights(n) * P) / (n * dv), np.mean(d**2), rtol=1e-12)  # Parseval
    L, d_perp = n * dv, (1.5, 2.0)  # comoving sides: the line of sight L, across n d_perp
    g = on_lattice(d, n_x, n_z)
    assert g.shape == (n_x, n, n_z) and np.array_equal(g[1, :, 2], d[1 * n_z + 2])
    F = np.fft.fftn(g)  # every mode: the lattice spectrum P3D = dV / N |F|^2 of this realisation
    dV = d_perp[0] * dv * d_perp[1]
    P3 = dV / g.size * np.abs(F) ** 2
    P1 = P3.sum(axis=(0, 2))[: n // 2 + 1] / (n_x * d_perp[0] * n_z * d_perp[1])
    np.testing.assert_allclose(P1, P, rtol=1e-12)
    gs = lattice_grid(n_x, n_z, n, L, d_perp, k_edges=np.linspace(0.1, 0.9, 5))
    assert gs.shape == (n_x, n, n_z) and np.allclose(gs.L, (n_x * 1.5, L, n_z * 2.0))
    np.testing.assert_array_equal(measure_lattice(d, gs)["P_iso"], gs.measure(g)["P_iso"])


def test_bin_centres():
    """``bin_centres``: the arithmetic midpoint ("linear"), the geometric mean ("log", the midpoint where the lower edge
    is 0), or the weighted centres of the measurement ("weighted", returned as given); a bad mode raises."""
    from skewerjax.summary_stats import X_BIN_MODES, bin_centres

    e = np.array([0.0, 1.0, 4.0, 16.0])
    np.testing.assert_allclose(bin_centres(e, "linear"), [0.5, 2.5, 10.0])
    np.testing.assert_allclose(bin_centres(e, "log"), [0.5, 2.0, 8.0])  # sqrt(1 x 4), sqrt(4 x 16); edge 0: linear
    w = np.array([0.7, 2.2, 9.1])
    np.testing.assert_array_equal(bin_centres(e, "weighted", w), w)
    assert X_BIN_MODES == ("weighted", "linear", "log")
    with pytest.raises(ValueError, match="x_bin"):
        bin_centres(e, "middle")
    with pytest.raises(ValueError, match="weighted"):
        bin_centres(e, "weighted")


def test_redshift_mean_groups_the_pixels_by_scale_factor():
    """``mean="redshift"`` is the mean over the pixels at the same scale factor (``redshift_groups``): on a grid of
    distinct scale factors the mean at each pixel, exactly as without a grid; on one snapshot (every pixel at the same a)
    the global mean; on a grid with repeated values the mean over each group. ``overflux_check`` with the same grid finds
    <delta> = 0 over each group."""
    from skewerjax.summary_stats import ensemble_mean_transmissions, mean_transmissions, overflux_check, redshift_groups

    rng = np.random.default_rng(11)
    tau = rng.gamma(2.0, 0.3, (3, 2, 20, 16))  # (n_real, N_lines, N_samples, N_pix)
    T = np.exp(-tau)
    distinct, snapshot = 0.25 * np.exp(-np.arange(16) * 1e-3), np.full(16, 0.2495)
    mixed = np.repeat([0.25, 0.24, 0.23, 0.22], 4)
    assert redshift_groups(distinct) is None and np.all(redshift_groups(snapshot) == 0)
    assert np.array_equal(redshift_groups(mixed[::-1]), np.repeat([3, 2, 1, 0], 4)[::-1])
    for grid in (None, distinct):  # bit for bit as before
        assert np.array_equal(ensemble_mean_transmissions(T, "redshift", grid=grid), ensemble_mean_transmissions(T, "redshift"))
        assert np.array_equal(np.asarray(mean_transmissions(T[0], "redshift", grid=grid)), np.asarray(mean_transmissions(T[0])))
    glob = ensemble_mean_transmissions(T, "global")
    np.testing.assert_allclose(ensemble_mean_transmissions(T, "redshift", grid=snapshot), glob, rtol=1e-14)
    np.testing.assert_allclose(
        np.asarray(mean_transmissions(T[0], "redshift", grid=snapshot)),
        np.asarray(mean_transmissions(T[0], "global")),
        rtol=1e-14,
    )
    Tb = ensemble_mean_transmissions(T, "redshift", grid=mixed)
    full = np.concatenate([T, T.prod(1, keepdims=True)], 1)
    for g in range(4):
        sl = slice(4 * g, 4 * g + 4)
        np.testing.assert_allclose(Tb[0, :, 0, sl], full[..., sl].mean(axis=(0, 2, 3))[:, None].repeat(4, 1), rtol=1e-14)
    delta = full / Tb - 1
    chk = overflux_check(delta, Tb, "redshift", grid=mixed)
    assert chk["mean_delta_level"].max() < 1e-15 and chk["C_error"] < 1e-15
    assert overflux_check(delta, Tb, "redshift")["mean_delta_level"].max() > 1e-3, "per pixel: not the set of this grid"
    ov = DeltaFieldBuilder(tau[0], mean="redshift", grid=snapshot).build()
    np.testing.assert_allclose(np.asarray(ov.mean_transmission), np.asarray(mean_transmissions(T[0], "global")), rtol=1e-14)


@pytest.mark.parametrize("n_lines", [3, 4])
@pytest.mark.parametrize("mean", ["redshift", "skewer"])
def test_decomposition_of_n_lines(n_lines, mean):
    """Any number of lines (``DeltaFieldBuilder`` with the default labels "1" .. "N"): every non-empty subset of lines is
    a term (2^N - 1), the correlator C = T_bar / prod_j T_bar_j - 1 equals sum_{|A| >= 2} <delta_A> over the set of the
    mode, the decomposition delta_tot = [sum_A delta_A - C] / (1 + C) is exact at every pixel, and P1D, xi1D (and P3D)
    of the total split into the orders 0 .. 2N, which add up to the total to rounding, with order 1 zero; the same
    through ``decompose_1d`` and ``overflux_check``."""
    from itertools import combinations

    from skewerjax.summary_stats import decompose_1d, overflux_check

    rng = np.random.default_rng(10 + n_lines)
    N_s, N_v = 30, 48
    base = 0.3 * rng.gamma(2.0, 0.5, (N_s, N_v))
    tau = np.stack([(0.4 + 0.2 * j) * base + 0.15 * rng.gamma(2.0, 0.5, (N_s, N_v)) for j in range(n_lines)])
    ov = DeltaFieldBuilder(tau, mean=mean).build()
    assert ov.labels == [str(j + 1) for j in range(n_lines)] and len(ov.subset_keys) == 2**n_lines - 1
    axes = MEAN_AXES[mean]
    corr = sum(
        np.asarray(ov.delta_terms["".join(A)]).mean(axes, keepdims=True)
        for r in range(2, n_lines + 1)
        for A in combinations(ov.labels, r)
    )
    C = np.asarray(ov.C_per_pixel)
    np.testing.assert_allclose(np.broadcast_to(C, (N_s, N_v)), np.broadcast_to(corr, (N_s, N_v)), rtol=1e-12, atol=1e-16)
    recon = (sum(np.asarray(v) for v in ov.delta_terms.values()) - C) / (1 + C)
    assert np.max(np.abs(recon - np.asarray(ov.total))) < 1e-14
    st = OverfluxStatistics1D(ov, 2.0).compute_xi1d()
    assert sorted(st.Pk_total_by_order) == list(range(2 * n_lines + 1))
    for tot, orders in ((st.Pk_total, st.Pk_total_by_order), (st.xi_total, st.xi_total_by_order)):
        sc = float(np.max(np.abs(np.asarray(tot))))
        assert np.max(np.abs(sum(np.asarray(v) for v in orders.values()) - np.asarray(tot))) < 1e-12 * sc
        assert np.max(np.abs(np.asarray(orders[1]))) < 1e-12 * sc, "order 1 vanishes"
        assert np.max(np.abs(np.asarray(orders[2 * n_lines]))) > 0, "the highest order is there"
    st3 = OverfluxStatistics3D(ov, 2.0, 1.0, grid_shape=(5, 6)).compute_p3d()
    sc = float(np.max(np.asarray(st3.Pk3d_total)))
    assert np.max(np.abs(np.asarray(st3.Pk3d_total_from_terms) - np.asarray(st3.Pk3d_total))) < 1e-12 * sc
    s = decompose_1d(tau, 2.0, mean=mean)
    assert s["P_order"].shape[0] == 2 * n_lines + 1
    np.testing.assert_allclose(s["P_total"], np.asarray(st.Pk_total), rtol=1e-12)
    T = np.exp(-tau)
    Tb = np.asarray(ov.mean_transmission)
    Tb = Tb if Tb.ndim == 3 else np.broadcast_to(Tb[:, None], (n_lines + 1, N_s, N_v))
    delta = np.concatenate([T, T.prod(0)[None]]) / Tb - 1
    chk = overflux_check(delta[None], Tb[None], mean)
    assert chk["C_error"] < 1e-14 and chk["decomposition_error"] < 1e-14 and np.max(chk["mean_delta_level"]) < 1e-14


def test_periodic_ranges_of_the_statistics():
    """The periodicity audit of the statistics: the 3D maps and wedges stop at L / 2 (minimum-image separations of
    ``GridStatistics3D``, the rows of ``lattice_grid`` up to half the period, bins beyond cut by ``full_coverage``); the
    1D lags of ``skewer_xi1d`` and ``OverfluxStatistics1D`` run up to v_per / 2 (the separation of the nearest images);
    the mesh wraps every deposit (``tests/test_mesh.py::test_mass_conservation_and_periodic_wrap``)."""
    from skewerjax.summary_stats import GridStatistics3D, lattice_grid, skewer_p1d, skewer_xi1d

    L, n = 12.0, 8
    gs = GridStatistics3D((n,) * 3, L)
    assert gs.r_edges[-1] <= L / 2 + 1e-12 and gs.r_mu_edges[-1] <= L / 2 + 1e-12
    assert np.nanmax(gs.rpar) <= L / 2 + 1e-12 and np.nanmax(gs.rpar_map) <= L / 2 + 1e-12, (
        "the rows of the map: |r_par| <= L / 2"
    )
    lat = lattice_grid(4, 4, 16, L, 3.0, k_edges=np.linspace(0.5, 1.0, 3), log=None)
    assert np.isclose(lat.rpar[-1], L / 2) and lat.rpar.size == 16 // 2 + 1, "one row per separation j dv, j = 0 .. N / 2"
    assert lat.r_edges[-1] <= L / 2 + 1e-12
    rng = np.random.default_rng(0)
    d, dv = rng.normal(size=(5, 64)), 2.5
    lags, _ = skewer_xi1d(d, dv)
    k, _ = skewer_p1d(d, dv)
    assert lags[-1] == pytest.approx(64 * dv / 2) and k[-1] == pytest.approx(np.pi / dv), "lags to v_per / 2, k to pi / dv"
    ov = DeltaFieldBuilder(rng.uniform(0.1, 1.0, (1, 5, 64))).build()
    st = OverfluxStatistics1D(ov, dv).compute_xi1d()
    assert np.max(np.abs(np.asarray(st.lags))) == pytest.approx(64 * dv / 2), "centred lags within +- v_per / 2"
