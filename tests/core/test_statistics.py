"""Summary statistics: P1D / xi1D / P3D recovery of Gaussian random fields, the exact decomposition by order and line
pair, the shift signature of a second line, the linear-flux McDonald form and the McDonald / Ma contamination fits
(review notes, P1D, xi1D and Sec. Overflux; formerly ``scripts/validation/check_summary_statistics.py``, same
tolerances; 100 skewers and a 16^3 grid by default, 500 and 32^3 with the ``slow`` marker)."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from skewerjax.core import DeltaFieldBuilder
from skewerjax.core.stats import (
    OverfluxStatistics1D,
    OverfluxStatistics3D,
    fit_ma_model_profiled,
    fit_mcdonald_a,
    mcdonald_factor,
    xi_from_power,
)
from skewerjax.simulations import PowerLawGRF
from skewerjax.utils.plotting import hybrid_k_edges, pk_sigma


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
    single = lambda d: SimpleNamespace(subset_keys=["a"], delta_terms={"a": d}, C_per_sample=jnp.zeros((d.shape[0], 1)), total=d)
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


def test_grid_statistics_conventions():
    """``GridStatistics3D``: xi is the brute-force periodic correlation (the FFT pair and the minimum-image binning), the
    rfft half-space weights give the mean over every full-grid mode of a bin, and the expectation of xi at r = 0 is the
    variance of the field."""
    from skewerjax.core.stats import GridStatistics3D

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
    from skewerjax.core.stats.two_point import bin_index

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
    """An isotropic GRF (power law + BAO, as the toy boxes of notebook 08): the mean over many realisations of P and xi,
    isotropic, in the (par, perp) plane and in wedges of mu, agrees with ``theory`` within the error of the mean."""
    from skewerjax.core.stats import GRID_KEYS, GridStatistics3D, ensemble_mean

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
    from skewerjax.core.stats import GridStatistics3D
    from skewerjax.core.stats.two_point import bin_index

    n, L = (6, 8, 5), (6.0, 8.0, 7.0)
    g = PowerLawGRF(1.0, 1.0, -1.0, L=L, N=n, A_bao=0.05, r_bao=2.0, sigma_bao=0.6, los_axis=1)
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
    from skewerjax.core.stats import GridStatistics3D

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


def test_summary_and_ensemble():
    """``OverfluxStatistics1D.summary``: the terms and the orders add up to the total; ``ensemble_1d`` of identical
    records has no scatter."""
    from skewerjax.core.stats import ensemble_1d

    grf = PowerLawGRF(0.1, 1.0, -0.5, L=200.0, N=256)
    d = jax.vmap(grf.sample)(jax.random.split(jax.random.PRNGKey(2), 20))
    f = jnp.stack([1.0 + 0.3 * d, 0.2 * (1.0 + 0.3 * jnp.roll(d, 30, axis=-1))])
    assert np.all(np.asarray(f) > 0)
    a = DeltaFieldBuilder(-jnp.log(f), labels=["1", "2"]).build()
    np.testing.assert_allclose(
        np.asarray(a.delta_terms["12"]),
        np.asarray(f[0] / f[0].mean(1, keepdims=True) - 1) * np.asarray(f[1] / f[1].mean(1, keepdims=True) - 1),
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
