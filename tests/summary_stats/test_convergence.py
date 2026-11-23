"""The statistics of lattices of skewers (``summary_stats.convergence``): ``flux_statistics`` is notebook 11's chain
call by call; ``lattice_convergence`` compares every lattice with the largest (ratio 1 for the largest itself, the bins of
the coarser lattices cut at their Nyquist frequency, xi masked near its zeros, the P1D summary within ``p1d_k_max``);
the runs of a resolution sweep (notebooks 18-21): ``run_statistics`` and its jackknife, ``compare_runs``, ``k_max_table``,
``rebin_pixels``, ``block_labels``, ``LatticeWedges`` (``lattice_grid``'s wedges); and the figures respect the style. Synthetic optical depths on an 8 x 8 lattice:
no data."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.collections
import matplotlib.pyplot as plt
import numpy as np

from skewerjax.core import lattice_columns
from skewerjax.summary_stats import (
    GRID_KEYS,
    LatticeWedges,
    block_labels,
    compare_runs,
    ensemble_mean,
    ensemble_mean_transmissions,
    flux_statistics,
    format_k_max_table,
    jackknife_error,
    k_max_converged,
    k_max_table,
    lattice_convergence,
    lattice_grid,
    lattice_power,
    measure_lattice,
    rebin_pixels,
    run_statistics,
    set_k_edges,
    skewer_p1d,
    skewer_xi1d,
)
from skewerjax.utils import plotting as pl

N_MAX, N_PIX, L, DV = 8, 64, 40.0, 5.0
LINE_IDS = ("HI-LyA", "SiIII-1206")


def _tau(seed=3):
    """(3 axes, 2 lines, N_MAX^2, N_PIX) positive optical depths, smooth along the line of sight."""
    rng = np.random.default_rng(seed)
    g = rng.normal(size=(3, 2, N_MAX**2, N_PIX))
    g = np.fft.irfft(np.fft.rfft(g, axis=-1) * np.exp(-0.5 * (np.arange(N_PIX // 2 + 1) / 6.0) ** 2), n=N_PIX, axis=-1)
    g /= g.std()
    return np.exp(g) * np.array([0.5, 0.01])[None, :, None, None]


def _grid(n, edges):
    return lattice_grid(n, n, N_PIX, L, L / n, **edges, xi_mu_edges=(0.0, 0.5, 1.0), pk_mu_edges=(0.0, 0.5, 1.0))


def _edges():
    k_nyq = min(np.pi * N_MAX / L, np.pi * N_PIX / L)
    d = L / N_MAX
    g = lattice_grid(
        N_MAX,
        N_MAX,
        N_PIX,
        L,
        d,
        k_edges=set_k_edges(2 * np.pi / L, k_1=3.5 * 2 * np.pi / L, k_2=k_nyq, k_max=k_nyq, dk_factor=0.4),
        r_edges=np.arange(0, L / 2 + d, d),
        rperp_edges=np.arange(0, L / 2 + d, d),
    )
    return dict(k_edges=g.k_edges, r_edges=g.r_edges, rperp_edges=g.rperp_edges)


def _subset(tau, n):
    idx = lattice_columns(n, (1, 1)) * (N_MAX // n)  # the n x n lattice nested in the 8 x 8 one, (i_a, i_b) a-major
    rows = idx[:, 0] * N_MAX + idx[:, 1]  # -> the row i_a 8 + i_b of the 8 x 8 lattice
    return tau[:, :, rows]


def test_flux_statistics_is_notebook_11s_chain():
    tau = _tau()
    edges = _edges()
    grid = _grid(N_MAX, edges)
    a_pix = np.full(N_PIX, 0.25)
    s = flux_statistics(tau, DV, grid, line_ids=LINE_IDS, a_pix=a_pix, keep_delta=True)
    T = np.exp(-tau)  # notebook 11, Part B and D1, call by call
    T_bar = ensemble_mean_transmissions(T, mean="redshift", grid=a_pix)
    delta = np.concatenate([T, T.prod(axis=1, keepdims=True)], axis=1) / T_bar - 1
    assert np.array_equal(s["delta"], delta) and s["fields"] == (*LINE_IDS, "total")
    p = [dict(p1d=skewer_p1d(delta[i], DV)[1]) for i in range(3)]
    x = [dict(xi1d=skewer_xi1d(delta[i], DV)[1]) for i in range(3)]
    f3 = [{f: measure_lattice(delta[i, j], grid) for j, f in enumerate(s["fields"])} for i in range(3)]
    assert np.array_equal(s["p1d"]["p1d"], ensemble_mean(p, ("p1d",))["p1d"])
    assert np.array_equal(s["xi1d"]["xi1d"], ensemble_mean(x, ("xi1d",))["xi1d"])
    for f in s["fields"]:
        ref = ensemble_mean([r[f] for r in f3], GRID_KEYS)
        for k in GRID_KEYS:
            assert np.array_equal(s["flux3d"][f][k], ref[k], equal_nan=True), (f, k)
    assert s["check"]["mean_delta_level"].max() < 1e-12


def _conv(**kw):
    tau, edges = _tau(), _edges()
    stats = {n: flux_statistics(_subset(tau, n), DV, _grid(n, edges), line_ids=LINE_IDS) for n in (2, 4, 8)}
    k_f = 2 * np.pi / (N_PIX * DV)
    k1 = set_k_edges(k_f, k_1=3.5 * k_f, k_2=0.1, k_max=np.pi / DV, dk_factor=0.5, n_log=6)
    return stats, lattice_convergence(stats, k_edges_1d=k1, **kw)


def test_the_convergence_against_the_largest_lattice():
    stats, c = _conv()
    assert c["n"] == [2, 4, 8] and c["n_max"] == 8
    assert np.allclose(c["p1d"]["ratio"][8], 1.0)  # the largest against itself
    for name, d in c["summary"].items():
        assert d["deviation"][-1] == 0.0 or np.isnan(d["deviation"][-1]), name
        assert np.isfinite(d["ref_err"]) and d["ref_err"] > 0, name
    P = c["P_mu"]["total"]
    nk = P["ref"].shape[-1]
    n2 = np.asarray(stats[2]["flux3d"]["total"]["P_mu"]).shape[-1]
    assert n2 < nk and np.all(np.isnan(P["ratio"][2][:, n2:]))  # beyond the Nyquist frequency of N = 2: no bin
    X = c["xi_mu"]["total"]
    assert np.all(np.isnan(np.where(X["mask"], np.nan, X["ratio"][4])))  # nothing outside the mask
    assert np.array_equal(X["mask"], np.isfinite(X["ref"]) & (np.abs(X["ref"]) > 2 * X["ref_err"]))
    # the P1D summary within p1d_k_max: the deviation over the bins below it
    k_max = float(c["p1d"]["k"][3])
    _, ck = _conv(p1d_k_max=k_max)
    keep = c["p1d"]["k"] <= k_max
    assert ck["p1d_k_max"] == k_max
    assert ck["summary"]["P1D total"]["deviation"][0] == np.max(np.abs(c["p1d"]["ratio"][2][-1][keep] - 1))


def test_the_convergence_figures():
    pl.use_style()
    _, c = _conv(p1d_k_max=0.05)
    fig, axs = pl.plot_p1d_convergence(c, title=r"$P_{\rm 1D}$")
    assert len(axs) == 3 and axs[0].get_yscale() == "log" and pl.check_fonts(fig) == []
    plt.close(fig)
    fig, axs = pl.plot_wedge_convergence(c, "P_mu", mu_edges=(0.0, 0.5, 1.0), ncols=2)
    assert len(axs) == 2 and axs[0].get_yscale() == "log" and pl.check_fonts(fig) == []
    plt.close(fig)
    fig, axs = pl.plot_wedge_convergence(c, "xi_mu", mu_edges=(0.0, 0.5, 1.0), ncols=2)
    assert axs[0].get_yscale() == "linear" and pl.check_fonts(fig) == []
    plt.close(fig)
    fig, ax = pl.plot_convergence_summary(c, title="summary")
    assert len(ax.lines) >= len(c["summary"]) and pl.check_fonts(fig) == []
    plt.close(fig)
    cols = pl.lattice_colors([2, 4, 8])
    assert len(set(cols.values())) == 3
    v = (np.arange(N_PIX) + 0.5) * DV
    tau = _tau()[0, :, 0]
    fig, axs, _ = pl.plot_toy_skewer(L, None, v, tau, mean_transmission=np.full((3, N_PIX), 0.6), line_ids=LINE_IDS)
    assert len(axs) == 3 and pl.check_fonts(fig) == []  # the flux rows only (no gas)
    plt.close(fig)


def test_the_lattice_figures_with_residuals():
    pl.use_style()
    _, c = _conv(p1d_k_max=0.05)
    from skewerjax.core import lattice_columns
    from skewerjax.summary_stats import decompose_1d, ensemble_1d

    tau = _tau()
    cols = lattice_columns(N_MAX, (1, 1))
    ords = {}
    for n in (2, 4, N_MAX):
        sub = np.flatnonzero((cols[:, 0] % (N_MAX // n) == 0) & (cols[:, 1] % (N_MAX // n) == 0))
        ords[n] = ensemble_1d([decompose_1d(tau[i][:, sub], DV) for i in range(tau.shape[0])])
    k_f = 2 * np.pi / (N_PIX * DV)
    edges = set_k_edges(k_f, k_1=3.5 * k_f, k_2=0.05, k_max=np.pi / DV, dk_factor=0.25, n_log=6)
    k_lim = (2 * np.pi / (N_PIX * DV), np.pi / DV)
    for kw in (dict(), dict(quantity="kP_pi", k_scale="linear", y_scale="linear", residual="max", total=False, residual_scale="linear")):
        fig, axs, res, kc = pl.plot_p1d_lattices(ords, k_edges=edges, k_limits=k_lim, title="P1D", **kw)
        panels = [2, 3, 4] + (["total"] if kw.get("total", True) else [])  # 2 lines: orders 2-4 (0 and 1 vanish), the total
        assert axs[0, 1].get_yscale() == kw.get("residual_scale", "symlog")
        assert axs.shape == (len(panels), 2) and list(res) == panels and set(res[2]) == {2, 4}
        assert axs[0, 0].get_yscale() == ("symlog" if not kw else "linear")
        assert not any(isinstance(c_, matplotlib.collections.PolyCollection) for c_ in axs[0, 1].collections)  # no band
        assert pl.check_fonts(fig) == []
        plt.close(fig)
    # the residual of order 2, by hand: the binned means of the lattices against the reference
    _, R, _ = pl._binned(np.asarray(ords[N_MAX]["k"]), ords[N_MAX]["P_order_each"], edges)
    _, X, _ = pl._binned(np.asarray(ords[2]["k"]), ords[2]["P_order_each"], edges)
    fig, axs, res, kc = pl.plot_p1d_lattices(ords, k_edges=edges)
    np.testing.assert_allclose(res[2][2], 100 * (X[2] - R[2]) / np.abs(R[2]))
    plt.close(fig)
    # the default: the native modes; the reference on the total panel only (one measurement: grey squares with errors)
    import pytest

    from skewerjax.utils.reference_data import P1D_DESI_DR1

    one = P1D_DESI_DR1[sorted(P1D_DESI_DR1)[0]]
    one = dict(one, kP_pi_err=0.1 * np.asarray(one["kP_pi"]))
    fig, axs, res, kc = pl.plot_p1d_lattices(ords, k_limits=k_lim, reference=one)
    np.testing.assert_allclose(kc, 2 * np.pi * np.arange(1, N_PIX // 2 + 1) / (N_PIX * DV))  # k > 0 up to pi / dv
    drawn = [[c_ for c_ in a.containers if c_.get_label() == one["label"]] for a in axs[:, 0]]
    assert [len(d) for d in drawn] == [0, 0, 0, 1], "on the total panel only"
    k_ref, y_ref = np.asarray(one["k"]), np.asarray(one["kP_pi"])
    np.testing.assert_allclose(drawn[-1][0].lines[0].get_ydata(), np.pi * y_ref / k_ref)  # P = pi kP_pi / k
    assert one["label"] in [t.get_text() for t in fig.legends[0].get_texts()]
    plt.close(fig)
    fig, axs, _, _ = pl.plot_p1d_lattices(ords, quantity="kP_pi", reference=P1D_DESI_DR1, kms_per_cMpc=100.0)
    small = [ln for ln in axs[-1, 0].lines if ln.get_marker() == "s" and ln.get_linewidth() < 1.0]
    assert len(small) == len(P1D_DESI_DR1) and len({str(ln.get_color()) for ln in small}) == len(P1D_DESI_DR1), "a colour per z"
    np.testing.assert_allclose(small[0].get_ydata(), P1D_DESI_DR1[sorted(P1D_DESI_DR1)[0]]["kP_pi"])
    assert sum("DESI DR1" in a.get_ylabel() for a in fig.axes) == 1, "one colour key naming the reference"
    assert not any(ln.get_marker() == "s" and ln.get_linewidth() < 1.0 for a in axs[:-1, 0] for ln in a.lines)
    plt.close(fig)
    with pytest.raises(ValueError, match="total"):
        pl.plot_p1d_lattices(ords, total=False, reference=one)
    fig, axs, res = pl.plot_pk_mu_wedges_lattices(c, mu_edges=(0.0, 0.5, 1.0), ncols=2, bao_k=[0.3])
    assert axs.shape == (2, 2) and axs[0, 1].get_yscale() == "symlog" and set(res) == {0, 1}
    np.testing.assert_allclose(res[1][4], 100 * (c["P_mu"]["total"]["ratio"][4][1] - 1))
    assert pl.check_fonts(fig) == []
    plt.close(fig)


# --------------------------------------------------------------------------- the runs of a resolution sweep
def _run_setup(seed=3):
    tau = _tau(seed)[0]  # (2 lines, 64 skewers, N_PIX): one line of sight
    cols = lattice_columns(N_MAX, (1, 1))
    lab = block_labels(cols, N_MAX, 2)
    k = set_k_edges(2 * np.pi / L, k_1=3.5 * 2 * np.pi / L, k_2=np.pi * N_MAX / L, k_max=np.pi * N_MAX / L, dk_factor=0.4)
    grid = lattice_grid(N_MAX, N_MAX, N_PIX, L, L / N_MAX, k_edges=k, pk_mu_edges=(0, 0.5, 1), xi_mu_edges=(0, 0.5, 1))
    return tau, cols, lab, grid


def test_block_labels_and_the_jackknife():
    cols = lattice_columns(8, (1, 1))
    lab = block_labels(cols, 8, 2)
    assert np.bincount(lab).tolist() == [16] * 4  # 2 x 2 squares of 4 x 4 skewers
    assert set(lab[(cols[:, 0] < 4) & (cols[:, 1] < 4)]) == {0} and set(lab[(cols[:, 0] >= 4) & (cols[:, 1] < 4)]) == {2}
    lab4 = block_labels(lattice_columns(4, (1, 1)), 4, 2)  # the same regions of the box on a coarser lattice
    assert np.array_equal(lab4, lab[np.flatnonzero((cols[:, 0] % 2 == 0) & (cols[:, 1] % 2 == 0))])
    for bad in (1, 3):
        try:
            block_labels(cols, 8, bad)
            raise AssertionError
        except ValueError:
            pass
    x = np.random.default_rng(0).normal(size=(20, 3))
    jk = np.stack([np.delete(x, i, 0).mean(0) for i in range(20)])
    assert np.allclose(jackknife_error(jk), x.std(0, ddof=1) / np.sqrt(20))  # the jackknife of a mean: its standard error


def test_run_statistics_and_its_jackknife():
    tau, cols, lab, grid = _run_setup()
    st = run_statistics(tau, DV, line_ids=LINE_IDS, labels=lab, grid=grid, a_pix=np.full(N_PIX, 0.25))
    T = np.exp(-tau)
    T = np.concatenate([T, T.prod(0, keepdims=True)])
    assert np.allclose(st["mean_flux"], T.mean(axis=(1, 2)))
    delta = T / T.mean(axis=(1, 2), keepdims=True) - 1
    k, p = skewer_p1d(delta, DV)
    assert np.allclose(st["k"], k) and np.allclose(st["p1d"], p, rtol=1e-10)
    assert np.allclose(st["p1d_jk"][1], skewer_p1d(delta[:, lab != 1], DV)[1], rtol=1e-10)  # block 1 left out
    full = measure_lattice(delta[2], grid)["P_mu"]
    assert np.allclose(st["P_mu"]["total"], full, rtol=1e-4, equal_nan=True)  # lattice_power: measure_lattice without xi
    assert np.allclose(lattice_power(delta[2], grid)["P_mu"], full, rtol=1e-4, equal_nan=True)
    keep = lab != 3
    assert np.allclose(
        st["P_mu_jk"]["total"][3],
        measure_lattice(delta[2] * keep[:, None], grid)["P_mu"] / keep.mean(),
        rtol=1e-4,
        equal_nan=True,
    )
    same = run_statistics(transmissions=T, dv=DV, line_ids=LINE_IDS, labels=lab, grid=grid, a_pix=np.full(N_PIX, 0.25))
    assert np.allclose(same["p1d"], st["p1d"]) and np.allclose(same["mean_flux"], st["mean_flux"])


def test_the_wedges_alone_are_lattice_grids_wedges():
    """``LatticeWedges``: the bins, centres, mode counts and P_mu of ``lattice_grid`` (its other bins left out), and
    ``run_statistics`` gives the same wedges with either."""
    tau, cols, lab, grid = _run_setup()
    k = set_k_edges(2 * np.pi / L, k_1=3.5 * 2 * np.pi / L, k_2=np.pi * N_MAX / L, k_max=np.pi * N_MAX / L, dk_factor=0.4)
    w = LatticeWedges(N_MAX, N_PIX, L, k_edges=k, mu_edges=(0, 0.5, 1))
    ax = grid.axes()
    assert np.array_equal(w.pk_mu_edges, ax["pk_mu_edges"]) and np.array_equal(w.k_mu_edges, grid.k_mu_edges)
    assert np.allclose(w.n_modes, np.nan_to_num(np.asarray(grid.n_modes["P_mu"])), rtol=1e-6, atol=0)
    assert np.allclose(w.k_mu, ax["k_mu"], rtol=1e-6, equal_nan=True)
    T = np.exp(-tau).prod(0)
    delta = T / T.mean() - 1
    assert np.allclose(w.power(delta), measure_lattice(delta, grid)["P_mu"], rtol=1e-5, equal_nan=True)
    kw = dict(line_ids=LINE_IDS, labels=lab, a_pix=np.full(N_PIX, 0.25))
    a, b = run_statistics(tau, DV, grid=grid, **kw), run_statistics(tau, DV, grid=w, **kw)
    assert np.allclose(a["P_mu_jk"]["total"], b["P_mu_jk"]["total"], rtol=1e-5, equal_nan=True)
    assert np.allclose(a["k_mu"], b["k_mu"], rtol=1e-6, equal_nan=True)
    assert np.array_equal(a["pk_mu_edges"], b["pk_mu_edges"])


def test_compare_runs_and_k_max():
    tau, cols, lab, grid = _run_setup()
    kw = dict(line_ids=LINE_IDS, labels=lab, grid=grid, a_pix=np.full(N_PIX, 0.25))
    ref = run_statistics(tau, DV, **kw)
    same = compare_runs(ref, ref)
    assert np.all(same["mean_flux"]["rel"] == 0) and np.all(same["p1d"]["rel"] == 0) and np.all(same["mean_flux"]["err"] == 0)
    run = run_statistics(tau * 1.01, DV, **kw)  # a 1 % deeper absorption: a small, smooth, systematic change
    c = compare_runs(run, ref)
    assert np.all(c["mean_flux"]["rel"] < 0) and np.all(np.isfinite(c["p1d"]["rel"])) and c["p1d"]["k"][0] > 0
    assert c["P_mu"]["total"]["rel"].shape == ref["P_mu"]["total"].shape and np.isfinite(c["P_mu"]["total"]["err"]).any()
    cb = compare_runs(run, ref, k_edges_1d=np.geomspace(c["p1d"]["k"][0] * 0.99, c["p1d"]["k"][-1] * 1.01, 6))
    assert cb["p1d"]["rel"].shape[1] <= 5 and np.all(np.diff(cb["p1d"]["k"]) > 0)
    k = np.array([0.1, 0.2, 0.3, 0.4])
    assert k_max_converged(k, [0.001, 0.002, 0.02, 0.001], 0.005) == 0.2
    assert k_max_converged(k, [0.01, 0.0, 0.0, 0.0], 0.005) == 0.0
    assert np.isinf(k_max_converged(k, [0.0, 0.0, 0.0, 0.0], 0.005))
    assert k_max_converged(k, [0.0, np.nan, 0.0, 0.0], 0.005) == 0.1  # a bin the run does not cover ends the range
    assert k_max_converged([np.nan, 0.2, 0.3], [5.0, 0.0, 0.0], 0.005) == np.inf  # a bin without modes is not a bin
    rows = k_max_table({"r": c}, (0.001, 0.005, 0.01))
    assert {r["statistic"] for r in rows} >= {"P1D total", "P1D HI-LyA", "P(k, mu) total mu 0.00-0.50"}
    assert all(set(r["k_max"]) == {0.001, 0.005, 0.01} for r in rows)
    assert all(r["k_max"][0.001] <= r["k_max"][0.005] <= r["k_max"][0.01] for r in rows)
    assert "P1D total" in format_k_max_table(rows)
    try:
        compare_runs(run, run_statistics(tau, DV, **dict(kw, labels=block_labels(cols, N_MAX, 4))))
        raise AssertionError
    except ValueError:
        pass


def test_rebin_pixels():
    rng = np.random.default_rng(1)
    F = rng.uniform(size=(2, 5, 100))
    for n in (100, 50, 33, 7, 1):
        R = rebin_pixels(F, n)
        assert R.shape == (2, 5, n) and np.allclose(R.mean(-1), F.mean(-1))  # the mean over the period is kept
    assert np.allclose(rebin_pixels(F, 100), F) and np.allclose(rebin_pixels(F, 50), F.reshape(2, 5, 50, 2).mean(-1))
    edges_f, edges_c = np.linspace(0, 1, 101), np.linspace(0, 1, 34)
    w = np.clip(np.minimum(edges_f[None, 1:], edges_c[1:, None]) - np.maximum(edges_f[None, :-1], edges_c[:-1, None]), 0, None)
    assert np.allclose(rebin_pixels(F, 33), (F @ w.T) / w.sum(1))  # the overlaps of comparison.scaling.rebin_average


def test_the_sweep_figures():
    tau, cols, lab, grid = _run_setup()
    kw = dict(line_ids=LINE_IDS, labels=lab, grid=grid, a_pix=np.full(N_PIX, 0.25))
    ref = run_statistics(tau, DV, **kw)
    comps = {
        f"r{i}": compare_runs(run_statistics(tau * (1 + 0.01 * i), DV, **kw), ref, k_edges_1d=np.geomspace(0.01, 0.6, 8))
        for i in (1, 2)
    }
    styles = pl.run_styles(["r1", "r2", "ref"], fiducial="ref")
    assert styles["ref"]["color"] == pl.FG and styles["r1"]["color"] != styles["r2"]["color"]
    assert pl.run_styles(["a", "b"])["a"] == pl.run_styles(["a", "c"])["a"]  # a rank has one colour in every sweep
    for fig, _ in (
        pl.plot_runs_p1d(comps, styles, "ref", title="P1D"),
        pl.plot_runs_wedges(comps, styles, "ref", title="wedges"),
        pl.plot_runs_mean_flux(comps, styles, "ref", title="mean flux"),
        pl.plot_runs_p1d(comps, styles, "ref", ref_per_run=True),
    ):
        fig.canvas.draw()
        assert pl.check_fonts(fig) == []
        plt.close(fig)
    rows = k_max_table({r: compare_runs(run_statistics(tau * (1 + 0.01 * i), DV, **kw), ref) for i, r in ((1, "r1"), (2, "r2"))})
    fig, axs = pl.plot_k_max_summary(rows, {"r1": 1.0, "r2": 2.0}, xlabel="x", p1d_k_ref=1e-3)
    fig.canvas.draw()
    assert pl.check_fonts(fig) == []
    lo, hi = axs[0].get_ylim()
    assert lo < 1e-3 < hi  # the marked k inside the P1D panel's limits
    plt.close(fig)
