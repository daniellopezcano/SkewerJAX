"""The statistics of nested lattices of skewers (``summary_stats.convergence``, notebook 17): ``flux_statistics`` is
notebook 11's chain call by call; ``lattice_convergence`` compares every lattice with the largest (ratio 1 for the
largest itself, the bins of the coarser lattices cut at their Nyquist frequency, xi masked near its zeros, the P1D
summary within ``p1d_k_max``); and the figures of notebook 17 respect the style. Synthetic optical depths on an 8 x 8
lattice: no data."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.collections
import matplotlib.pyplot as plt
import numpy as np

from skewerjax.simulations import skewer_store as ss
from skewerjax.summary_stats import (
    GRID_KEYS,
    ensemble_mean,
    ensemble_mean_transmissions,
    flux_statistics,
    lattice_convergence,
    lattice_grid,
    measure_lattice,
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
    rows = ss._keys(ss.lattice_indices(n, N_MAX), N_MAX)  # (i_a, i_b) on the 8 lattice -> row i_a 8 + i_b
    return tau[:, :, rows]


def test_flux_statistics_is_notebook_11s_chain():
    tau = _tau()
    edges = _edges()
    grid = _grid(N_MAX, edges)
    a_pix = np.full(N_PIX, 0.25)
    s = flux_statistics(tau, DV, grid, line_ids=LINE_IDS, a_pix=a_pix, keep_delta=True)
    T = np.exp(-tau)  # notebook 11, Part C and D1, call by call
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
