"""Smoke tests of the plotting module: the style and every tutorial figure function draw without error (Agg)."""

import matplotlib

matplotlib.use("Agg")
import jax.numpy as jnp  # noqa: E402
import matplotlib.legend  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from skewerjax.core import Cosmology, LineBundle, SkewerBundle, phys  # noqa: E402
from skewerjax.simulations import PowerLawGRF  # noqa: E402
from skewerjax.utils import plotting as pl  # noqa: E402


def setup_module():
    pl.use_style()


def teardown_function():
    plt.close("all")


def test_style_and_dag():
    # default: the paper style (white, no glow); "dark": black with the glow; same palette, FG / BG follow the style
    assert pl.THEME == "paper" and plt.rcParams["axes.facecolor"] == pl.BG == "white" and plt.rcParams["path.effects"] == []
    fig, ax = pl.plot_dag(("tau",))
    fig.canvas.draw()
    palette = dict(pl.PALETTE)
    pl.use_style("dark")
    assert pl.FG == "white" and pl.BG == "black" and isinstance(plt.rcParams["path.effects"][0], pl.NeonGlow)
    assert pl.ENTITY_COLORS["total"] == "white" and pl.CMAPS["sequential"] == "tol_iridescent_r" and pl.PALETTE == palette
    fig, ax = pl.plot_dag(("simulation", "fake_spectra"), external=True)
    fig.canvas.draw()
    pl.use_style()
    assert pl.FG == "black" and pl.ENTITY_COLORS["total"] == "black" and pl.CMAPS["sequential"] == "tol_iridescent"
    assert pl.CMAPS["density"] == "tol_rainbow_WhBr" and pl.CMAPS["density"] in matplotlib.colormaps
    nrm = pl._density_norm(np.array([-13.0, -12.0, -11.0]), decades=1.5)  # one norm for 04, 05, 07: tight, 0.05 dex pad
    assert np.isclose(nrm.vmin, -12.55) and np.isclose(nrm.vmax, -10.95)
    # the routes of the DAG: tests/utils/test_dag.py


def test_line_table_and_skewer_figures():
    lines = LineBundle(["HI-LyA", "SiIII-1206", "CIV-1548"], ref_line_id="HI-LyA")
    pl.plot_line_table(lines)[0].canvas.draw()
    t = jnp.linspace(0.0, 1.0, 50)
    pts = jnp.stack([5 + 3 * jnp.cos(6 * t), 5 + 3 * jnp.sin(6 * t), 10 * t], -1)[None]
    cosmo = Cosmology(H0=phys.H0, Omega_r_0=0.0, Omega_m_0=1.0, Omega_k_0=0.0, Omega_lambda_0=0.0)
    sb = SkewerBundle(spatial_points=pts, a0=0.5, x0=0.0, cosmologies=cosmo)
    pl.plot_skewer_anatomy(sb, 0)[0].canvas.draw()


def test_cylinder():
    v = np.linspace(0, 100, 64)
    s = np.linspace(0, 10, 40)
    cloud = dict(axis=s, s_perp=0.1 * np.sin(s), t_perp=0.1 * np.cos(s), value=np.exp(s / 3))
    flux = dict(v=v, s_of_v=v / 10, edges=np.linspace(0, 10, 5), tau_bins={"Lya": (np.ones((4, 64)) * 0.1, "C0")})
    fig, (ax3, axf) = pl.plot_cylinder(s, {"n": (np.log10(1 + s), "C1")}, cloud=cloud, flux=flux, progress=0.5, radius=0.3)
    fig.canvas.draw()


def test_palette_fonts_and_backend():
    # every discrete colour is Tol's "bright" scheme (tol_colors colorsets["bright"]), in Tol's order
    assert pl.PALETTE == pl._TOL_BRIGHT_VENDORED and list(pl.PALETTE) == [
        "blue",
        "red",
        "green",
        "yellow",
        "cyan",
        "purple",
        "grey",
    ]
    allowed = set(pl.PALETTE.values()) | {pl.FG, pl.BG}
    assert set(pl.ENTITY_COLORS.values()) <= allowed
    assert all(c in pl.PALETTE.values() for c in pl.ORDER_COLORS)
    assert {"tol_iridescent", "tol_iridescent_r", "tol_iridescentline"} <= set(matplotlib.colormaps)
    assert plt.rcParams["image.cmap"] == pl.CMAPS["sequential"]
    for hl, ext in ((("tau",), False), (("simulation", "fake_spectra"), True)):
        fig, _ = pl.plot_dag(hl, external=ext)
        assert pl.check_fonts(fig) == []
    assert plt.rcParams["legend.fontsize"] >= pl.FONT_MIN["legend"] and plt.rcParams["xtick.labelsize"] >= pl.FONT_MIN["tick"]
    assert pl.use_backend() == matplotlib.get_backend()  # outside IPython: unchanged


def test_fifth_pass_figures():
    from skewerjax.core import LineBundle as LB

    t = np.linspace(0, 1, 12)
    pts = np.stack([np.cos(3 * t), np.sin(3 * t), t], -1)
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    assert pl.plot_skewer_geometry_3d(ax, pts, highlight=5) is None  # scale_factors optional
    pl.plot_skewer_geometry_3d(ax, pts, scale_factors=1 - 0.01 * t)
    fig.canvas.draw()
    lines = LB(["HI-LyA", "SiIII-1206"], ref_line_id="HI-LyA")
    v = np.linspace(0, 400, 200)
    F = np.exp(-np.exp(-(((v - 200) / 10) ** 2)))[None] * np.ones((2, 1))
    for fig, _ in pl.plot_frames_axes(v, F, lines, v_ref=v, F_ref=F, a_dagger=0.25).values():
        fig.canvas.draw()
    for theme in ("dark", "paper"):
        pl.use_style(theme)
        pl.plot_dag(("lines", "cosmology", "skewers"))[0].canvas.draw()


def test_toy_box_figures():
    """The figures of notebooks 08 and 09 on a small toy box: the fields (08 A2), the 3D statistics of the realisations
    against their expectation and the continuum (08 A3), one skewer (09 C1), the 3D statistics of the overfluxes (09 D:
    maps in native units, isotropic quantities in comoving units, the echo curves, the wedges with their characteristic
    scales), the 1D statistics of the overflux by order and term (09 E), and the density along the skewers through the
    same figures (09 B, in comoving units, with their exact expectation): no titles, the line of sight vertical in the
    maps; the options cmap, quantity, color_scale, levels and the axis scales."""
    from dataclasses import replace

    from skewerjax.core import LineBundle
    from skewerjax.simulations import ToyBoxes
    from skewerjax.skewers import lattice_columns
    from skewerjax.summary_stats import bao_k_edges, box_grid, field_statistics, grid_limits
    from skewerjax.utils.reference_data import P1D_DESI_DR1_Z22

    from helpers import toy_lattice_statistics

    b = ToyBoxes(box_cMpc=8.0, n_cells=16, n_real=3, bao_ratio=1e-3)
    fig, axs, info = pl.plot_toy_fields(replace(b, velocity="linear").fields(0), b.box_cMpc, arrow_every=4)
    fig.canvas.draw()
    assert axs.shape == (2, 2) and info["arrow_scale"] > 0, "density and velocity rows (no temperature)"
    fig, axs, info = pl.plot_toy_fields(
        b.fields(0), b.box_cMpc, arrow_every=4, cmap="tol_iridescent"
    )  # zero velocities: no arrows
    fig.canvas.draw()
    assert info["arrow_scale"] == 0 and not any(ax.get_title() for ax in axs.ravel())
    n_pix, p = 64, 80  # the skewers: stride (4, 4), 80 points per cell, 64 pixels
    g, lim = b.grf(), grid_limits(b.n_cells, b.box_cMpc)
    gs = box_grid(
        b.n_cells,
        b.box_cMpc,
        k_edges=bao_k_edges(lim["k_min"], lim["k_max"], 3.0 / b.sigma_bao),
        r_edges=np.arange(0, b.box_cMpc / 2 + b.dx, b.dx),
        log=None,
    )
    res = field_statistics(
        (b.delta(i) for i in range(b.n_real)), gs, power=g.power, power_nw=lambda k: g.power(k, bao=False), log=None
    )
    run = toy_lattice_statistics(b, range(b.n_real), (4, 4), p, n_pix, decompose=True)
    for quantity, k_scale in (("xi", "linear"), ("r2xi", "log")):
        fig, axs, out = pl.plot_isotropic_recovery(
            res, quantity=quantity, r_bao=2.0, grf=b.grf(), k_scale=k_scale, k_marks={"$k_1$": 1.0, "$k_2$": 50.0}
        )
        fig.canvas.draw()
        assert axs[0, 0].get_xscale() == k_scale and axs[0, 1].get_yscale() == ("symlog" if quantity == "xi" else "linear")
    assert {"P_chi2_dof", "xi_chi2_dof", "xi_mean_pull", "xi_curve_r"} <= set(out)
    labels = [t.get_text() for t in axs[0, 0].get_legend().get_texts()]
    assert labels == ["per realisation", "expectation", "analytic input", r"mean $\pm$ error"], labels
    # x_bin: only the drawn x of the bins moves (default: the weighted centres of the measurement)
    a_ = res["axes"]
    for mode, xk in (
        ("weighted", a_["k_iso"]),
        ("linear", pl.bin_centres(a_["k_edges"], "linear")),
        ("log", pl.bin_centres(a_["k_edges"], "log")),
    ):
        f2, ax2, _ = pl.plot_isotropic_recovery(res, x_bin=mode)
        th = [ln for ln in ax2[0, 0].lines if ln.get_label() == "expectation"][0]
        ok = np.isfinite(res["P_iso"]) & np.isfinite(res["theory"]["P_iso"])
        np.testing.assert_allclose(th.get_xdata(), xk[ok]) and np.testing.assert_allclose(
            th.get_ydata(), res["theory"]["P_iso"][ok]
        )
        plt.close(f2)
    assert axs[0, 0].get_ylabel().startswith("$P(k)$") and len(axs[0, 0].lines) > len(res["P_iso_each"]) + 1, (
        "P, with the continuum"
    )
    assert any(ln.get_color() == pl.PALETTE["purple"] for ln in axs[0, 0].lines), "the continuum in a visible colour"
    texts = [t.get_text() for t in axs[0, 0].texts]
    assert "$k_1$ " in texts and "$k_2$ " not in texts, "k_1 labelled; k_2 beyond the axis range not drawn"
    assert any("k_{\\min}" in t for t in texts) and any("k_{\\max}" in t for t in texts), "the limits inside the grey bands"
    assert not any("full-coverage" in t.get_text() for t in axs[0, 0].get_legend().get_texts()), "no legend entry for the bands"
    assert axs[0, 0].get_yscale() == "log" and any("r_{\\rm BAO}" in t.get_text() for t in axs[0, 1].texts)
    lim = res["limits"]
    for ax, (lo, hi) in ((axs[0, 0], (lim["k_min"], lim["k_max"])), (axs[0, 1], (lim["r_min"], lim["r_max"]))):
        spans = [p for p in ax.patches if isinstance(p, matplotlib.patches.Rectangle)]
        edges = sorted(x for p in spans for x in (p.get_x(), p.get_x() + p.get_width()))
        assert len(spans) == 2 and np.isclose(edges[1], lo) and np.isclose(edges[2], hi), "shaded outside the full coverage"
    np.testing.assert_allclose(out["xi_curve_r"], (lim["r_min"], lim["r_max"]))
    with pytest.raises(ValueError, match="other k_edges"):
        pl.plot_isotropic_recovery(res, k_edges=res["axes"]["k_edges"][:-1])
    with pytest.raises(ValueError, match="quantity"):
        pl.plot_isotropic_recovery(res, quantity="r2")
    opts = {
        pl.plot_kpar_kperp_maps: dict(n_sigma=2, cmap="tol_iridescent", levels=[1e-3, 1e-2], k_marks={"$k_1$": 1.0}),
        pl.plot_rpar_rperp_maps: dict(n_sigma=2, r_bao=2.0, quantity="xi", color_scale="symlog", cmap="tol_BuRd", levels=[0.01]),
        pl.plot_xi_mu_wedges: dict(quantity="xi", y_scale="symlog", r_marks={"$a$": 1.0, "$b$": 100.0}),
    }
    for f, kw in opts.items():
        out, axs, chi2 = f(res, **kw)
        figs = list(out) if f is not pl.plot_xi_mu_wedges else [out]
        for fig in figs:
            fig.canvas.draw()
            assert not fig._suptitle and not fig.legends, "no titles, no figure-level legend"
        assert not any(ax.get_title() for ax in axs), "no titles"
        if f is not pl.plot_xi_mu_wedges:  # two figures, expected then measured, in the single-panel format
            assert len(figs) == 2 and figs[0] is not figs[1] and len(axs) == 2
            assert tuple(figs[0].get_size_inches()) == tuple(figs[1].get_size_inches()) == (14, 12.5)
            assert "perp" in axs[0].get_xlabel() and "parallel" in axs[0].get_ylabel(), "perpendicular across, parallel up"
            n0, n1 = axs[0].collections[0].norm, axs[1].collections[0].norm
            assert (n0.vmin, n0.vmax) == (n1.vmin, n1.vmax), "one colour norm for the pair"
            texts = [[t.get_text() for t in ax.get_legend().get_texts()] for ax in axs]
            assert texts[0][0].startswith("expected iso") and not any("measured" in t for t in texts[0])
            assert texts[1][0].startswith("expected iso") and texts[1][1].startswith("measured iso") and "sigma" in texts[1][1]
            leg = axs[1].get_legend()
            assert leg.get_zorder() >= pl.LEGEND_ZORDER > max(c.get_zorder() for c in axs[1].get_children() if c is not leg)
            assert leg.get_frame().get_alpha() == 1.0, "the legend on top, opaque"
        else:
            legs = [c for c in axs[0].get_children() if isinstance(c, matplotlib.legend.Legend)]
            assert len(legs) == 2, "two legends inside the panel: the wedges and the styles"
            assert [t.get_text() for t in legs[1].get_texts()] == ["measured", "expected", "per realisation"]
            assert legs[0].get_title().get_text() == "", "no title on the wedge legend"
            assert [t.get_text() for t in axs[0].texts] == ["$a$ "], "the marks inside the range only, labelled"
        if f is pl.plot_kpar_kperp_maps:  # P on the grid's own k_par: one row per mode, centred on it
            assert "P(k_" in figs[0].axes[-1].get_ylabel() and "k\\," not in figs[0].axes[-1].get_ylabel()
            assert axs[0].get_ylim()[0] == pytest.approx(0.5 * lim["k_min"])
            empty_cols = np.any(np.all(~np.isfinite(res["P_map"]), axis=0))
            assert any("no mode" in t for t in texts[1]) == bool(empty_cols), "the empty-bin entry only when there is one"
            assert len(axs[0].collections[0].get_array().compressed()) > 0 and not axs[0].get_legend().get_texts()[
                0
            ].get_text().startswith("measured")
        if f is pl.plot_rpar_rperp_maps:  # the wedge boundaries of plot_xi_mu_wedges, one ray per inner edge
            rays = [ln for ln in axs[1].lines if ln.get_linestyle() == "-." and len(ln.get_xdata()) == 2]
            assert len(rays) == len(pl.MU_WEDGES) - 2 and all(ln.get_color() == "white" for ln in rays)
            assert len(axs[1].patches) == len(pl.MU_WEDGES) - 1, "one coloured sector per wedge"
            assert len(texts[1]) == 2 and any("r_{\\rm BAO}" in t.get_text() for t in axs[1].texts)
            assert isinstance(axs[1].collections[0].norm, matplotlib.colors.SymLogNorm) and "xi" in figs[1].axes[-1].get_ylabel()
        for fig in figs:
            plt.close(fig)
    # the empty-bin entry appears when a k_perp column holds no mode, and only then
    sparse = pl.plot_kpar_kperp_maps(res, kperp_edges=None, kperp_scale="linear", kpar_scale="linear")
    with_gap = dict(res, P_map=np.where(np.arange(res["P_map"].shape[1]) == 1, np.nan, res["P_map"]))
    gap = pl.plot_kpar_kperp_maps(with_gap, kperp_scale="linear", kpar_scale="linear")
    has = lambda o: any("no mode" in t.get_text() for t in o[1][1].get_legend().get_texts())
    assert has(gap) and has(sparse) == bool(np.any(np.all(~np.isfinite(res["P_map"]), axis=0)))
    assert len(chi2) == len(pl.MU_WEDGES) - 1
    assert not hasattr(pl, "plot_pk_mu_wedges")
    sk = dict(tau=run["tau"][0], v=run["v"], period=run["period"], columns=lattice_columns(b.n_cells, (4, 4)))
    ix, iz = sk["columns"][0]
    fl = b.fields(0)
    n = {sp: fl["n"][sp][ix, :, iz] for sp in ("HI", "SiIII")}
    fig, axs, info = pl.plot_toy_skewer(
        b.box_cMpc, n, sk["v"], sk["tau"][:, 0], mean_transmission=run["mean_transmission"], points_per_cell=p
    )
    fig.canvas.draw()
    assert len(axs) == 5 and axs[0].get_xlabel().startswith("position") and axs[2].get_xlabel() == pl.LABELS["v_ref"]
    assert axs[2].get_ylabel() == pl.LABELS["transmission"] and "bar T" in axs[3].get_ylabel() and "bar T" in axs[4].get_ylabel()
    T = np.exp(-sk["tau"][:, 0])
    np.testing.assert_allclose(axs[3].lines[0].get_ydata(), run["mean_transmission"][0], rtol=1e-12)  # (d) T_bar, Lya
    np.testing.assert_allclose(axs[3].lines[2].get_ydata(), run["mean_transmission"][2], rtol=1e-12)  # the total
    np.testing.assert_allclose(axs[4].lines[0].get_ydata(), T[0] / run["mean_transmission"][0] - 1, rtol=1e-12)
    assert info["points"] in ("panel", "inset") and info["pixels"] in ("panel", "inset")
    fig, ax, info = pl.plot_skewer_slice(fl, b.box_cMpc, sk["columns"], highlight=(ix, iz), cmap="tol_iridescent")
    fig.canvas.draw()
    assert info["n_row"] == 4 and info["spacing_cMpc"] == pytest.approx(4 * b.dx)
    assert not fig.legends and ax.get_legend() is not None, "the legend inside the panel"
    arrows = [a for a in ax.get_legend().findobj(matplotlib.patches.FancyArrowPatch)]
    assert len(arrows) == 1, "the skewers' legend entry: a line with its arrowhead along it"
    (x0, y0), (x1, y1) = arrows[0]._posA_posB
    assert x1 > x0 and y1 == y0, "the arrow points along the (horizontal) line of the handle"
    u = float(run["period"]) / b.box_cMpc  # km/s per cMpc
    lim3 = dict(k_min=2 * np.pi / b.box_cMpc, k_max=np.pi / (4 * b.dx), r_min=4 * b.dx, r_max=b.box_cMpc / 2)
    fig, axs = pl.plot_overflux_3d(
        run["flux3d"], run["axes3d"], kms_per_cMpc=u, r_bao=2.0, r_marks={"$\\Delta$": 1.0}, limits=lim3
    )
    fig.canvas.draw()
    assert {"$r_{\\rm BAO}$ ", "$\\Delta$ "} <= {t.get_text() for t in axs["xi"].texts} and len(axs["P"].texts) == 2
    assert set(axs) == {"P", "xi"} and "cMpc" in axs["xi"].get_xlabel() and not axs["P"].get_title()
    assert axs["P"].get_ylabel().startswith("$P(k)$") and "cMpc$^2$" in axs["xi"].get_ylabel(), "P and r^2 xi in comoving units"
    assert "km" in axs["P"].child_axes[0].get_xlabel() and "km" in axs["xi"].child_axes[0].get_xlabel()
    tot = dict(run["flux3d"]["total"], axes=run["axes3d"])
    period = float(run["period"])
    fig, ax, drawn = pl.plot_overflux_rpar_rperp(tot, kms_per_cMpc=u, r_bao=2.0, delta_v=100.0, n_sigma=1.5, cmap="tol_BuRd")
    fig.canvas.draw()
    assert not fig.legends and ax.get_legend() is None, "no legend: the curves and the diamond are labelled"
    texts = [t.get_text() for t in ax.texts]
    assert pl.line_pair_label(math=False) in texts, "the pair next to the diamond"
    assert any(t.strip() == "echo " + pl.line_pair_label().replace("$\\Delta", "$+\\Delta") for t in texts), "G4 labels"
    assert np.all(np.asarray(ax.collections[0].get_edgecolor())[..., 3] == 0) or ax.collections[0].get_linewidth()[0] == 0
    assert len(ax.child_axes) == 2 and "cMpc" in ax.get_xlabel() and "perp" in ax.get_xlabel() and not ax.get_title()
    assert all("reference" not in c.get_xlabel() + c.get_ylabel() for c in ax.child_axes), (
        "no parenthetical on the reference axes"
    )
    assert "km" in ax.get_ylabel() and "parallel" in ax.get_ylabel(), "native units: perp in cMpc across, par in km/s up"
    dv = period / n_pix
    top = u * run["axes3d"]["rpar"][-1]  # the last row, centred on half the period [km/s]
    assert top == pytest.approx(period / 2)
    np.testing.assert_allclose(ax.get_ylim(), (-dv / 2, period / 2 + dv / 2))
    assert ax.get_xlim()[0] == 0, "one row per pixel separation j dv, j = 0 .. n_pix / 2, each centred on it"
    np.testing.assert_allclose(u * run["axes3d"]["rpar"], dv * np.arange(n_pix // 2 + 1), atol=1e-9)
    # the echoes: r_par = |sqrt(r_bao^2 - r_perp^2) +- Delta|, folded with the period L, in km/s
    d, L = 100.0 / u, b.box_cMpc
    c = pl.echo_curves(2.0, d, L)
    s_ = np.sqrt(2.0**2 - c["r_perp"] ** 2)
    np.testing.assert_allclose(c["plus"], pl.fold_periodic(s_ + d, L)) and np.all(c["plus"] <= L / 2)
    np.testing.assert_allclose(drawn["echo -"]["r_par"], u * pl.fold_periodic(np.abs(s_ - d), L))
    assert drawn["spot"]["r_par"][0] == pytest.approx(100.0) and set(drawn) == {"ring", "spot", "echo +", "echo -"}
    np.testing.assert_allclose(pl.fold_periodic([0.3, 0.7, 1.2, -0.2], 1.0), [0.3, 0.3, 0.2, 0.2])
    sc = pl.echo_scales(40.0, 28.0, 150.0)
    np.testing.assert_allclose(list(sc.values()), [40.0, 28.0, 12.0, 68.0, np.hypot(40.0, 28.0)])
    assert pl.echo_scales(40.0, 50.0, 150.0)["$r_{\\rm BAO} + \\Delta$"] == pytest.approx(60.0), "folded beyond L / 2"
    assert not hasattr(pl, "_reference_wedges")
    # the k-space map: k_perp in cMpc^-1 across, k_par in s/km upwards (native units), reference axes through u, no legend
    fig, ax, chi2k = pl.plot_overflux_kpar_kperp(tot, kms_per_cMpc=u, kperp_edges=run["axes3d"]["kperp_edges"])
    fig.canvas.draw()
    assert chi2k is None and not fig.legends and ax.get_legend() is None and len(ax.child_axes) == 2
    assert "cMpc" in ax.get_xlabel() and "s km" in ax.get_ylabel() and "parallel" in ax.get_ylabel()
    np.testing.assert_allclose(ax.get_ylim(), np.array(run["axes3d"]["kpar_edges"])[[0, -1]] / u)
    assert "P(k_" in fig.axes[-1].get_ylabel() and "cMpc$^3$" in fig.axes[-1].get_ylabel(), "P, not k P"
    # the wedges: r_BAO, the spot in the wedge of mu = 1 only, the echoes per wedge at its mean mu, every scale folded
    fig, axs, chi2 = pl.plot_xi_mu_wedges(tot, kms_per_cMpc=u, r_bao=2.0, delta_v=100.0, box=L)
    fig.canvas.draw()
    assert chi2 is None and len(axs) == 1 and "cMpc" in axs[0].get_xlabel() and "km" in axs[0].child_axes[0].get_xlabel()
    legs = [c for c in axs[0].get_children() if isinstance(c, matplotlib.legend.Legend)]
    assert not fig.legends and len(legs) == 3, "the echoes, the wedges and the styles, inside the panel"
    texts = [[t.get_text() for t in g.get_texts()] for g in legs]
    assert ["measured", "per realisation"] in texts
    echo = [t for t in texts if t[0].startswith("echo")][0]
    assert len(echo) == 2 and "+" in echo[0] and "-" in echo[1] and "Delta v" in echo[0]
    assert all(g.get_zorder() == pl.LEGEND_ZORDER for g in legs)
    assert not any("ticks at the top" in t.get_text() for t in axs[0].texts), "no note"
    ec = pl.wedge_echoes(2.0, 100.0 / u, pl.MU_WEDGES, L)
    mu_ = np.asarray(ec["mu"])
    root = np.sqrt(2.0**2 - (100.0 / u) ** 2 * (1 - mu_**2))
    np.testing.assert_allclose(ec["plus"], pl.fold_periodic(root + mu_ * 100.0 / u, L))
    assert ec["spot_wedge"] == len(pl.MU_WEDGES) - 2, "the spot in the wedge that holds mu = 1"
    drawn_x = {round(t.get_position()[0], 9) for t in axs[0].texts}
    assert round(ec["r_bao"], 9) in drawn_x and round(ec["delta"], 9) in drawn_x
    spot = [ln for ln in axs[0].lines if ln.get_linestyle() == "--" and np.allclose(ln.get_xdata(), ec["delta"])]
    assert len(spot) == 1 and spot[0].get_color() == pl.PALETTE[pl.MU_COLORS[ec["spot_wedge"]]]
    lo, hi = axs[0].get_xlim()
    dashes = {}
    for w in range(len(pl.MU_WEDGES) - 1):  # the echoes: vertical lines in the wedge colour, + dashed, - dotted
        for key in ("plus", "minus"):
            if lo <= ec[key][w] <= hi:
                col = pl.PALETTE[pl.MU_COLORS[w]]
                ln = [x for x in axs[0].lines if np.allclose(x.get_xdata(), ec[key][w]) and x.get_color() == col]
                assert ln, (w, key)
                dashes.setdefault(key, set()).add(tuple(ln[0]._unscaled_dash_pattern[1]))
    assert set(dashes) == {"plus", "minus"} and dashes["plus"].isdisjoint(dashes["minus"]), "two styles"
    with pytest.raises(TypeError):
        pl.plot_xi_mu_wedges(tot, reference={})
    # the density along the skewers (E, F): comoving line of sight, km/s for reference, with its exact expectation
    dens = dict(run["density3d"]["total"], axes=run["density_axes3d"])
    fig, ax, drawn = pl.plot_overflux_rpar_rperp(dens, kms_per_cMpc=u, r_bao=2.0, unit="cMpc", theory=run["density_theory"])
    fig.canvas.draw()
    assert "cMpc" in ax.get_ylabel() and any("km" in c.get_ylabel() for c in ax.child_axes)
    assert set(drawn) == {"ring"} and not fig.legends, "the density: the ring alone, nothing of the metal lines"
    fig, ax, chi2k = pl.plot_overflux_kpar_kperp(dens, kms_per_cMpc=u, unit="cMpc", theory=run["density_theory"])
    fig.canvas.draw()
    assert chi2k > 0 and "cMpc" in ax.get_ylabel() and any("s km" in c.get_ylabel() for c in ax.child_axes)
    fig, axs = pl.plot_overflux_3d(run["density3d"], run["density_axes3d"], kms_per_cMpc=u, theory=run["density_theory"])
    fig.canvas.draw()
    assert any(ln.get_linestyle() == "--" for ln in axs["P"].lines), "the expectation, dashed"
    fig, axs, chi2 = pl.plot_xi_mu_wedges(dict(dens, theory=run["density_theory"]), kms_per_cMpc=u)
    assert len(axs) == 2 and len(chi2) == len(pl.MU_WEDGES) - 1
    k_lim = (2 * np.pi / period, np.pi / (period / n_pix))
    fig, ax = pl.plot_p1d_reference(run["p1d"], P1D_DESI_DR1_Z22, kms_per_cMpc=u, k_limits=k_lim, k_max=1.2 * k_lim[1])
    fig.canvas.draw()
    assert not fig.legends and len(ax.get_legend().get_texts()) == 5  # three overfluxes, the boxes and the reference
    assert len(ax.child_axes) == 1 and "cMpc" in ax.child_axes[0].get_xlabel(), "one reference top axis"
    spans = sorted((p.get_x(), p.get_x() + p.get_width()) for p in ax.patches if isinstance(p, matplotlib.patches.Rectangle))
    assert np.isclose(spans[0][1], k_lim[0]) and np.isclose(spans[-1][0], k_lim[1]), "shaded below k_min, above k_max"
    fig, ax = pl.plot_p1d_reference(
        run["density_p1d"], P1D_DESI_DR1_Z22, fields=list(run["density3d"]), unit="cMpc", kms_per_cMpc=u, k_max=3.0
    )
    fig.canvas.draw()
    assert "cMpc" in ax.get_xlabel() and "km" in ax.child_axes[0].get_xlabel() and len(ax.get_legend().get_texts()) == 6
    dens1d = run["density_flux"]
    for orders_fn, terms_fn in ((pl.plot_p1d_orders, pl.plot_p1d_terms), (pl.plot_xi1d_orders, pl.plot_xi1d_terms)):
        fig, axs = orders_fn(dens1d, kms_per_cMpc=u, unit="cMpc")
        assert "cMpc" in axs[1].get_xlabel() and "km" in axs[0].child_axes[0].get_xlabel()
        fig, axs = terms_fn(dens1d, kms_per_cMpc=u, unit="cMpc")
        assert axs.shape == (1,), "the density: one order (2), its terms 1-1, 2-2, 1-2"
    ens = run["flux"]
    dv = [0.0, float(LineBundle(["HI-LyA", "SiIII-1206"]).delta_v[1])]
    k_lim = (2 * np.pi / period, np.pi / (period / n_pix))
    for kind, orders_fn, terms_fn, kw in (
        ("P", pl.plot_p1d_orders, pl.plot_p1d_terms, dict(k_limits=k_lim)),
        ("xi", pl.plot_xi1d_orders, pl.plot_xi1d_terms, dict(delta_v_list=dv)),
    ):
        fig, axs = orders_fn(ens, kms_per_cMpc=u, **kw)
        fig.canvas.draw()
        assert axs.shape == (2,) and not axs[0].get_title() and "cMpc" in axs[0].child_axes[0].get_xlabel()
        fig, axs = terms_fn(ens, kms_per_cMpc=u, **kw)
        fig.canvas.draw()
        assert axs.shape == (3,), f"{kind}: one row per order 2, 3, 4 (order 1 vanishes, order 0 not shown)"
        for ax in axs:  # the legend inside the panel
            bb, la = ax.get_legend().get_window_extent(), ax.get_window_extent()
            assert bb.x0 >= la.x0 - 1 and bb.x1 <= la.x1 + 1
        assert not any(ax.get_title() for ax in axs)
        if kind == "P":  # from k_min to k_max, shaded outside
            spans = sorted(
                (p.get_x(), p.get_x() + p.get_width()) for p in axs[0].patches if isinstance(p, matplotlib.patches.Rectangle)
            )
            assert np.isclose(spans[0][1], k_lim[0]) and np.isclose(spans[-1][0], k_lim[1])
        else:  # the separation folded into half the period, dashed, on every row
            assert all(any(ln.get_linestyle() == "--" for ln in ax.lines) for ax in axs)
    xi_lines = [c for c in pl.plot_xi1d_orders(ens)[1][0].lines if c.get_marker() == "o"]
    assert xi_lines, "xi1D: points with error bars too"
    line = [c for c in pl.plot_p1d_orders(ens)[1][0].lines if c.get_marker() == "o"][0]
    assert line.get_linestyle() == "-" and line.get_markersize() == pl._MS_SMALL, "small points joined by a thin line"
    # explicit display bins: P1D in the bins of bao_k_edges along the line of sight, xi1D at every lag
    k_f = 2 * np.pi / period
    e1 = bao_k_edges(k_f, np.pi / (period / n_pix), 3.0 / u, n_log=4)
    pts = [c for c in pl.plot_p1d_orders(ens, k_edges=e1)[1][0].lines if c.get_marker() == "o"][0]
    assert np.all(pts.get_xdata() >= e1[0]) and np.all(pts.get_xdata() <= e1[-1])
    old = pl.plot_orders_superimposed(ens, "P", k_edges=e1)[1][0]  # the data drawn: those of the building block
    ref = [c for c in old.lines if c.get_marker() == "o"][0]
    np.testing.assert_array_equal(pts.get_ydata(), ref.get_ydata())
    xs = [c for c in pl.plot_xi1d_orders(ens)[1][0].lines if c.get_marker() == "o"][0]
    np.testing.assert_allclose(np.diff(xs.get_xdata()), period / n_pix), "every lag j dv"
    assert not hasattr(pl, "plot_orders_compared") and not hasattr(pl, "plot_terms_compared")


def test_comoving_axis():
    """comoving_axis: v = aH x and k_v = k_x / (aH) on the secondary axes (notebook 09, Part D)."""
    u = 80.0  # km/s per cMpc
    fig, (a, b) = plt.subplots(1, 2)
    a.set_xlim(0.0, 800.0)
    b.set(xscale="log", xlim=(1e-3, 1e-1))
    ra, kb = pl.comoving_axis(a, u, "r"), pl.comoving_axis(b, u, "k")
    ry = pl.comoving_axis(a, u, "r", side="right", label="y [cMpc]")
    fig.canvas.draw()
    # the secondary axes alone draw the ticks of their side (the style puts in-ticks on all four sides)
    assert not any(t.tick2line.get_visible() for t in a.xaxis.get_major_ticks()), "no primary ticks on top"
    assert not any(t.tick2line.get_visible() for t in a.yaxis.get_major_ticks()), "no primary ticks on the right"
    assert all(t.tick1line.get_visible() for t in a.xaxis.get_major_ticks()), "the bottom ticks stay"
    assert any(t.tick2line.get_visible() for t in ra.xaxis.get_major_ticks()), "the secondary axis has its own (on top)"
    np.testing.assert_allclose(ra.get_xlim(), (0.0, 10.0))  # 800 km/s = 10 cMpc
    np.testing.assert_allclose(kb.get_xlim(), (0.08, 8.0))  # 0.001 s/km = 0.08 cMpc^-1
    assert ra.get_xlabel() == pl._units("cMpc")["r"] and ry.get_ylabel() == "y [cMpc]" and "log" in kb.get_xscale()
    with pytest.raises(ValueError):
        pl.comoving_axis(a, u, "v")
    plt.close(fig)


def test_gas_cells_along_a_skewer():
    """plot_gas_cells: the grid cubes and the Voronoi polyhedra crossed by one skewer through the periodic face."""
    from skewerjax.core import cells_from_values, cells_on_grid, select_cells, voronoi_volumes

    box, n = 4.0, 8
    rng = np.random.default_rng(0)
    vals = lambda m: dict(temperature=10 ** rng.uniform(4, 5, m), velocity=rng.normal(0, 50, (m, 3)))
    grid = cells_on_grid(box, {"HI": 10 ** rng.uniform(-13, -11, (n,) * 3), "SiIII": np.full((n,) * 3, 1e-14)}, **vals(n**3))
    gen = rng.uniform(0, box, (600, 3))
    vor = cells_from_values(gen, box, {"HI": 10 ** rng.uniform(-13, -11, 600)}, volumes=voronoi_volumes(gen, box), **vals(600))
    pts = np.array([3.0, 1.1, 2.3]) + np.linspace(0, 2.5, 30)[:, None] * np.array([1.0, 0.6, 0.4]) / np.linalg.norm(
        [1.0, 0.6, 0.4]
    )
    for cells, n_sp in ((grid, 2), (vor, 1)):
        near = select_cells(cells, pts[None])
        fig, axs, info = pl.plot_gas_cells(cells, near, pts)
        pl.add_panel_letters(axs)
        fig.canvas.draw()
        assert axs.shape == (n_sp, 3) and 0 < info["n_crossed"] <= info["n_selected"] == len(near)
        assert np.isclose(info["path_length"], 2.5)
        assert np.isfinite(info["arrow_scale"]) and info["arrow_scale"] > 0 and _arrows_outside(fig) <= 1e-9
        assert pl.round_speed(info["key_kms"]) == info["key_kms"]
    fig, _, info = pl.plot_gas_cells(vor, near, pts, arrow_scale=2e-3, view=(60.0, 10.0))  # the override, another view
    assert info["arrow_scale"] == 2e-3 and _arrows_outside(fig) <= 1e-9


def _arrows_outside(fig) -> float:
    """The largest distance of a drawn 3D arrow (quiver segments) beyond its panel's limits, in units of the span."""
    worst = 0.0
    for ax in fig.axes:
        if not hasattr(ax, "get_zlim"):
            continue
        lim = np.array([ax.get_xlim(), ax.get_ylim(), ax.get_zlim()])
        for c in ax.collections:
            seg = getattr(c, "_segments3d", None)
            if seg is not None and len(seg):
                p = np.asarray(seg, dtype=float).reshape(-1, 3)
                worst = max(worst, float((np.maximum(lim[:, 0] - p, p - lim[:, 1]) / (lim[:, 1] - lim[:, 0])).max()))
    return worst


def test_velocity_arrow_scale_and_fit():
    """velocity_arrow_scale: the 90th-percentile speed is 0.8 median cell sizes long (at least 6 % of the span), robust to
    an outlier, finite for any input, the override kept; round_speed: 1, 2 or 5 x 10^n; _fit_arrows: every tip inside
    the limits, which grow by at most 10 % of the span per side."""
    rng = np.random.default_rng(3)
    v = rng.normal(0, 100, (500, 3))
    h = rng.uniform(0.1, 0.3, 500)
    sc = pl.velocity_arrow_scale(v, h)
    assert np.isclose(sc * np.percentile(np.linalg.norm(v, axis=1), 90), 0.8 * np.median(h))
    v_out = v.copy()
    v_out[0] = 1e5  # one outlier barely changes the scale (a percentile, not the maximum)
    assert abs(pl.velocity_arrow_scale(v_out, h) / sc - 1) < 0.02
    assert np.isclose(pl.velocity_arrow_scale(v, h, span=100.0) * np.percentile(np.linalg.norm(v, axis=1), 90), 6.0)
    for bad in (np.zeros((4, 3)), np.full((4, 3), np.nan), np.zeros((0, 3))):
        x = pl.velocity_arrow_scale(bad, np.full(len(bad), 0.2))
        assert np.isfinite(x) and x > 0
    assert pl.velocity_arrow_scale(v, h, scale=0.003) == 0.003
    with pytest.raises(ValueError):
        pl.velocity_arrow_scale(v, h, scale=-1.0)
    assert [pl.round_speed(x) for x in (130.0, 170.0, 0.03, 4.0, 7.5e3)] == [100.0, 200.0, 0.02, 5.0, 1e4]
    assert pl.round_speed(0.6, below=True) == 0.5 and pl.round_speed(0.0) == 1.0
    assert pl.velocity_key_speed(v) in {50.0, 100.0, 200.0}
    starts = rng.uniform(0, 1, (200, 3))
    arrows = rng.normal(0, 0.5, (200, 3))
    fit, capped, lo, hi = pl._fit_arrows(starts, arrows, np.zeros(3), np.ones(3))
    tips = starts + fit
    assert np.all(tips >= lo - 1e-12) and np.all(tips <= hi + 1e-12) and capped.any()
    assert np.all(lo >= -0.1 - 1e-12) and np.all(hi <= 1.1 + 1e-12)
    assert np.allclose(fit[~capped], arrows[~capped])  # only the capped arrows are shortened, along their direction
    assert np.allclose(np.cross(fit[capped], arrows[capped]), 0.0, atol=1e-12)


def test_deposition_figures_on_voronoi_cells():
    """plot_deposition_cells, _pieces and _rings from the parts of cell_pieces: the pieces of a periodic skewer through
    random Voronoi cells within half_range_cMpc of its seam, every piece's tau on one period with velocity_period; the
    same pieces and labels in the three figures, one piece on the velocity seam, the ring windows centred on the seam,
    labels fit inside their elements or go next to them, stacked arrows never overlap each other or a label; plot_frames_axes relabels each line's own window on the global axis (the reference
    line stays within it) and, with ``wrapped``, draws wrap_to_ref_line's result."""
    from skewerjax.core import cell_pieces, cells_from_values, compute_optical_depth, voronoi_volumes, wrap_to_ref_line

    box = 4.0
    rng = np.random.default_rng(1)
    gen = rng.uniform(0, box, (800, 3))
    vel = rng.normal(0, 30, (800, 3))
    vel[:, 0] = 150.0  # everything moves away: the end wraps
    cells = cells_from_values(
        gen,
        box,
        {"HI": 10 ** rng.uniform(-12, -10, 800)},
        temperature=10 ** rng.uniform(4, 4.5, 800),
        velocity=vel,
        volumes=voronoi_volumes(gen, box),
    )
    x = np.linspace(0.0, box, 41)
    cosmo = Cosmology(H0=phys.H0, Omega_r_0=0.0, Omega_m_0=0.31, Omega_k_0=0.0, Omega_lambda_0=0.69)
    sb = SkewerBundle(
        spatial_points=jnp.asarray(np.stack([x, np.full_like(x, 2.0), np.full_like(x, 2.0)], -1)[None]),
        a0=0.25,
        ICs_defined_at_index=0,
        cosmologies=cosmo,
    )
    lya = LineBundle(["HI-LyA"])
    pieces, parts = cell_pieces(cells, lya, skewer_bundle=sb, return_parts=True)
    P = float(pieces.velocity_span_kms[0])
    v_ring = (np.arange(400) + 0.5) * P / 400
    dtau = np.asarray(
        compute_optical_depth(pieces, v_ring, lya, per_piece=True, velocity_period=True, resolution_tolerance=None)
    )[0, 0]
    infos = []
    for f in (
        lambda: pl.plot_deposition_cells(parts, sb, half_range_cMpc=0.6),
        lambda: pl.plot_deposition_pieces(parts, sb, half_range_cMpc=0.6),
        lambda: pl.plot_deposition_rings(parts, sb, v_ring, dtau, period=P, half_range_cMpc=0.6, half_range_kms=0.25 * P),
    ):
        fig, axs, info = f()
        fig.canvas.draw()
        assert not pl.check_fonts(fig)
        assert np.isfinite(info["arrow_scale"]) and info["arrow_scale"] > 0 and _arrows_outside(fig) <= 1e-9
        infos.append(info)
    assert (
        infos[1]["arrow_scale"] == infos[2]["arrow_scale"] and infos[1]["key_kms"] == infos[2]["key_kms"]
    )  # (b), (c): one scale
    assert all(i["piece_labels"] == info["piece_labels"] and i["seam_piece"] == info["seam_piece"] for i in infos)
    assert set(axs) == {"c", "d", "e"} and all(n > 1 for n in info["n_cells"].values())
    assert np.allclose(info["window_deg"]["s"], 360 * 0.6 / box) and np.allclose(
        info["window_deg"]["v"], 90.0
    )  # centred: +-l, +-v_range
    assert len(info["piece_labels"]) == sum(info["n_pieces"].values())
    assert all(
        lab.rstrip("abcdefghijklmnopqrstuvwxyz") in info["labels"]["start"] + info["labels"]["end"]
        for lab in info["piece_labels"]
    )
    assert info["seam_piece"] in info["piece_labels"] and abs(info["seam_v_c"]) < 0.5 * P
    assert [pl._label_fit(w, h, "S1a", 13)[0] for w, h in ((1.0, 0.5), (0.3, 1.0), (0.05, 1.0))] == ["h", "v", None]
    x0 = np.array([0.0, 0.05, 0.1, 0.5])
    x1 = x0 + 0.2
    y = np.zeros(4)
    lev = pl._arrow_levels(x0, x1, y, pad=0.0, dy=1.0)  # the three overlapping arrows on three levels
    assert sorted(lev[:3]) == [0, 1, 2] and lev[3] == 0
    assert pl._arrow_levels(x0[3:], x1[3:], y[3:], pad=0.0, dy=1.0, boxes=[(0.55, 0.6, -0.2, 1.2)])[0] == 2  # above a label
    two = LineBundle(["HI-LyA", "SiIII-1206"], ref_line_id="HI-LyA")
    v = np.arange(0.5, P, 1.0)
    F = np.exp(-np.stack([np.exp(-(((v - 50) / 5) ** 2)), 0.1 * np.exp(-(((v - 50) / 3) ** 2))]))
    figs = pl.plot_frames_axes(v, F, two, a_dagger=0.25, tau_scale={"SiIII-1206": 10.0})
    assert list(figs) == ["own", "global", "wavelength"]
    axs = [ax for _, ax in figs.values()]
    np.testing.assert_allclose(axs[0].get_lines()[1].get_ydata(), F[1] ** 10, rtol=1e-12)  # display only: exp(-10 tau)
    assert "for visibility" in axs[0].get_lines()[1].get_label()
    lya_line = axs[1].get_lines()[0].get_xdata()
    assert lya_line.min() >= 0.0 and lya_line.max() < P  # the reference line keeps its own window
    assert np.isclose(axs[1].get_lines()[3].get_xdata()[0], v[0] - float(two.delta_v[1]))  # Si III moved by -delta_v
    v1 = (np.arange(200) + 0.5) * P / 200  # (d): wrap_to_ref_line on one period
    tau1 = np.stack([np.exp(-(((v1 - 50) / 5) ** 2)), 0.1 * np.exp(-(((v1 - 50) / 3) ** 2))])[:, None]
    tau1_ref = np.asarray(wrap_to_ref_line(tau1, v1, two, velocity_period=P))
    figs = pl.plot_frames_axes(
        v,
        F,
        two,
        a_dagger=0.25,
        tau_scale={"SiIII-1206": 10.0},
        wrapped=dict(v=v1, tau=tau1[:, 0], tau_ref=tau1_ref[:, 0], period=P),
    )
    for fig, _ax in figs.values():  # one figure per case, all of the same layout
        fig.canvas.draw()
        assert not pl.check_fonts(fig) and tuple(fig.get_size_inches()) == (22, 7.5)
    assert len({tuple(np.round(ax.get_position().bounds, 9)) for _, ax in figs.values()}) == 1
    assert figs["own"][1].get_xlim() == figs["wrapped"][1].get_xlim() == (0.0, P)
    np.testing.assert_allclose(figs["wrapped"][1].get_lines()[2].get_ydata(), np.exp(-10.0 * tau1_ref[1, 0]), rtol=1e-12)


def test_quickstart_and_tutorial_figures():
    """The figure functions of notebooks 00, 00b and 04 on toy inputs, and the small helpers (custom panel labels, the
    shading outside a velocity window)."""
    import jax

    from skewerjax.core import (
        bundle_frame,
        cell_pieces,
        cells_from_values,
        compute_optical_depth,
        tau_by_path_bins,
    )
    from skewerjax.summary_stats import DeltaFieldBuilder, OverfluxStatistics1D, fit_mcdonald_a

    box = 4.0
    rng = np.random.default_rng(2)
    gen = rng.uniform(0, box, (500, 3))
    nH = 10 ** rng.uniform(-12, -10, 500)
    cells = cells_from_values(gen, box, {"HI": nH, "SiIII": 1e-4 * nH}, temperature=2e4, velocity=rng.normal(0, 30, (500, 3)))
    x = np.linspace(0.0, box, 61)
    cosmo = Cosmology(H0=phys.H0, Omega_r_0=0.0, Omega_m_0=0.31, Omega_k_0=0.0, Omega_lambda_0=0.69)
    pts = np.stack([np.stack([x, np.full_like(x, y), np.full_like(x, 2.0)], -1) for y in (1.0, 2.0, 3.0)])
    sb = SkewerBundle(spatial_points=jnp.asarray(pts), a0=0.25, ICs_defined_at_index=0, cosmologies=cosmo)
    lines = LineBundle(["HI-LyA", "SiIII-1206"], ref_line_id="HI-LyA")
    pieces, parts = cell_pieces(cells, lines, skewer_bundle=sb, return_parts=True)
    P = float(pieces.velocity_span_kms[0])
    v = (np.arange(128) + 0.5) * P / 128
    tau = np.asarray(
        compute_optical_depth(pieces, v, lines, velocity_period=True, line_frame="reference", resolution_tolerance=None)
    )
    stats = OverfluxStatistics1D(
        DeltaFieldBuilder(tau, labels=["Lya", "SiIII"], key_joiner="+").build(), float(v[1] - v[0])
    ).compute_p1d()
    fig, axs = pl.plot_toy_box(cells, sb, lines, v, np.exp(-tau), stats)
    fig.canvas.draw()
    assert axs[2].get_title() == "P1D over 3 skewers"
    frame = bundle_frame(sb)
    A_H = parts["absorbers"]["HI"]
    edges = np.linspace(0.0, box, 5)
    one = lambda lb: LineBundle([lb], ref_line_id=lb)
    tb = [
        np.asarray(tau_by_path_bins(parts["absorbers"][sp], one(l), v, frame, edges))[:, 0]
        for sp, l in (("HI", "HI-LyA"), ("SiIII", "SiIII-1206"))
    ]
    s = np.asarray(frame.s_grid[0])
    fig, axs, info = pl.plot_tau_stretches(
        edges, s, np.ones_like(s), v, tuple(tb), tuple(t.sum(0) for t in tb), frame, A_H, box=box
    )
    fig.canvas.draw()
    assert 0.0 <= info["s_max"] <= box and np.all(np.diff(info["vH_s"]) > 0)
    pv = {"n": nH[None, :61], "T": np.full((1, 61), 2e4), "v_los": np.zeros((1, 61))}
    slab = dict(sigma=rng.uniform(1.0, 10.0, (32, 32)), extent=(0.0, box, 0.0, box), axes=(1, 2))
    pl.plot_slab_and_skewer(slab, x, (pv, pv), box=box, y_sk=2.0, z_c=2.0, thick=0.5, x_mark=1.0, zoom=0.5)[0].canvas.draw()
    lo = np.sort(rng.uniform(0, box, 20))
    hi = lo + 0.05
    pl.plot_cell_crossings(lo, hi, nH[:20], np.full(20, 0.2), 0.01)[0].canvas.draw()
    pl.plot_voigt_of_piece(1e4, lines.lines)[0].canvas.draw()
    pl.plot_wrap_tiles(v, (tau[0, 0], tau[1, 0]), tau[:, :1], period=P, delta_v=float(lines.delta_v[1]), v_own=v[10], v_obs=v[5])[
        0
    ].canvas.draw()
    st = (
        OverfluxStatistics1D(DeltaFieldBuilder(tau, labels=["Lya", "SiIII"], key_joiner="+").build(), float(v[1] - v[0]))
        .compute_p1d()
        .compute_xi1d()
    )
    k = np.asarray(st.k)[1:]
    P_tot, P_ll = np.asarray(st.Pk_total)[1:].real, np.asarray(st.Pk_pair[("Lya", "Lya")])[1:].real
    fit = fit_mcdonald_a(k, P_tot, P_ll, delta_v=float(lines.delta_v[1]), k_range=(0.0, 0.1))
    ov = DeltaFieldBuilder(tau, labels=["Lya", "SiIII"], key_joiner="+").build()
    pl.plot_overflux_p1d_xi(v, ov, st, k, P_tot, P_ll, fit, skewer=1, period=P, delta_v=float(lines.delta_v[1]))[0].canvas.draw()
    g1 = PowerLawGRF(A_s=1.0, k_0=1.0, n_s=0.5, L=10.0, N=64)
    g2 = PowerLawGRF(A_s=1.0, k_0=1.0, n_s=0.5, L=(5.0, 10.0), N=(8, 16))
    g3 = PowerLawGRF(A_s=1.0, k_0=1.0, n_s=0.5, L=(5.0,) * 3, N=(8,) * 3)
    key = jax.random.PRNGKey(0)
    fig, axs = pl.plot_grf_draws([(g, g.sample(key)) for g in (g1, g2, g3)], ["1D", "2D", "3D"])
    fig.canvas.draw()
    assert axs[1].get_ylabel() == "y (line of sight) [Mpc]" and axs[2].get_ylabel() == "y [Mpc]"
    fig, ax2 = plt.subplots(1, 2)
    for k_, end in enumerate(("start", "end")):
        pl.shade_outside_window(ax2[k_], np.linspace(-50.0, 150.0, 11) + 100.0 * k_, 100.0, end)
    pl.add_panel_letters(ax2, labels=["a1", "a2"], box=False)
    assert ax2[1].get_xlim() == (50.0, 250.0) and [t.get_text() for a in ax2 for t in a.texts] == ["(a1)", "(a2)"]


def test_benchmark_figures():
    """The benchmark figures on synthetic rows (``comparison.benchmark.result_rows``): every series and stage drawn."""
    from skewerjax.comparison import benchmark as B

    rows = []
    for sim, cells in (("TNG50-4", 1e7), ("TNG50-2", 1e9)):
        for n in (1, 4, 16):
            for code, plat, mode in (
                ("fake_spectra", "cpu", None),
                ("skewerjax", "cpu", "default"),
                ("skewerjax", "gpu", "matched"),
            ):
                st = {s: (0.1 * n, 0.09 * n, 0.11 * n) for s in B.STAGES}
                rows.append(
                    dict(
                        code=code,
                        platform=plat,
                        mode=mode,
                        sim=sim,
                        n_skewers=n,
                        threads=32,
                        status="ok",
                        n_gas=cells,
                        total=(0.6 * n, 0.5 * n, 0.7 * n),
                        stages=st,
                        peak_rss_GB=1.0 + n,
                        compile_s=None if mode is None else 2.0,
                        gpu_peak_GB=0.5 if plat == "gpu" else None,
                        agreement={"HI-LyA": dict(max_dF=2e-5, rms_dF=3e-6, meanF_default=0.5, meanF_fake_spectra=0.52)},
                    )
                )
    fig, axs = plt.subplots(2, 3)
    s = pl.plot_bench_vs(axs[0, 0], rows, sim="TNG50-4", fit_from=1)
    assert np.isclose(s[("skewerjax", "cpu", "default")], 1.0)  # t ∝ N
    pl.plot_bench_vs(axs[0, 1], rows, x="n_gas", what="io+selection", n_skewers=16)
    pl.plot_bench_vs(axs[0, 2], rows, sim="TNG50-4", what="gpu")
    pl.plot_bench_speedup(axs[1, 0], rows, sim="TNG50-4")
    pl.plot_bench_stages(axs[1, 1], rows, sim="TNG50-2", n_skewers=4)
    pl.plot_bench_accuracy(
        axs[1, 2], [dict(r, mode="matched") for r in rows if r["code"] == "skewerjax" and r["platform"] == "cpu"]
    )
    fig.canvas.draw()
    assert len(axs[1, 1].patches) == 3 * len(B.STAGES) and np.isclose(pl.bench_value(rows[0], "throughput")[0], 1 / 0.6)
    assert [l.get_ydata()[0] for l in axs[1, 2].get_lines()[:2]] == [2e-5, 3e-6]  # max and rms |dF| of the matched mode
