"""Smoke tests of the plotting module: the style and every tutorial figure function draw without error (Agg)."""

import matplotlib

matplotlib.use("Agg")
import jax.numpy as jnp  # noqa: E402
import matplotlib.legend  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from skewerjax.core import Cosmology, LineBundle, SkewerBundle, phys  # noqa: E402
from skewerjax.simulations import PowerLawGRF, ToyBoxes  # noqa: E402
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
    """The figures of notebooks 08 and 10 on a small toy box: the fields (08 A2), the 3D statistics of the realisations
    against their expectation and the continuum (08 A3), one skewer (09 C1), the 3D statistics of the overfluxes (09 D:
    maps in native units, isotropic quantities in comoving units, the echo curves, the wedges with their characteristic
    scales), the 1D statistics of the overflux by order and term (09 E), and the density along the skewers through the
    same figures (09 B, in comoving units, with their exact expectation): no titles, the line of sight vertical in the
    maps; the options cmap, quantity, color_scale, levels and the axis scales."""
    from dataclasses import replace

    from skewerjax.core import LineBundle, lattice_columns
    from skewerjax.simulations import LINE_IDS, ToyBoxes
    from skewerjax.summary_stats import bao_k_edges, box_grid, field_statistics, grid_limits
    from skewerjax.utils.reference_data import P1D_DESI_DR1

    P1D_DESI_DR1_Z22 = P1D_DESI_DR1[2.2]

    from helpers import toy_lattice_statistics

    b = ToyBoxes(box_cMpc=8.0, n_cells=16, n_real=3, bao_ratio=1e-3)
    fig, axs, info = pl.plot_toy_fields(replace(b, velocity="linear").fields(0), b.box_cMpc, arrow_every=4)
    fig.canvas.draw()
    assert axs.shape == (2, len(b.species)) and info["arrow_scale"] > 0, (
        "density and velocity rows (no temperature), every species"
    )
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
        (b.cells(i) for i in range(b.n_real)), gs, species="HI", power=g.power, power_nw=lambda k: g.power(k, bao=False), log=None
    )
    run = toy_lattice_statistics(b, range(b.n_real), (4, 4), p, n_pix, decompose=True, line_ids=LINE_IDS[:2])  # two lines here
    for quantity, k_scale in (("xi", "linear"), ("r2xi", "log")):
        fig, axs, out = pl.plot_isotropic_recovery(
            res, quantity=quantity, r_bao=2.0, grf=b.grf(), k_scale=k_scale, k_marks={"$k_1$": 1.0, "$k_2$": 50.0}
        )
        fig.canvas.draw()
        assert axs[0, 0].get_xscale() == k_scale and axs[0, 1].get_yscale() == ("symlog" if quantity == "xi" else "linear")
    assert {"P_chi2_dof", "xi_chi2_dof", "xi_mean_pull", "xi_curve_r"} <= set(out)
    labels = [t.get_text() for t in axs[0, 0].get_legend().get_texts()]
    assert labels == ["per realisation", "expectation", r"mean $\pm$ error"], labels
    fig_a, axs_a, _ = pl.plot_isotropic_recovery(res, quantity="r2xi", r_bao=2.0, grf=b.grf(), show_analytic=True)
    labels = [t.get_text() for t in axs_a[0, 0].get_legend().get_texts()]
    assert labels == ["per realisation", "expectation", "analytic input", r"mean $\pm$ error"], labels
    assert len(axs_a[0, 0].lines) > len(res["P_iso_each"]) + 1, "P, with the continuum"
    assert any(ln.get_color() == pl.PALETTE["purple"] for ln in axs_a[0, 0].lines), "the continuum in a visible colour"
    plt.close(fig_a)
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
    assert axs[0, 0].get_ylabel().startswith("$P(k)$")
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
            assert tuple(figs[0].get_size_inches()) == tuple(figs[1].get_size_inches()), "the two figures alike"
            assert figs[0].get_size_inches()[0] <= 14 and figs[0].get_size_inches()[1] <= 12.5, "trimmed to the panel"
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
            marks = [t.get_text() for t in axs[0].texts if "_{\\" not in t.get_text()]
            assert marks == ["$a$ "], "the marks inside the range only, labelled"
            spans = [q for q in axs[0].patches if isinstance(q, matplotlib.patches.Rectangle)]
            edges = sorted(x for q in spans for x in (q.get_x(), q.get_x() + q.get_width()))
            assert len(spans) == 2 and np.isclose(edges[1], lim["r_min"]) and np.isclose(edges[2], lim["r_max"]), (
                "shaded outside the full coverage, as panel (b) of plot_isotropic_recovery"
            )
            assert any("r_{\\min}" in t.get_text() for t in axs[0].texts) and any(
                "r_{\\max}" in t.get_text() for t in axs[0].texts
            )
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
        b.box_cMpc,
        n,
        sk["v"],
        sk["tau"][:, 0],
        mean_transmission=run["mean_transmission"],
        points_per_cell=p,
        line_ids=LINE_IDS[:2],
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
    assert not fig.legends and [t.get_text() for t in ax.get_legend().get_texts()] == [pl.line_pair_label(math=False)]
    assert not any("echo" in t.get_text() or "Delta" in t.get_text() for t in ax.texts), "the pair's name, in the legend only"
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
    assert drawn["spot"]["r_par"][0] == pytest.approx(100.0)
    assert set(drawn) == {"ring", "spots", "spot", "echoes", "echo +", "echo -"} and len(drawn["spots"]) == 1
    np.testing.assert_allclose(pl.fold_periodic([0.3, 0.7, 1.2, -0.2], 1.0), [0.3, 0.3, 0.2, 0.2])
    assert not hasattr(pl, "_reference_wedges")
    # the k-space map: k_perp in cMpc^-1 across, k_par in s/km upwards (native units), reference axes through u, no legend
    fig, ax, chi2k = pl.plot_overflux_kpar_kperp(tot, kms_per_cMpc=u, kperp_edges=run["axes3d"]["kperp_edges"])
    fig.canvas.draw()
    assert chi2k is None and not fig.legends and ax.get_legend() is None and len(ax.child_axes) == 2
    assert "cMpc" in ax.get_xlabel() and "s km" in ax.get_ylabel() and "parallel" in ax.get_ylabel()
    np.testing.assert_allclose(ax.get_ylim(), np.array(run["axes3d"]["kpar_edges"])[[0, -1]] / u)
    assert "P(k_" in fig.axes[-1].get_ylabel() and "cMpc$^3$" in fig.axes[-1].get_ylabel(), "P, not k P"
    # the wedges: only the line-of-sight scales, each folded into [0, L / 2]: r_BAO, the spot and the echoes of every pair
    fig, axs, chi2 = pl.plot_xi_mu_wedges(tot, kms_per_cMpc=u, r_bao=2.0, delta_v=100.0, box=L)
    fig.canvas.draw()
    assert chi2 is None and len(axs) == 1 and "cMpc" in axs[0].get_xlabel() and "km" in axs[0].child_axes[0].get_xlabel()
    legs = [c for c in axs[0].get_children() if isinstance(c, matplotlib.legend.Legend)]
    assert not fig.legends and len(legs) == 3, "the line-of-sight scales, the wedges and the styles"
    texts = [[t.get_text() for t in g.get_texts()] for g in legs]
    assert ["measured", "per realisation"] in texts
    pairs_leg = [g for g in legs if "line-of-sight" in g.get_title().get_text()][0]
    assert [t.get_text() for t in pairs_leg.get_texts()][0] == pl.line_pair_label(math=False)
    assert all(g.get_zorder() == pl.LEGEND_ZORDER for g in legs)
    assert not any("ticks at the top" in t.get_text() for t in axs[0].texts), "no note"
    sc = pl.los_scales(2.0, list(pl.LINE_STYLES)[:2], np.array([0.0, 100.0 / u]), L)
    pr = sc["pairs"][0]
    assert sc["ring"] == pytest.approx(pl.fold_periodic(2.0, L))
    assert pr["spot"] == pytest.approx(pl.fold_periodic(100.0 / u, L))
    assert pr["plus"] == pytest.approx(pl.fold_periodic(2.0 + 100.0 / u, L))
    assert pr["minus"] == pytest.approx(pl.fold_periodic(abs(2.0 - 100.0 / u), L))
    drawn_x = {round(t.get_position()[0], 9) for t in axs[0].texts}
    assert round(sc["ring"], 9) in drawn_x
    lo, hi = axs[0].get_xlim()
    for key, ls in (("spot", "-"), ("plus", "--"), ("minus", ":")):
        if lo <= pr[key] <= hi:
            found = [ln for ln in axs[0].lines if np.allclose(ln.get_xdata(), pr[key]) and ln.get_color() == pr["color"]]
            assert found and found[0].get_linestyle() == ls, key
    assert not hasattr(pl, "wedge_echoes"), "the per-wedge echoes at the mean mu are gone"
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
    assert (
        "cMpc" in ax.get_xlabel()
        and "km" in ax.child_axes[0].get_xlabel()
        and len(ax.get_legend().get_texts()) == len(run["density3d"]) + 3
    )
    # the options of notebook 11 (one simulation): the legend of the ensemble, one reference per field, the spot without
    # r_BAO, a density map under the skewers, the error bars of the reference
    lab = "each of the 3 lines of sight (not independent)"
    ref = {f: {k: 1.1 * np.asarray(run["density3d"][f][k]) for k in ("P_iso", "xi_iso")} for f in run["density3d"]}
    fig, axs = pl.plot_overflux_3d(run["density3d"], run["density_axes3d"], theory=ref, theory_label="whole box", each_label=lab)
    texts = [t.get_text() for t in axs["P"].get_legend().get_texts()]
    assert "whole box" in texts and lab in texts
    dashed = [ln for ln in axs["P"].lines if ln.get_linestyle() == "--"]
    assert len(dashed) == len(ref) and len({ln.get_color() for ln in dashed}) == len(ref), "one reference per field, its colour"
    fig, axs, _ = pl.plot_xi_mu_wedges(tot, kms_per_cMpc=u, r_bao=None, delta_v=100.0, box=L, each_label=lab)
    legs = [c for c in axs[0].get_children() if isinstance(c, matplotlib.legend.Legend)]
    assert len(legs) == 3, "the spot without r_BAO (no echoes)"
    texts = [[t.get_text() for t in g.get_texts()] for g in legs]
    assert ["measured", lab] in texts and not any("echo" in t for g in texts for t in g)
    sep = pl.fold_periodic(100.0 / u, L)
    spot = [ln for ln in axs[0].lines if ln.get_linestyle() == "-" and np.allclose(ln.get_xdata(), sep)]
    assert len(spot) == 1, "the spot of the two lines, without r_BAO"
    ref_err = dict(P1D_DESI_DR1_Z22, kP_pi_err=0.1 * P1D_DESI_DR1_Z22["kP_pi"])
    fig, ax = pl.plot_p1d_reference(run["p1d"], ref_err, kms_per_cMpc=u, each_label=lab)
    assert lab in [t.get_text() for t in ax.get_legend().get_texts()]
    assert any(c.get_label() == ref_err["label"] for c in ax.containers), "the reference with its error bars"
    nmap = np.exp(np.random.default_rng(0).normal(size=(64, 64)))
    fig, ax, info = pl.plot_skewer_slice(nmap, L, sk["columns"], highlight=(ix, iz), spacing=b.dx)  # node indices x dx
    assert info["spacing_cMpc"] == pytest.approx(4 * b.dx) and np.allclose(ax.get_xlim(), (0.0, L))
    with pytest.raises(ValueError, match="spacing"):
        pl.plot_skewer_slice(nmap, L, sk["columns"])
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
        else:  # the separation folded into half the period, solid in the pair's colour, on every row
            spot = pl.line_pairs(("HI-LyA", "SiIII-1206"), dv, period)[0]
            assert all(
                any(np.allclose(ln.get_xdata(), spot["fold"]) and ln.get_color() == spot["color"] for ln in ax.lines)
                for ax in axs
            )
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
    """comoving_axis: v = aH x and k_v = k_x / (aH) on the secondary axes (notebook 10, Part D)."""
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
    pl.add_panel_letters(ax2, labels=["a1", "a2"], box=False)
    assert [t.get_text() for a in ax2 for t in a.texts] == ["(a1)", "(a2)"]


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


def test_n_line_figures():
    """Any number of lines (the default four, ``simulations.LINE_IDS``): one style per line from the registry
    (``line_style``: distinct colours and markers, the same in every figure), names from the line table with the
    wavelength when a species repeats (``line_names``), every pair (i, j) with its folded separation and label
    (``line_pairs``: "all", "reference", a list); the (r_perp, r_par) map with a diamond per pair and the echoes of the
    selected pairs, the wedges with a spot per pair, the decomposition by term readable for N = 4 (``max_terms``,
    ``terms``, the numbering of the lines)."""
    from skewerjax.core import LineBundle
    from skewerjax.simulations import LINE_IDS, ToyBoxes

    from helpers import toy_lattice_statistics

    ids = list(LINE_IDS)
    assert ids == ["HI-LyA", "SiIII-1206", "SiII-1193", "SiII-1190"]
    st = [pl.line_style(i, j) for j, i in enumerate(ids)]
    assert len({s["color"] for s in st}) == 4 and len({s["marker"] for s in st}) == 4 and len({str(s["ls"]) for s in st}) == 4
    assert pl.line_style("HI-LyA")["color"] == pl.ENTITY_COLORS["HI-LyA"] == pl.line_style("HI")["color"]
    assert pl.line_style("total")["color"] == pl.FG and pl.line_style("CIV-1548", 5)["marker"] == pl._MARKER_CYCLE[5]
    assert pl.line_names(ids) == ["Lyα", "Si III", "Si II 1193", "Si II 1190"]
    assert pl.line_names(["HI-LyA", "SiII-1193"]) == ["Lyα", "Si II"], "the wavelength only when the species repeats"
    assert pl.field_names(["HI", "SiII", "gas", "total"]) == {"HI": "H I", "SiII": "Si II", "gas": "gas", "total": "total"}
    lb = LineBundle(ids, ref_line_id=ids[0])
    dv = np.asarray(lb.delta_v)
    period = 3961.5
    allp = pl.line_pairs(ids, dv, period)
    assert [(p["i"], p["j"]) for p in allp] == [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    assert all(0 <= p["fold"] <= period / 2 for p in allp) and len({p["label"] for p in allp}) == 6
    assert allp[3]["dv"] == pytest.approx(abs(dv[2] - dv[1])) and allp[3]["fold"] == pytest.approx(
        pl.fold_periodic(dv[2] - dv[1], period)
    )
    assert [(p["i"], p["j"]) for p in pl.line_pairs(ids, dv, period, "reference")] == [(0, 1), (0, 2), (0, 3)]
    assert [(p["i"], p["j"]) for p in pl.line_pairs(ids, dv, period, [("SiII-1190", "SiII-1193")])] == [(2, 3)]
    assert pl.line_pairs(ids, dv, period, [(2, 3)])[0]["color"] == allp[5]["color"], "a pair keeps its colour in any selection"
    assert pl.line_pairs(ids[1:3], dv[1:3], period)[0]["color"] == allp[3]["color"], "and in any bundle"
    with pytest.raises(ValueError, match="pairs"):
        pl.line_pairs(ids, dv, period, "some")

    b = ToyBoxes(box_cMpc=8.0, n_cells=16, n_real=2, bao_ratio=1e-3)
    run = toy_lattice_statistics(b, range(b.n_real), (4, 4), 40, 64, decompose=True)
    period = float(run["period"])
    u, L = period / b.box_cMpc, b.box_cMpc
    tot = dict(run["flux3d"]["total"], axes=run["axes3d"])
    for pairs, n_echo in (("reference", 3), ("all", 6)):
        fig, ax, drawn = pl.plot_overflux_rpar_rperp(tot, kms_per_cMpc=u, r_bao=2.0, delta_v=dv, pairs=pairs)
        fig.canvas.draw()
        diamonds = [ln for ln in ax.lines if ln.get_marker() == "D"]
        assert len(diamonds) == 6 and len(drawn["spots"]) == 6, "a diamond for every pair"
        np.testing.assert_allclose(
            sorted(d["r_par"][0] for d in drawn["spots"]), sorted(u * pl.fold_periodic(p["dv"] / u, L) for p in allp), rtol=1e-9
        )
        assert len(drawn["echoes"]) == n_echo, "the echoes of the pairs asked for"
        leg = ax.get_legend()
        assert [t.get_text() for t in leg.get_texts()] == [p["text"] for p in allp], "every diamond named, no echo entry"
        assert [t.get_color() for t in leg.get_texts()] == [p["color"] for p in allp], "the names in the pairs' colours"
        assert leg._ncols == 1 and leg._loc == matplotlib.legend.Legend.codes["upper right"], "one column, top right"
        bb, la = leg.get_window_extent(), ax.get_window_extent()
        assert la.x0 <= bb.x0 and bb.x1 <= la.x1 and la.y0 <= bb.y0 and bb.y1 <= la.y1, "inside the panel"
        for d, p in zip(diamonds, allp):
            assert d.get_markeredgecolor() == p["color"], "the diamond in the pair's colour"
        plt.close(fig)
    assert len({p["color"] for p in allp}) == 6, "one colour per pair"
    fig, ax, drawn = pl.plot_overflux_rpar_rperp(tot, kms_per_cMpc=u, r_bao=2.0, delta_v=dv, pairs="all")
    fig.canvas.draw()
    for pr in pl.line_pairs(ids, dv / u, L):  # every pair, metal-metal included, folded with the period, in its colour
        e = drawn["echoes"][pr["text"]]
        np.testing.assert_allclose(e["plus"], u * pl.fold_periodic(np.abs(np.sqrt(2.0**2 - e["r_perp"] ** 2) + pr["dv"]), L))
        curves = [ln for ln in ax.lines if ln.get_color() == pr["color"] and len(ln.get_xdata()) == len(e["r_perp"])]
        assert [(ln.get_linestyle(), ln.get_linewidth()) for ln in curves] == [("--", 1.4), (":", 1.4)], pr["text"]
    plt.close(fig)
    fig, axs, _ = pl.plot_xi_mu_wedges(tot, kms_per_cMpc=u, r_bao=2.0, delta_v=dv, box=L, line_ids=ids)
    fig.canvas.draw()
    leg = [
        g
        for g in axs[0].get_children()
        if isinstance(g, matplotlib.legend.Legend) and "line-of-sight" in g.get_title().get_text()
    ]
    assert len(leg) == 1 and len(leg[0].get_texts()) == 6 + 3, "one colour per pair and the three styles"
    assert [t.get_color() for t in leg[0].get_texts()[:6]] == [p["color"] for p in allp], "the names in the pairs' colours"
    assert all(ln.get_color() != pl.DIM for ln in leg[0].get_lines()), "dotted grey is the BAO's alone"
    sc = pl.los_scales(2.0, ids, dv / u, L)
    lo, hi = axs[0].get_xlim()
    vertical = {round(float(ln.get_xdata()[0]), 9) for ln in axs[0].lines if len(set(ln.get_xdata())) == 1}
    for pr in sc["pairs"]:
        for key in ("spot", "plus", "minus"):
            assert 0 <= pr[key] <= L / 2
            assert not lo <= pr[key] <= hi or round(pr[key], 9) in vertical, (pr["text"], key)
    plt.close(fig)
    fig, axs = pl.plot_overflux_3d(run["flux3d"], run["axes3d"], kms_per_cMpc=u)
    labels = [t.get_text() for t in axs["P"].get_legend().get_texts()]
    assert [x for x in labels if not x.startswith("each")][:5] == ["Lyα", "Si III", "Si II 1193", "Si II 1190", "total"], labels
    markers = [c.lines[0].get_marker() for c in axs["P"].containers][:5]
    assert markers == [pl.line_style(i)["marker"] for i in ids] + ["*"], "the registry's markers"
    plt.close(fig)
    ens = run["flux"]
    assert ens["P_order"].shape[0] == 9, "orders 0 .. 8 for four lines"
    fig, axs = pl.plot_p1d_orders(ens, kms_per_cMpc=u)
    plt.close(fig)
    fig, axs = pl.plot_xi1d_terms(ens, kms_per_cMpc=u, max_terms=4, line_ids=ids, delta_v_list=dv)
    for ax in axs:
        names = [t.get_text() for t in ax.get_legend().get_texts()]
        assert sum(not (n.startswith("other") or "total" in n) for n in names) <= 4
    assert axs[0].get_legend().get_title().get_text() == "", "no legend title that names the lines"
    o2 = [t.get_text() for t in axs[1].get_legend().get_texts()]
    assert any(n.startswith("other terms") for n in o2), "the rest of order 2 summed"
    named = {t.get_text(): t for t in axs[0].texts}
    for p in pl.line_pairs(ids, dv, period):
        assert p["text"] in named and "Delta" not in p["text"], "every pair named, without Delta v"
        assert named[p["text"]].get_color() == p["color"] and named[p["text"]].get_fontsize() > pl.FONT_MIN["annotation"]
        assert [ln for ln in axs[1].lines if np.allclose(ln.get_xdata(), p["fold"]) and ln.get_color() == p["color"]], (
            "the spot, solid"
        )
    plt.close(fig)
    fig, axs = pl.plot_p1d_terms(ens, kms_per_cMpc=u, line_ids=ids, max_terms=4)
    assert axs[0].get_legend().get_title().get_text().startswith("lines: 1 = Lyα"), "the numbering stays in P1D"
    plt.close(fig)
    # quantity and y_scale of xi1D: r^2 xi with r = lag / u
    fig, axs = pl.plot_xi1d_orders(ens, kms_per_cMpc=u)
    plain = [c for c in axs[0].lines if c.get_marker() in pl.ORDER_MARKERS][0]
    x0, y0 = plain.get_xdata().copy(), plain.get_ydata().copy()
    plt.close(fig)
    fig, axs = pl.plot_xi1d_orders(ens, kms_per_cMpc=u, delta_v_list=dv, line_ids=ids, quantity="r2xi", y_scale="symlog")
    fig.canvas.draw()
    assert axs[0].get_yscale() == "symlog" and "r^2" in axs[0].get_ylabel() and "cMpc$^2$" in axs[0].get_ylabel()
    weighted = [c for c in axs[0].lines if c.get_marker() in pl.ORDER_MARKERS][0]
    np.testing.assert_allclose(weighted.get_ydata(), (x0 / u) ** 2 * y0, atol=1e-14)
    assert {t.get_text() for t in axs[0].texts} >= {pl.line_pairs(ids, dv, period)[0]["text"]}
    plt.close(fig)
    fig, axs = pl.plot_xi1d_terms(ens, kms_per_cMpc=u, max_terms=2, quantity="r2xi", y_scale="symlog")
    fig.canvas.draw()
    assert all(ax.get_yscale() == "symlog" and "r^2" in ax.get_ylabel() for ax in axs)
    plt.close(fig)
    with pytest.raises(ValueError, match="quantity"):
        pl.plot_xi1d_orders(ens, quantity="r3")
    with pytest.raises(ValueError, match="y_scale"):
        pl.plot_xi1d_orders(ens, y_scale="log2")
    # the BAO maxima: in the BAO style, in the drawn range only, one legend entry
    k_f = 2 * np.pi / L
    bao = np.array([0.5 * k_f, 3.0 * k_f, 5.0 * k_f, 1e3 * k_f])
    fig, axs = pl.plot_overflux_3d(
        run["flux3d"], run["axes3d"], kms_per_cMpc=u, bao_k=bao, limits=dict(k_min=k_f, k_max=6 * k_f, r_min=1.0, r_max=L / 2)
    )
    thin = [ln for ln in axs["P"].lines if ln.get_linewidth() == pl.SCALE_STYLES["bao"]["lw"] and ln.get_linestyle() == ":"]
    np.testing.assert_allclose(sorted(float(ln.get_xdata()[0]) for ln in thin), [3.0 * k_f, 5.0 * k_f])
    assert all(ln.get_color() == pl.DIM for ln in thin)
    assert [t.get_text() for t in axs["P"].get_legend().get_texts()].count("BAO maxima") == 1
    plt.close(fig)
    # every redshift of the reference alike, joined by a thin line, one colour key
    from skewerjax.utils.reference_data import P1D_DESI_DR1, nearest_reference

    assert nearest_reference(P1D_DESI_DR1, 3.008)["z"] == 3.0 and nearest_reference(P1D_DESI_DR1, 2.2)["z"] == 2.2
    fig, ax = pl.plot_p1d_reference(run["p1d"], P1D_DESI_DR1, kms_per_cMpc=u, k_max=0.042, bao_k=np.array([0.01, 0.02, 1.0]))
    fig.canvas.draw()
    assert len(fig.axes) == 2 and "DESI DR1" in fig.axes[1].get_ylabel(), "a colour key naming the reference once"
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert not any("Chaves" in x or "nearest" in x for x in labels), labels
    assert labels.count("BAO maxima (3D, reference)") == 1
    small = [ln for ln in ax.lines if ln.get_marker() == "s" and ln.get_linewidth() < 1.0]  # (Si III's points are squares too)
    assert len(small) == len(P1D_DESI_DR1) and len({str(ln.get_color()) for ln in small}) == len(P1D_DESI_DR1), "a colour per z"
    assert len({(ln.get_markersize(), ln.get_linewidth(), ln.get_linestyle()) for ln in small}) == 1, "every z alike"
    assert small[0].get_linestyle() == "-" and small[0].get_linewidth() < 1.0, "joined by a thin line"
    assert not ax.containers[len(run["p1d"]["p1d"]) :], "no error bars of one z"
    assert (
        len([ln for ln in ax.lines if ln.get_linestyle() == ":" and ln.get_linewidth() == pl.SCALE_STYLES["bao"]["lw"]]) == 2
    ), "inside the axis only"
    with pytest.raises(TypeError):
        pl.plot_p1d_reference(run["p1d"], P1D_DESI_DR1, z=3.0)
    plt.close(fig)
    fig, axs = pl.plot_p1d_terms(ens, kms_per_cMpc=u, orders=[2], terms=["1–2", "1–3"])
    shown = [t.get_text() for t in axs[0].get_legend().get_texts()]
    assert [x for x in shown if "–" in x] == ["1–2", "1–3"] and any(x.startswith("other") for x in shown), shown
    plt.close(fig)
    fig, axs, info = pl.plot_toy_skewer(
        L,
        {sp: np.ones(16) * (j + 1) for j, sp in enumerate(b.species)},
        run["v"],
        run["tau"][0][:, 0],
        mean_transmission=run["mean_transmission"],
    )
    assert [t.get_text() for t in axs[2].get_legend().get_texts()][:2] == ["Lyα", "Si III (on the Lyα axis)"]
    assert len(axs[2].lines) >= 5, "four lines and the total"
    plt.close(fig)


def test_bao_images_in_a_periodic_box():
    """xi_box(r) = sum_n xi(|r + n L|): the ring and its echoes appear as the rings about (0, n L) and (0, n L +- Delta)
    (n_perp = 0). ``echo_curves`` (any r_BAO, also > L / 2): every point of the folded curves lies on one of those image
    rings; ``los_scales``: on the line of sight the ring is at |r_BAO - n L| and the echoes at |r_BAO +- Delta - n L|,
    folded into [0, L / 2]."""
    L, r0, d = 51.7, 147.0, 21.0  # TNG50-4-like: r_BAO ~ 2.8 L
    c = pl.echo_curves(r0, d, L)
    assert np.all(c["ring"] <= L / 2 + 1e-12) and np.all(c["plus"] <= L / 2 + 1e-12)
    n = np.arange(-8, 9)
    for key, shift in (("ring", 0.0), ("plus", d), ("minus", d)):
        cen = np.concatenate([n * L + shift, -(n * L + shift), n * L - shift, -(n * L - shift)])
        resid = np.min(np.abs(c["r_perp"][:, None] ** 2 + (c[key][:, None] - cen[None]) ** 2 - r0**2), axis=1)
        assert np.max(resid) < 1e-6 * r0**2, key
    sc = pl.los_scales(r0, ["HI-LyA", "SiIII-1206"], np.array([0.0, d]), L)
    fold = lambda x: float(pl.fold_periodic(x, L))  # noqa: E731
    pr = sc["pairs"][0]
    assert sc["ring"] == pytest.approx(np.min(np.abs(r0 - n * L))) and sc["ring"] <= L / 2
    assert (
        pr["spot"] == pytest.approx(d)
        and pr["plus"] == pytest.approx(fold(r0 + d))
        and pr["minus"] == pytest.approx(fold(r0 - d))
    )
    assert pr["plus"] == pytest.approx(np.min(np.abs(r0 + d - n * L))), "the nearest image of the echo"
    far = pl.los_scales(2.0, ["HI-LyA", "SiIII-1206"], np.array([0.0, 7.0]), 8.0)["pairs"][0]  # Delta > L / 2
    assert far["spot"] == pytest.approx(1.0) and far["plus"] == pytest.approx(1.0) and far["minus"] == pytest.approx(3.0)
    none = pl.los_scales(None, ["HI-LyA", "SiIII-1206"], np.array([0.0, 7.0]), 8.0)
    assert none["ring"] is None and none["pairs"][0]["plus"] is None


def test_the_styles_of_the_characteristic_scales():
    """One registry: the BAO grey, dotted and broad; a pair of lines thinner, in one colour in every figure, spot solid, echo +
    dashed, echo - dotted; nothing else dotted grey, and no pair grey."""
    bao = pl.scale_style("bao")
    r, g, b = matplotlib.colors.to_rgb(bao["color"])
    assert r == g == b and bao["ls"] == ":" and bao["lw"] == 2.2, "grey, dotted, broad"
    assert all(pl.SCALE_STYLES[k]["lw"] < bao["lw"] for k in ("spot", "plus", "minus")), "the pairs thinner"
    assert [pl.scale_style(k, "#123456")["ls"] for k in ("spot", "plus", "minus")] == ["-", "--", ":"]
    assert pl.scale_style("spot", "#123456")["color"] == "#123456"
    with pytest.raises(ValueError, match="kind"):
        pl.scale_style("ring")
    assert pl.PALETTE["grey"] not in {pl.PALETTE[c] for c in pl.PAIR_COLORS.values()} and "grey" not in pl.PAIR_COLOR_ORDER
    assert len(set(pl.PAIR_COLORS.values())) == len(pl.PAIR_COLORS), "one colour per pair"
    assert pl.pair_color(("HI-LyA", "SiIII-1206")) == pl.pair_color(("SiIII-1206", "HI-LyA")), "either order"
    ax = plt.figure().add_subplot()
    ax.set_xlim(0, 3)
    assert pl._mark_lines(ax, {"$a$": 1.0}) == {"$a$": 1.0}
    assert ax.lines[0].get_linestyle() != ":", "the other marks are not dotted"
    assert pl._bao_mark(ax, 8.0, 6.0) == {"$|r_{\\rm BAO} - nL|$": 2.0}, "a scale beyond L / 2 by its folded image"
    assert ax.lines[1].get_linestyle() == ":" and ax.lines[1].get_color() == bao["color"]
    plt.close(ax.figure)


def test_the_symlog_colour_limits_are_symmetric():
    """color_scale="symlog": symmetric limits +- max |data|, 0 at the centre of the colours, whichever side is the larger."""
    for values, top in ((np.array([-3.0, -0.01, 0.0, 0.5, 40.0]), 40.0), (np.array([-40.0, 0.0, 3.0]), 40.0)):
        m, levels, extend = pl._map_scale(values, color_scale="symlog")
        assert isinstance(m, matplotlib.colors.SymLogNorm) and (m.vmin, m.vmax) == (-top, top) and extend == "neither"
        assert m(0.0) == pytest.approx(0.5) and m.linthresh == pytest.approx(0.01 * top)
        assert np.array_equal(levels, pl._map_scale(values)[1]), "the contours do not depend on the colour scale"
    assert pl._map_scale(np.array([1.0, 2.0]), color_scale="symlog", linthresh=0.5)[0].linthresh == 0.5
    linear = pl._map_scale(np.array([-3.0, 0.5, 40.0]))[0]
    assert (linear.vmin, linear.vmax) == (-40.0, 40.0) and not isinstance(linear, matplotlib.colors.SymLogNorm)
    for gone in ("TwoSidedSymLogNorm", "_map_colorbar", "_extend_of"):
        assert not hasattr(pl, gone), gone


def _map_inputs(n_cells=16, L=8.0):
    from skewerjax.summary_stats import bao_k_edges, box_grid, field_statistics, grid_limits

    b = ToyBoxes(box_cMpc=L, n_cells=n_cells, n_real=2, r_bao=2.5, sigma_bao=0.5, bao_ratio=1e-4, n_s=-0.4)
    lim = grid_limits(n_cells, L)
    k = bao_k_edges(lim["k_min"], lim["k_max"], lim["k_max"], k_1=None, dk_factor=1.0)
    r = np.arange(0, L / 2 + L / n_cells, L / n_cells)
    grid = box_grid(n_cells, L, k_edges=k, r_edges=r, log=None)
    return b, field_statistics((b.cells(i) for i in range(2)), grid, species="HI", power=b.grf().power, log=None)


def _physical_ratio(ax):
    ax.figure.canvas.draw()
    bb = ax.get_window_extent()
    return bb.height / bb.width


def test_the_maps_keep_the_physical_aspect_and_the_theme_colormaps():
    """Equal comoving spans at equal lengths, not deformed by the colour bar (as tall as the panel), and the theme's colour maps
    by default in every map."""
    b, res = _map_inputs()
    u = 100.0
    a = res["axes"]
    figs, axs, _ = pl.plot_rpar_rperp_maps(res, r_bao=2.5, quantity="xi", color_scale="symlog", kms_per_cMpc=u)
    for fig, ax in zip(figs, axs):
        ratio = (a["rpar_edges"][-1] - a["rpar_edges"][0]) / (a["rperp_edges"][-1] - a["rperp_edges"][0])
        assert ax.get_box_aspect() == pytest.approx(ratio) and _physical_ratio(ax) == pytest.approx(ratio, rel=1e-3)
        cax = fig.axes[1]
        assert cax.get_position().height == pytest.approx(ax.get_position().height), "the colour bar as tall as the panel"
        assert ax.collections[0].get_cmap().name == pl.CMAPS["diverging"]
    ring = [ln for ln in axs[1].lines if ln.get_linestyle() == ":"]
    assert ring and all(ln.get_color() == pl.scale_style("bao")["color"] for ln in ring), "the BAO circle in the BAO style"
    norm = axs[1].collections[0].norm
    expected = res["theory"]["xi_map"][np.isfinite(res["xi_map"]) & np.isfinite(res["theory"]["xi_map"])]
    assert norm.vmax == pytest.approx(np.abs(expected).max()) and norm.vmin == -norm.vmax, "symmetric, from the expectation"
    assert norm(0.0) == pytest.approx(0.5)
    for f in figs:
        plt.close(f)
    # r_par in km/s: the spans are compared in comoving units, through u
    tot = dict(res, axes=a)
    fig, ax, _ = pl.plot_overflux_rpar_rperp(tot, kms_per_cMpc=u, r_bao=2.5, delta_v=100.0)
    ymin, ymax, xmin, xmax = *ax.get_ylim(), *ax.get_xlim()
    assert _physical_ratio(ax) == pytest.approx(((ymax - ymin) / u) / (xmax - xmin), rel=1e-3), "r_par through u"
    plt.close(fig)
    figs, axs, _ = pl.plot_kpar_kperp_maps(res, kpar_scale="linear", kperp_scale="linear", kms_per_cMpc=u)
    ymin, ymax, xmin, xmax = *axs[1].get_ylim(), *axs[1].get_xlim()
    assert _physical_ratio(axs[1]) == pytest.approx((ymax - ymin) / (xmax - xmin), rel=1e-3)
    assert axs[1].collections[0].get_cmap().name == pl.CMAPS["diverging"]
    for f in figs:
        plt.close(f)
    figs, axs, _ = pl.plot_kpar_kperp_maps(res, kms_per_cMpc=u)
    assert _physical_ratio(axs[1]) == pytest.approx(1.0, rel=1e-3), "a logarithmic map: a square"
    for f in figs:
        plt.close(f)
    fl = b.fields(0)
    fig, ax, _ = pl.plot_skewer_slice(fl, 8.0, np.array([[0, 0], [8, 0]]))
    assert _physical_ratio(ax) == pytest.approx(1.0, rel=1e-3) and fig.axes[1].get_position().height == pytest.approx(
        ax.get_position().height
    )
    assert ax.images[0].get_cmap().name == pl.CMAPS["density"]
    plt.close(fig)
    # no default colour map outside the theme, and the dark theme follows
    import inspect

    for name, fn in inspect.getmembers(pl, inspect.isfunction):
        for key, par in inspect.signature(fn).parameters.items():
            if "cmap" in key:
                assert par.default is None or str(par.default).startswith("tol_"), (name, key, par.default)
    try:
        pl.use_style("dark")
        fig, ax, _ = pl.plot_skewer_slice(fl, 8.0, np.array([[0, 0], [8, 0]]))
        assert ax.images[0].get_cmap().name == pl.CMAPS["density"] == "tol_rainbow_WhBr_r", "following use_style"
        plt.close(fig)
    finally:
        pl.use_style()


def test_plot_toy_fields_true_squares_letters_and_the_arrow_key():
    b, _ = _map_inputs()
    fl = b.fields(0)
    fig, axs, info = pl.plot_toy_fields(fl, 8.0, arrow_every=4)
    pl.add_panel_letters(axs, labels="grid")
    fig.canvas.draw()
    assert axs.shape == (2, 3)
    sizes = {(round(ax.get_window_extent().width, 3), round(ax.get_window_extent().height, 3)) for ax in axs.ravel()}
    assert len(sizes) == 1 and next(iter(sizes))[0] == pytest.approx(next(iter(sizes))[1]), "true squares of one size"
    letters = [t.get_text() for ax in axs.ravel() for t in ax.texts if t.get_text().startswith("(")]
    assert letters == ["(a1)", "(a2)", "(a3)", "(b1)", "(b2)", "(b3)"], "the row's letter, the column's number"
    assert pl.check_fonts(fig) == []
    plt.close(fig)
    # with velocities: the key in a box inside its panel
    v = np.zeros_like(np.asarray(fl["v"]))
    v[..., 1] = 100.0 * np.sin(np.arange(16) * 2 * np.pi / 16)[None, :, None]
    fig, axs, info = pl.plot_toy_fields(dict(fl, v=v), 8.0, arrow_every=4)
    fig.canvas.draw()
    assert info["key_kms"] > 0
    for ax in axs[1]:
        box = [p for p in ax.patches if isinstance(p, matplotlib.patches.Rectangle) and p.get_zorder() == 12]
        assert len(box) == 1
        x0, y0, w, h = box[0].get_x(), box[0].get_y(), box[0].get_width(), box[0].get_height()
        assert 0 < x0 and x0 + w < 1 and 0 < y0 and y0 + h < 1, "inside the panel"
        key = [t for t in ax.texts if "km/s" in t.get_text()][0]
        tb = key.get_window_extent().transformed(ax.transAxes.inverted())
        assert x0 <= tb.x0 and tb.x1 <= x0 + w and y0 <= tb.y0 and tb.y1 <= y0 + h, "the speed inside its box"
    plt.close(fig)


def test_the_bao_and_the_imprints_in_the_figures():
    from skewerjax.summary_stats import pair_imprint

    b, res = _map_inputs()
    bao = np.array([1.0, 2.0])
    fig, axs, _ = pl.plot_isotropic_recovery(res, r_bao=2.5, bao_k=bao, k_scale="linear", k_marks={"$k_2$": 3.0})
    dotted = [ln for ln in axs[0, 0].lines if ln.get_linestyle() == ":"]
    assert len(dotted) == 2 and all(ln.get_linewidth() == 2.2 and ln.get_color() == pl.DIM for ln in dotted), "BAO maxima in (a)"
    assert len([ln for ln in axs[1, 0].lines if ln.get_linestyle() == ":"]) == 2, "and in its residuals"
    assert not any("BAO" in t.get_text() for t in axs[0, 0].get_legend().get_texts()), "no legend entry"
    for ax in axs[:, 1]:
        ring = [ln for ln in ax.lines if ln.get_linestyle() == ":" and ln.get_linewidth() == 2.2]
        assert len(ring) == 1 and ring[0].get_xdata()[0] == pytest.approx(2.5), "r_BAO in (b) and its residuals"
    assert any(ln.get_linestyle() == "-." for ln in axs[0, 0].lines), "the other marks, not dotted"
    plt.close(fig)
    fig, axs, _ = pl.plot_isotropic_recovery(
        dict(res, theory=None) if False else {k: v for k, v in res.items() if k != "theory"}, r_bao=13.0
    )
    assert axs[0, 1].lines[-1].get_xdata()[0] == pytest.approx(3.0), "beyond L / 2: its folded image"
    plt.close(fig)
    # the isotropic imprint in plot_overflux_3d: the maxima of j_0(k Delta_ij), solid, with a key in the pairs' colours
    ids = ("HI-LyA", "SiIII-1206", "SiII-1193")
    dv = np.array([0.0, 250.0, 330.0])
    u, L = 100.0, 8.0
    flux = {f: res for f in (*ids, "total")}
    fig, axs3 = pl.plot_overflux_3d(flux, res["axes"], kms_per_cMpc=u, delta_v=dv, line_ids=ids, box=L, bao_k=bao)
    lo, hi = axs3["P"].get_xlim()
    prs = pl.line_pairs(ids, dv / u, L)
    for pr in prs:
        lines = [ln for ln in axs3["P"].lines if ln.get_color() == pr["color"] and len(ln.get_xdata()) == 2]
        want = pair_imprint(pr["fold"], hi, kind="iso", k_min=lo)
        np.testing.assert_allclose(sorted(ln.get_xdata()[0] for ln in lines), want)
        assert all(ln.get_linestyle() == "-" and ln.get_linewidth() == 1.4 for ln in lines), "solid and thin"
        spot = [ln for ln in axs3["xi"].lines if ln.get_color() == pr["color"] and len(ln.get_xdata()) == 2]
        assert len(spot) == 1 and spot[0].get_xdata()[0] == pytest.approx(pr["fold"]) and spot[0].get_linestyle() == "-"
    key = axs3["P"].get_legend()
    assert [t.get_text() for t in key.get_texts()] == [p["text"] for p in prs], "the key names the pairs, no Delta v"
    assert [t.get_color() for t in key.get_texts()] == [p["color"] for p in prs]
    plt.close(fig)
    fig, axs3 = pl.plot_overflux_3d(flux, res["axes"], kms_per_cMpc=u, delta_v=dv, line_ids=ids, box=L, n_harmonics=1)
    assert len([ln for ln in axs3["P"].lines if ln.get_color() == prs[0]["color"] and len(ln.get_xdata()) == 2]) == 1, "the cap"
    plt.close(fig)


def test_the_imprints_of_the_pairs_on_the_p1d_figures():
    """k_par = 2 pi n / Delta v_fold, solid in the pair's colour, within the drawn range; ``n_harmonics`` the cap; a key below the
    panels names the pairs in their colours, at a larger size, without "Delta v"."""
    from skewerjax.simulations import LINE_IDS
    from skewerjax.summary_stats import pair_imprint
    from skewerjax.utils.reference_data import P1D_DESI_DR1

    from helpers import toy_lattice_statistics

    b = ToyBoxes(box_cMpc=8.0, n_cells=16, n_real=2, bao_ratio=1e-3)
    run = toy_lattice_statistics(b, range(b.n_real), (4, 4), 40, 64, decompose=True)
    period, ens = float(run["period"]), run["flux"]
    ids = list(LINE_IDS)
    dv = np.asarray(LineBundle(ids, ref_line_id=ids[0]).delta_v)
    prs = pl.line_pairs(ids, dv, period)
    prs = [p for p in prs if p["fold"] > 0]
    u = period / b.box_cMpc
    k_lim = (2 * np.pi / period, np.pi / (period / 64))
    fig, ax = pl.plot_p1d_reference(run["p1d"], P1D_DESI_DR1, kms_per_cMpc=u, delta_v=dv, line_ids=ids, k_max=0.5)
    fig.canvas.draw()
    lo, hi = ax.get_xlim()
    for p in prs:
        lines = [ln for ln in ax.lines if ln.get_color() == p["color"] and ln.get_linewidth() == 1.4]
        np.testing.assert_allclose(sorted(ln.get_xdata()[0] for ln in lines), pair_imprint(p["fold"], hi, k_min=lo))
        assert all(ln.get_linestyle() == "-" for ln in lines), "solid"

    def key_of(axis):
        return next(
            g
            for g in axis.get_children()
            if isinstance(g, matplotlib.legend.Legend) and g.get_texts()[0].get_text() == prs[0]["text"]
        )

    key = key_of(ax)
    assert [t.get_text() for t in key.get_texts()] == [p["text"] for p in prs] and all("Delta" not in p["text"] for p in prs)
    assert [t.get_color() for t in key.get_texts()] == [p["color"] for p in prs], "the names in the pairs' colours"
    assert key.get_texts()[0].get_fontsize() > pl.FONT_MIN["annotation"], "larger labels"
    assert key.get_window_extent().y1 < ax.get_tightbbox().y0 + key.get_window_extent().height + 1, "below the axis"
    plt.close(fig)
    fig, axs = pl.plot_p1d_orders(
        ens, kms_per_cMpc=u, k_limits=k_lim, k_scale="linear", delta_v_list=dv, line_ids=ids, n_harmonics=2
    )
    fig.canvas.draw()
    for a in axs:
        for p in prs:
            lines = [ln for ln in a.lines if ln.get_color() == p["color"] and ln.get_linewidth() == 1.4]
            assert len(lines) <= 2, "the cap"
            np.testing.assert_allclose(
                sorted(ln.get_xdata()[0] for ln in lines),
                pair_imprint(p["fold"], a.get_xlim()[1], k_min=a.get_xlim()[0], n_harmonics=2),
            )
    assert key_of(axs[-1]) is not None
    plt.close(fig)
    fig, axs = pl.plot_p1d_terms(
        ens, kms_per_cMpc=u, k_limits=k_lim, k_scale="linear", delta_v_list=dv, line_ids=ids, n_harmonics=1, orders=[2, 3]
    )
    assert all(len([ln for ln in a.lines if ln.get_linewidth() == 1.4]) <= len(prs) for a in axs) and key_of(axs[-1]) is not None
    plt.close(fig)


def _code_curves(nv=400):
    """The arrays of ``plot_code_comparison``'s curves on one skewer: a reference, and the same 1 % higher in n and tau."""
    v, y = np.arange(0.5, nv + 0.5), np.arange(0.5, nv + 0.5) / 80.0  # (aH) = 80 km/s per cMpc
    n, tau = 1e-10 * (2.0 + np.sin(v / 9)), 0.5 + 0.4 * np.cos(v / 17)
    ref = dict(v=v, y=y, n=n, tau=tau, F=np.exp(-tau))
    other = dict(v=v, y=y, n=n * 1.01, tau=tau * 1.01, F=np.exp(-tau * 1.01))
    return ref, other


def test_code_residual_definitions():
    """n and tau: the symmetric relative difference, bounded by 2, NaN where neither has gas; F: the absolute difference."""
    a, b = np.array([3.0, 1.0, 0.0, 0.0]), np.array([1.0, 3.0, 2.0, 0.0])
    for q in ("n", "tau"):
        r = pl.code_residual(q, a, b)
        np.testing.assert_allclose(r[:3], [1.0, -1.0, -2.0], rtol=1e-14)
        assert np.isnan(r[3])
        assert np.nanmax(np.abs(r)) <= 2.0
    np.testing.assert_array_equal(pl.code_residual("F", a, b), a - b)
    with pytest.raises(ValueError):
        pl.code_residual("T", a, b)
    with pytest.raises(ValueError):
        pl.code_residual("F", a, b[:2])


def test_code_modes_and_switch_colors():
    """A fixed style per mode: only fake_spectra dashed, the default solid and blue, every colour and style distinct and from
    the palette; 17 distinct colours for the switches, none of them a colour of the modes they are drawn against."""
    m = pl.CODE_MODES
    assert set(m) == {"default", "fake_spectra", "matched", "default+switch", "matched-switch"}
    assert m["default"]["ls"] == "-" and m["default"]["color"] == pl.PALETTE["blue"]
    assert m["fake_spectra"]["ls"] == "--" and m["fake_spectra"]["color"] != pl.PALETTE["blue"]
    assert [k for k, s in m.items() if s["ls"] == "--"] == ["fake_spectra"]
    assert len({str(s["ls"]) for s in m.values()}) == 5 and len({s["color"] for s in m.values()}) == 5
    assert all(s["color"] in pl.PALETTE.values() for s in m.values())
    colours = list(pl.SWITCH_COLORS.values())
    assert len(colours) == 17 == len(set(colours))
    assert not set(colours) & {m[k]["color"] for k in ("default", "fake_spectra", "matched")}


def test_plot_code_comparison_rows_and_axes():
    """Any subset of the rows, always in order, the residual rows shorter; n against y [cMpc], tau and F against v [km/s], on the
    same horizontal extent through y = v / aH; fixed residual limits, unless changed; the window; no residual rows without residuals."""
    ref, other = _code_curves()
    curves = [dict(mode="fake_spectra", **ref), dict(mode="default", **other)]
    pair = [("default", "fake_spectra")]
    fig, axs = pl.plot_code_comparison(curves, residuals=pair, aH=80.0, title="t")
    assert len(axs) == 6 and not pl.check_fonts(fig)
    assert [a.get_yscale() for a in axs] == ["log", "symlog", "log", "symlog", "linear", "symlog"]
    assert axs[1].get_position().height < axs[0].get_position().height
    assert axs[0].get_xlabel() == pl.LABELS["y"] and axs[5].get_xlabel() == pl.LABELS["v"]
    np.testing.assert_allclose(np.array(axs[0].get_xlim()) * 80.0, axs[5].get_xlim())
    np.testing.assert_allclose(axs[1].get_xlim(), axs[0].get_xlim())
    np.testing.assert_allclose(axs[2].get_xlim(), axs[5].get_xlim())
    assert axs[0].get_xlim() == (0.0, 5.0) and axs[5].get_xlim() == (0.0, 400.0)
    for row, ax in zip(("dn", "dtau", "dF"), axs[[1, 3, 5]]):
        assert ax.get_ylim() == (-pl.CODE_RESIDUAL_LIMITS[row][0], pl.CODE_RESIDUAL_LIMITS[row][0])
    plt.close(fig)

    fig, axs = pl.plot_code_comparison(curves, residuals=pair, rows=("dF", "n", "F", "dn"), aH=80.0)  # the order is fixed
    assert [a.get_ylabel() for a in axs[[1, 3]]] == [r"$\Delta n/\bar{n}$", pl.LABELS["dF"]] and axs[2].get_ylabel() == pl.LABELS[
        "F"
    ]
    plt.close(fig)
    fig, axs = pl.plot_code_comparison(curves, residuals=pair, aH=80.0, window=(100, 200), residual_limits=dict(dF=(1e-2, 1e-7)))
    assert axs[5].get_xlim() == (100, 200) and axs[0].get_xlim() == (1.25, 2.5) and axs[5].get_ylim() == (-1e-2, 1e-2)
    plt.close(fig)
    fig, axs = pl.plot_code_comparison(curves, aH=80.0)  # no residuals: n, tau, F
    assert len(axs) == 3
    plt.close(fig)
    deep = dict(
        ref, n=ref["n"] * 10.0 ** -(30.0 * (ref["v"] > 200))
    )  # decades of nearly empty pixels: the axis stops at log_decades
    fig, axs = pl.plot_code_comparison([dict(mode="default", **deep)], rows=("n",), aH=80.0, log_decades=6.0)
    lo, hi = axs[0].get_ylim()
    assert np.isclose(hi / lo, 1e6)
    plt.close(fig)
    fig, axs = pl.plot_code_comparison(curves, rows=("F",))  # no n row: no aH needed
    assert len(axs) == 1 and axs[0].get_xlabel() == pl.LABELS["v"]
    plt.close(fig)
    fig, axs = pl.plot_code_comparison(curves, rows=("n",))  # n alone: its own axis at the bottom
    assert axs[0].get_xlabel() == pl.LABELS["y"]
    plt.close(fig)
    for bad in (
        dict(rows=("T",)),
        dict(rows=("dF",)),
        dict(),
    ):  # an unknown row; a residual row alone, no residuals; n and F, no aH
        with pytest.raises(ValueError):
            pl.plot_code_comparison(curves, **bad)
        plt.close("all")


def test_plot_code_comparison_curves_and_residuals():
    """Each curve takes its mode's style, its own velocity grid and an optional colour; each residual its curve's style, its
    pair's label and the definition of ``code_residual``, with a thinner line on request."""
    ref, other = _code_curves()
    shifted = dict(ref, v=ref["v"] + 0.25)  # fake_spectra's own grid: drawn as it is, nothing corrected
    curves = [
        dict(mode="fake_spectra", **shifted),
        dict(mode="default", **ref),
        dict(mode="default+switch", label="a_eff", color=pl.SWITCH_COLORS["a_eff"], **other),
    ]
    pairs = [("a_eff", "default", dict(lw=0.7)), ("a_eff", "fake_spectra")]
    fig, axs = pl.plot_code_comparison(curves, residuals=pairs, aH=80.0)
    fs, de, sw = axs[4].get_lines()
    assert fs.get_linestyle() == "--" and fs.get_color() == pl.CODE_MODES["fake_spectra"]["color"]
    np.testing.assert_array_equal(fs.get_xdata(), shifted["v"])
    assert de.get_linestyle() == "-" and de.get_color() == pl.PALETTE["blue"]
    assert sw.get_color() == pl.SWITCH_COLORS["a_eff"] and sw.get_linestyle() == ":"
    thin, thick = [ln for ln in axs[3].get_lines() if len(ln.get_xdata()) > 2]
    assert thin.get_linewidth() < thick.get_linewidth() and thin.get_color() == pl.SWITCH_COLORS["a_eff"]
    assert thin.get_label() == "a_eff − default" and thick.get_label() == "a_eff − fake_spectra"
    np.testing.assert_allclose(thin.get_ydata(), 2 * 0.01 / 2.01, rtol=1e-12)
    np.testing.assert_allclose([ln for ln in axs[5].get_lines() if len(ln.get_xdata()) > 2][0].get_ydata(), other["F"] - ref["F"])
    assert {t.get_text() for t in axs[1].get_legend().get_texts()} == {"a_eff − default", "a_eff − fake_spectra"}
    plt.close(fig)
    two = [dict(mode="default+switch", **other), dict(mode="matched-switch", **ref)]
    fig, axs = pl.plot_code_comparison(two, residuals=[("default+switch", "matched-switch")], aH=80.0)
    assert axs[1].get_legend().get_texts()[0].get_text() == "(default + switch) − (matched − switch)"
    plt.close(fig)
    fig, axs = pl.plot_code_comparison(two, residuals=[("default+switch", "matched-switch")], aH=80.0, residual_note="each − x")
    assert axs[1].get_legend() is None and any(t.get_text() == "each − x" for t in axs[1].texts)
    plt.close(fig)
    for bad in (
        dict(residuals=[("nope", "default+switch")]),
        dict(curves=[dict(mode="nope", **ref)]),
        dict(curves=[dict(mode="default", **ref)] * 2),
    ):
        with pytest.raises(ValueError):
            pl.plot_code_comparison(bad.pop("curves", two), aH=80.0, **bad)
        plt.close("all")


def test_plot_code_comparison_linewidths():
    """The widths follow the plotting order, from 4 to 1 by default, each curve above the one before and under the legend; each
    residual takes the width of its curve; a curve's own ``lw`` and a residual's style go over them."""
    ref, other = _code_curves()
    three = [dict(mode="fake_spectra", **ref), dict(mode="default", **other), dict(mode="matched", **ref)]
    pairs = [("default", "fake_spectra"), ("matched", "fake_spectra")]

    def curves_and_residuals(axs):  # the lines of the F row and of the dF row, without the line at 0
        return axs[4].get_lines(), [ln for ln in axs[5].get_lines() if len(ln.get_xdata()) > 2]

    fig, axs = pl.plot_code_comparison(three, residuals=pairs, aH=80.0)
    lines, res = curves_and_residuals(axs)
    np.testing.assert_allclose([ln.get_linewidth() for ln in lines], [4.0, 2.0, 1.0])
    np.testing.assert_allclose([ln.get_linewidth() for ln in res], [2.0, 1.0])
    orders = [ln.get_zorder() for ln in lines]
    assert orders == sorted(set(orders)) and orders[-1] < axs[0].get_legend().get_zorder()
    plt.close(fig)

    four = [dict(three[0]), dict(three[1]), dict(three[2]), dict(mode="default+switch", **other)]
    fig, axs = pl.plot_code_comparison(four, aH=80.0)
    widths = [ln.get_linewidth() for ln in axs[2].get_lines()]  # no residuals: n, tau, F
    assert np.isclose(widths[0], 4.0) and np.isclose(widths[-1], 1.0) and widths == sorted(widths, reverse=True)
    plt.close(fig)

    fig, axs = pl.plot_code_comparison(three, residuals=pairs, aH=80.0, linewidths=(5, 3, 0.5))
    lines, res = curves_and_residuals(axs)
    assert [ln.get_linewidth() for ln in lines] == [5, 3, 0.5] and [ln.get_linewidth() for ln in res] == [3, 0.5]
    plt.close(fig)

    own = [dict(three[0]), dict(three[1], lw=1.2), dict(three[2])]
    styled = [("default", "fake_spectra"), ("matched", "fake_spectra", dict(lw=0.7))]
    fig, axs = pl.plot_code_comparison(own, residuals=styled, aH=80.0, linewidths=(5, 3, 0.5))
    lines, res = curves_and_residuals(axs)
    assert [ln.get_linewidth() for ln in lines] == [5, 1.2, 0.5] and [ln.get_linewidth() for ln in res] == [1.2, 0.7]
    plt.close(fig)

    with pytest.raises(ValueError, match="one width per curve"):
        pl.plot_code_comparison(three, aH=80.0, linewidths=(4, 2))
    plt.close("all")


def test_plot_code_comparison_shade_below():
    """The band |residual| < shade_below, in the residual rows only and under their lines, leaves their limits alone; None: no band."""
    ref, other = _code_curves()
    curves = [dict(mode="fake_spectra", **ref), dict(mode="default", **other)]
    pair = [("default", "fake_spectra")]

    def band(ax):  # (low, high) of the one shaded patch of a row, in data units
        assert len(ax.patches) == 1
        low, high = ax.transData.inverted().transform(ax.patches[0].get_window_extent().get_points())[:, 1]
        return low, high

    fig, axs = pl.plot_code_comparison(curves, residuals=pair, aH=80.0)
    fig.canvas.draw()
    for row, ax in zip(("dn", "dtau", "dF"), axs[[1, 3, 5]]):
        np.testing.assert_allclose(band(ax), (-1e-5, 1e-5), rtol=1e-6)
        assert ax.patches[0].get_zorder() < min(ln.get_zorder() for ln in ax.get_lines())
        assert 0.0 < ax.patches[0].get_alpha() < 1.0
        assert ax.get_ylim() == (-pl.CODE_RESIDUAL_LIMITS[row][0], pl.CODE_RESIDUAL_LIMITS[row][0])
    assert not any(ax.patches for ax in axs[[0, 2, 4]])
    plt.close(fig)

    fig, axs = pl.plot_code_comparison(curves, residuals=pair, aH=80.0, shade_below=1e-3)
    fig.canvas.draw()
    np.testing.assert_allclose(band(axs[5]), (-1e-3, 1e-3), rtol=1e-6)
    plt.close(fig)

    fig, axs = pl.plot_code_comparison(curves, residuals=pair, aH=80.0, shade_below=None)
    assert not any(ax.patches for ax in axs)
    plt.close(fig)


def test_no_matplotlib_colour_map_anywhere_in_the_package():
    """Every colour map of the package is one of the theme's (``CMAPS``, Tol's maps): no string naming another one
    (``cmap="seismic"``, ``"viridis"``, ...) in any source file, default or call, so that none can come back unnoticed."""
    import re
    from pathlib import Path

    import matplotlib

    root = Path(pl.__file__).resolve().parents[1]
    named = re.compile(r"""(?:cmap\w*\s*=\s*|get_cmap\(\s*|colormaps\[\s*)["']([A-Za-z_]+)["']""")
    builtin = set(matplotlib.colormaps) - {n for n in matplotlib.colormaps if n.startswith("tol_")}
    bad = []
    for path in root.rglob("*.py"):
        if "cloudy_tables" in path.parts:
            continue
        for k, line in enumerate(path.read_text().splitlines(), 1):
            bad += [f"{path.relative_to(root)}:{k}: {m}" for m in named.findall(line) if m in builtin or not m.startswith("tol_")]
    assert not bad, "colour maps outside the theme:\n" + "\n".join(bad)
