"""Smoke tests of the plotting module: the style and every tutorial figure function draw without error (Agg)."""
import matplotlib

matplotlib.use("Agg")
import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

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
    # every edge starts and ends on the drawn outline of its boxes (the box plus its pad)
    ew, eh = pl._DAG_W / 2 + pl._DAG_PAD, pl._DAG_H / 2 + pl._DAG_PAD
    on_outline = lambda p, k: (np.isclose(abs(p[0] - pl.DAG_NODES[k][0]), ew) and abs(p[1] - pl.DAG_NODES[k][1]) <= eh) or \
        (np.isclose(abs(p[1] - pl.DAG_NODES[k][1]), eh) and abs(p[0] - pl.DAG_NODES[k][0]) <= ew)
    for a, b, _, _, route in pl.DAG_EDGES:
        pts = pl._dag_route(a, b, route, pl.DAG_NODES, pl._DAG_LANES.get((a, b), 0.0), pl._DAG_END_DX.get((a, b), 0.0))
        assert on_outline(pts[0], a) and on_outline(pts[-1], b), (a, b, pts)


def test_line_table_and_skewer_figures():
    lines = LineBundle(["HI-LyA", "SiIII-1206", "CIV-1548"], ref_line_id="HI-LyA")
    pl.plot_line_table(lines)[0].canvas.draw()
    t = jnp.linspace(0.0, 1.0, 50)
    pts = jnp.stack([5 + 3 * jnp.cos(6 * t), 5 + 3 * jnp.sin(6 * t), 10 * t], -1)[None]
    cosmo = Cosmology(H0=phys.H0, Omega_r_0=0.0, Omega_m_0=1.0, Omega_k_0=0.0, Omega_lambda_0=0.0)
    sb = SkewerBundle(spatial_points=pts, a0=0.5, x0=0.0, cosmologies=cosmo)
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    assert pl.plot_paths_3d(ax, pts, values=np.asarray(sb.scale_factors[0]), box=10.0) is not None
    pl.plot_skewer_anatomy(sb, 0)[0].canvas.draw()


def test_voronoi_column_identity():
    fig, axs, nums = pl.plot_voronoi_discretisations()
    assert abs(nums["segments"] / nums["exact"] - 1) < 1e-12 and abs(nums["points"] / nums["exact"] - 1) > 1e-3


def test_grf_projection_and_p1d_figure():
    g = PowerLawGRF(A_s=1.0, k_0=1.0, n_s=-1.0, L=(10.0,) * 3, N=(8,) * 3)
    k = 2 * np.pi * np.arange(1, 5) / 10.0
    P, sig = pl.grf_p1d_along_axis(g, k, axis=1, return_sigma=True)
    assert np.all(P > 0) and np.all(sig > 0) and np.all(sig <= P)
    pl.plot_p1d_input_vs_measured(k, {"x": (P * 1.1, sig, "C0", {})}, P)[0].canvas.draw()


def test_frames_overflux_and_cylinder():
    lines = LineBundle(["HI-LyA", "SiIII-1206"], ref_line_id="HI-LyA")
    v = np.linspace(0, 100, 64)
    tau = np.exp(-((v - 50) / 5) ** 2)[None, None] * np.ones((2, 3, 1))
    phi = 1.0 / (1 + (v - 30) ** 2)[None] * np.ones((2, 1))
    pl.plot_line_frames(v, phi, phi, lines, 30.0, a_dagger=0.5)[0].canvas.draw()
    s = np.linspace(0, 10, 40)
    cloud = dict(axis=s, s_perp=0.1 * np.sin(s), t_perp=0.1 * np.cos(s), value=np.exp(s / 3))
    flux = dict(v=v, s_of_v=v / 10, edges=np.linspace(0, 10, 5), tau_bins={"Lya": (np.ones((4, 64)) * 0.1, "C0")})
    fig, (ax3, axf) = pl.plot_cylinder(s, {"n": (np.log10(1 + s), "C1")}, cloud=cloud, flux=flux, progress=0.5, radius=0.3)
    fig.canvas.draw()


def test_palette_fonts_and_backend():
    # every discrete colour is Tol's "bright" scheme (tol_colors colorsets["bright"]), in Tol's order
    assert pl.PALETTE == pl._TOL_BRIGHT_VENDORED and list(pl.PALETTE) == ["blue", "red", "green", "yellow", "cyan", "purple", "grey"]
    allowed = set(pl.PALETTE.values()) | {pl.FG, pl.BG}
    assert set(pl.ENTITY_COLORS.values()) <= allowed
    assert all(c in pl.PALETTE.values() for c in pl.ORDER_COLORS)
    assert {"tol_iridescent", "tol_iridescent_r", "tol_iridescentline"} <= set(matplotlib.colormaps)
    assert plt.rcParams["image.cmap"] == pl.CMAPS["sequential"]
    for hl, ext in ((("tau",), False), (("simulation", "fake_spectra"), True)):
        fig, _ = pl.plot_dag(hl, external=ext)
        assert pl.check_fonts(fig) == []
    assert plt.rcParams["legend.fontsize"] >= pl.FONT_MIN["legend"] and plt.rcParams["xtick.labelsize"] >= pl.FONT_MIN["tick"]
    assert pl.use_backend() == matplotlib.get_backend()      # outside IPython: unchanged


def test_fifth_pass_figures():
    import jax
    from skewerjax.utils import tutorial as tu
    from skewerjax.core import LineBundle as LB
    t = np.linspace(0, 1, 12)
    pts = np.stack([np.cos(3 * t), np.sin(3 * t), t], -1)
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    assert pl.plot_skewer_geometry_3d(ax, pts, highlight=5) is None              # scale_factors optional
    pl.plot_skewer_geometry_3d(ax, pts, scale_factors=1 - 0.01 * t)
    fig.canvas.draw()
    pl.plot_periodic_pieces(tu.periodic_pieces_demo(n_images=5))[0].canvas.draw()
    g1 = PowerLawGRF(A_s=1.0, k_0=1.0, n_s=-0.5, L=20.0, N=64)
    g2 = PowerLawGRF(A_s=1.0, k_0=1.0, n_s=-1.0, L=(10.0, 10.0), N=(16, 16))
    d1 = np.asarray(jax.vmap(g1.sample)(jax.random.split(jax.random.PRNGKey(0), 8)))
    d2 = np.asarray(jax.vmap(g2.sample)(jax.random.split(jax.random.PRNGKey(1), 8)))
    fig, axs, out = pl.plot_isotropic_recovery({"1D": (g1, d1), "2D": (g2, d2)})
    fig.canvas.draw()
    assert {"chi2_dof", "xi_chi2_dof"} <= set(out["2D"])
    r, xi, err, n, xin, cov = pl.correlation_binned(d2, g2.L, np.linspace(0, 2.5, 6), g2)
    assert np.isclose(xin[0], float(g2.variance()), rtol=1e-10), "input xi(0) = sigma^2"
    fig, axs, chi2 = pl.plot_kpar_kperp_maps({"2D": (g2, d2)})
    fig.canvas.draw()
    fig, axs, chi2w = pl.plot_pk_mu_wedges({"2D": (g2, d2)})
    fig.canvas.draw()
    assert len(chi2w["2D"]) == len(pl.MU_WEDGES) - 1
    lines = LB(["HI-LyA", "SiIII-1206"], ref_line_id="HI-LyA")
    v = np.linspace(0, 400, 200)
    F = np.exp(-np.exp(-((v - 200) / 10) ** 2))[None] * np.ones((2, 1))
    pl.plot_frames_axes(v, F, lines, v_ref=v, F_ref=F, a_dagger=0.25, centres=[200.0])[0].canvas.draw()
    for theme in ("dark", "paper"):
        pl.use_style(theme)
        pl.plot_dag(("lines", "cosmology", "skewers"))[0].canvas.draw()


def test_xi_covariance_is_the_scatter_of_the_draws():
    """The exact Gaussian covariance of the binned xi (``correlation_binned``) against the scatter of many draws."""
    import jax
    g = PowerLawGRF(A_s=1.0, k_0=1.0, n_s=-1.0, L=50.0, N=128)
    d = np.asarray(jax.vmap(g.sample)(jax.random.split(jax.random.PRNGKey(5), 3000)))
    edges = np.linspace(0.0, 12.5, 7)
    per = np.stack([pl.correlation_binned(d[i:i + 1], g.L, edges)[1] for i in range(d.shape[0])])
    cov1 = pl.correlation_binned(d[:1], g.L, edges, g)[5]
    np.testing.assert_allclose(per.std(0, ddof=1), np.sqrt(np.diag(cov1)), rtol=0.1)


def test_gas_cells_along_a_skewer():
    """plot_gas_cells: the grid cubes and the Voronoi polyhedra crossed by one skewer through the periodic face."""
    from skewerjax.core import cells_from_values, cells_on_grid, select_cells, voronoi_volumes
    box, n = 4.0, 8
    rng = np.random.default_rng(0)
    vals = lambda m: dict(temperature=10 ** rng.uniform(4, 5, m), velocity=rng.normal(0, 50, (m, 3)))
    grid = cells_on_grid(box, {"HI": 10 ** rng.uniform(-13, -11, (n,) * 3), "SiIII": np.full((n,) * 3, 1e-14)},
                         **vals(n ** 3))
    gen = rng.uniform(0, box, (600, 3))
    vor = cells_from_values(gen, box, {"HI": 10 ** rng.uniform(-13, -11, 600)}, volumes=voronoi_volumes(gen, box), **vals(600))
    pts = np.array([3.0, 1.1, 2.3]) + np.linspace(0, 2.5, 30)[:, None] * np.array([1.0, 0.6, 0.4]) / np.linalg.norm([1.0, 0.6, 0.4])
    for cells, n_sp in ((grid, 2), (vor, 1)):
        near = select_cells(cells, pts[None])
        fig, axs, info = pl.plot_gas_cells(cells, near, pts)
        pl.add_panel_letters(axs)
        fig.canvas.draw()
        assert axs.shape == (n_sp, 3) and 0 < info["n_crossed"] <= info["n_selected"] == len(near)
        assert np.isclose(info["path_length"], 2.5)
