"""Smoke tests of the plotting module: the style and every tutorial figure function draw without error (Agg)."""
import matplotlib

matplotlib.use("Agg")
import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from skewerjax.core import Cosmology, LineBundle, PowerLawGRF, SkewerBundle, phys  # noqa: E402
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


def test_fourth_pass_figures():
    from skewerjax.utils import tutorial as tu
    t = np.linspace(0, 1, 12)
    pts = np.stack([np.cos(3 * t), np.sin(3 * t), t], -1)
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    assert pl.plot_skewer_geometry_3d(ax, pts, scale_factors=1 - 0.01 * t, highlight=5) is not None
    fig.canvas.draw()
    pl.plot_two_elements(tu.two_elements(pieces=(1, 4), n_fine=2001, n_v=301))[0].canvas.draw()
    pl.plot_linear_detuning(tu.linear_detuning(n_v=801))[0].canvas.draw()
    pl.plot_periodicity_options(tu.periodic_box_demo())[0].canvas.draw()
    g = PowerLawGRF(A_s=1.0, k_0=1.0, n_s=-1.0, L=(10.0, 10.0), N=(16, 16))
    d = np.random.default_rng(0).normal(size=(2, 16, 16))
    fig, axs, out = pl.plot_pk_recovery_nd({"2D": (g, d)})
    fig.canvas.draw()
    assert "map_chi2_dof" in out["2D"]
    for theme in ("dark", "paper"):
        pl.use_style(theme)
        pl.plot_dag(("lines", "cosmology", "skewers"))[0].canvas.draw()
