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
    pl.plot_skewer_anatomy(sb, 0)[0].canvas.draw()


def test_grf_projection_and_p1d_figure():
    g = PowerLawGRF(A_s=1.0, k_0=1.0, n_s=-1.0, L=(10.0,) * 3, N=(8,) * 3)
    k = 2 * np.pi * np.arange(1, 5) / 10.0
    P, sig = pl.grf_p1d_along_axis(g, k, axis=1, return_sigma=True)
    assert np.all(P > 0) and np.all(sig > 0) and np.all(sig <= P)
    pl.plot_p1d_input_vs_measured(k, {"x": (P * 1.1, sig, "C0", {})}, P)[0].canvas.draw()


def test_cylinder():
    v = np.linspace(0, 100, 64)
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
    from skewerjax.core import LineBundle as LB
    t = np.linspace(0, 1, 12)
    pts = np.stack([np.cos(3 * t), np.sin(3 * t), t], -1)
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    assert pl.plot_skewer_geometry_3d(ax, pts, highlight=5) is None              # scale_factors optional
    pl.plot_skewer_geometry_3d(ax, pts, scale_factors=1 - 0.01 * t)
    fig.canvas.draw()
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


def test_deposition_steps_on_voronoi_cells():
    """plot_deposition_steps from the parts of cell_pieces: the two ends of a periodic skewer through random Voronoi cells,
    every piece's tau around v = 0 and v = P with and without velocity_period; plot_frames_axes relabels each line's own
    window on the global axis (the reference line stays within it)."""
    from skewerjax.core import cell_pieces, cells_from_values, compute_optical_depth, voronoi_volumes
    box = 4.0
    rng = np.random.default_rng(1)
    gen = rng.uniform(0, box, (800, 3))
    vel = rng.normal(0, 30, (800, 3)); vel[:, 0] = 150.0                       # everything moves away: the end wraps
    cells = cells_from_values(gen, box, {"HI": 10 ** rng.uniform(-12, -10, 800)}, temperature=10 ** rng.uniform(4, 4.5, 800),
                              velocity=vel, volumes=voronoi_volumes(gen, box))
    x = np.linspace(0.0, box, 41)
    cosmo = Cosmology(H0=phys.H0, Omega_r_0=0.0, Omega_m_0=0.31, Omega_k_0=0.0, Omega_lambda_0=0.69)
    sb = SkewerBundle(spatial_points=jnp.asarray(np.stack([x, np.full_like(x, 2.0), np.full_like(x, 2.0)], -1)[None]), a0=0.25,
                      ICs_defined_at_index=0, cosmologies=cosmo)
    lya = LineBundle(["HI-LyA"])
    pieces, parts = cell_pieces(cells, lya, skewer_bundle=sb, return_parts=True)
    P = float(pieces.velocity_span_kms[0])
    v_axes = (np.arange(-100.0, 300.0, 1.0), np.arange(P - 150.0, P + 250.0, 1.0))
    per = lambda vv, wrap: np.asarray(compute_optical_depth(pieces, vv, lya, per_piece=True, velocity_period=wrap,
                                                            resolution_tolerance=None))[0, 0]
    fig, axs, info = pl.plot_deposition_steps(parts, sb, ((0.0, 0.6), (box - 0.6, box)), v_axes, [per(vv, False) for vv in v_axes],
                                              [per(vv, True) for vv in v_axes], period=P)
    fig.canvas.draw()
    assert all(n > 1 for n in info["n_cells"]) and info["tracked"] and all(side == "end" for side, _ in info["tracked"])
    assert info["dtau"]["start"]["i"] > 0 and info["dtau"]["end"]["max"] < 1e-2 * info["dtau"]["start"]["max"]   # it all comes back at 0
    v_ring = np.arange(-100.0, P + 100.0, 2.0)                                # the same wrap on a ring
    fig, axs, ring = pl.plot_velocity_ring(parts, ((0.0, 0.6), (box - 0.6, box)), v_ring, per(v_ring, False), per(v_ring, True), period=P)
    fig.canvas.draw()
    assert ring["tracked"] and set(axs) == {"a", "b", "c_spiral", "c_ring", "d"}
    two = LineBundle(["HI-LyA", "SiIII-1206"], ref_line_id="HI-LyA")
    v = np.arange(0.5, P, 1.0)
    F = np.exp(-np.stack([np.exp(-((v - 50) / 5) ** 2), 0.1 * np.exp(-((v - 50) / 3) ** 2)]))
    fig, axs = pl.plot_frames_axes(v, F, two, a_dagger=0.25, tau_scale={"SiIII-1206": 10.0})
    fig.canvas.draw()
    np.testing.assert_allclose(axs[0].get_lines()[1].get_ydata(), F[1] ** 10, rtol=1e-12)        # display only: exp(-10 tau)
    assert "for visibility" in axs[0].get_lines()[1].get_label()
    lya_line = axs[1].get_lines()[0].get_xdata()
    assert lya_line.min() >= 0.0 and lya_line.max() < P                    # the reference line keeps its own window
    assert np.isclose(axs[1].get_lines()[3].get_xdata()[0], v[0] - float(two.delta_v[1]))   # Si III moved by -delta_v


def test_quickstart_and_tutorial_figures():
    """The figure functions of notebooks 00, 00b and 04 on toy inputs, and the small helpers (custom panel labels, the
    shading outside a velocity window)."""
    import jax
    from skewerjax.core import (DeltaFieldBuilder, bundle_frame, cell_pieces, cells_from_values, compute_optical_depth,
                                tau_by_path_bins)
    from skewerjax.core.stats import OverfluxStatistics1D, fit_mcdonald_a
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
    tau = np.asarray(compute_optical_depth(pieces, v, lines, velocity_period=True, line_frame="reference", resolution_tolerance=None))
    stats = OverfluxStatistics1D(DeltaFieldBuilder(tau, labels=["Lya", "SiIII"], key_joiner="+").build(), float(v[1] - v[0])).compute_p1d()
    fig, axs = pl.plot_toy_box(cells, sb, lines, v, np.exp(-tau), stats)
    fig.canvas.draw()
    assert axs[2].get_title() == "P1D over 3 skewers"
    frame = bundle_frame(sb)
    A_H = parts["absorbers"]["HI"]
    edges = np.linspace(0.0, box, 5)
    one = lambda lb: LineBundle([lb], ref_line_id=lb)
    tb = [np.asarray(tau_by_path_bins(parts["absorbers"][sp], one(l), v, frame, edges))[:, 0] for sp, l in (("HI", "HI-LyA"), ("SiIII", "SiIII-1206"))]
    s = np.asarray(frame.s_grid[0])
    fig, axs, info = pl.plot_tau_stretches(edges, s, np.ones_like(s), v, tuple(tb), tuple(t.sum(0) for t in tb), frame, A_H, box=box)
    fig.canvas.draw()
    assert 0.0 <= info["s_max"] <= box and np.all(np.diff(info["vH_s"]) > 0)
    pv = {"n": nH[None, :61], "T": np.full((1, 61), 2e4), "v_los": np.zeros((1, 61))}
    slab = dict(sigma=rng.uniform(1.0, 10.0, (32, 32)), extent=(0.0, box, 0.0, box), axes=(1, 2))
    pl.plot_slab_and_skewer(slab, x, (pv, pv), box=box, y_sk=2.0, z_c=2.0, thick=0.5, x_mark=1.0, zoom=0.5)[0].canvas.draw()
    lo = np.sort(rng.uniform(0, box, 20)); hi = lo + 0.05
    pl.plot_cell_crossings(lo, hi, nH[:20], np.full(20, 0.2), 0.01)[0].canvas.draw()
    pl.plot_voigt_of_piece(1e4, lines.lines)[0].canvas.draw()
    pl.plot_wrap_tiles(v, (tau[0, 0], tau[1, 0]), tau[:, :1], period=P, delta_v=float(lines.delta_v[1]), v_own=v[10], v_obs=v[5])[0].canvas.draw()
    st = OverfluxStatistics1D(DeltaFieldBuilder(tau, labels=["Lya", "SiIII"], key_joiner="+").build(), float(v[1] - v[0])).compute_p1d().compute_xi1d()
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
