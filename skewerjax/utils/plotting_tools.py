"""Plotting tools for SkewerJAX.

Sections
--------
1. Style and generic helpers: ``ten_power_formatter``, ``pk_style`` (package style from
   ``config.setup_matplotlib.matplotlib_default_config`` as an rc context).
2. 3D rendering of fields along skewers: ``sample_cylinder_cloud``, ``plot_3d_radar_cylinder``,
   ``sample_box_cloud``, ``plot_3d_neon_ridge_cube``, ``generate_all_simulation_plots``,
   ``create_animated_gifs_and_cleanup``; neon 2D figures on black: ``plot_neon_lines`` (spectra,
   overflux or densities along a skewer), ``plot_xi3d_rperp_pi`` (xi3D(r_perp, pi) heatmap,
   profiles and surface).
3. P1D / xi1D of composite overflux fields by line pair and order (``OverfluxStatistics1D``
   ``Pk_pair`` / ``xi_pair``): ``plot_p1d_by_order``, ``plot_p1d_total_recovery``,
   ``plot_xi_by_order``, ``plot_xi_total_recovery`` and their helpers (these four expect lags
   in units of L_box for the half-box line); ``plot_orders_superimposed`` (exact per-order
   decomposition, ``*_total_by_order``, all orders on one panel).
4. How P(k) is computed and recovered in 1D (``OverfluxStatistics1D``), plus helpers shared
   with 3D: ``hybrid_k_edges``, ``sample_mean_and_error``, ``pk_sigma``, ``recovery_summary``,
   ``plot_pk_recovery``, ``plot_xi_recovery``, ``plot_p1d_from_skewers``,
   ``plot_p1d_mode_binning``.
5. How P(k) is computed and recovered in 3D (``OverfluxStatistics3D``, ``PowerLawGRF``):
   ``plot_field_slices``, ``plot_kspace_slices``, ``plot_kspace_3d``,
   ``plot_kperp_kpar_binning``, ``plot_p3d_kperp_kpar``, ``plot_p3d_cells``,
   ``plot_p3d_recovery``, ``plot_xi3d_recovery``.

Functions in sections 1, 4 and 5, ``plot_neon_lines``, ``plot_xi3d_rperp_pi`` and
``plot_orders_superimposed`` return ``(fig, axes)``, accept ``ax`` and never save to disk.
Sections 2 and 3 keep their original interfaces (``show`` / ``savepath`` options; the order
plots of section 3 call ``plt.show()``).
"""
import gc
import glob
import os
import warnings
from contextlib import contextmanager

import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib import cm
from matplotlib.colors import LogNorm, Normalize
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from PIL import Image

from skewerjax.config.setup_matplotlib import matplotlib_default_config
from skewerjax.constants.physical_constants import phys


# ============================================================
# 1. Style and generic helpers
# ============================================================

def ten_power_formatter(y, pos):
    # Zero stays zero
    if np.isclose(y, 0.0):
        return "0"
    # Get exponent
    exp = int(np.round(np.log10(abs(y))))
    # Only label exact powers of ten, otherwise empty label
    if np.isclose(abs(y), 10.0**exp):
        return rf"$10^{{{exp}}}$"
    return ""


@contextmanager
def pk_style():
    """Package plotting style (``matplotlib_default_config``) as a temporary rc context.

    Draw (``plt.show()``) inside the ``with`` block so that ticks and labels use it. Not
    needed if the style is already set globally (``plt.rc`` / ``rcParams.update``).
    """
    font, rc = matplotlib_default_config()
    with plt.rc_context({**{f"font.{k}": v for k, v in font.items()}, **rc}):
        yield


def _figure(ax, nrows=1, ncols=1, figsize=(12, 9), layout="constrained", **kw):
    """(fig, flat axes array): a new figure if ``ax`` is None, else the given axes."""
    if ax is None:
        fig, axs = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False, layout=layout, **kw)
        return fig, axs.ravel()
    axs = np.empty(np.size(ax), dtype=object)
    axs[:] = list(np.ravel(np.array(ax, dtype=object)))
    if axs.size != nrows * ncols:
        raise ValueError(f"Expected {nrows * ncols} axes, got {axs.size}.")
    return axs[0].figure, axs


def _log_norm(values, pct=(2.0, 98.0)):
    """LogNorm from percentiles of the positive finite values.

    Single periodogram modes scatter as P chi^2_2 / 2, with dips near zero that would wash
    out a min/max colour scale.
    """
    v = np.asarray(values, dtype=float).ravel()
    v = v[np.isfinite(v) & (v > 0)]
    return LogNorm(*np.percentile(v, pct)) if v.size else LogNorm(1e-12, 1.0)


# ============================================================
# 2. 3D rendering of fields along skewers
# ============================================================

def _select_indices(points, skewer_index=0, n_samples_plot=1):
    n_skewers = np.asarray(points).shape[0]
    idx = np.arange(int(skewer_index), int(skewer_index) + max(1, int(n_samples_plot)))
    return np.clip(idx, 0, n_skewers - 1)


# ============================================================
# Sample the field in a cylinder around one skewer
# ============================================================
def sample_cylinder_cloud(
    quadri_vectors_skewer,
    field_fn,
    R_plot,
    *,
    los_axis=1,
    component=0,
    n_axis=240,
    n_perp=80,
    a_index=0,
    xyz_index=(1, 2, 3),
    length_fraction=1.0,
    seed=0,
):
    """Sample the field on a point cloud filling a cylinder around the skewer.

    Returns a dict with flat arrays in the skewer's LOCAL frame, ready to feed
    to plot_3d_radar_cylinder(..., surrounding_field=[cloud, ...]):

        s, t   : transverse offsets (physical, in [-R_plot, R_plot])
        axis   : along-axis coordinate == the LoS coordinate (xyz column
                 `los_axis`) at each sampled node -- matches the radar y-axis.
        value  : RAW field value at each point (the plot does log/clim/alpha).
        R      : R_plot (so the plot can map physical radius -> cylinder_radius).

    R_plot is in the coordinate units of the quadri-vectors (same units as
    `points`). n_perp is rounded up to a square stratified grid per axis node.
    Default layout is [a, x, y, z]; for [x, y, z, a] use a_index=3,
    xyz_index=(0, 1, 2).
    """
    quad = np.asarray(quadri_vectors_skewer, dtype=np.float64)
    xyz_index = list(xyz_index)
    path = quad[:, xyz_index]                       # (N, 3)
    a_col = quad[:, a_index]
    N = path.shape[0]

    # centred sub-segment of node indices
    lo = int(0.5 * (1.0 - length_fraction) * (N - 1))
    hi = N - 1 - lo
    node_idx = np.unique(np.linspace(lo, hi, n_axis).astype(int))
    n_axis = node_idx.size

    e1, e2 = _local_basis(path)                     # consistent transverse frame

    rng = np.random.default_rng(seed)
    # stratified (jittered-grid) disk samples per axis node -> uniform coverage,
    # far smoother than pure-random which leaves Poisson clumping.
    m = int(np.ceil(np.sqrt(n_perp)))
    lin = (np.arange(m) + 0.5) / m * 2.0 - 1.0          # cell centres in [-1, 1]
    gx, gy = np.meshgrid(lin, lin)                       # (m, m)
    jx = (rng.random((n_axis, m, m)) - 0.5) * (2.0 / m)  # per-node jitter
    jy = (rng.random((n_axis, m, m)) - 0.5) * (2.0 / m)
    s = ((gx[None] + jx) * R_plot).reshape(n_axis, m * m)
    t = ((gy[None] + jy) * R_plot).reshape(n_axis, m * m)
    n_perp = m * m

    P = path[node_idx]                              # (n_axis, 3)
    A = a_col[node_idx]                            # (n_axis,)
    axis_coord = path[node_idx, los_axis]          # (n_axis,) -> radar y

    pos = (P[:, None, :]
           + s[..., None] * e1[None, None, :]
           + t[..., None] * e2[None, None, :])      # (n_axis, n_perp, 3)

    quad_grid = np.empty((n_axis, n_perp, 4))
    quad_grid[..., a_index] = A[:, None]
    quad_grid[..., xyz_index[0]] = pos[..., 0]
    quad_grid[..., xyz_index[1]] = pos[..., 1]
    quad_grid[..., xyz_index[2]] = pos[..., 2]

    out = np.asarray(field_fn(quad_grid.reshape(-1, 4)))
    if out.ndim == 2:
        out = out[component]
    value = out.reshape(n_axis, n_perp).astype(float)

    # keep a clean cylinder: drop the grid corners outside the radius
    value[(s ** 2 + t ** 2) > R_plot ** 2] = np.nan

    axis_full = np.broadcast_to(axis_coord[:, None], (n_axis, n_perp))
    return dict(s=s.ravel(), t=t.ravel(), axis=axis_full.ravel(),
                value=value.ravel(), R=float(R_plot))


# ============================================================
# NEW: render one cloud into the radar axes
# ============================================================
def _draw_field_cloud(ax, x0, cloud, *, cmap, vmin, vmax, cylinder_radius,
                      radius_fill, logscale, alpha_max, alpha_gamma,
                      core_size, softness, size_growth, alpha_falloff, threshold):
    """Render the field cloud as smooth gas.

    Each sampled point is drawn as a soft Gaussian-like splat: a small bright
    core wrapped in `softness` progressively larger, fainter halos. With many
    low-alpha points the overlapping splats blend into a continuous haze instead
    of discrete bulbs.
    """
    s = np.asarray(cloud["s"]); t = np.asarray(cloud["t"])
    axis = np.asarray(cloud["axis"]); val = np.asarray(cloud["value"])
    R = cloud["R"]

    v = np.log10(np.maximum(val, 1e-300)) if logscale else val
    vr = max(vmax - vmin, 1e-30)
    scaled = np.clip((v - vmin) / vr, 0.0, 1.0)

    keep = np.isfinite(scaled) & (scaled > threshold)
    if not np.any(keep):
        return
    s, t, scaled = s[keep], t[keep], scaled[keep]
    axis = axis[keep]

    k = (cylinder_radius * radius_fill) / max(R, 1e-30)
    px = x0 + k * s
    py = axis
    pz = k * t

    base = cmap(scaled)                       # (M, 4) RGBA from the colormap
    a_pt = alpha_max * scaled ** alpha_gamma  # per-point peak opacity

    # soft splat: large faint halos first, small bright core last
    for j in reversed(range(max(int(softness), 1))):
        size_j = core_size * (size_growth ** j)
        rgba = base.copy()
        rgba[:, 3] = np.clip(a_pt * (alpha_falloff ** j), 0.0, 1.0)
        ax.scatter(px, py, pz, color=rgba, s=size_j, marker="o",
                   linewidths=0, depthshade=False, zorder=1)


# ============================================================
# radar-cylinder plot (your function + surrounding-field layer)
# ============================================================
def plot_3d_radar_cylinder(
    x_array=None,
    y_dict=None,
    *,
    points=None,
    los_axis=1,
    skewer_index=0,
    n_samples_plot=1,
    cylinder_radius=10.0,
    cylinder_separation=5.2,
    field_logscale=True,
    normalization="global",
    scale_box_x_factor=0.87,
    legend_y_factor=0.8,
    scale_units_xy=(0.50, 1.05),
    profile_angle_offset_deg=90.0,
    n_y_slices=4,
    field_units=None,
    x_axis_label=None,
    n_x_ticks=3,
    x_axis_pad=1.0,
    x_axis_tick_size=0.20,
    x_axis_fontsize=16,
    factor_ticks_x=3.0,
    factor_ticks_z=7.0,
    factor_x_label=1.2,
    view_elev=18,
    view_azim=-40,
    los_range=None,
    show_profile_points=False,
    profile_point_size=10.0,
    profile_linewidth=1.05,
    glow=True,
    show=True,
    profile_zorder=10,       # profiles drawn on top of cloud; raise if still hidden
    # ---------- NEW: surrounding-field cloud ----------
    surrounding_field=None,          # cloud dict or list of dicts (per selected sample)
    surrounding_cmap="magma",
    surrounding_logscale=True,
    surrounding_clim=None,           # (vmin, vmax) in (log) value units; else percentiles
    surrounding_clim_percentiles=(45.0, 99.7),
    surrounding_radius_fill=1.0,     # fraction of cylinder_radius the cloud fills
    surrounding_alpha_max=0.30,      # peak opacity of a splat core
    surrounding_alpha_gamma=1.2,     # >1 fades faint gas; lower -> smoother gradient
    surrounding_core_size=5.0,       # core splat size (points^2)
    surrounding_softness=5,          # number of soft halo layers (higher = smoother)
    surrounding_size_growth=1.7,     # halo size growth per layer
    surrounding_alpha_falloff=0.5,   # halo alpha falloff per layer
    surrounding_threshold=0.02,      # drop scaled values below this (transparent voids)
    surrounding_colorbar=True,
    surrounding_colorbar_label=None,
    colors=None,
):
    """Radar-cylinder plot for one or more 1D skewers.

    All original parameters are unchanged. The new ``surrounding_*`` options
    overlay the field around each skewer as a glowing 3D point cloud produced by
    ``sample_cylinder_cloud`` (pass one cloud per selected sample, in the same
    order as ``skewer_index .. skewer_index + n_samples_plot - 1``).
    """
    if x_array is None:
        if points is None:
            raise ValueError("Pass either `x_array` or `points`.")
        points = np.asarray(points)
        if points.ndim != 3:
            raise ValueError("`points` must have shape (N_samples, N_x, N_dim). "
                             f"Got {points.shape}.")
        if los_axis < 0 or los_axis >= points.shape[-1]:
            raise ValueError(f"`los_axis={los_axis}` invalid for N_dim={points.shape[-1]}.")
        x_array = points[:, :, los_axis]

    x_array = np.asarray(x_array)
    if x_array.ndim != 2:
        raise ValueError(f"Expected x_array (N_samples, N_x). Got {x_array.shape}.")
    if not isinstance(y_dict, dict) or len(y_dict) == 0:
        raise ValueError("Expected non-empty y_dict mapping labels to arrays.")

    labels = list(y_dict.keys())
    y_arrays = {lab: np.asarray(y_dict[lab]) for lab in labels}
    for lab, arr in y_arrays.items():
        if arr.shape != x_array.shape:
            raise ValueError(f"y_dict[{lab!r}] has shape {arr.shape}, expected {x_array.shape}.")

    selected = _select_indices(x_array, skewer_index, n_samples_plot)

    profiles = np.asarray([[y_arrays[lab][idx] for lab in labels] for idx in selected],
                          dtype=float)

    if normalization == "skewer":
        range_values = profiles
    elif normalization == "global":
        range_values = np.concatenate([y_arrays[lab].ravel() for lab in labels])
    else:
        raise ValueError("normalization must be either 'global' or 'skewer'.")

    vmin, vmax, vrange = _scaled_range(range_values, eps=1e-300,
                                       logscale=field_logscale, positive_only=False)
    eps = 1e-300
    positive_only = False

    n_samples = profiles.shape[0]
    n_profiles = profiles.shape[1]

    if colors is None:
        colors = _neon_colors(labels)
    profile_color = {lab: colors[lab] for lab in labels}
    angles = {lab: np.deg2rad(profile_angle_offset_deg) + 2.0 * np.pi * i / n_profiles
              for i, lab in enumerate(labels)}

    # ---- normalize surrounding_field input + global colour range ----------
    clouds = None
    if surrounding_field is not None:
        clouds = ([surrounding_field] if isinstance(surrounding_field, dict)
                  else list(surrounding_field))
        if len(clouds) < n_samples:
            clouds = clouds + [None] * (n_samples - len(clouds))
        s_cmap = plt.get_cmap(surrounding_cmap)
        if surrounding_clim is not None:
            s_vmin, s_vmax = float(surrounding_clim[0]), float(surrounding_clim[1])
        else:
            allv = np.concatenate([np.asarray(c["value"]).ravel()
                                   for c in clouds if c is not None])
            allv = (np.log10(np.maximum(allv, 1e-300)) if surrounding_logscale else allv)
            allv = allv[np.isfinite(allv)]
            s_vmin, s_vmax = np.percentile(allv, surrounding_clim_percentiles)
        if s_vmax <= s_vmin:
            s_vmax = s_vmin + 1e-6

    fig = plt.figure(figsize=(12.2 + 1.85 * max(n_samples - 1, 0), 7.0), facecolor="black")
    # computed_zorder=False: disable auto depth-sort so our manual zorder values
    # are respected, keeping profiles drawn on top of the cloud at this view angle.
    ax = fig.add_subplot(111, projection="3d", facecolor="black",
                         computed_zorder=False)
    ax.set_position((0.01, 0.02, 0.86, 0.90))

    centers = np.arange(n_samples, dtype=float) * cylinder_separation
    centers -= centers.mean()

    phi = np.linspace(0.0, 2.0 * np.pi, 360)
    rings = cylinder_radius * np.linspace(1.0 / 6.0, 1.0, 6)

    x_selected = x_array[selected]
    
    # --- Use los_range if provided, otherwise fallback to data limits ---
    if los_range is not None:
        x_min, x_max = float(los_range[0]), float(los_range[1])
    else:
        x_min = float(np.nanmin(x_selected))
        x_max = float(np.nanmax(x_selected))
        
    x_span = x_max - x_min
    if x_span <= 0:
        x_span = 1.0; x_min -= 0.5; x_max += 0.5

    axis_z = -cylinder_radius * (1.0 + x_axis_pad)
    tick_dx = cylinder_radius * x_axis_tick_size

    for i_sample, x0 in enumerate(centers):
        x_los = np.asarray(x_array[selected[i_sample]], dtype=float)

        # ---- NEW: glowing surrounding-field cloud (drawn first, underneath) ----
        if clouds is not None and clouds[i_sample] is not None:
            _draw_field_cloud(
                ax, x0, clouds[i_sample], cmap=s_cmap, vmin=s_vmin, vmax=s_vmax,
                cylinder_radius=cylinder_radius, radius_fill=surrounding_radius_fill,
                logscale=surrounding_logscale, alpha_max=surrounding_alpha_max,
                alpha_gamma=surrounding_alpha_gamma, core_size=surrounding_core_size,
                softness=surrounding_softness, size_growth=surrounding_size_growth,
                alpha_falloff=surrounding_alpha_falloff, threshold=surrounding_threshold,
            )

        y_slices = np.linspace(x_min, x_max, int(n_y_slices))

        ax.plot(np.full_like(y_slices, x0), y_slices, np.zeros_like(y_slices),
                color="white", lw=1.6, alpha=0.9, zorder=2)

        for yy in y_slices:
            for rr in rings:
                ax.plot(x0 + rr * np.cos(phi), np.full_like(phi, yy), rr * np.sin(phi),
                        color="white", lw=0.85, alpha=0.18, zorder=2)

        tick_vals = np.linspace(x_min, x_max, int(n_x_ticks))
        ax.plot(np.full_like(tick_vals, x0), tick_vals, np.full_like(tick_vals, axis_z),
                color="white", lw=1.2, alpha=0.75, zorder=2)

        for tt in tick_vals:
            ax.plot([x0 - tick_dx, x0 + tick_dx], [tt, tt], [axis_z, axis_z],
                    color="white", lw=1.0, alpha=0.75, zorder=2)
            ax.text(x0 - factor_ticks_x * tick_dx, tt, axis_z - factor_ticks_z,
                    f"{tt:.2f}", color="white", fontsize=x_axis_fontsize, ha="left", va="center",
                    zorder=2)

        if x_axis_label is not None:
            ax.text(x0, float(np.nanmean(tick_vals)), axis_z - factor_x_label * cylinder_radius,
                    x_axis_label, color="white", fontsize=x_axis_fontsize + 2, ha="center", va="top",
                    zorder=2)

        for i_profile, lab in enumerate(labels):
            th = angles[lab]
            c = profile_color[lab]
            r = cylinder_radius * _scale_values(profiles[i_sample, i_profile], vmin, vrange,
                                                eps, field_logscale, positive_only)

            ax.plot(x0 + cylinder_radius * np.cos(th) * np.ones_like(y_slices), y_slices,
                    cylinder_radius * np.sin(th) * np.ones_like(y_slices),
                    color=c, lw=0.9, alpha=0.18, zorder=profile_zorder - 1)

            if glow:
                ax.plot(x0 + r * np.cos(th), x_los, r * np.sin(th), color=c,
                        lw=8.5, alpha=0.10, zorder=profile_zorder)
                ax.plot(x0 + r * np.cos(th), x_los, r * np.sin(th), color=c,
                        lw=4.2, alpha=0.30, zorder=profile_zorder)

            ax.plot(x0 + r * np.cos(th), x_los, r * np.sin(th), color=c,
                    lw=profile_linewidth, alpha=1.00, zorder=profile_zorder + 1)

            if show_profile_points:
                ax.scatter(x0 + r * np.cos(th), x_los, r * np.sin(th), color=c,
                           s=profile_point_size, alpha=1.0, depthshade=False,
                           zorder=profile_zorder + 2)

    margin = 1.22 * cylinder_radius
    z_min = min(-margin, axis_z - 0.35 * cylinder_radius)

    ax.set_xlim(np.min(centers) - margin, np.max(centers) + margin)
    ax.set_ylim(x_min - 0.05 * x_span, x_max + 0.05 * x_span)
    ax.set_zlim(z_min, margin)
    ax.set_box_aspect((np.ptp(ax.get_xlim()), np.ptp(ax.get_ylim()), np.ptp(ax.get_zlim())))
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    _dark_3d_style(ax, hide_ticks=True)
    ax.view_init(elev=view_elev, azim=view_azim)

    # ---- profile scale bar (unchanged) ----
    sax = fig.add_axes((0.815 * scale_box_x_factor, 0.335, 0.125, 0.37), facecolor="black")
    sax.set_xlim(0.0, 1.0); sax.set_ylim(0.0, 1.0); sax.set_xticks([]); sax.set_yticks([])
    for spine in sax.spines.values():
        spine.set_color("white"); spine.set_linewidth(1.1)
    sax.plot([0.28, 0.28], [0.20, 0.80], color="white", lw=2.0)
    for rr, vv in zip(np.linspace(0.0, 1.0, 5), np.linspace(vmin, vmax, 5)):
        yy = 0.20 + 0.60 * rr
        sax.plot([0.20, 0.36], [yy, yy], color="white", lw=1.5)
        sax.text(0.88, yy, f"{vv:.2f}", color="white", fontsize=17, ha="right", va="center")
    if field_units is not None:
        sax.text(scale_units_xy[0], scale_units_xy[1], field_units, color="white",
                 fontsize=22, ha="center", va="bottom", transform=sax.transAxes)

    # ---- NEW: colorbar for the surrounding field ----
    if clouds is not None and surrounding_colorbar:
        cax = fig.add_axes((0.065, 0.30, 0.018, 0.42))
        sm = cm.ScalarMappable(norm=Normalize(s_vmin, s_vmax), cmap=s_cmap)
        sm.set_array([])
        cb = fig.colorbar(sm, cax=cax)
        cb.ax.yaxis.set_tick_params(color="white", labelsize=12)
        cb.outline.set_edgecolor("white")
        plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color="white")
        lbl = surrounding_colorbar_label or (
            "log10 field" if surrounding_logscale else "field")
        cb.set_label(lbl, color="white", fontsize=14)

    handles = [Line2D([0], [0], color=profile_color[lab], lw=5, label=lab) for lab in labels]
    leg = fig.legend(handles=handles, loc="upper center",
                     bbox_to_anchor=(0.44, 0.965 * legend_y_factor),
                     ncols=min(len(labels), 6), frameon=True, facecolor="black",
                     edgecolor="white", framealpha=1.0, fontsize=15)
    for text in leg.get_texts():
        text.set_color("white")

    if show:
        plt.show()
    return fig, ax


# ============================================================
# Private helpers
# ============================================================

def _neon_colors(labels):
    palette = [
        "#00FFFF", "#FF00FF", "#39FF14", "#FFEA00", "#FF5F1F", "#9D00FF",
        "#00BFFF", "#FF1493", "#7FFF00", "#FF073A", "#00FF7F", "#FF3131",
    ]
    return {lab: palette[i % len(palette)] for i, lab in enumerate(labels)}


def _dark_3d_style(ax, tick_fs=13, hide_ticks=False):
    ax.set_facecolor("black")
    ax.grid(False)
    ax.tick_params(
        axis="both", which="major", labelsize=tick_fs, colors="white",
        length=0 if hide_ticks else 5, width=0 if hide_ticks else 1.0,
    )
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.fill = False
        axis.pane.set_edgecolor("black")
        axis.label.set_color("white")
        try:
            axis._axinfo["grid"]["linewidth"] = 0.0
            axis._axinfo["axisline"]["linewidth"] = 0.0
            axis._axinfo["tick"]["color"] = "white"
        except Exception:
            pass


def _box_edges(x0, x1, y0, y1, z0, z1):
    return [
        ((x0,y0,z0),(x1,y0,z0)), ((x0,y1,z0),(x1,y1,z0)),
        ((x0,y0,z1),(x1,y0,z1)), ((x0,y1,z1),(x1,y1,z1)),
        ((x0,y0,z0),(x0,y1,z0)), ((x1,y0,z0),(x1,y1,z0)),
        ((x0,y0,z1),(x0,y1,z1)), ((x1,y0,z1),(x1,y1,z1)),
        ((x0,y0,z0),(x0,y0,z1)), ((x1,y0,z0),(x1,y0,z1)),
        ((x0,y1,z0),(x0,y1,z1)), ((x1,y1,z0),(x1,y1,z1)),
    ]


def _scaled_range(values, eps=1e-300, logscale=True, positive_only=False):
    values = np.asarray(values)
    mask = np.isfinite(values)
    if positive_only:
        mask &= values > 0
    data = values[mask]
    if data.size == 0:
        return (np.log10(eps), np.log10(eps) + 1.0, 1.0) if logscale else (0.0, 1.0, 1.0)
    data = np.log10(np.maximum(data, eps)) if logscale else data
    vmin, vmax = np.nanmin(data), np.nanmax(data)
    return vmin, vmax, max(vmax - vmin, 1e-30)


def _scale_values(values, vmin, vrange, eps=1e-300, logscale=True, positive_only=False):
    values = np.asarray(values)
    out = np.full(values.shape, vmin, dtype=float)
    mask = np.isfinite(values)
    if positive_only:
        mask &= values > 0
    out[mask] = np.log10(np.maximum(values[mask], eps)) if logscale else values[mask]
    return np.clip((out - vmin) / vrange, 0.0, 1.0)


def _resample_path(path, n_target):
    path = np.asarray(path)
    if path.shape[0] == n_target:
        return path
    old_t = np.linspace(0, 1, path.shape[0])
    new_t = np.linspace(0, 1, n_target)
    return np.column_stack([np.interp(new_t, old_t, path[:, i]) for i in range(3)])


def _local_basis(path):
    los = np.asarray(path[-1] - path[0], dtype=float)
    los /= np.linalg.norm(los) + 1e-30
    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(los, ref)) > 0.95:
        ref = np.array([1.0, 0.0, 0.0])
    e1 = np.cross(los, ref);  e1 /= np.linalg.norm(e1) + 1e-30
    e2 = np.cross(los, e1);   e2 /= np.linalg.norm(e2) + 1e-30
    return e1, e2


def _adaptive_tick_labels(vals, max_decimals=3):
    vals = np.asarray(vals, dtype=float)
    if vals.size == 0:
        return []
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return ["" for _ in vals]
    if finite.size == 1:
        vmax = np.abs(finite[0])
        return ([f"{finite[0]:.0f}"] if vmax >= 100 else
                [f"{finite[0]:.1f}"] if vmax >= 10 else
                [f"{finite[0]:.2f}"])
    step = np.nanmedian(np.abs(np.diff(finite))) if finite.size > 1 else 1.0
    vmax = np.nanmax(np.abs(finite))
    if vmax >= 100:   fmt = "{:.0f}"
    elif vmax >= 10:  fmt = "{:.1f}"
    elif step >= 1:   fmt = "{:.1f}"
    elif step >= 0.1: fmt = "{:.2f}"
    else:             fmt = "{:." + str(max_decimals) + "f}"
    return [fmt.format(v) for v in vals]


def _auto_axis_padding(tick_labels, axis_tick_fontsize, axis_label_fontsize,
                       tick_pad=None, label_pad=None,
                       tick_pad_factor=0.45, label_pad_factor=1.35):
    max_chars = max((len(str(t)) for t in tick_labels), default=4)
    if tick_pad is None:
        tick_pad = tick_pad_factor * axis_tick_fontsize + 0.35 * max_chars
    if label_pad is None:
        label_pad = (tick_pad + label_pad_factor * axis_tick_fontsize
                     + 0.45 * axis_label_fontsize + 0.45 * max_chars)
    return float(tick_pad), float(label_pad)


def _as_axis_tuple(value, name):
    if np.isscalar(value):
        return (value, value, value)
    if len(value) != 3:
        raise ValueError(f"{name} must be a scalar or a length-3 tuple.")
    return tuple(value)


def _smooth_grid_fourier(grid, sigma_vox):
    """Periodic Gaussian smoothing of a 3D grid via FFT (sigma in voxel units).

    This is the volume-rendering trick from bacco's ``np_convolve_gaussian``:
    multiply by a Gaussian in Fourier space.  Turns shot-noisy point samples
    into a smooth, continuous field.  Assumes a periodic (cosmological) box.
    """
    if sigma_vox is None or sigma_vox <= 0:
        return grid
    g = np.where(np.isfinite(grid), grid, 0.0).astype(np.float64)
    nx, ny, nz = g.shape
    gk = np.fft.rfftn(g)
    kx = np.fft.fftfreq(nx)[:, None, None]
    ky = np.fft.fftfreq(ny)[None, :, None]
    kz = np.fft.rfftfreq(nz)[None, None, :]
    k2 = (2.0 * np.pi) ** 2 * (kx * kx + ky * ky + kz * kz)
    gk *= np.exp(-0.5 * k2 * float(sigma_vox) ** 2)
    return np.fft.irfftn(gk, s=(nx, ny, nz))


def _eye_direction(elev, azim):
    """Unit vector pointing from the scene toward the matplotlib 3D camera."""
    el = np.deg2rad(elev)
    az = np.deg2rad(azim)
    return np.array([np.cos(el) * np.cos(az),
                     np.cos(el) * np.sin(az),
                     np.sin(el)])


# ============================================================
# Public: sample + smooth a scalar field over the full box
# ============================================================

def sample_box_cloud(
    quadri_vectors_one_skewer,
    field_fn,
    box_size,
    *,
    component=0,
    n=48,
    a_index=0,
    xyz_index=(1, 2, 3),
    log10=True,
    smoothing=1.0,
    jitter=0.5,
    seed=42,
):
    """Sample a scalar field on a regular grid over the full box, smooth it, and
    return a flat point cloud ``{x, y, z, value}`` in raw world coordinates.

    Pass the result directly as ``surrounding_field=cloud``.

    Parameters
    ----------
    quadri_vectors_one_skewer : (N_ppsk, 4) array
        Any single skewer — used only to read the scale factor at the midpoint.
    field_fn : callable
        ``(N, 4) -> (N,)`` or ``(N_species, N)``, e.g. ``fields.number_density``.
    box_size : float
        Full side length of the cubic box, same units as ``points``.  Must match
        the ``box_size`` passed to ``plot_3d_neon_ridge_cube``.
    component : int
        Species row when ``field_fn`` returns ``(N_species, N)`` (0 = HI).
    n : int
        Grid nodes per side; total evaluations = n**3 in one vectorised call.
        n=48 → ~110k pts (fast); n=64 → ~260k pts (more detail).
    a_index, xyz_index : int / sequence
        Quadri-vector column layout.  Default ``[a, x, y, z]``.
        Use ``a_index=3, xyz_index=(0,1,2)`` for ``[x, y, z, a]``.
    log10 : bool
        Store log10 of the (smoothed) field — recommended for densities.
    smoothing : float
        Gaussian smoothing sigma in VOXEL units, applied to the LINEAR field via
        FFT before taking the log.  0 disables.  ~0.8–1.5 gives a smooth nebula.
    jitter : float
        Sub-voxel position jitter (fraction of a cell) to suppress lattice
        aliasing in the scatter.  0 keeps points exactly on the grid.
    seed : int
        RNG seed for the jitter.
    """
    quad = np.asarray(quadri_vectors_one_skewer, dtype=np.float64)
    xyz_index = list(xyz_index)
    a0 = float(quad[quad.shape[0] // 2, a_index])

    h   = box_size / n
    lin = np.linspace(0.5 * h, box_size - 0.5 * h, n)
    Xg, Yg, Zg = np.meshgrid(lin, lin, lin, indexing="ij")   # regular grid

    # --- single vectorised field evaluation on the regular grid ----------
    N = n ** 3
    quad_grid = np.empty((N, 4), dtype=np.float64)
    quad_grid[:, a_index]      = a0
    quad_grid[:, xyz_index[0]] = Xg.ravel()
    quad_grid[:, xyz_index[1]] = Yg.ravel()
    quad_grid[:, xyz_index[2]] = Zg.ravel()

    out = np.asarray(field_fn(quad_grid))
    if out.ndim == 2:
        out = out[component]
    field = out.astype(np.float64).reshape(n, n, n)          # LINEAR density

    # --- Fourier-space Gaussian smoothing (on the linear field) ----------
    field = _smooth_grid_fourier(field, smoothing)
    field = np.maximum(field, 0.0)                           # kill tiny negatives

    if log10:
        with np.errstate(divide="ignore", invalid="ignore"):
            value = np.log10(np.where(field > 0, field, np.nan))
    else:
        value = field

    # --- sub-voxel jitter on positions only (not values) -----------------
    if jitter and jitter > 0:
        rng = np.random.default_rng(seed)
        disp = (rng.random((3, n, n, n)) - 0.5) * float(jitter) * h
        Xg = Xg + disp[0]; Yg = Yg + disp[1]; Zg = Zg + disp[2]

    return dict(x=Xg.ravel(), y=Yg.ravel(), z=Zg.ravel(),
                value=value.ravel(), box_size=box_size)


# ============================================================
# Private: render the cloud as a partially transparent volume
# ============================================================

def _draw_box_cloud(
    ax, cloud, *, cmap, vmin, vmax,
    alpha_max, alpha_gamma, core_size, threshold,
    glow, glow_size_mult, glow_alpha_mult,
    elev, azim, to_plot_x, to_plot_y, to_plot_z,
):
    """Render the field cloud with a transfer function and depth-sorted
    back-to-front alpha compositing (correct partial transparency).
    """
    val = np.asarray(cloud["value"], dtype=float)
    vr  = max(vmax - vmin, 1e-30)
    scaled = np.clip((val - vmin) / vr, 0.0, 1.0)            # colour/opacity coord

    keep = np.isfinite(scaled) & (scaled > threshold)
    if not np.any(keep):
        return

    px = to_plot_x(np.asarray(cloud["x"], dtype=float)[keep])
    py = to_plot_y(np.asarray(cloud["y"], dtype=float)[keep])
    pz = to_plot_z(np.asarray(cloud["z"], dtype=float)[keep])
    sc = scaled[keep]

    # ---- depth sort: far -> near along the camera direction -------------
    eye   = _eye_direction(elev, azim)
    depth = px * eye[0] + py * eye[1] + pz * eye[2]
    order = np.argsort(depth)                                # ascending = far first
    px, py, pz, sc = px[order], py[order], pz[order], sc[order]

    # ---- transfer function: colour (LogNorm-like) + opacity (gamma) -----
    rgba = cmap(sc)                                          # (M, 4)
    rgba[:, 3] = np.clip(alpha_max * sc ** alpha_gamma, 0.0, 1.0)

    # ---- soft emission glow halo (drawn under the cores) ----------------
    if glow:
        grgba = rgba.copy()
        grgba[:, 3] = np.clip(rgba[:, 3] * glow_alpha_mult, 0.0, 1.0)
        ax.scatter(px, py, pz, color=grgba, s=core_size * glow_size_mult,
                   marker="o", linewidths=0, depthshade=False, zorder=1)

    # ---- bright cores ---------------------------------------------------
    ax.scatter(px, py, pz, color=rgba, s=core_size,
               marker="o", linewidths=0, depthshade=False, zorder=1)


# ============================================================
# Public: 3D neon ridge-cube plot
# ============================================================

def plot_3d_neon_ridge_cube(
    points,
    y_dict,
    skewer_index=0,
    normalization="global",
    field_logscale=True,
    profile_angle_offset_deg=-90.0,
    ridge_scale=1.2,
    elev=13,
    azim=-70,
    normalize_scales=True,
    x_label=None,
    y_label=None,
    z_label=None,
    x_label_pad=None,
    y_label_pad=None,
    z_label_pad=None,
    x_tick_pad=None,
    y_tick_pad=None,
    z_tick_pad=None,
    axis_label_fontsize=16,
    n_axis_ticks=4,
    axis_tick_fontsize=None,
    tick_label_max_decimals=3,
    box_size=None,
    show=True,
    # ---- rendered background density field ----
    surrounding_field=None,
    surrounding_cmap="magma",
    surrounding_clim=None,
    surrounding_clim_percentiles=(45.0, 99.7),
    surrounding_alpha_max=0.35,
    surrounding_alpha_gamma=1.3,
    surrounding_core_size=6.0,
    surrounding_threshold=0.04,
    surrounding_glow=True,
    surrounding_glow_size_mult=3.5,
    surrounding_glow_alpha_mult=0.30,
):
    """3D neon ridge-cube for one or more skewers, with an optional rendered,
    partially transparent full-box background density field.

    Original parameters
    -------------------
    points : (N_samples, N_x, 3)
        Ridgeline coordinates: [..., 0] = box x, [..., 1] = LoS, [..., 2] = box z.
    y_dict : dict[str, (N_samples, N_x) array]
        Field profiles keyed by species/line label.
    skewer_index : int
    normalization : "global" | "skewer"
    field_logscale : bool
    profile_angle_offset_deg, ridge_scale, elev, azim : float
    normalize_scales : bool
        Display in normalised [-0.5, 0.5]^3 while keeping raw tick labels.
    x/y/z_label, x/y/z_label_pad, x/y/z_tick_pad : str / float
    axis_label_fontsize, n_axis_ticks, axis_tick_fontsize, tick_label_max_decimals
    box_size : float or None
        If given, box edges run [0, box_size]^3.  Match this to ``sample_box_cloud``.
    show : bool

    Rendered-background parameters
    ------------------------------
    surrounding_field : dict or None
        Cloud dict from ``sample_box_cloud`` (already smoothed / log-scaled).
        None disables the background.
    surrounding_cmap : str
        Matplotlib colormap ("magma", "inferno", "plasma", …).
    surrounding_clim : (vmin, vmax) or None
        Colour limits in the cloud's value units (log10 if you sampled with
        log10=True).  None → derived from ``surrounding_clim_percentiles``.
    surrounding_clim_percentiles : (lo, hi)
        Percentiles for automatic colour limits.  Raise the low end (e.g. 60) to
        darken voids and emphasise filaments.
    surrounding_alpha_max : float
        Peak opacity of the densest gas (0–1).  Keep < 1 for a see-through volume.
    surrounding_alpha_gamma : float
        Opacity curve exponent: alpha = alpha_max * scaled**gamma.  >1 makes faint
        gas more transparent (sharper filaments); <1 fills the volume more.
    surrounding_core_size : float
        Core splat size (pt²).
    surrounding_threshold : float
        Drop points whose scaled value is below this (transparent voids).
    surrounding_glow : bool
        Add a soft, larger, low-opacity emission halo under the cores.
    surrounding_glow_size_mult : float
        Halo size relative to the core size.
    surrounding_glow_alpha_mult : float
        Halo opacity relative to the core opacity.
    """
    points = np.asarray(points)
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError("Expected points with shape (N_samples, N_x, 3).")
    if not isinstance(y_dict, dict) or len(y_dict) == 0:
        raise ValueError("Expected non-empty y_dict mapping labels to arrays.")

    labels   = list(y_dict.keys())
    y_arrays = {lab: np.asarray(y_dict[lab]) for lab in labels}
    for lab, arr in y_arrays.items():
        if arr.shape != points.shape[:2]:
            raise ValueError(
                f"y_dict[{lab!r}] shape {arr.shape} != points.shape[:2] {points.shape[:2]}.")

    selected = int(np.clip(skewer_index, 0, points.shape[0] - 1))

    range_values = (
        np.asarray([y_arrays[lab][selected] for lab in labels])
        if normalization == "skewer"
        else np.concatenate([y_arrays[lab].ravel() for lab in labels])
    )
    if normalization not in ("global", "skewer"):
        raise ValueError("normalization must be 'global' or 'skewer'.")

    vmin, vmax, vrange = _scaled_range(range_values, eps=1e-300,
                                       logscale=field_logscale, positive_only=False)
    eps, positive_only = 1e-300, False

    colors        = _neon_colors(labels)
    profile_color = {lab: colors[lab] for lab in labels}
    angles = {
        lab: np.deg2rad(profile_angle_offset_deg) + 2 * np.pi * i / len(labels)
        for i, lab in enumerate(labels)
    }

    p_raw = points.astype(float, copy=False)

    if box_size is not None:
        mins_raw = np.array([0.0, 0.0, 0.0])
        maxs_raw = np.array([float(box_size)] * 3)
    else:
        mins_raw = np.nanmin(p_raw.reshape(-1, 3), axis=0)
        maxs_raw = np.nanmax(p_raw.reshape(-1, 3), axis=0)

    centers_raw = 0.5 * (mins_raw + maxs_raw)
    spans_raw   = maxs_raw - mins_raw
    safe_spans  = np.where(spans_raw > 0, spans_raw, 1.0)

    nx_ticks, ny_ticks, nz_ticks = [int(v) for v in _as_axis_tuple(n_axis_ticks, "n_axis_ticks")]
    x_ticks_raw = np.linspace(mins_raw[0], maxs_raw[0], nx_ticks)
    y_ticks_raw = np.linspace(mins_raw[1], maxs_raw[1], ny_ticks)
    z_ticks_raw = np.linspace(mins_raw[2], maxs_raw[2], nz_ticks)

    if normalize_scales:
        xz_extent = max(safe_spans[0], safe_spans[2], 1.0)

        def raw_to_plot_x(x): return (np.asarray(x) - centers_raw[0]) / xz_extent
        def raw_to_plot_y(y): return (np.asarray(y) - centers_raw[1]) / safe_spans[1]
        def raw_to_plot_z(z): return (np.asarray(z) - centers_raw[2]) / xz_extent

        p_plot = np.empty_like(p_raw, dtype=float)
        p_plot[..., 0] = raw_to_plot_x(p_raw[..., 0])
        p_plot[..., 1] = raw_to_plot_y(p_raw[..., 1])
        p_plot[..., 2] = raw_to_plot_z(p_raw[..., 2])

        x0, x1 = -0.5, 0.5
        y0, y1 = -0.5, 0.5
        z0, z1 = -0.5, 0.5
        extent  = 1.0

        x_ticks_plot = raw_to_plot_x(x_ticks_raw)
        y_ticks_plot = raw_to_plot_y(y_ticks_raw)
        z_ticks_plot = raw_to_plot_z(z_ticks_raw)

    else:
        def raw_to_plot_x(x): return np.asarray(x, dtype=float)
        def raw_to_plot_y(y): return np.asarray(y, dtype=float)
        def raw_to_plot_z(z): return np.asarray(z, dtype=float)

        p_plot = p_raw.copy()
        x0, y0, z0 = mins_raw
        x1, y1, z1 = maxs_raw
        if x1 <= x0: x0, x1 = x0 - 0.5, x1 + 0.5
        if y1 <= y0: y0, y1 = y0 - 0.5, y1 + 0.5
        if z1 <= z0: z0, z1 = z0 - 0.5, z1 + 0.5
        extent = max(x1 - x0, y1 - y0, z1 - z0, 1.0)

        x_ticks_plot = x_ticks_raw
        y_ticks_plot = y_ticks_raw
        z_ticks_plot = z_ticks_raw

    field_scale = 0.06 * ridge_scale * extent

    # ---- background colour range ----
    s_cmap = s_vmin = s_vmax = None
    if surrounding_field is not None:
        s_cmap = plt.get_cmap(surrounding_cmap)
        val = np.asarray(surrounding_field["value"], dtype=float)
        if surrounding_clim is not None:
            s_vmin, s_vmax = float(surrounding_clim[0]), float(surrounding_clim[1])
        else:
            finite = val[np.isfinite(val)]
            s_vmin, s_vmax = np.percentile(finite, surrounding_clim_percentiles)
        if s_vmax <= s_vmin:
            s_vmax = s_vmin + 1e-6

    # ---- figure ----
    fig = plt.figure(figsize=(10.2, 8.2), facecolor="black")
    ax  = fig.add_subplot(111, projection="3d", facecolor="black",
                          computed_zorder=False)
    ax.set_position((0.01, 0.01, 0.98, 0.98))

    # ---- rendered background density (drawn first, behind everything) ----
    if surrounding_field is not None:
        _draw_box_cloud(
            ax, surrounding_field,
            cmap=s_cmap, vmin=s_vmin, vmax=s_vmax,
            alpha_max=surrounding_alpha_max,
            alpha_gamma=surrounding_alpha_gamma,
            core_size=surrounding_core_size,
            threshold=surrounding_threshold,
            glow=surrounding_glow,
            glow_size_mult=surrounding_glow_size_mult,
            glow_alpha_mult=surrounding_glow_alpha_mult,
            elev=elev, azim=azim,
            to_plot_x=raw_to_plot_x,
            to_plot_y=raw_to_plot_y,
            to_plot_z=raw_to_plot_z,
        )

    # ---- box edges ----
    for a, b in _box_edges(x0, x1, y0, y1, z0, z1):
        ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color="white", lw=2.2)

    # ---- ridgelines ----
    for sk_idx, path in enumerate(p_plot):
        e1, e2 = _local_basis(path)
        for lab in labels:
            values = y_arrays[lab][sk_idx]
            p      = _resample_path(path, values.size)
            amp    = field_scale * _scale_values(
                values, vmin, vrange, eps, field_logscale, positive_only)
            direction = np.cos(angles[lab]) * e1 + np.sin(angles[lab]) * e2
            ridge = p + amp[:, None] * direction[None, :]
            c     = profile_color[lab]
            ax.plot(ridge[:, 0], ridge[:, 1], ridge[:, 2], color=c, lw=8.0, alpha=0.12)
            ax.plot(ridge[:, 0], ridge[:, 1], ridge[:, 2], color=c, lw=4.0, alpha=0.35)
            ax.plot(ridge[:, 0], ridge[:, 1], ridge[:, 2], color=c, lw=1.0, alpha=1.0)

    # ---- axis limits ----
    xm, ym, zm = 0.5*(x0+x1), 0.5*(y0+y1), 0.5*(z0+z1)
    half = 0.5 * extent + field_scale
    ax.set_xlim(xm - half, xm + half)
    ax.set_ylim(ym - half, ym + half)
    ax.set_zlim(zm - half, zm + half)
    ax.set_box_aspect((1, 1, 1))

    # ---- ticks / labels ----
    x_tick_labels = _adaptive_tick_labels(x_ticks_raw, max_decimals=tick_label_max_decimals)
    y_tick_labels = _adaptive_tick_labels(y_ticks_raw, max_decimals=tick_label_max_decimals)
    z_tick_labels = _adaptive_tick_labels(z_ticks_raw, max_decimals=tick_label_max_decimals)

    ax.set_xticks(x_ticks_plot); ax.set_xticklabels(x_tick_labels)
    ax.set_yticks(y_ticks_plot); ax.set_yticklabels(y_tick_labels)
    ax.set_zticks(z_ticks_plot); ax.set_zticklabels(z_tick_labels)

    _dark_3d_style(ax)

    if axis_tick_fontsize is None:
        axis_tick_fontsize = max(axis_label_fontsize - 5, 8)

    x_tick_pad, x_label_pad = _auto_axis_padding(
        x_tick_labels, axis_tick_fontsize, axis_label_fontsize,
        tick_pad=x_tick_pad, label_pad=x_label_pad)
    y_tick_pad, y_label_pad = _auto_axis_padding(
        y_tick_labels, axis_tick_fontsize, axis_label_fontsize,
        tick_pad=y_tick_pad, label_pad=y_label_pad)
    z_tick_pad, z_label_pad = _auto_axis_padding(
        z_tick_labels, axis_tick_fontsize, axis_label_fontsize,
        tick_pad=z_tick_pad, label_pad=z_label_pad)

    ax.tick_params(axis="x", colors="white", labelsize=axis_tick_fontsize, pad=x_tick_pad)
    ax.tick_params(axis="y", colors="white", labelsize=axis_tick_fontsize, pad=y_tick_pad)
    ax.tick_params(axis="z", colors="white", labelsize=axis_tick_fontsize, pad=z_tick_pad)

    ax.set_xlabel(x_label or "", labelpad=x_label_pad)
    ax.set_ylabel(y_label or "", labelpad=y_label_pad)
    ax.set_zlabel(z_label or "", labelpad=z_label_pad)

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.label.set_color("white")
        axis.label.set_fontsize(axis_label_fontsize)

    ax.view_init(elev=elev, azim=azim)

    if show:
        plt.show()
    return fig, ax






def generate_all_simulation_plots(
    # --- Core Data Objects ---
    line_bundle,
    skewer_bundle,
    fields,
    
    # --- Cosmology & Geometry Parameters ---
    L_box,
    
    # --- Grids & Optical Depths ---
    velocities_kms,
    wavelengths_per_line_A,
    velocities_per_line_kms,
    optical_depth,
    optical_depth_wrapped,
    
    # --- Scale Factors & Limits ---
    a_dagger,
    a_em,
    a_obs,
    v_min: float,
    v_max: float,
    
    # --- optional_parameters ---
    ii_cosmology = 0,
    ii_skewer = 0,
    
    # --- Plotting Aesthetics ---
    view_elev: float = 20.0,
    view_azim: float = -43.0,
    los_range=None,
    show: bool = False,
    
    plot_cylinders_R001=False,
    plot_cube=False,
    plot_normalized_velocity_units=True,
    plot_wrapped_velocity_units=True,
):
    """
    Generates all number density and transmittance plots for the simulation.
    
    Returns
    -------
    figures : list of matplotlib.figure.Figure
        A list containing all generated figure objects for further manipulation or saving.
    """
    figures = []

    # -------------------------------------------------------- #
    # ----------- PLOTS FOR NUMBER DENSITY FIELDS ------------ #
    # -------------------------------------------------------- #


    ii_line = 0
    key_line_plot = line_bundle.lines[ii_line].species

    quad_all = np.asarray(skewer_bundle.quadri_vectors[0])   # (N_sk, N_x, 4)
    points   = quad_all[..., 1:4]          

    field_plot = np.stack([np.asarray(fields.number_density(quad_all[i]))[ii_line] for i in range(quad_all.shape[0])])       # (N_sk, N_x)
    y_dict = {key_line_plot: field_plot}
    colors = {key_line_plot: "#00FFFF"}
    
    skewer_index, n_samples_plot = 0, 1
    
    if plot_cylinders_R001:
        # ------------------ plot cylinder HI (R=0.01) ----------------------- #
        R_plot = 0.01
        clouds = [
            sample_cylinder_cloud(
                quad_all[i], fields.number_density, R_plot, los_axis=1,
                component=ii_line, n_axis=512, n_perp=512,
                a_index=0, xyz_index=(1, 2, 3), length_fraction=1.0, seed=0,
            ) for i in range(skewer_index, skewer_index + n_samples_plot)
        ]

        fig, ax = plot_3d_radar_cylinder(
            points=points, y_dict=y_dict, los_axis=1, skewer_index=skewer_index, n_samples_plot=n_samples_plot,
            cylinder_radius=13.0, cylinder_separation=26.0, field_logscale=True, normalization="global",
            field_units=r"$\log_{10}\left(n \, \left[\mathrm{cm}^{-3}\right] \right)$",
            x_axis_label=r"$\mathrm{LoS}_{\parallel}\,[{\rm cMpc}]$", view_elev=view_elev, view_azim=view_azim, los_range=los_range,
            surrounding_field=clouds, surrounding_cmap="berlin", surrounding_logscale=True,
            surrounding_clim=(np.min(np.log10(field_plot)), np.max(np.log10(field_plot))), surrounding_radius_fill=0.1,
            surrounding_alpha_max=0.3, surrounding_alpha_gamma=1.2,
            surrounding_core_size=1.0, surrounding_softness=2, surrounding_size_growth=4.0, surrounding_alpha_falloff=0.7,
            surrounding_colorbar=True, surrounding_colorbar_label=r"$\log_{10}\left(n \, \left[\mathrm{cm}^{-3}\right] \right)$",
            colors=colors, show=show
        )
        figures.append(fig)

    # ------------------ plot cylinder HI (R=5.0) ----------------------- #
    R_plot = 5.0
    clouds = [
        sample_cylinder_cloud(
            quad_all[i], fields.number_density, R_plot, los_axis=1,
            component=ii_line, n_axis=512, n_perp=512,
            a_index=0, xyz_index=(1, 2, 3), length_fraction=1.0, seed=0,
        ) for i in range(skewer_index, skewer_index + n_samples_plot)
    ]

    fig, ax = plot_3d_radar_cylinder(
        points=points, y_dict=y_dict, los_axis=1, skewer_index=skewer_index, n_samples_plot=n_samples_plot,
        cylinder_radius=13.0, cylinder_separation=26.0, field_logscale=True, normalization="global",
        field_units=r"$\log_{10}\left(n \, \left[\mathrm{cm}^{-3}\right] \right)$",
        x_axis_label=r"$\mathrm{LoS}_{\parallel}\,[{\rm cMpc}]$", view_elev=view_elev, view_azim=view_azim, los_range=los_range,
        surrounding_field=clouds, surrounding_cmap="berlin", surrounding_logscale=True,
        surrounding_clim=(np.min(np.log10(field_plot)), np.max(np.log10(field_plot))), surrounding_radius_fill=1.0,
        surrounding_alpha_max=0.15, surrounding_alpha_gamma=1.2,
        surrounding_core_size=3.0, surrounding_softness=2, surrounding_size_growth=4.0, surrounding_alpha_falloff=0.7,
        surrounding_colorbar=True, surrounding_colorbar_label=r"$\log_{10}\left(n \, \left[\mathrm{cm}^{-3}\right] \right)$",
        colors=colors, show=show
    )
    figures.append(fig)

    if line_bundle.n_lines > 1:
        
        ii_line = 1
        key_line_plot = line_bundle.lines[ii_line].species

        quad_all = np.asarray(skewer_bundle.quadri_vectors[0])
        points   = quad_all[..., 1:4]          

        field_plot = np.stack([np.asarray(fields.number_density(quad_all[i]))[ii_line] for i in range(quad_all.shape[0])])
        y_dict = {key_line_plot: field_plot}
        colors = {key_line_plot: "#FF00FF"}

        skewer_index, n_samples_plot = 0, 1
        
        if plot_cylinders_R001:
            # ------------------ plot cylinder SiIII (R=0.01) ----------------------- #
            R_plot = 0.01
            clouds = [
                sample_cylinder_cloud(
                    quad_all[i], fields.number_density, R_plot, los_axis=1,
                    component=ii_line, n_axis=512, n_perp=512,
                    a_index=0, xyz_index=(1, 2, 3), length_fraction=1.0, seed=0,
                ) for i in range(skewer_index, skewer_index + n_samples_plot)
            ]
            fig, ax = plot_3d_radar_cylinder(
                points=points, y_dict=y_dict, los_axis=1, skewer_index=skewer_index, n_samples_plot=n_samples_plot,
                cylinder_radius=13.0, cylinder_separation=26.0, field_logscale=True, normalization="global",
                field_units=r"$\log_{10}\left(n \, \left[\mathrm{cm}^{-3}\right] \right)$",
                x_axis_label=r"$\mathrm{LoS}_{\parallel}\,[{\rm cMpc}]$", view_elev=view_elev, view_azim=view_azim, los_range=los_range,
                surrounding_field=clouds, surrounding_cmap="berlin", surrounding_logscale=True,
                surrounding_clim=(np.min(np.log10(field_plot)), np.max(np.log10(field_plot))), surrounding_radius_fill=0.1,
                surrounding_alpha_max=0.3, surrounding_alpha_gamma=1.2,
                surrounding_core_size=1.0, surrounding_softness=2, surrounding_size_growth=4.0, surrounding_alpha_falloff=0.7,
                surrounding_colorbar=True, surrounding_colorbar_label=r"$\log_{10}\left(n \, \left[\mathrm{cm}^{-3}\right] \right)$",
                colors=colors, show=show
            )
            figures.append(fig)

        # ------------------ plot cylinder SiIII (R=5.0) ----------------------- #
        R_plot = 5.0
        clouds = [
            sample_cylinder_cloud(
                quad_all[i], fields.number_density, R_plot, los_axis=1,
                component=ii_line, n_axis=512, n_perp=512,
                a_index=0, xyz_index=(1, 2, 3), length_fraction=1.0, seed=0,
            ) for i in range(skewer_index, skewer_index + n_samples_plot)
        ]

        fig, ax = plot_3d_radar_cylinder(
            points=points, y_dict=y_dict, los_axis=1, skewer_index=skewer_index, n_samples_plot=n_samples_plot,
            cylinder_radius=13.0, cylinder_separation=26.0, field_logscale=True, normalization="global",
            field_units=r"$\log_{10}\left(n \, \left[\mathrm{cm}^{-3}\right] \right)$",
            x_axis_label=r"$\mathrm{LoS}_{\parallel}\,[{\rm cMpc}]$", view_elev=view_elev, view_azim=view_azim, los_range=los_range,
            surrounding_field=clouds, surrounding_cmap="berlin", surrounding_logscale=True,
            surrounding_clim=(np.min(np.log10(field_plot)), np.max(np.log10(field_plot))), surrounding_radius_fill=1.0,
            surrounding_alpha_max=0.15, surrounding_alpha_gamma=1.2,
            surrounding_core_size=3.0, surrounding_softness=2, surrounding_size_growth=4.0, surrounding_alpha_falloff=0.7,
            surrounding_colorbar=True, surrounding_colorbar_label=r"$\log_{10}\left(n \, \left[\mathrm{cm}^{-3}\right] \right)$",
            colors=colors, show=show
        )
        figures.append(fig)

    # ------------------ plot box ----------------------- #
    
    if plot_cube:
        cloud = sample_box_cloud(
            skewer_bundle.quadri_vectors[ii_cosmology, 0],
            fields.number_density, box_size=L_box,
            component=0, n=150, a_index=0, xyz_index=(1, 2, 3), log10=True,
        )

        sampled_fields = fields.sample(skewer_bundle.quadri_vectors[ii_cosmology], skewer_bundle.los_directions)
        plot_dict = {key: jnp.log10(sampled_fields[key]['number_density']) for key in sampled_fields.keys()}

        fig, ax = plot_3d_neon_ridge_cube(
            points=points, y_dict=plot_dict, normalization="global", field_logscale=False, normalize_scales=False,
            x_label=r"$x\,[{\rm cMpc}]$", y_label=r"$r_{\parallel}\,[{\rm cMpc}]$", z_label=r"$z\,[{\rm cMpc}]$",
            axis_label_fontsize=22, x_label_pad=25.0, y_label_pad=25.0, z_label_pad=25.0,
            n_axis_ticks=(4, 4, 4), ridge_scale=1.0, box_size=L_box, elev=view_elev, azim=view_azim,
            surrounding_field=cloud, surrounding_cmap="berlin", surrounding_clim_percentiles=(45.0, 99.7),
            surrounding_alpha_max=0.5, surrounding_alpha_gamma=2.0, surrounding_core_size=3.0, surrounding_threshold=0.1, surrounding_glow=True,
            surrounding_glow_size_mult=3.5, surrounding_glow_alpha_mult=0.30, show=show
        )
        figures.append(fig)

    # -------------------------------------------------------- #
    # --------------- PLOTS FOR TRANSMITTANCE ---------------- #
    # -------------------------------------------------------- #

    colors = _neon_colors(line_bundle.species_keys)
    linestyles = ['-', '--', ':', '-.']

    if plot_normalized_velocity_units:
        
        # --------------------------------------------- Normalized Velocity Units --------------------------------------------- #
        fig, ax = plt.subplots(1, 1, figsize=(14, 4.5), sharex=True, sharey=True, facecolor="black")
        ax.set_facecolor("black")

        kk = 0
        handles = []
        for ii, key_species in enumerate(line_bundle.species_keys):
            color = colors[key_species]
            for jj, key_line in enumerate(line_bundle.lines_for_species(key_species)):
                ls = linestyles[jj]
                
                ax.plot(velocities_kms, jnp.exp(-optical_depth[kk, ii_skewer]), color=color, lw=6, ls=ls, alpha=0.2, zorder=2)
                ax.plot(velocities_kms, jnp.exp(-optical_depth[kk, ii_skewer]), color=color, lw=1.5, ls=ls, alpha=0.9, zorder=3)
                
                handles.append(Line2D([0], [0], color=color, lw=4, ls=ls, label=line_bundle.lines[kk].id))
                kk += 1
                
        ax.set_xlim([v_min-100, v_max+100])
        ax.set_ylim([-0.05, 1.05])
        ax.axvline(0, color='white', lw=4, ls=':', zorder=1)
        ax.axvline(phys.c_kms*jnp.log(a_dagger / a_em), color='white', lw=4, ls=':', zorder=2)
                
        leg = fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncols=min(len(handles), 7),
            frameon=True, fontsize=16, handlelength=2.6, facecolor="black", edgecolor="white", framealpha=0.9,
        )
        for text in leg.get_texts():
            text.set_color("white")
                
        ax.tick_params(axis="both", which="major", labelsize=20, colors="white", length=6, width=1.1)
        for side in ["top", "right", "bottom", "left"]:
            ax.spines[side].set_color("white")
            ax.spines[side].set_linewidth(1.0)
            
        ax.set_ylabel(r"Transmittance $\; \left[\mathrm{adim}\right]$", fontsize=20, color="white")
        ax.set_xlabel(r"Normalized Velocity Units $v \; \left[\mathrm{km}/\mathrm{s}\right]$", fontsize=20, color="white")

        fig.tight_layout()
        figures.append(fig)
        if show:
            plt.show()

    # --------------------------------------------- Observed wavelength --------------------------------------------- #
    fig, ax = plt.subplots(1, 1, figsize=(14, 4.5), sharex=True, sharey=True, facecolor="black")
    ax.set_facecolor("black")

    kk = 0
    handles = []
    _vlines_rest = []
    _vlines_obs = []
    for ii, key_species in enumerate(line_bundle.species_keys):
        color = colors[key_species]
        for jj, key_line in enumerate(line_bundle.lines_for_species(key_species)):
            ls = linestyles[jj]
            
            ax.plot(wavelengths_per_line_A[kk], jnp.exp(-optical_depth[kk, ii_skewer]), color=color, lw=6, ls=ls, alpha=0.2, zorder=2)
            ax.plot(wavelengths_per_line_A[kk], jnp.exp(-optical_depth[kk, ii_skewer]), color=color, lw=1.5, ls=ls, alpha=0.9, zorder=3)
            
            _vrest = line_bundle.lines[kk].lambda0_A * (a_obs / a_dagger)
            _vlines_rest.append(_vrest)
            ax.axvline(_vrest, color=color, lw=4, ls=':', zorder=1)
            
            _vobs = line_bundle.lines[kk].lambda0_A * (a_obs / a_em)
            _vlines_obs.append(_vobs)
            ax.axvline(_vobs, color=color, lw=4, ls=':', zorder=2)
            
            handles.append(Line2D([0], [0], color=color, lw=4, ls=ls, label=line_bundle.lines[kk].id))
            kk += 1
            
    ax.set_ylim([-0.05, 1.05])

    arrow_props = dict(arrowstyle="->", color="white", lw=3, mutation_scale=25)
    bbox_props = dict(boxstyle="round,pad=0.3", facecolor="black", edgecolor="white", alpha=1.0)
    
    # --- Top Arrows (Rest) ---
    _y_arrow_position = 0.5
    _delta_arrow = (_vlines_rest[0] - _vlines_rest[-1]) / 2
    _x_mid = _vlines_rest[-1] + _delta_arrow
    ax.annotate("", xy=(_x_mid + _delta_arrow, _y_arrow_position), xytext=(_x_mid, _y_arrow_position), arrowprops=arrow_props, zorder=10)
    ax.annotate("", xy=(_x_mid - _delta_arrow, _y_arrow_position), xytext=(_x_mid, _y_arrow_position), arrowprops=arrow_props, zorder=10)
    ax.text(_x_mid, _y_arrow_position, f"{2 * _delta_arrow:.2f}" + r"$\; \AA$",
            ha='center', va='center', color='white', fontsize=12, bbox=bbox_props, zorder=10)
            
    # --- Bottom Arrows (Obs) ---
    _y_arrow_position = 0.2
    _delta_arrow = (_vlines_obs[0] - _vlines_obs[-1]) / 2
    _x_mid = _vlines_obs[-1] + _delta_arrow
    ax.annotate("", xy=(_x_mid + _delta_arrow, _y_arrow_position), xytext=(_x_mid, _y_arrow_position), arrowprops=arrow_props, zorder=10)
    ax.annotate("", xy=(_x_mid - _delta_arrow, _y_arrow_position), xytext=(_x_mid, _y_arrow_position), arrowprops=arrow_props, zorder=10)
    ax.text(_x_mid, _y_arrow_position, f"{2 * _delta_arrow:.2f}" + r"$\; \AA$",
            ha='center', va='center', color='white', fontsize=12, bbox=bbox_props, zorder=10)

    leg = fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncols=min(len(handles), 7),
        frameon=True, fontsize=16, handlelength=2.6, facecolor="black", edgecolor="white", framealpha=0.9,
    )
    for text in leg.get_texts():
        text.set_color("white")
            
    ax.tick_params(axis="both", which="major", labelsize=20, colors="white", length=6, width=1.1)
    for side in ["top", "right", "bottom", "left"]:
        ax.spines[side].set_color("white")
        ax.spines[side].set_linewidth(1.0)
        
    ax.set_ylabel(r"Transmittance $\; \left[\mathrm{adim}\right]$", fontsize=20, color="white")
    ax.set_xlabel(r"Observed wavelength $\lambda_\mathrm{obs} \; \left[\AA\right]$", fontsize=20, color="white")

    fig.tight_layout()
    figures.append(fig)
    if show:
        plt.show()
        
    # --------------------------------------------- Shifted Velocity Units --------------------------------------------- #
    fig, ax = plt.subplots(1, 1, figsize=(14, 4.5), sharex=True, sharey=True, facecolor="black")
    ax.set_facecolor("black")

    kk = 0
    handles = []
    _vlines_rest = []
    _vlines_obs = []
    for ii, key_species in enumerate(line_bundle.species_keys):
        color = colors[key_species]
        for jj, line in enumerate(line_bundle.lines_for_species(key_species)):
            ls = linestyles[jj]
            
            ax.plot(velocities_per_line_kms[kk], jnp.exp(-optical_depth[kk, ii_skewer]), color=color, lw=6, ls=ls, alpha=0.2, zorder=2)
            ax.plot(velocities_per_line_kms[kk], jnp.exp(-optical_depth[kk, ii_skewer]), color=color, lw=1.5, ls=ls, alpha=0.9, zorder=3)

            _delta_v_for_line = -line_bundle.delta_v_for_line(line.id)
            
            _vrest = _delta_v_for_line
            _vlines_rest.append(_vrest)
            ax.axvline(_vrest, color=color, lw=4, ls=':', zorder=1)
            
            _vobs = _delta_v_for_line + phys.c_kms * jnp.log(a_dagger / a_em)
            _vlines_obs.append(_vobs)
            ax.axvline(_vobs, color=color, lw=4, ls=':', zorder=2)
            
            handles.append(Line2D([0], [0], color=color, lw=4, ls=ls, label=line_bundle.lines[kk].id))
            kk += 1

    ax.set_xlim([v_min-100, v_max+100])
    ax.set_ylim([-0.05, 1.05])

    arrow_props = dict(arrowstyle="->", color="white", lw=3, mutation_scale=25)
    bbox_props = dict(boxstyle="round,pad=0.3", facecolor="black", edgecolor="white", alpha=1.0)
    
    # --- Top Arrows (Rest) ---
    _y_arrow_position = 0.5
    _delta_arrow = (_vlines_rest[0] - _vlines_rest[-1]) / 2
    _x_mid = _vlines_rest[-1] + _delta_arrow
    ax.annotate("", xy=(_x_mid + _delta_arrow, _y_arrow_position), xytext=(_x_mid, _y_arrow_position), arrowprops=arrow_props, zorder=10)
    ax.annotate("", xy=(_x_mid - _delta_arrow, _y_arrow_position), xytext=(_x_mid, _y_arrow_position), arrowprops=arrow_props, zorder=10)
    ax.text(_x_mid, _y_arrow_position, f"{2 * _delta_arrow:.1f}" + r"$\; \mathrm{km}/\mathrm{s}$",
            ha='center', va='center', color='white', fontsize=12, bbox=bbox_props, zorder=10)
            
    # --- Bottom Arrows (Obs) ---
    _y_arrow_position = 0.2
    _delta_arrow = (_vlines_obs[0] - _vlines_obs[-1]) / 2
    _x_mid = _vlines_obs[-1] + _delta_arrow
    ax.annotate("", xy=(_x_mid + _delta_arrow, _y_arrow_position), xytext=(_x_mid, _y_arrow_position), arrowprops=arrow_props, zorder=10)
    ax.annotate("", xy=(_x_mid - _delta_arrow, _y_arrow_position), xytext=(_x_mid, _y_arrow_position), arrowprops=arrow_props, zorder=10)
    ax.text(_x_mid, _y_arrow_position, f"{2 * _delta_arrow:.1f}" + r"$\; \mathrm{km}/\mathrm{s}$",
            ha='center', va='center', color='white', fontsize=12, bbox=bbox_props, zorder=10)

    leg = fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncols=min(len(handles), 7),
        frameon=True, fontsize=16, handlelength=2.6, facecolor="black", edgecolor="white", framealpha=0.9,
    )
    for text in leg.get_texts():
        text.set_color("white")
            
    ax.tick_params(axis="both", which="major", labelsize=20, colors="white", length=6, width=1.1)
    for side in ["top", "right", "bottom", "left"]:
        ax.spines[side].set_color("white")
        ax.spines[side].set_linewidth(1.0)
        
    ax.set_ylabel(r"Transmittance $\; \left[\mathrm{adim}\right]$", fontsize=20, color="white")
    ax.set_xlabel(r"Shifted Velocity Units $v \; \left[\mathrm{km}/\mathrm{s}\right]$", fontsize=20, color="white")

    fig.tight_layout()
    figures.append(fig)
    if show:
        plt.show()

    if plot_wrapped_velocity_units:
            
        # --------------------------------------------- Wrapped Velocity Units --------------------------------------------- #
        fig, ax = plt.subplots(1, 1, figsize=(14, 4.5), sharex=True, sharey=True, facecolor="black")
        ax.set_facecolor("black")

        kk = 0
        handles = []
        for ii, key_species in enumerate(line_bundle.species_keys):
            color = colors[key_species]
            for jj, line in enumerate(line_bundle.lines_for_species(key_species)):
                ls = linestyles[jj]
                
                ax.plot(velocities_kms, jnp.exp(-optical_depth_wrapped[kk, ii_skewer]), color=color, lw=6, ls=ls, alpha=0.2, zorder=2)
                ax.plot(velocities_kms, jnp.exp(-optical_depth_wrapped[kk, ii_skewer]), color=color, lw=1.5, ls=ls, alpha=0.9, zorder=3)
                
                handles.append(Line2D([0], [0], color=color, lw=4, ls=ls, label=line_bundle.lines[kk].id))
                kk += 1
                
        ax.set_xlim([v_min-100, v_max+100])
        ax.set_ylim([-0.05, 1.05])

        ax.axvline(0, color='white', lw=4, ls=':', zorder=1)
        ax.axvline(phys.c_kms*jnp.log(a_dagger / a_em), color='white', lw=4, ls=':', zorder=2)

        leg = fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncols=min(len(handles), 7),
            frameon=True, fontsize=16, handlelength=2.6, facecolor="black", edgecolor="white", framealpha=0.9,
        )
        for text in leg.get_texts():
            text.set_color("white")
                
        ax.tick_params(axis="both", which="major", labelsize=20, colors="white", length=6, width=1.1)
        for side in ["top", "right", "bottom", "left"]:
            ax.spines[side].set_color("white")
            ax.spines[side].set_linewidth(1.0)
            
        ax.set_ylabel(r"Transmittance $\; \left[\mathrm{adim}\right]$", fontsize=20, color="white")
        ax.set_xlabel(r"Wrapped Velocity Units $v \; \left[\mathrm{km}/\mathrm{s}\right]$", fontsize=20, color="white")

        fig.tight_layout()
        figures.append(fig)
        if show:
            plt.show()
        
    # Explicitly delete local references to intermediate JAX arrays 
    # created during the plotting process (e.g., from fields.sample)
    try: del quad_all
    except NameError: pass
    try: del field_plot
    except NameError: pass
    try: del clouds
    except NameError: pass
    try: del cloud
    except NameError: pass
    try: del sampled_fields
    except NameError: pass
    try: del plot_dict
    except NameError: pass

    # Force Python garbage collection and clear JAX's compilation cache
    gc.collect()
    jax.clear_caches()
        
    return figures


def create_animated_gifs_and_cleanup(save_dir, duration_ms=400, loop=0, delete_originals=True):
    """
    Reads PNG files from save_dir, groups them by figure type, 
    creates an animated GIF for each type, and optionally deletes the original PNGs to save space.
    
    Parameters:
    - save_dir: Directory containing the saved PNGs.
    - duration_ms: Time each frame is displayed in milliseconds (e.g., 400 = 0.4s).
    - loop: 0 means loop infinitely, 1 means play once, etc.
    - delete_originals: bool, if True (default), deletes the original PNG frames after 
                        successfully creating the GIF. If False, keeps them on disk.
    """
    # 1. Find all PNG files in the directory
    png_files = glob.glob(os.path.join(save_dir, "*.png"))
    
    if not png_files:
        print(f"No PNG files found in '{save_dir}'.")
        return

    # 2. Group files by figure index (idx)
    # Structure: { idx: { iteration_ii: filepath } }
    grouped_images = {}

    for filepath in png_files:
        filename = os.path.basename(filepath)
        try:
            # Expected format: iterXXXX_figY.png
            name_without_ext = os.path.splitext(filename)[0]
            iter_part, fig_part = name_without_ext.split('_')
            
            ii = int(iter_part.replace('iter', ''))
            idx = int(fig_part.replace('fig', ''))
            
            if idx not in grouped_images:
                grouped_images[idx] = {}
            grouped_images[idx][ii] = filepath
            
        except ValueError:
            # Skip files that don't match the expected naming convention
            continue

    # 3. Create a GIF for each figure type and clean up
    print(f"Found {len(grouped_images)} unique figure types to animate.\n")
    
    for idx, iterations_dict in grouped_images.items():
        # Sort by iteration index (ii) to ensure correct chronological order
        sorted_iis = sorted(iterations_dict.keys())
        sorted_filepaths = [iterations_dict[ii] for ii in sorted_iis]
        
        if len(sorted_filepaths) < 2:
            print(f"⚠️ Skipping figure type {idx}: Not enough frames (found {len(sorted_filepaths)}) to make a GIF.")
            continue

        print(f"🎬 Creating GIF for figure type {idx} with {len(sorted_filepaths)} frames...")
        
        # Open all images for this figure type
        images = [Image.open(fp) for fp in sorted_filepaths]
        
        # Define output GIF path
        gif_filename = f"animation_fig{idx}.gif"
        gif_filepath = os.path.join(save_dir, gif_filename)
        
        try:
            # Save as animated GIF
            images[0].save(
                gif_filepath,
                save_all=True,
                append_images=images[1:],
                duration=duration_ms,
                loop=loop,
                optimize=True # Reduces file size
            )
            print(f"✅ Successfully saved: {gif_filepath}")
            
            # --- CLEANUP STEP ---
            if delete_originals:
                print(f"🧹 Cleaning up {len(sorted_filepaths)} original PNG frames to save space...")
                for fp in sorted_filepaths:
                    os.remove(fp)
                print(f"   🗑️ Deleted original frames for figure type {idx}.\n")
            else:
                print(f"   📂 Kept original frames for figure type {idx}.\n")
            
        except Exception as e:
            print(f"❌ Error creating GIF for figure type {idx}: {e}")
            print("   ⚠️ Skipping cleanup to preserve original frames.\n")


def _dark_2d_style(fig, ax):
    """Black background, white ticks, labels and spines (2D axes)."""
    fig.set_facecolor("black")
    ax.set_facecolor("black")
    ax.tick_params(axis="both", which="both", colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for side in ax.spines.values():
        side.set_color("white")


def plot_neon_lines(x, curves, *, total=None, total_label="Total", colors=None, linestyles=None,
                    vlines=(), xlabel="", ylabel="", logy=False, xlim=None, ylim=None, ax=None,
                    figsize=(14, 4.5)):
    """Neon line plot on a black background (spectra, overflux or densities along a skewer).

    Parameters
    ----------
    x : (N,) array
    curves : dict {label: (N,) array}, e.g. {line id: exp(-tau)} or {species: n(x)}.
    total : (N,) array or None
        Drawn as a white dotted line labelled ``total_label`` (e.g. the product of fluxes).
    colors, linestyles : dict {label: ...} or None
        Default: ``_neon_colors(labels)`` and "-".
    vlines : sequence of x positions drawn as thick white dotted lines (e.g. box edges).

    Returns fig, ax.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    else:
        fig = ax.figure
    colors = _neon_colors(list(curves)) if colors is None else colors
    linestyles = {} if linestyles is None else linestyles
    x = np.asarray(x)
    for lab, y in curves.items():
        y, c, ls = np.asarray(y), colors[lab], linestyles.get(lab, "-")
        ax.plot(x, y, color=c, lw=6, ls=ls, alpha=0.2, zorder=2)               # glow
        ax.plot(x, y, color=c, lw=1.5, ls=ls, alpha=0.9, zorder=3, label=lab)
    if total is not None:
        ax.plot(x, np.asarray(total), color="white", lw=2.0, ls=":", alpha=0.9, zorder=5, label=total_label)
    for xv in vlines:
        ax.axvline(float(xv), color="white", lw=4, ls=":", zorder=1)
    if logy:
        ax.set_yscale("log")
    ax.set(xlabel=xlabel, ylabel=ylabel)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    _dark_2d_style(fig, ax)
    leg = ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=min(len(curves) + 1, 7),
                    frameon=True, fontsize="small", facecolor="black", edgecolor="white")
    for t in leg.get_texts():
        t.set_color("white")
    return fig, ax


def plot_xi3d_rperp_pi(binned, *, scale="symlog", linthresh_frac=0.04, units=None, ax=None,
                       figsize=(36, 10)):
    """xi3D binned in r_perp, for pi >= 0, in three views (neon style on black).

    Left: heatmap xi(r_perp, pi); middle: xi(pi) for every r_perp bin; right: 3D surface.

    Parameters
    ----------
    binned : tuple
        ``(xi_b, r_perp_centers, r_perp_edges, pi_lags, n_pairs)`` from
        ``OverfluxStatistics3D.bin_xi3d_rperp``.
    scale : "linear", "log" or "symlog"
        Colour and z scale. "symlog" uses a linear threshold ``linthresh_frac`` max|xi|; the
        profile and surface heights are transformed the same way.
    units : dict or None
        Axis units (``UNITS_3D`` by default: r_perp [cMpc], pi [km/s]).
    ax : (Axes, Axes, Axes3D) or None

    Returns fig, axes (heatmap, profiles, surface).
    """
    units = UNITS_3D if units is None else units
    xi_b, r_c, r_e, pi_lags, _ = (np.asarray(v) for v in binned)
    half = len(pi_lags) // 2
    pi_pos, xi_pos = pi_lags[half:], xi_b[:, half:]
    vmax = np.nanmax(np.abs(xi_pos)) or 1.0
    if scale == "linear":
        norm, cmap, z = mpl.colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax), "YlOrRd", (lambda a: a)
    elif scale == "log":
        vmin = np.nanmin(xi_pos[xi_pos > 0]) if np.any(xi_pos > 0) else 1e-6 * vmax
        norm, cmap = mpl.colors.LogNorm(vmin=vmin, vmax=vmax), "plasma"
        z = lambda a: np.where(a > 0, np.log10(np.maximum(a, 1e-300)), np.nan)
    elif scale == "symlog":
        lt = linthresh_frac * vmax
        norm, cmap = mpl.colors.SymLogNorm(linthresh=lt, vmin=-vmax, vmax=vmax, base=10), "YlOrRd"
        z = lambda a: np.sign(a) * np.where(np.abs(a) <= lt, np.abs(a), lt + lt * np.log10(np.maximum(np.abs(a) / lt, 1.0)))
    else:
        raise ValueError(f"scale must be 'linear', 'log' or 'symlog', got {scale!r}.")
    zlabel = rf"$\xi_{{\rm 3D}}$ ({scale})"
    r_l, pi_l = _lbl(r"r_\perp", units["x_perp"]), _lbl(r"\pi", units["x_par"])

    if ax is None:
        fig = plt.figure(figsize=figsize, facecolor="black", layout="constrained")
        a0, a1 = fig.add_subplot(1, 3, 1), fig.add_subplot(1, 3, 2)
        a2 = fig.add_subplot(1, 3, 3, projection="3d")
    else:
        a0, a1, a2 = ax
        fig = a0.figure

    dpi = pi_pos[1] - pi_pos[0]
    pm = a0.pcolormesh(np.r_[pi_pos - dpi / 2, pi_pos[-1] + dpi / 2], r_e, xi_pos, norm=norm, cmap=cmap)
    a0.set(xlabel=pi_l, ylabel=r_l, title=r"$\hat\xi_{\rm 3D}(r_\perp,\pi)$")
    cb = fig.colorbar(pm, ax=a0, label=zlabel)
    cb.ax.yaxis.label.set_color("white")
    cb.ax.tick_params(colors="white")

    neon = list(_neon_colors(range(len(r_c))).values())
    for b, (y, rc) in enumerate(zip(z(xi_pos), r_c)):
        a1.plot(pi_pos, y, color=neon[b], lw=6, alpha=0.12)
        a1.plot(pi_pos, y, color=neon[b], lw=2.0, label=rf"$r_\perp={rc:.2f}$")
    a1.axhline(0.0, color="0.4", lw=1, ls="--")
    a1.set(xlabel=pi_l, ylabel=zlabel, title=r"$\hat\xi_{\rm 3D}(\pi)$ per $r_\perp$ bin")
    leg = a1.legend(fontsize="x-small", ncol=2, facecolor="black", edgecolor="0.3")
    for t in leg.get_texts():
        t.set_color("white")
    for a in (a0, a1):
        _dark_2d_style(fig, a)

    PI, R = np.meshgrid(pi_pos, r_c)
    Z = z(xi_pos)
    a2.plot_surface(PI, R, Z, facecolors=plt.get_cmap(cmap)(norm(xi_pos)), linewidth=0,
                    antialiased=False, shade=False, alpha=0.92)
    zmax = np.nanmax(np.abs(Z[np.isfinite(Z)]))
    a2.contourf(PI, R, Z, zdir="z", offset=-1.05 * zmax, levels=20, cmap=cmap, alpha=0.35)
    a2.set_zlim(-1.05 * zmax, 1.1 * zmax)
    _dark_3d_style(a2)
    a2.set_xlabel(pi_l, labelpad=20)
    a2.set_ylabel(r_l, labelpad=20)
    a2.set_zlabel(zlabel, labelpad=20)
    a2.view_init(elev=30, azim=225)
    return fig, np.array([a0, a1, a2], dtype=object)


# ============================================================
# 3. P1D / xi1D of composite overflux fields by line pair and order
# ============================================================


def pretty_subset(key):
    return key if len(key) == 1 else f"({key})"


def pretty_pair(pair):
    a, b = pair
    return rf"${pretty_subset(a)}{pretty_subset(b)}$"


def pretty_pair_sym(pair):
    label = pretty_pair(pair)
    return label[:-1] + r"_{\mathrm{sym}}$" if label.startswith("$") and label.endswith("$") else label + r"$_{\mathrm{sym}}$"


class VerticalLineHandler(HandlerBase):
    """Render a vertical line inside legend boxes."""
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        x = xdescent + width / 2
        line = Line2D(
            [x, x], [ydescent, ydescent + height],
            color=orig_handle.get_color(),
            linestyle=orig_handle.get_linestyle(),
            linewidth=orig_handle.get_linewidth(),
            transform=trans,
        )
        return [line]


# ============================================================
# Generic numeric / plotting helpers
# ============================================================

def _default_colors():
    return plt.rcParams["axes.prop_cycle"].by_key()["color"]


def _nonzero_min(arrays, default=1e-12):
    vals = []
    for arr in arrays:
        arr = np.asarray(arr)
        nz = np.abs(arr[np.abs(arr) > 0])
        if len(nz):
            vals.append(np.min(nz))
    return np.min(vals) if vals else default


def _safe_relative_residual(curve, ref):
    ref = np.asarray(ref)
    floor = 1e-12 * np.max(np.abs(ref)) + 1e-15
    den = np.maximum(np.abs(ref), floor)
    return (np.asarray(curve) - ref) / den


def _add_order_badge(
    ax,
    order,
    *,
    fontsize=22,
    boxstyle="round,pad=0.25",
    facecolor="white",
    edgecolor="black",
    alpha=0.9,
):
    ax.text(
        0.02, 0.96, rf"$\mathcal{{O}}({order})$",
        transform=ax.transAxes,
        va="top", ha="left",
        fontsize=fontsize,
        bbox=dict(
            boxstyle=boxstyle,
            facecolor=facecolor,
            edgecolor=edgecolor,
            alpha=alpha,
        ),
    )


# ============================================================
# Symlog formatting helpers
# ============================================================

def _sci_label(x):
    if x == 0:
        return r"$0$"
    sign = "-" if x < 0 else ""
    ax = abs(x)
    exponent = int(np.floor(np.log10(ax)))
    coeff = ax / 10**exponent
    coeff_round = round(coeff)
    if np.isclose(coeff, coeff_round, rtol=1e-10, atol=1e-14):
        coeff = coeff_round
    return rf"${sign}10^{{{exponent}}}$" if np.isclose(coeff, 1.0, rtol=1e-10, atol=1e-14) else rf"${sign}{coeff:g}\times 10^{{{exponent}}}$"


def _format_linear_number(x, linthresh, sci_cutoff=1e-2):
    if np.isclose(x, 0.0, atol=max(linthresh * 1e-12, 1e-30), rtol=0.0):
        return r"$0$"
    ax = abs(x)
    if ax < sci_cutoff:
        return _sci_label(x)
    if 1e-2 <= ax < 1e3:
        return f"{x:g}"
    return _sci_label(x)


def _make_symlog_formatter(linthresh=1e-3, base=10, label_minor=False, zero_tol=None, sci_cutoff=1e-2):
    zero_tol = max(linthresh * 1e-12, 1e-30) if zero_tol is None else zero_tol
    log_base = np.log(base)

    def _formatter(x, pos=None):
        if np.isclose(x, 0.0, atol=zero_tol, rtol=0.0):
            return r"$0$"

        ax = abs(x)
        if ax <= linthresh * (1.0 + 1e-12):
            return _format_linear_number(x, linthresh, sci_cutoff=sci_cutoff)

        exponent = int(np.floor(np.log(ax) / log_base))
        coeff = ax / (base ** exponent)
        coeff_round = round(coeff)
        if np.isclose(coeff, coeff_round, rtol=1e-10, atol=1e-14):
            coeff = coeff_round

        sign = "-" if x < 0 else ""
        if np.isclose(coeff, 1.0, rtol=1e-10, atol=1e-14):
            return rf"${sign}{base}^{{{exponent}}}$" if base != 10 else rf"${sign}10^{{{exponent}}}$"
        if not label_minor:
            return ""
        return rf"${sign}{coeff:g}\times {base}^{{{exponent}}}$" if base != 10 else rf"${sign}{coeff:g}\times 10^{{{exponent}}}$"

    return mticker.FuncFormatter(_formatter)


def apply_clean_symlog(ax, axis="y", linthresh=1e-3, linscale=1.0, base=10, subs=None, label_minor=False, sci_cutoff=1e-2):
    if subs is None:
        subs = tuple(range(2, int(base))) if int(base) == base and base >= 3 else ()

    if axis == "y":
        ax.set_yscale("symlog", linthresh=linthresh, linscale=linscale, base=base, subs=subs)
        axis_obj = ax.yaxis
    elif axis == "x":
        ax.set_xscale("symlog", linthresh=linthresh, linscale=linscale, base=base, subs=subs)
        axis_obj = ax.xaxis
    else:
        raise ValueError("axis must be 'x' or 'y'")

    axis_obj.set_major_locator(mticker.SymmetricalLogLocator(base=base, linthresh=linthresh, subs=(1.0,)))
    axis_obj.set_major_formatter(_make_symlog_formatter(linthresh=linthresh, base=base, label_minor=False, sci_cutoff=sci_cutoff))

    if len(subs):
        axis_obj.set_minor_locator(mticker.SymmetricalLogLocator(base=base, linthresh=linthresh, subs=subs))
        axis_obj.set_minor_formatter(
            _make_symlog_formatter(linthresh=linthresh, base=base, label_minor=True, sci_cutoff=sci_cutoff)
            if label_minor else mticker.NullFormatter()
        )
    else:
        axis_obj.set_minor_locator(mticker.NullLocator())
        axis_obj.set_minor_formatter(mticker.NullFormatter())

    ax.tick_params(which="major", length=6, width=1.0)
    ax.tick_params(which="minor", length=3, width=0.8)
    return ax


# ============================================================
# Pair / order helpers
# ============================================================

def pair_order(pair):
    return len(pair[0]) + len(pair[1])


def is_auto_pair(pair):
    return pair[0] == pair[1]


def build_order_sum_dict(curve_dict, ordered_pairs=None):
    if ordered_pairs is None:
        ordered_pairs = list(curve_dict.keys())
    if not ordered_pairs:
        raise ValueError("ordered_pairs cannot be empty.")

    template = curve_dict[ordered_pairs[0]]
    orders = sorted({pair_order(pair) for pair in ordered_pairs})
    out = {}

    for order in orders:
        acc = np.zeros_like(template, dtype=np.complex128)
        for pair in ordered_pairs:
            if pair_order(pair) == order:
                acc += curve_dict[pair]
        out[order] = acc

    return out, orders


# ============================================================
# Harmonics / delta-v difference helpers
# ============================================================

def draw_harmonics(
    ax,
    delta_v_list,
    L_box,
    Npix,
    *,
    colors=None,
    harmonic_color_mode="cycle",
    harmonic_main_lw=1.5,
    harmonic_higher_lw=1.5,
    harmonic_alpha=0.7,
):
    colors = _default_colors() if colors is None else colors
    dv = L_box / Npix

    for j, delta_v_0 in enumerate(delta_v_list):
        if delta_v_0 == 0.0:
            continue

        c = colors[j % len(colors)] if harmonic_color_mode == "cycle" else "grey"
        k0 = 2 * np.pi / delta_v_0   # Characteristic frequency from Delta v_0
        ax.axvline(k0, color=c, lw=harmonic_main_lw, ls="--")

        n_max_harmonic = int(np.floor((delta_v_0 / (2*dv))))
        for n in range(0, n_max_harmonic + 1):
            ax.axvline(n * k0, color=c, lw=harmonic_higher_lw, ls=":", alpha=harmonic_alpha)


def add_harmonic_legend(ax, *, fontsize=16):
    handles = [
        Line2D([], [], color="grey", lw=1.5, ls="--"),
        Line2D([], [], color="grey", lw=1.5, ls=":"),
    ]
    leg = ax.legend(
        handles=handles,
        labels=[r"$L_{\rm box}/\Delta v_0$", "harmonics"],
        handler_map={Line2D: VerticalLineHandler()},
        fontsize=fontsize,
        loc="lower right",
        ncol=2,
    )
    ax.add_artist(leg)
    return leg


def get_delta_v_difference_info(delta_v_list, L_box):
    out = []
    for i in range(len(delta_v_list)):
        for j in range(i + 1, len(delta_v_list)):
            d = abs(delta_v_list[j] - delta_v_list[i]) % L_box
            d = min(d, L_box - d)
            d_norm = d
            if d_norm > 0.0:
                out.append((i + 1, j + 1, d_norm))
    return out


def draw_delta_v_differences(ax, delta_v_list, L_box, *, cmap=None, lw=1.5, ls="--", alpha=0.9):
    cmap = plt.get_cmap("tab10") if cmap is None else cmap
    handles = []

    for n, (i, j, d_norm) in enumerate(get_delta_v_difference_info(delta_v_list, L_box)):
        c = cmap(n % 10)
        ax.axvline(d_norm, color=c, lw=lw, ls=ls, alpha=alpha)
        handles.append(Line2D([0], [0], color=c, lw=max(lw, 1.8), ls=ls, label=fr"$\Delta v_{{{i}{j}}}$"))

    return handles


# ============================================================
# P1D pair collapsing
# ============================================================

def collapse_order_pairs(pairs_this_order, Pk_pair, tolerance_threshold=1e-6):
    template = Pk_pair[pairs_this_order[0]]
    Pk_sum_full = np.zeros_like(template, dtype=np.complex128)
    for pair in pairs_this_order:
        Pk_sum_full += Pk_pair[pair]

    processed = set()
    entries = []
    Pk_sum = np.zeros_like(template, dtype=np.complex128)

    for pair in pairs_this_order:
        if pair in processed:
            continue

        A, B = pair
        rev = (B, A)

        if is_auto_pair(pair):
            curve = Pk_pair[pair]
            entries.append(dict(pair=pair, curve=curve, ls="-", label=pretty_pair(pair), is_auto=True))
            Pk_sum += curve
            processed.add(pair)
            continue

        if rev in pairs_this_order and rev not in processed:
            diff = np.max(np.abs(Pk_pair[pair][1:] - Pk_pair[rev][1:]))
            if diff <= tolerance_threshold:
                curve = 2.0 * Pk_pair[pair]
                entries.append(dict(pair=pair, curve=curve, ls="--", label=pretty_pair_sym(pair), is_auto=False))
                Pk_sum += curve
                processed.update([pair, rev])
            else:
                for p in [pair, rev]:
                    curve = Pk_pair[p]
                    entries.append(dict(pair=p, curve=curve, ls="--", label=pretty_pair(p), is_auto=False))
                    Pk_sum += curve
                processed.update([pair, rev])
        else:
            curve = Pk_pair[pair]
            entries.append(dict(pair=pair, curve=curve, ls="--", label=pretty_pair(pair), is_auto=False))
            Pk_sum += curve
            processed.add(pair)

    return entries, Pk_sum, Pk_sum_full


# ============================================================
# Xi helpers
# ============================================================

def xi_reverse_lag_raw(xi_raw):
    idx = (-np.arange(len(xi_raw))) % len(xi_raw)
    return xi_raw[idx]


def xi_plot_curve(xi_raw):
    return np.fft.fftshift(np.real(xi_raw))


def collapse_order_pairs_xi(pairs_this_order, xi_pair, tolerance_threshold=1e-6):
    template = xi_pair[pairs_this_order[0]]
    xi_sum_full = np.zeros_like(template, dtype=np.complex128)
    for pair in pairs_this_order:
        xi_sum_full += xi_pair[pair]

    processed = set()
    entries = []
    xi_sum_raw = np.zeros_like(template, dtype=np.complex128)

    for pair in pairs_this_order:
        if pair in processed:
            continue

        A, B = pair
        rev = (B, A)

        if is_auto_pair(pair):
            curve_raw = xi_pair[pair]
            entries.append(
                dict(
                    pair=pair,
                    curve_raw=curve_raw,
                    curve_plot=xi_plot_curve(curve_raw),
                    ls="-",
                    label=pretty_pair(pair),
                    is_auto=True,
                )
            )
            xi_sum_raw += curve_raw
            processed.add(pair)
            continue

        if rev in pairs_this_order and rev not in processed:
            curve_ab = xi_pair[pair]
            curve_ba = xi_pair[rev]
            diff = np.max(np.abs(curve_ba - xi_reverse_lag_raw(curve_ab)))

            if diff <= tolerance_threshold:
                curve_raw = curve_ab + curve_ba
                entries.append(
                    dict(
                        pair=pair,
                        curve_raw=curve_raw,
                        curve_plot=xi_plot_curve(curve_raw),
                        ls="--",
                        label=pretty_pair_sym(pair),
                        is_auto=False,
                    )
                )
                xi_sum_raw += curve_raw
                processed.update([pair, rev])
            else:
                for p, c in [(pair, curve_ab), (rev, curve_ba)]:
                    entries.append(
                        dict(
                            pair=p,
                            curve_raw=c,
                            curve_plot=xi_plot_curve(c),
                            ls="--",
                            label=pretty_pair(p),
                            is_auto=False,
                        )
                    )
                    xi_sum_raw += c
                processed.update([pair, rev])
        else:
            curve_raw = xi_pair[pair]
            entries.append(
                dict(
                    pair=pair,
                    curve_raw=curve_raw,
                    curve_plot=xi_plot_curve(curve_raw),
                    ls="--",
                    label=pretty_pair(pair),
                    is_auto=False,
                )
            )
            xi_sum_raw += curve_raw
            processed.add(pair)

    return entries, xi_sum_raw, xi_sum_full


# ============================================================
# P1D plotting
# ============================================================

def plot_p1d_by_order(
    ordered_pairs,
    Pk_pair,
    k,
    delta_v_list,
    L_box,
    Npix,
    *,
    tolerance_threshold=1e-6,
    residual_mode="auto_reference",
    show_harmonics=True,
    use_symlog_main=True,
    use_symlog_residual=False,
    use_log_x=False,
    xlim=None,
    symlog_linscale=1.0,
    symlog_label_minor=False,
    symlog_sci_cutoff=1e-2,
    legend_ncol_terms=2,
    xlabel=r"$k \, \left[ \mathrm{s} / \mathrm{km} \right]$",
    ylabel=r"$P_{\rm 1D}$",
    residual_ylabel=r"$\Delta$",
):
    if not ordered_pairs:
        raise ValueError("ordered_pairs cannot be empty.")

    # ------------------------------------------------------------------
    # Fixed defaults from the original function
    # ------------------------------------------------------------------
    cmap = plt.get_cmap("tab10")
    colors = _default_colors()
    figsize = None
    main_panel_height = 5.0
    residual_panel_height = 3.0
    hspace = 0.08
    main_curve_lw = 2.0
    collapsed_sum_lw = 1.5
    sum_color = "k"
    sum_ls = ":"
    harmonic_color_mode = "cycle"
    harmonic_main_lw = 1.5
    harmonic_higher_lw = 1.5
    harmonic_alpha = 0.7
    show_zero_line_main = True
    show_zero_line_residual = True
    symlog_base = 10
    order_badge_fontsize = 22
    order_badge_boxstyle = "round,pad=0.25"
    order_badge_facecolor = "white"
    order_badge_edgecolor = "black"
    order_badge_alpha = 0.9
    label_fontsize = 22
    residual_label_fontsize = 20
    tick_fontsize = 22
    legend_fontsize = 18
    harmonic_legend_fontsize = 16
    show_order_badge = True

    # ------------------------------------------------------------------
    # Harmonic-folding convention: if delta_v > L_box/2, display harmonics at L_box - delta_v
    # ------------------------------------------------------------------
    delta_v_array = np.asarray(delta_v_list, dtype=float)
    delta_v_harmonics = np.where(delta_v_array > 0.5 * L_box, L_box - delta_v_array, delta_v_array)
    folded_mask = delta_v_array > 0.5 * L_box

    if np.any(folded_mask):
        folded_entries = []
        for i, (orig, shown) in enumerate(zip(delta_v_array, delta_v_harmonics)):
            if folded_mask[i]:
                folded_entries.append(f"index {i}: delta_v={orig:.6g} -> displayed as {shown:.6g}")

        warnings.warn(
            "Some harmonic positions use the folded convention delta_v -> L_box - delta_v for delta_v > L_box/2. Affected entries: {', '.join(folded_entries)}",
            UserWarning,
        )

    orders = sorted({pair_order(pair) for pair in ordered_pairs})
    if figsize is None:
        figsize = (14, 6.0 * len(orders))

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(nrows=2 * len(orders), ncols=1,
        height_ratios=[v for _ in orders for v in (main_panel_height, residual_panel_height)], hspace=hspace,
    )

    axes, axes_res = [], []
    for i in range(len(orders)):
        ax = fig.add_subplot(gs[2 * i], sharex=axes[0] if axes else None)
        ax_res = fig.add_subplot(gs[2 * i + 1], sharex=ax)
        axes.append(ax)
        axes_res.append(ax_res)

    order_sum_dict = {}

    for ax, ax_res, order in zip(axes, axes_res, orders):
        pairs_this_order = [pair for pair in ordered_pairs if pair_order(pair) == order]

        if show_harmonics:
            draw_harmonics(ax, delta_v_harmonics, L_box, Npix, colors=colors, harmonic_color_mode=harmonic_color_mode,
                harmonic_main_lw=harmonic_main_lw, harmonic_higher_lw=harmonic_higher_lw, harmonic_alpha=harmonic_alpha,
            )

        entries, Pk_order_sum, Pk_order_sum_full = collapse_order_pairs(
            pairs_this_order, Pk_pair, tolerance_threshold=tolerance_threshold
        )

        sum_diff = np.max(np.abs(Pk_order_sum[1:] - Pk_order_sum_full[1:]))
        if sum_diff > tolerance_threshold:
            print(f"[warning] order {order}: collapsed/fallback representation differs from the full sum by max {sum_diff:.6e}")

        order_sum_dict[order] = Pk_order_sum

        legend_terms = [Line2D([0], [0], color=sum_color, lw=collapsed_sum_lw, ls=sum_ls, label="Sum")]

        for i, entry in enumerate(entries):
            entry["color"] = cmap(i % 10)
            ax.plot(k[1:], entry["curve"].real[1:], lw=main_curve_lw, ls=entry["ls"], color=entry["color"], alpha=0.95)
            legend_terms.append(Line2D([0], [0], color=entry["color"], lw=main_curve_lw, ls=entry["ls"], label=entry["label"]))
        ax.plot(k[1:], Pk_order_sum.real[1:], lw=collapsed_sum_lw, color=sum_color, ls=sum_ls, alpha=0.95)

        if show_zero_line_main:
            ax.axhline(0.0, color="k", lw=1.0, alpha=0.8, ls=":")

        leg_terms = ax.legend(
            handles=legend_terms,
            fontsize=legend_fontsize,
            loc="upper right",
            ncol=legend_ncol_terms,
        )
        ax.add_artist(leg_terms)

        if show_order_badge:
            _add_order_badge(ax, order, fontsize=order_badge_fontsize, boxstyle=order_badge_boxstyle,
                facecolor=order_badge_facecolor, edgecolor=order_badge_edgecolor, alpha=order_badge_alpha,
            )

        ax.set_ylabel(ylabel, fontsize=label_fontsize)
        ax.tick_params(axis="both", labelsize=tick_fontsize, labelbottom=False)

        if use_symlog_main:
            linthresh = _nonzero_min([Pk_order_sum.real[1:]])
            apply_clean_symlog(
                ax, axis="y", linthresh=linthresh, linscale=symlog_linscale, base=symlog_base, label_minor=symlog_label_minor, sci_cutoff=symlog_sci_cutoff
            )

        nonzero_residuals = []

        if residual_mode == "sum":
            Pk_ref = Pk_order_sum.real[1:]
            for entry in entries:
                res = _safe_relative_residual(entry["curve"].real[1:], Pk_ref)
                ax_res.plot(k[1:], res, lw=1.5, ls=entry["ls"], color=entry["color"], alpha=0.95)
                nz = np.abs(res[np.abs(res) > 0])
                if len(nz):
                    nonzero_residuals.append(np.min(nz))

        elif residual_mode == "auto_reference":
            auto_entries = [entry for entry in entries if entry["is_auto"]]
            if auto_entries:
                Pk_ref = auto_entries[0]["curve"].real[1:]
                for entry in auto_entries:
                    res = _safe_relative_residual(entry["curve"].real[1:], Pk_ref)
                    ax_res.plot(k[1:], res, lw=1.5, ls=entry["ls"], color=entry["color"], alpha=0.95)
                    nz = np.abs(res[np.abs(res) > 0])
                    if len(nz):
                        nonzero_residuals.append(np.min(nz))
            else:
                ax_res.text(0.5, 0.5, "No auto terms in this order", transform=ax_res.transAxes, fontsize=16, va="center", ha="center")
        else:
            raise ValueError("residual_mode must be 'sum' or 'auto_reference'")

        if show_zero_line_residual:
            ax_res.axhline(0.0, color="k", lw=1.0, alpha=0.8, ls=":")

        if show_harmonics:
            draw_harmonics(ax_res, delta_v_harmonics, L_box, Npix, colors=colors, harmonic_color_mode=harmonic_color_mode,
                harmonic_main_lw=harmonic_main_lw, harmonic_higher_lw=harmonic_higher_lw, harmonic_alpha=harmonic_alpha,
            )

        if use_symlog_residual:
            apply_clean_symlog(ax_res, axis="y", linthresh=(np.min(nonzero_residuals) if nonzero_residuals else 1e-12),
                linscale=symlog_linscale, base=symlog_base, label_minor=symlog_label_minor, sci_cutoff=symlog_sci_cutoff,
            )

        ax_res.set_ylabel(residual_ylabel, fontsize=residual_label_fontsize)
        ax_res.tick_params(axis="both", labelsize=tick_fontsize, labelbottom=(ax_res == axes_res[-1]))

        if use_log_x:
            ax.set_xscale("log")
            ax_res.set_xscale("log")

        if xlim is not None:
            ax.set_xlim(xlim)
            ax_res.set_xlim(xlim)

    if show_harmonics:
        add_harmonic_legend(axes[0], fontsize=harmonic_legend_fontsize)

    axes_res[-1].set_xlabel(xlabel, fontsize=label_fontsize)

    plt.tight_layout()
    plt.show()
    return fig, axes, axes_res, order_sum_dict


def plot_p1d_total_recovery(
    Pk_total,
    Pk_total_from_terms,
    Pk_pair,
    k,
    delta_v_list,
    L_box,
    Npix,
    *,
    ordered_pairs=None,
    cmap=None,
    colors=None,
    figsize=(14, 6.5),
    hspace=0.06,
    show_order_sums=False,
    show_order_sum_residuals=False,
    total_lw=3.0,
    recovered_lw=1.5,
    recovered_ls="--",
    total_color="k",
    recovered_color="red",
    order_sum_lw=2.0,
    order_sum_alpha=0.95,
    show_harmonics=True,
    harmonic_color_mode="cycle",
    harmonic_main_lw=1.5,
    harmonic_higher_lw=1.5,
    harmonic_alpha=0.7,
    use_symlog_main=True,
    use_symlog_residual=True,
    main_symlog_linthresh=None,
    residual_symlog_linthresh=1e-3,
    symlog_linscale=1.0,
    symlog_base=10,
    symlog_label_minor=False,
    symlog_sci_cutoff=1e-2,
    show_zero_line_main=False,
    show_zero_line_residual=True,
    label_fontsize=22,
    residual_label_fontsize=20,
    tick_fontsize=22,
    legend_fontsize=18,
    harmonic_legend_fontsize=16,
    xlabel=r"$k \, \left[ \mathrm{s} / \mathrm{km} \right]$",
    ylabel=r"$P_{\rm 1D}$",
    residual_ylabel=r"$\Delta$",
):
    ordered_pairs = list(Pk_pair.keys()) if ordered_pairs is None else ordered_pairs
    if not ordered_pairs:
        raise ValueError("ordered_pairs cannot be empty.")

    cmap = plt.get_cmap("tab10") if cmap is None else cmap
    colors = _default_colors() if colors is None else colors

    order_sum_dict, orders = build_order_sum_dict(Pk_pair, ordered_pairs=ordered_pairs)
    order_color_dict = {order: cmap(i % 10) for i, order in enumerate(orders)}

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[3.3, 1.2], hspace=hspace)
    ax = fig.add_subplot(gs[0])
    ax_res = fig.add_subplot(gs[1], sharex=ax)

    if show_harmonics:
        draw_harmonics(
            ax, delta_v_list, L_box, Npix,
            colors=colors,
            harmonic_color_mode=harmonic_color_mode,
            harmonic_main_lw=harmonic_main_lw,
            harmonic_higher_lw=harmonic_higher_lw,
            harmonic_alpha=harmonic_alpha,
        )

    order_sum_handles = []
    if show_order_sums:
        for order in orders:
            curve = order_sum_dict[order].real[1:]
            c = order_color_dict[order]
            ax.plot(k[1:], curve, lw=order_sum_lw, color=c, alpha=order_sum_alpha)
            order_sum_handles.append(Line2D([0], [0], color=c, lw=order_sum_lw, label=rf"$\mathcal{{O}}({order})$"))

    ax.plot(k[1:], Pk_total.real[1:], lw=total_lw, color=total_color)
    ax.plot(k[1:], Pk_total_from_terms[1:], lw=recovered_lw, color=recovered_color, ls=recovered_ls)

    if use_symlog_main:
        linthresh = _nonzero_min([Pk_total.real[1:]]) if main_symlog_linthresh is None else main_symlog_linthresh
        apply_clean_symlog(
            ax, axis="y", linthresh=linthresh, linscale=symlog_linscale,
            base=symlog_base, label_minor=symlog_label_minor, sci_cutoff=symlog_sci_cutoff,
        )

    if show_zero_line_main:
        ax.axhline(0.0, color="k", lw=1.0, alpha=0.8, ls=":")

    if show_order_sums:
        leg_orders = ax.legend(handles=order_sum_handles, fontsize=legend_fontsize, loc="upper right", ncol=1, title="Norm. Sum per order")
        ax.add_artist(leg_orders)

    leg_total = ax.legend(
        handles=[
            Line2D([0], [0], color=total_color, lw=total_lw, label=r"Tot."),
            Line2D([0], [0], color=recovered_color, lw=recovered_lw, ls=recovered_ls, label=r"Tot. Sum"),
        ],
        fontsize=legend_fontsize,
        loc="upper left",
        ncol=2,
    )
    ax.add_artist(leg_total)

    if show_harmonics:
        add_harmonic_legend(ax, fontsize=harmonic_legend_fontsize)

    ax.tick_params(axis="both", labelsize=tick_fontsize, labelbottom=False)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)

    Pk_ref = Pk_total.real[1:]
    if show_order_sum_residuals:
        for order in orders:
            c = order_color_dict[order]
            curve = order_sum_dict[order].real[1:]
            ax_res.plot(k[1:], _safe_relative_residual(curve, Pk_ref), lw=1.0, color=c, alpha=order_sum_alpha)

    ax_res.plot(k[1:], _safe_relative_residual(Pk_total_from_terms[1:], Pk_ref), lw=2.0, color=recovered_color, ls=recovered_ls)

    if show_harmonics:
        draw_harmonics(
            ax_res, delta_v_list, L_box, Npix,
            colors=colors,
            harmonic_color_mode=harmonic_color_mode,
            harmonic_main_lw=harmonic_main_lw,
            harmonic_higher_lw=harmonic_higher_lw,
            harmonic_alpha=harmonic_alpha,
        )

    if use_symlog_residual:
        apply_clean_symlog(
            ax_res, axis="y", linthresh=residual_symlog_linthresh, linscale=symlog_linscale,
            base=symlog_base, label_minor=symlog_label_minor, sci_cutoff=symlog_sci_cutoff,
        )

    if show_zero_line_residual:
        ax_res.axhline(0.0, color="k", lw=1.0, alpha=0.8, ls=":")

    ax_res.tick_params(axis="both", labelsize=tick_fontsize)
    ax_res.set_xlabel(xlabel, fontsize=label_fontsize)
    ax_res.set_ylabel(residual_ylabel, fontsize=residual_label_fontsize)

    plt.tight_layout()
    plt.show()
    return fig, (ax, ax_res), order_sum_dict


# ============================================================
# Xi plotting
# ============================================================

def plot_xi_by_order(
    ordered_pairs,
    xi_pair,
    lags,
    delta_v_list,
    L_box,
    *,
    tolerance_threshold=1e-6,
    residual_mode="auto_reference",
    cmap=None,
    figsize=None,
    main_panel_height=5.0,
    residual_panel_height=3.0,
    hspace=0.08,
    xlim=(-0.01, 0.51),
    main_curve_lw=2.0,
    collapsed_sum_lw=1.5,
    sum_color="k",
    sum_ls=":",
    show_half_line=True,
    show_zero_line_main=True,
    show_zero_line_residual=True,
    show_delta_v_diff_lines=True,
    delta_v_diff_lw=1.5,
    delta_v_diff_ls="--",
    delta_v_diff_alpha=0.9,
    use_symlog_main=True,
    use_symlog_residual=False,
    symlog_linscale=1.0,
    symlog_base=10,
    symlog_label_minor=False,
    symlog_sci_cutoff=1e-2,
    show_order_badge=True,
    order_badge_fontsize=22,
    order_badge_boxstyle="round,pad=0.25",
    order_badge_facecolor="white",
    order_badge_edgecolor="black",
    order_badge_alpha=0.9,
    label_fontsize=22,
    residual_label_fontsize=20,
    tick_fontsize=22,
    legend_fontsize=18,
    legend_ncol_terms=2,
    delta_legend_fontsize=16,
    xlabel=r"$\Delta v / L_\mathrm{box}$",
    ylabel=r"$\xi_{\rm 1D}$",
    residual_ylabel=r"$\Delta$",
):
    if not ordered_pairs:
        raise ValueError("ordered_pairs cannot be empty.")

    cmap = plt.get_cmap("tab10") if cmap is None else cmap
    orders = sorted({pair_order(pair) for pair in ordered_pairs})
    if figsize is None:
        figsize = (14, 6.0 * len(orders))

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        nrows=2 * len(orders),
        ncols=1,
        height_ratios=[v for _ in orders for v in (main_panel_height, residual_panel_height)],
        hspace=hspace,
    )

    axes, axes_res = [], []
    for i in range(len(orders)):
        ax = fig.add_subplot(gs[2 * i], sharex=axes[0] if axes else None)
        ax_res = fig.add_subplot(gs[2 * i + 1], sharex=ax)
        axes.append(ax)
        axes_res.append(ax_res)

    xi_order_sum_dict = {}
    xi_order_sum_plot_dict = {}
    delta_handles = []

    for ax, ax_res, order in zip(axes, axes_res, orders):
        pairs_this_order = [pair for pair in ordered_pairs if pair_order(pair) == order]

        if show_half_line:
            ax.axvline(0.5, color="k", lw=1.0, alpha=0.8, ls=":")
            ax_res.axvline(0.5, color="k", lw=1.0, alpha=0.8, ls=":")
        if show_zero_line_main:
            ax.axvline(0.0, color="k", lw=1.0, alpha=0.8, ls=":")
        if show_zero_line_residual:
            ax_res.axvline(0.0, color="k", lw=1.0, alpha=0.8, ls=":")

        if show_delta_v_diff_lines:
            delta_handles = draw_delta_v_differences(ax, delta_v_list, L_box, cmap=cmap, lw=delta_v_diff_lw, ls=delta_v_diff_ls, alpha=delta_v_diff_alpha)
            _ = draw_delta_v_differences(ax_res, delta_v_list, L_box, cmap=cmap, lw=delta_v_diff_lw, ls=delta_v_diff_ls, alpha=delta_v_diff_alpha)

        entries, xi_sum_raw, xi_sum_full = collapse_order_pairs_xi(
            pairs_this_order, xi_pair, tolerance_threshold=tolerance_threshold
        )

        sum_diff = np.max(np.abs(xi_sum_raw - xi_sum_full))
        if sum_diff > tolerance_threshold:
            print(f"[warning] order {order}: collapsed/fallback representation differs from the full sum by max {sum_diff:.6e}")

        xi_order_sum_dict[order] = xi_sum_raw
        xi_order_sum_plot = xi_plot_curve(xi_sum_raw)
        xi_order_sum_plot_dict[order] = xi_order_sum_plot

        legend_terms = [Line2D([0], [0], color=sum_color, lw=collapsed_sum_lw, ls=sum_ls, label="Sum")]
        for i, entry in enumerate(entries):
            entry["color"] = cmap(i % 10)
            ax.plot(lags, entry["curve_plot"], lw=main_curve_lw, ls=entry["ls"], color=entry["color"], alpha=0.95)
            legend_terms.append(Line2D([0], [0], color=entry["color"], lw=main_curve_lw, ls=entry["ls"], label=entry["label"]))

        ax.plot(lags, xi_order_sum_plot, lw=collapsed_sum_lw, color=sum_color, ls=sum_ls, alpha=0.95)
        ax.axhline(0.0, color="k", lw=1.0, alpha=0.8, ls=":")

        leg_terms = ax.legend(
            handles=legend_terms,
            fontsize=legend_fontsize,
            loc="upper right",
            ncol=legend_ncol_terms,
        )
        ax.add_artist(leg_terms)

        if show_order_badge:
            _add_order_badge(
                ax, order,
                fontsize=order_badge_fontsize,
                boxstyle=order_badge_boxstyle,
                facecolor=order_badge_facecolor,
                edgecolor=order_badge_edgecolor,
                alpha=order_badge_alpha,
            )

        ax.set_ylabel(ylabel, fontsize=label_fontsize)
        ax.tick_params(axis="both", labelsize=tick_fontsize, labelbottom=False)
        ax.set_xlim(*xlim)

        if use_symlog_main:
            linthresh = _nonzero_min([xi_order_sum_plot])
            apply_clean_symlog(
                ax, axis="y", linthresh=linthresh, linscale=symlog_linscale,
                base=symlog_base, label_minor=symlog_label_minor, sci_cutoff=symlog_sci_cutoff,
            )

        nonzero_residuals = []

        if residual_mode == "sum":
            xi_ref = xi_order_sum_plot
            for entry in entries:
                res = _safe_relative_residual(entry["curve_plot"], xi_ref)
                ax_res.plot(lags, res, lw=1.5, ls=entry["ls"], color=entry["color"], alpha=0.95)
                nz = np.abs(res[np.abs(res) > 0])
                if len(nz):
                    nonzero_residuals.append(np.min(nz))

        elif residual_mode == "auto_reference":
            auto_entries = [entry for entry in entries if entry["is_auto"]]
            if auto_entries:
                xi_ref = auto_entries[0]["curve_plot"]
                for entry in auto_entries:
                    res = _safe_relative_residual(entry["curve_plot"], xi_ref)
                    ax_res.plot(lags, res, lw=1.5, ls=entry["ls"], color=entry["color"], alpha=0.95)
                    nz = np.abs(res[np.abs(res) > 0])
                    if len(nz):
                        nonzero_residuals.append(np.min(nz))
            else:
                ax_res.text(0.5, 0.5, "No auto terms in this order", transform=ax_res.transAxes, fontsize=16, va="center", ha="center")
        else:
            raise ValueError("residual_mode must be 'sum' or 'auto_reference'")

        ax_res.axhline(0.0, color="k", lw=1.0, alpha=0.8, ls=":")
        ax_res.set_ylabel(residual_ylabel, fontsize=residual_label_fontsize)
        ax_res.set_xlim(*xlim)

        if use_symlog_residual:
            apply_clean_symlog(
                ax_res, axis="y", linthresh=(np.min(nonzero_residuals) if nonzero_residuals else 1e-12),
                linscale=symlog_linscale, base=symlog_base,
                label_minor=symlog_label_minor, sci_cutoff=symlog_sci_cutoff,
            )

        ax_res.tick_params(axis="both", labelsize=tick_fontsize, labelbottom=(ax_res == axes_res[-1]))

    if show_delta_v_diff_lines and delta_handles:
        leg_delta = axes[0].legend(handles=delta_handles, fontsize=delta_legend_fontsize, loc="lower right", ncol=len(delta_handles))
        axes[0].add_artist(leg_delta)

    axes_res[-1].set_xlabel(xlabel, fontsize=label_fontsize)

    plt.tight_layout()
    plt.show()
    return fig, axes, axes_res, xi_order_sum_dict, xi_order_sum_plot_dict


def plot_xi_total_recovery(
    xi_total,
    xi_total_from_terms,
    xi_pair,
    lags,
    delta_v_list,
    L_box,
    *,
    ordered_pairs=None,
    cmap=None,
    figsize=(14, 6.5),
    hspace=0.06,
    xlim=(-0.01, 0.51),
    show_order_sums=True,
    show_order_sum_residuals=True,
    total_lw=3.0,
    decomp_lw=1.5,
    decomp_ls="--",
    total_color="k",
    decomp_color="red",
    order_sum_lw=2.0,
    order_sum_alpha=0.95,
    show_half_line=True,
    show_zero_line_main=True,
    show_zero_line_residual=True,
    show_delta_v_diff_lines=True,
    delta_v_diff_lw=1.5,
    delta_v_diff_ls="--",
    delta_v_diff_alpha=0.9,
    annotate_half_line=True,
    half_line_text=r"$L_\mathrm{box}/2$",
    half_line_text_fontsize=16,
    use_symlog_main=True,
    use_symlog_residual=True,
    main_symlog_linthresh=None,
    residual_symlog_linthresh=1e-3,
    symlog_linscale=1.0,
    symlog_base=10,
    symlog_label_minor=False,
    symlog_sci_cutoff=1e-2,
    label_fontsize=22,
    residual_label_fontsize=20,
    tick_fontsize=22,
    legend_fontsize=18,
    delta_legend_fontsize=16,
    xlabel=r"$\Delta v / L_\mathrm{box}$",
    ylabel=r"$\xi_{\rm 1D}$",
    residual_ylabel=r"$\Delta$",
):
    ordered_pairs = list(xi_pair.keys()) if ordered_pairs is None else ordered_pairs
    if not ordered_pairs:
        raise ValueError("ordered_pairs cannot be empty.")

    cmap = plt.get_cmap("tab10") if cmap is None else cmap

    xi_order_sum_dict, orders = build_order_sum_dict(xi_pair, ordered_pairs=ordered_pairs)
    xi_order_sum_plot_dict = {order: xi_plot_curve(xi_order_sum_dict[order]) for order in orders}
    xi_order_norm_plot_dict = {}
    order_color_dict = {order: cmap(i % 10) for i, order in enumerate(orders)}

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[3.3, 1.2], hspace=hspace)
    ax = fig.add_subplot(gs[0])
    ax_res = fig.add_subplot(gs[1], sharex=ax)

    if show_half_line:
        ax.axvline(0.5, color="k", lw=1.0, alpha=0.8, ls=":")
        ax_res.axvline(0.5, color="k", lw=1.0, alpha=0.8, ls=":")
    if show_zero_line_main:
        ax.axvline(0.0, color="k", lw=1.0, alpha=0.8, ls=":")
    if show_zero_line_residual:
        ax_res.axvline(0.0, color="k", lw=1.0, alpha=0.8, ls=":")

    delta_handles = []
    if show_delta_v_diff_lines:
        delta_handles = draw_delta_v_differences(ax, delta_v_list, L_box, cmap=cmap, lw=delta_v_diff_lw, ls=delta_v_diff_ls, alpha=delta_v_diff_alpha)
        _ = draw_delta_v_differences(ax_res, delta_v_list, L_box, cmap=cmap, lw=delta_v_diff_lw, ls=delta_v_diff_ls, alpha=delta_v_diff_alpha)

    order_sum_handles = []
    if show_order_sums:
        for order in orders:
            xi_order_norm_plot = xi_order_sum_plot_dict[order]
            xi_order_norm_plot_dict[order] = xi_order_norm_plot
            c = order_color_dict[order]
            ax.plot(lags, xi_order_norm_plot, lw=order_sum_lw, color=c, alpha=order_sum_alpha)
            order_sum_handles.append(Line2D([0], [0], color=c, lw=order_sum_lw, label=rf"$\mathcal{{O}}({order})$"))
    else:
        for order in orders:
            xi_order_norm_plot_dict[order] = xi_order_sum_plot_dict[order]

    ax.plot(lags, xi_total, lw=total_lw, color=total_color)
    ax.plot(lags, xi_total_from_terms, lw=decomp_lw, color=decomp_color, ls=decomp_ls)
    ax.axhline(0.0, color="k", lw=1.0, alpha=0.8, ls=":")

    if annotate_half_line and show_half_line:
        ymin, ymax = ax.get_ylim()
        ytext = ymin + 0.92 * (ymax - ymin)
        ax.text(0.5, ytext, half_line_text, ha="center", va="top", fontsize=half_line_text_fontsize)

    if use_symlog_main:
        arrays_for_scale = [xi_total, xi_total_from_terms] + [xi_order_norm_plot_dict[o] for o in orders]
        linthresh = _nonzero_min(arrays_for_scale) if main_symlog_linthresh is None else main_symlog_linthresh
        apply_clean_symlog(
            ax, axis="y", linthresh=linthresh, linscale=symlog_linscale,
            base=symlog_base, label_minor=symlog_label_minor, sci_cutoff=symlog_sci_cutoff,
        )

    if show_order_sums:
        leg_orders = ax.legend(handles=order_sum_handles, fontsize=legend_fontsize, loc="upper right", ncol=1, title="Norm. Sum per order")
        ax.add_artist(leg_orders)

    leg_total = ax.legend(
        handles=[
            Line2D([0], [0], color=total_color, lw=total_lw, label=r"Tot."),
            Line2D([0], [0], color=decomp_color, lw=decomp_lw, ls=decomp_ls, label=r"Tot. Sum"),
        ],
        fontsize=legend_fontsize,
        loc="upper left",
        ncol=2,
    )
    ax.add_artist(leg_total)

    if delta_handles:
        leg_delta = ax.legend(handles=delta_handles, fontsize=delta_legend_fontsize, loc="lower right", ncol=2)
        ax.add_artist(leg_delta)

    ax.tick_params(axis="both", labelsize=tick_fontsize, labelbottom=False)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)
    ax.set_xlim(*xlim)

    xi_ref = xi_total
    if show_order_sum_residuals:
        for order in orders:
            c = order_color_dict[order]
            ax_res.plot(
                lags,
                _safe_relative_residual(xi_order_norm_plot_dict[order], xi_ref),
                lw=1.0, color=c, alpha=order_sum_alpha,
            )

    ax_res.plot(lags, _safe_relative_residual(xi_total_from_terms, xi_ref), lw=2.0, color=decomp_color, ls=decomp_ls)
    ax_res.axhline(0.0, color="k", lw=1.0, alpha=0.8, ls=":")

    if use_symlog_residual:
        apply_clean_symlog(
            ax_res, axis="y", linthresh=residual_symlog_linthresh, linscale=symlog_linscale,
            base=symlog_base, label_minor=symlog_label_minor, sci_cutoff=symlog_sci_cutoff,
        )

    ax_res.tick_params(axis="both", labelsize=tick_fontsize)
    ax_res.set_xlabel(xlabel, fontsize=label_fontsize)
    ax_res.set_ylabel(residual_ylabel, fontsize=residual_label_fontsize)
    ax_res.set_xlim(*xlim)

    plt.tight_layout()
    plt.show()
    return fig, (ax, ax_res), xi_order_sum_dict, xi_order_sum_plot_dict


def plot_orders_superimposed(stats, kind="P", *, k_weighted=False, max_lag=None, delta_v_list=None,
                             yscale=None, linthresh=None, xlabel=None, ylabel=None, ax=None,
                             figsize=(14, 10)):
    r"""Total P1D or xi1D of a composite overflux split by order, all orders on one panel.

    Uses the exact decomposition of ``OverfluxStatistics1D`` (``Pk_total_by_order`` /
    ``xi_total_by_order``): order n is the sum of the weighted terms (A, B) with
    |A| + |B| = n, including the order-0 term (𝒞, 𝒞) and the order-1 terms (𝒞, A). The
    order-1 terms vanish when every single-line field has zero mean per skewer, as for
    ``DeltaFieldBuilder``. Orders below 1e-12 max|total| are not drawn. The sum of the
    orders is drawn over the total (red dashed on black, as in ``plot_p1d_total_recovery``);
    the lower panel shows (sum - total) / max|total|, i.e. rounding error only.

    Parameters
    ----------
    stats : OverfluxStatistics1D (P1D / xi1D computed if needed).
    kind : "P" or "xi"
    k_weighted : bool
        P only: plot k P / pi (dimensionless) instead of P [km/s].
    max_lag : float or None
        xi only: show 0 <= Delta v <= max_lag (default: half the skewer, N_v dv / 2).
    delta_v_list : sequence or None
        Line offsets (e.g. ``LineBundle.delta_v``) [x units]; xi only: vertical lines at
        every separation |Delta v_i - Delta v_j| (folded into the periodic box).
    yscale : "symlog", "linear", "log" or None
        Main-panel y scale; None -> "symlog" for P (orders change sign), "linear" for xi.
    linthresh : float or None
        Symlog linear threshold; None -> 1e-3 max|total|.
    xlabel, ylabel : str or None
        Override the default labels (k [s/km], Delta v [km/s]), e.g. for box units.

    Returns
    -------
    fig, axes (main, residual)
    """
    if kind not in ("P", "xi"):
        raise ValueError(f"kind must be 'P' or 'xi', got {kind!r}.")
    L = stats.N_v * stats.dv
    if kind == "P":
        stats.compute_p1d()
        x = np.asarray(stats.k)[1:]
        sel = slice(1, None)
        w = x / np.pi if k_weighted else 1.0
        tot, by_order = stats.Pk_total, stats.Pk_total_by_order
    else:
        stats.compute_xi1d()
        lags = np.asarray(stats.lags)
        sel = (lags >= 0) & (lags <= (0.5 * L if max_lag is None else max_lag))
        x = lags[sel]
        w = 1.0
        tot, by_order = stats.xi_total, stats.xi_total_by_order
    tot = w * np.asarray(tot)[sel]
    by_order = {n: w * np.asarray(v)[sel] for n, v in by_order.items()}
    scale = np.max(np.abs(tot)) if np.any(tot) else 1.0
    total_sum = sum(by_order.values())

    fig, (ax0, ax1) = _figure(ax, 2, 1, figsize, sharex=True, gridspec_kw={"height_ratios": [3, 1.2]})
    cmap = plt.get_cmap("tab10")
    for n, y in sorted(by_order.items()):
        if np.max(np.abs(y)) > 1e-12 * scale:
            ax0.plot(x, y, lw=2.0, color=cmap(n % 10), alpha=0.9, label=rf"$\mathcal{{O}}({n})$")
    ax0.plot(x, tot, lw=3.0, color="k", label="total", zorder=3)
    ax0.plot(x, total_sum, lw=1.5, ls="--", color="red", label=r"$\sum$ orders", zorder=4)
    ax1.plot(x, (total_sum - tot) / scale, lw=1.5, color="red")
    ax1.axhline(0.0, color="k", lw=1.0, ls=":")
    if kind == "xi" and delta_v_list is not None:
        for a in (ax0, ax1):
            draw_delta_v_differences(a, list(np.asarray(delta_v_list, dtype=float)), L, lw=1.2)

    yscale = ("symlog" if kind == "P" else "linear") if yscale is None else yscale
    if yscale == "symlog":
        lt = 1e-3 * scale if linthresh is None else linthresh
        apply_clean_symlog(ax0, axis="y", linthresh=lt)
        lo = min(float(np.min(y)) for y in [tot, total_sum, *by_order.values()])
        ax0.set_ylim(bottom=3.0 * lo if lo < -lt else -lt, top=3.0 * scale)
    else:
        ax0.set_yscale(yscale)
    if kind == "P":
        ax0.set_xscale("log")
        ax0.set_ylabel(ylabel or (r"$k P_{\rm 1D}/\pi$" if k_weighted else r"$P_{\rm 1D}\ [\mathrm{km\,s^{-1}}]$"))
        ax1.set_xlabel(xlabel or r"$k\ [\mathrm{s\,km^{-1}}]$")
    else:
        ax0.set_ylabel(ylabel or r"$\xi_{\rm 1D}$")
        ax1.set_xlabel(xlabel or r"$\Delta v\ [\mathrm{km\,s^{-1}}]$")
    ax1.set_ylabel(r"$(\sum - {\rm tot})/{\rm max}|{\rm tot}|$")
    ax0.legend(frameon=False, fontsize="small", ncol=2)
    return fig, np.array([ax0, ax1])


# ============================================================
# 4. How P(k) is computed and recovered: 1D (OverfluxStatistics1D) and shared helpers
# ============================================================

def hybrid_k_edges(k_fund, k_max, n_lin=12, n_log=18):
    """k bin edges: linear, one mode per bin, up to (n_lin + 1/2) k_fund, then logarithmic.

    The linear edges sit at (j + 1/2) k_fund (j = 0..n_lin), so each low-k bin holds exactly
    one Fourier mode; then ``n_log`` log-spaced bins up to ``k_max``. Units of ``k_fund``.
    """
    lin = (np.arange(n_lin + 1) + 0.5) * k_fund
    lin = lin[lin < k_max]
    return np.concatenate([lin[:-1], np.geomspace(lin[-1], k_max, n_log + 1)])


def sample_mean_and_error(samples):
    """Mean over the first axis and its standard error, std(ddof=1) / sqrt(n).

    ``samples`` is a sequence of equally shaped arrays (e.g. one statistic per realisation)
    or an array whose first axis runs over samples. Returns (mean, error) as NumPy arrays.
    """
    s = np.asarray([np.asarray(x) for x in samples])
    n = s.shape[0]
    err = s.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.full(s.shape[1:], np.nan)
    return s.mean(axis=0), err


def pk_sigma(P_input, n_modes, n_real=1):
    """Gaussian error of a binned power spectrum, P sqrt(2 / (n_modes n_real)).

    ``n_modes`` counts +k and -k separately (as returned by ``bin_p1d`` / ``bin_p3d_kmod``),
    hence the factor 2. Not valid for skewers that share a box (they are correlated).
    """
    n = np.asarray(n_modes, dtype=float)
    return np.asarray(P_input, dtype=float) * np.sqrt(2.0 / (np.where(n > 0, n, np.nan) * n_real))


def recovery_summary(meas, expect, sigma, label="", ratio=True):
    """One-line summary of measured against expected values with 1-sigma errors.

    Reports chi^2 / dof (dof = number of finite points), the largest |pull|, the fraction of
    points within 2 sigma and, if ``ratio`` (use False for xi, whose input approaches zero),
    the mean ratio meas / expect.
    Points of correlated statistics (xi lags) are not independent: there chi^2 / dof is
    indicative only.
    """
    m, e, s = (np.ravel(np.asarray(x, dtype=float)) for x in (meas, expect, sigma))
    ok = np.isfinite(m) & np.isfinite(e) & np.isfinite(s) & (s > 0)
    pull = (m[ok] - e[ok]) / s[ok]
    out = (f"{label}: chi2/dof = {np.mean(pull ** 2):.2f} ({ok.sum()} points), "
           f"max|pull| = {np.max(np.abs(pull)):.2f}, {100 * np.mean(np.abs(pull) < 2):.0f}% within 2 sigma")
    if ratio and np.all(e[ok] != 0):
        out += f", <meas/input> = {np.mean(m[ok] / e[ok]):.3f}"
    return out


def plot_pk_recovery(binned, P_input, *, power=None, n_real=1, sigma=None, label="measured",
                     color=None, k_label=r"$k\ [\mathrm{s\,km^{-1}}]$",
                     P_label=r"$P_{\rm 1D}(k)\ [\mathrm{km\,s^{-1}}]$", ax=None, figsize=(13, 11)):
    """Binned P(k) against the input, with a ratio panel and 1-sigma errors.

    Parameters
    ----------
    binned : tuple
        ``(P_b, k_b, k_edges, n_modes)`` as returned by ``OverfluxStatistics1D.bin_p1d`` or
        ``OverfluxStatistics3D.bin_p3d_kmod``; ``P_b`` may be a mean over ``n_real``
        realisations with the same grid.
    P_input : array (n_bins,)
        Input spectrum averaged over the same bins with the same weights, e.g.
        ``stats.bin_p1d(edges, Pk=grf.power(stats.k))[0]``.
    power : callable or None
        Input P(k) drawn as a line (e.g. ``PowerLawGRF.power``).
    n_real : int
        Realisations averaged in ``P_b`` (used by the default errors).
    sigma : array (n_bins,) or None
        Errors of ``P_b``; None -> Gaussian ``pk_sigma(P_input, n_modes, n_real)``.
    ax : (Axes, Axes) or None
        Main and ratio panels; pass the returned axes again to overlay another measurement.

    Returns
    -------
    fig, axes (main, ratio)
    """
    P_b, k_b, edges, n_b = (np.asarray(x, dtype=float) for x in binned)
    P_in = np.asarray(P_input, dtype=float)
    sig = pk_sigma(P_in, n_b, n_real) if sigma is None else np.asarray(sigma, dtype=float)
    ok = np.isfinite(P_b) & (n_b > 0) & (P_in > 0)
    fig, (ax0, ax1) = _figure(ax, 2, 1, figsize, sharex=True, gridspec_kw={"height_ratios": [3, 1.2]})
    if power is not None:
        kk = np.geomspace(edges[edges > 0][0], edges[-1], 400)
        first = not any(l.get_label() == "input" for l in ax0.lines)
        ax0.plot(kk, np.asarray(power(kk)), color="k", lw=2, label="input" if first else None)
    eb = ax0.errorbar(k_b[ok], P_b[ok], yerr=sig[ok], fmt="o", ms=7, color=color, capsize=3, label=label)
    ax1.errorbar(k_b[ok], P_b[ok] / P_in[ok], yerr=sig[ok] / P_in[ok], fmt="o", ms=7,
                 color=eb[0].get_color(), capsize=3)
    ax1.axhline(1.0, color="k", ls="--", lw=1.5)
    ax0.set(xscale="log", yscale="log", ylabel=P_label)
    ax1.set(xscale="log", xlabel=k_label, ylabel=r"$\hat P / P_{\rm in}$")
    ax0.legend(frameon=False, fontsize="small")
    return fig, np.array([ax0, ax1])


def _xi_recovery_panels(ax0, ax1, r, xi, xi_in, sigma, r_max, label, color):
    """Measured xi (points) and input (black line) on ax0; pull (or difference) on ax1."""
    m = np.abs(r) <= (np.inf if r_max is None else r_max)
    ax0.plot(r[m], xi_in[m], color="k", lw=1.5, zorder=3)
    if sigma is None:
        pts, = ax0.plot(r[m], xi[m], "o", ms=5, color=color, label=label)
        ax1.plot(r[m], xi[m] - xi_in[m], "o", ms=5, color=pts.get_color())
    else:
        eb = ax0.errorbar(r[m], xi[m], yerr=sigma[m], fmt="o", ms=5, color=color, capsize=2, label=label)
        ax1.plot(r[m], (xi[m] - xi_in[m]) / sigma[m], "o", ms=5, color=eb[0].get_color())
    ax0.axhline(0.0, color="0.5", lw=1)
    ax1.axhline(0.0, color="k", ls="--", lw=1.5)
    ax1.set_ylabel(r"$\Delta\xi$" if sigma is None else r"$(\hat\xi-\xi_{\rm in})/\sigma$")


def plot_xi_recovery(stats, xi_input, *, sigma=None, max_lag=None, label="measured", color=None,
                     ax=None, figsize=(13, 11)):
    """xi1D (``stats.xi_total`` on the centred lags ``stats.lags``) against the input.

    Parameters
    ----------
    stats : OverfluxStatistics1D (xi1D is computed if needed).
    xi_input : (N_v,) array, centred, e.g.
        ``np.fft.fftshift(two_point.xi_from_power(grf.power(stats.k), N_v, dv))``.
    sigma : (N_v,) array or None
        None -> standard error of the mean over skewers, from ``stats.xi_total_per_sample``.
    max_lag : float or None
        Show only |Delta v| <= max_lag [km/s].

    The lower panel shows the pull (xi - xi_in) / sigma: a ratio is undefined where xi_in
    crosses zero. Lags are correlated, so the pulls are not independent.

    Returns fig, axes (main, pull).
    """
    stats.compute_xi1d()
    if sigma is None:
        _, sigma = sample_mean_and_error(stats.xi_total_per_sample)
    fig, (ax0, ax1) = _figure(ax, 2, 1, figsize, sharex=True, gridspec_kw={"height_ratios": [3, 1.2]})
    _xi_recovery_panels(ax0, ax1, np.asarray(stats.lags), np.asarray(stats.xi_total),
                        np.asarray(xi_input), np.asarray(sigma), max_lag, label, color)
    ax0.plot([], [], color="k", lw=1.5, label="input")
    ax0.set(ylabel=r"$\xi_{\rm 1D}(\Delta v)$")
    ax1.set(xlabel=r"$\Delta v\ [\mathrm{km\,s^{-1}}]$")
    ax0.legend(frameon=False, fontsize="small")
    return fig, np.array([ax0, ax1])


def plot_p1d_from_skewers(stats, *, power=None, n_show=3, ax=None, figsize=(28, 9)):
    """Real space -> Fourier space for a few skewers.

    Left: delta(v_j) of the first ``n_show`` skewers (``stats.total``), v_j = j dv [km/s],
    offset vertically. Right: their single-skewer periodograms
    P_s(k) = (dv / N_v) |rfft(delta_s)|^2 (``stats.Pk_total_per_sample``), the mean over all
    skewers (``stats.Pk_total``) and the input ``power(k)``. k = 0 is not shown (P = 0).

    Returns fig, axes (real, Fourier).
    """
    stats.compute_p1d()
    total = np.asarray(stats.total)
    k = np.asarray(stats.k)[1:]
    P_s = np.asarray(stats.Pk_total_per_sample)[:, 1:]
    v = np.arange(stats.N_v) * stats.dv
    off = 4.0 * total.std()
    fig, (ax0, ax1) = _figure(ax, 1, 2, figsize)
    for i in range(min(n_show, total.shape[0])):
        c = f"C{i}"
        ax0.plot(v, total[i] - i * off, color=c, lw=1.2)
        ax1.plot(k, P_s[i], ".-", color=c, ms=4, lw=0.6, alpha=0.8, label=f"skewer {i}")
    ax1.plot(k, np.asarray(stats.Pk_total)[1:], color="C3", lw=3,
             label=f"mean of {total.shape[0]} skewers")
    if power is not None:
        ax1.plot(k, np.asarray(power(k)), color="k", lw=2, ls="--", label="input")
    ax0.set(xlabel=r"$v\ [\mathrm{km\,s^{-1}}]$", ylabel=r"$\delta(v)$ + offset",
            title="real space", yticks=[])
    ax1.set(xscale="log", yscale="log", xlabel=r"$k\ [\mathrm{s\,km^{-1}}]$",
            ylabel=r"$\frac{dv}{N_v}|\mathrm{rfft}\,\delta|^2\ [\mathrm{km\,s^{-1}}]$",
            title="Fourier space", ylim=(np.percentile(P_s[:n_show], 1), 3 * P_s[:n_show].max()))
    ax1.legend(frameon=False, fontsize="small")
    return fig, np.array([ax0, ax1])


def plot_p1d_mode_binning(stats, k_edges, *, power=None, n_show=64, ax=None, figsize=(15, 14)):
    """Binning in k and mode counting.

    Top: every mode of the first ``n_show`` skewers (grey), the bin edges (vertical lines),
    the binned P1D of all skewers (``stats.bin_p1d(k_edges)``) and the input.
    Bottom: modes per bin (``n_modes`` of ``bin_p1d``): each skewer gives 2 modes (+k, -k)
    per k and 1 at Nyquist, so sigma[P_bin] / P = sqrt(2 / n_modes).

    Returns fig, axes (P, n_modes).
    """
    P_b, k_b, edges, n_b = (np.asarray(x) for x in stats.bin_p1d(k_edges))
    k = np.asarray(stats.k)[1:]
    P_s = np.asarray(stats.Pk_total_per_sample)[:n_show, 1:]
    fig, (ax0, ax1) = _figure(ax, 2, 1, figsize, sharex=True, gridspec_kw={"height_ratios": [3, 1.3]})
    ax0.plot(np.broadcast_to(k, P_s.shape).ravel(), P_s.ravel(), ".", color="0.6", ms=3,
             alpha=0.4, rasterized=True, label=f"modes of {P_s.shape[0]} skewers")
    for a in (ax0, ax1):
        for e in edges:
            a.axvline(e, color="tab:cyan", lw=0.8, alpha=0.7)
    ok = np.isfinite(P_b)
    ax0.plot(k_b[ok], P_b[ok], "o", color="C3", ms=8, label=f"bin mean, {stats.n_samples_kept} skewers")
    if power is not None:
        ax0.plot(k, np.asarray(power(k)), color="k", lw=2, label="input")
    ax0.set(xscale="log", yscale="log", ylabel=r"$P_{\rm 1D}(k)\ [\mathrm{km\,s^{-1}}]$",
            ylim=(np.percentile(P_s, 1), 3 * P_s.max()))
    ax0.legend(frameon=False, fontsize="small", loc="lower left")
    ax1.stairs(n_b, edges, fill=True, color="C0", alpha=0.6)
    ax1.set(xscale="log", yscale="log", xlabel=r"$k\ [\mathrm{s\,km^{-1}}]$", ylabel=r"$N_{\rm modes}$")
    return fig, np.array([ax0, ax1])


# ============================================================
# 5. How P(k) is computed and recovered: 3D (OverfluxStatistics3D)
# ============================================================
# z is the line of sight; spectra use the rfft layout (Nx, Ny, N_pix//2 + 1), k_par >= 0
# (the k_par < 0 half is the complex conjugate); k_perp = sqrt(k_x^2 + k_y^2). Units come
# from ``units``: UNITS_3D is the class convention (cMpc transverse, km/s along the LoS);
# UNITS_3D_COMOVING is for boxes with one length unit on all axes, where |k| and
# mu = k_par / |k| are defined.

UNITS_3D = {"x_perp": r"\mathrm{cMpc}", "x_par": r"\mathrm{km\,s^{-1}}",
            "k_perp": r"\mathrm{cMpc^{-1}}", "k_par": r"\mathrm{s\,km^{-1}}",
            "P": r"\mathrm{cMpc^2\,km\,s^{-1}}"}
UNITS_3D_COMOVING = {"x_perp": r"\mathrm{cMpc}", "x_par": r"\mathrm{cMpc}",
                     "k_perp": r"\mathrm{cMpc^{-1}}", "k_par": r"\mathrm{cMpc^{-1}}",
                     "P": r"\mathrm{cMpc^3}"}


def _lbl(sym, unit):
    return rf"${sym}\ [{unit}]$"


def _aspect(units):
    return "equal" if units["x_perp"] == units["x_par"] else "auto"


def _kpar_cell_edges(k_par):
    """Edges of the k_par cells (one rfft plane each), clipped at 0."""
    dk = float(k_par[1] - k_par[0])
    return np.clip((np.arange(k_par.size + 1) - 0.5) * dk, 0.0, None)


def _cells(binned):
    """Flatten a ``bin_p3d_kperp`` result into per-cell arrays (k_perp, k_par, P, n_modes)."""
    P_b, k_b, _, k_par, n_b = (np.asarray(x, dtype=float) for x in binned)
    kp, kz = np.meshgrid(k_b, k_par, indexing="ij")
    return kp, kz, P_b, n_b


def plot_field_slices(delta, grf, *, units=UNITS_3D, ax=None, figsize=(32, 10)):
    """Three central orthogonal slices of a 3D field (z = line of sight).

    Parameters
    ----------
    delta : (Nx, Ny, Nz) array, e.g. ``PowerLawGRF.sample(key)``.
    grf : object with the box sides ``L = (Lx, Ly, Lz)`` (e.g. the ``PowerLawGRF``).

    Returns fig, axes (z-slice, y-slice, x-slice).
    """
    d = np.asarray(delta)
    Nx, Ny, Nz = d.shape
    Lx, Ly, Lz = grf.L
    xp, xz = units["x_perp"], units["x_par"]
    panels = [(d[:, :, Nz // 2], Lx, Ly, _lbl("x", xp), _lbl("y", xp), r"transverse, $z=L_z/2$"),
              (d[:, Ny // 2, :], Lx, Lz, _lbl("x", xp), _lbl("z", xz), r"$y=L_y/2$ (contains LoS)"),
              (d[Nx // 2, :, :], Ly, Lz, _lbl("y", xp), _lbl("z", xz), r"$x=L_x/2$ (contains LoS)")]
    s = 3.0 * d.std()
    fig, axs = _figure(ax, 1, 3, figsize)
    for a, (img, La, Lb, xl, yl, title) in zip(axs, panels):
        im = a.imshow(img.T, origin="lower", cmap="RdBu_r", vmin=-s, vmax=s, extent=[0, La, 0, Lb],
                      interpolation="nearest", aspect=_aspect(units))
        a.set(xlabel=xl, ylabel=yl, title=title)
    fig.colorbar(im, ax=list(axs), label=r"$\delta$", shrink=0.8)
    return fig, axs


def plot_kspace_slices(stats, *, P=None, units=UNITS_3D, s=None, ax=None, figsize=(32, 10)):
    """|delta_k|^2 / V of the modes on three central planes of the rfft grid.

    Each mode is a marker at its (k_i, k_j), coloured by P (default ``stats.Pk3d_total``):
    the k_par = 0 plane (k_x, k_y), and the k_y = 0 and k_x = 0 planes (k_par >= 0). The
    mode lattice spacing is 2 pi / L_i per axis, so anisotropic boxes give rectangular
    lattices. Colours use the 2-98 percentiles (single modes scatter as chi^2_2).

    Returns fig, axes (k_par = 0, k_y = 0, k_x = 0).
    """
    stats.compute_p3d()
    P = np.asarray(stats.Pk3d_total if P is None else P)
    kx, ky, kz = (np.asarray(a) for a in (stats.kx, stats.ky, stats.k_par))
    kp, kl = units["k_perp"], units["k_par"]
    panels = [(kx, ky, P[:, :, 0], _lbl("k_x", kp), _lbl("k_y", kp), r"$k_\parallel=0$ (transverse)"),
              (kx, kz, P[:, 0, :], _lbl("k_x", kp), _lbl(r"k_\parallel", kl), r"$k_y=0$"),
              (ky, kz, P[0, :, :], _lbl("k_y", kp), _lbl(r"k_\parallel", kl), r"$k_x=0$")]
    norm = _log_norm(np.concatenate([p[2].ravel() for p in panels]))
    fig, axs = _figure(ax, 1, 3, figsize)
    for a, (ka, kb, Pp, xl, yl, title) in zip(axs, panels):
        A, B = np.meshgrid(ka, kb, indexing="ij")
        m = Pp > 0                                            # drops k = 0
        size = s if s is not None else min(120.0, 4.0e4 / m.sum())
        sc = a.scatter(A[m], B[m], c=Pp[m], s=size, cmap="magma", norm=norm, edgecolors="none")
        a.set(xlabel=xl, ylabel=yl, title=title, aspect=_aspect(units))
    fig.colorbar(sc, ax=list(axs), label=_lbl(r"\hat P(\mathbf{k})", units["P"]), shrink=0.8)
    return fig, axs


def plot_kspace_3d(stats, *, k_perp_radii=None, k_par_levels=None, P=None, n_show=6000, seed=0,
                   units=UNITS_3D, ax=None, figsize=(15, 13)):
    """All rfft modes (k_x, k_y, k_par >= 0) in 3D, coloured by P, with the binning geometry.

    Blue wireframe cylinders: k_perp = const (radii ``k_perp_radii``); green discs:
    k_par = const (``k_par_levels``); together they cut k space into the (k_perp, k_par)
    cells of ``bin_p3d_kperp``. ``n_show`` random modes are drawn (None = all). ``ax`` must be
    a 3D axes.

    Returns fig, ax.
    """
    stats.compute_p3d()
    P = np.asarray(stats.Pk3d_total if P is None else P)
    kx, ky, kz = (np.asarray(a) for a in (stats.kx, stats.ky, stats.k_par))
    K = np.stack(np.broadcast_arrays(kx[:, None, None], ky[None, :, None], kz[None, None, :]), -1)
    K, Pf = K.reshape(-1, 3), P.ravel()
    m = Pf > 0
    K, Pf = K[m], Pf[m]
    if n_show is not None and Pf.size > n_show:
        sel = np.random.default_rng(seed).choice(Pf.size, n_show, replace=False)
        K, Pf = K[sel], Pf[sel]
    kp_max, kz_max = np.hypot(np.abs(kx).max(), np.abs(ky).max()), kz.max()
    k_perp_radii = kp_max * np.array([0.25, 0.5, 0.9]) if k_perp_radii is None else k_perp_radii
    k_par_levels = kz_max * np.array([0.25, 0.5, 0.9]) if k_par_levels is None else k_par_levels

    fig, (a,) = _figure(ax, 1, 1, figsize, layout=None, subplot_kw={"projection": "3d"})
    sc = a.scatter(K[:, 0], K[:, 1], K[:, 2], c=Pf, cmap="magma", norm=_log_norm(Pf), s=10,
                   alpha=0.4, edgecolors="none", depthshade=False, rasterized=True)
    th = np.linspace(0, 2 * np.pi, 64)
    TH, Z = np.meshgrid(th, [0.0, kz_max])
    for r in k_perp_radii:
        a.plot_wireframe(r * np.cos(TH), r * np.sin(TH), Z, rstride=1, cstride=4,
                         color="deepskyblue", lw=1.2, alpha=0.7)
    TH, R = np.meshgrid(th, [0.0, max(k_perp_radii)])
    for z in k_par_levels:
        a.plot_surface(R * np.cos(TH), R * np.sin(TH), np.full_like(R, z), color="limegreen",
                       alpha=0.15, linewidth=0, shade=False)
    a.set_xlabel(_lbl("k_x", units["k_perp"]), labelpad=25)
    a.set_ylabel(_lbl("k_y", units["k_perp"]), labelpad=25)
    a.set_zlabel(_lbl(r"k_\parallel", units["k_par"]), labelpad=25)
    a.view_init(elev=18, azim=-55)
    fig.colorbar(sc, ax=a, shrink=0.6, pad=0.12, label=_lbl(r"\hat P(\mathbf{k})", units["P"]))
    return fig, a


def plot_kperp_kpar_binning(stats, k_perp_edges, *, P=None, units=UNITS_3D, ax=None,
                            figsize=(40, 11)):
    """(k_perp, k_par) binning and mode counting.

    Left: k_par = 0 plane with the k_perp edges as circles (modes outside the last edge
    are not binned). Middle: every rfft mode at (k_perp, k_par), with the k_perp edges
    (vertical) and the k_par cells (horizontal, one per rfft plane, as kept by
    ``bin_p3d_kperp``). Right: full-grid modes per cell, ``n_modes`` of
    ``stats.bin_p3d_kperp(k_perp_edges=k_perp_edges)``.

    Returns fig, axes (plane, cells, n_modes).
    """
    stats.compute_p3d()
    P = np.asarray(stats.Pk3d_total if P is None else P)
    kx, ky, kz = (np.asarray(a) for a in (stats.kx, stats.ky, stats.k_par))
    kperp = np.asarray(stats.k_perp_2d)
    edges = np.asarray(k_perp_edges, dtype=float)
    z_edges = _kpar_cell_edges(kz)
    *_, n_b = stats.bin_p3d_kperp(k_perp_edges=edges)
    norm = _log_norm(P)
    kp, kl = units["k_perp"], units["k_par"]
    fig, (a0, a1, a2) = _figure(ax, 1, 3, figsize)

    X, Y = np.meshgrid(kx, ky, indexing="ij")
    m = P[:, :, 0] > 0
    a0.scatter(X[m], Y[m], c=P[:, :, 0][m], cmap="magma", norm=norm, s=min(120.0, 4.0e4 / m.sum()),
               edgecolors="none")
    th = np.linspace(0, 2 * np.pi, 200)
    for e in edges:
        a0.plot(e * np.cos(th), e * np.sin(th), color="tab:cyan", lw=0.8)
    a0.set(xlabel=_lbl("k_x", kp), ylabel=_lbl("k_y", kp), title=r"$k_\parallel=0$ + $k_\perp$ edges",
           aspect="equal", xlim=(kx.min(), kx.max()), ylim=(ky.min(), ky.max()))

    KP = np.broadcast_to(kperp[:, :, None], P.shape)
    KZ = np.broadcast_to(kz[None, None, :], P.shape)
    m = P > 0
    sc = a1.scatter(KP[m], KZ[m], c=P[m], cmap="magma", norm=norm, s=16, alpha=0.8,
                    edgecolors="none", rasterized=True)
    for e in edges:
        a1.axvline(e, color="tab:cyan", lw=0.8)
    for e in z_edges:
        a1.axhline(e, color="limegreen", lw=0.5, alpha=0.6)
    a1.set(xlabel=_lbl(r"k_\perp", kp), ylabel=_lbl(r"k_\parallel", kl), title="all modes + cells",
           xlim=(0, KP.max()), ylim=(0, z_edges[-1]))
    fig.colorbar(sc, ax=[a0, a1], label=_lbl(r"\hat P(\mathbf{k})", units["P"]), shrink=0.8)

    n = np.where(np.asarray(n_b) > 0, np.asarray(n_b), np.nan)
    pm = a2.pcolormesh(edges, z_edges, n.T, cmap="viridis", norm=_log_norm(n, (0, 100)))
    a2.set(xlabel=_lbl(r"k_\perp", kp), ylabel=_lbl(r"k_\parallel", kl), title="modes per cell",
           xscale="log" if edges[0] > 0 else "linear")
    fig.colorbar(pm, ax=a2, label=r"$N_{\rm modes}$", shrink=0.8)
    return fig, np.array([a0, a1, a2])


def plot_p3d_kperp_kpar(binned, *, P_ref=None, units=UNITS_3D, ax=None, figsize=(15, 11)):
    """P(k_perp, k_par) map from ``bin_p3d_kperp`` (k_par at full rfft resolution).

    Parameters
    ----------
    binned : tuple
        ``(P_b, k_perp_centers, k_perp_edges, k_par, n_modes)`` from
        ``OverfluxStatistics3D.bin_p3d_kperp`` (P_b may be a mean over realisations).
    P_ref : (n_bins, N_kpar) array or None
        If given, the map shows P_b / P_ref, e.g. P_ref =
        ``stats.bin_p3d_kperp(k_perp_edges=e, Pk3d_input=P_iso)[0]``.

    Returns fig, ax.
    """
    P_b, _, edges, k_par, _ = binned
    Z = np.asarray(P_b, dtype=float) / (1.0 if P_ref is None else np.asarray(P_ref, dtype=float))
    edges = np.asarray(edges, dtype=float)
    fig, (a,) = _figure(ax, 1, 1, figsize)
    pm = a.pcolormesh(edges, _kpar_cell_edges(np.asarray(k_par)), Z.T, cmap="viridis",
                      norm=_log_norm(Z))
    a.set(xlabel=_lbl(r"k_\perp", units["k_perp"]), ylabel=_lbl(r"k_\parallel", units["k_par"]),
          xscale="log" if edges[0] > 0 else "linear")
    cl = r"$\hat P/P_{\rm ref}$" if P_ref is not None else _lbl(r"\hat P(k_\perp,k_\parallel)", units["P"])
    fig.colorbar(pm, ax=a, label=cl)
    return fig, a


def plot_p3d_cells(binned, *, x="kmod", P_ref=None, reference=None, n_real=1, k_range=None,
                   units=UNITS_3D_COMOVING, ax=None, figsize=(15, 11)):
    """Every (k_perp, k_par) cell of ``bin_p3d_kperp`` as one point.

    x = "kmod": P (or P / P_ref) against |k| = sqrt(k_perp^2 + k_par^2), coloured by
    mu^2 = (k_par / |k|)^2. For an isotropic field all cells collapse onto P(|k|).
    x = "mu2": against mu^2, coloured by |k| (anisotropy, e.g. Kaiser (1 + beta mu^2)^2).
    Needs one length unit on all axes (``UNITS_3D_COMOVING``).

    Parameters
    ----------
    binned : tuple from ``bin_p3d_kperp`` (see ``plot_p3d_kperp_kpar``).
    P_ref : array or None   Plot P_b / P_ref instead of P_b.
    reference : callable or None   Curve f(x) drawn in black (e.g. ``grf.power``).
    n_real : int   Realisations in P_b; errors sigma = y sqrt(2 / (n_modes n_real)).
    k_range : (k_lo, k_hi) or None   Only cells with k_lo <= |k| <= k_hi.

    Returns fig, ax.
    """
    kp, kz, P_b, n_b = _cells(binned)
    kmod = np.hypot(kp, kz)
    y = P_b / (1.0 if P_ref is None else np.asarray(P_ref, dtype=float))
    ok = np.isfinite(y) & (y > 0) & (kmod > 0) & (n_b > 0)
    if k_range is not None:
        ok &= (kmod >= k_range[0]) & (kmod <= k_range[1])
    mu2 = (kz[ok] / kmod[ok]) ** 2
    xv, cv = (kmod[ok], mu2) if x == "kmod" else (mu2, kmod[ok])
    yv = y[ok]
    fig, (a,) = _figure(ax, 1, 1, figsize)
    a.errorbar(xv, yv, yerr=yv * np.sqrt(2.0 / (n_b[ok] * n_real)), fmt="none", ecolor="0.7",
               lw=0.8, zorder=1)
    sc = a.scatter(xv, yv, c=cv, s=30, cmap="coolwarm" if x == "kmod" else "viridis",
                   norm=None if x == "kmod" else _log_norm(cv, (0, 100)), zorder=2)
    if reference is not None:
        xx = np.geomspace(xv.min(), xv.max(), 300) if x == "kmod" else np.linspace(0, 1, 200)
        a.plot(xx, np.asarray(reference(xx)), color="k", lw=2, zorder=3, label="reference")
        a.legend(frameon=False, fontsize="small")
    k_l = _lbl(r"|k|", units["k_perp"])
    yl = r"$\hat P/P_{\rm ref}$" if P_ref is not None else _lbl(r"\hat P", units["P"])
    if x == "kmod":
        a.set(xscale="log", yscale="log", xlabel=k_l, ylabel=yl)
        fig.colorbar(sc, ax=a, label=r"$\mu^2=(k_\parallel/|k|)^2$")
    else:
        a.set(xlabel=r"$\mu^2=(k_\parallel/|k|)^2$", ylabel=yl)
        fig.colorbar(sc, ax=a, label=k_l)
    return fig, a


def plot_p3d_recovery(binned, P_input, *, units=UNITS_3D_COMOVING, **kw):
    """``plot_pk_recovery`` for the spherical binning ``OverfluxStatistics3D.bin_p3d_kmod``
    (|k| needs one length unit on all axes)."""
    return plot_pk_recovery(binned, P_input, k_label=_lbl("|k|", units["k_perp"]),
                            P_label=_lbl(r"P_{\rm 3D}(|k|)", units["P"]), **kw)


def plot_xi3d_recovery(stats, xi_input, *, xi=None, sigma=None, max_lag=None, units=UNITS_3D,
                       ax=None, figsize=(26, 11)):
    """xi3D against the input along the three axes through r = 0.

    Left: transverse cuts xi(r_x, 0, 0) and xi(0, r_y, 0); right: LoS cut xi(0, 0, pi);
    lower panels: pull (xi - xi_in) / sigma (or the difference if ``sigma`` is None).

    Parameters
    ----------
    stats : OverfluxStatistics3D (gives the centred lag axes r_x, r_y, pi_lags).
    xi_input : (Nx, Ny, N_pix) array, centred, e.g.
        ``two_point.xi3d_from_power(grf.power(k_mod), grf.shape, grf.cell_volume)``.
    xi, sigma : (Nx, Ny, N_pix) arrays or None
        Measured xi (default ``stats.xi3d_total``) and its error, e.g. the mean and standard
        error over realisations from ``sample_mean_and_error``.
    max_lag : float or None
        Show only |r| <= max_lag (units of each axis).

    Returns fig, axes (transverse, LoS, transverse pull, LoS pull).
    """
    stats.compute_xi3d()
    xi = np.asarray(stats.xi3d_total if xi is None else xi)
    xi_in = np.asarray(xi_input)
    sg = None if sigma is None else np.asarray(sigma)
    cx, cy, cz = (n // 2 for n in xi.shape)                  # r = 0 of the centred lag grid
    cuts = [(np.asarray(stats.r_x), np.s_[:, cy, cz], r"$r_x$", 0),
            (np.asarray(stats.r_y), np.s_[cx, :, cz], r"$r_y$", 0),
            (np.asarray(stats.pi_lags), np.s_[cx, cy, :], r"$\pi$", 1)]
    fig, axs = _figure(ax, 2, 2, figsize, sharex="col", gridspec_kw={"height_ratios": [3, 1.2]})
    for r, sl, lab, col in cuts:
        _xi_recovery_panels(axs[col], axs[2 + col], r, xi[sl], xi_in[sl],
                            None if sg is None else sg[sl], max_lag, f"along {lab}", None)
    axs[0].set(ylabel=r"$\xi_{\rm 3D}$", title="transverse")
    axs[1].set(title="line of sight")
    axs[2].set(xlabel=_lbl(r"r_\perp", units["x_perp"]))
    axs[3].set(xlabel=_lbl(r"\pi", units["x_par"]))
    for a in axs[:2]:
        a.plot([], [], color="k", lw=1.5, label="input")
        a.legend(frameon=False, fontsize="small")
    return fig, axs
