"""
radar_cylinder_field.py
=======================

Drop-in superset of ``plot_3d_radar_cylinder`` that ADDS, on top of the existing
neon radar profiles, the actual scalar field *surrounding* the skewer -- sampled
in a cylinder around the line of sight and drawn as a glowing, alpha-weighted 3D
point cloud (a matplotlib "nebula") so it composites with the radar profiles in
the same Axes3D.

Two public objects:

    sample_cylinder_cloud(...)   -> samples fields.number_density in a cylinder
                                    around one skewer, returning a flat cloud
                                    {s, t, axis, value, R} in the local frame.
    plot_3d_radar_cylinder(...)  -> your original function, plus a `surrounding_*`
                                    layer that renders those clouds.

The other functions in your plotting module (ridge cube, samples plot) are
unchanged and can stay where they are. The private helpers below are copied
verbatim so this file is self-contained.
"""

import numpy as np
import gc
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import cm
from matplotlib.colors import Normalize
import os
import glob
from PIL import Image

from skewerjax.constants.physical_constants import phys


# ============================================================
# Shared helpers (copied from your module, unchanged)
# ============================================================
def _neon_colors(labels):
    palette = ["#00FFFF", "#FF00FF", "#39FF14", "#FFEA00", "#FF5F1F", "#9D00FF",
               "#00BFFF", "#FF1493", "#7FFF00", "#FF073A", "#00FF7F", "#FF3131"]
    return {lab: palette[i % len(palette)] for i, lab in enumerate(labels)}


def _dark_3d_style(ax, tick_fs=13, hide_ticks=False):
    ax.set_facecolor("black")
    ax.grid(False)
    ax.tick_params(axis="both", which="major", labelsize=tick_fs, colors="white",
                   length=0 if hide_ticks else 5, width=0 if hide_ticks else 1.0)
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


def _select_indices(points, skewer_index=0, n_samples_plot=1):
    n_skewers = np.asarray(points).shape[0]
    idx = np.arange(int(skewer_index), int(skewer_index) + max(1, int(n_samples_plot)))
    return np.clip(idx, 0, n_skewers - 1)


def _local_basis(path):
    los = np.asarray(path[-1] - path[0], dtype=float)
    los /= np.linalg.norm(los) + 1e-30
    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(los, ref)) > 0.95:
        ref = np.array([1.0, 0.0, 0.0])
    e1 = np.cross(los, ref); e1 /= np.linalg.norm(e1) + 1e-30
    e2 = np.cross(los, e1); e2 /= np.linalg.norm(e2) + 1e-30
    return e1, e2


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