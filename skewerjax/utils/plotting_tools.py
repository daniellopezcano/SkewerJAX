import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.lines import Line2D

from skewerjax.constants.physical_constants import phys


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


def plot_density_imshow_and_central_profile(
    *,
    points_central: np.ndarray,
    points_grid_imshow: np.ndarray,
    log10_number_density_imshow: np.ndarray,
    log10_number_density_central: np.ndarray,
    line_bundle,
    scale_factor_start: float,
    scale_factor_end: float,
    savepath_png: str | None = None,
    show: bool = False,
    extent_yz: tuple[float, float, float, float] | None = None,
    box_limits_yz: tuple[tuple[float, float], tuple[float, float]] | None = None,
    box_limits_los: tuple[float, float] | None = None,
    density_norm: mpl.colors.Normalize | None = None,
    skewer_cmap=None,
    skewer_norm: mpl.colors.Normalize | None = None,
    n_arrows: int = 4,
    save_pad_inches: float = 0.08,
):
    y_c = points_central[:, 1]
    z_c = points_central[:, 2]
    LoSx = y_c - y_c[0]

    if extent_yz is None:
        extent_yz = (
            float(points_grid_imshow[..., 1].min()),
            float(points_grid_imshow[..., 1].max()),
            float(points_grid_imshow[..., 2].min()),
            float(points_grid_imshow[..., 2].max()),
        )

    species_keys = list(line_bundle.species_keys)
    line_indices, line_objs = [], []
    for sp in species_keys:
        ln0 = line_bundle.lines_for_species(sp)[0]
        i0 = next(i for i, ln in enumerate(line_bundle.lines) if ln.id == ln0.id)
        line_indices.append(i0)
        line_objs.append(ln0)
    ncols = len(species_keys)

    if density_norm is None:
        vmin = float(np.min(log10_number_density_imshow))
        vmax = float(np.max(log10_number_density_imshow))
        density_norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    else:
        vmin = float(density_norm.vmin)
        vmax = float(density_norm.vmax)

    if skewer_cmap is None:
        skewer_cmap = mpl.cm.get_cmap("gist_rainbow")
    if skewer_norm is None:
        skewer_norm = mpl.colors.Normalize(vmin=0.0, vmax=max(1.0, float(len(y_c) - 1)))

    if log10_number_density_central.ndim == 1:
        _central = np.repeat(log10_number_density_central[None, :], len(line_bundle.lines), axis=0)
    else:
        _central = log10_number_density_central

    fig, axes = plt.subplots(
        2, ncols, figsize=(6.0 * ncols, 8.0),
        gridspec_kw=dict(height_ratios=[3.2, 1.0], hspace=0.08, wspace=0.10),
        sharex=True, sharey="row",
    )
    if ncols == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    idx = np.arange(len(y_c), dtype=float)

    for c, (sp, i_line) in enumerate(zip(species_keys, line_indices)):
        ax_img = axes[0, c]
        ax_prof = axes[1, c]

        ax_img.imshow(
            log10_number_density_imshow[i_line].T,
            origin="lower", extent=extent_yz, aspect="auto", norm=density_norm, cmap="cividis",
        )

        ax_img.scatter(y_c, z_c, c=idx, cmap=skewer_cmap, norm=skewer_norm, s=1, linewidths=1.0, zorder=5)

        # leftmost/rightmost along y_c
        i_left  = int(np.argmin(np.asarray(y_c)))
        i_right = int(np.argmax(np.asarray(y_c)))
        # endpoint colors (match scatter mapping)
        idx_arr = np.asarray(idx)
        if idx_arr.ndim == 0:
            c_left = c_right = skewer_cmap(skewer_norm(float(idx_arr)))
        else:
            c_left  = skewer_cmap(skewer_norm(float(idx_arr[i_left])))
            c_right = skewer_cmap(skewer_norm(float(idx_arr[i_right])))
        # endpoint coordinates
        xL, yL = float(np.asarray(y_c)[i_left]),  float(np.asarray(z_c)[i_left])
        xR, yR = float(np.asarray(y_c)[i_right]), float(np.asarray(z_c)[i_right])
        # labels with decimals
        txt_left  = rf"$a_\mathrm{{start}} = {float(scale_factor_start):.4f}$"
        txt_right = rf"$a_\mathrm{{end}} = {float(scale_factor_end):.4f}$"
        # centered above each point
        ax_img.annotate(
            txt_left, xy=(xL+6, yL), xycoords="data",
            xytext=(0, 8), textcoords="offset points",
            ha="center", va="bottom", color=c_left, fontsize=11,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=c_left, alpha=0.85),
            zorder=6,
        )
        ax_img.annotate(
            txt_right, xy=(xR-6, yR-3), xycoords="data",
            xytext=(0, 8), textcoords="offset points",
            ha="center", va="top", color=c_right, fontsize=11,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=c_right, alpha=0.85),
            zorder=6,
        )

        npts = len(y_c)
        centers = np.linspace(0.10, 0.90, n_arrows)
        for f in centers:
            k2 = int(f * (npts - 1))
            k1 = max(0, k2 - max(1, npts // 50))
            ax_img.annotate(
                "", xy=(y_c[k2], z_c[k2]), xytext=(y_c[k1], z_c[k1]),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=skewer_cmap(skewer_norm(idx[k2])),
                    lw=2.0, mutation_scale=18, shrinkA=0, shrinkB=0,
                ),
                zorder=6,
            )

        ax_img.set_title(f"{sp}", pad=10)
        if c == 0:
            ax_img.set_ylabel("$z$ [cMpc]")

        if box_limits_yz is not None:
            (ymin, ymax), (zmin, zmax) = box_limits_yz
            ax_img.set_xlim(ymin, ymax)
            ax_img.set_ylim(zmin, zmax)
        else:
            ax_img.set_xlim(extent_yz[0], extent_yz[1])
            ax_img.set_ylim(extent_yz[2], extent_yz[3])

        pts = np.column_stack([LoSx, _central[i_line]]).reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap=skewer_cmap, norm=skewer_norm)
        lc.set_array(idx[:-1])
        lc.set_linewidth(2.5)
        ax_prof.add_collection(lc)

        ax_prof.set_xlim(*(box_limits_los if box_limits_los is not None else (float(LoSx.min()), float(LoSx.max()))))
        ax_prof.set_ylim(vmin, vmax)  # <-- match the imshow normalization limits
        ax_prof.set_xlabel("$y$ [cMpc]")
        if c == 0:
            ax_prof.set_ylabel("$\log_{10} n \\,\\mathrm{[cm^{-3}]}$")

    fig.subplots_adjust(right=0.90)
    cax = fig.add_axes([0.92, 0.35, 0.015, 0.5])
    sm = mpl.cm.ScalarMappable(norm=density_norm, cmap=mpl.cm.get_cmap("cividis"))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(r"$\log_{10} n\ \mathrm{[cm^{-3}]}$")

    if savepath_png is not None:
        d = os.path.dirname(savepath_png)
        if d:
            os.makedirs(d, exist_ok=True)
        fig.canvas.draw()
        fig.savefig(
            savepath_png,
            dpi=200,
            bbox_inches="tight",
            pad_inches=save_pad_inches,
            bbox_extra_artists=[cax, cbar.ax, cbar.ax.yaxis.label],
        )

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, axes


def plot_flux_panels(
    *,
    wavelengths: np.ndarray,
    F_over_F0: np.ndarray,
    ic: int,
    jc: int,
    line_bundle,
    scale_factor_start: float,
    scale_factor_end: float,
    x_axis_key: str,
    min_x: float,
    max_x: float,
    lambda0_anchor: float | None = None,
    savepath_png: str | None = None,
    show: bool = False,
    skewer_cmap=None,
    skewer_norm: mpl.colors.Normalize | None = None,
    skewer_last_index: int | None = None,
    save_pad_inches: float = 0.08,
    ylim: tuple[float, float] = (-0.05, 1.05),
    ncols: int | None = None,
):

    linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1))]

    assert x_axis_key in ("wavelengths", "velocities", "log_lambda"), (
        "x_axis_key should be either 'wavelengths', 'velocities' or 'log_lambda'"
    )

    if x_axis_key == "wavelengths":
        x_label = r"$\lambda\ [\mathrm{\AA}]$"
        x_axis_array = np.asarray(wavelengths, dtype=float)
    elif x_axis_key == "velocities":
        if lambda0_anchor == 1:
            x_label = r"$c\, \ln\left(\lambda \left[\mathrm{\AA}\right]\right)\; \left[\mathrm{km\,/\,s}\right] $"
        else:
            x_label = r"$c\, \ln\left( \frac{\lambda}{\lambda_\dagger}\right)\; \left[\mathrm{km\,/\,s}\right]$"
        x_axis_array = phys.c*1e-3 * np.log(wavelengths / lambda0_anchor)

    F_np = np.asarray(F_over_F0, dtype=float)

    # Curve color selection
    if skewer_cmap is None:
        skewer_cmap = mpl.cm.get_cmap("gist_rainbow")
    if skewer_norm is None:
        skewer_norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
    curve_color = "k" if skewer_last_index is None else skewer_cmap(skewer_norm(float(skewer_last_index)))

    ymin, ymax = ylim

    lines = list(line_bundle.lines)
    n_lines = len(lines)
    if n_lines == 0:
        raise ValueError("line_bundle.lines is empty.")

    # Grid layout
    if ncols is None:
        ncols = len(lines)
    nrows = 1

    fig, axes = plt.subplots(nrows, ncols, figsize=(6.0 * ncols, 3.4 * nrows), sharex=True, sharey=True)
    axes = np.atleast_2d(axes)

    # Plot each line in its own panel
    for i_line, ln in enumerate(lines):
        r = i_line // ncols
        c = i_line % ncols
        ax = axes[r, c]

        ls = linestyles[i_line % len(linestyles)]
        F_line = F_np[i_line, ic, jc, :]
        ax.plot(x_axis_array, F_line, color=curve_color, lw=2.0, ls="-", alpha=0.95)

        lam0 = float(getattr(ln, "lambda0_A"))
        lam0_end = lam0 / scale_factor_end
        lam0_start = lam0 / scale_factor_start
        if x_axis_key == "wavelengths":
            x_end = lam0_end
            x_start = lam0_start
        elif x_axis_key == "velocities":
            x_end = phys.c*1e-3 * np.log(lam0_end / lambda0_anchor)
            x_start = phys.c*1e-3 * np.log(lam0_start / lambda0_anchor)
        ax.axvline(x_end, color="grey", linestyle=ls, lw=1.5, alpha=1.0)
        ax.axvline(x_start, color="grey", linestyle=ls, lw=1.5, alpha=1.0)

        # --- annotate a_i / a_f anchored to the vertical lines at y=0.5, rotated vertical ---
        c_ai = skewer_cmap(skewer_norm(0.0))
        c_af = curve_color
        txt_ai = rf"$a_\mathrm{{i}} = {float(scale_factor_start):.4f}$"
        txt_af = rf"$a_\mathrm{{f}} = {float(scale_factor_end):.4f}$"
        y_ax = 0.5
        ax.annotate(
            txt_ai, xy=(x_start, y_ax-0.3), xycoords=("data", "axes fraction"),
            xytext=(13, 0), textcoords="offset points", ha="left", va="center",
            rotation=90, rotation_mode="anchor", color=c_ai, fontsize=11, bbox=dict(boxstyle="round,pad=0.25",
            facecolor="white", edgecolor=c_ai, alpha=0.85), zorder=6,
        )
        ax.annotate(
            txt_af, xy=(x_end, y_ax+0.3), xycoords=("data", "axes fraction"),
            xytext=(-13, 0), textcoords="offset points", ha="right", va="center",
            rotation=90, rotation_mode="anchor", color=c_af, fontsize=11, bbox=dict(boxstyle="round,pad=0.25",
            facecolor="white", edgecolor=c_af, alpha=0.85), zorder=6,
        )

        name = getattr(ln, "name", None) or getattr(ln, "label", None) or f"Line {i_line}"
        ax.set_title(name, pad=8)

        ax.set_xlim(min_x, max_x)
        ax.set_ylim(ymin, ymax)

        ax.tick_params(which="both", direction="in", top=True, right=True, length=7, width=1.6)
        ax.tick_params(which="minor", length=4)
        for spine in ax.spines.values():
            spine.set_linewidth(1.8)

        # Axis labels only on outer edges to reduce clutter
        if r == nrows - 1:
            ax.set_xlabel(x_label)
        if c == 0:
            ax.set_ylabel(r"$e^{-\tau}$")

    # Hide unused panels if n_lines doesn't fill the grid
    for k in range(n_lines, nrows * ncols):
        r = k // ncols
        c = k % ncols
        axes[r, c].axis("off")

    fig.tight_layout()

    if savepath_png is not None:
        d = os.path.dirname(savepath_png)
        if d:
            os.makedirs(d, exist_ok=True)
        fig.canvas.draw()
        fig.savefig(savepath_png, dpi=200, bbox_inches="tight", pad_inches=save_pad_inches)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, axes


def plot_flux_panels_colored_by_los(
    *,
    wavelengths: np.ndarray,            # (N_lambda,)
    F_over_F0: np.ndarray,              # (N_lines, N_skx, N_sky, N_lambda)
    ic: int,
    jc: int,
    line_bundle,
    x_axis_key,
    scale_factor_start: float,
    scale_factor_end: float,
    min_x, max_x, skewer_last_index,
    lambda0_anchor: float | None = None,
    savepath_png: str | None = None,
    show: bool = False,
    skewer_cmap=None,
    skewer_norm: mpl.colors.Normalize | None = None,
    linestyles: list[str] | None = None,
    add_legend: bool = True,
    ylim: tuple[float, float] = (-0.05, 1.05),
):  
    
    assert x_axis_key in ("wavelengths", "velocities"), "x_axis_key should be either 'wavelengths', or 'velocities'"
    if x_axis_key == "wavelengths":
        x_label = r"$\lambda\ [\mathrm{\AA}]$"
        x_axis_array = np.asarray(wavelengths, dtype=float)
    elif x_axis_key == "velocities":
        if lambda0_anchor == 1:
            x_label = r"$c\, \ln\left(\lambda \left[\mathrm{\AA}\right]\right)\; \left[\mathrm{km\,/\,s}\right] $"
        else:
            x_label = r"$c\, \ln\left( \frac{\lambda}{\lambda_\dagger}\right)\; \left[\mathrm{km\,/\,s}\right]$"
        x_axis_array = phys.c*1e-3 * np.log(wavelengths / lambda0_anchor)

    F_np = np.asarray(F_over_F0, dtype=float)

    if skewer_cmap is None:
        skewer_cmap = mpl.cm.get_cmap("gist_rainbow")
    if linestyles is None:
        linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1))]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    legend_handles, legend_labels = [], []
    for i_line, line in enumerate(line_bundle.lines):
        ls = linestyles[i_line % len(linestyles)]

        lam0 = float(line.lambda0_A)
        lam0_end = lam0 / scale_factor_end
        lam0_start = lam0 / scale_factor_start
        if x_axis_key == "wavelengths":
            x_end = lam0_end
            x_start = lam0_start
        elif x_axis_key == "velocities":
            x_end = phys.c*1e-3 * np.log(lam0_end / lambda0_anchor)
            x_start = phys.c*1e-3 * np.log(lam0_start / lambda0_anchor)
        ax.axvline(x_end, color="grey", linestyle=linestyles[i_line], lw=1.5, alpha=1.0)
        ax.axvline(x_start, color="grey", linestyle=linestyles[i_line], lw=1.5, alpha=1.0)

        # --- annotate a_i / a_f anchored to the vertical lines at y=0.5, rotated vertical ---
        c_ai = skewer_cmap(skewer_norm(0.0))
        c_af = skewer_cmap(skewer_norm(float(skewer_last_index)))
        txt_ai = rf"$a_\mathrm{{i}} = {float(scale_factor_start):.4f}$"
        txt_af = rf"$a_\mathrm{{f}} = {float(scale_factor_end):.4f}$"
        y_ax = 0.5
        ax.annotate(
            txt_ai, xy=(x_start, y_ax-0.3), xycoords=("data", "axes fraction"),
            xytext=(13, 0), textcoords="offset points", ha="left", va="center",
            rotation=90, rotation_mode="anchor", color=c_ai, fontsize=11, bbox=dict(boxstyle="round,pad=0.25",
            facecolor="white", edgecolor=c_ai, alpha=0.85), zorder=6,
        )
        ax.annotate(
            txt_af, xy=(x_end, y_ax+0.3), xycoords=("data", "axes fraction"),
            xytext=(-13, 0), textcoords="offset points", ha="right", va="center",
            rotation=90, rotation_mode="anchor", color=c_af, fontsize=11, bbox=dict(boxstyle="round,pad=0.25",
            facecolor="white", edgecolor=c_af, alpha=0.85), zorder=6,
        )

        F_line = F_np[i_line, ic, jc, :]
        pts  = np.column_stack([x_axis_array, F_line]).reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)

        x_mid = 0.5 * (x_axis_array[:-1] + x_axis_array[1:])

        xmin, xmax = (min(x_end, x_start), max(x_end, x_start))
        norm_x = Normalize(vmin=xmin, vmax=xmax, clip=True)

        # --- build inverted colormap, still ending at the same "max color" defined by skewer_last_index ---
        if skewer_last_index is None:
            cmap_use = skewer_cmap.reversed()
        else:
            f_max = float(skewer_norm(float(skewer_last_index)))
            f_max = float(np.clip(f_max, 1e-6, 1.0))

            # truncate then invert
            cmap_trunc = LinearSegmentedColormap.from_list(
                "skewer_cmap_trunc",
                skewer_cmap(np.linspace(0.0, f_max, 256))
            )
            cmap_use = cmap_trunc.reversed()

        lc = LineCollection(segs, cmap=cmap_use, norm=norm_x, linestyle='-', alpha=0.7)
        lc.set_array(x_mid)
        lc.set_linewidth(1.5)
        ax.add_collection(lc)

        if add_legend:
            name = getattr(line, "name", None) or getattr(line, "label", None) or f"Line {i_line}"
            legend_handles.append(Line2D([0], [0], color="k", lw=2.2, linestyle=ls))
            legend_labels.append(name)

    ax.set_xlim((min_x, max_x))
    ax.set_ylim(*ylim)

    ax.set_xlabel(x_label, fontsize=20)
    ax.set_ylabel(r"$e^{-\tau}$", fontsize=20)

    ax.tick_params(which="both", direction="in", top=True, right=True, length=7, width=1.6, labelsize=18)
    ax.tick_params(which="minor", length=4)
    for spine in ax.spines.values():
        spine.set_linewidth(1.8)

    if add_legend and legend_handles:
        ax.legend(
            legend_handles, legend_labels, loc="lower left", fontsize=16, handlelength=2.6, handletextpad=0.6, borderaxespad=0.6, ncols=1
        )

    fig.tight_layout()

    if savepath_png is not None:
        d = os.path.dirname(savepath_png)
        if d:
            os.makedirs(d, exist_ok=True)
        fig.savefig(savepath_png, dpi=200, bbox_inches="tight")

    if show:
        plt.tight_layout()
        plt.show()
    else:
        plt.close(fig)

    return fig, ax







import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ============================================================
# Shared helpers
# ============================================================

def _neon_colors(labels):
    palette = ["#00FFFF", "#FF00FF", "#39FF14", "#FFEA00", "#FF5F1F", "#9D00FF", "#00BFFF", "#FF1493", "#7FFF00", "#FF073A", "#00FF7F", "#FF3131"]
    return {lab: palette[i % len(palette)] for i, lab in enumerate(labels)}


def _dark_3d_style(ax, tick_fs=13, hide_ticks=False):
    ax.set_facecolor("black")
    ax.grid(False)
    ax.tick_params(axis="both", which="major", labelsize=tick_fs, colors="white", length=0 if hide_ticks else 5, width=0 if hide_ticks else 1.0)
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
        ((x0, y0, z0), (x1, y0, z0)), ((x0, y1, z0), (x1, y1, z0)), ((x0, y0, z1), (x1, y0, z1)), ((x0, y1, z1), (x1, y1, z1)),
        ((x0, y0, z0), (x0, y1, z0)), ((x1, y0, z0), (x1, y1, z0)), ((x0, y0, z1), (x0, y1, z1)), ((x1, y0, z1), (x1, y1, z1)),
        ((x0, y0, z0), (x0, y0, z1)), ((x1, y0, z0), (x1, y0, z1)), ((x0, y1, z0), (x0, y1, z1)), ((x1, y1, z0), (x1, y1, z1)),
    ]


def _line_labels_and_species(line_bundle, n_lines):
    if line_bundle is None or not hasattr(line_bundle, "lines"):
        labels = [f"line {i}" for i in range(n_lines)]
        return labels, labels

    species = [getattr(line, "species", f"line {i}") for i, line in enumerate(line_bundle.lines)]
    if len(species) != n_lines:
        labels = [f"line {i}" for i in range(n_lines)]
        return labels, labels

    counts, seen, labels = {s: species.count(s) for s in set(species)}, {}, []
    for s in species:
        seen[s] = seen.get(s, 0) + 1
        labels.append(f"{s} {seen[s]}" if counts[s] > 1 else s)

    return labels, species


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
    old_t, new_t = np.linspace(0, 1, path.shape[0]), np.linspace(0, 1, n_target)
    return np.column_stack([np.interp(new_t, old_t, path[:, i]) for i in range(3)])


def _local_basis(path):
    los = np.asarray(path[-1] - path[0], dtype=float)
    los /= np.linalg.norm(los) + 1e-30
    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(los, ref)) > 0.95:
        ref = np.array([1.0, 0.0, 0.0])
    e1 = np.cross(los, ref)
    e1 /= np.linalg.norm(e1) + 1e-30
    e2 = np.cross(los, e1)
    e2 /= np.linalg.norm(e2) + 1e-30
    return e1, e2


def _select_indices(points, skewer_index=0, n_samples_plot=1):
    n_skewers = np.asarray(points).shape[0]
    idx = np.arange(int(skewer_index), int(skewer_index) + max(1, int(n_samples_plot)))
    return np.clip(idx, 0, n_skewers - 1)


def _grid_pairs_from_flat(flat, Nx, Nz):
    flat = np.clip(np.asarray(flat, dtype=int), 0, Nx * Nz - 1)
    return [(int(i // Nz), int(i % Nz)) for i in flat]


def _profile_setup(points, sampled_fields=None, field="number_density", mode="sampled_fields", optical_depth_grid=None, line_bundle=None, selected=None, normalization="global", field_logscale=True, eps=1e-300, optical_depth_eps=1e-30):
    selected = np.asarray(selected, dtype=int)

    if mode == "sampled_fields":
        labels = list(sampled_fields.keys())
        profiles = np.asarray([[np.asarray(sampled_fields[s][field][idx]) for s in labels] for idx in selected])
        range_values = profiles if normalization == "skewer" else np.concatenate([np.asarray(sampled_fields[s][field]).ravel() for s in labels])
        vmin, vmax, vrange = _scaled_range(range_values, eps=eps, logscale=field_logscale)
        return dict(labels=labels, color_labels=labels, profiles=profiles, selected=selected, vmin=vmin, vmax=vmax, vrange=vrange, eps=eps, positive_only=False)

    if mode == "optical_depth":
        tau = np.asarray(optical_depth_grid)
        if tau.ndim != 4:
            raise ValueError("optical_depth_grid must have shape (n_lines, Nx, Nz, N_lambda).")
        n_lines, Nx, Nz, _ = tau.shape
        grid_pairs = _grid_pairs_from_flat(selected, Nx, Nz)
        labels, species = _line_labels_and_species(line_bundle, n_lines)
        profiles = np.asarray([tau[:, ix, iz, :] for ix, iz in grid_pairs])
        range_values = profiles if normalization == "skewer" else tau
        vmin, vmax, vrange = _scaled_range(range_values, eps=optical_depth_eps, logscale=field_logscale, positive_only=True)
        return dict(labels=labels, color_labels=species, profiles=profiles, selected=selected, vmin=vmin, vmax=vmax, vrange=vrange, eps=optical_depth_eps, positive_only=True)

    raise ValueError("mode must be 'sampled_fields' or 'optical_depth'.")


# ============================================================
# radar-cylinder plot
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
    show_profile_points=False,
    profile_point_size=10.0,
    profile_linewidth=1.05,
    glow=True,
    show=True,
):
    """
    Radar-cylinder plot for one or more 1D skewers.

    This version always uses all points in the input arrays. It performs no
    downsampling, so narrow spikes in sampled fields are preserved.

    Parameters
    ----------
    x_array : array, optional, shape (N_samples, N_x)
        Projected LoS coordinate used as the cylinder-axis coordinate.

    points : array, optional, shape (N_samples, N_x, N_dim)
        Alternative input. If provided and x_array is None, the LoS coordinate
        is extracted as points[..., los_axis].

    y_dict : dict[str, array]
        Maps labels to arrays with shape (N_samples, N_x).

    los_axis : int
        Coordinate axis used from points when x_array is not provided.

    show_profile_points : bool
        If True, overlay scatter points on top of the profile curves.

    Returns
    -------
    fig, ax
    """
    if x_array is None:
        if points is None:
            raise ValueError("Pass either `x_array` or `points`.")
        points = np.asarray(points)
        if points.ndim != 3:
            raise ValueError(
                "`points` must have shape (N_samples, N_x, N_dim). "
                f"Got {points.shape}."
            )
        if los_axis < 0 or los_axis >= points.shape[-1]:
            raise ValueError(
                f"`los_axis={los_axis}` is invalid for points with "
                f"N_dim={points.shape[-1]}."
            )
        x_array = points[:, :, los_axis]

    x_array = np.asarray(x_array)

    if x_array.ndim != 2:
        raise ValueError(
            "Expected x_array with shape (N_samples, N_x). "
            f"Got {x_array.shape}."
        )

    if not isinstance(y_dict, dict) or len(y_dict) == 0:
        raise ValueError("Expected non-empty y_dict mapping labels to arrays.")

    labels = list(y_dict.keys())
    y_arrays = {lab: np.asarray(y_dict[lab]) for lab in labels}

    for lab, arr in y_arrays.items():
        if arr.shape != x_array.shape:
            raise ValueError(
                f"y_dict[{lab!r}] has shape {arr.shape}, but expected "
                f"{x_array.shape} to match x_array."
            )

    selected = _select_indices(x_array, skewer_index, n_samples_plot)

    profiles = np.asarray(
        [[y_arrays[lab][idx] for lab in labels] for idx in selected],
        dtype=float,
    )

    if normalization == "skewer":
        range_values = profiles
    elif normalization == "global":
        range_values = np.concatenate([y_arrays[lab].ravel() for lab in labels])
    else:
        raise ValueError("normalization must be either 'global' or 'skewer'.")

    vmin, vmax, vrange = _scaled_range(
        range_values,
        eps=1e-300,
        logscale=field_logscale,
        positive_only=False,
    )

    eps = 1e-300
    positive_only = False

    n_samples = profiles.shape[0]
    n_profiles = profiles.shape[1]

    colors = _neon_colors(labels)
    profile_color = {lab: colors[lab] for lab in labels}

    angles = {
        lab: np.deg2rad(profile_angle_offset_deg) + 2.0 * np.pi * i / n_profiles
        for i, lab in enumerate(labels)
    }

    fig = plt.figure(
        figsize=(12.2 + 1.85 * max(n_samples - 1, 0), 7.0),
        facecolor="black",
    )
    ax = fig.add_subplot(111, projection="3d", facecolor="black")
    ax.set_position((0.01, 0.02, 0.86, 0.90))

    centers = np.arange(n_samples, dtype=float) * cylinder_separation
    centers -= centers.mean()

    phi = np.linspace(0.0, 2.0 * np.pi, 360)
    rings = cylinder_radius * np.linspace(1.0 / 6.0, 1.0, 6)

    x_selected = x_array[selected]
    x_min = float(np.nanmin(x_selected))
    x_max = float(np.nanmax(x_selected))
    x_span = x_max - x_min

    if x_span <= 0:
        x_span = 1.0
        x_min -= 0.5
        x_max += 0.5

    axis_z = -cylinder_radius * (1.0 + x_axis_pad)
    tick_dx = cylinder_radius * x_axis_tick_size

    for i_sample, x0 in enumerate(centers):
        x_los = np.asarray(x_array[selected[i_sample]], dtype=float)

        y_slices = np.linspace(
            float(np.nanmin(x_los)),
            float(np.nanmax(x_los)),
            int(n_y_slices),
        )

        ax.plot(
            np.full_like(y_slices, x0),
            y_slices,
            np.zeros_like(y_slices),
            color="white",
            lw=1.6,
            alpha=0.9,
        )

        for yy in y_slices:
            for rr in rings:
                ax.plot(
                    x0 + rr * np.cos(phi),
                    np.full_like(phi, yy),
                    rr * np.sin(phi),
                    color="white",
                    lw=0.85,
                    alpha=0.18,
                )

        tick_vals = np.linspace(
            float(np.nanmin(x_los)),
            float(np.nanmax(x_los)),
            int(n_x_ticks),
        )

        ax.plot(
            np.full_like(tick_vals, x0),
            tick_vals,
            np.full_like(tick_vals, axis_z),
            color="white",
            lw=1.2,
            alpha=0.75,
        )

        for tt in tick_vals:
            ax.plot(
                [x0 - tick_dx, x0 + tick_dx],
                [tt, tt],
                [axis_z, axis_z],
                color="white",
                lw=1.0,
                alpha=0.75,
            )

            ax.text(
                x0 - factor_ticks_x * tick_dx,
                tt,
                axis_z - factor_ticks_z,
                f"{tt:.2f}",
                color="white",
                fontsize=x_axis_fontsize,
                ha="left",
                va="center",
            )

        if x_axis_label is not None:
            ax.text(
                x0,
                float(np.nanmean(tick_vals)),
                axis_z - factor_x_label * cylinder_radius,
                x_axis_label,
                color="white",
                fontsize=x_axis_fontsize + 2,
                ha="center",
                va="top",
            )

        for i_profile, lab in enumerate(labels):
            th = angles[lab]
            c = profile_color[lab]

            r = cylinder_radius * _scale_values(
                profiles[i_sample, i_profile],
                vmin,
                vrange,
                eps,
                field_logscale,
                positive_only,
            )

            ax.plot(
                x0 + cylinder_radius * np.cos(th) * np.ones_like(y_slices),
                y_slices,
                cylinder_radius * np.sin(th) * np.ones_like(y_slices),
                color=c,
                lw=0.9,
                alpha=0.18,
            )

            if glow:
                ax.plot(
                    x0 + r * np.cos(th),
                    x_los,
                    r * np.sin(th),
                    color=c,
                    lw=8.5,
                    alpha=0.10,
                )
                ax.plot(
                    x0 + r * np.cos(th),
                    x_los,
                    r * np.sin(th),
                    color=c,
                    lw=4.2,
                    alpha=0.30,
                )

            ax.plot(
                x0 + r * np.cos(th),
                x_los,
                r * np.sin(th),
                color=c,
                lw=profile_linewidth,
                alpha=1.00,
            )

            if show_profile_points:
                ax.scatter(
                    x0 + r * np.cos(th),
                    x_los,
                    r * np.sin(th),
                    color=c,
                    s=profile_point_size,
                    alpha=1.0,
                    depthshade=False,
                )

    margin = 1.22 * cylinder_radius
    z_min = min(-margin, axis_z - 0.35 * cylinder_radius)

    ax.set_xlim(np.min(centers) - margin, np.max(centers) + margin)
    ax.set_ylim(x_min - 0.05 * x_span, x_max + 0.05 * x_span)
    ax.set_zlim(z_min, margin)

    ax.set_box_aspect(
        (
            np.ptp(ax.get_xlim()),
            np.ptp(ax.get_ylim()),
            np.ptp(ax.get_zlim()),
        )
    )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    _dark_3d_style(ax, hide_ticks=True)
    ax.view_init(elev=view_elev, azim=view_azim)

    sax = fig.add_axes(
        (0.815 * scale_box_x_factor, 0.335, 0.125, 0.37),
        facecolor="black",
    )

    sax.set_xlim(0.0, 1.0)
    sax.set_ylim(0.0, 1.0)
    sax.set_xticks([])
    sax.set_yticks([])

    for spine in sax.spines.values():
        spine.set_color("white")
        spine.set_linewidth(1.1)

    sax.plot([0.28, 0.28], [0.20, 0.80], color="white", lw=2.0)

    for rr, vv in zip(np.linspace(0.0, 1.0, 5), np.linspace(vmin, vmax, 5)):
        yy = 0.20 + 0.60 * rr
        sax.plot([0.20, 0.36], [yy, yy], color="white", lw=1.5)
        sax.text(
            0.88,
            yy,
            f"{vv:.2f}",
            color="white",
            fontsize=17,
            ha="right",
            va="center",
        )

    if field_units is not None:
        sax.text(
            scale_units_xy[0],
            scale_units_xy[1],
            field_units,
            color="white",
            fontsize=22,
            ha="center",
            va="bottom",
            transform=sax.transAxes,
        )

    handles = [
        Line2D([0], [0], color=profile_color[lab], lw=5, label=lab)
        for lab in labels
    ]

    leg = fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.44, 0.965 * legend_y_factor),
        ncols=min(len(labels), 6),
        frameon=True,
        facecolor="black",
        edgecolor="white",
        framealpha=1.0,
        fontsize=15,
    )

    for text in leg.get_texts():
        text.set_color("white")

    if show:
        plt.show()

    return fig, ax

# ============================================================
# 3D ridge cube
# ============================================================

def _adaptive_tick_labels(vals, max_decimals=3):
    """
    Compact tick formatter for raw-coordinate labels.
    Keeps large values readable and avoids excessive trailing decimals.
    """
    vals = np.asarray(vals, dtype=float)

    if vals.size == 0:
        return []

    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return ["" for _ in vals]

    if finite.size == 1:
        vmax = np.abs(finite[0])
        if vmax >= 100:
            return [f"{finite[0]:.0f}"]
        if vmax >= 10:
            return [f"{finite[0]:.1f}"]
        return [f"{finite[0]:.2f}"]

    diffs = np.diff(finite)
    step = np.nanmedian(np.abs(diffs)) if diffs.size else 1.0
    vmax = np.nanmax(np.abs(finite))

    if vmax >= 100:
        fmt = "{:.0f}"
    elif vmax >= 10:
        fmt = "{:.1f}"
    elif step >= 1:
        fmt = "{:.1f}"
    elif step >= 0.1:
        fmt = "{:.2f}"
    else:
        fmt = "{:." + str(max_decimals) + "f}"

    return [fmt.format(v) for v in vals]


def _auto_axis_padding(
    tick_labels,
    axis_tick_fontsize,
    axis_label_fontsize,
    tick_pad=None,
    label_pad=None,
    tick_pad_factor=0.45,
    label_pad_factor=1.35,
):
    """
    Choose tick/label padding from label length and font sizes.
    User-provided tick_pad / label_pad override the automatic values.
    """
    max_chars = max((len(str(t)) for t in tick_labels), default=4)

    if tick_pad is None:
        tick_pad = tick_pad_factor * axis_tick_fontsize + 0.35 * max_chars

    if label_pad is None:
        label_pad = (
            tick_pad
            + label_pad_factor * axis_tick_fontsize
            + 0.45 * axis_label_fontsize
            + 0.45 * max_chars
        )

    return float(tick_pad), float(label_pad)


def _as_axis_tuple(value, name):
    """
    Allow either scalar or 3-tuple axis configuration.
    """
    if np.isscalar(value):
        return (value, value, value)

    if len(value) != 3:
        raise ValueError(f"{name} must be a scalar or a length-3 tuple.")

    return tuple(value)


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
    show=True,
):
    """
    3D neon ridge cube for one or more skewers.

    Parameters
    ----------
    points : array, shape (N_samples, N_x, 3)
        3D ridgeline coordinates.

        points[..., 0] : box x coordinate
        points[..., 1] : LoS / ridge coordinate, possibly in different units
        points[..., 2] : box z coordinate

    y_dict : dict[str, array]
        Maps species/line labels to arrays with shape (N_samples, N_x).

    normalize_scales : bool
        If True, display the geometry in normalized coordinates while keeping
        tick labels in the original raw coordinate units.
    """
    points = np.asarray(points)

    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError("Expected points with shape (N_samples, N_x, 3).")

    if not isinstance(y_dict, dict) or len(y_dict) == 0:
        raise ValueError("Expected non-empty y_dict mapping labels to arrays.")

    labels = list(y_dict.keys())
    y_arrays = {lab: np.asarray(y_dict[lab]) for lab in labels}

    expected_shape = points.shape[:2]
    for lab, arr in y_arrays.items():
        if arr.shape != expected_shape:
            raise ValueError(
                f"y_dict[{lab!r}] has shape {arr.shape}, "
                f"but expected {expected_shape} to match points.shape[:2]."
            )

    selected = int(np.clip(skewer_index, 0, points.shape[0] - 1))

    if normalization == "skewer":
        range_values = np.asarray([y_arrays[lab][selected] for lab in labels])
    elif normalization == "global":
        range_values = np.concatenate([y_arrays[lab].ravel() for lab in labels])
    else:
        raise ValueError("normalization must be either 'global' or 'skewer'.")

    vmin, vmax, vrange = _scaled_range(
        range_values,
        eps=1e-300,
        logscale=field_logscale,
        positive_only=False,
    )
    eps, positive_only = 1e-300, False

    colors = _neon_colors(labels)
    profile_color = {lab: colors[lab] for lab in labels}

    angles = {
        lab: np.deg2rad(profile_angle_offset_deg) + 2 * np.pi * i / len(labels)
        for i, lab in enumerate(labels)
    }

    p_raw = points.astype(float, copy=False)

    mins_raw = np.nanmin(p_raw.reshape(-1, 3), axis=0)
    maxs_raw = np.nanmax(p_raw.reshape(-1, 3), axis=0)
    centers_raw = 0.5 * (mins_raw + maxs_raw)
    spans_raw = maxs_raw - mins_raw
    safe_spans = np.where(spans_raw > 0, spans_raw, 1.0)

    n_axis_ticks_xyz = _as_axis_tuple(n_axis_ticks, "n_axis_ticks")
    nx_ticks, ny_ticks, nz_ticks = [int(v) for v in n_axis_ticks_xyz]

    x_ticks_raw = np.linspace(mins_raw[0], maxs_raw[0], nx_ticks)
    y_ticks_raw = np.linspace(mins_raw[1], maxs_raw[1], ny_ticks)
    z_ticks_raw = np.linspace(mins_raw[2], maxs_raw[2], nz_ticks)

    if normalize_scales:
        xz_extent = max(safe_spans[0], safe_spans[2], 1.0)

        def raw_to_plot_x(x):
            return (np.asarray(x) - centers_raw[0]) / xz_extent

        def raw_to_plot_y(y):
            return (np.asarray(y) - centers_raw[1]) / safe_spans[1]

        def raw_to_plot_z(z):
            return (np.asarray(z) - centers_raw[2]) / xz_extent

        p_plot = np.empty_like(p_raw, dtype=float)
        p_plot[..., 0] = raw_to_plot_x(p_raw[..., 0])
        p_plot[..., 1] = raw_to_plot_y(p_raw[..., 1])
        p_plot[..., 2] = raw_to_plot_z(p_raw[..., 2])

        x0, x1 = -0.5, 0.5
        y0, y1 = -0.5, 0.5
        z0, z1 = -0.5, 0.5
        extent = 1.0

        x_ticks_plot = raw_to_plot_x(x_ticks_raw)
        y_ticks_plot = raw_to_plot_y(y_ticks_raw)
        z_ticks_plot = raw_to_plot_z(z_ticks_raw)

    else:
        p_plot = p_raw.copy()

        x0, y0, z0 = mins_raw
        x1, y1, z1 = maxs_raw

        if x1 <= x0:
            x0, x1 = x0 - 0.5, x1 + 0.5
        if y1 <= y0:
            y0, y1 = y0 - 0.5, y1 + 0.5
        if z1 <= z0:
            z0, z1 = z0 - 0.5, z1 + 0.5

        extent = max(x1 - x0, y1 - y0, z1 - z0, 1.0)

        x_ticks_plot = x_ticks_raw
        y_ticks_plot = y_ticks_raw
        z_ticks_plot = z_ticks_raw

    field_scale = 0.06 * ridge_scale * extent

    fig = plt.figure(figsize=(10.2, 8.2), facecolor="black")
    ax = fig.add_subplot(111, projection="3d", facecolor="black")
    ax.set_position((0.01, 0.01, 0.98, 0.98))

    for a, b in _box_edges(x0, x1, y0, y1, z0, z1):
        ax.plot(
            [a[0], b[0]],
            [a[1], b[1]],
            [a[2], b[2]],
            color="white",
            lw=2.2,
        )

    for sk_idx, path in enumerate(p_plot):
        e1, e2 = _local_basis(path)

        for lab in labels:
            values = y_arrays[lab][sk_idx]
            p = _resample_path(path, values.size)

            amp = field_scale * _scale_values(
                values,
                vmin,
                vrange,
                eps,
                field_logscale,
                positive_only,
            )

            direction = (
                np.cos(angles[lab]) * e1
                + np.sin(angles[lab]) * e2
            )

            ridge = p + amp[:, None] * direction[None, :]
            c = profile_color[lab]

            ax.plot(
                ridge[:, 0],
                ridge[:, 1],
                ridge[:, 2],
                color=c,
                lw=8.0,
                alpha=0.12,
            )
            ax.plot(
                ridge[:, 0],
                ridge[:, 1],
                ridge[:, 2],
                color=c,
                lw=4.0,
                alpha=0.35,
            )
            ax.plot(
                ridge[:, 0],
                ridge[:, 1],
                ridge[:, 2],
                color=c,
                lw=1.0,
                alpha=1.0,
            )

    xm, ym, zm = 0.5 * (x0 + x1), 0.5 * (y0 + y1), 0.5 * (z0 + z1)
    half = 0.5 * extent + field_scale

    ax.set_xlim(xm - half, xm + half)
    ax.set_ylim(ym - half, ym + half)
    ax.set_zlim(zm - half, zm + half)
    ax.set_box_aspect((1, 1, 1))

    x_tick_labels = _adaptive_tick_labels(
        x_ticks_raw,
        max_decimals=tick_label_max_decimals,
    )
    y_tick_labels = _adaptive_tick_labels(
        y_ticks_raw,
        max_decimals=tick_label_max_decimals,
    )
    z_tick_labels = _adaptive_tick_labels(
        z_ticks_raw,
        max_decimals=tick_label_max_decimals,
    )

    ax.set_xticks(x_ticks_plot)
    ax.set_yticks(y_ticks_plot)
    ax.set_zticks(z_ticks_plot)

    ax.set_xticklabels(x_tick_labels)
    ax.set_yticklabels(y_tick_labels)
    ax.set_zticklabels(z_tick_labels)

    _dark_3d_style(ax)

    if axis_tick_fontsize is None:
        axis_tick_fontsize = max(axis_label_fontsize - 5, 8)

    x_tick_pad, x_label_pad = _auto_axis_padding(
        x_tick_labels,
        axis_tick_fontsize,
        axis_label_fontsize,
        tick_pad=x_tick_pad,
        label_pad=x_label_pad,
    )
    y_tick_pad, y_label_pad = _auto_axis_padding(
        y_tick_labels,
        axis_tick_fontsize,
        axis_label_fontsize,
        tick_pad=y_tick_pad,
        label_pad=y_label_pad,
    )
    z_tick_pad, z_label_pad = _auto_axis_padding(
        z_tick_labels,
        axis_tick_fontsize,
        axis_label_fontsize,
        tick_pad=z_tick_pad,
        label_pad=z_label_pad,
    )

    ax.tick_params(
        axis="x",
        colors="white",
        labelsize=axis_tick_fontsize,
        pad=x_tick_pad,
    )
    ax.tick_params(
        axis="y",
        colors="white",
        labelsize=axis_tick_fontsize,
        pad=y_tick_pad,
    )
    ax.tick_params(
        axis="z",
        colors="white",
        labelsize=axis_tick_fontsize,
        pad=z_tick_pad,
    )

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


# ============================================================
# Transmission plot
# ============================================================

def plot_transmission_samples_neon(
    F_over_F0,
    line_bundle,
    x_axis_array,
    x_label,
    N_plot_sample=4,
    seed=0,
    sample_indices=None,
    line_lw=2.2,
    line_glow_lw=6.0,
    line_alpha=0.95,
    line_glow_alpha=0.12,
    line_legend_lw=None,
    total_color="white",
    total_lw=1.05,
    total_glow_lw=None,
    total_alpha=1.0,
    total_glow_alpha=0.12,
):
    """
    Neon-style transmission plot for selected skewers.

    Notes
    -----
    - No vertical line markers are drawn.
    - Colors follow the same convention as the other plots:
      one neon color per species.
    - Linestyles are only used to distinguish multiple absorption lines
      from the same species.
    - The total transmitted flux is plotted last, on top of the individual
      line transmissions.

    Parameters
    ----------
    line_lw : float
        Main linewidth for individual absorption-line transmissions.

    line_glow_lw : float
        Glow linewidth for individual absorption-line transmissions.

    line_alpha : float
        Alpha for the main individual absorption-line curves.

    line_glow_alpha : float
        Alpha for the glow individual absorption-line curves.

    line_legend_lw : float or None
        Linewidth used for individual absorption-line legend handles.
        If None, defaults to line_lw.

    total_lw : float
        Main linewidth for total transmitted flux.

    total_glow_lw : float or None
        Glow linewidth for total transmitted flux.
        If None, defaults to 3 * total_lw.
    """
    F_over_F0 = np.asarray(F_over_F0)
    x_axis_array = np.asarray(x_axis_array)

    if F_over_F0.ndim != 3:
        raise ValueError("Expected F_over_F0 with shape (N_lines, N_skewers, N_x).")

    N_lines, N_skewers, N_x = F_over_F0.shape

    if x_axis_array.shape[-1] != N_x:
        raise ValueError(
            f"x_axis_array has length {x_axis_array.shape[-1]}, "
            f"but F_over_F0 has N_x={N_x}."
        )

    if sample_indices is None:
        rng = np.random.default_rng(seed)
        idx = (
            np.arange(N_skewers)
            if N_plot_sample >= N_skewers
            else np.sort(rng.choice(N_skewers, size=N_plot_sample, replace=False))
        )
    else:
        idx = np.clip(np.asarray(sample_indices, dtype=int), 0, N_skewers - 1)

    labels, species = _line_labels_and_species(line_bundle, N_lines)

    unique_species = list(dict.fromkeys(species))
    species_color = _neon_colors(unique_species)

    base_linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2))]

    species_counts = {sp: species.count(sp) for sp in unique_species}
    species_seen = {sp: 0 for sp in unique_species}

    line_style = []
    for sp in species:
        i_seen = species_seen[sp]
        species_seen[sp] += 1

        if species_counts[sp] == 1:
            line_style.append("-")
        else:
            line_style.append(base_linestyles[i_seen % len(base_linestyles)])

    if total_glow_lw is None:
        total_glow_lw = 3.0 * total_lw

    if line_legend_lw is None:
        line_legend_lw = line_lw

    fig, axes = plt.subplots(
        len(idx),
        1,
        figsize=(14, 2.2 * len(idx)),
        sharex=True,
        sharey=True,
        facecolor="black",
    )
    axes = np.atleast_1d(axes)

    for ax, sk_idx in zip(axes, idx):
        ax.set_facecolor("black")

        # Individual line transmissions first.
        for i_line, lab in enumerate(labels):
            sp = species[i_line]
            c = species_color[sp]
            ls = line_style[i_line]
            y = F_over_F0[i_line, sk_idx, :]

            ax.plot(
                x_axis_array,
                y,
                color=c,
                lw=line_glow_lw,
                ls=ls,
                alpha=line_glow_alpha,
                zorder=2,
            )
            ax.plot(
                x_axis_array,
                y,
                color=c,
                lw=line_lw,
                ls=ls,
                alpha=line_alpha,
                zorder=3,
            )

        # Total transmitted flux last, on top.
        F_total = np.prod(F_over_F0[:, sk_idx, :], axis=0)

        ax.plot(
            x_axis_array,
            F_total,
            color=total_color,
            lw=total_glow_lw,
            alpha=total_glow_alpha,
            zorder=8,
        )
        ax.plot(
            x_axis_array,
            F_total,
            color=total_color,
            lw=total_lw,
            alpha=total_alpha,
            zorder=9,
        )

        ax.text(
            0.015,
            0.12,
            f"sk. #{sk_idx}",
            transform=ax.transAxes,
            color="white",
            fontsize=12,
            ha="left",
            va="bottom",
        )

        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel(r"$e^{-\tau}$", fontsize=20, color="white")

        ax.tick_params(
            axis="both",
            which="major",
            labelsize=14,
            colors="white",
            length=6,
            width=1.1,
        )

        for side in ["top", "right", "bottom", "left"]:
            ax.spines[side].set_color("white")
            ax.spines[side].set_linewidth(1.0)

    handles = []

    for i_line, lab in enumerate(labels):
        sp = species[i_line]
        handles.append(
            Line2D(
                [0],
                [0],
                color=species_color[sp],
                lw=line_legend_lw,
                ls=line_style[i_line],
                label=lab,
            )
        )

    handles.append(
        Line2D(
            [0],
            [0],
            color=total_color,
            lw=total_lw,
            ls="-",
            label="Total",
        )
    )

    axes[-1].set_xlabel(x_label, fontsize=20, color="white")

    leg = fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncols=min(len(handles), 7),
        frameon=True,
        fontsize=16,
        handlelength=2.6,
        facecolor="black",
        edgecolor="white",
        framealpha=1,
    )

    for text in leg.get_texts():
        text.set_color("white")

    fig.subplots_adjust(top=0.88, hspace=0.08)

    plt.show()

    return fig, axes, idx