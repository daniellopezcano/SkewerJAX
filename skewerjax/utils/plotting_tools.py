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
            origin="lower",
            extent=extent_yz,
            aspect="auto",
            norm=density_norm,
            cmap="cividis",
        )

        ax_img.plot(y_c, z_c, color="k", lw=1.0, alpha=0.35, zorder=4)
        ax_img.scatter(y_c, z_c, c=idx, cmap=skewer_cmap, norm=skewer_norm, s=1, linewidths=1.0, zorder=5)

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
    scale_factor_ratio: float,
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
    """
    Plot one subpanel per *absorption line* (not per species), with the subplot title set to the line name.

    Conceptual behavior
    -------------------
    For each line object in `line_bundle.lines`:
      1) Extract the corresponding flux curve F_over_F0[i_line, ic, jc, :].
      2) Plot the curve against the chosen x-axis representation.
      3) Draw two vertical markers:
           - x0: line center at initial scale factor (rest value in the chosen x-axis)
           - xf: line center shifted by scale_factor_ratio
         using a linestyle unique to that line index (cycled).
      4) Title the panel with `line.name`.

    This matches the "one panel per transition" view you asked for, and correctly handles
    multiple transitions per species automatically because we iterate *all* lines.

    Parameters
    ----------
    wavelengths : np.ndarray, shape (N_lambda,)
        Rest wavelength grid.

    F_over_F0 : np.ndarray, shape (N_lines, N_skx, N_sky, N_lambda)
        Flux array aligned with `line_bundle.lines` along axis 0.

    ic, jc : int
        Skewer grid indices.

    line_bundle :
        Must provide `line_bundle.lines` iterable; each line should have:
          - line.lambda0_A
          - line.name (fallback: line.label or "Line i")

    scale_factor_ratio : float
        a0/ai factor used to compute the shifted marker xf.

    x_axis_key : {"wavelengths", "velocities", "log_lambda"}
        X-axis representation.

    lambda0_anchor : float | None
        Required for velocities/log_lambda transformations.

    ncols : int | None
        Number of columns in the subplot grid. If None, choose a reasonable default.

    Returns
    -------
    fig, axes
        Matplotlib figure and 2D axes array (nrows, ncols).
    """

    linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1))]

    assert x_axis_key in ("wavelengths", "velocities", "log_lambda"), (
        "x_axis_key should be either 'wavelengths', 'velocities' or 'log_lambda'"
    )

    if x_axis_key == "wavelengths":
        x_label = r"$\lambda\ [\mathrm{\AA}]$"
        x_axis_array = np.asarray(wavelengths, dtype=float)
    elif x_axis_key == "velocities":
        if lambda0_anchor is None:
            raise ValueError("lambda0_anchor must be provided when x_axis_key='velocities'")
        x_label = r"Velocity space [$\mathrm{km\,/\,s}$]"
        x_axis_array = phys.c*1e-3 * (wavelengths * scale_factor_ratio - lambda0_anchor) / (lambda0_anchor * scale_factor_ratio) # km/s
    else:  # log_lambda
        if lambda0_anchor is None:
            raise ValueError("lambda0_anchor must be provided when x_axis_key='log_lambda'")
        if lambda0_anchor == 1:
            x_label = r"$\log_{10}(\lambda \frac{a_0}{a_\mathrm{QSO}} \left[\mathrm{\AA}\right])$"
        else:
            x_label = r"$\log_{10}( \frac{\lambda a_0}{\lambda_\dagger a_\mathrm{QSO}}/ )$"
        x_axis_array = np.log10(wavelengths * scale_factor_ratio / lambda0_anchor)

    x_axis_array = np.asarray(x_axis_array, dtype=float)
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

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(6.0 * ncols, 3.4 * nrows),
        sharex=True, sharey=True
    )
    axes = np.atleast_2d(axes)

    # Plot each line in its own panel
    for i_line, ln in enumerate(lines):
        r = i_line // ncols
        c = i_line % ncols
        ax = axes[r, c]

        ls = linestyles[i_line % len(linestyles)]
        F_line = F_np[i_line, ic, jc, :]
        ax.plot(x_axis_array, F_line, color=curve_color, lw=2.0, ls="-", alpha=0.95)

        lam0_rest = float(getattr(ln, "lambda0_A"))
        lamf_rest = lam0_rest * scale_factor_ratio
        if x_axis_key == "wavelengths":
            xi = lam0_rest
            x0 = lam0_rest
            xf = lamf_rest
        elif x_axis_key == "velocities":
            xi = phys.c * 1e-3 * (lam0_rest - lambda0_anchor) / (lambda0_anchor)
            x0 = phys.c * 1e-3 * (lam0_rest * scale_factor_ratio - lambda0_anchor) / (lambda0_anchor * scale_factor_ratio)
            xf = phys.c * 1e-3 * (lamf_rest * scale_factor_ratio - lambda0_anchor) / (lambda0_anchor * scale_factor_ratio)
        else:  # log_lambda
            xi = np.log10(lam0_rest / lambda0_anchor)
            x0 = np.log10(lam0_rest * scale_factor_ratio / lambda0_anchor)
            xf = np.log10(lamf_rest * scale_factor_ratio / lambda0_anchor)
        ax.axvline(xi, color="grey", linestyle=ls, lw=1.5, alpha=1.0)
        ax.axvline(x0, color="grey", linestyle=ls, lw=1.5, alpha=1.0)
        ax.axvline(xf, color="grey", linestyle=ls, lw=1.5, alpha=1.0)

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
    skewer_bundle,
    min_x, max_x, skewer_last_index,
    N_grid_skewers: int,
    lambda0_anchor: float | None = None,
    savepath_png: str | None = None,
    show: bool = False,
    skewer_cmap=None,
    skewer_norm: mpl.colors.Normalize | None = None,
    linestyles: list[str] | None = None,
    add_legend: bool = True,
    ylim: tuple[float, float] = (-0.05, 1.05),
):  
    
    _scale_factors = np.asarray(skewer_bundle.scale_factors[0, ic * N_grid_skewers + jc], dtype=float)
    scale_factor_ratio = float(_scale_factors[-1] / _scale_factors[0])

    assert x_axis_key in ("wavelengths", "velocities", "log_lambda"), "x_axis_key should be either 'wavelengths', 'velocities' or 'log_lambda'"
    if x_axis_key == "wavelengths":
        x_label = r"$\lambda\ [\mathrm{\AA}]$"
        x_axis_array = np.asarray(wavelengths, dtype=float)
    elif x_axis_key == "velocities":
        x_label = r"Velocity space [$\mathrm{km\,/\,s}$]"
        x_axis_array = phys.c*1e-3 * (wavelengths*scale_factor_ratio - lambda0_anchor) / (lambda0_anchor*scale_factor_ratio) # km/s
    elif x_axis_key == "log_lambda":
        if lambda0_anchor == 1:
            x_label = r"$\log_{10}(\lambda \frac{a_0}{a_\mathrm{QSO}} \left[\mathrm{\AA}\right])$"
        else:
            x_label = r"$\log_{10}( \frac{\lambda a_0}{\lambda_\dagger a_\mathrm{QSO}})$"
        x_axis_array = np.log10(wavelengths*scale_factor_ratio / lambda0_anchor)

    x_axis_array = np.asarray(x_axis_array, dtype=float)
    F_np = np.asarray(F_over_F0, dtype=float)

    if skewer_cmap is None:
        skewer_cmap = mpl.cm.get_cmap("gist_rainbow")
    if linestyles is None:
        linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1))]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    legend_handles, legend_labels = [], []
    for i_line, line in enumerate(line_bundle.lines):
        ls = linestyles[i_line % len(linestyles)]

        lam0_rest = float(line.lambda0_A)
        lamf_rest = lam0_rest * scale_factor_ratio
        if x_axis_key == "wavelengths":
            xi = lam0_rest
            x0 = lam0_rest
            xf = lamf_rest
        elif x_axis_key == "velocities":
            xi = phys.c*1e-3 * (lam0_rest - lambda0_anchor) / (lambda0_anchor)
            x0 = phys.c*1e-3 * (lam0_rest*scale_factor_ratio - lambda0_anchor) / (lambda0_anchor * scale_factor_ratio)
            xf = phys.c*1e-3 * (lamf_rest*scale_factor_ratio - lambda0_anchor) / (lambda0_anchor * scale_factor_ratio)
        elif x_axis_key == "log_lambda":
            xi = np.log10(lam0_rest / lambda0_anchor)
            x0 = np.log10(lam0_rest*scale_factor_ratio / lambda0_anchor)
            xf = np.log10(lamf_rest*scale_factor_ratio / lambda0_anchor)
        ax.axvline(xi, color="grey", linestyle=linestyles[i_line], lw=1.5, alpha=1.0)
        ax.axvline(x0, color="grey", linestyle=linestyles[i_line], lw=1.5, alpha=1.0)
        ax.axvline(xf, color="grey", linestyle=linestyles[i_line], lw=1.5, alpha=1.0)

        F_line = F_np[i_line, ic, jc, :]
        pts  = np.column_stack([x_axis_array, F_line]).reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)

        x_mid = 0.5 * (x_axis_array[:-1] + x_axis_array[1:])

        xmin, xmax = (min(x0, xf), max(x0, xf))
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
            legend_handles, legend_labels, loc="lower right", fontsize=16, handlelength=2.6, handletextpad=0.6, borderaxespad=0.6, ncols=1
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