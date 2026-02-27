import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D


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
            ax_prof.set_ylabel("$\\log_{10} n \\,\\mathrm{[cm^{-3}]}$")

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
    savepath_png: str | None = None,
    show: bool = False,
    skewer_cmap=None,
    skewer_norm: mpl.colors.Normalize | None = None,
    skewer_last_index: int | None = None,
    save_pad_inches: float = 0.08,
):
    species_keys = list(line_bundle.species_keys)
    line_indices, line_objs = [], []
    for sp in species_keys:
        ln0 = line_bundle.lines_for_species(sp)[0]
        i0 = next(i for i, ln in enumerate(line_bundle.lines) if ln.id == ln0.id)
        line_indices.append(i0)
        line_objs.append(ln0)

    if skewer_cmap is None:
        skewer_cmap = mpl.cm.get_cmap("gist_rainbow")
    if skewer_norm is None:
        skewer_norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)

    curve_color = "k" if skewer_last_index is None else skewer_cmap(skewer_norm(float(skewer_last_index)))

    ncols = len(species_keys)
    fig, axes = plt.subplots(1, ncols, figsize=(5.8 * ncols, 3.5), sharex=True, sharey=True)
    if ncols == 1:
        axes = np.array([axes])

    ymin, ymax = -0.05, 1.05

    for c, (sp, i_line, ln0) in enumerate(zip(species_keys, line_indices, line_objs)):
        ax = axes[c]
        F_line = F_over_F0[i_line, ic, jc, :]

        ax.plot(wavelengths, F_line, color=curve_color, lw=2.6)

        lam0 = getattr(ln0, "lambda0_A", None)
        if lam0 is not None:
            ax.axvline(lam0, color="grey", linestyle=":", lw=1.5, alpha=1.0)

        ax.set_title(f"{sp}", pad=10)
        ax.set_xlim(float(np.min(wavelengths)), float(np.max(wavelengths)))
        ax.set_ylim(ymin, ymax)

        ax.set_xlabel(r"Wavelength $\lambda\ [\mathrm{\AA}]$")
        if c == 0:
            ax.set_ylabel(r"$e^{-\tau}$")

        ax.tick_params(which="both", direction="in", top=True, right=True, length=7, width=1.6)
        ax.tick_params(which="minor", length=4)
        for spine in ax.spines.values():
            spine.set_linewidth(1.8)

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
        )

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, axes

def _interp_index_from_scale_factors(a_target: np.ndarray, a_grid: np.ndarray) -> np.ndarray:
    a_grid = np.asarray(a_grid, dtype=float)
    f_grid = np.arange(a_grid.size, dtype=float)

    if a_grid[0] <= a_grid[-1]:
        a_sorted, f_sorted = a_grid, f_grid
    else:
        a_sorted, f_sorted = a_grid[::-1], f_grid[::-1]

    a_target = np.asarray(a_target, dtype=float)
    a_target = np.clip(a_target, a_sorted[0], a_sorted[-1])
    return np.interp(a_target, a_sorted, f_sorted)


def plot_flux_panels_colored_by_los(
    *,
    wavelengths: np.ndarray,            # (N_lambda,)
    F_over_F0: np.ndarray,              # (N_lines, N_skx, N_sky, N_lambda)
    ic: int,
    jc: int,
    line_bundle,
    skewer_bundle,
    N_grid_skewers: int,
    savepath_png: str | None = None,
    show: bool = False,
    skewer_cmap=None,
    skewer_norm: mpl.colors.Normalize | None = None,   # Normalize(vmin=0, vmax=N_ppsk-1)
    linestyles: list[str] | None = None,
    add_legend: bool = True,
    ylim: tuple[float, float] = (-0.05, 1.05),
):
    lam = np.asarray(wavelengths, dtype=float)
    F_np = np.asarray(F_over_F0, dtype=float)

    if skewer_cmap is None:
        skewer_cmap = mpl.cm.get_cmap("gist_rainbow")
    if skewer_norm is None:
        skewer_norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
    if linestyles is None:
        linestyles = ["-", "--", "-.", ":"]

    _scale_factors = np.asarray(skewer_bundle.scale_factors[0, ic * N_grid_skewers + jc], dtype=float)
    a0 = float(_scale_factors[0])
    a_last = float(_scale_factors[-1])

    fig, ax = plt.subplots(figsize=(10, 4.5))

    legend_handles, legend_labels = [], []

    for i_line, line in enumerate(line_bundle.lines):
        ls = linestyles[i_line % len(linestyles)]
        lambda0 = float(line.lambda0_A)
        lambda0_f = lambda0 * a_last / a0

        # vertical markers: same linestyle as the line
        ax.axvline(lambda0,   color="grey", linestyle=ls, linewidth=1.5, alpha=0.95, zorder=1)
        ax.axvline(lambda0_f, color="grey", linestyle=ls, linewidth=1.5, alpha=0.95, zorder=1)

        F_line = F_np[i_line, ic, jc, :]

        a_of_lam = a0 * (lam / lambda0)
        f_of_lam = _interp_index_from_scale_factors(a_of_lam, _scale_factors)

        pts = np.column_stack([lam, F_line]).reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)

        # colored curve (solid, to preserve colormap)
        lc = LineCollection(segs, cmap=skewer_cmap, norm=skewer_norm)
        lc.set_array(f_of_lam[:-1])
        lc.set_linewidth(2.5)
        lc.set_linestyle("solid")
        ax.add_collection(lc)

        if add_legend:
            name = getattr(line, "name", None) or getattr(line, "label", None) or f"Line {i_line}"
            legend_handles.append(Line2D([0], [0], color="k", lw=2.2, linestyle=ls))
            legend_labels.append(name)

    ax.set_xlim(float(np.min(lam)), float(np.max(lam)))
    ax.set_ylim(*ylim)

    ax.set_xlabel(r"Wavelength $\lambda\ [\mathrm{\AA}]$", fontsize=20)
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
        plt.show()
    else:
        plt.close(fig)

    return fig, ax