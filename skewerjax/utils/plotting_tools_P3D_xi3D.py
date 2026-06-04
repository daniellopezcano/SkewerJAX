"""
utils/plotting_tools_P3D_xi3D.py
=================================
Plotting utilities for 3D power spectrum P3D(k, μ) and 3D two-point
correlation function xi3D(r_perp, π) statistics produced by
summary_stats.P3D_xi3D.OverfluxStatistics3D.

Structure mirrors plotting_tools_P1D_xi1D.py so that the two modules
can be used side-by-side in the same analysis notebook.

Four main public functions
--------------------------
plot_p3d_by_order          – per-order (k, μ) wedge panels + residuals
plot_p3d_total_recovery    – total vs reconstructed P3D(k, μ) + residuals
plot_xi3d_rppi_by_order    – per-order xi(r_perp, π) 2-D maps
plot_xi3d_total_recovery   – total vs reconstructed xi(r_perp, π) maps

Plus convenience helpers shared with the 1D module (re-exported):
    pretty_subset, pretty_pair, pretty_pair_sym
    pair_order, is_auto_pair
    build_order_sum_dict
    collapse_order_pairs     (P3D version: operates on 2-D (k,μ) arrays)
    apply_clean_symlog
"""

from __future__ import annotations

import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerBase


# ============================================================
# Re-used helpers (identical to plotting_tools_P1D_xi1D)
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


def _add_order_badge(ax, order, *, fontsize=22, boxstyle="round,pad=0.25",
                     facecolor="white", edgecolor="black", alpha=0.9):
    ax.text(
        0.02, 0.96, rf"$\mathcal{{O}}({order})$",
        transform=ax.transAxes, va="top", ha="left", fontsize=fontsize,
        bbox=dict(boxstyle=boxstyle, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha),
    )


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
        acc = np.zeros_like(template, dtype=np.float64)
        for pair in ordered_pairs:
            if pair_order(pair) == order:
                acc += np.asarray(curve_dict[pair])
        out[order] = acc
    return out, orders


# ============================================================
# Symlog helpers (identical to P1D module)
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
    return rf"${sign}10^{{{exponent}}}$" if np.isclose(coeff, 1.0, rtol=1e-10, atol=1e-14) \
        else rf"${sign}{coeff:g}\times 10^{{{exponent}}}$"


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
            return rf"${sign}10^{{{exponent}}}$" if base == 10 else rf"${sign}{base}^{{{exponent}}}$"
        if not label_minor:
            return ""
        return rf"${sign}{coeff:g}\times 10^{{{exponent}}}$" if base == 10 \
            else rf"${sign}{coeff:g}\times {base}^{{{exponent}}}$"

    return mticker.FuncFormatter(_formatter)


def apply_clean_symlog(ax, axis="y", linthresh=1e-3, linscale=1.0, base=10,
                       subs=None, label_minor=False, sci_cutoff=1e-2):
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
    axis_obj.set_major_formatter(_make_symlog_formatter(linthresh=linthresh, base=base,
                                                        label_minor=False, sci_cutoff=sci_cutoff))
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
# P3D-specific pair collapsing (operates on 2D (n_k, n_mu) arrays)
# ============================================================

def collapse_order_pairs_p3d(pairs_this_order, Pk3d_pair, tolerance_threshold=1e-6):
    """
    Collapse symmetric off-diagonal pairs for a given decomposition order.

    Parameters
    ----------
    pairs_this_order : list of (str, str)
    Pk3d_pair        : dict mapping pair -> ndarray of shape (n_k, n_mu)

    Returns
    -------
    entries          : list of dicts with keys: pair, curve, ls, label, is_auto, color
    Pk3d_sum         : summed array  (n_k, n_mu)
    Pk3d_sum_full    : raw summed array before symmetry collapsing
    """
    template = np.asarray(Pk3d_pair[pairs_this_order[0]], dtype=float)
    Pk3d_sum_full = np.zeros_like(template)
    for pair in pairs_this_order:
        Pk3d_sum_full += np.asarray(Pk3d_pair[pair])

    processed = set()
    entries = []
    Pk3d_sum = np.zeros_like(template)

    for pair in pairs_this_order:
        if pair in processed:
            continue
        A, B = pair
        rev = (B, A)
        if is_auto_pair(pair):
            curve = np.asarray(Pk3d_pair[pair])
            entries.append(dict(pair=pair, curve=curve, ls="-", label=pretty_pair(pair), is_auto=True))
            Pk3d_sum += curve
            processed.add(pair)
            continue
        if rev in pairs_this_order and rev not in processed:
            diff = np.nanmax(np.abs(Pk3d_pair[pair] - Pk3d_pair[rev]))
            if diff <= tolerance_threshold:
                curve = 2.0 * np.asarray(Pk3d_pair[pair])
                entries.append(dict(pair=pair, curve=curve, ls="--", label=pretty_pair_sym(pair), is_auto=False))
                Pk3d_sum += curve
                processed.update([pair, rev])
            else:
                for p in [pair, rev]:
                    curve = np.asarray(Pk3d_pair[p])
                    entries.append(dict(pair=p, curve=curve, ls="--", label=pretty_pair(p), is_auto=False))
                    Pk3d_sum += curve
                processed.update([pair, rev])
        else:
            curve = np.asarray(Pk3d_pair[pair])
            entries.append(dict(pair=pair, curve=curve, ls="--", label=pretty_pair(pair), is_auto=False))
            Pk3d_sum += curve
            processed.add(pair)

    return entries, Pk3d_sum, Pk3d_sum_full


# ============================================================
# P3D wedge / multipole slice helpers
# ============================================================

def _mu_slice_label(mu_lo, mu_hi):
    return rf"$\mu \in [{mu_lo:.2f}, {mu_hi:.2f})$"


def _select_mu_bins(mu_bin_centers, n_mu_bins, mu_wedges):
    """
    Return a list of mu-bin column indices to plot.

    Parameters
    ----------
    mu_bin_centers : (n_k, n_mu)
    n_mu_bins      : int
    mu_wedges      : list of int (column indices) or None -> evenly spaced 4 wedges
    """
    if mu_wedges is None:
        step = max(1, n_mu_bins // 4)
        mu_wedges = list(range(0, n_mu_bins, step))
    return mu_wedges


def _p3d_plot_curve(p3d_2d, i_mu):
    """Return the 1D k-slice for a given mu-bin index, ignoring NaN."""
    return np.asarray(p3d_2d[:, i_mu], dtype=float)


# ============================================================
# P3D  – plot by order
# ============================================================

def plot_p3d_by_order(
    ordered_pairs,
    Pk3d_pair,
    k_bin_centers,
    mu_bin_centers,
    *,
    mu_wedges=None,
    tolerance_threshold=1e-6,
    residual_mode="sum",
    use_log_x=True,
    use_symlog_main=True,
    use_symlog_residual=False,
    symlog_linscale=1.0,
    symlog_base=10,
    symlog_label_minor=False,
    symlog_sci_cutoff=1e-2,
    xlim=None,
    legend_ncol_terms=2,
    show_order_badge=True,
    xlabel=r"$k$",
    ylabel=r"$P_{\rm 3D}$",
    residual_ylabel=r"$\Delta$",
    figsize=None,
    main_panel_height=5.0,
    residual_panel_height=3.0,
    hspace=0.08,
    label_fontsize=22,
    residual_label_fontsize=20,
    tick_fontsize=22,
    legend_fontsize=18,
    main_curve_lw=2.0,
    collapsed_sum_lw=1.5,
    sum_color="k",
    sum_ls=":",
):
    """
    Plot P3D(k) for each μ wedge, decomposed by overflux-field order.

    Mirrors plot_p1d_by_order: one (main + residual) panel pair per order.
    Within each panel, one line per μ wedge per pair term.

    Parameters
    ----------
    ordered_pairs   : list of (str, str) — from stats3d.ordered_pairs
    Pk3d_pair       : dict (a,b) -> (n_k, n_mu)  — stats3d.Pk3d_pair
    k_bin_centers   : (n_k, n_mu) — stats3d.k_bin_centers
    mu_bin_centers  : (n_k, n_mu) — stats3d.mu_bin_centers
    mu_wedges       : list of int column indices to show (default: 4 evenly spaced)
    """
    if not ordered_pairs:
        raise ValueError("ordered_pairs cannot be empty.")

    cmap = plt.get_cmap("tab10")
    orders = sorted({pair_order(pair) for pair in ordered_pairs})
    n_mu = mu_bin_centers.shape[1]
    mu_wedge_indices = _select_mu_bins(mu_bin_centers, n_mu, mu_wedges)

    if figsize is None:
        figsize = (14, 6.0 * len(orders))

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        nrows=2 * len(orders), ncols=1,
        height_ratios=[v for _ in orders for v in (main_panel_height, residual_panel_height)],
        hspace=hspace,
    )

    axes, axes_res = [], []
    for i in range(len(orders)):
        ax = fig.add_subplot(gs[2 * i], sharex=axes[0] if axes else None)
        ax_res = fig.add_subplot(gs[2 * i + 1], sharex=ax)
        axes.append(ax)
        axes_res.append(ax_res)

    order_sum_dict = {}

    for ax, ax_res, order in zip(axes, axes_res, orders):
        pairs_this_order = [p for p in ordered_pairs if pair_order(p) == order]
        entries, Pk3d_order_sum, Pk3d_order_sum_full = collapse_order_pairs_p3d(
            pairs_this_order, Pk3d_pair, tolerance_threshold=tolerance_threshold,
        )
        order_sum_dict[order] = Pk3d_order_sum

        # μ-wedge colour cycle: outer = pair entry, inner = mu wedge
        # Use a combined (entry_index * n_wedge + mu_index) colour for clarity
        legend_terms = [Line2D([0], [0], color=sum_color, lw=collapsed_sum_lw, ls=sum_ls, label="Sum")]
        mu_cmap = plt.get_cmap("viridis")

        for i_entry, entry in enumerate(entries):
            base_color = cmap(i_entry % 10)
            entry["color"] = base_color
            for i_w, i_mu in enumerate(mu_wedge_indices):
                k_vals    = k_bin_centers[:, i_mu]
                p_vals    = entry["curve"][:, i_mu]
                mu_lo = mu_bin_centers[0, i_mu]
                mu_hi_idx = min(i_mu + 1, n_mu - 1)
                mu_hi = mu_bin_centers[0, mu_hi_idx]

                alpha = 0.5 + 0.5 * (i_w / max(len(mu_wedge_indices) - 1, 1))
                ax.plot(k_vals, p_vals, lw=main_curve_lw, ls=entry["ls"],
                        color=base_color, alpha=alpha)

                if i_entry == 0:
                    legend_terms.append(
                        Line2D([0], [0], color="grey", lw=1.5, alpha=alpha,
                               label=_mu_slice_label(mu_lo, mu_hi))
                    )

            legend_terms.insert(
                1, Line2D([0], [0], color=base_color, lw=main_curve_lw,
                          ls=entry["ls"], label=entry["label"])
            )

        # Sum per wedge
        for i_w, i_mu in enumerate(mu_wedge_indices):
            k_vals = k_bin_centers[:, i_mu]
            alpha = 0.5 + 0.5 * (i_w / max(len(mu_wedge_indices) - 1, 1))
            ax.plot(k_vals, Pk3d_order_sum[:, i_mu], lw=collapsed_sum_lw,
                    color=sum_color, ls=sum_ls, alpha=alpha)

        ax.axhline(0.0, color="k", lw=0.8, alpha=0.6, ls=":")
        ax.legend(handles=legend_terms, fontsize=legend_fontsize,
                  loc="upper right", ncol=legend_ncol_terms)
        ax.set_ylabel(ylabel, fontsize=label_fontsize)
        ax.tick_params(axis="both", labelsize=tick_fontsize, labelbottom=False)

        if show_order_badge:
            _add_order_badge(ax, order)

        if use_symlog_main:
            linthresh = _nonzero_min([Pk3d_order_sum])
            apply_clean_symlog(ax, axis="y", linthresh=linthresh,
                               linscale=symlog_linscale, base=symlog_base,
                               label_minor=symlog_label_minor,
                               sci_cutoff=symlog_sci_cutoff)
        if use_log_x:
            ax.set_xscale("log")

        # Residuals
        nonzero_res = []
        if residual_mode == "sum":
            Pk3d_ref = Pk3d_order_sum
            for entry in entries:
                for i_w, i_mu in enumerate(mu_wedge_indices):
                    k_vals = k_bin_centers[:, i_mu]
                    mask = np.isfinite(k_vals) & np.isfinite(entry["curve"][:, i_mu])
                    if not mask.any():
                        continue
                    res = _safe_relative_residual(
                        entry["curve"][:, i_mu], Pk3d_ref[:, i_mu]
                    )
                    alpha = 0.5 + 0.5 * (i_w / max(len(mu_wedge_indices) - 1, 1))
                    ax_res.plot(k_vals, res, lw=1.5, ls=entry["ls"],
                                color=entry["color"], alpha=alpha)
                    nz = np.abs(res[np.abs(res) > 0])
                    if len(nz):
                        nonzero_res.append(np.min(nz))
        elif residual_mode == "auto_reference":
            auto = [e for e in entries if e["is_auto"]]
            if auto:
                for entry in auto:
                    Pk3d_ref = entry["curve"]
                    for i_w, i_mu in enumerate(mu_wedge_indices):
                        k_vals = k_bin_centers[:, i_mu]
                        res = _safe_relative_residual(
                            entry["curve"][:, i_mu], Pk3d_ref[:, i_mu]
                        )
                        alpha = 0.5 + 0.5 * (i_w / max(len(mu_wedge_indices) - 1, 1))
                        ax_res.plot(k_vals, res, lw=1.5, ls=entry["ls"],
                                    color=entry["color"], alpha=alpha)
            else:
                ax_res.text(0.5, 0.5, "No auto terms in this order",
                            transform=ax_res.transAxes, fontsize=16,
                            va="center", ha="center")
        else:
            raise ValueError("residual_mode must be 'sum' or 'auto_reference'")

        ax_res.axhline(0.0, color="k", lw=0.8, alpha=0.6, ls=":")
        ax_res.set_ylabel(residual_ylabel, fontsize=residual_label_fontsize)
        ax_res.tick_params(axis="both", labelsize=tick_fontsize,
                           labelbottom=(ax_res == axes_res[-1]))

        if use_symlog_residual:
            apply_clean_symlog(ax_res, axis="y",
                               linthresh=(np.min(nonzero_res) if nonzero_res else 1e-12),
                               linscale=symlog_linscale, base=symlog_base,
                               label_minor=symlog_label_minor,
                               sci_cutoff=symlog_sci_cutoff)
        if use_log_x:
            ax_res.set_xscale("log")

        if xlim is not None:
            ax.set_xlim(xlim)
            ax_res.set_xlim(xlim)

    axes_res[-1].set_xlabel(xlabel, fontsize=label_fontsize)
    plt.tight_layout()
    plt.show()
    return fig, axes, axes_res, order_sum_dict


# ============================================================
# P3D  – total recovery
# ============================================================

def plot_p3d_total_recovery(
    Pk3d_total,
    Pk3d_total_from_terms,
    Pk3d_pair,
    k_bin_centers,
    mu_bin_centers,
    *,
    ordered_pairs=None,
    mu_wedges=None,
    use_log_x=True,
    use_symlog_main=True,
    use_symlog_residual=False,
    main_symlog_linthresh=None,
    residual_symlog_linthresh=1e-3,
    symlog_linscale=1.0,
    symlog_base=10,
    symlog_label_minor=False,
    symlog_sci_cutoff=1e-2,
    xlim=None,
    show_order_sums=True,
    show_order_sum_residuals=True,
    total_lw=3.0,
    recovered_lw=1.5,
    recovered_ls="--",
    total_color="k",
    recovered_color="red",
    order_sum_lw=2.0,
    order_sum_alpha=0.95,
    figsize=(14, 7.0),
    hspace=0.06,
    label_fontsize=22,
    residual_label_fontsize=20,
    tick_fontsize=22,
    legend_fontsize=18,
    xlabel=r"$k$",
    ylabel=r"$P_{\rm 3D}$",
    residual_ylabel=r"$\Delta$",
):
    """
    Plot total P3D(k) vs reconstructed sum, showing μ wedges.

    Mirrors plot_p1d_total_recovery.

    Parameters
    ----------
    Pk3d_total           : (n_k, n_mu) — stats3d.Pk3d_total
    Pk3d_total_from_terms: (n_k, n_mu) — stats3d.Pk3d_total_from_terms
    Pk3d_pair            : dict (a,b) -> (n_k, n_mu)
    k_bin_centers        : (n_k, n_mu)
    mu_bin_centers       : (n_k, n_mu)
    """
    ordered_pairs = list(Pk3d_pair.keys()) if ordered_pairs is None else ordered_pairs
    cmap = plt.get_cmap("tab10")
    n_mu = mu_bin_centers.shape[1]
    mu_wedge_indices = _select_mu_bins(mu_bin_centers, n_mu, mu_wedges)

    order_sum_dict, orders = build_order_sum_dict(Pk3d_pair, ordered_pairs=ordered_pairs)
    order_color_dict = {order: cmap(i % 10) for i, order in enumerate(orders)}

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[3.3, 1.2], hspace=hspace)
    ax     = fig.add_subplot(gs[0])
    ax_res = fig.add_subplot(gs[1], sharex=ax)

    mu_cmap = plt.get_cmap("viridis")

    order_sum_handles = []
    if show_order_sums:
        for order in orders:
            c = order_color_dict[order]
            for i_w, i_mu in enumerate(mu_wedge_indices):
                alpha = 0.5 + 0.5 * (i_w / max(len(mu_wedge_indices) - 1, 1))
                ax.plot(k_bin_centers[:, i_mu],
                        order_sum_dict[order][:, i_mu],
                        lw=order_sum_lw, color=c, alpha=alpha * order_sum_alpha)
            order_sum_handles.append(
                Line2D([0], [0], color=c, lw=order_sum_lw,
                       label=rf"$\mathcal{{O}}({order})$")
            )

    for i_w, i_mu in enumerate(mu_wedge_indices):
        alpha = 0.5 + 0.5 * (i_w / max(len(mu_wedge_indices) - 1, 1))
        mu_lo = mu_bin_centers[0, i_mu]
        mu_hi = mu_bin_centers[0, min(i_mu + 1, n_mu - 1)]
        ax.plot(k_bin_centers[:, i_mu], Pk3d_total[:, i_mu],
                lw=total_lw, color=total_color, alpha=alpha)
        ax.plot(k_bin_centers[:, i_mu], Pk3d_total_from_terms[:, i_mu],
                lw=recovered_lw, color=recovered_color, ls=recovered_ls, alpha=alpha)

    if use_symlog_main:
        linthresh = (
            _nonzero_min([Pk3d_total])
            if main_symlog_linthresh is None else main_symlog_linthresh
        )
        apply_clean_symlog(ax, axis="y", linthresh=linthresh,
                           linscale=symlog_linscale, base=symlog_base,
                           label_minor=symlog_label_minor, sci_cutoff=symlog_sci_cutoff)

    ax.axhline(0.0, color="k", lw=0.8, alpha=0.6, ls=":")

    if show_order_sums and order_sum_handles:
        leg_ord = ax.legend(handles=order_sum_handles, fontsize=legend_fontsize,
                            loc="upper right", ncol=1, title="Norm. Sum per order")
        ax.add_artist(leg_ord)

    # μ-wedge legend
    mu_leg_handles = []
    for i_w, i_mu in enumerate(mu_wedge_indices):
        alpha = 0.5 + 0.5 * (i_w / max(len(mu_wedge_indices) - 1, 1))
        mu_lo = mu_bin_centers[0, i_mu]
        mu_hi = mu_bin_centers[0, min(i_mu + 1, n_mu - 1)]
        mu_leg_handles.append(
            Line2D([0], [0], color="grey", lw=2.0, alpha=alpha,
                   label=_mu_slice_label(mu_lo, mu_hi))
        )
    leg_mu = ax.legend(handles=mu_leg_handles, fontsize=legend_fontsize - 2,
                       loc="lower left", ncol=2, title=r"$\mu$ wedges")
    ax.add_artist(leg_mu)

    leg_tot = ax.legend(
        handles=[
            Line2D([0], [0], color=total_color, lw=total_lw, label=r"Tot."),
            Line2D([0], [0], color=recovered_color, lw=recovered_lw,
                   ls=recovered_ls, label=r"Tot. Sum"),
        ],
        fontsize=legend_fontsize, loc="upper left", ncol=2,
    )
    ax.add_artist(leg_tot)

    ax.tick_params(axis="both", labelsize=tick_fontsize, labelbottom=False)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)
    if use_log_x:
        ax.set_xscale("log")
    if xlim is not None:
        ax.set_xlim(xlim)

    # Residuals
    for i_w, i_mu in enumerate(mu_wedge_indices):
        alpha = 0.5 + 0.5 * (i_w / max(len(mu_wedge_indices) - 1, 1))
        k_vals  = k_bin_centers[:, i_mu]
        ref     = Pk3d_total[:, i_mu]
        if show_order_sum_residuals:
            for order in orders:
                c = order_color_dict[order]
                ax_res.plot(k_vals,
                            _safe_relative_residual(order_sum_dict[order][:, i_mu], ref),
                            lw=1.0, color=c, alpha=alpha * order_sum_alpha)
        ax_res.plot(k_vals,
                    _safe_relative_residual(Pk3d_total_from_terms[:, i_mu], ref),
                    lw=2.0, color=recovered_color, ls=recovered_ls, alpha=alpha)

    ax_res.axhline(0.0, color="k", lw=0.8, alpha=0.6, ls=":")
    if use_symlog_residual:
        apply_clean_symlog(ax_res, axis="y", linthresh=residual_symlog_linthresh,
                           linscale=symlog_linscale, base=symlog_base,
                           label_minor=symlog_label_minor, sci_cutoff=symlog_sci_cutoff)
    if use_log_x:
        ax_res.set_xscale("log")
    if xlim is not None:
        ax_res.set_xlim(xlim)

    ax_res.tick_params(axis="both", labelsize=tick_fontsize)
    ax_res.set_xlabel(xlabel, fontsize=label_fontsize)
    ax_res.set_ylabel(residual_ylabel, fontsize=residual_label_fontsize)

    plt.tight_layout()
    plt.show()
    return fig, (ax, ax_res), order_sum_dict


# ============================================================
# xi3D helpers
# ============================================================

def _xi3d_slice_label(r_perp_val, unit=""):
    return rf"$r_{{\perp}} = {r_perp_val:.1f}$" + (f" {unit}" if unit else "")


def _xi3d_imshow(
    ax,
    xi_rppi,
    r_perp,
    pi,
    *,
    cmap="RdBu_r",
    symlog=True,
    linthresh=None,
    vmax=None,
):
    """
    Display xi(r_perp, π) as a 2D color map.

    Parameters
    ----------
    xi_rppi : (n_rperp, n_pi) array
    r_perp  : (n_rperp,) array  — transverse lags (x-axis)
    pi      : (n_pi,)   array  — LoS lags (y-axis)
    """
    xi = np.asarray(xi_rppi)
    if linthresh is None:
        linthresh = _nonzero_min([xi]) if np.any(xi != 0) else 1e-4
    if vmax is None:
        vmax = np.nanmax(np.abs(xi))
    vmin = -vmax

    if symlog:
        norm = mcolors.SymLogNorm(linthresh=linthresh, linscale=1.0,
                                  vmin=vmin, vmax=vmax, base=10)
    else:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    extent = [r_perp.min(), r_perp.max(), pi.min(), pi.max()]
    im = ax.imshow(
        xi.T,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
    )
    return im


# ============================================================
# xi3D  – plot by order (2D r_perp / π maps)
# ============================================================

def plot_xi3d_rppi_by_order(
    ordered_pairs,
    xi3d_pair,
    r_perp,
    pi,
    *,
    stats3d=None,
    tolerance_threshold=1e-6,
    cmap_map="RdBu_r",
    symlog_map=True,
    use_symlog_1d=True,
    symlog_linscale=1.0,
    symlog_base=10,
    symlog_sci_cutoff=1e-2,
    r_perp_1d_indices=None,
    show_order_badge=True,
    label_fontsize=18,
    tick_fontsize=16,
    legend_fontsize=14,
    xlabel_map=r"$r_\perp$",
    ylabel_map=r"$\pi$",
    xlabel_1d=r"$\pi$",
    ylabel_1d=r"$\xi_{\rm 3D}$",
    unit_label="",
    figsize=None,
):
    """
    Display xi3D(r_perp, π) decomposed by overflux-field order.

    For each order, produce a row of panels:
      [2D map of sum] [1D slice(s) of each pair term]

    Parameters
    ----------
    ordered_pairs     : list of (str, str)
    xi3d_pair         : dict (a,b) -> (n_rperp, n_pi)  [from xi3d_rppi()]
    r_perp            : 1D ndarray — transverse lag axis
    pi                : 1D ndarray — LoS lag axis
    r_perp_1d_indices : list of int indices into r_perp for 1D line plots
                        (default: [0, n_rperp//4, n_rperp//2])
    stats3d           : OverfluxStatistics3D instance (optional, for xi_total)
    """
    if not ordered_pairs:
        raise ValueError("ordered_pairs cannot be empty.")

    cmap_cycle = plt.get_cmap("tab10")
    orders = sorted({pair_order(pair) for pair in ordered_pairs})
    n_rperp = len(r_perp)

    if r_perp_1d_indices is None:
        r_perp_1d_indices = sorted({0, n_rperp // 4, n_rperp // 2})
        r_perp_1d_indices = [i for i in r_perp_1d_indices if i < n_rperp]

    if figsize is None:
        figsize = (5 * (1 + len(r_perp_1d_indices)), 5.0 * len(orders))

    n_cols = 1 + len(r_perp_1d_indices)
    fig, axes_grid = plt.subplots(
        len(orders), n_cols,
        figsize=figsize,
        squeeze=False,
    )

    order_sum_dict = {}
    xi_sum_dict    = {}

    for i_order, order in enumerate(orders):
        pairs_this_order = [p for p in ordered_pairs if pair_order(p) == order]
        entries, xi_sum, xi_sum_full = collapse_order_pairs_p3d(
            pairs_this_order, xi3d_pair, tolerance_threshold=tolerance_threshold,
        )
        order_sum_dict[order] = xi_sum
        xi_sum_dict[order]    = xi_sum

        # ---- 2D map ----
        ax_map = axes_grid[i_order, 0]
        im = _xi3d_imshow(ax_map, xi_sum, r_perp, pi, cmap=cmap_map, symlog=symlog_map)
        plt.colorbar(im, ax=ax_map, fraction=0.046, pad=0.04)
        ax_map.set_xlabel(xlabel_map, fontsize=label_fontsize)
        ax_map.set_ylabel(ylabel_map, fontsize=label_fontsize)
        ax_map.tick_params(labelsize=tick_fontsize)
        if show_order_badge:
            _add_order_badge(ax_map, order, fontsize=label_fontsize)

        # ---- 1D slices ----
        for i_w, ir in enumerate(r_perp_1d_indices):
            ax_1d = axes_grid[i_order, 1 + i_w]
            ax_1d.axhline(0.0, color="k", lw=0.8, alpha=0.6, ls=":")
            ax_1d.set_title(_xi3d_slice_label(r_perp[ir], unit_label),
                            fontsize=label_fontsize - 2)

            for j, entry in enumerate(entries):
                c = cmap_cycle(j % 10)
                entry["color"] = c
                ax_1d.plot(pi, entry["curve"][ir, :], lw=2.0,
                           ls=entry["ls"], color=c, label=entry["label"])

            ax_1d.plot(pi, xi_sum[ir, :], lw=1.5, color="k", ls=":", label="Sum")
            ax_1d.set_xlabel(xlabel_1d, fontsize=label_fontsize)
            if i_w == 0:
                ax_1d.set_ylabel(ylabel_1d, fontsize=label_fontsize)
            ax_1d.tick_params(labelsize=tick_fontsize)

            if use_symlog_1d:
                linthresh = _nonzero_min([xi_sum[ir, :]])
                apply_clean_symlog(ax_1d, axis="y", linthresh=linthresh,
                                   linscale=symlog_linscale, base=symlog_base,
                                   sci_cutoff=symlog_sci_cutoff)
            if i_order == 0 and i_w == len(r_perp_1d_indices) - 1:
                ax_1d.legend(fontsize=legend_fontsize, loc="upper right")

    plt.tight_layout()
    plt.show()
    return fig, axes_grid, order_sum_dict


# ============================================================
# xi3D  – total recovery (2D maps + 1D slices)
# ============================================================

def plot_xi3d_rppi_total_recovery(
    xi3d_total_rppi,
    xi3d_total_from_terms_rppi,
    xi3d_pair_rppi,
    r_perp,
    pi,
    *,
    ordered_pairs=None,
    r_perp_1d_indices=None,
    cmap_map="RdBu_r",
    symlog_map=True,
    use_symlog_1d=True,
    symlog_linscale=1.0,
    symlog_base=10,
    symlog_sci_cutoff=1e-2,
    show_order_sums=True,
    show_order_sum_residuals=True,
    total_lw=3.0,
    decomp_lw=1.5,
    decomp_ls="--",
    total_color="k",
    decomp_color="red",
    figsize=None,
    label_fontsize=18,
    tick_fontsize=16,
    legend_fontsize=14,
    xlabel_map=r"$r_\perp$",
    ylabel_map=r"$\pi$",
    xlabel_1d=r"$\pi$",
    ylabel_1d=r"$\xi_{\rm 3D}$",
    residual_ylabel=r"$\Delta$",
    unit_label="",
):
    """
    Display xi3D total vs reconstructed sum.

    Layout per column (each column = one r_perp slice):
      [2D map : total]  [2D map : decomposed]  [1D slice : both + residual]

    Parameters
    ----------
    xi3d_total_rppi           : (n_rperp, n_pi)
    xi3d_total_from_terms_rppi: (n_rperp, n_pi)
    xi3d_pair_rppi            : dict (a,b) -> (n_rperp, n_pi)  [from xi3d_rppi()]
    r_perp                    : 1D ndarray
    pi                        : 1D ndarray
    """
    ordered_pairs = list(xi3d_pair_rppi.keys()) if ordered_pairs is None else ordered_pairs
    cmap_cycle = plt.get_cmap("tab10")
    n_rperp = len(r_perp)
    orders = sorted({pair_order(p) for p in ordered_pairs})
    order_color_dict = {order: cmap_cycle(i % 10) for i, order in enumerate(orders)}

    if r_perp_1d_indices is None:
        r_perp_1d_indices = sorted({0, n_rperp // 4, n_rperp // 2})
        r_perp_1d_indices = [i for i in r_perp_1d_indices if i < n_rperp]

    n_1d = len(r_perp_1d_indices)
    # Layout: row 0 = 2D maps (total + decomp) + 1D slices; row 1 = residuals for 1D slices
    n_cols = 2 + n_1d     # [map_total | map_decomp | slice_0 | slice_1 | ...]
    if figsize is None:
        figsize = (5 * n_cols, 9.0)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(nrows=2, ncols=n_cols,
                          height_ratios=[3.5, 1.2], hspace=0.08)

    ax_map_tot  = fig.add_subplot(gs[0, 0])
    ax_map_dec  = fig.add_subplot(gs[0, 1])
    axes_1d     = [fig.add_subplot(gs[0, 2 + i]) for i in range(n_1d)]
    axes_res    = [fig.add_subplot(gs[1, 2 + i], sharex=axes_1d[i]) for i in range(n_1d)]

    xi_total = np.asarray(xi3d_total_rppi)
    xi_decomp = np.asarray(xi3d_total_from_terms_rppi)

    vmax = np.nanmax(np.abs(xi_total))
    linthresh_global = _nonzero_min([xi_total])

    # 2D map: total
    im1 = _xi3d_imshow(ax_map_tot, xi_total, r_perp, pi,
                        cmap=cmap_map, symlog=symlog_map,
                        linthresh=linthresh_global, vmax=vmax)
    plt.colorbar(im1, ax=ax_map_tot, fraction=0.046, pad=0.04)
    ax_map_tot.set_title(r"$\xi_{\rm 3D}^{\rm tot}$", fontsize=label_fontsize)
    ax_map_tot.set_xlabel(xlabel_map, fontsize=label_fontsize)
    ax_map_tot.set_ylabel(ylabel_map, fontsize=label_fontsize)
    ax_map_tot.tick_params(labelsize=tick_fontsize)

    # 2D map: decomposed
    im2 = _xi3d_imshow(ax_map_dec, xi_decomp, r_perp, pi,
                        cmap=cmap_map, symlog=symlog_map,
                        linthresh=linthresh_global, vmax=vmax)
    plt.colorbar(im2, ax=ax_map_dec, fraction=0.046, pad=0.04)
    ax_map_dec.set_title(r"$\xi_{\rm 3D}^{\rm Sum}$", fontsize=label_fontsize)
    ax_map_dec.set_xlabel(xlabel_map, fontsize=label_fontsize)
    ax_map_dec.set_ylabel(ylabel_map, fontsize=label_fontsize)
    ax_map_dec.tick_params(labelsize=tick_fontsize)

    # Order sums
    order_sum_dict, _ = build_order_sum_dict(xi3d_pair_rppi, ordered_pairs=ordered_pairs)

    for i_w, ir in enumerate(r_perp_1d_indices):
        ax_1d  = axes_1d[i_w]
        ax_res = axes_res[i_w]

        ax_1d.axhline(0.0, color="k", lw=0.8, alpha=0.6, ls=":")
        ax_res.axhline(0.0, color="k", lw=0.8, alpha=0.6, ls=":")
        ax_1d.set_title(_xi3d_slice_label(r_perp[ir], unit_label),
                        fontsize=label_fontsize - 2)

        if show_order_sums:
            for order in orders:
                c = order_color_dict[order]
                ax_1d.plot(pi, order_sum_dict[order][ir, :],
                           lw=2.0, color=c, alpha=0.85,
                           label=rf"$\mathcal{{O}}({order})$")

        ax_1d.plot(pi, xi_total[ir, :],  lw=total_lw, color=total_color,  label="Tot.")
        ax_1d.plot(pi, xi_decomp[ir, :], lw=decomp_lw, color=decomp_color,
                   ls=decomp_ls, label="Tot. Sum")

        if use_symlog_1d:
            linthresh = _nonzero_min([xi_total[ir, :]])
            apply_clean_symlog(ax_1d, axis="y", linthresh=linthresh,
                               linscale=symlog_linscale, base=symlog_base,
                               sci_cutoff=symlog_sci_cutoff)

        ax_1d.set_xlabel(xlabel_1d, fontsize=label_fontsize)
        if i_w == 0:
            ax_1d.set_ylabel(ylabel_1d, fontsize=label_fontsize)
        ax_1d.tick_params(labelsize=tick_fontsize, labelbottom=False)
        if i_w == n_1d - 1:
            ax_1d.legend(fontsize=legend_fontsize, loc="upper right")

        # Residuals
        xi_ref = xi_total[ir, :]
        if show_order_sum_residuals:
            for order in orders:
                c = order_color_dict[order]
                ax_res.plot(pi, _safe_relative_residual(order_sum_dict[order][ir, :], xi_ref),
                            lw=1.0, color=c, alpha=0.85)
        ax_res.plot(pi, _safe_relative_residual(xi_decomp[ir, :], xi_ref),
                    lw=2.0, color=decomp_color, ls=decomp_ls)

        ax_res.set_xlabel(xlabel_1d, fontsize=label_fontsize)
        if i_w == 0:
            ax_res.set_ylabel(residual_ylabel, fontsize=label_fontsize - 2)
        ax_res.tick_params(labelsize=tick_fontsize)

    plt.tight_layout()
    plt.show()
    return fig, (ax_map_tot, ax_map_dec), axes_1d, axes_res, order_sum_dict