import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerBase


# ============================================================
# Labels / legend helpers
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
import warnings

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