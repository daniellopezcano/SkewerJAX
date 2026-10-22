"""Shared plotting style (Paul Tol's colour-blind-safe palette on the package's serif style) and a few figure helpers."""
from __future__ import annotations

from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler

# --------------------------------------------------------------------------- style
TOL_BRIGHT = {"blue": "#4477AA", "red": "#EE6677", "green": "#228833", "yellow": "#CCBB44",
              "cyan": "#66CCEE", "purple": "#AA3377", "grey": "#BBBBBB"}
TOL_BRIGHT_CYCLE = list(TOL_BRIGHT.values())
_TOL_CMAPS = {      # Tol's sequential and diverging schemes (https://tol-colors.readthedocs.io), with their "bad" colour
    "iridescent": (["#FEFBE9", "#FCF7D5", "#F5F3C1", "#EAF0B5", "#DDECBF", "#D0E7CA", "#C2E3D2", "#B5DDD8", "#A8D8DC",
                    "#9BD2E1", "#8DCBE4", "#81C4E7", "#7BBCE7", "#7EB2E4", "#88A5DD", "#9398D2", "#9B8AC4", "#9D7DB2",
                    "#9A709E", "#906388", "#805770", "#684957", "#46353A"], "#999999"),
    "YlOrBr": (["#FFFFE5", "#FFF7BC", "#FEE391", "#FEC44F", "#FB9A29", "#EC7014", "#CC4C02", "#993404", "#662506"], "#888888"),
    "sunset": (["#364B9A", "#4A7BB7", "#6EA6CD", "#98CAE1", "#C2E4EF", "#EAECCC", "#FEDA8B", "#FDB366", "#F67E4B",
                "#DD3D2D", "#A50026"], "#FFFFFF"),
    "BuRd": (["#2166AC", "#4393C3", "#92C5DE", "#D1E5F0", "#F7F7F7", "#FDDBC7", "#F4A582", "#D6604D", "#B2182B"], "#FFEE99"),
    "PRGn": (["#762A83", "#9970AB", "#C2A5CF", "#E7D4E8", "#F7F7F7", "#D9F0D3", "#ACD39E", "#5AAE61", "#1B7837"], "#FFEE99"),
}
# one colour per entity and one line style per mode, in every figure
ENTITY_COLORS = {
    "HI": TOL_BRIGHT["blue"], "HI-LyA": TOL_BRIGHT["blue"], "HI-LyB": TOL_BRIGHT["cyan"],
    "SiIII": TOL_BRIGHT["red"], "SiIII-1206": TOL_BRIGHT["red"], "SiII": TOL_BRIGHT["yellow"],
    "fake_spectra": TOL_BRIGHT["grey"], "analytic": TOL_BRIGHT["grey"], "reference": TOL_BRIGHT["grey"],
    "density": TOL_BRIGHT["green"], "temperature": TOL_BRIGHT["purple"], "velocity": TOL_BRIGHT["cyan"],
    "voronoi": TOL_BRIGHT["blue"], "tophat": TOL_BRIGHT["yellow"],
}
MODE_STYLES = {     # SkewerJAX default solid; matched (fake_spectra reproduced) dashed; references thick and light underneath
    "default": dict(ls="-", lw=2.0), "matched": dict(ls="--", lw=2.0), "reference": dict(ls="-", lw=5.0, alpha=0.55),
    "alternative": dict(ls=":", lw=2.2), "isolated": dict(ls="-.", lw=1.8),
}


def matplotlib_default_config():
    """(font, rc) of the package style: serif fonts, in-ticks on all sides, minor ticks, thick axes."""
    font = {"family": "serif", "weight": "normal", "size": 14}
    tick = lambda ax, extra: {f"{ax}tick.{k}": v for k, v in {
        "color": "black", "direction": "in", "labelsize": 24, "major.pad": 6.0, "major.size": 14.0, "major.width": 1.5,
        "minor.pad": 3.4, "minor.size": 7.0, "minor.visible": True, "minor.width": 1.0, **extra}.items()}
    rc = {"mathtext.fontset": "cm", "text.usetex": False, "figure.frameon": True, "axes.linewidth": 2.0,
          "axes.titlesize": 32, "axes.labelsize": 28, "legend.fontsize": 28, "legend.fancybox": True, "lines.linewidth": 2.5,
          **tick("x", {"alignment": "center", "bottom": True, "top": True, "labelbottom": True, "labeltop": False,
                       "major.bottom": True, "major.top": True, "minor.bottom": True, "minor.top": True}),
          **tick("y", {"alignment": "center_baseline", "left": True, "right": True, "labelleft": True, "labelright": False,
                       "major.left": True, "major.right": True, "minor.left": True, "minor.right": True})}
    return font, rc


def matplotlib_tol_config():
    """``matplotlib_default_config`` with Tol's bright cycle and sizes for multi-panel figures: (font, rc)."""
    font, rc = matplotlib_default_config()
    rc.update({"axes.prop_cycle": cycler(color=TOL_BRIGHT_CYCLE), "image.cmap": "cividis", "axes.titlesize": 22,
               "axes.labelsize": 22, "legend.fontsize": 15, "xtick.labelsize": 18, "ytick.labelsize": 18,
               "xtick.major.size": 9.0, "ytick.major.size": 9.0, "xtick.minor.size": 4.5, "ytick.minor.size": 4.5,
               "lines.linewidth": 2.0, "legend.framealpha": 0.9, "legend.edgecolor": "0.6", "axes.grid": False,
               "figure.dpi": 80, "savefig.dpi": 150})
    return font, rc


def use_style():
    """Apply ``matplotlib_tol_config`` globally (one call at the top of a notebook)."""
    font, rc = matplotlib_tol_config()
    plt.rc("font", **font)
    plt.rcParams.update(rc)


def tol_cmap(name: str = "iridescent", reverse: bool = False):
    """Tol colour map: sequential "iridescent", "YlOrBr"; diverging "sunset", "BuRd", "PRGn"."""
    colors, bad = _TOL_CMAPS[name]
    cm = mpl.colors.LinearSegmentedColormap.from_list(f"tol_{name}" + ("_r" if reverse else ""), colors[::-1] if reverse else colors)
    cm.set_bad(bad)
    return cm


# --------------------------------------------------------------------------- small helpers
def add_panel_letters(axes, x: float = 0.012, y: float = 0.975, size: int = 18, start: str = "a"):
    """(a), (b), ... in the upper-left corner of every axis."""
    for i, ax in enumerate(np.ravel(axes)):
        ax.text(x, y, f"({chr(ord(start) + i)})", transform=ax.transAxes, fontsize=size, fontweight="bold", zorder=20,
                va="top", ha="left", bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.85))


def residual_band(ax, tol: float, label: Optional[str] = None, zero: bool = True):
    """Zero line and a shaded +/- ``tol`` band on a residual panel."""
    if zero:
        ax.axhline(0.0, color="k", lw=1.0)
    ax.axhspan(-tol, tol, color=TOL_BRIGHT["grey"], alpha=0.35, lw=0, label=label)


def ten_power_formatter(y, pos):
    """Tick formatter that labels exact powers of ten only (for symlog axes)."""
    if np.isclose(y, 0.0):
        return "0"
    e = int(np.round(np.log10(abs(y))))
    return rf"${'-' if y < 0 else ''}10^{{{e}}}$" if np.isclose(abs(y), 10.0 ** e) else ""


def logbin(k, y, nb: int = 40):
    """Mean of y in ``nb`` log-spaced k bins (for display; checks use every mode). Returns (k_centres, y_means)."""
    k, y = np.asarray(k), np.asarray(y)
    e = np.geomspace(k[0], k[-1] * 1.0001, nb + 1)
    j = np.digitize(k, e) - 1
    n = np.bincount(j, minlength=nb)[:nb]
    return np.sqrt(e[1:] * e[:-1])[n > 0], (np.bincount(j, weights=y, minlength=nb)[:nb] / np.maximum(n, 1))[n > 0]


# --------------------------------------------------------------------------- power spectra
def hybrid_k_edges(k_fund, k_max, n_lin: int = 12, n_log: int = 18):
    """k bin edges: one mode per bin up to (n_lin + 1/2) k_fund (edges at (j + 1/2) k_fund), then n_log log bins."""
    lin = (np.arange(n_lin + 1) + 0.5) * k_fund
    lin = lin[lin < k_max]
    return np.concatenate([lin[:-1], np.geomspace(lin[-1], k_max, n_log + 1)])


def pk_sigma(P_input, n_modes, n_real: int = 1):
    """Gaussian error of a binned power spectrum, P sqrt(2 / (n_modes n_real)) (n_modes counts +k and -k)."""
    n = np.asarray(n_modes, dtype=float)
    return np.asarray(P_input, dtype=float) * np.sqrt(2.0 / (np.where(n > 0, n, np.nan) * n_real))


def recovery_summary(meas, expect, sigma, label: str = "") -> str:
    """chi^2 / dof, max |pull| and the fraction within 2 sigma of measured against expected values."""
    m, e, s = (np.ravel(np.asarray(x, dtype=float)) for x in (meas, expect, sigma))
    ok = np.isfinite(m) & np.isfinite(e) & np.isfinite(s) & (s > 0)
    pull = (m[ok] - e[ok]) / s[ok]
    return (f"{label}: chi2/dof = {np.mean(pull ** 2):.2f} ({ok.sum()} points), max|pull| = {np.max(np.abs(pull)):.2f}, "
            f"{100 * np.mean(np.abs(pull) < 2):.0f}% within 2 sigma")


def plot_pk_recovery(binned, P_input, *, power=None, n_real: int = 1, label: str = "measured", color=None,
                     k_label: str = r"$k\ [\mathrm{s\,km^{-1}}]$", P_label: str = r"$P_{\rm 1D}(k)\ [\mathrm{km\,s^{-1}}]$",
                     axes=None, figsize=(12, 9)):
    """Binned P(k) with Gaussian 1-sigma errors against the input, and their ratio.

    binned : (P_b, k_b, k_edges, n_modes) of ``OverfluxStatistics1D.bin_p1d`` / ``OverfluxStatistics3D.bin_p3d_kmod``.
    P_input : (n_bins,) the input averaged over the same bins; power : optional callable P(k) drawn as a line.
    Returns (fig, (main, ratio)); pass ``axes`` again to overlay another measurement.
    """
    P_b, k_b, edges, n_b = (np.asarray(x, dtype=float) for x in binned)
    P_in = np.asarray(P_input, dtype=float)
    sig = pk_sigma(P_in, n_b, n_real)
    ok = np.isfinite(P_b) & (n_b > 0) & (P_in > 0)
    if axes is None:
        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True, gridspec_kw={"height_ratios": [3, 1.2]}, layout="constrained")
    ax0, ax1 = axes
    if power is not None:
        kk = np.geomspace(edges[edges > 0][0], edges[-1], 400)
        ax0.plot(kk, np.asarray(power(kk)), color="k", lw=2, label="input")
    eb = ax0.errorbar(k_b[ok], P_b[ok], yerr=sig[ok], fmt="o", ms=6, color=color, capsize=3, label=label)
    ax1.errorbar(k_b[ok], P_b[ok] / P_in[ok], yerr=sig[ok] / P_in[ok], fmt="o", ms=6, color=eb[0].get_color(), capsize=3)
    ax1.axhline(1.0, color="k", ls="--", lw=1.5)
    ax0.set(xscale="log", yscale="log", ylabel=P_label)
    ax1.set(xscale="log", xlabel=k_label, ylabel=r"$\hat P / P_{\rm in}$")
    ax0.legend(frameon=False)
    return ax0.figure, np.array([ax0, ax1])


def plot_orders_superimposed(stats, kind: str = "P", *, k_weighted: bool = False, max_lag: Optional[float] = None,
                             delta_v_list=None, linthresh: Optional[float] = None, axes=None, figsize=(12, 9)):
    """P1D or xi1D of a composite overflux split by order (``OverfluxStatistics1D.*_total_by_order``), all orders on one
    panel; the sum of the orders over the total, and (sum - total) / max |total| below (rounding error only).

    kind : "P" (symlog y, log k) or "xi" (linear; lags 0..max_lag, default half the skewer); k_weighted : k P / pi;
    delta_v_list : line offsets [km/s]; for xi, vertical lines at every separation |dv_i - dv_j| (folded in the box).
    Returns (fig, (main, residual)).
    """
    if kind not in ("P", "xi"):
        raise ValueError(f"kind must be 'P' or 'xi', got {kind!r}.")
    L = stats.N_v * stats.dv
    if kind == "P":
        stats.compute_p1d()
        x, sel = np.asarray(stats.k)[1:], slice(1, None)
        w = x / np.pi if k_weighted else 1.0
        tot, by_order = stats.Pk_total, stats.Pk_total_by_order
    else:
        stats.compute_xi1d()
        lags = np.asarray(stats.lags)
        sel = (lags >= 0) & (lags <= (0.5 * L if max_lag is None else max_lag))
        x, w = lags[sel], 1.0
        tot, by_order = stats.xi_total, stats.xi_total_by_order
    tot = w * np.asarray(tot)[sel]
    by_order = {n: w * np.asarray(v)[sel] for n, v in by_order.items()}
    scale = np.max(np.abs(tot)) if np.any(tot) else 1.0
    total_sum = sum(by_order.values())
    if axes is None:
        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True, gridspec_kw={"height_ratios": [3, 1.2]}, layout="constrained")
    ax0, ax1 = axes
    for n, y in sorted(by_order.items()):
        if np.max(np.abs(y)) > 1e-12 * scale:
            ax0.plot(x, y, color=TOL_BRIGHT_CYCLE[n % len(TOL_BRIGHT_CYCLE)], label=rf"$\mathcal{{O}}({n})$")
    ax0.plot(x, tot, lw=3.0, color="k", label="total", zorder=3)
    ax0.plot(x, total_sum, lw=1.5, ls="--", color=TOL_BRIGHT["red"], label=r"$\sum$ orders", zorder=4)
    ax1.plot(x, (total_sum - tot) / scale, lw=1.5, color=TOL_BRIGHT["red"])
    ax1.axhline(0.0, color="k", lw=1.0, ls=":")
    if kind == "xi" and delta_v_list is not None:
        dv = np.asarray(delta_v_list, dtype=float)
        for d in np.unique(np.round(np.abs(np.mod(dv[:, None] - dv[None, :] + L / 2, L) - L / 2), 6)):
            if d > 0:
                for a in (ax0, ax1):
                    a.axvline(d, color=TOL_BRIGHT["grey"], ls="--", lw=1.2)
    if kind == "P":
        lt = 1e-3 * scale if linthresh is None else linthresh
        ax0.set_yscale("symlog", linthresh=lt)
        ax0.set_xscale("log")
        ax0.set_ylabel(r"$k P_{\rm 1D}/\pi$" if k_weighted else r"$P_{\rm 1D}\ [\mathrm{km\,s^{-1}}]$")
        ax1.set_xlabel(r"$k\ [\mathrm{s\,km^{-1}}]$")
    else:
        ax0.set_ylabel(r"$\xi_{\rm 1D}$")
        ax1.set_xlabel(r"$\Delta v\ [\mathrm{km\,s^{-1}}]$")
    ax1.set_ylabel(r"$\Delta / \max|{\rm tot}|$")
    ax0.legend(frameon=False, ncol=2)
    return ax0.figure, np.array([ax0, ax1])


# --------------------------------------------------------------------------- shared labels, sizes and the pipeline
LABELS = {  # one label (with its unit) per quantity, in every notebook
    "v": r"$v\ [\mathrm{km\,s^{-1}}]$", "v_ref": r"$v\ [\mathrm{km\,s^{-1}}]$ (Ly$\alpha$ frame)", "v_dagger": r"$v_\dagger\ [\mathrm{km\,s^{-1}}]$",
    "u": r"$u\ [\mathrm{km\,s^{-1}}]$ (snapshot frame)", "s": r"path length $s$ [cMpc]", "x": "x [cMpc]", "y": "y [cMpc]", "z": "z [cMpc]",
    "F": r"$F = e^{-\tau}$", "tau": r"$\tau$", "dF": r"$\Delta F$", "n": r"$n\ [\mathrm{cm^{-3}}]$", "T": r"$T$ [K]",
    "v_los": r"$v_\parallel\ [\mathrm{km\,s^{-1}}]$", "k": r"$k\ [\mathrm{s\,km^{-1}}]$", "P1D": r"$P_{\rm 1D}\ [\mathrm{km\,s^{-1}}]$",
    "kP": r"$k P_{\rm 1D}/\pi$", "xi": r"$\xi_{\rm 1D}$", "lag": r"$\Delta v\ [\mathrm{km\,s^{-1}}]$", "delta": r"$\delta_F$",
    "sigma": r"$\Sigma_{\rm gas}\ [\mathrm{M_\odot\,ckpc^{-2}}]$",
}
FIGSIZES = {"wide": (20, 6), "single": (13, 6), "tall": (13, 9), "triple": (22, 6), "grid": (22, 12), "pipeline": (20, 2.4)}
PIPELINE_STEPS = (("lines", "lines and\ncosmology"), ("skewers", "skewers\na(x)"), ("fields", "gas fields\nn, T, v"),
                  ("tau", r"$\tau(v)$ and $F$"), ("stats", r"overflux, P1D, $\xi$"))
PIPELINE_SIDE = (("simulation", "snapshot\n(Block B)", "fields"), ("comparison", "fake_spectra\n(Block C)", "tau"))


def plot_pipeline(highlight=(), *, ax=None, figsize=None, note: Optional[str] = None):
    """The SkewerJAX pipeline as a row of boxes, the steps in ``highlight`` filled (keys of ``PIPELINE_STEPS`` and
    ``PIPELINE_SIDE``: lines, skewers, fields, tau, stats, simulation, comparison). Returns (fig, ax)."""
    hl = {highlight} if isinstance(highlight, str) else set(highlight)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or FIGSIZES["pipeline"], layout="constrained")
    ax.set_axis_off()
    ax.set(xlim=(-0.6, len(PIPELINE_STEPS) - 0.4), ylim=(-1.55, 0.55))
    on, off = TOL_BRIGHT["blue"], "0.55"

    def box(x, y, text, key, w=0.78):
        hit = key in hl
        ax.add_patch(mpl.patches.FancyBboxPatch((x - w / 2, y - 0.32), w, 0.64, boxstyle="round,pad=0.02,rounding_size=0.12",
                                                facecolor=on if hit else "white", edgecolor=on if hit else off, lw=2.2 if hit else 1.3))
        ax.text(x, y, text, ha="center", va="center", fontsize=14, color="white" if hit else "0.35", fontweight="bold" if hit else "normal")

    arrow = dict(arrowstyle="-|>", color=off, lw=1.6, shrinkA=0, shrinkB=0, mutation_scale=16)
    pos = {}
    for i, (key, text) in enumerate(PIPELINE_STEPS):
        box(i, 0.0, text, key)
        pos[key] = i
        if i:
            ax.annotate("", (i - 0.39, 0.0), (i - 0.61, 0.0), arrowprops=arrow)
    for key, text, target in PIPELINE_SIDE:
        x = pos[target]
        box(x, -1.1, text, key, w=0.9)
        ax.annotate("", (x, -0.32), (x, -0.78), arrowprops=arrow)
    if note:
        ax.text(len(PIPELINE_STEPS) - 0.45, -1.1, note, ha="right", va="center", fontsize=13, color="0.35", style="italic")
    return ax.figure, ax


def plot_spectra(ax, v, F, line_ids, *, total: bool = True, mode: str = "default", label_suffix: str = "", **kw):
    """Flux of every line (colour per line, ``MODE_STYLES[mode]``) and, if ``total``, their product (black, thin)."""
    F = np.asarray(F)
    for j, lid in enumerate(line_ids):
        ax.plot(v, F[j], color=ENTITY_COLORS.get(lid), **{**MODE_STYLES[mode], **kw}, label=f"{lid}{label_suffix}")
    if total and len(line_ids) > 1:
        ax.plot(v, F.prod(axis=0), color="k", lw=1.2, label="product" + label_suffix)
    ax.set(xlabel=LABELS["v"], ylabel=LABELS["F"])
    return ax


def plot_skewer_profiles(axs, x, profiles: dict, *, xlabel: str = LABELS["s"], log=("n", "T"), styles=None):
    """One panel per entry of ``profiles`` = {quantity: (N,) or {label: (N,)}} along a skewer, coloured by quantity
    (``ENTITY_COLORS``: density, temperature, velocity, or per label); quantities in ``log`` on a log axis."""
    colour = {"n": ENTITY_COLORS["density"], "T": ENTITY_COLORS["temperature"], "v_los": ENTITY_COLORS["velocity"]}
    for ax, (q, y) in zip(np.ravel(axs), profiles.items()):
        curves = y if isinstance(y, dict) else {None: y}
        for i, (lab, yy) in enumerate(curves.items()):
            sty = (styles or {}).get(lab, MODE_STYLES["default"] if i == 0 else MODE_STYLES["matched"])
            ax.plot(x, np.asarray(yy), color=ENTITY_COLORS.get(lab, colour.get(q, "k")), label=lab, **sty)
        if q in log:
            ax.set_yscale("log")
        ax.set_ylabel(LABELS.get(q, q))
        if len(curves) > 1:
            ax.legend(fontsize=12)
    np.ravel(axs)[-1].set_xlabel(xlabel)
    return axs


def zoom_inset(ax, xlim, ylim=None, bounds=(0.58, 0.08, 0.4, 0.42), **kw):
    """An inset that redraws every line of ``ax`` inside ``xlim`` (and ``ylim``), with the zoomed region marked."""
    ins = ax.inset_axes(bounds)
    for ln in ax.get_lines():
        x, y = ln.get_xdata(), ln.get_ydata()
        ins.plot(x, y, color=ln.get_color(), ls=ln.get_linestyle(), lw=ln.get_linewidth(), alpha=ln.get_alpha())
    ins.set_xlim(*xlim)
    if ylim is not None:
        ins.set_ylim(*ylim)
    ins.tick_params(labelsize=11)
    ax.indicate_inset_zoom(ins, edgecolor="0.3", **kw)
    return ins


# --------------------------------------------------------------------------- the dark ("neon") style of the visualisations
NEON_COLORS = {"HI": "#00E5FF", "HI-LyA": "#00E5FF", "HI-LyB": "#7FFF00", "SiIII": "#FF2BD6", "SiIII-1206": "#FF2BD6",
               "SiII": "#FFEA00", "density": "#39FF14", "temperature": "#FF7F1F", "velocity": "#FFEA00", "total": "#FFFFFF"}
NEON_CYCLE = ["#00E5FF", "#FF2BD6", "#39FF14", "#FFEA00", "#FF7F1F", "#9D6BFF", "#00BFFF", "#FF5470"]


def dark_axes(axs, fig=None):
    """Black background with white ticks, labels and spines on every axis of ``axs`` (and on ``fig``)."""
    for ax in np.ravel(axs):
        ax.set_facecolor("black")
        ax.tick_params(axis="both", which="both", colors="white")
        for t in (ax.xaxis.label, ax.yaxis.label, ax.title):
            t.set_color("white")
        for sp in ax.spines.values():
            sp.set_color("white")
    fig = fig or np.ravel(axs)[0].figure
    fig.set_facecolor("black")
    return axs


def neon_line(ax, x, y, color, *, lw: float = 1.6, glow: float = 7.0, label=None, **kw):
    """A line with a soft glow underneath (two strokes)."""
    ax.plot(x, y, color=color, lw=glow, alpha=0.18, solid_capstyle="round", zorder=2, **kw)
    return ax.plot(x, y, color=color, lw=lw, alpha=0.95, zorder=3, label=label, **kw)[0]


def neon_legend(ax, **kw):
    leg = ax.legend(**{"frameon": True, "facecolor": "black", "edgecolor": "white", "fontsize": 13, **kw})
    for t in leg.get_texts():
        t.set_color("white")
    return leg


def plot_slab(ax, slab: dict, *, quantity: str = "sigma", cmap=None, vmin=None, vmax=None, colorbar: bool = True, label=None,
              smooth: float = 0.0):
    """``simulations.slab_projection`` map (log10) with its colour bar; axes in cMpc. Returns the image.

    smooth : Gaussian sigma [pixels] of a periodic smoothing before the log (it conserves the total of the map). Cell
        centres are binned, so large void cells leave empty pixels; these are drawn in the lowest colour of the map.
    """
    m = np.asarray(slab[quantity], dtype=float)
    if smooth > 0:
        from scipy.ndimage import gaussian_filter
        m = gaussian_filter(m, smooth, mode="wrap")
    lm = np.log10(np.where(m > 0, m, np.nan))
    names = "xyz"
    cmap = plt.get_cmap(cmap or ("magma" if quantity == "sigma" else "inferno"))
    cmap = cmap.with_extremes(bad=cmap(0.0))
    im = ax.imshow(lm.T, origin="lower", extent=slab["extent"], cmap=cmap,
                   vmin=vmin if vmin is not None else np.nanpercentile(lm, 1), vmax=vmax if vmax is not None else np.nanpercentile(lm, 99.8),
                   interpolation="nearest")
    ax.set(xlabel=f"{names[slab['axes'][0] - 1]} [cMpc]", ylabel=f"{names[slab['axes'][1] - 1]} [cMpc]")
    if colorbar:
        cb = ax.figure.colorbar(im, ax=ax, pad=0.01, fraction=0.046)
        cb.set_label(label or ("log10 " + LABELS.get(quantity, quantity)), color="white" if ax.get_facecolor()[0] < 0.5 else "black")
        cb.ax.tick_params(colors="white" if ax.get_facecolor()[0] < 0.5 else "black")
    return im


def plot_tau_buildup(ax_gas, ax_tau, s_edges, s, n, v, tau_bins, *, colors=None, F_total=None, ax_flux=None, line_label=""):
    """How the gas along a path builds up a spectrum: the density along s (top) with every path interval in its own
    colour, the tau of each interval in velocity space (middle, ``absorbers.tau_by_path_bins``: they add up to the
    total) and, optionally, the flux (bottom). Dark style; ``colors`` default ``NEON_CYCLE``; ``ax_gas=None`` skips the
    gas panel (a second line of the same gas)."""
    e = np.asarray(s_edges)
    cols = colors or [NEON_CYCLE[i % len(NEON_CYCLE)] for i in range(len(e) - 1)]
    s, n = np.asarray(s), np.asarray(n)
    if ax_gas is not None:
        ax_gas.plot(s, n, color="0.45", lw=0.8, zorder=1)
    for i, (lo, hi) in enumerate(zip(e[:-1], e[1:])):
        sel = (s >= lo) & (s < hi)
        if ax_gas is not None:
            neon_line(ax_gas, s[sel], n[sel], cols[i])
        neon_line(ax_tau, v, np.asarray(tau_bins[i]), cols[i])
    ax_tau.plot(v, np.asarray(tau_bins).sum(0), color="white", lw=1.0, ls=":", zorder=4, label=r"$\sum$ intervals")
    if ax_gas is not None:
        ax_gas.set(yscale="log", xlabel=LABELS["s"], ylabel=LABELS["n"])
    ax_tau.set(xlabel=LABELS["v"], ylabel=r"$\tau$ " + line_label)
    if ax_flux is not None and F_total is not None:
        neon_line(ax_flux, v, F_total, NEON_COLORS["total"], lw=1.4)
        ax_flux.set(xlabel=LABELS["v"], ylabel=LABELS["F"] + " " + line_label, ylim=(-0.05, 1.08))
    return cols


# --------------------------------------------------------------------------- benchmark figures (records of comparison.benchmark)
BENCH_STYLES = {  # (code, platform, dtype) -> style
    ("fake_spectra", "cpu", None): dict(color="0.25", marker="s", ls="-", label="fake_spectra (C, OpenMP)"),
    ("skewerjax", "cpu", None): dict(color=TOL_BRIGHT["blue"], marker="o", ls="-", label="SkewerJAX CPU, float64"),
    ("skewerjax", "gpu", None): dict(color=TOL_BRIGHT["green"], marker="^", ls="-", label="SkewerJAX GPU, float64"),
    ("skewerjax", "cpu", "float32"): dict(color=TOL_BRIGHT["blue"], marker="o", ls="--", mfc="white", label="SkewerJAX CPU, float32"),
    ("skewerjax", "gpu", "float32"): dict(color=TOL_BRIGHT["green"], marker="^", ls="--", mfc="white", label="SkewerJAX GPU, float32"),
}
BENCH_STAGE_COLORS = {"setup": TOL_BRIGHT["grey"], "io": TOL_BRIGHT["yellow"], "selection": TOL_BRIGHT["green"],
                      "physics": TOL_BRIGHT["purple"], "tau": TOL_BRIGHT["red"], "compile": TOL_BRIGHT["cyan"]}


def bench_key(rec: dict) -> tuple:
    c = rec["config"]
    return (c["code"], c["platform"] if c["code"] == "skewerjax" else "cpu", c.get("dtype") if c["code"] == "skewerjax" else None)


def bench_select(records, **match) -> list:
    """The successful records whose config matches every ``match`` item (sim = the snapshot directory name)."""
    out = []
    for r in records:
        c = dict(r["config"], sim=r["config"]["snap_base"].rstrip("/").split("/")[-1])
        if r.get("status") == "ok" and all(c.get(k) == v for k, v in match.items()):
            out.append(r)
    return out


def bench_stat(rec: dict, what: str = "total"):
    """(median, min, max) over the repeats of the total or of one stage."""
    x = np.array([r["total"] if what == "total" else r["stages"][what] for r in rec["repeats"]])
    return float(np.median(x)), float(x.min()), float(x.max())


def plot_bench_scaling(ax, records, *, x: str = "n_skewers", what: str = "total", **match):
    """Median wall time (min-max bars) against the number of skewers (or threads), one series per code/device/precision."""
    groups = {}
    for r in bench_select(records, **match):
        groups.setdefault(bench_key(r), []).append(r)
    for key, rs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        rs = sorted(rs, key=lambda r: r["config"][x])
        xs = np.array([r["config"][x] for r in rs])
        st = np.array([bench_stat(r, what) for r in rs])
        sty = dict(BENCH_STYLES.get(key, dict(label=str(key))))
        ax.errorbar(xs, st[:, 0], yerr=[st[:, 0] - st[:, 1], st[:, 2] - st[:, 0]], capsize=3, ms=8, lw=2, **sty)
    ax.set(xscale="log", yscale="log", xlabel={"n_skewers": "number of skewers", "threads": "CPU threads"}.get(x, x),
           ylabel=f"wall time ({what}) [s]")
    return ax


def plot_bench_stages(ax, records, *, compile_split: bool = True, fraction: bool = False, **match):
    """Stacked median time per stage, one bar per code/device/precision (SkewerJAX's first tau split into compile).

    fraction : True stacks the share of each stage in the total (%), False the seconds, with the total on each bar.
    Use a linear axis: a stacked bar on a log axis misstates the shares.
    """
    rs = sorted(bench_select(records, **match), key=lambda r: str(bench_key(r)))
    labels = []
    for i, r in enumerate(rs):
        parts = []
        for st in ("setup", "io", "selection", "physics", "tau"):
            t = bench_stat(r, st)[0]
            if st == "tau" and compile_split and r.get("compile_s") is not None and len(r["repeats"]) > 1:
                warm = float(np.median([x["stages"]["tau"] for x in r["repeats"][1:]]))
                parts += [("tau", warm), ("compile", max(r["compile_s"], 0.0))]
            else:
                parts.append((st, t))
        total = sum(t for _, t in parts)
        bottom = 0.0
        for st, t in parts:
            h = 100.0 * t / total if fraction else t
            name = "JIT compile" if st == "compile" else st
            ax.bar(i, h, bottom=bottom, color=BENCH_STAGE_COLORS[st],
                   label=name if name not in ax.get_legend_handles_labels()[1] else None)
            bottom += h
        if not fraction:
            ax.text(i, bottom, f"{total:.1f} s", ha="center", va="bottom", fontsize=12)
        labels.append(BENCH_STYLES.get(bench_key(r), {}).get("label", str(bench_key(r))).replace(", ", "\n"))
    ax.set_xticks(range(len(rs)), labels, fontsize=12)
    ax.set_ylabel("share of the median wall time [%]" if fraction else "median wall time [s]")
    ax.legend(fontsize=12)
    return ax


def plot_bench_memory(ax, records, *, device: bool = False, **match):
    """Peak host RSS (or, with ``device``, the GPU's peak bytes in use) against the number of skewers."""
    groups = {}
    for r in bench_select(records, **match):
        if device and "device_memory" not in r:
            continue
        groups.setdefault(bench_key(r), []).append(r)
    for key, rs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        rs = sorted(rs, key=lambda r: r["config"]["n_skewers"])
        y = [r["device_memory"]["peak_bytes_in_use"] / 1e9 if device else r["peak_rss_GB"] for r in rs]
        ax.plot([r["config"]["n_skewers"] for r in rs], y, ms=8, lw=2, **BENCH_STYLES.get(key, dict(label=str(key))))
    ax.set(xscale="log", xlabel="number of skewers", ylabel="peak GPU memory [GB]" if device else "peak host RSS [GB]")
    return ax


def plot_bench_speedup(ax, records, *, what: str = "total", **match):
    """fake_spectra's median time over each SkewerJAX series' at the same skewer count (> 1: SkewerJAX faster)."""
    fs = {r["config"]["n_skewers"]: bench_stat(r, what)[0] for r in bench_select(records, code="fake_spectra", **match)}
    groups = {}
    for r in bench_select(records, code="skewerjax", **match):
        if r["config"]["n_skewers"] in fs:
            groups.setdefault(bench_key(r), []).append(r)
    for key, rs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        rs = sorted(rs, key=lambda r: r["config"]["n_skewers"])
        n = [r["config"]["n_skewers"] for r in rs]
        ax.plot(n, [fs[k] / bench_stat(r, what)[0] for k, r in zip(n, rs)], ms=8, lw=2, **BENCH_STYLES.get(key, dict(label=str(key))))
    ax.axhline(1.0, color="k", lw=1, ls=":")
    ax.set(xscale="log", yscale="log", xlabel="number of skewers", ylabel=f"t(fake_spectra) / t(SkewerJAX), {what}")
    return ax


def plot_bench_resolution(ax, records, *, n_skewers: int = 16, what: str = "total", sims=("TNG50-4", "TNG50-3", "TNG50-2")):
    """Median wall time against the simulation resolution (gas cells, from SkewerJAX's selection) at fixed skewer count."""
    cells = {}
    for r in bench_select(records, code="skewerjax"):
        sim = r["config"]["snap_base"].rstrip("/").split("/")[-1]
        cells.setdefault(sim, {})[r["config"]["n_skewers"]] = r["repeats"][0].get("n_cells")
    groups = {}
    for sim in sims:
        for r in bench_select(records, sim=sim, n_skewers=n_skewers, threads=32):
            groups.setdefault(bench_key(r), []).append((sims.index(sim), bench_stat(r, what)))
    for key, pts in sorted(groups.items(), key=lambda kv: str(kv[0])):
        pts.sort()
        x = [p[0] for p in pts]
        st = np.array([p[1] for p in pts])
        ax.errorbar(x, st[:, 0], yerr=[st[:, 0] - st[:, 1], st[:, 2] - st[:, 0]], capsize=3, ms=8, lw=2,
                    **BENCH_STYLES.get(key, dict(label=str(key))))
    ax.set_xticks(range(len(sims)), [f"{s}\n{cells.get(s, {}).get(n_skewers) or 0:,} cells selected" for s in sims], fontsize=12)
    ax.set(yscale="log", ylabel=f"wall time ({what}) [s], {n_skewers} skewers")
    return ax


def plot_bench_accuracy(ax, records, *, line: str = "HI-LyA"):
    """Agreement with fake_spectra per snapshot and skewer count: the matched-mode max |dF| (filled) and the |d<F>| of
    SkewerJAX's default mode (open). The matched check runs in float64 whatever the timed device and precision, so
    one record per (snapshot, N) is shown: the CPU float64 one when present."""
    best = {}
    for r in records:
        if r.get("status") != "ok" or not r.get("agreement"):
            continue
        key = (r["config"]["snap_base"].rstrip("/").split("/")[-1], r["config"]["n_skewers"])
        if key not in best or bench_key(r) == ("skewerjax", "cpu", None):
            best[key] = r
    keys = sorted(best, key=lambda k: (-int(k[0][-1]) if k[0][-1].isdigit() else 0, k[1]))
    for i, key in enumerate(keys):
        a = best[key]["agreement"][line]
        ax.plot(i, a["max_dF"], "o", color=BENCH_STYLES[("skewerjax", "cpu", None)]["color"], ms=9,
                label="matched mode: max |ΔF|" if i == 0 else None)
        ax.plot(i, abs(a["meanF_skewerjax_default"] - a["meanF_fake_spectra"]), "o", mfc="white", ms=9,
                color=BENCH_STYLES[("skewerjax", "cpu", None)]["color"], label="default mode: |Δ⟨F⟩|" if i == 0 else None)
    ax.set_xticks(range(len(keys)), [f"{sim}\nN={n}" for sim, n in keys], fontsize=11)
    ax.set(yscale="log", ylabel="difference to fake_spectra")
    return ax
