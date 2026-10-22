"""Shared plotting style and figure helpers.

The style is dark (black background, a soft glow under every line) with Paul Tol's colour-blind-safe "light"
qualitative scheme, which stays luminous on black. ``use_style()`` at the top of a notebook applies it; every colour of
a quantity or line comes from ``ENTITY_COLORS`` and every unit-carrying label from ``LABELS``.
"""
from __future__ import annotations

from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from matplotlib.patheffects import AbstractPathEffect

# --------------------------------------------------------------------------- style
TOL_BRIGHT = {"blue": "#4477AA", "red": "#EE6677", "green": "#228833", "yellow": "#CCBB44",
              "cyan": "#66CCEE", "purple": "#AA3377", "grey": "#BBBBBB"}
TOL_BRIGHT_CYCLE = list(TOL_BRIGHT.values())
# Tol's "light" scheme (https://personal.sron.nl/~pault/, Fig. 4): distinct for colour-blind vision and bright on a
# dark background. Keyed like TOL_BRIGHT (blue, red, green, ... = the closest hue of the light scheme), plus its extras.
PALETTE = {"blue": "#77AADD", "red": "#EE8866", "green": "#44BB99", "yellow": "#EEDD88", "cyan": "#99DDFF",
           "purple": "#FFAABB", "grey": "#DDDDDD", "pear": "#BBCC33", "olive": "#AAAA00"}
PALETTE_CYCLE = [PALETTE[k] for k in ("blue", "red", "green", "yellow", "cyan", "purple", "pear", "olive", "grey")]
FG, BG = "white", "black"           # foreground (text, axes, totals) and background of the dark style
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
# one colour per entity (species, line, quantity, code) and one line style per mode, in every figure
ENTITY_COLORS = {
    "HI": PALETTE["blue"], "HI-LyA": PALETTE["blue"], "HI-LyB": PALETTE["cyan"],
    "SiIII": PALETTE["red"], "SiIII-1206": PALETTE["red"], "SiII": PALETTE["pear"],
    "fake_spectra": PALETTE["grey"], "analytic": PALETTE["grey"], "reference": PALETTE["grey"], "total": FG,
    "density": PALETTE["green"], "temperature": PALETTE["purple"], "velocity": PALETTE["yellow"],
    "voronoi": PALETTE["blue"], "tophat": PALETTE["olive"], "input": FG,
}
MODE_STYLES = {     # SkewerJAX default solid; matched (fake_spectra reproduced) dashed; references thick and light underneath
    "default": dict(ls="-", lw=2.0), "matched": dict(ls="--", lw=2.0), "reference": dict(ls="-", lw=5.0, alpha=0.55),
    "alternative": dict(ls=":", lw=2.2), "isolated": dict(ls="-.", lw=1.8),
}


class NeonGlow(AbstractPathEffect):
    """A soft glow under every stroked line (a wider, translucent copy in the line's own colour).

    Only unfilled paths with more than two vertices glow (data lines and outlines); text, markers, ticks, spines and
    filled patches are drawn unchanged. ``use_style`` installs it for every artist through ``rcParams["path.effects"]``.
    """

    def __init__(self, width: float = 5.5, alpha: float = 0.18):
        super().__init__()
        self.width, self.alpha = width, alpha

    def draw_path(self, renderer, gc, tpath, affine, rgbFace=None):
        if rgbFace is None and len(tpath.vertices) > 2:
            gc0 = renderer.new_gc()
            gc0.copy_properties(gc)
            gc0.set_linewidth(gc.get_linewidth() + self.width)
            gc0.set_alpha(self.alpha * (gc.get_alpha() if gc.get_alpha() is not None else 1.0))
            gc0.set_dashes(0, None)
            renderer.draw_path(gc0, tpath, affine, rgbFace)
            gc0.restore()
        renderer.draw_path(gc, tpath, affine, rgbFace)


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


def matplotlib_dark_config(glow: bool = True):
    """``matplotlib_tol_config`` on a black background, with ``PALETTE``, white text and axes and the ``NeonGlow``
    under every line: (font, rc). The look of the visualisations, for every figure."""
    font, rc = matplotlib_tol_config()
    rc.update({"axes.prop_cycle": cycler(color=PALETTE_CYCLE), "image.cmap": "magma",
               "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "figure.edgecolor": BG,
               "text.color": FG, "axes.labelcolor": FG, "axes.edgecolor": FG, "axes.titlecolor": FG,
               "xtick.color": FG, "ytick.color": FG, "xtick.labelcolor": FG, "ytick.labelcolor": FG,
               "legend.facecolor": BG, "legend.edgecolor": "0.55", "legend.labelcolor": FG, "legend.framealpha": 0.85,
               "grid.color": "0.35", "hatch.color": FG, "patch.edgecolor": FG, "boxplot.flierprops.markeredgecolor": FG,
               "path.effects": [NeonGlow()] if glow else []})
    return font, rc


def use_style(theme: str = "dark"):
    """Apply the package style globally: one call at the top of a notebook or script.

    theme : "dark" (default: ``matplotlib_dark_config``, the style of every notebook) or "light" (the previous
        white-background ``matplotlib_tol_config``; ``ENTITY_COLORS`` are tuned for the dark one).
    """
    if theme not in ("dark", "light"):
        raise ValueError(f"theme must be 'dark' or 'light', got {theme!r}.")
    plt.rcdefaults()
    font, rc = matplotlib_dark_config() if theme == "dark" else matplotlib_tol_config()
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
    fg, bg = plt.rcParams["text.color"], plt.rcParams["axes.facecolor"]
    for i, ax in enumerate(np.ravel(axes)):
        ax.text(x, y, f"({chr(ord(start) + i)})", transform=ax.transAxes, fontsize=size, fontweight="bold", zorder=20,
                va="top", ha="left", color=fg, bbox=dict(boxstyle="round,pad=0.15", facecolor=bg, edgecolor="none", alpha=0.85))


def residual_band(ax, tol: float, label: Optional[str] = None, zero: bool = True):
    """Zero line and a shaded +/- ``tol`` band on a residual panel."""
    if zero:
        ax.axhline(0.0, color=plt.rcParams["text.color"], lw=1.0)
    ax.axhspan(-tol, tol, color=PALETTE["grey"], alpha=0.25, lw=0, label=label)


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
        ax0.plot(kk, np.asarray(power(kk)), color=plt.rcParams["text.color"], lw=2, label="input")
    eb = ax0.errorbar(k_b[ok], P_b[ok], yerr=sig[ok], fmt="o", ms=6, color=color, capsize=3, label=label)
    ax1.errorbar(k_b[ok], P_b[ok] / P_in[ok], yerr=sig[ok] / P_in[ok], fmt="o", ms=6, color=eb[0].get_color(), capsize=3)
    ax1.axhline(1.0, color=plt.rcParams["text.color"], ls="--", lw=1.5)
    ax0.set(xscale="log", yscale="log", ylabel=P_label)
    ax1.set(xscale="log", xlabel=k_label, ylabel=r"$\hat P / P_{\rm in}$")
    ax0.legend(frameon=False)
    return ax0.figure, np.array([ax0, ax1])


ORDER_COLORS = [PALETTE[k] for k in ("grey", "purple", "blue", "green", "yellow", "cyan", "red", "pear")]   # by order n


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
            ax0.plot(x, y, color=ORDER_COLORS[n % len(ORDER_COLORS)], label=rf"$\mathcal{{O}}({n})$")
    fg = plt.rcParams["text.color"]
    ax0.plot(x, tot, lw=3.0, color=fg, label="total", zorder=3)
    ax0.plot(x, total_sum, lw=1.5, ls="--", color=PALETTE["red"], label=r"$\sum$ orders", zorder=4)
    ax1.plot(x, (total_sum - tot) / scale, lw=1.5, color=PALETTE["red"])
    ax1.axhline(0.0, color=fg, lw=1.0, ls=":")
    if kind == "xi" and delta_v_list is not None:
        dv = np.asarray(delta_v_list, dtype=float)
        for d in np.unique(np.round(np.abs(np.mod(dv[:, None] - dv[None, :] + L / 2, L) - L / 2), 6)):
            if d > 0:
                for a in (ax0, ax1):
                    a.axvline(d, color=PALETTE["grey"], ls="--", lw=1.2)
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


# --------------------------------------------------------------------------- shared labels, sizes and the dependency graph
LABELS = {  # one label (with its unit) per quantity, in every notebook
    "v": r"$v\ [\mathrm{km\,s^{-1}}]$", "v_ref": r"$v\ [\mathrm{km\,s^{-1}}]$ (Ly$\alpha$ frame)", "v_dagger": r"$v_\dagger\ [\mathrm{km\,s^{-1}}]$",
    "u": r"$u\ [\mathrm{km\,s^{-1}}]$ (snapshot frame)", "s": r"path length $s$ [cMpc]", "x": "x [cMpc]", "y": "y [cMpc]", "z": "z [cMpc]",
    "F": r"$F = e^{-\tau}$", "tau": r"$\tau$", "dF": r"$\Delta F$", "n": r"$n\ [\mathrm{cm^{-3}}]$", "T": r"$T$ [K]",
    "v_los": r"$v_\parallel\ [\mathrm{km\,s^{-1}}]$", "k": r"$k\ [\mathrm{s\,km^{-1}}]$", "P1D": r"$P_{\rm 1D}\ [\mathrm{km\,s^{-1}}]$",
    "kP": r"$k P_{\rm 1D}/\pi$", "xi": r"$\xi_{\rm 1D}$", "lag": r"$\Delta v\ [\mathrm{km\,s^{-1}}]$", "delta": r"$\delta_F$",
    "sigma": r"$\Sigma_{\rm gas}\ [\mathrm{M_\odot\,ckpc^{-2}}]$",
}
FIGSIZES = {"wide": (20, 6), "single": (13, 6), "tall": (13, 9), "triple": (22, 6), "grid": (22, 12), "dag": (20, 4.6)}

# The dependency graph of the ingredients, derived from the code (reply 20261022_1520, Sec. 2). Columns are the
# topological levels; ingredients in one column are independent of each other.
DAG_NODES = {   # key: (column, row, label)
    "lines": (0, 1.15, "lines\nLineBundle"),
    "samplers": (0, 0.0, "toy samplers\nPowerLawGRF, ramps"),
    "cosmology": (0, -1.15, "cosmology\nCosmology"),
    "skewers": (1, -1.15, "skewers\nSkewerBundle, a(x)"),
    "simulation": (2, -1.15, "simulation outputs\n(TNG snapshots) → gas"),
    "fields": (2, 0.0, "gas fields\nFields: n, T, v"),
    "tau": (3, 0.0, "optical depth\n$\\tau(v)$, $F = e^{-\\tau}$"),
    "overflux": (4, 0.0, "overflux\nδ_F per line, total"),
    "stats": (5, 0.0, "statistics\nP1D, ξ1D, P3D, ξ3D"),
    "fake_spectra": (3, -1.15, "fake_spectra\n(external reference)"),
}
DAG_EDGES = (   # (from, to, kind): "solid" required; "optional" (dashed) an alternative input or a data table
    ("cosmology", "skewers", "optional"), ("skewers", "simulation", "solid"), ("simulation", "fields", "solid"),
    ("samplers", "fields", "solid"), ("lines", "tau", "solid"), ("fields", "tau", "solid"), ("skewers", "tau", "solid"),
    ("tau", "overflux", "solid"), ("lines", "overflux", "optional"), ("overflux", "stats", "solid"),
    ("lines", "simulation", "optional"),
)
DAG_EXTERNAL = (("simulation", "fake_spectra"), ("fake_spectra", "tau"))
DAG_NOTEBOOKS = {"lines": "01", "cosmology": "02", "skewers": "03", "simulation": "04", "samplers": "05", "fields": "05",
                 "tau": "06", "overflux": "07", "stats": "07", "fake_spectra": "08"}


def plot_dag(highlight=(), *, external: bool = False, ax=None, figsize=None, notebooks: bool = True):
    """The dependency graph (DAG) of SkewerJAX's ingredients, ``highlight`` (keys of ``DAG_NODES``) filled and glowing.

    Arrows point from an ingredient to what needs it (dashed: optional, e.g. manual a(x) instead of a cosmology, the
    atomic table for ion masses, the line offsets for the reference frame); ingredients stacked in one column are
    independent. ``external`` adds fake_spectra, the external reference code (comparison notebooks only).
    ``notebooks`` writes the notebook number of every ingredient under its box. Returns (fig, ax).
    """
    hl = {highlight} if isinstance(highlight, str) else set(highlight)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or FIGSIZES["dag"], layout="constrained")
    ax.set_axis_off()
    ax.set(xlim=(-0.62, 5.62), ylim=(-1.8, 1.72))
    fg, dim, on = plt.rcParams["text.color"], "0.5", PALETTE["cyan"]
    w, h = 0.9, 0.62
    nodes = {k: v for k, v in DAG_NODES.items() if external or k != "fake_spectra"}

    def anchor(a, b):          # edge end points on the box borders
        (xa, ya, _), (xb, yb, _) = nodes[a], nodes[b]
        if abs(ya - yb) < 1e-6:
            return (xa + w / 2, ya), (xb - w / 2, yb)
        if abs(xa - xb) < 1e-6:
            return (xa, ya + np.sign(yb - ya) * h / 2), (xb, yb - np.sign(yb - ya) * h / 2)
        return (xa + w / 2, ya + np.sign(yb - ya) * 0.12), (xb - w / 2, yb - np.sign(yb - ya) * 0.12)

    edges = [(a, b, k) for a, b, k in DAG_EDGES] + ([(a, b, "external") for a, b in DAG_EXTERNAL] if external else [])
    for a, b, kind in edges:
        (x0, y0), (x1, y1) = anchor(a, b)
        hot = b in hl                               # the inputs of a highlighted ingredient light up
        rad = 0.0
        if (a, b) == ("skewers", "tau"):            # between the fields and the simulation boxes
            (x0, y0), (x1, y1) = (nodes[a][0] + 0.1, nodes[a][1] + h / 2), (nodes[b][0] - w / 2, nodes[b][1] - 0.2)
        if (a, b) == ("lines", "tau"):              # over the fields box
            (x0, y0), (x1, y1), rad = (nodes[a][0] + w / 2, nodes[a][1] - 0.05), (nodes[b][0] - 0.15, nodes[b][1] + h / 2), -0.12
        if (a, b) == ("lines", "overflux"):
            (x0, y0), (x1, y1), rad = (nodes[a][0] + w / 2, nodes[a][1] + 0.05), (nodes[b][0], nodes[b][1] + h / 2), -0.25
        if (a, b) == ("lines", "simulation"):
            (x0, y0), (x1, y1), rad = (nodes[a][0] + w / 2 - 0.05, nodes[a][1] - h / 2), (nodes[b][0] - w / 2, nodes[b][1] + 0.18), 0.0
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(
            arrowstyle="-|>", color=on if hot else (PALETTE["yellow"] if kind == "external" else dim), mutation_scale=18,
            lw=2.4 if hot else 1.6, ls="--" if kind in ("optional", "external") else "-", shrinkA=2, shrinkB=2,
            connectionstyle=f"arc3,rad={rad}"))
    for key, (x, y, text) in nodes.items():
        hit = key in hl
        ext = key == "fake_spectra"
        ec = on if hit else (PALETTE["yellow"] if ext else dim)
        if hit:
            for k, a_ in ((0.10, 0.10), (0.05, 0.18)):          # glow halo
                ax.add_patch(mpl.patches.FancyBboxPatch((x - w / 2 - k, y - h / 2 - k), w + 2 * k, h + 2 * k,
                             boxstyle="round,pad=0.02,rounding_size=0.14", facecolor=on, edgecolor="none", alpha=a_, path_effects=[]))
        ax.add_patch(mpl.patches.FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                     facecolor=on if hit else plt.rcParams["axes.facecolor"], edgecolor=ec, lw=2.4 if hit else 1.4,
                     ls="--" if ext else "-", path_effects=[]))
        name, sub = text.split("\n")
        tc = BG if hit else (fg if not ext else PALETTE["yellow"])
        ax.text(x, y + 0.11, name, ha="center", va="center", fontsize=14.5, color=tc, fontweight="bold")
        ax.text(x, y - 0.15, sub, ha="center", va="center", fontsize=11.5, color=tc)
        if notebooks and key in DAG_NOTEBOOKS and key != "samplers":
            ax.text(x + w / 2 - 0.02, y - h / 2 - 0.05, f"nb {DAG_NOTEBOOKS[key]}", ha="right", va="top", fontsize=10.5,
                    color=on if hit else dim)
    ax.text(-0.6, 1.66, "independent inputs", ha="left", va="top", fontsize=11, color=dim, style="italic")
    ax.annotate("", (-0.6, -1.65), (-0.6, 1.45), arrowprops=dict(arrowstyle="-", color=dim, lw=1, ls=":"))
    ax.text(5.6, -1.62, "arrow: needed by · dashed: optional / data table · filled: this notebook", ha="right", va="bottom",
            fontsize=11, color=dim, style="italic")
    return ax.figure, ax


def plot_spectra(ax, v, F, line_ids, *, total: bool = True, mode: str = "default", label_suffix: str = "", **kw):
    """Flux of every line (colour per line, ``MODE_STYLES[mode]``) and, if ``total``, their product (black, thin)."""
    F = np.asarray(F)
    for j, lid in enumerate(line_ids):
        ax.plot(v, F[j], color=ENTITY_COLORS.get(lid), **{**MODE_STYLES[mode], **kw}, label=f"{lid}{label_suffix}")
    if total and len(line_ids) > 1:
        ax.plot(v, F.prod(axis=0), color=ENTITY_COLORS["total"], lw=1.2, label="product" + label_suffix)
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
            ax.plot(x, np.asarray(yy), color=ENTITY_COLORS.get(lab, colour.get(q, FG)), label=lab, **sty)
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
    ax.indicate_inset_zoom(ins, edgecolor="0.7", **kw)
    return ins


# --------------------------------------------------------------------------- the dark ("neon") style of the visualisations
NEON_COLORS = ENTITY_COLORS          # the visualisations use the same colour-blind-safe colours as every other figure
NEON_CYCLE = PALETTE_CYCLE


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
    """A line with a soft glow underneath (two strokes; the global ``NeonGlow`` is switched off for them)."""
    ax.plot(x, y, color=color, lw=glow, alpha=0.18, solid_capstyle="round", zorder=2, path_effects=[], **kw)
    return ax.plot(x, y, color=color, lw=lw, alpha=0.95, zorder=3, label=label, path_effects=[], **kw)[0]


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
    ax_tau.plot(v, np.asarray(tau_bins).sum(0), color=FG, lw=1.0, ls=":", zorder=4, label=r"$\sum$ intervals")
    if ax_gas is not None:
        ax_gas.set(yscale="log", xlabel=LABELS["s"], ylabel=LABELS["n"])
    ax_tau.set(xlabel=LABELS["v"], ylabel=r"$\tau$ " + line_label)
    if ax_flux is not None and F_total is not None:
        neon_line(ax_flux, v, F_total, NEON_COLORS["total"], lw=1.4)
        ax_flux.set(xlabel=LABELS["v"], ylabel=LABELS["F"] + " " + line_label, ylim=(-0.05, 1.08))
    return cols


# --------------------------------------------------------------------------- benchmark figures (records of comparison.benchmark)
BENCH_STYLES = {  # (code, platform, dtype) -> style
    ("fake_spectra", "cpu", None): dict(color=PALETTE["grey"], marker="s", ls="-", label="fake_spectra (C, OpenMP)"),
    ("skewerjax", "cpu", None): dict(color=PALETTE["blue"], marker="o", ls="-", label="SkewerJAX CPU, float64"),
    ("skewerjax", "gpu", None): dict(color=PALETTE["green"], marker="^", ls="-", label="SkewerJAX GPU, float64"),
    ("skewerjax", "cpu", "float32"): dict(color=PALETTE["blue"], marker="o", ls="--", mfc=BG, label="SkewerJAX CPU, float32"),
    ("skewerjax", "gpu", "float32"): dict(color=PALETTE["green"], marker="^", ls="--", mfc=BG, label="SkewerJAX GPU, float32"),
}
BENCH_STAGE_COLORS = {"setup": PALETTE["grey"], "io": PALETTE["yellow"], "selection": PALETTE["green"],
                      "physics": PALETTE["purple"], "tau": PALETTE["red"], "compile": PALETTE["cyan"]}


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
    ax.axhline(1.0, color=plt.rcParams["text.color"], lw=1, ls=":")
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
        ax.plot(i, abs(a["meanF_skewerjax_default"] - a["meanF_fake_spectra"]), "o", mfc=BG, ms=9,
                color=BENCH_STYLES[("skewerjax", "cpu", None)]["color"], label="default mode: |Δ⟨F⟩|" if i == 0 else None)
    ax.set_xticks(range(len(keys)), [f"{sim}\nN={n}" for sim, n in keys], fontsize=11)
    ax.set(yscale="log", ylabel="difference to fake_spectra")
    return ax


# --------------------------------------------------------------------------- figures of the tutorial notebooks (01-09)
def species_color(species: str, i: int = 0) -> str:
    """``ENTITY_COLORS`` of a species or line id, else a colour of the cycle."""
    return ENTITY_COLORS.get(species, PALETTE_CYCLE[(i + 3) % len(PALETTE_CYCLE)])


def plot_line_table(line_bundle, *, T_K: float = 1e4, axs=None, figsize=(22, 6.5)):
    """The transitions of a ``LineBundle`` at a glance (notebook 01).

    (a) oscillator strength against rest wavelength (one stick per line, colour per species); (b) the offset
    delta_v_j = c ln(lambda_ref / lambda_j) of every line from the reference line, i.e. where line j's absorption
    lands on the reference frame; (c) the two widths of the Voigt profile in km/s: natural, lambda_0 A_21 / 4 pi, and
    thermal, b = sqrt(2 k_B T / m) at ``T_K``. Returns (fig, axs).
    """
    from skewerjax.core.constants import phys
    lines = line_bundle.lines
    sp = list(dict.fromkeys(l.species for l in lines))
    col = {s_: ENTITY_COLORS[s_] for s_ in sp if s_ in ENTITY_COLORS}
    spare = [c for c in PALETTE_CYCLE if c not in col.values()]
    col.update({s_: spare[i % len(spare)] for i, s_ in enumerate(s_ for s_ in sp if s_ not in col)})
    lam, f, A21, m = (np.asarray(x, dtype=float) for x in (line_bundle.lambda0_A, line_bundle.f_osc, line_bundle.A21_GHz,
                                                          line_bundle.mass_amu))
    dv = np.asarray(line_bundle.delta_v, dtype=float)
    if axs is None:
        fig, axs = plt.subplots(1, 3, figsize=figsize, layout="constrained", gridspec_kw={"width_ratios": [1.3, 1, 1]})
    order = np.argsort(lam)
    for r, j in enumerate(order):
        l = lines[j]
        axs[0].vlines(lam[j], 1e-3, f[j], color=col[l.species], lw=3)
        axs[0].annotate(l.id, (lam[j], f[j]), xytext=(0, 8 + 22 * (r % 3)), textcoords="offset points", rotation=90, ha="center",
                        va="bottom", fontsize=11, color=col[l.species], arrowprops=dict(arrowstyle="-", color=col[l.species], lw=0.6))
    axs[0].set(xscale="log", yscale="log", ylim=(1e-3, 300), xlabel=r"rest wavelength $\lambda_0$ [Å]", ylabel=r"oscillator strength $f$")
    axs[0].xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
    axs[0].xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    axs[0].set_xticks([1000, 1250, 1500, 2000, 2800])
    y = np.arange(len(lines))
    axs[1].barh(y, dv / 1e3, color=[col[l.species] for l in lines], height=0.6)
    axs[1].axvline(0, color=plt.rcParams["text.color"], lw=1)
    axs[1].set_yticks(y, [l.id for l in lines], fontsize=12)
    axs[1].invert_yaxis()
    axs[1].set(xlabel=rf"$\Delta v_j = c\,\ln(\lambda_{{\rm ref}}/\lambda_j)$ [$10^3$ km/s], ref = {line_bundle.ref_line_id}")
    b = np.sqrt(2 * phys.kB * T_K / m) / 1e3                        # phys.kB = k_B / m_u [m^2 s^-2 K^-1] -> [km/s]
    nat = lam * 1e-10 * A21 * 1e9 / (4 * np.pi) / 1e3                  # lambda_0 gamma [km/s]
    axs[2].scatter(nat, y, color=[col[l.species] for l in lines], s=80, marker="o", label=r"natural $\lambda_0 A_{21}/4\pi$")
    axs[2].scatter(b, y, color=[col[l.species] for l in lines], s=110, marker="D", label=rf"thermal $b$ at $T = 10^{{{np.log10(T_K):.0f}}}$ K")
    for j in range(len(lines)):
        axs[2].plot([nat[j], b[j]], [y[j], y[j]], color=col[lines[j].species], lw=1, alpha=0.6)
    axs[2].set_yticks(y, [l.id for l in lines], fontsize=12)
    axs[2].invert_yaxis()
    axs[2].set(xscale="log", xlabel="width [km/s]", xlim=(min(nat.min(), b.min()) / 3, max(nat.max(), b.max()) * 30))
    axs[2].legend(fontsize=12, loc="lower right")
    return axs[0].figure, axs


def box_edges_3d(ax, L, **kw):
    """The 12 edges of the cube [0, L]^3 on a 3D axis."""
    kw = {"color": "0.5", "lw": 0.8, "alpha": 0.7, **kw}
    for a in (0, L):
        for b in (0, L):
            ax.plot([0, L], [a, a], [b, b], **kw)
            ax.plot([a, a], [0, L], [b, b], **kw)
            ax.plot([a, a], [b, b], [0, L], **kw)


def plot_paths_3d(ax, points, *, values=None, cmap="magma", vmin=None, vmax=None, color=None, lw: float = 2.5,
                  box: Optional[float] = None, markers: bool = False):
    """Skewer trajectories (N_sk, N, 3) [cMpc] on a 3D axis, each coloured along its length by ``values`` (N_sk, N)
    (e.g. the scale factor a(x)) or in one ``color``; ``box`` draws the cube edges. Returns the ScalarMappable."""
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    P = np.asarray(points, dtype=float)
    if box is not None:
        box_edges_3d(ax, box)
    sm = None
    if values is not None:
        V = np.asarray(values, dtype=float)
        norm = mpl.colors.Normalize(np.min(V) if vmin is None else vmin, np.max(V) if vmax is None else vmax)
        cm_ = plt.get_cmap(cmap)
        for p, v in zip(P, V):
            seg = np.stack([p[:-1], p[1:]], 1)
            ax.add_collection3d(Line3DCollection(seg, colors=cm_(norm(0.5 * (v[1:] + v[:-1]))), lw=lw))
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cm_)
    else:
        for p in P:
            ax.plot(*p.T, color=color or PALETTE["cyan"], lw=lw)
    if markers:
        for p in P:
            ax.scatter(*p.T, color=FG, s=6, depthshade=False)
    lo, hi = (0.0, box) if box is not None else (P.min(), P.max())
    ax.set(xlim=(lo, hi), ylim=(lo, hi), zlim=(lo, hi), xlabel="x [cMpc]", ylabel="y [cMpc]", zlabel="z [cMpc]")
    dark_3d(ax)
    return sm


def dark_3d(ax):
    """Black panes, white ticks and labels on a 3D axis."""
    ax.set_facecolor(plt.rcParams["axes.facecolor"])
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.set_pane_color((0, 0, 0, 0))
        a.line.set_color("0.6")
        a.label.set_color(plt.rcParams["text.color"])
        a._axinfo["grid"]["color"] = (1, 1, 1, 0.08)
    ax.tick_params(colors=plt.rcParams["text.color"], labelsize=11)


def plot_skewer_anatomy(bundle, k: int = 0, *, ii_cosmology: int = 0, figsize=(22, 10)):
    """What a ``SkewerBundle`` stores, for skewer ``k``, one panel per attribute (notebook 03).

    (a) ``spatial_points`` (x, y, z) against the point index; (b) the path length s (``path_lengths``) and
    ``comoving_positions_from_ICs``; (c) ``line_elements`` w_i (the quadrature weights; they add up to the path
    length); (d) ``scale_factors`` a(s) with the IC point (a0 at x0); (e) ``los_directions`` (photon direction,
    with ``array_direction``); (f) the attributes, their shapes and the events ``quadri_vectors`` = (a, x, y, z).
    Returns (fig, axs).
    """
    pts = np.asarray(bundle.spatial_points[k])
    s = np.asarray(bundle.path_lengths[k])
    d = np.asarray(bundle.comoving_positions_from_ICs[k])
    w = np.asarray(bundle.line_elements[k])
    a = np.asarray(bundle.scale_factors[ii_cosmology, k])
    los = np.asarray(bundle.los_directions[k])
    ic = int(bundle.ICs_defined_at_index[k])
    i = np.arange(len(s))
    fig, axs = plt.subplots(2, 3, figsize=figsize, layout="constrained")
    ax = axs.ravel()
    for c, name, colr in zip(range(3), "xyz", (PALETTE["blue"], PALETTE["red"], PALETTE["green"])):
        ax[0].plot(i, pts[:, c], color=colr, label=name)
    ax[0].set(xlabel="point index i", ylabel="position [cMpc]", title="spatial_points (N_sk, N_ppsk, 3)")
    ax[0].legend(fontsize=12)
    ax[1].plot(i, s, color=PALETTE["cyan"], label="path_lengths s")
    ax[1].plot(i, d, color=PALETTE["yellow"], ls="--", label="comoving_positions_from_ICs")
    ax[1].set(xlabel="point index i", ylabel="[cMpc]", title="distances along the path")
    ax[1].legend(fontsize=12)
    ax[2].step(s, w, where="mid", color=PALETTE["green"])
    ax[2].fill_between(s, 0, w, step="mid", color=PALETTE["green"], alpha=0.25)
    ax[2].set(xlabel=LABELS["s"], ylabel=r"$w_i$ [cMpc]", title=rf"line_elements: $\sum_i w_i$ = {w.sum():.4g} = $s_{{\rm end}}$")
    ax[3].plot(s, a, color=PALETTE["purple"])
    ax[3].plot(s[ic], a[ic], "o", color=FG, ms=10)
    ax[3].annotate(f"ICs: a0 = {float(bundle.a0[k]):.4g} at x0 = {float(bundle.x0[k]):.4g}\n"
                   f"({'absorption' if bool(bundle.is_absorption[k]) else 'emission'} point)",
                   (s[ic], a[ic]), textcoords="offset points", xytext=(15 if ic == 0 else -15, -40),
                   ha="left" if ic == 0 else "right", fontsize=12, arrowprops=dict(arrowstyle="->", color=FG))
    ax[3].set(xlabel=LABELS["s"], ylabel="a", title="scale_factors (N_cosmo, N_sk, N_ppsk)")
    for c, name, colr in zip(range(3), "xyz", (PALETTE["blue"], PALETTE["red"], PALETTE["green"])):
        ax[4].plot(s, los[:, c], color=colr, label=rf"$\hat d_{name}$")
    ax[4].set(xlabel=LABELS["s"], ylabel="component", ylim=(-1.15, 1.15),
              title=f"los_directions; array_direction = {int(bundle.array_direction[k]):+d}")
    ax[4].legend(fontsize=12, ncol=3)
    ax[5].set_axis_off()
    rows = [(n, tuple(np.shape(getattr(bundle, n)))) for n in bundle._leaves]
    txt = "\n".join(f"{n:<28s}{str(sh):>18s}" for n, sh in rows)
    ax[5].text(0.0, 1.0, "SkewerBundle (a JAX pytree)\n\n" + txt, family="monospace", fontsize=12.5, va="top",
               transform=ax[5].transAxes, color=FG)
    ax[5].text(0.0, 0.02, "quadri_vectors[..., 0] = a, [..., 1:4] = (x, y, z):\nthe events every field sampler takes",
               fontsize=12.5, transform=ax[5].transAxes, color=PALETTE["cyan"])
    add_panel_letters(ax[:5])
    return fig, axs


def plot_voronoi_discretisations(seed: int = 3, n_cells: int = 90, every: int = 25, axs=None, figsize=(16, 15)):
    """The three grids of a simulation spectrum on a toy 2D Voronoi mesh (after the pre-refactor notebook
    ``3_fields_from_simulations``, cell 4).

    (a) the cells (nearest mesh-generating point), a curved path coloured by the cell it crosses, and the sampling
    points with the ends of their dual segments; (b) along the path the field is piecewise constant: the segment
    averages over w_i keep the column, the values at the points do not; (c) the spectral grid is independent: every
    segment adds its own profile (schematic: a Gaussian at a position set by its path coordinate).
    Returns (fig, axs, numbers) with the exact column and the column of both discretisations.
    """
    from scipy.spatial import Voronoi
    rng = np.random.default_rng(seed)
    gen = np.vstack([rng.uniform(0, 10, (int(0.6 * n_cells), 2)), rng.normal([5, 5], 0.8, (n_cells - int(0.6 * n_cells), 2))])
    tt = np.linspace(0, 10, 16 * every + 1)
    path = np.stack([tt, 5 + 1.2 * np.sin(tt / 2.2)], 1)
    sp = path[::every]
    mid = 0.5 * (sp[1:] + sp[:-1])
    nearest = lambda P: np.argmin(((P[:, None, :] - gen[None]) ** 2).sum(-1), axis=1)
    dens = 1 / (1 + 4 * np.exp(-((gen - 5) ** 2).sum(1)))
    if axs is None:
        fig, axs = plt.subplots(3, 1, figsize=figsize, gridspec_kw={"height_ratios": [3, 1.5, 1.5]}, layout="constrained")
    vor = Voronoi(np.vstack([gen, [[-50, -50], [-50, 60], [60, -50], [60, 60]]]))
    for r in vor.ridge_vertices:
        if -1 not in r:
            axs[0].plot(*vor.vertices[r].T, color="0.45", lw=1.0, path_effects=[])
    axs[0].plot(*gen.T, ".", color=FG, ms=6, label="mesh-generating points: one cell each")
    lab = nearest(path)
    cyc = np.array(PALETTE_CYCLE[:6])
    axs[0].scatter(*path.T, c=cyc[lab % len(cyc)], s=10, label="path, coloured by the cell it crosses")
    axs[0].plot(*sp.T, "o", mfc=BG, mec=FG, mew=2, ms=11, label="sampling points of the SkewerBundle")
    for m in mid:
        axs[0].plot(*m, "|", color=FG, ms=22, mew=2.5)
    axs[0].plot([], [], "|", color=FG, ms=14, mew=2.5, label="midpoints: ends of the dual segments")
    axs[0].set(xlim=(0, 10), ylim=(2.5, 8.0), aspect="equal", ylabel="y", xlabel="x")
    axs[0].set_title("(a) Cells and sampling points are different grids; each point stands for its dual segment")
    axs[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncols=2, fontsize=13)
    s_path = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
    n_path = dens[lab]
    s_sp = s_path[::every]
    edges = np.concatenate([[0], 0.5 * (s_sp[1:] + s_sp[:-1]), [s_path[-1]]])
    w = np.diff(edges)                        # the dual segments along the path (arc length; a polyline's are its line elements)
    ds = np.diff(s_path)
    n_mid = n_path[:-1]                                      # the cell of each path step
    s_mid = 0.5 * (s_path[1:] + s_path[:-1])
    seg_mean = np.array([np.sum(n_mid * ds * ((s_mid >= edges[i]) & (s_mid < edges[i + 1]))) / (edges[i + 1] - edges[i])
                         for i in range(len(sp))])
    n_pts = dens[nearest(sp)]
    axs[1].bar(s_sp, seg_mean, width=w, alpha=0.35, color=ENTITY_COLORS["density"], edgecolor=ENTITY_COLORS["density"],
               label=r"segment averages over $w_i$")
    axs[1].step(s_path, n_path, where="post", color=FG, lw=2, label="the field (cell values)")
    axs[1].plot(s_sp, n_pts, "D", color=PALETTE["red"], ms=9, mec=BG, label="values at the points")
    exact = float(np.sum(n_mid * ds))
    nums = dict(exact=exact, segments=float(np.sum(seg_mean * w)), points=float(np.sum(n_pts * w)))
    axs[1].set(ylabel="n", xlabel="path length s", ylim=(0, 1.6), xlim=(-0.5, s_path[-1] + 0.5))
    axs[1].set_title(f"(b) Column ∫n ds = {exact:.3f}: segment averages {nums['segments']:.3f}, point values {nums['points']:.3f}")
    axs[1].legend(loc="upper right", ncols=3, fontsize=12)
    vv = np.linspace(-60, 20 * s_path[-1] + 60, 900)
    tot = np.zeros_like(vv)
    for i, (s0, nn) in enumerate(zip(s_sp, seg_mean)):
        prof = nn * w[i] * np.exp(-((vv - 20 * s0) / 9) ** 2)
        tot += prof
        axs[2].plot(vv, prof, lw=2, color=cyc[i % len(cyc)])
    axs[2].plot(vv, tot, lw=1.2, ls=":", color=FG, label="sum: the optical depth")
    axs[2].vlines(np.arange(vv[0], vv[-1], 16), 0, 1.05 * tot.max(), color="0.4", lw=1, ls=":", path_effects=[])
    axs[2].set(xlabel=r"spectral coordinate (velocity)", ylabel=r"d$\tau$")
    axs[2].set_title("(c) The spectral grid (dotted) is independent; every segment adds its own profile")
    axs[2].legend(fontsize=12, loc="upper right")
    return axs[0].figure, axs, nums


def plot_line_frames(v, phi, phi_analytic, line_bundle, v_centre, *, a_dagger: float, a_obs: float = 1.0, band: float = 1e-6,
                     figsize=(22, 13)):
    """Line profiles in their own line-centred frames and on the observed-wavelength axis, against the analytic
    profile (notebook 06; after the pre-refactor ``5_line_profiles`` cells 16-17).

    v (N_v,) line-centred velocity v_dagger [km/s] of every line; phi, phi_analytic (N_lines, N_v) [ns];
    v_centre : the absorber's line-centre velocity v_c = c ln(a_dagger / a) [km/s] (the same in every frame).
    Top row: phi_j(v_dagger), one panel per line, with the rest-frame centre (v = 0, dash-dot), the absorber centre
    (v_c, dashed) and the peak 4 / A_21 of a Lorentzian (x); middle: the relative residual on a symlog axis with the
    +/- ``band`` band; bottom: every line on lambda_obs = lambda_j (a_obs / a_dagger) e^{v / c}. Returns (fig, axes).
    """
    from skewerjax.core.constants import phys
    lines = line_bundle.lines
    n = len(lines)
    lam0 = np.asarray(line_bundle.lambda0_A, dtype=float)
    fig = plt.figure(figsize=figsize, layout="constrained")
    gs = fig.add_gridspec(3, n, height_ratios=[2.2, 1.0, 2.0])
    top = [fig.add_subplot(gs[0, j]) for j in range(n)]
    res = [fig.add_subplot(gs[1, j], sharex=top[j]) for j in range(n)]
    low = fig.add_subplot(gs[2, :])
    cols = [species_color(l.id, j) for j, l in enumerate(lines)]
    for j, l in enumerate(lines):
        c = cols[j]
        top[j].plot(v, phi[j], color=c, lw=2.2, label="SkewerJAX")
        top[j].plot(v, phi_analytic[j], color=ENTITY_COLORS["analytic"], **MODE_STYLES["reference"], label="analytic")
        top[j].axvline(0.0, color=c, ls="-.", lw=1.3)
        top[j].axvline(v_centre, color=c, ls="--", lw=1.3)
        top[j].plot([v_centre], [4.0 / float(line_bundle.A21_GHz[j])], "x", color=FG, ms=12, mew=2.5)
        top[j].set(yscale="log", title=rf"{l.id}, $\lambda_0$ = {l.lambda0_A:.3f} Å")
        r = phi[j] / phi_analytic[j] - 1.0
        res[j].plot(v, r, color=c, lw=1.2)
        residual_band(res[j], band)
        res[j].set_yscale("symlog", linthresh=band)
        res[j].set_yticks([-1e-2, -1e-4, 0.0, 1e-4, 1e-2])
        res[j].yaxis.set_major_formatter(mpl.ticker.FuncFormatter(ten_power_formatter))
        res[j].set(xlabel=r"line-centred $v_\dagger$ [km/s] (own frame)")
        lam = lam0[j] * (a_obs / a_dagger) * np.exp(np.asarray(v) / phys.c_kms)
        lam_c = lam0[j] * (a_obs / a_dagger) * np.exp(v_centre / phys.c_kms)
        low.plot(lam, phi[j], color=c, lw=2.2, label=f"{l.id}")
        low.plot(lam, phi_analytic[j], color=ENTITY_COLORS["analytic"], **{**MODE_STYLES["reference"], "lw": 4})
        low.axvline(lam0[j] * a_obs / a_dagger, color=c, ls="-.", lw=1.2)
        low.axvline(lam_c, color=c, ls="--", lw=1.2)
    top[0].set_ylabel(r"$\phi$ [ns]")
    res[0].set_ylabel(r"$\phi/\phi_{\rm an} - 1$")
    from matplotlib.lines import Line2D
    top[0].legend(handles=[Line2D([], [], color=FG, lw=2, label="SkewerJAX"),
                           Line2D([], [], color=ENTITY_COLORS["analytic"], **MODE_STYLES["reference"], label="analytic"),
                           Line2D([], [], color=FG, ls="-.", lw=1.3, label="rest-frame centre"),
                           Line2D([], [], color=FG, ls="--", lw=1.3, label="absorber centre"),
                           Line2D([], [], color=FG, marker="x", ls="none", ms=10, mew=2, label=r"$4/A_{21}$")], fontsize=11)
    low.set(yscale="log", xlabel=r"observed wavelength $\lambda_{\rm obs}$ [Å]", ylabel=r"$\phi$ [ns]")
    low.legend(fontsize=12, ncol=n, loc="upper center")
    add_panel_letters(top + res + [low], size=15)
    return fig, (top, res, low)


def plot_flux_vs_analytic(v, F, F_analytic, line_bundle, lam_obs, lam_centre, *, zoom_kms: float = 80.0,
                          skewer_labels=None, band: float = 1e-2, figsize=(22, 13)):
    """F against the closed form on the observed-wavelength axis, and a zoom on every line centre with its
    residual (notebook 06; after the pre-refactor ``6_optical_depth`` cells 31-32).

    v (N_v,) line-centred v_dagger [km/s]; F, F_analytic (N_lines, N_sk, N_v); lam_obs (N_lines, N_sk, N_v) the observed
    wavelength of every pixel; lam_centre (N_lines, N_sk) where the absorption is centred (dashed) on that axis. The rest
    wavelength lambda_0 is dash-dotted. Colours: one per skewer; solid SkewerJAX, thick light the analytic F.
    Returns (fig, (main, zooms, residuals)).
    """
    lines = line_bundle.lines
    nl, ns = np.shape(F)[:2]
    cols = [PALETTE_CYCLE[i % len(PALETTE_CYCLE)] for i in range(ns)]
    fig = plt.figure(figsize=figsize, layout="constrained")
    gs = fig.add_gridspec(3, nl, height_ratios=[2.0, 1.6, 0.9])
    main = fig.add_subplot(gs[0, :])
    zoom = [fig.add_subplot(gs[1, j]) for j in range(nl)]
    resid = [fig.add_subplot(gs[2, j], sharex=zoom[j]) for j in range(nl)]
    v = np.asarray(v)
    for j, l in enumerate(lines):
        main.axvline(l.lambda0_A, color="0.6", ls="-.", lw=1.3)
        sel = np.abs(v) <= zoom_kms
        for k in range(ns):
            lab = (skewer_labels[k] if skewer_labels is not None else f"skewer {k}") if j == 0 else None
            main.plot(lam_obs[j, k], F_analytic[j, k], color=cols[k], **MODE_STYLES["reference"])
            main.plot(lam_obs[j, k], F[j, k], color=cols[k], lw=1.8, label=lab)
            main.axvline(lam_centre[j, k], color=cols[k], ls="--", lw=1.2)
            zoom[j].plot(v[sel], F_analytic[j, k, sel], color=cols[k], **MODE_STYLES["reference"])
            zoom[j].plot(v[sel], F[j, k, sel], color=cols[k], lw=1.8)
            resid[j].plot(v[sel], F[j, k, sel] - F_analytic[j, k, sel], color=cols[k], lw=1.5)
        zoom[j].axvline(0.0, color="0.6", ls="-.", lw=1.2)
        zoom[j].set_title(rf"{l.id}, $\lambda_0$ = {l.lambda0_A:.3f} Å", fontsize=17)
        residual_band(resid[j], band)
        resid[j].set(xlabel=r"line-centred $v_\dagger$ [km/s]")
    main.set(xlabel=r"observed wavelength $\lambda_{\rm obs}$ [Å]", ylabel=r"$F/F_0$")
    main.legend(fontsize=12, loc="lower left")
    zoom[0].set_ylabel(r"$F/F_0$")
    resid[0].set_ylabel(r"$F - F_{\rm an}$")
    add_panel_letters([main] + zoom + resid, size=15)
    return fig, (main, zoom, resid)


def plot_raw_to_overflux(v, tau_own, tau_ref, line_ids, *, skewer: int = 0, period: Optional[float] = None, figsize=(22, 15)):
    """From the integrator's raw output to the overflux, one panel per step (notebook 07).

    v (N_v,) [km/s]; tau_own (N_lines, N_sk, N_v): every line in its own line-centred frame (the output of
    ``compute_optical_depth``); tau_ref: the same on the reference frame (``wrap_to_ref_line``).
    (a) F_j = e^{-tau_j} in each own frame; (b) on the reference frame, every line moved by -delta_v_j modulo the
    period (the box tiled three times when ``period`` is given); (c) the total F = prod_j F_j; (d) the overflux
    delta_F = F / <F> - 1 of every line and of the total, <F> over all skewers and pixels. Returns (fig, axs).
    """
    v = np.asarray(v)
    Fo, Fr = np.exp(-np.asarray(tau_own)), np.exp(-np.asarray(tau_ref))
    F_tot = Fr.prod(0)
    mF, mT = Fr.mean(axis=(1, 2)), F_tot.mean()
    cols = [species_color(l, j) for j, l in enumerate(line_ids)]
    fig, axs = plt.subplots(4, 1, figsize=figsize, sharex=False, layout="constrained")
    for j, l in enumerate(line_ids):
        axs[0].plot(v, Fo[j, skewer], color=cols[j], label=f"{l} (own frame)")
    axs[0].set(ylabel=LABELS["F"], xlabel=LABELS["v_dagger"] + " — each line in its own frame", xlim=(v[0], v[-1]))
    tiles = (-1, 0, 1) if period is not None else (0,)
    for n in tiles:
        for j, l in enumerate(line_ids):
            axs[1].plot(v + n * (period or 0), Fr[j, skewer], color=cols[j], alpha=1.0 if n == 0 else 0.35,
                        label=f"{l} on the {line_ids[0]} frame" if n == 0 else None)
        if period is not None:
            axs[1].axvline(v[0] - 0.5 * (v[1] - v[0]) + n * period, color="0.6", ls=":", lw=1.2)
    axs[1].set(ylabel=LABELS["F"], xlabel=LABELS["v_ref"] + (" — three periods of the box" if period else ""),
               xlim=((v[0] - period, v[-1] + period) if period else (v[0], v[-1])))
    for j, l in enumerate(line_ids):
        axs[2].plot(v, Fr[j, skewer], color=cols[j], lw=1.0, alpha=0.5)
    axs[2].plot(v, F_tot[skewer], color=ENTITY_COLORS["total"], lw=1.8, label=r"$F_{\rm tot} = \prod_j F_j$")
    axs[2].set(ylabel=LABELS["F"], xlabel=LABELS["v_ref"], xlim=(v[0], v[-1]))
    for j, l in enumerate(line_ids):
        axs[3].plot(v, Fr[j, skewer] / mF[j] - 1, color=cols[j], lw=1.2, label=rf"$\delta_F$ {l}, $\langle F\rangle$ = {mF[j]:.4f}")
    axs[3].plot(v, F_tot[skewer] / mT - 1, color=ENTITY_COLORS["total"], lw=1.6, label=rf"$\delta_F$ total, $\langle F\rangle$ = {mT:.4f}")
    axs[3].set(ylabel=LABELS["delta"], xlabel=LABELS["v_ref"], xlim=(v[0], v[-1]))
    titles = ["(a) the integrator's output: one F per line, each in its own line-centred frame",
              "(b) wrap_to_ref_line: every line moved by −Δv_j onto the reference frame, modulo the period",
              "(c) the total flux: the product of the lines",
              r"(d) the overflux $\delta_F = F/\langle F\rangle - 1$ ($\langle F\rangle$ over every skewer and pixel)"]
    for ax, t in zip(axs, titles):
        ax.set_title(t, fontsize=17, loc="left")
        ax.legend(fontsize=11.5, loc="lower right", ncol=len(line_ids) + 1)
    return fig, axs


def _pair_label(pair):
    from skewerjax.core.stats.two_point import ZERO_ORDER_KEY
    a, b = (r"$\mathcal{C}$" if t == ZERO_ORDER_KEY else t for t in pair)
    return f"{a}–{b}"


def plot_terms_by_order(stats, kind: str = "P", *, orders=None, max_lag: Optional[float] = None, linthresh: Optional[float] = None,
                        figsize=None):
    """P1D or xi1D of the total overflux split by order and, within each order, into its terms (notebook 07).

    One panel per order n (default every order with a non-zero contribution, the order-0 normalisation term
    included); the terms (A, B) and (B, A) of ``stats.*_total_from_terms_contributions`` are drawn as one curve "A–B"
    (their sum), the order total ``*_total_by_order[n]`` thick and white. P: symlog y against log k; xi: linear, lags
    0..max_lag. Returns (fig, axs).
    """
    if kind not in ("P", "xi"):
        raise ValueError(f"kind must be 'P' or 'xi', got {kind!r}.")
    if kind == "P":
        stats.compute_p1d()
        x, sel = np.asarray(stats.k)[1:], slice(1, None)
        contrib, by_order = stats.Pk_total_from_terms_contributions, stats.Pk_total_by_order
    else:
        stats.compute_xi1d()
        lags = np.asarray(stats.lags)
        L = stats.N_v * stats.dv
        sel = (lags >= 0) & (lags <= (0.5 * L if max_lag is None else max_lag))
        x = lags[sel]
        contrib, by_order = stats.xi_total_from_terms_contributions, stats.xi_total_by_order
    groups = {}
    for (a, b), y in contrib.items():
        key = tuple(sorted((a, b), key=lambda t: (stats.term_keys.index(t))))
        groups.setdefault(stats.pair_orders[(a, b)], {}).setdefault(key, 0.0)
        groups[stats.pair_orders[(a, b)]][key] = groups[stats.pair_orders[(a, b)]][key] + np.asarray(y)[sel]
    scale = max(np.max(np.abs(np.asarray(v)[sel])) for v in by_order.values())
    if orders is None:
        orders = [n for n in sorted(by_order) if np.max(np.abs(np.asarray(by_order[n])[sel])) > 1e-12 * scale]
    fig, axs = plt.subplots(1, len(orders), figsize=figsize or (6.5 * len(orders), 6.5), layout="constrained", sharey=(kind == "P"))
    axs = np.atleast_1d(axs)
    for ax, n in zip(axs, orders):
        for i, (key, y) in enumerate(groups.get(n, {}).items()):
            ax.plot(x, y, color=PALETTE_CYCLE[i % len(PALETTE_CYCLE)], lw=1.6, label=_pair_label(key))
        ax.plot(x, np.asarray(by_order[n])[sel], color=FG, lw=3.0, alpha=0.8, label=rf"$\mathcal{{O}}({n})$ total", zorder=1)
        ax.set_title(rf"order {n}: $|A| + |B| = {n}$", fontsize=18)
        if kind == "P":
            ax.set_xscale("log")
            ax.set_yscale("symlog", linthresh=1e-3 * scale if linthresh is None else linthresh)
            ax.set_xlabel(LABELS["k"])
        else:
            ax.set_xlabel(LABELS["lag"])
        ax.axhline(0, color="0.5", lw=0.8)
        ax.legend(fontsize=11, ncol=2)
    axs[0].set_ylabel(LABELS["P1D"] if kind == "P" else LABELS["xi"])
    add_panel_letters(axs)
    return fig, axs


def grf_p1d_along_axis(grf, k_par, axis: int = 1, return_sigma: bool = False):
    """Expected 1D power spectrum of a 3D ``PowerLawGRF`` along grid lines parallel to ``axis``, sampled at the grid nodes:
    P1D(k_par) = (1 / L_perp^2) sum_{k_perp} P3D(sqrt(k_par^2 + k_perp^2)) over the transverse modes of the grid
    (the discrete projection; same normalisation as ``OverfluxStatistics1D``). k_par [1/length]; returns [length].

    return_sigma : also return the standard deviation of the P1D averaged over *all* the grid lines of one realisation,
        sqrt(sum_{k_perp} P3D^2) / L_perp^2: every 3D mode is an independent complex Gaussian (|delta_k|^2 exponential), so
        a steep spectrum, dominated by a few low k_perp modes, scatters more than P sqrt(2 / n_modes).
    """
    tr = [i for i in range(3) if i != axis]
    ka, kb = (2 * np.pi * np.fft.fftfreq(grf.shape[i], d=grf.dx[i]) for i in tr)
    kperp2 = (ka[:, None] ** 2 + kb[None, :] ** 2).ravel()
    kp = np.asarray(k_par, dtype=float)
    P = np.asarray(grf.power(np.sqrt(kp[:, None] ** 2 + kperp2[None, :])))
    area = grf.L[tr[0]] * grf.L[tr[1]]
    if return_sigma:
        return P.sum(-1) / area, np.sqrt((P ** 2).sum(-1)) / area
    return P.sum(-1) / area


def plot_p1d_input_vs_measured(k, curves: dict, reference, *, ref_label: str = "input", axs=None, k_range=None, figsize=(15, 10)):
    """Measured P1D curves against an input spectrum, with their ratios (notebook 07).

    curves : {label: (P (N_k,), sigma (N_k,) or None, colour, style dict)}; reference (N_k,): the expected spectrum on
    the same k. Top: k P / pi on log axes; bottom: P / reference. Returns (fig, axs)."""
    k = np.asarray(k)
    if axs is None:
        fig, axs = plt.subplots(2, 1, figsize=figsize, sharex=True, gridspec_kw={"height_ratios": [2.2, 1.3]}, layout="constrained")
    ref = np.asarray(reference)
    axs[0].plot(k, k * ref / np.pi, color=FG, lw=4, alpha=0.55, label=ref_label)
    for lab, (P, sig, c, sty) in curves.items():
        P = np.asarray(P)
        axs[0].plot(k, k * P / np.pi, color=c, label=lab, **sty)
        r = P / ref
        axs[1].plot(k, r, color=c, **sty)
        if sig is not None:
            axs[1].fill_between(k, r - np.asarray(sig) / ref, r + np.asarray(sig) / ref, color=c, alpha=0.2, lw=0)
    axs[1].axhline(1.0, color=FG, ls="--", lw=1.2)
    axs[0].set(xscale="log", yscale="log", ylabel=LABELS["kP"])
    axs[1].set(xscale="log", yscale="log", xlabel=LABELS["k"], ylabel=f"P / {ref_label}")
    if k_range is not None:
        axs[1].set_xlim(*k_range)
    axs[0].legend(fontsize=12, ncol=2)
    return axs[0].figure, axs


def plot_cylinder(s, profiles: dict, *, cloud=None, flux=None, progress: float = 1.0, radius: float = 1.0, view=(14.0, -62.0),
                  cloud_cmap: str = "magma", cloud_clim=None, n_rings: int = 5, fig=None, figsize=(22, 13)):
    """The gas around a skewer wrapped in a cylinder, and the flux it has absorbed so far (notebook 09; after the
    radar-cylinder plot of the pre-refactor visualisations, ``plot_3d_radar_cylinder``).

    s (N,) path coordinate along the skewer [cMpc] (the cylinder axis).
    profiles : {label: (y (N,), colour)} quantities along the skewer (e.g. the ion densities, log-scaled for display),
        each drawn as a glowing radar line on the cylinder surface at its own angle (radius ∝ the scaled value).
    cloud : optional dict(axis (M,), s_perp (M,), t_perp (M,), value (M,)) of the gas around the skewer (e.g. the
        selected cells: along-axis coordinate, transverse offsets [cMpc], density), drawn as a soft point cloud.
    flux : optional dict(v (N_v,), s_of_v (N_v,), edges (n_bins + 1,), tau_bins {label: ((n_bins, N_v), colour)}): the
        tau of every stretch of the path (``tau_by_path_bins``); the absorbed flux is drawn on the cylinder surface
        (radius ∝ 1 - F) at the position s(v) of every pixel, and in a panel below against v.
    progress : 0..1, the fraction of the path already crossed by the light: gas beyond s = s_0 + progress (s_N - s_0)
        is dimmed and only the stretches before it absorb (one call per frame drives an animation).
    Returns (fig, (ax3d, ax_flux)).
    """
    s = np.asarray(s, dtype=float)
    s0, s1 = s[0], s[-1]
    s_p = s0 + float(np.clip(progress, 0.0, 1.0)) * (s1 - s0)
    if fig is None:
        fig = plt.figure(figsize=figsize)
    ax = fig.add_axes((-0.12, 0.26, 1.24, 0.80), projection="3d", computed_zorder=False)
    axf = fig.add_axes((0.07, 0.06, 0.88, 0.25)) if flux is not None else None
    ax.set_facecolor(BG)
    phi = np.linspace(0, 2 * np.pi, 180)
    for y in np.linspace(s0, s1, n_rings):
        for rr in radius * np.array([0.5, 1.0]):
            ax.plot(np.full_like(phi, y), rr * np.cos(phi), rr * np.sin(phi), color=FG, lw=0.7, alpha=0.18, path_effects=[])
    ax.plot([s0, s1], [0, 0], [0, 0], color=FG, lw=1.2, alpha=0.6, path_effects=[])
    ax.plot(np.full_like(phi, s_p), 1.08 * radius * np.cos(phi), 1.08 * radius * np.sin(phi), color=PALETTE["yellow"], lw=2.0,
            alpha=0.95, label="light front")                    # where the light is at this progress
    if cloud is not None:
        val = np.asarray(cloud["value"], dtype=float)
        lv = np.log10(np.maximum(val, 1e-300))
        lo, hi = cloud_clim if cloud_clim is not None else np.percentile(lv[np.isfinite(lv)], (30, 99.5))
        sc = np.clip((lv - lo) / max(hi - lo, 1e-9), 0, 1)
        ax_c = np.asarray(cloud["axis"])
        keep = sc > 0.02
        rgba = plt.get_cmap(cloud_cmap)(sc[keep])
        rgba[:, 3] = np.where(ax_c[keep] <= s_p, 0.15 + 0.6 * sc[keep] ** 1.2, 0.04)
        for size in (90.0, 18.0):
            ax.scatter(ax_c[keep], np.asarray(cloud["s_perp"])[keep], np.asarray(cloud["t_perp"])[keep], color=rgba, s=size,
                       lw=0, depthshade=False, zorder=1)
            rgba = rgba.copy()
            rgba[:, 3] = np.minimum(1.0, rgba[:, 3] * 1.8)
    n_prof = len(profiles) + (len(flux["tau_bins"]) if flux is not None else 0)
    angles = np.pi / 2 + 2 * np.pi * np.arange(max(n_prof, 1)) / max(n_prof, 1)
    for (lab, (y, c)), th in zip(profiles.items(), angles):
        y = np.asarray(y, dtype=float)
        r = radius * (0.15 + 0.85 * (y - np.nanmin(y)) / max(np.nanmax(y) - np.nanmin(y), 1e-30))
        on = s <= s_p
        for msk, alpha in ((on, 1.0), (~on, 0.15)):
            yy = np.where(msk, r, np.nan)
            ax.plot(s, yy * np.cos(th), yy * np.sin(th), color=c, lw=1.4, alpha=alpha, zorder=5, label=lab if alpha == 1.0 else None)
    if flux is not None:
        v, s_of_v, e = (np.asarray(flux[k], dtype=float) for k in ("v", "s_of_v", "edges"))
        frac = np.clip((s_p - e[:-1]) / np.maximum(np.diff(e), 1e-30), 0.0, 1.0)      # how much of each stretch is crossed
        for (lab, (tb, c)), th in zip(flux["tau_bins"].items(), angles[len(profiles):]):
            tb = np.asarray(tb, dtype=float)
            F_now, F_end = np.exp(-(frac[:, None] * tb).sum(0)), np.exp(-tb.sum(0))
            r = radius * (1.0 - F_now)
            ax.plot(s_of_v, r * np.cos(th), r * np.sin(th), color=c, lw=1.4, zorder=6, label=f"1 − F, {lab}")
            axf.plot(v, F_end, color=c, lw=1.0, alpha=0.3, path_effects=[])
            axf.plot(v, F_now, color=c, lw=1.6, label=f"{lab}: absorbed up to s = {s_p:.1f} cMpc")
        axf.set(xlim=(v[0], v[-1]), ylim=(-0.05, 1.1), xlabel=LABELS["v_dagger"] + ", each line in its own frame", ylabel=LABELS["F"])
        axf.legend(fontsize=12, loc="lower left", ncol=2)
    ax.set(xlim=(s0, s1), ylim=(-1.25 * radius, 1.25 * radius), zlim=(-1.25 * radius, 1.25 * radius))
    ax.set_box_aspect((3.2, 1.0, 1.0), zoom=1.25)
    ax.set_xlabel(LABELS["s"], labelpad=14)
    ax.set_yticks([]); ax.set_zticks([])
    dark_3d(ax)
    ax.view_init(*view)
    h, lab = ax.get_legend_handles_labels()
    fig.legend(h, lab, loc="upper left", bbox_to_anchor=(0.01, 0.93), fontsize=13, frameon=True, ncol=1)
    return fig, (ax, axf)
