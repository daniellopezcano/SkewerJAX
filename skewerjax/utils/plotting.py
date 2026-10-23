"""Shared plotting style and figure helpers.

Two styles, switched with one call at the top of a notebook or script: ``use_style("paper")`` (the default: white
background, black text and axes, publication look) and ``use_style("dark")`` (black background, white text, a soft glow
under every line). Both use Paul Tol's colour-blind-safe "bright" qualitative scheme (package ``tol_colors``,
``colorsets["bright"]``; vendored below with identical values when the package is not installed) in the same order, and
Tol's sequential / diverging colour maps: only the background, the foreground (``FG``, ``BG``) and the direction of the
sequential maps (low values merge with the background) change. ``use_style`` also sets minimum font sizes for every kind
of text (``FONT_MIN``); ``use_backend()`` picks the interactive backend (ipympl) in a live notebook and inline otherwise.
Every colour of a quantity or line comes from ``ENTITY_COLORS`` (a fixed assignment, the same in every notebook) and
every unit-carrying label from ``LABELS``. Import ``FG``, ``BG`` after ``use_style`` (they follow the style).
"""
from __future__ import annotations

import os
from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from matplotlib.patheffects import AbstractPathEffect

# --------------------------------------------------------------------------- style
# Tol's "bright" scheme (https://personal.sron.nl/~pault/, Fig. 1; tol_colors 2.2.0 ``colorsets["bright"]``), in Tol's
# order: the only discrete colours of the package's figures.
_TOL_BRIGHT_VENDORED = {"blue": "#4477AA", "red": "#EE6677", "green": "#228833", "yellow": "#CCBB44",
                        "cyan": "#66CCEE", "purple": "#AA3377", "grey": "#BBBBBB"}
try:                                   # the package, when installed, is the source of the values
    import tol_colors as _tc
    TOL_BRIGHT = {k: v.upper() for k, v in _tc.colorsets["bright"]._asdict().items()}
except Exception:                      # ImportError, or an API change: the vendored copy (same values)
    TOL_BRIGHT = dict(_TOL_BRIGHT_VENDORED)
TOL_BRIGHT_CYCLE = list(TOL_BRIGHT.values())
PALETTE = TOL_BRIGHT                   # keyed by hue: blue, red, green, yellow, cyan, purple, grey
PALETTE_CYCLE = TOL_BRIGHT_CYCLE       # fixed order of the colour cycle
THEMES = {"paper": dict(FG="black", BG="white", DIM="#8C8C8C"), "dark": dict(FG="white", BG="black", DIM=PALETTE["grey"])}
FG, BG = THEMES["paper"]["FG"], THEMES["paper"]["BG"]   # foreground (text, axes, totals) and background; set by use_style
THEME = "paper"
DIM = THEMES["paper"]["DIM"]           # secondary elements (guide lines, inactive boxes); a darker grey on white
_TOL_CMAPS = {      # Tol's sequential and diverging schemes (https://personal.sron.nl/~pault/, Figs. 12-20), "bad" colour
    "iridescent": (["#FEFBE9", "#FCF7D5", "#F5F3C1", "#EAF0B5", "#DDECBF", "#D0E7CA", "#C2E3D2", "#B5DDD8", "#A8D8DC",
                    "#9BD2E1", "#8DCBE4", "#81C4E7", "#7BBCE7", "#7EB2E4", "#88A5DD", "#9398D2", "#9B8AC4", "#9D7DB2",
                    "#9A709E", "#906388", "#805770", "#684957", "#46353A"], "#999999"),
    "incandescent": (["#CEFFFF", "#C6F7D6", "#A2F49B", "#BBE453", "#D5CE04", "#E7B503", "#F19903", "#F6790B", "#F94902",
                      "#E40515", "#A80003"], "#888888"),
    "YlOrBr": (["#FFFFE5", "#FFF7BC", "#FEE391", "#FEC44F", "#FB9A29", "#EC7014", "#CC4C02", "#993404", "#662506"], "#888888"),
    "sunset": (["#364B9A", "#4A7BB7", "#6EA6CD", "#98CAE1", "#C2E4EF", "#EAECCC", "#FEDA8B", "#FDB366", "#F67E4B",
                "#DD3D2D", "#A50026"], "#FFFFFF"),
    "nightfall": (["#125A56", "#00767B", "#238F9D", "#42A7C6", "#60BCE9", "#9DCCEF", "#C6DBED", "#DEE6E7", "#ECEADA",
                   "#F0E6B2", "#F9D576", "#FFB954", "#FD9A44", "#F57634", "#E94C1F", "#D11807", "#A01813"], "#FFFFFF"),
    "BuRd": (["#2166AC", "#4393C3", "#92C5DE", "#D1E5F0", "#F7F7F7", "#FDDBC7", "#F4A582", "#D6604D", "#B2182B"], "#FFEE99"),
    "PRGn": (["#762A83", "#9970AB", "#C2A5CF", "#E7D4E8", "#F7F7F7", "#D9F0D3", "#ACD39E", "#5AAE61", "#1B7837"], "#FFEE99"),
}
_TOL_CMAPS["iridescentline"] = (_TOL_CMAPS["iridescent"][0][7:-1], "#999999")          # visible on white
_TOL_CMAPS["iridescentline_dark"] = (_TOL_CMAPS["iridescent"][0][1:-7], "#999999")     # visible on black (reversed)
# Sequential maps run from the background colour to the strongest one (low values merge with the background): light to
# dark in the paper style, dark to bright in the dark style (``use_style`` switches them); diverging maps are the same.
# "line": the sequential map for coloured lines and markers, without the end that would vanish on the background.
_CMAPS_THEME = {"paper": {"sequential": "tol_iridescent", "hot": "tol_incandescent", "line": "tol_iridescentline"},
                "dark": {"sequential": "tol_iridescent_r", "hot": "tol_incandescent_r", "line": "tol_iridescentline_dark_r"}}
CMAPS = {**_CMAPS_THEME["paper"], "diverging": "tol_sunset", "diverging_alt": "tol_BuRd"}
# Fixed colour of every entity (species and lines, quantities, codes, modes), the same in every notebook. Lines take
# the bright colours in Tol's order (Lya blue, Si III red, ...); field quantities take the remaining ones.
ENTITY_COLORS = {
    "HI": PALETTE["blue"], "HI-LyA": PALETTE["blue"], "SiIII": PALETTE["red"], "SiIII-1206": PALETTE["red"],
    "SiII": PALETTE["green"], "HI-LyB": PALETTE["cyan"], "CIV": PALETTE["yellow"], "SiIV": PALETTE["purple"],
    "fake_spectra": PALETTE["grey"], "analytic": PALETTE["grey"], "reference": PALETTE["grey"], "total": FG, "input": FG,
    "density": PALETTE["green"], "temperature": PALETTE["purple"], "velocity": PALETTE["yellow"],
    "voronoi": PALETTE["cyan"], "tophat": PALETTE["yellow"], "crossings": PALETTE["purple"], "points": PALETTE["red"],
}
LINE_COLOR_ORDER = ("blue", "red", "green", "cyan", "yellow", "purple")   # further lines of a bundle, in this order
MODE_STYLES = {     # SkewerJAX default solid; matched (fake_spectra reproduced) dashed; references thick and light underneath
    "default": dict(ls="-", lw=2.0), "matched": dict(ls="--", lw=2.0), "reference": dict(ls="-", lw=6.0, alpha=0.3),
    "alternative": dict(ls=":", lw=2.2), "isolated": dict(ls="-.", lw=1.8),
}
# Minimum font sizes [pt] of the style: rc defaults, and the floor every figure function respects (``check_fonts``).
FONT_MIN = {"tick": 16, "label": 20, "title": 20, "legend": 15, "annotation": 15, "textbox": 15, "floor": 13}


class NeonGlow(AbstractPathEffect):
    """A soft glow under every stroked line (a wider, translucent copy in the line's own colour).

    Only unfilled paths with more than two vertices glow (data lines and outlines); text, markers, ticks, spines and
    filled patches are drawn unchanged. ``use_style`` installs it for every artist through ``rcParams["path.effects"]``.
    """

    def __init__(self, width: float = 5.5, alpha: float = 0.22):
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
    font = {"family": "serif", "weight": "normal", "size": FONT_MIN["annotation"]}
    tick = lambda ax, extra: {f"{ax}tick.{k}": v for k, v in {
        "color": BG, "direction": "in", "labelsize": 24, "major.pad": 6.0, "major.size": 14.0, "major.width": 1.5,
        "minor.pad": 3.4, "minor.size": 7.0, "minor.visible": True, "minor.width": 1.0, **extra}.items()}
    rc = {"mathtext.fontset": "cm", "text.usetex": False, "figure.frameon": True, "axes.linewidth": 2.0,
          "axes.titlesize": 32, "axes.labelsize": 28, "legend.fontsize": 28, "legend.fancybox": True, "lines.linewidth": 2.5,
          **tick("x", {"alignment": "center", "bottom": True, "top": True, "labelbottom": True, "labeltop": False,
                       "major.bottom": True, "major.top": True, "minor.bottom": True, "minor.top": True}),
          **tick("y", {"alignment": "center_baseline", "left": True, "right": True, "labelleft": True, "labelright": False,
                       "major.left": True, "major.right": True, "minor.left": True, "minor.right": True})}
    return font, rc


def matplotlib_tol_config():
    """``matplotlib_default_config`` with Tol's bright cycle and the minimum font sizes of ``FONT_MIN``: (font, rc)."""
    font, rc = matplotlib_default_config()
    rc.update({"axes.prop_cycle": cycler(color=PALETTE_CYCLE), "image.cmap": "tol_iridescent",
               "axes.titlesize": FONT_MIN["title"] + 2, "axes.labelsize": FONT_MIN["label"] + 1,
               "figure.titlesize": FONT_MIN["title"] + 3, "legend.fontsize": FONT_MIN["legend"],
               "legend.title_fontsize": FONT_MIN["legend"], "xtick.labelsize": FONT_MIN["tick"] + 1,
               "ytick.labelsize": FONT_MIN["tick"] + 1, "xtick.major.size": 9.0, "ytick.major.size": 9.0,
               "xtick.minor.size": 4.5, "ytick.minor.size": 4.5, "lines.linewidth": 2.0, "legend.framealpha": 0.9,
               "legend.edgecolor": DIM, "axes.grid": False, "figure.dpi": 80, "savefig.dpi": 150})
    return font, rc


def matplotlib_paper_config():
    """``matplotlib_tol_config`` on a white background with black text, ticks and axes, no glow: (font, rc). The
    default look of every figure."""
    font, rc = matplotlib_tol_config()
    fg, bg, dim = (THEMES["paper"][k] for k in ("FG", "BG", "DIM"))
    rc.update({"image.cmap": _CMAPS_THEME["paper"]["sequential"],
               "figure.facecolor": bg, "axes.facecolor": bg, "savefig.facecolor": bg, "figure.edgecolor": bg,
               "text.color": fg, "axes.labelcolor": fg, "axes.edgecolor": fg, "axes.titlecolor": fg,
               "xtick.color": fg, "ytick.color": fg, "xtick.labelcolor": fg, "ytick.labelcolor": fg,
               "legend.facecolor": bg, "legend.edgecolor": dim, "legend.labelcolor": fg, "legend.framealpha": 0.9,
               "grid.color": dim, "grid.alpha": 0.5, "hatch.color": fg, "patch.edgecolor": fg,
               "boxplot.flierprops.markeredgecolor": fg, "path.effects": []})
    return font, rc


def matplotlib_dark_config(glow: bool = True):
    """``matplotlib_tol_config`` on a black background, white text and axes and the ``NeonGlow`` under every line:
    (font, rc). The optional dark look."""
    font, rc = matplotlib_tol_config()
    FG, BG, DIM = (THEMES["dark"][k] for k in ("FG", "BG", "DIM"))
    rc.update({"image.cmap": _CMAPS_THEME["dark"]["sequential"],
               "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "figure.edgecolor": BG,
               "text.color": FG, "axes.labelcolor": FG, "axes.edgecolor": FG, "axes.titlecolor": FG,
               "xtick.color": FG, "ytick.color": FG, "xtick.labelcolor": FG, "ytick.labelcolor": FG,
               "legend.facecolor": BG, "legend.edgecolor": DIM, "legend.labelcolor": FG, "legend.framealpha": 0.85,
               "grid.color": DIM, "grid.alpha": 0.35, "hatch.color": FG, "patch.edgecolor": FG,
               "boxplot.flierprops.markeredgecolor": FG, "path.effects": [NeonGlow()] if glow else []})
    return font, rc


def _register_tol_cmaps():
    """Register Tol's maps (and their reverses) with matplotlib as ``tol_<name>`` / ``tol_<name>_r``."""
    for name in _TOL_CMAPS:
        for rev in (False, True):
            cm = tol_cmap(name, reverse=rev)
            if cm.name not in mpl.colormaps:
                mpl.colormaps.register(cm)


def use_style(theme: str = "paper"):
    """Apply the package style globally: one call at the top of a notebook or script.

    theme : "paper" (default: ``matplotlib_paper_config``, white background, publication look) or "dark"
        (``matplotlib_dark_config``: black background and a glow under the lines). The palette, its order and
        ``ENTITY_COLORS`` are the same in both; ``FG``, ``BG``, the ``"total"`` / ``"input"`` entity colours and the
        sequential entries of ``CMAPS`` follow the theme ("light" is accepted as an alias of "paper").
    """
    global FG, BG, DIM, THEME
    theme = "paper" if theme == "light" else theme
    if theme not in THEMES:
        raise ValueError(f"theme must be 'paper' or 'dark', got {theme!r}.")
    _register_tol_cmaps()
    plt.rcdefaults()
    font, rc = matplotlib_dark_config() if theme == "dark" else matplotlib_paper_config()
    THEME, FG, BG, DIM = theme, THEMES[theme]["FG"], THEMES[theme]["BG"], THEMES[theme]["DIM"]
    ENTITY_COLORS.update(total=FG, input=FG)
    CMAPS.update(_CMAPS_THEME[theme])
    plt.rc("font", **font)
    plt.rcParams.update(rc)
    return theme


def _interactive_frontend() -> bool:
    """True in a live Jupyter frontend (JupyterLab / Notebook / VS Code), False in a script or headless execution
    (nbconvert / nbclient set none of these variables; ``SKEWERJAX_HEADLESS=1`` forces False)."""
    if os.environ.get("SKEWERJAX_HEADLESS", "0") not in ("", "0"):
        return False
    try:
        from IPython import get_ipython
        ip = get_ipython()
    except ImportError:
        return False
    if ip is None or type(ip).__name__ != "ZMQInteractiveShell":
        return False
    return any(k in os.environ for k in ("JPY_SESSION_NAME", "VSCODE_PID", "VSCODE_CWD"))


def use_backend(interactive: Optional[bool] = None) -> str:
    """Select the Matplotlib backend of a notebook: "widget" (ipympl: zoom, pan, rotate 3D) in a live frontend when
    ipympl is installed, "inline" otherwise (scripts, headless execution, no ipympl). Returns the backend chosen.

    interactive : None (decide automatically), True (widget if possible) or False (inline).
    """
    try:
        from IPython import get_ipython
        ip = get_ipython()
    except ImportError:
        ip = None
    if ip is None:
        return mpl.get_backend()
    want = _interactive_frontend() if interactive is None else bool(interactive)
    if want:
        try:
            import ipympl  # noqa: F401
            ip.run_line_magic("matplotlib", "widget")
            return "widget"
        except Exception:
            pass
    ip.run_line_magic("matplotlib", "inline")
    return "inline"


def tol_cmap(name: str = "iridescent", reverse: bool = False):
    """Tol colour map: sequential "iridescent", "incandescent", "YlOrBr"; diverging "sunset", "nightfall", "BuRd", "PRGn"."""
    colors, bad = _TOL_CMAPS[name]
    cm = mpl.colors.LinearSegmentedColormap.from_list(f"tol_{name}" + ("_r" if reverse else ""), colors[::-1] if reverse else colors)
    cm.set_bad(bad)
    return cm


def check_fonts(fig, minimum: Optional[float] = None) -> list:
    """Texts of ``fig`` smaller than ``minimum`` (default ``FONT_MIN["floor"]``): [(size, text)] (empty = readable)."""
    m = FONT_MIN["floor"] if minimum is None else minimum
    fig.canvas.draw()
    out = []
    for t in fig.findobj(mpl.text.Text):
        if t.get_visible() and t.get_text().strip() and t.get_fontsize() < m - 1e-9:
            out.append((t.get_fontsize(), t.get_text()[:40]))
    return out


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


ORDER_COLORS = [PALETTE[k] for k in ("grey", "purple", "blue", "green", "yellow", "cyan", "red")]   # by order n (cycled)


def plot_orders_superimposed(stats, kind: str = "P", *, k_weighted: bool = False, max_lag: Optional[float] = None,
                             delta_v_list=None, linthresh: Optional[float] = None, axes=None, figsize=(12, 9)):
    """P1D or xi1D of a composite overflux split by order (``OverfluxStatistics1D.*_total_by_order``), all orders on one
    panel; the sum of the orders over the total, and (sum - total) / max |total| below (rounding error only).

    kind : "P" (symlog y, log k) or "xi" (linear; lags 0..max_lag, default half the skewer); k_weighted : k P / pi;
    delta_v_list : line offsets [km/s]; vertical lines at every separation dv = |dv_i - dv_j| (xi, folded in the box)
        and, for P, at its harmonics k = 2 pi n / dv, where the cross terms cos(k dv) peak.
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
    if kind == "P" and delta_v_list is not None:          # the cross terms oscillate as cos(k dv): features at k = 2 pi n / dv
        dv = np.asarray(delta_v_list, dtype=float)
        first = True
        for d in np.unique(np.round(np.abs(dv[:, None] - dv[None, :]), 6)):
            if d <= 0:
                continue
            for n_ in range(1, int(x.max() * d / (2 * np.pi)) + 1):
                kn = 2 * np.pi * n_ / d
                if kn >= x.min():
                    for a in (ax0, ax1):
                        a.axvline(kn, color=PALETTE["grey"], ls="--", lw=1.0, alpha=0.7, path_effects=[],
                                  label=rf"$k = 2\pi n/\Delta v$ ($\Delta v$ = {d:g} km/s)" if (first and a is ax0) else None)
                    first = False
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
FIGSIZES = {"wide": (20, 6), "single": (13, 6), "tall": (13, 9), "triple": (22, 6), "grid": (22, 12), "dag": (22, 6.0)}

# The dependency graph of the ingredients, derived from the code. Columns are topological levels; the two field sources
# (toy samplers and simulation outputs) are interchangeable inputs of the gas fields, side by side in one column.
DAG_NODES = {   # key: (column, row, name, detail)
    "lines": (0, 1.45, "lines", "LineBundle"),
    "cosmology": (0, -1.2, "cosmology", "Cosmology"),
    "skewers": (1, -1.2, "skewers", "SkewerBundle, a(x)"),
    "samplers": (2, 0.52, "toy samplers", "PowerLawGRF, ramps"),
    "simulation": (2, -0.52, "simulation outputs", "TNG snapshot → gas"),
    "fields": (3, 0.0, "gas fields", "n, T, v at the points"),
    "tau": (4, 0.0, "optical depth", r"$\tau(v)$, $F=e^{-\tau}$"),
    "overflux": (5, 0.0, "overflux", r"$\delta_F$ per line, total"),
    "stats": (6, 0.0, "statistics", "P1D, ξ1D, P3D, ξ3D"),
    "fake_spectra": (3, -0.8, "fake_spectra", "external reference"),
}
# (from, to, kind, label, route): kind "solid" required | "optional" optional input or data table | "external";
# route = waypoints between the two ends: "h" leaves horizontally then turns, "v" leaves vertically then turns.
DAG_EDGES = (
    ("cosmology", "skewers", "optional", "a(x) (or manual)", None),
    ("skewers", "simulation", "solid", "paths", "v"),
    ("skewers", "tau", "solid", "points, a(x), w", "h"),
    ("lines", "simulation", "optional", "ion masses", "v"),
    ("samplers", "fields", "solid", None, None),
    ("simulation", "fields", "solid", None, None),
    ("fields", "tau", "solid", None, None),
    ("lines", "tau", "solid", None, "h"),
    ("tau", "overflux", "solid", None, None),
    ("lines", "overflux", "optional", "line offsets Δv (frames)", "h"),
    ("overflux", "stats", "solid", None, None),
)
DAG_EXTERNAL = (("simulation", "fake_spectra", None), ("fake_spectra", "tau", "compared"))   # same snapshot, compared τ
DAG_NOTEBOOKS = {"lines": "01", "cosmology": "02", "skewers": "03", "samplers": "04", "fields": "04", "simulation": "05",
                 "tau": "06", "overflux": "07", "stats": "07", "fake_spectra": "08"}
_DAG_W, _DAG_H = 0.80, 0.56


def _dag_route(a, b, route, nodes, lane):
    """Polyline from box a to box b. None: straight between facing sides; "h": out of the right side of a at height
    ``lane`` above its centre, horizontally, then vertically into b; "v": out of the top / bottom of a, vertically to
    height ``lane`` above b's centre, then horizontally into the left side of b."""
    (xa, ya), (xb, yb) = nodes[a][:2], nodes[b][:2]
    w, h = _DAG_W / 2, _DAG_H / 2
    if route is None:
        return [(xa + w, ya), (xb - w, yb + 0.25 * (ya - yb))]
    if route == "h":
        y0 = ya + lane
        return [(xa + w, y0), (xb, y0), (xb, yb + (h if y0 > yb else -h))]
    y1, x0 = yb + lane, xa + 0.18
    return [(x0, ya + (h if y1 > ya else -h)), (x0, y1), (xb - w, y1)]


def plot_dag(highlight=(), *, external: bool = False, ax=None, figsize=None, notebooks: bool = True):
    """The dependency graph (DAG) of SkewerJAX's ingredients; ``highlight`` (keys of ``DAG_NODES``: the ingredients a
    notebook uses) are drawn with one light fill and a coloured edge, and the edges into them in the same colour.

    Arrows point from an ingredient to what needs it. Solid: a required input. Dashed: optional, or data only (the
    cosmology can be replaced by manual a(x); the simulation reads only ion masses from the line table; the overflux
    reads only the line offsets Δv to put every line on the reference frame). The two field sources are interchangeable
    inputs of the gas fields. ``external`` adds fake_spectra, the external reference code (comparison notebook only).
    ``notebooks`` writes the notebook of every ingredient under its box. Returns (fig, ax).
    """
    from matplotlib.lines import Line2D
    hl = {highlight} if isinstance(highlight, str) else set(highlight)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or FIGSIZES["dag"], layout="constrained")
    ax.set_axis_off()
    ax.set(xlim=(-0.5, 6.5), ylim=(-1.62, 1.86))
    fg, on, ext_c = plt.rcParams["text.color"], PALETTE["cyan"], PALETTE["yellow"]
    on_fill = mpl.colors.to_rgba(on, 0.30)                      # the one highlight: a light fill and a coloured edge
    w, h = _DAG_W, _DAG_H
    nodes = {k: v for k, v in DAG_NODES.items() if external or k != "fake_spectra"}
    lanes = {("lines", "tau"): -0.2, ("lines", "overflux"): 0.2, ("lines", "simulation"): 0.12,
             ("skewers", "simulation"): -0.12, ("skewers", "tau"): 0.0}
    # the group of interchangeable field sources
    gx, gy = DAG_NODES["samplers"][0], 0.5 * (DAG_NODES["samplers"][1] + DAG_NODES["simulation"][1])
    gh = DAG_NODES["samplers"][1] - DAG_NODES["simulation"][1] + h + 0.34
    ax.add_patch(mpl.patches.FancyBboxPatch((gx - w / 2 - 0.08, gy - gh / 2), w + 0.16, gh, boxstyle="round,pad=0.02,rounding_size=0.1",
                 facecolor="none", edgecolor=DIM, lw=1.2, ls=":", path_effects=[]))
    ax.text(gx, gy + gh / 2 + 0.04, "field sources (interchangeable)", ha="center", va="bottom", fontsize=FONT_MIN["floor"],
            color=DIM, style="italic")
    edges = [e for e in DAG_EDGES] + ([(a, b, "external", lab, None) for a, b, lab in DAG_EXTERNAL] if external else [])
    for a, b, kind, label, route in edges:
        if a not in nodes or b not in nodes:
            continue
        if (a, b) == ("fake_spectra", "tau"):
            pts = [(nodes[a][0] + w / 2, nodes[a][1]), (nodes[b][0] - 0.2, nodes[a][1]), (nodes[b][0] - 0.2, nodes[b][1] - h / 2)]
        elif (a, b) == ("simulation", "fake_spectra"):
            pts = [(nodes[a][0] + w / 2, nodes[a][1] - 0.15), (nodes[b][0] - w / 2, nodes[a][1] - 0.15)]
        elif (a, b) == ("skewers", "tau"):
            pts = [(nodes[a][0] + w / 2, nodes[a][1]), (nodes[b][0], nodes[a][1]), (nodes[b][0], nodes[b][1] - h / 2)]
        else:
            pts = _dag_route(a, b, route, nodes, lanes.get((a, b), 0.0))
        hot = b in hl                                           # the inputs of a highlighted ingredient light up
        col = on if hot else (ext_c if kind == "external" else DIM)
        dashed = kind in ("optional", "external")
        lw = 2.4 if hot else 1.6
        xs, ys = zip(*pts)
        # one polyline with a fixed dash pattern (in points: the same in every orientation and segment length)
        ax.plot(xs, ys, color=col, lw=lw, path_effects=[], solid_capstyle="butt", dash_capstyle="butt",
                **(dict(ls=(0, (4.0, 3.0))) if dashed else dict(ls="-")))
        (xe0, ye0), (xe1, ye1) = pts[-2], pts[-1]               # the arrow head only, on a short stub of the last segment
        d = np.hypot(xe1 - xe0, ye1 - ye0)
        stub = (xe1 - 0.06 * (xe1 - xe0) / d, ye1 - 0.06 * (ye1 - ye0) / d)
        ax.annotate("", (xe1, ye1), stub, arrowprops=dict(arrowstyle="-|>", color=col, mutation_scale=18, lw=lw,
                    ls="-", shrinkA=0, shrinkB=1))
        if label:                                               # on the longest segment (above the boxes if it is short)
            seg = max(range(len(pts) - 1), key=lambda k: np.hypot(pts[k + 1][0] - pts[k][0], pts[k + 1][1] - pts[k][1]))
            (x0, y0), (x1, y1) = pts[seg], pts[seg + 1]
            vert = abs(x1 - x0) < 1e-9
            f = 0.3 if (a, b) == ("skewers", "tau") else 0.5
            xl, yl = x0 + f * (x1 - x0) + (0.03 if vert else 0.0), 0.5 * (y0 + y1) + (0.0 if vert else 0.035)
            if route is None and not vert and abs(x1 - x0) < 0.5:
                yl = max(nodes[a][1], nodes[b][1]) + h / 2 + 0.04
            ax.text(xl, yl, label,
                    ha="left" if vert else "center", va="center" if vert else "bottom", fontsize=FONT_MIN["floor"],
                    color=col, style="italic", rotation=0,
                    bbox=dict(boxstyle="round,pad=0.12", facecolor=plt.rcParams["axes.facecolor"], edgecolor="none", alpha=0.85))
    for key, (x, y, name, sub) in nodes.items():
        hit = key in hl
        ext = key == "fake_spectra"
        ec = on if hit else (ext_c if ext else DIM)
        ax.add_patch(mpl.patches.FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                     facecolor=on_fill if hit else plt.rcParams["axes.facecolor"], edgecolor=ec, lw=2.6 if hit else 1.4,
                     ls=(0, (4.0, 3.0)) if ext else "-", path_effects=[], zorder=3))
        tc = fg if hit else (ext_c if ext else fg)
        ax.text(x, y + 0.1, name, ha="center", va="center", fontsize=FONT_MIN["legend"], color=tc, fontweight="bold", zorder=4)
        ax.text(x, y - 0.13, sub, ha="center", va="center", fontsize=FONT_MIN["floor"], color=tc, zorder=4)
        if notebooks and key in DAG_NOTEBOOKS:
            ax.text(x + w / 2, y - h / 2 - 0.03, f"nb {DAG_NOTEBOOKS[key]}", ha="right", va="top", fontsize=FONT_MIN["floor"],
                    color=on if hit else DIM, zorder=4)
    handles = [Line2D([], [], color=DIM, lw=1.8, ls="-", label="required input"),
               Line2D([], [], color=DIM, lw=1.8, ls=(0, (4.0, 3.0)), label="optional / data only"),
               Line2D([], [], lw=0, marker="s", ms=13, markerfacecolor=on_fill, markeredgecolor=on, markeredgewidth=2.0,
                      label="used in this notebook")]
    if external:
        handles.append(Line2D([], [], color=ext_c, lw=1.8, ls=(0, (4.0, 3.0)), label="external reference"))
    for hd in handles:
        hd.set_path_effects([])
    ax.legend(handles=handles, loc="lower right", ncol=len(handles), frameon=False, fontsize=FONT_MIN["floor"],
              bbox_to_anchor=(1.0, -0.04), handlelength=2.4)
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
            ax.legend(fontsize=FONT_MIN["legend"])
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
    ins.tick_params(labelsize=FONT_MIN["tick"])
    ax.indicate_inset_zoom(ins, edgecolor=DIM, **kw)
    return ins


# --------------------------------------------------------------------------- the dark ("neon") style of the visualisations
NEON_COLORS = ENTITY_COLORS          # the visualisations use the same colour-blind-safe colours as every other figure
NEON_CYCLE = PALETTE_CYCLE


def dark_axes(axs, fig=None):
    """The style's background (``BG``) with ``FG`` ticks, labels and spines on every axis of ``axs`` (and on ``fig``);
    the name is historical: it follows the current style."""
    for ax in np.ravel(axs):
        ax.set_facecolor(BG)
        ax.tick_params(axis="both", which="both", colors=FG)
        for t in (ax.xaxis.label, ax.yaxis.label, ax.title):
            t.set_color(FG)
        for sp in ax.spines.values():
            sp.set_color(FG)
    fig = fig or np.ravel(axs)[0].figure
    fig.set_facecolor(BG)
    return axs


def neon_line(ax, x, y, color, *, lw: float = 1.6, glow: float = 7.0, label=None, **kw):
    """A line with a soft glow underneath in the dark style (two strokes; the global ``NeonGlow`` is switched off for
    them); a plain line in the paper style."""
    if THEME == "dark":
        ax.plot(x, y, color=color, lw=glow, alpha=0.18, solid_capstyle="round", zorder=2, path_effects=[], **kw)
    return ax.plot(x, y, color=color, lw=lw, alpha=0.95, zorder=3, label=label, path_effects=[], **kw)[0]


def neon_legend(ax, **kw):
    leg = ax.legend(**{"frameon": True, "facecolor": BG, "edgecolor": FG, "fontsize": FONT_MIN["legend"], **kw})
    for t in leg.get_texts():
        t.set_color(FG)
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
    cmap = plt.get_cmap(cmap or (CMAPS["sequential"] if quantity == "sigma" else CMAPS["hot"]))
    cmap = cmap.with_extremes(bad=cmap(0.0))
    im = ax.imshow(lm.T, origin="lower", extent=slab["extent"], cmap=cmap,
                   vmin=vmin if vmin is not None else np.nanpercentile(lm, 1), vmax=vmax if vmax is not None else np.nanpercentile(lm, 99.8),
                   interpolation="nearest")
    ax.set(xlabel=f"{names[slab['axes'][0] - 1]} [cMpc]", ylabel=f"{names[slab['axes'][1] - 1]} [cMpc]")
    if colorbar:
        cb = ax.figure.colorbar(im, ax=ax, pad=0.01, fraction=0.046)
        cb.set_label(label or ("log10 " + LABELS.get(quantity, quantity)), color=FG if ax.get_facecolor()[0] < 0.5 else BG)
        cb.ax.tick_params(colors=FG if ax.get_facecolor()[0] < 0.5 else BG)
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
        ax_gas.plot(s, n, color=DIM, lw=0.8, zorder=1)
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
            ax.text(i, bottom, f"{total:.1f} s", ha="center", va="bottom", fontsize=FONT_MIN["legend"])
        labels.append(BENCH_STYLES.get(bench_key(r), {}).get("label", str(bench_key(r))).replace(", ", "\n"))
    ax.set_xticks(range(len(rs)), labels, fontsize=FONT_MIN["legend"])
    ax.set_ylabel("share of the median wall time [%]" if fraction else "median wall time [s]")
    ax.legend(fontsize=FONT_MIN["legend"])
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
    ax.set_xticks(range(len(sims)), [f"{s}\n{cells.get(s, {}).get(n_skewers) or 0:,} cells selected" for s in sims], fontsize=FONT_MIN["legend"])
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
    ax.set_xticks(range(len(keys)), [f"{sim}\nN={n}" for sim, n in keys], fontsize=FONT_MIN["legend"])
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
    ids = [l.id for l in lines]
    col = {i_: ENTITY_COLORS.get(i_, ENTITY_COLORS.get(l.species)) for i_, l in zip(ids, lines)}
    col = {i_: c for i_, c in col.items() if c is not None}
    spare = [PALETTE[k] for k in LINE_COLOR_ORDER if PALETTE[k] not in col.values()] or PALETTE_CYCLE
    col.update({i_: spare[k % len(spare)] for k, i_ in enumerate(i_ for i_ in ids if i_ not in col)})
    lam, f, A21, m = (np.asarray(x, dtype=float) for x in (line_bundle.lambda0_A, line_bundle.f_osc, line_bundle.A21_GHz,
                                                          line_bundle.mass_amu))
    dv = np.asarray(line_bundle.delta_v, dtype=float)
    if axs is None:
        fig, axs = plt.subplots(1, 3, figsize=figsize, layout="constrained", gridspec_kw={"width_ratios": [1.3, 1, 1]})
    order = np.argsort(lam)
    xl = np.log10(lam[order])                                    # label positions: at least 0.03 dex apart
    for r in range(1, xl.size):
        xl[r] = max(xl[r], xl[r - 1] + 0.03)
    xl -= max(0.0, xl[-1] - np.log10(lam.max()) - 0.02)
    for r, j in enumerate(order):
        c = col[ids[j]]
        axs[0].vlines(lam[j], 1e-3, f[j], color=c, lw=3)
        axs[0].annotate(ids[j], (lam[j], f[j]), xytext=(10 ** xl[r], 25.0), textcoords="data", rotation=90, ha="center",
                        va="bottom", fontsize=FONT_MIN["legend"], color=c,
                        arrowprops=dict(arrowstyle="-", color=c, lw=0.8, shrinkA=0, shrinkB=2))
    axs[0].set(xscale="log", yscale="log", ylim=(1e-3, 3e4), xlabel=r"rest wavelength $\lambda_0$ [Å]", ylabel=r"oscillator strength $f$")
    axs[0].xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
    axs[0].xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    axs[0].set_xticks([1000, 1250, 1500, 2000, 2800])
    y = np.arange(len(lines))
    axs[1].barh(y, dv / 1e3, color=[col[i_] for i_ in ids], height=0.6)
    axs[1].axvline(0, color=plt.rcParams["text.color"], lw=1)
    axs[1].set_yticks(y, [l.id for l in lines], fontsize=FONT_MIN["legend"])
    axs[1].invert_yaxis()
    axs[1].set(xlabel=rf"$\Delta v_j = c\,\ln(\lambda_{{\rm ref}}/\lambda_j)$ [$10^3$ km/s], ref = {line_bundle.ref_line_id}")
    b = np.sqrt(2 * phys.kB * T_K / m) / 1e3                        # phys.kB = k_B / m_u [m^2 s^-2 K^-1] -> [km/s]
    nat = lam * 1e-10 * A21 * 1e9 / (4 * np.pi) / 1e3                  # lambda_0 gamma [km/s]
    axs[2].scatter(nat, y, color=[col[i_] for i_ in ids], s=80, marker="o", label=r"natural $\lambda_0 A_{21}/4\pi$")
    axs[2].scatter(b, y, color=[col[i_] for i_ in ids], s=110, marker="D", label=rf"thermal $b$ at $T = 10^{{{np.log10(T_K):.0f}}}$ K")
    for j in range(len(lines)):
        axs[2].plot([nat[j], b[j]], [y[j], y[j]], color=col[ids[j]], lw=1, alpha=0.6)
    axs[2].set_yticks(y, [l.id for l in lines], fontsize=FONT_MIN["legend"])
    axs[2].invert_yaxis()
    axs[2].set(xscale="log", xlabel="width [km/s]", xlim=(min(nat.min(), b.min()) / 3, max(nat.max(), b.max()) * 30))
    axs[2].legend(fontsize=FONT_MIN["legend"], loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=1, frameon=False)
    return axs[0].figure, axs


def box_edges_3d(ax, L, **kw):
    """The 12 edges of the cube [0, L]^3 on a 3D axis."""
    kw = {"color": DIM, "lw": 0.8, "alpha": 0.7, **kw}
    for a in (0, L):
        for b in (0, L):
            ax.plot([0, L], [a, a], [b, b], **kw)
            ax.plot([a, a], [0, L], [b, b], **kw)
            ax.plot([a, a], [b, b], [0, L], **kw)


def plot_paths_3d(ax, points, *, values=None, cmap=None, vmin=None, vmax=None, color=None, lw: float = 2.5,
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
        cm_ = plt.get_cmap(CMAPS["line"] if cmap is None else cmap)
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
    """Transparent panes, the style's ticks and labels and a faint grid on a 3D axis (follows the current style)."""
    ax.set_facecolor(plt.rcParams["axes.facecolor"])
    grid = (1, 1, 1, 0.08) if THEME == "dark" else (0, 0, 0, 0.10)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.set_pane_color((0, 0, 0, 0))
        a.line.set_color(DIM)
        a.label.set_color(plt.rcParams["text.color"])
        a._axinfo["grid"]["color"] = grid
    ax.tick_params(colors=plt.rcParams["text.color"], labelsize=FONT_MIN["tick"])


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
    ax[0].legend(fontsize=FONT_MIN["legend"])
    ax[1].plot(i, s, color=PALETTE["cyan"], label="path_lengths s")
    ax[1].plot(i, d, color=PALETTE["yellow"], ls="--", label="comoving_positions_from_ICs")
    ax[1].set(xlabel="point index i", ylabel="[cMpc]", title="distances along the path")
    ax[1].legend(fontsize=FONT_MIN["legend"])
    ax[2].step(s, w, where="mid", color=PALETTE["green"])
    ax[2].fill_between(s, 0, w, step="mid", color=PALETTE["green"], alpha=0.25)
    ax[2].set(xlabel=LABELS["s"], ylabel=r"$w_i$ [cMpc]", title=rf"line_elements: $\sum_i w_i$ = {w.sum():.4g} = $s_{{\rm end}}$")
    ax[3].plot(s, a, color=PALETTE["purple"])
    ax[3].plot(s[ic], a[ic], "o", color=FG, ms=10)
    ax[3].annotate(f"ICs: a0 = {float(bundle.a0[k]):.4g} at x0 = {float(bundle.x0[k]):.4g}\n"
                   f"({'absorption' if bool(bundle.is_absorption[k]) else 'emission'} point)",
                   (s[ic], a[ic]), textcoords="offset points", xytext=(15 if ic == 0 else -15, -40),
                   ha="left" if ic == 0 else "right", fontsize=FONT_MIN["legend"], arrowprops=dict(arrowstyle="->", color=FG))
    ax[3].set(xlabel=LABELS["s"], ylabel="a", title="scale_factors (N_cosmo, N_sk, N_ppsk)")
    for c, name, colr in zip(range(3), "xyz", (PALETTE["blue"], PALETTE["red"], PALETTE["green"])):
        ax[4].plot(s, los[:, c], color=colr, label=rf"$\hat d_{name}$")
    ax[4].set(xlabel=LABELS["s"], ylabel="component", ylim=(-1.15, 1.15),
              title=f"los_directions; array_direction = {int(bundle.array_direction[k]):+d}")
    ax[4].legend(fontsize=FONT_MIN["legend"], ncol=3)
    ax[5].set_axis_off()
    rows = [(n, tuple(np.shape(getattr(bundle, n)))) for n in bundle._leaves]
    txt = "\n".join(f"{n:<28s}{str(sh):>18s}" for n, sh in rows)
    ax[5].text(0.0, 1.0, "SkewerBundle (a JAX pytree)\n\n" + txt, family="monospace", fontsize=FONT_MIN["legend"], va="top",
               transform=ax[5].transAxes, color=FG)
    ax[5].text(0.0, 0.02, "quadri_vectors[..., 0] = a, [..., 1:4] = (x, y, z):\nthe events every field sampler takes",
               fontsize=FONT_MIN["legend"], transform=ax[5].transAxes, color=PALETTE["cyan"])
    add_panel_letters(ax[:5])
    return fig, axs


def plot_voronoi_discretisations(seed: int = 3, n_cells: int = 90, every: int = 25, axs=None, figsize=(16, 15)):
    """The three grids of a simulation spectrum on a toy 2D Voronoi mesh.

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
            axs[0].plot(*vor.vertices[r].T, color=DIM, lw=1.0, path_effects=[])
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
    axs[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncols=2, fontsize=FONT_MIN["legend"])
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
    axs[1].legend(loc="upper right", ncols=3, fontsize=FONT_MIN["legend"])
    vv = np.linspace(-60, 20 * s_path[-1] + 60, 900)
    tot = np.zeros_like(vv)
    for i, (s0, nn) in enumerate(zip(s_sp, seg_mean)):
        prof = nn * w[i] * np.exp(-((vv - 20 * s0) / 9) ** 2)
        tot += prof
        axs[2].plot(vv, prof, lw=2, color=cyc[i % len(cyc)])
    axs[2].plot(vv, tot, lw=1.2, ls=":", color=FG, label="sum: the optical depth")
    axs[2].vlines(np.arange(vv[0], vv[-1], 16), 0, 1.05 * tot.max(), color=DIM, lw=1, ls=":", path_effects=[])
    axs[2].set(xlabel=r"spectral coordinate (velocity)", ylabel=r"d$\tau$")
    axs[2].set_title("(c) The spectral grid (dotted) is independent; every segment adds its own profile")
    axs[2].legend(fontsize=FONT_MIN["legend"], loc="upper right")
    return axs[0].figure, axs, nums


def plot_line_frames(v, phi, phi_analytic, line_bundle, v_centre, *, a_dagger: float, a_obs: float = 1.0, band: float = 1e-6,
                     figsize=(22, 13)):
    """Line profiles in their own line-centred frames and on the observed-wavelength axis, against the analytic
    profile (notebook 06).

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
                           Line2D([], [], color=FG, marker="x", ls="none", ms=10, mew=2, label=r"$4/A_{21}$")], fontsize=FONT_MIN["legend"])
    low.set(yscale="log", xlabel=r"observed wavelength $\lambda_{\rm obs}$ [Å]", ylabel=r"$\phi$ [ns]")
    low.legend(fontsize=FONT_MIN["legend"], ncol=n, loc="upper center")
    add_panel_letters(top + res + [low], size=15)
    return fig, (top, res, low)


def plot_flux_vs_analytic(v, F, F_analytic, line_bundle, lam_obs, lam_centre, *, zoom_kms: float = 80.0,
                          skewer_labels=None, band: float = 1e-2, figsize=(22, 13)):
    """F against the closed form on the observed-wavelength axis, and a zoom on every line centre with its
    residual (notebook 06).

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
        main.axvline(l.lambda0_A, color=DIM, ls="-.", lw=1.3)
        sel = np.abs(v) <= zoom_kms
        for k in range(ns):
            lab = (skewer_labels[k] if skewer_labels is not None else f"skewer {k}") if j == 0 else None
            main.plot(lam_obs[j, k], F_analytic[j, k], color=cols[k], **MODE_STYLES["reference"])
            main.plot(lam_obs[j, k], F[j, k], color=cols[k], lw=1.8, label=lab)
            main.axvline(lam_centre[j, k], color=cols[k], ls="--", lw=1.2)
            zoom[j].plot(v[sel], F_analytic[j, k, sel], color=cols[k], **MODE_STYLES["reference"])
            zoom[j].plot(v[sel], F[j, k, sel], color=cols[k], lw=1.8)
            resid[j].plot(v[sel], F[j, k, sel] - F_analytic[j, k, sel], color=cols[k], lw=1.5)
        zoom[j].axvline(0.0, color=DIM, ls="-.", lw=1.2)
        zoom[j].set_title(rf"{l.id}, $\lambda_0$ = {l.lambda0_A:.3f} Å", fontsize=17)
        residual_band(resid[j], band)
        resid[j].set(xlabel=r"line-centred $v_\dagger$ [km/s]")
    main.set(xlabel=r"observed wavelength $\lambda_{\rm obs}$ [Å]", ylabel=r"$F/F_0$")
    main.legend(fontsize=FONT_MIN["legend"], loc="lower left")
    zoom[0].set_ylabel(r"$F/F_0$")
    resid[0].set_ylabel(r"$F - F_{\rm an}$")
    add_panel_letters([main] + zoom + resid, size=15)
    return fig, (main, zoom, resid)


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
        ax.axhline(0, color=DIM, lw=0.8)
        ax.legend(fontsize=FONT_MIN["legend"], ncol=2)
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
    axs[0].legend(fontsize=FONT_MIN["legend"], ncol=2)
    return axs[0].figure, axs


def plot_cylinder(s, profiles: dict, *, cloud=None, flux=None, progress: float = 1.0, radius: float = 1.0, view=(14.0, -62.0),
                  cloud_cmap: str = None, cloud_clim=None, n_rings: int = 5, flux_on_cylinder: bool = False, fig=None,
                  figsize=(22, 13)):
    """The gas around a skewer wrapped in a cylinder, and the flux it has absorbed so far (notebook
    00_quickstart_step_by_step).

    s (N,) path coordinate along the skewer [cMpc] (the cylinder axis).
    profiles : {label: (y (N,), colour)} quantities along the skewer (e.g. the ion densities, log-scaled for display),
        each drawn as a glowing radar line on the cylinder surface at its own angle (radius ∝ the scaled value).
    cloud : optional dict(axis (M,), s_perp (M,), t_perp (M,), value (M,)) of the gas around the skewer (e.g. the
        selected cells: along-axis coordinate, transverse offsets [cMpc], density), drawn as a soft point cloud.
    flux : optional dict(v (N_v,), s_of_v (N_v,), edges (n_bins + 1,), tau_bins {label: ((n_bins, N_v), colour)}): the
        tau of every stretch of the path (``tau_by_path_bins``); the absorbed flux is drawn in a panel below against v
        (and, with ``flux_on_cylinder``, also on the cylinder surface, radius ∝ 1 - F, at the position s(v) of every
        pixel; by default the cylinder shows only the ``profiles`` and the ``cloud``).
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
    cloud_cmap = CMAPS["sequential"] if cloud_cmap is None else cloud_cmap
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
    n_prof = len(profiles) + (len(flux["tau_bins"]) if (flux is not None and flux_on_cylinder) else 0)
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
        for (lab, (tb, c)), th in zip(flux["tau_bins"].items(), list(angles[len(profiles):]) + [0.0] * len(flux["tau_bins"])):
            tb = np.asarray(tb, dtype=float)
            F_now, F_end = np.exp(-(frac[:, None] * tb).sum(0)), np.exp(-tb.sum(0))
            if flux_on_cylinder:
                r = radius * (1.0 - F_now)
                ax.plot(s_of_v, r * np.cos(th), r * np.sin(th), color=c, lw=1.4, zorder=6, label=f"1 − F, {lab}")
            axf.plot(v, F_end, color=c, lw=1.0, alpha=0.3, path_effects=[])
            axf.plot(v, F_now, color=c, lw=1.6, label=f"{lab}: absorbed up to s = {s_p:.1f} cMpc")
        axf.set(xlim=(v[0], v[-1]), ylim=(-0.05, 1.1), xlabel=LABELS["v_dagger"] + ", each line in its own frame", ylabel=LABELS["F"])
        axf.legend(fontsize=FONT_MIN["legend"], loc="lower left", ncol=2)
    ax.set(xlim=(s0, s1), ylim=(-1.25 * radius, 1.25 * radius), zlim=(-1.25 * radius, 1.25 * radius))
    ax.set_box_aspect((3.2, 1.0, 1.0), zoom=1.25)
    ax.set_xlabel(LABELS["s"], labelpad=14)
    ax.set_yticks([]); ax.set_zticks([])
    dark_3d(ax)
    ax.view_init(*view)
    h, lab = ax.get_legend_handles_labels()
    fig.legend(h, lab, loc="upper left", bbox_to_anchor=(0.01, 0.93), fontsize=FONT_MIN["legend"], frameon=True, ncol=1)
    return fig, (ax, axf)


# --------------------------------------------------------------------------- tutorial figures (third pass)
def _textbox(ax, x, y, s, *, color=None, transform=None, **kw):
    """A short label in a dark rounded box (readable on any background)."""
    kw = {"fontsize": FONT_MIN["textbox"], "ha": "left", "va": "center", "zorder": 30, **kw}
    return ax.text(x, y, s, color=color or FG, transform=transform or ax.transData,
                   bbox=dict(boxstyle="round,pad=0.25", facecolor=BG, edgecolor=color or DIM, alpha=0.88, lw=1.0), **kw)


def plot_scale_factor_solutions(x, a_num, exact: dict, colors: dict, *, x0: float = 0.0, a0: float = 1.0, h: float = None,
                                linthresh: float = 1e-5, axs=None, figsize=(14, 10)):
    """a(x) along a light ray for several backgrounds: numerical (solid) on the closed forms (thick, light), the present
    day (x0, a0) with guide lines, each background's special point (vertical line in its colour, labelled), and the
    residual numerical / closed form - 1 on a symlog axis (``linthresh``; ticks 0, +-linthresh, +-1).

    x : (N,) comoving distance [Mpc] (x > x0 towards the future); a_num : {name: (N,)}; exact : {name: (N,) or None};
    colors : {name: colour}; h : H0/c [1/Mpc]. Special points of the one-component closed forms of da/dx (x0, a0 the
    initial conditions): radiation, da/dx = h, a = a0 + h (x - x0) vanishes at x0 - a0/h; matter, da/dx = h a^(1/2),
    a = (a0^(1/2) + h (x - x0) / 2)^2 reaches a = a' = 0 at x0 - 2 a0^(1/2)/h; Lambda, da/dx = h a^2,
    a = a0 / (1 - a0 h (x - x0)) diverges at x0 + 1/(a0 h). Returns (fig, (main, residual)).
    """
    from matplotlib.ticker import FixedLocator, FuncFormatter
    if axs is None:
        fig, axs = plt.subplots(2, 1, figsize=figsize, sharex=True, gridspec_kw={"height_ratios": [3, 1.4]}, layout="constrained")
    ax, axr = axs
    specials = {} if h is None else {"radiation": (x0 - a0 / h, "radiation: a → 0"),
                                     "matter": (x0 - 2.0 * np.sqrt(a0) / h, "matter: a′ = 0"),
                                     "Lambda": (x0 + 1.0 / (a0 * h), "Λ: a → ∞")}
    for name, a in a_num.items():
        c = colors[name]
        ax.plot(x, a, color=c, label=name, zorder=3)
        ex = exact.get(name)
        if ex is not None:
            ok = np.isfinite(ex) & (ex > 0)
            ax.plot(x[ok], ex[ok], color=c, **MODE_STYLES["reference"])
            axr.plot(x[ok], a[ok] / ex[ok] - 1.0, color=c)
        if name in specials:
            xs, txt = specials[name]
            for a_ in (ax, axr):
                a_.axvline(xs, color=c, ls="--", lw=1.4, path_effects=[])
            ax.text(xs, 1.5e-3, " " + txt, rotation=90, color=c, va="bottom", ha="right", fontsize=FONT_MIN["annotation"])
    ax.axvline(x0, color=DIM, ls=":", lw=1.2, path_effects=[])
    ax.axhline(a0, color=DIM, ls=":", lw=1.2, path_effects=[])
    ax.plot(x0, a0, "x", color=FG, ms=14, mew=3, zorder=5)
    _textbox(ax, x0 + 0.03 * (x[-1] - x[0]), a0 * 3.0, "present day (x₀, a₀)")
    ax.set(yscale="log", ylim=(1e-3, 1e2), ylabel="scale factor $a$")
    ax.legend(loc="upper center", bbox_to_anchor=(0.36, 1.0), ncols=2)
    axr.set_yscale("symlog", linthresh=linthresh)
    ticks = [-1.0, -linthresh, 0.0, linthresh, 1.0]
    axr.yaxis.set_major_locator(FixedLocator(ticks))
    axr.yaxis.set_major_formatter(FuncFormatter(lambda y, p: "0" if y == 0 else (f"{y:+.0f}" if abs(y) >= 1 else f"{y:+.0e}")))
    axr.set_ylim(-2.0, 2.0)
    axr.axhline(0.0, color=FG, lw=1.0)
    axr.set(xlabel="comoving distance $x$ [Mpc] (towards the future)", ylabel="numerical / exact − 1")
    return ax.figure, np.array([ax, axr])


def plot_skewer_geometry_3d(ax, points, *, scale_factors=None, label_w=None, tangent_length: float = None,
                            highlight: int = None):
    """The geometry of one skewer in 3D (notebook 03): sampling points x_i, midpoints m_{i+-1/2}, the dual segment of
    every point (from m_{i-1/2} to m_{i+1/2}; coloured by a(x) when ``scale_factors`` (N,) is given, else in
    alternating colours), unit tangents t_i and the line elements w_i of the points ``label_w`` (default: about six).
    ``highlight`` names one point whose dual segment is annotated. Returns the ScalarMappable of a(x) (or None).
    """
    from skewerjax.core.skewers import path_geometry
    p = np.asarray(points, dtype=float)
    g = {k: np.asarray(v) for k, v in path_geometry(p).items()}
    n = len(p)
    sm = None
    if scale_factors is not None:
        a = np.asarray(scale_factors, dtype=float)
        norm = mpl.colors.Normalize(a.min(), a.max())
        cm_ = plt.get_cmap(CMAPS["line"])
        cols = [cm_(norm(x)) for x in a]
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cm_)
    else:
        cols = [(PALETTE["blue"], PALETTE["cyan"])[i % 2] for i in range(n)]
    for i, (lo, x, hi) in enumerate(zip(g["mid_lo"], p, g["mid_hi"])):
        ax.plot(*np.stack([lo, x, hi]).T, color=cols[i], lw=9, alpha=0.9, solid_capstyle="butt", path_effects=[],
                label=r"dual segment of $x_i$ (length $w_i$)" if i == 0 else None)
    ax.scatter(*p.T, color=FG, s=45, depthshade=False, zorder=5, label=r"sampling points $x_i$")
    mids = 0.5 * (p[1:] + p[:-1])
    ax.scatter(*mids.T, color=PALETTE["yellow"], edgecolor=FG, linewidth=0.6, s=40, marker="D", depthshade=False, zorder=6,
               label=r"midpoints $m_{i\pm1/2}$")
    L = tangent_length or 0.9 * np.median(g["line_elements"])
    ax.quiver(*p.T, *(L * g["tangents"]).T, color=PALETTE["red"], lw=1.8, arrow_length_ratio=0.3,
              label=r"unit tangents $\hat t_i$")
    for i in (np.linspace(0, n - 1, 6).round().astype(int) if label_w is None else label_w):
        ax.text(*p[i], f"   $w_{{{i}}}$ = {g['line_elements'][i]:.2f}", color=FG, fontsize=FONT_MIN["annotation"], zorder=7)
    if highlight is not None:
        k = int(highlight)
        ax.plot(*np.stack([g["mid_lo"][k], p[k], g["mid_hi"][k]]).T, color=FG, lw=2.2, ls=(0, (3, 2)), path_effects=[], zorder=8)
        ax.text(*g["mid_hi"][k], f"  dual segment of $x_{{{k}}}$:\n  $m_{{{k}-1/2}} \\to m_{{{k}+1/2}}$", color=FG,
                fontsize=FONT_MIN["annotation"], zorder=8)
    lo, hi = p.min(0), p.max(0)
    pad = 0.12 * (hi - lo).max()
    ax.set(xlim=(lo[0] - pad, hi[0] + pad), ylim=(lo[1] - pad, hi[1] + pad), zlim=(lo[2] - pad, hi[2] + pad),
           xlabel="x [cMpc]", ylabel="y [cMpc]", zlabel="z [cMpc]")
    dark_3d(ax)
    return sm


def _bin_modes(k, P, w, edges):
    """Weighted mean of P (N_real, modes) and its error over realisations in bins of k; returns (k_b, P_b, err, n_modes)."""
    j = np.digitize(k, edges) - 1
    ok = (j >= 0) & (j < len(edges) - 1) & (k > 0)
    nb = len(edges) - 1
    W = np.bincount(j[ok], weights=w[ok], minlength=nb)
    kb = np.bincount(j[ok], weights=(w * k)[ok], minlength=nb) / np.maximum(W, 1e-300)
    Pr = np.stack([np.bincount(j[ok], weights=(w * Pi)[ok], minlength=nb) / np.maximum(W, 1e-300) for Pi in P])
    Pm = Pr.mean(0)
    err = Pr.std(0, ddof=1) / np.sqrt(P.shape[0]) if P.shape[0] > 1 else Pm * np.sqrt(2.0 / np.maximum(W, 1))
    keep = W > 0
    return kb[keep], Pm[keep], err[keep], W[keep]


def power_map_kpar_kperp(m, edges_par, edges_perp):
    """Mean P over draws and modes in bins of (k_par = |k| along the last axis, k_perp = |k| across it).

    m : ``field_power_modes`` output. Returns (P (n_par, n_perp) with NaN where a bin has no mode, n_modes (weighted),
    error of P (scatter between draws), n_indep (independent complex modes of the half space, for the Gaussian error))."""
    kv = m["kvec"]
    kpar, kperp = np.abs(kv[:, -1]), np.linalg.norm(kv[:, :-1], axis=-1)
    ip = np.digitize(kpar, edges_par) - 1
    iq = np.digitize(kperp, edges_perp) - 1
    ok = (ip >= 0) & (ip < len(edges_par) - 1) & (iq >= 0) & (iq < len(edges_perp) - 1) & (m["k"] > 0)
    flat = ip[ok] * (len(edges_perp) - 1) + iq[ok]
    nb = (len(edges_par) - 1) * (len(edges_perp) - 1)
    W = np.bincount(flat, weights=m["w"][ok], minlength=nb)
    Pr = np.stack([np.bincount(flat, weights=(m["w"] * Pi)[ok], minlength=nb) for Pi in m["P"]]) / np.maximum(W, 1e-300)
    Pm = np.where(W > 0, Pr.mean(0), np.nan)
    err = Pr.std(0, ddof=1) / np.sqrt(Pr.shape[0]) if Pr.shape[0] > 1 else Pm * np.sqrt(2.0 / np.maximum(W, 1))
    shp = (len(edges_par) - 1, len(edges_perp) - 1)
    n_ind = np.bincount(flat, minlength=nb)
    return Pm.reshape(shp), W.reshape(shp), np.where(W > 0, err, np.nan).reshape(shp), n_ind.reshape(shp)


def plot_pk_recovery_nd(cases: dict, *, n_bins: int = 18, figsize=(24, 15)):
    """P(k) recovered from GRF realisations in 1D, 2D and 3D. Top: the isotropic P(|k|) against the input, with
    residuals. Bottom (2D and 3D): the measured power in the (k_par, k_perp) plane (k_par along the last axis), as
    colour, with iso-power contours of the input on the same colour levels: for an isotropic spectrum the measured
    colours follow circles, which is what justifies the isotropic average above.

    cases : {title: (grf, deltas (N_real, *shape))}. Returns (fig, axs (3, n_cases), summary dict)."""
    n = len(cases)
    fig, axs = plt.subplots(3, n, figsize=figsize, gridspec_kw={"height_ratios": [2.2, 1, 2.6]}, layout="constrained", squeeze=False)
    out = {}
    for c, (title, (grf, deltas)) in enumerate(cases.items()):
        m = field_power_modes(deltas, grf.L)
        kmin = 2 * np.pi / max(grf.L)
        kmax = min(np.pi * N / L for N, L in zip(grf.shape, grf.L))
        edges = np.geomspace(0.999 * kmin, kmax, n_bins + 1)
        kb, Pb, eb, nb = _bin_modes(m["k"], m["P"], m["w"], edges)
        Pin_b = _bin_modes(m["k"], np.asarray(grf.power(m["k"]))[None], m["w"], edges)[1]
        kk = np.geomspace(kb[0], kb[-1], 300)
        axs[0, c].plot(kk, np.asarray(grf.power(kk)), color=FG, lw=2, label="input P(k)")
        axs[0, c].errorbar(kb, Pb, eb, fmt="o", ms=6, capsize=3, color=ENTITY_COLORS["density"], label=f"measured, |k| ({deltas.shape[0]} draws)")
        axs[1, c].errorbar(kb, Pb / Pin_b - 1, eb / Pin_b, fmt="o", ms=6, capsize=3, color=ENTITY_COLORS["density"])
        out[title] = dict(chi2_dof=float(np.mean(((Pb - Pin_b) / eb) ** 2)), max_rel=float(np.max(np.abs(Pb / Pin_b - 1))))
        axs[0, c].set(xscale="log", yscale="log", title=title, ylabel="P(k)")
        axs[1, c].set(xscale="log", ylabel="measured / input − 1", xlabel=r"$|k|$ [Mpc$^{-1}$]")
        residual_band(axs[1, c], 0.05)
        axs[0, c].legend(fontsize=FONT_MIN["legend"])
        axs[1, c].set_ylim(-0.3, 0.3)
        ax = axs[2, c]
        if grf.ndim == 1:
            ax.set_axis_off()
            ax.text(0.5, 0.5, "1D: the only direction is the line\nof sight, so P(|k|) = P(k$_\\parallel$)",
                    ha="center", va="center", transform=ax.transAxes, fontsize=FONT_MIN["annotation"], color=DIM)
            continue
        dk = 2 * np.pi / min(grf.L)                                  # the coarsest fundamental mode
        e = hybrid_k_edges(dk, kmax, n_lin=6, n_log=20)              # one mode per bin at low k, then log bins
        Pm, W, Em, Nind = power_map_kpar_kperp(m, e, e)
        kc = np.sqrt(e[1:] * e[:-1])
        good = np.isfinite(Pm) & (Pm > 0)
        norm = mpl.colors.LogNorm(np.nanmin(Pm[good]), np.nanmax(Pm[good]))
        cmap_ = plt.get_cmap(CMAPS["sequential"]).copy()
        cmap_.set_bad((0, 0, 0, 0))
        im = ax.pcolormesh(e, e, np.ma.masked_invalid(Pm).T, cmap=cmap_, norm=norm, shading="flat")
        kf = np.geomspace(e[0], e[-1], 300)
        KP, KQ = np.meshgrid(kf, kf, indexing="ij")
        levels = 10.0 ** np.arange(np.ceil(np.log10(norm.vmin)), np.log10(norm.vmax))
        cs = ax.contour(kf, kf, np.asarray(grf.power(np.hypot(KP, KQ))).T, levels=levels, colors=FG, linewidths=1.3,
                        linestyles="--")
        ax.clabel(cs, fmt=lambda x: f"$10^{{{np.log10(x):.0f}}}$", fontsize=FONT_MIN["floor"], inline=False)
        fig.colorbar(im, ax=ax, label="measured P [Mpc$^{d}$]", shrink=0.9)
        Pin_map = np.asarray(grf.power(np.hypot(*np.meshgrid(kc, kc, indexing="ij"))))
        # Gaussian expectation: |delta_k|^2 is exponential, so the mean of n independent modes has error P / sqrt(n)
        sig = Pin_map / np.sqrt(np.maximum(Nind * deltas.shape[0], 1))
        well = good & (Nind > 0)
        out[title]["map_chi2_dof"] = float(np.mean(((Pm[well] - Pin_map[well]) / sig[well]) ** 2))
        ax.set(xscale="log", yscale="log", xlim=(e[0], e[-1]), ylim=(e[0], e[-1]), aspect="equal",
               xlabel=r"$k_\parallel$ (last axis) [Mpc$^{-1}$]",
               ylabel=r"$k_\perp$ [Mpc$^{-1}$]", title="measured P in the $(k_\\parallel, k_\\perp)$ plane; dashed: input")
    return fig, axs, out


def plot_nH_T_fractions(n_H, T, x_HI, f_SiIII, sf, cloudy_grid, *, n_thr: float = None, bins: int = 110, figsize=(24, 9.5)):
    """The n_H-T plane of the cells, H I and Si III kept apart.

    (a) H I: every (n_H, T) bin coloured by the median log10 x_HI of its cells, with contours of the number of cells;
    (b) Si III: the precomputed CLOUDY table log10 f_SiIII(n_H, T) as the background, the same cell contours on top.
    Text boxes name the visible features: the star-formation threshold (``n_thr``), the effective equation of state of
    the star-forming cells (a power-law fit T(n_H), printed), the photo-heated gas near 10^4 K and the hot halo gas.

    n_H [cm^-3], T [K], x_HI, f_SiIII, sf (bool) : per cell; cloudy_grid : (log10 hden (N_d,), log10 T (N_T,),
    log10 f (N_d, N_T)). Returns (fig, axs, info) with the EOS fit and the CLOUDY maximum.
    """
    from scipy.stats import binned_statistic_2d
    lx, ly = np.log10(n_H), np.log10(T)
    ex = np.linspace(max(lx.min(), -7.5), min(lx.max(), 4.0), bins + 1)
    ey = np.linspace(max(ly.min(), 3.0), min(ly.max(), 8.0), bins + 1)
    cnt, _, _ = np.histogram2d(lx, ly, [ex, ey])
    cx, cy = 0.5 * (ex[1:] + ex[:-1]), 0.5 * (ey[1:] + ey[:-1])
    fig, axs = plt.subplots(1, 2, figsize=figsize, layout="constrained")
    med = binned_statistic_2d(lx, ly, np.log10(np.maximum(x_HI, 1e-12)), "median", bins=[ex, ey]).statistic
    cm0 = mpl.colormaps[CMAPS["sequential"]].copy()
    cm0.set_bad((0, 0, 0, 0))                                    # empty bins: the background
    im0 = axs[0].pcolormesh(ex, ey, np.ma.masked_invalid(med.T), cmap=cm0, vmin=-7, vmax=0, shading="flat", rasterized=True)
    fig.colorbar(im0, ax=axs[0], label=r"median $\log_{10} x_{\rm HI}$ of the cells in the bin")
    ld, lt, lf = cloudy_grid
    im1 = axs[1].pcolormesh(ld, lt, np.maximum(lf, -8).T, cmap=CMAPS["hot"], vmin=-8, vmax=0, shading="nearest", rasterized=True)
    fig.colorbar(im1, ax=axs[1], label=r"CLOUDY $\log_{10} f_{\rm SiIII}(n_{\rm H}, T)$ (table, at the snapshot redshift)")
    lev = np.log10(np.maximum(cnt, 0.5))
    for ax in axs:
        ax.contour(cx, cy, lev.T, levels=[0.0, 1.0, 2.0, 3.0], colors=[FG], linewidths=[0.8, 1.1, 1.4, 1.8])
        ax.set(xlabel=r"$\log_{10} n_{\rm H}$ [cm$^{-3}$]", ylabel=r"$\log_{10} T$ [K]", xlim=(ex[0], ex[-1]), ylim=(ey[0], ey[-1]))
    info = {}
    if sf is not None and np.any(sf):
        p = np.polyfit(lx[sf], ly[sf], 1)
        info["eos"] = dict(slope=float(p[0]), intercept=float(p[1]), scatter=float(np.std(ly[sf] - np.polyval(p, lx[sf]))))
        xs = np.array([lx[sf].min(), ex[-1]])
        for ax in axs:
            ax.plot(xs, np.polyval(p, xs), color=PALETTE["cyan"], lw=2.2, ls="--", path_effects=[])
        _textbox(axs[0], xs[0] + 0.25, np.polyval(p, xs[0] + 0.25) + 0.45,
                 f"star-forming cells: effective EOS\nT ∝ n_H^{p[0]:.2f} (±{info['eos']['scatter']:.2f} dex)", color=PALETTE["cyan"])
    if n_thr is not None:
        for ax in axs:
            ax.axvline(np.log10(n_thr), color=PALETTE["yellow"], ls=":", lw=2.0, path_effects=[])
        _textbox(axs[0], np.log10(n_thr) + 0.1, ey[-1] - 0.35, f"star-formation\nthreshold {n_thr:.2f} cm⁻³", color=PALETTE["yellow"])
    _textbox(axs[0], ex[0] + 0.3, 3.75, "photo-heated gas:\nfloor near 10⁴ K", color=FG)
    _textbox(axs[0], ex[0] + 0.3, 6.6, "hot, shock-heated\nhalo gas (x_HI << 1)", color=FG)
    i, j = np.unravel_index(np.argmax(lf), lf.shape)
    info["cloudy_max"] = dict(log_n_H=float(ld[i]), log_T=float(lt[j]), log_f=float(lf[i, j]))
    _textbox(axs[1], max(ex[0] + 0.3, min(ld[i], ex[-1]) - 2.2), lt[j] - 0.55,
             f"Si III ridge: most abundant near\nlog T = {lt[j]:.2f} (log n_H = {ld[i]:.1f})", color=FG)
    j_lo = np.argmax(lf[np.argmin(np.abs(ld + 3.0))])
    _textbox(axs[1], ex[0] + 0.3, ey[0] + 0.25, f"photoionised at low n_H (at log n_H = −3: peak near log T = {lt[j_lo]:.2f})", color=FG)
    axs[1].set_xlim(ex[0], ex[-1]); axs[1].set_ylim(ey[0], ey[-1])
    axs[0].set_title(r"H I: stored ionisation equilibrium (star-forming cells: cold phase)")
    axs[1].set_title(r"Si III: CLOUDY table (background) and the cells (contours)")
    return fig, axs, info


def plot_deposition_story(seed: int = 3, n_cells: int = 90, every: int = 25, H_kms_per_unit: float = 20.0, b: float = 9.0,
                          figsize=(20, 22)):
    """Selection -> deposition -> integration on a toy 2D Voronoi mesh, one figure.

    (a) the cells, filled by their density; the path, its sampling points and the ends of their dual segments; every
        piece of a dual segment drawn in the colour of the cell that fills it;
    (b) along the path: the cell values (pieces, same colours), and the four ways to assign the gas to the points with
        their column: exact pieces (Voronoi default) = segment averages x w_i, point values x w_i, top-hat discs of
        radius sqrt(A) (top-hat spheres, in 2D), and the cell crossings (Voronoi-adapted);
    (c) the integration: every piece p adds d tau_p(v) = C n_p l_p a_p^-2 phi_p(v), centred at the line centre of its
        centroid (v = H s here: no peculiar velocity, a = 1), on a pixel grid that knows nothing of the cells or the
        points (the derivation is notebook 06).
    Returns (fig, axs, columns).
    """
    from matplotlib.collections import LineCollection, PolyCollection
    from scipy.spatial import Voronoi
    rng = np.random.default_rng(seed)
    gen = np.vstack([rng.uniform(0, 10, (int(0.6 * n_cells), 2)), rng.normal([5, 5], 0.8, (n_cells - int(0.6 * n_cells), 2))])
    dens = (0.12 + 1.2 * np.exp(-((gen - 5) ** 2).sum(1) / 1.5)) * rng.uniform(0.7, 1.0, len(gen))   # a dense filament
    cmap = mpl.colormaps[CMAPS["sequential"]]
    norm = mpl.colors.Normalize(0, dens.max())
    col = lambda c: cmap(0.12 + 0.88 * norm(dens[c]))
    tt = np.linspace(0, 10, 16 * every + 1)
    path = np.stack([tt, 5 + 1.2 * np.sin(tt / 2.2)], 1)
    nearest = lambda P: np.argmin(((P[:, None, :] - gen[None]) ** 2).sum(-1), axis=1)
    sp = path[::every]
    s_path = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
    s_sp = s_path[::every]
    edges = np.concatenate([[0], 0.5 * (s_sp[1:] + s_sp[:-1]), [s_path[-1]]])
    w = np.diff(edges)
    # fine sampling of the path: pieces = runs of one cell inside one dual segment
    fine_s = np.linspace(0, s_path[-1], 20001)
    fine_xy = np.stack([np.interp(fine_s, s_path, path[:, 0]), np.interp(fine_s, s_path, path[:, 1])], 1)
    lab = nearest(fine_xy)
    seg = np.clip(np.searchsorted(edges, fine_s, side="right") - 1, 0, len(w) - 1)
    ds = np.diff(fine_s)
    lab_m, seg_m, s_m = lab[:-1], seg[:-1], 0.5 * (fine_s[1:] + fine_s[:-1])
    cuts = np.nonzero((lab_m[1:] != lab_m[:-1]) | (seg_m[1:] != seg_m[:-1]))[0] + 1
    starts, ends = np.concatenate([[0], cuts]), np.concatenate([cuts, [lab_m.size]])
    pieces = [(fine_s[a], fine_s[e], lab_m[a], seg_m[a]) for a, e in zip(starts, ends)]
    cuts_c = np.nonzero(lab_m[1:] != lab_m[:-1])[0] + 1
    cross = [(fine_s[a], fine_s[e], lab_m[a]) for a, e in zip(np.concatenate([[0], cuts_c]), np.concatenate([cuts_c, [lab_m.size]]))]
    exact = float(np.sum(dens[lab_m] * ds))
    seg_mean = np.array([sum(dens[c] * (e - a) for a, e, c, k in pieces if k == i) / w[i] for i in range(len(w))])
    n_pts = dens[nearest(sp)]
    # top hats: discs of radius sqrt(A_c) and density n_c / pi (each holds the cell's n A), cell areas by a fine grid
    gx = np.linspace(0, 10, 500)
    G = np.stack(np.meshgrid(gx, gx), -1).reshape(-1, 2)
    area = np.bincount(np.concatenate([nearest(G[i:i + 50000]) for i in range(0, len(G), 50000)]), minlength=len(gen)) * (gx[1] - gx[0]) ** 2
    h = np.sqrt(area)
    d2 = ((fine_xy[:, None, :] - gen[None]) ** 2).sum(-1)
    n_top = (np.where(d2 <= h[None] ** 2, dens[None] / np.pi, 0.0)).sum(1)
    top = float(np.sum(0.5 * (n_top[1:] + n_top[:-1]) * ds))
    cols = dict(exact=exact, segments=float(np.sum(seg_mean * w)), points=float(np.sum(n_pts * w)), tophat=top,
                crossings=float(sum(dens[c] * (e - a) for a, e, c in cross)), n_pieces=len(pieces), n_crossings=len(cross))
    fig, axs = plt.subplots(3, 1, figsize=figsize, gridspec_kw={"height_ratios": [2.6, 1.7, 1.5]}, layout="constrained")
    ax = axs[0]
    vor = Voronoi(np.vstack([gen, [[-50, -50], [-50, 60], [60, -50], [60, 60]]]))
    polys, fc = [], []
    for c in range(len(gen)):
        reg = vor.regions[vor.point_region[c]]
        if -1 not in reg and reg:
            polys.append(vor.vertices[reg]); fc.append(col(c))
    ax.add_collection(PolyCollection(polys, facecolors=fc, edgecolors=BG, linewidths=0.8, alpha=0.55))
    ax.plot(*gen.T, ".", color=FG, ms=5)
    segs = [np.stack([np.interp([a, e], s_path, path[:, 0]), np.interp([a, e], s_path, path[:, 1])], 1) for a, e, c, k in pieces]
    ax.add_collection(LineCollection(segs, colors=BG, linewidths=13, path_effects=[]))
    ax.add_collection(LineCollection(segs, colors=[col(c) for a, e, c, k in pieces], linewidths=8, path_effects=[]))
    ax.plot(*path.T, color=FG, lw=1.0)
    ax.plot(*sp.T, "o", mfc=BG, mec=FG, mew=2, ms=12, zorder=5)
    mids = np.stack([np.interp(edges[1:-1], s_path, path[:, 0]), np.interp(edges[1:-1], s_path, path[:, 1])], 1)
    ax.plot(*mids.T, "|", color=PALETTE["yellow"], ms=26, mew=3, zorder=6)
    ax.set(xlim=(0, 10), ylim=(2.3, 8.0), aspect="equal", xlabel="x", ylabel="y")
    sm = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, dens.max()), cmap=mpl.colors.ListedColormap([cmap(0.12 + 0.88 * u) for u in np.linspace(0, 1, 256)]))
    fig.colorbar(sm, ax=ax, label="cell density n", shrink=0.8)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker=".", ls="none", color=FG, ms=10, label="mesh-generating points (one cell each)"),
                       Line2D([], [], marker="o", ls="none", mfc=BG, mec=FG, mew=2, ms=12, label="sampling points $x_i$"),
                       Line2D([], [], marker="|", ls="none", color=PALETTE["yellow"], ms=18, mew=3, label="midpoints: ends of the dual segments"),
                       Line2D([], [], color=cmap(0.7), lw=9, label="pieces: cell ∩ dual segment, in the cell's colour")],
              loc="upper center", bbox_to_anchor=(0.5, -0.1), ncols=2, fontsize=FONT_MIN["legend"])
    ax.set_title("(a) Selection and deposition: which cell fills which piece of every dual segment")
    ax = axs[1]
    for a, e, c, k in pieces:
        ax.fill_between([a, e], 0, dens[c], color=col(c), alpha=0.9, lw=0, step="pre", path_effects=[])
    ax.bar(s_sp, seg_mean, width=w, fill=False, edgecolor=PALETTE["green"], lw=2.2, label=rf"segment averages $\bar n_i$ over $w_i$: N = {cols['segments']:.3f}")
    ax.plot(s_sp, n_pts, "D", color=PALETTE["red"], ms=10, mec=BG, label=f"values at the points × $w_i$: N = {cols['points']:.3f}")
    ax.plot(fine_s, n_top, color=PALETTE["yellow"], lw=1.6, label=f"top-hat discs (each cell a uniform disc): N = {cols['tophat']:.3f}")
    for a, e, c in cross:
        ax.plot([a, e], [-0.09 * dens.max()] * 2, color=col(c), lw=6, solid_capstyle="butt", path_effects=[])
        ax.plot([e, e], [-0.14 * dens.max(), -0.04 * dens.max()], color=FG, lw=1.0, path_effects=[])
    ax.plot([], [], color=cmap(0.7), lw=6, label=f"cell crossings (bar below 0; Voronoi-adapted, {len(cross)} pieces): N = {cols['crossings']:.3f}")
    for e in edges:
        ax.axvline(e, color=PALETTE["yellow"], lw=0.8, ls=":", path_effects=[])
    ax.axhline(0, color=FG, lw=0.8)
    ax.set(xlabel="path length s", ylabel="n", xlim=(0, s_path[-1]), ylim=(-0.16 * dens.max(), 1.5 * dens.max()))
    ax.set_title(f"(b) Along the path: the exact column N = ∫n ds = {exact:.3f}; how each assignment reproduces it")
    ax.legend(loc="upper center", fontsize=FONT_MIN["legend"], ncols=2)
    ax = axs[2]
    vv = np.linspace(-3 * b, H_kms_per_unit * s_path[-1] + 3 * b, 1500)
    tot = np.zeros_like(vv)
    for a, e, c, k in pieces:
        prof = dens[c] * (e - a) * np.exp(-((vv - H_kms_per_unit * 0.5 * (a + e)) / b) ** 2) / (np.sqrt(np.pi) * b)
        tot += prof
        ax.plot(vv, prof, color=col(c), lw=1.6)
    ax.plot(vv, tot, color=FG, lw=1.6, ls="--", label=r"$\tau(v) = \sum_p d\tau_p(v)$")
    ax.plot(H_kms_per_unit * np.array([0.5 * (a + e) for a, e, c, k in pieces]), np.full(len(pieces), -0.03 * tot.max()), "|",
            color=FG, ms=12, mew=1.5, label=r"line centres $v_{c,p}$ of the piece centroids")
    ax.vlines(np.arange(vv[0], vv[-1], 2.2 * b), 0, 1.05 * tot.max(), color=DIM, lw=1, ls=":", path_effects=[], label="pixels of the spectral grid")
    ax.set(xlabel=r"$v$ [km s$^{-1}$]; piece $p$ at $v_{c,p} = c\,\ln(a_\dagger/a(s_p)) + v_{\parallel,p}$ ($= H s_p$ here)",
           ylabel=r"$d\tau_p(v) = \mathcal{C}\, n_p\, \ell_p\, a_p^{-2}\, \phi_p(v)$", ylim=(-0.07 * tot.max(), None))
    ax.set_title("(c) Integration: each piece adds its own profile, centred where its centroid absorbs; the pixels are independent")
    ax.legend(loc="upper right", fontsize=FONT_MIN["legend"])
    return fig, axs, cols


def plot_voigt_limits(x, sigma, gamma, curves: dict, ref_key: str = "scipy_wofz", *, linthresh: float = 1e-6, figsize=(15, 11)):
    """Voigt profiles V(x; sigma, gamma) for extreme (sigma, gamma) pairs, every implementation in ``curves`` = {name:
    (N_pairs, N_x)} on the reference ``ref_key`` (thick, light), and their relative residuals (symlog, ``linthresh``).
    The pairs show the Gaussian limit (gamma << sigma), the Lorentzian limit (sigma << gamma) and the mixed cases."""
    from matplotlib.lines import Line2D
    fig, (ax, axr) = plt.subplots(2, 1, figsize=figsize, sharex=True, gridspec_kw={"height_ratios": [2.2, 1]}, layout="constrained")
    cols = [PALETTE[k] for k in ("blue", "red", "green", "yellow", "cyan", "purple")]
    styles = {name: st for name, st in zip([k for k in curves if k != ref_key], (MODE_STYLES["default"], MODE_STYLES["matched"], MODE_STYLES["alternative"]))}
    ref = np.asarray(curves[ref_key])
    for i, (s, g) in enumerate(zip(np.asarray(sigma), np.asarray(gamma))):
        c = cols[i % len(cols)]
        kind = "Gaussian limit" if g < 0.1 * s else ("Lorentzian limit" if s < 0.1 * g else "mixed")
        ax.semilogy(x, ref[i], color=c, **MODE_STYLES["reference"], label=rf"$\sigma$ = {s:g}, $\gamma$ = {g:g} ({kind})")
        for name, st in styles.items():
            y = np.asarray(curves[name])[i]
            ax.semilogy(x, y, color=c, **st)
            axr.plot(x, y / ref[i] - 1, color=c, **st)
    ax.set(ylim=(1e-6, 1e3), ylabel=r"$V(x;\sigma,\gamma)$")
    leg = ax.legend(loc="upper left", fontsize=FONT_MIN["legend"])
    ax.add_artist(leg)
    ax.legend(handles=[Line2D([], [], color=FG, **MODE_STYLES["reference"], label=f"{ref_key} (reference)")] +
                      [Line2D([], [], color=FG, **st, label=name) for name, st in styles.items()], loc="upper right", fontsize=FONT_MIN["legend"])
    axr.set_yscale("symlog", linthresh=linthresh)
    axr.set_yticks([-1e-2, -1e-4, -linthresh, 0, linthresh, 1e-4, 1e-2])
    axr.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(ten_power_formatter))
    axr.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    axr.set(xlabel="x", ylabel="V / reference − 1", ylim=(-0.05, 0.05))
    axr.axhline(0, color=FG, lw=1)
    return fig, np.array([ax, axr])


def plot_frames_axes(v, F_own, line_bundle, *, a_dagger: float, a_obs: float = 1.0, centres=None, figsize=(22, 15)):
    """How line-centred velocities relate to the global velocity axis and to the observed wavelength (flux only).

    v : (N_v,) uniform grid of line-centred velocities v_dagger [km/s] (one period of the box); F_own : (N_lines, N_v)
    the flux of every line in its own frame. (a) observed wavelength lambda_obs = lambda_j (a_obs / a_dagger) e^{v/c};
    (b) the global (reference-line) axis: ``wrap_to_ref_line`` (solid, periodic on the grid) against the unwrapped shift
    v - Delta v_j (dashed, outside the grid where it leaves it; the grid is shaded); (c, larger) the line-centred frames:
    the same gas absorbs at the same v_dagger in every line (dotted: the absorber centres).
    ``centres`` : optional absorber centres [km/s] to mark. Returns (fig, axs).
    """
    from skewerjax.core.constants import phys
    from skewerjax.core.optical_depth import wrap_to_ref_line
    v = np.asarray(v, dtype=float)
    F_own = np.asarray(F_own, dtype=float)
    lines = line_bundle.lines
    dv_j = np.asarray(line_bundle.delta_v, dtype=float)
    F_wrap = np.exp(-np.asarray(wrap_to_ref_line(-np.log(np.maximum(F_own, 1e-300))[:, None], v, line_bundle))[:, 0])
    cols = [ENTITY_COLORS.get(l.id, species_color(l.id, j)) for j, l in enumerate(lines)]
    fig, axs = plt.subplots(3, 1, figsize=figsize, gridspec_kw={"height_ratios": [1.2, 1.4, 2.2]}, layout="constrained")
    for j, l in enumerate(lines):
        lam = l.lambda0_A * (a_obs / a_dagger) * np.exp(v / phys.c_kms)
        axs[0].plot(lam, F_own[j], color=cols[j], label=f"{l.id}: λ₀ (a_obs/a†) e^(v†/c)")
        axs[0].axvline(l.lambda0_A * a_obs / a_dagger, color=cols[j], ls="-.", lw=1.2, path_effects=[])
        axs[1].plot(v, F_wrap[j], color=cols[j], lw=2.2, label=f"{l.id}: wrap_to_ref_line (periodic)")
        axs[1].plot(v - dv_j[j], F_own[j], color=cols[j], lw=1.6, ls="--", alpha=0.8,
                    label=f"{l.id}: unwrapped, v† − Δv = v† {-dv_j[j]:+.0f} km/s" if dv_j[j] != 0 else None)
        axs[2].plot(v, F_own[j], color=cols[j], lw=2.4 if j == 0 else 1.8, ls="-" if j == 0 else "--", label=f"{l.id}, own frame")
    axs[1].axvspan(v[0], v[-1], color=DIM, alpha=0.10, lw=0)
    _textbox(axs[1], v[0] + 0.01 * (v[-1] - v[0]), 0.3, "the spectral grid (shaded)", color=FG)
    if centres is not None:
        for c0 in np.atleast_1d(centres):
            axs[2].axvline(c0, color=FG, ls=":", lw=1.2, path_effects=[])
    for j in range(1, len(lines)):
        _textbox(axs[2], 0.02, 0.12 + 0.1 * j, f"{lines[j].id} sits Δv = {dv_j[j]:+.1f} km/s from {lines[0].id} on the global axis "
                 f"(c ln(λ_ref/λ_j)); in its own frame it absorbs at the same v† as {lines[0].id}", color=cols[j],
                 transform=axs[2].transAxes)
    axs[0].set(xlabel=r"observed wavelength $\lambda_{\rm obs}$ [Å]", ylabel=LABELS["F"])
    axs[1].set(xlabel=f"global velocity axis: v on the {lines[0].id} frame [km/s]", ylabel=LABELS["F"])
    axs[2].set(xlabel=r"line-centred velocity $v_\dagger$ [km/s] (each line in its own frame)", ylabel=LABELS["F"], xlim=(v[0], v[-1]))
    for ax in axs:
        ax.set_ylim(-0.03, 1.08)
        ax.legend(loc="lower left" if ax is axs[1] else "lower right", fontsize=FONT_MIN["legend"])
    return fig, axs


def plot_two_elements(r: dict, *, figsize=(22, 15)):
    """Two gas elements on opposite sides of a reference point with opposite peculiar velocities (``tutorial.two_elements``):
    (a) where they are and where they absorb (the Hubble line centre v_H(x) and the line centres with the peculiar
    velocity); (b) tau from a fine quadrature of the integral, the package's sum and the two per-element parts;
    (c) the error of the sum with K pieces per element. Returns (fig, axs)."""
    fig, axs = plt.subplots(3, 1, figsize=figsize, gridspec_kw={"height_ratios": [1.2, 1.5, 1.0]}, layout="constrained")
    cA, cB = PALETTE["blue"], PALETTE["red"]
    ax = axs[0]
    x, vH = r["x_grid"], r["v_hubble"]
    sel = (x > r["x_c"][0] - 1.2) & (x < r["x_c"][1] + 1.2)
    ax.plot(x[sel], vH[sel], color=DIM, lw=2, label=r"$v_H(x) = c\ln(a_\dagger/a(x))$: gas at rest")
    for e, (c, name) in enumerate(((cA, "A"), (cB, "B"))):
        xc, h = r["x_c"][e], 0.5 * r["length"]
        ax.axvspan(xc - h, xc + h, color=c, alpha=0.25, lw=0)
        vh = np.interp(xc, x, vH)
        ax.annotate("", (xc, r["v_star"]), (xc, vh), arrowprops=dict(arrowstyle="-|>", color=c, lw=2.2))
        ax.plot([xc], [r["v_star"]], "o", color=c, ms=11, mec=FG,
                label=rf"{name}: $n$ = {r['n_com'][e]:.0e}, $T$ = {r['T'][e]:.0e} K, $u$ = {r['u'][e]:+.0f} km/s")
    ax.axhline(r["v_star"], color=FG, ls=":", lw=1.2)
    ax.text(r["x_c"].mean(), r["v_star"] + 8, rf"both line centres: $v_\dagger$ = {r['v_star']:.1f} km/s", ha="center",
            va="bottom", color=FG, fontsize=FONT_MIN["annotation"])
    ax.set(xlabel=r"comoving distance from the observer $x$ [cMpc]", ylabel=r"line centre $v_c$ [km s$^{-1}$]",
           title="(a) A (nearer, receding) and B (farther, approaching): the arrows are their peculiar velocities")
    ax.legend(loc="upper left", fontsize=FONT_MIN["legend"])
    ax = axs[1]
    v = r["v"]
    Kmax = max(r["tau_sum"])
    ax.plot(v, r["tau_ref"], color=FG, **MODE_STYLES["reference"], label="integral: fine quadrature (scipy wofz)")
    ax.plot(v, r["tau_sum"][Kmax], color=FG, lw=1.8, label=rf"the code's sum, {Kmax} pieces per element")
    for e, c, name in ((0, cA, "A"), (1, cB, "B")):
        ax.plot(v, r["tau_ref_parts"][e], color=c, lw=4, alpha=0.35)
        ax.plot(v, r["tau_sum_parts"][Kmax][e], color=c, ls="--", lw=1.8, label=f"element {name} alone")
    ax.set(xlabel=r"$v_\dagger$ [km s$^{-1}$]", ylabel=r"$\tau$", yscale="log", ylim=(1e-3 * r["tau_ref"].max(), 2 * r["tau_ref"].max()),
           title=r"(b) $\tau = \tau_A + \tau_B$: each element with its own $n$, $T$, $u$ and $a$, at the same $v_\dagger$")
    ax.legend(loc="upper right", fontsize=FONT_MIN["legend"])
    ax = axs[2]
    for K, c in zip(sorted(r["tau_sum"]), (PALETTE["purple"], PALETTE["green"], PALETTE["cyan"], PALETTE["yellow"])):
        err = (r["tau_sum"][K] - r["tau_ref"]) / r["tau_ref"].max()
        ax.plot(v, err, color=c, lw=1.8, label=rf"$K$ = {K}: $W/b_A$ = {r['hubble_width_kms'] / K / r['b_kms'][0]:.2f}, max = {np.abs(err).max():.1e}")
    ax.axhline(0, color=FG, lw=0.8)
    ax.set(xlabel=r"$v_\dagger$ [km s$^{-1}$]", ylabel=r"(sum $-$ integral) / $\tau_{\max}$", yscale="symlog",
           title=r"(c) The error of the sum: the midpoint rule on every piece, $\propto (W/b)^2$ with $W = (dv_H/dx)\,\ell_p$")
    ax.yaxis.set_major_locator(mpl.ticker.SymmetricalLogLocator(base=10, linthresh=1e-5, subs=[1.0]))
    ax.set_yscale("symlog", linthresh=1e-5)
    ax.legend(loc="upper right", fontsize=FONT_MIN["legend"])
    return fig, axs


def plot_linear_detuning(d: dict, *, figsize=(22, 7.5)):
    """The linear detuning against the exact one for one absorber (``tutorial.linear_detuning``): (a) F of the
    strongest column, exact and linear; (b) tau_linear / tau_exact - 1 against v - v_c, with the wing prediction
    -(v - v_c)/c; (c) |F_linear - F_exact| for every column. Returns (fig, axs)."""
    from skewerjax.core.constants import phys
    v, te, tl = d["v"], d["tau_exact"], d["tau_linear"]
    cols = [PALETTE[k] for k in ("blue", "green", "red", "purple")]
    fig, axs = plt.subplots(1, 3, figsize=figsize, layout="constrained")
    axs[0].plot(v, np.exp(-te[-1]), color=FG, **MODE_STYLES["reference"], label="exact detuning")
    axs[0].plot(v, np.exp(-tl[-1]), color=cols[2], ls="--", lw=1.8, label="linear detuning")
    axs[0].set(xlabel=r"$v_\dagger - v_c$ [km s$^{-1}$]", ylabel=LABELS["F"], ylim=(-0.03, 1.05),
               title=rf"(a) $\log_{{10}} N$ = {d['log_N'][-1]:.1f}: the damping wings")
    rel = tl[0] / te[0] - 1
    ok = te[0] > 1e-300
    axs[1].plot(v[ok], rel[ok], color=cols[0], lw=2, label=r"$\tau_{\rm lin}/\tau_{\rm exact} - 1$ (any column)")
    axs[1].plot(v, -v / phys.c_kms, color=FG, ls=":", lw=1.6, label=r"wings: $-(v_\dagger - v_c)/c$")
    axs[1].set(xlabel=r"$v_\dagger - v_c$ [km s$^{-1}$]", ylabel="relative change of $\\tau$", yscale="symlog",
               title="(b) the relative error of $\\tau$")
    axs[1].set_yscale("symlog", linthresh=1e-4)
    for i, lN in enumerate(d["log_N"]):
        axs[2].semilogy(v, np.abs(np.exp(-tl[i]) - np.exp(-te[i])) + 1e-12, color=cols[i], lw=1.8, label=rf"$\log_{{10}} N$ = {lN:.1f}")
    axs[2].set(xlabel=r"$v_\dagger - v_c$ [km s$^{-1}$]", ylabel=r"$|F_{\rm lin} - F_{\rm exact}|$", ylim=(1e-8, 1e-2),
               title="(c) what it does to the flux")
    for ax in axs:
        ax.legend(loc="lower center" if ax is axs[1] else "lower right", fontsize=FONT_MIN["legend"])
    return fig, axs


def plot_periodicity_options(p: dict, *, figsize=(22, 17)):
    """The periodicity options on one absorber near the far edge of a periodic box (``tutorial.periodic_box_demo``):
    (a) no wrap, (b) detunings wrapped modulo P, (c) a buffer of periodic gas, (d) the post-processing shift of a
    second line onto the reference frame (``wrap_to_ref_line``). Returns (fig, axs)."""
    v, P, t = p["v"], p["period_kms"], p["tau"]
    fig, axs = plt.subplots(4, 1, figsize=figsize, layout="constrained")
    shade = lambda ax: ax.axvspan(0, P, color=DIM, alpha=0.12, lw=0)
    ext = (p["v_ext"] >= -0.6 * P) & (p["v_ext"] <= 1.6 * P)
    full = p["tau_full"]
    ax = axs[0]; shade(ax)
    ax.plot(p["v_full"], full, color=FG, ls=":", lw=1.4, label="the absorber's whole profile (it does not stop at $P$)")
    ax.plot(v, t["none"], color=PALETTE["blue"], lw=2.4, label="no wrap: what lands on the grid $[0, P)$")
    ax.set_title(r"(a) No wrap: the profile past $v = P$ falls off the grid and is lost")
    ax = axs[1]; shade(ax)
    ax.plot(v, t["wrapped"], color=PALETTE["green"], lw=2.4, label=r"detunings wrapped: $v_\dagger - v_c \to$ its image in $[-P/2, P/2)$")
    ax.plot(v, t["none"], color=PALETTE["blue"], lw=1.2, ls="--", label="no wrap")
    ax.annotate("", (0.03 * P, 0.6 * t["wrapped"].max()), (0.97 * P, 0.6 * t["wrapped"].max()),
                arrowprops=dict(arrowstyle="-|>", color=FG, lw=1.6, connectionstyle="arc3,rad=0.12"))
    ax.text(0.5 * P, 0.68 * t["wrapped"].max(), "the overflow re-enters at the near end", ha="center", color=FG,
            fontsize=FONT_MIN["annotation"])
    ax.set_title(r"(b) Detunings wrapped modulo the box period $P$ (velocity_period $= P$)")
    ax = axs[2]; shade(ax)
    ax.plot(p["v_ext"][ext], p["tau_ext"][ext], color=PALETTE["purple"], lw=1.4, label="buffer: the path extended by periodic copies of the gas")
    ax.plot(v, t["buffer"], color=PALETTE["purple"], lw=2.6, label=r"read on $[0, P)$")
    ax.plot(v, t["wrapped"], color=PALETTE["green"], lw=1.2, ls="--", label="detunings wrapped (b)")
    for xc in p["copies"]:
        ax.axvline(p["v_of_x"](xc), color=PALETTE["purple"], ls=":", lw=1.2)
    ax.set_title("(c) A buffer: a longer, non-periodic path through the periodic gas, read on one period")
    ax = axs[3]; shade(ax)
    cH, cS = ENTITY_COLORS["HI-LyA"], ENTITY_COLORS["SiIII-1206"]
    ax.plot(v, p["two_lines_own"][1], color=cS, lw=1.4, ls="--", label="Si III in its own frame")
    ax.plot(v, p["two_lines_ref"][0], color=cH, lw=2.2, label=r"Ly$\alpha$ (reference frame)")
    ax.plot(v, p["two_lines_ref"][1], color=cS, lw=2.4, label=r"Si III shifted by $-\Delta v$ modulo $P$ (wrap_to_ref_line)")
    ax.plot(v - p["delta_v"], p["two_lines_own"][1], color=cS, lw=1.2, ls=":", label=r"the shift without the modulo")
    ax.set_title(rf"(d) One axis for both lines: $-\Delta v$ = {-p['delta_v']:.0f} km/s is {abs(p['delta_v']) / P:.2f} periods")
    for ax in axs:
        ax.set(xlim=(-0.55 * P, 1.55 * P) if ax is not axs[3] else (-1.15 * P, 1.1 * P),
               xlabel=r"$v_\dagger$ [km s$^{-1}$] (grey: the grid, one period $[0, P)$)", ylabel=r"$\tau$")
        ax.legend(loc="upper left" if ax is not axs[3] else "upper right", fontsize=FONT_MIN["legend"])
    return fig, axs


def field_power_modes(deltas, L):
    """|delta_k|^2 / V of every mode of a batch of periodic fields (``PowerLawGRF``'s normalisation).

    deltas : (N_real, n_1, ..., n_d) on a box of sides ``L`` (scalar or (d,)). Returns dict: ``k`` |k| (modes,),
    ``kvec`` (modes, d), ``P`` (N_real, modes), ``w`` (modes,) half-space multiplicity (1 on self-conjugate planes)."""
    d = np.asarray(deltas, dtype=float)
    shape = d.shape[1:]
    Ls = np.broadcast_to(np.asarray(L, dtype=float), (len(shape),))
    dV = np.prod(Ls / np.asarray(shape))
    V = np.prod(Ls)
    dk = np.fft.rfftn(d, axes=tuple(range(1, d.ndim))) * dV
    P = np.abs(dk) ** 2 / V
    axes_k = [2 * np.pi * np.fft.fftfreq(n, l / n) for n, l in zip(shape[:-1], Ls[:-1])]
    axes_k.append(2 * np.pi * np.fft.rfftfreq(shape[-1], Ls[-1] / shape[-1]))
    kv = np.meshgrid(*axes_k, indexing="ij")
    kvec = np.stack([k.ravel() for k in kv], -1)
    last = kv[-1].ravel()
    nyq = np.pi * shape[-1] / Ls[-1]
    w = np.where((last == 0) | (np.isclose(last, nyq) & (shape[-1] % 2 == 0)), 1.0, 2.0)
    return dict(k=np.linalg.norm(kvec, axis=-1), kvec=kvec, P=P.reshape(d.shape[0], -1), w=w)


def binned_power(deltas, L, edges, *, which: str = "iso"):
    """Binned P of a batch of periodic fields: (k_b, P_b, err, n_modes). which: "iso" (|k|), "par" (modes with
    k_perp = 0, along the last axis) or "perp" (modes with k_par = 0)."""
    m = field_power_modes(deltas, L)
    k, P, w = m["k"], m["P"], m["w"]
    if which != "iso":
        perp = np.linalg.norm(m["kvec"][:, :-1], axis=-1)
        sel = np.isclose(perp, 0.0) if which == "par" else np.isclose(m["kvec"][:, -1], 0.0)
        k = np.abs(m["kvec"][sel, -1]) if which == "par" else perp[sel]
        P, w = P[:, sel], w[sel]
    return _bin_modes(k, P, w, edges)


def plot_effect_fluxes(v, tau_base, isolated: dict, *, window=None, skewer: int = 0, ncols: int = 4, figsize=None,
                       color=None, base_label: str = "fake_spectra options"):
    """One pair of panels per isolated switch: the flux with fake_spectra's options (thick, light) and with that one
    switch set to SkewerJAX's value (solid) on ``window`` [km/s], and below their difference. isolated : {label: tau}
    of ``run_effects(...)["isolated"]``, (N_sk, N_v) or (N_lines, N_sk, N_v); tau_base : the same shape. Returns (fig, axs)."""
    labels = list(isolated)
    n = len(labels)
    nrows = -(-n // ncols)
    fig, axs = plt.subplots(2 * nrows, ncols, figsize=figsize or (6 * ncols, 5.2 * nrows), sharex=True,
                            gridspec_kw={"height_ratios": [2, 1] * nrows}, layout="constrained", squeeze=False)
    v = np.asarray(v)
    m = slice(None) if window is None else (v >= window[0]) & (v <= window[1])
    pick = lambda t: np.asarray(t)[0, skewer] if np.ndim(t) == 3 else np.asarray(t)[skewer]   # (N_lines, N_sk, N_v) or (N_sk, N_v)
    Fb = np.exp(-pick(tau_base))
    c = color or ENTITY_COLORS["HI"]
    for i, lab in enumerate(labels):
        r, q = 2 * (i // ncols), i % ncols
        Fi = np.exp(-pick(isolated[lab]))
        axs[r, q].plot(v[m], Fb[m], color=ENTITY_COLORS["fake_spectra"], **MODE_STYLES["reference"])
        axs[r, q].plot(v[m], Fi[m], color=c, lw=1.6)
        d = Fi - Fb
        axs[r + 1, q].plot(v[m], d[m], color=c, lw=1.4)
        axs[r + 1, q].axhline(0, color=FG, lw=0.8)
        axs[r, q].set_title(lab, fontsize=FONT_MIN["legend"] + 1)
        _textbox(axs[r + 1, q], 0.02, 0.85, f"max|ΔF| = {np.abs(d).max():.1e}\nΔ⟨F⟩ = {d.mean():+.1e}", transform=axs[r + 1, q].transAxes,
                 va="top")
        axs[r, q].set_ylim(-0.03, 1.05)
        if q == 0:
            axs[r, q].set_ylabel(LABELS["F"]); axs[r + 1, q].set_ylabel(LABELS["dF"])
    for i in range(n, nrows * ncols):
        r, q = 2 * (i // ncols), i % ncols
        axs[r, q].set_axis_off(); axs[r + 1, q].set_axis_off()
    for q in range(ncols):
        axs[-1, q].set_xlabel("u [km/s]")
    return fig, axs
