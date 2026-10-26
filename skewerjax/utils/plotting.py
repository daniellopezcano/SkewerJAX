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
    "voronoi": PALETTE["cyan"], "tophat": PALETTE["yellow"], "crossings": PALETTE["purple"],
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
    """(a), (b), ... in the upper-left corner of every axis (2D or 3D)."""
    fg, bg = plt.rcParams["text.color"], plt.rcParams["axes.facecolor"]
    for i, ax in enumerate(np.ravel(axes)):
        (ax.text2D if hasattr(ax, "text2D") else ax.text)(x, y, f"({chr(ord(start) + i)})", transform=ax.transAxes, fontsize=size, fontweight="bold", zorder=20,
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
FIGSIZES = {"wide": (20, 6), "single": (13, 6), "tall": (13, 9), "triple": (22, 6), "grid": (22, 12), "dag": (24, 6.6)}

# The dependency graph of the ingredients, derived from the code. Columns are topological levels; the two field sources
# (toy fields and simulation outputs) are interchangeable inputs of the gas cells, side by side in one column.
DAG_NODES = {   # key: (column, row, name, detail)
    "lines": (0, 1.45, "lines", "LineBundle"),
    "cosmology": (0, -1.2, "cosmology", "Cosmology"),
    "skewers": (1, -1.2, "skewers", "SkewerBundle, a(x)"),
    "samplers": (2, 0.52, "toy fields", "PowerLawGRF → cells"),
    "simulation": (2, -0.52, "simulation outputs", "TNG snapshot → cells"),
    "fields": (3, 0.0, "gas cells", "Voronoi cells: n, T, v"),
    "profiles": (4, 1.45, "line profiles", r"Voigt $\phi_j(v)$ of the gas"),
    "deposition": (4, 0.0, "deposition", "cells → pieces"),
    "tau": (5, 0.0, "optical depth", r"$\tau(v)$, $F=e^{-\tau}$"),
    "overflux": (6, 0.0, "overflux", r"$\delta_F$ per line, total"),
    "stats": (7, 0.0, "statistics", "P1D, ξ1D, P3D, ξ3D"),
    "fake_spectra": (3, -0.8, "fake_spectra", "external reference"),
}
# (from, to, kind, label, route): kind "solid" required | "optional" optional input or data table | "external";
# route = waypoints between the two ends: "h" leaves horizontally then turns, "v" leaves vertically then turns.
DAG_EDGES = (
    ("cosmology", "skewers", "optional", "a(x) (or manual)", None),
    ("lines", "simulation", "optional", "ion masses", "v"),
    ("samplers", "fields", "solid", None, None),
    ("simulation", "fields", "solid", None, None),
    ("fields", "deposition", "solid", None, None),
    ("skewers", "deposition", "solid", "points, a(x)", "h"),
    ("deposition", "tau", "solid", "pieces", None),
    ("lines", "profiles", "solid", "atomic data", None),
    ("profiles", "tau", "solid", None, "h"),
    ("tau", "overflux", "solid", None, None),
    ("lines", "overflux", "optional", "line offsets Δv (frames)", "h"),
    ("overflux", "stats", "solid", None, None),
)
DAG_EXTERNAL = (("simulation", "fake_spectra", None), ("fake_spectra", "tau", "compared"))   # same snapshot, compared τ
DAG_NOTEBOOKS = {"lines": "01", "cosmology": "02", "skewers": "03", "samplers": "04", "fields": "04, 05", "simulation": "05",
                 "profiles": "06", "deposition": "07", "tau": "07", "overflux": "08", "stats": "08", "fake_spectra": "09"}
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
    reads only the line offsets Δv to put every line on the reference frame). The two sources of gas are interchangeable:
    both produce the same container, Voronoi gas cells, which the deposition cuts into pieces along the skewers; the
    optical depth sums the line profiles of the pieces. ``external`` adds fake_spectra, the external reference code
    (comparison notebook only).
    ``notebooks`` writes the notebook of every ingredient under its box. Returns (fig, ax).
    """
    from matplotlib.lines import Line2D
    hl = {highlight} if isinstance(highlight, str) else set(highlight)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or FIGSIZES["dag"], layout="constrained")
    ax.set_axis_off()
    ax.set(xlim=(-0.5, 7.5), ylim=(-1.62, 2.1))
    fg, on, ext_c = plt.rcParams["text.color"], PALETTE["cyan"], PALETTE["yellow"]
    on_fill = mpl.colors.to_rgba(on, 0.30)                      # the one highlight: a light fill and a coloured edge
    w, h = _DAG_W, _DAG_H
    nodes = {k: v for k, v in DAG_NODES.items() if external or k != "fake_spectra"}
    lanes = {("lines", "overflux"): 0.44, ("lines", "simulation"): 0.12, ("profiles", "tau"): 0.0}
    # the group of interchangeable field sources
    gx, gy = DAG_NODES["samplers"][0], 0.5 * (DAG_NODES["samplers"][1] + DAG_NODES["simulation"][1])
    gh = DAG_NODES["samplers"][1] - DAG_NODES["simulation"][1] + h + 0.34
    ax.add_patch(mpl.patches.FancyBboxPatch((gx - w / 2 - 0.08, gy - gh / 2), w + 0.16, gh, boxstyle="round,pad=0.02,rounding_size=0.1",
                 facecolor="none", edgecolor=DIM, lw=1.2, ls=":", path_effects=[]))
    ax.text(gx, gy + gh / 2 + 0.04, "sources of cells (interchangeable)", ha="center", va="bottom", fontsize=FONT_MIN["floor"],
            color=DIM, style="italic")
    edges = [e for e in DAG_EDGES] + ([(a, b, "external", lab, None) for a, b, lab in DAG_EXTERNAL] if external else [])
    for a, b, kind, label, route in edges:
        if a not in nodes or b not in nodes:
            continue
        if (a, b) == ("fake_spectra", "tau"):
            pts = [(nodes[a][0] + w / 2, nodes[a][1]), (nodes[b][0] - 0.2, nodes[a][1]), (nodes[b][0] - 0.2, nodes[b][1] - h / 2)]
        elif (a, b) == ("simulation", "fake_spectra"):
            pts = [(nodes[a][0] + w / 2, nodes[a][1] - 0.15), (nodes[b][0] - w / 2, nodes[a][1] - 0.15)]
        elif (a, b) == ("skewers", "deposition"):
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
            f = 0.3 if (a, b) == ("skewers", "deposition") else 0.5
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
    ax[5].text(0.0, 0.02, "quadri_vectors[..., 0] = a, [..., 1:4] = (x, y, z):\nthe events (a, x, y, z) of the sampling points",
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
    profile (notebook 07).

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
    residual (notebook 07).

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
    """P1D or xi1D of the total overflux split by order and, within each order, into its terms (notebook 08).

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
    kp = np.asarray(k_par, dtype=float)
    comps = {axis: kp[:, None, None], tr[0]: ka[None, :, None], tr[1]: kb[None, None, :]}
    P = np.asarray(grf.power_at(*(comps[i] for i in range(3)))).reshape(kp.size, -1)     # P(k, mu): mu along grf.los_axis
    area = grf.L[tr[0]] * grf.L[tr[1]]
    if return_sigma:
        return P.sum(-1) / area, np.sqrt((P ** 2).sum(-1)) / area
    return P.sum(-1) / area


def plot_p1d_input_vs_measured(k, curves: dict, reference, *, ref_label: str = "input", axs=None, k_range=None, figsize=(15, 10)):
    """Measured P1D curves against an input spectrum, with their ratios (notebook 08).

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


def power_map_kpar_kperp(m, edges_par, edges_perp, los_axis: int = -1):
    """Mean P over draws and modes in bins of (k_par = |k| along ``los_axis``, k_perp = |k| across it).

    m : ``field_power_modes`` output. Returns (P (n_par, n_perp) with NaN where a bin has no mode, n_modes (weighted),
    error of P (scatter between draws), n_indep (independent complex modes of the half space, for the Gaussian error))."""
    kv = m["kvec"]
    los = los_axis % kv.shape[1]
    kpar, kperp = np.abs(kv[:, los]), np.linalg.norm(np.delete(kv, los, axis=1), axis=-1)
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



def plot_nH_T_fractions(n_H, T, x_HI, f_SiIII, sf, cloudy_grid, *, n_thr: float = None, bins: int = 110, figsize=(24, 9.5)):
    """The n_H-T plane of the cells, H I and Si III kept apart.

    (a) H I: every (n_H, T) bin coloured by the median log10 x_HI of its cells, with contours of the number of cells;
    (b) Si III: the precomputed CLOUDY table log10 f_SiIII(n_H, T) as the background, the same cell contours on top.
    Text boxes name the visible features: the star-formation threshold (``n_thr``), the effective equation of state of
    the star-forming cells (a power-law fit T(n_H), printed, and drawn over the densities of 99.5 % of those cells: the
    densest few hundred bend below it), the photo-heated gas near 10^4 K and the hot halo gas.

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
        xs = np.array([lx[sf].min(), np.percentile(lx[sf], 99.5)])           # where the power law describes the cells
        for ax in axs:
            ax.plot(xs, np.polyval(p, xs), color=PALETTE["cyan"], lw=2.2, ls="--", path_effects=[])
        x_box = xs[0] + 1.3                                       # below the strip, clear of the cells
        _textbox(axs[0], x_box, np.polyval(p, x_box) - 0.95,
                 "star-forming cells: effective EOS\n" + rf"$T \propto n_{{\rm H}}^{{{p[0]:.2f}}}$ ($\pm${info['eos']['scatter']:.2f} dex)",
                 color=PALETTE["cyan"])
    if n_thr is not None:
        for ax in axs:
            ax.axvline(np.log10(n_thr), color=PALETTE["yellow"], ls=":", lw=2.0, path_effects=[])
        _textbox(axs[0], np.log10(n_thr) + 0.1, ey[-1] - 0.35, "star-formation\nthreshold " + rf"{n_thr:.2f} cm$^{{-3}}$", color=PALETTE["yellow"])
    _textbox(axs[0], ex[0] + 0.3, 3.75, "photo-heated gas:\n" + r"floor near $10^4$ K", color=FG)
    _textbox(axs[0], ex[0] + 0.3, 6.6, "hot, shock-heated\n" + r"halo gas ($x_{\rm HI} \ll 1$)", color=FG)
    i, j = np.unravel_index(np.argmax(lf), lf.shape)
    info["cloudy_max"] = dict(log_n_H=float(ld[i]), log_T=float(lt[j]), log_f=float(lf[i, j]))
    _textbox(axs[1], max(ex[0] + 0.3, min(ld[i], ex[-1]) - 2.2), lt[j] - 0.55,
             "Si III ridge: most abundant near\n" + rf"$\log T = {lt[j]:.2f}$ ($\log n_{{\rm H}} = {ld[i]:.1f}$)", color=FG)
    j_lo = np.argmax(lf[np.argmin(np.abs(ld + 3.0))])
    _textbox(axs[1], ex[0] + 0.3, ey[0] + 0.25, r"photoionised at low $n_{\rm H}$ (at $\log n_{\rm H} = -3$: peak near "
             + rf"$\log T = {lt[j_lo]:.2f}$)", color=FG)
    axs[1].set_xlim(ex[0], ex[-1]); axs[1].set_ylim(ey[0], ey[-1])
    axs[0].set_title(r"H I: stored ionisation equilibrium (star-forming cells: cold phase)")
    axs[1].set_title(r"Si III: CLOUDY table (background) and the cells (contours)")
    return fig, axs, info


def _cells_on_path(selected, path, box):
    """The cells of ``selected`` that a polyline ``path`` (N, 3) [cMpc, unwrapped] passes through, in path order.

    The path is resampled at a quarter of the smallest cell size and every sample takes the cell of its nearest
    generating point (periodic when ``box`` is given): the Voronoi definition. Returns (index into ``selected`` per
    crossed cell, its generating point moved next to the path (minimum image), its unit tangent there).
    """
    from scipy.spatial import cKDTree
    path = np.asarray(path, dtype=float)
    seg = np.diff(path, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    step = 0.25 * float(np.min(selected.size_cMpc))
    n_sub = np.maximum(np.ceil(seg_len / step).astype(int), 1)
    pts = np.concatenate([path[i] + np.linspace(0.0, 1.0, n, endpoint=False)[:, None] * seg[i] for i, n in enumerate(n_sub)] + [path[-1:]])
    tang = np.concatenate([np.repeat(seg / seg_len[:, None], n_sub, axis=0), seg[-1:] / seg_len[-1]])
    pos = np.asarray(selected.positions, dtype=float)
    tree = cKDTree(np.mod(pos, box) if box is not None else pos, boxsize=box)
    owner = tree.query(np.mod(pts, box) if box is not None else pts, workers=-1)[1]
    idx, first = np.unique(owner, return_index=True)
    order = np.argsort(first)
    idx, first = idx[order], first[order]
    g = pos[idx]
    if box is not None:
        g = g + box * np.round((pts[first] - g) / box)
    return idx, g, tang[first]


def _cube_faces(centre, side):
    """The 6 faces (4, 3) of an axis-aligned box of sides ``side`` (3,) centred on ``centre``."""
    c = np.asarray(centre, dtype=float)
    h = 0.5 * np.asarray(side, dtype=float)
    v = c + h * np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float)
    quads = ((0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4), (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5))
    return [v[list(q)] for q in quads]


def _voronoi_faces(cells, centres, sizes):
    """The faces (n_v, 3) of the Voronoi cells of the generating points ``centres`` (K, 3) [cMpc, unwrapped], among
    every generating point of ``cells`` within 4 cell sizes of them (periodic images when ``cells.box`` is given).

    scipy's ``Voronoi`` on that local neighbourhood; every face is a ridge between two generators, its vertices ordered
    by angle around their mean in the plane of the face. Returns one list of faces per centre.
    """
    from scipy.spatial import Voronoi, cKDTree
    box = cells.box
    pos = np.asarray(cells.positions, dtype=float)
    tree = cKDTree(np.mod(pos, box) if box is not None else pos, boxsize=box)
    pts, keys = [], {}
    for c, h in zip(centres, sizes):
        for j in tree.query_ball_point(np.mod(c, box) if box is not None else c, 4.0 * h):
            p = pos[j] if box is None else pos[j] + box * np.round((c - pos[j]) / box)
            key = (j,) + tuple(np.round(p, 6))
            if key not in keys:
                keys[key] = len(pts)
                pts.append(p)
    pts = np.asarray(pts)
    vor = Voronoi(pts)
    near = cKDTree(pts)
    own = near.query(np.asarray(centres, dtype=float))[1]
    faces = {int(i): [] for i in own}
    for (a, b), rv in zip(vor.ridge_points, vor.ridge_vertices):
        for i in (a, b):
            if i in faces and -1 not in rv:
                v = vor.vertices[rv]
                n = pts[b] - pts[a]
                n /= np.linalg.norm(n)
                e1 = v[0] - v.mean(0)
                e1 -= n * (e1 @ n)
                e1 /= np.linalg.norm(e1)
                e2 = np.cross(n, e1)
                d = v - v.mean(0)
                faces[i].append(v[np.argsort(np.arctan2(d @ e2, d @ e1))])
    return [faces[int(i)] for i in own]


def plot_gas_cells(cells, selected, path, species=None, *, skewer: int = 0, view=(24.0, -62.0), zoom: float = 1.1,
                   decades: float = 12.0, alpha: float = 0.55, figsize=None):
    """What a ``core.Cells`` holds for the optical depth, along one skewer: the same 3D view for any source (04, 05).

    cells : the ``core.Cells`` (its geometry: a grid of cubes when ``cells.grid`` is set, else Voronoi cells).
    selected : the cells within reach of the skewer, as the optical depth selects them
        (``core.select_cells(cells, bundle.spatial_points)``).
    path : a ``SkewerBundle`` (its skewer ``skewer``) or the (N, 3) points of one skewer [cMpc].
    species : a key, a sequence, or None for every species of ``cells``; one row each.

    The cells drawn are those of ``selected`` that the skewer passes through (each sample of the path belongs to its
    nearest generating point), as 3D volumes: cubes for grid cells; for Voronoi cells, polyhedra computed here only for
    the drawing (scipy's ``Voronoi`` on the generating points around them, periodic). Every panel shows the cells
    coloured by its column's quantity, with their edges, their generating points (dots) and the skewer (line):
    (1) the comoving number density of the species (log, ``CMAPS["sequential"]``, its own colour bar, at most ``decades``
    below its maximum: lower values take the lowest colour); (2) the
    temperature (log, ``CMAPS["hot"]``); (3) the peculiar velocity: colour the component along the skewer v_par
    (``CMAPS["diverging"]``, symmetric) and one 3D arrow per cell, from its generating point. The temperature and
    velocity colour bars, and the arrow scale, are shared by the rows. ``view`` = (elevation, azimuth) [deg]; ``zoom``
    scales the 3D boxes (data aspect 1:1:1); ``alpha`` is the opacity of the cell faces.

    Returns (fig, axs (n_species, 3), info {"n_selected", "n_crossed", "crossed": indices into ``selected``,
    "path_length": [cMpc], "arrow_kms": the speed of an arrow one median cell size long [km/s]}).
    """
    import re
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    sp_list = list(cells.species_keys) if species is None else ([species] if isinstance(species, str) else list(species))
    pts = np.asarray(path.spatial_points[skewer] if hasattr(path, "spatial_points") else path, dtype=float)
    box = cells.box
    idx, g, tang = _cells_on_path(selected, pts, box)
    size = np.asarray(selected.size_cMpc, dtype=float)[idx]
    if cells.grid is not None:
        side = float(box) / np.asarray(cells.grid, dtype=float)
        faces = [_cube_faces(c, side) for c in g]
    else:
        faces = _voronoi_faces(cells, g, size)
    s0 = np.cumsum(np.r_[0.0, np.linalg.norm(np.diff(pts, axis=0), axis=1)])
    allv = np.concatenate([np.concatenate(f) for f in faces] + [pts])
    lo, hi = allv.min(0), allv.max(0)
    fields = {sp: {k: np.asarray(selected.species[sp][k], dtype=float)[idx] for k in ("number_density", "temperature", "velocity")}
              for sp in sp_list}
    logT = {sp: np.log10(np.maximum(f["temperature"], 1e-300)) for sp, f in fields.items()}
    t_lim = (min(x.min() for x in logT.values()), max(x.max() for x in logT.values()))
    t_lim = (t_lim[0], max(t_lim[1], t_lim[0] + 1e-3))
    v_par = {sp: np.sum(f["velocity"] * tang, axis=1) for sp, f in fields.items()}
    v_lim = max(max(float(np.abs(x).max()) for x in v_par.values()), 1e-6)
    speed = max(max(float(np.linalg.norm(f["velocity"], axis=1).max()) for f in fields.values()), 1e-6)
    h_med = float(np.median(size))
    arrow_kms = speed / 1.2                                          # the fastest cell: an arrow of 1.2 median cell sizes
    n = len(sp_list)
    fig = plt.figure(figsize=figsize or (24, 7.6 * n + 1.6), layout="constrained")
    fig.get_layout_engine().set(rect=(0.0, 0.0, 0.96, 0.97), wspace=0.08)   # room for a suptitle and the z labels (3D labels are not in the layout)
    fig.text(1.0, 0.5, " ", alpha=0.0)                                   # keeps that room under bbox_inches="tight" (inline)
    axs = np.array([[fig.add_subplot(n, 3, 3 * r + c + 1, projection="3d", computed_zorder=False) for c in range(3)]
                    for r in range(n)])
    norm_t = mpl.colors.Normalize(*t_lim)
    norm_v = mpl.colors.Normalize(-v_lim, v_lim)
    for r, sp in enumerate(sp_list):
        name = re.sub(r"^([A-Z][a-z]?)([IVX]+)$", r"\1 \2", sp)        # "SiIII" -> "Si III"
        f = fields[sp]
        ln = np.log10(np.maximum(f["number_density"], 1e-300))
        norm_n = mpl.colors.Normalize(max(ln.min(), ln.max() - decades), max(ln.max(), ln.min() + 1e-3))
        cols = ((CMAPS["sequential"], norm_n, ln), (CMAPS["hot"], norm_t, logT[sp]), (CMAPS["diverging"], norm_v, v_par[sp]))
        titles = (f"{name} density", f"{name} temperature", f"{name} peculiar velocity")
        labels = (rf"$\log_{{10}} n_{{\rm com}}$ [cm$^{{-3}}$] ({name})", r"$\log_{10} T$ [K] (shared by the rows)",
                  r"$v_\parallel$ along the skewer [km s$^{-1}$] (shared)")
        for c, (cmap, norm, val) in enumerate(cols):
            ax = axs[r, c]
            rgba = mpl.colormaps[cmap](norm(val))
            poly = [q for fc in faces for q in fc]
            fcol = np.repeat(rgba, [len(fc) for fc in faces], axis=0)
            fcol[:, 3] = alpha
            ax.add_collection3d(Poly3DCollection(poly, facecolors=fcol, edgecolors=FG, linewidths=0.7))
            ax.plot(*pts.T, color=FG, lw=2.6, label="skewer")
            ax.scatter(*g.T, s=22, color=FG, depthshade=False, label="generating points")
            if c == 2:
                vv = f["velocity"] / arrow_kms * h_med
                ax.quiver(*g.T, *vv.T, color=FG, lw=2.0, arrow_length_ratio=0.3,
                          normalize=False)
            ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), zlim=(lo[2], hi[2]))
            ax.set_box_aspect(hi - lo, zoom=zoom)
            ax.view_init(*view)
            for a_, lab in zip((ax.xaxis, ax.yaxis, ax.zaxis), "xyz"):
                a_.set_major_locator(mpl.ticker.MaxNLocator(3))
                a_.set_label_text(f"{lab} [cMpc]")
                a_.labelpad = 10 if lab == "z" else 12
            ax.set_title(titles[c], pad=0)
            if c == 2:
                ax.text2D(0.99, 0.93, f"arrows: {arrow_kms:.0f} km/s per {h_med:.2g} cMpc", transform=ax.transAxes, ha="right",
                          va="top", fontsize=FONT_MIN["annotation"], color=FG)
            cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, location="bottom", shrink=0.7, aspect=25, pad=0.0,
                              label=labels[c], extend="min" if c == 0 and ln.min() < norm.vmin else "neither")
            cb.locator = mpl.ticker.MaxNLocator(4)
            cb.update_ticks()
    info = dict(n_selected=len(selected), n_crossed=int(idx.size), crossed=idx, path_length=float(s0[-1]), arrow_kms=arrow_kms)
    return fig, axs, info


def plot_voigt_limits(x, sigma, gamma, curves: dict, ref_key: str = "scipy_wofz", *, linthresh: float = 1e-6,
                      expansion: dict = None, figsize=None):
    """Voigt profiles V(x; sigma, gamma) for extreme (sigma, gamma) pairs, every implementation in ``curves`` = {name:
    (N_pairs, N_x)} on the reference ``ref_key`` (thick, light), and their relative residuals (symlog, ``linthresh``).
    The pairs show the Gaussian limit (gamma << sigma), the Lorentzian limit (sigma << gamma) and the mixed cases.

    expansion : optional {"u": (N_u,) u >= 0, "residuals": {a: (N_u,) |H_approx / H - 1|}, "label": str,
        "u_switch": float}: panel (c), the relative error of an approximation of H(a, u) = Re w(u + i a) against the exact
        one, one line per damping parameter a (sequential colours), with its core (|u| < 3, Gaussian), transition and
        wings (|u| > ``u_switch``) marked. Returns (fig, axs): 2 axes, or 3 with ``expansion``.
    """
    from matplotlib.lines import Line2D
    n_rows = 2 if expansion is None else 3
    figsize = figsize or ((15, 11) if expansion is None else (15, 16))
    fig, axs = plt.subplots(n_rows, 1, figsize=figsize, gridspec_kw={"height_ratios": [2.2, 1, 1.25][:n_rows]}, layout="constrained")
    ax, axr = axs[0], axs[1]
    axr.sharex(ax)
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
    if expansion is not None:
        axe = axs[2]
        u = np.asarray(expansion["u"])
        u_sw = float(expansion.get("u_switch", 6.0))
        res = expansion["residuals"]
        cmap = plt.get_cmap(CMAPS["line"])
        la = np.log10(np.asarray(list(res), dtype=float))
        for k, (a, r) in enumerate(res.items()):
            f = 0.5 if np.ptp(la) == 0 else (la[k] - la.min()) / np.ptp(la)
            axe.loglog(u, np.maximum(np.asarray(r), 1e-12), color=cmap(0.3 + 0.7 * f), lw=2.2, label=rf"$a$ = {a:g}")
        axe.axvspan(u.min(), 3.0, color=PALETTE["grey"], alpha=0.18, lw=0)
        axe.axvline(u_sw, color=DIM, ls=":", lw=1.8)
        top = 0.93
        for xt, txt in ((np.sqrt(u.min() * 3.0), "core (Gaussian)"), (np.sqrt(3.0 * u_sw), "transition"),
                        (np.sqrt(u_sw * u.max()), "wings (series)")):
            axe.text(xt, top, txt, transform=axe.get_xaxis_transform(), ha="center", va="top", fontsize=FONT_MIN["annotation"], color=FG)
        axe.set(xlim=(u.min(), u.max()), ylim=(1e-10, 1e-3), xlabel=r"$u = x/(\sqrt{2}\,\sigma)$",
                ylabel=r"$|H_{\rm approx}/H - 1|$")
        axe.set_title(expansion.get("label", "approximation"), fontsize=FONT_MIN["title"])
        axe.legend(loc="lower left", fontsize=FONT_MIN["legend"], ncols=len(res))
    return fig, np.asarray(axs)






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


def input_power_modes(grf, m):
    """The input P of ``grf`` at the modes of a ``field_power_modes`` output (|k| and, for an anisotropic input, mu
    along ``grf.los_axis``), shaped as one draw (1, modes): the exact expectation of every measured mode."""
    return np.asarray(grf.power_at(*m["kvec"].T))[None]


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


# --------------------------------------------------------------------------- path geometry (notebook 03)
def geometry_style(name: str) -> dict:
    """The style of one element of the path geometry, the same in every figure (03 in 3D, 05 in 2D).

    name : "point" (sampling points x_i), "midpoint" (m_{i+-1/2}), "dual_segment" (from midpoint to midpoint; two
    alternating colours), "tangent" (unit tangents), "line_element" (w_i labels and brackets), "piece_end" (the ends of
    the pieces of the tau sum), "generator" (the mesh-generating points of the cells). Follows the current theme.
    """
    return {"point": dict(marker="o", s=75, color=FG, edgecolors=BG, linewidths=1.2, zorder=9),
            "midpoint": dict(marker="D", s=60, color=PALETTE["yellow"], edgecolors=FG, linewidths=1.0, zorder=10),
            "dual_segment": dict(colors=(PALETTE["blue"], PALETTE["cyan"]), lw=10, alpha=0.6),
            "tangent": dict(color=PALETTE["red"], lw=2.4),
            "line_element": dict(color=PALETTE["purple"], lw=2.0),
            "piece_end": dict(marker="|", s=260, color=FG, linewidths=2.2, zorder=11),
            "generator": dict(marker="+", s=70, color=FG, linewidths=1.6, zorder=6)}[name]


def plot_skewer_geometry_3d(ax, points, *, scale_factors=None, label_w=None, tangent_length: float = None,
                            highlight: int = None):
    """The geometry of one skewer in 3D (notebook 03), each element in its own style (``geometry_style``): sampling
    points x_i (dots), midpoints m_{i+-1/2} (diamonds), the dual segment of every point (thick bands, alternating
    colours, from m_{i-1/2} to m_{i+1/2}), unit tangents t_i (arrows) and the line elements w_i (purple brackets beside
    the dual segments of the points ``label_w``, default about six, with their lengths). ``scale_factors`` (N,), if
    given, adds a(x_i) to those labels. ``highlight`` names one point whose dual segment is annotated. Returns None.
    """
    from skewerjax.core.skewers import path_geometry
    p = np.asarray(points, dtype=float)
    g = {k: np.asarray(v) for k, v in path_geometry(p).items()}
    n = len(p)
    ax.computed_zorder = False                                    # draw in the z-order below, not by depth
    ds = geometry_style("dual_segment")
    for i, (lo, x, hi) in enumerate(zip(g["mid_lo"], p, g["mid_hi"])):
        ax.plot(*np.stack([lo, x, hi]).T, color=ds["colors"][i % 2], lw=ds["lw"], alpha=ds["alpha"], solid_capstyle="butt",
                path_effects=[], zorder=2, label=r"dual segments $[m_{i-1/2}, m_{i+1/2}]$" if i == 0 else None)
    ax.plot(*p.T, color=DIM, lw=1.0, path_effects=[], zorder=3)
    st = geometry_style("point")
    ax.scatter(*p.T, depthshade=False, label=r"sampling points $x_i$", **st)
    mids = 0.5 * (p[1:] + p[:-1])
    ax.scatter(*mids.T, depthshade=False, label=r"midpoints $m_{i\pm1/2}$", **geometry_style("midpoint"))
    L = tangent_length or 1.4 * np.median(g["line_elements"])
    tg = geometry_style("tangent")
    every = slice(0, None, 2)                                     # every other point: the arrows stand out of the bands
    ax.quiver(*p[every].T, *(L * g["tangents"][every]).T, color=tg["color"], lw=tg["lw"], arrow_length_ratio=0.35, zorder=7,
              label=r"unit tangents $\hat t_i$ (every other point)")
    le = geometry_style("line_element")
    span = (p.max(0) - p.min(0)).max()
    for j, i in enumerate(np.linspace(0, n - 1, 6).round().astype(int) if label_w is None else label_w):
        lo, hi, t = g["mid_lo"][i], g["mid_hi"][i], g["tangents"][i]
        nrm = np.cross(t, [0.0, 0.0, 1.0])
        nrm = nrm if np.linalg.norm(nrm) > 0.3 else np.cross(t, [1.0, 0.0, 0.0])
        out = 0.05 * span * nrm / np.linalg.norm(nrm)                  # a bracket beside the band, across the path
        br = np.stack([lo, lo + out, hi + out, hi])
        ax.plot(*br.T, color=le["color"], lw=le["lw"], path_effects=[], zorder=8,
                label=r"line elements $w_i$ (dual-segment lengths)" if j == 0 else None)
        txt = f"  $w_{{{i}}}$ = {g['line_elements'][i]:.2f}"
        if scale_factors is not None:
            txt += f", a = {float(np.asarray(scale_factors)[i]):.4f}"
        ax.text(*(0.5 * (lo + hi) + 1.8 * out), txt, color=le["color"], fontsize=FONT_MIN["annotation"], zorder=12)
    if highlight is not None:
        k = int(highlight)
        ax.plot(*np.stack([g["mid_lo"][k], p[k], g["mid_hi"][k]]).T, color=FG, lw=2.2, ls=(0, (3, 2)), path_effects=[], zorder=8)
        ax.text(*g["mid_hi"][k], f"  dual segment of $x_{{{k}}}$:\n  $m_{{{k}-1/2}} \\to m_{{{k}+1/2}}$", color=FG,
                fontsize=FONT_MIN["annotation"], zorder=12)
    lo, hi = p.min(0), p.max(0)
    pad = 0.08 * (hi - lo).max()
    ax.set(xlim=(lo[0] - pad, hi[0] + pad), ylim=(lo[1] - pad, hi[1] + pad), zlim=(lo[2] - pad, hi[2] + pad),
           xlabel="x [cMpc]", ylabel="y [cMpc]", zlabel="z [cMpc]")
    dark_3d(ax)
    return None


# --------------------------------------------------------------------------- GRF recovery (notebook 04)
def _wrapped_offsets(n: int, dx: float) -> np.ndarray:
    """Signed minimum-image separations of a periodic axis of n cells: dx (j for j < n/2, j - n above)."""
    j = np.arange(n)
    return dx * np.where(j <= n // 2, j, j - n)


def correlation_binned(deltas, L, edges, grf=None, *, r_max_fraction: float = 0.25, cov_k_cut: float = 16.0,
                       max_exact: float = 2e8):
    """xi(|r|) of periodic fields, binned in |r| over the minimum-image separations of the grid.

    Measured: xi(r) = (1/N) sum_x delta(x) delta(x + r) = irfftn(|rfftn(delta)|^2) / N (Wiener-Khinchin), averaged over
    the separations of each bin and over draws. Input (``grf``): xi(r) = (1/V) sum_k P(k) e^{i k.r} = irfftn(P on the
    rfft grid) / dV, the exact correlation of the discrete periodic field with spectrum P (the box's own modes, no
    k = 0), binned alike, so <xi_measured> = xi_input exactly. Only separations with every |r_i| <= ``r_max_fraction``
    L_i are used.

    Errors (with ``grf``): the exact Gaussian covariance of the binned mean over the N_real draws,

        Cov(xi_b, xi_b') = 2 / (V^2 N_real) sum_{k != 0} P(k)^2 c_b(k) c_b'(k),   c_b(k) = <cos(k.r)>_{r in bin b},

    (|delta_k|^2 is exponential with variance P^2; the modes k and -k are the same). Every bin is dominated by the few
    largest-scale modes of the box, so the bins are strongly correlated: they move together. The diagonal is exact;
    when the grid has more than ``max_exact`` (modes x bins; 2e8 is about 1.6 GB of work arrays), the off-diagonal terms
    keep the modes with |k| <= ``cov_k_cut`` fundamental modes only (the dominant ones for a red spectrum; a bluer one
    needs the exact sum). Without ``grf``: the scatter between draws / sqrt(N_real).
    Returns (r_b, xi_b, err, n_sep, xi_input_b or None, cov or None).
    """
    d = np.asarray(deltas, dtype=float)
    shape = d.shape[1:]
    Ls = np.broadcast_to(np.asarray(L, dtype=float), (len(shape),))
    dx = Ls / np.asarray(shape)
    m = [max(int(r_max_fraction * n), 1) for n in shape]
    take = [np.r_[0:mi + 1, n - mi:n] for mi, n in zip(m, shape)]
    sub = np.ix_(*take)
    r = np.sqrt(sum(np.meshgrid(*[_wrapped_offsets(n, h)[t] ** 2 for n, h, t in zip(shape, dx, take)], indexing="ij")))
    j = np.digitize(r.ravel(), edges) - 1
    ok = (j >= 0) & (j < len(edges) - 1)
    nb = len(edges) - 1
    cnt = np.bincount(j[ok], minlength=nb)
    rb = np.bincount(j[ok], weights=r.ravel()[ok], minlength=nb) / np.maximum(cnt, 1)
    axes = tuple(range(len(shape)))
    xs = []
    for di in d:
        f = np.fft.rfftn(di)
        xi = np.fft.irfftn(np.abs(f) ** 2, s=shape, axes=axes)[sub] / np.prod(shape)
        xs.append(np.bincount(j[ok], weights=xi.ravel()[ok], minlength=nb) / np.maximum(cnt, 1))
    xs = np.stack(xs)
    err = xs.std(0, ddof=1) / np.sqrt(len(xs)) if len(xs) > 1 else np.full(nb, np.nan)
    xin, cov = None, None
    keep = cnt > 0
    if grf is not None:
        import scipy.fft as sfft
        kk = [2 * np.pi * np.fft.fftfreq(n, h) for n, h in zip(shape[:-1], dx[:-1])] + [2 * np.pi * np.fft.rfftfreq(shape[-1], dx[-1])]
        kmod = np.sqrt(sum(np.meshgrid(*[k ** 2 for k in kk], indexing="ij")))
        Pk = np.asarray(grf.power_at(*np.meshgrid(*kk, indexing="ij", sparse=True)))     # P(k, mu) of every mode
        xi_in = np.fft.irfftn(Pk, s=shape, axes=axes)[sub] / np.prod(dx)
        xin = np.bincount(j[ok], weights=xi_in.ravel()[ok], minlength=nb) / np.maximum(cnt, 1)
        # c_b(k) = Re FFT of the bin's indicator over the separations / n_b; w: half-space multiplicity
        last = kk[-1][(None,) * (len(shape) - 1) + (slice(None),)]
        w = np.where((last == 0) | (np.isclose(last, np.pi / dx[-1]) & (shape[-1] % 2 == 0)), 1.0, 2.0)
        wP2 = np.broadcast_to(w * Pk ** 2, kmod.shape)
        exact = kmod.size * nb <= max_exact
        low = np.ones(kmod.shape, bool) if exact else (kmod > 0) & (kmod <= cov_k_cut * 2 * np.pi / Ls.max() * 1.0001)
        jj = np.full(r.shape, -1)
        jj.ravel()[ok] = j[ok]
        C_low, diag = np.zeros((nb, int(low.sum()))), np.zeros(nb)
        for b in np.nonzero(keep)[0]:
            ind = np.zeros(shape, np.float32)
            ind[sub] = (jj == b)
            c = sfft.rfftn(ind, workers=-1).real.astype(np.float64) / cnt[b]
            diag[b] = np.sum(wP2 * c ** 2)
            C_low[b] = c[low]
        cov = C_low @ (C_low * wP2[low]).T
        np.fill_diagonal(cov, diag)
        cov = cov * 2.0 / (np.prod(Ls) ** 2 * len(xs))
        err = np.sqrt(np.diag(cov))
        cov = cov[np.ix_(keep, keep)]
    return rb[keep], xs.mean(0)[keep], err[keep], cnt[keep], None if xin is None else xin[keep], cov


def chi2_cov(res, cov, rcond: float = 1e-10):
    """res^T C^-1 res / dof over the eigenvectors of C above ``rcond`` x its largest eigenvalue (C is nearly singular
    when the bins are strongly correlated): (chi2/dof, dof)."""
    lam, U = np.linalg.eigh(cov)
    good = lam > rcond * lam.max()
    y = U[:, good].T @ res
    return float(np.sum(y ** 2 / lam[good]) / good.sum()), int(good.sum())


def plot_isotropic_recovery(cases: dict, *, n_bins: int = 18, n_lin: int = 0, n_r_bins: int = 16, figsize=(24, 19)):
    """Isotropic statistics of GRF draws in 1D, 2D and 3D against the input (notebook 04).

    Rows: (1) P(|k|) measured (points, error bars: scatter between draws / sqrt(N_real)) on the input averaged over the
    modes of each bin, at the same k (the mode-weighted mean |k| of the bin), with the input at every k as a light guide;
    (2) measured / input - 1, the same input and errors; (3) xi(|r|) measured and the input xi of the box's discrete
    modes (``correlation_binned``), error bars from its exact Gaussian covariance; (4) the difference in units of that
    error. The xi bins are strongly correlated (the same largest-scale modes set all of them): the title of (4) and the
    summary give the fraction of the variance in the leading eigenmode of their correlation matrix (a common shift of
    every bin) and chi2/dof with the full covariance.

    The comparison in (1) and (2) is with the bin average of the input, not with the curve at the bin's k: over a bin
    the input changes (its curvature, and the BAO wiggles, which a wide bin averages out), so the curve can miss the
    points by many error bars where the average does not. n_bins : logarithmic bins of |k| from the fundamental mode
    to the Nyquist frequency; n_lin > 0: the first ``n_lin`` modes in bins of their own (``hybrid_k_edges``), then
    ``n_bins`` logarithmic bins. cases : {title: (grf, deltas (N_real, *shape))}. Returns (fig, axs (4, n_cases),
    summary dict).
    """
    n = len(cases)
    fig, axs = plt.subplots(4, n, figsize=figsize, gridspec_kw={"height_ratios": [2.0, 1.0, 2.0, 1.0]}, layout="constrained",
                            squeeze=False)
    out = {}
    col = ENTITY_COLORS["density"]
    for c, (title, (grf, deltas)) in enumerate(cases.items()):
        m = field_power_modes(deltas, grf.L)
        kmin = 2 * np.pi / max(grf.L)
        kmax = min(np.pi * N / L for N, L in zip(grf.shape, grf.L))
        edges = hybrid_k_edges(kmin, kmax, n_lin=n_lin, n_log=n_bins) if n_lin > 0 else np.geomspace(0.999 * kmin, kmax, n_bins + 1)
        kb, Pb, eb, _ = _bin_modes(m["k"], m["P"], m["w"], edges)
        Pin_b = _bin_modes(m["k"], input_power_modes(grf, m), m["w"], edges)[1]
        kk = np.geomspace(kb[0], kb[-1], 600)
        guide = "input at every k, averaged over directions" if getattr(grf, "anisotropic", False) and grf.ndim > 1 else "input at every k"
        axs[0, c].plot(kk, np.asarray(grf.power_monopole(kk)), color=FG, **MODE_STYLES["reference"], label=guide)
        axs[0, c].plot(kb, Pin_b, color=FG, lw=1.6, marker="_", ms=14, mew=2, label="input, averaged over the modes of each bin")
        axs[0, c].errorbar(kb, Pb, eb, fmt="o", ms=5, capsize=3, color=col, label=f"measured ({deltas.shape[0]} draws)", zorder=5)
        axs[1, c].errorbar(kb, Pb / Pin_b - 1, eb / Pin_b, fmt="o", ms=5, capsize=3, color=col)
        lim = max(0.1, 1.2 * float(np.max(np.abs(Pb / Pin_b - 1) + eb / Pin_b)))   # every point and its error bar
        axs[0, c].set(xscale="log", yscale="log", title=title, ylabel=r"$P(k)$")
        axs[1, c].set(xscale="log", ylabel="measured / input − 1", xlabel=r"$|k|$ [Mpc$^{-1}$]", ylim=(-lim, lim))
        axs[1, c].sharex(axs[0, c])
        residual_band(axs[1, c], 0.05)
        axs[0, c].legend(fontsize=FONT_MIN["legend"])
        r_edges = np.linspace(0.0, 0.25 * min(grf.L), n_r_bins + 1)
        rb, xb, xe, _, xin, cov = correlation_binned(deltas, grf.L, r_edges, grf)
        axs[2, c].plot(rb, xin, color=FG, lw=2, label=r"input: $\frac{1}{V}\sum_k P(k)\,e^{i\mathbf{k}\cdot\mathbf{r}}$")
        axs[2, c].errorbar(rb, xb, xe, fmt="o", ms=6, capsize=3, color=col, label="measured")
        axs[3, c].errorbar(rb, (xb - xin) / xe, np.ones_like(rb), fmt="o", ms=6, capsize=3, color=col)
        axs[3, c].axhline(0.0, color=FG, lw=1.0, path_effects=[])
        axs[3, c].axhspan(-2, 2, color=DIM, alpha=0.12, lw=0)
        axs[2, c].set(ylabel=r"$\xi(r)$")
        axs[3, c].sharex(axs[2, c])
        chi2c, dof = chi2_cov(xb - xin, cov)
        corr = cov / np.sqrt(np.outer(np.diag(cov), np.diag(cov)))
        lead = float(np.linalg.eigvalsh(corr).max() / corr.shape[0])          # variance in the common (leading) mode
        axs[3, c].set(xlabel=r"$|r|$ [Mpc]", ylabel=r"$(\hat\xi - \xi_{\rm in})/\sigma_\xi$", ylim=(-5, 5))
        axs[3, c].set_title(f"common mode: {lead:.0%} of the variance; " + rf"$\chi^2/\mathrm{{dof}}$ (cov.) = {chi2c:.2f}",
                            fontsize=FONT_MIN["legend"])
        axs[2, c].legend(fontsize=FONT_MIN["legend"])
        out[title] = dict(chi2_dof=float(np.mean(((Pb - Pin_b) / eb) ** 2)), max_rel=float(np.max(np.abs(Pb / Pin_b - 1))),
                          n_kbins=int(kb.size), xi_chi2_dof=float(np.mean(((xb - xin) / xe) ** 2)), xi_chi2_cov=chi2c, xi_dof=dof,
                          xi_lead=lead, xi_mean_pull=float(np.mean((xb - xin) / xe)))
    return fig, axs, out


def power_kmu(m, k_edges, mu_edges, los_axis: int = -1):
    """P in bins of (|k|, mu = |k_par| / |k|), k_par along ``los_axis``: mean over draws and modes.

    m : ``field_power_modes`` output. Returns (k_b (n_mu, n_k), P_b, err (scatter between draws / sqrt(N_real), or the
    Gaussian P / sqrt(n) for one draw), n_indep (half-space modes))."""
    k, kv, w = m["k"], m["kvec"], m["w"]
    mu = np.abs(kv[:, los_axis]) / np.maximum(k, 1e-300)
    ik = np.digitize(k, k_edges) - 1
    im = np.clip(np.digitize(mu, mu_edges) - 1, 0, len(mu_edges) - 2)        # mu = 1 into the last wedge
    ok = (ik >= 0) & (ik < len(k_edges) - 1) & (k > 0)
    nk, nm = len(k_edges) - 1, len(mu_edges) - 1
    flat = im[ok] * nk + ik[ok]
    W = np.bincount(flat, weights=w[ok], minlength=nk * nm)
    kb = np.bincount(flat, weights=(w * k)[ok], minlength=nk * nm) / np.maximum(W, 1e-300)
    Pr = np.stack([np.bincount(flat, weights=(w * Pi)[ok], minlength=nk * nm) for Pi in m["P"]]) / np.maximum(W, 1e-300)
    Pm = Pr.mean(0)
    n_ind = np.bincount(flat, minlength=nk * nm)
    err = Pr.std(0, ddof=1) / np.sqrt(Pr.shape[0]) if Pr.shape[0] > 1 else Pm / np.sqrt(np.maximum(n_ind, 1))
    shp = (nm, nk)
    bad = (W == 0).reshape(shp)
    f = lambda x: np.where(bad, np.nan, x.reshape(shp))
    return f(kb), f(Pm), f(err), n_ind.reshape(shp)


def _occupied_edges(values, edges):
    """The bin edges with every empty bin merged into the next one, so that each bin holds at least one of ``values``
    (a log bin narrower than the spacing of the discrete modes along an axis would be empty: a blank row or column)."""
    v = np.unique(np.asarray(values, dtype=float))
    out = [edges[0]]
    for hi in edges[1:]:
        if np.any((v >= out[-1]) & (v < hi)):
            out.append(hi)
    out[-1] = edges[-1]                                    # trailing empty bins join the last occupied one
    return np.asarray(out)


def plot_kpar_kperp_maps(cases: dict, *, n_lin: int = 6, n_log: int = 20, figsize=(20, 9)):
    """The measured P of 2D and 3D GRF draws in the (k_par, k_perp) plane (notebook 04).

    The measured P as colour, with measured iso-power contours (solid) on the input ones (dashed), at the same levels.
    The input is binned exactly like the measurement (its mean over the modes of every bin, at the same bin centres),
    so the two sets of contours differ only by the noise. One mode per bin for the first ``n_lin`` modes, then ``n_log``
    logarithmic bins (``hybrid_k_edges``), so k_par = 0 and k_perp = 0 are not shown; along each axis an empty bin is
    merged into the next (``_occupied_edges``), so no row or column is blank. The returned chi2/dof is that of
    (P_meas - P_in) / sigma_P per bin, sigma_P = P_in / sqrt(n_modes N_real) (mode counting: the mean of n independent
    exponential |delta_k|^2). cases : {title: (grf, deltas)} of 2D / 3D fields. Returns (fig, axs (n_cases,),
    {title: chi2/dof}).
    """
    n = len(cases)
    fig, axs = plt.subplots(1, n, figsize=figsize, layout="constrained", squeeze=False)
    out = {}
    for c, (title, (grf, deltas)) in enumerate(cases.items()):
        m = field_power_modes(deltas, grf.L)
        kmax = min(np.pi * N / L for N, L in zip(grf.shape, grf.L))
        e = hybrid_k_edges(2 * np.pi / min(grf.L), kmax, n_lin=n_lin, n_log=n_log)
        los = grf.los_axis % grf.ndim
        kv = m["kvec"][m["k"] > 0]
        ep = _occupied_edges(np.abs(kv[:, los]), e)                             # k_par: the modes along the line of sight
        eq = _occupied_edges(np.linalg.norm(np.delete(kv, los, axis=1), axis=-1), e)   # k_perp
        Pm, W, _, Nind = power_map_kpar_kperp(m, ep, eq, grf.los_axis)
        Pin = power_map_kpar_kperp(dict(m, P=input_power_modes(grf, m)), ep, eq, grf.los_axis)[0]
        kcp, kcq = np.sqrt(ep[1:] * ep[:-1]), np.sqrt(eq[1:] * eq[:-1])
        good = np.isfinite(Pm) & (Pm > 0) & np.isfinite(Pin)
        norm = mpl.colors.LogNorm(np.nanmin(Pm[good]), np.nanmax(Pm[good]))
        cmap_ = plt.get_cmap(CMAPS["sequential"]).copy()
        cmap_.set_bad((0, 0, 0, 0))
        ax = axs[0, c]
        im = ax.pcolormesh(ep, eq, np.ma.masked_invalid(Pm).T, cmap=cmap_, norm=norm, shading="flat")
        levels = 10.0 ** np.arange(np.ceil(np.log10(norm.vmin)), np.log10(norm.vmax))
        ci = ax.contour(kcp, kcq, np.ma.masked_invalid(Pin).T, levels=levels, colors=FG, linewidths=1.4, linestyles="--")
        cm = ax.contour(kcp, kcq, np.ma.masked_invalid(Pm).T, levels=levels, colors=PALETTE["red"], linewidths=1.6, linestyles="-")
        ax.clabel(ci, fmt=lambda x: f"$10^{{{np.log10(x):.0f}}}$", fontsize=FONT_MIN["floor"], inline=False)
        ax.plot([], [], color=FG, ls="--", label="input iso-power contours (binned alike)")
        ax.plot([], [], color=PALETTE["red"], ls="-", label="measured contours, same levels")
        ax.legend(loc="lower left", fontsize=FONT_MIN["legend"])
        fig.colorbar(im, ax=ax, label=r"measured $P$ [Mpc$^{d}$]", shrink=0.9)
        sig = Pin / np.sqrt(np.maximum(Nind * deltas.shape[0], 1))
        pull = np.where(good & (Nind > 0), (Pm - Pin) / sig, np.nan)
        chi2 = float(np.nanmean(pull ** 2))
        out[title] = chi2
        ax.set(xscale="log", yscale="log", xlim=(e[0], e[-1]), ylim=(e[0], e[-1]), aspect="equal", title=title,
               xlabel=r"$k_\parallel$ (" + ("last axis" if grf.los_axis == grf.ndim - 1 else f"axis {grf.los_axis}") + r") [Mpc$^{-1}$]",
               ylabel=r"$k_\perp$ [Mpc$^{-1}$]")
    return fig, axs[0], out


MU_WEDGES = (0.0, 0.5, 0.8, 0.95, 1.0)   # mu = k_par / |k|: the four wedges of Lya 3D analyses (arXiv:2308.10950, Fig. 4)


MU_COLORS = ("blue", "cyan", "green", "red")             # one colour per wedge, in every panel (PALETTE keys)


def plot_pk_mu_wedges(cases: dict, *, mu_edges=MU_WEDGES, n_bins: int = 14, figsize=(24, 13)):
    """P(k) in wedges of mu = |k_par| / |k| (k_par along ``grf.los_axis``), measured against the input (notebook 04); the
    way Lya 3D analyses show the anisotropic signal, wedge by wedge.

    One column per case: on top, every wedge in one panel (points with error bars: measured, scatter between draws /
    sqrt(N_real); dashed line through markers: the input averaged over the modes of each (k, mu) bin), each wedge in
    one colour (``MU_COLORS``); below, measured / input - 1 per wedge in the same colours. Every wedge is offset by a
    small factor in k so that its points show; its input is drawn at the same offset k, so a point and its input share
    one x in both panels. For an isotropic input every wedge recovers P(k). cases : {title: (grf, deltas)}. Returns
    (fig, axs (2, n_cases), {title: chi2/dof per wedge}).
    """
    nm = len(mu_edges) - 1
    n = len(cases)
    fig, axs = plt.subplots(2, n, figsize=figsize, layout="constrained", squeeze=False, sharex="col",
                            gridspec_kw={"height_ratios": [2.0, 1.0]})
    out = {}
    cols = [PALETTE[MU_COLORS[w % len(MU_COLORS)]] for w in range(nm)]
    for c, (title, (grf, deltas)) in enumerate(cases.items()):
        m = field_power_modes(deltas, grf.L)
        kmax = min(np.pi * N / L for N, L in zip(grf.shape, grf.L))
        edges = np.geomspace(2 * np.pi / min(grf.L), kmax, n_bins + 1)
        kb, Pb, eb, _ = power_kmu(m, edges, mu_edges, grf.los_axis)
        Pin = power_kmu(dict(m, P=input_power_modes(grf, m)), edges, mu_edges, grf.los_axis)[1]
        out[title] = []
        a0, a1 = axs[0, c], axs[1, c]
        shift = np.exp(0.025 * (np.arange(nm) - (nm - 1) / 2))      # a small offset in k per wedge, so every point shows
        for w in range(nm):
            ok = np.isfinite(Pb[w]) & np.isfinite(eb[w]) & (eb[w] > 0)
            chi2 = float(np.mean(((Pb[w][ok] - Pin[w][ok]) / eb[w][ok]) ** 2))
            out[title].append(chi2)
            lab = rf"${mu_edges[w]:g} \leq \mu < {mu_edges[w + 1]:g}$ ($\chi^2/\mathrm{{dof}}$ = {chi2:.2f})"
            x = kb[w][ok] * shift[w]
            a0.plot(x, Pin[w][ok], color=cols[w], lw=1.8, ls="--", marker="_", ms=12, mew=1.8)
            a0.errorbar(x, Pb[w][ok], eb[w][ok], fmt="o", ms=5, capsize=3, color=cols[w], label=lab)
            a1.errorbar(x, Pb[w][ok] / Pin[w][ok] - 1, eb[w][ok] / Pin[w][ok], fmt="o-", ms=5, capsize=3, lw=1.2, color=cols[w])
        a0.plot([], [], color=FG, ls="--", lw=1.8, marker="_", ms=12, label="input, averaged over the modes of each bin (one per wedge)")
        residual_band(a1, 0.05)
        a0.set(xscale="log", yscale="log", title=title, ylabel=r"$P(k, \mu)$")
        a1.set(xscale="log", ylim=(-0.3, 0.3), xlabel=r"$|k|$ [Mpc$^{-1}$]", ylabel="measured / input − 1")
        a0.legend(fontsize=FONT_MIN["legend"])
    return fig, axs, out


# --------------------------------------------------------------------------- selection -> deposition -> tau (notebook 07)
def cell_colors(cells) -> dict:
    """{cell index: colour} for the crossed cells, in the order given: Tol's bright palette without its grey, cycled
    (colour-blind safe, qualitative; neighbours along a path never share a colour unless more than six are crossed)."""
    cyc = [PALETTE[k] for k in ("blue", "red", "green", "yellow", "cyan", "purple")]
    out = {}
    for c in cells:
        c = int(c)
        if c not in out:
            out[c] = cyc[len(out) % len(cyc)]
    return out


def voronoi_polygons_2d(generators, box: float) -> list:
    """The periodic Voronoi cell (a polygon (n_v, 2)) of every 2D generator in [0, box)^2 (scipy on 3 x 3 copies)."""
    from scipy.spatial import Voronoi
    g = np.asarray(generators, dtype=float)
    shifts = np.array([(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1)], dtype=float) * box
    tiled = np.concatenate([g + s for s in shifts])
    vor = Voronoi(tiled)
    n0 = 4 * len(g)                                              # the copy with shift (0, 0)
    return [vor.vertices[vor.regions[vor.point_region[n0 + i]]] for i in range(len(g))]


def plot_deposition_steps(cells, selected, bundle, seg, ab, frame, v, tau_pieces, *, tau_pieces_periodic=None,
                          period: float = None, plane_axes=(0, 1), figsize=None):
    """Selection -> deposition -> tau on toy cells, drawn from the objects of each step (notebook 07).

    cells : every ``core.Cells`` of the toy box (generators in one plane: the cells are its 2D Voronoi cells), with
    ``ids`` = their indices (the velocities are those of the species of ``ab``); selected : the cells kept by the
    selection (``seg`` and ``ab`` index them); bundle : the
    ``SkewerBundle`` (one skewer in that plane, along ``plane_axes[0]``); seg : the ``SegmentComponents`` of the Voronoi
    deposition; ab : its ``Absorbers``; frame : the ``ExactFrame``; v : (N_v,) [km/s]; tau_pieces : (N_p, N_v) the
    contribution of every piece (``compute_optical_depth(..., per_piece=True)``), in the order of ``ab``.
    tau_pieces_periodic : optional (N_p, N_v), the same with ``velocity_period`` (period ``period`` [km/s]): adds (d).

    (a) the plane: every cell outlined, the crossed ones filled in their own colour (``cell_colors``), their generators
    and peculiar velocities (arrows, in-plane); the skewer with the styles of 03 (points, midpoints, dual segments) and
    the ends of the pieces. (b) along the same axis: n of every piece (bars in its cell's colour) and the dual segments.
    (c) tau: every piece's profile in its cell's colour and their sum; per crossed cell, a marker at the Hubble
    velocity of its centroid and an arrow to its line centre, shifted by the line-of-sight projection of its peculiar
    velocity. (d) the periodic spectrum on the axis of (c): the period [0, P) and, shaded, its copies; every piece's
    wrapped profile in its cell's colour and their sum, on the sum of (c) (thick, light); per cell whose line centre
    is outside the period, an arrow from it to its image inside (effect (i)); the wings that cross an end re-enter at
    the other (effect (ii)). Returns (fig, axs, colours).
    """
    from matplotlib.collections import PolyCollection
    from skewerjax.core.constants import phys
    from skewerjax.core.skewers import path_geometry
    periodic = tau_pieces_periodic is not None
    if periodic and period is None:
        raise ValueError("tau_pieces_periodic needs its period [km/s].")
    figsize = figsize or ((22, 27) if periodic else (22, 21))
    i0, i1 = plane_axes
    box = cells.box
    ids = np.asarray(selected.ids)                                # selected index -> index in ``cells``
    gen = np.asarray(cells.positions)[:, [i0, i1]]
    vel = np.asarray(cells.species[ab.species]["velocity"] if hasattr(cells, "species") else cells.fields["velocity"])[:, [i0, i1]]
    pts3 = np.asarray(bundle.spatial_points[0], dtype=float)
    pts = pts3[:, [i0, i1]]
    g = {k: np.asarray(x) for k, x in path_geometry(pts3).items()}
    live = ab.n[0] > 0
    order = np.argsort(ab.s[0][live])
    s_p, l_p, c_p, n_p, u_p = (np.asarray(getattr(ab, f))[0][live][order] for f in ("s", "length", "cell", "n", "v_los"))
    c_p = ids[c_p]
    tau_p = np.asarray(tau_pieces)[live][order]
    colors = cell_colors(c_p)
    x0 = pts[0, 0]
    ratios = [2.0, 1.0, 1.7] + ([1.7] if periodic else [])
    fig, axs = plt.subplots(len(ratios), 1, figsize=figsize, gridspec_kw={"height_ratios": ratios}, layout="constrained")
    axs[1].sharex(axs[0])                                         # (a) and (b): the same x axis [cMpc]
    polys = voronoi_polygons_2d(gen, box)
    axs[0].add_collection(PolyCollection(polys, facecolors="none", edgecolors=DIM, linewidths=0.8))
    for c, col in colors.items():
        axs[0].add_collection(PolyCollection([polys[c]], facecolors=col, alpha=0.35, edgecolors=col, linewidths=1.5))
    axs[0].scatter(*gen.T, **geometry_style("generator"), label="mesh-generating points (one cell each)")
    crossed = np.array(list(colors))
    sc = 0.15 * box / max(np.abs(vel[crossed]).max(), 1e-30)
    axs[0].quiver(*gen[crossed].T, *(sc * vel[crossed]).T, color=[colors[c] for c in crossed], angles="xy", scale_units="xy",
                  scale=1.0, width=0.004, zorder=7)
    axs[0].plot([], [], color=FG, lw=2, label="peculiar velocity of a crossed cell (at its generator)")
    ds = geometry_style("dual_segment")
    for i, (lo, x, hi) in enumerate(zip(g["mid_lo"], pts3, g["mid_hi"])):
        q = np.stack([lo, x, hi])[:, [i0, i1]]
        axs[0].plot(*q.T, color=ds["colors"][i % 2], lw=ds["lw"], alpha=ds["alpha"], solid_capstyle="butt", path_effects=[],
                    zorder=4, label="dual segments" if i == 0 else None)
    tan = g["tangents"][0][[i0, i1]]
    ends = np.unique(np.concatenate([s_p - 0.5 * l_p, s_p + 0.5 * l_p]))
    axs[0].scatter(*(pts[0][:, None] + tan[:, None] * ends[None]), **geometry_style("piece_end"), label="ends of the pieces")
    axs[0].scatter(*pts.T, **geometry_style("point"), label=r"sampling points $x_i$")
    axs[0].scatter(*(0.5 * (pts[1:] + pts[:-1])).T, **geometry_style("midpoint"), label=r"midpoints $m_{i\pm1/2}$")
    pad = 0.08 * box
    axs[0].set(xlim=(x0 - pad, pts[-1, 0] + pad), ylim=(pts[:, 1].mean() - 0.25 * box, pts[:, 1].mean() + 0.25 * box),
               xlabel="x [cMpc]", ylabel="y [cMpc]")
    axs[0].set_aspect("equal", adjustable="datalim")             # equal scales, the x range of (b)
    axs[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncols=3, fontsize=FONT_MIN["legend"])
    xs = x0 + s_p * tan[0]
    for x, l, c, nn in zip(xs, l_p, c_p, n_p):
        axs[1].bar(x, nn, width=l * tan[0], color=colors[int(c)], alpha=0.75, edgecolor=FG, linewidth=0.6)
    for i, (lo, hi) in enumerate(zip(g["mid_lo"][:, i0], g["mid_hi"][:, i0])):
        axs[1].plot([lo, hi], [0, 0], color=ds["colors"][i % 2], lw=ds["lw"], alpha=ds["alpha"], solid_capstyle="butt",
                    path_effects=[], clip_on=False)
    axs[1].scatter(pts[:, 0], np.zeros(len(pts)), **geometry_style("point"))
    axs[1].scatter(0.5 * (pts[1:, 0] + pts[:-1, 0]), np.zeros(len(pts) - 1), **geometry_style("midpoint"))
    axs[1].set(xlabel="x [cMpc] (along the skewer)", ylabel=r"$n$ of each piece [cm$^{-3}$]")
    axs[1].set_ylim(bottom=0)
    a_d = frame.a_dagger
    v_H = phys.c_kms * np.log(a_d / frame.scale_factors(0, s_p))
    beta = -frame.velocity(0, s_p, u_p) / phys.c_kms
    v_c = v_H + 0.5 * phys.c_kms * (np.log1p(-beta) - np.log1p(beta))
    tot = tau_p.sum(0)
    axs[2].plot(v, tot, color=FG, lw=2.6, label=r"$\tau$: the sum over the pieces")
    for t, c in zip(tau_p, c_p):
        axs[2].plot(v, t, color=colors[int(c)], lw=1.6, alpha=0.9)
    rows = {}
    for r, c in enumerate(colors):                                # one row per cell, in the order of the path
        y = tot.max() * (1.1 + 0.09 * (r % 4))
        k = np.nonzero(c_p == c)[0]
        w = l_p[k] * n_p[k]
        vh, vc = np.average(v_H[k], weights=w), np.average(v_c[k], weights=w)
        rows[c] = (y, vc)
        axs[2].scatter([vh], [y], marker="o", s=70, facecolors="none", edgecolors=colors[c], linewidths=2, zorder=8)
        axs[2].annotate("", (vc, y), (vh, y), arrowprops=dict(arrowstyle="-|>", color=colors[c], lw=2.0, shrinkA=4, shrinkB=0))
    axs[2].scatter([], [], marker="o", s=70, facecolors="none", edgecolors=FG, linewidths=2,
                   label=r"Hubble velocity of the cell's pieces, $v_H = c\ln(a_\dagger/a)$; arrow: $+\,v_\parallel$")
    axs[2].set(xlabel=LABELS["v_dagger"], ylabel=r"$\tau$ ($\mathrm{Ly}\alpha$)", ylim=(0, 1.75 * tot.max()))
    axs[2].legend(loc="upper right", fontsize=FONT_MIN["legend"], ncols=2)
    if periodic:
        axd = axs[3]
        axd.sharex(axs[2])
        P = float(period)
        tau_w = np.asarray(tau_pieces_periodic)[live][order]
        tot_w = tau_w.sum(0)
        for lo, hi in ((min(v[0], 0.0), 0.0), (P, max(v[-1], P))):
            axd.axvspan(lo, hi, color=PALETTE["grey"], alpha=0.22, lw=0)
        for vb in (0.0, P):
            axd.axvline(vb, color=DIM, ls=":", lw=1.8)
        axd.plot(v, tot, color=FG, **MODE_STYLES["reference"], label=r"$\tau$ without wrap, as in (c)")
        axd.plot(v, tot_w, color=FG, lw=2.6, label=r"$\tau$ with velocity_period=True: periodic, period $P$")
        for t, c in zip(tau_w, c_p):
            axd.plot(v, t, color=colors[int(c)], lw=1.6, alpha=0.9)
        top_w = max(tot_w.max(), tot.max())
        out = [c for c, (_, vc) in rows.items() if vc < 0.0 or vc >= P]
        for r, c in enumerate(out):                               # effect (i): the line centre's image in [0, P), above the curves
            vc = rows[c][1]
            y = top_w * (1.1 + 0.08 * r)
            img = vc - P * np.floor(vc / P)
            axd.scatter([vc], [y], marker="o", s=70, facecolors="none", edgecolors=colors[c], linewidths=2, zorder=8)
            axd.annotate("", (img, y), (vc, y), arrowprops=dict(arrowstyle="-|>", color=colors[c], lw=2.0, ls="--", shrinkA=4, shrinkB=0))
        if out:
            axd.plot([], [], color=FG, lw=2.0, ls="--", label=r"(i) line centre $v_c$ outside the period $\to$ $v_c \mp P$")
        ytop = top_w * (1.3 + 0.08 * len(out))
        for xt, txt in ((0.5 * (v[0] + 0.0), "copy"), (0.5 * P, r"one period $[0, P)$"), (0.5 * (P + v[-1]), "copy")):
            axd.text(xt, 0.97 * ytop, txt, ha="center", va="top", fontsize=FONT_MIN["annotation"], color=FG)
        axd.set(xlabel=LABELS["v_dagger"], ylabel=r"$\tau$ ($\mathrm{Ly}\alpha$), periodic", ylim=(0, ytop))
        axd.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncols=3, fontsize=FONT_MIN["legend"])
    return fig, axs, colors


# --------------------------------------------------------------------------- frames and periodicity (notebook 07)
def plot_frames_axes(v, F_own, line_bundle, *, v_ref=None, F_ref=None, a_dagger: float, a_obs: float = 1.0, centres=None,
                     figsize=(22, 15)):
    """The same flux in each line's own frame, on the global velocity axis and in observed wavelength: three axes.

    v : (N_v,) line-centred velocities v_dagger [km/s]; F_own : (N_lines, N_v) every line in its own frame. (a) the
    line-centred frames: the same gas absorbs at the same v_dagger in every line (dotted: ``centres``); (b) the global
    velocity axis of the reference line: ``F_ref`` (N_lines, N_v_ref) on ``v_ref``, e.g. from
    ``compute_optical_depth(..., line_frame="reference")``; (c) the observed wavelength of every pixel,
    ``spectral.line_wavelength_grid_A`` (lambda_obs = lambda_j (a_obs / a_dagger) e^{v/c}). Returns (fig, axs).
    """
    from skewerjax.core.spectral import line_wavelength_grid_A
    v = np.asarray(v, dtype=float)
    F_own = np.asarray(F_own, dtype=float)
    lines = line_bundle.lines
    dv_j = np.asarray(line_bundle.delta_v, dtype=float)
    cols = [ENTITY_COLORS.get(l.id, species_color(l.id, j)) for j, l in enumerate(lines)]
    lam = np.asarray(line_wavelength_grid_A(v, line_bundle, a_dagger, a_obs))
    fig, axs = plt.subplots(3, 1, figsize=figsize, gridspec_kw={"height_ratios": [2.0, 1.4, 1.4]}, layout="constrained")
    for j, l in enumerate(lines):
        axs[0].plot(v, F_own[j], color=cols[j], lw=2.4 if j == 0 else 1.8, ls="-" if j == 0 else "--", label=f"{l.id}, own frame")
        if F_ref is not None:
            axs[1].plot(v_ref, F_ref[j], color=cols[j], lw=2.0,
                        label=f"{l.id}" + (rf": at $v + \Delta v_j$ = $v$ {dv_j[j]:+.0f} km/s" if dv_j[j] else ""))
        axs[2].plot(lam[j], F_own[j], color=cols[j], lw=2.0, label=f"{l.id}: " + r"$\lambda_j\,(a_{\rm obs}/a_\dagger)\,e^{v_\dagger/c}$")
        axs[2].axvline(l.lambda0_A * a_obs / a_dagger, color=cols[j], ls="-.", lw=1.2, path_effects=[])
    if centres is not None:
        for c0 in np.atleast_1d(centres):
            axs[0].axvline(c0, color=FG, ls=":", lw=1.2, path_effects=[])
    for j in range(1, len(lines)):
        _textbox(axs[0], 0.02, 0.12 + 0.1 * j, rf"{lines[j].id} sits $\Delta v_j = c\ln(\lambda_{{\rm ref}}/\lambda_j)$ = "
                 f"{dv_j[j]:+.1f} km/s from {lines[0].id} on the global axis; in its own frame it absorbs at the same "
                 rf"$v_\dagger$ as {lines[0].id}", color=cols[j], transform=axs[0].transAxes)
    axs[0].set(xlabel=r"line-centred velocity $v_\dagger$ [km/s] (each line in its own frame)", ylabel=LABELS["F"], xlim=(v[0], v[-1]))
    axs[1].set(xlabel=f"global velocity axis: v on the {lines[0].id} frame [km/s]", ylabel=LABELS["F"])
    axs[2].set(xlabel=r"observed wavelength $\lambda_{\rm obs}$ [Å]", ylabel=LABELS["F"])
    for ax in axs:
        ax.set_ylim(-0.03, 1.08)
        ax.legend(loc="lower right", fontsize=FONT_MIN["legend"])
    return fig, axs


def plot_periodic_pieces(d: dict, *, figsize=(22, 24)):
    """Where every piece absorbs, without wrap and with ``velocity_period=True`` (``tutorial.periodic_pieces_demo``).

    (a) the path: the pieces at their positions s (bars, one colour each; height log10 N), with their peculiar
    velocities (arrows); top axis: the Hubble velocity v_H(s), up to P at the far end. (b) velocity space, before any
    wrap: every piece goes from its Hubble velocity v_H (open marker) to its line centre v_c = v_H + its peculiar part
    (arrow); the period [0, P) is shaded; a piece whose line centre is beyond P is effect (i). (c) no wrap: every
    piece's tau on [0, P); what falls outside is lost (faint, dashed: the profiles on the extended axis). (d) wrapped
    detunings: effect (i), the piece beyond P absorbs at v_c - P (arrow); effect (ii), the damped wing past P re-enters
    near v = 0 (arrow). (e) the damped piece alone: the nearest image (what the code keeps) against the exact periodic
    sum over images, in F. Returns (fig, axs).
    """
    v, P, s, L = d["v"], d["P"], d["s"], d["L"]
    n = len(s)
    cyc = [PALETTE[k] for k in ("blue", "red", "green", "yellow", "purple", "cyan")]
    cols = [cyc[i % len(cyc)] for i in range(n)]
    fig, axs = plt.subplots(5, 1, figsize=figsize, gridspec_kw={"height_ratios": [1.0, 1.0, 1.4, 1.4, 1.1]}, layout="constrained")
    ax = axs[0]
    for i in range(n):
        ax.bar(s[i], d["logN"][i] - 12, width=max(d["length"][i], 0.12), bottom=12, color=cols[i], edgecolor=FG, linewidth=0.8)
        if d["u"][i]:
            ax.annotate("", (s[i] + 0.01 * L * d["u"][i] / 10, d["logN"][i] + 0.4), (s[i], d["logN"][i] + 0.4),
                        arrowprops=dict(arrowstyle="-|>", color=cols[i], lw=2))
        crowded = i + 1 < n and s[i + 1] - s[i] < 0.05 * L          # a close neighbour on the right: label to the left
        ax.text(s[i] - (0.01 * L if crowded else 0.0), d["logN"][i] + 0.8, f"{i}: u = {d['u'][i]:+.0f}", color=cols[i],
                ha="right" if crowded else "center", fontsize=FONT_MIN["annotation"])
    ax.set(xlim=(0, L), ylim=(12, 21.5), xlabel=r"path coordinate $s$ [cMpc] (observer at $s = 0$)", ylabel=r"$\log_{10} N$")
    top = ax.secondary_xaxis("top", functions=(lambda x: np.interp(x, np.linspace(0, L, 400), d["v_of_s"](np.linspace(0, L, 400))),
                                               lambda y: np.interp(y, d["v_of_s"](np.linspace(0, L, 400)), np.linspace(0, L, 400))))
    top.set_xlabel(r"$v_H(s) = c\ln[a(0)/a(s)]$ [km/s]")
    ax = axs[1]
    ax.axvspan(0, P, color=DIM, alpha=0.12, lw=0)
    ax.text(0.5 * P, 0.15 * (n + 1), "one period [0, P): the spectral grid", ha="center", color=FG, fontsize=FONT_MIN["annotation"])
    for i in range(n):
        ax.scatter([d["v_H"][i]], [i + 1], marker="o", s=80, facecolors="none", edgecolors=cols[i], linewidths=2, zorder=5)
        ax.annotate("", (d["v_c"][i], i + 1), (d["v_H"][i], i + 1), arrowprops=dict(arrowstyle="-|>", color=cols[i], lw=2.2))
        ax.scatter([d["v_c"][i]], [i + 1], marker="o", s=70, color=cols[i], zorder=6)
    j = d["strong"]
    for i in range(n):                                           # the wrap step: (i) a line centre outside [0, P) moves by -+P
        if not 0 <= d["v_c"][i] < P:
            w = d["v_c"][i] - P * np.floor(d["v_c"][i] / P)
            ax.annotate("", (w, i + 1), (d["v_c"][i], i + 1),
                        arrowprops=dict(arrowstyle="-|>", color=cols[i], lw=1.8, ls="--", connectionstyle="arc3,rad=-0.12"))
            ax.scatter([w], [i + 1], marker="D", s=70, color=cols[i], edgecolors=FG, zorder=7)
            ax.text(w + 25, i + 1.25, f"(i) wrapped: $v_c - P$ = {w:.0f}", color=cols[i], fontsize=FONT_MIN["annotation"])
    wing = d["v_ext"][d["tau_ext"][j] > 1.0]                      # (ii) the damped wing where tau > 1, and its part past P
    ax.plot([wing.min(), wing.max()], [j + 1, j + 1], color=cols[j], lw=8, alpha=0.35, solid_capstyle="butt", path_effects=[])
    past = wing[wing >= P]
    if past.size:
        ax.plot([past.min() - P, past.max() - P], [j + 1, j + 1], color=cols[j], lw=8, alpha=0.35, ls=(0, (2, 1)),
                solid_capstyle="butt", path_effects=[])
        ax.text(past.max() - P + 25, j + 1.25, r"(ii) the wing past $P$ ($\tau > 1$) re-enters here", color=cols[j],
                fontsize=FONT_MIN["annotation"])
    for x in (0, P):
        ax.axvline(x, color=FG, lw=1.4, ls="--", path_effects=[])
    ax.set(xlim=(d["v_ext"][0], d["v_ext"][-1]), ylim=(0.3, n + 0.9), yticks=range(1, n + 1), yticklabels=[f"piece {i}" for i in range(n)],
           xlabel=r"velocity [km/s]: $v_H$ (open) $\to$ line centre $v_c$ (filled); diamonds and dashed: where the wrap puts it")
    for k, (ax, key) in enumerate(((axs[2], "tau_nowrap"), (axs[3], "tau_wrap"))):
        ax.axvspan(0, P, color=DIM, alpha=0.08, lw=0)
        for i in range(n):
            if k == 0:
                ax.plot(d["v_ext"], d["tau_ext"][i], color=cols[i], lw=1.2, ls="--", alpha=0.6)
            ax.plot(v, d[key][i], color=cols[i], lw=2.0, label=f"piece {i}")
        ax.plot(v, d[key].sum(0), color=FG, lw=1.2, label="total")
        for x in (0, P):
            ax.axvline(x, color=FG, lw=1.4, ls="--", path_effects=[])
        ax.set(yscale="symlog", ylim=(0, 5e6), xlim=(d["v_ext"][0], d["v_ext"][-1]), ylabel=r"$\tau$ of each piece")
        ax.yaxis.set_major_locator(mpl.ticker.SymmetricalLogLocator(base=100, linthresh=1.0))
    i_out = int(np.argmax(d["v_c"] - P))
    j = d["strong"]
    axs[2].set(xlabel=r"$v_\dagger$ [km/s]: velocity_period=False (no wrap); dashed: the profiles beyond the grid, lost")
    axs[3].set(xlabel=r"$v_\dagger$ [km/s]: velocity_period=True (nearest image modulo P)")
    if d["v_c"][i_out] > P:
        k = int(np.argmax(d["tau_wrap"][i_out]))
        axs[3].annotate(f"(i) piece {i_out}, line centre beyond $P$:\nabsorbs at $v_c - P$", (v[k], d["tau_wrap"][i_out][k]),
                        (v[k] + 0.12 * P, 3e2), arrowprops=dict(arrowstyle="-|>", color=cols[i_out], lw=2.0),
                        color=cols[i_out], fontsize=FONT_MIN["annotation"], ha="left", va="bottom")
    axs[3].annotate(f"(ii) the damped wing of piece {j}\npast $P$ re-enters near $v = 0$", (8.0, 3e2), (-0.33 * P, 3e3),
                    arrowprops=dict(arrowstyle="-|>", color=cols[j], lw=2.0), color=cols[j], fontsize=FONT_MIN["annotation"],
                    ha="left", va="bottom")
    axs[2].legend(loc="upper center", ncols=n + 1, fontsize=FONT_MIN["legend"])
    ax = axs[4]
    Fe, Fn = np.exp(-d["tau_strong_exact"]), np.exp(-d["tau_strong_nearest"])
    ax.plot(v, Fe, color=FG, lw=2.4, label=r"exact periodic sum $\sum_n \tau(v + nP)$")
    ax.plot(v, Fn, color=cols[j], lw=1.8, ls="--", label="nearest image only (velocity_period)")
    ax.plot(v, Fn - Fe, color=PALETTE["grey"], lw=1.6, label=r"$F_{\rm nearest} - F_{\rm exact}$")
    ax.axvline(np.mod(d["v_c"][j] - 0.5 * P, P), color=FG, ls=":", lw=1.2, path_effects=[])
    ax.set(xlim=(0, P), ylim=(-0.05, 1.05), xlabel=r"$v_\dagger$ [km/s] (dotted: half a period from the damped piece)",
           ylabel=f"F of piece {j} alone")
    ax.legend(loc="center right", fontsize=FONT_MIN["legend"])
    return fig, axs
