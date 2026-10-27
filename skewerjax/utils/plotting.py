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

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from matplotlib.patheffects import AbstractPathEffect

# --------------------------------------------------------------------------- style
# Tol's "bright" scheme (https://personal.sron.nl/~pault/, Fig. 1; tol_colors 2.2.0 ``colorsets["bright"]``), in Tol's
# order: the only discrete colours of the package's figures.
_TOL_BRIGHT_VENDORED = {
    "blue": "#4477AA",
    "red": "#EE6677",
    "green": "#228833",
    "yellow": "#CCBB44",
    "cyan": "#66CCEE",
    "purple": "#AA3377",
    "grey": "#BBBBBB",
}
try:  # the package, when installed, is the source of the values
    import tol_colors as _tc

    TOL_BRIGHT = {k: v.upper() for k, v in _tc.colorsets["bright"]._asdict().items()}
except Exception:  # ImportError, or an API change: the vendored copy (same values)
    TOL_BRIGHT = dict(_TOL_BRIGHT_VENDORED)
TOL_BRIGHT_CYCLE = list(TOL_BRIGHT.values())
PALETTE = TOL_BRIGHT  # keyed by hue: blue, red, green, yellow, cyan, purple, grey
PALETTE_CYCLE = TOL_BRIGHT_CYCLE  # fixed order of the colour cycle
THEMES = {"paper": dict(FG="black", BG="white", DIM="#8C8C8C"), "dark": dict(FG="white", BG="black", DIM=PALETTE["grey"])}
FG, BG = THEMES["paper"]["FG"], THEMES["paper"]["BG"]  # foreground (text, axes, totals) and background; set by use_style
THEME = "paper"
DIM = THEMES["paper"]["DIM"]  # secondary elements (guide lines, inactive boxes); a darker grey on white
_TOL_CMAPS = {  # Tol's sequential and diverging schemes (https://personal.sron.nl/~pault/, Figs. 12-20), "bad" colour
    "iridescent": (
        [
            "#FEFBE9",
            "#FCF7D5",
            "#F5F3C1",
            "#EAF0B5",
            "#DDECBF",
            "#D0E7CA",
            "#C2E3D2",
            "#B5DDD8",
            "#A8D8DC",
            "#9BD2E1",
            "#8DCBE4",
            "#81C4E7",
            "#7BBCE7",
            "#7EB2E4",
            "#88A5DD",
            "#9398D2",
            "#9B8AC4",
            "#9D7DB2",
            "#9A709E",
            "#906388",
            "#805770",
            "#684957",
            "#46353A",
        ],
        "#999999",
    ),
    "incandescent": (
        ["#CEFFFF", "#C6F7D6", "#A2F49B", "#BBE453", "#D5CE04", "#E7B503", "#F19903", "#F6790B", "#F94902", "#E40515", "#A80003"],
        "#888888",
    ),
    "YlOrBr": (["#FFFFE5", "#FFF7BC", "#FEE391", "#FEC44F", "#FB9A29", "#EC7014", "#CC4C02", "#993404", "#662506"], "#888888"),
    "sunset": (
        ["#364B9A", "#4A7BB7", "#6EA6CD", "#98CAE1", "#C2E4EF", "#EAECCC", "#FEDA8B", "#FDB366", "#F67E4B", "#DD3D2D", "#A50026"],
        "#FFFFFF",
    ),
    "nightfall": (
        [
            "#125A56",
            "#00767B",
            "#238F9D",
            "#42A7C6",
            "#60BCE9",
            "#9DCCEF",
            "#C6DBED",
            "#DEE6E7",
            "#ECEADA",
            "#F0E6B2",
            "#F9D576",
            "#FFB954",
            "#FD9A44",
            "#F57634",
            "#E94C1F",
            "#D11807",
            "#A01813",
        ],
        "#FFFFFF",
    ),
    "BuRd": (["#2166AC", "#4393C3", "#92C5DE", "#D1E5F0", "#F7F7F7", "#FDDBC7", "#F4A582", "#D6604D", "#B2182B"], "#FFEE99"),
    "PRGn": (["#762A83", "#9970AB", "#C2A5CF", "#E7D4E8", "#F7F7F7", "#D9F0D3", "#ACD39E", "#5AAE61", "#1B7837"], "#FFEE99"),
}
# Tol's smooth rainbow (Fig. 21; tol_colors 2.2.0 ``rainbow_linear``, i.e. "rainbow_WhBr"): about 2.8x the perceptual
# length (CIELAB) of iridescent, so cells of close densities stay distinct; reasonably clear in colour-blind vision.
_TOL_CMAPS["rainbow_WhBr"] = (
    [
        "#E8ECFB",
        "#DDD8EF",
        "#D1C1E1",
        "#C3A8D1",
        "#B58FC2",
        "#A778B4",
        "#9B62A7",
        "#8C4E99",
        "#6F4C9B",
        "#6059A9",
        "#5568B8",
        "#4E79C5",
        "#4D8AC6",
        "#4E96BC",
        "#549EB3",
        "#59A5A9",
        "#60AB9E",
        "#69B190",
        "#77B77D",
        "#8CBC68",
        "#A6BE54",
        "#BEBC48",
        "#D1B541",
        "#DDAA3C",
        "#E49C39",
        "#E78C35",
        "#E67932",
        "#E4632D",
        "#DF4828",
        "#DA2222",
        "#B8221E",
        "#95211B",
        "#721E17",
        "#521A13",
    ],
    "#666666",
)
_TOL_CMAPS["iridescentline"] = (_TOL_CMAPS["iridescent"][0][7:-1], "#999999")  # visible on white
_TOL_CMAPS["iridescentline_dark"] = (_TOL_CMAPS["iridescent"][0][1:-7], "#999999")  # visible on black (reversed)
# Sequential maps run from the background colour to the strongest one (low values merge with the background): light to
# dark in the paper style, dark to bright in the dark style (``use_style`` switches them); diverging maps are the same.
# "line": the sequential map for coloured lines and markers, without the end that would vanish on the background.
# "density": the number density of the cells (04, 05, 07), whole range, where neighbouring cells must stay distinct.
_CMAPS_THEME = {
    "paper": {
        "sequential": "tol_iridescent",
        "hot": "tol_incandescent",
        "line": "tol_iridescentline",
        "density": "tol_rainbow_WhBr",
    },
    "dark": {
        "sequential": "tol_iridescent_r",
        "hot": "tol_incandescent_r",
        "line": "tol_iridescentline_dark_r",
        "density": "tol_rainbow_WhBr_r",
    },
}
CMAPS = {**_CMAPS_THEME["paper"], "diverging": "tol_sunset", "diverging_alt": "tol_BuRd"}
# Fixed colour of every entity (species and lines, quantities, codes, modes), the same in every notebook. Lines take
# the bright colours in Tol's order (Lya blue, Si III red, ...); field quantities take the remaining ones.
ENTITY_COLORS = {
    "HI": PALETTE["blue"],
    "HI-LyA": PALETTE["blue"],
    "SiIII": PALETTE["red"],
    "SiIII-1206": PALETTE["red"],
    "SiII": PALETTE["green"],
    "HI-LyB": PALETTE["cyan"],
    "CIV": PALETTE["yellow"],
    "SiIV": PALETTE["purple"],
    "fake_spectra": PALETTE["grey"],
    "analytic": PALETTE["grey"],
    "reference": PALETTE["grey"],
    "total": FG,
    "input": FG,
    "density": PALETTE["green"],
    "temperature": PALETTE["purple"],
    "velocity": PALETTE["yellow"],
    "voronoi": PALETTE["cyan"],
    "tophat": PALETTE["yellow"],
    "crossings": PALETTE["purple"],
}
LINE_COLOR_ORDER = ("blue", "red", "green", "cyan", "yellow", "purple")  # further lines of a bundle, in this order
MODE_STYLES = {  # SkewerJAX default solid; matched (fake_spectra reproduced) dashed; references thick and light underneath
    "default": dict(ls="-", lw=2.0),
    "matched": dict(ls="--", lw=2.0),
    "reference": dict(ls="-", lw=6.0, alpha=0.3),
    "alternative": dict(ls=":", lw=2.2),
    "isolated": dict(ls="-.", lw=1.8),
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
    tick = lambda ax, extra: {
        f"{ax}tick.{k}": v
        for k, v in {
            "color": BG,
            "direction": "in",
            "labelsize": 24,
            "major.pad": 6.0,
            "major.size": 14.0,
            "major.width": 1.5,
            "minor.pad": 3.4,
            "minor.size": 7.0,
            "minor.visible": True,
            "minor.width": 1.0,
            **extra,
        }.items()
    }
    rc = {
        "mathtext.fontset": "cm",
        "text.usetex": False,
        "figure.frameon": True,
        "axes.linewidth": 2.0,
        "axes.titlesize": 32,
        "axes.labelsize": 28,
        "legend.fontsize": 28,
        "legend.fancybox": True,
        "lines.linewidth": 2.5,
        **tick(
            "x",
            {
                "alignment": "center",
                "bottom": True,
                "top": True,
                "labelbottom": True,
                "labeltop": False,
                "major.bottom": True,
                "major.top": True,
                "minor.bottom": True,
                "minor.top": True,
            },
        ),
        **tick(
            "y",
            {
                "alignment": "center_baseline",
                "left": True,
                "right": True,
                "labelleft": True,
                "labelright": False,
                "major.left": True,
                "major.right": True,
                "minor.left": True,
                "minor.right": True,
            },
        ),
    }
    return font, rc


def matplotlib_tol_config():
    """``matplotlib_default_config`` with Tol's bright cycle and the minimum font sizes of ``FONT_MIN``: (font, rc)."""
    font, rc = matplotlib_default_config()
    rc.update(
        {
            "axes.prop_cycle": cycler(color=PALETTE_CYCLE),
            "image.cmap": "tol_iridescent",
            "axes.titlesize": FONT_MIN["title"] + 2,
            "axes.labelsize": FONT_MIN["label"] + 1,
            "figure.titlesize": FONT_MIN["title"] + 3,
            "legend.fontsize": FONT_MIN["legend"],
            "legend.title_fontsize": FONT_MIN["legend"],
            "xtick.labelsize": FONT_MIN["tick"] + 1,
            "ytick.labelsize": FONT_MIN["tick"] + 1,
            "xtick.major.size": 9.0,
            "ytick.major.size": 9.0,
            "xtick.minor.size": 4.5,
            "ytick.minor.size": 4.5,
            "lines.linewidth": 2.0,
            "legend.framealpha": 0.9,
            "legend.edgecolor": DIM,
            "axes.grid": False,
            "figure.dpi": 80,
            "savefig.dpi": 150,
        }
    )
    return font, rc


def matplotlib_paper_config():
    """``matplotlib_tol_config`` on a white background with black text, ticks and axes, no glow: (font, rc). The
    default look of every figure."""
    font, rc = matplotlib_tol_config()
    fg, bg, dim = (THEMES["paper"][k] for k in ("FG", "BG", "DIM"))
    rc.update(
        {
            "image.cmap": _CMAPS_THEME["paper"]["sequential"],
            "figure.facecolor": bg,
            "axes.facecolor": bg,
            "savefig.facecolor": bg,
            "figure.edgecolor": bg,
            "text.color": fg,
            "axes.labelcolor": fg,
            "axes.edgecolor": fg,
            "axes.titlecolor": fg,
            "xtick.color": fg,
            "ytick.color": fg,
            "xtick.labelcolor": fg,
            "ytick.labelcolor": fg,
            "legend.facecolor": bg,
            "legend.edgecolor": dim,
            "legend.labelcolor": fg,
            "legend.framealpha": 0.9,
            "grid.color": dim,
            "grid.alpha": 0.5,
            "hatch.color": fg,
            "patch.edgecolor": fg,
            "boxplot.flierprops.markeredgecolor": fg,
            "path.effects": [],
        }
    )
    return font, rc


def matplotlib_dark_config(glow: bool = True):
    """``matplotlib_tol_config`` on a black background, white text and axes and the ``NeonGlow`` under every line:
    (font, rc). The optional dark look."""
    font, rc = matplotlib_tol_config()
    FG, BG, DIM = (THEMES["dark"][k] for k in ("FG", "BG", "DIM"))
    rc.update(
        {
            "image.cmap": _CMAPS_THEME["dark"]["sequential"],
            "figure.facecolor": BG,
            "axes.facecolor": BG,
            "savefig.facecolor": BG,
            "figure.edgecolor": BG,
            "text.color": FG,
            "axes.labelcolor": FG,
            "axes.edgecolor": FG,
            "axes.titlecolor": FG,
            "xtick.color": FG,
            "ytick.color": FG,
            "xtick.labelcolor": FG,
            "ytick.labelcolor": FG,
            "legend.facecolor": BG,
            "legend.edgecolor": DIM,
            "legend.labelcolor": FG,
            "legend.framealpha": 0.85,
            "grid.color": DIM,
            "grid.alpha": 0.35,
            "hatch.color": FG,
            "patch.edgecolor": FG,
            "boxplot.flierprops.markeredgecolor": FG,
            "path.effects": [NeonGlow()] if glow else [],
        }
    )
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


def use_backend(interactive: bool | None = None) -> str:
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
    """Tol colour map: sequential "iridescent", "incandescent", "YlOrBr", "rainbow_WhBr" (smooth rainbow); diverging "sunset",
    "nightfall", "BuRd", "PRGn"."""
    colors, bad = _TOL_CMAPS[name]
    cm = mpl.colors.LinearSegmentedColormap.from_list(
        f"tol_{name}" + ("_r" if reverse else ""), colors[::-1] if reverse else colors
    )
    cm.set_bad(bad)
    return cm


def check_fonts(fig, minimum: float | None = None) -> list:
    """Texts of ``fig`` smaller than ``minimum`` (default ``FONT_MIN["floor"]``): [(size, text)] (empty = readable)."""
    m = FONT_MIN["floor"] if minimum is None else minimum
    fig.canvas.draw()
    out = []
    for t in fig.findobj(mpl.text.Text):
        if t.get_visible() and t.get_text().strip() and t.get_fontsize() < m - 1e-9:
            out.append((t.get_fontsize(), t.get_text()[:40]))
    return out


# --------------------------------------------------------------------------- small helpers
def add_panel_letters(axes, x: float = 0.012, y: float = 0.975, size: int = 18, start: str = "a", labels=None, box: bool = True):
    """(a), (b), ... in the upper-left corner of every axis (2D or 3D); ``labels``: the texts instead (e.g. "a1", "a2"
    for rows and columns); ``box=False``: no background box."""
    fg, bg = plt.rcParams["text.color"], plt.rcParams["axes.facecolor"]
    axes = list(np.ravel(axes))
    labels = [chr(ord(start) + i) for i in range(len(axes))] if labels is None else list(labels)
    bbox = dict(boxstyle="round,pad=0.15", facecolor=bg, edgecolor="none", alpha=0.85) if box else None
    for ax, lab in zip(axes, labels):
        (ax.text2D if hasattr(ax, "text2D") else ax.text)(
            x,
            y,
            f"({lab})",
            transform=ax.transAxes,
            fontsize=size,
            fontweight="bold",
            zorder=20,
            va="top",
            ha="left",
            color=fg,
            bbox=bbox,
        )


def shade_outside_window(ax, v, period: float, end: str, **shade):
    """Shade the part of a velocity axis ``v`` [km/s] outside the window [0, P) at one ``end`` ("start": v < 0, "end":
    v >= P) and mark the window's edge (dotted); the x limits are set to the axis."""
    v = np.asarray(v, dtype=float)
    shade = dict(dict(color=PALETTE["grey"], alpha=0.22, lw=0), **shade)
    edge = 0.0 if end == "start" else float(period)
    ax.axvspan(*((v[0], edge) if end == "start" else (edge, v[-1])), **shade)
    ax.axvline(edge, color=DIM, ls=":", lw=1.8)
    ax.set_xlim(v[0], v[-1])


def residual_band(ax, tol: float, label: str | None = None, zero: bool = True):
    """Zero line and a shaded +/- ``tol`` band on a residual panel."""
    if zero:
        ax.axhline(0.0, color=plt.rcParams["text.color"], lw=1.0)
    ax.axhspan(-tol, tol, color=PALETTE["grey"], alpha=0.25, lw=0, label=label)


def ten_power_formatter(y, pos):
    """Tick formatter that labels exact powers of ten only (for symlog axes)."""
    if np.isclose(y, 0.0):
        return "0"
    e = int(np.round(np.log10(abs(y))))
    return rf"${'-' if y < 0 else ''}10^{{{e}}}$" if np.isclose(abs(y), 10.0**e) else ""


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


def plot_pk_recovery(
    binned,
    P_input,
    *,
    power=None,
    n_real: int = 1,
    label: str = "measured",
    color=None,
    k_label: str = r"$k\ [\mathrm{s\,km^{-1}}]$",
    P_label: str = r"$P_{\rm 1D}(k)\ [\mathrm{km\,s^{-1}}]$",
    axes=None,
    figsize=(12, 9),
):
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
        fig, axes = plt.subplots(
            2, 1, figsize=figsize, sharex=True, gridspec_kw={"height_ratios": [3, 1.2]}, layout="constrained"
        )
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


ORDER_COLORS = [PALETTE[k] for k in ("grey", "purple", "blue", "green", "yellow", "cyan", "red")]  # by order n (cycled)


def plot_orders_superimposed(
    stats,
    kind: str = "P",
    *,
    k_weighted: bool = False,
    max_lag: float | None = None,
    delta_v_list=None,
    linthresh: float | None = None,
    axes=None,
    figsize=(12, 9),
):
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
        fig, axes = plt.subplots(
            2, 1, figsize=figsize, sharex=True, gridspec_kw={"height_ratios": [3, 1.2]}, layout="constrained"
        )
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
    if kind == "P" and delta_v_list is not None:  # the cross terms oscillate as cos(k dv): features at k = 2 pi n / dv
        dv = np.asarray(delta_v_list, dtype=float)
        first = True
        for d in np.unique(np.round(np.abs(dv[:, None] - dv[None, :]), 6)):
            if d <= 0:
                continue
            for n_ in range(1, int(x.max() * d / (2 * np.pi)) + 1):
                kn = 2 * np.pi * n_ / d
                if kn >= x.min():
                    for a in (ax0, ax1):
                        a.axvline(
                            kn,
                            color=PALETTE["grey"],
                            ls="--",
                            lw=1.0,
                            alpha=0.7,
                            path_effects=[],
                            label=rf"$k = 2\pi n/\Delta v$ ($\Delta v$ = {d:g} km/s)" if (first and a is ax0) else None,
                        )
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
    "v": r"$v\ [\mathrm{km\,s^{-1}}]$",
    "v_ref": r"$v\ [\mathrm{km\,s^{-1}}]$ (Ly$\alpha$ frame)",
    "v_dagger": r"$v_\dagger\ [\mathrm{km\,s^{-1}}]$",
    "u": r"$u\ [\mathrm{km\,s^{-1}}]$ (snapshot frame)",
    "s": r"path length $s$ [cMpc]",
    "x": "x [cMpc]",
    "y": "y [cMpc]",
    "z": "z [cMpc]",
    "F": r"$F = e^{-\tau}$",
    "tau": r"$\tau$",
    "dF": r"$\Delta F$",
    "n": r"$n\ [\mathrm{cm^{-3}}]$",
    "T": r"$T$ [K]",
    "v_los": r"$v_\parallel\ [\mathrm{km\,s^{-1}}]$",
    "k": r"$k\ [\mathrm{s\,km^{-1}}]$",
    "P1D": r"$P_{\rm 1D}\ [\mathrm{km\,s^{-1}}]$",
    "kP": r"$k P_{\rm 1D}/\pi$",
    "xi": r"$\xi_{\rm 1D}$",
    "lag": r"$\Delta v\ [\mathrm{km\,s^{-1}}]$",
    "delta": r"$\delta_F$",
    "sigma": r"$\Sigma_{\rm gas}\ [\mathrm{M_\odot\,ckpc^{-2}}]$",
}
FIGSIZES = {"wide": (20, 6), "single": (13, 6), "tall": (13, 9), "triple": (22, 6), "grid": (22, 12), "dag": (26, 6.6)}

# The dependency graph of the ingredients, derived from the code. Columns are topological levels; the two field sources
# (toy fields and simulation outputs) are interchangeable inputs of the gas cells, side by side in one column.
DAG_NODES = {  # key: (column, row, name, detail)
    "lines": (0, 1.45, "lines", "LineBundle"),
    "cosmology": (0, -1.2, "cosmology", "Cosmology"),
    "skewers": (1, -1.2, "skewers", "SkewerBundle, a(x)"),
    "samplers": (2, 0.52, "toy fields", "PowerLawGRF → cells"),
    "simulation": (2, -0.52, "simulation outputs", "TNG snapshot → cells"),
    "fields": (3, 0.0, "gas cells", "Voronoi cells: n, T, v"),
    "deposition": (4, 0.0, "deposition", "cells → pieces"),
    "profiles": (5, 0.0, "line profiles", r"Voigt $\phi_j(v)$ per piece"),
    "tau": (6, 0.0, "optical depth", r"$\tau(v)$, $F=e^{-\tau}$"),
    "overflux": (7, 0.0, "overflux", r"$\delta_F$ per line, total"),
    "stats": (8, 0.0, "statistics", "P1D, ξ1D, P3D, ξ3D"),
    "fake_spectra": (3, -0.8, "fake_spectra", "external reference"),
}
# (from, to, kind, label, route): kind "solid" required | "optional" optional input or data table | "external";
# route = waypoints between the two ends (``_dag_route``). Every edge is a dependency of the code:
#   cosmology -> skewers   ``SkewerBundle`` solves a(x) with ``Cosmology.solve_a_of_x`` (or takes manual_scale_factors);
#   lines -> simulation    ``gas_physics.Ion.mass_amu`` (``core.lines.get_mass_amu``);
#   sources -> gas cells   ``grf_cells`` / ``snapshot_cells``, ``SimulationSkewers.to_cells`` return ``core.Cells``;
#   cells, skewers -> deposition   ``core.cell_pieces(cells, lines, skewer_bundle=...)``: points and a(x) (``bundle_frame``);
#   deposition -> line profiles    the profiles are evaluated at the pieces' T, v_par and a (``optical_depth_kernel``);
#   lines -> line profiles         lambda_j, A_21, m (``voigt_widths_GHz``);
#   deposition, profiles, lines -> tau   tau = C f_j sum_p (n l / a^2)_p phi_jp: f_j in the prefactor, lambda_j in the
#                                  detuning (``frequency_offset_GHz``), delta_v_j for the reference axis
#                                  (``line_frame="reference"``, ``wrap_to_ref_line``);
#   tau -> overflux -> statistics  ``DeltaFieldBuilder(tau)``, ``OverfluxStatistics1D/3D`` (no line data: the lines are
#                                  already on one axis).
DAG_EDGES = (
    ("cosmology", "skewers", "optional", "a(x) (or manual)", None),
    ("lines", "simulation", "optional", "ion masses", "v"),
    ("samplers", "fields", "solid", None, None),
    ("simulation", "fields", "solid", None, None),
    ("fields", "deposition", "solid", None, None),
    ("skewers", "deposition", "solid", "points, a(x)", "h"),
    ("deposition", "profiles", "solid", r"T, $v_\parallel$, a", None),
    ("deposition", "tau", "solid", r"$n\,\ell/a^2$ of the pieces", "u"),
    ("lines", "profiles", "solid", "atomic data: λ, A₂₁, m", "h"),
    ("lines", "tau", "solid", "f, λ, line offsets Δv", "h"),
    ("profiles", "tau", "solid", None, None),
    ("tau", "overflux", "solid", None, None),
    ("overflux", "stats", "solid", None, None),
)
DAG_EXTERNAL = (("simulation", "fake_spectra", None), ("fake_spectra", "tau", "compared"))  # same snapshot, compared τ
DAG_NOTEBOOKS = {
    "lines": "01",
    "cosmology": "02",
    "skewers": "03",
    "samplers": "04",
    "fields": "04, 05",
    "simulation": "05",
    "profiles": "06",
    "deposition": "07",
    "tau": "07",
    "overflux": "08",
    "stats": "08",
    "fake_spectra": "09, 10",
}
_DAG_W, _DAG_H, _DAG_PAD = 0.80, 0.56, 0.02  # box width and height, and the pad of its rounded outline [data]
# per edge: the lane of its route (``_dag_route``), and the x offset of its end from the centre of the target box
_DAG_LANES = {("lines", "profiles"): -0.12, ("lines", "tau"): 0.12, ("lines", "simulation"): 0.12, ("deposition", "tau"): 0.34}
_DAG_END_DX = {("skewers", "deposition"): -0.2, ("deposition", "tau"): -0.2}


def _dag_route(a, b, route, nodes, lane, dx=0.0):
    """Polyline from box a to box b, starting and ending on the drawn outlines (the box plus its pad).

    None: straight between facing sides; "h": out of the right side of a at height ``lane`` above its centre,
    horizontally, then vertically into the top or bottom of b (``dx`` from its centre); "v": out of the top / bottom of a, vertically to height
    ``lane`` above b's centre, then horizontally into the left side of b; "u": out of the bottom of a (its centre), down
    by ``lane``, horizontally, then up into the bottom of b (``dx`` from its centre), under the boxes in between.
    """
    (xa, ya), (xb, yb) = nodes[a][:2], nodes[b][:2]
    w, h = _DAG_W / 2 + _DAG_PAD, _DAG_H / 2 + _DAG_PAD
    if route is None:
        return [(xa + w, ya), (xb - w, yb + 0.25 * (ya - yb))]
    if route == "h":
        y0 = ya + lane
        return [(xa + w, y0), (xb + dx, y0), (xb + dx, yb + (h if y0 > yb else -h))]
    if route == "u":
        x0, x1, y0 = xa, xb + dx, min(ya, yb) - h - lane
        return [(x0, ya - h), (x0, y0), (x1, y0), (x1, yb - h)]
    y1, x0 = yb + lane, xa + 0.18
    return [(x0, ya + (h if y1 > ya else -h)), (x0, y1), (xb - w, y1)]


def plot_dag(highlight=(), *, external: bool = False, ax=None, figsize=None, notebooks: bool = True):
    """The dependency graph (DAG) of SkewerJAX's ingredients; ``highlight`` (keys of ``DAG_NODES``: the ingredients a
    notebook uses) are drawn with one light fill and a coloured edge, and the edges into them in the same colour.

    Arrows point from an ingredient to what needs it, and each is a dependency of the code (``DAG_EDGES``). Solid: a
    required input. Dashed: optional, or data only (the cosmology can be replaced by manual a(x); the simulation reads
    only ion masses from the line table). The two sources of gas are interchangeable: both produce the same container,
    Voronoi gas cells, which the deposition cuts into pieces along the skewers. The line profiles are evaluated at the
    pieces' temperature, velocity and scale factor with the lines' atomic data; the optical depth sums them, weighted by
    each piece's n l / a^2, with the oscillator strengths f and wavelengths λ of the lines, and puts every line on the
    reference axis with the offsets Δv. ``external`` adds fake_spectra, the external reference code (comparison notebook
    only). ``notebooks`` writes the notebook of every ingredient under its box. Returns (fig, ax).
    """
    from matplotlib.lines import Line2D

    hl = {highlight} if isinstance(highlight, str) else set(highlight)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or FIGSIZES["dag"], layout="constrained")
    ax.set_axis_off()
    ax.set(xlim=(-0.5, 8.5), ylim=(-1.62, 1.86))
    fg, on, ext_c = plt.rcParams["text.color"], PALETTE["cyan"], PALETTE["yellow"]
    on_fill = mpl.colors.to_rgba(on, 0.30)  # the one highlight: a light fill and a coloured edge
    w, h, pad = _DAG_W, _DAG_H, _DAG_PAD
    nodes = {k: v for k, v in DAG_NODES.items() if external or k != "fake_spectra"}
    # the group of interchangeable field sources
    gx, gy = DAG_NODES["samplers"][0], 0.5 * (DAG_NODES["samplers"][1] + DAG_NODES["simulation"][1])
    gh = DAG_NODES["samplers"][1] - DAG_NODES["simulation"][1] + h + 0.34
    ax.add_patch(
        mpl.patches.FancyBboxPatch(
            (gx - w / 2 - 0.08, gy - gh / 2),
            w + 0.16,
            gh,
            boxstyle="round,pad=0.02,rounding_size=0.1",
            facecolor="none",
            edgecolor=DIM,
            lw=1.2,
            ls=":",
            path_effects=[],
        )
    )
    ax.text(
        gx,
        gy + gh / 2 + 0.04,
        "sources of cells (interchangeable)",
        ha="center",
        va="bottom",
        fontsize=FONT_MIN["floor"],
        color=DIM,
        style="italic",
    )
    edges = [e for e in DAG_EDGES] + ([(a, b, "external", lab, None) for a, b, lab in DAG_EXTERNAL] if external else [])
    ew, eh = w / 2 + pad, h / 2 + pad  # the drawn outline of a box, from its centre
    for a, b, kind, label, route in edges:
        if a not in nodes or b not in nodes:
            continue
        if (a, b) == ("fake_spectra", "tau"):
            y0 = nodes[a][1] - 0.12  # below the deposition -> tau route
            pts = [(nodes[a][0] + ew, y0), (nodes[b][0], y0), (nodes[b][0], nodes[b][1] - eh)]
        elif (a, b) == ("simulation", "fake_spectra"):
            pts = [(nodes[a][0] + ew, nodes[a][1] - 0.15), (nodes[b][0] - ew, nodes[a][1] - 0.15)]
        else:
            pts = _dag_route(a, b, route, nodes, _DAG_LANES.get((a, b), 0.0), _DAG_END_DX.get((a, b), 0.0))
        hot = b in hl  # the inputs of a highlighted ingredient light up
        col = on if hot else (ext_c if kind == "external" else DIM)
        dashed = kind in ("optional", "external")
        lw = 2.4 if hot else 1.6
        xs, ys = zip(*pts)
        # one polyline with a fixed dash pattern (in points: the same in every orientation and segment length)
        ax.plot(
            xs,
            ys,
            color=col,
            lw=lw,
            path_effects=[],
            solid_capstyle="butt",
            dash_capstyle="butt",
            **(dict(ls=(0, (4.0, 3.0))) if dashed else dict(ls="-")),
        )
        (xe0, ye0), (xe1, ye1) = pts[-2], pts[-1]  # the arrow head only, on a short stub of the last segment
        d = np.hypot(xe1 - xe0, ye1 - ye0)
        stub = (xe1 - 0.06 * (xe1 - xe0) / d, ye1 - 0.06 * (ye1 - ye0) / d)
        ax.annotate(
            "",
            (xe1, ye1),
            stub,
            arrowprops=dict(arrowstyle="-|>", color=col, mutation_scale=18, lw=lw, ls="-", shrinkA=0, shrinkB=0),
            zorder=4,
        )
        if label:  # on the longest segment (above the boxes if it is short)
            seg = max(range(len(pts) - 1), key=lambda k: np.hypot(pts[k + 1][0] - pts[k][0], pts[k + 1][1] - pts[k][1]))
            (x0, y0), (x1, y1) = pts[seg], pts[seg + 1]
            vert = abs(x1 - x0) < 1e-9
            f = 0.3 if (a, b) == ("skewers", "deposition") else 0.5
            xl, yl = x0 + f * (x1 - x0) + (0.03 if vert else 0.0), 0.5 * (y0 + y1) + (0.0 if vert else 0.035)
            if route is None and not vert and abs(x1 - x0) < 0.5:
                yl = max(nodes[a][1], nodes[b][1]) + h / 2 + 0.04
            ax.text(
                xl,
                yl,
                label,
                ha="left" if vert else "center",
                va="center" if vert else "bottom",
                fontsize=FONT_MIN["floor"],
                color=col,
                style="italic",
                rotation=0,
                bbox=dict(boxstyle="round,pad=0.12", facecolor=plt.rcParams["axes.facecolor"], edgecolor="none", alpha=0.85),
            )
    for key, (x, y, name, sub) in nodes.items():
        hit = key in hl
        ext = key == "fake_spectra"
        ec = on if hit else (ext_c if ext else DIM)
        ax.add_patch(
            mpl.patches.FancyBboxPatch(
                (x - w / 2, y - h / 2),
                w,
                h,
                boxstyle=f"round,pad={pad},rounding_size=0.12",
                facecolor=on_fill if hit else plt.rcParams["axes.facecolor"],
                edgecolor=ec,
                lw=2.6 if hit else 1.4,
                ls=(0, (4.0, 3.0)) if ext else "-",
                path_effects=[],
                zorder=3,
            )
        )
        tc = fg if hit else (ext_c if ext else fg)
        ax.text(x, y + 0.1, name, ha="center", va="center", fontsize=FONT_MIN["legend"], color=tc, fontweight="bold", zorder=4)
        ax.text(x, y - 0.13, sub, ha="center", va="center", fontsize=FONT_MIN["floor"], color=tc, zorder=4)
        if notebooks and key in DAG_NOTEBOOKS:
            ax.text(
                x + w / 2,
                y - h / 2 - 0.03,
                f"nb {DAG_NOTEBOOKS[key]}",
                ha="right",
                va="top",
                fontsize=FONT_MIN["floor"],
                color=on if hit else DIM,
                zorder=4,
            )
    handles = [
        Line2D([], [], color=DIM, lw=1.8, ls="-", label="required input"),
        Line2D([], [], color=DIM, lw=1.8, ls=(0, (4.0, 3.0)), label="optional / data only"),
        Line2D(
            [],
            [],
            lw=0,
            marker="s",
            ms=13,
            markerfacecolor=on_fill,
            markeredgecolor=on,
            markeredgewidth=2.0,
            label="used in this notebook",
        ),
    ]
    if external:
        handles.append(Line2D([], [], color=ext_c, lw=1.8, ls=(0, (4.0, 3.0)), label="external reference"))
    for hd in handles:
        hd.set_path_effects([])
    ax.legend(
        handles=handles,
        loc="lower right",
        ncol=len(handles),
        frameon=False,
        fontsize=FONT_MIN["floor"],
        bbox_to_anchor=(1.0, -0.04),
        handlelength=2.4,
    )
    return ax.figure, ax


# --------------------------------------------------------------------------- the dark ("neon") style of the visualisations
NEON_COLORS = ENTITY_COLORS  # the visualisations use the same colour-blind-safe colours as every other figure
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
    """A legend with the style's background and foreground colours (``ax.legend`` keywords override)."""
    leg = ax.legend(**{"frameon": True, "facecolor": BG, "edgecolor": FG, "fontsize": FONT_MIN["legend"], **kw})
    for t in leg.get_texts():
        t.set_color(FG)
    return leg


def plot_slab(
    ax,
    slab: dict,
    *,
    quantity: str = "sigma",
    cmap=None,
    vmin=None,
    vmax=None,
    colorbar: bool = True,
    label=None,
    smooth: float = 0.0,
):
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
    im = ax.imshow(
        lm.T,
        origin="lower",
        extent=slab["extent"],
        cmap=cmap,
        vmin=vmin if vmin is not None else np.nanpercentile(lm, 1),
        vmax=vmax if vmax is not None else np.nanpercentile(lm, 99.8),
        interpolation="nearest",
    )
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


# --------------------------------------------------------------------------- the quickstarts (notebooks 00 and 00b)
def plot_toy_box(cells, skewers, lines, v, F, stats, *, skewer: int = 0, axis: int = 1, ref_label: str = "Lya", figsize=(22, 6)):
    """The quickstart's summary (notebook 00): (a) the comoving density of every species along one skewer (the deposited
    column per unit length of each dual segment, ``core.deposit`` + ``core.segment_fields``), against its coordinate
    ``axis``; (b) the flux of every line (``F``: (N_lines, N_sk, N_v)) and their product; (c) k P1D / pi of the total
    overflux and of the reference line alone (``stats``: ``OverfluxStatistics1D`` after ``compute_p1d``, labels
    ``ref_label``). Returns (fig, axs)."""
    import jax.numpy as jnp

    from skewerjax.core import deposit, line_elements, segment_fields

    k = np.asarray(stats.k)[1:]
    fig, axs = plt.subplots(1, 3, figsize=figsize, layout="constrained")
    pts0 = np.asarray(skewers.spatial_points[skewer : skewer + 1], dtype=float)
    used, seg = deposit(cells, pts0)
    w0 = np.asarray(line_elements(jnp.asarray(pts0)))
    for j, sp in enumerate(lines.species_keys):
        n = (
            segment_fields(seg, used.species[sp]["number_density"], used.species[sp]["temperature"], 0.0)["column_length"][0]
            / w0[0]
        )
        axs[0].semilogy(
            pts0[0, :, axis],
            n,
            color=ENTITY_COLORS[sp],
            **(MODE_STYLES["default"] if j == 0 else MODE_STYLES["matched"]),
            label=sp,
        )
    axs[0].set(xlabel=f"{'xyz'[axis]} [cMpc]", ylabel=r"$n\ [\mathrm{cm^{-3}}]$", title=f"gas along skewer {skewer}")
    for j, l in enumerate(lines.lines):
        axs[1].plot(v, F[j, skewer], color=ENTITY_COLORS[l.id], label=l.id)
    axs[1].plot(v, F[:, skewer].prod(0), color=FG, lw=1.2, label="total")
    axs[1].set(xlabel=r"$v$ [km/s] (Ly$\alpha$ frame)", ylabel="F", title="transmitted flux")
    axs[2].loglog(k, k * np.asarray(stats.Pk_total)[1:] / np.pi, color=FG, label="total")
    axs[2].loglog(
        k,
        k * np.asarray(stats.Pk_pair[(ref_label, ref_label)])[1:] / np.pi,
        color=ENTITY_COLORS[lines.lines[0].species],
        **MODE_STYLES["matched"],
        label=r"Ly$\alpha$ alone",
    )
    axs[2].set(xlabel=r"$k$ [s/km]", ylabel=r"$k P_{\rm 1D}/\pi$", title=f"P1D over {np.shape(F)[1]} skewers")
    for ax in axs:
        ax.legend()
    add_panel_letters(axs)
    return fig, axs


def plot_slab_and_skewer(
    slab, s, fields, *, box: float, y_sk: float, z_c: float, thick: float, x_mark: float, zoom: float = 3.0, figsize=(22, 12)
):
    """The gas around a skewer along x (left) and along it (right) (notebook 00b, (1, 3)).

    slab : ``simulations.slab_projection`` of the slab |z - z_c| < thick / 2; s : (N,) path lengths [cMpc]; fields :
    (H I, Si III) point values of ``SimulationSkewers.point_values`` (skewer 0); box [cMpc]; the skewer at y = ``y_sk``;
    ``x_mark``: a position on it (the inset, ± ``zoom`` cMpc, and the dotted lines). Right: the density of both species
    (Si III clipped at a hundredth of its 10th percentile: the empty-cell dips), the temperature and the peculiar
    velocity. Returns (fig, (axm, axp)).
    """
    pv_H, pv_S = fields
    fig = plt.figure(figsize=figsize, layout="constrained")
    gs = fig.add_gridspec(3, 2, width_ratios=[1.1, 1])
    axm = fig.add_subplot(gs[:, 0])
    axp = [fig.add_subplot(gs[i, 1]) for i in range(3)]
    dark_axes([axm] + axp, fig)
    im = plot_slab(axm, slab, smooth=1.0, label=r"log$_{10}\ \Sigma_{\rm gas}$ [M$_\odot$ ckpc$^{-2}$]")
    neon_line(axm, [0, box], [y_sk, y_sk], NEON_COLORS["HI"], lw=1.8)
    ins = axm.inset_axes((0.62, 0.62, 0.36, 0.36))
    plot_slab(ins, slab, smooth=1.0, colorbar=False, vmin=im.norm.vmin, vmax=im.norm.vmax)
    neon_line(ins, [0, box], [y_sk, y_sk], NEON_COLORS["HI"], lw=1.4)
    ins.set(xlim=(x_mark - zoom, x_mark + zoom), ylim=(y_sk - zoom, y_sk + zoom), xlabel="", ylabel="")
    ins.tick_params(colors=FG, labelsize=16)
    axm.indicate_inset_zoom(ins, edgecolor=FG)
    axm.set_title(f"gas in |z − {z_c:.1f}| < {thick / 2} cMpc, and the skewer", color=FG)
    neon_line(axp[0], s, pv_H["n"][0], NEON_COLORS["HI"], label="H I")
    neon_line(axp[0], s, pv_S["n"][0], NEON_COLORS["SiIII"], label="Si III")
    neon_line(axp[1], s, pv_H["T"][0], NEON_COLORS["temperature"])
    neon_line(axp[2], s, pv_H["v_los"][0], NEON_COLORS["velocity"])
    n_pos = pv_S["n"][0][pv_S["n"][0] > 0]
    axp[0].set(yscale="log", ylabel=LABELS["n"], ylim=(np.percentile(n_pos, 10) / 100, None))
    axp[1].set(yscale="log", ylabel=LABELS["T"])
    axp[2].set(ylabel=LABELS["v_los"], xlabel=LABELS["s"])
    for ax in axp:
        ax.axvline(x_mark, color=FG, ls=":", lw=1.2)
        ax.set_xlim(0, box)
    neon_legend(axp[0], loc="upper left", ncol=2)
    dark_axes([axm] + axp, fig)
    return fig, (axm, axp)


def plot_cell_crossings(lo, hi, n_c, size, spacing: float, *, figsize=(20, 8)):
    """The cells crossed by a skewer (notebook 00b, §2): (a) the crossed length l_c of every crossing [lo, hi) [cMpc],
    coloured by the cell's density ``n_c`` [cm^-3] (log, ``CMAPS["diverging"]``: both ends distinct); (b) the cell size
    ``size`` (V^(1/3)) and l_c, against the ``spacing`` of the sampling points. Returns (fig, axs)."""
    lo, hi = np.asarray(lo), np.asarray(hi)
    fig, axs = plt.subplots(2, 1, figsize=figsize, sharex=True, gridspec_kw={"height_ratios": [1.3, 1]}, layout="constrained")
    norm = mpl.colors.LogNorm(np.percentile(n_c, 2), n_c.max())
    cmap = plt.get_cmap(CMAPS["diverging"])
    axs[0].bar(0.5 * (lo + hi), hi - lo, width=hi - lo, color=cmap(norm(n_c)), edgecolor=BG, lw=0.3)
    axs[0].set(
        ylabel=r"crossed length $\ell_c$ [cMpc]",
        yscale="log",
        title=f"{lo.size} cell crossings along the skewer, coloured by n_HI",
    )
    axs[1].semilogy(0.5 * (lo + hi), size, ".", color=PALETTE["cyan"], ms=4, label=r"cell size $V^{1/3}$")
    axs[1].semilogy(0.5 * (lo + hi), hi - lo, ".", color=PALETTE["yellow"], ms=3, alpha=0.6, label=r"crossed length $\ell_c$")
    axs[1].axhline(spacing, color=FG, ls=":", lw=1.4, label="spacing of the sampling points (0.25 km/s)")
    axs[1].set(xlabel=LABELS["s"], ylabel="[cMpc]")
    axs[1].legend(fontsize=15, ncol=3)
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axs[0], label=r"$n_{\rm HI}$ [cm$^{-3}$]")
    add_panel_letters(axs)
    return fig, axs


def plot_voigt_of_piece(T_K: float, line_list, dv=np.linspace(-300, 300, 6001), *, figsize=(16, 7)):
    """The Voigt profiles phi / phi_max of gas at temperature ``T_K`` [K] at rest, in every line of ``line_list``
    (``core.Line``) against v_dagger - v_c [km/s] (notebook 00b, §4): Gaussian cores of width b, Lorentzian wings.
    Returns (fig, ax)."""
    import jax.numpy as jnp

    from skewerjax.core import voigt as V
    from skewerjax.core.constants import phys
    from skewerjax.core.line_profiles import voigt_widths_GHz

    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    for ln in line_list:
        sg, gm = voigt_widths_GHz(
            jnp.array([T_K]),
            jnp.zeros(1),
            jnp.zeros(1),
            jnp.array([ln.lambda0_A]),
            jnp.array([ln.A21_GHz]),
            jnp.array([ln.mass_amu]),
        )
        nu = phys.c * 10.0 / ln.lambda0_A * np.expm1(-dv / phys.c_kms)
        phi = np.asarray(V.voigt_exojax(jnp.asarray(nu), float(np.ravel(sg)[0]), float(np.ravel(gm)[0])))
        b = np.sqrt(2 * phys.kB * T_K / ln.mass_amu) / 1e3
        ax.semilogy(
            dv,
            phi / phi.max(),
            color=NEON_COLORS[ln.id],
            label=f"{ln.id}: b = {b:.1f} km/s, γ λ = {float(np.ravel(gm)[0]) * ln.lambda0_A * 1e-4:.1e} km/s",
        )
    ax.set(
        xlabel=r"$v_\dagger - v_c$ [km/s]",
        ylabel=r"$\phi / \phi_{\max}$",
        ylim=(1e-7, 2),
        title=f"the Voigt profiles of the strongest piece (T = {T_K:.2e} K): Gaussian cores, Lorentzian wings",
    )
    ax.legend(fontsize=15)
    return fig, ax


def plot_tau_stretches(edges, s, n, v, tau_bins, tau, frame, absorbers, *, box: float, figsize=(20, 16)):
    """Stretches of gas along one skewer and the absorption each one makes (notebook 00b, (5)).

    edges : (M + 1,) path intervals [cMpc]; s, n : the H I density at the sampling points; v : (N_v,) velocities;
    tau_bins : (Lya, Si III) the tau of every interval (M, N_v) (``absorbers.tau_by_path_bins``); tau : (Lya, Si III)
    the whole tau (N_v,); frame : the ``ExactFrame``; absorbers : the H I ``Absorbers`` (its strongest piece is marked:
    position dashed in the top panel; its Hubble velocity dotted and line centre dashed below). The top axis maps s
    to the Hubble velocity v_H(s) = c ln(a_dagger / a(s)). Returns (fig, axs, info {vH_s, s_max, u_max, vH_max, vc_max}).
    """
    from skewerjax.core.constants import phys
    from skewerjax.core.spectral import line_centre_velocity_kms

    (tb_H, tb_S), (t_H, t_S) = tau_bins, tau
    fig, axs = plt.subplots(4, 1, figsize=figsize, layout="constrained", gridspec_kw={"height_ratios": [1.2, 1, 1, 1]})
    dark_axes(axs, fig)
    cols = plot_tau_buildup(axs[0], axs[1], edges, s, n, v, tb_H, line_label=r"Ly$\alpha$")
    plot_tau_buildup(None, axs[2], edges, s, None, v, tb_S, colors=cols, line_label="Si III")
    neon_line(axs[3], v, np.exp(-t_H), NEON_COLORS["HI-LyA"], label=r"Ly$\alpha$")
    neon_line(axs[3], v, np.exp(-t_S), NEON_COLORS["SiIII-1206"], label="Si III 1206")
    axs[1].set_yscale("symlog", linthresh=1e-2)
    axs[2].set_yscale("symlog", linthresh=1e-4)
    axs[3].set(xlabel=LABELS["v_dagger"] + ", each line in its own frame", ylabel=LABELS["F"], ylim=(-0.05, 1.08))
    for ax in axs[1:]:
        ax.set_xlim(v[0], v[-1])
    axs[0].set_xlim(0, box)
    axs[0].set_ylabel(r"$n_{\rm HI}\ [\mathrm{cm^{-3}}]$")
    # the shared axis: the Hubble velocity of every path position, v_H(s) = c ln(a_dagger / a(s)), on top of the density panel
    vH_s = phys.c_kms * np.log(frame.a_dagger / frame.scale_factors(0, s))
    top = axs[0].secondary_xaxis("top", functions=(lambda x: np.interp(x, s, vH_s), lambda u: np.interp(u, vH_s, s)))
    top.set_xlabel(r"$v_H(s) = c\,\ln(a_\dagger/a(s))$ [km s$^{-1}$]: where gas at rest would absorb")
    axs[0].annotate(
        "", xy=(0.005, 0.86), xytext=(0.11, 0.86), xycoords="axes fraction", arrowprops=dict(arrowstyle="-|>", color=FG, lw=2)
    )
    axs[0].text(
        0.115,
        0.86,
        "towards the observer (at s < 0): the light travels to smaller s",
        transform=axs[0].transAxes,
        va="center",
        color=FG,
        fontsize=15,
    )
    # the strongest H I piece: its position, its Hubble velocity and its line centre (Hubble flow + peculiar velocity)
    A = absorbers
    p_max = int(np.argmax(A.n[0] * A.length[0]))
    s_max, u_max = float(A.s[0, p_max]), float(A.v_los[0, p_max])
    vH_max = float(np.interp(s_max, s, vH_s))
    vc_max = float(
        line_centre_velocity_kms(frame.a_dagger, frame.scale_factors(0, np.array([s_max])), -np.array([[u_max]]))[0, 0]
    )
    axs[0].axvline(s_max, color=FG, ls="--", lw=1.4)
    for ax in axs[1:]:
        ax.axvline(vH_max, color=FG, ls=":", lw=1.4)
        ax.axvline(vc_max, color=FG, ls="--", lw=1.4)
    neon_legend(axs[1], loc="upper right")
    neon_legend(axs[3], loc="lower left", ncol=2)
    dark_axes(axs, fig)
    return fig, axs, dict(vH_s=vH_s, s_max=s_max, u_max=u_max, vH_max=vH_max, vc_max=vc_max)


def plot_wrap_tiles(
    v, tau_own, tau_ref, *, period: float, delta_v: float, v_own: float, v_obs: float, zoom: float = 300.0, figsize=(22, 7)
):
    """Wrapping onto the reference frame (notebook 00b, (5)): (a) the tau of Lya and Si III in their own frames
    (``tau_own``: two (N_v,)), normalised, around ``v_own`` (± ``zoom`` km/s); (b) their flux on the Lya frame
    (``tau_ref``: (2, 1, N_v), ``wrap_to_ref_line``) tiled over three periods ``period`` [km/s], with the shift
    -``delta_v`` of Si III from ``v_own`` to ``v_obs`` (its peak on the Lya frame, modulo P). Returns (fig, axs)."""
    P, dV = float(period), float(delta_v)
    fig, axs = plt.subplots(1, 2, figsize=figsize, layout="constrained", gridspec_kw={"width_ratios": [1, 1.6]})
    dark_axes(axs, fig)
    for lid, t in zip(["HI-LyA", "SiIII-1206"], tau_own):
        neon_line(axs[0], v, t / t.max(), NEON_COLORS[lid], label=lid)
    axs[0].set(
        xlabel=LABELS["v_dagger"] + " (own frame)",
        ylabel=r"$\tau / \tau_{\max}$",
        title=r"same gas, same $v_\dagger$ (zoom)",
        xlim=(v_own - zoom, v_own + zoom),
    )
    for n in np.arange(-1, 2):
        neon_line(
            axs[1], v + n * P, np.exp(-tau_ref[0, 0]), NEON_COLORS["HI-LyA"], label=r"Ly$\alpha$" if n == 0 else None, lw=1.2
        )
        neon_line(
            axs[1],
            v + n * P,
            np.exp(-tau_ref[1, 0]),
            NEON_COLORS["SiIII-1206"],
            label="Si III on the Ly$\\alpha$ frame" if n == 0 else None,
            lw=1.6,
        )
        axs[1].axvspan(v[0] + n * P, v[-1] + n * P, color=FG, alpha=0.04 if n == 0 else 0.0)
        axs[1].axvline(v[0] + n * P, color=FG, ls=":", lw=1)
    axs[1].annotate("", (v_own - dV, 0.12), (v_own, 0.12), arrowprops=dict(arrowstyle="-|>", color=FG, lw=2))
    axs[1].text(
        v_own - dV / 2, 0.16, f"−Δv = −{dV:.0f} km/s", color=FG, ha="center", fontsize=15, bbox=dict(fc=BG, ec="none", alpha=0.85)
    )
    axs[1].plot([v_obs], [0.05], "v", color=NEON_COLORS["SiIII-1206"], ms=12)
    axs[1].set(
        xlim=(v[0] - P, v[-1] + P),
        ylim=(-0.05, 1.12),
        xlabel=LABELS["v_ref"],
        ylabel=LABELS["F"],
        title=f"three periods (P = {P:.0f} km/s); the middle one is the box",
    )
    neon_legend(axs[0], loc="upper right")
    neon_legend(axs[1], loc="lower right", ncol=2)
    dark_axes(axs, fig)
    return fig, axs


def plot_overflux_p1d_xi(v, overflux, stats, k, P_tot, P_ll, fit, *, skewer: int, period: float, delta_v: float, figsize=(24, 7)):
    """Overflux, P1D with the Si III model and the Lya x Si III correlation (notebook 00b, (6, 7)): (a) the overflux of
    each line and the total along ``skewer`` (``DeltaFieldBuilder``); (b) k P / pi of Lya alone, the total and the
    McDonald model with ``fit["a"]`` (k <= 0.1 s/km); (c) the cross correlation for lags in [0, P/2], dotted at
    |delta_v| modulo the ``period``. Returns (fig, axs)."""
    from skewerjax.core.stats import mcdonald_factor

    P, dV = float(period), float(delta_v)
    fig, axs = plt.subplots(1, 3, figsize=figsize, layout="constrained")
    dark_axes(axs, fig)
    for lab, key, c in (("Lya", "Lya", NEON_COLORS["HI-LyA"]), ("Si III", "SiIII", NEON_COLORS["SiIII-1206"])):
        neon_line(axs[0], v, overflux.delta_terms[key][skewer], c, label=rf"$\delta_{{\rm {lab}}}$", lw=1.2)
    neon_line(axs[0], v, overflux.total[skewer], NEON_COLORS["total"], label=r"$\delta_{\rm tot}$", lw=1.0)
    axs[0].set(xlabel=LABELS["v_ref"], ylabel=LABELS["delta"], title=f"overflux of skewer {skewer}")
    neon_line(axs[1], k, k * P_ll / np.pi, NEON_COLORS["HI-LyA"], label=r"Ly$\alpha$ alone")
    neon_line(axs[1], k, k * P_tot / np.pi, NEON_COLORS["total"], label="total", lw=1.2)
    neon_line(
        axs[1],
        k,
        k * P_ll * mcdonald_factor(k, fit["a"], dV) / np.pi,
        NEON_COLORS["SiIII-1206"],
        label=f"McDonald, a = {fit['a']:.3f}",
        lw=1.0,
    )
    kp = k * P_tot / np.pi
    inr = k <= 0.1
    axs[1].set(
        xscale="log",
        yscale="log",
        xlim=(k[0], 0.1),
        ylim=(kp[inr].min() / 3, kp[inr].max() * 3),
        xlabel=LABELS["k"],
        ylabel=LABELS["kP"],
        title=f"P1D, {np.shape(overflux.total)[0]} skewers",
    )
    lags = np.asarray(stats.lags)
    xc = np.asarray(stats.xi_pair[("Lya", "SiIII")]).real
    sel = (lags >= 0) & (lags <= P / 2)
    neon_line(axs[2], lags[sel], xc[sel], NEON_COLORS["SiIII-1206"], label=r"$\xi_{\rm Ly\alpha \times SiIII}$")
    axs[2].axvline(abs(np.mod(dV + P / 2, P) - P / 2), color=FG, ls=":", lw=1.5)
    axs[2].set(xlabel=LABELS["lag"], ylabel=LABELS["xi"], title="cross term (dotted: |Δv| mod P)")
    for ax in axs:
        neon_legend(ax, loc="best")
    dark_axes(axs, fig)
    return fig, axs


# --------------------------------------------------------------------------- Gaussian random fields (notebook 04)
def plot_grf_draws(draws, titles, *, width_ratios=(1.3, 0.75, 1.0), figsize=(24, 7.5)):
    """One draw of a ``PowerLawGRF`` per panel (notebook 04): a 1D field as a line, a 2D field as a map, a 3D field as
    its slice through the middle of the last axis (diverging maps centred on delta = 0; "line of sight" on the y label
    when it is the map's y axis). draws : [(grf, field)]; titles : one per panel. Returns (fig, axs)."""
    fig, axs = plt.subplots(
        1, len(draws), figsize=figsize, layout="constrained", gridspec_kw={"width_ratios": list(width_ratios)}
    )
    for ax, (g, d), title in zip(np.ravel(axs), draws, titles):
        if np.ndim(d) == 1:
            ax.plot(g.x_grid()[0], d, color=ENTITY_COLORS["density"], lw=1)
            ax.set(xlabel="x [Mpc]", ylabel=r"$\delta$", title=title)
            continue
        m = np.asarray(d if np.ndim(d) == 2 else d[:, :, np.shape(d)[2] // 2])
        vmax = float(np.abs(m).max())
        ax.imshow(
            m.T, origin="lower", extent=(0, g.L[0], 0, g.L[1]), cmap=CMAPS["diverging"], aspect="equal", vmin=-vmax, vmax=vmax
        )
        ax.set(xlabel="x [Mpc]", ylabel="y (line of sight) [Mpc]" if g.los_axis == 1 else "y [Mpc]", title=title)
    add_panel_letters(axs)
    return fig, axs


# --------------------------------------------------------------------------- benchmark figures (rows of comparison.benchmark)
# Colour = code and device, line style = SkewerJAX mode (MODE_STYLES: default solid, matched dashed, float32 dotted,
# crossings dash-dot), marker = device. Rows come from ``comparison.benchmark.result_rows``.
BENCH_DEVICES = {
    ("fake_spectra", "cpu"): dict(color=PALETTE["grey"], marker="s", name="fake_spectra"),
    ("skewerjax", "cpu"): dict(color=PALETTE["blue"], marker="o", name="SkewerJAX CPU"),
    ("skewerjax", "gpu"): dict(color=PALETTE["green"], marker="^", name="SkewerJAX GPU"),
}
BENCH_MODES = {None: "default", "default": "default", "matched": "matched", "float32": "alternative", "crossings": "isolated"}
BENCH_STAGE_COLORS = {
    "setup": PALETTE["grey"],
    "io": PALETTE["yellow"],
    "selection": PALETTE["green"],
    "physics": PALETTE["purple"],
    "pieces": PALETTE["cyan"],
    "tau": PALETTE["red"],
}
BENCH_STAGE_LABELS = {
    "setup": "setup",
    "io": "I/O",
    "selection": "cell selection",
    "physics": "gas physics",
    "pieces": "pieces",
    "tau": r"$\tau$ sum",
}
BENCH_SIM_MARKERS = {"TNG50-4": "o", "TNG50-3": "D", "TNG50-2": "*"}
BENCH_SIM_COLORS = {"TNG50-4": PALETTE["cyan"], "TNG50-3": PALETTE["purple"], "TNG50-2": PALETTE["red"]}


def bench_series(row: dict) -> tuple:
    """(code, platform, mode) of a benchmark row: one line in the figures."""
    return row["code"], row["platform"], row["mode"]


def bench_style(key: tuple, *, fill: bool = True) -> dict:
    """Line style of one series (code, platform, mode): colour and marker of the device, line style of the mode."""
    code, plat, mode = key
    d = BENCH_DEVICES[(code, plat)]
    ms = MODE_STYLES[BENCH_MODES[mode]]
    label = d["name"] if code == "fake_spectra" else f"{d['name']}, {mode}"
    return dict(
        color=d["color"],
        marker=d["marker"],
        ls=ms["ls"],
        lw=2.0,
        ms=9,
        label=label,
        mfc=d["color"] if fill and mode in (None, "default") else BG,
    )


def bench_value(row: dict, what: str = "total") -> tuple:
    """(median, min, max) over the timed runs: ``total``, a stage, ``compute`` (physics + pieces + tau),
    ``io+selection`` (setup + io + selection) or ``throughput`` (skewers per second); ``rss`` / ``gpu`` [GB] (one
    value)."""
    if what == "total":
        return tuple(row["total"])
    if what in row["stages"]:
        return tuple(row["stages"][what])
    if what in ("compute", "io+selection"):
        parts = ("physics", "pieces", "tau") if what == "compute" else ("setup", "io", "selection")
        return tuple(sum(row["stages"][s][i] for s in parts) for i in range(3))
    if what == "throughput":
        m, lo, hi = row["total"]
        return row["n_skewers"] / m, row["n_skewers"] / hi, row["n_skewers"] / lo
    if what in ("rss", "gpu"):
        v = row["peak_rss_GB"] if what == "rss" else row["gpu_peak_GB"]
        return (np.nan,) * 3 if v is None else (v, v, v)
    raise ValueError(f"unknown quantity {what!r}")


BENCH_YLABELS = {
    "total": "wall time per run [s]",
    "compute": "physics + pieces + τ [s]",
    "io+selection": "I/O + selection [s]",
    "throughput": "throughput [skewers / s]",
    "rss": "peak host memory (RSS) [GB]",
    "gpu": "peak GPU memory [GB]",
    "tau": "τ sum [s]",
}


def bench_slope(x, y, x_min: float = 0.0) -> float | None:
    """Log-log slope of y(x) fitted on the points with x >= ``x_min`` (None with fewer than two)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    k = (x >= x_min) & np.isfinite(y) & (y > 0)
    return float(np.polyfit(np.log(x[k]), np.log(y[k]), 1)[0]) if k.sum() >= 2 else None


def _bench_groups(rows, **match) -> dict:
    groups = {}
    for r in rows:
        if r["status"] == "ok" and all(r.get(k) == v for k, v in match.items()):
            groups.setdefault(bench_series(r), []).append(r)
    order = lambda k: (k[0] != "fake_spectra", k[1], list(BENCH_MODES).index(k[2]))
    return {k: groups[k] for k in sorted(groups, key=order)}


def plot_bench_vs(ax, rows, *, x: str = "n_skewers", what: str = "total", fit_from: float | None = None, modes=None, **match):
    """One line per series (code, device, mode): the median (min-max bars) of ``what`` (``bench_value``) against
    ``x`` (``n_skewers``, ``n_gas`` or ``threads``), log-log. ``fit_from``: fit the log-log slope on x >= fit_from and
    add it to the label. ``modes``: the SkewerJAX modes to show (default all). Returns {series: slope}."""
    slopes = {}
    for key, rs in _bench_groups(rows, **match).items():
        if modes is not None and key[0] == "skewerjax" and key[2] not in modes:
            continue
        rs = sorted(rs, key=lambda r: r[x])
        xs = np.array([r[x] for r in rs], float)
        st = np.array([bench_value(r, what) for r in rs])
        if not np.isfinite(st[:, 0]).any():
            continue
        sty = bench_style(key)
        if fit_from is not None:
            slopes[key] = bench_slope(xs, st[:, 0], fit_from)
            if slopes[key] is not None:
                sty["label"] += f"  (slope {slopes[key]:.2f})"
        ax.errorbar(xs, st[:, 0], yerr=[st[:, 0] - st[:, 1], st[:, 2] - st[:, 0]], capsize=3, **sty)
    ax.set(
        xscale="log",
        yscale="log",
        ylabel=BENCH_YLABELS.get(what, f"{what} [s]"),
        xlabel={"n_skewers": "number of skewers N", "n_gas": "gas cells in the snapshot", "threads": "CPU threads"}.get(x, x),
    )
    return slopes


def plot_bench_speedup(ax, rows, *, x: str = "n_skewers", what: str = "total", modes=None, **match):
    """t(fake_spectra) / t(SkewerJAX series) at the same point (> 1: SkewerJAX faster), against ``x``."""
    groups = _bench_groups(rows, **match)
    fs = {r[x]: bench_value(r, what)[0] for k, rs in groups.items() if k[0] == "fake_spectra" for r in rs}
    for key, rs in groups.items():
        if key[0] == "fake_spectra" or (modes is not None and key[2] not in modes):
            continue
        rs = sorted((r for r in rs if r[x] in fs), key=lambda r: r[x])
        if rs:
            ax.plot([r[x] for r in rs], [fs[r[x]] / bench_value(r, what)[0] for r in rs], **bench_style(key))
    ax.axhline(1.0, color=FG, lw=1.2, ls=":")
    ax.set(
        xscale="log",
        yscale="log",
        ylabel="speed-up over fake_spectra (< 1: slower)",
        xlabel={"n_skewers": "number of skewers N", "n_gas": "gas cells in the snapshot"}.get(x, x),
    )
    return ax


def plot_bench_stages(ax, rows, *, fraction: bool = False, modes=None, labels: bool = True, **match):
    """Horizontal bars stacked by stage (median over the timed runs), one per series at one point (``match``: e.g.
    sim, n_skewers, threads). The total, and SkewerJAX's JIT compilation (first run only, not in the bars), are written
    after each bar. ``fraction``: the share of each stage [%] instead of seconds. Linear axis (a stacked bar on a log
    axis misstates the shares)."""
    from skewerjax.comparison.benchmark import STAGES

    groups = [
        (k, rs[0]) for k, rs in _bench_groups(rows, **match).items() if k[0] == "fake_spectra" or modes is None or k[2] in modes
    ]
    seen = set(ax.get_legend_handles_labels()[1])
    for i, (_key, r) in enumerate(groups):
        tot = sum(bench_value(r, s)[0] for s in STAGES)
        left = 0.0
        for s in STAGES:
            w = bench_value(r, s)[0] * (100.0 / tot if fraction else 1.0)
            lab = BENCH_STAGE_LABELS[s]
            ax.barh(i, w, left=left, color=BENCH_STAGE_COLORS[s], edgecolor=BG, lw=0.5, label=None if lab in seen else lab)
            seen.add(lab)
            left += w
        if not fraction:
            txt = f" {tot:.3g} s" + (
                f" (+{r['compile_s']:.2g} s compile)" if (r.get("compile_s") or 0) > 0 else ""
            )  # < 0: within the noise
            ax.text(left, i, txt, va="center", ha="left", fontsize=FONT_MIN["legend"])
    ax.set_yticks(range(len(groups)), [bench_style(k)["label"] for k, _ in groups] if labels else [], fontsize=FONT_MIN["legend"])
    ax.invert_yaxis()
    ax.set_xlabel("share of the run [%]" if fraction else "median wall time per run [s]")
    return ax


def plot_bench_accuracy(ax, rows, *, line: str = "HI-LyA", sims=("TNG50-4", "TNG50-3", "TNG50-2")):
    """Agreement with fake_spectra against N, one colour and marker per simulation: the matched mode's max |dF| over
    the stored skewers (filled, solid) and its rms over their pixels (filled, dashed), and the default mode's |d<F>|
    (open, dotted). From the SkewerJAX CPU rows (the GPU ones agree to float64)."""
    for sim in sims:
        c = BENCH_SIM_COLORS.get(sim, FG)
        rs = sorted(
            (
                r
                for r in rows
                if r["status"] == "ok"
                and r["sim"] == sim
                and r["code"] == "skewerjax"
                and r["platform"] == "cpu"
                and r["mode"] == "matched"
                and r["threads"] == 32
                and r.get("agreement")
            ),
            key=lambda r: r["n_skewers"],
        )
        if not rs:
            continue
        n = [r["n_skewers"] for r in rs]
        a = [r["agreement"][line] for r in rs]
        mk = BENCH_SIM_MARKERS.get(sim, "o")
        ax.plot(n, [x["max_dF"] for x in a], marker=mk, ms=10, color=c, ls="-", lw=1.5, label=f"{sim}: matched, max |ΔF|")
        ax.plot(n, [x["rms_dF"] for x in a], marker=mk, ms=7, color=c, ls="--", lw=1.5, label=f"{sim}: matched, rms ΔF")
        ax.plot(
            n,
            [abs(x["meanF_default"] - x["meanF_fake_spectra"]) for x in a],
            marker=mk,
            ms=10,
            color=c,
            mfc=BG,
            ls=":",
            lw=1.5,
            label=f"{sim}: default, |Δ⟨F⟩|",
        )
    ax.set(xscale="log", yscale="log", xlabel="number of skewers N", ylabel="difference to fake_spectra")
    return ax


# --------------------------------------------------------------------------- figures of the tutorial notebooks (01-10)
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
    lam, f, A21, m = (
        np.asarray(x, dtype=float) for x in (line_bundle.lambda0_A, line_bundle.f_osc, line_bundle.A21_GHz, line_bundle.mass_amu)
    )
    dv = np.asarray(line_bundle.delta_v, dtype=float)
    if axs is None:
        fig, axs = plt.subplots(1, 3, figsize=figsize, layout="constrained", gridspec_kw={"width_ratios": [1.3, 1, 1]})
    order = np.argsort(lam)
    xl = np.log10(lam[order])  # label positions: at least 0.03 dex apart
    for r in range(1, xl.size):
        xl[r] = max(xl[r], xl[r - 1] + 0.03)
    xl -= max(0.0, xl[-1] - np.log10(lam.max()) - 0.02)
    for r, j in enumerate(order):
        c = col[ids[j]]
        axs[0].vlines(lam[j], 1e-3, f[j], color=c, lw=3)
        axs[0].annotate(
            ids[j],
            (lam[j], f[j]),
            xytext=(10 ** xl[r], 25.0),
            textcoords="data",
            rotation=90,
            ha="center",
            va="bottom",
            fontsize=FONT_MIN["legend"],
            color=c,
            arrowprops=dict(arrowstyle="-", color=c, lw=0.8, shrinkA=0, shrinkB=2),
        )
    axs[0].set(
        xscale="log", yscale="log", ylim=(1e-3, 3e4), xlabel=r"rest wavelength $\lambda_0$ [Å]", ylabel=r"oscillator strength $f$"
    )
    axs[0].xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
    axs[0].xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    axs[0].set_xticks([1000, 1250, 1500, 2000, 2800])
    y = np.arange(len(lines))
    axs[1].barh(y, dv / 1e3, color=[col[i_] for i_ in ids], height=0.6)
    axs[1].axvline(0, color=plt.rcParams["text.color"], lw=1)
    axs[1].set_yticks(y, [l.id for l in lines], fontsize=FONT_MIN["legend"])
    axs[1].invert_yaxis()
    axs[1].set(xlabel=rf"$\Delta v_j = c\,\ln(\lambda_{{\rm ref}}/\lambda_j)$ [$10^3$ km/s], ref = {line_bundle.ref_line_id}")
    b = np.sqrt(2 * phys.kB * T_K / m) / 1e3  # phys.kB = k_B / m_u [m^2 s^-2 K^-1] -> [km/s]
    nat = lam * 1e-10 * A21 * 1e9 / (4 * np.pi) / 1e3  # lambda_0 gamma [km/s]
    axs[2].scatter(nat, y, color=[col[i_] for i_ in ids], s=80, marker="o", label=r"natural $\lambda_0 A_{21}/4\pi$")
    axs[2].scatter(
        b, y, color=[col[i_] for i_ in ids], s=110, marker="D", label=rf"thermal $b$ at $T = 10^{{{np.log10(T_K):.0f}}}$ K"
    )
    for j in range(len(lines)):
        axs[2].plot([nat[j], b[j]], [y[j], y[j]], color=col[ids[j]], lw=1, alpha=0.6)
    axs[2].set_yticks(y, [l.id for l in lines], fontsize=FONT_MIN["legend"])
    axs[2].invert_yaxis()
    axs[2].set(xscale="log", xlabel="width [km/s]", xlim=(min(nat.min(), b.min()) / 3, max(nat.max(), b.max()) * 30))
    axs[2].legend(fontsize=FONT_MIN["legend"], loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=1, frameon=False)
    return axs[0].figure, axs


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
    ax[3].annotate(
        f"ICs: a0 = {float(bundle.a0[k]):.4g} at x0 = {float(bundle.x0[k]):.4g}\n"
        f"({'absorption' if bool(bundle.is_absorption[k]) else 'emission'} point)",
        (s[ic], a[ic]),
        textcoords="offset points",
        xytext=(15 if ic == 0 else -15, -40),
        ha="left" if ic == 0 else "right",
        fontsize=FONT_MIN["legend"],
        arrowprops=dict(arrowstyle="->", color=FG),
    )
    ax[3].set(xlabel=LABELS["s"], ylabel="a", title="scale_factors (N_cosmo, N_sk, N_ppsk)")
    for c, name, colr in zip(range(3), "xyz", (PALETTE["blue"], PALETTE["red"], PALETTE["green"])):
        ax[4].plot(s, los[:, c], color=colr, label=rf"$\hat d_{name}$")
    ax[4].set(
        xlabel=LABELS["s"],
        ylabel="component",
        ylim=(-1.15, 1.15),
        title=f"los_directions; array_direction = {int(bundle.array_direction[k]):+d}",
    )
    ax[4].legend(fontsize=FONT_MIN["legend"], ncol=3)
    ax[5].set_axis_off()
    rows = [(n, tuple(np.shape(getattr(bundle, n)))) for n in bundle._leaves]
    txt = "\n".join(f"{n:<28s}{str(sh):>18s}" for n, sh in rows)
    ax[5].text(
        0.0,
        1.0,
        "SkewerBundle (a JAX pytree)\n\n" + txt,
        family="monospace",
        fontsize=FONT_MIN["legend"],
        va="top",
        transform=ax[5].transAxes,
        color=FG,
    )
    ax[5].text(
        0.0,
        0.02,
        "quadri_vectors[..., 0] = a, [..., 1:4] = (x, y, z):\nthe events (a, x, y, z) of the sampling points",
        fontsize=FONT_MIN["legend"],
        transform=ax[5].transAxes,
        color=PALETTE["cyan"],
    )
    add_panel_letters(ax[:5])
    return fig, axs


def _pair_label(pair):
    from skewerjax.core.stats.two_point import ZERO_ORDER_KEY

    a, b = (r"$\mathcal{C}$" if t == ZERO_ORDER_KEY else t for t in pair)
    return f"{a}–{b}"


def plot_terms_by_order(
    stats, kind: str = "P", *, orders=None, max_lag: float | None = None, linthresh: float | None = None, figsize=None
):
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
        key = tuple(sorted((a, b), key=lambda t: stats.term_keys.index(t)))
        groups.setdefault(stats.pair_orders[(a, b)], {}).setdefault(key, 0.0)
        groups[stats.pair_orders[(a, b)]][key] = groups[stats.pair_orders[(a, b)]][key] + np.asarray(y)[sel]
    scale = max(np.max(np.abs(np.asarray(v)[sel])) for v in by_order.values())
    if orders is None:
        orders = [n for n in sorted(by_order) if np.max(np.abs(np.asarray(by_order[n])[sel])) > 1e-12 * scale]
    fig, axs = plt.subplots(
        1, len(orders), figsize=figsize or (6.5 * len(orders), 6.5), layout="constrained", sharey=(kind == "P")
    )
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
    P = np.asarray(grf.power_at(*(comps[i] for i in range(3)))).reshape(kp.size, -1)  # P(k, mu): mu along grf.los_axis
    area = grf.L[tr[0]] * grf.L[tr[1]]
    if return_sigma:
        return P.sum(-1) / area, np.sqrt((P**2).sum(-1)) / area
    return P.sum(-1) / area


def plot_p1d_input_vs_measured(k, curves: dict, reference, *, ref_label: str = "input", axs=None, k_range=None, figsize=(15, 10)):
    """Measured P1D curves against an input spectrum, with their ratios (notebook 08).

    curves : {label: (P (N_k,), sigma (N_k,) or None, colour, style dict)}; reference (N_k,): the expected spectrum on
    the same k. Top: k P / pi on log axes; bottom: P / reference. Returns (fig, axs)."""
    k = np.asarray(k)
    if axs is None:
        fig, axs = plt.subplots(
            2, 1, figsize=figsize, sharex=True, gridspec_kw={"height_ratios": [2.2, 1.3]}, layout="constrained"
        )
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


def plot_cylinder(
    s,
    profiles: dict,
    *,
    cloud=None,
    flux=None,
    progress: float = 1.0,
    radius: float = 1.0,
    view=(14.0, -62.0),
    cloud_cmap: str = None,
    cloud_clim=None,
    n_rings: int = 5,
    flux_on_cylinder: bool = False,
    fig=None,
    figsize=(22, 13),
):
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
    ax.plot(
        np.full_like(phi, s_p),
        1.08 * radius * np.cos(phi),
        1.08 * radius * np.sin(phi),
        color=PALETTE["yellow"],
        lw=2.0,
        alpha=0.95,
        label="light front",
    )  # where the light is at this progress
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
            ax.scatter(
                ax_c[keep],
                np.asarray(cloud["s_perp"])[keep],
                np.asarray(cloud["t_perp"])[keep],
                color=rgba,
                s=size,
                lw=0,
                depthshade=False,
                zorder=1,
            )
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
            ax.plot(
                s, yy * np.cos(th), yy * np.sin(th), color=c, lw=1.4, alpha=alpha, zorder=5, label=lab if alpha == 1.0 else None
            )
    if flux is not None:
        v, s_of_v, e = (np.asarray(flux[k], dtype=float) for k in ("v", "s_of_v", "edges"))
        frac = np.clip((s_p - e[:-1]) / np.maximum(np.diff(e), 1e-30), 0.0, 1.0)  # how much of each stretch is crossed
        for (lab, (tb, c)), th in zip(flux["tau_bins"].items(), list(angles[len(profiles) :]) + [0.0] * len(flux["tau_bins"])):
            tb = np.asarray(tb, dtype=float)
            F_now, F_end = np.exp(-(frac[:, None] * tb).sum(0)), np.exp(-tb.sum(0))
            if flux_on_cylinder:
                r = radius * (1.0 - F_now)
                ax.plot(s_of_v, r * np.cos(th), r * np.sin(th), color=c, lw=1.4, zorder=6, label=f"1 − F, {lab}")
            axf.plot(v, F_end, color=c, lw=1.0, alpha=0.3, path_effects=[])
            axf.plot(v, F_now, color=c, lw=1.6, label=f"{lab}: absorbed up to s = {s_p:.1f} cMpc")
        axf.set(
            xlim=(v[0], v[-1]), ylim=(-0.05, 1.1), xlabel=LABELS["v_dagger"] + ", each line in its own frame", ylabel=LABELS["F"]
        )
        axf.legend(fontsize=FONT_MIN["legend"], loc="lower left", ncol=2)
    ax.set(xlim=(s0, s1), ylim=(-1.25 * radius, 1.25 * radius), zlim=(-1.25 * radius, 1.25 * radius))
    ax.set_box_aspect((3.2, 1.0, 1.0), zoom=1.25)
    ax.set_xlabel(LABELS["s"], labelpad=14)
    ax.set_yticks([])
    ax.set_zticks([])
    dark_3d(ax)
    ax.view_init(*view)
    h, lab = ax.get_legend_handles_labels()
    fig.legend(h, lab, loc="upper left", bbox_to_anchor=(0.01, 0.93), fontsize=FONT_MIN["legend"], frameon=True, ncol=1)
    return fig, (ax, axf)


# --------------------------------------------------------------------------- tutorial figures (third pass)
def _textbox(ax, x, y, s, *, color=None, transform=None, **kw):
    """A short label in a dark rounded box (readable on any background)."""
    kw = {"fontsize": FONT_MIN["textbox"], "ha": "left", "va": "center", "zorder": 30, **kw}
    return ax.text(
        x,
        y,
        s,
        color=color or FG,
        transform=transform or ax.transData,
        bbox=dict(boxstyle="round,pad=0.25", facecolor=BG, edgecolor=color or DIM, alpha=0.88, lw=1.0),
        **kw,
    )


def plot_scale_factor_solutions(
    x,
    a_num,
    exact: dict,
    colors: dict,
    *,
    x0: float = 0.0,
    a0: float = 1.0,
    h: float = None,
    linthresh: float = 1e-5,
    axs=None,
    figsize=(14, 10),
):
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
    specials = (
        {}
        if h is None
        else {
            "radiation": (x0 - a0 / h, "radiation: a → 0"),
            "matter": (x0 - 2.0 * np.sqrt(a0) / h, "matter: a′ = 0"),
            "Lambda": (x0 + 1.0 / (a0 * h), "Λ: a → ∞"),
        }
    )
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
    cm0.set_bad((0, 0, 0, 0))  # empty bins: the background
    im0 = axs[0].pcolormesh(ex, ey, np.ma.masked_invalid(med.T), cmap=cm0, vmin=-7, vmax=0, shading="flat", rasterized=True)
    fig.colorbar(im0, ax=axs[0], label=r"median $\log_{10} x_{\rm HI}$ of the cells in the bin")
    ld, lt, lf = cloudy_grid
    im1 = axs[1].pcolormesh(ld, lt, np.maximum(lf, -8).T, cmap=CMAPS["hot"], vmin=-8, vmax=0, shading="nearest", rasterized=True)
    fig.colorbar(im1, ax=axs[1], label=r"CLOUDY $\log_{10} f_{\rm SiIII}(n_{\rm H}, T)$ (table, at the snapshot redshift)")
    lev = np.log10(np.maximum(cnt, 0.5))
    for ax in axs:
        ax.contour(cx, cy, lev.T, levels=[0.0, 1.0, 2.0, 3.0], colors=[FG], linewidths=[0.8, 1.1, 1.4, 1.8])
        ax.set(
            xlabel=r"$\log_{10} n_{\rm H}$ [cm$^{-3}$]", ylabel=r"$\log_{10} T$ [K]", xlim=(ex[0], ex[-1]), ylim=(ey[0], ey[-1])
        )
    info = {}
    if sf is not None and np.any(sf):
        p = np.polyfit(lx[sf], ly[sf], 1)
        info["eos"] = dict(slope=float(p[0]), intercept=float(p[1]), scatter=float(np.std(ly[sf] - np.polyval(p, lx[sf]))))
        xs = np.array([lx[sf].min(), np.percentile(lx[sf], 99.5)])  # where the power law describes the cells
        for ax in axs:
            ax.plot(xs, np.polyval(p, xs), color=PALETTE["cyan"], lw=2.2, ls="--", path_effects=[])
        x_box = xs[0] + 1.3  # below the strip, clear of the cells
        _textbox(
            axs[0],
            x_box,
            np.polyval(p, x_box) - 0.95,
            "star-forming cells: effective EOS\n"
            + rf"$T \propto n_{{\rm H}}^{{{p[0]:.2f}}}$ ($\pm${info['eos']['scatter']:.2f} dex)",
            color=PALETTE["cyan"],
        )
    if n_thr is not None:
        for ax in axs:
            ax.axvline(np.log10(n_thr), color=PALETTE["yellow"], ls=":", lw=2.0, path_effects=[])
        _textbox(
            axs[0],
            np.log10(n_thr) + 0.1,
            ey[-1] - 0.35,
            "star-formation\nthreshold " + rf"{n_thr:.2f} cm$^{{-3}}$",
            color=PALETTE["yellow"],
        )
    _textbox(axs[0], ex[0] + 0.3, 3.75, "photo-heated gas:\n" + r"floor near $10^4$ K", color=FG)
    _textbox(axs[0], ex[0] + 0.3, 6.6, "hot, shock-heated\n" + r"halo gas ($x_{\rm HI} \ll 1$)", color=FG)
    i, j = np.unravel_index(np.argmax(lf), lf.shape)
    info["cloudy_max"] = dict(log_n_H=float(ld[i]), log_T=float(lt[j]), log_f=float(lf[i, j]))
    _textbox(
        axs[1],
        max(ex[0] + 0.3, min(ld[i], ex[-1]) - 2.2),
        lt[j] - 0.55,
        "Si III ridge: most abundant near\n" + rf"$\log T = {lt[j]:.2f}$ ($\log n_{{\rm H}} = {ld[i]:.1f}$)",
        color=FG,
    )
    j_lo = np.argmax(lf[np.argmin(np.abs(ld + 3.0))])
    _textbox(
        axs[1],
        ex[0] + 0.3,
        ey[0] + 0.25,
        r"photoionised at low $n_{\rm H}$ (at $\log n_{\rm H} = -3$: peak near " + rf"$\log T = {lt[j_lo]:.2f}$)",
        color=FG,
    )
    axs[1].set_xlim(ex[0], ex[-1])
    axs[1].set_ylim(ey[0], ey[-1])
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
    pts = np.concatenate(
        [path[i] + np.linspace(0.0, 1.0, n, endpoint=False)[:, None] * seg[i] for i, n in enumerate(n_sub)] + [path[-1:]]
    )
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
    js, ps = [], []
    for c, h in zip(centres, sizes):  # the neighbours of every centre, nearest image
        j = np.asarray(tree.query_ball_point(np.mod(c, box) if box is not None else c, 4.0 * h), dtype=np.int64)
        js.append(j)
        ps.append(pos[j] if box is None else pos[j] + box * np.round((c - pos[j]) / box))
    js, ps = np.concatenate(js), np.concatenate(ps)
    key = np.column_stack([js.astype(float), np.round(ps, 6) + 0.0])  # one point per (generator, image), first kept
    pts = ps[np.sort(np.unique(key, axis=0, return_index=True)[1])]
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


def _cell_faces(cells, centres, sizes):
    """The faces of the cells of generating points ``centres`` (K, 3) [cMpc, unwrapped]: cubes for a grid of cells
    (``cells.grid``), else Voronoi polyhedra (``_voronoi_faces``). One list of (n_v, 3) faces per cell."""
    if cells.grid is not None:
        side = float(cells.box) / np.asarray(cells.grid, dtype=float)
        return [_cube_faces(c, side) for c in centres]
    return _voronoi_faces(cells, centres, sizes)


# --------------------------------------------------------------------------- velocity arrows (04, 05, 07)
ARROW_FRACTION, ARROW_PERCENTILE = 0.8, 90.0  # the 90th-percentile speed is drawn 0.8 median cell sizes long
ARROW_EXTEND = 0.1  # the axis limits may grow by this fraction of the largest span per side to hold the arrows
ARROW_MIN_SPAN = 0.06  # ... and that arrow is at least this fraction of the panel's largest span (many small cells)


def velocity_arrow_scale(
    velocities,
    sizes,
    *,
    span=None,
    fraction: float = ARROW_FRACTION,
    percentile: float = ARROW_PERCENTILE,
    min_span_fraction: float = ARROW_MIN_SPAN,
    scale=None,
):
    """The length of the velocity arrows per unit speed [cMpc per km/s] for the cells shown in a figure.

    The ``percentile``-th percentile of their speeds is drawn ``fraction`` x the median of their sizes long: the arrows
    stand out without swamping the cells, and one fast outlier does not shrink every other arrow (a percentile, not
    the maximum); with ``span`` (the largest extent of the panels [cMpc]) that arrow is at least ``min_span_fraction``
    x span long, so that the arrows stay visible when many small cells fill a large panel. velocities : (K, 3) vectors
    or (K,) speeds or signed components [km/s]; sizes : (K,) cell sizes V^(1/3) [cMpc]; scale : an explicit value
    [cMpc per km/s], returned as is (the override). Finite and > 0 for any input: without moving cells 1 km/s stands in
    for the percentile, without sizes 1 cMpc for the median.
    """
    if scale is not None:
        if not (np.isfinite(scale) and scale > 0):
            raise ValueError(f"the arrow scale must be a finite number > 0 [cMpc per km/s], got {scale!r}.")
        return float(scale)
    v = np.asarray(velocities, dtype=float)
    speed = np.linalg.norm(v, axis=-1) if v.ndim == 2 else np.abs(v.ravel())
    speed = speed[np.isfinite(speed) & (speed > 0)]
    h = np.asarray(sizes, dtype=float).ravel()
    h = h[np.isfinite(h) & (h > 0)]
    v_p = float(np.percentile(speed, percentile)) if speed.size else 1.0
    length = fraction * (float(np.median(h)) if h.size else 1.0)
    if span is not None and np.isfinite(span) and span > 0:
        length = max(length, min_span_fraction * float(span))
    return length / v_p


def round_speed(v: float, *, below: bool = False) -> float:
    """The number of the form 1, 2 or 5 x 10^n nearest to ``v`` > 0 in log (``below``: the largest one <= v); 1 for
    v <= 0 or not finite. The speed of the reference arrows [km/s]."""
    if not (np.isfinite(v) and v > 0):
        return 1.0
    n = np.floor(np.log10(v))
    cand = np.array([m * 10.0**e for e in (n - 1, n, n + 1) for m in (1.0, 2.0, 5.0)])
    if below:
        return float(cand[cand <= v * (1 + 1e-12)].max())
    return float(cand[np.argmin(np.abs(np.log(cand / v)))])


def velocity_key_speed(velocities) -> float:
    """The speed of the reference arrow [km/s]: the median speed of the moving cells, rounded to 1, 2 or 5 x 10^n."""
    v = np.asarray(velocities, dtype=float)
    speed = np.linalg.norm(v, axis=-1) if v.ndim == 2 else np.abs(v.ravel())
    speed = speed[np.isfinite(speed) & (speed > 0)]
    return round_speed(float(np.median(speed)) if speed.size else 1.0)


def _fit_arrows(starts, arrows, lo, hi, *, extend: float = ARROW_EXTEND):
    """Arrows from ``starts`` (K, d) that stay inside the limits ``lo``, ``hi`` (d,): the limits grow to hold the tips,
    by at most ``extend`` x the largest span per side, and an arrow that still leaves them is shortened so that its tip
    lies on the limit (``capped``). Returns (arrows (K, d), capped (K,) bool, lo, hi)."""
    starts, arrows = np.atleast_2d(np.asarray(starts, dtype=float)), np.atleast_2d(np.asarray(arrows, dtype=float))
    lo, hi = np.array(lo, dtype=float), np.array(hi, dtype=float)
    tips = starts + arrows
    margin = extend * float(np.max(hi - lo))
    lo, hi = np.minimum(lo, np.maximum(tips.min(0), lo - margin)), np.maximum(hi, np.minimum(tips.max(0), hi + margin))
    with np.errstate(divide="ignore", invalid="ignore"):
        t_hi = np.where(arrows > 0, (hi - starts) / arrows, np.inf)
        t_lo = np.where(arrows < 0, (lo - starts) / arrows, np.inf)
    t = np.clip(np.minimum(t_hi, t_lo).min(1), 0.0, 1.0)
    return arrows * t[:, None], t < 1.0 - 1e-12, lo, hi


def _screen_axis(view) -> int:
    """The data axis most parallel to the screen for a 3D ``view`` = (elevation, azimuth) [deg]: the one with the
    smallest component along the line of sight, so its projection shortens a length the least."""
    el, az = np.radians(view[0]), np.radians(view[1])
    sight = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    return int(np.argmin(np.abs(sight)))


def _velocity_key_3d(ax, lo, hi, scale: float, v_key: float, view, avoid, *, note: str = "lengths in projection") -> float:
    """The reference arrow of a 3D panel, inside the scene: ``v_key`` km/s at the arrows' ``scale`` [cMpc per km/s],
    along the data axis most parallel to the screen (``_screen_axis``), so that the projection shortens it exactly as it
    does the arrows. It starts from the box corner (of the upper half on the screen) whose arrow and label are, in projection,
    farthest from the points ``avoid`` (cell vertices, path), and points inwards; its label ("200 km/s" and ``note``) sits in a
    small box past its tip or beside its middle (inwards, above or below on the screen), wherever it is clearest. The speed is lowered to a rounder value when the arrow would be longer than 60 % of
    the box along its axis. Returns the speed drawn [km/s]."""
    from matplotlib import patheffects as pe

    lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    span = hi - lo
    i = _screen_axis(view)
    if v_key * scale > 0.6 * span[i]:
        v_key = round_speed(0.6 * span[i] / scale, below=True)
    length = v_key * scale
    el, az = np.radians(view[0]), np.radians(view[1])
    up = np.array([-np.sin(el) * np.cos(az), -np.sin(el) * np.sin(az), np.cos(el)])  # the screen's up, in data space
    inset = 0.06 * span
    corners = np.array(np.meshgrid(*[(0, 1)] * 3, indexing="ij")).reshape(3, -1).T
    pos = np.where(corners == 1, hi - inset, lo + inset)
    avoid = np.asarray(avoid, dtype=float).reshape(-1, 3)

    def label_at(start, direction, side="inward"):  # past the tip, or beside the middle: inwards, above or below on the screen
        if side == "ahead":
            return start + (length + 0.12 * span[i]) * direction
        mid = start + 0.5 * length * direction
        inward = 0.5 * (lo + hi) - mid
        inward[i] = 0.0
        inward = inward / span.max()
        inward = -up if not np.any(inward) else inward / np.linalg.norm(inward)
        offset = {"inward": inward, "above": up, "below": -up}[side]
        return np.clip(mid + 0.09 * span.max() * offset, lo, hi)

    from mpl_toolkits.mplot3d import proj3d

    M = ax.get_proj()  # the panel's projection (limits, box aspect and view are set): clearance on the screen

    def screen(p):
        x, y, _ = proj3d.proj_transform(*np.atleast_2d(p).T, M)
        return np.stack([x, y], -1)

    av = screen(avoid) if avoid.size else np.zeros((0, 2))
    box2 = screen(pos)
    scale2 = float(np.max(np.ptp(box2, axis=0)))
    y2 = (box2[:, 1] - box2[:, 1].min()) / scale2  # the height of every corner on the screen
    best = None
    for k in range(len(corners)):  # the upper half on the screen: the clearest of arrow and label, then the highest
        if y2[k] < np.median(y2):
            continue
        d = np.zeros(3)
        d[i] = -1.0 if corners[k][i] == 1 else 1.0  # inwards
        for side in ("inward", "ahead", "above", "below"):
            pts_k = screen(np.vstack([pos[k] + np.linspace(0, 1, 9)[:, None] * (length * d), label_at(pos[k], d, side)[None]]))
            dist = float(np.min(np.linalg.norm(av[None] - pts_k[:, None], axis=-1))) / scale2 if av.size else 1.0
            score = dist + 0.05 * y2[k]
            if best is None or score > best[0]:
                best = (score, pos[k], d, side)
    _, start, direction, side = best
    ax.quiver(
        *start,
        *(length * direction),
        color=FG,
        lw=2.8,
        arrow_length_ratio=0.3,
        normalize=False,
        zorder=30,
        path_effects=[pe.Stroke(linewidth=5.0, foreground=BG), pe.Normal()],
    )
    label = f"{v_key:g} km/s" + (f"\n{note}" if note else "")
    at = label_at(start, direction, side)  # the label never leaves the scene
    ax.text(
        *at,
        label,
        fontsize=FONT_MIN["annotation"],
        color=FG,
        ha="center",
        va="center",
        zorder=31,
        bbox=dict(boxstyle="round,pad=0.2", facecolor=BG, edgecolor=DIM, alpha=0.85),
    )
    return v_key


def _velocity_key_2d(ax, v_key: float, scale: float, *, x: float = 0.97, y: float = 0.93):
    """The reference arrow of a 2D panel whose x axis is a length [cMpc]: ``v_key`` km/s at ``scale`` [cMpc per km/s],
    horizontal, ending at ``x`` (axes fraction) at height ``y``, labelled in a small box to its left."""
    from matplotlib import patheffects as pe

    x0, x1 = ax.get_xlim()
    tip = x0 + x * (x1 - x0)
    tr = ax.get_xaxis_transform()
    ax.annotate(
        "",
        (tip, y),
        (tip - v_key * scale, y),
        xycoords=tr,
        zorder=12,
        arrowprops=dict(
            arrowstyle="-|>",
            color=FG,
            lw=2.4,
            mutation_scale=14,
            shrinkA=0,
            shrinkB=0,
            path_effects=[pe.withStroke(linewidth=4.0, foreground=BG)],
        ),
    )
    ax.text(
        tip - v_key * scale - 0.01 * (x1 - x0),
        y,
        f"{v_key:g} km/s",
        transform=tr,
        ha="right",
        va="center",
        zorder=12,
        fontsize=FONT_MIN["annotation"],
        color=FG,
        bbox=dict(boxstyle="round,pad=0.25", facecolor=BG, edgecolor=DIM, alpha=0.9),
    )


def _mark_capped(ax, tips, capped):
    """An open circle at the tip of every capped arrow (longer than drawn: cut at the panel edge)."""
    if np.any(capped):
        t = np.atleast_2d(tips)[capped]
        ax.scatter(*t.T, s=60, facecolors="none", edgecolors=FG, linewidths=1.6, depthshade=False, zorder=29)


def _draw_cells_3d(
    ax,
    faces,
    rgba,
    g,
    path,
    lo,
    hi,
    *,
    arrows=None,
    alpha: float = 0.55,
    view=(24.0, -62.0),
    zoom: float = 1.1,
    path_color=None,
    arrow_color=None,
):
    """Cells as 3D volumes on ``ax``: ``faces`` (one list of (n_v, 3) faces per cell) filled with ``rgba`` (one colour per
    cell, opacity ``alpha``), edges in the foreground colour, the generating points ``g`` (dots), the skewer ``path``
    (N, 3) as a line and, optionally, ``arrows`` (K, 3) from the generating points [cMpc]; limits ``lo``, ``hi`` (3,),
    aspect 1:1:1, ``view`` = (elevation, azimuth); ``path_color``: default the foreground colour; ``arrow_color``: one
    colour, or one per arrow (then drawn with a thin dark outline, visible over a cell of the same colour). The drawing
    of ``plot_gas_cells`` and ``plot_deposition_cells``."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    poly = [q for fc in faces for q in fc]
    fcol = np.repeat(np.asarray(rgba, dtype=float), [len(fc) for fc in faces], axis=0)
    fcol[:, 3] = alpha
    ax.add_collection3d(Poly3DCollection(poly, facecolors=fcol, edgecolors=FG, linewidths=0.7))
    ax.plot(*np.asarray(path).T, color=path_color or FG, lw=2.6, label="skewer")
    ax.scatter(*np.asarray(g).T, s=22, color=FG, depthshade=False, label="generating points")
    if arrows is not None:
        per_arrow = arrow_color is not None and not isinstance(arrow_color, str) and np.ndim(arrow_color) == 2
        q = ax.quiver(
            *np.asarray(g).T,
            *np.asarray(arrows).T,
            color=arrow_color if arrow_color is not None else FG,
            lw=2.4 if per_arrow else 2.0,
            arrow_length_ratio=0.3,
            normalize=False,
        )
        if per_arrow:  # the lines of quiver: every shaft, every first head side, every second
            from matplotlib import patheffects as pe

            c = np.asarray(arrow_color, dtype=float)
            q.set_color(np.concatenate([c, c, c]))
            q.set_path_effects([pe.Stroke(linewidth=4.2, foreground=FG), pe.Normal()])
    ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), zlim=(lo[2], hi[2]))
    ax.set_box_aspect(np.asarray(hi) - np.asarray(lo), zoom=zoom)
    ax.view_init(*view)
    for a_, lab in zip((ax.xaxis, ax.yaxis, ax.zaxis), "xyz"):
        a_.set_major_locator(mpl.ticker.MaxNLocator(3))
        a_.set_label_text(f"{lab} [cMpc]")
        a_.labelpad = 10 if lab == "z" else 12


def _density_norm(log_n, decades: float = np.inf):
    """The norm of ``CMAPS["density"]`` for the cell densities (``plot_gas_cells``: 04, 05; the deposition figures: 07):
    log10 n_com over the cells drawn, at most ``decades`` below the largest, widened by 0.05 dex at both ends so that
    the extreme cells keep a colour off the ends of the map."""
    hi = float(np.max(log_n))
    lo = max(float(np.min(log_n)), hi - decades)
    return mpl.colors.Normalize(lo - 0.05, max(hi, lo + 1e-3) + 0.05)


def plot_gas_cells(
    cells,
    selected,
    path,
    species=None,
    *,
    skewer: int = 0,
    view=(24.0, -62.0),
    zoom: float = 1.1,
    decades: float = 12.0,
    alpha: float = 0.55,
    arrow_scale=None,
    figsize=None,
):
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
    (1) the comoving number density of the species (log, ``CMAPS["density"]`` with ``_density_norm`` as in 07, its own
    colour bar, at most ``decades`` below its maximum: lower values take the lowest colour); (2) the
    temperature (log, ``CMAPS["hot"]``); (3) the peculiar velocity: colour the component along the skewer v_par
    (``CMAPS["diverging"]``, symmetric) and one 3D arrow per cell, from its generating point. The temperature and
    velocity colour bars, and the arrow scale, are shared by the rows. ``view`` = (elevation, azimuth) [deg]; ``zoom``
    scales the 3D boxes (data aspect 1:1:1); ``alpha`` is the opacity of the cell faces.

    Arrows: one scale for the figure, ``velocity_arrow_scale`` of the cells drawn (their 90th-percentile speed is 0.8
    median cell sizes long, and at least 6 % of the box), or ``arrow_scale`` [cMpc per km/s]; every arrow inside the panel (the limits grow a little,
    an arrow still too long is cut at the edge and ends with an open circle); a reference arrow of a round speed inside
    each velocity panel (``_velocity_key_3d``; lengths are shown in projection).

    Returns (fig, axs (n_species, 3), info {"n_selected", "n_crossed", "crossed": indices into ``selected``,
    "path_length": [cMpc], "arrow_scale": [cMpc per km/s], "arrow_kms": the speed of an arrow one median cell size long
    [km/s], "key_kms": the reference arrow [km/s], "n_capped": arrows cut at the panel edge}).
    """
    import re

    sp_list = list(cells.species_keys) if species is None else ([species] if isinstance(species, str) else list(species))
    pts = np.asarray(path.spatial_points[skewer] if hasattr(path, "spatial_points") else path, dtype=float)
    box = cells.box
    idx, g, tang = _cells_on_path(selected, pts, box)
    size = np.asarray(selected.size_cMpc, dtype=float)[idx]
    faces = _cell_faces(cells, g, size)
    s0 = np.cumsum(np.r_[0.0, np.linalg.norm(np.diff(pts, axis=0), axis=1)])
    allv = np.concatenate([np.concatenate(f) for f in faces] + [pts])
    lo, hi = allv.min(0), allv.max(0)
    fields = {
        sp: {k: np.asarray(selected.species[sp][k], dtype=float)[idx] for k in ("number_density", "temperature", "velocity")}
        for sp in sp_list
    }
    logT = {sp: np.log10(np.maximum(f["temperature"], 1e-300)) for sp, f in fields.items()}
    t_lim = (min(x.min() for x in logT.values()), max(x.max() for x in logT.values()))
    t_lim = (t_lim[0], max(t_lim[1], t_lim[0] + 1e-3))
    v_par = {sp: np.sum(f["velocity"] * tang, axis=1) for sp, f in fields.items()}
    v_lim = max(max(float(np.abs(x).max()) for x in v_par.values()), 1e-6)
    vel_all = np.concatenate([f["velocity"] for f in fields.values()])
    scale = velocity_arrow_scale(
        vel_all, np.tile(size, len(sp_list)), span=float(np.max(hi - lo)), scale=arrow_scale
    )  # one per figure
    h_med = float(np.median(size))
    arrows, capped = {}, {}
    for sp, f in fields.items():  # the limits grow once, for every panel (the same view in each)
        arrows[sp], capped[sp], lo, hi = _fit_arrows(g, f["velocity"] * scale, lo, hi)
    for sp, f in fields.items():  # the final limits: shorten again against them
        arrows[sp], capped[sp], _, _ = _fit_arrows(g, f["velocity"] * scale, lo, hi, extend=0.0)
    v_key = velocity_key_speed(vel_all)
    n = len(sp_list)
    fig = plt.figure(figsize=figsize or (24, 7.6 * n + 1.6), layout="constrained")
    fig.get_layout_engine().set(
        rect=(0.0, 0.0, 0.96, 0.97), wspace=0.08
    )  # room for a suptitle and the z labels (3D labels are not in the layout)
    fig.text(1.0, 0.5, " ", alpha=0.0)  # keeps that room under bbox_inches="tight" (inline)
    axs = np.array(
        [[fig.add_subplot(n, 3, 3 * r + c + 1, projection="3d", computed_zorder=False) for c in range(3)] for r in range(n)]
    )
    norm_t = mpl.colors.Normalize(*t_lim)
    norm_v = mpl.colors.Normalize(-v_lim, v_lim)
    for r, sp in enumerate(sp_list):
        name = re.sub(r"^([A-Z][a-z]?)([IVX]+)$", r"\1 \2", sp)  # "SiIII" -> "Si III"
        f = fields[sp]
        ln = np.log10(np.maximum(f["number_density"], 1e-300))
        norm_n = _density_norm(ln, decades)
        cols = ((CMAPS["density"], norm_n, ln), (CMAPS["hot"], norm_t, logT[sp]), (CMAPS["diverging"], norm_v, v_par[sp]))
        titles = (f"{name} density", f"{name} temperature", f"{name} peculiar velocity")
        labels = (
            rf"$\log_{{10}} n_{{\rm com}}$ [cm$^{{-3}}$] ({name})",
            r"$\log_{10} T$ [K] (shared by the rows)",
            r"$v_\parallel$ along the skewer [km s$^{-1}$] (shared)",
        )
        for c, (cmap, norm, val) in enumerate(cols):
            ax = axs[r, c]
            _draw_cells_3d(
                ax,
                faces,
                mpl.colormaps[cmap](norm(val)),
                g,
                pts,
                lo,
                hi,
                alpha=alpha,
                view=view,
                zoom=zoom,
                arrows=arrows[sp] if c == 2 else None,
            )
            if c == 2:  # the arrows: their key in the scene, the caption in the title
                _mark_capped(ax, g + arrows[sp], capped[sp])
                v_key = _velocity_key_3d(
                    ax, lo, hi, scale, v_key, view, np.concatenate([g, pts] + [np.concatenate(f) for f in faces]), note=""
                )
                ax.set_title(
                    titles[c] + "\n(arrows: lengths in projection" + ("; o: cut at the edge)" if np.any(capped[sp]) else ")"),
                    pad=0,
                )
            else:
                ax.set_title(titles[c], pad=0)
            cb = fig.colorbar(
                mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                ax=ax,
                location="bottom",
                shrink=0.7,
                aspect=25,
                pad=0.0,
                label=labels[c],
                extend="min" if c == 0 and ln.min() < norm.vmin else "neither",
            )
            cb.locator = mpl.ticker.MaxNLocator(4)
            cb.update_ticks()
    info = dict(
        n_selected=len(selected),
        n_crossed=int(idx.size),
        crossed=idx,
        path_length=float(s0[-1]),
        arrow_scale=scale,
        arrow_kms=h_med / scale,
        key_kms=v_key,
        n_capped=int(max(np.count_nonzero(c) for c in capped.values())),
    )
    return fig, axs, info


def plot_voigt_limits(x, sigma, gamma, curves: dict, ref_key: str = "scipy_wofz", *, linthresh: float = 1e-6, figsize=None):
    """Voigt profiles V(x; sigma, gamma) for extreme (sigma, gamma) pairs, every implementation in ``curves`` = {name:
    (N_pairs, N_x)} on the reference ``ref_key`` (thick, light), and their relative residuals (symlog, ``linthresh``).
    The pairs show the Gaussian limit (gamma << sigma), the Lorentzian limit (sigma << gamma) and the mixed cases.
    Returns (fig, axs): (a) the profiles, (b) the residuals below, sharing x.
    """
    from matplotlib.lines import Line2D

    fig, axs = plt.subplots(2, 1, figsize=figsize or (15, 11), gridspec_kw={"height_ratios": [2.2, 1]}, layout="constrained")
    ax, axr = axs[0], axs[1]
    axr.sharex(ax)
    cols = [PALETTE[k] for k in ("blue", "red", "green", "yellow", "cyan", "purple")]
    styles = {
        name: st
        for name, st in zip(
            [k for k in curves if k != ref_key], (MODE_STYLES["default"], MODE_STYLES["matched"], MODE_STYLES["alternative"])
        )
    }
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
    ax.legend(
        handles=[Line2D([], [], color=FG, **MODE_STYLES["reference"], label=f"{ref_key} (reference)")]
        + [Line2D([], [], color=FG, **st, label=name) for name, st in styles.items()],
        loc="upper right",
        fontsize=FONT_MIN["legend"],
    )
    axr.set_yscale("symlog", linthresh=linthresh)
    axr.set_yticks([-1e-2, -1e-4, -linthresh, 0, linthresh, 1e-4, 1e-2])
    axr.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(ten_power_formatter))
    axr.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    axr.set(xlabel="x", ylabel="V / reference − 1", ylim=(-0.05, 0.05))
    axr.axhline(0, color=FG, lw=1)
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


def plot_effect_fluxes(
    v,
    tau_base,
    isolated: dict,
    *,
    window=None,
    skewer: int = 0,
    ncols: int = 4,
    figsize=None,
    color=None,
    base_label: str = "fake_spectra options",
):
    """One pair of panels per isolated switch: the flux with fake_spectra's options (thick, light) and with that one
    switch set to SkewerJAX's value (solid) on ``window`` [km/s], and below their difference. isolated : {label: tau}
    of ``run_effects(...)["isolated"]``, (N_sk, N_v) or (N_lines, N_sk, N_v); tau_base : the same shape. Returns (fig, axs)."""
    labels = list(isolated)
    n = len(labels)
    nrows = -(-n // ncols)
    fig, axs = plt.subplots(
        2 * nrows,
        ncols,
        figsize=figsize or (6 * ncols, 5.2 * nrows),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1] * nrows},
        layout="constrained",
        squeeze=False,
    )
    v = np.asarray(v)
    m = slice(None) if window is None else (v >= window[0]) & (v <= window[1])
    pick = lambda t: np.asarray(t)[0, skewer] if np.ndim(t) == 3 else np.asarray(t)[skewer]  # (N_lines, N_sk, N_v) or (N_sk, N_v)
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
        _textbox(
            axs[r + 1, q],
            0.02,
            0.85,
            f"max|ΔF| = {np.abs(d).max():.1e}\nΔ⟨F⟩ = {d.mean():+.1e}",
            transform=axs[r + 1, q].transAxes,
            va="top",
        )
        axs[r, q].set_ylim(-0.03, 1.05)
        if q == 0:
            axs[r, q].set_ylabel(LABELS["F"])
            axs[r + 1, q].set_ylabel(LABELS["dF"])
    for i in range(n, nrows * ncols):
        r, q = 2 * (i // ncols), i % ncols
        axs[r, q].set_axis_off()
        axs[r + 1, q].set_axis_off()
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
    return {
        "point": dict(marker="o", s=75, color=FG, edgecolors=BG, linewidths=1.2, zorder=9),
        "midpoint": dict(marker="D", s=60, color=PALETTE["yellow"], edgecolors=FG, linewidths=1.0, zorder=10),
        "dual_segment": dict(colors=(PALETTE["blue"], PALETTE["cyan"]), lw=10, alpha=0.6),
        "tangent": dict(color=PALETTE["red"], lw=2.4),
        "line_element": dict(color=PALETTE["purple"], lw=2.0),
        "piece_end": dict(marker="|", s=260, color=FG, linewidths=2.2, zorder=11),
        "generator": dict(marker="+", s=70, color=FG, linewidths=1.6, zorder=6),
    }[name]


def plot_skewer_geometry_3d(ax, points, *, scale_factors=None, label_w=None, tangent_length: float = None, highlight: int = None):
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
    ax.computed_zorder = False  # draw in the z-order below, not by depth
    ds = geometry_style("dual_segment")
    for i, (lo, x, hi) in enumerate(zip(g["mid_lo"], p, g["mid_hi"])):
        ax.plot(
            *np.stack([lo, x, hi]).T,
            color=ds["colors"][i % 2],
            lw=ds["lw"],
            alpha=ds["alpha"],
            solid_capstyle="butt",
            path_effects=[],
            zorder=2,
            label=r"dual segments $[m_{i-1/2}, m_{i+1/2}]$" if i == 0 else None,
        )
    ax.plot(*p.T, color=DIM, lw=1.0, path_effects=[], zorder=3)
    st = geometry_style("point")
    ax.scatter(*p.T, depthshade=False, label=r"sampling points $x_i$", **st)
    mids = 0.5 * (p[1:] + p[:-1])
    ax.scatter(*mids.T, depthshade=False, label=r"midpoints $m_{i\pm1/2}$", **geometry_style("midpoint"))
    L = tangent_length or 1.4 * np.median(g["line_elements"])
    tg = geometry_style("tangent")
    every = slice(0, None, 2)  # every other point: the arrows stand out of the bands
    ax.quiver(
        *p[every].T,
        *(L * g["tangents"][every]).T,
        color=tg["color"],
        lw=tg["lw"],
        arrow_length_ratio=0.35,
        zorder=7,
        label=r"unit tangents $\hat t_i$ (every other point)",
    )
    le = geometry_style("line_element")
    span = (p.max(0) - p.min(0)).max()
    for j, i in enumerate(np.linspace(0, n - 1, 6).round().astype(int) if label_w is None else label_w):
        lo, hi, t = g["mid_lo"][i], g["mid_hi"][i], g["tangents"][i]
        nrm = np.cross(t, [0.0, 0.0, 1.0])
        nrm = nrm if np.linalg.norm(nrm) > 0.3 else np.cross(t, [1.0, 0.0, 0.0])
        out = 0.05 * span * nrm / np.linalg.norm(nrm)  # a bracket beside the band, across the path
        br = np.stack([lo, lo + out, hi + out, hi])
        ax.plot(
            *br.T,
            color=le["color"],
            lw=le["lw"],
            path_effects=[],
            zorder=8,
            label=r"line elements $w_i$ (dual-segment lengths)" if j == 0 else None,
        )
        txt = f"  $w_{{{i}}}$ = {g['line_elements'][i]:.2f}"
        if scale_factors is not None:
            txt += f", a = {float(np.asarray(scale_factors)[i]):.4f}"
        ax.text(*(0.5 * (lo + hi) + 1.8 * out), txt, color=le["color"], fontsize=FONT_MIN["annotation"], zorder=12)
    if highlight is not None:
        k = int(highlight)
        ax.plot(*np.stack([g["mid_lo"][k], p[k], g["mid_hi"][k]]).T, color=FG, lw=2.2, ls=(0, (3, 2)), path_effects=[], zorder=8)
        ax.text(
            *g["mid_hi"][k],
            f"  dual segment of $x_{{{k}}}$:\n  $m_{{{k}-1/2}} \\to m_{{{k}+1/2}}$",
            color=FG,
            fontsize=FONT_MIN["annotation"],
            zorder=12,
        )
    lo, hi = p.min(0), p.max(0)
    pad = 0.08 * (hi - lo).max()
    ax.set(
        xlim=(lo[0] - pad, hi[0] + pad),
        ylim=(lo[1] - pad, hi[1] + pad),
        zlim=(lo[2] - pad, hi[2] + pad),
        xlabel="x [cMpc]",
        ylabel="y [cMpc]",
        zlabel="z [cMpc]",
    )
    dark_3d(ax)
    return None


# --------------------------------------------------------------------------- GRF recovery (notebook 04)
def _wrapped_offsets(n: int, dx: float) -> np.ndarray:
    """Signed minimum-image separations of a periodic axis of n cells: dx (j for j < n/2, j - n above)."""
    j = np.arange(n)
    return dx * np.where(j <= n // 2, j, j - n)


def correlation_binned(
    deltas, L, edges, grf=None, *, r_max_fraction: float = 0.25, cov_k_cut: float = 16.0, max_exact: float = 2e8
):
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
    take = [np.r_[0 : mi + 1, n - mi : n] for mi, n in zip(m, shape)]
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

        kk = [2 * np.pi * np.fft.fftfreq(n, h) for n, h in zip(shape[:-1], dx[:-1])] + [
            2 * np.pi * np.fft.rfftfreq(shape[-1], dx[-1])
        ]
        kmod = np.sqrt(sum(np.meshgrid(*[k**2 for k in kk], indexing="ij")))
        Pk = np.asarray(grf.power_at(*np.meshgrid(*kk, indexing="ij", sparse=True)))  # P(k, mu) of every mode
        xi_in = np.fft.irfftn(Pk, s=shape, axes=axes)[sub] / np.prod(dx)
        xin = np.bincount(j[ok], weights=xi_in.ravel()[ok], minlength=nb) / np.maximum(cnt, 1)
        # c_b(k) = Re FFT of the bin's indicator over the separations / n_b; w: half-space multiplicity
        last = kk[-1][(None,) * (len(shape) - 1) + (slice(None),)]
        w = np.where((last == 0) | (np.isclose(last, np.pi / dx[-1]) & (shape[-1] % 2 == 0)), 1.0, 2.0)
        wP2 = np.broadcast_to(w * Pk**2, kmod.shape)
        exact = kmod.size * nb <= max_exact
        low = np.ones(kmod.shape, bool) if exact else (kmod > 0) & (kmod <= cov_k_cut * 2 * np.pi / Ls.max() * 1.0001)
        jj = np.full(r.shape, -1)
        jj.ravel()[ok] = j[ok]
        C_low, diag = np.zeros((nb, int(low.sum()))), np.zeros(nb)
        for b in np.nonzero(keep)[0]:
            ind = np.zeros(shape, np.float32)
            ind[sub] = jj == b
            c = sfft.rfftn(ind, workers=-1).real.astype(np.float64) / cnt[b]
            diag[b] = np.sum(wP2 * c**2)
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
    return float(np.sum(y**2 / lam[good]) / good.sum()), int(good.sum())


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
    fig, axs = plt.subplots(
        4, n, figsize=figsize, gridspec_kw={"height_ratios": [2.0, 1.0, 2.0, 1.0]}, layout="constrained", squeeze=False
    )
    out = {}
    col = ENTITY_COLORS["density"]
    for c, (title, (grf, deltas)) in enumerate(cases.items()):
        m = field_power_modes(deltas, grf.L)
        kmin = 2 * np.pi / max(grf.L)
        kmax = min(np.pi * N / L for N, L in zip(grf.shape, grf.L))
        edges = (
            hybrid_k_edges(kmin, kmax, n_lin=n_lin, n_log=n_bins) if n_lin > 0 else np.geomspace(0.999 * kmin, kmax, n_bins + 1)
        )
        kb, Pb, eb, _ = _bin_modes(m["k"], m["P"], m["w"], edges)
        Pin_b = _bin_modes(m["k"], input_power_modes(grf, m), m["w"], edges)[1]
        kk = np.geomspace(kb[0], kb[-1], 600)
        guide = (
            "input at every k, averaged over directions"
            if getattr(grf, "anisotropic", False) and grf.ndim > 1
            else "input at every k"
        )
        axs[0, c].plot(kk, np.asarray(grf.power_monopole(kk)), color=FG, **MODE_STYLES["reference"], label=guide)
        axs[0, c].plot(kb, Pin_b, color=FG, lw=1.6, marker="_", ms=14, mew=2, label="input, averaged over the modes of each bin")
        axs[0, c].errorbar(kb, Pb, eb, fmt="o", ms=5, capsize=3, color=col, label=f"measured ({deltas.shape[0]} draws)", zorder=5)
        axs[1, c].errorbar(kb, Pb / Pin_b - 1, eb / Pin_b, fmt="o", ms=5, capsize=3, color=col)
        lim = max(0.1, 1.2 * float(np.max(np.abs(Pb / Pin_b - 1) + eb / Pin_b)))  # every point and its error bar
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
        lead = float(np.linalg.eigvalsh(corr).max() / corr.shape[0])  # variance in the common (leading) mode
        axs[3, c].set(xlabel=r"$|r|$ [Mpc]", ylabel=r"$(\hat\xi - \xi_{\rm in})/\sigma_\xi$", ylim=(-5, 5))
        axs[3, c].set_title(
            f"common mode: {lead:.0%} of the variance; " + rf"$\chi^2/\mathrm{{dof}}$ (cov.) = {chi2c:.2f}",
            fontsize=FONT_MIN["legend"],
        )
        axs[2, c].legend(fontsize=FONT_MIN["legend"])
        out[title] = dict(
            chi2_dof=float(np.mean(((Pb - Pin_b) / eb) ** 2)),
            max_rel=float(np.max(np.abs(Pb / Pin_b - 1))),
            n_kbins=int(kb.size),
            xi_chi2_dof=float(np.mean(((xb - xin) / xe) ** 2)),
            xi_chi2_cov=chi2c,
            xi_dof=dof,
            xi_lead=lead,
            xi_mean_pull=float(np.mean((xb - xin) / xe)),
        )
    return fig, axs, out


def power_kmu(m, k_edges, mu_edges, los_axis: int = -1):
    """P in bins of (|k|, mu = |k_par| / |k|), k_par along ``los_axis``: mean over draws and modes.

    m : ``field_power_modes`` output. Returns (k_b (n_mu, n_k), P_b, err (scatter between draws / sqrt(N_real), or the
    Gaussian P / sqrt(n) for one draw), n_indep (half-space modes))."""
    k, kv, w = m["k"], m["kvec"], m["w"]
    mu = np.abs(kv[:, los_axis]) / np.maximum(k, 1e-300)
    ik = np.digitize(k, k_edges) - 1
    im = np.clip(np.digitize(mu, mu_edges) - 1, 0, len(mu_edges) - 2)  # mu = 1 into the last wedge
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
    out[-1] = edges[-1]  # trailing empty bins join the last occupied one
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
        ep = _occupied_edges(np.abs(kv[:, los]), e)  # k_par: the modes along the line of sight
        eq = _occupied_edges(np.linalg.norm(np.delete(kv, los, axis=1), axis=-1), e)  # k_perp
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
        ax.contour(kcp, kcq, np.ma.masked_invalid(Pm).T, levels=levels, colors=PALETTE["red"], linewidths=1.6, linestyles="-")
        ax.clabel(ci, fmt=lambda x: f"$10^{{{np.log10(x):.0f}}}$", fontsize=FONT_MIN["floor"], inline=False)
        ax.plot([], [], color=FG, ls="--", label="input iso-power contours (binned alike)")
        ax.plot([], [], color=PALETTE["red"], ls="-", label="measured contours, same levels")
        ax.legend(loc="lower left", fontsize=FONT_MIN["legend"])
        fig.colorbar(im, ax=ax, label=r"measured $P$ [Mpc$^{d}$]", shrink=0.9)
        sig = Pin / np.sqrt(np.maximum(Nind * deltas.shape[0], 1))
        pull = np.where(good & (Nind > 0), (Pm - Pin) / sig, np.nan)
        chi2 = float(np.nanmean(pull**2))
        out[title] = chi2
        ax.set(
            xscale="log",
            yscale="log",
            xlim=(e[0], e[-1]),
            ylim=(e[0], e[-1]),
            aspect="equal",
            title=title,
            xlabel=r"$k_\parallel$ ("
            + ("last axis" if grf.los_axis == grf.ndim - 1 else f"axis {grf.los_axis}")
            + r") [Mpc$^{-1}$]",
            ylabel=r"$k_\perp$ [Mpc$^{-1}$]",
        )
    return fig, axs[0], out


MU_WEDGES = (0.0, 0.5, 0.8, 0.95, 1.0)  # mu = k_par / |k|: the four wedges of Lya 3D analyses (arXiv:2308.10950, Fig. 4)


MU_COLORS = ("blue", "cyan", "green", "red")  # one colour per wedge, in every panel (PALETTE keys)


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
    fig, axs = plt.subplots(
        2, n, figsize=figsize, layout="constrained", squeeze=False, sharex="col", gridspec_kw={"height_ratios": [2.0, 1.0]}
    )
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
        shift = np.exp(0.025 * (np.arange(nm) - (nm - 1) / 2))  # a small offset in k per wedge, so every point shows
        for w in range(nm):
            ok = np.isfinite(Pb[w]) & np.isfinite(eb[w]) & (eb[w] > 0)
            chi2 = float(np.mean(((Pb[w][ok] - Pin[w][ok]) / eb[w][ok]) ** 2))
            out[title].append(chi2)
            lab = rf"${mu_edges[w]:g} \leq \mu < {mu_edges[w + 1]:g}$ ($\chi^2/\mathrm{{dof}}$ = {chi2:.2f})"
            x = kb[w][ok] * shift[w]
            a0.plot(x, Pin[w][ok], color=cols[w], lw=1.8, ls="--", marker="_", ms=12, mew=1.8)
            a0.errorbar(x, Pb[w][ok], eb[w][ok], fmt="o", ms=5, capsize=3, color=cols[w], label=lab)
            a1.errorbar(x, Pb[w][ok] / Pin[w][ok] - 1, eb[w][ok] / Pin[w][ok], fmt="o-", ms=5, capsize=3, lw=1.2, color=cols[w])
        a0.plot(
            [],
            [],
            color=FG,
            ls="--",
            lw=1.8,
            marker="_",
            ms=12,
            label="input, averaged over the modes of each bin (one per wedge)",
        )
        residual_band(a1, 0.05)
        a0.set(xscale="log", yscale="log", title=title, ylabel=r"$P(k, \mu)$")
        a1.set(xscale="log", ylim=(-0.3, 0.3), xlabel=r"$|k|$ [Mpc$^{-1}$]", ylabel="measured / input − 1")
        a0.legend(fontsize=FONT_MIN["legend"])
    return fig, axs, out


# --------------------------------------------------------------------------- selection -> deposition -> tau (notebook 07)
def _path_at(points, s_pts, s):
    """Positions (K, 3) at path lengths ``s`` (K,) along a polyline ``points`` (N, 3) with path lengths ``s_pts``."""
    return np.stack([np.interp(s, s_pts, points[:, k]) for k in range(3)], -1)


def _skewer_cells(parts, regions, species: str) -> dict:
    """The pieces of one skewer and its cells as the deposition figures (``plot_deposition_cells``, ...) draw them.

    parts : the parts of ``core.cell_pieces(..., return_parts=True)``; regions : ((s0, s1), (s2, s3)) [cMpc] two
    stretches of the path. Per piece (the absorbers of ``species``, skewer 0): live, s, l, cell, n, v_los, the Hubble
    velocity v_H of its centroid and its line centre v_c [km/s] (NaN where not live); cols: the live pieces of each
    stretch, in path order; labels {cell: "S1", "S2", ... for the first stretch, "E1", ... for the second} (a cell met in
    both keeps its first); cells: the labelled cells; cmap, norm: log10 density of the labelled cells (their range only),
    on the whole of ``CMAPS["density"]`` (as ``plot_gas_cells``); colour(c): the colour of any cell (clipped to the norm).
    """
    from skewerjax.core.constants import phys

    ab, use, frame = parts["absorbers"][species], parts["cells"], parts["frame"]
    live = np.asarray(ab.n[0]) > 0
    s_all, l_all, c_all, n_all, u_all = (np.asarray(getattr(ab, f))[0] for f in ("s", "length", "cell", "n", "v_los"))
    s_ok = np.where(live, s_all, 0.0)
    v_H = np.where(live, phys.c_kms * np.log(frame.a_dagger / frame.scale_factors(0, s_ok)), np.nan)
    beta = -frame.velocity(0, s_ok, u_all) / phys.c_kms
    v_c = v_H + 0.5 * phys.c_kms * (np.log1p(-beta) - np.log1p(beta))
    labels, n_lab, cols = {}, {"S": 0, "E": 0}, []
    for k, (lo, hi) in enumerate(regions):
        idx = np.nonzero(live & (s_all >= lo) & (s_all < hi))[0]
        idx = idx[np.argsort(s_all[idx])]
        cols.append(idx)
        for c in c_all[idx]:
            if int(c) not in labels:
                key = "SE"[k]
                n_lab[key] += 1
                labels[int(c)] = f"{key}{n_lab[key]}"
    cells_all = np.array(list(labels))
    cmap = mpl.colormaps[CMAPS["density"]]
    dens = np.asarray(use.species[species]["number_density"], dtype=float)
    log_n = np.log10(dens[cells_all])
    norm = _density_norm(log_n)
    col = {int(c): cmap(norm(x)) for c, x in zip(cells_all, log_n)}

    def colour(c):
        c = int(c)
        if c not in col:
            col[c] = cmap(norm(np.clip(np.log10(dens[c]), norm.vmin, norm.vmax)))
        return col[c]

    return dict(
        live=live,
        s=s_all,
        l=l_all,
        cell=c_all,
        n=n_all,
        v_los=u_all,
        v_H=v_H,
        v_c=v_c,
        cols=cols,
        labels=labels,
        cells=cells_all,
        cmap=cmap,
        norm=norm,
        colour=colour,
    )


def _piece_labels(R) -> dict:
    """{piece index: "E3a"}: per labelled piece of ``_skewer_cells``, its cell's label and one letter per piece of that
    cell, in path order."""
    import string

    idx = np.concatenate(R["cols"])
    count, out = {}, {}
    for i in idx[np.argsort(R["s"][idx])]:
        c = int(R["cell"][i])
        k = count.get(c, 0)
        count[c] = k + 1
        out[int(i)] = R["labels"][c] + (string.ascii_lowercase[k] if k < 26 else str(k))
    return out


def _arrow_levels(x0, x1, y, *, pad: float, dy: float, boxes=()) -> np.ndarray:
    """Greedy stacking of horizontal arrows spanning [x0, x1] at base heights ``y``: each arrow, in order of its left end,
    takes the lowest level k (drawn at y + k dy) that is at least ``dy`` away in height from every arrow already placed
    whose span, widened by ``pad``, overlaps its own, and at least ``dy / 2`` outside every fixed box (x0, x1, y0, y1)
    of ``boxes`` (e.g. labels) that it overlaps. Returns the level of every arrow."""
    lo, hi = np.minimum(x0, x1), np.maximum(x0, x1)
    y = np.asarray(y, dtype=float)
    lev = np.zeros(lo.size, dtype=int)
    placed = []
    for i in np.argsort(lo):
        k = 0
        while any(
            lo[i] < hi[j] + pad and lo[j] < hi[i] + pad and abs(y[i] + k * dy - y[j] - lev[j] * dy) < dy - 1e-12 for j in placed
        ) or any(
            lo[i] < bx1 + pad and bx0 < hi[i] + pad and by0 - 0.5 * dy < y[i] + k * dy < by1 + 0.5 * dy
            for bx0, bx1, by0, by1 in boxes
        ):
            k += 1
        lev[i] = k
        placed.append(i)
    return lev


def _label_fit(w_in: float, h_in: float, txt: str, fontsize: float):
    """How a label fits in a box of ``w_in`` x ``h_in`` inches: "h" (horizontal), "v" (rotated by 90 degrees) or None
    (it does not fit), with the label's estimated width and height [in]."""
    tw, th = 0.62 * fontsize / 72.0 * len(txt) + 0.06, 1.05 * fontsize / 72.0 + 0.04
    how = "h" if (w_in >= tw and h_in >= th) else ("v" if (w_in >= th and h_in >= tw) else None)
    return how, tw, th


class _RingSector:
    """A sector of a ring on Cartesian axes (equal aspect): angle th [rad] clockwise from 12 o'clock, the middle of the
    window [th_lo, th_hi] at 12 o'clock, radius r in [r_in, r_out] drawn at the display radius
    ``depth`` (r_out - r_in) + (r - r_in). ``aspect``: the width / height of the band, which sets the depth (a narrow
    window gets a larger depth, so that every window fills an axis of the same shape)."""

    def __init__(self, th_lo, th_hi, r_in, r_out, aspect=3.4):
        self.lo, self.hi, self.r_in, self.r_out = th_lo, th_hi, r_in, r_out
        self.mid = 0.5 * (th_lo + th_hi)
        sn, cs = np.sin(0.5 * (th_hi - th_lo)), np.cos(0.5 * (th_hi - th_lo))
        depth = (aspect - 2 * sn) / max(2 * sn - aspect * (1 - cs), 1e-3)  # 2 (d + 1) sin / (d + 1 - d cos) = aspect
        self.k = max(depth, 1.0) * (r_out - r_in)

    def xy(self, th, r):
        rho = self.k + (np.asarray(r, dtype=float) - self.r_in)
        phi = np.asarray(th, dtype=float) - self.mid
        return rho * np.sin(phi), rho * np.cos(phi)

    def frame(self, ax, r_ring):
        """The band's outline, the baseline ``r_ring`` (grey), equal aspect, no Cartesian axes."""
        th = np.linspace(self.lo, self.hi, 300)
        for r in (self.r_in, self.r_out):
            ax.plot(*self.xy(th, np.full_like(th, r)), color=FG, lw=1.0)
        for t in (self.lo, self.hi):
            ax.plot(*self.xy([t, t], [self.r_in, self.r_out]), color=FG, lw=1.0)
        ax.plot(*self.xy(th, np.full_like(th, r_ring)), color=DIM, lw=0.8)
        x, y = self.xy(np.r_[th, th[::-1]], np.r_[np.full_like(th, self.r_in), np.full_like(th, self.r_out)])
        ax.set_aspect("equal")
        ax.axis("off")
        pad = 0.02 * (x.max() - x.min())
        ax.set(xlim=(x.min() - 4.0 * pad, x.max() + 2.5 * pad), ylim=(y.min() - 0.2 * pad, y.max() + 3.2 * pad))

    def inch(self, ax):
        """Inches per data unit of ``ax`` (equal aspect, the box shrunk to fit its limits)."""
        fig = ax.get_figure()
        pos = ax.get_position()
        w, h = pos.width * fig.get_figwidth(), pos.height * fig.get_figheight()
        return min(w / np.ptp(ax.get_xlim()), h / np.ptp(ax.get_ylim()))

    def wedge(self, ax, t0, t1, r0, r1, **kw):
        th = np.linspace(t0, t1, max(int(abs(t1 - t0) / 0.002) + 2, 3))
        x, y = self.xy(np.r_[th, th[::-1]], np.r_[np.full_like(th, r0), np.full_like(th, r1)])
        ax.fill(x, y, **kw)

    def band(self, ax, th, r0, r1, **kw):
        x, y = self.xy(np.r_[th, th[::-1]], np.r_[np.broadcast_to(r0, th.shape), np.asarray(r1)[::-1]])
        ax.fill(x, y, **kw)

    def line(self, ax, th, r, **kw):
        ax.plot(*self.xy(th, r), **kw)

    def ray(self, ax, th, r0=None, r1=None, **kw):
        ax.plot(*self.xy([th, th], [self.r_in if r0 is None else r0, self.r_out if r1 is None else r1]), **kw)

    def arrow(self, ax, t0, t1, r, *, color):
        """An arrow along the ring, from angle t0 to t1 at radius r (an arc, with its head at t1)."""
        from matplotlib import patheffects as pe

        th = np.linspace(t0, t1, 12)
        stroke = [pe.withStroke(linewidth=3.6, foreground=FG)]
        ax.plot(
            *self.xy(th[:-1], np.full(th.size - 1, r)), color=color, lw=2.2, solid_capstyle="butt", path_effects=stroke, zorder=6
        )
        ax.annotate(
            "",
            self.xy(t1, r),
            self.xy(th[-2], r),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=2.2, mutation_scale=13, shrinkA=0, shrinkB=0, path_effects=stroke),
            zorder=6,
        )

    def piece_labels(self, ax, items, *, fontsize=FONT_MIN["floor"], placed=None) -> dict:
        """Labels without leader lines: each inside its element (along the arc, or radial when the element is narrow),
        else radial just outside it. ``items``: (th, dth, r0, r1, text, bold) per element (angle and angular width [rad],
        radial extent), in order of priority; an element thinner than a line or outside the window gets no label, and a
        label that would overlap one already drawn (``placed``: data boxes, updated) or leave the band is skipped.
        Returns {text: (th, half angular width, r0, r1)} of the labels drawn."""
        inch = self.inch(ax)
        placed = [] if placed is None else placed
        out = {}
        for th, dth, r0, r1, txt, bold in items:
            if not self.lo <= th <= self.hi:
                continue
            w_in = (self.k + 0.5 * (r0 + r1) - self.r_in) * abs(dth) * inch
            how, tw, th_ = _label_fit(w_in, (r1 - r0) * inch, txt, fontsize)
            if how is None and w_in < 0.02 and r1 - r0 > 0:
                continue
            arc, rad = (tw, th_) if how == "h" else (th_, tw)  # the label's extent along the arc and the radius [in]
            rm = 0.5 * (r0 + r1) if how else r1 + 0.03 + 0.5 * rad / inch
            if rm + 0.5 * rad / inch > self.r_out:
                continue
            half = 0.5 * arc / inch / (self.k + rm - self.r_in)
            x, y = self.xy(
                np.array([th - half, th + half, th + half, th - half]), rm + 0.5 * rad / inch * np.array([-1, -1, 1, 1])
            )
            bx = (x.min(), x.max(), y.min(), y.max())
            if any(bx[0] < q[1] and q[0] < bx[1] and bx[2] < q[3] and q[2] < bx[3] for q in placed):
                continue
            placed.append(bx)
            ax.text(
                *self.xy(th, rm),
                txt,
                rotation=-np.degrees(th - self.mid) + (0.0 if how == "h" else 90.0),
                rotation_mode="anchor",
                ha="center",
                va="center",
                fontsize=fontsize,
                color=FG,
                zorder=12,
                path_effects=_halo(),
                fontweight="bold" if bold else "normal",
            )
            out[txt] = (th, half, rm - 0.5 * rad / inch, rm + 0.5 * rad / inch)
        return out

    def ticks(self, ax, per, step, unit):
        """Angular ticks every ``step`` of the periodic coordinate (period ``per``), labelled in its units, outside; the
        seam 0 = per in bold, values before it as per - x."""
        lo_, hi_ = self.lo * per / (2 * np.pi), self.hi * per / (2 * np.pi)
        fmt = (lambda x: f"{x:.2f}") if unit == "cMpc" else (lambda x: f"{x:.0f}")
        xs = np.arange(np.ceil(lo_ / step) * step, hi_ + 1e-9, step)
        for x in xs:
            th = 2 * np.pi * x / per
            ax.plot(*self.xy([th, th], [self.r_out, self.r_out + 0.05]), color=FG, lw=1.2)
            seam = abs(x) < 1e-9
            txt = f"0 ≡ {fmt(per)}" if seam else (fmt(x + per) if x < 0 else f"{x:g}")
            ax.text(
                *self.xy(th, self.r_out + 0.08),
                txt + (f" {unit}" if x == xs[-1] else ""),
                rotation=-np.degrees(th - self.mid),
                rotation_mode="anchor",
                ha="center",
                va="bottom",
                fontsize=FONT_MIN["tick"],
                color=FG,
                fontweight="bold" if seam else "normal",
            )

    def rticks(self, ax, r, labs, title):
        """Radial ticks and their title along the left edge of the sector."""
        d = 1.0 / (self.inch(ax) * (self.k + 0.5 * (self.r_out - self.r_in)))  # one inch along the arc [rad]
        for rr, lab in zip(r, labs):
            ax.plot(*self.xy([self.lo, self.lo - 0.06 * d], [rr, rr]), color=FG, lw=1.2)
            ax.text(
                *self.xy(self.lo - 0.09 * d, rr),
                lab,
                ha="right",
                va="center",
                fontsize=FONT_MIN["tick"],
                color=FG,
                rotation=-np.degrees(self.lo - self.mid),
                rotation_mode="anchor",
            )
        x, y = self.xy(self.lo - 0.75 * d, 0.5 * (min(r) + max(r)))
        ax.text(
            x,
            y,
            title,
            ha="right",
            va="center",
            rotation=90.0 - np.degrees(self.lo - self.mid),
            fontsize=FONT_MIN["label"],
            color=FG,
        )


def _full_ring(ins, ring, per: float, unit: str, draw, title: str):
    """The whole ring on its own axes ``ins`` (unit circle = the ring): the window of ``ring`` (grey wedge), the seam at
    12 o'clock with its value, the opposite point; ``draw(ins)`` adds the content; ``title`` below."""
    ins.set_aspect("equal", anchor="C")
    ins.axis("off")
    th = np.linspace(0, 2 * np.pi, 361)
    ins.plot(np.sin(th), np.cos(th), color=DIM, lw=1.2)
    tw_ = np.linspace(ring.lo, ring.hi, 40)
    ins.fill(np.r_[0.0, 1.6 * np.sin(tw_), 0.0], np.r_[0.0, 1.6 * np.cos(tw_), 0.0], color=PALETTE["grey"], alpha=0.4, lw=0)
    draw(ins)
    fmt = (lambda x: f"{x:.2f}") if unit == "cMpc" else (lambda x: f"{x:.0f}")
    ins.text(0, 1.8, f"0 ≡ {fmt(per)}", ha="center", va="bottom", fontsize=FONT_MIN["annotation"], color=FG, fontweight="bold")
    ins.text(0, -1.78, f"{fmt(0.5 * per)} {unit}", ha="center", va="top", fontsize=FONT_MIN["annotation"], color=FG)
    ins.text(0, -2.45, title, ha="center", va="top", fontsize=FONT_MIN["annotation"], color=FG, style="italic")
    ins.set(xlim=(-1.75, 1.75), ylim=(-2.95, 2.4))
    return ins


def _deposition_view(parts, bundle, half_range_cMpc: float, species: str, period=None) -> dict:
    """What the three deposition figures of notebook 07 share: the pieces of one skewer whose centroids lie within
    ``half_range_cMpc`` of its seam s = 0 = L (``_skewer_cells`` with the stretches [0, l) and [L - l, L)), their labels,
    colours, Hubble widths W_p [km/s], the piece on the velocity seam and the common info of the figures."""
    from skewerjax.core.constants import phys
    from skewerjax.core.skewers import path_geometry

    frame = parts["frame"]
    pts = np.asarray(bundle.spatial_points[0], dtype=float)
    geo = path_geometry(pts)
    s_pts, tan_pts = np.asarray(geo["s"]), np.asarray(geo["tangents"])
    L, hr = float(s_pts[-1]), float(half_range_cMpc)
    R = _skewer_cells(parts, ((0.0, hr), (L - hr, L)), species)
    P = (
        float(period)
        if period is not None
        else float(phys.c_kms * np.log(frame.a_dagger / frame.scale_factors(0, np.array([L]))[0]))
    )
    live, s_all, l_all = R["live"], R["s"], R["l"]
    W = np.where(live, frame.hubble_rate(0, np.where(live, s_all, 0.0)) * l_all, 0.0)  # W_p, as absorber_pieces
    plab = _piece_labels(R)
    shown = np.array(sorted(plab, key=lambda i: s_all[i]), dtype=int)
    wrap = lambda x, per: np.mod(np.asarray(x) + 0.5 * per, per) - 0.5 * per
    dist = np.abs(wrap(R["v_c"][shown], P)) - 0.5 * W[shown]  # the piece on the velocity seam: [v_c -+ W/2] holds 0 = P
    seam = int(shown[np.argmin(dist)])
    head = species.rstrip("IVX")
    n_sh = R["n"][shown]
    info = dict(
        P=P,
        L=L,
        half_range_cMpc=hr,
        piece_labels={plab[i]: int(i) for i in shown},
        n_pieces={},
        n_cells={},
        labels={},
        seam_piece=plab[seam],
        seam_v_c=float(wrap(R["v_c"][seam], P)),
        seam_W=float(W[seam]),
    )
    for k, side in enumerate(("start", "end")):
        cells = R["cell"][R["cols"][k]]
        _, first = np.unique(cells, return_index=True)
        info["n_pieces"][side], info["n_cells"][side] = int(cells.size), int(first.size)
        info["labels"][side] = [R["labels"][int(c)] for c in cells[np.sort(first)]]
    return dict(
        R=R,
        L=L,
        P=P,
        hr=hr,
        pts=pts,
        s_pts=s_pts,
        tan_pts=tan_pts,
        W=W,
        plab=plab,
        shown=shown,
        seam=seam,
        wrap=wrap,
        name=f"{head} {species[len(head) :]}" if head != species else species,
        info=info,
        floor=10 ** (np.log10(n_sh.min()) - 0.35),
        top=10 ** (np.log10(n_sh.max()) + 0.15),
    )


def _deposition_arrows(D, cells, species: str, arrow_kms=None, arrow_cMpc=None, *, parallel: bool = False):
    """The arrow scale [cMpc per km/s] and the reference speed [km/s] of the deposition figures, from what their arrows
    show (``velocity_arrow_scale``, ``velocity_key_speed``): the vectors v of the cells of the pieces shown (3D, figure
    (a)), or the components v_par = v . t of those pieces (``parallel``: along s, figures (b) and (c), which so share
    one scale). ``arrow_cMpc`` / ``arrow_kms`` when both are given (the override)."""
    if (arrow_kms is None) != (arrow_cMpc is None):
        raise ValueError("give both arrow_kms and arrow_cMpc (an arrow of arrow_cMpc cMpc per arrow_kms km/s), or neither.")
    R, shown = D["R"], D["shown"]
    cell = np.unique(R["cell"][shown])
    vel = R["v_los"][shown] if parallel else np.asarray(cells.species[species]["velocity"], dtype=float)[cell]
    size = np.asarray(cells.size_cMpc, dtype=float)[R["cell"][shown] if parallel else cell]
    over = None if arrow_kms is None else float(arrow_cMpc) / float(arrow_kms)
    return velocity_arrow_scale(vel, size, span=D["hr"], scale=over), velocity_key_speed(vel)  # each panel: l of path


def _velocity_arrow_handle(
    key_kms: float, scale: float, what: str = r"$v_\parallel = \mathbf{v}\cdot\hat t$ (colour of its piece)", extra: str = ""
):
    from matplotlib import patheffects as pe
    from matplotlib.lines import Line2D

    return Line2D(
        [],
        [],
        color=PALETTE["grey"],
        lw=2.5,
        marker=">",
        markevery=[1],
        markersize=9,
        markeredgecolor=FG,
        path_effects=[pe.Stroke(linewidth=4.0, foreground=FG), pe.Normal()],
        label=rf"{what}; key: {key_kms:g} km/s{extra}",
    )


def _halo(width: float = 3.0):
    """A thin outline in the background colour around a text: legible over any fill colour."""
    from matplotlib import patheffects as pe

    return [pe.withStroke(linewidth=width, foreground=BG)]


def plot_deposition_cells(
    parts,
    bundle,
    *,
    half_range_cMpc: float = 2.0,
    species: str = "HI",
    arrow_kms=None,
    arrow_cMpc=None,
    view=(20.0, -76.0),
    zoom: float = 1.3,
    figsize=(22, 7.5),
):
    """Row (a) of the deposition figures of notebook 07: the cells of one simulation skewer around its seam, in 3D.

    parts : the parts of ``core.cell_pieces(cells, lines, skewer_bundle=bundle, return_parts=True)``; bundle : the
    ``SkewerBundle`` (one straight skewer through a periodic box, from one face to the opposite one); half_range_cMpc :
    l [cMpc], the cells of the pieces with centroids in s in [-l, l) around the seam s = 0 = L; arrow_kms, arrow_cMpc :
    an arrow of ``arrow_cMpc`` per ``arrow_kms`` (both or neither); default ``velocity_arrow_scale`` of the cells shown
    (their 90th-percentile speed 0.8 median cell sizes long); ``plot_deposition_pieces`` and ``plot_deposition_rings``
    share one scale of their own, from the components v_par that they draw. Every arrow stays inside its panel (an arrow still too long after the limits grow a
    little is cut at the edge and ends with an open circle); each panel holds a reference arrow of a round speed
    (``_velocity_key_3d``: lengths are shown in projection, and the two panels' boxes differ in size). ``info`` holds
    ``arrow_scale`` [cMpc per km/s], ``key_kms`` and ``n_capped``.

    (a1) the end of the box (s in [L - l, L)), (a2) its start (s in [0, l)): the cells in the colour of their density of
    ``species`` (one label on each: E1, ... at the end, S1, ... at the start), their peculiar velocities (arrows from
    the generating points, in the colour of their cell), the skewer and its unit tangents t (red), the box face where
    the path ends or starts (green plane). Returns (fig, (ax_end, ax_start), info).
    """
    from matplotlib.lines import Line2D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    D = _deposition_view(parts, bundle, half_range_cMpc, species)
    R, use, box = D["R"], parts["cells"], parts["cells"].box
    pts, s_pts, tan_pts, L = D["pts"], D["s_pts"], D["tan_pts"], D["L"]
    green, red = PALETTE["green"], PALETTE["red"]
    kpc, v_key = _deposition_arrows(D, use, species, arrow_kms, arrow_cMpc)  # cMpc of arrow per km/s, key [km/s]
    h_med = float(np.median(np.asarray(use.size_cMpc)[R["cell"][D["shown"]]]))
    n_capped, keys = 0, []
    fig = plt.figure(figsize=figsize)
    axs = [fig.add_axes([-0.085 + 0.495 * k, 0.18, 0.6, 0.73], projection="3d", computed_zorder=False) for k in range(2)]
    for kp, k in enumerate((1, 0)):  # panel 0: the end (stretch 1), panel 1: the start
        idx = R["cols"][k]
        s_p, l_p, c_p = R["s"][idx], R["l"][idx], R["cell"][idx]
        cells, first = np.unique(c_p, return_index=True)
        cells, first = cells[np.argsort(first)], np.sort(first)
        a3 = axs[kp]
        pos = np.asarray(use.positions, dtype=float)[cells]
        near = _path_at(pts, s_pts, s_p[first])
        g = pos + box * np.round((near - pos) / box) if box is not None else pos
        faces = _cell_faces(use, g, np.asarray(use.size_cMpc, dtype=float)[cells])
        vel = np.asarray(use.species[species]["velocity"], dtype=float)[cells]
        s_lo, s_hi = s_p[0] - 0.5 * l_p[0], s_p[-1] + 0.5 * l_p[-1]
        seg_pts = _path_at(pts, s_pts, np.linspace(s_lo, s_hi, 60))
        x_face = float(
            _path_at(pts, s_pts, np.array([0.0 if k == 0 else L]))[0, 0]
        )  # the box face of this end (a plane x = const)
        allv = np.concatenate([np.concatenate(f) for f in faces] + [seg_pts])
        lo, hi = allv.min(0), allv.max(0)
        lo[0], hi[0] = min(lo[0], x_face - 0.1), max(hi[0], x_face + 0.1)
        arrows, capped, lo, hi = _fit_arrows(g, vel * kpc, lo, hi)  # every arrow inside the panel
        n_capped += int(np.count_nonzero(capped))
        _draw_cells_3d(
            a3,
            faces,
            [R["colour"](c) for c in cells],
            g,
            seg_pts,
            lo,
            hi,
            arrows=arrows,
            view=view,
            zoom=zoom,
            path_color=red,
            arrow_color=[R["colour"](c) for c in cells],
        )
        _mark_capped(a3, g + arrows, capped)
        keys.append(
            _velocity_key_3d(
                a3, lo, hi, kpc, v_key, view, np.concatenate([g, seg_pts] + [np.concatenate(f) for f in faces]), note=""
            )
        )
        quad = np.array([[x_face, lo[1], lo[2]], [x_face, hi[1], lo[2]], [x_face, hi[1], hi[2]], [x_face, lo[1], hi[2]]])
        a3.add_collection3d(Poly3DCollection([quad], facecolors=[mpl.colors.to_rgba(green, 0.22)], edgecolors="none", zorder=0))
        a3.plot(*quad[[0, 1, 2, 3, 0]].T, color=green, lw=4.0, zorder=14)  # the outline of the box face, on top
        t_s = np.linspace(s_lo + 0.15 * (s_hi - s_lo), s_hi - 0.25 * (s_hi - s_lo), 3)
        t0 = _path_at(pts, s_pts, t_s)
        t_dir = tan_pts[np.clip(np.searchsorted(s_pts, t_s), 0, len(s_pts) - 1)]
        a3.quiver(*t0.T, *(0.25 * t_dir).T, color=red, lw=3.4, arrow_length_ratio=0.35, zorder=15)
        a3.text(
            *(t0[1] + 0.12 * t_dir[1] + np.array([0.0, 0.0, -0.07])),
            r"$\hat t$",
            color=red,
            fontsize=FONT_MIN["label"] + 2,
            va="top",
            zorder=20,
            bbox=dict(boxstyle="round,pad=0.1", facecolor=BG, edgecolor="none", alpha=0.8),
        )
        for c, gg in zip(cells, g):  # each label on its cell, above its generating point
            a3.text(
                gg[0],
                gg[1],
                gg[2] + 0.12 * h_med,
                R["labels"][int(c)],
                fontsize=FONT_MIN["annotation"],
                color=FG,
                fontweight="bold",
                ha="center",
                va="bottom",
                zorder=20,
                path_effects=_halo(),
            )
        for a_ in (a3.xaxis, a3.yaxis, a3.zaxis):  # axes, ticks and labels only: no panes, no grid
            a_.pane.fill = False
            a_.pane.set_edgecolor("none")
        a3.grid(False)
        a3.xaxis.set_major_locator(mpl.ticker.MaxNLocator(3))
        for a_ in (a3.yaxis, a3.zaxis):
            a_.set_major_locator(mpl.ticker.MaxNLocator(2))
            a_.set_major_formatter(mpl.ticker.FormatStrFormatter("%.1f"))
        for a_ in (a3.xaxis, a3.yaxis, a3.zaxis):
            a_.set_minor_locator(mpl.ticker.NullLocator())
        a3.xaxis.labelpad, a3.yaxis.labelpad, a3.zaxis.labelpad = 28, 22, 16
        a3.tick_params(axis="y", pad=6)
        a3.tick_params(axis="z", pad=8)
        a3.set_title(f"{('start', 'end')[k]} of the box: {cells.size} cells", pad=-6)
        for art in a3.collections + a3.lines + a3.texts:  # a 3D axis clips to a square box: draw on its whole rectangle
            art.set_clip_on(False)
    cax = fig.add_axes([0.56, 0.1, 0.36, 0.035])
    fig.colorbar(
        mpl.cm.ScalarMappable(norm=R["norm"], cmap=R["cmap"]),
        cax=cax,
        orientation="horizontal",
        label=rf"$\log_{{10}} n_{{\rm com}}$ of {D['name']} [cm$^{{-3}}$]: the colour of a cell and of its pieces",
    )
    handles = [
        Line2D([], [], color=red, lw=2.6, marker=">", markevery=[1], markersize=9, label=r"skewer and its unit tangent $\hat t$"),
        _velocity_arrow_handle(
            min(keys),
            kpc,
            r"cell velocity $\mathbf{v}$",
            " in each panel, projected" + ("; o: cut at the edge" if n_capped else ""),
        ),
        Line2D([], [], color=green, lw=4.0, label="box face (the seam)"),
        Line2D([], [], color=FG, ls="none", marker="o", markersize=5, label="generating point"),
    ]
    fig.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.02, 0.01),
        ncols=2,
        fontsize=FONT_MIN["legend"],
        frameon=False,
        columnspacing=1.5,
    )
    add_panel_letters(axs, labels=["a1", "a2"], x=0.06, y=0.97, box=False)
    fig.patches.append(
        mpl.patches.Rectangle((0, 0), 1, 1, transform=fig.transFigure, fill=False, edgecolor="none")
    )  # bbox_inches="tight"
    info = dict(D["info"], arrow_scale=kpc, key_kms=min(keys), n_capped=n_capped)
    return fig, tuple(axs), info  # misses 3D axis labels: keep the page


def plot_deposition_pieces(
    parts,
    bundle,
    *,
    half_range_cMpc: float = 2.0,
    species: str = "HI",
    arrow_kms=None,
    arrow_cMpc=None,
    figsize=(22, 8.5),
):
    """Row (b) of the deposition figures of notebook 07: the pieces of one simulation skewer around its seam, along s.

    parts, bundle, half_range_cMpc, arrow_kms, arrow_cMpc : as ``plot_deposition_cells`` (the pieces with centroids in
    s in [-l, l) around the seam). (b1) the end of the box, (b2) its start: each piece a bar of height n_p and width
    l_p in the colour of its cell, labelled inside it by its cell and a letter in path order (E3a, E3b, ...; rotated in
    a narrow bar, just above a bar too narrow for it; bold: the piece on the velocity seam), the cell crossings
    (solid), the ends of the dual segments (dotted), the box face (green), and each piece's v_par = v . t as an arrow
    in its colour from the centre of the piece along s (with its sign); arrows that would overlap each other or a
    label are stacked (``_arrow_levels``), at one scale with ``plot_deposition_rings`` (from the v_par drawn: the
    90th-percentile |v_par| 0.8 median cell sizes long; every arrow inside its panel; one
    still too long after the limits grow a little is cut at the edge and ends with an open circle), with a reference
    arrow of a round speed in a box at the top of (b2). Returns (fig, (ax_end, ax_start), info).
    """
    from matplotlib import patheffects as pe
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    D = _deposition_view(parts, bundle, half_range_cMpc, species)
    R, s_pts, L, plab, seam = D["R"], D["s_pts"], D["L"], D["plab"], D["seam"]
    s_all, l_all, c_all, n_all, u_all, col = R["s"], R["l"], R["cell"], R["n"], R["v_los"], R["colour"]
    floor, top = D["floor"], D["top"]
    green = PALETTE["green"]
    kpc, v_key = _deposition_arrows(D, parts["cells"], species, arrow_kms, arrow_cMpc, parallel=True)
    n_capped = 0
    fs = FONT_MIN["floor"]  # piece labels: small, in their bars
    fig, axs = plt.subplots(1, 2, figsize=figsize, sharey=True, layout="constrained")
    panels = []
    for kp, k in enumerate((1, 0)):
        idx = R["cols"][k]
        s_p, l_p, c_p, n_p = s_all[idx], l_all[idx], c_all[idx], n_all[idx]
        s_lo, s_hi = s_p[0] - 0.5 * l_p[0], s_p[-1] + 0.5 * l_p[-1]
        x_face = 0.0 if k == 0 else L
        b = axs[kp]
        b.bar(
            s_p - 0.5 * l_p,
            n_p - floor,
            bottom=floor,
            width=l_p,
            align="edge",
            color=[col(c) for c in c_p],
            edgecolor=FG,
            linewidth=0.8,
        )
        if seam in idx:
            b.bar(
                [s_all[seam] - 0.5 * l_all[seam]],
                [n_all[seam] - floor],
                bottom=floor,
                width=l_all[seam],
                align="edge",
                fill=False,
                edgecolor=FG,
                linewidth=3.0,
                zorder=3,
            )
        d = u_all[idx] * kpc  # the arrows: from the centre s_p to s_p + v_par kpc
        b.axvline(x_face, color=green, lw=3.5)
        b.text(
            x_face,
            0.03,
            " s = 0" if k == 0 else "s = L ",
            transform=b.get_xaxis_transform(),
            color=green,
            ha="left" if k == 0 else "right",
            va="bottom",
            fontsize=FONT_MIN["annotation"],
            fontweight="bold",
        )
        pad = 0.03 * (s_hi - s_lo)
        d_fit, cap, (a_lo,), (a_hi,) = _fit_arrows(s_p[:, None], d[:, None], [s_lo], [s_hi])  # every arrow inside the axis
        d, n_capped = d_fit[:, 0], n_capped + int(np.count_nonzero(cap))
        xl = (a_lo - (pad if k == 0 else 0.3 * pad), a_hi + (pad if k == 1 else 0.3 * pad))
        b.set(xlim=xl, yscale="log", xlabel=LABELS["s"])
        panels.append((b, idx, d, s_lo, s_hi, xl, cap))
    axs[0].set_ylim(floor, top * 10**1.0)  # provisional: the labels and arrows set the top below
    axs[0].set_ylabel(rf"$n_{{\rm com}}$ of {D['name']} [cm$^{{-3}}$]")
    axs[0].set_title(f"end of the box: s from L − {D['hr']:g} cMpc to L", pad=10)
    axs[1].set_title(f"start of the box: s from 0 to {D['hr']:g} cMpc", pad=10)
    handles = [
        Patch(facecolor=PALETTE["grey"], edgecolor=FG, label=r"piece $p$: a cell $\cap$ a dual segment ($n_p$, $\ell_p$)"),
        Patch(facecolor="none", edgecolor=FG, lw=3.0, label=f"{plab[seam]}: the piece on the velocity seam"),
        Line2D([], [], color=FG, lw=2.2, label="cell crossing"),
        Line2D([], [], color=DIM, lw=1.3, ls=(0, (3.0, 2.5)), label="end of a dual segment"),
        Line2D([], [], color=green, lw=3.5, label="box face (the seam)"),
        _velocity_arrow_handle(v_key, kpc, extra=" (top right of b2)" + ("; o: cut at the edge" if n_capped else "")),
    ]
    fig.legend(
        handles=handles, loc="outside lower center", ncols=3, fontsize=FONT_MIN["legend"], columnspacing=1.6, handlelength=2.0
    )
    add_panel_letters(axs, labels=["b1", "b2"], box=True)
    fig.get_layout_engine().execute(fig)
    y_top = np.log10(top) + 1.0
    for _ in range(2):  # the label sizes in dex depend on the top of the axis
        dex_in = (y_top - np.log10(floor)) / (axs[0].get_position().height * figsize[1])  # dex per inch
        out, y_top_new = [], np.log10(top)
        for b, idx, d, s_lo, s_hi, xl, _cap in panels:
            in_x = b.get_position().width * figsize[0] / (xl[1] - xl[0])  # inches per cMpc
            place, boxes = [], []
            for i in idx:
                txt = plab[int(i)]
                y_n = np.log10(n_all[i])
                how, tw, th = _label_fit(l_all[i] * in_x, (y_n - np.log10(floor)) / dex_in, txt, fs)
                if how is None:  # too narrow: just above the bar, rotated
                    y0_ = y_n + 0.02
                    for x0_, x1_, _, yb1 in boxes:  # above another label it would touch
                        if abs(s_all[i] - 0.5 * (x0_ + x1_)) * in_x < th:
                            y0_ = max(y0_, yb1 + 0.02)
                    boxes.append((s_all[i] - 0.5 * th / in_x, s_all[i] + 0.5 * th / in_x, y0_, y0_ + tw * dex_in))
                    place.append((i, "above", y0_))
                else:
                    place.append((i, how, 0.5 * (y_n + np.log10(floor))))
            y0 = np.log10(n_all[idx]) + 0.12  # an arrow above its bar, or above its label
            lev = _arrow_levels(s_all[idx], s_all[idx] + d, y0, pad=0.012 * (s_hi - s_lo), dy=0.16, boxes=boxes)
            y_ar = y0 + 0.16 * lev
            out.append((place, y_ar))
            y_top_new = max(y_top_new, y_ar.max(), max((bx[3] for bx in boxes), default=-np.inf))
        y_top = y_top_new + 0.25
    y_top += 0.3  # room for the reference arrow
    axs[0].set_ylim(floor, 10**y_top)
    for (b, idx, d, s_lo, s_hi, _xl, cap), (place, y_ar) in zip(panels, out):
        for i_cap, dd, ya in zip(idx[cap], d[cap], y_ar[cap]):  # cut at the edge: an open circle at the tip
            b.plot([s_all[i_cap] + dd], [10**ya], "o", mfc="none", mec=FG, ms=8, mew=1.6, zorder=9)
        for (i, how, y_), dd, ya in zip(place, d, y_ar):
            cc = col(c_all[i])
            b.annotate(
                "",
                (s_all[i] + dd, 10**ya),
                (s_all[i], 10**ya),
                zorder=6,
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=cc,
                    lw=2.4,
                    mutation_scale=14,
                    shrinkA=0,
                    shrinkB=0,
                    path_effects=[pe.withStroke(linewidth=4.0, foreground=FG)],
                ),
            )
            b.plot([s_all[i]], [10**ya], "o", color=cc, markersize=5, markeredgecolor=FG, zorder=7)
            b.text(
                s_all[i],
                10**y_,
                plab[int(i)],
                rotation=0 if how == "h" else 90,
                ha="center",
                va="bottom" if how == "above" else "center",
                fontsize=fs,
                color=FG,
                zorder=8,
                path_effects=_halo(),
                fontweight="bold" if i == seam else "normal",
            )
        y_cut = 10 ** (y_ar.max() + 0.12)
        mids = 0.5 * (s_pts[1:] + s_pts[:-1])
        b.vlines(mids[(mids > s_lo) & (mids < s_hi)], floor, y_cut, color=DIM, lw=1.3, ls=(0, (3.0, 2.5)), zorder=1)
        c_p, s_p, l_p = c_all[idx], s_all[idx], l_all[idx]
        brk = np.nonzero(np.diff(c_p))[0]
        b.vlines(s_p[brk] + 0.5 * l_p[brk], floor, y_cut, color=FG, lw=2.2, zorder=2)
    _velocity_key_2d(axs[1], v_key, kpc)
    info = dict(D["info"], arrow_scale=kpc, key_kms=v_key, n_capped=n_capped)
    return fig, tuple(axs), info


def plot_deposition_rings(
    parts,
    bundle,
    v,
    tau_pieces,
    *,
    period: float,
    half_range_cMpc: float = 2.0,
    half_range_kms: float = 300.0,
    species: str = "HI",
    line_label: str = r"Ly$\alpha$",
    arrow_kms=None,
    arrow_cMpc=None,
    tau_clip=None,
    tau_min: float = 1e-3,
    figsize=(22, 21),
):
    """Rows (c)-(e) of the deposition figures of notebook 07: the pieces of ``plot_deposition_pieces`` on the periodic
    rings of s and of v_dagger, and their optical depths.

    parts, bundle, half_range_cMpc, arrow_kms, arrow_cMpc : as ``plot_deposition_cells`` (the same pieces, colours,
    arrow scale (a reference arrow at the top right of (c); an arrow beyond the window is cut at its edge) and
    labels as ``plot_deposition_pieces``); v : (N_v,) a velocity grid over one period [0, P) [km/s]; tau_pieces :
    (N_p, N_v) every piece's tau on it with ``velocity_period=True`` (``compute_optical_depth(pieces, v, line,
    per_piece=True, velocity_period=True)``, in the order of the absorbers); period : P [km/s]; half_range_kms : the
    window of (d) and (e), v_dagger in [-v_range, v_range] around the seam 0 = P [km/s] (fixed: it does not follow the
    pieces; choose it to hold their line centres after the peculiar-velocity shift; pieces beyond it are cut at its
    edges); tau_clip : the top of the radial axis of (e) (default: the largest tau in the window, rounded up to a power
    of ten); tau_min : its bottom (log axis; smaller tau, e.g. the zeros away from a profile, sit on the ring).

    (c) the ring of s (period L), window [-l, l) around the seam s = 0 = L: angle s, angular width l_p, height n_p,
    v_par along the ring from the centre of each piece (the scale of (b), in the colour of the piece); (d) the ring of
    v_dagger (period P): each piece at its line centre v_c, angular width W_p (they may overlap); (e) every piece's
    d tau_p(v_dagger) as ``velocity_period=True`` evaluates it, and the total tau of all the pieces of the skewer (radius
    ~ log tau). Each element carries its piece label, inside it or next to it (at the peak of a profile), when it is
    large enough and the label does not overlap another (``_RingSector.piece_labels``). On the left of every row, the
    whole ring with the window in grey. Returns (fig, {"c", "d", "e"}, info).
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    D = _deposition_view(parts, bundle, half_range_cMpc, species, period=period)
    R, L, P, plab, shown, wrap = D["R"], D["L"], D["P"], D["plab"], D["shown"], D["wrap"]
    s_all, l_all, c_all, n_all, u_all, col, live, W_all = (
        R["s"],
        R["l"],
        R["cell"],
        R["n"],
        R["v_los"],
        R["colour"],
        R["live"],
        D["W"],
    )
    floor, top = D["floor"], D["top"]
    green = PALETTE["green"]
    kpc, v_key = _deposition_arrows(D, parts["cells"], species, arrow_kms, arrow_cMpc, parallel=True)
    v = np.asarray(v, dtype=float)
    tw = np.asarray(tau_pieces, dtype=float)
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        3, 2, left=0.005, right=0.995, top=0.965, bottom=0.075, width_ratios=[1.15, 4.6], wspace=0.0, hspace=0.16
    )
    ax = {r_: fig.add_subplot(gs[i, 1]) for i, r_ in enumerate("cde")}
    keys = {r_: fig.add_subplot(gs[i, 0]) for i, r_ in enumerate("cde")}
    R0, H = 1.0, 1.0  # the ring and the height of the blocks
    r_out = R0 + H + 0.8  # room above the blocks: arrows and labels
    r_n = lambda n: R0 + H * (np.log10(n) - np.log10(floor)) / (np.log10(top) - np.log10(floor))
    th_s = 2 * np.pi * D["hr"] / L  # the windows: [-l, l) on the ring of s, [-v_range, v_range] on that of v
    th_v = 2 * np.pi * float(half_range_kms) / P
    ring_s = _RingSector(-th_s, th_s, R0 - 0.05, r_out)
    ring_v = _RingSector(-th_v, th_v, R0 - 0.05, r_out)
    tick_n = 10.0 ** np.arange(np.ceil(np.log10(floor)), np.log10(top) + 1e-9)
    n_lab = [rf"$10^{{{int(np.log10(t))}}}$" for t in tick_n]

    def blocks(a_, ring, centre, width, per):
        """The pieces as blocks, cut at the edges of the window; their labels (largest first), inside or next to them."""
        items = []
        for i in shown:
            cc = col(c_all[i])
            t0, t1 = 2 * np.pi * (centre[i] - 0.5 * width[i]) / per, 2 * np.pi * (centre[i] + 0.5 * width[i]) / per
            t0, t1 = max(t0, ring.lo), min(t1, ring.hi)
            if t1 <= t0:
                continue
            ring.wedge(a_, t0, t1, R0, r_n(n_all[i]), facecolor=mpl.colors.to_rgba(cc, 0.45), edgecolor=cc, lw=1.8, zorder=3)
            items.append((0.5 * (t0 + t1), t1 - t0, R0, float(r_n(n_all[i])), plab[int(i)], False))
        items.sort(key=lambda it: -it[1] * (it[3] - it[2]))
        return items

    def radial(ins, th, r, **kw):
        ins.plot(r * np.sin(th), r * np.cos(th), **kw)

    def ins_seam(ins, seam_kw):
        radial(ins, 0.0, np.array([0.45, 1.75]), **seam_kw)

    def ins_pieces(ins, centre, per):
        for i in shown:  # the selected pieces, as short coloured ticks on the ring
            radial(ins, 2 * np.pi * centre[i] / per, np.array([0.82, 1.18]), color=col(c_all[i]), lw=2.0)

    # (c) the ring of s
    su = wrap(s_all, L)  # s with the seam at 0: the end at s - L < 0
    c_ = ax["c"]
    ring_s.frame(c_, R0)
    labs = ring_s.piece_labels(c_, blocks(c_, ring_s, su, l_all, L))
    th0 = 2 * np.pi * su[shown] / L
    th1_full = 2 * np.pi * (su[shown] + u_all[shown] * kpc) / L  # v_par from the centre of each piece, scale of (b)
    th1 = np.clip(th1_full, ring_s.lo, ring_s.hi)  # every arrow inside the window: cut at its edge
    cut = th1 != th1_full
    y0 = r_n(n_all[shown]) + 0.06
    lev = _arrow_levels(
        th0,
        th1,
        y0,
        pad=0.006 * (2 * th_s),
        dy=0.085,  # the arrows clear the labels
        boxes=[(t - h, t + h, r0, r1) for t, h, r0, r1 in labs.values()],
    )
    r_ar = y0 + 0.085 * lev
    for i, t0_, t1_, ra, cu in zip(shown, th0, th1, r_ar, cut):
        ring_s.arrow(c_, t0_, t1_, ra, color=col(c_all[i]))
        c_.plot(*ring_s.xy(t0_, ra), "o", color=col(c_all[i]), markersize=4.5, markeredgecolor=FG, zorder=7)
        if cu:  # cut at the edge of the window: an open circle at the tip
            c_.plot(*ring_s.xy(t1_, ra), "o", mfc="none", mec=FG, ms=8, mew=1.6, zorder=9)
    t_end = ring_s.hi - 0.04 * (ring_s.hi - ring_s.lo)  # the reference arrow, at the top right of the window
    t_key, r_key = t_end - 2 * np.pi * v_key * kpc / L, r_out - 0.18
    ring_s.arrow(c_, t_key, t_end, r_key, color=FG)
    c_.text(
        *ring_s.xy(t_key - 0.01 * (ring_s.hi - ring_s.lo), r_key),
        f"{v_key:g} km/s",
        ha="right",
        va="center",
        zorder=12,
        rotation=-np.degrees(t_key - ring_s.mid),
        rotation_mode="anchor",
        fontsize=FONT_MIN["annotation"],
        color=FG,
        bbox=dict(boxstyle="round,pad=0.25", facecolor=BG, edgecolor=DIM, alpha=0.9),
    )
    ring_s.ray(c_, 0.0, color=green, lw=3.5)
    ring_s.ticks(c_, L, 1.0 if D["hr"] >= 1.0 else 0.25, "cMpc")
    ring_s.rticks(c_, r_n(tick_n), n_lab, r"$n_{\rm com}$ [cm$^{-3}$]")
    c_.set_title(
        rf"(c) the pieces on the ring of $s$ (period $L$), $s \in [-{D['hr']:g}, {D['hr']:g}]$ cMpc: "
        r"height $n_p$, width $\ell_p$, $v_\parallel$",
        pad=0,
    )
    _full_ring(
        keys["c"], ring_s, L, "cMpc", lambda i_: (ins_pieces(i_, su, L), ins_seam(i_, dict(color=green, lw=3.0))), "the ring of s"
    )
    # (d) the ring of v_dagger: the same pieces at their line centres
    vu = wrap(R["v_c"], P)
    d_ = ax["d"]
    ring_v.frame(d_, R0)
    ring_v.piece_labels(d_, blocks(d_, ring_v, vu, W_all, P))
    ring_v.ray(d_, 0.0, color=FG, lw=2.0, ls=":")
    step = 100.0 if half_range_kms > 150 else 50.0
    ring_v.ticks(d_, P, step, "km/s")
    ring_v.rticks(d_, r_n(tick_n), n_lab, r"$n_{\rm com}$ [cm$^{-3}$]")
    d_.set_title(
        rf"(d) the same pieces on the ring of $v_\dagger$ (period $P$), $v_\dagger \in [-{half_range_kms:g}, {half_range_kms:g}]$ "
        r"km/s: at $v_c = v_H + v_\parallel$, width $W_p$",
        pad=0,
    )
    _full_ring(
        keys["d"],
        ring_v,
        P,
        "km/s",
        lambda i_: (ins_pieces(i_, vu, P), ins_seam(i_, dict(color=FG, lw=2.0, ls=":"))),
        r"the ring of $v_\dagger$",
    )
    # (e) every piece's d tau_p on the ring of v_dagger, and the total
    e_ = ax["e"]
    ring_v.frame(e_, R0)
    vw = wrap(v, P)
    o = np.argsort(vw)
    vw, two = vw[o], tw[:, o]
    th_all = 2 * np.pi * vw / P
    win = np.abs(th_all) <= th_v
    tot = two[live].sum(0)
    t_top = 1.5 * tot[win].max() if tau_clip is None else float(tau_clip)
    lt0, lt1 = np.log10(tau_min), np.log10(t_top)
    H_e = r_out - R0 - 0.35  # the band, but room for the labels above the peaks
    r_t = lambda t: R0 + H_e * (np.log10(np.clip(np.asarray(t), tau_min, t_top)) - lt0) / (lt1 - lt0)  # log radius
    th_w = th_all[win]
    peaks = []
    for i in shown:
        t = two[i][win]
        cc = col(c_all[i])
        ring_v.band(e_, th_w, R0, r_t(t), facecolor=mpl.colors.to_rgba(cc, 0.25), lw=0)
        ring_v.line(e_, th_w, np.where(t > tau_min, r_t(t), np.nan), color=cc, lw=1.6, zorder=3)  # not along the floor
        j = int(np.argmax(t))
        if t[j] > 3 * tau_min:  # a label at the peak of every visible profile
            peaks.append((th_w[j], 0.0, float(r_t(t[j])), float(r_t(t[j])), plab[int(i)], False, t[j]))
    peaks.sort(key=lambda it: -it[6])
    ring_v.piece_labels(e_, [it[:6] for it in peaks])
    ring_v.line(e_, th_w, r_t(tot[win]), color=FG, lw=3.0, zorder=5)
    ring_v.ray(e_, 0.0, color=FG, lw=2.0, ls=":")
    ring_v.ticks(e_, P, step, "km/s")
    tt = 10.0 ** np.arange(np.ceil(lt0), lt1 + 1e-9)
    ring_v.rticks(e_, r_t(tt), [rf"$10^{{{int(round(np.log10(x)))}}}$" for x in tt], r"$d\tau_p$, $\tau$")
    e_.set_title(
        rf"(e) from pieces to $\tau$: each piece's $d\tau_p(v_\dagger)$ ({line_label}, Voigt at its $T$, "
        r"velocity_period=True) and the total $\tau$",
        pad=0,
    )

    def ins_e(i_):
        th = 2 * np.pi * np.mod(vw, P) / P
        oo = np.argsort(th)
        rr = 1.0 + 0.6 * (np.log10(np.clip(tot[oo], tau_min, tot.max())) - lt0) / (np.log10(tot.max()) - lt0)
        i_.plot(rr * np.sin(th[oo]), rr * np.cos(th[oo]), color=FG, lw=1.2)
        ins_seam(i_, dict(color=FG, lw=2.0, ls=":"))

    _full_ring(keys["e"], ring_v, P, "km/s", ins_e, r"$\tau$ on the whole ring")
    handles = [
        Patch(facecolor=PALETTE["grey"], alpha=0.6, edgecolor=FG, label="a piece of (b), in the colour of its cell"),
        _velocity_arrow_handle(v_key, kpc, extra=" (top right of c)" + ("; o: cut at the edge" if np.any(cut) else "")),
        Line2D([], [], color=green, lw=3.5, label=r"seam $s = 0 \equiv L$ (the box faces)"),
        Line2D([], [], color=FG, lw=2.0, ls=":", label=r"seam $v_\dagger = 0 \equiv P$"),
        Line2D(
            [],
            [],
            color=PALETTE["grey"],
            lw=1.6,
            label=rf"$d\tau_p$ of a piece (radius $\propto \log\tau$, from $\tau$ = {tau_min:g})",
        ),
        Line2D([], [], color=FG, lw=3.0, label=r"total $\tau$: every piece of the skewer"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncols=3,
        fontsize=FONT_MIN["legend"],
        frameon=False,
        columnspacing=1.6,
    )
    info = dict(
        D["info"],
        window_deg={"s": float(np.degrees(th_s)), "v": float(np.degrees(th_v))},
        arrow_scale=kpc,
        key_kms=v_key,
        n_capped=int(np.count_nonzero(cut)),
        tau_window_max=float(tot[win].max()),
        v_c_range=(float(np.nanmin(vu[shown] - 0.5 * W_all[shown])), float(np.nanmax(vu[shown] + 0.5 * W_all[shown]))),
    )
    return fig, {"c": ax["c"], "d": ax["d"], "e": ax["e"]}, info


# --------------------------------------------------------------------------- frames and periodicity (notebook 07)
def plot_frames_axes(
    v,
    F_own,
    line_bundle,
    *,
    v_ref=None,
    F_ref=None,
    a_dagger: float,
    a_obs: float = 1.0,
    tau_scale=None,
    wrapped=None,
    figsize=(22, 7.5),
) -> dict:
    """The same flux in each line's own frame, on the global velocity axis and in observed wavelength: one figure per case.

    v : (N_v,) line-centred velocities v_dagger [km/s]; F_own : (N_lines, N_v) every line in its own frame. "own": the
    line-centred frames, the same gas absorbs at the same v_dagger in every line; "global": the global velocity axis of
    the reference line, by default every line's own-frame flux relabelled at v - delta_v_j (a change of axis only: each
    line keeps exactly its computed range, shown by its two dash-dotted edges), or ``F_ref`` (N_lines, N_v_ref) on one
    grid ``v_ref``, e.g. from ``compute_optical_depth(..., line_frame="reference")``, which evaluates every line over the
    whole of that grid; "wavelength": the observed wavelength of every pixel, ``spectral.line_wavelength_grid_A``
    (lambda_obs = lambda_j (a_obs / a_dagger) e^{v/c}). ``tau_scale`` : {line id: factor k}, display only: that line is
    drawn as F^k = exp(-k tau) in every figure and labelled "tau x k for visibility" (a weak line next to Lya).
    ``wrapped`` : optional dict(v, tau, tau_ref, period) adding "wrapped", what ``optical_depth.wrap_to_ref_line`` does:
    ``tau`` (N_lines, N_v) every line in its own frame on a uniform grid ``v`` of exactly one period P
    (``velocity_period=P``), ``tau_ref`` = ``wrap_to_ref_line(tau, v, line_bundle, velocity_period=P)``; every line other
    than the reference drawn before (dashed) and after (solid) the shift by -delta_v_j modulo P, the reference line thin
    for context, an arrow from the strongest feature before to after.

    Every figure has the same size ``figsize``, the same axis box, y range, fonts and a legend band below the axis, so
    that they can sit side by side; "own" and "wrapped" share the x range [0, P) (the period of ``wrapped``, else the
    edges of the pixels of ``v``). Returns {case: (fig, ax)}.
    """
    from skewerjax.core.spectral import line_wavelength_grid_A

    v = np.asarray(v, dtype=float)
    F_own = np.asarray(F_own, dtype=float)
    lines = line_bundle.lines
    k_ = [float((tau_scale or {}).get(l.id, 1.0)) for l in lines]
    sfx = [f" (τ × {k:.0e} for visibility)" if k != 1.0 else "" for k in k_]
    F_own = np.stack([F_own[j] ** k_[j] for j in range(len(lines))])  # exp(-k tau): display only
    F_ref = None if F_ref is None else np.stack([np.asarray(F_ref, dtype=float)[j] ** k_[j] for j in range(len(lines))])
    dv_j = np.asarray(line_bundle.delta_v, dtype=float)
    cols = [ENTITY_COLORS.get(l.id, species_color(l.id, j)) for j, l in enumerate(lines)]
    lam = np.asarray(line_wavelength_grid_A(v, line_bundle, a_dagger, a_obs))
    W, Hf = figsize
    top, below, band, left, right = 0.85, 0.95, 0.9, 1.15, 0.35  # [in]: title, ticks + x label, legend, margins
    rect = (left / W, (below + band) / Hf, 1 - (left + right) / W, 1 - (top + below + band) / Hf)
    dv = float(np.median(np.diff(v))) if v.size > 1 else 1.0
    x_frame = (0.0, float(wrapped["period"])) if wrapped is not None else (v[0] - 0.5 * dv, v[-1] + 0.5 * dv)

    def new():
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes(rect)
        fig.patches.append(
            mpl.patches.Rectangle((0, 0), 1, 1, transform=fig.transFigure, fill=False, edgecolor="none")
        )  # whole page
        return fig, ax  # when inline

    def finish(fig, ax, xlabel, ncols=2):
        ax.set(xlabel=xlabel, ylabel=LABELS["F"], ylim=(-0.03, 1.08))
        ax.margins(x=0.0)
        h, lab = ax.get_legend_handles_labels()
        fig.legend(
            h,
            lab,
            loc="upper center",
            bbox_to_anchor=(0.5, band / Hf),
            ncols=ncols,
            fontsize=FONT_MIN["legend"],
            frameon=False,
            columnspacing=2.0,
        )

    figs = {}
    fig, ax = new()  # own frames
    for j, l in enumerate(lines):
        ax.plot(
            v, F_own[j], color=cols[j], lw=2.4 if j == 0 else 1.8, ls="-" if j == 0 else "--", label=f"{l.id}{sfx[j]}, own frame"
        )
    finish(fig, ax, r"line-centred velocity $v_\dagger$ [km/s] (each line in its own frame)")
    ax.set_xlim(*x_frame)
    figs["own"] = (fig, ax)
    fig, ax = new()  # global velocity axis
    for j, l in enumerate(lines):
        shift = rf": at $v - \Delta v_j$ = $v$ {-dv_j[j]:+.0f} km/s" if dv_j[j] else ": unchanged (reference)"
        if F_ref is None:  # a change of axis: the own-frame flux, relabelled
            ax.plot(v - dv_j[j], F_own[j], color=cols[j], lw=2.0, label=f"{l.id}{sfx[j]}{shift}")
            for edge in (v[0], v[-1]):
                ax.axvline(edge - dv_j[j], color=cols[j], ls="-.", lw=1.2, path_effects=[])
        else:
            ax.plot(v_ref, F_ref[j], color=cols[j], lw=2.0, label=f"{l.id}{sfx[j]}{shift}")
    for j in range(1, len(lines)):
        _textbox(
            ax,
            0.02,
            0.62 - 0.22 * (j - 1),
            rf"{lines[j].id}: $\Delta v_j = c\ln(\lambda_{{\rm ref}}/\lambda_j)$ = "
            f"{dv_j[j]:+.1f} km/s;"
            "\n"
            rf"in its own frame it absorbs at the same $v_\dagger$ as {lines[0].id}",
            color=cols[j],
            transform=ax.transAxes,
        )
    finish(fig, ax, f"global velocity axis: v on the {lines[0].id} frame [km/s]")
    figs["global"] = (fig, ax)
    fig, ax = new()  # observed wavelength
    for j, l in enumerate(lines):
        ax.plot(
            lam[j],
            F_own[j],
            color=cols[j],
            lw=2.0,
            label=f"{l.id}{sfx[j]}: " + r"$\lambda_j\,(a_{\rm obs}/a_\dagger)\,e^{v_\dagger/c}$",
        )
        ax.axvline(l.lambda0_A * a_obs / a_dagger, color=cols[j], ls="-.", lw=1.2, path_effects=[])
    finish(fig, ax, r"observed wavelength $\lambda_{\rm obs}$ [Å]")
    figs["wavelength"] = (fig, ax)
    if wrapped is not None:  # wrap_to_ref_line: own frame -> reference frame, modulo P
        vw, P = np.asarray(wrapped["v"], dtype=float), float(wrapped["period"])
        tw, tr = np.asarray(wrapped["tau"], dtype=float), np.asarray(wrapped["tau_ref"], dtype=float)
        ref = line_bundle.ref_index() if line_bundle.ref_index() is not None else 0
        fig, ax = new()
        ax.plot(vw, np.exp(-tw[ref]), color=cols[ref], lw=1.0, alpha=0.5, label=f"{lines[ref].id} (reference), for context")
        for j in range(len(lines)):
            if j == ref:
                continue
            ax.plot(
                vw, np.exp(-k_[j] * tw[j]), color=cols[j], lw=1.6, ls="--", label=f"{lines[j].id}{sfx[j]}: own frame (before)"
            )
            ax.plot(
                vw,
                np.exp(-k_[j] * tr[j]),
                color=cols[j],
                lw=2.2,
                label=f"{lines[j].id}{sfx[j]}: wrap_to_ref_line, at $v - \\Delta v_j$ modulo $P$ (after)",
            )
            i0 = int(np.argmax(tw[j]))
            v1 = float(np.mod(vw[i0] - dv_j[j] - vw[0], P) + vw[0])
            ax.annotate(
                "",
                (v1, 0.35),
                (vw[i0], 0.35),
                arrowprops=dict(arrowstyle="-|>", color=cols[j], lw=2.2, mutation_scale=20, connectionstyle="arc3,rad=-0.15"),
            )
            _textbox(
                ax,
                0.5 * (vw[i0] + v1),
                0.62,
                rf"$-\Delta v_j$ = {-dv_j[j]:+.0f} km/s, modulo $P$ = {P:.0f} km/s",
                color=cols[j],
                ha="center",
            )
        finish(fig, ax, f"v [km/s] on one period [0, P): each line's own frame (dashed), the {lines[ref].id} frame (solid)")
        ax.set_xlim(*x_frame)
        figs["wrapped"] = (fig, ax)
    return figs
