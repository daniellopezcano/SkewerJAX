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

from skewerjax.summary_stats.binning import bin_index, hybrid_k_edges
from skewerjax.summary_stats.grid3d import MU_WEDGES  # noqa: F401 (MU_WEDGES: re-exported)
from skewerjax.summary_stats.orders import ZERO_ORDER_KEY

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


# --------------------------------------------------------------------------- power spectra
def pk_sigma(P_input, n_modes, n_real: int = 1):
    """Gaussian error of a binned power spectrum, P sqrt(2 / (n_modes n_real)) (n_modes counts +k and -k)."""
    n = np.asarray(n_modes, dtype=float)
    return np.asarray(P_input, dtype=float) * np.sqrt(2.0 / (np.where(n > 0, n, np.nan) * n_real))


# --------------------------------------------------------------------------- 1D statistics by order and term (notebook 08)
ORDER_COLORS = [PALETTE[k] for k in ("grey", "purple", "blue", "green", "yellow", "cyan", "red")]  # by order n (cycled)


def _as_ensemble(stats):
    """An ``OverfluxStatistics1D`` (one set of skewers: no error) or a ``summary_stats.ensemble_1d`` result, as the latter."""
    if hasattr(stats, "summary"):
        from skewerjax.summary_stats import ensemble_1d

        return ensemble_1d([stats.summary()])
    return stats


def _display_axis(ens, kind: str, max_lag, n_lin: int, n_log: int, lag_bin: int):
    """The display bins of P1D (one mode per bin for the first ``n_lin`` modes, then ``n_log`` log bins; k = 0 left
    out) or xi1D (lags 0 .. max_lag in bins of ``lag_bin`` pixels): (x, edges, slice of the stored axis)."""
    if kind == "P":
        k = np.asarray(ens["k"])
        return k, hybrid_k_edges(k[1], k[-1] * 1.0001, n_lin=n_lin, n_log=n_log)
    lags = np.asarray(ens["lags"])
    d = float(lags[1] - lags[0])
    top = 0.5 * (lags.size * d) if max_lag is None else float(max_lag)
    return lags, np.arange(-0.5 * d, top + d, lag_bin * d)


def _binned(x, each, edges, *, per_box: bool = False):
    """Mean over the realisations (n_real, ..., N) of the means in the bins of x: (centres, mean, error of the mean) and,
    with ``per_box``, the binned values of every realisation (n_real, ..., n_bins)."""
    j = bin_index(x, edges)
    nb = edges.size - 1
    ok = j >= 0
    cnt = np.bincount(j[ok], minlength=nb).astype(float)
    xc = np.bincount(j[ok], weights=x[ok], minlength=nb) / np.maximum(cnt, 1)
    flat = np.asarray(each)[..., ok].reshape(-1, int(ok.sum()))
    b = np.stack([np.bincount(j[ok], weights=f, minlength=nb) for f in flat]) / np.maximum(cnt, 1)
    b = b.reshape(np.shape(each)[:-1] + (nb,))
    n = b.shape[0]
    err = b.std(0, ddof=1) / np.sqrt(n) if n > 1 else np.full(b.shape[1:], np.nan)
    keep = cnt > 0
    if per_box:
        return xc[keep], b.mean(0)[..., keep], err[..., keep], b[..., keep]
    return xc[keep], b.mean(0)[..., keep], err[..., keep]


def _with_errors(ax, x, y, e, color, kind: str, label=None, each=None, **kw):
    """The mean with its error as points with error bars, joined by a thin line of the same colour, above every
    realisation ``each`` (n, N) as a thin faint line and above the order totals; smaller points for xi (dense lags)."""
    if each is not None:
        for yy in each:
            ax.plot(x, yy, color=color, **_EACH)
    e = None if np.all(~np.isfinite(e)) else e
    size = dict(ms=5, capsize=2.5, elinewidth=1.6) if kind == "P" else dict(ms=3.5, capsize=1.5, elinewidth=1.2)
    ax.errorbar(x, y, e, color=color, fmt="o-", lw=1.0, label=label, zorder=5, **size, **kw)


def _symlog_decades(ax, linthresh: float):
    """A symmetric-log y axis with ticks at 0 and at +/- the decades above the linear threshold only (the labels of
    +/- linthresh and 0 would overlap), labelled 10^n."""
    ax.set_yscale("symlog", linthresh=linthresh)
    lo, hi = ax.get_ylim()
    top = max(abs(lo), abs(hi), linthresh)
    first = np.ceil(np.log10(linthresh))
    exps = np.arange(first + 1, np.floor(np.log10(top)) + 1)
    if exps.size == 0:
        exps = np.array([first])
    ticks = [0.0] + [sg * 10.0**e for e in exps for sg in (1.0, -1.0) if lo <= sg * 10.0**e <= hi]
    ax.set_yticks(sorted(ticks))
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(ten_power_formatter))
    ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())


def _symlog_linear(ax, linthresh: float, linscale: float = 2.5):
    """A symmetric-log y axis whose linear part, |y| < linthresh, keeps most of the height (``linscale``), with nice
    linear ticks inside, decade ticks outside and dashed lines at +/- linthresh."""
    ax.set_yscale("symlog", linthresh=linthresh, linscale=linscale)
    lo, hi = ax.get_ylim()
    inner = mpl.ticker.MaxNLocator(nbins=4, symmetric=True).tick_values(-linthresh, linthresh)
    inner = inner[np.abs(inner) < linthresh]
    top = max(abs(lo), abs(hi))
    exps = np.arange(np.ceil(np.log10(linthresh)), np.floor(np.log10(top)) + 1)
    outer = [sg * 10.0**e for e in exps for sg in (1.0, -1.0) if 10.0**e > linthresh and lo <= sg * 10.0**e <= hi]
    ax.set_yticks(sorted(list(inner) + outer))
    ax.yaxis.set_major_formatter(
        mpl.ticker.FuncFormatter(
            lambda y, pos: ten_power_formatter(y, pos) if abs(y) >= linthresh else f"{y:g}".replace("-", "−")
        )
    )
    ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())
    for s in (-1.0, 1.0):
        ax.axhline(s * linthresh, color=DIM, ls="--", lw=0.8)


def plot_orders_superimposed(
    stats,
    kind: str = "P",
    *,
    max_lag=None,
    delta_v_list=None,
    linthresh=None,
    n_lin: int = 16,
    n_log: int = 30,
    lag_bin: int = 4,
    each: bool = True,
    axes=None,
    figsize=(12, 9),
):
    """P1D or xi1D of a composite overflux split by order |A| + |B|, all orders on one panel; the sum of the orders over
    the total, and (sum - total) / max |total| below (rounding error only).

    stats : ``summary_stats.ensemble_1d`` of many realisations (the mean, with the error of the mean: points with error
    bars, joined by thin lines; every realisation faint with ``each``) or one ``OverfluxStatistics1D``. The total and
    the sum of the orders, the reference the orders add up to, are lines. P: symlog y against log k, binned for display
    (one mode per bin for the first ``n_lin`` modes, then ``n_log`` log bins); xi: linear, lags 0 .. max_lag (default
    half the period) in bins of ``lag_bin`` pixels. delta_v_list : line offsets [km/s]: for xi, dashed lines at every
    separation folded into half the period. Returns (fig, (main, residual)).
    """
    if kind not in ("P", "xi"):
        raise ValueError(f"kind must be 'P' or 'xi', got {kind!r}.")
    ens = _as_ensemble(stats)
    x, edges = _display_axis(ens, kind, max_lag, n_lin, n_log, lag_bin)
    xc, tot, tot_e = _binned(x, ens[f"{kind}_total_each"], edges)
    _, orders, orders_e, orders_b = _binned(x, ens[f"{kind}_order_each"], edges, per_box=True)
    show_each = each and orders_b.shape[0] > 1
    scale = float(np.max(np.abs(tot))) or 1.0
    if axes is None:
        fig, axes = plt.subplots(
            2, 1, figsize=figsize, sharex=True, gridspec_kw={"height_ratios": [3, 1.2]}, layout="constrained"
        )
    ax0, ax1 = axes
    for n, (y, e) in enumerate(zip(orders, orders_e)):
        if np.max(np.abs(y)) > 1e-12 * scale:
            c = ORDER_COLORS[n % len(ORDER_COLORS)]
            _with_errors(ax0, xc, y, e, c, kind, label=rf"$\mathcal{{O}}({n})$", each=orders_b[:, n] if show_each else None)
    ax0.plot(xc, tot, lw=3.0, color=FG, alpha=0.6, label="total", zorder=3)
    ax0.plot(xc, orders.sum(0), lw=1.5, ls="--", color=PALETTE["red"], label=r"$\sum$ orders", zorder=4)
    total_sum = np.asarray(ens[f"{kind}_order"]).sum(0)
    full = np.asarray(ens[f"{kind}_total"])
    sel = (x > 0) & (x <= edges[-1]) if kind == "P" else (x >= 0) & (x <= edges[-1])
    ax1.plot(x[sel], ((total_sum - full) / scale)[sel], lw=1.2, color=PALETTE["red"])
    ax1.axhline(0.0, color=FG, lw=1.0, ls=":")
    if kind == "xi" and delta_v_list is not None:
        period = float(len(ens["lags"]) * (ens["lags"][1] - ens["lags"][0]))
        dv = np.asarray(delta_v_list, dtype=float)
        seps = np.unique(np.round(np.abs(np.mod(dv[:, None] - dv[None, :] + period / 2, period) - period / 2), 3))
        for d in seps[seps > 0]:
            for a in (ax0, ax1):
                a.axvline(d, color=PALETTE["grey"], ls="--", lw=1.4)
    if kind == "P":
        ax0.set_xscale("log")
        _symlog_decades(ax0, 1e-3 * scale if linthresh is None else linthresh)
        ax0.set_ylabel(LABELS["P1D"])
        ax1.set_xlabel(LABELS["k"])
    else:
        ax0.set_ylabel(LABELS["xi"])
        ax1.set_xlabel(LABELS["lag"])
        ax0.set_xlim(0.0, 0.5 * x.size * float(x[1] - x[0]) if max_lag is None else float(max_lag))  # up to half the period
    ax0.axhline(0.0, color=DIM, lw=0.8)
    ax1.set_ylabel(r"$\Delta / \max|{\rm tot}|$")
    if show_each:
        ax0.plot([], [], color=DIM, lw=1.2, alpha=0.5, label=f"each box ({orders_b.shape[0]})")
    ax0.legend(frameon=False, ncol=2, fontsize=FONT_MIN["legend"])
    return ax0.figure, np.array([ax0, ax1])


def _pair_label(pair):
    a, b = (r"$\mathcal{C}$" if t == ZERO_ORDER_KEY else t for t in pair)
    return f"{a}–{b}"


def plot_terms_by_order(
    stats,
    kind: str = "P",
    *,
    orders=None,
    max_lag=None,
    linthresh=None,
    n_lin: int = 16,
    n_log: int = 30,
    lag_bin: int = 4,
    each: bool = True,
    legend: str = "below",
    axs=None,
    figsize=None,
):
    """P1D or xi1D of the total overflux split by order and, within each order, into its terms.

    One panel per order n (default every order with a non-zero contribution where it is drawn: for xi the order-0
    normalisation term too, for P not, as it lives at k = 0 only; no titles: the order is the thick total's legend
    entry), each on its own scale (P: symmetric log, linear
    within 1 % of the panel's order total); the terms (A, B) and (B, A) are drawn as one curve "A–B" (their sum, real), the order total thick. stats :
    ``summary_stats.ensemble_1d`` (mean and error of the mean: points with error bars joined by thin lines; every
    realisation faint with ``each``) or one ``OverfluxStatistics1D``. P: symlog y against log k (display bins as
    ``plot_orders_superimposed``); xi: linear, lags 0 .. max_lag. ``axs``: draw into these axes (one per order);
    ``legend``: "below" each panel or on its "right". Returns (fig, axs).
    """
    if kind not in ("P", "xi"):
        raise ValueError(f"kind must be 'P' or 'xi', got {kind!r}.")
    ens = _as_ensemble(stats)
    x, edges = _display_axis(ens, kind, max_lag, n_lin, n_log, lag_bin)
    pairs = [tuple(p) for p in np.asarray(ens["pairs"])]
    keys = list(dict.fromkeys(a for a, _ in pairs))  # the term keys in their order (order-0 term first)
    po = np.asarray(ens["pair_orders"])
    terms_each = np.asarray(ens[f"{kind}_terms_each"])
    groups = {}
    for i, (a, b) in enumerate(pairs):
        key = tuple(sorted((a, b), key=keys.index))
        groups.setdefault(int(po[i]), {}).setdefault(key, []).append(i)
    xc, by_order, by_order_e = _binned(x, ens[f"{kind}_order_each"], edges)
    scale = float(np.max(np.abs(by_order))) or 1.0
    if orders is None:  # those drawn (P: k > 0, so not the order-0 term)
        orders = [n for n in range(by_order.shape[0]) if np.max(np.abs(by_order[n])) > 1e-12 * scale]
    if axs is None:
        fig, axs = plt.subplots(1, len(orders), figsize=figsize or (6.5 * len(orders), 6.5), layout="constrained")
    axs = np.atleast_1d(axs)
    for ax, n in zip(axs, orders):
        for i, (key, idx) in enumerate(groups.get(n, {}).items()):
            _, y, e, b = _binned(x, terms_each[:, idx].sum(1), edges, per_box=True)
            if np.max(np.abs(y)) > 1e-12 * scale:
                c = PALETTE_CYCLE[i % len(PALETTE_CYCLE)]
                _with_errors(ax, xc, y, e, c, kind, label=_pair_label(key), each=b if each and b.shape[0] > 1 else None)
        ax.plot(xc, by_order[n], color=FG, lw=3.0, alpha=0.6, label=rf"$\mathcal{{O}}({n})$ total", zorder=1)
        if kind == "P":
            ax.set_xscale("log")
            _symlog_decades(ax, 1e-2 * float(np.max(np.abs(by_order[n]))) if linthresh is None else linthresh)
            ax.set_xlabel(LABELS["k"])
        else:
            ax.set_xlabel(LABELS["lag"])
            ax.set_xlim(0.0, 0.5 * x.size * float(x[1] - x[0]) if max_lag is None else float(max_lag))  # up to half the period
        ax.axhline(0, color=DIM, lw=0.8)
        if legend == "right":
            ax.legend(fontsize=FONT_MIN["legend"], loc="center left", bbox_to_anchor=(1.01, 0.5))
        else:
            ax.legend(fontsize=FONT_MIN["legend"], ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2))  # the panels are dense
    axs[0].set_ylabel(LABELS["P1D"] if kind == "P" else LABELS["xi"])
    return axs[0].figure, axs


def _comoving_1d(ax, kind: str, kms_per_cMpc: float | None):
    """The cMpc counterpart of a 1D axis in velocity units on top of ``ax`` (``comoving_axis``): k_par [cMpc^-1] for
    P1D, the separation [cMpc] for xi1D."""
    if kms_per_cMpc is not None:
        if kind == "P":
            comoving_axis(ax, kms_per_cMpc, "k", label=r"$k_\parallel$ [cMpc$^{-1}$]")
        else:
            comoving_axis(ax, kms_per_cMpc, "r", label=r"$\Delta x_\parallel$ [cMpc]")


def plot_orders_compared(ensembles: dict, kind: str = "P", *, kms_per_cMpc: float | None = None, figsize=None, **kw):
    """``plot_orders_superimposed`` of one or several ensembles side by side, one column each, with identical layouts
    (notebook 08, Part D: the flux contrast). ensembles : {title: ``ensemble_1d``} (the titles are drawn only for more
    than one ensemble); kms_per_cMpc : a top axis in comoving units (``comoving_axis``, k_x = u k_v or x = v / u); kw:
    the options of ``plot_orders_superimposed``. Returns (fig, axs (2, n))."""
    n = len(ensembles)
    fig, axs = plt.subplots(
        2,
        n,
        figsize=figsize or (12 * n, 10),
        sharex="col",
        gridspec_kw={"height_ratios": [3, 1.2]},
        layout="constrained",
        squeeze=False,
    )
    for c, (title, ens) in enumerate(ensembles.items()):
        plot_orders_superimposed(ens, kind, axes=axs[:, c], **kw)
        _comoving_1d(axs[0, c], kind, kms_per_cMpc)
        if n > 1:
            axs[0, c].set_title(title, fontsize=FONT_MIN["title"] + 2)
    return fig, axs


def plot_terms_compared(ensembles: dict, kind: str = "P", *, orders=None, kms_per_cMpc: float | None = None, figsize=None, **kw):
    """``plot_terms_by_order`` of one or several ensembles, stacked: one row per order (sharing the x axis), one column
    per ensemble, with identical layouts (notebook 08, Part D). ensembles : {title: ``ensemble_1d``}; orders : the
    orders shown (default: those with a non-zero contribution in the first ensemble, without order 0: in P1D it lives
    at k = 0 only, in xi1D it is a constant); kms_per_cMpc : a top axis in comoving units on the first row
    (``comoving_axis``). No titles for one ensemble. Returns (fig, axs (n_orders, n))."""
    first = _as_ensemble(next(iter(ensembles.values())))
    if orders is None:
        o = np.abs(np.asarray(first[f"{kind}_order"]))
        o = o[:, 1:] if kind == "P" else o
        orders = [n for n in range(1, o.shape[0]) if o[n].max() > 1e-12 * o.max()]
    n = len(ensembles)
    fig, axs = plt.subplots(
        len(orders),
        n,
        figsize=figsize or (16 if n == 1 else 13 * n, 5.5 * len(orders)),
        sharex=True,
        layout="constrained",
        squeeze=False,
    )
    ylab = LABELS["P1D"] if kind == "P" else LABELS["xi"]
    for c, (title, ens) in enumerate(ensembles.items()):
        plot_terms_by_order(ens, kind, orders=orders, axs=axs[:, c], legend="right", **kw)
        for r, ax in enumerate(axs[:, c]):
            ax.set_ylabel(ylab)
            if r < len(orders) - 1:
                ax.set_xlabel("")
        _comoving_1d(axs[0, c], kind, kms_per_cMpc)
        if n > 1:
            axs[0, c].set_title(title, fontsize=FONT_MIN["title"])
    return fig, axs


_GRID = dict(color=PALETTE["grey"], lw=0.35, alpha=0.7, zorder=0)  # the sampling grid of plot_toy_skewer, very thin


def _grid_or_inset(ax, positions, draw, window, *, min_gap_in: float = 0.05, label: str = ""):
    """Very thin vertical lines at ``positions`` on ``ax``: over the whole panel when two neighbours are at least
    ``min_gap_in`` inches apart, otherwise only in a zoomed inset over ``window`` (x0, x1), in which ``draw(ax)`` redraws
    the data, with the zoomed range marked on the panel. Returns "panel" or "inset"."""
    pos = np.asarray(positions, dtype=float)
    x0, x1 = ax.get_xlim()
    width_in = ax.get_position().width * ax.figure.get_size_inches()[0]
    gap = float(np.median(np.diff(pos))) / (x1 - x0) * width_in if pos.size > 1 else np.inf
    if gap >= min_gap_in:
        ax.vlines(pos[(pos >= x0) & (pos <= x1)], 0, 1, transform=ax.get_xaxis_transform(), **_GRID)
        return "panel"
    ins = ax.inset_axes([0.06, 0.50, 0.26, 0.46])
    draw(ins)
    ins.set_xlim(*window)
    sel = (pos >= window[0]) & (pos <= window[1])
    ins.vlines(pos[sel], 0, 1, transform=ins.get_xaxis_transform(), **dict(_GRID, lw=0.6, alpha=0.9))
    ins.set_yscale(ax.get_yscale())
    ins.relim()
    ins.autoscale_view(scalex=False)
    ins.tick_params(labelsize=FONT_MIN["floor"])
    ins.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ins.xaxis.set_major_locator(mpl.ticker.MaxNLocator(3))
    ins.set_facecolor(BG)
    ins.text(
        0.98,
        0.04,
        label,
        transform=ins.transAxes,
        ha="right",
        va="bottom",
        fontsize=FONT_MIN["floor"],
        color=FG,
        path_effects=_halo(),
    )
    ax.indicate_inset_zoom(ins, edgecolor=FG, alpha=0.8)
    return "inset"


def plot_toy_skewer(
    L: float,
    n: dict,
    v,
    tau,
    *,
    mean_transmission,
    line_ids=("HI-LyA", "SiIII-1206"),
    points_per_cell: int | None = None,
    zoom_cells: float = 2.0,
    zoom_pixels: int = 12,
    figsize=(22, 20),
):
    """One skewer of a toy box, from the gas to the overflux (notebook 08, B2). Four rows:

    (a) n of each species against the comoving position y along the skewer (the cell values, as steps); (b) the
    overdensities n / <n> - 1 (<.> along this skewer); (c) the transmission T = exp(-tau) of each line and the total
    T = prod_j T_j, on the Lya axis (every line evaluated there: Si III at v + delta_v, wrapped into the period); (d) the
    overfluxes delta = T / T_bar - 1 on the same axis, with T_bar(v) = ``mean_transmission`` (n_lines + 1, N_pix), the
    ensemble mean at each pixel of each line and of the total (``run_skewers``'s ``mean_transmission``).

    The grids, as very thin grey vertical lines: in (a) and (b) the points of the path, y_j = j L / (N p) with p =
    ``points_per_cell`` (None: not drawn), in (c) and (d) the pixel centres v, where tau is evaluated. When the lines of
    a panel would lie closer than about 0.05 inch, they are drawn only in a zoomed inset (``zoom_cells`` cells from
    y = L / 2, or ``zoom_pixels`` pixels from the middle of the axis), marked on the panel.

    L : box side [cMpc]; n : {species: (N,)} the cells along y, centred on j L / N; v : (N_pix,) the Lya axis [km/s];
    tau : (n_lines, N_pix) on it. Species and lines in the same order (H I / Lya, Si III / Si III 1206). Returns
    (fig, axs (4,), info {"points": "panel" | "inset" | None, "pixels": "panel" | "inset"}).
    """
    sp = list(n)
    N = len(n[sp[0]])
    dx = L / N
    edges = (np.arange(N + 2) - 0.5) * dx  # cell j centred on j dx, and cell 0 again at the far end (periodic)
    wrap = lambda a: np.append(a, a[0])
    fig, axs = plt.subplots(4, 1, figsize=figsize, layout="constrained")
    sty = [dict(lw=2.4), dict(lw=2.0, ls="--")]
    name = {"HI": "H I", "SiIII": "Si III"}
    v = np.asarray(v)
    F = np.exp(-np.asarray(tau))
    Ft = F.prod(0)
    T_bar = np.asarray(mean_transmission)

    def gas(row):
        def draw(ax):
            for j, s in enumerate(sp):
                c = ENTITY_COLORS.get(s, PALETTE_CYCLE[j])
                y = n[s] if row == 0 else n[s] / np.mean(n[s]) - 1
                lab = name.get(s, s) if row == 0 else rf"$\delta_{{\rm {name.get(s, s).replace(' ', '')}}}$"
                ax.stairs(wrap(y), edges, baseline=None, color=c, label=lab, **sty[j % 2])

        return draw

    def flux(row):
        def draw(ax):
            for j, lid in enumerate(line_ids):
                c = ENTITY_COLORS.get(lid, PALETTE_CYCLE[j])
                lab = {"HI-LyA": r"H I Ly$\alpha$", "SiIII-1206": r"Si III 1206 (on the Ly$\alpha$ axis)"}.get(lid, lid)
                ax.plot(v, F[j] if row == 2 else F[j] / T_bar[j] - 1, color=c, label=lab, **sty[j % 2])
            ax.plot(v, Ft if row == 2 else Ft / T_bar[-1] - 1, color=FG, lw=1.4, label="total", zorder=5)

        return draw

    draws = [gas(0), gas(1), flux(2), flux(3)]
    pos = "position along the skewer, y [cMpc]"
    v_end = float(v[-1] + 0.5 * (v[1] - v[0]))
    for i, (ylab, xlab) in enumerate(
        (
            (r"$n\ [\mathrm{cm^{-3}}]$", pos),
            (r"$\delta = n/\langle n\rangle - 1$", pos),
            (LABELS["transmission"], LABELS["v_ref"]),
            (r"$\delta = T/\bar T - 1$", LABELS["v_ref"]),
        )
    ):
        draws[i](axs[i])
        axs[i].set(ylabel=ylab, xlabel=xlab)
        axs[i].set_xlim(0, L if xlab == pos else v_end)
        axs[i].legend(fontsize=FONT_MIN["legend"], ncol=3, loc="upper right")
    axs[0].set_yscale("log")
    for i in (1, 3):
        axs[i].axhline(0, color=DIM, lw=0.8)
    fig.canvas.draw()  # the axes positions, for the spacing of the grid lines
    info = {"points": None}
    if points_per_cell:
        y_pts = np.arange(N * int(points_per_cell) + 1) * dx / int(points_per_cell)
        w0 = 0.5 * L
        for i in (0, 1):
            info["points"] = _grid_or_inset(
                axs[i], y_pts, draws[i], (w0, w0 + zoom_cells * dx), label=f"points of the path, {int(points_per_cell)} per cell"
            )
    dv = float(v[1] - v[0])
    w0 = float(v[v.size // 2]) - 0.5 * dv
    for i in (2, 3):
        info["pixels"] = _grid_or_inset(axs[i], v, draws[i], (w0, w0 + zoom_pixels * dv), label="pixel centres")
    return fig, axs, info


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
    "transmission": r"$T = e^{-\tau}$",
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
    from skewerjax.summary_stats import mcdonald_factor

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


# --------------------------------------------------------------------------- toy boxes: the fields and their 3D statistics (08)
MU_COLORS = ("blue", "cyan", "green", "red")  # one colour per wedge, in every panel (PALETTE keys)
_EACH = dict(lw=0.8, alpha=0.12)  # one realisation, faint, in the background
_MEAN = dict(fmt="o-", lw=1.0, ms=7, capsize=4, elinewidth=2.2, capthick=2.0, mew=1.5, zorder=6)  # the mean and its error, joined
_BAND = dict(color=PALETTE["green"], alpha=0.28)  # the uncertainty band of a measured iso-contour
_LEGEND_08 = FONT_MIN["legend"] + 2  # the custom legends of the wedge figures


def _slice_image(ax, fields, L: float, *, species: str = "HI", z_index: int = 0, colorbar: bool = True):
    """The log number density of one species in the plane z = z_index dx (x across, y upwards), as panel (a1) of
    ``plot_toy_fields``; returns the image."""
    N = fields["v"].shape[0]
    dx = L / N
    ext = (-dx / 2, L - dx / 2, -dx / 2, L - dx / 2)  # cell j is centred on j dx
    ln = np.log10(np.asarray(fields["n"][species])[:, :, z_index].T)  # (y, x): imshow rows are y
    im = ax.imshow(ln, origin="lower", extent=ext, cmap=CMAPS["density"], norm=_density_norm(ln), interpolation="nearest")
    if colorbar:
        ax.figure.colorbar(im, ax=ax, label=rf"$\log_{{10}} n_{{\rm {species}}}\ [\mathrm{{cm^{{-3}}}}]$", shrink=0.85)
    ax.set(xlabel=LABELS["x"], ylabel="y [cMpc] (line of sight)", aspect="equal")
    return im


def plot_toy_fields(fields, L: float, *, z_index: int = 0, species=("HI", "SiIII"), arrow_every: int = 8, figsize=(22, 18)):
    """The gas of one toy box in the plane of the cell layer ``z_index`` of the third axis (notebook 08, A2): rows number
    density (log, ``CMAPS["density"]``) and peculiar velocity (colour: v_y, the component along the skewers,
    ``CMAPS["diverging"]``; arrows: (v_x, v_y) in the plane, every ``arrow_every``-th cell, one scale for the figure from
    ``velocity_arrow_scale``, with a reference arrow; when every velocity is zero, the row says so and has no arrows);
    one column per species, named by the colour-bar labels (no titles). The temperature of the toy boxes is a function
    of the density (T = T0 (1 + delta)^(gamma - 1)), so it is not drawn.

    fields : ``ToyBoxes.fields`` (n {species: (N, N, N)}, v (N, N, N, 3)); L : box side [cMpc]. x is horizontal and
    y, the line of sight of the skewers, vertical. Returns (fig, axs (2, n_species), info {arrow_scale, key_kms}).
    """
    N = fields["v"].shape[0]
    dx = L / N
    ext = (-dx / 2, L - dx / 2, -dx / 2, L - dx / 2)
    sl = lambda a: np.asarray(a)[:, :, z_index].T  # (y, x): imshow rows are y
    fig, axs = plt.subplots(2, len(species), figsize=figsize, layout="constrained", squeeze=False)
    vy = sl(fields["v"][..., 1])
    v_lim = float(np.percentile(np.abs(vy), 99.5))
    still = v_lim == 0.0
    norm_v = mpl.colors.Normalize(-v_lim, v_lim) if not still else mpl.colors.Normalize(-1.0, 1.0)
    s = slice(arrow_every // 2, N, arrow_every)
    xa = np.arange(N)[s] * dx
    X, Y = np.meshgrid(xa, xa, indexing="xy")
    vx_a, vy_a = sl(fields["v"][..., 0])[s, s], vy[s, s]
    scale = key = 0.0
    if not still:
        scale = velocity_arrow_scale(np.stack([vx_a.ravel(), vy_a.ravel()], -1), np.full(vx_a.size, arrow_every * dx))
        key = velocity_key_speed(np.stack([vx_a.ravel(), vy_a.ravel()], -1))
    for c, sp in enumerate(species):
        _slice_image(axs[0, c], fields, L, species=sp, z_index=z_index)
        ax = axs[1, c]
        im = ax.imshow(vy, origin="lower", extent=ext, cmap=CMAPS["diverging"], norm=norm_v, interpolation="nearest")
        fig.colorbar(im, ax=ax, label=r"$v_y$ (along the skewers) [km s$^{-1}$]", shrink=0.85)
        ax.set(xlabel=LABELS["x"], ylabel="y [cMpc] (line of sight)", aspect="equal", xlim=ext[:2], ylim=ext[2:])
        if still:
            ax.text(
                0.5,
                0.5,
                "no peculiar velocities: v = 0",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color=FG,
                fontsize=FONT_MIN["annotation"] + 3,
                path_effects=_halo(),
            )
        else:
            ax.quiver(
                X, Y, vx_a * scale, vy_a * scale, angles="xy", scale_units="xy", scale=1.0, color=FG, width=0.0025, zorder=5
            )
            _velocity_key_2d(ax, key, scale, x=0.97, y=1.03)
    return fig, axs, dict(arrow_scale=scale, key_kms=key)


def plot_skewer_slice(
    fields, L: float, columns, *, z_index: int = 0, species: str = "HI", highlight=None, n_arrows: int = 4, figsize=(12, 10.5)
):
    """Where the skewers run (notebook 08, B2): the number density of ``species`` in the plane of the cell layer
    ``z_index`` of the third axis (as panel (a1) of ``plot_toy_fields``), a plane that contains the line of sight (y) and
    the row of skewers with i_z = z_index; those skewers as thin black lines along y, each with ``n_arrows`` tiny arrows
    along its tangent (+y, the direction of the photons' path), and the spacing of two neighbours marked at the top.

    columns : (N_sk, 2) the node indices (i_x, i_z) of the skewers (``ToyBoxes.columns``); highlight : (i_x, i_z) of
    one skewer drawn thicker (the skewer of ``plot_toy_skewer``). Returns (fig, ax, info {n_row, spacing_cMpc})."""
    N = fields["v"].shape[0]
    dx = L / N
    cols = np.atleast_2d(np.asarray(columns))
    row = np.sort(cols[cols[:, 1] == z_index, 0])
    if row.size == 0:
        raise ValueError(f"no skewer in the layer i_z = {z_index}.")
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    _slice_image(ax, fields, L, species=species, z_index=z_index)
    x = row * dx
    y0, y1 = -dx / 2, L - dx / 2
    ax.vlines(x, y0, y1, colors="black", lw=0.6, zorder=5)
    ya = (np.arange(n_arrows) + 0.5) / n_arrows * L
    Xa, Ya = np.meshgrid(x, ya)
    ax.quiver(
        Xa,
        Ya,
        np.zeros_like(Xa),
        np.full_like(Ya, 0.025 * L),
        angles="xy",
        scale_units="xy",
        scale=1.0,
        color="black",
        width=0.0022,
        headwidth=4.5,
        headlength=5,
        headaxislength=4.5,
        zorder=6,
    )
    ax.plot([], [], color="black", lw=1.0, marker=r"$\uparrow$", ms=12, label=f"skewers of this layer ({row.size}), along +y")
    if highlight is not None and int(highlight[1]) == z_index:
        ax.axvline(int(highlight[0]) * dx, color="black", lw=2.6, zorder=6, label="the skewer of the next figure")
    spacing = float(np.median(np.diff(x))) if row.size > 1 else float(L)
    if row.size > 1:
        j = row.size // 2
        ax.annotate(
            "",
            xy=(x[j + 1], 1.012 * L),
            xytext=(x[j], 1.012 * L),
            xycoords="data",
            annotation_clip=False,
            arrowprops=dict(arrowstyle="<->", color=FG, lw=1.4, shrinkA=0, shrinkB=0),
        )
        ax.text(
            0.5 * (x[j] + x[j + 1]),
            1.03 * L,
            f"{spacing:.3g} cMpc",
            ha="center",
            va="bottom",
            color=FG,
            fontsize=FONT_MIN["annotation"],
            clip_on=False,
        )
    ax.set(xlim=(-dx / 2, L - dx / 2), ylim=(y0, y1))
    fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center", ncol=2, fontsize=FONT_MIN["legend"])
    return fig, ax, dict(n_row=int(row.size), spacing_cMpc=spacing)


def _ensemble_xy(res, key, theory=True):
    """(mean, error, each, theory) of one statistic of a ``simulations.field_statistics`` result."""
    return res[key], res[key + "_err"], res[key + "_each"], (res["theory"][key] if theory else None)


def plot_isotropic_recovery(
    res, *, xi_view: str = "r2", r_bao: float | None = None, grf=None, n_curve: int = 600, figsize=(22, 12)
):
    """k P(|k|) and xi(|r|) of many realisations of a 3D field against their exact expectation (notebook 08, A3).

    res : ``simulations.field_statistics`` (the mean over the realisations, its error, every realisation, the
    expectation ``theory`` binned like the measurement, the ``axes``). Columns: k P(|k|) (k the mode-weighted centre of
    each bin, for the measurement and the expectation alike) and xi(|r|); rows: the statistic (every realisation faint,
    the mean with the error of the mean, the expectation on top) and its residual: measured / expected - 1 for P,
    (measured - expected) / error for xi. No titles. ``xi_view``: "r2" draws r^2 xi on linear axes (the standard BAO
    view), "symlog" xi on a symmetric-log axis. ``r_bao``: a dotted line at the BAO scale [cMpc]. ``grf``: the
    ``PowerLawGRF`` of the realisations: its continuous k P(k) (``power``) and xi(r) (``correlation``, the Hankel
    transform of the full spectrum) as a thin grey line, at ``n_curve`` points across each x range.
    Returns (fig, axs (2, 2), summary dict: chi2/dof of P, max |P / P_th - 1|, mean pull and chi2/dof of xi (diagonal)).
    """
    if xi_view not in ("r2", "symlog"):
        raise ValueError(f"xi_view must be 'r2' or 'symlog', got {xi_view!r}.")
    fig, axs = plt.subplots(2, 2, figsize=figsize, sharex="col", gridspec_kw={"height_ratios": [2.0, 1.0]}, layout="constrained")
    col, n = ENTITY_COLORS["density"], res["n_real"]
    k = res["axes"]["k_iso"]
    P, Pe, Peach, Pth = _ensemble_xy(res, "P_iso")
    ok = np.isfinite(P) & np.isfinite(Pth)
    for y in Peach:
        axs[0, 0].plot(k[ok], k[ok] * y[ok], color=col, **_EACH)
    axs[0, 0].plot([], [], color=col, lw=1.2, alpha=0.5, label=f"each realisation ({n})")
    axs[0, 0].errorbar(k[ok], k[ok] * P[ok], k[ok] * Pe[ok], color=col, label="mean, error of the mean", **_MEAN)
    axs[0, 0].plot(k[ok], k[ok] * Pth[ok], color=FG, lw=2.2, zorder=8, label="expectation (input averaged over each bin's modes)")
    cont = dict(color=ENTITY_COLORS["analytic"], lw=1.3, zorder=9)
    if grf is not None:
        kc = np.geomspace(float(k[ok].min()) / 1.05, float(k[ok].max()) * 1.05, int(n_curve))
        axs[0, 0].plot(kc, kc * np.asarray(grf.power(kc)), label=r"continuous $k P(k)$ and $\xi(r)$ (infinite volume)", **cont)
    rel, rel_e = P[ok] / Pth[ok] - 1, Pe[ok] / Pth[ok]
    axs[1, 0].errorbar(k[ok], rel, rel_e, color=col, **_MEAN)
    residual_band(axs[1, 0], 0.02)
    lim = max(0.03, 1.2 * float(np.max(np.abs(rel) + rel_e)))
    axs[0, 0].set(xscale="log", yscale="log", ylabel=r"$k\,P(k)$ [cMpc$^2$]")
    axs[1, 0].set(xscale="log", xlabel=r"$|k|$ [cMpc$^{-1}$]", ylabel="measured / expected − 1", ylim=(-lim, lim))
    axs[0, 0].legend(fontsize=FONT_MIN["legend"], loc="lower left")

    r = res["axes"]["r_iso"]
    X, Xe, Xeach, Xth = _ensemble_xy(res, "xi_iso")
    okr = np.isfinite(X) & (r > 0)
    w = r**2 if xi_view == "r2" else np.ones_like(r)
    for y in Xeach:
        axs[0, 1].plot(r[okr], (w * y)[okr], color=col, **_EACH)
    axs[0, 1].errorbar(r[okr], (w * X)[okr], (w * Xe)[okr], color=col, **_MEAN)
    axs[0, 1].plot(r[okr], (w * Xth)[okr], color=FG, lw=2.2, zorder=8)
    if grf is not None:
        rc = np.linspace(float(r[okr].min()), float(r[okr].max()), int(n_curve))
        yc = (rc**2 if xi_view == "r2" else 1.0) * grf.correlation(rc)
        axs[0, 1].plot(rc, yc, **cont)
        if xi_view == "r2":  # the axis follows the measurement; the continuum may leave it (the missing k = 0 mode)
            y = np.concatenate([(w * (X - Xe))[okr], (w * (X + Xe))[okr], (w * Xth)[okr]])
            lo, hi = float(np.nanmin(y)), float(np.nanmax(y))
            top = hi + 0.15 * (hi - lo)
            axs[0, 1].set_ylim(lo - 0.1 * (hi - lo), top)
            if np.any(yc > top):
                x_out = float(rc[np.argmax(yc > top)])
                axs[0, 1].annotate(
                    "continuous $r^2\\xi(r)$:\nabove, off scale",
                    xy=(x_out, top),
                    xytext=(rc[0] + 0.06 * (rc[-1] - rc[0]), top - 0.1 * (top - lo)),
                    color=ENTITY_COLORS["analytic"],
                    fontsize=FONT_MIN["annotation"],
                    arrowprops=dict(arrowstyle="->", color=ENTITY_COLORS["analytic"], lw=1.2),
                    ha="left",
                    va="top",
                )
    pull = (X - Xth)[okr] / Xe[okr]
    axs[1, 1].errorbar(r[okr], pull, np.ones_like(pull), color=col, **_MEAN)
    axs[1, 1].axhline(0.0, color=FG, lw=1.0)
    axs[1, 1].axhspan(-2, 2, color=DIM, alpha=0.15, lw=0)
    if xi_view == "r2":
        axs[0, 1].set(ylabel=r"$r^2 \xi(r)$ [cMpc$^2$]")
    else:
        lt = float(np.nanmin(np.abs(Xth[okr]))) * 3 or 1e-6
        axs[0, 1].set_yscale("symlog", linthresh=lt)
        axs[0, 1].set(ylabel=r"$\xi(r)$")
    axs[0, 1].axhline(0.0, color=DIM, lw=0.8)
    axs[1, 1].set(xlabel=r"$|r|$ [cMpc]", ylabel=r"$(\hat\xi - \xi_{\rm th})/\sigma$", ylim=(-5, 5))
    if r_bao is not None:
        for a in axs[:, 1]:
            a.axvline(r_bao, color=DIM, ls=":", lw=1.8)
        axs[0, 1].text(
            r_bao,
            0.04,
            r" $r_{\rm BAO}$",
            transform=axs[0, 1].get_xaxis_transform(),
            color=FG,
            fontsize=FONT_MIN["annotation"],
            ha="left",
            va="bottom",
        )
    summary = dict(
        P_chi2_dof=float(np.mean(((P[ok] - Pth[ok]) / Pe[ok]) ** 2)),
        P_max_rel=float(np.max(np.abs(rel))),
        xi_mean_pull=float(np.mean(pull)),
        xi_chi2_dof=float(np.mean(pull**2)),
        n_real=int(n),
    )
    return fig, axs, summary


def _nice(x: float) -> float:
    """x rounded to one significant figure."""
    return float(f"{x:.1g}") if x != 0 else 0.0


def _three_levels(vmin: float, vmax: float, *, log: bool = False) -> np.ndarray:
    """Three iso-contour levels, rounded to one significant figure: at 1/4, 1/2 and 3/4 of the decades between ``vmin``
    and ``vmax`` (``log``: P), or at 1/4, 1/2 and 3/4 of ``vmax`` (r^2 xi: the positive side, where the correlation and
    the BAO ring are)."""
    if log:
        lo, hi = np.log10(vmin), np.log10(vmax)
        lv = [_nice(10.0 ** (lo + f * (hi - lo))) for f in (0.25, 0.5, 0.75)]
    else:
        lv = [_nice(f * vmax) for f in (0.25, 0.5, 0.75)]
    return np.unique(lv)


def _contour_band(ax, x, y, M, E, levels, n_sigma: float = 1.0, **kw):
    """The uncertainty of measured iso-contours: for each level, the region where the level lies within M +- n_sigma E
    (the values M and their errors E on the bin centres x (columns) and y (rows), M[y, x]); nothing on a map smaller
    than 2 x 2 bins."""
    if min(np.shape(M)) < 2:
        return
    for lv in levels:
        Z = (np.asarray(M) - lv) / np.where(np.asarray(E) > 0, E, np.nan)
        ax.contourf(
            x,
            y,
            np.ma.masked_invalid(Z),
            levels=[-float(n_sigma), float(n_sigma)],
            colors=[_BAND["color"]],
            alpha=_BAND["alpha"],
            zorder=4,
            **kw,
        )


def _band_handle(n: int, what: str = "realisations", n_sigma: float = 1.0):
    """The legend entry of ``_contour_band``."""
    return mpl.patches.Patch(**_BAND, label=rf"measured $\pm {float(n_sigma):g}\sigma$ (error of the mean over {n} {what})")


def _contours(ax, x, y, M, levels, fmt, **kw):
    """Iso-contours of M[y, x] on the bin centres, labelled with their value (None on a map smaller than 2 x 2 bins)."""
    if min(np.shape(M)) < 2:
        return None
    ci = ax.contour(x, y, np.ma.masked_invalid(M), levels=levels, **kw)
    ax.clabel(ci, fmt=fmt, fontsize=FONT_MIN["floor"], inline=False)
    return ci


def _fmt_r2xi(x):
    return f"{x:g}".replace("-", "−")


def plot_kpar_kperp_maps(res, *, n_sigma: float = 1.0, figsize=(20, 9.5)):
    """k P(k_perp, k_par) of many realisations, k = |k| the mode-weighted centre of each cell of the map: the
    expectation (left) and the mean measured (right) on one logarithmic colour scale with the colour map of the r^2 xi
    maps (``CMAPS["diverging"]``), k_perp across (x) and k_par = |k_y|
    along the line of sight (y), both axes logarithmic (one mode per bin at low k, then log bins;
    ``GridStatistics3D``), so k_par = 0 and k_perp = 0 are not shown (notebook 08, A3). Three iso-power levels
    (``_three_levels``), the same for both: the expectation's dashed in both panels, the measured ones green in the
    right panel with a transparent green band where the level lies within the mean +- ``n_sigma`` times its error (the
    error of the mean of the 1D figures). No titles; the panels share their y axis. res :
    ``simulations.field_statistics``. Returns (fig, axs (2,), chi2/dof of (measured - expected) / error per bin)."""
    a = res["axes"]
    ep, eq = a["kpar_edges"], a["kperp_edges"]
    kc = np.hypot(a["kpar_map"], a["kperp_map"])  # |k| of every cell of the map
    Pm, Pe, Pth = kc * res["P_map"], kc * res["P_map_err"], kc * res["theory"]["P_map"]
    good = np.isfinite(Pm) & np.isfinite(Pth) & (Pm > 0)
    norm = mpl.colors.LogNorm(np.nanmin(Pth[good]), np.nanmax(Pth[good]))
    cmap_ = plt.get_cmap(CMAPS["diverging"]).copy()
    cmap_.set_bad((0, 0, 0, 0))
    levels = _three_levels(norm.vmin, norm.vmax, log=True)
    kcp, kcq = np.sqrt(ep[1:] * ep[:-1]), np.sqrt(eq[1:] * eq[:-1])
    fig, axs = plt.subplots(1, 2, figsize=figsize, sharey=True, layout="constrained")
    for ax, M in ((axs[0], Pth), (axs[1], Pm)):
        im = ax.pcolormesh(eq, ep, np.ma.masked_invalid(np.where(good, M, np.nan)), cmap=cmap_, norm=norm, shading="flat")
        _contours(ax, kcq, kcp, Pth, levels, _fmt_r2xi, colors=FG, linewidths=1.4, linestyles="--")
        ax.set(
            xscale="log",
            yscale="log",
            xlim=(eq[0], eq[-1]),
            ylim=(ep[0], ep[-1]),
            aspect="equal",
            xlabel=r"$k_\perp$ [cMpc$^{-1}$]",
        )
    axs[0].set_ylabel(r"$k_\parallel$ (line of sight, y) [cMpc$^{-1}$]")
    _contour_band(axs[1], kcq, kcp, Pm, Pe, levels, n_sigma)
    _contours(axs[1], kcq, kcp, Pm, levels, _fmt_r2xi, colors=PALETTE["green"], linewidths=1.8)
    handles = [
        mpl.lines.Line2D([], [], color=FG, ls="--", label=r"expected iso-$kP$ contours (both panels)"),
        mpl.lines.Line2D([], [], color=PALETTE["green"], lw=1.8, label="measured, same levels"),
        _band_handle(res["n_real"], n_sigma=n_sigma),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, fontsize=FONT_MIN["legend"])
    fig.colorbar(im, ax=axs, label=r"$k\,P(k_\perp, k_\parallel)$ [cMpc$^2$]", shrink=0.9)
    pull = (Pm - Pth)[good] / Pe[good]
    return fig, axs, float(np.mean(pull**2))


def _mu_rays(
    ax, mu_edges, r_max: float, *, labels: bool = True, sector: float | None = 0.14, y_scale: float = 1.0, kind: str = "r"
):
    """The wedges of mu = r_par / r in the (r_perp, r_par) plane, r_perp horizontal and r_par vertical: one white
    dash-dot ray from the origin to the edge of the square [0, r_max]^2 per inner edge of ``mu_edges`` (mu = 0 and 1 are
    the axes), labelled with its mu, and near the origin a small transparent sector per wedge in its colour of
    ``plot_xi_mu_wedges`` (``MU_COLORS``; radius ``sector`` r_max; None: none). The same edges as
    ``plot_xi_mu_wedges`` (the ``mu_edges`` of the measurement, ``MU_WEDGES``). mu and r_max are in comoving units on
    both axes; ``y_scale`` converts the comoving r_par (or k_par) to the units of the y axis (u for r_par in km/s, 1 / u
    for k_par in s/km; the rays and sectors are drawn in those units). ``kind``: "r" or "k" (the legend label). Returns
    the legend handle of the rays."""
    from matplotlib import patheffects as pe

    edge = [pe.withStroke(linewidth=3.2, foreground=(0, 0, 0, 0.55))]
    mu = np.asarray(mu_edges, dtype=float)
    ys = float(y_scale)
    for m in mu[1:-1]:
        d = np.array([np.sqrt(1.0 - m * m), m])  # (r_perp, r_par)
        end = r_max / d.max() * d * np.array([1.0, ys])
        ax.plot([0.0, end[0]], [0.0, end[1]], color="white", ls="-.", lw=1.6, zorder=7, path_effects=edge)
        if labels:
            ax.text(
                *(0.86 * end),
                rf"$\mu = {m:g}$",
                color="white",
                fontsize=FONT_MIN["annotation"],
                ha="left",
                va="top",
                zorder=9,
                path_effects=[pe.withStroke(linewidth=3, foreground="black")],
            )
    if sector:
        for w in range(mu.size - 1):
            th = np.linspace(np.arcsin(mu[w]), np.arcsin(mu[w + 1]), 40)
            rho = sector * r_max
            xy = np.vstack([[0.0, 0.0], np.column_stack([rho * np.cos(th), ys * rho * np.sin(th)])])
            ax.add_patch(
                mpl.patches.Polygon(
                    xy,
                    closed=True,
                    facecolor=PALETTE[MU_COLORS[w % len(MU_COLORS)]],
                    alpha=0.8,
                    edgecolor="white",
                    lw=1.0,
                    zorder=8,
                )
            )
    what = r"$\mu = r_\parallel / r$" if kind == "r" else r"$\mu = k_\parallel / k$"
    return mpl.lines.Line2D(
        [], [], color="white", ls="-.", lw=1.6, path_effects=edge, label=f"wedges of {what} (sectors: their colours)"
    )


def _r2xi_scale(values):
    """The linear, diverging colour scale of r^2 xi centred on 0, from minus to plus the largest positive value (the BAO
    ring, or the strongest correlation), so that the large negative r^2 xi towards the corners of the box (r^2 grows)
    saturates instead of compressing the ring; its three iso-contour levels (``_three_levels``) and the colour bar's
    ``extend``."""
    v = np.asarray(values, dtype=float)
    top = float(np.nanmax(v))
    vmax = top if top > 0 else float(np.nanmax(np.abs(v)))
    extend = "min" if float(np.nanmin(v)) < -vmax else "neither"
    return mpl.colors.Normalize(-vmax, vmax), _three_levels(-vmax, vmax), extend


def plot_rpar_rperp_maps(res, *, r_bao: float | None = None, mu_edges=None, n_sigma: float = 1.0, figsize=(20, 9.5)):
    """r^2 xi(r_perp, r_par) of many realisations, the real-space analogue of ``plot_kpar_kperp_maps``: the expectation
    (left) and the mean measured (right) on one linear, diverging colour scale centred on 0 (``CMAPS["diverging"]``, up
    to the largest positive r^2 xi: the more negative values towards the corners saturate; r^2 compensates the steep fall
    of xi, so the BAO ring stands out), r_perp across (x) and r_par = |r_y| along the line of sight (y), linear axes up to
    half the box, r^2 = r_par^2 + r_perp^2 at the bin centres. Three iso-r^2 xi levels, the same for both panels: the
    expectation's dashed in both, the measured ones green in the right panel with a transparent green band where the
    level lies within the mean +- ``n_sigma`` times its error (the error of the mean of the 1D figures); the boundaries of the wedges of
    ``plot_xi_mu_wedges`` as white rays with a coloured sector per wedge near the origin (``_mu_rays``; ``mu_edges``,
    default those of the measurement, ``MU_WEDGES``). No titles; the panels share their y axis. ``r_bao``: a dotted
    circle at the BAO scale [cMpc]. res : ``simulations.field_statistics``. Returns (fig, axs (2,), chi2/dof of (measured
    - expected) / error of xi per bin)."""
    a = res["axes"]
    ep, eq = a["rpar_edges"], a["rperp_edges"]
    Xm, Xe, Xth = res["xi_map"], res["xi_map_err"], res["theory"]["xi_map"]
    good = np.isfinite(Xm) & np.isfinite(Xth)
    rcp, rcq = 0.5 * (ep[1:] + ep[:-1]), 0.5 * (eq[1:] + eq[:-1])
    r2 = rcp[:, None] ** 2 + rcq[None, :] ** 2
    norm, levels, extend = _r2xi_scale((r2 * Xth)[good])
    mu_edges = a["mu_edges"] if mu_edges is None else mu_edges
    fig, axs = plt.subplots(1, 2, figsize=figsize, sharey=True, layout="constrained")
    for ax, M in ((axs[0], Xth), (axs[1], Xm)):
        im = ax.pcolormesh(
            eq, ep, np.ma.masked_invalid(np.where(good, r2 * M, np.nan)), cmap=CMAPS["diverging"], norm=norm, shading="flat"
        )
        _contours(ax, rcq, rcp, r2 * Xth, levels, _fmt_r2xi, colors=FG, linewidths=1.4, linestyles="--")
        ray = _mu_rays(ax, mu_edges, min(ep[-1], eq[-1]))
        if r_bao is not None:
            t = np.linspace(0, np.pi / 2, 200)
            ax.plot(r_bao * np.cos(t), r_bao * np.sin(t), color=FG, ls=":", lw=1.8)
        ax.set(xlim=(eq[0], eq[-1]), ylim=(ep[0], ep[-1]), aspect="equal", xlabel=r"$r_\perp$ [cMpc]")
    axs[0].set_ylabel(r"$r_\parallel$ (line of sight, y) [cMpc]")
    _contour_band(axs[1], rcq, rcp, r2 * Xm, r2 * Xe, levels, n_sigma)
    _contours(axs[1], rcq, rcp, r2 * Xm, levels, _fmt_r2xi, colors=PALETTE["green"], linewidths=1.8)
    handles = [
        mpl.lines.Line2D([], [], color=FG, ls="--", label=r"expected iso-$r^2\xi$ contours (both panels)"),
        mpl.lines.Line2D([], [], color=PALETTE["green"], lw=1.8, label="measured, same levels"),
        _band_handle(res["n_real"], n_sigma=n_sigma),
        ray,
    ]
    if r_bao is not None:
        handles.append(mpl.lines.Line2D([], [], color=FG, ls=":", lw=1.8, label=r"$r = r_{\rm BAO}$"))
    fig.legend(handles=handles, loc="outside lower center", ncol=3, fontsize=FONT_MIN["legend"])
    fig.colorbar(im, ax=axs, label=r"$r^2\xi(r_\perp, r_\parallel)$ [cMpc$^2$]", shrink=0.9, extend=extend)
    pull = (Xm - Xth)[good] / Xe[good]
    return fig, axs, float(np.mean(pull**2))


def _wedge_label(lo: float, hi: float) -> str:
    return rf"${lo:.2f} < \mu < {hi:.2f}$"


_UNITS = {  # the r and k labels of the 3D statistics: comoving (the fields, and every isotropic quantity) or velocity
    "cMpc": dict(
        r=r"$r$ [cMpc]",
        r2=r"$r^2 \xi(r)$ [cMpc$^2$]",
        k=r"$|k|$ [cMpc$^{-1}$]",
        P=r"$P(k)$ [cMpc$^3$]",
        kP=r"$k\,P(k)$ [cMpc$^2$]",
    ),
    "km/s": dict(
        r=r"$r$ [km s$^{-1}$]",
        r2=r"$r^2 \xi(r)$ [(km s$^{-1}$)$^2$]",
        k=r"$|k|$ [s km$^{-1}$]",
        P=r"$P(k)$ [(km s$^{-1}$)$^3$]",
        kP=r"$k\,P(k)$ [(km s$^{-1}$)$^2$]",
    ),
}


def _units(unit: str) -> dict:
    if unit not in _UNITS:
        raise ValueError(f"unit must be one of {tuple(_UNITS)}, got {unit!r}.")
    return _UNITS[unit]


def _secondary(ax, functions, side: str, label: str):
    """A secondary x (``side`` "top") or y ("right") axis with the conversion ``functions`` (forward, inverse)."""
    if side == "top":
        sec = ax.secondary_xaxis("top", functions=functions)
        sec.set_xlabel(label)
    elif side == "right":
        sec = ax.secondary_yaxis("right", functions=functions)
        sec.set_ylabel(label)
    else:
        raise ValueError(f"side must be 'top' or 'right', got {side!r}.")
    return sec


def comoving_axis(ax, kms_per_cMpc: float, kind: str = "r", *, side: str = "top", label: str | None = None):
    """A secondary axis in comoving units on an axis in velocity units (notebook 08, Parts C and D), with one conversion
    factor, v = u x and k_v = k_x / u: u = ``kms_per_cMpc`` [km/s per cMpc] (u = period / L = (1/L) int_0^L aH dy, the
    mean aH along a skewer, ``SkewerRun.derived``).

    kind : "r" (km/s below, cMpc on the new axis) or "k" (s/km below, cMpc^-1); side : "top" (on the x axis) or "right"
    (on the y axis); label : default the comoving label of ``_UNITS``. Returns the secondary axis."""
    u = float(kms_per_cMpc)
    if kind not in ("r", "k"):
        raise ValueError(f"kind must be 'r' or 'k', got {kind!r}.")
    functions = (lambda v: v / u, lambda x: x * u) if kind == "r" else (lambda k: k * u, lambda k: k / u)
    return _secondary(ax, functions, side, label or _units("cMpc")[kind])


def velocity_axis(ax, kms_per_cMpc: float, kind: str = "r", *, side: str = "top", label: str | None = None):
    """The reverse of ``comoving_axis``: a secondary axis in velocity units on an axis in comoving units, v = u x and
    k_v = k_x / u (kind "r": cMpc below, km/s on the new axis; "k": cMpc^-1 below, s/km). Returns the secondary axis."""
    u = float(kms_per_cMpc)
    if kind not in ("r", "k"):
        raise ValueError(f"kind must be 'r' or 'k', got {kind!r}.")
    functions = (lambda x: x * u, lambda v: v / u) if kind == "r" else (lambda k: k / u, lambda k: k * u)
    return _secondary(ax, functions, side, label or _units("km/s")[kind])


def _reference_wedges(reference, h: float | None):
    """The wedges of a ``utils.reference_data`` correlation block (r [h^-1 Mpc], r^2 xi [h^-2 Mpc^2]) in comoving units:
    r / h [cMpc], r^2 xi / h^2 [cMpc^2] (``h`` default H0 / 100 of ``core.phys``)."""
    from skewerjax.core import phys

    h = phys.H0_km_s_mpc / 100.0 if h is None else float(h)
    r = np.asarray(reference["r"], dtype=float) / h
    return [
        dict(mu_min=w["mu_min"], mu_max=w["mu_max"], r=r, r2xi=np.asarray(w["r2xi"]) / h**2, sigma=np.asarray(w["sigma"]) / h**2)
        for w in reference["wedges"]
    ], h


def plot_xi_mu_wedges(
    res,
    *,
    r_bao: float | None = None,
    unit: str = "cMpc",
    kms_per_cMpc: float | None = None,
    reference=None,
    reference_h: float | None = None,
    figsize=None,
):
    """xi(r) in wedges of mu = r_par / r (``MU_WEDGES``) of many realisations, as DESI DR2 shows the Lya auto-correlation
    (arXiv:2503.14739, Fig. 3, top): r^2 xi(r) against r on linear axes, one colour per wedge (``MU_COLORS``), every
    realisation as a faint line, the mean with the error of the mean as points joined by thin lines (notebook 08, A3 and
    C). With an expectation (``res["theory"]``, ``simulations.field_statistics``): the expectation of each wedge as a
    dashed line in the wedge's colour and, below, (measured - expected) / error. Without (the overflux of
    ``run_skewers``, ``res`` = one entry of its ``flux3d`` with ``res["axes"]`` = its ``axes3d``): one panel, the
    measurement only. The wedges are offset by a few per cent of a bin in r so that every point shows. Legends above the
    axes: the wedges (colour: mu interval), the styles, and the reference. No titles.

    r_bao : a dotted line; unit : "cMpc" or "km/s" (the axis labels); kms_per_cMpc : a secondary top axis in the other
    unit (``comoving_axis`` / ``velocity_axis``). reference : a correlation block of ``utils.reference_data`` (e.g.
    ``XI_LYA_LYA_AA_WEDGES``, r [h^-1 Mpc], r^2 xi [h^-2 Mpc^2]), converted to cMpc and cMpc^2 with ``reference_h``
    (default H0 / 100 of ``core.phys``; ``unit`` must be "cMpc") and drawn as open squares with their error bars, in the
    colour of the measured wedge whose mu edges coincide (grey otherwise), labelled with their own mu ranges. Returns
    (fig, axs (2,) with an expectation, (1,) without, chi2/dof per wedge against the expectation (diagonal), or None)."""
    lab = _units(unit)
    a = res["axes"]
    mu, r = a["mu_edges"], a["r_mu"]
    X, Xe = res["xi_mu"], res["xi_mu_err"]
    each = res.get("xi_mu_each")
    theory = "theory" in res
    Xth = res["theory"]["xi_mu"] if theory else None
    nm = len(mu) - 1
    cols = [PALETTE[MU_COLORS[w % len(MU_COLORS)]] for w in range(nm)]
    dr = float(np.diff(a["r_mu_edges"]).mean())
    off = 0.08 * dr * (np.arange(nm) - (nm - 1) / 2)
    if theory:
        fig, axs = plt.subplots(
            2, 1, figsize=figsize or (12, 12), sharex=True, gridspec_kw={"height_ratios": [2.0, 1.0]}, layout="constrained"
        )
    else:
        fig, ax = plt.subplots(figsize=figsize or (12, 9), layout="constrained")
        axs = np.array([ax])
    out = [] if theory else None
    for w in range(nm):
        top = np.isfinite(X[w]) & (Xe[w] > 0)
        if theory:
            top &= np.isfinite(Xth[w])
            x, r2 = r[w][top], r[w][top] ** 2
            pull = (X[w][top] - Xth[w][top]) / Xe[w][top]
            out.append(float(np.mean(pull**2)))
            axs[0].plot(x, r2 * Xth[w][top], color=cols[w], ls="--", lw=2.0, zorder=7)
            axs[1].errorbar(x + off[w], pull, np.ones_like(pull), color=cols[w], **dict(_MEAN, ms=6))
        xt, r2t = r[w][top], r[w][top] ** 2
        if each is not None:
            for y in each[:, w]:
                axs[0].plot(xt + off[w], r2t * y[top], color=cols[w], **_EACH)
        axs[0].errorbar(xt + off[w], r2t * X[w][top], r2t * Xe[w][top], color=cols[w], **dict(_MEAN, ms=6))
    grey = PALETTE["grey"]
    styles = [mpl.lines.Line2D([], [], color=grey, marker="o", ms=6, lw=1.0, label="measured: mean, error of the mean")]
    if theory:
        styles.append(mpl.lines.Line2D([], [], color=grey, ls="--", lw=2.0, label="expectation of the wedge"))
        axs[1].set(ylabel=r"$(\hat\xi - \xi_{\rm th})/\sigma$", ylim=(-5, 5))
        axs[1].axhline(0.0, color=FG, lw=1.0)
        axs[1].axhspan(-2, 2, color=DIM, alpha=0.15, lw=0)
    if each is not None:
        styles.append(
            mpl.lines.Line2D(
                [], [], color=grey, lw=1.2, alpha=0.5, label=f"each {'realisation' if theory else 'box'} ({each.shape[0]})"
            )
        )
    if r_bao is not None:
        for ax in axs:
            ax.axvline(r_bao, color=DIM, ls=":", lw=1.8)
        styles.append(mpl.lines.Line2D([], [], color=DIM, ls=":", lw=1.8, label=r"$r_{\rm BAO}$"))
    r2 = r**2
    y = [(r2 * (X - Xe)).ravel(), (r2 * (X + Xe)).ravel()] + ([(r2 * Xth).ravel()] if theory else [])
    ref_handles = []
    if reference is not None:
        if unit != "cMpc":
            raise ValueError("the reference is drawn in comoving units: unit='cMpc'.")
        wedges, h = _reference_wedges(reference, reference_h)
        x_max = float(np.nanmax(r)) * 1.02
        for rw in wedges:
            match = [w for w in range(nm) if np.isclose(rw["mu_min"], mu[w]) and np.isclose(rw["mu_max"], mu[w + 1])]
            c = cols[match[0]] if match else grey
            ok = np.isfinite(rw["r2xi"]) & (rw["r"] <= x_max)
            e = np.where(np.isfinite(rw["sigma"]), rw["sigma"], 0.0)[ok]
            axs[0].errorbar(
                rw["r"][ok],
                rw["r2xi"][ok],
                e,
                fmt="s",
                ms=7,
                mfc="none",
                mec=c,
                mew=1.6,
                color=c,
                elinewidth=1.2,
                capsize=2,
                zorder=5,
            )
            y += [rw["r2xi"][ok] - e, rw["r2xi"][ok] + e]
            ref_handles.append(
                mpl.lines.Line2D(
                    [],
                    [],
                    color=c,
                    marker="s",
                    mfc="none",
                    mew=1.6,
                    ms=7,
                    ls="none",
                    label=_wedge_label(rw["mu_min"], rw["mu_max"]),
                )
            )
    if each is not None or reference is not None:  # the limits of the means, their errors and the references
        yy = np.concatenate(y)
        lo, hi = float(np.nanmin(yy)), float(np.nanmax(yy))
        axs[0].set_ylim(lo - 0.12 * (hi - lo), hi + 0.12 * (hi - lo))
    axs[0].axhline(0.0, color=DIM, lw=0.8)
    axs[0].set(ylabel=lab["r2"])
    axs[-1].set(xlabel=lab["r"])
    if kms_per_cMpc is not None:
        (velocity_axis if unit == "cMpc" else comoving_axis)(axs[0], kms_per_cMpc, "r")
    wedges_h = [
        mpl.lines.Line2D([], [], color=cols[w], marker="o", ms=7, lw=2.0, label=_wedge_label(mu[w], mu[w + 1])) for w in range(nm)
    ]
    fig.legend(handles=wedges_h, loc="outside upper left", ncol=2, fontsize=_LEGEND_08, title="wedges", title_fontsize=_LEGEND_08)
    fig.legend(handles=styles, loc="outside upper right", ncol=1, fontsize=_LEGEND_08 - 1)
    if ref_handles:
        fig.legend(
            handles=ref_handles,
            loc="outside lower center",
            ncol=len(ref_handles),
            fontsize=_LEGEND_08 - 1,
            title=f"reference (open squares): {reference['label']}, h = {h:g}",
            title_fontsize=_LEGEND_08 - 1,
        )
    return fig, axs, out


def plot_overflux_3d(
    flux3d: dict,
    axes: dict,
    *,
    kms_per_cMpc: float | None = None,
    r_bao: float | None = None,
    names=None,
    each: str | None = "all",
    figsize=(22, 9),
):
    """k P(|k|) and r^2 xi(|r|) of the overflux of each line and of the total on the lattice of skewers, in comoving
    units (notebook 08, C; the line of sight converted with u = period / L, ``SkewerRun.grid3d``): the overflux
    counterpart of ``plot_isotropic_recovery``, with its estimators (every mode of the lattice inside the sphere of its
    smallest Nyquist frequency in |k| bins, every pair of pixels up to half the box in |r| bins) and its styles (every box
    faint, the mean over the boxes with the error of the mean as points joined by thin lines; one colour per line,
    ``ENTITY_COLORS``, the total ``FG``); no residual row, as the overflux has no expectation; no titles.

    flux3d : {field: ``ensemble_mean`` of the ``GRID_KEYS`` statistics} and axes : the bins (``run_skewers``'s ``flux3d``
    and ``axes3d``; k [1/cMpc], r [cMpc]); the last field is the total. names : {field: legend label}. each : the boxes
    drawn faint, "all" (every field), "total" (the total only) or None. kms_per_cMpc : top axes in s/km and km/s for
    reference (``velocity_axis``). ``r_bao`` [cMpc]: a dotted line. Returns (fig, axs dict(P, xi))."""
    lab = _units("cMpc")
    fields = list(flux3d)
    names = names or {}
    total = fields[-1]
    fig, (axP, axX) = plt.subplots(1, 2, figsize=figsize, layout="constrained")
    k, r = axes["k_iso"], axes["r_iso"]
    n = flux3d[total]["n_real"]
    for j, f in enumerate(fields):
        c = FG if f == total else ENTITY_COLORS.get(f, PALETTE_CYCLE[j])
        e = flux3d[f]
        faint = each == "all" or (each == "total" and f == total)
        ok = np.isfinite(e["P_iso"]) & (e["P_iso"] > 0)
        okr = np.isfinite(e["xi_iso"]) & (r > 0)
        lbl = names.get(f, f)
        if faint:
            for y in e["P_iso_each"]:
                axP.plot(k[ok], (k * y)[ok], color=c, **_EACH)
            for y in e["xi_iso_each"]:
                axX.plot(r[okr], (r**2 * y)[okr], color=c, **_EACH)
        axP.errorbar(k[ok], (k * e["P_iso"])[ok], (k * e["P_iso_err"])[ok], color=c, label=lbl, **dict(_MEAN, ms=6))
        axX.errorbar(r[okr], (r**2 * e["xi_iso"])[okr], (r**2 * e["xi_iso_err"])[okr], color=c, **dict(_MEAN, ms=6))
    if each:
        axP.plot([], [], color=DIM, lw=1.2, alpha=0.5, label=f"each box ({n})")
    axP.set(xscale="log", yscale="log", xlabel=lab["k"], ylabel=lab["kP"])
    axX.axhline(0.0, color=DIM, lw=0.8)
    axX.set(xlabel=lab["r"], ylabel=lab["r2"])
    if r_bao is not None:
        axX.axvline(r_bao, color=DIM, ls=":", lw=1.8)
        axX.text(
            r_bao,
            0.04,
            r" $r_{\rm BAO}$",
            transform=axX.get_xaxis_transform(),
            color=FG,
            fontsize=FONT_MIN["annotation"],
            ha="left",
            va="bottom",
        )
    if kms_per_cMpc is not None:
        velocity_axis(axP, kms_per_cMpc, "k")
        velocity_axis(axX, kms_per_cMpc, "r")
    axP.legend(fontsize=FONT_MIN["legend"], loc="lower left")
    return fig, dict(P=axP, xi=axX)


def _native_map_axes(ax, u: float, kind: str, x_max: float, y_max: float):
    """The axes of the (perp, par) maps of the overflux (notebook 08, C): perpendicular across in comoving units,
    parallel upwards in velocity units, the origin at the bottom-left corner, equal comoving lengths (aspect), and the
    reference axes: the perpendicular in velocity units on top, the parallel in comoving units on the right. kind "r":
    cMpc and km/s; "k": cMpc^-1 and s/km. x_max, y_max: the axis limits in the units of the axes."""
    if kind == "r":
        ax.set(xlabel=r"$r_\perp$ [cMpc]", ylabel=r"$r_\parallel$ (line of sight) [km s$^{-1}$]", aspect=1.0 / u)
        _secondary(ax, (lambda x: x * u, lambda v: v / u), "top", r"$r_\perp$ [km s$^{-1}$] (reference, $v = u\,x$)")
        _secondary(ax, (lambda v: v / u, lambda x: x * u), "right", r"$r_\parallel$ [cMpc] (reference)")
    else:
        ax.set(xlabel=r"$k_\perp$ [cMpc$^{-1}$]", ylabel=r"$k_\parallel$ (line of sight) [s km$^{-1}$]", aspect=u)
        _secondary(ax, (lambda k: k / u, lambda k: k * u), "top", r"$k_\perp$ [s km$^{-1}$] (reference, $k_v = k_x/u$)")
        _secondary(ax, (lambda k: k * u, lambda k: k / u), "right", r"$k_\parallel$ [cMpc$^{-1}$] (reference)")
    ax.set(xlim=(0.0, x_max), ylim=(0.0, y_max))


def plot_overflux_rpar_rperp(
    res,
    *,
    kms_per_cMpc: float,
    r_bao: float | None = None,
    delta_v: float | None = None,
    mu_edges=None,
    n_sigma: float = 1.0,
    figsize=(14, 12.5),
):
    """r^2 xi(r_perp, r_par) of one overflux on the lattice of skewers (notebook 08, C), the measured panel of
    ``plot_rpar_rperp_maps``: a linear, diverging colour scale centred on 0, three iso-r^2 xi levels in green with the
    band where the level lies within the mean +- ``n_sigma`` times its error (the error of the mean over the boxes), the
    boundaries of the wedges of mu as white rays with a coloured sector per wedge near the origin; no expectation; no
    title.

    Axes: r_perp across in cMpc and r_par upwards in km/s, their native units, the origin at the bottom-left corner, equal
    comoving lengths; for reference, r_perp in km/s on top and r_par in cMpc on the right, through u =
    ``kms_per_cMpc``. r^2 xi [cMpc^2] and mu combine both in comoving units (r_par / u). res : one entry of
    ``run_skewers``'s ``flux3d`` with ``res["axes"]`` = its ``axes3d`` (comoving; r_par in bins of 3 cells, r_perp in
    bins of the transverse spacing, both up to half the box). ``r_bao`` [cMpc]: a dotted circle (comoving). ``delta_v``
    [km/s]: the separation of two lines (Si III - Lya) folded into [0, period / 2], where their cross term correlates the
    same gas: diamonds on the line of sight (r_perp = 0) at r_par = delta_v and, with ``r_bao``, at the BAO echo of the
    cross term, delta_v + u r_bao, and at |u r_bao - delta_v|, each only when it lies inside the map. Returns (fig, ax,
    markers {label: r_par [km/s]} of the drawn markers)."""
    u = float(kms_per_cMpc)
    a = res["axes"]
    ep, eq = a["rpar_edges"], a["rperp_edges"]
    X, Xe = res["xi_map"], res["xi_map_err"]
    rcp, rcq = 0.5 * (ep[1:] + ep[:-1]), 0.5 * (eq[1:] + eq[:-1])
    r2 = rcp[:, None] ** 2 + rcq[None, :] ** 2
    M = r2 * X
    norm, levels, extend = _r2xi_scale(M[np.isfinite(M)])
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    im = ax.pcolormesh(eq, u * ep, np.ma.masked_invalid(M), cmap=CMAPS["diverging"], norm=norm, shading="flat")
    _contour_band(ax, rcq, u * rcp, M, r2 * Xe, levels, n_sigma)
    _contours(ax, rcq, u * rcp, M, levels, _fmt_r2xi, colors=PALETTE["green"], linewidths=1.8)
    handles = [
        mpl.lines.Line2D([], [], color=PALETTE["green"], lw=1.8, label=r"iso-$r^2\xi$ contours"),
        _band_handle(res["n_real"], "boxes", n_sigma),
        _mu_rays(ax, a["mu_edges"] if mu_edges is None else mu_edges, min(ep[-1], eq[-1]), y_scale=u),
    ]
    if r_bao is not None:
        t = np.linspace(0, np.pi / 2, 200)
        ax.plot(r_bao * np.cos(t), u * r_bao * np.sin(t), color=FG, ls=":", lw=1.8)
        handles.append(mpl.lines.Line2D([], [], color=FG, ls=":", lw=1.8, label=r"$r = r_{\rm BAO}$"))
    drawn = {}
    if delta_v is not None:
        d = abs(float(delta_v))
        marks = [(d, "D", "none", r"Si III–Ly$\alpha$: $r_\parallel = \Delta v$")]
        if r_bao is not None:
            marks += [
                (d + u * r_bao, "d", "none", r"BAO echo: $r_\parallel = \Delta v + u\,r_{\rm BAO}$"),
                (abs(u * r_bao - d), "d", FG, r"$r_\parallel = |u\,r_{\rm BAO} - \Delta v|$"),
            ]
        for y, m, fc, lbl in marks:
            if 0.0 <= y <= u * ep[-1]:
                ax.plot([0.0], [y], marker=m, ms=14, mfc=fc, mec=FG, mew=2.4, zorder=11, clip_on=False, ls="none")
                handles.append(mpl.lines.Line2D([], [], color=FG, marker=m, ms=12, mfc=fc, mew=2.2, ls="none", label=lbl))
                drawn[lbl] = y
    _native_map_axes(ax, u, "r", eq[-1], u * ep[-1])
    fig.legend(handles=handles, loc="outside lower center", ncol=2, fontsize=FONT_MIN["legend"])
    fig.colorbar(im, ax=ax, label=r"$r^2\xi(r_\perp, r_\parallel)$ [cMpc$^2$]", shrink=0.9, pad=0.04, extend=extend)
    return fig, ax, drawn


def plot_overflux_kpar_kperp(res, *, kms_per_cMpc: float, mu_edges=None, n_sigma: float = 1.0, figsize=(14, 12.5)):
    """k P(k_perp, k_par) of one overflux on the lattice of skewers, measured (notebook 08, C): the companion of
    ``plot_overflux_rpar_rperp`` in Fourier space, in the style of ``plot_kpar_kperp_maps`` (logarithmic colour scale,
    three iso-kP levels in green with the band where the level lies within the mean +- ``n_sigma`` times its error, the
    wedges of mu = k_par / k as white rays with coloured sectors), measurement only; no title.

    Axes: k_perp across in cMpc^-1 and k_par upwards in s/km, their native units, linear from the origin at the
    bottom-left corner, equal comoving lengths; for reference, k_perp in s/km on top and k_par in cMpc^-1 on the right,
    through u = ``kms_per_cMpc`` (k_v = k_x / u). k = |k| [cMpc^-1] (the mode-weighted centre of each cell) and mu
    combine both in comoving units. res : one entry of ``run_skewers``'s ``flux3d`` with ``res["axes"]`` = its ``axes3d``
    (the map in linear bins of the fundamental mode up to the smallest Nyquist frequency, ``SkewerRun.grid3d``). Returns
    (fig, ax)."""
    u = float(kms_per_cMpc)
    a = res["axes"]
    ep, eq = a["kpar_edges"], a["kperp_edges"]
    kc = np.hypot(a["kpar_map"], a["kperp_map"])
    M, E = kc * res["P_map"], kc * res["P_map_err"]
    good = np.isfinite(M) & (M > 0)
    norm = mpl.colors.LogNorm(np.nanmin(M[good]), np.nanmax(M[good]))
    levels = _three_levels(norm.vmin, norm.vmax, log=True)
    kcp, kcq = 0.5 * (ep[1:] + ep[:-1]), 0.5 * (eq[1:] + eq[:-1])
    cmap_ = plt.get_cmap(CMAPS["diverging"]).copy()
    cmap_.set_bad((0, 0, 0, 0))
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    im = ax.pcolormesh(eq, ep / u, np.ma.masked_invalid(np.where(good, M, np.nan)), cmap=cmap_, norm=norm, shading="flat")
    _contour_band(ax, kcq, kcp / u, M, E, levels, n_sigma)
    _contours(ax, kcq, kcp / u, M, levels, _fmt_r2xi, colors=PALETTE["green"], linewidths=1.8)
    handles = [
        mpl.lines.Line2D([], [], color=PALETTE["green"], lw=1.8, label=r"iso-$kP$ contours"),
        _band_handle(res["n_real"], "boxes", n_sigma),
        _mu_rays(ax, a["mu_edges"] if mu_edges is None else mu_edges, min(ep[-1], eq[-1]), y_scale=1.0 / u, kind="k"),
    ]
    _native_map_axes(ax, u, "k", eq[-1], ep[-1] / u)
    fig.legend(handles=handles, loc="outside lower center", ncol=2, fontsize=FONT_MIN["legend"])
    fig.colorbar(im, ax=ax, label=r"$k\,P(k_\perp, k_\parallel)$ [cMpc$^2$]", shrink=0.9, pad=0.04)
    return fig, ax


def _binned_ensemble(k, each, edges, *, per_box: bool = False):
    """Mean over the realisations of the means of ``each`` (n, N_k) in the bins of k: (centres, mean, error of the mean)
    and, with ``per_box``, the binned values of every realisation (n, n_bins)."""
    return _binned(np.asarray(k, dtype=float), np.asarray(each), np.asarray(edges, dtype=float), per_box=per_box)


def plot_p1d_reference(
    p1d: dict,
    reference: dict,
    *,
    names=None,
    fields=None,
    k_max: float | None = None,
    each: bool = True,
    kms_per_cMpc: float | None = None,
    figsize=(12, 10),
):
    """k P1D(k) / pi against k [s/km], linear k and logarithmic P, the convention of Fig. 1 of the reference (Chaves-Montero
    et al., DESI DR1 P1D; notebook 08, D1): the P1D of each overflux of the toy boxes (``run_skewers``'s ``p1d``,
    binned in k to the spacing of the reference: every box faint (``each``), the mean over the boxes with the error of
    the mean as points joined by thin lines; one colour per line, the total ``FG``) and the reference measurement
    (``utils.reference_data``: dict(k, kP_pi, label)) as grey points. fields : the rows of ``p1d["p1d"]`` (default
    ``simulations.toy_boxes.OVERFLUX_FIELDS``); names : {field: legend label}; k_max : the right end of the k axis
    (default 1.6 times the largest reference k); kms_per_cMpc : a top axis k_x = u k_v [cMpc^-1] (``comoving_axis``).
    No title. Returns (fig, ax)."""
    from skewerjax.simulations.toy_boxes import OVERFLUX_FIELDS

    fields = list(fields or OVERFLUX_FIELDS)
    names = names or {}
    k_ref, y_ref = np.asarray(reference["k"]), np.asarray(reference["kP_pi"])
    k_max = float(k_max or 1.6 * k_ref.max())
    k = np.asarray(p1d["k"])
    step = float(np.median(np.diff(k_ref)))
    edges = np.arange(0.5 * k[1], k_max + step, step)
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    for j, f in enumerate(fields):
        kc, m, e, b = _binned_ensemble(k, np.asarray(p1d["p1d_each"])[:, j], edges, per_box=True)
        c = FG if f == fields[-1] else ENTITY_COLORS.get(f, PALETTE_CYCLE[j])
        if each:
            for y in b:
                ax.plot(kc, kc * y / np.pi, color=c, **_EACH)
        ax.errorbar(kc, kc * m / np.pi, kc * e / np.pi, color=c, label=names.get(f, f), **dict(_MEAN, ms=5))
    if each:
        ax.plot([], [], color=DIM, lw=1.2, alpha=0.5, label=f"each box ({np.shape(p1d['p1d_each'])[0]})")
    ax.plot(k_ref, y_ref, "s", color=ENTITY_COLORS["reference"], ms=7, mec=FG, mew=0.8, label=reference["label"], zorder=7)
    ax.set(
        xlim=(0.0, k_max),
        yscale="log",
        xlabel=r"$k_\parallel$ [s km$^{-1}$]",
        ylabel=r"$k_\parallel P_{\rm 1D}(k_\parallel) / \pi$",
    )
    if kms_per_cMpc is not None:
        comoving_axis(ax, kms_per_cMpc, "k", label=r"$k_\parallel$ [cMpc$^{-1}$]")
    fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center", ncol=2, fontsize=FONT_MIN["legend"])
    return fig, ax


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
