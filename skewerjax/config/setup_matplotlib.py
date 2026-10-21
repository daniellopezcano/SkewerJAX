def matplotlib_default_config():

    font = {
        'family' : 'serif',
        'weight' : 'normal',
        'size'   : 14
    }
    
    rcnew = {
        "mathtext.fontset" : "cm", 
        "text.usetex": False,

        'figure.frameon': True,
        'axes.linewidth': 2.,

        "axes.titlesize" : 32, 
        "axes.labelsize" : 28,
        "legend.fontsize" : 28,
        'legend.fancybox': True,
        'lines.linewidth': 2.5,

        'xtick.alignment': 'center',
        'xtick.bottom': True,
        'xtick.color': 'black',
        'xtick.direction': 'in',
        'xtick.labelbottom': True,
        'xtick.labelsize': 24, #17.5,
        'xtick.labeltop': False,
        'xtick.major.bottom': True,
        'xtick.major.pad': 6.0,
        'xtick.major.size': 14.0,
        'xtick.major.top': True,
        'xtick.major.width': 1.5,
        'xtick.minor.bottom': True,
        'xtick.minor.pad': 3.4,
        'xtick.minor.size': 7.0,
        'xtick.minor.top': True,
        'xtick.minor.visible': True,
        'xtick.minor.width': 1.0,
        'xtick.top': True,

        'ytick.alignment': 'center_baseline',
        'ytick.color': 'black',
        'ytick.direction': 'in',
        'ytick.labelleft': True,
        'ytick.labelright': False,
        'ytick.labelsize': 24, #17.5,
        'ytick.left': True,
        'ytick.major.left': True,
        'ytick.major.pad': 6.0,
        'ytick.major.right': True,
        'ytick.major.size': 14.0,
        'ytick.major.width': 1.5,
        'ytick.minor.left': True,
        'ytick.minor.pad': 3.4,
        'ytick.minor.right': True,
        'ytick.minor.size': 7.0,
        'ytick.minor.visible': True,
        'ytick.minor.width': 1.0,
        'ytick.right': True
    }
    
    return font, rcnew


# =============================================================================
# Paul Tol's colour-blind-safe palettes (https://tol-colors.readthedocs.io), defined here so that no package is
# needed. Additive: `matplotlib_default_config` above is unchanged; notebooks opt in with `matplotlib_tol_config`.
# =============================================================================
import matplotlib as _mpl
from cycler import cycler as _cycler

TOL_BRIGHT = {
    "blue": "#4477AA", "red": "#EE6677", "green": "#228833", "yellow": "#CCBB44",
    "cyan": "#66CCEE", "purple": "#AA3377", "grey": "#BBBBBB",
}
TOL_BRIGHT_CYCLE = list(TOL_BRIGHT.values())

# Tol's sequential and diverging schemes (colour lists from the tol-colors documentation)
_TOL_CMAPS = {
    "iridescent": (["#FEFBE9", "#FCF7D5", "#F5F3C1", "#EAF0B5", "#DDECBF", "#D0E7CA", "#C2E3D2", "#B5DDD8", "#A8D8DC",
                    "#9BD2E1", "#8DCBE4", "#81C4E7", "#7BBCE7", "#7EB2E4", "#88A5DD", "#9398D2", "#9B8AC4", "#9D7DB2",
                    "#9A709E", "#906388", "#805770", "#684957", "#46353A"], "#999999"),
    "YlOrBr": (["#FFFFE5", "#FFF7BC", "#FEE391", "#FEC44F", "#FB9A29", "#EC7014", "#CC4C02", "#993404", "#662506"], "#888888"),
    "sunset": (["#364B9A", "#4A7BB7", "#6EA6CD", "#98CAE1", "#C2E4EF", "#EAECCC", "#FEDA8B", "#FDB366", "#F67E4B",
                "#DD3D2D", "#A50026"], "#FFFFFF"),
    "BuRd": (["#2166AC", "#4393C3", "#92C5DE", "#D1E5F0", "#F7F7F7", "#FDDBC7", "#F4A582", "#D6604D", "#B2182B"], "#FFEE99"),
    "PRGn": (["#762A83", "#9970AB", "#C2A5CF", "#E7D4E8", "#F7F7F7", "#D9F0D3", "#ACD39E", "#5AAE61", "#1B7837"], "#FFEE99"),
}


def tol_cmap(name: str = "iridescent", reverse: bool = False):
    """A Tol colour map as a matplotlib ``LinearSegmentedColormap``: sequential ``"iridescent"``, ``"YlOrBr"``;
    diverging ``"sunset"``, ``"BuRd"``, ``"PRGn"``. ``reverse`` flips it."""
    colors, bad = _TOL_CMAPS[name]
    cm = _mpl.colors.LinearSegmentedColormap.from_list(f"tol_{name}" + ("_r" if reverse else ""), colors[::-1] if reverse else colors)
    cm.set_bad(bad)
    return cm


# One colour per entity and one line style per mode, shared by all notebooks.
ENTITY_COLORS = {
    # spectral lines / species
    "HI": TOL_BRIGHT["blue"], "Lya": TOL_BRIGHT["blue"], "HI-LyA": TOL_BRIGHT["blue"],
    "SiIII": TOL_BRIGHT["red"], "SiIII-1206": TOL_BRIGHT["red"],
    "HI-LyB": TOL_BRIGHT["cyan"], "LyB": TOL_BRIGHT["cyan"],
    # codes and references
    "fake_spectra": TOL_BRIGHT["grey"], "analytic": TOL_BRIGHT["grey"], "reference": TOL_BRIGHT["grey"],
    # gas quantities
    "density": TOL_BRIGHT["green"], "temperature": TOL_BRIGHT["purple"], "velocity": TOL_BRIGHT["cyan"],
    # geometry / extraction
    "voronoi": TOL_BRIGHT["blue"], "tophat": TOL_BRIGHT["yellow"],
}
MODE_STYLES = {
    # SkewerJAX default: solid; fake_spectra reproduced ("matched"): dashed; fake_spectra itself / analytic
    # references: thick, light, solid underneath; alternatives and isolated effects: dotted / dash-dotted
    "default": dict(ls="-", lw=2.0),
    "matched": dict(ls="--", lw=2.0),
    "reference": dict(ls="-", lw=5.0, alpha=0.55),
    "alternative": dict(ls=":", lw=2.2),
    "isolated": dict(ls="-.", lw=1.8),
}


def matplotlib_tol_config():
    """``matplotlib_default_config`` with Tol's bright colour cycle and legend / title sizes that suit
    multi-panel figures. Returns (font, rcnew) like ``matplotlib_default_config``; use as
    ``font, rc = matplotlib_tol_config(); plt.rc('font', **font); plt.rcParams.update(rc)``."""
    font, rcnew = matplotlib_default_config()
    rcnew = dict(rcnew)
    rcnew.update({
        "axes.prop_cycle": _cycler(color=TOL_BRIGHT_CYCLE),
        "image.cmap": "cividis",
        "axes.titlesize": 22, "axes.labelsize": 22, "legend.fontsize": 15,
        "xtick.labelsize": 18, "ytick.labelsize": 18,
        "xtick.major.size": 9.0, "ytick.major.size": 9.0, "xtick.minor.size": 4.5, "ytick.minor.size": 4.5,
        "lines.linewidth": 2.0, "legend.framealpha": 0.9, "legend.edgecolor": "0.6",
        "axes.grid": False, "figure.dpi": 80, "savefig.dpi": 150,
    })
    return font, rcnew


def add_panel_letters(axes, x: float = 0.012, y: float = 0.975, size: int = 20, start: str = "a"):
    """Write (a), (b), ... in the upper-left corner of every axis of ``axes`` (any iterable or array)."""
    import numpy as _np
    for i, ax in enumerate(_np.ravel(axes)):
        ax.text(x, y, f"({chr(ord(start) + i)})", transform=ax.transAxes, fontsize=size, fontweight="bold", zorder=20,
                va="top", ha="left", bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.85))


def residual_band(ax, tol: float, label: str = None, zero: bool = True):
    """Zero line and a shaded +/- ``tol`` band on a residual panel."""
    if zero:
        ax.axhline(0.0, color="k", lw=1.0)
    ax.axhspan(-tol, tol, color=TOL_BRIGHT["grey"], alpha=0.35, lw=0, label=label)


def takeaway(fig, text: str, size: int = 20):
    """A one-line takeaway as the figure title."""
    fig.suptitle(text, fontsize=size)
