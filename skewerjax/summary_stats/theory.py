"""Theory helpers of the statistics of a periodic box: what an isotropic xi(r) or P(k) predicts, from callables only.

* ``sphere_area_in_cube``, ``integral_constraint``: the mean of xi over a cube of side L, xi_bar_V = V^-1 int_cube xi(|x|)
  d^3x, the constant the k = 0 mode of a periodic box removes (Peebles 1980, sec. 31; Landy & Szalay 1993; Bagla & Prasad
  2006, arXiv:astro-ph/0601320; de Mattia & Ruhlmann-Kleider 2019, arXiv:1904.08851): a 1D integral with the exact area of
  the sphere inside the cube;
* ``bao_maxima``: the positions of the BAO maxima in k, of the bump term W(k) of a model or of j_0(k r_d) (notebooks 08 to
  11; the sound horizon r_d: ``core.sound_horizon_drag``);
* ``pair_imprint``: the Fourier imprint of the separation of two lines, in P1D and in the isotropic P(|k|) (notebooks 10, 11).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.special import spherical_jn


def sphere_area_in_cube(r, L: float):
    """The area [units of L^2] of the part inside the cube [-L/2, L/2]^3 of the sphere of radius r centred on the cube.

    With a = L / 2 and c = a / r: 4 pi r^2 for r <= a; minus six caps of area 2 pi r (r - a) for a <= r <= sqrt(2) a; for
    sqrt(2) a <= r <= sqrt(3) a the caps overlap and each of the eight octants holds a curved triangle bounded by three
    small circles, of solid angle (Gauss-Bonnet) 3 beta - pi + 3 c phi, with beta = arccos[c^2 / (1 - c^2)] the angle at
    its vertices and phi = pi / 2 - 2 arctan[sqrt(1 - 2 c^2) / c] the angle of each arc about its axis; 0 beyond sqrt(3) a."""
    r = np.asarray(r, dtype=float)
    a = 0.5 * float(L)
    out = np.zeros_like(r)
    inner = r <= a
    out[inner] = 4.0 * np.pi * r[inner] ** 2
    mid = (r > a) & (r <= np.sqrt(2.0) * a)
    out[mid] = 4.0 * np.pi * r[mid] ** 2 - 12.0 * np.pi * r[mid] * (r[mid] - a)
    far = (r > np.sqrt(2.0) * a) & (r < np.sqrt(3.0) * a)
    c = a / r[far]
    s = np.sqrt(np.maximum(1.0 - 2.0 * c**2, 0.0))
    beta = np.arccos(np.clip(c**2 / (1.0 - c**2), -1.0, 1.0))
    phi = 0.5 * np.pi - 2.0 * np.arctan(s / c)
    out[far] = 8.0 * r[far] ** 2 * (3.0 * beta - np.pi + 3.0 * c * phi)
    return out


def _j0_maxima(x_max: float) -> np.ndarray:
    """The positive maxima of j_0(x) at 1 < x <= x_max, x = 7.725, 14.066, 20.371, ...: j_1(x) = 0 with j_0(x) > 0."""
    x = np.linspace(1.0, float(x_max) + 1.0, int(max(64, 8 * float(x_max))))
    j1 = spherical_jn(1, x)
    roots = [brentq(lambda t: spherical_jn(1, t), x[i], x[i + 1]) for i in np.flatnonzero(j1[:-1] * j1[1:] < 0)]
    return np.array([t for t in roots if spherical_jn(0, t) > 0])


def integral_constraint(xi: Callable, L: float, *, n_panels: int = 2048, n_geom: int = 64, nodes: int = 8) -> float:
    """xi_bar_V = V^-1 int_{[-L/2, L/2]^3} xi(|x|) d^3x = V^-1 int_0^{sqrt(3) L / 2} xi(r) A(r) dr (``sphere_area_in_cube``).

    The term of the k = 0 mode of a periodic box (notes above, module docstring): the expectation of xi on the box is
    xi(r) - xi_bar_V plus the fluctuation of the images. Finite for a red spectrum with xi ~ r^-gamma, gamma < 3, while
    P(0) / V itself diverges.

    xi : a vectorised callable of r [the units of L], any scale of feature (a narrow BAO bump is resolved by the panels);
    n_panels : panels of ``nodes``-point Gauss-Legendre quadrature, uniform and split at the kinks of A(r); n_geom : extra
    geometric panels from L * 1e-8 to the first uniform panel, for an integrable divergence at r = 0."""
    L = float(L)
    breaks = np.array([0.0, 0.5, np.sqrt(0.5), 0.5 * np.sqrt(3.0)]) * L
    edges = [
        np.linspace(breaks[i], breaks[i + 1], max(2, int(n_panels * (breaks[i + 1] - breaks[i]) / breaks[-1])) + 1)
        for i in range(3)
    ]
    grid = np.unique(np.concatenate(edges))
    first = grid[1]
    grid = np.unique(np.concatenate([np.geomspace(L * 1e-8, first, int(n_geom)), grid[1:]]))
    t, w = np.polynomial.legendre.leggauss(int(nodes))
    mid, half = 0.5 * (grid[1:] + grid[:-1]), 0.5 * np.diff(grid)
    r = (mid[:, None] + half[:, None] * t).ravel()
    wt = (half[:, None] * w).ravel()
    return float(np.sum(wt * np.asarray(xi(r), dtype=float) * sphere_area_in_cube(r, L)) / L**3)


def bao_maxima(
    k_max: float, *, window: Callable | None = None, r_bao: float | None = None, k_min: float = 0.0, kms_per_cMpc=None
) -> np.ndarray:
    """The positions of the BAO maxima in k, sorted, in k_min < k <= ``k_max`` [1/cMpc]; divided by ``kms_per_cMpc`` (u)
    in velocity units, k_par = k / u [s/km] (``k_max`` and ``k_min`` stay in 1/cMpc).

    * window : a callable W(k), e.g. ``PowerLawGRF.bao_window``: the positive local maxima of the model's own bump term
      A_BAO W(k) at k > 0, found numerically (a scan with 32 points per BAO period when ``r_bao`` is given as well, else
      4000 points; refined by a bounded minimisation);
    * r_bao alone : the BAO scale r_d [cMpc], the maxima of j_0(k r_d), that is j_1(x) = 0 with j_0(x) > 0: x = 7.725,
      14.066, 20.371, ... (the BAO feature of a sharp shell of radius r_d)."""
    k_max, k_min = float(k_max), float(k_min)
    if window is None:
        if r_bao is None:
            raise ValueError("bao_maxima needs window (a callable W(k)) or r_bao.")
        r = float(r_bao)
        k = _j0_maxima(k_max * r) / r
    else:
        n = 4000 if r_bao is None else int(max(64, 32 * k_max * float(r_bao) / (2.0 * np.pi)))
        g = np.linspace(max(k_min, 1e-6 * k_max), k_max, n)
        W = lambda q: np.asarray(window(q), dtype=float)  # noqa: E731
        y = W(g)
        i = np.flatnonzero((y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:]) & (y[1:-1] > 0)) + 1
        k = np.array(
            [minimize_scalar(lambda q: -float(W(np.array([q]))[0]), bounds=(g[j - 1], g[j + 1]), method="bounded").x for j in i]
        )
    k = k[(k > k_min) & (k <= k_max)]
    return k if kms_per_cMpc is None else k / float(kms_per_cMpc)


PAIR_IMPRINTS = ("p1d", "iso")


def pair_imprint(
    separation: float,
    k_max: float,
    *,
    kind: str = "p1d",
    period: float | None = None,
    n_harmonics: int | None = None,
    k_min: float = 0.0,
) -> np.ndarray:
    """The maxima in k of the cross term of two lines a ``separation`` apart along the line of sight, sorted, in
    k_min < k <= ``k_max``. One line is a copy of the other shifted by Delta, delta_2(x) ~ delta_1(x + Delta), so their cross
    term multiplies the spectrum by a factor that oscillates with k (McDonald et al. 2006, Si III; notebooks 10 and 11, D5):

    * kind "p1d", along the line of sight: cos(k_par Delta), with maxima at k_par = 2 pi n / Delta, n = 1, 2, ...;
    * kind "iso", the isotropic P(|k|): the average of cos(k mu Delta) over the directions is j_0(k Delta), with maxima at
      k = x_m / Delta, x_m = 7.725, 14.066, 20.371, ... (``_j0_maxima``).

    separation : Delta, the unit of the result being its inverse (km/s -> s/km, cMpc -> 1/cMpc). period : the period of the
    line of sight in the same unit: Delta is folded into [0, period / 2] first, since the modes k = 2 pi m / period of a
    periodic box cannot tell a separation from its nearest image (cos(k Delta) = cos(k Delta_fold) on them). n_harmonics :
    keep the first n maxima only (None: all within the range). A zero separation has no imprint (an empty array)."""
    if kind not in PAIR_IMPRINTS:
        raise ValueError(f"kind must be one of {PAIR_IMPRINTS}, got {kind!r}.")
    if n_harmonics is not None and int(n_harmonics) < 1:
        raise ValueError(f"n_harmonics must be >= 1 or None, got {n_harmonics!r}.")
    d = abs(float(separation))
    if period is not None:
        m = d % float(period)
        d = min(m, float(period) - m)
    k_max, k_min = float(k_max), float(k_min)
    if d == 0.0 or k_max <= 0.0:
        return np.zeros(0)
    if kind == "p1d":
        k = 2.0 * np.pi * np.arange(1, int(k_max * d / (2.0 * np.pi)) + 2) / d
    else:
        k = _j0_maxima(k_max * d) / d
    k = k[(k > k_min) & (k <= k_max)]
    return k if n_harmonics is None else k[: int(n_harmonics)]
