"""Toy boxes (notebook 08): periodic 3D Gaussian fields of known spectrum, their gas, skewers through them, and the
statistics of many realisations, streamed (generate, measure, discard) and returned in memory: nothing is stored, every
call recomputes.

* ``ToyBoxes``: the configuration and realisation i, regenerated from its seed ``base_seed + i`` whenever it is needed
  (``fields``, ``cells``): the fields are never stored. The spectrum is a power law plus a BAO term (a Gaussian bump in
  xi, its exact transform A_bao W(k) in P, ``PowerLawGRF``; valid for bao_ratio <= R_max), isotropic (no Kaiser or
  Fingers-of-God term), normalised to ``sigma_delta``; the gas of every cell, both species being the same gas (one temperature, one velocity):

      n_HI = n_bar_HI (1 + delta),   n_SiIII = r n_HI,   T = T0 (1 + delta)^(gamma - 1),
      v = 0 (``velocity="zero"``, the default)  or  v_k = i a H f (k / k^2) delta_k (``"linear"``: linear theory, notebook 04).

* ``field_statistics``: P and xi of every realisation on its grid (``summary_stats.GridStatistics3D``), in explicit bins
  (``ToyBoxes.bin_edges``: |k| and |r| edges, the map rows on the grid's own modes and separations, every bin inside the
  full-coverage limits ``ToyBoxes.limits``), against their exact expectations (with and without the BAO wiggle) and the
  exact Gaussian covariance of the isotropic xi bins;
  ``xi_chi2`` and ``xi_subset_offsets`` the chi^2 and the offset of the mean xi from its expectation.
* ``run_skewers``: per realisation, a regular sub-grid of skewers along y through the centres of the cell columns
  (``SkewerRun``) and tau of Lya and Si III (periodic, ``velocity_period=True``, both on the Lya axis,
  ``line_frame="reference"``; ``skewer_tau``); then, with the ensemble mean transmission at each pixel over every skewer
  of every realisation, delta = T / T_bar(v) - 1 (``summary_stats.overflux``), the 1D statistics by term and order
  (``OverfluxStatistics1D.summary``) and the 3D statistics of the overflux of each line and of the total on the lattice
  of skewers, in comoving units (``SkewerRun.grid3d``; ``skewer_statistics``), averaged over the realisations
  (``summary_stats.ensemble_1d``, ``ensemble_mean``); and the density sampled on the same lattice through the same 3D
  estimator, against its exact expectation (``lattice_density_theory``).
"""

from __future__ import annotations

import logging
import time
import warnings
from dataclasses import asdict, dataclass
from functools import cached_property

import jax
import jax.numpy as jnp
import numpy as np

from skewerjax.core import Cosmology, LineBundle, SkewerBundle, compute_optical_depth, phys
from skewerjax.core.absorbers import cell_pieces
from skewerjax.simulations.grf import PowerLawGRF, _bao_ratio_max
from skewerjax.simulations.toy_cells import grf_cells
from skewerjax.summary_stats import (
    GRID_KEYS,
    DeltaFieldBuilder,
    GridStatistics3D,
    OverfluxStatistics1D,
    ensemble_1d,
    ensemble_mean,
    mean_transmissions,
)
from skewerjax.summary_stats.grid3d import _rfft_weights

logger = logging.getLogger(__name__)

LINE_IDS = ("HI-LyA", "SiIII-1206")  # line 1, line 2 of the term labels "1", "2", "12"
TERM_LABELS = ("1", "2")
OVERFLUX_FIELDS = (*LINE_IDS, "total")  # the overfluxes of the 3D statistics: each line and the total
OVERFLUX_KEYS = GRID_KEYS  # their statistics: those of the fields
VELOCITY_MODES = ("zero", "linear")  # ToyBoxes.velocity


def bao_k_edges(k_f: float, k_max: float, k_w: float, *, k_1: float | None = None, n_log: int = 12, info: bool = False):
    """|k| bin edges that resolve BAO wiggles of period ~2 k_f [same units as the inputs]:

    * one bin per mode up to k_1 (edges at (j + 1/2) k_f; default k_1 = 3.5 k_f, rounded to such an edge);
    * bins of k_f / 4 from k_1 to k_2, the end of the wiggles: k_w (e.g. 3 / sigma_bao) rounded to the nearest multiple
      of k_f / 4 above k_1, so that every such bin is a full quarter;
    * ``n_log`` logarithmic bins from k_2 to k_max: the quarter bins end and the log bins start at the same k_2.

    When k_2 >= k_max, quarter bins up to k_max; when k_w <= k_1, no quarter bins (k_2 = k_1); when k_max <= k_1, one
    mode per bin up to k_max. ``info``: also return dict(k_1, k_2, k_w, n_mode, n_quarter, n_log) (the numbers of
    bins)."""
    k_1 = 3.5 * k_f if k_1 is None else (np.floor(k_1 / k_f - 0.5) + 0.5) * k_f
    if k_max <= k_1:
        lin = np.arange(0.5, k_max / k_f, 1.0) * k_f
        e = np.append(lin, k_max) if k_max - lin[-1] > 0.1 * k_f else lin
        k_1 = k_2 = float(e[-1])
        parts = (e, np.empty(0), np.empty(0))
    else:
        mode = np.arange(0.5, k_1 / k_f + 0.25, 1.0) * k_f
        k_2 = k_1 + max(0, int(np.round((k_w - k_1) / (0.25 * k_f)))) * 0.25 * k_f
        if k_2 >= k_max:
            q = np.arange(k_1 + 0.25 * k_f, k_max - 0.1 * k_f, 0.25 * k_f)
            parts = (mode, np.append(q, k_max), np.empty(0))
            k_2 = float(k_max)
        else:
            q = np.arange(k_1 + 0.25 * k_f, k_2 + 0.1 * k_f, 0.25 * k_f)
            log_edges = np.geomspace(k_2, k_max, int(n_log) + 1)[1:]
            parts = (mode, q, log_edges)
    e = np.concatenate(parts)
    if not info:
        return e
    n = dict(n_mode=parts[0].size - 1, n_quarter=parts[1].size, n_log=parts[2].size)
    return e, dict(k_1=float(k_1), k_2=float(k_2), k_w=float(k_w), **n)


def native_edges(spacing: float, n_max: int, start: int = 0) -> np.ndarray:
    """The edges of one bin per grid value j spacing, j = ``start`` .. ``n_max``, each centred on its value:
    (j -+ 1/2) spacing (the rows of the maps on a grid's own modes or separations)."""
    return (np.arange(int(start), int(n_max) + 2) - 0.5) * float(spacing)


def full_coverage(edges, hi: float, name: str = "bins", log=print) -> np.ndarray:
    """``edges`` without the bins that extend beyond ``hi``, the end of the range where every direction is available
    (|k| <= the smallest Nyquist frequency, |r| <= half the box); ``log`` reports a cut in one line."""
    e = np.asarray(edges, dtype=float)
    keep = e <= hi * (1 + 1e-9)
    if keep.all():
        return e
    if log:
        log(f"{name}: {int((~keep).sum())} edge(s) beyond the full-coverage limit {hi:.4g} cut (last edge {e[-1]:.4g})")
    return e[keep]


@dataclass(frozen=True)
class ToyBoxes:
    """The toy boxes of notebook 08: ``n_real`` periodic cubes of side ``box_cMpc`` on ``n_cells``^3 grids.

    Spectrum: P(k) = A_s (k / k_0)^(n_s - 1) + A_bao W(k), isotropic (``PowerLawGRF`` with A_bao = bao_ratio A_s,
    r_bao, sigma_bao [cMpc]): ``bao_ratio`` = R_bao is the height of the BAO bump in xi in units of A_s [cMpc^-3], and
    W the exact transform of the bump, which changes sign: the spectrum is valid (P >= 0 at every mode of the grid) for
    bao_ratio <= R_max (``bao_ratio_max``, independent of sigma_delta); a larger ratio raises a ``ValueError`` quoting
    R_max. A_s is set by ``sigma_delta`` = sqrt(variance on the grid). Gas: the module
    docstring, with n_bar_HI [comoving cm^-3], ``siiii_to_hi`` = r, T0 [K], ``gamma_minus_1``. ``z`` and ``Omega_m``
    (a flat cosmology, H0 = ``phys.H0_km_s_mpc``) set a(z) H(z), the Hubble flow of the skewers (their velocity axis).
    Realisation i is ``grf().sample(jax.random.PRNGKey(base_seed + i))``.

    velocity : "zero" (default) or "linear"
        The peculiar velocities of the gas: "zero" sets them to 0 in every cell (the flux then follows the density
        and the temperature only); "linear" uses linear theory, v_k = i a H f (k / k^2) delta_k [km/s] (notebook 04),
        with the growth rate ``f_growth``, which matters for this option only.
    """

    box_cMpc: float
    n_cells: int = 256
    n_real: int = 96
    base_seed: int = 1000
    sigma_delta: float = 0.15
    n_s: float = -1.0
    k_0: float = 2.0
    r_bao: float = 147.1
    sigma_bao: float = 1.0
    bao_ratio: float = 0.002
    n_bar_HI: float = 4e-13
    siiii_to_hi: float = 0.2
    T0: float = 1.5e4
    gamma_minus_1: float = 0.6
    z: float = 3.0
    Omega_m: float = 0.31
    f_growth: float = 1.0
    velocity: str = "zero"

    def __post_init__(self):
        if self.velocity not in VELOCITY_MODES:
            raise ValueError(f"velocity must be one of {VELOCITY_MODES}, got {self.velocity!r}.")
        r_max = self.bao_ratio_max
        if self.bao_ratio > r_max * (1.0 + 1e-12):
            raise ValueError(
                f"bao_ratio = {self.bao_ratio:g} exceeds R_max = {r_max:.6g}, the largest BAO ratio with P(k) >= 0 at every "
                f"mode of this grid (L = {self.box_cMpc:g} cMpc, N = {self.n_cells}): lower bao_ratio."
            )

    # ------------------------------------------------------------------ the spectrum and the realisations
    @property
    def bao_ratio_max(self) -> float:
        """R_max [cMpc^-3]: the largest bao_ratio with P(k) = A_s [(k / k_0)^(n_s - 1) + R_bao W(k)] >= 0 at every mode
        of the grid (``PowerLawGRF.bao_ratio_max``); independent of A_s, so of sigma_delta."""
        L, N = (float(self.box_cMpc),) * 3, (int(self.n_cells),) * 3
        return _bao_ratio_max(N, L, float(self.k_0), float(self.n_s), float(self.r_bao), float(self.sigma_bao))

    @cached_property
    def A_s(self) -> float:
        """The amplitude that gives sigma_delta on the grid (the variance is linear in A_s at fixed bao_ratio)."""
        return self.sigma_delta**2 / float(self._grf(1.0).variance())

    def _grf(self, A_s: float) -> PowerLawGRF:
        L, N = (float(self.box_cMpc),) * 3, (int(self.n_cells),) * 3
        return PowerLawGRF(
            A_s, self.k_0, self.n_s, L=L, N=N, A_bao=self.bao_ratio * A_s, r_bao=self.r_bao, sigma_bao=self.sigma_bao, los_axis=1
        )

    def grf(self) -> PowerLawGRF:
        """The ``PowerLawGRF`` of every realisation (line of sight: axis 1, y)."""
        return self._grf(self.A_s)

    @property
    def seeds(self) -> np.ndarray:
        """The seed of every realisation: base_seed + i, i = 0 .. n_real - 1."""
        return self.base_seed + np.arange(self.n_real)

    def key(self, i: int):
        return jax.random.PRNGKey(int(self.base_seed + i))

    def delta(self, i: int) -> np.ndarray:
        """Realisation i (n_cells^3), regenerated from its seed."""
        return np.asarray(self.grf().sample(self.key(i)))

    @property
    def dx(self) -> float:
        """Cell side [cMpc]."""
        return float(self.box_cMpc) / self.n_cells

    @property
    def aH(self) -> float:
        """a H(z) [km/s per cMpc] (flat cosmology, H0 = phys.H0_km_s_mpc): the Hubble flow at the start of a skewer."""
        E = np.sqrt(self.Omega_m * (1 + self.z) ** 3 + 1 - self.Omega_m)
        return phys.H0_km_s_mpc * E / (1 + self.z)

    @property
    def aHf(self) -> float:
        """a H(z) f [km/s per cMpc]: the factor of the linear-theory velocities (``velocity="linear"``)."""
        return self.aH * self.f_growth

    def cosmology(self) -> Cosmology:
        return Cosmology(H0=phys.H0, Omega_r_0=0.0, Omega_m_0=self.Omega_m, Omega_k_0=0.0, Omega_lambda_0=1.0 - self.Omega_m)

    def fields(self, i: int) -> dict:
        """The gas of realisation i on the grid (NumPy): delta, n {species: comoving cm^-3}, T [K], v (N, N, N, 3) [km/s]
        (zeros, or linear theory: ``velocity``). Both species are the same gas: one T and one v."""
        g = self.grf()
        d = g.sample(self.key(i))
        if self.velocity == "linear":
            k_vec, k_mod = g.k_grid(real=True)
            k2 = k_mod**2
            fk = jnp.fft.rfftn(d)
            v = jnp.stack([jnp.fft.irfftn(1j * self.aHf * kv / jnp.where(k2 > 0, k2, 1.0) * fk, s=g.shape) for kv in k_vec], -1)
        else:
            v = np.zeros(g.shape + (3,), dtype=np.asarray(d).dtype)
        d = np.asarray(d)
        if d.min() <= -1.0:  # no clipping: a negative density has no temperature; lower sigma_delta instead
            raise ValueError(
                f"realisation {i}: min(1 + delta) = {1 + d.min():.3f} <= 0 (sigma_delta = {self.sigma_delta}); lower sigma_delta."
            )
        n_HI = self.n_bar_HI * (1.0 + d)
        return dict(
            delta=d,
            n={"HI": n_HI, "SiIII": self.siiii_to_hi * n_HI},
            T=self.T0 * (1.0 + d) ** self.gamma_minus_1,
            v=np.asarray(v),
        )

    def cells(self, i: int, fields: dict | None = None):
        """``core.Cells`` of realisation i: the grid cells (``grf_cells``), n of both species, the shared T and v."""
        f = self.fields(i) if fields is None else fields
        n_bar = {"HI": self.n_bar_HI, "SiIII": self.siiii_to_hi * self.n_bar_HI}
        return grf_cells(f["delta"], self.grf(), n_bar, temperature=f["T"], velocity=f["v"])

    def limits(self) -> dict:
        """The full-coverage limits of the grid [1/cMpc], [cMpc]: k_min = 2 pi / L (the fundamental mode) and k_max =
        pi / dx (the Nyquist frequency of each axis: inside the sphere |k| <= pi / dx every direction is available,
        beyond it only the corners of the cube of modes); r_min = dx and r_max = L / 2 (|r| <= L / 2 holds every
        direction of the minimum-image separations, beyond it only the corners)."""
        L, dx = float(self.box_cMpc), self.dx
        return dict(k_min=2 * np.pi / L, k_max=np.pi / dx, r_min=dx, r_max=0.5 * L)

    def k_edges(self, *, k_1: float | None = None, k_2_factor: float = 3.0, n_log: int = 12, k_max=None, info: bool = False):
        """The |k| bins [1/cMpc] of ``bao_k_edges`` with k_f = 2 pi / L: one per mode up to k_1 (default 3.5 k_f), bins of
        k_f / 4 up to k_2 ~= ``k_2_factor`` / sigma_bao (the end of the BAO wiggles), ``n_log`` log bins up to ``k_max``
        (default pi / dx, ``limits``). ``info``: also the dict of ``bao_k_edges``."""
        lim = self.limits()
        k_max = lim["k_max"] if k_max is None else float(k_max)
        return bao_k_edges(lim["k_min"], k_max, k_2_factor / self.sigma_bao, k_1=k_1, n_log=n_log, info=info)

    def r_edges(self) -> np.ndarray:
        """The default |r| bins [cMpc]: width dx (the finest meaningful) from 0 to L / 2, so the first bin holds r = 0
        only."""
        return full_coverage(np.arange(0.0, 0.5 * self.box_cMpc + 0.5 * self.dx, self.dx), self.limits()["r_max"], log=None)

    def bin_edges(self, k_edges=None, r_edges=None, log=print) -> dict:
        """The bins of ``field_statistics`` (``GridStatistics3D`` keywords) [1/cMpc], [cMpc]:

        * |k| ``k_edges`` (default ``k_edges()``), also the k_perp columns of the (k_par, k_perp) map; the map's rows
          are the grid's own k_par = 2 pi m / L, m = 1 .. N / 2, one row per mode centred on it;
        * |r| ``r_edges`` (default ``r_edges()``: width dx up to L / 2), also the |r| bins of the mu wedges and the
          r_perp columns of the (r_par, r_perp) map; the map's rows are the grid's own r_par = j dx, j = 0 .. N / 2.

        Every |k|, k_perp, |r| and r_perp bin that extends beyond the full-coverage limits (``limits``: pi / dx, L / 2)
        is cut (``full_coverage``, reported by ``log``)."""
        lim = self.limits()
        k = self.k_edges() if k_edges is None else np.asarray(k_edges, dtype=float)
        r = self.r_edges() if r_edges is None else np.asarray(r_edges, dtype=float)
        k = full_coverage(k, lim["k_max"], "|k| bins", log)
        r = full_coverage(r, lim["r_max"], "|r| bins", log)
        half = self.n_cells // 2
        return dict(
            k_edges=k,
            kpar_edges=native_edges(lim["k_min"], half, start=1),
            kperp_edges=k,
            r_edges=r,
            r_mu_edges=r,
            rpar_edges=native_edges(self.dx, half),
            rperp_edges=r,
        )

    # ------------------------------------------------------------------ skewers
    def columns(self, stride=(1, 1)) -> np.ndarray:
        """(N_sk, 2) node indices (i_x, i_z) of the skewer columns: every stride[0]-th in x, every stride[1]-th in z
        (row-major, x outer)."""
        ix, iz = np.arange(0, self.n_cells, stride[0]), np.arange(0, self.n_cells, stride[1])
        return np.stack(np.meshgrid(ix, iz, indexing="ij"), -1).reshape(-1, 2)

    def skewer_bundle(self, columns, points_per_cell: int) -> SkewerBundle:
        """Skewers along y through the centres x = i_x dx, z = i_z dx of the cell columns (the nodes of ``grf_cells``),
        from y = 0 to y = L, points_per_cell points per cell: the first and the last point are the same place of the
        periodic box, so ``velocity_period=True`` is one box."""
        cols = np.atleast_2d(columns)
        y = np.linspace(0.0, self.box_cMpc, self.n_cells * points_per_cell + 1)
        pts = np.empty((cols.shape[0], y.size, 3))
        pts[..., 0] = cols[:, :1] * self.dx
        pts[..., 1] = y
        pts[..., 2] = cols[:, 1:] * self.dx
        return SkewerBundle(
            spatial_points=jnp.asarray(pts), cosmologies=self.cosmology(), a0=1.0 / (1.0 + self.z), x0=0.0, ICs_defined_at_index=0
        )


def hubble_velocity(bundle: SkewerBundle) -> np.ndarray:
    """v_c = c ln(a_dagger / a) [km/s] of the points of the first skewer (the line centre without peculiar velocity,
    ``core.spectral.line_centre_velocity_kms``), a_dagger = the bundle's a0."""
    a = np.asarray(bundle.scale_factors[0, 0], dtype=np.float64)
    return phys.c_kms * np.log(float(np.ravel(bundle.a0)[0]) / a)


# ---------------------------------------------------------------------- the fields: P and xi of every realisation
def field_statistics(boxes: ToyBoxes, *, k_edges=None, r_edges=None, realisations=None, grid=None, log=print) -> dict:
    """P and xi of the realisations on their grid (``GridStatistics3D``, line of sight y, the bins of
    ``ToyBoxes.bin_edges(k_edges, r_edges)`` unless ``grid`` is given: |k| and the k_perp columns ``k_edges``, |r|, the
    mu wedges and the r_perp columns ``r_edges``, the map rows on the grid's own modes and separations, every bin inside
    the full-coverage limits), streamed: each realisation is regenerated from its seed, measured and discarded. Every
    call recomputes; nothing is stored.

    Returns ``ensemble_mean`` over the realisations of every ``GRID_KEYS`` statistic (mean, ``_err``: the error of the
    mean, std / sqrt(n) with ddof = 1, of each bin alone; ``_each``), the expectation ``theory`` {key: array}, the
    no-BAO expectation ``theory_nw`` (the power law alone, ``PowerLawGRF.power_grid(bao=False)``: xi without the bump,
    so ``theory - theory_nw`` is the bump on the grid),
    ``cov_xi_iso`` (the exact Gaussian covariance of the isotropic xi bins of one realisation,
    ``GridStatistics3D.xi_iso_covariance``), the bin ``axes`` (``GridStatistics3D.axes``), ``min_one_plus_delta`` per
    realisation, ``realisations``, ``limits`` (``ToyBoxes.limits``), ``time_s`` (the measurements) and ``time_cov_s``
    (the covariance).
    """
    idx = np.arange(boxes.n_real) if realisations is None else np.asarray(realisations)
    if grid is None:
        grid = GridStatistics3D((boxes.n_cells,) * 3, boxes.box_cMpc, los_axis=1, **boxes.bin_edges(k_edges, r_edges, log=log))
    gs = grid
    g = boxes.grf()
    t0 = time.perf_counter()
    recs, mins = [], []
    for n, i in enumerate(idx):
        d = g.sample(boxes.key(int(i)))
        mins.append(float(1.0 + jnp.min(d)))
        recs.append(gs.measure(d))
        if log and (n + 1) % 16 == 0:
            log(f"field statistics: {n + 1}/{idx.size} realisations, {time.perf_counter() - t0:.0f} s")
    t_meas = time.perf_counter() - t0
    res = ensemble_mean(recs, GRID_KEYS)
    res["theory"] = gs.theory(g.power_grid())
    res["theory_nw"] = gs.theory(g.power_grid(bao=False))
    t0 = time.perf_counter()
    res["cov_xi_iso"] = gs.xi_iso_covariance(g.power_grid())
    res.update(
        axes=gs.axes(),
        min_one_plus_delta=np.array(mins),
        realisations=idx,
        limits=boxes.limits(),
        time_s=t_meas,
        time_cov_s=time.perf_counter() - t0,
    )
    return res


def _xi_iso_offset(res):
    """(r, mean - expectation, expectation, each realisation, covariance) of the isotropic xi bins with r > 0."""
    r, th = res["axes"]["r_iso"], res["theory"]["xi_iso"]
    ok = np.isfinite(th) & np.isfinite(res["xi_iso"]) & (r > 0)
    C = res["cov_xi_iso"][np.ix_(ok, ok)]
    return r[ok], (res["xi_iso"] - th)[ok], th[ok], res["xi_iso_each"][:, ok], C


def xi_chi2(res, r_range=None) -> dict:
    """chi^2 of the mean isotropic xi of a ``field_statistics`` result against its expectation, with the exact covariance
    of the mean (``cov_xi_iso`` / n) and with its diagonal only, over the bins with r in ``r_range`` [cMpc] (default:
    every bin with r > 0). Returns dict(chi2_full, chi2_diag, dof (the number of bins), p_full (the probability of a
    larger chi2), mode1_share (the share of the variance of one realisation in the first eigenvector of the covariance),
    offset_top3_share (the share of the offset |mean - expectation|^2 along the 3 eigenvectors of largest variance),
    pulls_top3 (the offset along those 3 eigenvectors in units of their standard deviation for the mean), chi2_each (n,)
    (each realisation against the covariance of one realisation))."""
    from scipy.stats import chi2 as chi2_dist

    r, d, th, each, C = _xi_iso_offset(res)
    sel = np.ones(r.size, bool) if r_range is None else (r >= r_range[0]) & (r <= r_range[1])
    d, each, C = d[sel], each[:, sel], C[np.ix_(sel, sel)]
    n = each.shape[0]
    w, U = np.linalg.eigh(C)  # ascending: the last eigenvectors have the largest variance
    proj = U.T @ d
    chi2_full = float(np.sum(proj**2 / (w / n)))
    return dict(
        chi2_full=chi2_full,
        chi2_diag=float(np.sum(d**2 / (np.diag(C) / n))),
        dof=int(d.size),
        p_full=float(chi2_dist.sf(chi2_full, d.size)),
        mode1_share=float(w[-1] / w.sum()),
        offset_top3_share=float(np.sum(proj[-3:] ** 2) / np.sum(proj**2)),
        pulls_top3=(proj / np.sqrt(w / n))[::-1][:3],
        chi2_each=np.array([float((x - th[sel]) @ np.linalg.solve(C, x - th[sel])) for x in each]),
    )


def xi_subset_offsets(res, sizes=(4, 8, 16), r_range=None) -> list:
    """The offset of the mean r^2 xi(|r|) of a ``field_statistics`` result from its expectation, averaged over the bins
    with r in ``r_range`` [cMpc] (default every bin with r > 0), for disjoint subsets of ``size`` consecutive
    realisations, and the standard deviation of that offset for a mean of ``size`` realisations from the exact
    covariance, sqrt(w^T C w / size) with w = r^2 / n_bins. Returns [dict(size, offsets (n_subsets,), sigma)] [cMpc^2]."""
    r, _, th, each, C = _xi_iso_offset(res)
    sel = np.ones(r.size, bool) if r_range is None else (r >= r_range[0]) & (r <= r_range[1])
    w = np.where(sel, r**2, 0.0) / sel.sum()
    sig1 = float(np.sqrt(w @ C @ w))
    out = []
    for m in sizes:
        if m <= each.shape[0]:
            offs = [float(w @ (each[j : j + m].mean(0) - th)) for j in range(0, each.shape[0] - m + 1, m)]
            out.append(dict(size=int(m), offsets=np.array(offs), sigma=sig1 / np.sqrt(m)))
    return out


# ---------------------------------------------------------------------- the skewers: tau, and the 1D and 3D statistics
@dataclass(frozen=True)
class SkewerRun:
    """The skewer set-up of ``run_skewers``.

    Parameters
    ----------
    stride : (s_x, s_z)
        Which cell columns become skewers: every s_x-th column in x and every s_z-th in z, from column 0. The skewers run
        along y exactly through the cell centres x = i s_x dx, z = j s_z dx (the nodes of ``grf_cells``):
        ceil(N / s_x) ceil(N / s_z) skewers per box, a transverse spacing s dx [cMpc]. Changes the result (the number of
        skewers, so the noise of every statistic; the transverse sampling of the 3D statistics, whose transverse Nyquist
        frequency is pi / (s dx)) and the cost (linear in the number of skewers). The 3D statistics need N to be a
        multiple of both strides (a periodic lattice of skewers).
    points_per_cell : int
        Points of the path per cell length along y: N points_per_cell + 1 points from y = 0 to y = L, so the pieces of
        the cells (``cell_pieces``) have the width dx / points_per_cell, W = period / (N points_per_cell) in velocity.
        Changes the result until it converges (tau sums the pieces at their centroids, accurate while W << b: the
        resolution check of ``compute_optical_depth`` asks W <= b / 4 for all but a small fraction of the column); the
        cost of tau is linear in it.
    n_pix : int
        Velocity pixels over the period (one box): width dv = period / n_pix, tau evaluated at the pixel centres. Sets the
        velocity resolution, the k range of P1D (k_Nyquist = pi / dv) and the line-of-sight resolution of the 3D
        statistics: it changes the result; the cost of tau is linear in it.
    batch : int
        Skewers per call of ``cell_pieces`` / ``compute_optical_depth``: memory only, the results are identical.
    """

    stride: tuple = (8, 16)
    points_per_cell: int = 28
    n_pix: int = 2048
    batch: int = 256

    def periodic(self, boxes: ToyBoxes) -> bool:
        """True if the columns form a periodic lattice (N a multiple of both strides): the 3D statistics need it."""
        return boxes.n_cells % self.stride[0] == 0 and boxes.n_cells % self.stride[1] == 0

    def derived(self, boxes: ToyBoxes) -> dict:
        """The quantities that follow from this set-up and the boxes: skewers per box (n_x, n_z, per_box) and in all
        (total, n_real boxes), points per skewer, the velocity period of one box [km/s] (``hubble_velocity`` of one
        skewer: c ln(a_dagger / a) from y = 0 to L), the pixel width dv and the piece width W [km/s], the conversion
        u = period / L [km/s per cMpc] (the mean a H along the skewer, from ``aH_start`` at y = 0 to ``aH_end`` at y = L),
        and the transverse spacing s dx [cMpc] and s dx u [km/s] per axis."""
        n_x, n_z = (int(np.ceil(boxes.n_cells / s)) for s in self.stride)
        v = hubble_velocity(boxes.skewer_bundle(np.array([[0, 0]]), self.points_per_cell))
        y = np.linspace(0.0, boxes.box_cMpc, v.size)
        period = float(abs(v[-1] - v[0]))
        u = period / float(boxes.box_cMpc)
        d_perp = np.array(self.stride, dtype=float) * boxes.dx
        return dict(
            n_x=n_x,
            n_z=n_z,
            per_box=n_x * n_z,
            total=n_x * n_z * boxes.n_real,
            points_per_skewer=boxes.n_cells * self.points_per_cell + 1,
            period_kms=period,
            dv_kms=period / self.n_pix,
            piece_kms=period / (boxes.n_cells * self.points_per_cell),
            kms_per_cMpc=u,
            aH_start=float(abs(v[1] - v[0]) / (y[1] - y[0])),
            aH_end=float(abs(v[-1] - v[-2]) / (y[-1] - y[-2])),
            d_perp_cMpc=d_perp,
            d_perp_kms=d_perp * u,
        )

    def grid3d(self, boxes: ToyBoxes, n_los: int | None = None, *, edges: dict | None = None, log=None) -> GridStatistics3D:
        """The 3D statistics on the lattice of skewers (``GridStatistics3D``), in comoving units, on exactly the lattice
        the skewers produce (no regridding): grid (n_x, n_los, n_z) (n_los = n_pix by default: the pixels of the
        overflux; ``n_cells`` for the density sampled at the cells of the skewers, ``lattice_density_theory``), line of
        sight along axis 1, sides (n_x s_x dx, L, n_z s_z dx) = L. The line of sight is a velocity, converted with the one
        factor u = period / L (``derived``): a pixel dv spans dv / u = L / n_pix [cMpc], so |r|, |k| and mu combine both
        axes in comoving units; k_x, k_z are multiples of 2 pi / L up to pi / (s dx), k_par up to pi n_los / L (pi / dv in
        s/km), and xi is the inverse transform of P on the same lattice. Bins [1/cMpc], [cMpc]:

        * the (r_par, r_perp) plane: one row per separation of the lattice along the line of sight, r_par = j L / n_los,
          j = 0 .. n_los / 2, centred on it (j dv in km/s); r_perp columns ``edges["rperp_edges"]`` (default width the
          transverse spacing s dx, from 0 to L / 2: the first column holds the pairs on the same skewer);
        * |r| and the mu wedges ``edges["r_edges"]`` (default width the coarser lattice spacing, max(s dx, L / n_los),
          up to L / 2: xi of a periodic field is periodic, so the separations beyond L / 2 repeat those inside);
        * |k| (and the mu wedges) ``edges["k_edges"]`` (default ``bao_k_edges`` up to the smallest Nyquist frequency of
          the lattice, k_N = min(pi n_los / L, pi / (s_x dx), pi / (s_z dx)): inside that sphere a |k| shell holds every
          direction, beyond it only the directions closer to the finer axes); also the k_perp columns of the (k_par,
          k_perp) plane, whose rows are the lattice's own k_par = 2 pi m / L, m = 1 .. n_los / 2.

        Every bin beyond the full-coverage limits (k_N, L / 2) is cut (``full_coverage``, reported by ``log``)."""
        if not self.periodic(boxes):
            raise ValueError(f"the 3D statistics need n_cells = {boxes.n_cells} to be a multiple of both strides {self.stride}.")
        L = float(boxes.box_cMpc)
        n_los = self.n_pix if n_los is None else int(n_los)
        n_x, n_z = boxes.n_cells // self.stride[0], boxes.n_cells // self.stride[1]
        d_perp = np.array(self.stride, dtype=float) * boxes.dx
        k_nyq = float(min(np.pi * n_los / L, *(np.pi / d_perp)))
        half = 0.5 * L
        e = dict(edges or {})
        dr = max(float(d_perp.max()), L / n_los)
        k = e.get("k_edges")
        k = bao_k_edges(2 * np.pi / L, k_nyq, 3.0 / boxes.sigma_bao) if k is None else full_coverage(k, k_nyq, "|k| bins", log)
        r = e.get("r_edges")
        r = np.arange(0.0, half + 0.5 * dr, dr) if r is None else np.asarray(r, dtype=float)
        rq = e.get("rperp_edges")
        rq = np.arange(0.0, half + 0.5 * d_perp.min(), float(d_perp.min())) if rq is None else np.asarray(rq, dtype=float)
        r, rq = full_coverage(r, half, "|r| bins", log), full_coverage(rq, half, "r_perp bins", log)
        return GridStatistics3D(
            (n_x, n_los, n_z),
            (n_x * d_perp[0], L, n_z * d_perp[1]),
            los_axis=1,
            k_edges=k,
            k_mu_edges=k,
            kpar_edges=native_edges(2 * np.pi / L, n_los // 2, start=1),
            kperp_edges=k,
            r_edges=r,
            r_mu_edges=r,
            rpar_edges=native_edges(L / n_los, n_los // 2),
            rperp_edges=rq,
        )

    def limits(self, boxes: ToyBoxes) -> dict:
        """The lattice of the 3D statistics of ``grid3d`` and its full-coverage limits.

        Native units: the transverse spacing d_perp = s dx [cMpc] per axis, k_x, k_z from k_fund = 2 pi / L to
        k_nyq_perp = pi / (s dx) [1/cMpc]; along the line of sight the pixel dv = v_per / n_pix [km/s], k_par from
        k_par_min_skm = 2 pi / v_per to k_par_max_skm = pi / dv [s/km], r_par from dv to v_per / 2 [km/s] (period_kms =
        v_per, kms_per_cMpc = u). Comoving units (the isotropic quantities): k_nyq_par = pi u / dv = pi n_pix / L, k_nyq =
        the smallest Nyquist frequency (the radius of the |k| bins), dr_iso = max(s dx, dv / u) (the coarser lattice
        spacing, the default |r| bin width), r_max = L / 2, r_perp_min = the smaller transverse spacing; and the
        full-coverage limits of the isotropic bins as ``ToyBoxes.limits`` names them (``plot_overflux_3d(limits=)``):
        k_min = k_fund, k_max = k_nyq, r_min = dr_iso (below it only the separations along the finer axis), r_max."""
        L = float(boxes.box_cMpc)
        d = self.derived(boxes)
        d_perp = np.array(self.stride, dtype=float) * boxes.dx
        k_par, k_perp = np.pi * self.n_pix / L, np.pi / d_perp
        return dict(
            d_perp=d_perp,
            k_nyq_par=float(k_par),
            k_nyq_perp=k_perp,
            k_nyq=float(min(k_par, *k_perp)),
            k_fund=2 * np.pi / L,
            dv_kms=d["dv_kms"],
            period_kms=d["period_kms"],
            kms_per_cMpc=d["kms_per_cMpc"],
            k_par_min_skm=2 * np.pi / d["period_kms"],
            k_par_max_skm=np.pi / d["dv_kms"],
            dr_iso=float(max(d_perp.max(), L / self.n_pix)),
            r_max=0.5 * L,
            r_perp_min=float(d_perp.min()),
            k_min=2 * np.pi / L,
            k_max=float(min(k_par, *k_perp)),
            r_min=float(max(d_perp.max(), L / self.n_pix)),
        )


def on_points(columns, points_per_cell: int) -> np.ndarray:
    """The values at the points of the skewers of cell values ``columns`` (..., N, ...) along axis 1 (y): (..., N p, ...)
    at y_m = m dx / p, m = 0 .. N p - 1 (p = ``points_per_cell``; the end point y = L, the same place of the periodic
    box as y = 0, dropped, so the lattice is exactly periodic). The gas is uniform in each cell, so a point takes the
    value of its cell, that of the nearest node j = floor(m / p + 1/2) mod N (a point on a cell boundary goes to the
    upper cell): every cell holds exactly p points."""
    p = int(points_per_cell)
    return np.roll(np.repeat(np.asarray(columns), p, axis=1), -(p // 2), axis=1)


def _lattice_power(boxes: ToyBoxes, setup: SkewerRun, bao: bool = True, points_per_cell: int = 1) -> np.ndarray:
    """The aliased spectrum P_lat on the rfft modes of the lattice of ``grid3d(boxes, n_los=n_cells p)`` (p =
    ``points_per_cell``), its k = 0 entry set to 0 as in the estimator: xi_lat = xi_grid at the lattice separations
    (``lattice_density_theory``) and P_lat = dV_lat rfftn(xi_lat) [cMpc^3]."""
    g = boxes.grf()
    p, N = int(points_per_cell), int(boxes.n_cells)
    xi = np.fft.irfftn(np.asarray(g.power_grid(bao=bao)), s=g.shape) / g.cell_volume
    xi = xi[:: setup.stride[0], :, :: setup.stride[1]]
    if p > 1:  # two points Delta m = q p + s apart lie q cells apart for a fraction 1 - s / p of the pairs, q + 1 for s / p
        q, s = np.divmod(np.arange(N * p), p)
        t = (s / p)[None, :, None]
        xi = (1.0 - t) * xi[:, q, :] + t * xi[:, (q + 1) % N, :]
    P = np.fft.rfftn(xi).real * (setup.stride[0] * boxes.dx) * (boxes.dx / p) * (setup.stride[1] * boxes.dx)
    P[0, 0, 0] = 0.0
    return P


def _lattice_p1d(P, gs: GridStatistics3D) -> np.ndarray:
    """The 1D spectrum along the skewers of a lattice spectrum P (rfft layout of ``gs``, line of sight axis 1):
    P1D(k_par) = (1 / (L_x L_z)) sum_{k_perp} P(k_perp, k_par) at k_par = 2 pi j / L, j = 0 .. n_los / 2 [cMpc]."""
    return (P * _rfft_weights(gs.shape[2])).sum(axis=(0, 2))[: gs.shape[1] // 2 + 1] / (gs.L[0] * gs.L[2])


def lattice_density_theory(
    boxes: ToyBoxes, setup: SkewerRun, *, bao: bool = True, edges: dict | None = None, points_per_cell: int = 1, p1d=False
):
    """The exact expectation of the statistics of ``grid3d(boxes, n_los=n_cells p)`` (p = ``points_per_cell``) for the
    density overdensity sampled on the lattice of skewers: every s_x-th column in x, every s_z-th in z, and along y
    every cell (p = 1) or every point of the skewers (``on_points``: the cell of each point). The lattice samples xi_grid
    at its own separations (``ToyBoxes``: xi_grid = irfftn(P on the modes) / dV); along y, two points Delta m = q p + s
    apart lie q or q + 1 cells apart, for the fractions 1 - s / p and s / p of the pairs, so xi_lat is xi_grid
    interpolated linearly between the cells, exactly. Its spectrum is the aliased P_lat = dV_lat rfftn(xi_lat) (the input
    folded over the aliases of the transverse lattice and, along y, the spectrum of the piecewise-constant columns),
    binned like the measurement (``GridStatistics3D.theory``), without the k = 0 mode. ``bao``: True, or False (the
    power law alone, ``PowerLawGRF.power_grid``); ``edges``: the bins of ``SkewerRun.grid3d``. ``p1d``: also return the
    1D spectrum along the skewers, P1D(k_par) = (1 / (L_x L_z)) sum_{k_perp} P_lat(k_perp, k_par), k_par = 2 pi j / L,
    j = 0 .. n_los / 2 [cMpc] (returns (theory, P1D))."""
    gs = setup.grid3d(boxes, n_los=boxes.n_cells * int(points_per_cell), edges=edges)
    P = _lattice_power(boxes, setup, bao, points_per_cell)
    th = gs.theory(P)
    return (th, _lattice_p1d(P, gs)) if p1d else th


def bao_height(r, r2xi, r_bao: float, sigma_bao: float, err=None) -> dict:
    """The BAO peak of r^2 xi(r): its value in the bin nearest r_bao and its height above the straight line through the
    bins nearest r_bao -+ 4 sigma_bao (the sidebands), with the error of that height from the errors ``err`` of the
    three bins (independent errors assumed). Returns dict(r, value, height, height_err (NaN without ``err``))."""
    r, y = np.asarray(r, dtype=float), np.asarray(r2xi, dtype=float)
    j, lo, hi = (int(np.argmin(np.abs(r - x))) for x in (r_bao, r_bao - 4 * sigma_bao, r_bao + 4 * sigma_bao))
    if hi == lo:  # the bins are too coarse to hold sidebands
        return dict(r=float(r[j]), value=float(y[j]), height=np.nan, height_err=np.nan)
    t = (r[j] - r[lo]) / (r[hi] - r[lo])
    h = y[j] - (y[lo] + t * (y[hi] - y[lo]))
    he = np.nan if err is None else float(np.sqrt(err[j] ** 2 + ((1 - t) * err[lo]) ** 2 + (t * err[hi]) ** 2))
    return dict(r=float(r[j]), value=float(y[j]), height=float(h), height_err=he)


def lattice_density_check(run: dict, boxes: ToyBoxes, setup: SkewerRun, field: str = "total") -> dict:
    """The density overdensity sampled at the points of the skewers (``run_skewers``'s ``density3d[field]``, measured with
    the 3D estimator of the overflux on the lattice of the points) against its exact expectation (``lattice_density_theory``
    with ``points_per_cell``, scaled by the global-mean normalisation, ``density_scale``), with exact Gaussian errors,
    valid for any number of realisations n: P(|k|) bin by bin, sigma = P sqrt(2 / (n_modes n)) (n_modes counts +k and
    -k); xi(|r|) with the exact covariance of the bins / n (``GridStatistics3D.xi_iso_covariance`` on the aliased
    spectrum), against the expectation with and without the BAO bump.

    Returns dict(P_chi2, P_dof, xi_chi2, xi_chi2_nw (against the expectation without the bump), xi_dof, bao,
    bao_expected, bao_nw (``bao_height`` of r^2 xi [cMpc^2], measured with the error from the covariance, expected,
    without the bump), n_real)."""
    d = run["density3d"][field]
    n = int(d["n_real"])
    edges, p, c = run.get("edges3d"), int(setup.points_per_cell), float(run["density_scale"])
    gs = setup.grid3d(boxes, n_los=boxes.n_cells * p, edges=edges)
    P = c * _lattice_power(boxes, setup, points_per_cell=p)
    th, nw = gs.theory(P), gs.theory(c * _lattice_power(boxes, setup, bao=False, points_per_cell=p))
    n_modes = np.asarray(gs.n_modes["P_iso"], dtype=float)
    sig = th["P_iso"] * np.sqrt(2.0 / (np.where(n_modes > 0, n_modes, np.nan) * n))
    okP = np.isfinite(d["P_iso"]) & np.isfinite(sig)
    r = gs.r_iso
    C = gs.xi_iso_covariance(P) / n
    ok = np.isfinite(d["xi_iso"]) & (r > 0) & np.all(np.isfinite(C), axis=1)
    Ci = np.linalg.inv(C[np.ix_(ok, ok)])
    chi = lambda m: float((d["xi_iso"] - m)[ok] @ Ci @ (d["xi_iso"] - m)[ok])
    err = r**2 * np.sqrt(np.diag(C))
    return dict(
        P_chi2=float(np.sum(((d["P_iso"] - th["P_iso"])[okP] / sig[okP]) ** 2)),
        P_dof=int(okP.sum()),
        xi_chi2=chi(th["xi_iso"]),
        xi_chi2_nw=chi(nw["xi_iso"]),
        xi_dof=int(ok.sum()),
        bao=bao_height(r, r**2 * d["xi_iso"], boxes.r_bao, boxes.sigma_bao, err),
        bao_expected=bao_height(r, r**2 * th["xi_iso"], boxes.r_bao, boxes.sigma_bao),
        bao_nw=bao_height(r, r**2 * nw["xi_iso"], boxes.r_bao, boxes.sigma_bao),
        n_real=n,
    )


def skewer_tau(boxes: ToyBoxes, i: int, setup: SkewerRun = SkewerRun(), *, columns=None, fields=None) -> dict:
    """tau (2, N_sk, n_pix) of Lya and Si III along the skewers of ``setup`` (or of ``columns``) through realisation i,
    periodic over one box (``velocity_period=True``), both on the Lya axis (``line_frame="reference"``; Si III at
    v + delta_v). Returns dict(tau, columns, v (n_pix,) [km/s], period, dv [km/s], resolution (n_batches, 2, 2): max W / b
    and column fraction above tolerance per batch and line, warnings (list of str), density {species: (n_x, n_cells, n_z)}
    (the number densities [comoving cm^-3] of the cells crossed by the lattice of skewers, every cell along y: the gas
    the skewers sample, ``on_points``; None for ``columns``), time_s {fields, pieces, tau})."""
    lines = LineBundle(list(LINE_IDS), ref_line_id=LINE_IDS[0])
    t = {"fields": time.perf_counter()}
    f = boxes.fields(i) if fields is None else fields
    cells = boxes.cells(i, f)
    t["fields"] = time.perf_counter() - t["fields"]
    lattice = columns is None
    cols = boxes.columns(setup.stride) if lattice else np.atleast_2d(columns)
    taus, res, msgs = [], [], []
    t["pieces"] = t["tau"] = 0.0
    for b0 in range(0, cols.shape[0], setup.batch):
        t0 = time.perf_counter()
        sb = boxes.skewer_bundle(cols[b0 : b0 + setup.batch], setup.points_per_cell)
        pieces = cell_pieces(cells, lines, skewer_bundle=sb)
        period = float(pieces.velocity_span_kms[0])
        v = (np.arange(setup.n_pix) + 0.5) * period / setup.n_pix
        t1 = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            tau, info = compute_optical_depth(pieces, v, lines, velocity_period=True, line_frame="reference", return_info=True)
        t["pieces"] += t1 - t0
        t["tau"] += time.perf_counter() - t1
        taus.append(np.asarray(tau))
        r = info["resolution"]
        res.append(np.stack([r["max_ratio"], r["column_fraction"]], -1))
        msgs += [str(w.message) for w in caught]
    density = {sp: np.asarray(n)[:: setup.stride[0], :, :: setup.stride[1]] for sp, n in f["n"].items()} if lattice else None
    return dict(
        tau=np.concatenate(taus, axis=1),
        columns=cols,
        v=v,
        period=period,
        dv=period / setup.n_pix,
        resolution=np.stack(res),
        warnings=sorted(set(msgs)),
        density=density,
        time_s=t,
    )


def skewer_statistics(
    tau,
    boxes: ToyBoxes,
    setup: SkewerRun,
    period: float,
    *,
    mean_transmission=None,
    mean: str = "global",
    lattice=True,
    edges: dict | None = None,
) -> dict:
    """The statistics of the skewers of one realisation (tau (2, N_sk, n_pix) of ``skewer_tau``), every overflux
    normalised by the mean transmission of the ensemble, delta = T / T_bar - 1 (``summary_stats.overflux``):
    ``mean_transmission`` (3, n_pix) = T_bar_Lya, T_bar_SiIII, T_bar_total over every skewer of every realisation of the
    ensemble (``run_skewers``), or, if None, over these skewers alone, ``mean`` "global" (one number per line, the same
    in every column) or "per_pixel".

    1D: the overflux of each line, its products and the total (``DeltaFieldBuilder``) and their
    ``OverfluxStatistics1D.summary``. 3D (``lattice`` and ``SkewerRun.periodic``): the overflux of each line and of the
    total (``OVERFLUX_FIELDS``), their statistics on the lattice in comoving units (``SkewerRun.grid3d`` with the bins
    ``edges``, ``GridStatistics3D.measure``) and their P1D, the mean over the skewers of (dv / n_pix) |rfft(delta)|^2 at
    k = 2 pi rfftfreq(n_pix, dv) [s/km]. Returns dict(flux (the 1D summary), flux3d {field: {OVERFLUX_KEYS: array}},
    axes3d (``GridStatistics3D.axes``), k1d, p1d (3, N_k) (None for the 3D entries without a lattice), delta_sum (3,
    n_pix): the sum over the skewers of each overflux, delta12_sum (n_pix,): of the product delta_1 delta_2,
    decomposition_error: max |[sum_A delta_A - C] / (1 + C) - delta_tot| over every pixel, time_s)."""
    t0 = time.perf_counter()
    dv = float(period) / setup.n_pix
    ov = DeltaFieldBuilder(tau, labels=list(TERM_LABELS), mean=mean, mean_transmission=mean_transmission).build()
    flux = OverfluxStatistics1D(ov, dv).summary()
    delta = np.stack([np.asarray(ov.delta_terms[k]) for k in TERM_LABELS] + [np.asarray(ov.total)])  # (3, N_sk, n_pix)
    C = np.asarray(ov.C_per_pixel)
    recon = (sum(np.asarray(v) for v in ov.delta_terms.values()) - C) / (1.0 + C)
    out = dict(
        flux=flux,
        flux3d=None,
        axes3d=None,
        k1d=None,
        p1d=None,
        delta_sum=delta.sum(axis=1),
        delta12_sum=np.asarray(ov.delta_terms["".join(TERM_LABELS)]).sum(axis=0),
        decomposition_error=float(np.max(np.abs(recon - delta[-1]))),
    )
    if lattice and setup.periodic(boxes):
        gs = setup.grid3d(boxes, edges=edges)
        n_x, n_z = gs.shape[0], gs.shape[2]
        on_grid = lambda d: d.reshape(n_x, n_z, setup.n_pix).transpose(0, 2, 1)  # columns are x-major: (x, y = los, z)
        out.update(
            flux3d={name: gs.measure(on_grid(d)) for name, d in zip(OVERFLUX_FIELDS, delta)},
            axes3d=gs.axes(),
            k1d=2 * np.pi * np.fft.rfftfreq(setup.n_pix, dv),
            p1d=(dv / setup.n_pix) * np.mean(np.abs(np.fft.rfft(delta, axis=-1)) ** 2, axis=1),
        )
    out["time_s"] = time.perf_counter() - t0
    return out


def skewer_box(
    boxes: ToyBoxes, i: int, setup: SkewerRun = SkewerRun(), *, columns=None, fields=None, mean: str = "global"
) -> dict:
    """``skewer_tau`` and ``skewer_statistics`` of realisation i alone: the ensemble of the normalisation is the skewers
    of this realisation (``mean``). Returns the entries of both (``time_s`` {fields, pieces, tau, stats}), mean_F (2,) and
    mean_F_total, the mean transmission of each line and of their product over the box."""
    r = skewer_tau(boxes, i, setup, columns=columns, fields=fields)
    s = skewer_statistics(r["tau"], boxes, setup, r["period"], mean=mean, lattice=columns is None)
    T = np.exp(-r["tau"])
    r["time_s"]["stats"] = s.pop("time_s")
    return dict(r, **s, mean_F=T.mean(axis=(1, 2)), mean_F_total=float(T.prod(0).mean()))


class _LinearTerms:
    """The density counterpart of a built ``DeltaFieldBuilder`` for ``OverfluxStatistics1D``: the total overdensity of a
    sum of species, delta_tot = sum_j w_j delta_j (w_j = the mean of species j over the mean of the sum), is linear, so
    its terms are the weighted single species, "1", "2", with no product and C = 0 (every pair (j, l) has order 2)."""

    def __init__(self, terms, total):
        self.labels, self.key_joiner = list(TERM_LABELS), ""
        self.subset_keys = list(TERM_LABELS)
        self.delta_terms = dict(zip(TERM_LABELS, terms))
        self.total = total
        self.C_per_pixel = np.zeros((1, total.shape[-1]))


DENSITY_FIELDS = ("HI", "SiIII", "total")  # the overdensities of Parts E and F (notebook 08): each species and their sum


def _density_statistics(cols: dict, means: dict, gs: GridStatistics3D, setup: SkewerRun, L: float) -> dict:
    """The 3D and 1D statistics of one realisation's density skewers at their points (``on_points``): the overdensity of
    each species and of their sum against its global mean, delta_X = n_X / <n_X> - 1 (``means``, absolute), on the lattice
    of the points (``gs``) and along every skewer (P1D, and the 1D summary with the weighted species as terms). Returns
    dict(flux3d {DENSITY_FIELDS: GRID_KEYS}, flux (``OverfluxStatistics1D.summary``), p1d (3, N_k) [cMpc], k1d [cMpc^-1],
    identity: max |delta_HI - delta_SiIII|, max |delta_HI - delta_tot|)."""
    p = int(setup.points_per_cell)
    n = {sp: on_points(cols[sp], p) for sp in ("HI", "SiIII")}
    n["total"] = n["HI"] + n["SiIII"]
    delta = {f: n[f] / means[f] - 1.0 for f in DENSITY_FIELDS}
    n_los = delta["HI"].shape[1]
    dx = float(L) / n_los
    flat = {f: d.transpose(0, 2, 1).reshape(-1, n_los) for f, d in delta.items()}  # (N_sk, n_los), x-major as the columns
    w = {sp: means[sp] / means["total"] for sp in ("HI", "SiIII")}
    lin = _LinearTerms([w["HI"] * flat["HI"], w["SiIII"] * flat["SiIII"]], flat["total"])
    D = np.stack([flat[f] for f in DENSITY_FIELDS])
    return dict(
        flux3d={f: gs.measure(delta[f]) for f in DENSITY_FIELDS},
        flux=OverfluxStatistics1D(lin, dx).summary(),
        k1d=2 * np.pi * np.fft.rfftfreq(n_los, dx),
        p1d=(dx / n_los) * np.mean(np.abs(np.fft.rfft(D, axis=-1)) ** 2, axis=1),
        identity=np.array([np.abs(delta["HI"] - delta["SiIII"]).max(), np.abs(delta["HI"] - delta["total"]).max()]),
    )


def run_skewers(
    boxes: ToyBoxes,
    setup: SkewerRun = SkewerRun(),
    *,
    edges: dict | None = None,
    realisations=None,
    mean: str = "global",
    keep=(0,),
    log=print,
) -> dict:
    """The skewers of every realisation (default all) and their statistics, in memory, in two passes:

    1. each realisation is generated, tau computed along its skewers (``skewer_tau``), and its tau kept, with the
       number densities of the cells the skewers cross (the gas they sample, at cell resolution);
    2. the mean transmission of the ensemble, T_bar, of each line and of the total over every skewer of every
       realisation (``mean``: "global", one number per line over every pixel, the default; or "per_pixel", T_bar(v) at
       each pixel), normalises the overflux of every realisation (``skewer_statistics``), so <delta> = 0 over the
       ensemble (at every pixel with "per_pixel"); and the density of every species, sampled at the points of the
       skewers (``on_points``: the end point dropped, spacing dx / points_per_cell), against its global mean, delta_X =
       n_X / <n_X> - 1, goes through the same 3D estimator on the lattice of the points (``SkewerRun.grid3d`` with
       n_los = n_cells points_per_cell, the bins ``edges``) and the same 1D statistics (notebook 08, Parts E and F).

    ``edges``: the bins of the 3D statistics (``SkewerRun.grid3d``: k_edges, r_edges, rperp_edges), for the overflux and
    the density alike (kept as ``edges3d``, for ``lattice_density_check``). The tau of every realisation is held until
    the end (2 n_real N_sk n_pix floats), and the densities of the cells along the skewers (2 n_real N_sk n_cells). The
    warnings of the optical depth (the resolution check) are issued again with the realisation. Every call recomputes;
    nothing is stored.

    Returns dict(flux (``ensemble_1d`` over the realisations), flux3d {field: ``ensemble_mean`` of the ``OVERFLUX_KEYS``
    statistics} for the ``OVERFLUX_FIELDS``, axes3d (the bins, comoving units), p1d (``ensemble_mean`` of "p1d" (3, N_k),
    with "k" [s/km]); the density at the points: density3d {``DENSITY_FIELDS``: ``ensemble_mean`` of the ``GRID_KEYS``},
    density_axes3d, density_theory (the exact expectation on that lattice, ``lattice_density_theory`` scaled by
    density_scale = (n_bar / <n>)^2, the global-mean normalisation), density_flux (``ensemble_1d`` of the 1D summary
    with the weighted species as terms), density_p1d (as p1d, with "k" [cMpc^-1] and the exact expectation "theory"
    [cMpc]), density_mean {field: <n_X> / n_bar_X}, density_identity (max over the realisations of |delta_HI -
    delta_SiIII| and |delta_HI - delta_total|) (the 3D entries None without a lattice); mean_transmission (3, n_pix) (T_bar
    of Lya, Si III and the total, as used: ``mean``), mean_transmission_per_pixel (3, n_pix) (T_bar(v) at each pixel,
    whatever ``mean``: the drift along the skewers), mean_delta (3, n_pix) (the mean of each overflux over the ensemble at
    each pixel: the shared pattern, zero with "per_pixel"), mean_delta_global (3,) (over every pixel: zero to rounding),
    C (3,) (T_bar / (T_bar_1 T_bar_2) - 1, its mean over the pixels) and C_error (max |C - <delta_1 delta_2>| over the
    ensemble, globally or at each pixel), decomposition_error (max over the realisations), mean_F (n, 2), mean_F_total
    (n,), resolution (n, n_batches, 2, 2), warnings {i: [str]}, time_s (n, 5) [fields, pieces, tau, stats, density],
    device (n,), realisations (n,), skewers {i: dict(tau, columns, v, period)} of the realisations in ``keep`` (the
    figures), edges3d, mean, config (the two set-ups as dicts))."""
    idx = np.arange(boxes.n_real) if realisations is None else np.asarray(realisations)
    keep = {int(k) for k in keep}
    taus, recs, cols, warns = [], [], [], {}
    lattice = setup.periodic(boxes)
    if lattice:
        setup.grid3d(boxes, edges=edges, log=log)  # reports any bin cut by the full-coverage limits, once
    for i in idx:
        t0 = time.perf_counter()
        r = skewer_tau(boxes, int(i), setup)
        taus.append(r.pop("tau"))
        cols.append(r.pop("density"))
        recs.append(r)
        warns[int(i)] = r["warnings"]
        for m in r["warnings"]:
            warnings.warn(f"realisation {int(i)}: {m}", stacklevel=2)
        if log:
            log(
                f"realisation {int(i)}: {r['columns'].shape[0]} skewers in {time.perf_counter() - t0:.0f} s "
                f"({jax.devices()[0].platform}); max column fraction above tolerance {r['resolution'][..., 1].max():.1e}"
            )
    period = recs[0]["period"]
    T = [np.exp(-t) for t in taus]
    T_pix = np.mean([np.asarray(mean_transmissions(x, "per_pixel")) for x in T], axis=0)  # equal skewers per realisation
    T_bar = np.broadcast_to(T_pix.mean(axis=1, keepdims=True), T_pix.shape).copy() if mean == "global" else T_pix
    stats = []
    for r, t in zip(recs, taus):
        s = skewer_statistics(t, boxes, setup, period, mean_transmission=T_bar, mean=mean, lattice=lattice, edges=edges)
        r["time_s"]["stats"] = s.pop("time_s")
        stats.append(s)
    n_sk = taus[0].shape[1]
    mean_delta = sum(s["delta_sum"] for s in stats) / (n_sk * len(stats))
    d12 = sum(s["delta12_sum"] for s in stats) / (n_sk * len(stats))  # <delta_1 delta_2> at each pixel
    C = T_bar[-1] / (T_bar[0] * T_bar[1]) - 1.0
    out = dict(
        flux=ensemble_1d([s["flux"] for s in stats]),
        flux3d=None,
        axes3d=None,
        p1d=None,
        density3d=None,
        mean_transmission=T_bar,
        mean_transmission_per_pixel=T_pix,
        mean_delta=mean_delta,
        mean_delta_global=mean_delta.mean(axis=1),
        C=float(C.mean()),
        C_error=float(np.abs(C.mean() - d12.mean())) if mean == "global" else float(np.abs(C - d12).max()),
        decomposition_error=max(s["decomposition_error"] for s in stats),
        mean_F=np.stack([x.mean(axis=(1, 2)) for x in T]),
        mean_F_total=np.array([float(x.prod(0).mean()) for x in T]),
        resolution=np.stack([r["resolution"] for r in recs]),
        warnings=warns,
        device=np.array([jax.devices()[0].platform] * len(recs)),
        realisations=idx,
        skewers={
            int(i): dict(tau=t, columns=r["columns"], v=r["v"], period=r["period"])
            for i, r, t in zip(idx, recs, taus)
            if int(i) in keep
        },
        edges3d=edges,
        mean=mean,
        config=dict(boxes=asdict(boxes), setup=asdict(setup)),
    )
    if lattice:
        out["flux3d"] = {name: ensemble_mean([s["flux3d"][name] for s in stats], OVERFLUX_KEYS) for name in OVERFLUX_FIELDS}
        out["axes3d"] = stats[0]["axes3d"]
        out["p1d"] = dict(ensemble_mean(stats, ("p1d",)), k=stats[0]["k1d"])
        # the density at the points of the skewers, against its global mean (every cell holds points_per_cell points)
        p = int(setup.points_per_cell)
        n_bar = {"HI": boxes.n_bar_HI, "SiIII": boxes.siiii_to_hi * boxes.n_bar_HI}
        means = {sp: float(np.mean([c[sp].mean() for c in cols])) for sp in n_bar}
        means["total"] = means["HI"] + means["SiIII"]
        n_bar["total"] = n_bar["HI"] + n_bar["SiIII"]
        gs_rho = setup.grid3d(boxes, n_los=boxes.n_cells * p, edges=edges)
        dens = []
        for r, c in zip(recs, cols):
            t0 = time.perf_counter()
            dens.append(_density_statistics(c, means, gs_rho, setup, boxes.box_cMpc))
            r["time_s"]["density"] = time.perf_counter() - t0
        scale = (n_bar["total"] / means["total"]) ** 2  # delta_X = (n_bar_X / <n_X>) (1 + delta) - 1, the same for every X
        P = scale * _lattice_power(boxes, setup, points_per_cell=p)
        out.update(
            density3d={f: ensemble_mean([d["flux3d"][f] for d in dens], GRID_KEYS) for f in DENSITY_FIELDS},
            density_axes3d=gs_rho.axes(),
            density_theory=gs_rho.theory(P),
            density_scale=scale,
            density_flux=ensemble_1d([d["flux"] for d in dens]),
            density_p1d=dict(ensemble_mean(dens, ("p1d",)), k=dens[0]["k1d"], theory=_lattice_p1d(P, gs_rho)),
            density_mean={f: means[f] / n_bar[f] for f in DENSITY_FIELDS},
            density_identity=np.max([d["identity"] for d in dens], axis=0),
        )
    keys = ("fields", "pieces", "tau", "stats", "density")
    out["time_s"] = np.array([[r["time_s"].get(k, 0.0) for k in keys] for r in recs])
    return out
