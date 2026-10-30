"""Toy boxes (notebook 08): periodic 3D Gaussian fields of known spectrum, their gas, skewers through them, and the
statistics of many realisations, streamed (generate, measure, discard) and returned in memory: nothing is stored, every
call recomputes.

* ``ToyBoxes``: the configuration and realisation i, regenerated from its seed ``base_seed + i`` whenever it is needed
  (``fields``, ``cells``): the fields are never stored. The spectrum is a power law plus a BAO term (a Gaussian bump in
  xi with its envelope in P, ``PowerLawGRF``), isotropic (no Kaiser or Fingers-of-God term), normalised to
  ``sigma_delta``; the gas of every cell, both species being the same gas (one temperature, one velocity):

      n_HI = n_bar_HI (1 + delta),   n_SiIII = r n_HI,   T = T0 (1 + delta)^(gamma - 1),
      v = 0 (``velocity="zero"``, the default)  or  v_k = i a H f (k / k^2) delta_k (``"linear"``: linear theory, notebook 04).

* ``field_statistics``: P and xi of every realisation on its grid (``summary_stats.GridStatistics3D``) against their exact
  expectations (with and without the BAO wiggle) and the exact Gaussian covariance of the isotropic xi bins;
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
from skewerjax.simulations.grf import PowerLawGRF
from skewerjax.simulations.toy_cells import grf_cells
from skewerjax.summary_stats import (
    GRID_KEYS,
    DeltaFieldBuilder,
    GridStatistics3D,
    OverfluxStatistics1D,
    ensemble_1d,
    ensemble_mean,
    hybrid_k_edges,
    mean_transmissions,
)

logger = logging.getLogger(__name__)

LINE_IDS = ("HI-LyA", "SiIII-1206")  # line 1, line 2 of the term labels "1", "2", "12"
TERM_LABELS = ("1", "2")
OVERFLUX_FIELDS = (*LINE_IDS, "total")  # the overfluxes of the 3D statistics: each line and the total
OVERFLUX_KEYS = GRID_KEYS  # their statistics: those of the fields
VELOCITY_MODES = ("zero", "linear")  # ToyBoxes.velocity


def bao_k_edges(k_f: float, k_max: float, k_w: float) -> np.ndarray:
    """|k| bin edges that resolve BAO wiggles of period ~2 k_f: one mode per bin up to 3.5 k_f, then bins of k_f / 4 up
    to k_w (the end of the wiggles), then 12 log bins to k_max; when k_w >= k_max, bins of k_f / 4 up to k_max; one mode
    per bin when k_max <= 4 k_f; ``hybrid_k_edges`` when k_w <= 4 k_f < k_max. Same units as the inputs."""
    if k_max <= 4.0 * k_f:
        lin = np.arange(0.5, k_max / k_f, 1.0) * k_f
        return np.append(lin, k_max) if k_max - lin[-1] > 0.1 * k_f else lin
    if 4.0 * k_f < k_w < k_max:
        return np.concatenate(
            [np.arange(0.5, 3.6, 1.0) * k_f, np.arange(3.75 * k_f, k_w, 0.25 * k_f), np.geomspace(k_w, k_max, 13)]
        )
    if k_w >= k_max > 4.0 * k_f:
        fine = np.arange(3.75 * k_f, k_max - 0.1 * k_f, 0.25 * k_f)
        return np.concatenate([np.arange(0.5, 3.6, 1.0) * k_f, fine, [k_max]])
    return hybrid_k_edges(k_f, k_max, 12, 24)


@dataclass(frozen=True)
class ToyBoxes:
    """The toy boxes of notebook 08: ``n_real`` periodic cubes of side ``box_cMpc`` on ``n_cells``^3 grids.

    Spectrum: P(k) = A_s (k / k_0)^(n_s - 1) + P_BAO(k), isotropic (``PowerLawGRF`` with A_bao = bao_ratio A_s,
    r_bao, sigma_bao [cMpc]): ``bao_ratio`` is the height of the BAO bump in xi in units of A_s [cMpc^-3], and P_BAO =
    A_bao E(k) [1 + f(k)] >= 0 (the bump's transform on its envelope, |f| <= 1), so every bao_ratio >= 0 is a valid
    spectrum; A_s is set by ``sigma_delta`` = sqrt(variance on the grid). Gas: the module
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
    n_s: float = -1.25
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

    # ------------------------------------------------------------------ the spectrum and the realisations
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

    def bin_edges(self) -> dict:
        """The bins of ``field_statistics`` (``GridStatistics3D`` keywords) that resolve the BAO term: |k| one mode per
        bin up to 3.5 k_f, then bins of k_f / 4 up to 3 / sigma_bao (the wiggles, period 2 pi / r_bao), then 12 log bins
        to the Nyquist frequency (``bao_k_edges``); |r| (and the mu wedges) in bins of 3 cells up to max(L / 2, r_bao +
        4 sigma_bao), so the whole BAO peak shows (beyond L / 2 a bin holds only the separations of the cube's minimum
        image: its expectation is binned alike); the (r_par, r_perp) plane up to L / 2. [1/cMpc], [cMpc]."""
        L, dx = float(self.box_cMpc), self.dx
        k = bao_k_edges(2 * np.pi / L, np.pi / dx, 3.0 / self.sigma_bao)
        dr, half = 3 * dx, 0.5 * L
        top = max(half, min(self.r_bao + 4 * self.sigma_bao, np.sqrt(3.0) * half))
        r, r_half = np.arange(0.0, top + 1e-9 * top, dr), np.arange(0.0, half + 1e-9 * half, dr)
        return dict(k_edges=k, r_edges=r, r_mu_edges=r, rpar_edges=r_half, rperp_edges=r_half)

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
def field_statistics(boxes: ToyBoxes, *, realisations=None, grid=None, log=print) -> dict:
    """P and xi of the realisations on their grid (``GridStatistics3D``, line of sight y, the bins of
    ``ToyBoxes.bin_edges`` unless ``grid`` is given), streamed: each realisation is regenerated from its seed, measured
    and discarded. Every call recomputes; nothing is stored.

    Returns ``ensemble_mean`` over the realisations of every ``GRID_KEYS`` statistic (mean, ``_err``: the error of the
    mean, std / sqrt(n) with ddof = 1, of each bin alone; ``_each``), the expectation ``theory`` {key: array}, the
    no-wiggle expectation ``theory_nw`` (the same spectrum without the BAO wiggle,
    ``PowerLawGRF.power_grid(bao="envelope")``: xi without the bump, so ``theory - theory_nw`` is the bump on the grid),
    ``cov_xi_iso`` (the exact Gaussian covariance of the isotropic xi bins of one realisation,
    ``GridStatistics3D.xi_iso_covariance``), the bin ``axes`` (``GridStatistics3D.axes``), ``min_one_plus_delta`` per
    realisation, ``realisations``, ``time_s`` (the measurements) and ``time_cov_s`` (the covariance).
    """
    idx = np.arange(boxes.n_real) if realisations is None else np.asarray(realisations)
    gs = GridStatistics3D((boxes.n_cells,) * 3, boxes.box_cMpc, los_axis=1, **boxes.bin_edges()) if grid is None else grid
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
    res["theory_nw"] = gs.theory(g.power_grid(bao="envelope"))
    t0 = time.perf_counter()
    res["cov_xi_iso"] = gs.xi_iso_covariance(g.power_grid())
    res.update(
        axes=gs.axes(),
        min_one_plus_delta=np.array(mins),
        realisations=idx,
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

    def grid3d(self, boxes: ToyBoxes, n_los: int | None = None) -> GridStatistics3D:
        """The 3D statistics on the lattice of skewers (``GridStatistics3D``), in comoving units: grid (n_x, n_los, n_z)
        (n_los = n_pix by default: the pixels of the overflux; ``n_cells`` for the density sampled at the cells of the
        skewers, ``lattice_density_theory``), line of sight along axis 1, sides (n_x s_x dx, L, n_z s_z dx) = L. The line
        of sight is a velocity, converted with the one factor u = period / L (``derived``): a pixel dv spans dv / u =
        L / n_pix [cMpc], so |r|, |k| and mu combine both axes in comoving units. Bins [1/cMpc], [cMpc]:

        * |k| (and the mu wedges) as ``bao_k_edges`` up to the smallest Nyquist frequency of the lattice, k_N =
          min(pi n_los / L, pi / (s_x dx), pi / (s_z dx)): inside that sphere a |k| shell holds every direction, beyond it
          the modes of the coarser axes stop and a shell only holds the directions closer to the finer axis;
        * the (k_par, k_perp) plane in linear bins of the fundamental mode 2 pi / L, centred on its multiples, up to k_N
          (a map with its origin at a corner);
        * |r|, the mu wedges and the (r_par, r_perp) plane up to half the box, L / 2: xi of a periodic field is
          periodic, xi(r + L) = xi(r), so the separations beyond L / 2 repeat (mirrored) those inside; |r| and r_par in
          bins of 3 cells, 3 dx, r_perp in bins of the transverse spacing (the smaller one)."""
        if not self.periodic(boxes):
            raise ValueError(f"the 3D statistics need n_cells = {boxes.n_cells} to be a multiple of both strides {self.stride}.")
        L = float(boxes.box_cMpc)
        n_los = self.n_pix if n_los is None else int(n_los)
        n_x, n_z = boxes.n_cells // self.stride[0], boxes.n_cells // self.stride[1]
        d_perp = np.array(self.stride, dtype=float) * boxes.dx
        k_nyq = float(min(np.pi * n_los / L, *(np.pi / d_perp)))
        k = bao_k_edges(2 * np.pi / L, k_nyq, 3.0 / boxes.sigma_bao)
        k_lin = (np.arange(0.5, k_nyq / (2 * np.pi / L), 1.0)) * (2 * np.pi / L)
        half = 0.5 * L
        r = np.arange(0.0, half + 1e-9 * half, 3 * boxes.dx)
        return GridStatistics3D(
            (n_x, n_los, n_z),
            (n_x * d_perp[0], L, n_z * d_perp[1]),
            los_axis=1,
            k_edges=k,
            k_mu_edges=k,
            kpar_edges=k_lin,
            kperp_edges=k_lin,
            r_edges=r,
            r_mu_edges=r,
            rpar_edges=r,
            rperp_edges=np.arange(0.0, half + 1e-9 * half, float(d_perp.min())),
        )

    def limits(self, boxes: ToyBoxes) -> dict:
        """The ranges of the 3D statistics of ``grid3d`` [1/cMpc], [cMpc]: k_nyquist (pi n_pix / L along the line of
        sight, i.e. pi / dv in velocity units, pi / (s dx) across it per axis, and their minimum, the radius of the |k|
        bins), k_fund = 2 pi / L, r_max = L / 2 and r_perp_min = the smaller transverse spacing."""
        L = float(boxes.box_cMpc)
        d_perp = np.array(self.stride, dtype=float) * boxes.dx
        k_par, k_perp = np.pi * self.n_pix / L, np.pi / d_perp
        return dict(
            k_nyq_par=float(k_par),
            k_nyq_perp=k_perp,
            k_nyq=float(min(k_par, *k_perp)),
            k_fund=2 * np.pi / L,
            r_max=0.5 * L,
            r_perp_min=float(d_perp.min()),
        )


def _lattice_power(boxes: ToyBoxes, setup: SkewerRun, bao: str = "total"):
    """(the grid of ``grid3d(boxes, n_los=n_cells)``, the aliased spectrum P_lat on its rfft modes): xi_lat = xi_grid at
    the lattice separations and P_lat = dV_lat rfftn(xi_lat) (``lattice_density_theory``)."""
    g = boxes.grf()
    gs = setup.grid3d(boxes, n_los=boxes.n_cells)
    xi_grid = np.fft.irfftn(np.asarray(g.power_grid(bao=bao)), s=g.shape) / g.cell_volume
    xi_lat = xi_grid[:: setup.stride[0], :, :: setup.stride[1]]
    return gs, np.fft.rfftn(xi_lat).real * gs.dV


def lattice_density_theory(boxes: ToyBoxes, setup: SkewerRun, *, bao: str = "total") -> dict:
    """The exact expectation of the statistics of ``grid3d(boxes, n_los=n_cells)`` for the density overdensity sampled
    on the lattice of skewers (every s_x-th column in x, every s_z-th in z, every cell along y): the lattice samples
    xi_grid at its own separations, xi_lat(r) = xi_grid(r) (``ToyBoxes``: xi_grid = irfftn(P on the modes) / dV), so its
    spectrum is the aliased P_lat = dV_lat rfftn(xi_lat), the input folded over the aliases of the transverse lattice,
    binned like the measurement (``GridStatistics3D.theory``). ``bao``: "total" or "envelope" (no wiggle,
    ``PowerLawGRF.power_grid``)."""
    gs, P = _lattice_power(boxes, setup, bao)
    return gs.theory(P)


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


def lattice_density_check(run: dict, boxes: ToyBoxes, setup: SkewerRun) -> dict:
    """The check of the 3D estimator of Part C (notebook 08): the density sampled on the lattice of skewers, measured
    with the same estimator by ``run_skewers`` (``density3d``), against its exact expectation (``lattice_density_theory``)
    with exact Gaussian errors, valid for any number of realisations n: P(|k|) bin by bin, sigma = P sqrt(2 / (n_modes
    n)) (n_modes counts +k and -k); xi(|r|) with the exact covariance of the bins / n (``GridStatistics3D.
    xi_iso_covariance`` on the aliased spectrum), against the expectation with and without the BAO wiggle.

    Returns dict(P_chi2, P_dof, xi_chi2, xi_chi2_nw (against the no-wiggle expectation), xi_dof, bao, bao_expected,
    bao_nw (``bao_height`` of r^2 xi [cMpc^2], measured with the error from the covariance, expected, no wiggle),
    n_real)."""
    d = run["density3d"]
    n = int(d["n_real"])
    gs, P = _lattice_power(boxes, setup)
    th, nw = gs.theory(P), lattice_density_theory(boxes, setup, bao="envelope")
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
    and column fraction above tolerance per batch and line, warnings (list of str), density (the overdensity on the
    lattice of skewers (n_x, n_cells, n_z), every cell along y; None for ``columns``), time_s {fields, pieces, tau})."""
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
    density = np.asarray(f["delta"])[:: setup.stride[0], :, :: setup.stride[1]] if lattice else None
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


def skewer_statistics(tau, boxes: ToyBoxes, setup: SkewerRun, period: float, *, mean_transmission=None, lattice=True) -> dict:
    """The statistics of the skewers of one realisation (tau (2, N_sk, n_pix) of ``skewer_tau``), every overflux
    normalised by the ensemble mean transmission at each pixel, delta = T / T_bar(v) - 1 (``summary_stats.overflux``):
    ``mean_transmission`` (3, n_pix) = T_bar_Lya, T_bar_SiIII, T_bar_total over every skewer of every realisation of the
    ensemble (``run_skewers``), or, if None, over these skewers alone.

    1D: the overflux of each line, its products and the total (``DeltaFieldBuilder``) and their
    ``OverfluxStatistics1D.summary``. 3D (``lattice`` and ``SkewerRun.periodic``): the overflux of each line and of the
    total (``OVERFLUX_FIELDS``), their statistics on the lattice in comoving units (``SkewerRun.grid3d``,
    ``GridStatistics3D.measure``) and their P1D, the mean over the skewers of (dv / n_pix) |rfft(delta)|^2 at
    k = 2 pi rfftfreq(n_pix, dv) [s/km]. Returns dict(flux (the 1D summary), flux3d {field: {OVERFLUX_KEYS: array}},
    axes3d (``GridStatistics3D.axes``), k1d, p1d (3, N_k) (None for the 3D entries without a lattice), delta_sum (3,
    n_pix): the sum over the skewers of each overflux, decomposition_error: max |[sum_A delta_A - C] / (1 + C) -
    delta_tot| over every pixel, time_s)."""
    t0 = time.perf_counter()
    dv = float(period) / setup.n_pix
    ov = DeltaFieldBuilder(tau, labels=list(TERM_LABELS), mean_transmission=mean_transmission).build()
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
        decomposition_error=float(np.max(np.abs(recon - delta[-1]))),
    )
    if lattice and setup.periodic(boxes):
        gs = setup.grid3d(boxes)
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


def skewer_box(boxes: ToyBoxes, i: int, setup: SkewerRun = SkewerRun(), *, columns=None, fields=None) -> dict:
    """``skewer_tau`` and ``skewer_statistics`` of realisation i alone: the ensemble of the normalisation is the skewers
    of this realisation. Returns the entries of both (``time_s`` {fields, pieces, tau, stats}), mean_F (2,) and
    mean_F_total, the mean transmission of each line and of their product over the box."""
    r = skewer_tau(boxes, i, setup, columns=columns, fields=fields)
    s = skewer_statistics(r["tau"], boxes, setup, r["period"], lattice=columns is None)
    T = np.exp(-r["tau"])
    r["time_s"]["stats"] = s.pop("time_s")
    return dict(r, **s, mean_F=T.mean(axis=(1, 2)), mean_F_total=float(T.prod(0).mean()))


def run_skewers(boxes: ToyBoxes, setup: SkewerRun = SkewerRun(), *, realisations=None, keep=(0,), log=print) -> dict:
    """The skewers of every realisation (default all) and their statistics, in memory, in two passes:

    1. each realisation is generated, tau computed along its skewers (``skewer_tau``), and its tau kept; the density on
       the lattice of skewers is measured with the 3D estimator of the overflux (``SkewerRun.grid3d`` with n_los =
       n_cells; the check of the estimator, against ``lattice_density_theory``);
    2. the ensemble mean transmission at each pixel, T_bar(v), of each line and of the total over every skewer of every
       realisation, normalises the overflux of every realisation (``skewer_statistics``), so <delta>_ens = 0 at every
       pixel.

    The tau of every realisation is held until the end (2 n_real N_sk n_pix floats). The warnings of the optical depth
    (the resolution check) are issued again with the realisation. Every call recomputes; nothing is stored.

    Returns dict(flux (``ensemble_1d`` over the realisations), flux3d {field: ``ensemble_mean`` of the ``OVERFLUX_KEYS``
    statistics} for the ``OVERFLUX_FIELDS``, axes3d (the bins, comoving units), p1d (``ensemble_mean`` of "p1d" (3, N_k),
    with "k" [s/km]), density3d (``ensemble_mean`` of the density's ``GRID_KEYS`` statistics on the lattice, with its
    "axes" and the exact expectation "theory", ``lattice_density_theory``) (flux3d, axes3d, p1d, density3d None without a
    lattice), mean_transmission (3, n_pix) (T_bar of Lya, Si III and the total at each pixel), mean_delta (3, n_pix) (the
    ensemble mean of each overflux at each pixel: zero to rounding), decomposition_error (max over the realisations),
    mean_F (n, 2), mean_F_total (n,), resolution (n, n_batches, 2, 2), warnings {i: [str]}, time_s (n, 4) [fields,
    pieces, tau, stats], device (n,), realisations (n,), skewers {i: dict(tau, columns, v, period)} of the realisations in
    ``keep`` (the figures), config (the two set-ups as dicts))."""
    idx = np.arange(boxes.n_real) if realisations is None else np.asarray(realisations)
    keep = {int(k) for k in keep}
    taus, recs, dens, warns = [], [], [], {}
    lattice = setup.periodic(boxes)
    gs_rho = setup.grid3d(boxes, n_los=boxes.n_cells) if lattice else None
    for i in idx:
        t0 = time.perf_counter()
        r = skewer_tau(boxes, int(i), setup)
        taus.append(r.pop("tau"))
        if lattice:
            dens.append(gs_rho.measure(r["density"]))
        r["density"] = None
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
    T_bar = np.mean([np.asarray(mean_transmissions(np.exp(-t))) for t in taus], axis=0)  # equal skewers per realisation
    stats = []
    for r, t in zip(recs, taus):
        s = skewer_statistics(t, boxes, setup, period, mean_transmission=T_bar, lattice=lattice)
        r["time_s"]["stats"] = s.pop("time_s")
        stats.append(s)
    n_sk = taus[0].shape[1]
    T = [np.exp(-t) for t in taus]
    out = dict(
        flux=ensemble_1d([s["flux"] for s in stats]),
        flux3d=None,
        axes3d=None,
        p1d=None,
        density3d=None,
        mean_transmission=T_bar,
        mean_delta=sum(s["delta_sum"] for s in stats) / (n_sk * len(stats)),
        decomposition_error=max(s["decomposition_error"] for s in stats),
        mean_F=np.stack([x.mean(axis=(1, 2)) for x in T]),
        mean_F_total=np.array([float(x.prod(0).mean()) for x in T]),
        resolution=np.stack([r["resolution"] for r in recs]),
        warnings=warns,
        time_s=np.array([[r["time_s"][k] for k in ("fields", "pieces", "tau", "stats")] for r in recs]),
        device=np.array([jax.devices()[0].platform] * len(recs)),
        realisations=idx,
        skewers={
            int(i): dict(tau=t, columns=r["columns"], v=r["v"], period=r["period"])
            for i, r, t in zip(idx, recs, taus)
            if int(i) in keep
        },
        config=dict(boxes=asdict(boxes), setup=asdict(setup)),
    )
    if lattice:
        out["flux3d"] = {name: ensemble_mean([s["flux3d"][name] for s in stats], OVERFLUX_KEYS) for name in OVERFLUX_FIELDS}
        out["axes3d"] = stats[0]["axes3d"]
        out["p1d"] = dict(ensemble_mean(stats, ("p1d",)), k=stats[0]["k1d"])
        out["density3d"] = dict(ensemble_mean(dens, GRID_KEYS), axes=gs_rho.axes(), theory=lattice_density_theory(boxes, setup))
    return out
