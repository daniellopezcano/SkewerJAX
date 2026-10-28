"""Toy boxes (notebook 08): periodic 3D Gaussian fields of known spectrum, their gas, skewers through them, and the
statistics of many realisations, streamed (generate, measure, discard) and cached.

* ``ToyBoxes``: the configuration and realisation i, regenerated from its seed ``base_seed + i`` whenever it is needed
  (``fields``, ``cells``): the fields are never stored. The spectrum is a power law plus a BAO term (a Gaussian bump in
  xi with its envelope in P, ``PowerLawGRF``), isotropic (no Kaiser or Fingers-of-God term), normalised to
  ``sigma_delta``; the gas of every cell, both species being the same gas (one temperature, one velocity):

      n_HI = n_bar_HI (1 + delta),   n_SiIII = r n_HI,   T = T0 (1 + delta)^(gamma - 1),
      v_k = i a H f (k / k^2) delta_k  (linear theory; notebook 04).

* ``field_statistics``: P and xi of every realisation on its grid (``core.stats.GridStatistics3D``) against their exact
  expectations (with and without the BAO wiggle) and the exact Gaussian covariance of the isotropic xi bins, cached in
  one file (read back when it holds this configuration, computed otherwise); ``xi_subset_offsets`` the offset of the
  mean xi from its expectation for disjoint subsets of the realisations.
* ``run_skewers``: per realisation, a regular sub-grid of skewers along y through the centres of the cell columns
  (``SkewerRun``), tau of Lya and Si III (periodic, ``velocity_period=True``, both on the Lya axis,
  ``line_frame="reference"``), the 1D statistics by term and order of the flux contrast (``OverfluxStatistics1D.summary``)
  and the 3D statistics of the overflux of each line and of the total on the lattice of skewers, in velocity units
  (``SkewerRun.grid3d``); one file per realisation, so a run is resumable and several processes can share it (a
  realisation is claimed with a lock file), or recomputed from scratch (``overwrite=True``). ``load_skewer_run``
  averages them (``core.stats.ensemble_1d``, ``ensemble_mean``).

Caches are keyed by ``ToyBoxes.tag`` (every field of the configuration, ``CACHE_VERSION`` and the float precision) and,
for the skewers, by the ``SkewerRun`` set-up: a cache of another configuration is never read.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import warnings
from dataclasses import asdict, dataclass
from functools import cached_property
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from skewerjax.core import Cosmology, DeltaFieldBuilder, LineBundle, SkewerBundle, compute_optical_depth, phys
from skewerjax.core.absorbers import cell_pieces
from skewerjax.core.stats import GRID_KEYS, GridStatistics3D, OverfluxStatistics1D, ensemble_1d, ensemble_mean
from skewerjax.core.stats.grid3d import hybrid_k_edges
from skewerjax.simulations.grf import PowerLawGRF
from skewerjax.simulations.toy_cells import grf_cells

logger = logging.getLogger(__name__)

LINE_IDS = ("HI-LyA", "SiIII-1206")  # line 1, line 2 of the term labels "1", "2", "12"
TERM_LABELS = ("1", "2")
OVERFLUX_FIELDS = (*LINE_IDS, "total")  # the overfluxes of the 3D statistics: each line and the total
OVERFLUX_KEYS = (*GRID_KEYS, "xi_ell")  # their statistics: those of the fields and the multipoles of xi
CACHE_VERSION = 3  # bump when the cached results of a configuration change (2: the BAO envelope; 3: covariance, 3D overflux)


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
    docstring, with n_bar_HI [comoving cm^-3], ``siiii_to_hi`` = r, T0 [K], ``gamma_minus_1``; the velocities of
    linear theory with a H f [km/s per cMpc] at redshift ``z`` of a flat cosmology (H0 = ``phys.H0_km_s_mpc``,
    ``Omega_m``), ``f_growth`` (notebook 04). Realisation i is ``grf().sample(jax.random.PRNGKey(base_seed + i))``.
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
    def aHf(self) -> float:
        """a H(z) f [km/s per cMpc] (flat cosmology, H0 = phys.H0_km_s_mpc)."""
        E = np.sqrt(self.Omega_m * (1 + self.z) ** 3 + 1 - self.Omega_m)
        return phys.H0_km_s_mpc * E / (1 + self.z) * self.f_growth

    def cosmology(self) -> Cosmology:
        return Cosmology(H0=phys.H0, Omega_r_0=0.0, Omega_m_0=self.Omega_m, Omega_k_0=0.0, Omega_lambda_0=1.0 - self.Omega_m)

    def fields(self, i: int) -> dict:
        """The gas of realisation i on the grid (NumPy): delta, n {species: comoving cm^-3}, T [K], v (N, N, N, 3) [km/s].
        Both species are the same gas: one T and one v."""
        g = self.grf()
        d = g.sample(self.key(i))
        k_vec, k_mod = g.k_grid(real=True)
        k2 = k_mod**2
        fk = jnp.fft.rfftn(d)
        v = jnp.stack([jnp.fft.irfftn(1j * self.aHf * kv / jnp.where(k2 > 0, k2, 1.0) * fk, s=g.shape) for kv in k_vec], -1)
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

    def tag(self) -> str:
        """A short hash of the configuration (every field), ``CACHE_VERSION`` and the float precision: the cache key of
        its results."""
        key = dict(asdict(self), cache_version=CACHE_VERSION, dtype=str(jnp.result_type(float)))
        return hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:10]

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
def field_statistics(boxes: ToyBoxes, cache_dir, *, realisations=None, grid=None, run: bool = False, log=print) -> dict:
    """P and xi of the realisations on their grid (``GridStatistics3D``, line of sight y, the bins of
    ``ToyBoxes.bin_edges`` unless ``grid`` is given), streamed, cached in ``cache_dir/fields_<tag>.npz``: read back when
    it holds these realisations with the same bins, computed and written otherwise, or always with ``run=True``.

    Returns ``ensemble_mean`` over the realisations of every ``GRID_KEYS`` statistic (mean, ``_err``, ``_each``), the
    expectation ``theory`` {key: array}, the no-wiggle expectation ``theory_nw`` (the same spectrum without the BAO
    wiggle, ``PowerLawGRF.power_grid(bao="envelope")``: xi without the bump, so ``theory - theory_nw`` is the bump on
    the grid), ``cov_xi_iso`` (the exact Gaussian covariance of the isotropic xi bins of one realisation,
    ``GridStatistics3D.xi_iso_covariance``), the bin ``axes`` (``GridStatistics3D.axes``), ``min_one_plus_delta`` per
    realisation, ``realisations``, ``time_s`` (the measurements) and ``time_cov_s`` (the covariance).
    """
    idx = np.arange(boxes.n_real) if realisations is None else np.asarray(realisations)
    path = Path(cache_dir) / f"fields_{boxes.tag()}.npz"
    if path.exists() and not run:
        z = np.load(path, allow_pickle=False)
        want = boxes.bin_edges() if grid is None else {k: v for k, v in grid.axes().items() if k.endswith("edges")}
        same_bins = all(f"axis_{k}" in z and np.array_equal(z[f"axis_{k}"], v) for k, v in want.items())
        if np.array_equal(z["realisations"], idx) and same_bins and "cov_xi_iso" in z:
            return _field_result(z)
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
    out = {f"each_{k}": np.stack([r[k] for r in recs]) for k in GRID_KEYS}
    out.update({f"theory_{k}": v for k, v in gs.theory(g.power_grid()).items()})
    out.update({f"theory_nw_{k}": v for k, v in gs.theory(g.power_grid(bao="envelope")).items()})
    t0 = time.perf_counter()
    out["cov_xi_iso"] = gs.xi_iso_covariance(g.power_grid())
    out.update({f"axis_{k}": v for k, v in gs.axes().items()})
    out.update(
        min_one_plus_delta=np.array(mins),
        realisations=idx,
        time_s=np.array(t_meas),
        time_cov_s=np.array(time.perf_counter() - t0),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    _save_npz(path, out)
    return _field_result(out)


def _field_result(z) -> dict:
    keys = list(z.keys())
    res = ensemble_mean([{k: z[f"each_{k}"][j] for k in GRID_KEYS} for j in range(len(z["realisations"]))], GRID_KEYS)
    res["theory"] = {k: z[f"theory_{k}"] for k in GRID_KEYS}
    res["theory_nw"] = {k: z[f"theory_nw_{k}"] for k in GRID_KEYS}
    res["axes"] = {k[5:]: z[k] for k in keys if k.startswith("axis_")}
    res.update(
        cov_xi_iso=z["cov_xi_iso"],
        min_one_plus_delta=z["min_one_plus_delta"],
        realisations=z["realisations"],
        time_s=float(z["time_s"]),
        time_cov_s=float(z["time_cov_s"]),
    )
    return res


def _save_npz(path: Path, arrays: dict):
    """Write atomically (a temporary file, then a rename), so a reader never sees half a file."""
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


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

    def folder(self, boxes: ToyBoxes, cache_dir) -> Path:
        """The cache folder of these skewers through these boxes: keyed by ``boxes.tag()``, the stride, the points per
        cell and the pixels (``batch`` does not change the results)."""
        tag = hashlib.sha1(json.dumps([boxes.tag(), asdict(self)["stride"], self.points_per_cell, self.n_pix]).encode())
        return Path(cache_dir) / f"skewers_{boxes.tag()}_{tag.hexdigest()[:8]}"

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

    def grid3d(self, boxes: ToyBoxes, period: float) -> GridStatistics3D:
        """The 3D statistics of the overflux on the lattice of skewers (``GridStatistics3D``), in velocity units: grid
        (n_x, n_pix, n_z), line of sight along axis 1, sides (n_x s_x dx u, period, n_z s_z dx u) = period, with u =
        period / L [km/s per cMpc] the conversion of the line of sight (``derived``) applied to the transverse comoving
        separations. Bins [s/km], [km/s]: |k| (and the mu wedges) as ``bao_k_edges`` up to the transverse Nyquist
        frequency pi / (s dx u), beyond which the transverse modes alias; |r| and the mu wedges in bins of 3 cells, 3 dx
        u, up to max(period / 2, (r_bao + 4 sigma_bao) u), with the multipoles l = 0, 2, 4 of xi in the same bins (the
        lattice is coarse across the line of sight, so the plain |r| average mixes the anisotropy of xi into it); the
        (r_par, r_perp) plane up to period / 2, r_perp in bins of the transverse spacing."""
        if not self.periodic(boxes):
            raise ValueError(f"the 3D statistics need n_cells = {boxes.n_cells} to be a multiple of both strides {self.stride}.")
        u = float(period) / float(boxes.box_cMpc)
        n_x, n_z = boxes.n_cells // self.stride[0], boxes.n_cells // self.stride[1]
        d_perp = np.array(self.stride, dtype=float) * boxes.dx * u
        k = bao_k_edges(2 * np.pi / period, float(np.pi / d_perp.max()), 3.0 / (boxes.sigma_bao * u))
        dr, half = 3 * boxes.dx * u, 0.5 * period
        top = max(half, min((boxes.r_bao + 4 * boxes.sigma_bao) * u, np.sqrt(3.0) * half))
        r = np.arange(0.0, top + 1e-9 * top, dr)
        return GridStatistics3D(
            (n_x, self.n_pix, n_z),
            (n_x * d_perp[0], float(period), n_z * d_perp[1]),
            los_axis=1,
            k_edges=k,
            k_mu_edges=k,
            r_edges=r,
            r_mu_edges=r,
            rpar_edges=np.arange(0.0, half + 1e-9 * half, dr),
            rperp_edges=np.arange(0.0, half + 1e-9 * half, float(d_perp.min())),
            multipoles=True,
        )


def _claim(path: Path) -> bool:
    """Claim a realisation with a lock file holding the pid; a lock of a process that no longer runs is taken over."""
    lock = path.with_suffix(".lock")
    for _ in range(2):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return True
        except FileExistsError:
            try:
                pid = int(lock.read_text() or 0)
                os.kill(pid, 0)
                return False  # another live process works on it
            except (ProcessLookupError, ValueError):
                lock.unlink(missing_ok=True)  # stale: take it over
            except PermissionError:
                return False
    return False


def skewer_box(boxes: ToyBoxes, i: int, setup: SkewerRun = SkewerRun(), *, columns=None, fields=None) -> dict:
    """tau and the statistics of realisation i along the skewers of ``setup`` (one call of ``run_skewers``).

    tau (2, N_sk, n_pix) of Lya and Si III, periodic over one box, both on the Lya axis (Si III at v + delta_v). 1D: the
    flux contrast of each line, delta_F = F / <F> - 1 (per-skewer means), its products and total (``DeltaFieldBuilder``)
    and their ``OverfluxStatistics1D.summary``. 3D (the lattice of ``setup``, i.e. ``columns`` None, and
    ``SkewerRun.periodic``): the overflux of each line and of the total against the mean of the box, delta = F /
    <F>_box - 1 (``OVERFLUX_FIELDS``), its statistics on the lattice in velocity units (``SkewerRun.grid3d``,
    ``GridStatistics3D.measure``) and its P1D, the mean over the skewers of (dv / n_pix) |rfft(delta)|^2 at k = 2 pi
    rfftfreq(n_pix, dv) [s/km].
    Returns dict(tau, columns, v, period, dv, flux (the 1D summary), flux3d {field: {OVERFLUX_KEYS: array}}, axes3d
    (``GridStatistics3D.axes``), k1d, p1d (3, N_k) (None for the 3D entries without a lattice), mean_F (2,),
    mean_F_total, resolution (n_batches, 2, 2): max W / b and column fraction above tolerance per batch and line,
    warnings (list of str), time_s {fields, pieces, tau, stats}).
    """
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
        tau = np.asarray(tau)
        t["pieces"] += t1 - t0
        t["tau"] += time.perf_counter() - t1
        taus.append(tau)
        r = info["resolution"]
        res.append(np.stack([r["max_ratio"], r["column_fraction"]], -1))
        msgs += [str(w.message) for w in caught]
    tau = np.concatenate(taus, axis=1)
    t0 = time.perf_counter()
    dv = period / setup.n_pix
    flux = OverfluxStatistics1D(DeltaFieldBuilder(tau, labels=list(TERM_LABELS)).build(), dv).summary()
    F = np.exp(-tau)
    flux3d = axes3d = k1d = p1d = None
    if lattice and setup.periodic(boxes):
        f3 = np.concatenate([F, F.prod(0)[None]])  # (3, N_sk, n_pix): each line, then the total
        delta = f3 / f3.mean(axis=(1, 2), keepdims=True) - 1.0
        gs = setup.grid3d(boxes, period)
        n_x, n_z = gs.shape[0], gs.shape[2]
        on_grid = lambda d: d.reshape(n_x, n_z, setup.n_pix).transpose(0, 2, 1)  # columns are x-major: (x, y = los, z)
        flux3d = {name: gs.measure(on_grid(d)) for name, d in zip(OVERFLUX_FIELDS, delta)}
        axes3d = gs.axes()
        k1d = 2 * np.pi * np.fft.rfftfreq(setup.n_pix, dv)
        p1d = (dv / setup.n_pix) * np.mean(np.abs(np.fft.rfft(delta, axis=-1)) ** 2, axis=1)
    t["stats"] = time.perf_counter() - t0
    return dict(
        tau=tau,
        columns=cols,
        v=v,
        period=period,
        dv=dv,
        flux=flux,
        flux3d=flux3d,
        axes3d=axes3d,
        k1d=k1d,
        p1d=p1d,
        mean_F=F.mean(axis=(1, 2)),
        mean_F_total=float(F.prod(0).mean()),
        resolution=np.stack(res),
        warnings=sorted(set(msgs)),
        time_s=t,
    )


def run_skewers(
    boxes: ToyBoxes,
    cache_dir,
    setup: SkewerRun = SkewerRun(),
    *,
    realisations=None,
    keep=(0,),
    overwrite: bool = False,
    log=print,
) -> Path:
    """``skewer_box`` for every realisation (default all), one file each in ``setup.folder(boxes, cache_dir)``:
    ``box_<i>.npz`` (the 1D summary, the 3D statistics and P1D of the overfluxes, the mean fluxes, the resolution check,
    the warnings and the timings) and, for the realisations in ``keep``, ``skewers_<i>.npz`` (tau, the columns and the
    axis: the figures). Realisations already done are skipped, and one claimed by another live process too (lock
    files), so the run is resumable and several processes (e.g. CPU and GPU) can share it. ``overwrite=True`` first
    deletes every result of the folder (``box_*.npz``, ``skewers_*.npz``), so the run recomputes the realisations from
    scratch and the folder holds only them. The warnings of the optical depth (the resolution check) are stored and
    issued again. Returns the folder."""
    folder = setup.folder(boxes, cache_dir)
    folder.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for p in list(folder.glob("box_*.npz")) + list(folder.glob("skewers_*.npz")):
            p.unlink()
    (folder / "config.json").write_text(json.dumps(dict(boxes=asdict(boxes), setup=asdict(setup)), indent=1))
    idx = np.arange(boxes.n_real) if realisations is None else np.asarray(realisations)
    for i in idx:
        path = folder / f"box_{int(i):03d}.npz"
        if path.exists() or not _claim(path):
            continue
        try:
            t0 = time.perf_counter()
            r = skewer_box(boxes, int(i), setup)
            out = {f"flux_{k}": v for k, v in r["flux"].items()}
            if r["flux3d"] is not None:
                for name, rec in r["flux3d"].items():
                    out.update({f"f3d_{name}_{k}": v for k, v in rec.items()})
                out.update({f"f3d_axis_{k}": v for k, v in r["axes3d"].items()})
                out.update(k1d=r["k1d"], p1d=r["p1d"])
            out.update(
                mean_F=r["mean_F"],
                mean_F_total=np.array(r["mean_F_total"]),
                resolution=r["resolution"],
                warnings=np.array(r["warnings"] or [""], dtype=str),
                time_s=np.array([r["time_s"][k] for k in ("fields", "pieces", "tau", "stats")]),
                device=np.array(jax.devices()[0].platform),
                period=np.array(r["period"]),
            )
            if int(i) in set(int(k) for k in keep):
                _save_npz(
                    folder / f"skewers_{int(i):03d}.npz",
                    dict(tau=r["tau"], columns=r["columns"], v=r["v"], period=np.array(r["period"])),
                )
            _save_npz(path, out)
            for m in r["warnings"]:
                warnings.warn(f"realisation {int(i)}: {m}", stacklevel=2)
            if log:
                log(
                    f"realisation {int(i)}: {r['columns'].shape[0]} skewers in {time.perf_counter() - t0:.0f} s "
                    f"({jax.devices()[0].platform}); <F> = {np.round(r['mean_F'], 3)}; max column fraction above "
                    f"tolerance {r['resolution'][..., 1].max():.1e}"
                )
        finally:
            path.with_suffix(".lock").unlink(missing_ok=True)
    return folder


def load_skewer_run(folder, *, realisations=None) -> dict:
    """The ensemble of a ``run_skewers`` folder: dict(flux (``ensemble_1d`` over the realisations), flux3d {field:
    ``ensemble_mean`` of the ``OVERFLUX_KEYS`` statistics} for the ``OVERFLUX_FIELDS``, axes3d (the bins, velocity units),
    p1d (``ensemble_mean`` of "p1d" (3, N_k), with "k" [s/km]) (flux3d, axes3d, p1d None without a lattice), mean_F
    (n, 2), mean_F_total (n,), resolution (n, n_batches, 2, 2), warnings {i: [str]}, time_s (n, 4) [fields, pieces,
    tau, stats], device (n,), realisations (n,), skewers {i: dict} of the kept realisations, config)."""
    folder = Path(folder)
    files = sorted(folder.glob("box_*.npz"))
    if realisations is not None:
        want = {int(i) for i in realisations}
        files = [p for p in files if int(p.stem[4:]) in want]
    if not files:
        raise FileNotFoundError(f"no realisation in {folder}: run run_skewers first.")
    zs = [dict(np.load(p, allow_pickle=False)) for p in files]
    part = lambda z, pre: {k[len(pre) :]: v for k, v in z.items() if k.startswith(pre)}
    kept = {int(p.stem[8:]): dict(np.load(p, allow_pickle=False)) for p in sorted(folder.glob("skewers_*.npz"))}
    flux3d = axes3d = p1d = None
    if all("p1d" in z for z in zs):
        flux3d = {
            name: ensemble_mean([{k: z[f"f3d_{name}_{k}"] for k in OVERFLUX_KEYS} for z in zs], OVERFLUX_KEYS)
            for name in OVERFLUX_FIELDS
        }
        axes3d = part(zs[0], "f3d_axis_")
        p1d = dict(ensemble_mean(zs, ("p1d",)), k=zs[0]["k1d"])
    return dict(
        flux=ensemble_1d([part(z, "flux_") for z in zs]),
        flux3d=flux3d,
        axes3d=axes3d,
        p1d=p1d,
        mean_F=np.stack([z["mean_F"] for z in zs]),
        mean_F_total=np.array([float(z["mean_F_total"]) for z in zs]),
        resolution=np.stack([z["resolution"] for z in zs]),
        warnings={int(p.stem[4:]): [m for m in z["warnings"] if m] for p, z in zip(files, zs)},
        time_s=np.stack([z["time_s"] for z in zs]),
        device=np.array([str(z["device"]) for z in zs]),
        realisations=np.array([int(p.stem[4:]) for p in files]),
        skewers=kept,
        config=json.loads((folder / "config.json").read_text()),
    )
