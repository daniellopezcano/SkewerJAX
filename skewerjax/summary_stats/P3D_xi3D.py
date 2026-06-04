"""
summary_stats/P3D_xi3D.py
=========================
3D power spectrum (P3D) and 3D two-point correlation function (xi3D)
for a regular grid of absorption skewers.

Design mirrors P1D_xi1D.OverfluxStatistics1D so that the two classes can
be used side-by-side in the same analysis.

Geometry convention
-------------------
The skewer grid is assumed to cover a cubic (or rectangular) periodic box
with N_xy × N_xy skewers in the transverse plane and N_z = Npix pixels along
the line-of-sight (LoS) axis.  The skewer index ordering follows the
row-major (C) convention:

    skewer index  i = ix * N_xy + iy,   ix, iy ∈ {0, …, N_xy - 1}

The field array fed to this class has shape

    delta_terms[key]  :  (N_sk, Npix)   with  N_sk = N_xy * N_xy

and is promoted internally to a 3D cube of shape (N_xy, N_xy, Npix) for
3D FFT computation.

FFT / normalisation conventions (identical to LaCe / measure_flux_power.py)
---------------------------------------------------------------------------
The 3D power spectrum is defined so that for white noise with variance σ²:

    P3D ≈ σ² * dx_perp² * dv

where dx_perp is the transverse pixel size and dv the LoS pixel size.

Concretely:

    P̂(k)  =  V_box / N_tot²  *  |FFT(δ)|²   [units of volume]

with V_box = L_perp² * L_los  and  N_tot = N_xy² * Npix.

The (k, μ) decomposition uses

    k   = |k_vec|
    μ   = |k_z| / k        (cosine of angle to LoS)

and bins power into (k, μ) cells following the same approach as
measure_p3d_Mpc in LaCe.

The 3D 2PCF is the inverse FFT of the isotropically-averaged P3D
(Wiener–Khinchin), evaluated on a (r, π) grid where r is the transverse
separation and π the LoS separation.

Usage
-----
    stats3d = OverfluxStatistics3D(
        overflux,
        dv,               # LoS pixel size [km/s or Mpc]
        dx_perp,          # transverse pixel / skewer spacing [same units]
        N_xy,             # number of skewers per transverse side
    )
    stats3d.compute_p3d()
    stats3d.compute_xi3d()

Example calls are given at the bottom of this file.
"""

from __future__ import annotations

import logging
from itertools import product as iterproduct

import numpy as np
import scipy.stats as spt
import jax.numpy as jnp
from jax.numpy.fft import fftn, ifftn, fftfreq, fftshift

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: build 3D wavenumber arrays for a grid
# ---------------------------------------------------------------------------

def _build_k_arrays(N_xy: int, Npix: int, dx_perp: float, dv: float):
    """
    Build 3D (k_perp_x, k_perp_y, k_z) grids and derived (k, mu) grids.

    Returns
    -------
    k_box : ndarray, shape (N_xy, N_xy, Npix)
        |k| for every mode.
    mu_box : ndarray, shape (N_xy, N_xy, Npix)
        |k_z| / k (NaN for k=0).
    k_perp_box : ndarray, shape (N_xy, N_xy, Npix)
        sqrt(k_x² + k_y²).
    k_los_box  : ndarray, shape (N_xy, N_xy, Npix)
        |k_z|.
    """
    k_xy = np.fft.fftfreq(N_xy, d=dx_perp) * 2.0 * np.pi   # (N_xy,)
    k_z  = np.fft.fftfreq(Npix,  d=dv)     * 2.0 * np.pi   # (Npix,)

    kx = k_xy[:, np.newaxis, np.newaxis]   # (N_xy, 1, 1)
    ky = k_xy[np.newaxis, :, np.newaxis]   # (1, N_xy, 1)
    kz = k_z[np.newaxis, np.newaxis, :]    # (1, 1, Npix)

    k_perp_box = np.sqrt(kx**2 + ky**2)
    k_los_box  = np.abs(kz) * np.ones_like(k_perp_box)
    k_box      = np.sqrt(k_perp_box**2 + k_los_box**2)

    # mu = |k_z| / k, with NaN at k=0
    mu_box = np.where(k_box > 0.0, k_los_box / k_box, np.nan)

    return k_box, mu_box, k_perp_box, k_los_box


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class OverfluxStatistics3D:
    """
    Compute P3D(k, μ) and xi3D(r_perp, π) from a built DeltaFieldBuilder.

    Parameters
    ----------
    overflux : DeltaFieldBuilder (after .build())
        Must expose: subset_keys, delta_terms, C_per_sample, total.
        The 'samples' axis (axis 0 of each delta_terms array) is the skewer
        axis here: N_samples = N_sk = N_xy².
    dv : float
        Line-of-sight pixel size (km/s, Mpc, or any consistent unit).
    dx_perp : float
        Transverse spacing between skewers (same units as dv).
    N_xy : int
        Number of skewers per transverse side.  Must satisfy N_xy² == N_sk.
    c_norm_min : float
        Minimum |C_norm| threshold for validity masking (same role as in 1D).
    k_max : float or None
        Maximum |k| for binned P3D outputs.  If None, uses the Nyquist limit
        along the LoS axis.
    n_k_bins : int
        Number of logarithmic |k| bins for the binned P3D.
    n_mu_bins : int
        Number of linear μ ∈ [0, 1] bins for the binned P3D.

    Notes
    -----
    * The 3D FFT convention follows measure_flux_power.py / LaCe:
          P3D = V_box / N_tot² * |FFT(δ)|²
    * xi3D is computed via inverse FFT of the *full* (un-binned)
      3D power, then evaluated on a (r_perp, π) grid by azimuthal
      averaging over the transverse plane.
    * The pair decomposition (cross-spectra between line components)
      is carried through in the same way as the 1D class.
    """

    def __init__(
        self,
        overflux,
        dv: float,
        dx_perp: float,
        N_xy: int,
        *,
        c_norm_min: float = 1e-12,
        k_max: float | None = None,
        n_k_bins: int = 20,
        n_mu_bins: int = 16,
    ):
        # ------------------------------------------------------------------ #
        # 1. Validate required attributes
        # ------------------------------------------------------------------ #
        for attr in ("subset_keys", "delta_terms", "C_per_sample", "total"):
            if not hasattr(overflux, attr):
                raise ValueError(
                    f"`overflux` is missing required attribute '{attr}'."
                )

        self.dv      = float(dv)
        self.dx_perp = float(dx_perp)
        self.N_xy    = int(N_xy)
        self.c_norm_min = float(c_norm_min)
        self.n_k_bins   = int(n_k_bins)
        self.n_mu_bins  = int(n_mu_bins)

        # ------------------------------------------------------------------ #
        # 2. Load arrays and check shapes
        # ------------------------------------------------------------------ #
        total        = jnp.asarray(overflux.total)          # (N_sk, Npix)
        C_per_sample = jnp.asarray(overflux.C_per_sample)   # (N_sk, 1)
        C_norm       = (1.0 + C_per_sample) ** 2            # (N_sk, 1)
        delta_terms  = {
            key: jnp.asarray(overflux.delta_terms[key])
            for key in overflux.subset_keys
        }

        N_sk, Npix = total.shape
        expected_Nsk = self.N_xy ** 2
        if N_sk != expected_Nsk:
            raise ValueError(
                f"N_xy={N_xy} implies N_sk={expected_Nsk} skewers, "
                f"but delta_terms have N_sk={N_sk}. "
                "Check N_xy or the overflux object."
            )

        self.N_sk  = N_sk
        self.Npix  = Npix

        # ------------------------------------------------------------------ #
        # 3. Validity mask (same logic as OverfluxStatistics1D)
        # ------------------------------------------------------------------ #
        valid_mask = (
            jnp.all(jnp.isfinite(C_norm), axis=1)
            & jnp.all(jnp.abs(C_norm) > self.c_norm_min, axis=1)
            & jnp.all(jnp.isfinite(total), axis=1)
        )
        for key in overflux.subset_keys:
            valid_mask &= jnp.all(jnp.isfinite(delta_terms[key]), axis=1)

        self.valid_mask           = valid_mask
        self.n_samples_initial    = N_sk
        self.n_samples_kept       = int(jnp.sum(valid_mask))
        self.n_samples_removed    = N_sk - self.n_samples_kept

        if self.n_samples_kept == 0:
            raise ValueError("All skewers were removed by validity checks.")
        if self.n_samples_removed > 0:
            logger.warning(
                "Removed %d invalid skewers out of %d.",
                self.n_samples_removed,
                N_sk,
            )

        # ------------------------------------------------------------------ #
        # 4. Keep only valid skewers and reshape into 3D cubes
        # ------------------------------------------------------------------ #
        # After masking we can no longer guarantee a square transverse grid.
        # We keep the original (N_sk, Npix) layout for cross-spectra, but
        # promote only the *total* field to 3D for the isotropic P3D.
        # The pair cross-spectra are computed directly in 1D over the skewer
        # axis following the same (dv/Npix) normalization as OverfluxStatistics1D,
        # then the 3D power is assembled from those 1D spectra.
        #
        # If ALL skewers are valid (the common case) we reshape to a true 3D
        # cube and perform a genuine 3D FFT.  The full-cube path is flagged
        # by self._full_cube.

        self._full_cube = (self.n_samples_removed == 0)

        self.total       = total[valid_mask]           # (N_sk_valid, Npix)
        self.C_per_sample = C_per_sample[valid_mask]
        self.C_norm      = C_norm[valid_mask]
        self.delta_terms = {
            key: delta_terms[key][valid_mask]
            for key in overflux.subset_keys
        }

        self.subset_keys   = overflux.subset_keys
        self.ordered_pairs = list(iterproduct(self.subset_keys, repeat=2))

        # ------------------------------------------------------------------ #
        # 5. Box geometry
        # ------------------------------------------------------------------ #
        self.L_perp  = self.N_xy * self.dx_perp          # transverse box size
        self.L_los   = self.Npix  * self.dv               # LoS box size
        self.V_box   = self.L_perp**2 * self.L_los        # box volume
        self.N_tot   = self.N_xy**2   * self.Npix         # total voxels

        # ------------------------------------------------------------------ #
        # 6. Wavenumber grids (NumPy, for indexing)
        # ------------------------------------------------------------------ #
        self.k_box, self.mu_box, self.k_perp_box, self.k_los_box = (
            _build_k_arrays(self.N_xy, self.Npix, self.dx_perp, self.dv)
        )

        # Nyquist-based k_max if not provided
        if k_max is None:
            k_max = np.pi / max(self.dx_perp, self.dv)
        self.k_max = float(k_max)

        # 1D wavenumber arrays for output (along each axis)
        self.k_los_1d = (
            np.fft.fftfreq(self.Npix, d=self.dv) * 2.0 * np.pi
        )
        self.k_xy_1d = (
            np.fft.fftfreq(self.N_xy, d=self.dx_perp) * 2.0 * np.pi
        )

        # Lag arrays for xi3D output
        self.lags_los  = (
            np.arange(self.Npix)  - self.Npix  // 2
        ) * self.dv
        self.lags_perp = (
            np.arange(self.N_xy)  - self.N_xy  // 2
        ) * self.dx_perp

        # ------------------------------------------------------------------ #
        # 7. Output containers (populated by compute_p3d / compute_xi3d)
        # ------------------------------------------------------------------ #
        # --- isotropic binned P3D ---
        self.k_bin_centers    = None   # (n_k_bins,)
        self.mu_bin_centers   = None   # (n_mu_bins,)
        self.Pk3d_total       = None   # (n_k_bins, n_mu_bins)
        self.Pk3d_counts      = None   # (n_k_bins, n_mu_bins)

        self.Pk3d_pair        = None   # dict (a,b) -> (n_k_bins, n_mu_bins)

        # --- full-cube 3D power (all k modes) ---
        self.Pk3d_total_cube  = None   # (N_xy, N_xy, Npix)
        self.Pk3d_pair_cube   = None   # dict (a,b) -> (N_xy, N_xy, Npix)

        # --- 3D 2PCF on (r_perp, pi) grid ---
        self.xi3d_total       = None   # (N_xy, N_xy, Npix)
        self.xi3d_pair        = None   # dict (a,b) -> (N_xy, N_xy, Npix)

        # mean-subtracted C correction (analogous to 1D xi)
        self.Pk3d_total_from_terms        = None
        self.Pk3d_total_from_terms_contributions = None

    # ---------------------------------------------------------------------- #
    # Private helpers
    # ---------------------------------------------------------------------- #

    def _to_cube(self, field_2d: jnp.ndarray) -> jnp.ndarray:
        """
        Reshape (N_sk, Npix) -> (N_xy, N_xy, Npix).

        Requires that no skewers were removed (self._full_cube is True).
        Caller is responsible for checking this.
        """
        return field_2d.reshape(self.N_xy, self.N_xy, self.Npix)

    def _p3d_from_cube(self, cube: jnp.ndarray) -> jnp.ndarray:
        """
        Compute raw (un-binned) P3D cube from a real-space delta cube.

        P3D(k) = V_box / N_tot² * |FFT(δ)|²

        Returns
        -------
        power_cube : jnp.ndarray, shape (N_xy, N_xy, Npix)
        """
        modes = fftn(cube) / self.N_tot
        return (jnp.abs(modes) ** 2 * self.V_box).real

    def _fft_of_field(self, field_2d: jnp.ndarray) -> jnp.ndarray:
        """
        FFT of a 2D (N_sk, Npix) skewer field, promoted to the 3D cube.
        """
        cube = self._to_cube(field_2d)
        return fftn(cube) / self.N_tot

    def _bin_p3d(
        self,
        power_cube: np.ndarray,
    ):
        """
        Bin a raw P3D cube into (k, mu) cells.

        Parameters
        ----------
        power_cube : ndarray or jnp.ndarray, shape (N_xy, N_xy, Npix)

        Returns
        -------
        binned_p3d     : ndarray, shape (n_k_bins, n_mu_bins)
        binned_k       : ndarray  (mean |k| per bin)
        binned_mu      : ndarray  (mean μ per bin)
        binned_counts  : ndarray
        k_bin_edges    : ndarray
        mu_bin_edges   : ndarray
        """
        power_cube = np.asarray(power_cube)

        k_flat   = self.k_box.flatten()[1:]     # skip k=0
        mu_flat  = np.abs(self.mu_box).flatten()[1:]
        p_flat   = power_cube.flatten()[1:]

        # Filter out NaN mu (k=0 mode, already skipped above)
        finite_mask = np.isfinite(mu_flat) & np.isfinite(p_flat)
        k_flat  = k_flat[finite_mask]
        mu_flat = mu_flat[finite_mask]
        p_flat  = p_flat[finite_mask]

        # Logarithmic k bins
        k_pos = k_flat[k_flat > 0.0]
        lnk_min = np.log(0.9999 * k_pos.min())
        lnk_max = np.log(self.k_max)
        lnk_bin_max = lnk_max + (lnk_max - lnk_min) / (self.n_k_bins - 1)
        lnk_edges = np.linspace(lnk_min, lnk_bin_max, self.n_k_bins + 1)
        k_bin_edges = np.exp(lnk_edges)

        mu_bin_edges = np.linspace(0.0, 1.0, self.n_mu_bins + 1)

        bins = [k_bin_edges, mu_bin_edges]

        binned_p3d    = spt.binned_statistic_2d(
            k_flat, mu_flat, p_flat, statistic="mean", bins=bins
        )[0]
        binned_k      = spt.binned_statistic_2d(
            k_flat, mu_flat, k_flat, statistic="mean", bins=bins
        )[0]
        binned_mu     = spt.binned_statistic_2d(
            k_flat, mu_flat, mu_flat, statistic="mean", bins=bins
        )[0]
        binned_counts = spt.binned_statistic_2d(
            k_flat, mu_flat, p_flat, statistic="count", bins=bins
        )[0]

        return binned_p3d, binned_k, binned_mu, binned_counts, k_bin_edges, mu_bin_edges

    # ---------------------------------------------------------------------- #
    # Public methods
    # ---------------------------------------------------------------------- #

    def compute_p3d(self):
        """
        Compute 3D power spectra for the total field and all component pairs.

        Populates
        ---------
        self.Pk3d_total_cube   : raw P3D cube of the total overflux field
        self.Pk3d_pair_cube    : dict of raw P3D cross-spectra for each pair
        self.Pk3d_total        : (n_k_bins, n_mu_bins)  binned P3D of total field
        self.Pk3d_pair         : dict, binned cross P3D per ordered pair
        self.k_bin_centers     : (n_k_bins, n_mu_bins)  mean |k| per bin
        self.mu_bin_centers    : (n_k_bins, n_mu_bins)  mean μ per bin
        self.Pk3d_counts       : (n_k_bins, n_mu_bins)  number of modes per bin
        self.Pk3d_total_from_terms          : (n_k_bins, n_mu_bins)
        self.Pk3d_total_from_terms_contributions : dict per ordered pair

        Returns
        -------
        self  (for chaining)
        """
        if self.Pk3d_total is not None:
            return self

        if not self._full_cube:
            raise RuntimeError(
                "compute_p3d() in full-cube mode requires all N_xy² skewers "
                "to be valid.  Some skewers were removed; the 3D cube cannot "
                "be assembled without them.  Consider padding with zeros or "
                "using only the per-skewer 1D statistics."
            )

        logger.info("Starting compute_p3d().")

        # ------------------------------------------------------------------ #
        # Total field P3D
        # ------------------------------------------------------------------ #
        total_cube       = self._to_cube(self.total)
        total_modes      = self._fft_of_field(self.total)
        p3d_total_cube   = (jnp.abs(total_modes) ** 2 * self.V_box).real
        self.Pk3d_total_cube = p3d_total_cube

        # ------------------------------------------------------------------ #
        # Component FFTs (one per subset_key)
        # ------------------------------------------------------------------ #
        f_terms = {
            key: self._fft_of_field(self.delta_terms[key])
            for key in self.subset_keys
        }

        # ------------------------------------------------------------------ #
        # Cross-power cubes for every ordered pair
        # ------------------------------------------------------------------ #
        self.Pk3d_pair_cube = {}
        self.Pk3d_total_from_terms_contributions = {}
        total_from_terms_cube = jnp.zeros_like(p3d_total_cube)

        # Mean C_norm over skewers (scalar, for the decomposition correction)
        C_norm_mean = float(jnp.mean(self.C_norm))

        for pair in self.ordered_pairs:
            a, b = pair
            p_ab_cube = (
                self.V_box
                * (f_terms[a] * jnp.conjugate(f_terms[b])).real
            )
            self.Pk3d_pair_cube[pair] = p_ab_cube
            contribution = p_ab_cube / C_norm_mean
            self.Pk3d_total_from_terms_contributions[pair] = contribution
            total_from_terms_cube = total_from_terms_cube + contribution

        self.Pk3d_total_from_terms_cube = total_from_terms_cube

        # ------------------------------------------------------------------ #
        # Bin everything into (k, mu) cells
        # ------------------------------------------------------------------ #
        (
            binned_total,
            binned_k,
            binned_mu,
            binned_counts,
            k_bin_edges,
            mu_bin_edges,
        ) = self._bin_p3d(np.asarray(p3d_total_cube))

        self.Pk3d_total      = binned_total
        self.k_bin_centers   = binned_k
        self.mu_bin_centers  = binned_mu
        self.Pk3d_counts     = binned_counts
        self.k_bin_edges     = k_bin_edges
        self.mu_bin_edges    = mu_bin_edges

        self.Pk3d_pair = {}
        for pair in self.ordered_pairs:
            self.Pk3d_pair[pair] = self._bin_p3d(
                np.asarray(self.Pk3d_pair_cube[pair])
            )[0]

        self.Pk3d_total_from_terms = self._bin_p3d(
            np.asarray(total_from_terms_cube)
        )[0]
        # Also bin each contribution
        self.Pk3d_total_from_terms_contributions = {
            pair: self._bin_p3d(
                np.asarray(self.Pk3d_total_from_terms_contributions[pair])
            )[0]
            for pair in self.ordered_pairs
        }

        logger.info("Finished compute_p3d().")
        return self

    def compute_xi3d(self):
        """
        Compute 3D two-point correlation function via inverse FFT of P3D.

        Must call compute_p3d() first (or it will be called automatically).

        The full 3D xi cube has shape (N_xy, N_xy, Npix) with axes
        (r_perp_x, r_perp_y, pi).  The fftshift brings the zero-lag to the
        cube centre.

        Populates
        ---------
        self.xi3d_total  : (N_xy, N_xy, Npix)  — total field 2PCF
        self.xi3d_pair   : dict (a,b) -> (N_xy, N_xy, Npix)
        self.xi3d_total_from_terms : (N_xy, N_xy, Npix)
        self.r_perp_grid : 2D ndarray (N_xy, N_xy) — transverse lag |r_perp|
        self.pi_grid     : 1D ndarray (Npix,)        — LoS lag π
        self.r3d_grid    : 3D ndarray (N_xy, N_xy, Npix) — |r| = sqrt(r_perp² + π²)

        Returns
        -------
        self  (for chaining)
        """
        if self.xi3d_total is not None:
            return self

        self.compute_p3d()

        logger.info("Starting compute_xi3d().")

        def _p3d_to_xi3d(p3d_cube: jnp.ndarray) -> jnp.ndarray:
            """Inverse FFT of a P3D cube -> xi3D, centred with fftshift."""
            # xi(r) = 1/V * sum_k P(k) * exp(i k.r)
            # Numerically: irfft of (P / V_box) gives xi with correct normalization.
            xi_raw = jnp.real(ifftn(p3d_cube / self.V_box)) * self.N_tot
            return jnp.fft.fftshift(xi_raw)

        # Total
        self.xi3d_total = _p3d_to_xi3d(self.Pk3d_total_cube)

        # Pairs
        self.xi3d_pair = {
            pair: _p3d_to_xi3d(self.Pk3d_pair_cube[pair])
            for pair in self.ordered_pairs
        }

        # Decomposition total
        C_norm_mean = float(jnp.mean(self.C_norm))
        xi_from_terms = jnp.zeros_like(self.xi3d_total)
        for pair in self.ordered_pairs:
            xi_from_terms = xi_from_terms + _p3d_to_xi3d(
                self.Pk3d_pair_cube[pair]
            ) / C_norm_mean
        # Subtract mean-field correction (analogous to the 1D correction)
        correction = float(
            jnp.mean(jnp.asarray(self.C_per_sample) ** 2 / self.C_norm)
        )
        self.xi3d_total_from_terms = xi_from_terms - correction

        # ------------------------------------------------------------------ #
        # Lag grids for convenience
        # ------------------------------------------------------------------ #
        # 1D lag arrays (centred)
        pi_arr    = self.lags_los                   # (Npix,)
        rperp_arr = self.lags_perp                  # (N_xy,)

        rx = rperp_arr[:, np.newaxis, np.newaxis]   # (N_xy, 1, 1)
        ry = rperp_arr[np.newaxis, :, np.newaxis]   # (1, N_xy, 1)
        rz = pi_arr[np.newaxis, np.newaxis, :]      # (1, 1, Npix)

        self.r3d_grid    = np.sqrt(rx**2 + ry**2 + rz**2)  # (N_xy, N_xy, Npix)
        self.r_perp_grid = np.sqrt(rx**2 + ry**2)[..., 0]  # (N_xy, N_xy)
        self.pi_grid     = pi_arr                           # (Npix,)

        logger.info("Finished compute_xi3d().")
        return self

    # ---------------------------------------------------------------------- #
    # Convenience: isotropic 1D P(k) by averaging over mu bins
    # ---------------------------------------------------------------------- #

    def isotropic_p3d(self, field: str = "total") -> tuple[np.ndarray, np.ndarray]:
        """
        Return isotropically averaged P(k) by summing over all μ bins.

        Parameters
        ----------
        field : "total" or a valid subset_key

        Returns
        -------
        k_iso : ndarray, shape (n_k_bins,)
        Pk_iso : ndarray, shape (n_k_bins,)
        """
        if self.Pk3d_total is None:
            self.compute_p3d()

        if field == "total":
            p_table = self.Pk3d_total
        else:
            p_table = self.Pk3d_pair.get((field, field))
            if p_table is None:
                raise KeyError(
                    f"No auto-power for key '{field}'. "
                    "Use a valid subset_key or 'total'."
                )

        counts = self.Pk3d_counts
        with np.errstate(invalid="ignore"):
            k_iso  = np.nansum(self.k_bin_centers * counts, axis=1) / np.nansum(counts, axis=1)
            Pk_iso = np.nansum(p_table            * counts, axis=1) / np.nansum(counts, axis=1)
        return k_iso, Pk_iso

    # ---------------------------------------------------------------------- #
    # Convenience: xi3D projected onto (r_perp, pi) wedges
    # ---------------------------------------------------------------------- #

    def xi3d_rppi(
        self,
        field: str = "total",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return xi(r_perp, pi) by azimuthally averaging xi3D over the two
        transverse axes.

        Parameters
        ----------
        field : "total" or a valid subset_key

        Returns
        -------
        r_perp : 1D ndarray, shape (N_xy // 2,)    unique positive r_perp values
        pi     : 1D ndarray, shape (Npix,)          LoS lag grid
        xi_rppi : 2D ndarray, shape (N_xy // 2, Npix)
        """
        if self.xi3d_total is None:
            self.compute_xi3d()

        if field == "total":
            xi_cube = np.asarray(self.xi3d_total)
        else:
            xi_cube = np.asarray(self.xi3d_pair.get((field, field)))
            if xi_cube is None:
                raise KeyError(
                    f"No auto-xi for key '{field}'. "
                    "Use a valid subset_key or 'total'."
                )

        # Azimuthal average: bin by r_perp distance
        rperp_flat = self.r_perp_grid.flatten()   # (N_xy²,)
        r_unique   = np.unique(np.round(rperp_flat, 8))
        r_unique   = r_unique[r_unique >= 0.0]

        xi_rppi = np.zeros((len(r_unique), self.Npix))
        for ir, r in enumerate(r_unique):
            mask = np.abs(rperp_flat - r) < 1e-10 * max(r, self.dx_perp)
            xi_rppi[ir] = np.mean(
                xi_cube.reshape(self.N_xy * self.N_xy, self.Npix)[mask], axis=0
            )

        return r_unique, self.pi_grid, xi_rppi
