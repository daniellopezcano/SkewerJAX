import logging
from itertools import product

import numpy as np
import jax.numpy as jnp
from jax.numpy.fft import fftn, ifftn, fftfreq, fftshift

logger = logging.getLogger(__name__)


class OverfluxStatistics3D:
    """
    Compute P3D and xi3D summary statistics from a built DeltaFieldBuilder
    object on a regular square skewer grid.

    Geometry
    --------
    The overflux object must contain delta_terms whose first axis indexes
    skewers arranged on a square N_xy x N_xy transverse grid, so that:

        delta_terms[key].shape == (N_sk, N_pix)   with N_sk = N_xy ** 2

    The full 3D field is obtained by reshaping to (N_xy, N_xy, N_pix).

    Axes and units
    --------------
    Transverse axes (x, y):  spacing d_perp  [cMpc]
                              wavenumbers k_x, k_y   [cMpc^-1]
                              lags r_x, r_y, r_perp  [cMpc]
    LoS axis (parallel):      spacing dv      [km/s]
                              wavenumbers k_par  [s/km]
                              lag pi             [km/s]

    The two sets of axes intentionally carry different units and are never
    mixed.  Unit conversion (Eq. xi3D_pi_to_Mpc in the notes) is left to
    a separate utility.

    P3D normalization  [Eq. P3D_DFT_power]
    ----------------------------------------
    DFT performed with jnp.fft.fftn (no internal normalization):

        P_ij(k) = d_perp^2 * dv  *  tilde_delta_i(k) * conj(tilde_delta_j(k))

    Units: [cMpc^2 * km/s].

    xi3D normalization  [Eq. xi3D_sample_definition]
    --------------------------------------------------
    The 3D correlation function is the inverse FFT of P3D, divided by the
    voxel volume to recover the continuous-limit estimator:

        xi_ij(r_perp, pi) = IFFT{ P_ij(k) } / (d_perp^2 * dv)

    Because IFFT(FFT(f) * conj(FFT(g))) = N * (f ★ g) (cross-correlation via
    the convolution theorem, where N = N_xy^2 * N_pix is the total number of
    voxels), the combination

        xi_ij = IFFT{ P_ij } / (d_perp^2 * dv)
              = (d_perp^2 * dv) * IFFT{ F_i * conj(F_j) } / (d_perp^2 * dv)
              = IFFT{ F_i * conj(F_j) }

    where F_i = fftn(delta_i).  The jnp.fft.ifftn call carries the 1/N
    normalization internally, reproducing Eq. (xi3D_sample_definition):

        xi_ij(r_perp, pi) = (1 / N_sk*N_pix) sum_{x,y,v}
                            delta_i(x,y,v) * delta_j(x+r_perp, y+r_perp, v+pi)

    Units: dimensionless (same as the delta fields).

    Parameters
    ----------
    overflux : DeltaFieldBuilder (built)
        Must expose: subset_keys, delta_terms, C_per_sample, total.
        delta_terms[key].shape must be (N_sk, N_pix) with N_sk = N_xy**2.
    dv : float
        Pixel spacing along the LoS [km/s].
    d_perp : float
        Pixel spacing in the transverse plane [cMpc].
    c_norm_min : float, optional
        Minimum |C_norm| = |(1 + C_per_sample)^2| for validity filtering.
    """

    def __init__(self, overflux, dv, d_perp, c_norm_min=1e-12):
        self.dv     = float(dv)
        self.d_perp = float(d_perp)
        self.c_norm_min = float(c_norm_min)

        # ------------------------------------------------------------------ #
        # Validate required attributes from overflux
        # ------------------------------------------------------------------ #
        for attr in ("subset_keys", "delta_terms", "C_per_sample", "total"):
            if not hasattr(overflux, attr):
                raise ValueError(
                    f"`overflux` is missing required attribute '{attr}'."
                )

        # ------------------------------------------------------------------ #
        # Load arrays and derive grid dimensions
        # ------------------------------------------------------------------ #
        total        = jnp.asarray(overflux.total)         # (N_sk, N_pix)
        C_per_sample = jnp.asarray(overflux.C_per_sample)  # (N_sk, 1)
        C_norm       = (1.0 + C_per_sample) ** 2           # (N_sk, 1)

        delta_terms = {
            key: jnp.asarray(overflux.delta_terms[key])
            for key in overflux.subset_keys
        }   # each: (N_sk, N_pix)

        N_sk, N_pix = total.shape

        # Infer N_xy from N_sk = N_xy^2
        N_xy = int(round(N_sk ** 0.5))
        if N_xy * N_xy != N_sk:
            raise ValueError(
                f"N_sk={N_sk} is not a perfect square. "
                "delta_terms must correspond to a square N_xy x N_xy skewer grid."
            )

        self.N_sk   = N_sk
        self.N_xy   = N_xy
        self.N_pix  = N_pix
        self.L_perp  = N_xy  * d_perp   # [cMpc]
        self.L_Lambda = N_pix * dv       # [km/s]
        self.A_perp  = self.L_perp ** 2  # [cMpc^2]

        # ------------------------------------------------------------------ #
        # Sample validity mask  (flat skewer axis)
        # ------------------------------------------------------------------ #
        valid_mask = (
            jnp.all(jnp.isfinite(C_norm), axis=1)
            & jnp.all(jnp.abs(C_norm) > self.c_norm_min, axis=1)
        )
        valid_mask &= jnp.all(jnp.isfinite(total), axis=1)
        for key in overflux.subset_keys:
            valid_mask &= jnp.all(jnp.isfinite(delta_terms[key]), axis=1)

        self.valid_mask        = valid_mask
        self.n_samples_initial = N_sk
        self.n_samples_kept    = int(jnp.sum(valid_mask))
        self.n_samples_removed = N_sk - self.n_samples_kept

        if self.n_samples_kept == 0:
            raise ValueError(
                "All samples were removed by the initialization validity checks."
            )

        if self.n_samples_removed > 0:
            logger.warning(
                "Removed %d invalid skewers out of %d. "
                "They are zeroed (not dropped) to preserve the grid shape.",
                self.n_samples_removed, N_sk,
            )
            zero_row = jnp.zeros((1, N_pix), dtype=total.dtype)
            total = jnp.where(valid_mask[:, None], total, zero_row)
            for key in overflux.subset_keys:
                delta_terms[key] = jnp.where(
                    valid_mask[:, None], delta_terms[key], zero_row
                )

        self.total        = total        # (N_sk, N_pix)
        self.C_per_sample = C_per_sample # (N_sk, 1)
        self.C_norm       = C_norm       # (N_sk, 1)
        self.delta_terms  = delta_terms  # {key: (N_sk, N_pix)}
        self.subset_keys   = overflux.subset_keys
        self.ordered_pairs = list(product(self.subset_keys, repeat=2))

        # ------------------------------------------------------------------ #
        # Wavenumber and lag grids  (computed once at init)
        #
        # Convention:  k = 2π * freq   (same as OverfluxStatistics1D)
        #
        # k_perp_1d / k_par:   1D wavenumber axes
        # r_perp_1d / pi_lags: 1D lag axes after fftshift centering
        #
        # The 3D lag grid stored here is the (r_x, r_y, pi) cube;
        # r_perp = sqrt(r_x^2 + r_y^2) is computed where needed.
        # ------------------------------------------------------------------ #
        self.k_perp_1d = 2.0 * jnp.pi * fftfreq(N_xy,  d=d_perp)  # (N_xy,)  [cMpc^-1]
        self.k_par     = 2.0 * jnp.pi * fftfreq(N_pix, d=dv)      # (N_pix,) [s/km]

        # 3D k arrays broadcast to (N_xy, N_xy, N_pix)
        kx = self.k_perp_1d[:, None, None]
        ky = self.k_perp_1d[None, :, None]
        kz = self.k_par    [None, None, :]
        self.k_perp_3d = jnp.sqrt(kx ** 2 + ky ** 2)
        self.k_par_3d  = jnp.broadcast_to(kz, (N_xy, N_xy, N_pix))

        # Lag axes (fftshift-centered so zero lag is at the centre)
        # r_perp_1d:  shape (N_xy,),  lags in [−L_perp/2, +L_perp/2)  [cMpc]
        # pi_lags:    shape (N_pix,),  lags in [−L_Lambda/2, +L_Lambda/2) [km/s]
        self.r_perp_1d = (
            jnp.array(np.fft.fftshift(np.fft.fftfreq(N_xy)))  * self.L_perp
        )   # (N_xy,)  [cMpc]
        self.pi_lags   = (
            jnp.array(np.fft.fftshift(np.fft.fftfreq(N_pix))) * self.L_Lambda
        )   # (N_pix,) [km/s]

        # ------------------------------------------------------------------ #
        # Output containers  (populated lazily)
        # ------------------------------------------------------------------ #
        # --- P3D ---
        self.Pk3d_total                          = None  # (N_xy, N_xy, N_pix)
        self.Pk3d_pair                           = None  # dict
        self.Pk3d_total_from_terms               = None  # (N_xy, N_xy, N_pix)
        self.Pk3d_total_from_terms_contributions = None  # dict

        # --- xi3D ---
        self.xi3d_total                          = None  # (N_xy, N_xy, N_pix)
        self.xi3d_pair                           = None  # dict
        self.xi3d_total_from_terms               = None  # (N_xy, N_xy, N_pix)
        self.xi3d_total_from_terms_contributions = None  # dict

        logger.info(
            "OverfluxStatistics3D initialized: N_xy=%d, N_pix=%d, "
            "d_perp=%.4f cMpc, dv=%.4f km/s.",
            N_xy, N_pix, d_perp, dv,
        )

    # ---------------------------------------------------------------------- #
    # Internal helpers
    # ---------------------------------------------------------------------- #

    def _reshape_to_3d(self, arr_2d):
        """
        Reshape a (N_sk, N_pix) array to (N_xy, N_xy, N_pix).
        Flat skewer index s maps to (s // N_xy, s % N_xy) (row-major).
        """
        return arr_2d.reshape(self.N_xy, self.N_xy, self.N_pix)

    @staticmethod
    def _p3d_to_xi3d(Pk3d):
        """
        Convert a P3D array (N_xy, N_xy, N_pix) to xi3D via inverse FFT.

        Derivation
        ----------
        From the Wiener–Khinchin theorem for a periodic discrete field:

            xi_ij(r) = IFFT{ P_ij(k) } / (d_perp^2 * dv)

        But P_ij(k) = d_perp^2 * dv * F_i(k) * conj(F_j(k)), so:

            xi_ij(r) = IFFT{ F_i(k) * conj(F_j(k)) }

        jnp.fft.ifftn carries the 1/N normalization internally, which
        reproduces the 1/(N_sk * N_pix) factor in Eq. (xi3D_sample_definition).

        The output is fftshift-ed so that the zero-lag voxel is at the
        centre of the array along all three axes.

        Returns
        -------
        xi3d : jnp.ndarray, shape (N_xy, N_xy, N_pix), real, centered.
        """
        # Recover F_i * conj(F_j) from P_ij by dividing the prefactor,
        # but since ifftn(P_ij / prefactor) = ifftn(F_i conj(F_j)) and
        # ifftn already handles 1/N, we note:
        #
        #   P_ij = prefactor * F_i conj(F_j)
        #   ifftn(P_ij) = prefactor * ifftn(F_i conj(F_j))
        #               = prefactor * xi_ij          [by Wiener-Khinchin]
        #
        # Therefore:
        #   xi_ij = ifftn(P_ij) / prefactor
        #         = ifftn(P_ij) / (d_perp^2 * dv)
        #
        # However, since we store P_ij already multiplied by prefactor,
        # the cleanest route is to keep this division explicit here.
        # The caller must pass d_perp and dv; we defer that to compute_xi3d.
        # Here we just do the IFFT — the caller divides by the prefactor.
        return jnp.fft.fftshift(ifftn(Pk3d).real)

    # ---------------------------------------------------------------------- #
    # compute_p3d
    # ---------------------------------------------------------------------- #

    def compute_p3d(self):
        """
        Compute the 3D power spectrum on the raw (k_x, k_y, k_par) grid.

        Normalization:  P_ij(k) = d_perp^2 * dv * F_i(k) * conj(F_j(k))
        Units: [cMpc^2 * km/s].

        Outputs stored on self
        ----------------------
        Pk3d_total : (N_xy, N_xy, N_pix), real
        Pk3d_pair  : dict {(a,b): (N_xy, N_xy, N_pix)}, real
        Pk3d_total_from_terms : (N_xy, N_xy, N_pix)
            sum_{(a,b)} Pk3d_pair[(a,b)] / mean(C_norm)
        Pk3d_total_from_terms_contributions : dict {(a,b): (N_xy, N_xy, N_pix)}

        Returns self (chainable).
        """
        if self.Pk3d_total is not None:
            return self

        prefactor = self.d_perp ** 2 * self.dv   # [cMpc^2 * km/s]

        f_total = fftn(self._reshape_to_3d(self.total))
        f_terms = {
            key: fftn(self._reshape_to_3d(self.delta_terms[key]))
            for key in self.subset_keys
        }

        self.Pk3d_total = (prefactor * f_total * jnp.conj(f_total)).real

        mean_C_norm = jnp.mean(self.C_norm)   # scalar: (1 + C_12)^2 estimator

        self.Pk3d_pair                           = {}
        self.Pk3d_total_from_terms_contributions = {}
        self.Pk3d_total_from_terms               = jnp.zeros_like(self.Pk3d_total)

        for pair in self.ordered_pairs:
            a, b = pair
            Pk3d_ab = (prefactor * f_terms[a] * jnp.conj(f_terms[b])).real
            self.Pk3d_pair[pair] = Pk3d_ab
            contribution = Pk3d_ab / mean_C_norm
            self.Pk3d_total_from_terms_contributions[pair] = contribution
            self.Pk3d_total_from_terms += contribution

        logger.info(
            "compute_p3d() done: prefactor=%.4e [cMpc^2 km/s], "
            "shape=(%d, %d, %d), n_pairs=%d.",
            float(prefactor), self.N_xy, self.N_xy, self.N_pix,
            len(self.ordered_pairs),
        )
        return self

    # ---------------------------------------------------------------------- #
    # compute_xi3d
    # ---------------------------------------------------------------------- #

    def compute_xi3d(self):
        """
        Compute the 3D two-point correlation function xi3D.

        Relation to P3D  [Eq. xi3D_sample_definition via Wiener-Khinchin]
        ------------------------------------------------------------------
        For a periodic box:

            xi_ij(r_x, r_y, pi) = IFFT_3D{ P_ij(k) } / (d_perp^2 * dv)

        where IFFT carries the 1/(N_xy^2 * N_pix) normalization, giving

            xi_ij = (1 / N_sk * N_pix)
                    sum_{x,y,v} delta_i(x,y,v) * delta_j(x+r_x, y+r_y, v+pi)

        exactly as in Eq. (xi3D_sample_definition).

        The output arrays are fftshift-ed so that the zero-lag voxel is at
        the centre of the array:
            axis 0: r_x  in [-L_perp/2, +L_perp/2)  [cMpc],  grid: r_perp_1d
            axis 1: r_y  in [-L_perp/2, +L_perp/2)  [cMpc],  grid: r_perp_1d
            axis 2: pi   in [-L_Lambda/2, +L_Lambda/2) [km/s], grid: pi_lags

        Decomposition  [Eq. xi3D_ensemble_L2_decomposition]
        -----------------------------------------------------
        Under the constant-C_12 approximation:

            xi_total(r_perp, pi) = [ sum_{(a,b)} xi_pair[(a,b)] - C_12^2 ]
                                   / (1 + C_12)^2

        implemented as:

            xi3d_total_from_terms = sum_{(a,b)} xi3d_pair[(a,b)] / mean_C_norm
                                    - correction

        where correction = mean(C_per_sample^2 / C_norm) accounts for the
        -C_12^2 / (1+C_12)^2 zero-lag term (same structure as xi1D).

        Outputs stored on self
        ----------------------
        xi3d_total : (N_xy, N_xy, N_pix), real, fftshift-centered
            Correlation function of the total overflux field.
            Lag axes: (r_x, r_y, pi).

        xi3d_pair : dict {(a,b): (N_xy, N_xy, N_pix)}, real, fftshift-centered
            Cross-correlation for every ordered pair (a, b).

        xi3d_total_from_terms : (N_xy, N_xy, N_pix), fftshift-centered
            Reconstruction of total xi3D from the subset decomposition.

        xi3d_total_from_terms_contributions : dict {(a,b): (N_xy, N_xy, N_pix)}
            Individual pair contributions to xi3d_total_from_terms.

        Returns self (chainable).
        """
        if self.xi3d_total is not None:
            return self

        self.compute_p3d()

        prefactor = self.d_perp ** 2 * self.dv   # [cMpc^2 * km/s]

        # ------------------------------------------------------------------ #
        # Total xi3D:  xi = IFFT(P) / prefactor
        # ------------------------------------------------------------------ #
        self.xi3d_total = (
            jnp.fft.fftshift(ifftn(self.Pk3d_total).real) / prefactor
        )   # (N_xy, N_xy, N_pix), dimensionless

        # ------------------------------------------------------------------ #
        # Component xi3D for all ordered pairs + decomposition
        # ------------------------------------------------------------------ #
        mean_C_norm = jnp.mean(self.C_norm)   # scalar

        self.xi3d_pair                           = {}
        self.xi3d_total_from_terms_contributions = {}
        self.xi3d_total_from_terms               = jnp.zeros_like(self.xi3d_total)

        for pair in self.ordered_pairs:
            xi3d_ab = (
                jnp.fft.fftshift(ifftn(self.Pk3d_pair[pair]).real) / prefactor
            )   # (N_xy, N_xy, N_pix)
            self.xi3d_pair[pair] = xi3d_ab

            contribution = xi3d_ab / mean_C_norm
            self.xi3d_total_from_terms_contributions[pair] = contribution
            self.xi3d_total_from_terms += contribution

        # Subtract the -C_12^2 / (1+C_12)^2 zero-separation term.
        # This correction is a constant offset identical in structure to the
        # 1D case in OverfluxStatistics1D.compute_xi1d().
        correction = jnp.mean(self.C_per_sample ** 2 / self.C_norm)
        self.xi3d_total_from_terms -= float(correction)

        logger.info(
            "compute_xi3d() done: output shape=(%d, %d, %d), correction=%.4e.",
            self.N_xy, self.N_xy, self.N_pix, float(correction),
        )
        return self

    # ---------------------------------------------------------------------- #
    # bin_p3d_kperp
    # ---------------------------------------------------------------------- #

    def bin_p3d_kperp(self, n_kperp_bins=6, Pk3d_input=None):
        """
        Bin P3D along k_perp, returning a 2D array (n_kperp_bins, N_pix).

        For each k_perp bin [k_lo, k_hi), averages P3D over all (k_x, k_y)
        modes with k_perp = sqrt(k_x^2 + k_y^2) in that range.
        The k_par axis is kept at full resolution.

        Parameters
        ----------
        n_kperp_bins : int
            Number of bins. Default 6.
        Pk3d_input : ndarray (N_xy, N_xy, N_pix) or None
            Array to bin.  None → self.Pk3d_total.

        Returns
        -------
        Pk_binned        : np.ndarray (n_kperp_bins, N_pix)
        k_perp_centers   : np.ndarray (n_kperp_bins,)  [cMpc^-1]
        k_perp_edges     : np.ndarray (n_kperp_bins+1,) [cMpc^-1]
        k_par            : np.ndarray (N_pix,)           [s/km]
        n_modes          : np.ndarray (n_kperp_bins,)
        """
        self.compute_p3d()

        Pk         = np.array(self.Pk3d_total if Pk3d_input is None else Pk3d_input)
        kp2d       = np.array(self.k_perp_3d[:, :, 0])   # (N_xy, N_xy)
        k_par_np   = np.array(self.k_par)

        k_perp_max   = np.pi / self.d_perp
        k_perp_edges = np.linspace(0.0, k_perp_max, n_kperp_bins + 1)

        Pk_binned      = np.full((n_kperp_bins, self.N_pix), np.nan)
        k_perp_centers = np.full(n_kperp_bins, np.nan)
        n_modes        = np.zeros(n_kperp_bins, dtype=int)

        for i in range(n_kperp_bins):
            lo, hi = k_perp_edges[i], k_perp_edges[i + 1]
            mask = (kp2d >= lo) & (kp2d < hi if i < n_kperp_bins - 1 else kp2d <= hi)
            n_modes[i] = int(mask.sum())
            if n_modes[i] == 0:
                continue
            Pk_binned[i]      = Pk[mask].mean(axis=0)
            k_perp_centers[i] = kp2d[mask].mean()

        return Pk_binned, k_perp_centers, k_perp_edges, k_par_np, n_modes

    # ---------------------------------------------------------------------- #
    # bin_xi3d_rperp
    # ---------------------------------------------------------------------- #

    def bin_xi3d_rperp(self, n_rperp_bins=None, xi3d_input=None):
        """
        Bin xi3D along r_perp = sqrt(r_x^2 + r_y^2), returning a 2D array
        (n_rperp_bins, N_pix) as a function of (r_perp_bin, pi).

        For each r_perp bin the method averages xi3D over all (r_x, r_y)
        displacement vectors whose magnitude falls within the bin.
        The pi (LoS lag) axis is kept at full resolution.

        Bin edges are linearly spaced from 0 to L_perp/2 (the maximum
        unambiguous transverse separation on a periodic grid).

        Parameters
        ----------
        n_rperp_bins : int or None
            Number of bins.  None → N_xy // 2 (one bin per unique grid radius).
        xi3d_input : ndarray (N_xy, N_xy, N_pix) or None
            Array to bin (must be fftshift-centered on all axes).
            None → self.xi3d_total.

        Returns
        -------
        xi_binned       : np.ndarray (n_rperp_bins, N_pix)
        r_perp_centers  : np.ndarray (n_rperp_bins,)  [cMpc]
        r_perp_edges    : np.ndarray (n_rperp_bins+1,) [cMpc]
        pi_lags         : np.ndarray (N_pix,)           [km/s]
        n_pairs         : np.ndarray (n_rperp_bins,)
        """
        self.compute_xi3d()

        xi       = np.array(self.xi3d_total if xi3d_input is None else xi3d_input)
        pi_np    = np.array(self.pi_lags)

        if n_rperp_bins is None:
            n_rperp_bins = self.N_xy // 2

        # Build the r_perp grid from the centered lag axes
        rp1d = np.array(self.r_perp_1d)                  # (N_xy,) centered
        rx   = rp1d[:, None]                               # (N_xy, 1)
        ry   = rp1d[None, :]                               # (1, N_xy)
        rp2d = np.sqrt(rx ** 2 + ry ** 2)                 # (N_xy, N_xy)

        # Bin edges: 0 to L_perp/2 (max unambiguous separation on periodic grid)
        r_perp_max   = self.L_perp / 2.0
        r_perp_edges = np.linspace(0.0, r_perp_max, n_rperp_bins + 1)

        xi_binned      = np.full((n_rperp_bins, self.N_pix), np.nan)
        r_perp_centers = np.full(n_rperp_bins, np.nan)
        n_pairs        = np.zeros(n_rperp_bins, dtype=int)

        for i in range(n_rperp_bins):
            lo, hi = r_perp_edges[i], r_perp_edges[i + 1]
            mask = (rp2d >= lo) & (rp2d < hi if i < n_rperp_bins - 1 else rp2d <= hi)
            n_pairs[i] = int(mask.sum())
            if n_pairs[i] == 0:
                continue
            xi_binned[i]      = xi[mask].mean(axis=0)
            r_perp_centers[i] = rp2d[mask].mean()

        return xi_binned, r_perp_centers, r_perp_edges, pi_np, n_pairs
