"""P3D and xi3D of overflux fields on a regular grid of skewers (``OverfluxStatistics3D``)."""

from __future__ import annotations

import logging
from itertools import product

import jax.numpy as jnp
import numpy as np

from skewerjax.core.stats.two_point import (
    ZERO_ORDER_KEY,
    LazyPairs,
    bin_index,
    binned_mean,
    lag_axis,
    p3d_kernel,
    p3d_order_spectra,
    rfft_multiplicity,
    subset_orders,
    xi3d_from_power,
)

logger = logging.getLogger(__name__)


class OverfluxStatistics3D:
    r"""P3D and xi3D of a built ``DeltaFieldBuilder`` whose skewers lie on a regular grid.

    Geometry
    --------
    ``delta_terms[key]`` has shape (N_sk, N_pix) with N_sk = Nx * Ny skewers ordered
    row-major on the transverse grid (skewer s -> (s // Ny, s % Ny)); the field is
    reshaped to (Nx, Ny, N_pix). Transverse spacings dx, dy [cMpc]; LoS spacing dv [km/s].
    The axes keep their own units (the notes do not define a 3D unit convention yet):

        k_x, k_y = 2 pi fftfreq(N, d)    [1/cMpc];   k_par = 2 pi rfftfreq(N_pix, dv)  [s/km]
        r_x, r_y  [cMpc] and pi [km/s] lag axes, centred (fftshift).

    Normalisation (d-dimensional form of the review notes, Eq. P1D_sample_definition;
    dV = dx dy dv, N = Nx Ny N_pix, V = N dV)
    ---------------------------------------------------------------------------------------
        P_ab(k)  = (dV / N) F_a F_b^*,  F = rfftn(delta)          [cMpc^2 km/s]
        xi_ab(r) = (1/N) sum_x a(x) b(x + r) = irfftn(P_ab^*) / dV  (exact discrete pair)

    Spectra use the rfft layout (Nx, Ny, N_pix//2 + 1), k_par >= 0.

    Decomposition: as in ``OverfluxStatistics1D`` with per-skewer w = 1/(1+C_s) and
    c = C_s/(1+C_s); the weighted terms (incl. the order-0 term ``ZERO_ORDER_KEY`` = -c,
    which varies across skewers and so contributes on the k_par = 0 plane) sum exactly to
    the total P3D and xi3D.

    Attributes after ``compute_p3d`` / ``compute_xi3d``
    -------------------------------------------------
    Pk3d_total (Nx, Ny, K) real; Pk3d_pair lazy {(a,b): Re P_ab} raw;
    Pk3d_pair_complex lazy {(a,b): P_ab}; Pk3d_total_by_order {n: (Nx, Ny, K)};
    Pk3d_total_from_terms; Pk3d_total_from_terms_contributions lazy {(A,B)} weighted.
    xi3d_total, xi3d_total_from_terms (Nx, Ny, N_pix) centred; xi3d_pair lazy {(a,b)} raw,
    centred, unsymmetrised; xi3d_total_by_order {n}; xi3d_total_from_terms_contributions lazy.

    Parameters
    ----------
    overflux : DeltaFieldBuilder (built)
    dv : float
        LoS pixel width [km/s].
    d_perp : float or (float, float)
        Transverse skewer spacing (dx, dy) [cMpc].
    c_norm_min : float
        Invalid skewers are zeroed (not dropped) to keep the grid.
    grid_shape : (int, int) or None
        (Nx, Ny); None infers a square grid N_sk = N_xy^2.
    """

    def __init__(self, overflux, dv, d_perp, c_norm_min=1e-12, *, grid_shape=None):
        self.dv = float(dv)
        dxy = np.broadcast_to(np.asarray(d_perp, dtype=float), (2,))
        self.dx, self.dy = float(dxy[0]), float(dxy[1])
        self.d_perp = self.dx if self.dx == self.dy else (self.dx, self.dy)
        self.c_norm_min = float(c_norm_min)

        for attr in ("subset_keys", "delta_terms", "C_per_sample", "total"):
            if not hasattr(overflux, attr):
                raise ValueError(f"`overflux` is missing required attribute '{attr}'.")

        total = jnp.asarray(overflux.total)
        C_per_sample = jnp.asarray(overflux.C_per_sample).reshape(-1, 1)
        C_norm = (1.0 + C_per_sample) ** 2
        self.subset_keys = list(overflux.subset_keys)
        terms = jnp.stack([jnp.asarray(overflux.delta_terms[k]) for k in self.subset_keys])

        N_sk, N_pix = total.shape
        if grid_shape is None:
            N_xy = int(round(N_sk**0.5))
            if N_xy * N_xy != N_sk:
                raise ValueError(f"N_sk={N_sk} is not a perfect square; pass grid_shape=(Nx, Ny).")
            grid_shape = (N_xy, N_xy)
        Nx, Ny = (int(n) for n in grid_shape)
        if Nx * Ny != N_sk:
            raise ValueError(f"grid_shape={grid_shape} does not match N_sk={N_sk}.")

        self.N_sk, self.Nx, self.Ny, self.N_pix = N_sk, Nx, Ny, N_pix
        self.N_xy = Nx if Nx == Ny else None
        self.shape = (Nx, Ny, N_pix)
        self.L_x, self.L_y = Nx * self.dx, Ny * self.dy  # [cMpc]
        self.L_perp = self.L_x if self.N_xy is not None else (self.L_x, self.L_y)
        self.L_Lambda = N_pix * self.dv  # [km/s]
        self.A_perp = self.L_x * self.L_y  # [cMpc^2]
        self.dV = self.dx * self.dy * self.dv
        self.N_vox = Nx * Ny * N_pix

        valid_mask = jnp.all(jnp.isfinite(C_norm), axis=1) & jnp.all(jnp.abs(C_norm) > self.c_norm_min, axis=1)
        valid_mask &= jnp.all(jnp.isfinite(total), axis=1) & jnp.all(jnp.isfinite(terms), axis=(0, 2))
        self.valid_mask = valid_mask
        self.n_samples_initial = N_sk
        self.n_samples_kept = int(jnp.sum(valid_mask))
        self.n_samples_removed = N_sk - self.n_samples_kept
        if self.n_samples_kept == 0:
            raise ValueError("All samples were removed by the initialization validity checks.")
        if self.n_samples_removed > 0:
            logger.warning(
                "Removed %d invalid skewers out of %d. They are zeroed (not dropped) to preserve the grid shape.",
                self.n_samples_removed,
                N_sk,
            )
            total = jnp.where(valid_mask[:, None], total, 0.0)
            terms = jnp.where(valid_mask[None, :, None], terms, 0.0)
            C_per_sample = jnp.where(valid_mask[:, None], C_per_sample, 0.0)
            C_norm = (1.0 + C_per_sample) ** 2

        self.total = total
        self.C_per_sample = C_per_sample
        self.C_norm = C_norm
        self._terms = terms
        self.delta_terms = {k: terms[i] for i, k in enumerate(self.subset_keys)}
        C = C_per_sample[:, 0]
        self._w = jnp.where(valid_mask, 1.0 / (1.0 + C), 0.0).reshape(Nx, Ny)
        self._c = jnp.where(valid_mask, C / (1.0 + C), 0.0).reshape(Nx, Ny)

        self.ordered_pairs = list(product(self.subset_keys, repeat=2))
        orders = subset_orders(overflux, self.subset_keys)
        self.term_keys = [ZERO_ORDER_KEY] + self.subset_keys
        self.term_orders = [0] + [orders[k] for k in self.subset_keys]
        self.term_pairs = list(product(self.term_keys, repeat=2))
        self.pair_orders = {(a, b): oa + ob for (a, oa), (b, ob) in product(zip(self.term_keys, self.term_orders), repeat=2)}

        # wavenumber and lag axes
        self.kx = 2.0 * jnp.pi * jnp.fft.fftfreq(Nx, d=self.dx)  # [1/cMpc]
        self.ky = 2.0 * jnp.pi * jnp.fft.fftfreq(Ny, d=self.dy)
        self.k_perp_1d = self.kx
        self.k_par = 2.0 * jnp.pi * jnp.fft.rfftfreq(N_pix, d=self.dv)  # [s/km], >= 0
        self.k_perp_2d = jnp.sqrt(self.kx[:, None] ** 2 + self.ky[None, :] ** 2)
        self.r_x = jnp.asarray(lag_axis(Nx, self.dx))  # [cMpc], centred
        self.r_y = jnp.asarray(lag_axis(Ny, self.dy))
        self.r_perp_1d = self.r_x
        self.pi_lags = jnp.asarray(lag_axis(N_pix, self.dv))  # [km/s], centred
        self._mult_par = jnp.asarray(rfft_multiplicity(N_pix))

        for name in (
            "Pk3d_total",
            "Pk3d_pair",
            "Pk3d_pair_complex",
            "Pk3d_total_by_order",
            "Pk3d_total_from_terms",
            "Pk3d_total_from_terms_contributions",
            "xi3d_total",
            "xi3d_pair",
            "xi3d_total_by_order",
            "xi3d_total_from_terms",
            "xi3d_total_from_terms_contributions",
        ):
            setattr(self, name, None)

        logger.info(
            "OverfluxStatistics3D initialized: grid=(%d, %d, %d), dx=%.4g, dy=%.4g cMpc, dv=%.4g km/s.",
            Nx,
            Ny,
            N_pix,
            self.dx,
            self.dy,
            self.dv,
        )

    # ------------------------------------------------------------------ P3D
    def compute_p3d(self):
        """Compute P3D of the total field, per pair (lazy), per order and from terms."""
        if self.Pk3d_total is not None:
            return self
        pref = self.dV / self.N_vox
        f_tot, f_raw, f_w = p3d_kernel(self.total.reshape(self.shape), self._terms.reshape((-1,) + self.shape), self._w, self._c)
        self._f_raw, self._f_w = f_raw, f_w
        self.Pk3d_total = pref * jnp.abs(f_tot) ** 2

        n_orders = max(self.term_orders) + 1
        P_ord = p3d_order_spectra(f_w, jnp.asarray(self.term_orders), pref, n_orders)
        self._P_ord = P_ord
        self.Pk3d_total_by_order = {n: P_ord[n].real for n in range(P_ord.shape[0])}
        self.Pk3d_total_from_terms = jnp.sum(P_ord.real, axis=0)

        idx = {k: i for i, k in enumerate(self.subset_keys)}
        tidx = {k: i for i, k in enumerate(self.term_keys)}
        raw = lambda p: pref * f_raw[idx[p[0]]] * jnp.conj(f_raw[idx[p[1]]])
        wtd = lambda p: pref * f_w[tidx[p[0]]] * jnp.conj(f_w[tidx[p[1]]])
        self._raw_pair, self._w_pair = raw, wtd
        self.Pk3d_pair_complex = LazyPairs(self.ordered_pairs, raw)
        self.Pk3d_pair = LazyPairs(self.ordered_pairs, lambda p: raw(p).real)
        self.Pk3d_total_from_terms_contributions = LazyPairs(self.term_pairs, lambda p: wtd(p).real)
        logger.info("compute_p3d() done: pref=%.4e, %d subset terms, %d orders.", pref, len(self.subset_keys), P_ord.shape[0])
        return self

    # ------------------------------------------------------------------ xi3D
    def compute_xi3d(self):
        """xi3D = irfftn(P^*) / dV, centred on (r_x, r_y, pi); exact pair with the P3D above."""
        if self.xi3d_total is not None:
            return self
        self.compute_p3d()
        to_xi = lambda P: xi3d_from_power(P, self.shape, self.dV)
        self.xi3d_total = to_xi(self.Pk3d_total)
        xi_ord = to_xi(self._P_ord)
        self.xi3d_total_by_order = {n: xi_ord[n] for n in range(xi_ord.shape[0])}
        self.xi3d_total_from_terms = jnp.sum(xi_ord, axis=0)
        self.xi3d_pair = LazyPairs(self.ordered_pairs, lambda p: to_xi(self._raw_pair(p)))
        self.xi3d_total_from_terms_contributions = LazyPairs(self.term_pairs, lambda p: to_xi(self._w_pair(p)))
        return self

    # ------------------------------------------------------------------ binning
    def bin_p3d_kperp(self, n_kperp_bins=6, Pk3d_input=None, k_perp_edges=None):
        """Bin P3D in k_perp, keeping k_par at full resolution.

        Parameters
        ----------
        n_kperp_bins : int
            Linear bins on [0, min(pi/dx, pi/dy)] unless ``k_perp_edges`` is given.
        Pk3d_input : array (Nx, Ny, N_pix//2+1) or None
            Spectrum to bin (rfft layout); None -> ``Pk3d_total``.

        Returns
        -------
        Pk_binned (n_bins, N_kpar), k_perp_centers (n_bins,) [1/cMpc], k_perp_edges,
        k_par (N_kpar,) [s/km], n_modes (n_bins, N_kpar) full-grid modes per cell.
        """
        self.compute_p3d()
        Pk = self.Pk3d_total if Pk3d_input is None else jnp.asarray(Pk3d_input)
        if k_perp_edges is None:
            k_perp_edges = np.linspace(0.0, np.pi / max(self.dx, self.dy), n_kperp_bins + 1)
        k_perp_edges = np.asarray(k_perp_edges, dtype=float)
        nb = len(k_perp_edges) - 1
        kp = np.asarray(self.k_perp_2d).ravel()
        idx = jnp.asarray(bin_index(kp, k_perp_edges))
        vals = Pk.reshape(-1, Pk.shape[-1])
        P_b, cnt = binned_mean(vals, idx, jnp.ones_like(vals), nb)
        k_b, _ = binned_mean(jnp.asarray(kp), idx, jnp.ones(kp.shape), nb)
        return P_b, k_b, k_perp_edges, self.k_par, cnt * self._mult_par[None, :]

    def bin_p3d_kmod(self, k_edges=None, n_bins=20, Pk3d_input=None):
        """Spherically averaged P3D in |k| = sqrt(k_perp^2 + k_par^2).

        Only meaningful if the transverse and LoS axes use the same units (e.g. after
        converting dv to cMpc, or for test boxes). Default edges: log-spaced from the
        smallest fundamental mode to the smallest Nyquist frequency.

        Returns P_binned (n_bins,), k_centers (n_bins,), k_edges, n_modes (n_bins,).
        """
        self.compute_p3d()
        Pk = self.Pk3d_total if Pk3d_input is None else jnp.asarray(Pk3d_input)
        k_mod = np.asarray(jnp.sqrt(self.k_perp_2d[:, :, None] ** 2 + self.k_par[None, None, :] ** 2))
        if k_edges is None:
            k_lo = 2 * np.pi / max(self.L_x, self.L_y, self.L_Lambda)
            k_hi = np.pi / max(self.dx, self.dy, self.dv)
            k_edges = np.geomspace(0.999 * k_lo, k_hi, n_bins + 1)
        k_edges = np.asarray(k_edges, dtype=float)
        idx = bin_index(k_mod.ravel(), k_edges)
        idx = jnp.asarray(np.where(k_mod.ravel() > 0, idx, -1))
        w = jnp.broadcast_to(self._mult_par[None, None, :], k_mod.shape).ravel()
        P_b, cnt = binned_mean(Pk.ravel(), idx, w, len(k_edges) - 1)
        k_b, _ = binned_mean(jnp.asarray(k_mod.ravel()), idx, w, len(k_edges) - 1)
        return P_b, k_b, k_edges, cnt

    def bin_xi3d_rperp(self, n_rperp_bins=None, xi3d_input=None):
        """Bin xi3D in r_perp = sqrt(r_x^2 + r_y^2), keeping the pi axis.

        Edges are linear on [0, min(L_x, L_y)/2]; default n_rperp_bins = min(Nx, Ny)//2.

        Returns
        -------
        xi_binned (n_bins, N_pix), r_perp_centers (n_bins,) [cMpc], r_perp_edges,
        pi_lags (N_pix,) [km/s], n_pairs (n_bins,) lag cells per bin.
        """
        self.compute_xi3d()
        xi = self.xi3d_total if xi3d_input is None else jnp.asarray(xi3d_input)
        nb = min(self.Nx, self.Ny) // 2 if n_rperp_bins is None else int(n_rperp_bins)
        rp = np.sqrt(np.asarray(self.r_x)[:, None] ** 2 + np.asarray(self.r_y)[None, :] ** 2).ravel()
        r_edges = np.linspace(0.0, min(self.L_x, self.L_y) / 2.0, nb + 1)
        idx = jnp.asarray(bin_index(rp, r_edges))
        vals = xi.reshape(-1, xi.shape[-1])
        xi_b, cnt = binned_mean(vals, idx, jnp.ones_like(vals), nb)
        r_b, _ = binned_mean(jnp.asarray(rp), idx, jnp.ones(rp.shape), nb)
        return xi_b, r_b, r_edges, self.pi_lags, cnt[:, 0]
