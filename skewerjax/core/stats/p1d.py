"""P1D and xi1D of overflux fields: total, per line pair and per order (``OverfluxStatistics1D``)."""
from __future__ import annotations

import logging
from itertools import product

import jax.numpy as jnp
import numpy as np

from skewerjax.core.stats.two_point import (
    ZERO_ORDER_KEY, bin_index, binned_mean, lag_axis, order_sum, p1d_kernel,
    rfft_multiplicity, subset_orders, xi_from_power,
)

logger = logging.getLogger(__name__)


class OverfluxStatistics1D:
    r"""P1D and xi1D of a built ``DeltaFieldBuilder``: total, per line pair, and per order.

    Conventions (notes ``subsec_P1D.tex``, Eqs. P1D_delta_tilde_sample, P1D_sample_definition,
    xi1D_from_P1D_ensemble; periodic skewers of N_v pixels of width dv [km/s])
    -------------------------------------------------------------------------------------
        k = 2 pi rfftfreq(N_v, dv)                     [s/km], k >= 0
        P_ab(k)   = (dv / N_v) F_a(k) F_b(k)^*,  F = rfft(delta)          [km/s]
        xi_ab(Dv) = (1/N_v) sum_j a(v_j) b(v_j + Dv) = irfft(P_ab^*) / dv  [dimensionless]

    P and xi are an exact discrete Fourier pair; for a != b, P_ab is complex and xi_ab is
    not symmetric (xi_ba(Dv) = xi_ab(-Dv)); ``Pk_pair`` stores Re P_ab = (P_ab + P_ba)/2.
    Every statistic is the mean over the kept samples (skewers).

    Decomposition (notes ``sec_flux.tex`` Eq. total_overflux_expansion_general;
    ``subsec_xi1D.tex`` Eq. xi1D_ensemble_L2_decomposition)
    ---------------------------------------------------------------------------
    Per sample, delta_tot = sum_A w delta_A - c with w = 1/(1+C_s), c = C_s/(1+C_s) (exact
    for ``DeltaFieldBuilder``). The weighted terms (A, B), including the order-0 term
    ``ZERO_ORDER_KEY`` = -c, have order |A|+|B| and sum exactly to the total:

        Pk_total  = sum_n Pk_total_by_order[n] = Pk_total_from_terms   (all k, incl. k = 0)
        xi_total  = sum_n xi_total_by_order[n] = xi_total_from_terms   (all lags)

    Attributes after ``compute_p1d`` / ``compute_xi1d``
    -------------------------------------------------
    k (N_k,), lags (N_v,) [km/s, centred], n_modes (N_k,)
    Pk_total, Pk_total_err (N_k,), Pk_total_per_sample (N_s, N_k)
    Pk_pair {(a,b): (N_k,)} raw Re P_ab;  Pk_pair_complex {(a,b): (N_k,)} complex
    Pk_total_from_terms (N_k,), Pk_total_by_order {n: (N_k,)},
    Pk_total_from_terms_contributions {(A,B): (N_k,)} weighted, incl. ZERO_ORDER_KEY pairs
    xi_total, xi_total_from_terms (N_v,) centred; xi_total_per_sample (N_s, N_v) centred
    xi_pair {(a,b): (N_v,)} raw FFT lag order (lag j dv, j = 0..N_v-1), as before
    xi_total_by_order {n: (N_v,)}, xi_total_from_terms_contributions {(A,B): (N_v,)} centred

    Error bars: ``n_modes[k]`` counts full-grid modes (+k and -k separately) over all
    samples, so a Gaussian field gives sigma[P(k)] = P(k) sqrt(2 / n_modes[k]).

    Parameters
    ----------
    overflux : DeltaFieldBuilder (built)
        Must expose subset_keys, delta_terms, C_per_sample, total.
    dv : float
        Pixel width [km/s].
    c_norm_min : float
        Samples with |(1 + C_s)^2| <= c_norm_min or non-finite values are removed.
    chunk_size : int
        Skewers per FFT batch (bounds memory to ~n_terms * chunk_size * N_v).
    """

    def __init__(self, overflux, dv, c_norm_min=1e-12, *, chunk_size=4096):
        self.overflux = overflux
        self.dv = float(dv)
        self.c_norm_min = float(c_norm_min)
        self.chunk_size = int(chunk_size)

        for attr in ("subset_keys", "delta_terms", "C_per_sample", "total"):
            if not hasattr(overflux, attr):
                raise ValueError(f"`overflux` is missing required attribute '{attr}'.")

        total = jnp.asarray(overflux.total)
        C_per_sample = jnp.asarray(overflux.C_per_sample).reshape(-1, 1)
        C_norm = (1.0 + C_per_sample) ** 2
        self.subset_keys = list(overflux.subset_keys)
        terms = jnp.stack([jnp.asarray(overflux.delta_terms[k]) for k in self.subset_keys])

        valid_mask = jnp.all(jnp.isfinite(C_norm), axis=1) & jnp.all(jnp.abs(C_norm) > self.c_norm_min, axis=1)
        valid_mask &= jnp.all(jnp.isfinite(total), axis=1) & jnp.all(jnp.isfinite(terms), axis=(0, 2))

        self.valid_mask = valid_mask
        self.n_samples_initial = total.shape[0]
        self.n_samples_kept = int(jnp.sum(valid_mask))
        self.n_samples_removed = self.n_samples_initial - self.n_samples_kept
        if self.n_samples_kept == 0:
            raise ValueError("All samples were removed by the initialization validity checks.")
        if self.n_samples_removed > 0:
            logger.warning("Removed %d invalid samples out of %d during initialization.",
                           self.n_samples_removed, self.n_samples_initial)
            keep = np.asarray(valid_mask)
            total, C_per_sample, C_norm, terms = total[keep], C_per_sample[keep], C_norm[keep], terms[:, keep]

        self.total = total
        self.C_per_sample = C_per_sample
        self.C_norm = C_norm
        self._terms = terms
        self.delta_terms = {k: terms[i] for i, k in enumerate(self.subset_keys)}

        self.N_v = total.shape[1]
        self.k = 2.0 * jnp.pi * jnp.fft.rfftfreq(self.N_v, d=self.dv)      # [s/km]
        self.lags = jnp.asarray(lag_axis(self.N_v, self.dv))               # [km/s], centred
        self.n_modes = jnp.asarray(rfft_multiplicity(self.N_v) * self.n_samples_kept)
        self.ordered_pairs = list(product(self.subset_keys, repeat=2))

        orders = subset_orders(overflux, self.subset_keys)
        self.term_keys = [ZERO_ORDER_KEY] + self.subset_keys
        self.term_orders = [0] + [orders[k] for k in self.subset_keys]
        self.term_pairs = list(product(self.term_keys, repeat=2))
        self.pair_orders = {(a, b): oa + ob for (a, oa), (b, ob)
                            in product(zip(self.term_keys, self.term_orders), repeat=2)}

        for name in ("Pk_total_per_sample", "Pk_total", "Pk_total_err", "Pk_pair", "Pk_pair_complex",
                     "Pk_total_from_terms", "Pk_total_by_order", "Pk_total_from_terms_contributions",
                     "xi_total_per_sample", "xi_total", "xi_pair", "xi_total_from_terms",
                     "xi_total_by_order", "xi_total_from_terms_contributions"):
            setattr(self, name, None)
        self._P_raw = self._P_w = None

    # ------------------------------------------------------------------ P1D
    def compute_p1d(self):
        """Compute all P1D statistics (one jitted, chunked pass over the skewers)."""
        if self.Pk_total is not None:
            return self
        C = self.C_per_sample[:, 0]
        w, c = 1.0 / (1.0 + C), C / (1.0 + C)
        P_raw, P_w, P_tot = p1d_kernel(self.total, self._terms, w, c, self.dv / self.N_v,
                                       min(self.chunk_size, self.n_samples_kept))
        self._P_raw, self._P_w = P_raw, P_w

        self.Pk_total_per_sample = P_tot
        self.Pk_total = jnp.mean(P_tot, axis=0)
        self.Pk_total_err = jnp.std(P_tot, axis=0, ddof=1) / np.sqrt(P_tot.shape[0]) \
            if P_tot.shape[0] > 1 else jnp.full_like(self.Pk_total, jnp.nan)

        idx = {k: i for i, k in enumerate(self.subset_keys)}
        self.Pk_pair_complex = {(a, b): P_raw[idx[a], idx[b]] for a, b in self.ordered_pairs}
        self.Pk_pair = {p: v.real for p, v in self.Pk_pair_complex.items()}

        tidx = {k: i for i, k in enumerate(self.term_keys)}
        self.Pk_total_from_terms_contributions = {(a, b): P_w[tidx[a], tidx[b]].real
                                                  for a, b in self.term_pairs}
        by_order = order_sum(P_w, self.term_orders).real
        self.Pk_total_by_order = {n: by_order[n] for n in range(by_order.shape[0])}
        self.Pk_total_from_terms = jnp.sum(by_order, axis=0)
        return self

    # ------------------------------------------------------------------ xi1D
    def compute_xi1d(self):
        """xi1D by Wiener-Khinchin: xi = irfft(P^*) / dv (exact pair with the P1D above)."""
        if self.xi_total is not None:
            return self
        self.compute_p1d()
        shift = lambda x: jnp.fft.fftshift(x, axes=-1)
        N, dv = self.N_v, self.dv

        self.xi_total_per_sample = shift(xi_from_power(self.Pk_total_per_sample, N, dv))
        self.xi_total = jnp.mean(self.xi_total_per_sample, axis=0)

        xi_raw = xi_from_power(self._P_raw, N, dv)                          # (n, n, N_v)
        idx = {k: i for i, k in enumerate(self.subset_keys)}
        self.xi_pair = {(a, b): xi_raw[idx[a], idx[b]] for a, b in self.ordered_pairs}

        xi_w = shift(xi_from_power(self._P_w, N, dv))
        tidx = {k: i for i, k in enumerate(self.term_keys)}
        self.xi_total_from_terms_contributions = {(a, b): xi_w[tidx[a], tidx[b]]
                                                  for a, b in self.term_pairs}
        by_order = order_sum(xi_w, self.term_orders)
        self.xi_total_by_order = {n: by_order[n] for n in range(by_order.shape[0])}
        self.xi_total_from_terms = jnp.sum(by_order, axis=0)
        return self

    # ------------------------------------------------------------------ binning
    def bin_p1d(self, k_edges, Pk=None):
        """Bin a P1D (default ``Pk_total``) in k with fixed edges [s/km].

        Returns
        -------
        Pk_binned (n_bins,), k_centers (n_bins,) mode-weighted, k_edges, n_modes (n_bins,)
        (full-grid modes summed over samples; sigma = P sqrt(2 / n_modes) for a Gaussian field).
        """
        self.compute_p1d()
        Pk = self.Pk_total if Pk is None else jnp.asarray(Pk)
        k_edges = np.asarray(k_edges, dtype=float)
        idx = jnp.asarray(bin_index(np.asarray(self.k), k_edges))
        w = self.n_modes
        P_b, n_b = binned_mean(Pk, idx, w, len(k_edges) - 1)
        k_b, _ = binned_mean(self.k, idx, w, len(k_edges) - 1)
        return P_b, k_b, k_edges, n_b


def p1d_from_tau(tau, dv: float):
    """P1D of delta_F = F / <F> - 1 of one line over all skewers, without the k = 0 mode.

    tau : (N_sk, N_v); dv : pixel width [km/s]. Returns (k [s/km], P [km/s]) as NumPy arrays.
    """
    from skewerjax.core.overflux import DeltaFieldBuilder
    st = OverfluxStatistics1D(DeltaFieldBuilder(np.asarray(tau)[None]).build(), dv).compute_p1d()
    return np.asarray(st.k)[1:], np.asarray(st.Pk_total)[1:]
