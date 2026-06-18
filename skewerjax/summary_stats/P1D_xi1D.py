import logging
from itertools import product

import numpy as jnp
from numpy.fft import rfft, rfftfreq, irfft


logger = logging.getLogger(__name__)


class OverfluxStatistics1D:
    """
    Compute P1D and xi1D summary statistics from a built DeltaFieldBuilder object.
    Invalid samples are removed at initialization.
    """

    def __init__(self, overflux, dv, c_norm_min=1e-12):
        self.overflux = overflux
        self.dv = float(dv)
        self.c_norm_min = float(c_norm_min)

        for attr in ("subset_keys", "delta_terms", "C_per_sample", "total"):
            if not hasattr(overflux, attr):
                raise ValueError(f"`overflux` is missing required attribute '{attr}'.")

        # Load arrays
        total = jnp.asarray(overflux.total)
        C_per_sample = jnp.asarray(overflux.C_per_sample)
        C_norm = (1.0 + C_per_sample) ** 2
        delta_terms = {key: jnp.asarray(overflux.delta_terms[key]) for key in overflux.subset_keys}

        # Build validity mask sample-by-sample
        valid_mask = jnp.all(jnp.isfinite(C_norm), axis=1) & jnp.all(jnp.abs(C_norm) > self.c_norm_min, axis=1)
        valid_mask &= jnp.all(jnp.isfinite(total), axis=1)

        for key in overflux.subset_keys:
            valid_mask &= jnp.all(jnp.isfinite(delta_terms[key]), axis=1)

        self.valid_mask = valid_mask
        self.n_samples_initial = total.shape[0]
        self.n_samples_kept = int(jnp.sum(valid_mask))
        self.n_samples_removed = self.n_samples_initial - self.n_samples_kept

        if self.n_samples_kept == 0:
            raise ValueError("All samples were removed by the initialization validity checks.")

        if self.n_samples_removed > 0:
            logger.warning(
                "Removed %d invalid samples out of %d during initialization.",
                self.n_samples_removed,
                self.n_samples_initial,
            )

        # Keep only valid samples
        self.total = total[valid_mask]
        self.C_per_sample = C_per_sample[valid_mask]
        self.C_norm = C_norm[valid_mask]
        self.delta_terms = {key: delta_terms[key][valid_mask] for key in overflux.subset_keys}

        self.subset_keys = overflux.subset_keys
        self.N_v = self.total.shape[1]
        self.k = 2.0 * jnp.pi * rfftfreq(self.N_v, d=self.dv) # 2 pi \frac{m}{N_v dv} with m \in \{0, ..., N_v/2\}
        self.lags = (jnp.arange(self.N_v) - self.N_v // 2) * self.dv
        self.ordered_pairs = list(product(self.subset_keys, repeat=2))

        self.Pk_total_per_sample = None
        self.Pk_total = None

        self.xi_total_per_sample = None
        self.xi_total = None

        self.Pk_pair_per_sample = None
        self.Pk_pair = None

        self.Pk_total_from_terms = None
        self.Pk_total_from_terms_contributions = None

        self.xi_pair_per_sample = None
        self.xi_pair = None
        
        self.xi_total_from_terms = None

    def compute_p1d(self):
        if self.Pk_total is not None:
            return self

        prefactor = self.dv / self.N_v

        # -------------------- Total P1D -------------------- #
        f_total = rfft(self.total, axis=1)
        self.Pk_total_per_sample = (prefactor * f_total * jnp.conjugate(f_total)).real
        self.Pk_total = jnp.mean(self.Pk_total_per_sample, axis=0)

        # -------------------- Component P1D -------------------- #
        f_terms = {key: rfft(self.delta_terms[key], axis=1) for key in self.subset_keys}

        self.Pk_pair_per_sample = {}
        self.Pk_pair = {}
        self.Pk_total_from_terms = jnp.zeros_like(self.Pk_total, dtype=self.Pk_total.dtype)
        self.Pk_total_from_terms_contributions = {}

        for pair in self.ordered_pairs:
            a, b = pair
            Pk_ab_per_sample = (prefactor * f_terms[a] * jnp.conjugate(f_terms[b])).real

            self.Pk_pair_per_sample[pair] = Pk_ab_per_sample
            self.Pk_pair[pair] = jnp.mean(Pk_ab_per_sample, axis=0)

            contribution = jnp.mean(Pk_ab_per_sample / self.C_norm, axis=0)
            self.Pk_total_from_terms_contributions[pair] = contribution
            self.Pk_total_from_terms += contribution

        return self

    def compute_xi1d(self):
        if self.xi_total is not None:
            return self

        self.compute_p1d()

        # -------------------- Total xi1D -------------------- #
        self.xi_total_per_sample = (
            jnp.fft.fftshift(
                irfft(self.Pk_total_per_sample, n=self.N_v, axis=1).real, axes=1
            ) / self.dv
        )
        self.xi_total = jnp.mean(self.xi_total_per_sample, axis=0)

        # -------------------- Component xi1D -------------------- #
        self.xi_pair_per_sample = {}
        self.xi_pair = {}
        self.xi_total_from_terms = jnp.zeros(self.N_v, dtype=self.xi_total.dtype)

        for pair in self.ordered_pairs:
            xi_ab_per_sample = irfft(self.Pk_pair_per_sample[pair], n=self.N_v, axis=1) / self.dv

            self.xi_pair_per_sample[pair] = xi_ab_per_sample
            self.xi_pair[pair] = jnp.mean(xi_ab_per_sample, axis=0).real
            self.xi_total_from_terms += jnp.fft.fftshift(
                jnp.mean(xi_ab_per_sample / self.C_norm, axis=0).real
            )

        correction = jnp.mean(jnp.asarray(self.C_per_sample)**2 / self.C_norm)
        self.xi_total_from_terms -= correction

        return self