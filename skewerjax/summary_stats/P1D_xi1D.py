import logging
from itertools import product

import jax.numpy as jnp
from jax.numpy.fft import rfft, rfftfreq, irfft


logger = logging.getLogger(__name__)


class OverfluxStatistics1D:
    """
    Compute P1D and xi1D summary statistics from a built DeltaFieldBuilder object.
    """

    def __init__(self, overflux, dv):
        self.overflux = overflux
        self.dv = float(dv)

        for attr in ("subset_keys", "delta_terms", "C_per_sample", "total"):
            if not hasattr(overflux, attr):
                raise ValueError(f"`overflux` is missing required attribute '{attr}'.")

        Npix = overflux.total.shape[1]

        self.k = 2.0 * jnp.pi * rfftfreq(Npix, d=self.dv)
        self.lags = (jnp.arange(Npix) - Npix // 2) * self.dv
        self.ordered_pairs = list(product(overflux.subset_keys, repeat=2))

        self.Pk_total_per_sample = None
        self.Pk_total = None
        self.xi_total_per_sample = None
        self.xi_total = None
        self.Pk_pair_per_sample = None
        self.Pk_pair = None
        self.Pk_total_from_terms = None
        self.xi_pair_per_sample = None
        self.xi_pair = None
        self.xi_total_from_terms = None

        logger.info(
            "Initialized OverfluxStatistics1D with dv=%.6g, Npix=%d, n_subset_keys=%d, n_ordered_pairs=%d",
            self.dv,
            Npix,
            len(overflux.subset_keys),
            len(self.ordered_pairs),
        )

        self._log_nan_summary("overflux.total", overflux.total)
        self._log_nan_summary("overflux.C_per_sample", overflux.C_per_sample)
        for key in overflux.subset_keys:
            self._log_nan_summary(f"overflux.delta_terms[{key!r}]", overflux.delta_terms[key])

    @staticmethod
    def _count_nans(x):
        x = jnp.asarray(x)
        return int(jnp.sum(jnp.isnan(x)))

    def _log_nan_summary(self, name, x):
        x = jnp.asarray(x)
        n_nan = self._count_nans(x)
        if n_nan > 0:
            logger.warning(
                "NaNs detected in %s: count=%d, shape=%s",
                name,
                n_nan,
                x.shape,
            )

    def compute_p1d(self):
        if self.Pk_total is not None:
            logger.info("P1D already computed; returning cached results.")
            return self

        logger.info("Starting compute_p1d().")

        delta_total = jnp.asarray(self.overflux.total)
        delta_terms = {
            key: jnp.asarray(self.overflux.delta_terms[key])
            for key in self.overflux.subset_keys
        }
        C_per_sample = jnp.asarray(self.overflux.C_per_sample)

        Npix = delta_total.shape[1]
        prefactor = self.dv / Npix
        weight = 1.0 / (1.0 + C_per_sample) ** 2

        logger.info(
            "compute_p1d(): Npix=%d, prefactor=%.6g, weight_shape=%s",
            Npix,
            prefactor,
            weight.shape,
        )

        self._log_nan_summary("delta_total", delta_total)
        self._log_nan_summary("C_per_sample", C_per_sample)
        self._log_nan_summary("weight", weight)

        # -------------------- Total P1D -------------------- #
        f_total = rfft(delta_total, axis=1)
        self.Pk_total_per_sample = (prefactor * f_total * jnp.conjugate(f_total)).real
        self.Pk_total = jnp.nanmean(self.Pk_total_per_sample, axis=0)

        self._log_nan_summary("Pk_total_per_sample", self.Pk_total_per_sample)
        self._log_nan_summary("Pk_total", self.Pk_total)

        # -------------------- Component P1D -------------------- #
        f_terms = {key: rfft(delta_terms[key], axis=1) for key in self.overflux.subset_keys}

        self.Pk_pair_per_sample = {}
        self.Pk_pair = {}
        self.Pk_total_from_terms = jnp.zeros_like(self.Pk_total, dtype=self.Pk_total.dtype)

        for pair in self.ordered_pairs:
            a, b = pair
            Pk_ab_per_sample = (prefactor * f_terms[a] * jnp.conjugate(f_terms[b])).real

            self.Pk_pair_per_sample[pair] = Pk_ab_per_sample
            self.Pk_pair[pair] = jnp.nanmean(Pk_ab_per_sample, axis=0)
            self.Pk_total_from_terms = self.Pk_total_from_terms + jnp.nanmean(
                Pk_ab_per_sample * weight, axis=0
            )

            self._log_nan_summary(f"Pk_pair_per_sample[{pair}]", Pk_ab_per_sample)
            self._log_nan_summary(f"Pk_pair[{pair}]", self.Pk_pair[pair])

        self._log_nan_summary("Pk_total_from_terms", self.Pk_total_from_terms)

        logger.info("Finished compute_p1d().")
        return self

    def compute_xi1d(self):
        if self.xi_total is not None:
            logger.info("xi1D already computed; returning cached results.")
            return self

        logger.info("Starting compute_xi1d().")
        self.compute_p1d()

        Npix = self.overflux.total.shape[1]
        weight = 1.0 / (1.0 + jnp.asarray(self.overflux.C_per_sample)) ** 2

        logger.info(
            "compute_xi1d(): Npix=%d, weight_shape=%s",
            Npix,
            weight.shape,
        )

        self._log_nan_summary("weight (xi1d)", weight)

        # -------------------- Total xi1D -------------------- #
        self.xi_total_per_sample = (
            jnp.fft.fftshift(
                irfft(self.Pk_total_per_sample, n=Npix, axis=1).real,
                axes=1,
            ) / self.dv
        )
        self.xi_total = jnp.nanmean(self.xi_total_per_sample, axis=0)

        self._log_nan_summary("xi_total_per_sample", self.xi_total_per_sample)
        self._log_nan_summary("xi_total", self.xi_total)

        # -------------------- Component xi1D -------------------- #
        self.xi_pair_per_sample = {}
        self.xi_pair = {}
        self.xi_total_from_terms = jnp.zeros(Npix, dtype=self.xi_total.dtype)

        for pair in self.ordered_pairs:
            xi_ab_per_sample = irfft(self.Pk_pair_per_sample[pair], n=Npix, axis=1) / self.dv

            self.xi_pair_per_sample[pair] = xi_ab_per_sample
            self.xi_pair[pair] = jnp.nanmean(xi_ab_per_sample, axis=0).real
            self.xi_total_from_terms = self.xi_total_from_terms + jnp.fft.fftshift(
                jnp.nanmean(xi_ab_per_sample * weight, axis=0).real
            )

            self._log_nan_summary(f"xi_pair_per_sample[{pair}]", xi_ab_per_sample)
            self._log_nan_summary(f"xi_pair[{pair}]", self.xi_pair[pair])

        # This is equivalent to your last subtraction term, but simplified
        correction = jnp.nanmean(jnp.asarray(self.overflux.C_per_sample)[:, 0] ** 2)
        self._log_nan_summary("xi_total_from_terms correction", correction)

        self.xi_total_from_terms = self.xi_total_from_terms - correction
        self._log_nan_summary("xi_total_from_terms", self.xi_total_from_terms)

        logger.info("Finished compute_xi1d().")
        return self