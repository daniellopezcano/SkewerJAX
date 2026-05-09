import jax.numpy as jnp
from itertools import product
from jax.numpy.fft import rfft, rfftfreq, irfft


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

    def compute_p1d(self):
        if self.Pk_total is not None:
            return self

        delta_total = jnp.asarray(self.overflux.total)
        delta_terms = {
            key: jnp.asarray(self.overflux.delta_terms[key])
            for key in self.overflux.subset_keys
        }
        C_per_sample = jnp.asarray(self.overflux.C_per_sample)

        Npix = delta_total.shape[1]
        prefactor = self.dv / Npix
        weight = 1.0 / (1.0 + C_per_sample) ** 2

        # -------------------- Total P1D -------------------- #
        f_total = rfft(delta_total, axis=1)
        self.Pk_total_per_sample = prefactor * f_total * jnp.conjugate(f_total)
        self.Pk_total = jnp.mean(self.Pk_total_per_sample, axis=0)

        # -------------------- Component P1D -------------------- #
        f_terms = {key: rfft(delta_terms[key], axis=1) for key in self.overflux.subset_keys}

        self.Pk_pair_per_sample = {}
        self.Pk_pair = {}
        self.Pk_total_from_terms = jnp.zeros_like(self.Pk_total, dtype=self.Pk_total.dtype)

        for pair in self.ordered_pairs:
            a, b = pair
            Pk_ab_per_sample = prefactor * f_terms[a] * jnp.conjugate(f_terms[b])

            self.Pk_pair_per_sample[pair] = Pk_ab_per_sample
            self.Pk_pair[pair] = jnp.mean(Pk_ab_per_sample, axis=0)
            self.Pk_total_from_terms = self.Pk_total_from_terms + jnp.mean(
                Pk_ab_per_sample * weight, axis=0
            )

        return self

    def compute_xi1d(self):
        if self.xi_total is not None:
            return self

        self.compute_p1d()

        Npix = self.overflux.total.shape[1]
        weight = 1.0 / (1.0 + jnp.asarray(self.overflux.C_per_sample)) ** 2

        # -------------------- Total xi1D -------------------- #
        self.xi_total_per_sample = (
            jnp.fft.fftshift(
                irfft(self.Pk_total_per_sample, n=Npix, axis=1).real,
                axes=1,
            ) / self.dv
        )
        self.xi_total = jnp.mean(self.xi_total_per_sample, axis=0)

        # -------------------- Component xi1D -------------------- #
        self.xi_pair_per_sample = {}
        self.xi_pair = {}
        self.xi_total_from_terms = jnp.zeros(Npix, dtype=self.xi_total.dtype)

        for pair in self.ordered_pairs:
            xi_ab_per_sample = irfft(self.Pk_pair_per_sample[pair], n=Npix, axis=1) / self.dv

            self.xi_pair_per_sample[pair] = xi_ab_per_sample
            self.xi_pair[pair] = jnp.mean(xi_ab_per_sample, axis=0).real
            self.xi_total_from_terms = self.xi_total_from_terms + jnp.fft.fftshift(
                jnp.mean(xi_ab_per_sample * weight, axis=0).real
            )

        # This is equivalent to your last subtraction term, but simplified
        self.xi_total_from_terms = self.xi_total_from_terms - jnp.mean(
            jnp.asarray(self.overflux.C_per_sample)[:, 0] ** 2
        )

        return self