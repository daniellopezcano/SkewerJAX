import logging
from itertools import combinations
import jax.numpy as jnp

logger = logging.getLogger(__name__)


class DeltaFieldBuilder:
    """
    Build single-line and composite overflux fields directly from an
    optical-depth array.

    Input
    -----
    optical_depth : array, shape (N_lines, N_samples, N_points)
        Optical depth per line, sample/skewer, and spectral pixel.

    Internally this class computes

        F_i = exp(-tau_i)
        delta_i = F_i / mean(F_i, axis=1) - 1

    and then all products of the delta_i fields.
    """

    def __init__(self, optical_depth, labels=None, key_joiner="", dtype=float):
        self.optical_depth = jnp.asarray(optical_depth, dtype=dtype)

        if self.optical_depth.ndim != 3:
            raise ValueError(
                "`optical_depth` must have shape "
                "(N_lines, N_samples, N_points). "
                f"Got {self.optical_depth.shape}."
            )

        self.N_lines = self.optical_depth.shape[0]
        self.N_samples = self.optical_depth.shape[1]
        self.N_points = self.optical_depth.shape[2]

        if labels is None:
            self.labels = [str(i + 1) for i in range(self.N_lines)]
        else:
            self.labels = list(labels)
            if len(self.labels) != self.N_lines:
                raise ValueError(
                    f"`labels` must have length N_lines={self.N_lines}. "
                    f"Got {len(self.labels)}."
                )

        self.key_joiner = key_joiner
        self.dtype = dtype

        self.fields = jnp.exp(-self.optical_depth)

        self._filter_bad_input_samples()

        self.subset_keys = None
        self.composite_keys = None
        self.delta_terms = None
        self.C_per_sample = None
        self.field_total = None
        self.total = None

        logger.info(
            "Initialized DeltaFieldBuilder with %d lines, N_samples=%d, N_points=%d.",
            self.N_lines,
            self.N_samples,
            self.N_points,
        )

    def _filter_samples(self, bad_mask, reason):
        n_bad = int(jnp.sum(bad_mask))
        if n_bad == 0:
            return

        n_tot = self.fields.shape[1]
        keep_mask = ~bad_mask

        logger.warning("Removing %d/%d samples: %s", n_bad, n_tot, reason)

        self.optical_depth = self.optical_depth[:, keep_mask, :]
        self.fields = self.fields[:, keep_mask, :]

        self.N_samples = self.fields.shape[1]

        if self.N_samples == 0:
            raise ValueError("All samples were removed during filtering.")

        logger.info("Remaining samples after filtering: %d", self.N_samples)

    def _filter_bad_input_samples(self):
        bad_mask = jnp.any(jnp.isnan(self.optical_depth), axis=(0, 2))
        bad_mask = bad_mask | jnp.any(jnp.isnan(self.fields), axis=(0, 2))
        self._filter_samples(bad_mask, "input NaNs")

    def _make_key(self, members):
        return self.key_joiner.join(members)

    def build(self):
        logger.info("Starting build().")

        # --------------------------------------------------------------
        # 1. Remove samples with zero mean transmission
        # --------------------------------------------------------------
        single_means = jnp.mean(self.fields, axis=2)  # (N_lines, N_samples)
        zero_mean_mask = jnp.any(single_means == 0.0, axis=0)

        stacked_fields = self.fields
        field_total = jnp.prod(stacked_fields, axis=0)  # (N_samples, N_points)
        total_mean = jnp.mean(field_total, axis=1)
        zero_mean_mask = zero_mean_mask | (total_mean == 0.0)

        self._filter_samples(zero_mean_mask, "zero mean in single field or total field")
        self.N_samples = self.fields.shape[1]

        # --------------------------------------------------------------
        # 2. Initialize output containers
        # --------------------------------------------------------------
        self.delta_terms = {}
        self.subset_keys = []
        self.composite_keys = []
        self.C_per_sample = jnp.zeros((self.N_samples, 1), dtype=self.fields.dtype)

        # --------------------------------------------------------------
        # 3. Single-line overflux fields
        # --------------------------------------------------------------
        for i_line, key in enumerate(self.labels):
            mean = jnp.mean(self.fields[i_line], axis=1, keepdims=True)
            term = self.fields[i_line] / mean - 1.0

            self.delta_terms[key] = term
            self.subset_keys.append(key)

            if int(jnp.sum(jnp.isnan(term))):
                logger.warning("NaNs detected in delta_terms[%r].", key)

        # --------------------------------------------------------------
        # 4. Total field and total overflux
        # --------------------------------------------------------------
        self.field_total = jnp.prod(self.fields, axis=0)

        field_total_mean = jnp.mean(self.field_total, axis=1, keepdims=True)
        self.total = self.field_total / field_total_mean - 1.0

        if int(jnp.sum(jnp.isnan(self.field_total))):
            logger.warning("NaNs detected in field_total.")

        if int(jnp.sum(jnp.isnan(self.total))):
            logger.warning("NaNs detected in total.")

        # --------------------------------------------------------------
        # 5. Composite overflux products
        # --------------------------------------------------------------
        for r in range(2, len(self.labels) + 1):
            for members in combinations(self.labels, r):
                key = self._make_key(members)

                term = self.delta_terms[members[0]]
                for member in members[1:]:
                    term = term * self.delta_terms[member]

                self.delta_terms[key] = term
                self.subset_keys.append(key)
                self.composite_keys.append(key)
                self.C_per_sample += jnp.mean(term, axis=1, keepdims=True)

                if int(jnp.sum(jnp.isnan(term))):
                    logger.warning("NaNs detected in delta_terms[%r].", key)

        if int(jnp.sum(jnp.isnan(self.C_per_sample))):
            logger.warning("NaNs detected in C_per_sample.")

        logger.info(
            "Finished build(): N_samples=%d, subset_keys=%d, composite_keys=%d.",
            self.N_samples,
            len(self.subset_keys),
            len(self.composite_keys),
        )

        return self