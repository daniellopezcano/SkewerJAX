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

    Memory policy
    -------------
    By default, this class does not keep the input optical depth array.
    It also releases the intermediate transmission fields after build().
    The retained heavy arrays are only the quantities normally needed
    downstream: delta_terms and total.
    """

    def __init__(
        self,
        optical_depth,
        labels=None,
        key_joiner="",
        dtype=float,
        *,
        keep_fields=False,
        keep_field_total=False,
    ):
        optical_depth = jnp.asarray(optical_depth, dtype=dtype)

        if optical_depth.ndim != 3:
            raise ValueError(
                "`optical_depth` must have shape "
                "(N_lines, N_samples, N_points). "
                f"Got {optical_depth.shape}."
            )

        self.N_lines = optical_depth.shape[0]
        self.N_samples = optical_depth.shape[1]
        self.N_points = optical_depth.shape[2]

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
        self.keep_fields = keep_fields
        self.keep_field_total = keep_field_total

        # --------------------------------------------------------------
        # Heavy temporary array: transmission fields.
        # We keep this only until build() finishes, unless keep_fields=True.
        # --------------------------------------------------------------
        fields = jnp.exp(-optical_depth)

        # Remove bad samples before storing anything heavy on self.
        bad_mask = jnp.any(jnp.isnan(optical_depth), axis=(0, 2))
        bad_mask = bad_mask | jnp.any(jnp.isnan(fields), axis=(0, 2))

        n_bad = int(jnp.sum(bad_mask))
        if n_bad > 0:
            n_tot = fields.shape[1]
            keep_mask = ~bad_mask

            logger.warning("Removing %d/%d samples: input NaNs", n_bad, n_tot)

            fields = fields[:, keep_mask, :]
            self.N_samples = fields.shape[1]

            if self.N_samples == 0:
                raise ValueError("All samples were removed during filtering.")

            logger.info("Remaining samples after filtering: %d", self.N_samples)

        # Store only the temporary transmission fields.
        # Do not store optical_depth.
        self.fields = fields
        self.optical_depth = None

        # Output containers.
        self.subset_keys = None
        self.composite_keys = None
        self.delta_terms = None
        self.C_per_sample = None
        self.field_total = None
        self.total = None

        self._built = False

        logger.info(
            "Initialized DeltaFieldBuilder with %d lines, N_samples=%d, N_points=%d.",
            self.N_lines,
            self.N_samples,
            self.N_points,
        )

    def _make_key(self, members):
        return self.key_joiner.join(members)

    def _filter_samples_from_fields(self, bad_mask, reason):
        n_bad = int(jnp.sum(bad_mask))
        if n_bad == 0:
            return

        n_tot = self.fields.shape[1]
        keep_mask = ~bad_mask

        logger.warning("Removing %d/%d samples: %s", n_bad, n_tot, reason)

        self.fields = self.fields[:, keep_mask, :]
        self.N_samples = self.fields.shape[1]

        if self.N_samples == 0:
            raise ValueError("All samples were removed during filtering.")

        logger.info("Remaining samples after filtering: %d", self.N_samples)

    def build(self):
        if self._built and self.fields is None:
            raise RuntimeError(
                "This DeltaFieldBuilder has already been built and its "
                "temporary fields were released. Create a new instance to rebuild."
            )

        logger.info("Starting build().")

        fields = self.fields

        # --------------------------------------------------------------
        # 1. Remove samples with zero mean transmission
        # --------------------------------------------------------------
        single_means = jnp.mean(fields, axis=2)  # (N_lines, N_samples)
        zero_mean_mask = jnp.any(single_means == 0.0, axis=0)

        field_total = jnp.prod(fields, axis=0)  # (N_samples, N_points)
        total_mean = jnp.mean(field_total, axis=1)
        zero_mean_mask = zero_mean_mask | (total_mean == 0.0)

        self._filter_samples_from_fields(
            zero_mean_mask,
            "zero mean in single field or total field",
        )

        fields = self.fields
        self.N_samples = fields.shape[1]

        # --------------------------------------------------------------
        # 2. Initialize output containers
        # --------------------------------------------------------------
        delta_terms = {}
        subset_keys = []
        composite_keys = []
        C_per_sample = jnp.zeros((self.N_samples, 1), dtype=fields.dtype)

        # --------------------------------------------------------------
        # 3. Single-line overflux fields
        # --------------------------------------------------------------
        for i_line, key in enumerate(self.labels):
            mean = jnp.mean(fields[i_line], axis=1, keepdims=True)
            term = fields[i_line] / mean - 1.0

            delta_terms[key] = term
            subset_keys.append(key)

            if int(jnp.sum(jnp.isnan(term))):
                logger.warning("NaNs detected in delta_terms[%r].", key)

        # --------------------------------------------------------------
        # 4. Total field and total overflux
        # --------------------------------------------------------------
        field_total = jnp.prod(fields, axis=0)
        field_total_mean = jnp.mean(field_total, axis=1, keepdims=True)
        total = field_total / field_total_mean - 1.0

        if int(jnp.sum(jnp.isnan(total))):
            logger.warning("NaNs detected in total.")

        # Only keep field_total if explicitly requested.
        self.field_total = field_total if self.keep_field_total else None
        self.total = total

        # --------------------------------------------------------------
        # 5. Composite overflux products
        # --------------------------------------------------------------
        for r in range(2, len(self.labels) + 1):
            for members in combinations(self.labels, r):
                key = self._make_key(members)

                term = delta_terms[members[0]]
                for member in members[1:]:
                    term = term * delta_terms[member]

                delta_terms[key] = term
                subset_keys.append(key)
                composite_keys.append(key)
                C_per_sample += jnp.mean(term, axis=1, keepdims=True)

                if int(jnp.sum(jnp.isnan(term))):
                    logger.warning("NaNs detected in delta_terms[%r].", key)

        if int(jnp.sum(jnp.isnan(C_per_sample))):
            logger.warning("NaNs detected in C_per_sample.")

        # --------------------------------------------------------------
        # 6. Store final outputs
        # --------------------------------------------------------------
        self.delta_terms = delta_terms
        self.subset_keys = subset_keys
        self.composite_keys = composite_keys
        self.C_per_sample = C_per_sample

        # Release heavy intermediate transmission fields unless requested.
        if not self.keep_fields:
            self.fields = None

        self.optical_depth = None
        self._built = True

        logger.info(
            "Finished build(): N_samples=%d, subset_keys=%d, composite_keys=%d.",
            self.N_samples,
            len(self.subset_keys),
            len(self.composite_keys),
        )

        return self