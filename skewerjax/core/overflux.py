"""Transmitted flux F = exp(-tau) and the overflux (flux-contrast) fields delta_F of single lines and line subsets.

Review notes, Sec. Overflux (Eqs. delta_j, F_total_product_lines_delta, overflux_product_expansion_general):
per sample (skewer) s, delta_j = F_j / <F_j>_s - 1, the subset products delta_A = prod_{j in A} delta_j, and
C_s = sum_{|A| >= 2} <delta_A>_s, so that the total overflux delta_tot = prod_j F_j / <prod_j F_j> - 1 decomposes
exactly into the subset terms (``core.stats.two_point``).
"""
from __future__ import annotations

import logging
from itertools import combinations
from typing import Sequence

import jax.numpy as jnp

logger = logging.getLogger(__name__)


def overflux_terms(fields: jnp.ndarray, labels: Sequence[str], key_joiner: str = "") -> dict:
    """Overflux terms of transmissions ``fields`` (N_lines, N_samples, N_pix) (pure; jit with static ``labels``).

    Returns dict(delta_terms {key: (N_samples, N_pix)} for every non-empty line subset (single lines first, then
    subsets of 2, 3, ... lines; keys join the labels), total (N_samples, N_pix), C_per_sample (N_samples, 1),
    field_total (N_samples, N_pix) = prod_j F_j).
    """
    delta = {}
    for i, key in enumerate(labels):
        delta[key] = fields[i] / jnp.mean(fields[i], axis=1, keepdims=True) - 1.0
    field_total = jnp.prod(fields, axis=0)
    total = field_total / jnp.mean(field_total, axis=1, keepdims=True) - 1.0
    C = jnp.zeros((fields.shape[1], 1), dtype=fields.dtype)
    for r in range(2, len(labels) + 1):
        for members in combinations(labels, r):
            term = delta[members[0]]
            for m in members[1:]:
                term = term * delta[m]
            delta[key_joiner.join(members)] = term
            C += jnp.mean(term, axis=1, keepdims=True)
    return dict(delta_terms=delta, total=total, C_per_sample=C, field_total=field_total)


class DeltaFieldBuilder:
    """Single-line and composite overflux fields of an optical-depth array (eager wrapper of ``overflux_terms``).

    Parameters
    ----------
    optical_depth : (N_lines, N_samples, N_pix).
    labels : one label per line (default "1", "2", ...); key_joiner joins them into subset keys.
    dtype : of the computation (``float``: float64 under x64).
    keep_fields, keep_field_total : keep F_j and prod_j F_j after ``build`` (released by default).

    Samples with NaN in tau or F, or with a zero mean transmission (of a line or of the total), are dropped
    (data-dependent shapes: this filtering is the eager part). After ``build()``: ``subset_keys``,
    ``composite_keys``, ``delta_terms``, ``total``, ``C_per_sample``, ``N_samples``.
    """

    def __init__(self, optical_depth, labels=None, key_joiner="", dtype=float, *, keep_fields=False, keep_field_total=False):
        optical_depth = jnp.asarray(optical_depth, dtype=dtype)
        if optical_depth.ndim != 3:
            raise ValueError(f"`optical_depth` must have shape (N_lines, N_samples, N_points). Got {optical_depth.shape}.")
        self.N_lines, self.N_samples, self.N_points = optical_depth.shape
        self.labels = [str(i + 1) for i in range(self.N_lines)] if labels is None else list(labels)
        if len(self.labels) != self.N_lines:
            raise ValueError(f"`labels` must have length N_lines={self.N_lines}. Got {len(self.labels)}.")
        self.key_joiner, self.dtype = key_joiner, dtype
        self.keep_fields, self.keep_field_total = keep_fields, keep_field_total
        self.fields = jnp.exp(-optical_depth)
        self._drop(jnp.any(jnp.isnan(optical_depth), axis=(0, 2)) | jnp.any(jnp.isnan(self.fields), axis=(0, 2)), "input NaNs")
        self.optical_depth = None
        self.subset_keys = self.composite_keys = self.delta_terms = self.C_per_sample = self.field_total = self.total = None
        self._built = False

    def _drop(self, bad, reason):
        n_bad = int(jnp.sum(bad))
        if n_bad == 0:
            return
        logger.warning("Removing %d/%d samples: %s", n_bad, self.fields.shape[1], reason)
        self.fields = self.fields[:, ~bad, :]
        self.N_samples = self.fields.shape[1]
        if self.N_samples == 0:
            raise ValueError("All samples were removed during filtering.")

    def build(self):
        if self._built and self.fields is None:
            raise RuntimeError("This DeltaFieldBuilder has already been built and its temporary fields were released. "
                               "Create a new instance to rebuild.")
        f = self.fields
        zero = jnp.any(jnp.mean(f, axis=2) == 0.0, axis=0) | (jnp.mean(jnp.prod(f, axis=0), axis=1) == 0.0)
        self._drop(zero, "zero mean in single field or total field")
        out = overflux_terms(self.fields, self.labels, self.key_joiner)
        for k, v in out["delta_terms"].items():
            if int(jnp.sum(jnp.isnan(v))):
                logger.warning("NaNs detected in delta_terms[%r].", k)
        self.delta_terms, self.total, self.C_per_sample = out["delta_terms"], out["total"], out["C_per_sample"]
        self.subset_keys = list(self.delta_terms)
        self.composite_keys = self.subset_keys[self.N_lines:]
        self.field_total = out["field_total"] if self.keep_field_total else None
        if not self.keep_fields:
            self.fields = None
        self._built = True
        return self


def equivalent_width_kms(tau, dv: float):
    """Velocity equivalent width sum(1 - e^-tau) dv [km/s] along the last axis."""
    return jnp.sum(-jnp.expm1(-jnp.asarray(tau)), axis=-1) * dv
