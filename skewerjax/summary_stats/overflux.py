"""The transmission T = exp(-tau) and the overflux (flux-contrast) fields delta of single lines and line subsets.

Terminology: T = e^{-tau} is the **transmission**; the flux is the observed F = F_0 T (continuum times transmission).
With the continuum known, the overflux of DESI's continuum-fitting convention is

    delta = F / (F_0 T_bar) - 1 = T / T_bar - 1,    T_bar(v) = <T>_ens,

the ensemble mean at each velocity pixel v over every sample (skewer) of the ensemble, so <delta>_ens = 0 holds exactly
at every pixel (the mean along one skewer is not zero in general). Per line j, delta_j = T_j / T_bar_j - 1 with
T_bar_j(v) = <T_j>_ens; for the total, T = prod_j T_j and T_bar = <prod_j T_j>_ens != prod_j T_bar_j. The subset
products delta_A = prod_{j in A} delta_j and, at every pixel,

    C(v) = T_bar(v) / prod_j T_bar_j(v) - 1 = sum_{|A| >= 2} <delta_A>_ens(v)

give the exact decomposition (review notes, Sec. Overflux, Eqs. F_total_product_lines_delta,
overflux_product_expansion_general, with the per-sample mean replaced by the ensemble mean at each pixel)

    delta_tot = [sum_{A != {}} delta_A - C] / (1 + C)

(``summary_stats.orders``). The order-1 terms <delta_j>_ens of the denominator <prod_j (1 + delta_j)>_ens vanish
because each T_bar_j is the mean over the same ensemble. In observed spectra, continuum fitting also removes the mean and
the slope of each forest; a known continuum does not.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from itertools import combinations

import jax.numpy as jnp

logger = logging.getLogger(__name__)


def mean_transmissions(transmissions: jnp.ndarray) -> jnp.ndarray:
    """The ensemble mean at each pixel of every line and of the total: (N_lines + 1, N_pix), rows T_bar_j(v) = <T_j>
    over the samples and, last, T_bar(v) = <prod_j T_j> (``transmissions`` (N_lines, N_samples, N_pix)). The means of
    an ensemble split over several arrays (e.g. boxes) are the sample-weighted means of these."""
    return jnp.concatenate([jnp.mean(transmissions, axis=1), jnp.mean(jnp.prod(transmissions, axis=0), axis=0)[None]])


def overflux_terms(
    transmissions: jnp.ndarray, labels: Sequence[str], key_joiner: str = "", mean_transmission: jnp.ndarray | None = None
) -> dict:
    """Overflux terms of transmissions (N_lines, N_samples, N_pix) (pure; jit with static ``labels``), normalised by the
    ensemble mean at each pixel (module docstring).

    mean_transmission : (N_lines + 1, N_pix), optional
        T_bar_j(v) of every line and T_bar(v) of the total (``mean_transmissions``) when the ensemble is larger than
        these samples (e.g. every box of a run); default: the mean over these samples.

    Returns dict(delta_terms {key: (N_samples, N_pix)} for every non-empty line subset (single lines first, then subsets
    of 2, 3, ... lines; keys join the labels), total (N_samples, N_pix), C_per_pixel (1, N_pix) = T_bar / prod_j T_bar_j
    - 1, mean_transmission (N_lines + 1, N_pix), transmission_total (N_samples, N_pix) = prod_j T_j).
    """
    T_bar = mean_transmissions(transmissions) if mean_transmission is None else jnp.asarray(mean_transmission)
    delta = {}
    for i, key in enumerate(labels):
        delta[key] = transmissions[i] / T_bar[i][None] - 1.0
    transmission_total = jnp.prod(transmissions, axis=0)
    total = transmission_total / T_bar[-1][None] - 1.0
    C = (T_bar[-1] / jnp.prod(T_bar[:-1], axis=0) - 1.0)[None]
    for r in range(2, len(labels) + 1):
        for members in combinations(labels, r):
            term = delta[members[0]]
            for m in members[1:]:
                term = term * delta[m]
            delta[key_joiner.join(members)] = term
    return dict(delta_terms=delta, total=total, C_per_pixel=C, mean_transmission=T_bar, transmission_total=transmission_total)


class DeltaFieldBuilder:
    """Single-line and composite overflux fields of an optical-depth array (eager wrapper of ``overflux_terms``), each
    normalised by its ensemble mean transmission at every pixel (module docstring).

    Parameters
    ----------
    optical_depth : (N_lines, N_samples, N_pix).
    labels : one label per line (default "1", "2", ...); key_joiner joins them into subset keys.
    dtype : of the computation (``float``: float64 under x64).
    mean_transmission : (N_lines + 1, N_pix), optional
        The ensemble means T_bar_j(v) and T_bar(v) (``mean_transmissions``) when the ensemble holds more samples than
        ``optical_depth`` (e.g. every box of a run); default: the mean over the kept samples of ``optical_depth``. With
        few samples, the mean of the samples is a noisy estimate of the ensemble mean.
    keep_transmissions, keep_transmission_total : keep T_j and prod_j T_j after ``build`` (released by default).

    Samples with NaN in tau or T are dropped before the means are taken (data-dependent shapes: this filtering is the
    eager part); a zero mean transmission at a pixel (of a line or of the total) raises. After ``build()``:
    ``subset_keys``, ``composite_keys``, ``delta_terms``, ``total``, ``C_per_pixel``, ``mean_transmission``,
    ``N_samples``.
    """

    def __init__(
        self,
        optical_depth,
        labels=None,
        key_joiner="",
        dtype=float,
        *,
        mean_transmission=None,
        keep_transmissions=False,
        keep_transmission_total=False,
    ):
        optical_depth = jnp.asarray(optical_depth, dtype=dtype)
        if optical_depth.ndim != 3:
            raise ValueError(f"`optical_depth` must have shape (N_lines, N_samples, N_points). Got {optical_depth.shape}.")
        self.N_lines, self.N_samples, self.N_points = optical_depth.shape
        self.labels = [str(i + 1) for i in range(self.N_lines)] if labels is None else list(labels)
        if len(self.labels) != self.N_lines:
            raise ValueError(f"`labels` must have length N_lines={self.N_lines}. Got {len(self.labels)}.")
        self.key_joiner, self.dtype = key_joiner, dtype
        self.keep_transmissions, self.keep_transmission_total = keep_transmissions, keep_transmission_total
        self.mean_transmission = None if mean_transmission is None else jnp.asarray(mean_transmission, dtype=dtype)
        if self.mean_transmission is not None and self.mean_transmission.shape != (self.N_lines + 1, self.N_points):
            raise ValueError(
                f"`mean_transmission` must have shape (N_lines + 1, N_pix) = {(self.N_lines + 1, self.N_points)}. "
                f"Got {self.mean_transmission.shape}."
            )
        self.transmissions = jnp.exp(-optical_depth)
        bad = jnp.any(jnp.isnan(optical_depth), axis=(0, 2)) | jnp.any(jnp.isnan(self.transmissions), axis=(0, 2))
        self._drop(bad, "input NaNs")
        self.optical_depth = None
        self.subset_keys = self.composite_keys = self.delta_terms = self.C_per_pixel = self.transmission_total = None
        self.total = None
        self._built = False

    def _drop(self, bad, reason):
        n_bad = int(jnp.sum(bad))
        if n_bad == 0:
            return
        logger.warning("Removing %d/%d samples: %s", n_bad, self.transmissions.shape[1], reason)
        self.transmissions = self.transmissions[:, ~bad, :]
        self.N_samples = self.transmissions.shape[1]
        if self.N_samples == 0:
            raise ValueError("All samples were removed during filtering.")

    def build(self):
        if self._built and self.transmissions is None:
            raise RuntimeError(
                "This DeltaFieldBuilder has already been built and its temporary transmissions were released. "
                "Create a new instance to rebuild."
            )
        T_bar = mean_transmissions(self.transmissions) if self.mean_transmission is None else self.mean_transmission
        if bool(jnp.any(T_bar == 0.0)):
            raise ValueError("A mean transmission is zero at some pixel (of a line or of the total): delta is undefined there.")
        out = overflux_terms(self.transmissions, self.labels, self.key_joiner, T_bar)
        for k, v in out["delta_terms"].items():
            if int(jnp.sum(jnp.isnan(v))):
                logger.warning("NaNs detected in delta_terms[%r].", k)
        self.delta_terms, self.total, self.C_per_pixel = out["delta_terms"], out["total"], out["C_per_pixel"]
        self.mean_transmission = out["mean_transmission"]
        self.subset_keys = list(self.delta_terms)
        self.composite_keys = self.subset_keys[self.N_lines :]
        self.transmission_total = out["transmission_total"] if self.keep_transmission_total else None
        if not self.keep_transmissions:
            self.transmissions = None
        self._built = True
        return self


def equivalent_width_kms(tau, dv: float):
    """Velocity equivalent width sum(1 - e^-tau) dv [km/s] along the last axis."""
    return jnp.sum(-jnp.expm1(-jnp.asarray(tau)), axis=-1) * dv
