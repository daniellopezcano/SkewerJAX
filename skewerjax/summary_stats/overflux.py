"""The transmission T = exp(-tau) and the overflux (flux-contrast) fields delta of single lines and line subsets.

Terminology: T = e^{-tau} is the **transmission**; the flux is the observed F = F_0 T (continuum times transmission).
Here the continuum is known (in effect F / F_0 = T), so the overflux of DESI's continuum-fitting convention is exactly

    delta = F / (F_0 T_bar) - 1 = T / T_bar - 1,

with T_bar the mean transmission (``mean``, ``MEAN_MODES``), per line and for the total, over one of four sets of
pixels (each mode gives <delta> = 0 over its own set, and removes what is constant over it):

| ``mean`` | T_bar: the mean of T over | removes | use it for |
|---|---|---|---|
| "redshift" (default) | every pixel at the same redshift, of every sample: T_bar(z) | the mean-flux evolution along the path (the shared pattern s(v) of an exact frame) | the observational convention, mean flux as a function of z |
| "global" | every pixel of every sample: one number | only the overall mean; a drift of T_bar along v stays in delta | one redshift with no evolution along the path (a snapshot frame) |
| "box" | every pixel of every sample of one realisation (box): one number per box | the box-to-box scatter of the mean (k = 0 of each box) | per-realisation normalisation, as for independent simulations |
| "skewer" | the pixels of each sample: one number per skewer | the k_par = 0 mode of every skewer, as continuum fitting does | mimicking a fitted continuum amplitude |

"redshift" groups the pixels by their scale factor: samples on a shared grid (every row of ``grid`` the same, as on
the toy lattice, where all skewers share one v -> a grid) give T_bar(v) at each pixel; pixels of the grid at the same
scale factor (to ``GRID_RTOL``) share one mean (``redshift_groups``: in one snapshot every pixel has the same a, so
T_bar(z) is the global mean); samples whose grids differ raise a ``ValueError`` (``check_shared_grid``). Without
``grid`` every pixel is its own redshift. On one array of samples "global" and "box" are the same mean; an ensemble
split over realisations (``ensemble_mean_transmissions``, notebook 10) distinguishes them, and ``overflux_check`` checks
the normalisation over the sets of the mode.

Per line j, delta_j = T_j / T_bar_j - 1; for the total, T = prod_j T_j and T_bar = <prod_j T_j> != prod_j T_bar_j. The
subset products delta_A = prod_{j in A} delta_j and

    C = T_bar / prod_j T_bar_j - 1 = sum_{|A| >= 2} <delta_A>      (over the set of pixels of the mode: one number
                                                                   per pixel, per ensemble, per box or per skewer)

give the exact decomposition, at every pixel of every sample (review notes, Sec. Overflux, Eqs.
F_total_product_lines_delta, overflux_product_expansion_general, with the mean of each sample replaced by the mean of
the set of the mode)

    delta_tot = [sum_{A != {}} delta_A - C] / (1 + C)

(``summary_stats.orders``). It follows from T = prod_j T_bar_j (1 + delta_j) and its mean, T_bar = prod_j T_bar_j
(1 + sum_j <delta_j> + C): the order-1 terms <delta_j> vanish because each T_bar_j is the mean over the same set of
pixels. In observed spectra the continuum F_0 of each quasar must be estimated: a change of its
amplitude or slope is indistinguishable from a large-scale fluctuation of the transmission along that sightline, so
even a correct continuum model absorbs those modes of delta (analyses correct for it with a projection); with a known
continuum this does not arise.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from itertools import combinations

import jax.numpy as jnp
import numpy as np

logger = logging.getLogger(__name__)

MEAN_MODES = ("redshift", "global", "box", "skewer")  # the mean transmission of the overflux (module docstring)
GRID_RTOL = 1e-12  # "redshift": the relative tolerance of a shared pixel grid (rounding only)


def _check_mean(mean: str) -> str:
    if mean not in MEAN_MODES:
        raise ValueError(f"mean must be one of {MEAN_MODES}, got {mean!r}.")
    return mean


def check_shared_grid(grid, n_pix: int | None = None, rtol: float = GRID_RTOL, what: str = "samples") -> np.ndarray:
    """The one pixel grid shared by every row of ``grid`` (the scale factor, or the redshift, of every pixel of every
    sample: (N_pix,) or (N_rows, N_pix)), for ``mean="redshift"``: T_bar(z) is the mean over the pixels at the same z,
    which is the mean at each pixel only when every sample has the same grid. Raises ``ValueError`` if the rows differ
    by more than ``rtol`` (relative), or if their length is not ``n_pix``. Returns the grid (N_pix,)."""
    g = np.atleast_2d(np.asarray(grid, dtype=np.float64))
    if g.ndim != 2 or (n_pix is not None and g.shape[1] != int(n_pix)):
        raise ValueError(f"the pixel grid of the {what} must be (N_pix,) or (N_rows, N_pix) with N_pix = {n_pix}; got {g.shape}.")
    dev = float(np.max(np.abs(g - g[:1]) / np.maximum(np.abs(g[:1]), 1e-300)))
    if not np.isfinite(dev) or dev > rtol:
        raise ValueError(
            f'mean="redshift" needs one pixel grid (scale factor or redshift) shared by all the {what}: their grids differ '
            f"by up to {dev:.2e} (relative; tolerance {rtol:.0e}). Put the samples on a common grid first, or use "
            f'mean="global", "box" or "skewer".'
        )
    return g[0]


def redshift_groups(grid, n_pix: int | None = None, rtol: float = GRID_RTOL, what: str = "samples"):
    """The pixels at the same scale factor of a shared grid (``check_shared_grid``), for ``mean="redshift"``: the label
    (N_pix,) of the group of every pixel (groups numbered by increasing grid value; equal values to ``rtol``, relative),
    or None when every pixel has its own scale factor (then T_bar(z) is the mean at each pixel)."""
    g = check_shared_grid(grid, n_pix, rtol, what)
    order = np.argsort(g, kind="stable")
    gs = g[order]
    new = np.concatenate([[True], np.abs(np.diff(gs)) > rtol * np.maximum(np.abs(gs[1:]), 1e-300)])
    if new.all():
        return None
    labels = np.empty(g.size, dtype=np.int64)
    labels[order] = np.cumsum(new) - 1
    return labels


def _group_mean(m, labels):
    """The mean of ``m`` (..., N_pix) over each group of pixels (``redshift_groups``), at every pixel of the group
    (``labels`` None: ``m`` itself)."""
    if labels is None:
        return m
    onehot = np.eye(int(labels.max()) + 1)[labels]  # (N_pix, n_groups)
    return (m @ (onehot / onehot.sum(0)))[..., labels]


def mean_transmissions(transmissions: jnp.ndarray, mean: str = "redshift", *, grid=None) -> jnp.ndarray:
    """The mean transmission of every line and of the total, rows T_bar_j = <T_j> and, last, T_bar = <prod_j T_j>, of
    ``transmissions`` (N_lines, N_samples, N_pix), over the set of pixels of ``mean`` (module docstring):

    * "redshift" (default): over the samples at each pixel, T_bar(z): (N_lines + 1, N_pix). ``grid`` (N_pix,) or
      (N_samples, N_pix), optional: the scale factor (or redshift) of every pixel, checked to be shared by all the
      samples (``check_shared_grid``; without it the samples are assumed to share one grid); pixels at the same scale
      factor share their mean (``redshift_groups``);
    * "global", "box": over every pixel of every sample (the same here: one array is one box), the same value in every
      column: (N_lines + 1, N_pix);
    * "skewer": over the pixels of each sample, the same value along it: (N_lines + 1, N_samples, N_pix).

    The means of an ensemble split over several arrays of equal shape (e.g. boxes) are the means of these. A NumPy input
    is averaged in float64 on the host (``DeltaFieldBuilder``, ``ensemble_mean_transmissions``: C = T_bar / prod_j T_bar_j
    - 1 is a small difference of means, ~1e-7 for weak metal lines, below the rounding of float32 means); a JAX input
    stays in JAX, in its dtype (jit)."""
    mean = _check_mean(mean)
    xp = np if isinstance(transmissions, np.ndarray) else jnp
    if xp is np:
        transmissions = np.asarray(transmissions, dtype=np.float64)
    T = xp.concatenate([transmissions, xp.prod(transmissions, axis=0)[None]])  # (N_lines + 1, N_samples, N_pix)
    if mean == "skewer":
        return xp.broadcast_to(xp.mean(T, axis=2, keepdims=True), T.shape)
    m = xp.mean(T, axis=1)
    if mean == "redshift":
        return m if grid is None else _group_mean(m, redshift_groups(grid, T.shape[-1]))
    return xp.broadcast_to(xp.mean(m, axis=1, keepdims=True), m.shape)


def ensemble_mean_transmissions(transmissions, mean: str = "redshift", *, grid=None) -> np.ndarray:
    """The mean transmission T_bar of every line and of the total over the set of pixels of ``mean`` (module docstring),
    for an ensemble of realisations (boxes) with the same number of samples each.

    transmissions : (n_real, N_lines, N_samples, N_pix) T_j = exp(-tau_j) of every realisation (or a list of
        (N_lines, N_samples, N_pix)).
    mean : "redshift" (default): at each pixel over every sample of every realisation, T_bar(z) (the mean of the
        per-realisation means at each pixel, ``mean_transmissions``; pixels at the same scale factor of ``grid`` share
        their mean, ``redshift_groups``); "global": over every pixel of every sample of every
        realisation; "box": over every pixel of every sample of each realisation; "skewer": over the pixels of each sample.
    grid : (N_pix,) or (n_real, N_pix), optional, with ``mean="redshift"``: the scale factor (or redshift) of every
        pixel of every realisation, checked to be shared (``check_shared_grid``, ``ValueError`` otherwise).

    Returns (n_real, N_lines + 1, N_samples, N_pix), the rows T_bar_j of the lines and, last, T_bar of the total
    prod_j T_j: a read-only broadcast view of the means (no copy), so that delta = T / T_bar - 1 is elementwise for every
    mode, and T_bar[i][:, s] is the mean seen by sample s of realisation i. The means are float64 whatever the dtype of
    the transmissions (``mean_transmissions``)."""
    mean = _check_mean(mean)
    T = [np.asarray(x, dtype=np.float64) for x in transmissions]
    n, (n_l, n_s, n_pix) = len(T), T[0].shape
    groups = redshift_groups(grid, n_pix, what="realisations") if mean == "redshift" and grid is not None else None
    shape = (n, n_l + 1, n_s, n_pix)
    if mean == "skewer":
        m = np.stack([np.concatenate([x, x.prod(axis=0, keepdims=True)]).mean(axis=2, keepdims=True) for x in T])
        return np.broadcast_to(m, shape)
    per = np.stack([np.asarray(mean_transmissions(x, "redshift")) for x in T])  # (n, N_lines + 1, N_pix): T_bar(v) per box
    m = {
        "redshift": lambda: _group_mean(per.mean(axis=0), groups)[
            None
        ],  # equal samples per realisation: T_bar(v) of the ensemble
        "global": lambda: per.mean(axis=0).mean(axis=-1, keepdims=True)[None],
        "box": lambda: per.mean(axis=-1, keepdims=True),
    }[mean]()
    return np.broadcast_to(m[:, :, None, :], shape)


_SET_AXES = {"redshift": (0, 2), "global": (0, 2, 3), "box": (2, 3), "skewer": (3,)}  # (n_real, field, sample, pixel)


def overflux_check(delta, mean_transmission, mean: str = "redshift", *, grid=None) -> dict:
    """Checks of the normalisation of an ensemble overflux over the sets of pixels of ``mean`` (module docstring).

    delta : (n_real, N_lines + 1, N_samples, N_pix) the overflux of every line and, last, of the total, delta = T / T_bar
        - 1 with mean_transmission (the same shape, ``ensemble_mean_transmissions``). grid : with ``mean="redshift"``,
        the pixel grid of the means (``ensemble_mean_transmissions``): the sets are the pixels at the same scale factor.

    Returns dict(
        mean_delta (N_lines + 1, N_pix): <delta> at each pixel over every sample of every realisation (the pattern shared
            by the samples: zero with "redshift" on a grid of distinct scale factors, a diagnostic in every mode),
        mean_delta_level (N_lines + 1,): max |<delta>| over the sets of the mode (zero to rounding),
        C (float): the mean of C = T_bar / prod_j T_bar_j - 1,
        C_error: max |C - sum_{|A| >= 2} <delta_A>| over the sets of the mode (delta_A = prod_{j in A} delta_j),
        decomposition_error: max |[sum_A delta_A - C] / (1 + C) - delta_tot| over every pixel of every sample (the exact
            decomposition, ``summary_stats.orders``))."""
    mean = _check_mean(mean)
    d = np.asarray(delta, dtype=np.float64)  # the checks in float64: C ~ 1e-7 is below float32's rounding of the means
    Tb = np.broadcast_to(np.asarray(mean_transmission, dtype=np.float64), d.shape)
    n, m = d.shape[0], d.shape[1] - 1
    H, C, err = np.empty((n,) + d.shape[2:]), np.empty((n,) + d.shape[2:]), 0.0
    for i in range(n):  # one realisation at a time: the products of the lines, C and the decomposition
        dl = d[i, :m]
        h = 0.0
        for r in range(2, m + 1):
            for A in combinations(range(m), r):
                h = h + np.prod(dl[list(A)], axis=0)
        H[i] = h
        C[i] = Tb[i, -1] / np.prod(Tb[i, :m], axis=0) - 1.0
        err = max(err, float(np.max(np.abs((sum(dl) + h - C[i]) / (1.0 + C[i]) - d[i, -1]))))
    ax = _SET_AXES[mean]
    groups = redshift_groups(grid, d.shape[-1], what="realisations") if mean == "redshift" and grid is not None else None
    lvl = np.abs(_group_mean(d.mean(axis=ax, keepdims=True), groups))
    ax3 = tuple(a - (a > 1) for a in ax)  # the same sets on (n_real, sample, pixel)
    return dict(
        mean_delta=d.mean(axis=(0, 2)),
        mean_delta_level=lvl.max(axis=tuple(a for a in range(4) if a != 1)).ravel(),
        C=float(C.mean()),
        C_error=float(np.max(np.abs(_group_mean(H.mean(axis=ax3) - C.mean(axis=ax3), groups)))),
        decomposition_error=err,
    )


def overflux_terms(
    transmissions: jnp.ndarray,
    labels: Sequence[str],
    key_joiner: str = "",
    mean_transmission: jnp.ndarray | None = None,
    mean: str = "redshift",
) -> dict:
    """Overflux terms of transmissions (N_lines, N_samples, N_pix) (pure; jit with static ``labels`` and ``mean``),
    normalised by the mean transmission of the ensemble (module docstring).

    mean_transmission : (N_lines + 1, N_pix) or (N_lines + 1, N_samples, N_pix), optional
        T_bar_j of every line and T_bar of the total (``mean_transmissions``) when the ensemble is larger than these
        samples (e.g. every box of a run); default: the mean over these samples, ``mean`` (``MEAN_MODES``).

    Returns dict(delta_terms {key: (N_samples, N_pix)} for every non-empty line subset (single lines first, then subsets
    of 2, 3, ... lines; keys join the labels), total (N_samples, N_pix), C_per_pixel = T_bar / prod_j T_bar_j - 1, (1,
    N_pix) for a mean shared by the samples (constant along v with "global" or "box"), (N_samples, N_pix) with one per
    sample ("skewer"), mean_transmission, transmission_total (N_samples, N_pix) = prod_j T_j).
    """
    T_bar = mean_transmissions(transmissions, mean) if mean_transmission is None else jnp.asarray(mean_transmission)
    per = (lambda x: x[None]) if T_bar.ndim == 2 else (lambda x: x)  # one mean for every sample, or one per sample
    delta = {}
    for i, key in enumerate(labels):
        delta[key] = transmissions[i] / per(T_bar[i]) - 1.0
    transmission_total = jnp.prod(transmissions, axis=0)
    total = transmission_total / per(T_bar[-1]) - 1.0
    C = per(T_bar[-1] / jnp.prod(T_bar[:-1], axis=0) - 1.0)
    for r in range(2, len(labels) + 1):
        for members in combinations(labels, r):
            term = delta[members[0]]
            for m in members[1:]:
                term = term * delta[m]
            delta[key_joiner.join(members)] = term
    return dict(delta_terms=delta, total=total, C_per_pixel=C, mean_transmission=T_bar, transmission_total=transmission_total)


class DeltaFieldBuilder:
    """Single-line and composite overflux fields of an optical-depth array (eager wrapper of ``overflux_terms``), each
    normalised by the mean transmission of its line (or of the total) over the ensemble (module docstring).

    Parameters
    ----------
    optical_depth : (N_lines, N_samples, N_pix).
    labels : one label per line (default "1", "2", ...); key_joiner joins them into subset keys.
    dtype : of the computation (``float``: float64 under x64).
    mean : "redshift" (default), "global", "box" or "skewer" (module docstring, ``MEAN_MODES``): the set of pixels of
        the mean transmission.
    grid : (N_pix,) or (N_samples, N_pix), optional: the scale factor (or redshift) of every pixel, checked to be shared
        by the samples with ``mean="redshift"`` (``check_shared_grid``); pixels at the same scale factor share their mean.
    mean_transmission : (N_lines + 1, N_pix) or, for "skewer", (N_lines + 1, N_samples, N_pix), optional
        The means T_bar_j and T_bar (``mean_transmissions``) when the ensemble holds more samples than ``optical_depth``
        (e.g. every box of a run); default: the mean over the kept samples of ``optical_depth`` (``mean``). With few
        samples, the mean of the samples is a noisy estimate of the ensemble mean.
    keep_transmissions, keep_transmission_total : keep T_j and prod_j T_j after ``build`` (released by default).

    Samples with NaN in tau or T are dropped before the means are taken (data-dependent shapes: this filtering is the
    eager part); a zero mean transmission at a pixel (of a line or of the total) raises. After ``build()``:
    ``subset_keys``, ``composite_keys``, ``delta_terms``, ``total``, ``C_per_pixel`` and ``mean_transmission`` (NumPy,
    float64: the means are taken on the host in float64, so that C = T_bar / prod_j T_bar_j - 1, ~1e-7 for weak metal
    lines, is resolved whatever the precision of the overflux fields), ``N_samples``.
    """

    def __init__(
        self,
        optical_depth,
        labels=None,
        key_joiner="",
        dtype=float,
        *,
        mean: str = "redshift",
        grid=None,
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
        self.key_joiner, self.dtype, self.mean = key_joiner, dtype, _check_mean(mean)
        self.keep_transmissions, self.keep_transmission_total = keep_transmissions, keep_transmission_total
        self.mean_transmission = None if mean_transmission is None else np.asarray(mean_transmission, dtype=np.float64)
        shapes = ((self.N_lines + 1, self.N_points), (self.N_lines + 1, self.N_samples, self.N_points))
        if self.mean_transmission is not None and self.mean_transmission.shape not in shapes:
            raise ValueError(
                f"`mean_transmission` must have shape (N_lines + 1, N_pix) = {shapes[0]} or (N_lines + 1, N_samples, "
                f"N_pix) = {shapes[1]}. Got {self.mean_transmission.shape}."
            )
        self.grid = None
        if grid is not None and self.mean == "redshift":
            check_shared_grid(grid, self.N_points)
            self.grid = np.asarray(grid)
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
        if self.mean_transmission is not None and self.mean_transmission.ndim == 3:  # one mean per sample ("skewer")
            self.mean_transmission = self.mean_transmission[:, ~bad, :]
        self.N_samples = self.transmissions.shape[1]
        if self.N_samples == 0:
            raise ValueError("All samples were removed during filtering.")

    def build(self):
        if self._built and self.transmissions is None:
            raise RuntimeError(
                "This DeltaFieldBuilder has already been built and its temporary transmissions were released. "
                "Create a new instance to rebuild."
            )
        T_bar = (  # float64 on the host, whatever the precision of the transmissions (``mean_transmissions``)
            mean_transmissions(np.asarray(self.transmissions), self.mean, grid=self.grid)
            if self.mean_transmission is None
            else self.mean_transmission
        )
        if bool(np.any(T_bar == 0.0)):
            raise ValueError("A mean transmission is zero at some pixel (of a line or of the total): delta is undefined there.")
        out = overflux_terms(self.transmissions, self.labels, self.key_joiner, jnp.asarray(T_bar, self.transmissions.dtype), self.mean)
        for k, v in out["delta_terms"].items():
            if int(jnp.sum(jnp.isnan(v))):
                logger.warning("NaNs detected in delta_terms[%r].", k)
        self.delta_terms, self.total = out["delta_terms"], out["total"]
        C = T_bar[-1] / np.prod(T_bar[:-1], axis=0) - 1.0  # in float64: a small difference of means
        self.C_per_pixel = C[None] if C.ndim == 1 else C
        self.mean_transmission = T_bar
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
