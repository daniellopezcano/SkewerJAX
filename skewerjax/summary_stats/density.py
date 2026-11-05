"""The density sampled along a lattice of periodic skewers (notebook 10, Part B), from arrays and geometry only: its 3D
statistics through the estimator of the overflux on the lattice of the points (``lattice.lattice_grid``), its 1D
statistics along the skewers, and the exact expectation of both on that lattice from the spectrum of the field, passed
as a callable P(k).

* ``skewer_density_statistics``: the overdensity of every species and of their sum along the skewers of every
  realisation, against its global mean, through the 3D estimator and along each skewer;
* ``lattice_density_theory``: the exact expectation of those statistics, xi_grid at the separations of the lattice
  (interpolated linearly between the cells along the line of sight, exactly) and its aliased spectrum.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np

from skewerjax.summary_stats.binning import ensemble_mean
from skewerjax.summary_stats.field_stats import grid_power
from skewerjax.summary_stats.grid3d import GRID_KEYS, GridStatistics3D, _rfft_weights
from skewerjax.summary_stats.lattice import on_points
from skewerjax.summary_stats.mesh import lattice_only
from skewerjax.summary_stats.stats_1d import OverfluxStatistics1D, ensemble_1d


# ---------------------------------------------------------------------- the exact expectation on the lattice
def _lattice_power(power: Callable, n_cells: int, L: float, stride, points_per_cell: int = 1) -> np.ndarray:
    """The aliased spectrum P_lat on the rfft modes of the lattice of skewers with ``stride`` (s_x, s_z) through an
    ``n_cells``^3 grid of side ``L`` [cMpc], with n_cells p points along y (p = ``points_per_cell``), its k = 0 entry set
    to 0 as in the estimator: xi_lat = xi_grid at the lattice separations (``lattice_density_theory``), xi_grid =
    irfftn(P on the grid's modes) / dV, and P_lat = dV_lat rfftn(xi_lat) [cMpc^3]. ``power``: callable P(k) of |k|."""
    N, p = int(n_cells), int(points_per_cell)
    sx, sz = (int(s) for s in stride)
    shape = (N,) * 3
    dx = float(L) / N
    dV = float(np.prod(tuple(float(L) / n for n in shape)))
    xi = np.fft.irfftn(np.asarray(grid_power(power, shape, L)), s=shape) / dV
    xi = xi[::sx, :, ::sz]
    if p > 1:  # two points Delta m = q p + s apart lie q cells apart for a fraction 1 - s / p of the pairs, q + 1 for s / p
        q, s = np.divmod(np.arange(N * p), p)
        t = (s / p)[None, :, None]
        xi = (1.0 - t) * xi[:, q, :] + t * xi[:, (q + 1) % N, :]
    P = np.fft.rfftn(xi).real * (sx * dx) * (dx / p) * (sz * dx)
    P[0, 0, 0] = 0.0
    return P


def _lattice_p1d(P, gs: GridStatistics3D) -> np.ndarray:
    """The 1D spectrum along the skewers of a lattice spectrum P (rfft layout of ``gs``, line of sight axis 1):
    P1D(k_par) = (1 / (L_x L_z)) sum_{k_perp} P(k_perp, k_par) at k_par = 2 pi j / L, j = 0 .. n_los / 2 [cMpc]."""
    return (P * _rfft_weights(gs.shape[2])).sum(axis=(0, 2))[: gs.shape[1] // 2 + 1] / (gs.L[0] * gs.L[2])


def lattice_density_theory(
    power: Callable,
    grid: GridStatistics3D,
    n_cells: int,
    stride,
    *,
    points_per_cell: int = 1,
    p1d: bool = False,
    scale: float = 1.0,
):
    """The exact expectation of the 3D statistics on ``grid`` of a field of spectrum ``power`` sampled on the lattice of
    skewers with ``stride`` (s_x, s_z) through an ``n_cells``^3 periodic grid: every s_x-th column in x, every s_z-th
    in z, and along y every cell (p = ``points_per_cell`` = 1) or every point of the skewers (``lattice.on_points``: the
    cell of each point).

    power : callable P(k) [cMpc^3] of |k| [1/cMpc] (e.g. ``PowerLawGRF.power``; ``lambda k: g.power(k, bao=False)`` for
        the power law alone): the spectrum on the grid's modes (``grid_power``).
    grid : the ``lattice_grid`` of the points, (n_cells / s_x, n_cells p, n_cells / s_z), whose line-of-sight side is
        the box side L [cMpc] and whose bins are those of the expectation (checked against n_cells, stride and p).
    scale : a factor of the spectrum (the global-mean normalisation of the overdensity, ``skewer_density_statistics``'s
        ``density_scale``).

    The lattice samples xi_grid = irfftn(P on the modes) / dV at its own separations; along y, two points Delta m =
    q p + s apart lie q or q + 1 cells apart, for the fractions 1 - s / p and s / p of the pairs, so xi_lat is xi_grid
    interpolated linearly between the cells, exactly. Its spectrum is the aliased P_lat = dV_lat rfftn(xi_lat) (the input
    folded over the aliases of the transverse lattice and, along y, the spectrum of the piecewise-constant columns),
    binned like the measurement (``GridStatistics3D.theory``), without the k = 0 mode. ``p1d``: also return the 1D
    spectrum along the skewers, P1D(k_par) = (1 / (L_x L_z)) sum_{k_perp} P_lat(k_perp, k_par), k_par = 2 pi j / L,
    j = 0 .. n_los / 2 [cMpc] (returns (theory, P1D))."""
    N, p = int(n_cells), int(points_per_cell)
    sx, sz = (int(s) for s in stride)
    if N % sx or N % sz:
        raise ValueError(f"the lattice needs n_cells = {N} to be a multiple of both strides {tuple(stride)}.")
    expect = (N // sx, N * p, N // sz)
    if tuple(grid.shape) != expect:
        raise ValueError(f"grid.shape = {tuple(grid.shape)}, but n_cells, stride, points_per_cell give the lattice {expect}.")
    P = _lattice_power(power, N, float(grid.L[1]), (sx, sz), p)
    P = scale * P if scale != 1.0 else P
    th = grid.theory(P)
    return (th, _lattice_p1d(P, grid)) if p1d else th


# ---------------------------------------------------------------------- the measurement along the skewers
class _LinearTerms:
    """The density counterpart of a built ``DeltaFieldBuilder`` for ``OverfluxStatistics1D``: the total overdensity of a
    sum of species, delta_tot = sum_j w_j delta_j (w_j = the mean of species j over the mean of the sum), is linear, so
    its terms are the weighted single species, "1", "2", ..., with no product and C = 0 (every pair (j, l) has order
    2)."""

    def __init__(self, terms, total):
        labels = [str(j + 1) for j in range(len(terms))]
        self.labels, self.key_joiner = labels, ""
        self.subset_keys = list(labels)
        self.delta_terms = dict(zip(labels, terms))
        self.total = total
        self.C_per_pixel = np.zeros((1, total.shape[-1]))


def _density_statistics(cols: dict, means: dict, gs: GridStatistics3D, points_per_cell: int, L: float) -> dict:
    """The 3D and 1D statistics of one realisation's density skewers at their points (``on_points``): the overdensity of
    each species and of their sum against its global mean, delta_X = n_X / <n_X> - 1 (``means``, absolute), on the
    lattice of the points (``gs``) and along every skewer (P1D, and the 1D summary with the weighted species as terms).
    Returns dict(flux3d {field: GRID_KEYS}, flux (``OverfluxStatistics1D.summary``), p1d (n_species + 1, N_k) [cMpc],
    k1d [cMpc^-1], identity: max |delta_1 - delta_j| over the other species, max |delta_1 - delta_tot|)."""
    p = int(points_per_cell)
    species = list(cols)
    fields = (*species, "total")
    n = {sp: on_points(cols[sp], p) for sp in species}
    n["total"] = sum(n[sp] for sp in species)
    delta = {f: n[f] / means[f] - 1.0 for f in fields}
    n_los = delta[species[0]].shape[1]
    dx = float(L) / n_los
    flat = {f: d.transpose(0, 2, 1).reshape(-1, n_los) for f, d in delta.items()}  # (N_sk, n_los), x-major as the columns
    w = {sp: means[sp] / means["total"] for sp in species}
    lin = _LinearTerms([w[sp] * flat[sp] for sp in species], flat["total"])
    D = np.stack([flat[f] for f in fields])
    return dict(
        flux3d={f: gs.measure(delta[f]) for f in fields},
        flux=OverfluxStatistics1D(lin, dx).summary(),
        k1d=2 * np.pi * np.fft.rfftfreq(n_los, dx),
        p1d=(dx / n_los) * np.mean(np.abs(np.fft.rfft(D, axis=-1)) ** 2, axis=1),
        identity=np.array(
            [
                max((np.abs(delta[species[0]] - delta[sp]).max() for sp in species[1:]), default=0.0),
                np.abs(delta[species[0]] - delta["total"]).max(),
            ]
        ),
    )


@lattice_only
def skewer_density_statistics(columns, points_per_cell: int, grid: GridStatistics3D, L: float, *, n_bar=None) -> dict:
    """The density of every species along the skewers of every realisation, sampled at the points of the skewers
    (``on_points``: the end point dropped, spacing dx / points_per_cell, every cell holding points_per_cell points),
    against its global mean over every point of every skewer of every realisation, delta_X = n_X / <n_X> - 1, through
    the 3D estimator of the overflux on the lattice of the points (``grid``: ``lattice_grid`` with n_los = n_cells
    points_per_cell) and the 1D statistics along the skewers (notebook 10, Part B). Arrays and geometry only.

    columns : [{species: (n_x, n_cells, n_z)}] per realisation, one or more species (their order: the terms "1", "2", ...
        and the fields (species_1, species_2, ..., "total")), the number densities of the cells the lattice of skewers crosses
        [comoving cm^-3]; L : the box side [cMpc]; n_bar : {species: n_bar} (optional): the mean densities of the
        model, for density_mean and density_scale. The points of the skewers are the nodes of the lattice of ``grid``, so
        the FFT is exact, with no mass assignment, window or interlacing (Sefusatti et al. 2016, §2.3); ``mas`` or
        ``interlace`` raise (``mesh.lattice_only``).

    Returns dict(density3d {field: ``ensemble_mean`` of the ``GRID_KEYS``}, density_axes3d, density_flux (``ensemble_1d``
    of the 1D summary with the weighted species as terms), density_p1d (``ensemble_mean`` of "p1d", with "k"
    [cMpc^-1]), density_means {field: <n_X>} [comoving cm^-3], density_identity (max over the realisations of
    |delta_1 - delta_j| over the other species, and of |delta_1 - delta_total|), time_s (n,); with ``n_bar``: density_mean {field: <n_X> /
    n_bar_X} and density_scale = (n_bar / <n>)^2 of the total (delta_X = (n_bar_X / <n_X>)(1 + delta) - 1: the factor of
    the expected spectrum, ``lattice_density_theory(scale=)``))."""
    species = list(columns[0])
    if not species:
        raise ValueError("skewer_density_statistics needs at least one species.")
    fields = (*species, "total")
    means = {sp: float(np.mean([c[sp].mean() for c in columns])) for sp in species}
    means["total"] = sum(means[sp] for sp in species)
    dens, ts = [], []
    for c in columns:
        t0 = time.perf_counter()
        dens.append(_density_statistics(c, means, grid, points_per_cell, L))
        ts.append(time.perf_counter() - t0)
    out = dict(
        density3d={f: ensemble_mean([d["flux3d"][f] for d in dens], GRID_KEYS) for f in fields},
        density_axes3d=grid.axes(),
        density_flux=ensemble_1d([d["flux"] for d in dens]),
        density_p1d=dict(ensemble_mean(dens, ("p1d",)), k=dens[0]["k1d"]),
        density_means=means,
        density_identity=np.max([d["identity"] for d in dens], axis=0),
        time_s=np.array(ts),
    )
    if n_bar is not None:
        nb = dict(n_bar, total=sum(n_bar[sp] for sp in species))
        out.update(density_mean={f: means[f] / nb[f] for f in fields}, density_scale=(nb["total"] / means["total"]) ** 2)
    return out
