"""Deposition: the pieces of the path, i.e. which cell lies on the dual segment of every sampling point, and where.

The gas is a set of Voronoi cells (``core.cells.Cells``, from a toy model or a simulation). The sampling points are
those of the bundle (any spacing, straight or curved). Point i stands for its *dual segment* m_{i-1/2} -> x_i ->
m_{i+1/2} (``core.skewers.path_geometry``), of comoving length w_i. A deposition answers one question: **which gas
lies on that segment, and where?** It returns for every point up to K *pieces*: a cell index, the comoving length of
the cell's piece of the segment, the signed path offset of that piece's centroid from x_i, and a density weight.
Inside a piece the gas properties are constant; the core sums the optical depth over the pieces
(``core.absorbers.cell_pieces``, ``core.optical_depth``).

* ``voronoi_segments`` (SkewerJAX default): Arepo's own geometry. A point of space belongs to the cell of its nearest
  mesh-generating point (TNG ``Coordinates``), so along the path the gas is piecewise constant and single valued. On a
  straight line x_0 + s t the squared distance to generator g is (s - u)^2 + r^2, u = t.(g - x_0), r its distance
  from the line: s^2 is common to all generators, so the nearest one is the lower envelope of *straight lines*
  -2 u s + u^2 + r^2. **Straight paths** (``straight_paths``) take that envelope directly: the cells crossed and their
  exact crossings for the whole path in O(K log K) (K candidate generators), cut at the boundaries of the dual
  segments. **Curved paths** are straight between consecutive points and their midpoints: every half-segment is
  sampled at ``n_sub`` sub-points labelled by a periodic k-d tree, and every change of label is located at the exact
  crossing of the two lines. Weight 1. ``cell_crossings`` joins the pieces of one cell in consecutive dual segments:
  the exact intersections of the path with the cells.
* ``grid_segments``: the same pieces for a regular grid of cells (``cells.cells_on_grid``), whose cells are cubes
  centred on the nodes: the boundaries are the planes halfway between nodes, crossed at exact parameters along
  every straight half-segment, and the cell of a piece is the node nearest to its midpoint. No k-d tree and no
  selection; ``tests/test_cells.py`` checks that it gives the pieces of ``voronoi_segments``.
* ``tophat_segments`` (fake_spectra's geometry): each cell is a sphere of radius h = V^(1/3) and
  uniform density (3 / 4 pi) n (``absorption.cpp:tophat_kern_frac`` L112-L118). The overlap of
  every sphere with every half-segment is solved exactly (a quadratic in t). Spheres overlap, so a
  segment has several pieces with *different* T and v: the representation is multi-valued.
* ``tophat_chord_segments`` (fake_spectra's quadrature of those spheres, ``singleabs.h`` L100-L118): every chord of a
  sphere with a straight path is sampled at its 7 interior points, instead of the exact overlaps. With its weights
  (1/8 per point) a chord holds 7/8 of its column, the one deposition that does not conserve it (``chord_weight``).

Every other deposition conserves the column by construction (sum of piece lengths x density), for
any spacing, because the lengths are exact segment integrals; only *where* inside the piece the
absorption sits is approximated (at the piece centroid). That approximation is what the sampling
criterion of ``core.absorbers.sampling_check`` monitors.

Periodicity of the box (the gas, not the spectrum): generator positions are always taken as the minimum image
relative to each point, x_i + mi(g - x_i), mi(d) = d - L round(d / L), so paths may cross the box faces any number
of times.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial import cKDTree

from skewerjax.core.selection import VORONOI_FACTOR, wrap_positions
from skewerjax.core.skewers import path_geometry

TOPHAT_NORM = 3.0 / (4.0 * np.pi)  # density inside a sphere of radius V^(1/3), per n_cell


@dataclass
class SegmentComponents:
    """The pieces of the dual segment of every sampling point (host arrays).

    Attributes
    ----------
    cell : (N_sk, N, K) int, index into the cells used (-1: empty slot).
    length : (N_sk, N, K) comoving length of the piece [cMpc].
    offset : (N_sk, N, K) signed path offset of the piece centroid from x_i [cMpc].
    weight : float, density per n_cell inside the piece (1 Voronoi, 3/4pi top hat).
    kind : "voronoi", "tophat" or "tophat_chords".
    diagnostics : dict (e.g. the k-nearest-neighbour guarantee, the largest nearest-generator distance).
    """

    cell: np.ndarray
    length: np.ndarray
    offset: np.ndarray
    weight: float
    kind: str
    diagnostics: dict

    @property
    def n_components(self) -> np.ndarray:
        """(N_sk, N) number of pieces of every dual segment."""
        return (self.cell >= 0).sum(axis=-1)

    def compact(self) -> SegmentComponents:
        """Drop the trailing empty slots (K -> the largest number of components of any point)."""
        order = np.argsort(-(self.length * (self.cell >= 0)), axis=-1, kind="stable")
        take = lambda a: np.take_along_axis(a, order, axis=-1)
        k = max(int(self.n_components.max()), 1)
        return SegmentComponents(
            take(self.cell)[..., :k],
            take(self.length)[..., :k],
            take(self.offset)[..., :k],
            self.weight,
            self.kind,
            self.diagnostics,
        )

    def column_lengths(self, n_cells) -> np.ndarray:
        """(N_sk, N) sum over components of weight x n_cell x length [n units x cMpc]."""
        n = np.asarray(n_cells, dtype=np.float64)
        return (self.weight * np.where(self.cell >= 0, n[np.maximum(self.cell, 0)], 0.0) * self.length).sum(axis=-1)


# --------------------------------------------------------------------------- helpers
def _min_image(d, box):
    return d if box is None else d - box * jnp.round(d / box)


def _halves(points):
    """Half-segments of every point: (A, B_back, B_fwd), each (P, 3) [cMpc]: the point and the midpoints around it."""
    A = np.asarray(points, dtype=np.float64)
    g = path_geometry(A)  # float64 NumPy, whatever the precision policy
    return A.reshape(-1, 3), g["mid_lo"].reshape(-1, 3), g["mid_hi"].reshape(-1, 3)


# --------------------------------------------------------------------------- Voronoi
def _crossing_np(A, D, ga, gb, t0, t1):
    """``_crossing`` in float64 NumPy (the host back end of ``voronoi_segments``)."""
    da, db = A[:, None, :] - ga, A[:, None, :] - gb
    alpha_a, alpha_b = np.einsum("psk,psk->ps", da, da), np.einsum("psk,psk->ps", db, db)
    beta_a, beta_b = 2.0 * np.einsum("psk,pk->ps", da, D), 2.0 * np.einsum("psk,pk->ps", db, D)
    den = beta_b - beta_a
    same = np.all(ga == gb, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        tc = np.where(same | (den == 0), t1[None, :], (alpha_a - alpha_b) / np.where(den == 0, 1.0, den))
    return np.clip(tc, t0[None, :], t1[None, :])


def _min_image_np(d, box):
    return d if box is None else d - box * np.round(d / box)


@jax.jit
def _crossing(A, D, ga, gb, t0, t1):
    """Exact parameter t in [t0, t1] where the nearest generator changes from ga to gb on A + t D.

    |A + tD - g|^2 = |A - g|^2 + 2 t (A - g).D + t^2 |D|^2; the t^2 term is common, so the two
    distances are equal where two straight lines in t cross. Arrays: A, D (P, 3); ga, gb (P, S, 3)
    (minimum images w.r.t. A); t0, t1 (S,). Returns (P, S); t1 where ga == gb.
    """
    da, db = A[:, None, :] - ga, A[:, None, :] - gb
    alpha_a, alpha_b = jnp.sum(da * da, -1), jnp.sum(db * db, -1)
    beta_a, beta_b = 2.0 * jnp.sum(da * D[:, None, :], -1), 2.0 * jnp.sum(db * D[:, None, :], -1)
    den = beta_b - beta_a
    same = jnp.all(ga == gb, axis=-1)
    tc = jnp.where(same | (den == 0), t1[None, :], (alpha_a - alpha_b) / jnp.where(den == 0, 1.0, den))
    return jnp.clip(tc, t0[None, :], t1[None, :])


def voronoi_segments(
    points,
    gas_positions,
    box: float | None,
    *,
    gas_size=None,
    method: str = "auto",
    n_sub: int | None = None,
    max_sub: int = 4096,
    backend: str = "numpy",
    chunk: int = 64,
    k_chunk: int = 512,
    threads: int | None = None,
) -> SegmentComponents:
    """Nearest-generating-point (Voronoi) deposition on the dual segments of ``points``.

    Two constructions of the same pieces:

    * **envelope** (straight paths, exact): along a straight path the nearest generator at every s is the lower envelope
      of straight lines in s (``_voronoi_envelope``): the cells crossed and their exact crossings, in O(K log K) per
      path for K candidate generators, then cut at the boundaries of the dual segments.
    * **sampled** (any path): every half-segment is sampled at ``n_sub`` + 1 sub-points, each labelled with its nearest
      generator (periodic k-d tree over ``gas_positions``), and wherever two consecutive labels differ the boundary
      between the two cells is located exactly (``_crossing``). Exact as long as no cell is crossed in a chord shorter
      than a sub-interval; with ``gas_size`` and ``n_sub=None``, ``n_sub`` makes every sub-interval shorter than 1/16 of
      the smallest cell of ``gas_positions`` (capped at ``max_sub``); a grazed cell can then be missed only over less
      than that length.

    Parameters
    ----------
    points : (N_sk, N, 3) sampling points [cMpc] (unwrapped along the path).
    gas_positions : (M, 3) mesh-generating points [cMpc] in [0, box).
    box : periodic box [cMpc] or None.
    gas_size : (M,) cell sizes V^(1/3) [cMpc] (to choose ``n_sub``, and the diagnostic below).
    method : "auto" (default: the envelope for the straight paths, ``straight_paths``, the sampled construction for the
        others), "envelope" (every path must be straight) or "sampled" (every path).
    n_sub, max_sub, backend, chunk, k_chunk : the sampled construction only. ``backend``: "numpy" (SciPy k-d tree
        labels, NumPy merging) or "jax" (the same algorithm in JAX with static shapes, ``_voronoi_segments_jax``;
        identical cells, lengths equal to round-off); ``chunk``, ``k_chunk`` ("jax"): consecutive points of a path per
        chunk and candidate generators per chunk (doubled automatically until the candidate sets are provably complete).
    threads : host threads over chunks of skewers (default: one per core, at most one per skewer); the result does not
        depend on it.

    Returns
    -------
    SegmentComponents (weight 1); ``diagnostics`` holds ``n_envelope`` and ``n_sampled`` (paths of each construction),
    for the envelope the candidate images per path (``candidates_per_path``, mean), for the sampled one ``n_sub`` and
    the sub-interval relative to the smallest cell; with ``gas_size``, ``max_distance_over_size``, the largest distance
    from a point (envelope) or sub-point (sampled, "numpy") to its nearest generating point in units of that cell's size
    (a warning when it exceeds ``VORONOI_FACTOR`` without a periodic box: the paths leave the gas).
    """
    if method not in ("auto", "envelope", "sampled"):
        raise ValueError(f"method must be 'auto', 'envelope' or 'sampled', got {method!r}.")
    if backend not in ("numpy", "jax"):
        raise ValueError(f"backend must be 'numpy' or 'jax', got {backend!r}.")
    pts = np.asarray(points, dtype=np.float64)
    n_sk, n_p = pts.shape[:2]
    gpos = wrap_positions(gas_positions, box)
    size = None if gas_size is None else np.asarray(gas_size, dtype=np.float64)
    straight = np.zeros(n_sk, bool) if method == "sampled" else straight_paths(pts)
    if method == "envelope" and not straight.all():
        raise ValueError(f"method='envelope' needs straight paths; {int((~straight).sum())} of {n_sk} are not (straight_paths).")
    tree = cKDTree(gpos, boxsize=box) if box is not None else cKDTree(gpos)
    n_thr = max(1, min(os.cpu_count() or 1, n_sk)) if threads is None else max(1, int(threads))
    diag = dict(n_envelope=int(straight.sum()), n_sampled=int((~straight).sum()))
    parts, worst = [], []
    if straight.any():
        idx = np.flatnonzero(straight)
        res = _threaded(lambda sl: _voronoi_envelope(pts[idx[sl]], tree, gpos, box, size), idx.size, n_thr)
        parts.append((idx, _stack_segments([r[0] for r in res])))
        worst += [r[1] for r in res]
        diag["candidates_per_path"] = float(np.concatenate([r[2] for r in res]).mean())
    if not straight.all():
        idx = np.flatnonzero(~straight)
        seg, w, d = _voronoi_sampled(pts[idx], tree, gpos, box, size, n_sub, max_sub, backend, chunk, k_chunk, n_thr)
        parts.append((idx, seg))
        worst += [w]
        diag.update(d)
    if size is not None and worst:  # how far the path gets from the gas, in sizes of its nearest cell
        diag["max_distance_over_size"] = float(max(worst))
        if box is None and diag["max_distance_over_size"] > VORONOI_FACTOR:
            warnings.warn(
                f"voronoi_segments: parts of the paths lie {diag['max_distance_over_size']:.1f} cell sizes from the nearest "
                f"generating point (more than {VORONOI_FACTOR}): the paths leave the (non-periodic) gas, and those "
                "stretches take the nearest cell's gas.",
                stacklevel=2,
            )
    if len(parts) == 1 and np.array_equal(parts[0][0], np.arange(n_sk)):
        seg = parts[0][1]
    else:  # straight and curved paths together: back in the order of the paths
        K = max(p[1].cell.shape[-1] for p in parts)
        seg = SegmentComponents(-np.ones((n_sk, n_p, K), np.int64), np.zeros((n_sk, n_p, K)), np.zeros((n_sk, n_p, K)), 1.0, "voronoi", {})
        for idx, sg in parts:
            k = sg.cell.shape[-1]
            seg.cell[idx, :, :k], seg.length[idx, :, :k], seg.offset[idx, :, :k] = sg.cell, sg.length, sg.offset
    return SegmentComponents(seg.cell, seg.length, seg.offset, 1.0, "voronoi", diag).compact()


def _threaded(job, n: int, n_thr: int) -> list:
    """``job(slice)`` over chunks of ``range(n)`` (about 4 per thread), on ``n_thr`` threads (one shared k-d tree; NumPy,
    numba's nogil kernels and the tree queries release the GIL): the results in order."""
    bounds = np.linspace(0, n, min(n, 4 * n_thr) + 1).astype(int)
    chunks = [slice(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]
    if n_thr > 1 and len(chunks) > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=n_thr) as pool:
            return list(pool.map(job, chunks))
    return [job(sl) for sl in chunks]


def _stack_segments(parts) -> SegmentComponents:
    """The ``SegmentComponents`` of consecutive chunks of paths in one (the slots padded to the largest K)."""
    K = max(p.cell.shape[-1] for p in parts)
    pad = lambda x, fill: np.concatenate([x, np.full(x.shape[:-1] + (K - x.shape[-1],), fill, x.dtype)], axis=-1)
    return SegmentComponents(
        np.concatenate([pad(p.cell, -1) for p in parts]),
        np.concatenate([pad(p.length, 0.0) for p in parts]),
        np.concatenate([pad(p.offset, 0.0) for p in parts]),
        1.0,
        "voronoi",
        {},
    )


def _voronoi_sampled(pts, tree, gpos, box, size, n_sub, max_sub, backend, chunk, k_chunk, n_thr):
    """The sampled construction of ``voronoi_segments`` for the paths ``pts``: (SegmentComponents, the largest distance
    from a sub-point to its generator in cell sizes, diagnostics)."""
    n_sk, n_p = pts.shape[:2]
    A, Bl, Bh = _halves(pts)
    L_half = np.stack([np.linalg.norm(Bl - A, axis=-1), np.linalg.norm(Bh - A, axis=-1)], axis=1)  # (P, 2)
    if n_sub is not None:
        S = int(n_sub)
    elif size is None:
        S = 8
    else:  # sub-interval <= 1/16 of the smallest cell near the paths
        S = int(np.clip(np.ceil(16.0 * L_half.max() / max(size.min(), 1e-12)), 8, max_sub))
    diag = dict(n_sub=S, sub_over_min_cell=float(L_half.max() / S / size.min()) if size is not None else None)
    if backend == "jax":
        cell, length, offset, kd = _voronoi_segments_jax(A, Bl, Bh, L_half, gpos, box, S, n_sk, chunk, k_chunk)
        shp = (n_sk, n_p, cell.shape[-1])
        return SegmentComponents(cell.reshape(shp), length.reshape(shp), offset.reshape(shp), 1.0, "voronoi", {}).compact(), 0.0, {**diag, **kd}
    res = _threaded(lambda sl: _voronoi_numpy(pts[sl], tree, gpos, box, size, S), n_sk, n_thr)
    return _stack_segments([r[0] for r in res]), max(r[1] for r in res), diag


def _voronoi_numpy(pts, tree, gpos, box, size, S: int):
    """``voronoi_segments`` (host back end) of the skewers ``pts`` (n, N, 3) with the k-d tree of all the generators:
    (SegmentComponents, the largest distance from a sub-point to its generator in units of that cell's size (0 without
    ``size``))."""
    n_sk, n_p = pts.shape[:2]
    A, Bl, Bh = _halves(pts)
    P = A.shape[0]
    L_half = np.stack([np.linalg.norm(Bl - A, axis=-1), np.linalg.norm(Bh - A, axis=-1)], axis=1)  # (P, 2)
    t = np.linspace(0.0, 1.0, S + 1)
    sub = np.stack([A[:, None, :] + t[None, :, None] * (B - A)[:, None, :] for B in (Bl, Bh)], axis=1)  # (P, 2, S+1, 3)
    dist, lab = tree.query(wrap_positions(sub.reshape(-1, 3), box), k=1, workers=1)
    lab = lab.reshape(P, 2, S + 1)
    rows, labs, lens, moms = [], [], [], []
    t0, t1 = t[:-1], t[1:]
    for h, (B, sign) in enumerate(((Bl, -1.0), (Bh, 1.0))):
        D = B - A
        la, lb = lab[:, h, :-1], lab[:, h, 1:]
        ga = A[:, None, :] + _min_image_np(gpos[la] - A[:, None, :], box)
        gb = A[:, None, :] + _min_image_np(gpos[lb] - A[:, None, :], box)
        tc = _crossing_np(A, D, ga, gb, t0, t1)
        Lh = L_half[:, h][:, None]
        T0, T1 = t[None, :-1], t[None, 1:]
        for lab_, lo, hi in ((la, T0, tc), (lb, tc, T1)):
            rows.append(np.broadcast_to(np.arange(P)[:, None], lab_.shape).ravel())
            labs.append(lab_.ravel())
            lens.append(((hi - lo) * Lh).ravel())
            moms.append((sign * 0.5 * (hi**2 - lo**2) * Lh**2).ravel())
    worst = float(np.max(dist / size[lab.reshape(-1)])) if size is not None else 0.0
    return _merge_pieces(rows, labs, lens, moms, P, len(gpos), (n_sk, n_p), {}), worst


def _merge_pieces(rows, labs, lens, moms, P: int, M: int, shape, diag) -> SegmentComponents:
    """Voronoi ``SegmentComponents`` from the contributions (point row, cell label, length, moment) of every part of
    every half-segment: the parts of one cell in one dual segment are merged (lengths and moments summed)."""
    n_sk, n_p = shape
    rows, labs, lens, moms = (np.concatenate(x) for x in (rows, labs, lens, moms))
    keep = lens > 0
    key = rows[keep].astype(np.int64) * M + labs[keep]
    ukey, inv = np.unique(key, return_inverse=True)
    L_ = np.bincount(inv, weights=lens[keep])
    M_ = np.bincount(inv, weights=moms[keep])
    prow, plab = ukey // M, ukey % M
    counts = np.bincount(prow, minlength=P)
    K = max(int(counts.max()), 1)
    start = np.concatenate([[0], np.cumsum(counts)[:-1]])
    slot = np.arange(prow.size) - start[prow]
    cell, length, offset = -np.ones((P, K), np.int64), np.zeros((P, K)), np.zeros((P, K))
    cell[prow, slot] = plab
    length[prow, slot] = L_
    offset[prow, slot] = np.where(L_ > 0, M_ / np.where(L_ > 0, L_, 1.0), 0.0)
    shp = (n_sk, n_p, K)
    return SegmentComponents(cell.reshape(shp), length.reshape(shp), offset.reshape(shp), 1.0, "voronoi", diag).compact()


# --------------------------------------------------------------------------- Voronoi: straight paths, exact
STRAIGHT_RTOL = 1e-7  # a path is straight when no point is farther than this x its length from the chord of its ends


def straight_paths(points, rtol: float = STRAIGHT_RTOL) -> np.ndarray:
    """(N_sk,) bool: the paths of ``points`` (N_sk, N, 3) [cMpc] that are straight, i.e. whose points lie on the chord
    from their first to their last point within ``rtol`` x its length, in increasing order along it."""
    p = np.asarray(points, dtype=np.float64)
    x0, chord = p[:, :1, :], p[:, -1, :] - p[:, 0, :]
    length = np.linalg.norm(chord, axis=-1)
    t = chord / np.maximum(length, 1e-300)[:, None]
    rel = p - x0
    s = np.einsum("knc,kc->kn", rel, t)
    dev = np.linalg.norm(rel - s[..., None] * t[:, None, :], axis=-1).max(axis=-1)
    return (length > 0) & (dev <= rtol * length) & np.all(np.diff(s, axis=-1) >= 0, axis=-1)


def _envelope_kernel():
    """The lower envelope of the squared distances along a line (numba, no GIL): ``run(u, r2)`` with the candidates
    sorted by u (ties by r2) returns (k (m,) the candidates on the envelope in order, start (m,) the path coordinate
    where each takes over (-inf for the first))."""
    import numba

    @numba.njit(nogil=True)  # compiled per process (no disk cache)
    def run(u, r2):
        n = u.shape[0]
        stack = np.empty(n, np.int64)
        start = np.empty(n, np.float64)
        m = 0
        for i in range(n):
            if m > 0 and u[stack[m - 1]] == u[i]:  # the same slope, a larger or equal r^2: never below
                continue
            x = -np.inf
            while m > 0:
                j = stack[m - 1]
                # (s - u_i)^2 + r2_i < (s - u_j)^2 + r2_j for s > x (u_i > u_j)
                x = 0.5 * (u[i] + u[j]) + (r2[i] - r2[j]) / (2.0 * (u[i] - u[j]))
                if x <= start[m - 1]:  # j is never the lowest: drop it
                    m -= 1
                    x = -np.inf
                else:
                    break
            stack[m] = i
            start[m] = x
            m += 1
        return stack[:m], start[:m]

    return run


_ENVELOPE = {}


def _knn_candidates(tree, x, delta: float, k0: int = 16):
    """The generators that can be nearest to some point within ``delta`` / 2 of each of the points ``x`` (P, 3): every
    generator within d_1(x_j) + delta of x_j, d_1 the distance to the nearest one (d_1 is 1-Lipschitz along the path, so
    a generator nearest to a point y with |y - x_j| <= delta / 2 is within d_1(y) + delta / 2 <= d_1(x_j) + delta of x_j).
    Returns (point index, generator index) of every pair, and d_1 and the nearest generator of every point."""
    n = tree.n
    k = min(k0, n)
    while True:
        d, i = tree.query(x, k=k, workers=1)
        d, i = d.reshape(len(x), -1), i.reshape(len(x), -1)
        keep = d <= d[:, :1] + delta
        if k == n or not keep[:, -1].any():  # the k-th neighbour is beyond the bound: complete
            break
        k = min(2 * k, n)
    row = np.broadcast_to(np.arange(len(x))[:, None], keep.shape)[keep]
    return row, i[keep], d[:, 0], i[:, 0]


def _voronoi_envelope(pts, tree, gpos, box, size):
    """The exact Voronoi pieces of the dual segments of straight paths ``pts`` (n, N, 3) [cMpc] (``straight_paths``).

    Along x(s) = x_0 + s t the squared distance to a generator image p is (s - u)^2 + r^2, with u = t.(p - x_0) and
    r^2 = |p - x_0 - u t|^2, so the nearest generator at every s is the lowest of the straight lines -2 u s + u^2 + r^2:
    their lower envelope (``_envelope_kernel``, O(K log K) for K candidates) gives the cells crossed and the exact
    crossings s* = (u_i + u_j) / 2 + (r_i^2 - r_j^2) / (2 (u_i - u_j)). The candidates are the generator images within
    d_1 + delta of the points (``_knn_candidates``, delta the largest point spacing), each as the minimum image relative
    to the point that found it, so a path may cross the box faces. The crossings are then cut at the boundaries of the
    dual segments (the midpoints between points): a piece per cell and segment, its length and its centroid's offset
    from the point. Returns (SegmentComponents, the largest d_1 / size of the points' nearest cells (0 without
    ``size``), the number of candidate images per path (n,))."""
    run = _ENVELOPE.setdefault("numba", _envelope_kernel())
    n_sk, n_p = pts.shape[:2]
    cells, lens, offs, counts, worst, n_cand = [], [], [], [], 0.0, np.zeros(n_sk, np.int64)
    for k in range(n_sk):
        x = pts[k]
        x0 = x[0]
        chord = x[-1] - x0
        t = chord / np.linalg.norm(chord)
        s = (x - x0) @ t  # path coordinate of the points [cMpc]
        delta = float(np.max(np.diff(s))) if n_p > 1 else 0.0
        row, gi, d1, i1 = _knn_candidates(tree, wrap_positions(x, box), delta)
        if size is not None:
            worst = max(worst, float(np.max(d1 / size[i1])))
        p = x[row] + _min_image_np(gpos[gi] - x[row], box)  # the image near the point that found it
        if box is not None:  # one entry per (generator, image)
            shift = np.round((p - gpos[gi]) / box).astype(np.int64) + 512
            key = (gi.astype(np.int64) << 30) | (shift[:, 0] << 20) | (shift[:, 1] << 10) | shift[:, 2]
        else:
            key = gi.astype(np.int64)
        _, first = np.unique(key, return_index=True)
        gi, p = gi[first], p[first]
        n_cand[k] = gi.size
        rel = p - x0
        u = rel @ t
        r2 = np.sum((rel - u[:, None] * t) ** 2, axis=-1)
        order = np.lexsort((gi, r2, u))
        win, start = run(u[order], r2[order])
        lo = np.maximum(np.concatenate([[-np.inf], start[1:]]), s[0])
        hi = np.minimum(np.concatenate([start[1:], [np.inf]]), s[-1])
        ok = hi > lo
        cell_c, lo, hi = gi[order][win][ok], lo[ok], hi[ok]
        # cut the crossings at the dual-segment boundaries b_0 = s_0, b_i = (s_{i-1} + s_i) / 2, b_N = s_{N-1}
        b = np.concatenate([[s[0]], 0.5 * (s[1:] + s[:-1]), [s[-1]]])
        i0 = np.clip(np.searchsorted(b, lo, side="right") - 1, 0, n_p - 1)
        i1_ = np.clip(np.searchsorted(b, hi, side="left") - 1, 0, n_p - 1)
        rep = i1_ - i0 + 1
        c_idx = np.repeat(np.arange(cell_c.size), rep)
        seg = np.repeat(i0, rep) + (np.arange(c_idx.size) - np.repeat(np.cumsum(rep) - rep, rep))
        a_ = np.maximum(lo[c_idx], b[seg])
        b_ = np.minimum(hi[c_idx], b[seg + 1])
        keep = b_ > a_
        seg, c_idx, a_, b_ = seg[keep], c_idx[keep], a_[keep], b_[keep]
        cells.append((seg, cell_c[c_idx]))
        lens.append(b_ - a_)
        offs.append(0.5 * (a_ + b_) - s[seg])
        counts.append(np.bincount(seg, minlength=n_p))
    K = max(1, max(int(c.max()) for c in counts))
    cell = -np.ones((n_sk, n_p, K), np.int64)
    length, offset = np.zeros((n_sk, n_p, K)), np.zeros((n_sk, n_p, K))
    for k, ((seg, cc), ln, of, cnt) in enumerate(zip(cells, lens, offs, counts)):
        slot = np.arange(seg.size) - (np.cumsum(cnt) - cnt)[seg]  # the pieces come in order of s, so of segment
        cell[k, seg, slot], length[k, seg, slot], offset[k, seg, slot] = cc, ln, of
    return SegmentComponents(cell, length, offset, 1.0, "voronoi", {}), worst, n_cand


def grid_segments(points, shape, box: float, *, chunk: int = 1_000_000) -> SegmentComponents:
    """The Voronoi pieces of the dual segments of ``points`` for a periodic regular grid of cells (fast path).

    The cells are those of ``cells.cells_on_grid``: generators at the nodes x_j = j box / N_j (C order), so the cell
    of node j is the cube |x - x_j| <= box / (2 N) per axis, and a position belongs to the node round(x N / box) mod N.
    Every straight half-segment A + t (B - A) crosses the planes halfway between nodes at exact parameters t; the
    parts between consecutive crossings take the cell of their midpoint, and the parts of one cell in one dual
    segment are merged as in ``voronoi_segments``. Exact (no sub-sampling, no k-d tree, no selection).

    Parameters
    ----------
    points : (N_sk, N, 3) sampling points [cMpc] (unwrapped along the path).
    shape : (N_x, N_y, N_z) grid shape; box : periodic box side [cMpc].
    chunk : half-segments per vectorised block (memory).

    Returns
    -------
    SegmentComponents (weight 1, kind "voronoi"), cell = the flat C-order index of the grid cell.
    """
    pts = np.asarray(points, dtype=np.float64)
    n_sk, n_p = pts.shape[:2]
    A, Bl, Bh = _halves(pts)
    P = A.shape[0]
    N = np.asarray(shape, dtype=np.int64)
    dx = float(box) / N.astype(np.float64)
    M = int(np.prod(N))
    rows, labs, lens, moms = [], [], [], []
    for B, sign in ((Bl, -1.0), (Bh, 1.0)):
        D = B - A
        Lh = np.linalg.norm(D, axis=-1)
        for c0 in range(0, P, chunk):
            sl = slice(c0, c0 + chunk)
            u0, du = A[sl] / dx, D[sl] / dx  # positions in units of the spacing
            f0 = np.floor(u0 + 0.5)  # the node of the start
            nmax = np.abs(np.floor(u0 + du + 0.5) - f0).max(axis=0).astype(np.int64)  # crossings per axis (at most)
            ts = [np.zeros((u0.shape[0], 1)), np.ones((u0.shape[0], 1))]
            for i in range(3):
                if nmax[i] == 0:
                    continue
                m = np.arange(nmax[i])[None, :]
                s_i = np.sign(du[:, i : i + 1])
                face = f0[:, i : i + 1] + s_i * (0.5 + m)  # the planes ahead of the start
                with np.errstate(divide="ignore", invalid="ignore"):
                    t = (face - u0[:, i : i + 1]) / du[:, i : i + 1]
                ts.append(np.where((s_i != 0) & (t > 0) & (t < 1), t, 1.0))
            T = np.sort(np.concatenate(ts, axis=1), axis=1)
            lo, hi = T[:, :-1], T[:, 1:]
            mid = 0.5 * (lo + hi)
            idx = np.mod(np.floor(u0[:, None, :] + mid[..., None] * du[:, None, :] + 0.5).astype(np.int64), N)
            lab = (idx[..., 0] * N[1] + idx[..., 1]) * N[2] + idx[..., 2]
            lh = Lh[sl][:, None]
            rows.append(np.broadcast_to(np.arange(c0, c0 + u0.shape[0])[:, None], lab.shape).ravel())
            labs.append(lab.ravel())
            lens.append(((hi - lo) * lh).ravel())
            moms.append((sign * 0.5 * (hi**2 - lo**2) * lh**2).ravel())
    return _merge_pieces(rows, labs, lens, moms, P, M, (n_sk, n_p), dict(grid=tuple(int(n) for n in N)))


@partial(jax.jit, static_argnames=("S", "k_chunk"))
def _voronoi_chunk(A, Bl, Bh, L_half, gpos, box, S: int, k_chunk: int):
    """Voronoi pieces of the dual segments of one chunk of C consecutive points of one path (static shapes).

    1. Candidates: the ``k_chunk`` generators nearest to the chunk centre c (``lax.top_k``).
    2. Labels of the S + 1 sub-points of each half-segment (argmin over the candidates), exact crossings
       (``_crossing``), and the merge of the pieces of equal cell (sort + segment sum).

    Completeness: on a half-segment of length <= L_i the nearest generator of any sub-point is within
    rho_i = 2 L_i + d1_i of x_i (d1_i: distance from x_i to its nearest generator). The candidates hold
    every such generator if the k_chunk-th distance from c exceeds max_i (rho_i + |x_i - c|); the flag
    is returned and the caller doubles ``k_chunk`` until it holds (labels = exact nearest generators).

    Returns (cell (C, 4S) int, length (C, 4S), moment (C, 4S), complete).
    """
    mi = lambda d: _min_image(d, box)
    c = jnp.mean(A, axis=0)
    neg, cand = jax.lax.top_k(-jnp.sum(mi(gpos - c) ** 2, axis=-1), k_chunk)
    cut = jnp.sqrt(-neg[-1])
    gimg = A[:, None, :] + mi(gpos[cand][None, :, :] - A[:, None, :])  # (C, K, 3) images near x_i
    d1 = jnp.sqrt(jnp.min(jnp.sum((gimg - A[:, None, :]) ** 2, axis=-1), axis=-1))
    rho = 2.0 * jnp.max(L_half, axis=-1) + d1
    complete = cut > jnp.max(rho + jnp.linalg.norm(A - c, axis=-1))
    t = jnp.linspace(0.0, 1.0, S + 1)
    t0, t1 = t[:-1], t[1:]
    cells, lens, moms = [], [], []
    for h, (B, sign) in enumerate(((Bl, -1.0), (Bh, 1.0))):
        D = B - A
        sub = A[:, None, :] + t[None, :, None] * D[:, None, :]  # (C, S+1, 3)
        lab = jax.vmap(lambda sb, g: jnp.argmin(jnp.sum((sb[:, None, :] - g[None]) ** 2, -1), axis=-1))(sub, gimg)
        la, lb = lab[:, :-1], lab[:, 1:]
        ga = jnp.take_along_axis(gimg, la[..., None], axis=1)
        gb = jnp.take_along_axis(gimg, lb[..., None], axis=1)
        tc = _crossing(A, D, ga, gb, t0, t1)
        Lh = L_half[:, h][:, None]
        for lab_, lo, hi in ((la, t0[None, :], tc), (lb, tc, t1[None, :])):
            cells.append(cand[lab_])
            lens.append((hi - lo) * Lh)
            moms.append(sign * 0.5 * (hi**2 - lo**2) * Lh**2)
    cell, ln, mo = (jnp.concatenate(x, axis=1) for x in (cells, lens, moms))  # (C, 4S)
    M = gpos.shape[0]
    key = jnp.where(ln > 0, cell, M)  # zero-length pieces: sentinel
    order = jnp.argsort(key, axis=1)
    key, ln, mo = (jnp.take_along_axis(x, order, axis=1) for x in (key, ln, mo))
    new = jnp.concatenate([jnp.ones_like(key[:, :1], bool), key[:, 1:] != key[:, :-1]], axis=1)
    grp = jnp.cumsum(new, axis=1) - 1  # group of every piece
    seg_sum = jax.vmap(lambda g, x: jnp.zeros(x.shape, x.dtype).at[g].add(x))
    L_, M_ = seg_sum(grp, ln), seg_sum(grp, mo)
    lab_g = jax.vmap(lambda g, k: jnp.full(k.shape, M, k.dtype).at[g].set(k))(grp, key)
    cell_out = jnp.where((lab_g < M) & (L_ > 0), lab_g, -1)
    return cell_out, jnp.where(cell_out >= 0, L_, 0.0), jnp.where(cell_out >= 0, M_, 0.0), complete


def _voronoi_segments_jax(A, Bl, Bh, L_half, gpos, box, S: int, n_sk: int, chunk: int, k_chunk: int):
    """``voronoi_segments`` with JAX labels and merging: (cell, length, offset) (P, 4S) and diagnostics.

    Each path is cut into chunks of ``chunk`` consecutive points (padded by repeating its last point), so a
    chunk stays local; the chunks run through ``lax.map``. ``k_chunk`` is doubled (and the chunks recomputed)
    until every candidate set is provably complete, so the labels are those of an exact nearest-generator
    search. Only the trimming of the empty slots (``SegmentComponents.compact``) runs on the host, because
    its output size depends on the data.
    """
    P, M = A.shape[0], gpos.shape[0]
    n_p = P // n_sk
    nch = -(-n_p // chunk)
    pad = nch * chunk - n_p

    def per_path(x):
        x = x.reshape((n_sk, n_p) + x.shape[1:])
        if pad:
            x = np.concatenate([x, np.repeat(x[:, -1:], pad, axis=1)], axis=1)
        return jnp.asarray(x.reshape((n_sk * nch, chunk) + x.shape[2:]))

    Aj, Blj, Bhj, Lj = (per_path(x) for x in (A, Bl, Bh, L_half))
    g = jnp.asarray(gpos)
    bx = None if box is None else float(box)
    k = min(k_chunk, M)
    while True:
        cell, ln, mo, ok = jax.lax.map(lambda xs: _voronoi_chunk(*xs, g, bx, S, k), (Aj, Blj, Bhj, Lj))
        if bool(jnp.all(ok)) or k == M:
            break
        k = min(2 * k, M)
    back = lambda x: np.asarray(x).reshape(n_sk, nch * chunk, -1)[:, :n_p].reshape(P, -1)
    cell, ln, mo = back(cell), back(ln), back(mo)
    off = np.where(ln > 0, mo / np.where(ln > 0, ln, 1.0), 0.0)
    return cell.astype(np.int64), ln, off, dict(k_chunk=k)


# --------------------------------------------------------------------------- top hat
def _tophat_half(A, B, G, h, valid):
    """Exact overlap of the straight half A -> B with spheres (G, h): (length, moment) (P, K), in float64 NumPy (host
    geometry, whatever the precision policy)."""
    D = B - A
    L2 = np.sum(D * D, axis=-1)[:, None]
    AG = A[:, None, :] - G
    bq = np.einsum("pkc,pc->pk", AG, D)
    cq = np.einsum("pkc,pkc->pk", AG, AG) - h * h
    disc = bq * bq - L2 * cq
    good = valid & (disc > 0) & (L2 > 0)
    sq = np.sqrt(np.where(good, disc, 0.0))
    L2s = np.where(L2 > 0, L2, 1.0)
    t1 = np.clip((-bq - sq) / L2s, 0.0, 1.0)
    t2 = np.clip((-bq + sq) / L2s, 0.0, 1.0)
    dt = np.where(good, np.maximum(t2 - t1, 0.0), 0.0)
    length = dt * np.sqrt(L2s)
    moment = np.where(good & (t2 > t1), 0.5 * (t2**2 - t1**2) * L2s, 0.0)
    return length, moment


def _tophat_pairs(A, r_half, gpos, size, box):
    """Candidate (point, cell) pairs with |x_i - g_c| <= h_c + r_i, as a padded (P, K) index array."""
    tree = cKDTree(wrap_positions(A, box), boxsize=box)
    lists = tree.query_ball_point(gpos, r=size + r_half.max(), workers=-1)
    cells = np.repeat(np.arange(len(gpos)), [len(l) for l in lists])
    pts = np.concatenate([np.asarray(l, dtype=np.int64) for l in lists]) if len(cells) else np.zeros(0, np.int64)
    order = np.argsort(pts, kind="stable")
    pts, cells = pts[order], cells[order]
    counts = np.bincount(pts, minlength=A.shape[0])
    K = max(int(counts.max()) if counts.size else 1, 1)
    start = np.concatenate([[0], np.cumsum(counts)[:-1]])
    slot = np.arange(pts.size) - start[pts]
    idx = -np.ones((A.shape[0], K), np.int64)
    idx[pts, slot] = cells
    return idx


def tophat_segments(points, gas_positions, gas_size, box: float | None, *, chunk: int = 4096) -> SegmentComponents:
    """fake_spectra's top-hat spheres (radius ``gas_size`` = V^(1/3)) on the dual segments.

    Returns SegmentComponents with weight 3 / (4 pi): a sphere of radius V^(1/3) and density
    (3/4pi) n holds exactly the cell's atoms.
    """
    pts = np.asarray(points, dtype=np.float64)
    n_sk, n_p = pts.shape[:2]
    A, Bl, Bh = _halves(pts)
    r = np.maximum(np.linalg.norm(Bl - A, axis=-1), np.linalg.norm(Bh - A, axis=-1))
    gpos, size = wrap_positions(gas_positions, box), np.asarray(gas_size, dtype=np.float64)
    idx = _tophat_pairs(A, r, gpos, size, box)
    valid = idx >= 0
    safe = np.maximum(idx, 0)
    G = A[:, None, :] + _min_image_np(gpos[safe] - A[:, None, :], box)
    H = size[safe]
    length = np.zeros(idx.shape)
    moment = np.zeros(idx.shape)
    for c0 in range(0, A.shape[0], chunk):
        sl = slice(c0, c0 + chunk)
        for B, sign in ((Bl, -1.0), (Bh, 1.0)):
            l_, m_ = _tophat_half(A[sl], B[sl], G[sl], H[sl], valid[sl])
            length[sl] += l_
            moment[sl] += sign * m_
    offset = np.where(length > 0, moment / np.where(length > 0, length, 1.0), 0.0)
    cell = np.where(length > 0, idx, -1)
    shp = (n_sk, n_p, idx.shape[1])
    return SegmentComponents(
        cell.reshape(shp), length.reshape(shp), offset.reshape(shp), TOPHAT_NORM, "tophat", dict(max_candidates=int(idx.shape[1]))
    ).compact()


CHORD_POINTS = 7  # the interior points of fake_spectra's chord quadrature: NGRID - 1 (singleabs.h L9, L102)


def tophat_chord_segments(
    points, gas_positions, gas_size, box: float | None, *, chord_weight: float | None = None
) -> SegmentComponents:
    """fake_spectra's chord quadrature of its top-hat spheres (radius ``gas_size`` = V^(1/3)) along straight paths.

    The chord of half-length z_r = sqrt(h^2 - dr^2) of every sphere touching the path (``absorption.cpp`` L235) is
    sampled at its interior points s_c + z_r (k / 4 - 1), k = 1..7 (``singleabs.h`` L100-L118; NGRID = 8), each one a
    piece at that position, of density (3 / 4 pi) n (L106) and length 2 z_r w. The chords come from the candidate pairs
    of ``core.crossings.straight_chords`` (index arithmetic on the paths' plane: no scan of every cell per path).

    chord_weight : the weight w of every point. None (default): 1/7, so that the chord keeps its whole column. 1/8:
        fake_spectra's (``deltav`` = 2 v_high / NGRID, L100, L116): the two end points, which have zero weight for an SPH
        kernel, are dropped (L97-L99), so a top hat integrates 7/8 of its column.

    Parameters
    ----------
    points : (N_sk, N, 3) sampling points of straight paths [cMpc]; a chord point belongs to the nearest one, with its
        offset (only where the absorption sits matters: the pieces are not cut at the dual segments).
    gas_positions (M, 3), gas_size (M,) [cMpc]; box : the periodic box (the image nearest to the middle of the path) or None.

    A path across a whole periodic box keeps the chord points beyond its ends (s < 0, s > L: the spectrum wraps them,
    ``velocity_period``); a shorter path drops them. Returns SegmentComponents with weight 3 / (4 pi), kind "tophat_chords".
    """
    from skewerjax.core import crossings as cx

    if chord_weight is not None and not chord_weight > 0:
        raise ValueError(f"chord_weight must be None or > 0, got {chord_weight!r}.")
    w = 1.0 / cx.CHORD_POINTS if chord_weight is None else float(chord_weight)
    pts = np.asarray(points, dtype=np.float64)
    n_sk, n_p = pts.shape[:2]
    if not np.all(cx.straight_paths_mask(pts)):
        raise ValueError("deposition='tophat_chords' needs straight paths (fake_spectra's sightlines).")
    paths = cx.paths_from_points(pts, box)
    ch = cx.straight_chords(paths, np.asarray(gas_positions, dtype=np.float64), np.asarray(gas_size, dtype=np.float64))
    idx, s, ln = ch.points(w)
    k_of = ch.path[idx]
    c = ch.cell[idx]
    L = paths.length[k_of]
    if box is None or not np.allclose(paths.length, box, rtol=1e-9):
        inside = (s >= 0.0) & (s <= L)
        k_of, c, s, ln = k_of[inside], c[inside], s[inside], ln[inside]
    s_pt = path_geometry(pts)["s"]
    order = np.lexsort((s, k_of))
    k_of, c, s, ln = k_of[order], c[order], s[order], ln[order]
    i = np.empty(s.size, np.int64)
    for k in np.unique(k_of):
        sel = np.flatnonzero(k_of == k)
        i[sel] = np.clip(np.searchsorted(0.5 * (s_pt[k, 1:] + s_pt[k, :-1]), s[sel]), 0, n_p - 1)
    key = k_of * n_p + i
    cnt = np.bincount(key, minlength=n_sk * n_p)
    K = max(int(cnt.max()) if cnt.size else 1, 1)
    o2 = np.argsort(key, kind="stable")
    slot = np.arange(key.size) - np.concatenate([[0], np.cumsum(cnt)[:-1]])[key[o2]]
    cell = np.full((n_sk * n_p, K), -1, dtype=np.int64)
    length, offset = np.zeros(cell.shape), np.zeros(cell.shape)
    cell[key[o2], slot] = c[o2]
    length[key[o2], slot] = ln[o2]
    offset[key[o2], slot] = (s - s_pt[k_of, i])[o2]
    shp = (n_sk, n_p, K)
    return SegmentComponents(
        cell.reshape(shp), length.reshape(shp), offset.reshape(shp), TOPHAT_NORM, "tophat_chords", dict(chord_points=cx.CHORD_POINTS, chord_weight=w)
    )


# --------------------------------------------------------------------------- cell crossings (Voronoi-adapted segmentation)
@dataclass
class CellCrossings:
    """The path cut at the Voronoi cell boundaries: one interval per crossing of a cell (host arrays).

    Derived from a Voronoi ``SegmentComponents`` by joining the pieces of the same cell in consecutive dual
    segments, so the boundaries are the cell crossings only (the midpoints between sampling points are gone).
    It is a property of the geometry: the same for any spacing of the points that resolves the cells.

    Attributes
    ----------
    s_lo, s_hi : (N_sk, M) path coordinates [cMpc] of the ends of every crossing (padded: s_lo = s_hi = 0).
    cell : (N_sk, M) int, index into the cells (-1: padding).
    diagnostics : dict (``max_gap``: largest mismatch [cMpc] between consecutive ends, 0 for straight paths).
    """

    s_lo: np.ndarray
    s_hi: np.ndarray
    cell: np.ndarray
    diagnostics: dict

    @property
    def length(self) -> np.ndarray:
        """(N_sk, K) length of every crossing [cMpc] (0 in the empty slots)."""
        return np.where(self.cell >= 0, self.s_hi - self.s_lo, 0.0)

    @property
    def n_crossings(self) -> np.ndarray:
        """(N_sk,) number of cell crossings of every skewer."""
        return (self.cell >= 0).sum(axis=-1)


def cell_crossings(seg: SegmentComponents, points) -> CellCrossings:
    """``CellCrossings`` of a Voronoi deposition along ``points`` (N_sk, N, 3) [cMpc].

    Every piece p of the dual segment of point i is the interval s_i + offset_p -+ length_p / 2 of the path
    (a straight piece of one convex cell); consecutive intervals of the same cell are joined. Exact for
    straight paths; on a polyline, a cell that holds both arms of a bend but not its vertex would make two
    pieces of one point, and ``diagnostics["max_gap"]`` reports the mismatch.
    """
    if seg.kind != "voronoi":
        raise ValueError(f"cell crossings need a Voronoi deposition, got {seg.kind!r}.")
    g = path_geometry(np.asarray(points, dtype=np.float64))
    s_pt = np.asarray(g["s"])
    rows_lo, rows_hi, rows_c = [], [], []
    gap = 0.0
    for k in range(seg.cell.shape[0]):
        pi, ki = np.nonzero(seg.cell[k] >= 0)
        mid = s_pt[k][pi] + seg.offset[k][pi, ki]
        half = 0.5 * seg.length[k][pi, ki]
        c = seg.cell[k][pi, ki]
        order = np.lexsort((mid, pi))
        lo, hi, c = (mid - half)[order], (mid + half)[order], c[order]
        if lo.size > 1:
            gap = max(gap, float(np.max(np.abs(lo[1:] - hi[:-1]))))
        new = np.concatenate([[True], c[1:] != c[:-1]])
        start = np.nonzero(new)[0]
        end = np.concatenate([start[1:], [c.size]]) - 1
        rows_lo.append(lo[start])
        rows_hi.append(hi[end])
        rows_c.append(c[start])
    m = max(max(len(r) for r in rows_c), 1)
    pad = lambda rows, f: np.stack([np.concatenate([r, np.full(m - len(r), f)]) for r in rows])
    return CellCrossings(pad(rows_lo, 0.0), pad(rows_hi, 0.0), pad(rows_c, -1).astype(np.int64), dict(max_gap=gap))


# --------------------------------------------------------------------------- along-skewer fields
def segment_fields(seg: SegmentComponents, n, T, v_los) -> dict:
    """Column-weighted fields of every dual segment (diagnostics and plots).

    Parameters
    ----------
    n : (M,) per-cell density [cm^-3]; T : (M,) [K]; v_los : (N_sk, N, K) component LoS velocities.

    Returns
    -------
    dict of (N_sk, N): ``column_length`` (sum of weight n length; the segment average density is
    column / w), ``T`` and ``v_los`` (column-weighted means), ``n_components``.
    """
    live = seg.cell >= 0
    c = np.maximum(seg.cell, 0)
    col = np.where(live, seg.weight * np.asarray(n)[c] * seg.length, 0.0)
    tot = col.sum(-1)
    safe = np.where(tot > 0, tot, 1.0)
    return dict(
        column_length=tot,
        T=np.where(tot > 0, (col * np.asarray(T)[c]).sum(-1) / safe, 0.0),
        v_los=np.where(tot > 0, (col * v_los).sum(-1) / safe, 0.0),
        n_components=live.sum(-1),
    )
