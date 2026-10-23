"""Deposition (layer 5): which gas lies on the dual segment of every sampling point, and where.

The sampling points are those of the bundle (any spacing, straight or curved). Point i stands for
its *dual segment* m_{i-1/2} -> x_i -> m_{i+1/2} (``core.skewers.path_geometry``), of comoving length
w_i. A deposition answers one question: **which gas lies on that segment, and where?** It returns
for every point up to K *components*: a cell index, the comoving length of the cell's piece of the
segment, the signed path offset of that piece's centroid from x_i, and a density weight.

* ``voronoi_segments`` (SkewerJAX default): Arepo's own geometry. A point of space belongs to the
  cell of its nearest mesh-generating point (TNG ``Coordinates``), so along the segment the gas is
  piecewise constant and single valued. The pieces are found exactly: on a straight half-segment
  A + t (B - A) the squared distance to generator g is |A - g|^2 + 2 t (A - g).(B - A) + t^2 |B - A|^2,
  and the last term is common to all generators, so the nearest generator is the lower envelope of
  *straight lines* in t. Every half-segment is sampled at ``n_sub`` sub-points labelled by a
  periodic k-d tree, and every change of label is located at the exact crossing of the two
  lines. Weight 1.
* ``tophat_segments`` (fake_spectra's geometry): each cell is a sphere of radius h = V^(1/3) and
  uniform density (3 / 4 pi) n (``absorption.cpp:tophat_kern_frac`` L112-L118). The overlap of
  every sphere with every half-segment is solved exactly (a quadratic in t). Spheres overlap, so a
  segment has several components with *different* T and v: the representation is multi-valued.
* ``tophat_point_values`` (the previous "points" deposition): one value per point, n = sum of the
  sphere densities containing x_i, and n-weighted T and v. It is a point sample (no line element:
  the column is conserved only if the spacing resolves the cells) and it *merges* the overlapping
  spheres into one profile per point, which is what broke Si III (narrow lines are displaced and
  reshaped when components with different v are averaged). Kept only for comparison.

Every deposition conserves the column by construction (sum of component lengths x density), for
any spacing, because the lengths are exact segment integrals; only *where* inside the segment the
absorption sits is approximated (at the piece centroid). That approximation is what the sampling
criterion of ``skewerjax.simulations.absorbers.sampling_check`` monitors.

Periodicity: generator positions are always taken as the minimum image relative to each point,
x_i + mi(g - x_i), mi(d) = d - L round(d / L), so paths may cross the box faces any number of times.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Optional

import numpy as np
import jax
import jax.numpy as jnp
from scipy.spatial import cKDTree

from skewerjax.core.skewers import path_geometry

TOPHAT_NORM = 3.0 / (4.0 * np.pi)          # density inside a sphere of radius V^(1/3), per n_cell


@dataclass
class SegmentComponents:
    """Components of the dual segment of every sampling point (host arrays).

    Attributes
    ----------
    cell : (N_sk, N, K) int, index into the GasCells used (-1: empty slot).
    length : (N_sk, N, K) comoving length of the piece [cMpc].
    offset : (N_sk, N, K) signed path offset of the piece centroid from x_i [cMpc].
    weight : float, density per n_cell inside the piece (1 Voronoi, 3/4pi top hat).
    kind : "voronoi" or "tophat".
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
        return (self.cell >= 0).sum(axis=-1)

    def compact(self) -> "SegmentComponents":
        """Drop the trailing empty slots (K -> the largest number of components of any point)."""
        order = np.argsort(-(self.length * (self.cell >= 0)), axis=-1, kind="stable")
        take = lambda a: np.take_along_axis(a, order, axis=-1)
        k = max(int(self.n_components.max()), 1)
        return SegmentComponents(take(self.cell)[..., :k], take(self.length)[..., :k], take(self.offset)[..., :k],
                                 self.weight, self.kind, self.diagnostics)

    def column_lengths(self, n_cells) -> np.ndarray:
        """(N_sk, N) sum over components of weight x n_cell x length [n units x cMpc]."""
        n = np.asarray(n_cells, dtype=np.float64)
        return (self.weight * np.where(self.cell >= 0, n[np.maximum(self.cell, 0)], 0.0) * self.length).sum(axis=-1)


# --------------------------------------------------------------------------- helpers
def _min_image(d, box):
    return d if box is None else d - box * jnp.round(d / box)


def _wrap(x, box):
    if box is None:
        return np.asarray(x, dtype=np.float64)
    y = np.mod(np.asarray(x, dtype=np.float64), box)
    return np.where(y >= box, 0.0, y)


def _halves(points):
    """Half-segments of every point: (A, B_back, B_fwd), each (P, 3), and the geometry dict."""
    g = path_geometry(jnp.asarray(points, dtype=jnp.float64))
    A = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return A, np.asarray(g["mid_lo"]).reshape(-1, 3), np.asarray(g["mid_hi"]).reshape(-1, 3), g


# --------------------------------------------------------------------------- Voronoi
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


def voronoi_segments(points, gas_positions, box: Optional[float], *, gas_size=None, n_sub: Optional[int] = None,
                     max_sub: int = 4096, backend: str = "numpy", chunk: int = 64,
                     k_chunk: int = 512) -> SegmentComponents:
    """Nearest-generating-point (Voronoi) deposition on the dual segments of ``points``.

    Every half-segment is sampled at ``n_sub`` + 1 sub-points; each sub-point is labelled with its
    nearest generator (periodic k-d tree over ``gas_positions``), and wherever two consecutive
    labels differ the boundary between the two cells is located exactly (``_crossing``). The
    result is exact as long as no cell is crossed in a chord shorter than a sub-interval; with
    ``gas_size`` and ``n_sub=None``, ``n_sub`` makes every sub-interval shorter than 1/16 of
    the smallest cell of ``gas_positions`` (capped at ``max_sub``); a grazed cell can then be
    missed only over less than that length.

    Parameters
    ----------
    points : (N_sk, N, 3) sampling points [cMpc] (unwrapped along the path).
    gas_positions : (M, 3) mesh-generating points [cMpc] in [0, box).
    box : periodic box [cMpc] or None.
    gas_size : (M,) cell sizes V^(1/3) [cMpc] (to choose ``n_sub``).
    backend : "numpy" (default: SciPy k-d tree labels, NumPy merging) or "jax" (the same algorithm in
        JAX with static shapes, ``_voronoi_segments_jax``; identical cells, lengths equal to round-off).
    chunk, k_chunk : "jax" only: consecutive points of a path per chunk and candidate generators per chunk
        (doubled automatically until the candidate sets are provably complete).

    Returns
    -------
    SegmentComponents (weight 1); ``diagnostics`` holds ``n_sub`` and the sub-interval relative to
    the smallest cell (and, for "jax", the candidate sizes used).
    """
    if backend not in ("numpy", "jax"):
        raise ValueError(f"backend must be 'numpy' or 'jax', got {backend!r}.")
    pts = np.asarray(points, dtype=np.float64)
    n_sk, n_p = pts.shape[:2]
    A, Bl, Bh, _ = _halves(pts)
    P = A.shape[0]
    gpos = _wrap(gas_positions, box)
    size = None if gas_size is None else np.asarray(gas_size, dtype=np.float64)
    L_half = np.stack([np.linalg.norm(Bl - A, axis=-1), np.linalg.norm(Bh - A, axis=-1)], axis=1)     # (P, 2)
    if n_sub is not None:
        S = int(n_sub)
    elif size is None:
        S = 8
    else:                                   # sub-interval <= 1/16 of the smallest cell near the paths
        S = int(np.clip(np.ceil(16.0 * L_half.max() / max(size.min(), 1e-12)), 8, max_sub))
    if backend == "jax":
        cell, length, offset, kd = _voronoi_segments_jax(A, Bl, Bh, L_half, gpos, box, S, n_sk, chunk, k_chunk)
        diag = dict(n_sub=S, sub_over_min_cell=float(L_half.max() / S / size.min()) if size is not None else None, **kd)
        shp = (n_sk, n_p, cell.shape[-1])
        return SegmentComponents(cell.reshape(shp), length.reshape(shp), offset.reshape(shp), 1.0, "voronoi", diag).compact()
    tree = cKDTree(gpos, boxsize=box) if box is not None else cKDTree(gpos)
    t = np.linspace(0.0, 1.0, S + 1)
    sub = np.stack([A[:, None, :] + t[None, :, None] * (B - A)[:, None, :] for B in (Bl, Bh)], axis=1)   # (P, 2, S+1, 3)
    lab = tree.query(_wrap(sub.reshape(-1, 3), box), k=1, workers=-1)[1].reshape(P, 2, S + 1)
    rows, labs, lens, moms = [], [], [], []
    t0, t1 = jnp.asarray(t[:-1]), jnp.asarray(t[1:])
    for h, (B, sign) in enumerate(((Bl, -1.0), (Bh, 1.0))):
        D = B - A
        la, lb = lab[:, h, :-1], lab[:, h, 1:]
        ga = A[:, None, :] + np.asarray(_min_image(jnp.asarray(gpos[la]) - A[:, None, :], box))
        gb = A[:, None, :] + np.asarray(_min_image(jnp.asarray(gpos[lb]) - A[:, None, :], box))
        tc = np.asarray(_crossing(jnp.asarray(A), jnp.asarray(D), jnp.asarray(ga), jnp.asarray(gb), t0, t1))
        Lh = L_half[:, h][:, None]
        T0, T1 = t[None, :-1], t[None, 1:]
        for lab_, lo, hi in ((la, T0, tc), (lb, tc, T1)):
            rows.append(np.broadcast_to(np.arange(P)[:, None], lab_.shape).ravel())
            labs.append(lab_.ravel())
            lens.append(((hi - lo) * Lh).ravel())
            moms.append((sign * 0.5 * (hi ** 2 - lo ** 2) * Lh ** 2).ravel())
    rows, labs, lens, moms = (np.concatenate(x) for x in (rows, labs, lens, moms))
    keep = lens > 0
    key = rows[keep].astype(np.int64) * len(gpos) + labs[keep]
    ukey, inv = np.unique(key, return_inverse=True)
    L_ = np.bincount(inv, weights=lens[keep]); M_ = np.bincount(inv, weights=moms[keep])
    prow, plab = ukey // len(gpos), ukey % len(gpos)
    counts = np.bincount(prow, minlength=P)
    K = max(int(counts.max()), 1)
    start = np.concatenate([[0], np.cumsum(counts)[:-1]])
    slot = np.arange(prow.size) - start[prow]
    cell, length, offset = -np.ones((P, K), np.int64), np.zeros((P, K)), np.zeros((P, K))
    cell[prow, slot] = plab
    length[prow, slot] = L_
    offset[prow, slot] = np.where(L_ > 0, M_ / np.where(L_ > 0, L_, 1.0), 0.0)
    diag = dict(n_sub=S, sub_over_min_cell=float(L_half.max() / S / size.min()) if size is not None else None)
    shp = (n_sk, n_p, K)
    return SegmentComponents(cell.reshape(shp), length.reshape(shp), offset.reshape(shp), 1.0, "voronoi", diag).compact()


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
    mi = lambda d: d if box is None else d - box * jnp.round(d / box)
    c = jnp.mean(A, axis=0)
    neg, cand = jax.lax.top_k(-jnp.sum(mi(gpos - c) ** 2, axis=-1), k_chunk)
    cut = jnp.sqrt(-neg[-1])
    gimg = A[:, None, :] + mi(gpos[cand][None, :, :] - A[:, None, :])           # (C, K, 3) images near x_i
    d1 = jnp.sqrt(jnp.min(jnp.sum((gimg - A[:, None, :]) ** 2, axis=-1), axis=-1))
    rho = 2.0 * jnp.max(L_half, axis=-1) + d1
    complete = cut > jnp.max(rho + jnp.linalg.norm(A - c, axis=-1))
    t = jnp.linspace(0.0, 1.0, S + 1)
    t0, t1 = t[:-1], t[1:]
    cells, lens, moms = [], [], []
    for h, (B, sign) in enumerate(((Bl, -1.0), (Bh, 1.0))):
        D = B - A
        sub = A[:, None, :] + t[None, :, None] * D[:, None, :]                   # (C, S+1, 3)
        lab = jax.vmap(lambda sb, g: jnp.argmin(jnp.sum((sb[:, None, :] - g[None]) ** 2, -1), axis=-1))(sub, gimg)
        la, lb = lab[:, :-1], lab[:, 1:]
        ga = jnp.take_along_axis(gimg, la[..., None], axis=1)
        gb = jnp.take_along_axis(gimg, lb[..., None], axis=1)
        tc = _crossing(A, D, ga, gb, t0, t1)
        Lh = L_half[:, h][:, None]
        for lab_, lo, hi in ((la, t0[None, :], tc), (lb, tc, t1[None, :])):
            cells.append(cand[lab_])
            lens.append((hi - lo) * Lh)
            moms.append(sign * 0.5 * (hi ** 2 - lo ** 2) * Lh ** 2)
    cell, ln, mo = (jnp.concatenate(x, axis=1) for x in (cells, lens, moms))    # (C, 4S)
    M = gpos.shape[0]
    key = jnp.where(ln > 0, cell, M)                                             # zero-length pieces: sentinel
    order = jnp.argsort(key, axis=1)
    key, ln, mo = (jnp.take_along_axis(x, order, axis=1) for x in (key, ln, mo))
    new = jnp.concatenate([jnp.ones_like(key[:, :1], bool), key[:, 1:] != key[:, :-1]], axis=1)
    grp = jnp.cumsum(new, axis=1) - 1                                            # group of every piece
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
@jax.jit
def _tophat_half(A, B, G, h, valid):
    """Exact overlap of the straight half A -> B with spheres (G, h): (length, moment) (P, K)."""
    D = B - A
    L2 = jnp.sum(D * D, axis=-1)[:, None]
    AG = A[:, None, :] - G
    bq = jnp.sum(AG * D[:, None, :], axis=-1)
    cq = jnp.sum(AG * AG, axis=-1) - h * h
    disc = bq * bq - L2 * cq
    good = valid & (disc > 0) & (L2 > 0)
    sq = jnp.sqrt(jnp.where(good, disc, 0.0))
    L2s = jnp.where(L2 > 0, L2, 1.0)
    t1 = jnp.clip((-bq - sq) / L2s, 0.0, 1.0)
    t2 = jnp.clip((-bq + sq) / L2s, 0.0, 1.0)
    dt = jnp.where(good, jnp.maximum(t2 - t1, 0.0), 0.0)
    length = dt * jnp.sqrt(L2s)
    moment = jnp.where(good & (t2 > t1), 0.5 * (t2 ** 2 - t1 ** 2) * L2s, 0.0)
    return length, moment


def _tophat_pairs(A, r_half, gpos, size, box):
    """Candidate (point, cell) pairs with |x_i - g_c| <= h_c + r_i, as a padded (P, K) index array."""
    tree = cKDTree(_wrap(A, box), boxsize=box) if box is not None else cKDTree(_wrap(A, box))
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


def tophat_segments(points, gas_positions, gas_size, box: Optional[float], *, chunk: int = 4096) -> SegmentComponents:
    """fake_spectra's top-hat spheres (radius ``gas_size`` = V^(1/3)) on the dual segments.

    Returns SegmentComponents with weight 3 / (4 pi): a sphere of radius V^(1/3) and density
    (3/4pi) n holds exactly the cell's atoms.
    """
    pts = np.asarray(points, dtype=np.float64)
    n_sk, n_p = pts.shape[:2]
    A, Bl, Bh, _ = _halves(pts)
    r = np.maximum(np.linalg.norm(Bl - A, axis=-1), np.linalg.norm(Bh - A, axis=-1))
    gpos, size = _wrap(gas_positions, box), np.asarray(gas_size, dtype=np.float64)
    idx = _tophat_pairs(A, r, gpos, size, box)
    valid = idx >= 0
    safe = np.maximum(idx, 0)
    G = A[:, None, :] + np.asarray(_min_image(jnp.asarray(gpos[safe]) - A[:, None, :], box))
    H = size[safe]
    length = np.zeros(idx.shape)
    moment = np.zeros(idx.shape)
    for c0 in range(0, A.shape[0], chunk):
        sl = slice(c0, c0 + chunk)
        for B, sign in ((Bl, -1.0), (Bh, 1.0)):
            l_, m_ = _tophat_half(jnp.asarray(A[sl]), jnp.asarray(B[sl]), jnp.asarray(G[sl]), jnp.asarray(H[sl]),
                                  jnp.asarray(valid[sl]))
            length[sl] += np.asarray(l_)
            moment[sl] += sign * np.asarray(m_)
    offset = np.where(length > 0, moment / np.where(length > 0, length, 1.0), 0.0)
    cell = np.where(length > 0, idx, -1)
    shp = (n_sk, n_p, idx.shape[1])
    return SegmentComponents(cell.reshape(shp), length.reshape(shp), offset.reshape(shp), TOPHAT_NORM, "tophat",
                             dict(max_candidates=int(idx.shape[1]))).compact()


def tophat_point_values(points, gas_positions, gas_size, box: Optional[float], n, T, v_los_of) -> dict:
    """The previous "points" deposition: point values of the overlapping spheres, merged.

    Parameters
    ----------
    n, T : (M,) per-cell ion density and temperature; v_los_of : callable(cell_idx, point_idx) ->
        line-of-sight velocity of those cells projected on the tangent at those points [km/s].

    Returns
    -------
    dict of (N_sk, N) arrays: ``n`` (sum of 3/4pi n_c over spheres containing x_i), ``T`` and
    ``v_los`` (n-weighted means; T = 1e4 K and v = 0 where empty).
    """
    pts = np.asarray(points, dtype=np.float64)
    A = pts.reshape(-1, 3)
    gpos, size = _wrap(gas_positions, box), np.asarray(gas_size, dtype=np.float64)
    idx = _tophat_pairs(A, np.zeros(A.shape[0]), gpos, size, box)
    valid = idx >= 0
    safe = np.maximum(idx, 0)
    d = np.asarray(_min_image(jnp.asarray(gpos[safe]) - A[:, None, :], box))
    inside = valid & (np.sum(d * d, axis=-1) <= size[safe] ** 2)
    w = np.where(inside, TOPHAT_NORM * np.asarray(n)[safe], 0.0)
    ntot = w.sum(axis=1)
    pidx = np.broadcast_to(np.arange(A.shape[0])[:, None], idx.shape)
    vv = np.where(inside, v_los_of(safe, pidx), 0.0)
    safe_n = np.where(ntot > 0, ntot, 1.0)
    T_m = np.where(ntot > 0, (w * np.asarray(T)[safe]).sum(1) / safe_n, 1e4)
    v_m = np.where(ntot > 0, (w * vv).sum(1) / safe_n, 0.0)
    shp = pts.shape[:2]
    return dict(n=ntot.reshape(shp), T=T_m.reshape(shp), v_los=v_m.reshape(shp))


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
    cell : (N_sk, M) int, index into the GasCells (-1: padding).
    diagnostics : dict (``max_gap``: largest mismatch [cMpc] between consecutive ends, 0 for straight paths).
    """
    s_lo: np.ndarray
    s_hi: np.ndarray
    cell: np.ndarray
    diagnostics: dict

    @property
    def length(self) -> np.ndarray:
        return np.where(self.cell >= 0, self.s_hi - self.s_lo, 0.0)

    @property
    def n_crossings(self) -> np.ndarray:
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
    g = path_geometry(jnp.asarray(points, dtype=jnp.float64))
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
        rows_lo.append(lo[start]); rows_hi.append(hi[end]); rows_c.append(c[start])
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
    dict of (N_sk, N): ``n_mean`` (segment average density = column / w), ``column_length``
    (sum of weight n length), ``T`` and ``v_los`` (column-weighted means), ``n_components``.
    """
    live = seg.cell >= 0
    c = np.maximum(seg.cell, 0)
    col = np.where(live, seg.weight * np.asarray(n)[c] * seg.length, 0.0)
    tot = col.sum(-1)
    safe = np.where(tot > 0, tot, 1.0)
    return dict(column_length=tot, T=np.where(tot > 0, (col * np.asarray(T)[c]).sum(-1) / safe, 0.0),
                v_los=np.where(tot > 0, (col * v_los).sum(-1) / safe, 0.0), n_components=live.sum(-1))
