"""Layer 5 — deposition of gas cells onto the sampling points of a ``SkewerBundle``.

The sampling points are those of the bundle (any spacing, straight or curved). Point i stands for
its *dual segment* m_{i-1/2} -> x_i -> m_{i+1/2} (``skewers.path_geometry``), of comoving length
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
criterion of ``skewerjax.optical_depth.absorbers.sampling_check`` monitors.

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

from skewerjax.skewers.skewers import path_geometry

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
                     max_sub: int = 4096) -> SegmentComponents:
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

    Returns
    -------
    SegmentComponents (weight 1); ``diagnostics`` holds ``n_sub`` and the sub-interval relative to
    the smallest cell.
    """
    pts = np.asarray(points, dtype=np.float64)
    n_sk, n_p = pts.shape[:2]
    A, Bl, Bh, _ = _halves(pts)
    P = A.shape[0]
    gpos = _wrap(gas_positions, box)
    tree = cKDTree(gpos, boxsize=box) if box is not None else cKDTree(gpos)
    size = None if gas_size is None else np.asarray(gas_size, dtype=np.float64)
    L_half = np.stack([np.linalg.norm(Bl - A, axis=-1), np.linalg.norm(Bh - A, axis=-1)], axis=1)     # (P, 2)
    if n_sub is not None:
        S = int(n_sub)
    elif size is None:
        S = 8
    else:                                   # sub-interval <= 1/16 of the smallest cell near the paths
        S = int(np.clip(np.ceil(16.0 * L_half.max() / max(size.min(), 1e-12)), 8, max_sub))
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


def voronoi_point_values(points, gas_positions, box: Optional[float]) -> SegmentComponents:
    """Point sampling of the Voronoi field: point i gets the cell of its nearest generator, with the
    whole line element w_i and no offset. This is the classical "field at points + trapezoid"
    quadrature: exact only when every cell is much longer than the spacing (it does not conserve
    the column of under-sampled cells). Kept to quantify why segment integrals are needed."""
    pts = np.asarray(points, dtype=np.float64)
    g = path_geometry(jnp.asarray(pts))
    gpos = _wrap(gas_positions, box)
    tree = cKDTree(gpos, boxsize=box) if box is not None else cKDTree(gpos)
    _, idx = tree.query(_wrap(pts.reshape(-1, 3), box), k=1, workers=-1)
    shp = pts.shape[:2] + (1,)
    return SegmentComponents(idx.reshape(shp), np.asarray(g["line_elements"]).reshape(shp), np.zeros(shp), 1.0,
                             "voronoi_points", {})
