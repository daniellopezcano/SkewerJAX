"""Cell crossings of straight paths: which Voronoi cells a straight skewer crosses, and where (``docs/performance.md``,
section D2).

A straight path x(s) = x_0 + s t, 0 <= s <= L, meets the Voronoi cell of generator g where g is the nearest generator.
The squared distance to g is (s - u_g)^2 + r_g^2, with u_g = t.(g - x_0) and r_g the distance of g from the line, so
the nearest generator at every s is the lowest of the straight lines -2 u_g s + u_g^2 + r_g^2: their **lower envelope**
gives the cells crossed and the exact crossing points, s* = (u_i + u_j) / 2 + (r_i^2 - r_j^2) / (2 (u_i - u_j)), in
O(K log K) for K candidate generators. A crossing is one interval [s_lo, s_hi] of the path inside one cell, where the
density, temperature and peculiar velocity are constant.

**Candidates.** A cell can be crossed only if its generator lies within its circumradius of the line, which the reach
R = ``VORONOI_FACTOR`` V^(1/3) bounds (``core.selection``: the rule of every selection of the package). The candidates of
a path are therefore the cells with r_g <= R_g, found without any tree:

* ``line_pairs``: every (cell, path) pair with r <= R. The paths are bucketed on a grid of the plane perpendicular to
  their common axis (axis-aligned paths, e.g. a lattice: a cell at (x, z) of reach R looks only at the buckets of
  [x - R, x + R] x [z - R, z + R], with periodic wrap) or, for any other straight paths, tested against every path
  within the bounding boxes (``line_pairs`` falls back to a vectorised test by chunks);
* each pair carries the generator's image nearest to the path in a periodic box, its u and r^2.

``envelope_crossings`` sorts the candidates of every path by u and runs the envelope (numba, parallel over paths).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from skewerjax.core.selection import VORONOI_FACTOR

STRAIGHT_RTOL = 1e-7  # a path is straight when no point is farther than this x its length from the chord of its ends


@dataclass
class StraightPaths:
    """N straight paths x_0 + s t, 0 <= s <= L [cMpc] (float64 host arrays): ``origin`` (N, 3), ``direction`` (N, 3)
    unit vectors, ``length`` (N,), the periodic ``box`` (or None); ``axis`` the common axis (0, 1, 2) when every
    direction is +-e_axis (else None)."""

    origin: np.ndarray
    direction: np.ndarray
    length: np.ndarray
    box: float | None
    axis: int | None = None

    def __len__(self) -> int:
        return int(self.length.size)

    def take(self, idx) -> StraightPaths:
        return StraightPaths(self.origin[idx], self.direction[idx], self.length[idx], self.box, self.axis)


def straight_paths_mask(points, rtol: float = STRAIGHT_RTOL) -> np.ndarray:
    """(N_sk,) bool: the paths of ``points`` (N_sk, N, 3) [cMpc] whose points lie on the chord from their first to their
    last point within ``rtol`` x its length, in increasing order along it."""
    p = np.asarray(points, dtype=np.float64)
    x0, chord = p[:, :1, :], p[:, -1, :] - p[:, 0, :]
    length = np.linalg.norm(chord, axis=-1)
    t = chord / np.maximum(length, 1e-300)[:, None]
    rel = p - x0
    s = np.einsum("knc,kc->kn", rel, t)
    dev = np.linalg.norm(rel - s[..., None] * t[:, None, :], axis=-1).max(axis=-1)
    return (length > 0) & (dev <= rtol * length) & np.all(np.diff(s, axis=-1) >= 0, axis=-1)


def paths_from_points(points, box: float | None) -> StraightPaths:
    """The ``StraightPaths`` of straight sampled paths ``points`` (N_sk, N, 3) [cMpc]: from the first to the last point
    of each (``straight_paths_mask`` must hold; ``ValueError`` otherwise)."""
    p = np.asarray(points, dtype=np.float64)
    ok = straight_paths_mask(p)
    if not ok.all():
        raise ValueError(f"{int((~ok).sum())} of {ok.size} paths are not straight (straight_paths_mask).")
    chord = p[:, -1] - p[:, 0]
    L = np.linalg.norm(chord, axis=-1)
    t = chord / L[:, None]
    axis = None
    for k in range(3):
        if np.all(np.abs(np.abs(t[:, k]) - 1.0) < 1e-12):
            axis = k
            t = np.zeros_like(t)
            t[:, k] = np.sign(chord[:, k])
    return StraightPaths(p[:, 0].copy(), t, L, box, axis)


# --------------------------------------------------------------------------- numba kernels
_KERNELS = {}


def _kernels():
    """The numba kernels (compiled once per process; no disk cache)."""
    if _KERNELS:
        return _KERNELS
    import numba

    @numba.njit(nogil=True)
    def _wrap(d, box):
        if box > 0.0:
            return d - box * np.floor(d / box + 0.5)
        return d

    @numba.njit(nogil=True)
    def _n_images(y, r, o, sgn, L, box, us):
        """The images of a generator at axis coordinate y along a path (origin o, sign sgn, length L): their u with
        -r <= u <= L + r, written to ``us``; returns their number."""
        u0 = sgn * (y - o)
        n = 0
        if box > 0.0:
            u0 = u0 - box * np.floor(u0 / box)  # in [0, box)
            m = -1
            while u0 + m * box <= L + r:
                uu = u0 + m * box
                if uu >= -r:
                    us[n] = uu
                    n += 1
                m += 1
        elif u0 >= -r and u0 <= L + r:
            us[0] = u0
            n = 1
        return n

    @numba.njit(nogil=True)
    def axis_pairs(
        pos,
        R,
        ax0,
        ax1,
        axis,
        box,
        bstart,
        bcount,
        bitems,
        sk_c0,
        sk_c1,
        sk_o,
        sk_sign,
        sk_len,
        nb0,
        nb1,
        g0,
        g1,
        offs,
        cell_out,
        sk_out,
        u_out,
        r2_out,
        count_only,
        counts,
    ):
        """(cell, path, u, r^2) of every generator image within reach of an axis-aligned path (bucketed on the (ax0,
        ax1) plane): counted per cell (count_only), or written from offs[i]."""
        us = np.empty(64, np.float64)
        for i in range(pos.shape[0]):
            x, z, y, r = pos[i, ax0], pos[i, ax1], pos[i, axis], R[i]
            r2c = r * r
            ia, ib = int(np.floor((x - r) / g0)), int(np.floor((x + r) / g0))
            ja, jb = int(np.floor((z - r) / g1)), int(np.floor((z + r) / g1))
            if box > 0.0:
                ib = min(ib, ia + nb0 - 1)
                jb = min(jb, ja + nb1 - 1)
            else:  # no wrap: only the buckets of the grid
                ia, ib = max(ia, 0), min(ib, nb0 - 1)
                ja, jb = max(ja, 0), min(jb, nb1 - 1)
            o = offs[i] if not count_only else 0
            c = 0
            for bi in range(ia, ib + 1):
                bb0 = bi % nb0 if box > 0.0 else bi
                if bb0 < 0 or bb0 >= nb0:
                    continue
                for bj in range(ja, jb + 1):
                    bb1 = bj % nb1 if box > 0.0 else bj
                    if bb1 < 0 or bb1 >= nb1:
                        continue
                    b = bb0 * nb1 + bb1
                    for q in range(bstart[b], bstart[b] + bcount[b]):
                        k = bitems[q]
                        dx = _wrap(x - sk_c0[k], box)
                        dz = _wrap(z - sk_c1[k], box)
                        d2 = dx * dx + dz * dz
                        if d2 <= r2c:
                            ni = _n_images(y, r, sk_o[k], sk_sign[k], sk_len[k], box, us)
                            if not count_only:
                                for m in range(ni):
                                    cell_out[o + c + m] = i
                                    sk_out[o + c + m] = k
                                    r2_out[o + c + m] = d2
                                    u_out[o + c + m] = us[m]
                            c += ni
            counts[i] = c

    @numba.njit(nogil=True)
    def envelope(u, r2):
        """The lower envelope of (s - u)^2 + r2 over candidates sorted by u: (indices on the envelope in order, the s
        where each takes over (-inf for the first)). Equal u: the smaller r2 wins."""
        n = u.shape[0]
        stack = np.empty(n, np.int64)
        start = np.empty(n, np.float64)
        m = 0
        for i in range(n):
            if m > 0 and u[stack[m - 1]] == u[i]:
                if r2[i] < r2[stack[m - 1]]:
                    m -= 1  # replace the top (same slope, lower line)
                else:
                    continue
            x = -np.inf
            while m > 0:
                j = stack[m - 1]
                x = 0.5 * (u[i] + u[j]) + (r2[i] - r2[j]) / (2.0 * (u[i] - u[j]))
                if x <= start[m - 1]:
                    m -= 1
                    x = -np.inf
                else:
                    break
            stack[m] = i
            start[m] = x
            m += 1
        return stack[:m], start[:m]

    @numba.njit(nogil=True, parallel=True)
    def crossings_csr(pstart, u, r2, cell, length, out_start, out_cell, out_lo, out_hi, count_only):
        """For every path k (candidates pstart[k]:pstart[k+1], unsorted): sort by u, envelope, clip to [0, L_k]; either
        count the crossings (count_only) or write them at out_start[k]."""
        n_sk = pstart.shape[0] - 1
        counts = np.zeros(n_sk, np.int64)
        for k in numba.prange(n_sk):
            a, b = pstart[k], pstart[k + 1]
            if b <= a:
                continue
            uu = u[a:b]
            order = np.argsort(uu, kind="mergesort")
            us = uu[order]
            rs = r2[a:b][order]
            win, st = envelope(us, rs)
            m = win.shape[0]
            L = length[k]
            c = 0
            o = out_start[k] if not count_only else 0
            for q in range(m):
                lo = st[q] if q > 0 else -np.inf
                hi = st[q + 1] if q + 1 < m else np.inf
                lo = max(lo, 0.0)
                hi = min(hi, L)
                if hi > lo:
                    if not count_only:
                        out_cell[o + c] = cell[a + order[win[q]]]
                        out_lo[o + c] = lo
                        out_hi[o + c] = hi
                    c += 1
            counts[k] = c
        return counts

    @numba.njit(nogil=True)
    def counting_order(keys, n_keys):
        """The stable order that sorts integer ``keys`` in [0, n_keys) (a counting sort, O(n)); and the bucket starts."""
        cnt = np.zeros(n_keys + 1, np.int64)
        for k in keys:
            cnt[k + 1] += 1
        for i in range(n_keys):
            cnt[i + 1] += cnt[i]
        start = cnt.copy()
        order = np.empty(keys.size, np.int64)
        for i in range(keys.size):
            k = keys[i]
            order[cnt[k]] = i
            cnt[k] += 1
        return order, start

    _KERNELS.update(axis_pairs=axis_pairs, envelope=envelope, crossings_csr=crossings_csr, counting_order=counting_order)
    return _KERNELS


# --------------------------------------------------------------------------- pairs
@dataclass
class CandidatePairs:
    """(cell, path) candidate pairs (host arrays (n,)): the cell (an index into the positions given), the path, the
    along-path coordinate u of the generator image nearest to the path [cMpc] and its squared distance r2 from the line
    [cMpc^2]. Sorted by path (``start`` (N_sk + 1,): the CSR offsets)."""

    cell: np.ndarray
    path: np.ndarray
    u: np.ndarray
    r2: np.ndarray
    start: np.ndarray


def _bucket_paths(c0, c1, box, n_sk, extent0, extent1):
    """A grid of buckets on the plane of the axis-aligned paths (about one path per bucket): (bstart, bcount, bitems,
    nb0, nb1, g0, g1)."""
    lo0, lo1 = (0.0, 0.0) if box else (float(c0.min()), float(c1.min()))
    nb = max(1, int(np.sqrt(n_sk)))
    nb0 = nb if (box or extent0 > 0) else 1
    nb1 = nb if (box or extent1 > 0) else 1
    # without a box the buckets cover the paths' extent (one bucket, of width 1, when they share a coordinate)
    g0 = box / nb0 if box else (extent0 * (1 + 1e-9) / nb0 if extent0 > 0 else 1.0)
    g1 = box / nb1 if box else (extent1 * (1 + 1e-9) / nb1 if extent1 > 0 else 1.0)
    b0 = np.floor((c0 - lo0) / g0).astype(np.int64)
    b1 = np.floor((c1 - lo1) / g1).astype(np.int64)
    if box:
        b0 %= nb0
        b1 %= nb1
    else:
        b0 = np.clip(b0, 0, nb0 - 1)
        b1 = np.clip(b1, 0, nb1 - 1)
    b = b0 * nb1 + b1
    order = np.argsort(b, kind="stable")
    bcount = np.bincount(b, minlength=nb0 * nb1)
    bstart = np.concatenate([[0], np.cumsum(bcount)[:-1]])
    return bstart, bcount, order.astype(np.int64), nb0, nb1, g0, g1, lo0, lo1


def line_pairs(paths: StraightPaths, positions, reach, *, threads: int | None = None) -> CandidatePairs:
    """Every (cell, path) pair whose generator lies within the cell's ``reach`` R (N_c,) [cMpc] of the path's line
    (module docstring), with u and r^2 of the generator's image nearest to the middle of the path. ``positions`` (N_c, 3)
    [cMpc] in [0, box). Axis-aligned paths only (``paths.axis``) for now; ``NotImplementedError`` otherwise."""
    if paths.axis is None:
        return _general_pairs(paths, positions, reach)
    K = _kernels()
    pos = np.ascontiguousarray(positions, dtype=np.float64)
    R = np.ascontiguousarray(reach, dtype=np.float64)
    ax = paths.axis
    ax0, ax1 = [k for k in range(3) if k != ax]
    box = float(paths.box) if paths.box else 0.0
    n_sk = len(paths)
    c0, c1 = paths.origin[:, ax0].copy(), paths.origin[:, ax1].copy()
    if box:
        c0, c1 = np.mod(c0, box), np.mod(c1, box)
    bstart, bcount, bitems, nb0, nb1, g0, g1, lo0, lo1 = _bucket_paths(c0, c1, box, n_sk, np.ptp(c0), np.ptp(c1))
    sk_o = paths.origin[:, ax].astype(np.float64)
    sk_sign = paths.direction[:, ax].astype(np.float64)
    # the cells are shifted to the bucket origin (lo0, lo1 = 0 in a periodic box)
    pos_b = pos.copy()
    pos_b[:, ax0] -= lo0
    pos_b[:, ax1] -= lo1
    c0b, c1b = c0 - lo0, c1 - lo1
    n_thr = max(1, min(os.cpu_count() or 1, threads or (os.cpu_count() or 1)))
    chunks = (
        np.linspace(0, pos.shape[0], min(pos.shape[0], 4 * n_thr) + 1).astype(np.int64) if pos.shape[0] else np.zeros(1, np.int64)
    )
    counts = np.zeros(pos.shape[0], np.int64)
    lens = paths.length.astype(np.float64)
    di, df = np.zeros(1, np.int64), np.zeros(1)
    args = (ax0, ax1, ax, box, bstart, bcount, bitems, c0b, c1b, sk_o, sk_sign, lens, nb0, nb1, g0, g1)

    def count(j):
        a, b = chunks[j], chunks[j + 1]
        K["axis_pairs"](pos_b[a:b], R[a:b], *args, di, di, di, df, df, True, counts[a:b])

    _run_threads(count, len(chunks) - 1, n_thr)
    offs = np.concatenate([[0], np.cumsum(counts)])
    n = int(offs[-1])
    cell, sk = np.empty(n, np.int64), np.empty(n, np.int64)
    u, r2 = np.empty(n), np.empty(n)

    def fill(j):
        a, b = chunks[j], chunks[j + 1]
        K["axis_pairs"](pos_b[a:b], R[a:b], *args, offs[a:b], cell, sk, u, r2, False, np.empty(b - a, np.int64))
        cell[offs[a] : offs[b]] += a

    _run_threads(fill, len(chunks) - 1, n_thr)
    return _sort_by_path(cell, sk, u, r2, n_sk)


def _general_pairs(paths: StraightPaths, positions, reach, chunk: int = 1 << 16) -> CandidatePairs:
    """``line_pairs`` for straight paths of any direction: each path against every cell, by chunks of cells, with every
    periodic image of a generator (shifts of -1, 0, 1 box per axis from its image nearest to the middle of the path)
    whose distance to the line is within reach and whose u lies in [-R, L + R]. O(N_c N_sk): for few paths."""
    pos = np.asarray(positions, dtype=np.float64)
    R = np.asarray(reach, dtype=np.float64)
    box = paths.box
    shifts = np.array([(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)], float) if box else np.zeros((1, 3))
    cells, sks, us, r2s = [], [], [], []
    for k in range(len(paths)):
        x0, t, L = paths.origin[k], paths.direction[k], float(paths.length[k])
        mid = x0 + 0.5 * L * t
        for c0 in range(0, pos.shape[0], chunk):
            d0 = pos[c0 : c0 + chunk] - mid
            if box:
                d0 -= box * np.round(d0 / box)
            Rc = R[c0 : c0 + chunk]
            for sh in shifts:
                d = d0 + (box * sh if box else 0.0)
                uu = d @ t + 0.5 * L
                rr = np.maximum(np.sum(d * d, axis=1) - (uu - 0.5 * L) ** 2, 0.0)
                idx = np.flatnonzero((rr <= Rc * Rc) & (uu >= -Rc) & (uu <= L + Rc))
                cells.append(idx + c0)
                sks.append(np.full(idx.size, k))
                us.append(uu[idx])
                r2s.append(rr[idx])
    cat = lambda x, dt: np.concatenate(x).astype(dt) if x else np.zeros(0, dt)
    return _sort_by_path(cat(cells, np.int64), cat(sks, np.int64), cat(us, np.float64), cat(r2s, np.float64), len(paths))


def counting_order(keys, n_keys: int):
    """(order, start): the stable order that sorts the integer ``keys`` in [0, n_keys), and the start of every key
    (n_keys + 1,) (a counting sort, O(n): the pairs of a tile are tens of millions)."""
    return _kernels()["counting_order"](np.ascontiguousarray(keys, dtype=np.int64), int(n_keys))


def _sort_by_path(cell, sk, u, r2, n_sk) -> CandidatePairs:
    order, start = counting_order(sk, n_sk)
    return CandidatePairs(cell[order], sk[order], u[order], r2[order], start)


def _run_threads(job, n: int, n_thr: int) -> None:
    if n_thr > 1 and n > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=n_thr) as pool:
            list(pool.map(job, range(n)))
    else:
        for j in range(n):
            job(j)


# --------------------------------------------------------------------------- crossings
@dataclass
class Crossings:
    """The cell crossings of N_sk straight paths, as CSR arrays: path k holds the crossings ``start[k]:start[k+1]``,
    each with its ``cell`` (an index into the positions given), its ends ``s_lo`` < ``s_hi`` [cMpc] along the path
    (0 <= s <= L), in increasing s. The crossings tile every path exactly: s_lo of one is s_hi of the previous one."""

    start: np.ndarray
    cell: np.ndarray
    s_lo: np.ndarray
    s_hi: np.ndarray

    @property
    def n_paths(self) -> int:
        return int(self.start.size - 1)

    @property
    def counts(self) -> np.ndarray:
        return np.diff(self.start)

    @property
    def path(self) -> np.ndarray:
        """(n,) the path of every crossing."""
        return np.repeat(np.arange(self.n_paths), self.counts)

    @property
    def length(self) -> np.ndarray:
        return self.s_hi - self.s_lo


def envelope_crossings(pairs: CandidatePairs, lengths) -> Crossings:
    """The crossings of every path from its candidate pairs (lower envelope, module docstring), clipped to [0, L]."""
    K = _kernels()
    L = np.ascontiguousarray(lengths, dtype=np.float64)
    n_sk = L.size
    dummy_i, dummy_f = np.zeros(1, np.int64), np.zeros(1)
    counts = K["crossings_csr"](
        pairs.start, pairs.u, pairs.r2, pairs.cell, L, np.zeros(n_sk, np.int64), dummy_i, dummy_f, dummy_f, True
    )
    start = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    n = int(start[-1])
    cell, lo, hi = np.empty(n, np.int64), np.empty(n), np.empty(n)
    K["crossings_csr"](pairs.start, pairs.u, pairs.r2, pairs.cell, L, start[:-1].copy(), cell, lo, hi, False)
    return Crossings(start, cell, lo, hi)


def straight_crossings(paths: StraightPaths, positions, sizes, *, threads: int | None = None, return_pairs: bool = False):
    """The cell crossings of ``paths`` through the Voronoi cells of generators ``positions`` (N_c, 3) [cMpc] of sizes
    ``sizes`` = V^(1/3) [cMpc]: candidates within reach ``VORONOI_FACTOR`` x size of each line (``line_pairs``), then the
    lower envelope (``envelope_crossings``). Returns ``Crossings`` (and the ``CandidatePairs``)."""
    reach = VORONOI_FACTOR * np.asarray(sizes, dtype=np.float64)
    pairs = line_pairs(paths, positions, reach, threads=threads)
    cr = envelope_crossings(pairs, paths.length)
    return (cr, pairs) if return_pairs else cr


# --------------------------------------------------------------------------- top-hat chords (fake_spectra)
CHORD_POINTS = 7  # the interior points of fake_spectra's chord quadrature: NGRID - 1 (singleabs.h L9, L102)


@dataclass
class Chords:
    """The chords of fake_spectra's top-hat spheres (radius h = V^(1/3)) with straight paths (host arrays (n,)): the
    ``cell``, the ``path``, the path coordinate ``u`` of the sphere's centre [cMpc] and the chord's half length
    ``z`` = sqrt(h^2 - r^2) [cMpc], r the impact parameter."""

    cell: np.ndarray
    path: np.ndarray
    u: np.ndarray
    z: np.ndarray

    def points(self, chord_weight: float | None = None):
        """fake_spectra's quadrature (``singleabs.h`` L100-L118): the 7 interior points s = u + z (k / 4 - 1), k = 1..7,
        of every chord, each of length 2 z w (w: ``chord_weight``, default 1/7: the whole column; fake_spectra's 1/8):
        (index of the chord (7n,), s [cMpc] (7n,), length [cMpc] (7n,))."""
        w = 1.0 / CHORD_POINTS if chord_weight is None else float(chord_weight)
        k = np.arange(1, CHORD_POINTS + 1) / (0.5 * (CHORD_POINTS + 1)) - 1.0
        idx = np.repeat(np.arange(self.cell.size), CHORD_POINTS)
        s = (self.u[:, None] + self.z[:, None] * k[None, :]).ravel()
        return idx, s, np.repeat(2.0 * self.z * w, CHORD_POINTS)


def straight_chords(paths: StraightPaths, positions, sizes, *, threads: int | None = None) -> Chords:
    """The chords of the spheres of radius ``sizes`` (V^(1/3)) [cMpc] centred on ``positions`` with the straight
    ``paths``, as fake_spectra takes them: every sphere touching the line once, its image nearest to the middle of the
    path in a periodic box (``absorption.cpp`` L235; the old ``core.deposition.tophat_chord_segments``). From the
    candidate pairs of ``line_pairs`` with reach h: no tree, no scan of the cells per path."""
    pairs = line_pairs(paths, positions, np.asarray(sizes, dtype=np.float64), threads=threads)
    L = paths.length[pairs.path]
    keep = pairs.r2 < np.asarray(sizes, dtype=np.float64)[pairs.cell] ** 2
    if paths.box:  # the image nearest to the middle of the path: u - L/2 in [-box/2, box/2)
        d = pairs.u - 0.5 * L
        keep &= (d >= -0.5 * paths.box) & (d < 0.5 * paths.box)
    h = np.asarray(sizes, dtype=np.float64)[pairs.cell[keep]]
    return Chords(pairs.cell[keep], pairs.path[keep], pairs.u[keep], np.sqrt(h * h - pairs.r2[keep]))
