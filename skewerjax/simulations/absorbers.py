"""Absorbers (layer 6): the deposited pieces of every skewer, their frame a(s), and tau through the core.

What the optical-depth integral needs (review notes, Eq. tau_straight_los) is, along the path, the
comoving density of the absorbing ion, its temperature, its *peculiar* velocity and the length of each piece. The
Hubble flow is not a field: it enters through the scale factor a(s), set by the frame at every piece's centroid.

``Absorbers`` flattens the pieces of all skewers (cell pieces, sphere overlaps, ...) into one list per skewer (path
coordinate s of the centroid, density, piece length, T, line-of-sight peculiar velocity). ``absorber_pieces`` turns
them into ``core.Pieces``, and ``tau_absorbers`` integrates them with ``core.compute_optical_depth``, the same entry
point as every other field source:

    tau(v) = sum_p  C f  n_com,p a_p^-2  phi_p(v)  l_p .

Periodicity is not a property of the frame or of the pieces: it is ``compute_optical_depth``'s ``velocity_period``
(default: none), passed through by ``tau_absorbers``.

Frames (NumPy, host side):

* ``ExactFrame`` (SkewerJAX default): a(s) from the cosmology along the path; the spectral coordinate is the notes'
  v_dagger with a_dagger = a(s = 0).
* ``SnapshotFrame``: one scale factor a_snap for every point and the linear Hubble flow u = velfac s + v_pec as a
  line-of-sight velocity c tanh(u / c) (fake_spectra's frame, ``spectra.py`` L221-L228; matched option).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import jax.numpy as jnp
import numpy as np

from skewerjax.core.constants import phys
from skewerjax.core.optical_depth import Pieces, compute_optical_depth
from skewerjax.core.skewers import path_geometry

MPC_CM = 3.085678e24


# --------------------------------------------------------------------------- frames
@dataclass
class SnapshotFrame:
    """fake_spectra's frame: a = a_snap everywhere, linear Hubble flow velfac * s [km/s].

    path_length : (N_sk,) path length of every skewer [cMpc] (its velocity span is velfac x path_length).
    """
    a_snap: float
    velfac: float                      # [km/s per cMpc] = a H(z)
    path_length: np.ndarray

    def scale_factors(self, skewer: int, s):
        return np.full(np.shape(s), self.a_snap)

    def velocity(self, skewer: int, s, v_los):
        """Line-of-sight velocity that puts the line centre at velfac s + v_pec exactly."""
        return phys.c_kms * np.tanh((self.velfac * np.asarray(s) + np.asarray(v_los)) / phys.c_kms)

    def velocity_span(self, skewer: int) -> float:
        """Hubble-flow span of the path [km/s], first to last point."""
        return float(self.velfac * np.asarray(self.path_length)[skewer])

    @property
    def a_dagger(self):
        return self.a_snap

    def hubble_rate(self, skewer: int, s):
        """dv/ds of the frame [km/s per cMpc]."""
        return np.full(np.shape(s), self.velfac)


@dataclass
class ExactFrame:
    """a(s) from the cosmology along each path (review notes).

    Parameters
    ----------
    s_grid : (N_sk, N) path coordinates [cMpc] of the sampling points (increasing).
    a_grid : (N_sk, N) scale factors there (``scale_factors_along_paths``).
    a_dagger : reference scale factor of the spectral coordinate (default a(s = 0) of skewer 0).
    """
    s_grid: np.ndarray
    a_grid: np.ndarray
    a_dagger_value: Optional[float] = None

    def scale_factors(self, skewer: int, s):
        lna = np.interp(np.asarray(s), self.s_grid[skewer], np.log(self.a_grid[skewer]))
        return np.exp(lna)

    def velocity(self, skewer: int, s, v_los):
        return np.asarray(v_los, dtype=np.float64)

    @property
    def a_dagger(self):
        return float(self.a_grid[0, 0]) if self.a_dagger_value is None else float(self.a_dagger_value)

    def hubble_rate(self, skewer: int, s):
        lna = np.log(self.a_grid[skewer])
        dlna = np.gradient(lna, self.s_grid[skewer])
        return phys.c_kms * np.abs(np.interp(np.asarray(s), self.s_grid[skewer], dlna))

    def velocity_span(self, skewer: int) -> float:
        """c |ln(a_first / a_last)| [km/s]: the line-centre span of the path without peculiar velocities."""
        return float(phys.c_kms * abs(np.log(self.a_grid[skewer, 0] / self.a_grid[skewer, -1])))


def scale_factors_along_paths(s, a_ref: float, cosmology, s_ref=None) -> np.ndarray:
    """a(s) along paths with a(s_ref) = a_ref; s increases away from the observer.

    Uses ``Cosmology.solve_a_of_x`` (x = comoving distance towards the observer, da/dx > 0), forward
    and backward from s_ref. s : (N_sk, N) [cMpc]; s_ref : (N_sk,) or None (path midpoints).
    """
    s = np.atleast_2d(np.asarray(s, dtype=np.float64))
    s_ref = 0.5 * (s[:, 0] + s[:, -1]) if s_ref is None else np.broadcast_to(np.asarray(s_ref, float), (s.shape[0],))
    out = np.empty_like(s)
    for k in range(s.shape[0]):
        x = s_ref[k] - s[k]                                # towards the observer
        for m in (x >= 0, x < 0):
            if not m.any():
                continue
            xs = x[m]
            order = np.argsort(np.abs(xs))
            grid = np.concatenate([[0.0], xs[order]])
            a = np.asarray(cosmology.solve_a_of_x(jnp.array([a_ref]), jnp.asarray(grid), rtol=1e-10, atol=1e-12))[0, 1:]
            res = np.empty_like(xs)
            res[order] = a
            out[k, m] = res
    return out


# --------------------------------------------------------------------------- absorbers
@dataclass
class Absorbers:
    """Flattened components of the sampling points of N_sk skewers (padded per skewer).

    Arrays are (N_sk, M); padded entries have ``n = 0`` and ``cell = -1``.

    s [cMpc] path coordinate; n [physical cm^-3 at the data epoch] density inside the piece
    (deposition weight included); length [cMpc] piece length; T [K]; v_los [km/s] peculiar velocity
    along the tangent (positive away from the observer); cell, point: provenance indices;
    a_data: scale factor of the data.
    """
    s: np.ndarray
    n: np.ndarray
    length: np.ndarray
    T: np.ndarray
    v_los: np.ndarray
    cell: np.ndarray
    point: np.ndarray
    a_data: float
    species: str
    mass_amu: float
    meta: dict = field(default_factory=dict)

    @property
    def n_skewers(self) -> int:
        return self.s.shape[0]

    def column_cm2(self) -> np.ndarray:
        """(N_sk, M) physical column of every absorber at the data epoch [cm^-2]."""
        return self.n * self.length * MPC_CM * self.a_data

    def with_values(self, **kw) -> "Absorbers":
        d = {k: getattr(self, k) for k in ("s", "n", "length", "T", "v_los", "cell", "point")}
        d.update(kw)
        return Absorbers(**d, a_data=self.a_data, species=self.species, mass_amu=self.mass_amu, meta=dict(self.meta))


def _pad_stack(rows, fill):
    m = max(max((len(r) for r in rows), default=1), 1)
    return np.stack([np.concatenate([np.asarray(r, dtype=np.result_type(np.asarray(r), type(fill))),
                                     np.full(m - len(r), fill)]) for r in rows])


def absorbers_from_components(seg, points, n_cell, T_cell, velocity_cell, a_data: float, species: str,
                              mass_amu: float) -> Absorbers:
    """Turn a ``SegmentComponents`` (deposition) into ``Absorbers``.

    Parameters
    ----------
    seg : SegmentComponents for ``points`` (N_sk, N, 3).
    n_cell, T_cell : (M_cells,) ion density [physical cm^-3] and temperature [K] of the cells.
    velocity_cell : (M_cells, 3) physical peculiar velocities [km/s].
    """
    g = path_geometry(jnp.asarray(points, dtype=jnp.float64))
    s_pt, tan = np.asarray(g["s"]), np.asarray(g["tangents"])
    n_cell, T_cell, vel = (np.asarray(x, dtype=np.float64) for x in (n_cell, T_cell, velocity_cell))
    rows = {k: [] for k in ("s", "n", "length", "T", "v_los", "cell", "point")}
    for k in range(seg.cell.shape[0]):
        live = seg.cell[k] >= 0
        pi, ki = np.nonzero(live)
        c = seg.cell[k][pi, ki]
        rows["s"].append(s_pt[k][pi] + seg.offset[k][pi, ki])
        rows["n"].append(seg.weight * n_cell[c])
        rows["length"].append(seg.length[k][pi, ki])
        rows["T"].append(T_cell[c])
        rows["v_los"].append(np.sum(vel[c] * tan[k][pi], axis=-1))
        rows["cell"].append(c)
        rows["point"].append(pi)
    fills = dict(s=0.0, n=0.0, length=0.0, T=1e4, v_los=0.0, cell=-1, point=-1)
    return Absorbers(**{k: _pad_stack(v, fills[k]) for k, v in rows.items()}, a_data=float(a_data),
                     species=species, mass_amu=float(mass_amu), meta=dict(kind=seg.kind))


def absorbers_from_crossings(cross, points, n_cell, T_cell, velocity_cell, a_data: float, species: str, mass_amu: float,
                             *, frame=None, tolerance: Optional[float] = 0.25, max_split: int = 64) -> Absorbers:
    """Absorbers of the Voronoi-adapted segmentation: every cell crossing, split into equal sub-pieces.

    A crossing of length l holds one cell, so n, T and v_pec are constant on it; only a(s) and the Hubble
    flow vary. It is split into q equal sub-pieces at their midpoints (midpoint rule), with
    q = ceil(W / (tolerance b)), W = (dv/ds) l its velocity width and b the thermal width (the criterion of
    ``sampling_check``), capped at ``max_split``; ``frame=None`` or ``tolerance=None`` keeps q = 1. The
    column sum_p n_p l_p is exact for any q; the profile of each crossing converges as q grows.

    Parameters
    ----------
    cross : ``CellCrossings`` of ``points`` (N_sk, N, 3) [cMpc]; n_cell, T_cell, velocity_cell : as
        ``absorbers_from_components``; frame : the frame of the integral (its ``hubble_rate``).
    """
    g = path_geometry(jnp.asarray(points, dtype=jnp.float64))
    s_pt, tan = np.asarray(g["s"]), np.asarray(g["tangents"])
    n_cell, T_cell, vel = (np.asarray(x, dtype=np.float64) for x in (n_cell, T_cell, velocity_cell))
    rows = {k: [] for k in ("s", "n", "length", "T", "v_los", "cell", "point")}
    n_split = []
    for k in range(cross.cell.shape[0]):
        live = cross.cell[k] >= 0
        lo, hi, c = cross.s_lo[k][live], cross.s_hi[k][live], cross.cell[k][live]
        l = hi - lo
        if frame is None or tolerance is None:
            q = np.ones(c.size, np.int64)
        else:
            b = np.sqrt(2.0 * phys.kB * T_cell[c] / mass_amu) / 1e3
            W = frame.hubble_rate(k, 0.5 * (lo + hi)) * l
            q = np.clip(np.ceil(W / (tolerance * np.maximum(b, 1e-30))), 1, max_split).astype(np.int64)
        n_split.append(q)
        rep = np.repeat(np.arange(c.size), q)
        j = np.arange(rep.size) - np.repeat(np.cumsum(q) - q, q)
        s = lo[rep] + (j + 0.5) * l[rep] / q[rep]
        pt = np.clip(np.searchsorted(0.5 * (s_pt[k][1:] + s_pt[k][:-1]), s), 0, s_pt.shape[1] - 1)
        cc = c[rep]
        rows["s"].append(s); rows["n"].append(n_cell[cc]); rows["length"].append(l[rep] / q[rep])
        rows["T"].append(T_cell[cc]); rows["v_los"].append(np.sum(vel[cc] * tan[k][pt], axis=-1))
        rows["cell"].append(cc); rows["point"].append(pt)
    fills = dict(s=0.0, n=0.0, length=0.0, T=1e4, v_los=0.0, cell=-1, point=-1)
    q_all = np.concatenate(n_split) if n_split else np.zeros(0, np.int64)
    meta = dict(kind="voronoi_crossings", max_split=int(q_all.max()) if q_all.size else 0,
                mean_split=float(q_all.mean()) if q_all.size else 0.0)
    return Absorbers(**{k: _pad_stack(v, fills[k]) for k, v in rows.items()}, a_data=float(a_data),
                     species=species, mass_amu=float(mass_amu), meta=meta)


# --------------------------------------------------------------------------- sampling check
def sampling_check(ab: Absorbers, frame, *, T_scale: float = 1.0, tolerance: float = 0.25,
                   warn: bool = True, lorentz_hwhm_kms: float = 0.0) -> dict:
    """Is every absorber short enough in velocity to be represented at its centroid?

    The quadrature puts the absorption of a piece of length l at its centroid, which replaces a
    top hat of velocity width W = (dv/ds) l by a delta; the peak of a Doppler profile of width b
    then changes by W^2 / (12 b^2). The check flags W > tolerance x b (0.5 % for the default 1/4)
    and reports the fraction of the column affected. It depends only on the bundle spacing (which
    bounds l), the frame's Hubble rate and T: for a line of mass A it is the same for every ion
    with the same T, and heavier ions need finer sampling (b ~ A^-1/2).

    ``lorentz_hwhm_kms`` is the natural half width of the line in velocity, lambda (A_21 / 4 pi):
    the width compared with W is that of the Voigt profile (``optical_depth.voigt_width_kms``),
    which is the thermal b except for very cold gas (6e-3 km/s for Lya).
    """
    W = np.stack([frame.hubble_rate(k, ab.s[k]) for k in range(ab.n_skewers)]) * ab.length
    b_G = np.sqrt(2.0 * phys.kB * ab.T * T_scale / ab.mass_amu) / 1e3
    f_G, f_L = 2.0 * np.sqrt(np.log(2.0)) * b_G, 2.0 * lorentz_hwhm_kms
    b = (0.5346 * f_L + np.sqrt(0.2166 * f_L ** 2 + f_G ** 2)) / (2.0 * np.sqrt(np.log(2.0)))
    ratio = W / np.maximum(b, 1e-30)
    col = ab.n * ab.length
    frac = float((col * (ratio > tolerance)).sum() / max(col.sum(), 1e-300))
    rep = dict(max_ratio=float(ratio[col > 0].max()) if np.any(col > 0) else 0.0, column_fraction=frac,
               b_min=float(b[col > 0].min()) if np.any(col > 0) else np.nan, ok=frac < 1e-3, tolerance=tolerance)
    if warn and not rep["ok"]:
        warnings.warn(f"{ab.species}: {frac:.2%} of the column sits in pieces wider than {tolerance} b in velocity "
                      f"(max W/b = {rep['max_ratio']:.2f}, b_min = {rep['b_min']:.2f} km/s). Sample the skewer more "
                      f"densely: spacing <= {tolerance} b_min / (dv/ds).", stacklevel=2)
    return rep


# --------------------------------------------------------------------------- optical depth
def absorber_pieces(ab: Absorbers, frame, n_lines: int, *, skewers=None, density_scaling: str = "comoving", T_scale: float = 1.0,
                    n_scale: float = 1.0) -> Pieces:
    """The absorbers of ``skewers`` (default all) as ``core.Pieces`` (N_sk, M), padded with empty pieces.

    Every absorber is one piece: at its centroid s, with the frame's a(s), the velocity the frame injects along the
    path (photons travel towards decreasing s), T x ``T_scale``, the comoving density n_com = n a_data^3 x ``n_scale``
    (``density_scaling="fixed_column"``: n a_data a(s)^2) and its length. Its width in velocity is the frame's Hubble
    rate times its length (n, T and v_pec are constant inside a cell). ``T_scale`` and ``n_scale`` are matched-mode
    knobs (fake_spectra's b constants and 7/8 chords).
    """
    ks = range(ab.n_skewers) if skewers is None else [int(k) for k in np.atleast_1d(skewers)]
    s, n, l, T, vl = (np.asarray(getattr(ab, f))[list(ks)] for f in ("s", "n", "length", "T", "v_los"))
    a = np.stack([frame.scale_factors(k, s[i]) for i, k in enumerate(ks)])
    u = np.stack([frame.velocity(k, s[i], vl[i]) for i, k in enumerate(ks)])
    W = np.stack([frame.hubble_rate(k, s[i]) for i, k in enumerate(ks)]) * l
    n_com = n * n_scale * (ab.a_data ** 3 if density_scaling == "comoving" else ab.a_data * a ** 2)
    lines = lambda x: jnp.broadcast_to(jnp.asarray(x)[None], (n_lines,) + x.shape)
    zeros = np.zeros(T.shape)
    return Pieces(a_dagger=jnp.full(s.shape, frame.a_dagger), scale_factor=jnp.asarray(a), length_m=jnp.asarray(l) * phys.Mpc_to_m,
                  temperature=lines(T * T_scale), turbulence=lines(zeros), collisional=lines(zeros), los_velocity=lines(-u),
                  number_density=lines(n_com), width_kms=lines(W),
                  velocity_span_kms=jnp.asarray([frame.velocity_span(k) for k in ks]))


def tau_absorbers(ab: Absorbers, line_bundle, velocities_kms, frame, *, velocity_period=False, density_scaling: str = "comoving",
                  T_scale: float = 1.0, n_scale: float = 1.0, detuning: str = "exact",
                  pixel_offsets=None, pixel_weights=None, chunk: int = 2048, v_chunk: Optional[int] = 4096,
                  voigt: str = "ExoJAX", check: Optional[float] = 0.25, per_absorber: bool = False,
                  dtype: Optional[str] = None, devices=None, skewers=None) -> np.ndarray:
    """tau(v) of every skewer (or of the ``skewers`` given) from its absorbers, through ``core.compute_optical_depth``.

    Parameters
    ----------
    line_bundle : LineBundle whose species is ``ab.species``.
    velocities_kms : (N_v,) spectral coordinate [km/s] (v_dagger in ``ExactFrame``; u in ``SnapshotFrame``), or
        (N_lines, N_v), one grid per line.
    frame : ExactFrame or SnapshotFrame (sets a(s), the injected velocity and a_dagger).
    velocity_period : ``compute_optical_depth``'s option (default False: no wrap; True: each skewer periodic over
        its own velocity span; a number: that period [km/s]).
    density_scaling, T_scale, n_scale : see ``absorber_pieces``.
    detuning : "exact" (default) or "linear" (matched mode).
    pixel_offsets, pixel_weights : sub-pixel offsets [km/s] and weights (sum 1) to average each pixel; None: centres.
    chunk : pieces per kernel call (``compute_optical_depth``'s ``piece_chunk``); v_chunk : velocities per call.
    check : tolerance of ``sampling_check`` (None: no check).
    per_absorber : True returns (N_lines, N_sk, M, N_v) contributions (memory!).
    dtype : opt-in kernel precision, e.g. "float32" (``core.optical_depth_from_pieces``).
    devices : None (one device) or devices to split the rows of pieces of each skewer over (``core.optical_depth_sharded``).
    skewers : optional indices of the skewers to integrate (default all); the output follows their order.

    Returns
    -------
    (N_lines, N_sk, N_v) optical depth [float64] (N_sk = len(skewers) if given).
    """
    if ab.species not in tuple(line_bundle.species_keys) or len(tuple(line_bundle.species_keys)) != 1:
        raise ValueError(f"line_bundle must hold lines of {ab.species!r} only, got {tuple(line_bundle.species_keys)}.")
    if density_scaling not in ("comoving", "fixed_column"):
        raise ValueError(f"density_scaling must be 'comoving' or 'fixed_column', got {density_scaling!r}.")
    if check is not None:
        hwhm = float(np.min(np.asarray(line_bundle.lambda0_A) * np.asarray(line_bundle.A21_GHz) / (4.0 * np.pi))) * 1e-4
        sampling_check(ab, frame, T_scale=T_scale, tolerance=check, lorentz_hwhm_kms=hwhm)
    v = np.asarray(velocities_kms, dtype=np.float64)
    off = np.zeros(1) if pixel_offsets is None else np.asarray(pixel_offsets, float)
    wts = np.ones(1) if pixel_weights is None else np.asarray(pixel_weights, float)
    v_all = (v[..., :, None] + off).reshape(v.shape[:-1] + (-1,))
    n_lines, M, n_v = line_bundle.n_lines, ab.s.shape[1], v.shape[-1]
    ks = range(ab.n_skewers) if skewers is None else [int(k) for k in np.atleast_1d(skewers)]
    out = np.zeros((n_lines, len(ks), M if per_absorber else 1, n_v))
    kw = dict(velocity_period=velocity_period, voigt=voigt, detuning=detuning, resolution_tolerance=None, v_chunk=v_chunk,
              piece_chunk=chunk, per_piece=per_absorber, dtype=None if per_absorber else dtype, devices=devices)
    for i, k in enumerate(ks):
        pieces = absorber_pieces(ab, frame, n_lines, skewers=[k], density_scaling=density_scaling, T_scale=T_scale,
                                 n_scale=n_scale)
        t = np.asarray(compute_optical_depth(pieces, v_all if v.ndim == 1 else jnp.asarray(v_all), line_bundle, **kw))[:, 0]
        if per_absorber:
            out[:, i] = (t.reshape(n_lines, M, n_v, off.size) * wts).sum(-1)
        else:
            out[:, i, 0] = (t.reshape(n_lines, n_v, off.size) * wts).sum(-1)
    return out if per_absorber else out[:, :, 0]


def tau_by_path_bins(ab: Absorbers, line_bundle, velocities_kms, frame, s_edges, *, skewer: int = 0, **kw) -> np.ndarray:
    """(N_bins, N_lines, N_v) tau of one skewer from the absorbers in each path interval [s_edges[i], s_edges[i + 1])
    [cMpc]: which stretch of gas makes which absorption. tau is linear in the densities, so the bins add up to the
    tau of the absorbers inside [s_edges[0], s_edges[-1]). ``kw`` go to ``tau_absorbers`` (default ``check=None``)."""
    kw.setdefault("check", None)
    e = np.asarray(s_edges, dtype=np.float64)
    out = []
    for lo, hi in zip(e[:-1], e[1:]):
        inside = (ab.s >= lo) & (ab.s < hi)
        out.append(tau_absorbers(ab.with_values(n=np.where(inside, ab.n, 0.0)), line_bundle, velocities_kms, frame,
                                 skewers=[skewer], **kw)[:, 0])
    return np.stack(out)


# --------------------------------------------------------------------------- columns
def column_per_pixel(ab: Absorbers, path_length: float, nbins: int) -> np.ndarray:
    """(N_sk, nbins) physical column [cm^-2] of the absorbers binned exactly into ``nbins`` equal
    position pixels of a periodic path of length ``path_length`` [cMpc]. Each absorber is the
    interval [s - l/2, s + l/2] (a piece of one convex cell or sphere inside one segment)."""
    out = np.zeros((ab.n_skewers, nbins))
    dz = path_length / nbins
    for k in range(ab.n_skewers):
        live = ab.n[k] > 0
        lo = ab.s[k][live] - 0.5 * ab.length[k][live]
        hi = lo + ab.length[k][live]
        dens = ab.n[k][live] * MPC_CM * ab.a_data
        j0, j1 = np.floor(lo / dz).astype(np.int64), np.floor(hi / dz).astype(np.int64)
        n = j1 - j0 + 1
        cell = np.repeat(np.arange(lo.size), n)
        j = np.repeat(j0, n) + (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n))
        length = np.minimum(hi[cell], (j + 1) * dz) - np.maximum(lo[cell], j * dz)
        keep = length > 0
        out[k] = np.bincount(np.mod(j[keep], nbins), weights=dens[cell[keep]] * length[keep], minlength=nbins)
    return out
