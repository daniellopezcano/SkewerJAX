"""Absorbers: the deposited pieces of every skewer, their frame a(s), and the pieces of the optical-depth sum.

What the optical-depth integral needs (review notes, Eq. tau_straight_los) is, along the path, the comoving density
of the absorbing species, its temperature, its *peculiar* velocity and the length of each piece. The Hubble flow is
not a field: it enters through the scale factor a(s), set by the frame at every piece's centroid.

The route from the gas to tau is the same for every source of gas (``core.cells.Cells``, from a toy model or from a
simulation):

    Cells + sampling points --selection--> cells within reach --deposition--> pieces of every dual segment
          --``absorbers_from_components``--> ``Absorbers`` (one species) --``absorber_pieces``--> ``core.Pieces``
          --``core.compute_optical_depth``--> tau(v) = sum_p  C f  n_com,p a_p^-2  phi_p(v)  l_p .

``cell_pieces`` runs the whole chain for every species of a ``LineBundle``; ``compute_optical_depth(bundle, v, lines,
cells)`` calls it. ``tau_absorbers`` integrates the absorbers of one species skewer by skewer (bounded memory,
optional pixel averaging).

Periodicity is not a property of the frame or of the pieces: it is ``compute_optical_depth``'s ``velocity_period``
(default: none), passed through by ``tau_absorbers``.

Frames (NumPy, host side):

* ``ExactFrame`` (SkewerJAX default): a(s) along the path (from the bundle's scale factors, or from the cosmology);
  the spectral coordinate is the notes' v_dagger.
* ``SnapshotFrame``: one scale factor a_snap for every point and the linear Hubble flow u = velfac s + v_pec as a
  line-of-sight velocity c tanh(u / c) (fake_spectra's frame, ``spectra.py`` L221-L228; matched option).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import jax.numpy as jnp
import numpy as np

from skewerjax.core import deposition as dep
from skewerjax.core.constants import phys
from skewerjax.core.optical_depth import Pieces, compute_optical_depth
from skewerjax.core.selection import select_cells
from skewerjax.core.skewers import path_geometry


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
        """The scale factor at path lengths ``s`` [cMpc]: a_snap everywhere."""
        return np.full(np.shape(s), self.a_snap)

    def velocity(self, skewer: int, s, v_los):
        """Line-of-sight velocity that puts the line centre at velfac s + v_pec exactly."""
        return phys.c_kms * np.tanh((self.velfac * np.asarray(s) + np.asarray(v_los)) / phys.c_kms)

    def velocity_span(self, skewer: int) -> float:
        """Hubble-flow span of the path [km/s], first to last point."""
        return float(self.velfac * np.asarray(self.path_length)[skewer])

    @property
    def a_dagger(self):
        """The scale factor of the spectral axis: a_snap."""
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
    a_grid : (N_sk, N) scale factors there (``scale_factors_along_paths``, or a ``SkewerBundle``'s).
    a_dagger_value : reference scale factor of the spectral coordinate (default a(s = 0) of skewer 0).
    photon_sign : None (the simulation convention: the observer is before the first point and the photons travel
        towards decreasing s) or (N_sk,) +1 / -1, the direction of the photons along s of every skewer
        (``SkewerBundle.array_direction``).
    a_dagger_per_skewer : optional (N_sk,) a_dagger of every skewer (instead of one value).
    """
    s_grid: np.ndarray
    a_grid: np.ndarray
    a_dagger_value: Optional[float] = None
    photon_sign: Optional[np.ndarray] = None
    a_dagger_per_skewer: Optional[np.ndarray] = None

    def scale_factors(self, skewer: int, s):
        """a(s) at path lengths ``s`` [cMpc] of skewer ``skewer``, interpolated in ln a."""
        lna = np.interp(np.asarray(s), self.s_grid[skewer], np.log(self.a_grid[skewer]))
        return np.exp(lna)

    def velocity(self, skewer: int, s, v_los):
        """The velocity along -(photon direction) [km/s] of gas with velocity ``v_los`` along +s."""
        if self.photon_sign is None:
            return np.asarray(v_los, dtype=np.float64)
        return -float(self.photon_sign[skewer]) * np.asarray(v_los, dtype=np.float64)

    @property
    def a_dagger(self):
        """The scale factor a_dagger of the spectral axis (default: a at the first point)."""
        return float(self.a_grid[0, 0]) if self.a_dagger_value is None else float(self.a_dagger_value)

    def hubble_rate(self, skewer: int, s):
        """|dv_H/ds| = c |d ln a / ds| of the frame [km/s per cMpc]."""
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
    """The pieces of N_sk skewers for one species, flattened per skewer (padded).

    Arrays are (N_sk, M); padded entries have ``n = 0`` and ``cell = -1``.

    s [cMpc] path coordinate of the centroid; n [comoving cm^-3] density inside the piece (deposition weight
    included); length [cMpc] piece length; T [K]; v_los [km/s] peculiar velocity along the tangent (positive towards
    increasing s); turbulence [m/s], collisional [GHz] (None: 0); cell, point: provenance indices (cell of the
    ``Cells`` used, sampling point of the dual segment); a_data: the scale factor of the data (a snapshot's), or None.
    """
    s: np.ndarray
    n: np.ndarray
    length: np.ndarray
    T: np.ndarray
    v_los: np.ndarray
    cell: np.ndarray
    point: np.ndarray
    a_data: Optional[float]
    species: str
    mass_amu: float
    meta: dict = field(default_factory=dict)
    turbulence: Optional[np.ndarray] = None
    collisional: Optional[np.ndarray] = None

    @property
    def n_skewers(self) -> int:
        """Number of skewers N_sk."""
        return self.s.shape[0]

    def column_cm2(self, a: Optional[float] = None) -> np.ndarray:
        """(N_sk, M) physical column of every absorber at scale factor ``a`` (default: the data's) [cm^-2]:
        n_com l / a^2."""
        a = self.a_data if a is None else a
        if a is None:
            raise ValueError("column_cm2 needs a scale factor (these absorbers have no a_data).")
        return self.n * self.length * phys.Mpc_to_cm / a ** 2

    def with_values(self, **kw) -> "Absorbers":
        """A copy with some per-absorber arrays replaced (e.g. ``n=...``)."""
        d = {k: getattr(self, k) for k in ("s", "n", "length", "T", "v_los", "cell", "point", "turbulence", "collisional")}
        d.update(kw)
        return Absorbers(**d, a_data=self.a_data, species=self.species, mass_amu=self.mass_amu, meta=dict(self.meta))


def _pad_stack(rows, fill):
    m = max(max((len(r) for r in rows), default=1), 1)
    return np.stack([np.concatenate([np.asarray(r, dtype=np.result_type(np.asarray(r), type(fill))),
                                     np.full(m - len(r), fill)]) for r in rows])


def _species_values(cells, species: str):
    """(n [comoving cm^-3], T [K], velocity (M, 3) [km/s], turbulence [m/s] or None, collisional [GHz] or None)."""
    f = cells.species[species]
    v = lambda k: np.asarray(f[k], dtype=np.float64)
    return (v("number_density"), v("temperature"), v("velocity"), v("turbulence") if "turbulence" in f else None,
            v("collisional") if "collisional" in f else None)


def absorbers_from_components(seg, points, cells, species: str, mass_amu: float, *, a_data: Optional[float] = None) -> Absorbers:
    """Turn a ``SegmentComponents`` (deposition) of ``cells`` into the ``Absorbers`` of ``species``.

    Parameters
    ----------
    seg : SegmentComponents for ``points`` (N_sk, N, 3), with cell indices into ``cells``.
    cells : ``core.Cells`` (the species' comoving density, temperature, peculiar velocity and, if given, turbulence
        and collisional term).
    mass_amu : mass of the species [u] (thermal widths of ``sampling_check``); a_data : scale factor of the data.
    """
    g = path_geometry(jnp.asarray(points, dtype=jnp.float64))
    s_pt, tan = np.asarray(g["s"]), np.asarray(g["tangents"])
    n_cell, T_cell, vel, turb, coll = _species_values(cells, species)
    keys = ("s", "n", "length", "T", "v_los", "cell", "point") + (("turbulence",) if turb is not None else ()) \
        + (("collisional",) if coll is not None else ())
    rows = {k: [] for k in keys}
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
        if turb is not None:
            rows["turbulence"].append(turb[c])
        if coll is not None:
            rows["collisional"].append(coll[c])
    fills = dict(s=0.0, n=0.0, length=0.0, T=1e4, v_los=0.0, cell=-1, point=-1, turbulence=0.0, collisional=0.0)
    return Absorbers(**{k: _pad_stack(v, fills[k]) for k, v in rows.items()}, a_data=None if a_data is None else float(a_data),
                     species=species, mass_amu=float(mass_amu), meta=dict(kind=seg.kind))


def absorbers_from_crossings(cross, points, cells, species: str, mass_amu: float, *, a_data: Optional[float] = None, frame=None,
                             tolerance: Optional[float] = 0.25, max_split: int = 64, split=None) -> Absorbers:
    """Absorbers of the Voronoi-adapted segmentation: every cell crossing, split into equal sub-pieces.

    A crossing of length l holds one cell, so n, T and v_pec are constant on it; only a(s) and the Hubble
    flow vary. It is split into q equal sub-pieces at their midpoints (midpoint rule), with
    q = ceil(W / (tolerance b)), W = (dv/ds) l its velocity width and b the thermal width (the criterion of
    ``sampling_check``), capped at ``max_split``; ``frame=None`` or ``tolerance=None`` keeps q = 1. The
    column sum_p n_p l_p is exact for any q; the profile of each crossing converges as q grows.

    Parameters
    ----------
    cross : ``CellCrossings`` of ``points`` (N_sk, N, 3) [cMpc]; cells, species, mass_amu, a_data : as
        ``absorbers_from_components``; frame : the frame of the integral (its ``hubble_rate``).
    split : optional list of (n_crossings_k,) integer splits per skewer, used instead of the criterion (the same
        sub-pieces for several species: ``crossing_splits``).
    """
    g = path_geometry(jnp.asarray(points, dtype=jnp.float64))
    s_pt, tan = np.asarray(g["s"]), np.asarray(g["tangents"])
    n_cell, T_cell, vel, turb, coll = _species_values(cells, species)
    keys = ("s", "n", "length", "T", "v_los", "cell", "point") + (("turbulence",) if turb is not None else ()) \
        + (("collisional",) if coll is not None else ())
    rows = {k: [] for k in keys}
    q_list = crossing_splits(cross, T_cell, mass_amu, frame, tolerance, max_split) if split is None else split
    for k in range(cross.cell.shape[0]):
        live = cross.cell[k] >= 0
        lo, hi, c = cross.s_lo[k][live], cross.s_hi[k][live], cross.cell[k][live]
        l = hi - lo
        q = q_list[k]
        rep = np.repeat(np.arange(c.size), q)
        j = np.arange(rep.size) - np.repeat(np.cumsum(q) - q, q)
        s = lo[rep] + (j + 0.5) * l[rep] / q[rep]
        pt = np.clip(np.searchsorted(0.5 * (s_pt[k][1:] + s_pt[k][:-1]), s), 0, s_pt.shape[1] - 1)
        cc = c[rep]
        rows["s"].append(s); rows["n"].append(n_cell[cc]); rows["length"].append(l[rep] / q[rep])
        rows["T"].append(T_cell[cc]); rows["v_los"].append(np.sum(vel[cc] * tan[k][pt], axis=-1))
        rows["cell"].append(cc); rows["point"].append(pt)
        if turb is not None:
            rows["turbulence"].append(turb[cc])
        if coll is not None:
            rows["collisional"].append(coll[cc])
    fills = dict(s=0.0, n=0.0, length=0.0, T=1e4, v_los=0.0, cell=-1, point=-1, turbulence=0.0, collisional=0.0)
    q_all = np.concatenate(q_list) if len(q_list) else np.zeros(0, np.int64)
    meta = dict(kind="voronoi_crossings", max_split=int(q_all.max()) if q_all.size else 0,
                mean_split=float(q_all.mean()) if q_all.size else 0.0)
    return Absorbers(**{k: _pad_stack(v, fills[k]) for k, v in rows.items()}, a_data=None if a_data is None else float(a_data),
                     species=species, mass_amu=float(mass_amu), meta=meta)


def crossing_splits(cross, T_cell, mass_amu: float, frame=None, tolerance: Optional[float] = 0.25, max_split: int = 64) -> list:
    """Per skewer, the number of equal sub-pieces q of every cell crossing (``absorbers_from_crossings``)."""
    out = []
    for k in range(cross.cell.shape[0]):
        live = cross.cell[k] >= 0
        lo, hi, c = cross.s_lo[k][live], cross.s_hi[k][live], cross.cell[k][live]
        l = hi - lo
        if frame is None or tolerance is None:
            q = np.ones(c.size, np.int64)
        else:
            b = np.sqrt(2.0 * phys.kB * np.asarray(T_cell)[c] / mass_amu) / 1e3
            W = frame.hubble_rate(k, 0.5 * (lo + hi)) * l
            q = np.clip(np.ceil(W / (tolerance * np.maximum(b, 1e-30))), 1, max_split).astype(np.int64)
        out.append(q)
    return out


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
    the width compared with W is that of the Voigt profile (``line_profiles.voigt_width_kms``),
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

    Every absorber is one piece: at its centroid s, with the frame's a(s), the velocity along the photon (the frame's
    injected velocity; photons travel towards decreasing s unless the frame says otherwise), T x ``T_scale``, the
    comoving density n x ``n_scale`` (``density_scaling="fixed_column"``: n (a(s) / a_data)^2, fake_spectra's column at
    the data epoch), its turbulence and collisional term, and its length. Its width in velocity is the frame's Hubble
    rate times its length (n, T and v_pec are constant inside a cell). ``T_scale`` and ``n_scale`` are matched-mode
    knobs (fake_spectra's b constants and 7/8 chords).
    """
    ks = range(ab.n_skewers) if skewers is None else [int(k) for k in np.atleast_1d(skewers)]
    s, n, l, T, vl = (np.asarray(getattr(ab, f))[list(ks)] for f in ("s", "n", "length", "T", "v_los"))
    a = np.stack([frame.scale_factors(k, s[i]) for i, k in enumerate(ks)])
    u = np.stack([frame.velocity(k, s[i], vl[i]) for i, k in enumerate(ks)])
    W = np.stack([frame.hubble_rate(k, s[i]) for i, k in enumerate(ks)]) * l
    if density_scaling == "comoving":
        n_com = n * n_scale
    else:
        if ab.a_data is None:
            raise ValueError("density_scaling='fixed_column' needs absorbers with a_data.")
        n_com = n * n_scale * a ** 2 / ab.a_data ** 2
    lines = lambda x: jnp.broadcast_to(jnp.asarray(x)[None], (n_lines,) + x.shape)
    opt = lambda x: np.zeros(T.shape) if x is None else np.asarray(x)[list(ks)]
    a_sk = getattr(frame, "a_dagger_per_skewer", None)
    a_d = jnp.full(s.shape, frame.a_dagger) if a_sk is None else jnp.broadcast_to(jnp.asarray(np.asarray(a_sk)[list(ks)])[:, None], s.shape)
    return Pieces(a_dagger=a_d, scale_factor=jnp.asarray(a), length_m=jnp.asarray(l) * phys.Mpc_to_m,
                  temperature=lines(T * T_scale), turbulence=lines(opt(ab.turbulence)), collisional=lines(opt(ab.collisional)),
                  los_velocity=lines(-u), number_density=lines(n_com), width_kms=lines(W),
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
    """(N_sk, nbins) physical column [cm^-2] at the data epoch (``a_data``) of the absorbers binned exactly into
    ``nbins`` equal position pixels of a periodic path of length ``path_length`` [cMpc]. Each absorber is the
    interval [s - l/2, s + l/2] (a piece of one convex cell or sphere inside one segment)."""
    out = np.zeros((ab.n_skewers, nbins))
    dz = path_length / nbins
    for k in range(ab.n_skewers):
        live = ab.n[k] > 0
        lo = ab.s[k][live] - 0.5 * ab.length[k][live]
        hi = lo + ab.length[k][live]
        dens = ab.n[k][live] * phys.Mpc_to_cm / ab.a_data ** 2
        j0, j1 = np.floor(lo / dz).astype(np.int64), np.floor(hi / dz).astype(np.int64)
        n = j1 - j0 + 1
        cell = np.repeat(np.arange(lo.size), n)
        j = np.repeat(j0, n) + (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n))
        length = np.minimum(hi[cell], (j + 1) * dz) - np.maximum(lo[cell], j * dz)
        keep = length > 0
        out[k] = np.bincount(np.mod(j[keep], nbins), weights=dens[cell[keep]] * length[keep], minlength=nbins)
    return out


# --------------------------------------------------------------------------- the route: cells -> pieces
DEPOSITIONS = ("voronoi", "tophat", "voronoi_crossings")


def deposit(cells, points, *, deposition: str = "voronoi", selection_backend: str = "cpu", grid_fast_path: bool = True):
    """Selection and deposition of ``cells`` along ``points`` (N_sk, N, 3) [cMpc]: (the cells the indices refer to,
    ``SegmentComponents``).

    deposition : "voronoi" (default: the exact pieces of the Voronoi cells in every dual segment; for a grid of cells,
        ``grid_segments``, unless ``grid_fast_path=False``), "tophat" (spheres of radius V^(1/3), the fake_spectra
        option) or "voronoi_crossings" (the Voronoi pieces, joined per cell crossing by ``cell_pieces``).
    selection_backend : backend of ``core.selection.select_cells``, which first keeps the cells within reach of the
        points (skipped by the grid fast path).
    """
    if deposition not in DEPOSITIONS:
        raise ValueError(f"deposition must be one of {DEPOSITIONS}, got {deposition!r}.")
    pts = np.asarray(points, dtype=np.float64)
    if deposition != "tophat" and grid_fast_path and cells.grid is not None and cells.box is not None:
        return cells, dep.grid_segments(pts, cells.grid, cells.box)
    mode = "tophat" if deposition == "tophat" else "voronoi"
    use = select_cells(cells, pts, mode=mode, backend=selection_backend)
    if deposition == "tophat":
        return use, dep.tophat_segments(pts, use.positions, use.size_cMpc, use.box)
    return use, dep.voronoi_segments(pts, use.positions, use.box, gas_size=use.size_cMpc)


def bundle_frame(skewer_bundle, *, a_dagger=None, ii_cosmology: int = 0) -> ExactFrame:
    """The ``ExactFrame`` of a ``SkewerBundle``: a(s) interpolated in ln a between its points (where a(x) was solved),
    its photon directions, and a_dagger (default: the bundle's ``a0``; a scalar, or (N_sk,) per skewer)."""
    pts = np.asarray(skewer_bundle.spatial_points, dtype=np.float64)
    s = np.asarray(path_geometry(jnp.asarray(pts))["s"])
    a = np.asarray(skewer_bundle.quadri_vectors[ii_cosmology][..., 0], dtype=np.float64)
    a_d = np.asarray(skewer_bundle.a0 if a_dagger is None else a_dagger, dtype=np.float64)
    if a_d.ndim > 1 or (a_d.ndim == 1 and a_d.size not in (1, pts.shape[0])):
        raise ValueError(f"a_dagger must be a scalar or one value per skewer ({pts.shape[0]},), got shape {a_d.shape}.")
    per = None if a_d.size <= 1 or np.all(a_d == a_d.ravel()[0]) else np.broadcast_to(a_d, (pts.shape[0],)).copy()
    return ExactFrame(s, a, a_dagger_value=float(np.ravel(a_d)[0]), photon_sign=np.asarray(skewer_bundle.array_direction),
                      a_dagger_per_skewer=per)


def cell_pieces(cells, line_bundle, *, skewer_bundle=None, points=None, frame=None, a_dagger=None, ii_cosmology: int = 0,
                deposition: str = "voronoi", segments=None, selection_backend: str = "cpu",
                grid_fast_path: bool = True, crossing_tolerance: Optional[float] = 0.25, T_scale: float = 1.0,
                n_scale: float = 1.0, density_scaling: str = "comoving", a_data: Optional[float] = None, skewers=None,
                return_parts: bool = False):
    """The pieces of the optical-depth sum for every line of ``line_bundle``, from ``cells`` along a set of paths.

    The one route from the gas to tau, for every source of cells (toy models, simulations):

    1. selection of the cells within reach of the points, and deposition into the pieces of every dual segment
       (``deposit``; for ``deposition="voronoi_crossings"`` the pieces are joined per cell crossing and split into
       equal sub-pieces of velocity width <= ``crossing_tolerance`` b of the narrowest species);
    2. per species, the ``Absorbers`` of its cells (``absorbers_from_components`` / ``absorbers_from_crossings``);
    3. ``absorber_pieces`` in the frame: a(s) at every centroid and the velocity along the photon.

    Every species shares the same pieces (one deposition), so a line's fields are those of its species.

    Parameters
    ----------
    cells : ``core.Cells`` holding every species of ``line_bundle``.
    skewer_bundle : the paths and their a(x) (``bundle_frame``: a_dagger default its ``a0``); or give ``points``
        (N_sk, N, 3) [cMpc] and a ``frame`` (``ExactFrame`` / ``SnapshotFrame``) instead.
    deposition, selection_backend, grid_fast_path : see ``deposit``; segments : a ``SegmentComponents`` already
        computed for ``points`` (its cell indices refer to ``cells``), used instead.
    T_scale, n_scale, density_scaling, a_data : see ``absorber_pieces`` (matched-mode knobs; defaults change nothing).
    skewers : optional indices of the skewers to return.
    return_parts : also return dict(cells, segments, absorbers {species: Absorbers}, frame, crossings).

    Returns
    -------
    ``core.Pieces`` (N_lines, N_sk, N_p) (and the parts).
    """
    from dataclasses import replace
    if skewer_bundle is not None:
        points = np.asarray(skewer_bundle.spatial_points, dtype=np.float64)
        frame = bundle_frame(skewer_bundle, a_dagger=a_dagger, ii_cosmology=ii_cosmology) if frame is None else frame
    elif points is None or frame is None:
        raise ValueError("give a skewer_bundle, or points and a frame.")
    pts = np.asarray(points, dtype=np.float64)
    species = list(line_bundle.species_keys)
    missing = [sp for sp in species if sp not in cells.species]
    if missing:
        raise ValueError(f"the cells hold no {missing} (they hold {cells.species_keys}).")
    if segments is None:
        use, seg = deposit(cells, pts, deposition="voronoi" if deposition == "voronoi_crossings" else deposition,
                           selection_backend=selection_backend, grid_fast_path=grid_fast_path)
    else:
        use, seg = cells, segments
    mass = {sp: float(np.asarray(line_bundle.mass_amu)[line_bundle.indices_for_species(sp)[0]]) for sp in species}
    cross = None
    if deposition == "voronoi_crossings":
        cross = dep.cell_crossings(seg, pts)
        per_sp = [crossing_splits(cross, np.asarray(use.species[sp]["temperature"]), mass[sp], frame, crossing_tolerance)
                  for sp in species]
        split = [np.max(np.stack([q[k] for q in per_sp]), axis=0) for k in range(len(per_sp[0]))]
        abs_ = {sp: absorbers_from_crossings(cross, pts, use, sp, mass[sp], a_data=a_data, split=split) for sp in species}
    else:
        abs_ = {sp: absorbers_from_components(seg, pts, use, sp, mass[sp], a_data=a_data) for sp in species}
    kw = dict(skewers=skewers, density_scaling=density_scaling, T_scale=T_scale, n_scale=n_scale)
    per = {sp: absorber_pieces(abs_[sp], frame, 1, **kw) for sp in species}
    p0 = per[species[0]]
    n_l = line_bundle.n_lines
    rows = {f: [None] * n_l for f in ("temperature", "turbulence", "collisional", "los_velocity", "number_density", "width_kms")}
    for sp in species:
        for j in line_bundle.indices_for_species(sp):
            for f in rows:
                rows[f][j] = getattr(per[sp], f)[0]
    pieces = replace(p0, **{f: jnp.stack(v) for f, v in rows.items()})
    if return_parts:
        return pieces, dict(cells=use, segments=seg, absorbers=abs_, frame=frame, crossings=cross)
    return pieces
