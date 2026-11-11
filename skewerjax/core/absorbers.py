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
optional sub-pixel points).

Periodicity is not a property of the frame or of the pieces: it is ``compute_optical_depth``'s ``velocity_period``
(default: none), passed through by ``tau_absorbers``.

Frames (NumPy, host side):

* ``ExactFrame`` (SkewerJAX default): a(s) along the path (from the bundle's scale factors, or from the cosmology);
  the spectral coordinate is the notes' v_dagger.
* ``SnapshotFrame``: one scale factor a_snap for every point and the linear Hubble flow u = velfac s + v_pec as a
  line-of-sight velocity c tanh(u / c) (fake_spectra's frame, ``spectra.py`` L221-L228, ``absorption.cpp`` L237).
"""

from __future__ import annotations

import logging
import time
import warnings
from dataclasses import dataclass, field

import jax.numpy as jnp
import numpy as np

from skewerjax.config import device_array
from skewerjax.core import deposition as dep
from skewerjax.core.constants import phys
from skewerjax.core.line_profiles import check_b_constants
from skewerjax.core.optical_depth import Pieces, compute_optical_depth, resolution_report
from skewerjax.core.selection import select_cells
from skewerjax.core.skewers import path_geometry

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- frames
@dataclass
class SnapshotFrame:
    """fake_spectra's frame: a = a_snap everywhere, linear Hubble flow velfac * s [km/s].

    path_length : (N_sk,) path length of every skewer [cMpc] (its velocity span is velfac x path_length).
    """

    a_snap: float
    velfac: float  # [km/s per cMpc] = a H(z)
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

    def hubble_coordinate(self, skewer: int, s):
        """The path coordinate along which the frame's c ln(a_dagger / a) runs: 0 here (a = a_snap everywhere; the
        Hubble flow velfac s is in the injected velocity), so ``hubble_flow="linear"`` and "exact" coincide."""
        return np.zeros(np.shape(s))

    def path_length_of(self, skewer: int) -> float:
        """The comoving length of the path [cMpc]."""
        return float(np.asarray(self.path_length)[skewer])

    def first_hubble_rate(self, skewer: int) -> float:
        """dV_H/dy at the first point [km/s per cMpc]: velfac."""
        return float(self.velfac)

    def first_hubble_velocity(self, skewer: int) -> float:
        """V_H at the first point of the frame's c ln(a_dagger / a): 0."""
        return 0.0


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
    a_dagger_value: float | None = None
    photon_sign: np.ndarray | None = None
    a_dagger_per_skewer: np.ndarray | None = None

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

    def hubble_coordinate(self, skewer: int, s):
        """The comoving path coordinate y = s - s_first [cMpc], along which c ln(a_dagger / a(y)) runs."""
        return np.asarray(s, dtype=np.float64) - float(self.s_grid[skewer, 0])

    def path_length_of(self, skewer: int) -> float:
        """The comoving length of the path, first to last point [cMpc]."""
        return float(self.s_grid[skewer, -1] - self.s_grid[skewer, 0])

    def first_hubble_rate(self, skewer: int) -> float:
        """dV_H/dy = -c d ln a / dy at the first point (the position of a_dagger) [km/s per cMpc], signed (> 0 when a
        decreases along the path): the default (aH)_ref of ``hubble_flow="linear"``. Second-order one-sided difference
        of ln a on the sampling points."""
        lna, sg = np.log(self.a_grid[skewer]), self.s_grid[skewer]
        g = np.gradient(lna[:3], sg[:3], edge_order=2)[0] if lna.size >= 3 else (lna[1] - lna[0]) / (sg[1] - sg[0])
        return float(-phys.c_kms * g)

    def first_hubble_velocity(self, skewer: int) -> float:
        """V_H at the first point, c ln(a_dagger / a_first) [km/s] (0 when a_dagger is a at the first point)."""
        a_d = self.a_dagger if self.a_dagger_per_skewer is None else float(np.asarray(self.a_dagger_per_skewer)[skewer])
        return float(phys.c_kms * np.log(a_d / self.a_grid[skewer, 0]))


def velocity_span(source, hubble_flow: str = "exact", aH_ref=None, *, ii_cosmology: int = 0):
    """(N_sk,) the line-centre span of every path without peculiar velocities [km/s] under ``hubble_flow`` (the period
    of ``velocity_period=True``, and the length of the pixel axis of one box): "exact" c |ln(a_first / a_last)| of the
    frame, "linear" |(aH)_ref| L with ``aH_ref`` (None: the frame's aH at the first point, ``first_hubble_rate``).

    source : ``Pieces`` (``optical_depth.velocity_span``), a ``SkewerBundle`` (its ``bundle_frame``) or a frame
    (``ExactFrame``, ``SnapshotFrame``)."""
    from skewerjax.core.optical_depth import HUBBLE_FLOWS
    from skewerjax.core.optical_depth import velocity_span as pieces_span

    if isinstance(source, Pieces):
        return np.asarray(pieces_span(source, hubble_flow, aH_ref))
    if hubble_flow not in HUBBLE_FLOWS:
        raise ValueError(f"hubble_flow must be one of {HUBBLE_FLOWS}, got {hubble_flow!r}.")
    frame = source if hasattr(source, "velocity_span") else bundle_frame(source, ii_cosmology=ii_cosmology)
    n_sk = np.shape(frame.s_grid)[0] if hasattr(frame, "s_grid") else np.size(frame.path_length)
    if hubble_flow == "exact":
        return np.array([frame.velocity_span(k) for k in range(n_sk)])
    rate = lambda k: frame.first_hubble_rate(k) if aH_ref is None else float(aH_ref)
    return np.array([abs(rate(k)) * frame.path_length_of(k) for k in range(n_sk)])


def scale_factors_along_paths(s, a_ref: float, cosmology, s_ref=None) -> np.ndarray:
    """a(s) along paths with a(s_ref) = a_ref; s increases away from the observer.

    Solved in float64 on the host (``Cosmology.solve_a_of_x_host``; x = comoving distance towards the observer,
    da/dx > 0), forward and backward from s_ref. s : (N_sk, N) [cMpc]; s_ref : (N_sk,) or None (path midpoints).
    Paths with the same s grid and s_ref (parallel skewers through a box) share one solve (the box mode).
    """
    s = np.atleast_2d(np.asarray(s, dtype=np.float64))
    s_ref = 0.5 * (s[:, 0] + s[:, -1]) if s_ref is None else np.broadcast_to(np.asarray(s_ref, float), (s.shape[0],))
    key = np.concatenate([s_ref[:, None], s], axis=1)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    sols = np.empty((uniq.shape[0], s.shape[1]))
    for u, row in enumerate(uniq):
        x = row[0] - row[1:]  # towards the observer
        res = np.empty_like(x)
        for m in (x >= 0, x < 0):
            if not m.any():
                continue
            xs = x[m]
            order = np.argsort(np.abs(xs))
            grid = np.concatenate([[0.0], xs[order]])
            a = cosmology.solve_a_of_x_host(np.full(1, a_ref), grid)[0, 1:]
            out = np.empty_like(xs)
            out[order] = a
            res[m] = out
        sols[u] = res
    return sols[np.ravel(inv)]


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
    a_data: float | None
    species: str
    mass_amu: float
    meta: dict = field(default_factory=dict)
    turbulence: np.ndarray | None = None
    collisional: np.ndarray | None = None

    @property
    def n_skewers(self) -> int:
        """Number of skewers N_sk."""
        return self.s.shape[0]

    def column_cm2(self, a: float | None = None) -> np.ndarray:
        """(N_sk, M) physical column of every absorber at scale factor ``a`` (default: the data's) [cm^-2]:
        n_com l / a^2."""
        a = self.a_data if a is None else a
        if a is None:
            raise ValueError("column_cm2 needs a scale factor (these absorbers have no a_data).")
        return self.n * self.length * phys.Mpc_to_cm / a**2

    def with_values(self, **kw) -> Absorbers:
        """A copy with some per-absorber arrays replaced (e.g. ``n=...``)."""
        d = {k: getattr(self, k) for k in ("s", "n", "length", "T", "v_los", "cell", "point", "turbulence", "collisional")}
        d.update(kw)
        return Absorbers(**d, a_data=self.a_data, species=self.species, mass_amu=self.mass_amu, meta=dict(self.meta))


_FILLS = dict(s=0.0, n=0.0, length=0.0, T=1e4, v_los=0.0, cell=-1, point=-1, turbulence=0.0, collisional=0.0)  # padding


def _pad_stack(rows, fill):
    m = max(max((len(r) for r in rows), default=1), 1)
    return np.stack(
        [
            np.concatenate([np.asarray(r, dtype=np.result_type(np.asarray(r), type(fill))), np.full(m - len(r), fill)])
            for r in rows
        ]
    )


def _species_values(cells, species: str):
    """(n [comoving cm^-3], T [K], velocity (M, 3) [km/s], turbulence [m/s] or None, collisional [GHz] or None)."""
    f = cells.species[species]
    v = lambda k: np.asarray(f[k], dtype=np.float64)
    return (
        v("number_density"),
        v("temperature"),
        v("velocity"),
        v("turbulence") if "turbulence" in f else None,
        v("collisional") if "collisional" in f else None,
    )


def _path_s_tangents(points):
    """(s (N_sk, N) [cMpc], tangents (N_sk, N, 3)) of the sampling points (``path_geometry``)."""
    g = path_geometry(np.asarray(points, dtype=np.float64))  # float64 NumPy, whatever the precision policy
    return g["s"], g["tangents"]


class _AbsorberRows:
    """Per-skewer rows of the absorbers of one species, padded into ``Absorbers`` by ``build`` (the optional turbulence
    and collisional fields only when the cells carry them)."""

    def __init__(self, turb, coll):
        self.turb, self.coll = turb, coll
        keys = (
            ("s", "n", "length", "T", "v_los", "cell", "point")
            + (("turbulence",) if turb is not None else ())
            + (("collisional",) if coll is not None else ())
        )
        self.rows = {k: [] for k in keys}

    def add(self, c, **values):
        """One skewer: the cells ``c`` of its absorbers and their per-absorber values (s, n, length, T, v_los, point)."""
        for k, v in values.items():
            self.rows[k].append(v)
        self.rows["cell"].append(c)
        if self.turb is not None:
            self.rows["turbulence"].append(self.turb[c])
        if self.coll is not None:
            self.rows["collisional"].append(self.coll[c])

    def build(self, species: str, mass_amu: float, a_data: float | None, meta: dict) -> Absorbers:
        return Absorbers(
            **{k: _pad_stack(v, _FILLS[k]) for k, v in self.rows.items()},
            a_data=None if a_data is None else float(a_data),
            species=species,
            mass_amu=float(mass_amu),
            meta=meta,
        )


def absorbers_from_components(seg, points, cells, species: str, mass_amu: float, *, a_data: float | None = None) -> Absorbers:
    """Turn a ``SegmentComponents`` (deposition) of ``cells`` into the ``Absorbers`` of ``species``.

    Parameters
    ----------
    seg : SegmentComponents for ``points`` (N_sk, N, 3), with cell indices into ``cells``.
    cells : ``core.Cells`` (the species' comoving density, temperature, peculiar velocity and, if given, turbulence
        and collisional term).
    mass_amu : mass of the species [u] (thermal widths of ``sampling_check``); a_data : scale factor of the data.
    """
    s_pt, tan = _path_s_tangents(points)
    n_cell, T_cell, vel, turb, coll = _species_values(cells, species)
    rows = _AbsorberRows(turb, coll)
    for k in range(seg.cell.shape[0]):
        pi, ki = np.nonzero(seg.cell[k] >= 0)
        c = seg.cell[k][pi, ki]
        rows.add(
            c,
            s=s_pt[k][pi] + seg.offset[k][pi, ki],
            n=seg.weight * n_cell[c],
            length=seg.length[k][pi, ki],
            T=T_cell[c],
            v_los=np.sum(vel[c] * tan[k][pi], axis=-1),
            point=pi,
        )
    return rows.build(species, mass_amu, a_data, dict(kind=seg.kind))


def absorbers_from_crossings(
    cross,
    points,
    cells,
    species: str,
    mass_amu: float,
    *,
    a_data: float | None = None,
    frame=None,
    tolerance: float | None = 0.25,
    max_split: int = 64,
    split=None,
) -> Absorbers:
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
    s_pt, tan = _path_s_tangents(points)
    n_cell, T_cell, vel, turb, coll = _species_values(cells, species)
    rows = _AbsorberRows(turb, coll)
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
        rows.add(
            cc, s=s, n=n_cell[cc], length=l[rep] / q[rep], T=T_cell[cc], v_los=np.sum(vel[cc] * tan[k][pt], axis=-1), point=pt
        )
    q_all = np.concatenate(q_list) if len(q_list) else np.zeros(0, np.int64)
    meta = dict(
        kind="voronoi_crossings",
        max_split=int(q_all.max()) if q_all.size else 0,
        mean_split=float(q_all.mean()) if q_all.size else 0.0,
    )
    return rows.build(species, mass_amu, a_data, meta)


def crossing_splits(cross, T_cell, mass_amu: float, frame=None, tolerance: float | None = 0.25, max_split: int = 64) -> list:
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
def sampling_check(
    ab: Absorbers, frame, *, T_scale: float = 1.0, tolerance: float = 0.25, warn: bool = True, lorentz_hwhm_kms: float = 0.0
) -> dict:
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
    b = (0.5346 * f_L + np.sqrt(0.2166 * f_L**2 + f_G**2)) / (2.0 * np.sqrt(np.log(2.0)))
    ratio = W / np.maximum(b, 1e-30)
    col = ab.n * ab.length
    frac = float((col * (ratio > tolerance)).sum() / max(col.sum(), 1e-300))
    rep = dict(
        max_ratio=float(ratio[col > 0].max()) if np.any(col > 0) else 0.0,
        column_fraction=frac,
        b_min=float(b[col > 0].min()) if np.any(col > 0) else np.nan,
        ok=frac < 1e-3,
        tolerance=tolerance,
    )
    if warn and not rep["ok"]:
        warnings.warn(
            f"{ab.species}: {frac:.2%} of the column sits in pieces wider than {tolerance} b in velocity "
            f"(max W/b = {rep['max_ratio']:.2f}, b_min = {rep['b_min']:.2f} km/s). Sample the skewer more "
            f"densely: spacing <= {tolerance} b_min / (dv/ds).",
            stacklevel=2,
        )
    return rep


# --------------------------------------------------------------------------- optical depth
def absorber_pieces(
    ab: Absorbers,
    frame,
    n_lines: int,
    *,
    skewers=None,
) -> Pieces:
    """The absorbers of ``skewers`` (default all) as ``core.Pieces`` (N_sk, M), padded with empty pieces.

    Every absorber is one piece: at its centroid s, with the frame's a(s), the velocity along the photon (the frame's
    injected velocity; photons travel towards decreasing s unless the frame says otherwise), T, the comoving density
    n, its turbulence and collisional term, its length, its cell (``Pieces.cell``: the per-cell options of
    ``compute_optical_depth``), and the path fields of ``hubble_flow="linear"`` (``hubble_coordinate``,
    ``path_length_of``, ``first_hubble_rate``, ``first_hubble_velocity`` of the frame). A fixed comoving-to-physical
    conversion, e.g. fake_spectra's column at the data epoch a_data, is ``compute_optical_depth(a_eff=a_data)``. Its
    width in velocity is the span of its line centres (n, T and v_pec are constant inside a cell), with the weight
    gradients of the integrated profile (``Pieces.tilt_a2``, ``tilt_flow``).
    """
    ks = list(range(ab.n_skewers) if skewers is None else [int(k) for k in np.atleast_1d(skewers)])
    s, n, l, T, vl = (np.asarray(getattr(ab, f), dtype=np.float64)[ks] for f in ("s", "n", "length", "T", "v_los"))
    a = np.stack([frame.scale_factors(k, s[i]) for i, k in enumerate(ks)])
    u = np.stack([frame.velocity(k, s[i], vl[i]) for i, k in enumerate(ks)])
    opt = lambda x: np.zeros(T.shape) if x is None else np.asarray(x, dtype=np.float64)[ks]
    a_sk = getattr(frame, "a_dagger_per_skewer", None)
    a_d = (
        np.full(s.shape, frame.a_dagger)
        if a_sk is None
        else np.broadcast_to(np.asarray(a_sk, dtype=np.float64)[ks][:, None], s.shape)
    )
    # The line centres at the two ends and the middle of every piece (float64): the span of the piece's line centres,
    # W = |v_c(hi) - v_c(lo)|, its centre (the Hubble part kept in hubble_kms, the peculiar part in los_velocity), and
    # the gradients of the weight per unit line centre, a^-2 ds / dv_c, along it (the integrated profile,
    # ``core.tau_engine``). In float64 here: in float32 the kernel would take the log of a ratio rounded to ~6e-8, i.e.
    # ~0.02 km/s of error (docs/precision.md).
    path = np.repeat(np.arange(len(ks)), s.shape[1])
    sub = _SkewerSubset(frame, ks)
    ends = [line_centres(sub, path, (s + d * l).ravel(), vl.ravel()) for d in (-0.5, 0.0, 0.5)]
    (v_lo, a_lo), (v_mid, _), (v_hi, a_hi) = [(x.reshape(s.shape), y.reshape(s.shape)) for x, y in ends]
    W = np.abs(v_hi - v_lo)
    beta = -u / phys.c_kms
    pec = phys.c_kms * 0.5 * (np.log1p(-beta) - np.log1p(beta))
    hub = 0.5 * (v_lo + v_hi) - pec  # the Hubble part of the span's centre
    dvc = v_hi - v_lo
    with np.errstate(divide="ignore", invalid="ignore"):
        slope_lo = np.abs(-3.0 * v_lo + 4.0 * v_mid - v_hi)
        slope_hi = np.abs(v_lo - 4.0 * v_mid + 3.0 * v_hi)
        ok = (dvc != 0) & (l > 0)
        tilt_a2 = np.where(ok, 2.0 * np.log(a_lo / a_hi) / dvc, 0.0)
        tilt_flow = np.where(ok & (slope_lo > 0) & (slope_hi > 0), np.log(slope_lo / slope_hi) / dvc, 0.0)
    dev = lambda x: device_array(np.ascontiguousarray(x))
    lines = lambda x: jnp.broadcast_to(dev(x)[None], (n_lines,) + x.shape)
    return Pieces(
        a_dagger=dev(a_d),
        scale_factor=dev(a),
        length_m=dev(l * phys.Mpc_to_m),
        temperature=lines(T),
        turbulence=lines(opt(ab.turbulence)),
        collisional=lines(opt(ab.collisional)),
        los_velocity=lines(-u),
        number_density=lines(n),
        width_kms=lines(W),
        velocity_span_kms=dev(np.array([frame.velocity_span(k) for k in ks])),
        path_cMpc=dev(np.stack([frame.hubble_coordinate(k, s[i]) for i, k in enumerate(ks)])),
        path_length_cMpc=dev(np.array([frame.path_length_of(k) for k in ks])),
        aH_first=dev(np.array([frame.first_hubble_rate(k) for k in ks])),
        vH_first=dev(np.array([frame.first_hubble_velocity(k) for k in ks])),
        cell=jnp.asarray(np.asarray(ab.cell)[ks]),
        hubble_kms=dev(hub),
        tilt_a2=dev(tilt_a2),
        tilt_flow=dev(tilt_flow),
    )


class _SkewerSubset:
    """A frame seen through the skewers ``ks`` (row i is skewer ks[i]); its methods take the row."""

    def __init__(self, frame, ks):
        self.f, self.ks = frame, list(ks)
        self.a_dagger = frame.a_dagger
        self.a_dagger_per_skewer = (
            None if getattr(frame, "a_dagger_per_skewer", None) is None else np.asarray(frame.a_dagger_per_skewer)[self.ks]
        )

    def scale_factors(self, k, s):
        return self.f.scale_factors(self.ks[k], s)

    def velocity(self, k, s, v):
        return self.f.velocity(self.ks[k], s, v)

    def first_hubble_rate(self, k):
        return self.f.first_hubble_rate(self.ks[k])

    def first_hubble_velocity(self, k):
        return self.f.first_hubble_velocity(self.ks[k])

    def hubble_coordinate(self, k, s):
        return self.f.hubble_coordinate(self.ks[k], s)


def tau_absorbers(
    ab: Absorbers,
    line_bundle,
    velocities_kms,
    frame,
    *,
    velocity_period=False,
    a_eff: float | None = None,
    b_constants: str = "codata",
    detuning: str = "exact",
    hubble_flow: str = "exact",
    aH_ref=None,
    pixel_offsets=None,
    pixel_weights=None,
    chunk: int = 2048,
    v_chunk: int | None = 4096,
    voigt: str = "harris",
    check: float | None = 0.25,
    per_absorber: bool = False,
    dtype: str | None = None,
    devices=None,
    skewers=None,
) -> np.ndarray:
    """tau(v) of every skewer (or of the ``skewers`` given) from its absorbers, through ``core.compute_optical_depth``.

    Parameters
    ----------
    line_bundle : LineBundle whose species is ``ab.species``.
    velocities_kms : (N_v,) spectral coordinate [km/s] (v_dagger in ``ExactFrame``; u in ``SnapshotFrame``), or
        (N_lines, N_v), one grid per line.
    frame : ExactFrame or SnapshotFrame (sets a(s), the injected velocity and a_dagger).
    velocity_period : ``compute_optical_depth``'s option (default False: no wrap; True: each skewer periodic over
        its own velocity span; a number: that period [km/s]).
    a_eff : None (default: a_p^-2 of every piece) or a scalar, the effective scale factor of the comoving-to-physical
        conversion (``core.compute_optical_depth``; e.g. a_eff = ``ab.a_data``, fake_spectra's column at the data epoch).
    b_constants : the constants of the thermal width (``core.compute_optical_depth``; "codata" default).
    detuning : "exact" (default) or "linear" (fake_spectra's).
    hubble_flow, aH_ref : the Hubble part of the line centres (``core.compute_optical_depth``; "exact" default).
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
    if check is not None:
        hwhm = float(np.min(np.asarray(line_bundle.lambda0_A) * np.asarray(line_bundle.A21_GHz) / (4.0 * np.pi))) * 1e-4
        sampling_check(ab, frame, T_scale=check_b_constants(b_constants), tolerance=check, lorentz_hwhm_kms=hwhm)
    v = np.asarray(velocities_kms, dtype=np.float64)
    off = np.zeros(1) if pixel_offsets is None else np.asarray(pixel_offsets, float)
    wts = np.ones(1) if pixel_weights is None else np.asarray(pixel_weights, float)
    v_all = (v[..., :, None] + off).reshape(v.shape[:-1] + (-1,))
    n_lines, M, n_v = line_bundle.n_lines, ab.s.shape[1], v.shape[-1]
    ks = range(ab.n_skewers) if skewers is None else [int(k) for k in np.atleast_1d(skewers)]
    out = np.zeros((n_lines, len(ks), M if per_absorber else 1, n_v))
    kw = dict(
        velocity_period=velocity_period,
        voigt=voigt,
        detuning=detuning,
        resolution_tolerance=None,
        v_chunk=v_chunk,
        piece_chunk=chunk,
        per_piece=per_absorber,
        dtype=None if per_absorber else dtype,
        devices=devices,
        a_eff=a_eff,
        hubble_flow=hubble_flow,
        aH_ref=aH_ref,
        b_constants=b_constants,
    )
    for i, k in enumerate(ks):
        pieces = absorber_pieces(ab, frame, n_lines, skewers=[k])
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
        out.append(
            tau_absorbers(
                ab.with_values(n=np.where(inside, ab.n, 0.0)), line_bundle, velocities_kms, frame, skewers=[skewer], **kw
            )[:, 0]
        )
    return np.stack(out)


# --------------------------------------------------------------------------- contributions of cell crossings
def _shared_grid(frame) -> bool:
    """True for an ``ExactFrame`` whose skewers share one s grid and one a(s) (the box mode)."""
    if not isinstance(frame, ExactFrame):
        return False
    s, a = np.asarray(frame.s_grid), np.asarray(frame.a_grid)
    return bool(np.array_equal(s, np.broadcast_to(s[:1], s.shape)) and np.array_equal(a, np.broadcast_to(a[:1], a.shape)))


def line_centres(frame, path, s, v_los, *, hubble_flow: str = "exact", aH_ref=None):
    """(v_c [km/s], a) at path coordinates ``s`` [cMpc] of the paths ``path`` (n,), for gas of velocity ``v_los``
    [km/s, along +s] (float64): v_c = V_H(s) + c (ln(1 - beta) - ln(1 + beta)) / 2, beta = -u / c, u the frame's
    velocity along the photon (``absorber_pieces``, ``spectral.line_centre_velocity_kms``); V_H = c ln(a_dagger / a(s))
    ("exact") or V_H(0) + (aH)_ref y(s) ("linear"). Vectorised over the paths in box mode (``_shared_grid``)."""
    path = np.asarray(path, dtype=np.int64)
    s = np.asarray(s, dtype=np.float64)
    v_los = np.asarray(v_los, dtype=np.float64)
    a = np.empty(s.shape)
    u = np.empty(s.shape)
    VH = np.empty(s.shape)
    a_sk = getattr(frame, "a_dagger_per_skewer", None)
    if isinstance(frame, SnapshotFrame):  # a = a_snap, u = c tanh((velfac s + v) / c): v_c = velfac s + v exactly
        a[:] = frame.a_snap
        u[:] = phys.c_kms * np.tanh((frame.velfac * s + v_los) / phys.c_kms)
        VH[:] = 0.0
    elif _shared_grid(frame):
        lna = np.interp(s, frame.s_grid[0], np.log(frame.a_grid[0]))
        a[:] = np.exp(lna)
        a_d = np.full(s.shape, frame.a_dagger) if a_sk is None else np.asarray(a_sk, dtype=np.float64)[path]
        VH[:] = phys.c_kms * (np.log(a_d) - lna)
        sign = None if frame.photon_sign is None else np.asarray(frame.photon_sign, dtype=np.float64)[path]
        u[:] = v_los if sign is None else -sign * v_los
        if hubble_flow == "linear":
            y = s - float(frame.s_grid[0, 0])
            rate = np.array([frame.first_hubble_rate(0)]) if aH_ref is None else np.array([float(aH_ref)])
            v0 = (
                np.array([frame.first_hubble_velocity(k) for k in range(np.max(path) + 1)])[path]
                if a_sk is not None
                else frame.first_hubble_velocity(0)
            )
            VH[:] = v0 + rate[0] * y
    else:
        order = np.argsort(path, kind="stable")
        ps = path[order]
        bounds = np.flatnonzero(np.diff(np.concatenate([[-1], ps, [ps[-1] + 1 if ps.size else 0]])))
        for i0, i1 in zip(bounds[:-1], bounds[1:]):
            idx = order[i0:i1]
            k = int(path[idx[0]])
            ss = s[idx]
            ak = frame.scale_factors(k, ss)
            a[idx] = ak
            u[idx] = frame.velocity(k, ss, v_los[idx])
            if hubble_flow == "linear":
                rate = frame.first_hubble_rate(k) if aH_ref is None else float(aH_ref)
                VH[idx] = frame.first_hubble_velocity(k) + rate * frame.hubble_coordinate(k, ss)
            else:
                a_d = frame.a_dagger if a_sk is None else float(np.asarray(a_sk)[k])
                VH[idx] = phys.c_kms * np.log(a_d / ak)
    beta = -u / phys.c_kms
    return VH + phys.c_kms * 0.5 * (np.log1p(-beta) - np.log1p(beta)), a


def crossing_contributions(
    crossings,
    cells,
    line_bundle,
    frame,
    directions,
    *,
    a_eff=None,
    b_constants: str = "codata",
    hubble_flow: str = "exact",
    aH_ref=None,
    paths=None,
):
    """The contributions of the cell crossings of straight paths to every line of ``line_bundle``
    (``core.tau_engine.LineContributions``, host float64), from ``core.crossings.Crossings`` (cell indices into
    ``cells``, a ``core.Cells``) and the ``frame`` of the paths.

    Every crossing [s_lo, s_hi] of cell c absorbs over the line centres v_c(s) of its path, from v_c(s_lo) to v_c(s_hi):
    the centre of that span, its extent W, the amplitude C f n int a(s)^-2 ds (Simpson's rule; ``a_eff``: a_eff^-2 l) and
    the logarithmic gradient of the weight per unit line centre, a^-2 ds / dv_c, along the span
    (``LineContributions.tilt``: the comoving-to-physical factor and the curvature of the Hubble flow), with n, T and the peculiar
    velocity of the cell (its component along ``directions`` (N_sk, 3), the unit vectors of the paths).
    ``paths``: optional (N_sk,) indices of the rows of the result (default: the crossing's path).

    Returns a list of ``LineContributions``, one per line.
    """
    from skewerjax.core.tau_engine import LineContributions, line_widths

    path = crossings.path
    cell = crossings.cell
    lo, hi = crossings.s_lo, crossings.s_hi
    mid = 0.5 * (lo + hi)
    length = hi - lo
    b2 = check_b_constants(b_constants)
    out = [None] * line_bundle.n_lines
    lam, A21, mass, f_osc = (
        np.asarray(x, dtype=np.float64)
        for x in (line_bundle.lambda0_A, line_bundle.A21_GHz, line_bundle.mass_amu, line_bundle.f_osc)
    )
    # the geometry of the line centres is the same for every species (the cells' velocity): computed once
    sp0 = list(line_bundle.species_keys)[0]
    vel = np.asarray(cells.species[sp0]["velocity"], dtype=np.float64)
    v_los = np.einsum("ij,ij->i", vel[cell], np.asarray(directions, dtype=np.float64)[path])
    n3 = path.size
    vc3, a3 = line_centres(
        frame, np.tile(path, 3), np.concatenate([lo, mid, hi]), np.tile(v_los, 3), hubble_flow=hubble_flow, aH_ref=aH_ref
    )
    (v_lo, v_mid, v_hi), (a_lo, a_mid, a_hi) = np.split(vc3, 3), np.split(a3, 3)
    del vc3, a3
    # the line centres of the crossing span [v_lo, v_hi]: centre and extent; the weight per unit line centre,
    # a(s)^-2 ds / dv_c (the quadratic through the three points gives dv_c/ds at the ends), varies along it: its
    # logarithmic gradient is the tilt (``core.tau_engine``); the column is Simpson's int a^-2 ds
    vc = 0.5 * (v_lo + v_hi)
    W = np.abs(v_hi - v_lo)
    dv = v_hi - v_lo
    with np.errstate(divide="ignore", invalid="ignore"):
        slope_lo = np.abs(-3.0 * v_lo + 4.0 * v_mid - v_hi) / length
        slope_hi = np.abs(v_lo - 4.0 * v_mid + 3.0 * v_hi) / length
        if a_eff is None:
            a2 = (a_lo**-2 + 4.0 * a_mid**-2 + a_hi**-2) / 6.0
            w_lo, w_hi = a_lo**-2 / slope_lo, a_hi**-2 / slope_hi
        else:
            a2 = np.full(n3, float(a_eff) ** -2)
            w_lo, w_hi = 1.0 / slope_lo, 1.0 / slope_hi
        tilt = np.where((dv != 0) & (w_lo > 0) & (w_hi > 0), np.log(w_hi / w_lo) / dv, 0.0)
    colgeo = length * phys.Mpc_to_m * a2
    row = path if paths is None else np.asarray(paths)[path]
    for sp in line_bundle.species_keys:
        n_c, T_c, _, turb, coll = _species_values(cells, sp)
        col = n_c[cell] * colgeo
        T = T_c[cell] * b2
        tb = np.zeros(cell.size) if turb is None else turb[cell]
        co = np.zeros(cell.size) if coll is None else coll[cell]
        for j in line_bundle.indices_for_species(sp):
            sig, gam = line_widths(T, tb, co, lam[j], A21[j], mass[j])
            amp = (phys.cross_section_prefactor * 1e2) * f_osc[j] * col
            out[j] = LineContributions(row, vc, W, amp, sig, np.ascontiguousarray(gam), cell, tilt)
    return out


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
        dens = ab.n[k][live] * phys.Mpc_to_cm / ab.a_data**2
        j0, j1 = np.floor(lo / dz).astype(np.int64), np.floor(hi / dz).astype(np.int64)
        n = j1 - j0 + 1
        cell = np.repeat(np.arange(lo.size), n)
        j = np.repeat(j0, n) + (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n))
        length = np.minimum(hi[cell], (j + 1) * dz) - np.maximum(lo[cell], j * dz)
        keep = length > 0
        out[k] = np.bincount(np.mod(j[keep], nbins), weights=dens[cell[keep]] * length[keep], minlength=nbins)
    return out


def density_per_pixel(ab: Absorbers, path_length: float, nbins: int) -> np.ndarray:
    """(N_sk, nbins) mean physical number density [cm^-3] at the data epoch in each of ``nbins`` equal position pixels:
    ``column_per_pixel`` over the physical pixel length a_data L / nbins. On a position pixel, velocity pixels of width
    dv = (a H) L / nbins (fake_spectra's ``Spectra.get_density``, ``spectra.py`` L912-L916)."""
    return column_per_pixel(ab, path_length, nbins) / (path_length / nbins * ab.a_data * phys.Mpc_to_cm)


# --------------------------------------------------------------------------- the route: cells -> pieces
DEPOSITIONS = ("voronoi", "voronoi_segments", "tophat", "voronoi_crossings", "tophat_chords")


def deposit(
    cells,
    points,
    *,
    deposition: str = "voronoi",
    grid_fast_path: bool = True,
    chord_weight: float | None = None,
    return_info: bool = False,
):
    """Selection and deposition of ``cells`` along ``points`` (N_sk, N, 3) [cMpc]: (the cells the indices refer to,
    ``SegmentComponents``).

    deposition : "voronoi" or "voronoi_segments" (the exact pieces of the Voronoi cells in every dual segment; for a grid
        of cells, ``grid_segments``, unless ``grid_fast_path=False``), "tophat" (spheres of radius V^(1/3), the fake_spectra
        option: their exact overlaps with the dual segments), "tophat_chords" (those spheres through fake_spectra's
        quadrature of 7 points per chord, ``tophat_chord_segments``: straight paths) or "voronoi_crossings" (the
        Voronoi pieces, joined per cell crossing by ``cell_pieces``).
    chord_weight : "tophat_chords" only: the weight of each chord point (``tophat_chord_segments``; None: 1/7, the
        column kept; 1/8: fake_spectra's, 7/8 of the column).
    return_info : also return dict(selection (``select_cells``'s info; for the grid fast path every cell), deposition
        (kind, the ``SegmentComponents.diagnostics``, time_s)).
    """
    if deposition not in DEPOSITIONS:
        raise ValueError(f"deposition must be one of {DEPOSITIONS}, got {deposition!r}.")
    pts = np.asarray(points, dtype=np.float64)
    t0 = time.perf_counter()
    if chord_weight is not None and deposition != "tophat_chords":
        raise ValueError(f"chord_weight applies to deposition='tophat_chords', not {deposition!r}.")
    deposition = "voronoi" if deposition == "voronoi_segments" else deposition
    spheres = deposition in ("tophat", "tophat_chords")
    if not spheres and grid_fast_path and cells.grid is not None and cells.box is not None:
        use, sel = cells, dict(n_cells=len(cells), n_selected=len(cells), indices=np.arange(len(cells)), mode="grid", time_s=0.0)
        seg = dep.grid_segments(pts, cells.grid, cells.box)
    else:
        use, sel = select_cells(cells, pts, mode="tophat" if spheres else "voronoi", return_info=True)
        t0 = time.perf_counter()
        if deposition == "tophat":
            seg = dep.tophat_segments(pts, use.positions, use.size_cMpc, use.box)
        elif deposition == "tophat_chords":
            seg = dep.tophat_chord_segments(pts, use.positions, use.size_cMpc, use.box, chord_weight=chord_weight)
        else:
            seg = dep.voronoi_segments(pts, use.positions, use.box, gas_size=use.size_cMpc)
    info = dict(selection=sel, deposition=dict(kind=seg.kind, **seg.diagnostics, time_s=time.perf_counter() - t0))
    logger.debug(
        "deposit: %s pieces of %d dual segments from %d cells (%s), %.3f s",
        seg.kind,
        int(np.prod(pts.shape[:2])),
        len(use),
        ", ".join(f"{k}={v}" for k, v in seg.diagnostics.items()),
        info["deposition"]["time_s"],
    )
    return (use, seg, info) if return_info else (use, seg)


def _bundle_host(skewer_bundle, ii_cosmology: int = 0):
    """(points (N_sk, N, 3) [cMpc], a (N_sk, N)) of a ``SkewerBundle`` in float64: its host copies (``SkewerBundle.host``)
    when it has them, else its arrays (in the precision they hold)."""
    h = skewer_bundle.host
    if h is not None:
        return h["points"], h["scale_factors"][ii_cosmology]
    return (
        np.asarray(skewer_bundle.spatial_points, dtype=np.float64),
        np.asarray(skewer_bundle.scale_factors[ii_cosmology], dtype=np.float64),
    )


def bundle_frame(skewer_bundle, *, a_dagger=None, ii_cosmology: int = 0) -> ExactFrame:
    """The ``ExactFrame`` of a ``SkewerBundle``: a(s) interpolated in ln a between its points (where a(x) was solved),
    its photon directions, and a_dagger (default: the bundle's ``a0``; a scalar, or (N_sk,) per skewer)."""
    pts, a = _bundle_host(skewer_bundle, ii_cosmology)
    s = path_geometry(pts)["s"]
    # a_dagger in float64 (the host copy): rounded to float32 it would shift every line centre by c x 6e-8 ~ 0.02 km/s
    h = skewer_bundle.host
    a0 = skewer_bundle.a0 if h is None or "a0" not in h else h["a0"]
    a_d = np.asarray(a0 if a_dagger is None else a_dagger, dtype=np.float64)
    if a_d.ndim > 1 or (a_d.ndim == 1 and a_d.size not in (1, pts.shape[0])):
        raise ValueError(f"a_dagger must be a scalar or one value per skewer ({pts.shape[0]},), got shape {a_d.shape}.")
    per = None if a_d.size <= 1 or np.all(a_d == a_d.ravel()[0]) else np.broadcast_to(a_d, (pts.shape[0],)).copy()
    return ExactFrame(
        s,
        a,
        a_dagger_value=float(np.ravel(a_d)[0]),
        photon_sign=np.asarray(skewer_bundle.array_direction),
        a_dagger_per_skewer=per,
    )


def _crossing_pieces(cells, line_bundle, pts, frame, species, a_data, skewers, return_parts, return_info, t0):
    """``cell_pieces`` for straight paths with ``deposition="voronoi"``: one piece per cell crossing."""
    from dataclasses import replace

    from skewerjax.core import crossings as cx

    paths = cx.paths_from_points(pts, cells.box)
    cr = cx.straight_crossings(
        paths, np.asarray(cells.positions, dtype=np.float64), np.asarray(cells.size_cMpc, dtype=np.float64)
    )
    t1 = time.perf_counter()
    mass = {sp: float(np.asarray(line_bundle.mass_amu)[line_bundle.indices_for_species(sp)[0]]) for sp in species}
    s_pt = path_geometry(pts)["s"]
    mid = 0.5 * (cr.s_lo + cr.s_hi)
    abs_ = {}
    for sp in species:
        n_c, T_c, vel, turb, coll = _species_values(cells, sp)
        rows = _AbsorberRows(turb, coll)
        for k in range(cr.n_paths):
            sl = slice(cr.start[k], cr.start[k + 1])
            c = cr.cell[sl]
            pt = np.clip(np.searchsorted(0.5 * (s_pt[k][1:] + s_pt[k][:-1]), mid[sl]), 0, s_pt.shape[1] - 1)
            rows.add(
                c, s=mid[sl], n=n_c[c], length=cr.s_hi[sl] - cr.s_lo[sl], T=T_c[c], v_los=vel[c] @ paths.direction[k], point=pt
            )
        abs_[sp] = rows.build(sp, mass[sp], a_data, dict(kind="voronoi_crossings_exact"))
    t2 = time.perf_counter()
    per = {sp: absorber_pieces(abs_[sp], frame, 1, skewers=skewers) for sp in species}
    p0 = per[species[0]]
    n_l = line_bundle.n_lines
    fields = ("temperature", "turbulence", "collisional", "los_velocity", "number_density", "width_kms")
    rows_ = {f: [None] * n_l for f in fields}
    for sp in species:
        for j in line_bundle.indices_for_species(sp):
            for f in fields:
                rows_[f][j] = getattr(per[sp], f)[0]
    pieces = replace(p0, **{f: jnp.stack(v) for f, v in rows_.items()})
    out = [pieces]
    if return_parts:
        out.append(dict(cells=cells, segments=None, absorbers=abs_, frame=frame, crossings=cr, paths=paths))
    if return_info:
        t3 = time.perf_counter()
        ks = list(range(pts.shape[0])) if skewers is None else [int(k) for k in np.atleast_1d(skewers)]
        A0 = abs_[species[0]]
        out.append(
            dict(
                selection=None,
                deposition=dict(kind="voronoi_crossings_exact", n_crossings=cr.counts, time_s=t1 - t0),
                n_pieces=(A0.cell[ks] >= 0).sum(-1),
                path_length_cMpc=paths.length[ks],
                piece_length_cMpc=A0.length[ks].sum(-1),
                column_cm2={sp: (abs_[sp].n[ks] * abs_[sp].length[ks]).sum(-1) * phys.Mpc_to_cm for sp in species},
                resolution=None,
                timings=dict(deposit=t1 - t0, absorbers=t2 - t1, pieces=t3 - t2),
                time_s=t3 - t0,
            )
        )
    return out[0] if len(out) == 1 else tuple(out)


def cell_pieces(
    cells,
    line_bundle,
    *,
    skewer_bundle=None,
    points=None,
    frame=None,
    a_dagger=None,
    ii_cosmology: int = 0,
    deposition: str = "voronoi",
    segments=None,
    grid_fast_path: bool = True,
    crossing_tolerance: float | None = 0.25,
    chord_weight: float | None = None,
    a_data: float | None = None,
    skewers=None,
    return_parts: bool = False,
    return_info: bool = False,
):
    """The pieces of the optical-depth sum for every line of ``line_bundle``, from ``cells`` along a set of paths.

    The one route from the gas to tau, for every source of cells (toy models, simulations):

    1. ``deposition="voronoi"`` (default) on straight paths: the **cell crossings** of every path, one piece per crossing
       (``core.crossings.straight_crossings``: the lower envelope of the generators within reach of the path; no
       selection, no sampling points); on curved paths, and with ``"voronoi_segments"`` on any path: selection of the
       cells within reach of the points and deposition into the pieces of every dual segment (``deposit``; for
       ``deposition="voronoi_crossings"`` the dual-segment pieces are joined per cell crossing and split into equal
       sub-pieces of velocity width <= ``crossing_tolerance`` b of the narrowest species);
    2. per species, the ``Absorbers`` of its cells (``absorbers_from_components`` / ``absorbers_from_crossings``);
    3. ``absorber_pieces`` in the frame: a(s) at every centroid and the velocity along the photon.

    Every species shares the same pieces (one deposition), so a line's fields are those of its species.

    Parameters
    ----------
    cells : ``core.Cells`` holding every species of ``line_bundle``.
    skewer_bundle : the paths and their a(x) (``bundle_frame``: a_dagger default its ``a0``); or give ``points``
        (N_sk, N, 3) [cMpc] and a ``frame`` (``ExactFrame`` / ``SnapshotFrame``) instead.
    deposition, grid_fast_path : see ``deposit``; segments : a ``SegmentComponents`` already
        computed for ``points`` (its cell indices refer to ``cells``), used instead.
    chord_weight : ``deposition="tophat_chords"`` only, see ``deposit``.
    a_data : the scale factor of the data (kept on the ``Absorbers``).
    skewers : optional indices of the skewers to return.
    return_parts : also return dict(cells, segments, absorbers {species: Absorbers}, frame, crossings).
    return_info : also return dict(selection, deposition (``deposit``'s info; selection None when ``segments`` are
        given), n_pieces (N_sk,) live pieces, path_length_cMpc (N_sk,), piece_length_cMpc (N_sk,) their sum (= the path
        length), column_cm2 {species: (N_sk,)} comoving column sum_p n_p l_p, resolution (``resolution_report``),
        timings {deposit, absorbers, pieces} [s], time_s).

    Returns
    -------
    ``core.Pieces`` (N_lines, N_sk, N_p); with ``return_parts`` and / or ``return_info``: (pieces, parts, info), in
    that order, each only when asked for.
    """
    t0 = time.perf_counter()
    from dataclasses import replace

    if skewer_bundle is not None:
        points = _bundle_host(skewer_bundle, ii_cosmology)[0]
        frame = bundle_frame(skewer_bundle, a_dagger=a_dagger, ii_cosmology=ii_cosmology) if frame is None else frame
    elif points is None or frame is None:
        raise ValueError("give a skewer_bundle, or points and a frame.")
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 3 or pts.shape[-1] != 3 or pts.shape[1] < 2:
        raise ValueError(f"points must be (N_sk, N >= 2, 3) [cMpc], got {pts.shape}.")
    if deposition not in DEPOSITIONS:
        raise ValueError(f"deposition must be one of {DEPOSITIONS}, got {deposition!r}.")
    species = list(line_bundle.species_keys)
    missing = [sp for sp in species if sp not in cells.species]
    if missing:
        raise ValueError(f"the cells hold no {missing} (they hold {cells.species_keys}).")
    if segments is not None and segments.cell.shape[:2] != pts.shape[:2]:
        raise ValueError(f"segments were deposited for {segments.cell.shape[:2]} points, the paths have {pts.shape[:2]}.")
    from skewerjax.core import crossings as cx

    if chord_weight is not None and deposition != "tophat_chords":
        raise ValueError(f"chord_weight applies to deposition='tophat_chords', not {deposition!r}.")
    if deposition == "voronoi" and segments is None and bool(np.all(cx.straight_paths_mask(pts))):
        return _crossing_pieces(cells, line_bundle, pts, frame, species, a_data, skewers, return_parts, return_info, t0)
    if segments is None:
        use, seg, dinfo = deposit(
            cells,
            pts,
            deposition="voronoi" if deposition in ("voronoi_crossings", "voronoi_segments") else deposition,
            grid_fast_path=grid_fast_path,
            chord_weight=chord_weight,
            return_info=True,
        )
    else:
        use, seg, dinfo = cells, segments, dict(selection=None, deposition=dict(kind=segments.kind, **segments.diagnostics))
    t1 = time.perf_counter()
    mass = {sp: float(np.asarray(line_bundle.mass_amu)[line_bundle.indices_for_species(sp)[0]]) for sp in species}
    cross = None
    if deposition == "voronoi_crossings":
        cross = dep.cell_crossings(seg, pts)
        per_sp = [
            crossing_splits(cross, np.asarray(use.species[sp]["temperature"]), mass[sp], frame, crossing_tolerance)
            for sp in species
        ]
        split = [np.max(np.stack([q[k] for q in per_sp]), axis=0) for k in range(len(per_sp[0]))]
        abs_ = {sp: absorbers_from_crossings(cross, pts, use, sp, mass[sp], a_data=a_data, split=split) for sp in species}
    else:
        abs_ = {sp: absorbers_from_components(seg, pts, use, sp, mass[sp], a_data=a_data) for sp in species}
    t2 = time.perf_counter()
    per = {sp: absorber_pieces(abs_[sp], frame, 1, skewers=skewers) for sp in species}
    p0 = per[species[0]]
    n_l = line_bundle.n_lines
    rows = {f: [None] * n_l for f in ("temperature", "turbulence", "collisional", "los_velocity", "number_density", "width_kms")}
    for sp in species:
        for j in line_bundle.indices_for_species(sp):
            for f in rows:
                rows[f][j] = getattr(per[sp], f)[0]
    pieces = replace(p0, **{f: jnp.stack(v) for f, v in rows.items()})
    if seg.kind == "tophat_chords":  # fake_spectra's chord points are points: their length is a quadrature weight
        z = jnp.zeros_like(pieces.length_m)
        pieces = replace(pieces, width_kms=jnp.zeros_like(pieces.width_kms), tilt_a2=z, tilt_flow=z)
    out = [pieces]
    if return_parts:
        out.append(dict(cells=use, segments=seg, absorbers=abs_, frame=frame, crossings=cross))
    if return_info or logger.isEnabledFor(logging.DEBUG):
        t3 = time.perf_counter()
        ks = list(range(pts.shape[0])) if skewers is None else [int(k) for k in np.atleast_1d(skewers)]
        A0 = abs_[species[0]]
        info = dict(
            **dinfo,
            n_pieces=(A0.cell[ks] >= 0).sum(-1),
            path_length_cMpc=np.linalg.norm(np.diff(pts[ks], axis=1), axis=-1).sum(-1),
            piece_length_cMpc=A0.length[ks].sum(-1),
            column_cm2={sp: (abs_[sp].n[ks] * abs_[sp].length[ks]).sum(-1) * phys.Mpc_to_cm for sp in species},
            resolution=resolution_report(pieces, line_bundle),
            timings=dict(deposit=t1 - t0, absorbers=t2 - t1, pieces=t3 - t2),
            time_s=t3 - t0,
        )
        logger.debug(
            "cell_pieces: %s, %d skewers, %s pieces, %.3f s (deposit %.3f, absorbers %.3f, pieces %.3f)",
            deposition,
            len(ks),
            pieces.shape[-1],
            info["time_s"],
            *info["timings"].values(),
        )
        if return_info:
            out.append(info)
    return out[0] if len(out) == 1 else tuple(out)


def cell_contributions(
    cells,
    line_bundle,
    *,
    skewer_bundle=None,
    points=None,
    frame=None,
    a_dagger=None,
    ii_cosmology: int = 0,
    a_eff=None,
    b_constants: str = "codata",
    hubble_flow: str = "exact",
    aH_ref=None,
    return_info: bool = False,
):
    """The contributions (``core.tau_engine.LineContributions``, one per line, host float64) of the Voronoi cells of
    ``cells`` crossed by the straight paths of ``skewer_bundle`` (or ``points`` (N_sk, N, 3) [cMpc] and a ``frame``):
    the default route of ``core.compute_optical_depth`` for straight paths. The crossings come from the lower envelope
    of the candidate generators within reach of each path (``core.crossings.straight_crossings``: no selection, no
    sampling points: only the ends of each path and its frame are used); each crossing is one contribution
    (``crossing_contributions``). With ``return_info``: (contributions, dict(n_crossings (N_sk,), n_candidates,
    n_cells, timings {crossings, contributions} [s], time_s))."""
    from skewerjax.core import crossings as cx

    t0 = time.perf_counter()
    if skewer_bundle is not None:
        points = _bundle_host(skewer_bundle, ii_cosmology)[0]
        frame = bundle_frame(skewer_bundle, a_dagger=a_dagger, ii_cosmology=ii_cosmology) if frame is None else frame
    elif points is None or frame is None:
        raise ValueError("give a skewer_bundle, or points and a frame.")
    species = list(line_bundle.species_keys)
    missing = [sp for sp in species if sp not in cells.species]
    if missing:
        raise ValueError(f"the cells hold no {missing} (they hold {cells.species_keys}).")
    paths = cx.paths_from_points(points, cells.box)
    cr, pairs = cx.straight_crossings(
        paths, np.asarray(cells.positions, dtype=np.float64), np.asarray(cells.size_cMpc, dtype=np.float64), return_pairs=True
    )
    t1 = time.perf_counter()
    out = crossing_contributions(
        cr,
        cells,
        line_bundle,
        frame,
        paths.direction,
        a_eff=a_eff,
        b_constants=b_constants,
        hubble_flow=hubble_flow,
        aH_ref=aH_ref,
    )
    t2 = time.perf_counter()
    if not return_info:
        return out
    info = dict(
        n_crossings=cr.counts,
        n_candidates=int(pairs.cell.size),
        n_cells=int(len(cells)),
        n_cells_crossed=int(np.unique(cr.cell).size),
        crossings=cr,
        timings=dict(crossings=t1 - t0, contributions=t2 - t1),
        time_s=t2 - t0,
    )
    return out, info


def chord_contributions(
    chords,
    cells,
    line_bundle,
    frame,
    directions,
    lengths,
    box,
    *,
    chord_weight: float | None = None,
    a_eff=None,
    b_constants: str = "codata",
    hubble_flow: str = "exact",
    aH_ref=None,
    compact: bool = True,
):
    """The contributions of fake_spectra's chord quadrature (``core.crossings.Chords.points``) to every line: every
    chord point a point piece (width 0) of density (3 / 4 pi) n and length 2 z w, at its line centre in ``frame``
    (fake_spectra's ``SnapshotFrame``: velfac s + v_pec), grouped by cell (``LineContributions.group``: the per-cell
    schemes). A path across the whole periodic box keeps the points beyond its ends (the spectrum wraps them with
    ``velocity_period``); a shorter path drops them (``core.deposition.tophat_chord_segments``). ``compact`` (paths across
    the box): one contribution per chord, its 7 points at vc + (k - 4) spacing generated on the device
    (``LineContributions.spacing``; the same sum)."""
    from skewerjax.core.deposition import TOPHAT_NORM
    from skewerjax.core.tau_engine import LineContributions, line_widths

    box_paths = bool(box) and bool(np.allclose(np.asarray(lengths), box, rtol=1e-9))
    if compact and box_paths:  # one contribution per chord: its 7 points are generated by the per-cell kernel
        return _compact_chords(
            chords, cells, line_bundle, frame, directions, chord_weight, a_eff, b_constants, hubble_flow, aH_ref
        )
    idx, s, ln = chords.points(chord_weight)
    path = chords.path[idx]
    cell = chords.cell[idx]
    L = np.asarray(lengths, dtype=np.float64)[path]
    if not (box and np.allclose(np.asarray(lengths), box, rtol=1e-9)):
        inside = (s >= 0.0) & (s <= L)
        idx, s, ln, path, cell = idx[inside], s[inside], ln[inside], path[inside], cell[inside]
    b2 = check_b_constants(b_constants)
    lam, A21, mass, f_osc = (
        np.asarray(x, dtype=np.float64)
        for x in (line_bundle.lambda0_A, line_bundle.A21_GHz, line_bundle.mass_amu, line_bundle.f_osc)
    )
    out = [None] * line_bundle.n_lines
    for sp in line_bundle.species_keys:
        n_c, T_c, vel, turb, coll = _species_values(cells, sp)
        v_los = np.einsum("ij,ij->i", vel[cell], np.asarray(directions, dtype=np.float64)[path])
        vc, a = line_centres(frame, path, s, v_los, hubble_flow=hubble_flow, aH_ref=aH_ref)
        a2 = a**-2 if a_eff is None else float(a_eff) ** -2
        col = TOPHAT_NORM * n_c[cell] * ln * phys.Mpc_to_m * a2
        T = T_c[cell] * b2
        tb = np.zeros(cell.size) if turb is None else turb[cell]
        co = np.zeros(cell.size) if coll is None else coll[cell]
        for j in line_bundle.indices_for_species(sp):
            sig, gam = line_widths(T, tb, co, lam[j], A21[j], mass[j])
            amp = (phys.cross_section_prefactor * 1e2) * f_osc[j] * col
            out[j] = LineContributions(path, vc, np.zeros(cell.size), amp, sig, np.ascontiguousarray(gam), cell)
    return out


def _compact_chords(chords, cells, line_bundle, frame, directions, chord_weight, a_eff, b_constants, hubble_flow, aH_ref):
    """``chord_contributions`` with one contribution per chord: the line centre of its middle point, the spacing of its 7
    points in line centre (from its ends, s = u -+ 3 z / 4) and its 7 points' total amplitude."""
    from skewerjax.core.crossings import CHORD_POINTS
    from skewerjax.core.deposition import TOPHAT_NORM
    from skewerjax.core.tau_engine import LineContributions, line_widths

    w = 1.0 / CHORD_POINTS if chord_weight is None else float(chord_weight)
    path, cell = chords.path, chords.cell
    sp0 = list(line_bundle.species_keys)[0]
    vel = np.asarray(cells.species[sp0]["velocity"], dtype=np.float64)
    v_los = np.einsum("ij,ij->i", vel[cell], np.asarray(directions, dtype=np.float64)[path])
    n = path.size
    s3 = np.concatenate([chords.u - 0.75 * chords.z, chords.u, chords.u + 0.75 * chords.z])
    vc3, a3 = line_centres(frame, np.tile(path, 3), s3, np.tile(v_los, 3), hubble_flow=hubble_flow, aH_ref=aH_ref)
    v_lo, vc, v_hi = vc3[:n], vc3[n : 2 * n], vc3[2 * n :]
    a_mid = a3[n : 2 * n]
    spacing = (v_hi - v_lo) / 6.0
    a2 = a_mid**-2 if a_eff is None else float(a_eff) ** -2
    colgeo = TOPHAT_NORM * CHORD_POINTS * 2.0 * chords.z * w * phys.Mpc_to_m * a2
    b2 = check_b_constants(b_constants)
    lam, A21, mass, f_osc = (
        np.asarray(x, dtype=np.float64)
        for x in (line_bundle.lambda0_A, line_bundle.A21_GHz, line_bundle.mass_amu, line_bundle.f_osc)
    )
    out = [None] * line_bundle.n_lines
    for sp in line_bundle.species_keys:
        n_c, T_c, _, turb, coll = _species_values(cells, sp)
        col = n_c[cell] * colgeo
        T = T_c[cell] * b2
        tb = np.zeros(n) if turb is None else turb[cell]
        co = np.zeros(n) if coll is None else coll[cell]
        for j in line_bundle.indices_for_species(sp):
            sig, gam = line_widths(T, tb, co, lam[j], A21[j], mass[j])
            amp = (phys.cross_section_prefactor * 1e2) * f_osc[j] * col
            out[j] = LineContributions(path, vc, np.zeros(n), amp, sig, np.ascontiguousarray(gam), cell, None, spacing)
    return out
