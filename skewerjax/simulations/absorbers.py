"""Absorbers (layer 6): the deposited pieces of every skewer, their frame a(s), and tau through the core kernel.

What the optical-depth integral needs (notes ``sec_flux.tex`` Eq. tau_straight_los) is, along the path, the
comoving density of the absorbing ion, its temperature, its *peculiar* velocity and the line element. The Hubble
flow is not a field: it enters through the scale factor a(s) of every point, set by the frame.

``Absorbers`` flattens the components of all sampling points of a skewer (cell pieces, sphere overlaps, ...) into
one list (path coordinate s, density, piece length, T, line-of-sight peculiar velocity). ``absorber_samples`` turns
the absorbers of one skewer into a ``core.LoSSamples`` (fixed-size chunks: static shapes) and ``tau_absorbers``
integrates them with ``core.optical_depth_from_samples``, the pieces' lengths as line elements:

    tau(v) = sum_p  C f  n_com,p a_p^-2  phi_p(v)  l_p .

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
from skewerjax.core.optical_depth import LoSSamples, optical_depth_from_samples, optical_depth_kernel
from skewerjax.core.skewers import path_geometry

MPC_CM = 3.085678e24


# --------------------------------------------------------------------------- frames
@dataclass
class SnapshotFrame:
    """fake_spectra's frame: a = a_snap everywhere, linear Hubble flow velfac * s [km/s]."""
    a_snap: float
    velfac: float                      # [km/s per cMpc] = a H(z)
    period_kms: Optional[float] = None
    s_origin: float = 0.0              # path coordinate of the box origin (u = 0) [cMpc]

    def scale_factors(self, skewer: int, s):
        return np.full(np.shape(s), self.a_snap)

    def velocity(self, skewer: int, s, v_los):
        """Line-of-sight velocity that puts the line centre at velfac (s - s_origin) + v_pec exactly."""
        return phys.c_kms * np.tanh((self.velfac * (np.asarray(s) - self.s_origin) + np.asarray(v_los)) / phys.c_kms)

    @property
    def a_dagger(self):
        return self.a_snap

    def hubble_rate(self, skewer: int, s):
        """dv/ds of the frame [km/s per cMpc]."""
        return np.full(np.shape(s), self.velfac)


@dataclass
class ExactFrame:
    """a(s) from the cosmology along each path (notes ``app_velocity_units.tex``).

    Parameters
    ----------
    s_grid : (N_sk, N) path coordinates [cMpc] of the sampling points (increasing).
    a_grid : (N_sk, N) scale factors there (``scale_factors_along_paths``).
    a_dagger : reference scale factor of the spectral coordinate (default a(s = 0) of skewer 0).
    period_kms : velocity period of a periodic box (None: no wrap).
    """
    s_grid: np.ndarray
    a_grid: np.ndarray
    a_dagger_value: Optional[float] = None
    period_kms: Optional[float] = None

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


def absorbers_from_point_values(points, pv: dict, a_data: float, species: str, mass_amu: float) -> Absorbers:
    """Absorbers of a point-valued deposition (``tophat_point_values``): one per point, with the
    trapezoid line element of the point (the previous "points" mode)."""
    g = path_geometry(jnp.asarray(points, dtype=jnp.float64))
    n_sk, n_p = np.shape(pv["n"])
    return Absorbers(s=np.asarray(g["s"]), n=np.asarray(pv["n"], float), length=np.asarray(g["line_elements"]),
                     T=np.asarray(pv["T"], float), v_los=np.asarray(pv["v_los"], float),
                     cell=-np.ones((n_sk, n_p), int), point=np.broadcast_to(np.arange(n_p), (n_sk, n_p)).copy(),
                     a_data=float(a_data), species=species, mass_amu=float(mass_amu), meta=dict(kind="points"))


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
def absorber_samples(ab: Absorbers, frame, skewer: int, n_lines: int, *, density_scaling: str = "comoving", T_scale: float = 1.0,
                     n_scale: float = 1.0, chunk: int = 2048) -> LoSSamples:
    """The absorbers of one skewer as a ``core.LoSSamples`` of ``nchunk`` rows of ``chunk`` points (padded with n = 0).

    Every absorber is one point: at its path coordinate s, with the frame's a(s), the velocity the frame injects
    along the path (photons travel towards decreasing s), T x ``T_scale``, the comoving density
    n_com = n a_data^3 x ``n_scale`` (``density_scaling="fixed_column"``: n a_data a(s)^2) and its length as line
    element. ``T_scale`` and ``n_scale`` are matched-mode knobs (fake_spectra's b constants and 7/8 chords).
    """
    M = ab.s.shape[1]
    nchunk = -(-M // chunk)
    pad = nchunk * chunk - M
    P = lambda x, f: np.concatenate([x[skewer], np.full(pad, f)]).reshape(nchunk, chunk)
    s, n, l, T, vl = P(ab.s, 0.0), P(ab.n, 0.0), P(ab.length, 0.0), P(ab.T, 1e4), P(ab.v_los, 0.0)
    a = frame.scale_factors(skewer, s)
    n_com = n * n_scale * (ab.a_data ** 3 if density_scaling == "comoving" else ab.a_data * a ** 2)
    u = frame.velocity(skewer, s, vl)
    lines = lambda x: jnp.broadcast_to(jnp.asarray(x)[None], (n_lines,) + x.shape)
    zeros = np.zeros(T.shape)
    return LoSSamples(a_dagger=jnp.full((nchunk, chunk), frame.a_dagger), scale_factor=jnp.asarray(a),
                      weights_m=jnp.asarray(l) * phys.Mpc_to_m, temperature=lines(T * T_scale), turbulence=lines(zeros),
                      collisional=lines(zeros), los_velocity=lines(-np.asarray(u)), number_density=lines(n_com))


def tau_absorbers(ab: Absorbers, line_bundle, velocities_kms, frame, *, density_scaling: str = "comoving",
                  T_scale: float = 1.0, n_scale: float = 1.0, detuning: str = "exact",
                  pixel_offsets=None, pixel_weights=None, chunk: int = 2048, v_chunk: Optional[int] = 4096,
                  voigt: str = "ExoJAX", check: Optional[float] = 0.25, per_absorber: bool = False) -> np.ndarray:
    """tau(v) of every skewer from its absorbers.

    Parameters
    ----------
    line_bundle : LineBundle whose species is ``ab.species``.
    velocities_kms : (N_v,) spectral coordinate [km/s] (v_dagger in ``ExactFrame``; u in ``SnapshotFrame``).
    frame : ExactFrame or SnapshotFrame (sets a(s), the injected velocity, a_dagger and the period).
    density_scaling, T_scale, n_scale : see ``absorber_samples``.
    detuning : "exact" (default) or "linear" (matched mode).
    pixel_offsets, pixel_weights : sub-pixel offsets [km/s] and weights (sum 1) to average each pixel; None: centres.
    chunk : absorbers per row of the samples (static shape); v_chunk : velocities per kernel call.
    check : tolerance of ``sampling_check`` (None: no check).
    per_absorber : True returns (N_lines, N_sk, M, N_v) contributions (memory!).

    Returns
    -------
    (N_lines, N_sk, N_v) optical depth [float64].
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
    v_all = (v[:, None] + off[None, :]).ravel()
    n_lines, n_sk, M = line_bundle.n_lines, ab.n_skewers, ab.s.shape[1]
    nchunk = -(-M // chunk)
    out = np.zeros((n_lines, n_sk, nchunk * chunk if per_absorber else 1, v.size))
    for k in range(n_sk):
        smp = absorber_samples(ab, frame, k, n_lines, density_scaling=density_scaling, T_scale=T_scale, n_scale=n_scale, chunk=chunk)
        if per_absorber:
            t = _tau_per_absorber(smp, line_bundle, v_all, frame, detuning, voigt, v_chunk)
            out[:, k] = (t.reshape(n_lines, nchunk * chunk, v.size, off.size) * wts).sum(-1)
        else:
            t = np.asarray(optical_depth_from_samples(smp, jnp.asarray(v_all), line_bundle, voigt=voigt, detuning=detuning,
                                                      velocity_period=frame.period_kms, v_chunk=v_chunk)).sum(axis=1)
            out[:, k, 0] = (t.reshape(n_lines, v.size, off.size) * wts).sum(-1)
    return out if per_absorber else out[:, :, 0]


def _tau_per_absorber(smp: LoSSamples, line_bundle, v_all, frame, detuning, voigt, v_chunk):
    """(N_lines, N_points, N_v') contribution of every absorber (all rows of ``smp`` at once)."""
    n_l = line_bundle.n_lines
    flat = lambda x: x.reshape(n_l, -1)
    nv = len(v_all) if v_chunk is None else v_chunk
    parts = [optical_depth_kernel(jnp.asarray(v_all[v0:v0 + nv]), line_bundle.lambda0_A, line_bundle.A21_GHz, line_bundle.mass_amu,
                                  line_bundle.f_osc, frame.a_dagger, smp.scale_factor.reshape(-1), flat(smp.los_velocity),
                                  flat(smp.temperature), flat(smp.turbulence), flat(smp.collisional), flat(smp.number_density),
                                  smp.weights_m.reshape(-1), voigt=voigt, detuning=detuning, velocity_period=frame.period_kms,
                                  per_point=True)
             for v0 in range(0, len(v_all), nv)]
    return np.asarray(jnp.concatenate(parts, axis=-1))


# --------------------------------------------------------------------------- columns
def column_per_pixel(ab: Absorbers, path_length: float, nbins: int, s_origin: float = 0.0) -> np.ndarray:
    """(N_sk, nbins) physical column [cm^-2] of the absorbers binned exactly into ``nbins`` equal
    position pixels of a periodic path of length ``path_length`` [cMpc]. Each absorber is the
    interval [s - l/2, s + l/2] (a piece of one convex cell or sphere inside one segment)."""
    out = np.zeros((ab.n_skewers, nbins))
    dz = path_length / nbins
    for k in range(ab.n_skewers):
        live = ab.n[k] > 0
        lo = ab.s[k][live] - s_origin - 0.5 * ab.length[k][live]
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
