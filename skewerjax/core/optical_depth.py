"""Line profiles and the optical depth tau(v) along skewers (review notes, Eqs. line_profile, tau_straight_los).

The optical depth is a line integral along the path (x comoving, dl_phys = a dx),

    tau_j(v) = C_j f_j int n_com(x) a(x)^-2 phi_j(v; x) dx,   C_j = q_e^2 / (4 eps_0 m_e c),   F = exp(-tau),

evaluated as a sum over **pieces** p, fragments of the path within which the gas properties are constant:

    tau_j(v) = C_j f_j sum_p l_p n_com,p a_p^-2 phi_pj(v),

with every quantity of a piece, its scale factor a_p included, evaluated at the piece's centroid (the midpoint rule
on each piece). The gas is a set of Voronoi cells (``core.cells.Cells``, from a toy model or a simulation), constant
inside every cell, and the pieces are the exact intersections of the path with the cells, cut also at the ends of
the dual segments of the sampling points (``core.absorbers.cell_pieces``). So the column sum_p n_p l_p is exact; the
profile has an error O((W_p / b_p)^2), W_p the variation of the line centre across the piece (the Hubble flow only:
the peculiar velocity is constant in a cell).

The line profiles phi_pj come from ``core.line_profiles``. Layers (all pure; the last one only adds eager checks and
options):

* ``optical_depth_kernel``: the pieces of one skewer, arrays in, (N_lines, N_v) out (jit);
* ``optical_depth_from_pieces``: every skewer of a ``Pieces`` pytree, ``lax.map`` over skewers (jit);
* ``compute_optical_depth``: the entry point: the pieces (from a bundle and cells, or given), the resolution check,
  the spectral axis (each line's own frame, or the reference line's) and the one periodicity option of the package,
  ``velocity_period`` (default: none).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec
from jax.typing import ArrayLike

from skewerjax.core.constants import phys
from skewerjax.core.line_profiles import voigt_line_profiles, voigt_width_kms, voigt_widths_GHz
from skewerjax.core.spectral import frequency_offset_GHz, line_centre_velocity_kms


# --------------------------------------------------------------------------- the optical depth of one skewer
@partial(jax.jit, static_argnames=("voigt", "detuning", "per_piece"))
def optical_depth_kernel(velocities_kms, lambda0_A, A21_GHz, mass_amu, f_osc, a_dagger, scale_factor, los_velocity, temperature,
                         turbulence, collisional, number_density, length_m, voigt: str = "ExoJAX", detuning: str = "exact",
                         velocity_period=None, per_piece: bool = False):
    """tau = sum_p l_p C f n_p a_p^-2 phi_p(v) for the pieces of one skewer (Eq. tau_straight_los, as a sum over pieces).

    Parameters
    ----------
    velocities_kms : (N_v,) or (N_lines, N_v) (one grid per line); lambda0_A, A21_GHz [GHz], mass_amu, f_osc : (N_lines,)
    a_dagger : scalar or (N_p,); scale_factor : (N_p,) a at the piece centroids
    los_velocity [km/s], temperature [K], turbulence [m/s], collisional [GHz], number_density [comoving cm^-3] :
        (N_lines, N_p)
    length_m : (N_p,) piece lengths [comoving m]
    velocity_period : None (no wrap) or the period [km/s] (a number, possibly traced) of the detuning's nearest image.
    per_piece : True returns every piece's contribution (N_lines, N_p, N_v) instead of the sum.

    Returns
    -------
    (N_lines, N_v) optical depth.
    """
    sigma, gamma = voigt_widths_GHz(temperature, turbulence, collisional, lambda0_A, A21_GHz, mass_amu)
    nu = frequency_offset_GHz(velocities_kms, lambda0_A, a_dagger, scale_factor, los_velocity,
                              velocity_period=velocity_period, detuning=detuning)
    n_l, n_p, n_v = nu.shape
    phi = voigt_line_profiles(nu.reshape(n_l * n_p, n_v), sigma.reshape(-1), gamma.reshape(-1), voigt).reshape(n_l, n_p, n_v)
    # dtau/dx [1/m] = C f phi n a^-2 (x comoving); 1e2 turns cm^2 cm^-3 m into a number
    amp = phys.cross_section_prefactor * f_osc[:, None] * number_density * scale_factor[None, :] ** -2 * 1e2 * length_m[None, :]
    if per_piece:
        return amp[..., None] * phi
    return jnp.einsum("lp,lpv->lv", amp, phi)


# --------------------------------------------------------------------------- pieces
@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Pieces:
    """The pieces of N_sk skewers: what the optical-depth sum runs over (the boundary between field sources and tau).

    A piece is a fragment of the path within which the gas properties are constant; every quantity below is its value
    at the piece's centroid. Padding pieces have zero length and density.

    a_dagger, scale_factor : (N_sk, N_p) reference scale factor of v_dagger and the scale factor a at the centroid;
    length_m : (N_sk, N_p) comoving length [m];
    temperature [K], turbulence [m/s], collisional [GHz], los_velocity [km/s, along the photon direction],
    number_density [comoving cm^-3] : (N_lines, N_sk, N_p);
    width_kms : (N_lines, N_sk, N_p) the variation of the line centre across the piece [km/s] (``resolution_report``);
    velocity_span_kms : (N_sk,) the line-centre span of the whole path without peculiar velocities [km/s], from its
        first point to its last one (c |ln(a_first / a_last)|): the period of ``velocity_period=True``.
    JAX pytree (every field a leaf).
    """
    a_dagger: jnp.ndarray
    scale_factor: jnp.ndarray
    length_m: jnp.ndarray
    temperature: jnp.ndarray
    turbulence: jnp.ndarray
    collisional: jnp.ndarray
    los_velocity: jnp.ndarray
    number_density: jnp.ndarray
    width_kms: jnp.ndarray
    velocity_span_kms: jnp.ndarray

    @property
    def shape(self) -> tuple:
        """(N_lines, N_sk, N_p)."""
        return self.temperature.shape

    def rows(self, chunk: int) -> "Pieces":
        """The pieces of every skewer in rows of ``chunk`` (N_sk * n_rows, chunk), padded with empty pieces: summing a
        row's tau over the rows of one skewer gives the skewer's tau (bounds the memory of the kernel)."""
        n_sk, m = self.scale_factor.shape
        n_rows = -(-m // chunk)
        pad = n_rows * chunk - m

        def r(x, empty):
            lead = x.shape[:-2]
            x = jnp.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, pad)], mode="constant" if empty else "edge")
            return x.reshape(lead + (n_sk * n_rows, chunk))

        return Pieces(a_dagger=r(self.a_dagger, False), scale_factor=r(self.scale_factor, False), length_m=r(self.length_m, True),
                      temperature=r(self.temperature, False), turbulence=r(self.turbulence, False),
                      collisional=r(self.collisional, False), los_velocity=r(self.los_velocity, False),
                      number_density=r(self.number_density, True), width_kms=r(self.width_kms, True),
                      velocity_span_kms=jnp.repeat(self.velocity_span_kms, n_rows))


# --------------------------------------------------------------------------- every skewer
@partial(jax.jit, static_argnames=("voigt", "detuning", "v_chunk", "batch_size", "per_piece", "dtype"))
def optical_depth_from_pieces(pieces: Pieces, velocities_kms, line_bundle, *, voigt: str = "ExoJAX", detuning: str = "exact",
                              velocity_period=None, v_chunk: Optional[int] = None, batch_size: Optional[int] = None,
                              per_piece: bool = False, dtype: Optional[str] = None) -> jnp.ndarray:
    """tau (N_lines, N_sk, N_v) of every skewer of ``pieces``: ``optical_depth_kernel`` mapped over skewers.

    velocities_kms : (N_v,) or (N_lines, N_v) (one grid per line).
    velocity_period : None (no wrap) or (N_sk,) periods [km/s] (``compute_optical_depth`` resolves the option).
    v_chunk : velocities per kernel call (memory ~ N_lines x N_p x v_chunk); batch_size : skewers vmapped together
    inside the sequential map (None: one at a time, the lowest memory). per_piece : (N_lines, N_sk, N_p, N_v).
    dtype : None (default: the dtype of the inputs) or e.g. "float32": opt-in lower-precision kernel (2-3x faster on
        the CPU; max |dF| ~1e-3 on TNG50-4 skewers); the result is cast back to the input dtype.
    """
    out_dtype = pieces.temperature.dtype
    if dtype is not None:
        pieces, velocities_kms, velocity_period = jax.tree_util.tree_map(lambda x: x.astype(dtype),
                                                                         (pieces, velocities_kms, velocity_period))
    lam, A21, mass, f_osc = (x if dtype is None else x.astype(dtype)
                             for x in (line_bundle.lambda0_A, line_bundle.A21_GHz, line_bundle.mass_amu, line_bundle.f_osc))
    n_v = velocities_kms.shape[-1]
    nv = n_v if v_chunk is None else int(v_chunk)

    def one_skewer(s):
        a_d, a, w, T, tb, co, lv, n, P = s
        chunks = [optical_depth_kernel(velocities_kms[..., v0:v0 + nv], lam, A21, mass, f_osc, a_d, a, lv, T, tb, co, n, w,
                                       voigt=voigt, detuning=detuning, velocity_period=P, per_piece=per_piece)
                  for v0 in range(0, n_v, nv)]
        return jnp.concatenate(chunks, axis=-1)

    sk = lambda x: jnp.moveaxis(x, 1, 0)                          # (N_lines, N_sk, N_p) -> (N_sk, N_lines, N_p)
    xs = (pieces.a_dagger, pieces.scale_factor, pieces.length_m, sk(pieces.temperature), sk(pieces.turbulence),
          sk(pieces.collisional), sk(pieces.los_velocity), sk(pieces.number_density), velocity_period)
    return jnp.moveaxis(jax.lax.map(one_skewer, xs, batch_size=batch_size), 0, 1).astype(out_dtype)


def optical_depth_sharded(pieces: Pieces, velocities_kms, line_bundle, *, devices=None, velocity_period=None, **kw) -> jnp.ndarray:
    """``optical_depth_from_pieces`` with the skewers split over ``devices`` (default: all ``jax.devices()``).

    The skewer axis is padded to a multiple of the number of devices (copies of the last skewer, dropped from the
    result), split with ``shard_map`` over a 1D mesh, and every device maps its own skewers; the velocity grid and the
    line data are replicated. Each skewer is computed exactly as on one device, so the result is the same; with one
    device this is ``optical_depth_from_pieces``. kw: its options (``voigt``, ``detuning``, ``v_chunk``, ``dtype``, ...).
    CPU cores can be exposed as devices with ``XLA_FLAGS=--xla_force_host_platform_device_count=N`` (before importing JAX).
    """
    devices = list(jax.devices() if devices is None else devices)
    velocities_kms = jnp.asarray(velocities_kms)
    if len(devices) == 1:
        return optical_depth_from_pieces(pieces, velocities_kms, line_bundle, velocity_period=velocity_period, **kw)
    n_sk, n_dev = pieces.scale_factor.shape[0], len(devices)
    pad = -n_sk % n_dev
    skewer_axis = lambda x: 0 if x.ndim <= 2 else 1                 # (N_sk,), (N_sk, N_p) or (N_lines, N_sk, N_p)
    args = pieces if velocity_period is None else (pieces, jnp.asarray(velocity_period))
    padded = jax.tree_util.tree_map(
        lambda x: jnp.pad(x, [(0, pad) if i == skewer_axis(x) else (0, 0) for i in range(x.ndim)], mode="edge"), args)
    spec = jax.tree_util.tree_map(lambda x: PartitionSpec("sk") if x.ndim <= 2 else PartitionSpec(None, "sk"), padded)

    def body(s, v, lb):
        p, P = (s, None) if velocity_period is None else s
        return optical_depth_from_pieces(p, v, lb, velocity_period=P, **kw)

    run = shard_map(body, mesh=Mesh(np.array(devices), ("sk",)), in_specs=(spec, PartitionSpec(), PartitionSpec()),
                    out_specs=PartitionSpec(None, "sk"), check_rep=False)
    return jax.jit(run)(padded, velocities_kms, line_bundle)[:, :n_sk]


# --------------------------------------------------------------------------- resolution
def resolution_report(pieces: Pieces, line_bundle, tolerance: float = 0.25) -> dict:
    """Are the pieces short enough in velocity to be represented at their centroids? (NumPy, diagnostics only.)

    The sum puts each piece's absorption at its centroid's line centre: a top hat of velocity width W (the line-centre
    variation across the piece, ``Pieces.width_kms``) replaced by a delta, which changes the peak of a Doppler profile
    of width b by W^2 / (12 b^2) (0.5 % for W = b / 4). b is the Voigt width (``voigt_width_kms``).

    Returns
    -------
    dict(max_ratio (N_lines,) of W / b, column_fraction (N_lines,) of the column in pieces with W / b > tolerance,
    ok (every column fraction < 1e-3), b_min (N_lines,)).
    """
    n_l = pieces.shape[0]
    flat = lambda y: np.asarray(y).reshape(n_l, -1)
    sigma, gamma = voigt_widths_GHz(jnp.asarray(flat(pieces.temperature)), jnp.asarray(flat(pieces.turbulence)),
                                    jnp.asarray(flat(pieces.collisional)), line_bundle.lambda0_A, line_bundle.A21_GHz,
                                    line_bundle.mass_amu)
    b = np.asarray(voigt_width_kms(sigma, gamma, line_bundle.lambda0_A))
    col = flat(pieces.number_density) * np.asarray(pieces.length_m).reshape(1, -1)
    ratio = flat(pieces.width_kms) / np.maximum(b, 1e-30)
    live = col > 0
    frac = (col * (ratio > tolerance)).sum(-1) / np.maximum(col.sum(-1), 1e-300)
    return dict(max_ratio=np.where(live, ratio, 0.0).max(-1), column_fraction=frac, ok=bool(np.all(frac < 1e-3)),
                b_min=np.where(live, b, np.inf).min(-1))


# --------------------------------------------------------------------------- the entry point
def _velocity_periods(velocity_period, pieces: Pieces) -> Optional[jnp.ndarray]:
    """(N_sk,) periods [km/s] of the option ``velocity_period``, or None (no wrap)."""
    if velocity_period is None or velocity_period is False:
        return None
    n_sk = pieces.scale_factor.shape[0]
    if velocity_period is True:
        P = jnp.asarray(pieces.velocity_span_kms)
        if not isinstance(P, jax.core.Tracer) and bool(jnp.any(P <= 0)):
            raise ValueError("velocity_period=True: a skewer has no velocity span (a(first point) = a(last point)); "
                             "give the period explicitly.")
        return P
    P = jnp.broadcast_to(jnp.asarray(velocity_period, dtype=pieces.scale_factor.dtype), (n_sk,))
    if not isinstance(P, jax.core.Tracer) and bool(jnp.any(P <= 0)):
        raise ValueError(f"velocity_period must be positive, got {velocity_period}.")
    return P


def compute_optical_depth(source, velocities_kms: ArrayLike, line_bundle, cells=None, *, a_dagger: Optional[ArrayLike] = None,
                          ii_cosmology: int = 0, deposition: str = "voronoi", selection_backend: str = "cpu",
                          line_frame: str = "own", velocity_period=False, voigt: str = "ExoJAX", detuning: str = "exact",
                          resolution_tolerance: Optional[float] = 0.25, v_chunk: Optional[int] = None,
                          piece_chunk: Optional[int] = None, batch_size: Optional[int] = None, per_piece: bool = False,
                          dtype: Optional[str] = None, devices=None) -> jnp.ndarray:
    """tau (N_lines, N_sk, N_v) of every line along every skewer (F = exp(-tau)): the sum over the pieces of ``source``.

    Parameters
    ----------
    source : a ``SkewerBundle`` (with ``cells``: the pieces of ``core.absorbers.cell_pieces``) or ``Pieces`` (e.g.
        ``SimulationSkewers.pieces``).
    velocities_kms : (N_v,) velocity grid [km/s]: the line-centred velocity v_dagger of each line (``line_frame="own"``)
        or of the reference line (``"reference"``).
    cells : ``core.Cells`` holding every species of ``line_bundle`` (bundle sources only).
    a_dagger, ii_cosmology : reference scale factor of v_dagger (default: the bundle's ``a0``; scalar or (N_sk,)) and
        which cosmology of the bundle's scale factors (bundle sources only).
    deposition, selection_backend : see ``core.absorbers.deposit`` (bundle sources only; default "voronoi", the exact
        pieces of the Voronoi cells in every dual segment).
    line_frame : "own" (default): each line on its own line-centred axis; "reference": every line evaluated directly on
        the reference line's axis (``LineBundle.ref_line_id``), at v + delta_v_j: the lines share one axis exactly, with
        no shift, interpolation or wrap (review notes, relative line-centred velocity units).
    velocity_period : the only periodicity option of the package. False (default): no wrap, the spectrum is the direct
        integral along the path. True: each skewer is periodic over its own velocity span
        (``Pieces.velocity_span_kms``, set by the scale factors of its first and last points). A number: that period
        [km/s] for every skewer. With a period P, every detuning v - v_c is replaced by its nearest image in
        [-P/2, P/2): a piece whose line centre falls outside the period absorbs at the other end, and the wings that
        cross one end re-enter at the other (only the nearest image is kept).
    voigt : "ExoJAX" (default), "convolution" or "harris" (first order in the damping parameter, the matched
        fast mode; ``core.voigt.voigt_harris``) ("scipy_wofz" cannot run inside the jitted kernel).
    detuning : "exact" (default) or "linear" (fake_spectra's first-order mapping; matched mode).
    resolution_tolerance : warn when more than 1e-3 of the column sits in pieces wider than this fraction of the line
        width in velocity (``resolution_report``); None disables the check.
    v_chunk, batch_size : memory controls of ``optical_depth_from_pieces``; piece_chunk : pieces per kernel call (the
        pieces of each skewer are summed in rows of ``piece_chunk``, ``Pieces.rows``).
    per_piece : True returns every piece's contribution (N_lines, N_sk, N_p, N_v).
    dtype : opt-in kernel precision, e.g. "float32" (``optical_depth_from_pieces``); None keeps the input precision.
    devices : None (one device, the default) or a list of devices to split the skewers over (``optical_depth_sharded``).
    """
    if isinstance(source, Pieces):
        if cells is not None or a_dagger is not None:
            raise ValueError("cells and a_dagger apply to a SkewerBundle source, not to Pieces.")
        pieces = source
    else:
        if cells is None:
            raise ValueError("a SkewerBundle source needs the gas: compute_optical_depth(bundle, v, lines, cells).")
        from skewerjax.core.absorbers import cell_pieces          # (absorbers imports this module)
        pieces = cell_pieces(cells, line_bundle, skewer_bundle=source, a_dagger=a_dagger, ii_cosmology=ii_cosmology,
                             deposition=deposition, selection_backend=selection_backend)
    if resolution_tolerance is not None:
        rep = resolution_report(pieces, line_bundle, resolution_tolerance)
        if not rep["ok"]:
            warnings.warn(
                "compute_optical_depth: the pieces are too wide in velocity for the line widths: "
                + ", ".join(f"line {j}: max W / b = {rep['max_ratio'][j]:.2f}, column fraction "
                            f"{rep['column_fraction'][j]:.1e} above tolerance {resolution_tolerance}"
                            for j in range(line_bundle.n_lines))
                + ". Sample the skewers more densely.", stacklevel=2)
    v = jnp.asarray(velocities_kms)
    if line_frame == "reference":
        v = v[None, :] + jnp.asarray(line_bundle.delta_v)[:, None]
    elif line_frame != "own":
        raise ValueError(f"line_frame must be 'own' or 'reference', got {line_frame!r}.")
    P = _velocity_periods(velocity_period, pieces)
    n_sk, m = pieces.scale_factor.shape
    if piece_chunk is not None:
        pieces = pieces.rows(int(piece_chunk))
        P = None if P is None else jnp.repeat(P, pieces.scale_factor.shape[0] // n_sk)
    kw = dict(voigt=voigt, detuning=detuning, velocity_period=P, v_chunk=v_chunk, batch_size=batch_size, per_piece=per_piece,
              dtype=dtype)
    tau = (optical_depth_sharded(pieces, v, line_bundle, devices=devices, **kw) if devices is not None
           else optical_depth_from_pieces(pieces, v, line_bundle, **kw))
    if piece_chunk is None:
        return tau
    n_l, n_v = tau.shape[0], tau.shape[-1]
    if per_piece:
        return tau.reshape(n_l, n_sk, -1, n_v)[:, :, :m]
    return tau.reshape(n_l, n_sk, -1, n_v).sum(axis=2)


def wrap_to_ref_line(optical_depth: ArrayLike, velocities_kms: ArrayLike, line_bundle, *, velocity_period: Optional[float] = None,
                     fill_value: float = np.nan) -> jnp.ndarray:
    """Put every line's tau, computed in its own frame, on the reference line's axis: tau_j^ref(v) = tau_j(v + delta_v_j).

    A feature at v_j in line j's own frame appears at v_j - delta_v_j, delta_v_j = c ln(lambda_ref / lambda_j)
    (lines bluer than the reference move to lower velocities). This is a change of axis, by linear interpolation;
    ``compute_optical_depth(..., line_frame="reference")`` evaluates every line on the reference axis directly instead
    (exact, no interpolation, no gaps).

    velocity_period : None (default): no wrap; where v + delta_v_j falls outside the grid, tau_j is unknown and
        ``fill_value`` (NaN) is returned. A number P: only for a spectrum computed with ``velocity_period=P`` on a
        uniform grid of exactly one period (N_v dv = P): the shift then wraps modulo P.
    optical_depth : (N_lines, N_sk, N_v); returns the same shape.
    """
    tau, v = jnp.asarray(optical_depth), jnp.asarray(velocities_kms)
    if velocity_period is None:
        shift = lambda tau_1d, dv_j: jnp.interp(v + dv_j, v, tau_1d, left=fill_value, right=fill_value)
    else:
        dv, v0, P = v[1] - v[0], v[0], float(velocity_period)
        if not isinstance(dv, jax.core.Tracer) and abs(float(v.size * dv) / P - 1.0) > 1e-6:
            raise ValueError(f"velocity_period={P}: the grid must be uniform and span one period (N_v dv = {float(v.size * dv)}).")
        shift = lambda tau_1d, dv_j: jnp.interp(jnp.mod(v + dv_j - v0, P) + v0, v, tau_1d, period=P)
    return jax.vmap(lambda t, s: jax.vmap(lambda t1: shift(t1, s))(t), in_axes=(0, 0))(tau, jnp.asarray(line_bundle.delta_v))
