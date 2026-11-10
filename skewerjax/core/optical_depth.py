"""Line profiles and the optical depth tau(v) along skewers (review notes, Eqs. line_profile, tau_straight_los).

The optical depth is a line integral along the path (x comoving, dl_phys = a dx),

    tau_j(v) = C_j f_j int n_com(x) a(x)^-2 phi_j(v; x) dx,   C_j = q_e^2 / (4 eps_0 m_e c),   F = exp(-tau),

evaluated as a sum over **pieces** p, fragments of the path within which the gas properties are constant:

    tau_j(v) = C_j f_j sum_p l_p n_com,p a_p^-2 phi_pj(v),

with every quantity of a piece, its scale factor a_p included, evaluated at the piece's centroid (the midpoint rule
on each piece). The factor a_p^-2 converts the comoving density and length to physical ones; the option ``a_eff``
(``compute_optical_depth``) replaces it by one a_eff^-2 for every piece (a fixed comoving-to-physical conversion,
e.g. a_eff = 1 / (1 + z) of a snapshot) and touches nothing else: the line centres v_c,p, the widths and the detuning
keep the a(x) of the frame. Through the line centres the frame still sets the Jacobian of the column into velocity:
a piece of length l_p spreads over a width (a H)_p l_p, so the tau per pixel of a uniform gas also scales as
1 / (a H) along the path, whatever ``a_eff``. The option ``hubble_flow`` sets the Hubble part of the line centres,
v_c,p = V_H(y_p) + v_pec,p with y_p the comoving path coordinate of the centroid from the first point of the skewer:
"exact" (default) V_H(y) = c ln(a_dagger / a(y)) = V_H(0) + int_0^y aH dy', the frame's a(y); "linear" V_H(y) =
V_H(0) + (aH)_ref y, its first order about the first point (V_H(0) = 0 when a_dagger is a at the first point, as for
the toy boxes), fake_spectra's Hubble flow (``absorption.cpp`` L237: vel = velfac pos + pvel; ``spectra.py`` L227-L228: velfac =
a H(z)), which makes the Jacobian constant. It changes nothing else (the density factor, the peculiar velocity, the
widths, the lengths, the detuning), and the period of ``velocity_period=True`` and the piece widths of the resolution
check follow it (``velocity_span``). The gas is a set of Voronoi cells (``core.cells.Cells``, from a toy model or a simulation), constant
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

Two optional schemes evaluate the sum per cell instead (``core.per_cell``): fake_spectra's pixel average and its early
stop (``pixel_average``, ``tautail`` of ``compute_optical_depth``), on windows around each cell's line centre.
"""

from __future__ import annotations

import dataclasses
import logging
import time
import warnings
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec
from jax.typing import ArrayLike

from skewerjax.config import real_dtype
from skewerjax.core.constants import phys
from skewerjax.core.line_profiles import (
    check_b_constants,
    check_profile_mode,
    voigt_line_profiles,
    voigt_width_kms,
    voigt_widths_GHz,
)
from skewerjax.core.spectral import frequency_offset_GHz
from skewerjax.core.validation import check_positive_int, check_shape, check_values, is_traced

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- the optical depth of one skewer
@partial(jax.jit, static_argnames=("voigt", "detuning", "per_piece"))
def optical_depth_kernel(
    velocities_kms,
    lambda0_A,
    A21_GHz,
    mass_amu,
    f_osc,
    a_dagger,
    scale_factor,
    los_velocity,
    temperature,
    turbulence,
    collisional,
    number_density,
    length_m,
    voigt: str = "harris",
    detuning: str = "exact",
    velocity_period=None,
    per_piece: bool = False,
    a_eff=None,
    hubble_kms=None,
):
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
    a_eff : None (default: a_p^-2 of each piece) or a scalar (possibly traced): a_eff^-2 for every piece in the
        comoving-to-physical factor only (the line centres and the detuning keep ``scale_factor``).
    hubble_kms : None (default: the Hubble part of the line centres from the frame, c ln(a_dagger / a_p)) or (N_p,) the
        Hubble part V_H of every piece [km/s] (``hubble_flow="linear"``: (aH)_ref y_p, possibly traced).

    Returns
    -------
    (N_lines, N_v) optical depth.
    """
    sigma, gamma = voigt_widths_GHz(temperature, turbulence, collisional, lambda0_A, A21_GHz, mass_amu)
    nu = frequency_offset_GHz(
        velocities_kms,
        lambda0_A,
        a_dagger,
        scale_factor,
        los_velocity,
        velocity_period=velocity_period,
        detuning=detuning,
        hubble_kms=hubble_kms,
    )
    # one profile batch per line: voigt="harris" decides its exact fallback per batch (``voigt_harris``), and the lines
    # differ in their damping parameters (every Lya element below HARRIS_A_MAX, Si III above it near 1e4 K)
    phi = jnp.stack([voigt_line_profiles(nu[j], sigma[j], gamma[j], voigt) for j in range(nu.shape[0])])
    # dtau/dx [1/m] = C f phi n a^-2 (x comoving); 1e2 turns cm^2 cm^-3 m into a number. The column n l first (the
    # density alone reaches 1e-30 cm^-3 for metals in hot gas, whose product with the prefactor would leave the float32
    # range), then the constant of each line.
    a2 = scale_factor**-2 if a_eff is None else jnp.broadcast_to(jnp.asarray(a_eff, scale_factor.dtype) ** -2, scale_factor.shape)
    const = (phys.cross_section_prefactor * 1e2) * f_osc
    amp = const[:, None] * (number_density * (length_m * a2)[None, :])
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

    Optional, for ``hubble_flow="linear"`` (``compute_optical_depth``; filled by ``core.absorbers.cell_pieces``, None
    otherwise):

    path_cMpc : (N_sk, N_p) the comoving path coordinate y_p of every centroid from the first point of its skewer, along
        which the frame's Hubble flow c ln(a_dagger / a) runs [cMpc] (0 for a frame that carries its Hubble flow in
        ``los_velocity`` instead, ``SnapshotFrame``: the two options then coincide);
    path_length_cMpc : (N_sk,) the comoving length of every path, first to last point [cMpc];
    aH_first : (N_sk,) the frame's dV_H/dy = -c d ln a / dy at the first point [km/s per cMpc], the default (aH)_ref;
    vH_first : (N_sk,) the frame's V_H at the first point, c ln(a_dagger / a_first) [km/s] (0 when a_dagger is the scale
        factor of the first point, as for the toy boxes; 0 for ``SnapshotFrame``).

    Optional, for the per-cell options ``tautail`` and ``pixel_average`` (filled by ``core.absorbers.cell_pieces``, None
    otherwise; the sum over pieces never reads it):

    cell : (N_sk, N_p) int, the cell of every piece (-1: padding).

    Optional, for precision (filled by ``core.absorbers.absorber_pieces``; None: computed in the kernel):

    hubble_kms : (N_sk, N_p) the Hubble part of every line centre, V_H = c ln(a_dagger / a_p) [km/s], evaluated in float64
        on the host and stored in the policy dtype: ``hubble_flow="exact"`` uses it instead of the log of a ratio of
        two rounded scale factors (in float32 an error of ~0.02 km/s; docs/precision.md).

    JAX pytree (every field a leaf; a None field has none).
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
    path_cMpc: jnp.ndarray | None = None
    path_length_cMpc: jnp.ndarray | None = None
    aH_first: jnp.ndarray | None = None
    vH_first: jnp.ndarray | None = None
    cell: jnp.ndarray | None = None
    hubble_kms: jnp.ndarray | None = None

    @property
    def shape(self) -> tuple:
        """(N_lines, N_sk, N_p)."""
        return self.temperature.shape

    def rows(self, chunk: int) -> Pieces:
        """The pieces of every skewer in rows of ``chunk`` (N_sk * n_rows, chunk), padded with empty pieces: summing a
        row's tau over the rows of one skewer gives the skewer's tau (bounds the memory of the kernel)."""
        n_sk, m = self.scale_factor.shape
        n_rows = -(-m // chunk)
        pad = n_rows * chunk - m

        def r(x, empty):
            lead = x.shape[:-2]
            x = jnp.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, pad)], mode="constant" if empty else "edge")
            return x.reshape(lead + (n_sk * n_rows, chunk))

        return Pieces(
            a_dagger=r(self.a_dagger, False),
            scale_factor=r(self.scale_factor, False),
            length_m=r(self.length_m, True),
            temperature=r(self.temperature, False),
            turbulence=r(self.turbulence, False),
            collisional=r(self.collisional, False),
            los_velocity=r(self.los_velocity, False),
            number_density=r(self.number_density, True),
            width_kms=r(self.width_kms, True),
            velocity_span_kms=jnp.repeat(self.velocity_span_kms, n_rows),
            path_cMpc=None if self.path_cMpc is None else r(self.path_cMpc, False),
            path_length_cMpc=None if self.path_length_cMpc is None else jnp.repeat(self.path_length_cMpc, n_rows),
            aH_first=None if self.aH_first is None else jnp.repeat(self.aH_first, n_rows),
            vH_first=None if self.vH_first is None else jnp.repeat(self.vH_first, n_rows),
            cell=None if self.cell is None else r(self.cell, False),
            hubble_kms=None if self.hubble_kms is None else r(self.hubble_kms, False),
        )

    def lines(self, sel) -> Pieces:
        """The pieces of the lines ``sel`` (a slice or an index array of the N_lines axis): the per-line fields sliced,
        the per-skewer ones shared."""
        pick = lambda x: None if x is None else (x[sel] if x.ndim == 3 else x)
        return Pieces(**{f: pick(getattr(self, f)) for f in self.__dataclass_fields__})

    def skewers(self, sel) -> Pieces:
        """The pieces of the skewers ``sel`` (an index array or a slice)."""
        pick = lambda x: None if x is None else (x[sel] if x.ndim < 3 else x[:, sel])
        return Pieces(**{f: pick(getattr(self, f)) for f in self.__dataclass_fields__})


# --------------------------------------------------------------------------- every skewer
HUBBLE_FLOWS = ("exact", "linear")  # the Hubble part of the line centres (module docstring)


def _hubble_rates(pieces: Pieces, aH_ref):
    """(N_sk,) (aH)_ref of every skewer of ``pieces`` [km/s per cMpc]: ``aH_ref`` (a scalar, possibly traced) or, if
    None, the frame's ``Pieces.aH_first``."""
    n_sk = pieces.scale_factor.shape[0]
    if aH_ref is None:
        if pieces.aH_first is None:
            raise ValueError('hubble_flow="linear" needs Pieces.aH_first (core.absorbers.cell_pieces fills it) or aH_ref.')
        return jnp.asarray(pieces.aH_first)
    return jnp.broadcast_to(jnp.asarray(aH_ref, dtype=pieces.scale_factor.dtype), (n_sk,))


def velocity_span(pieces: Pieces, hubble_flow: str = "exact", aH_ref=None) -> jnp.ndarray:
    """(N_sk,) the line-centre span of every path without peculiar velocities [km/s], the period of
    ``velocity_period=True``: "exact" c |ln(a_first / a_last)| (``Pieces.velocity_span_kms``); "linear" |(aH)_ref| L,
    L = ``Pieces.path_length_cMpc``. ``core.absorbers.velocity_span`` gives it for a ``SkewerBundle`` or a frame too."""
    if hubble_flow not in HUBBLE_FLOWS:
        raise ValueError(f"hubble_flow must be one of {HUBBLE_FLOWS}, got {hubble_flow!r}.")
    if hubble_flow == "exact":
        return jnp.asarray(pieces.velocity_span_kms)
    if pieces.path_length_cMpc is None:
        raise ValueError('hubble_flow="linear" needs Pieces.path_length_cMpc (core.absorbers.cell_pieces fills it).')
    return jnp.abs(_hubble_rates(pieces, aH_ref)) * jnp.asarray(pieces.path_length_cMpc)


@partial(jax.jit, static_argnames=("voigt", "detuning", "v_chunk", "batch_size", "per_piece", "dtype", "hubble_flow"))
def optical_depth_from_pieces(
    pieces: Pieces,
    velocities_kms,
    line_bundle,
    *,
    voigt: str = "harris",
    detuning: str = "exact",
    velocity_period=None,
    v_chunk: int | None = None,
    batch_size: int | None = None,
    per_piece: bool = False,
    dtype: str | None = None,
    a_eff=None,
    hubble_flow: str = "exact",
    aH_ref=None,
) -> jnp.ndarray:
    """tau (N_lines, N_sk, N_v) of every skewer of ``pieces``: ``optical_depth_kernel`` mapped over skewers.

    velocities_kms : (N_v,) or (N_lines, N_v) (one grid per line).
    velocity_period : None (no wrap) or (N_sk,) periods [km/s] (``compute_optical_depth`` resolves the option).
    v_chunk : velocities per kernel call (memory ~ N_lines x N_p x v_chunk); batch_size : skewers vmapped together
    inside the sequential map (None: one at a time, the lowest memory). per_piece : (N_lines, N_sk, N_p, N_v).
    dtype : None (default: the dtype of the inputs; ``compute_optical_depth`` casts them to the package's precision) or
        "float32" / "float64": the kernel's precision; the result is cast back to the input dtype.
    a_eff : None or a scalar (traced: no recompilation per value; differentiable): ``optical_depth_kernel``.
    hubble_flow : "exact" (default) or "linear": V_H = (aH)_ref y_p with ``aH_ref`` (None: ``Pieces.aH_first`` of each
        skewer; a scalar, traced and differentiable), y_p = ``Pieces.path_cMpc`` and V_H(0) = ``Pieces.vH_first`` (0 if
        None) (module docstring).
    """
    if hubble_flow not in HUBBLE_FLOWS:
        raise ValueError(f"hubble_flow must be one of {HUBBLE_FLOWS}, got {hubble_flow!r}.")
    out_dtype = pieces.temperature.dtype
    if dtype is not None:
        pieces, velocities_kms, velocity_period, a_eff, aH_ref = jax.tree_util.tree_map(
            lambda x: jnp.asarray(x).astype(dtype), (pieces, velocities_kms, velocity_period, a_eff, aH_ref)
        )
    linear = hubble_flow == "linear"
    if linear and pieces.path_cMpc is None:
        raise ValueError('hubble_flow="linear" needs Pieces.path_cMpc (core.absorbers.cell_pieces fills it).')
    aH = _hubble_rates(pieces, aH_ref) if linear else None
    lam, A21, mass, f_osc = (
        x if dtype is None else x.astype(dtype)
        for x in (line_bundle.lambda0_A, line_bundle.A21_GHz, line_bundle.mass_amu, line_bundle.f_osc)
    )
    n_v = velocities_kms.shape[-1]
    nv = n_v if v_chunk is None else int(v_chunk)

    def one_skewer(s):
        a_d, a, w, T, tb, co, lv, n, P, y, h, h0, hx = s
        hub = hx if y is None else h0 + h * y  # V_H = V_H(0) + (aH)_ref y_p ("linear"); exact: Pieces.hubble_kms, else in-kernel
        chunks = [
            optical_depth_kernel(
                velocities_kms[..., v0 : v0 + nv],
                lam,
                A21,
                mass,
                f_osc,
                a_d,
                a,
                lv,
                T,
                tb,
                co,
                n,
                w,
                voigt=voigt,
                detuning=detuning,
                velocity_period=P,
                per_piece=per_piece,
                a_eff=a_eff,
                hubble_kms=hub,
            )
            for v0 in range(0, n_v, nv)
        ]
        return jnp.concatenate(chunks, axis=-1)

    sk = lambda x: jnp.moveaxis(x, 1, 0)  # (N_lines, N_sk, N_p) -> (N_sk, N_lines, N_p)
    xs = (
        pieces.a_dagger,
        pieces.scale_factor,
        pieces.length_m,
        sk(pieces.temperature),
        sk(pieces.turbulence),
        sk(pieces.collisional),
        sk(pieces.los_velocity),
        sk(pieces.number_density),
        velocity_period,
        pieces.path_cMpc if linear else None,
        aH,
        (jnp.zeros_like(aH) if pieces.vH_first is None else jnp.asarray(pieces.vH_first, dtype=aH.dtype)) if linear else None,
        None if linear else pieces.hubble_kms,
    )
    return jnp.moveaxis(jax.lax.map(one_skewer, xs, batch_size=batch_size), 0, 1).astype(out_dtype)


def optical_depth_sharded(
    pieces: Pieces, velocities_kms, line_bundle, *, devices=None, velocity_period=None, **kw
) -> jnp.ndarray:
    """``optical_depth_from_pieces`` with the skewers split over ``devices`` (default: all ``jax.devices()``).

    The skewer axis is padded to a multiple of the number of devices (copies of the last skewer, dropped from the
    result), split with ``shard_map`` over a 1D mesh, and every device maps its own skewers; the velocity grid and the
    line data are replicated. Each skewer is computed exactly as on one device, so the result is the same; with one
    device this is ``optical_depth_from_pieces``. kw: its options (``voigt``, ``detuning``, ``v_chunk``, ``dtype``, ...).
    CPU cores can be exposed as devices with ``XLA_FLAGS=--xla_force_host_platform_device_count=N`` (before importing JAX).
    """
    devices = list(jax.devices() if devices is None else devices)
    velocities_kms = jnp.asarray(velocities_kms)
    if len(devices) == 1:  # on that device, whatever the default backend (e.g. the CPU next to a GPU)
        pieces, velocities_kms, velocity_period = jax.device_put((pieces, velocities_kms, velocity_period), devices[0])
        with jax.default_device(devices[0]):
            return optical_depth_from_pieces(pieces, velocities_kms, line_bundle, velocity_period=velocity_period, **kw)
    n_sk, n_dev = pieces.scale_factor.shape[0], len(devices)
    a_eff = kw.pop("a_eff", None)
    aH_ref = kw.pop("aH_ref", None)
    pad = -n_sk % n_dev
    skewer_axis = lambda x: 0 if x.ndim <= 2 else 1  # (N_sk,), (N_sk, N_p) or (N_lines, N_sk, N_p)
    args = pieces if velocity_period is None else (pieces, jnp.asarray(velocity_period))
    padded = jax.tree_util.tree_map(
        lambda x: jnp.pad(x, [(0, pad) if i == skewer_axis(x) else (0, 0) for i in range(x.ndim)], mode="edge"), args
    )
    spec = jax.tree_util.tree_map(lambda x: PartitionSpec("sk") if x.ndim <= 2 else PartitionSpec(None, "sk"), padded)

    def body(s, v, lb, ae, ah):
        p, P = (s, None) if velocity_period is None else s
        return optical_depth_from_pieces(p, v, lb, velocity_period=P, a_eff=ae, aH_ref=ah, **kw)

    run = shard_map(
        body,
        mesh=Mesh(np.array(devices), ("sk",)),
        in_specs=(spec, PartitionSpec(), PartitionSpec(), PartitionSpec(), PartitionSpec()),
        out_specs=PartitionSpec(None, "sk"),
        check_rep=False,
    )
    return jax.jit(run)(padded, velocities_kms, line_bundle, a_eff, aH_ref)[:, :n_sk]


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
    sigma, gamma = voigt_widths_GHz(
        jnp.asarray(flat(pieces.temperature)),
        jnp.asarray(flat(pieces.turbulence)),
        jnp.asarray(flat(pieces.collisional)),
        line_bundle.lambda0_A,
        line_bundle.A21_GHz,
        line_bundle.mass_amu,
    )
    b = np.asarray(voigt_width_kms(sigma, gamma, line_bundle.lambda0_A))
    col = flat(pieces.number_density) * np.asarray(pieces.length_m).reshape(1, -1)
    ratio = flat(pieces.width_kms) / np.maximum(b, 1e-30)
    live = col > 0
    frac = (col * (ratio > tolerance)).sum(-1) / np.maximum(col.sum(-1), 1e-300)
    return dict(
        max_ratio=np.where(live, ratio, 0.0).max(-1),
        column_fraction=frac,
        ok=bool(np.all(frac < 1e-3)),
        b_min=np.where(live, b, np.inf).min(-1),
    )


# --------------------------------------------------------------------------- the entry point
def _velocity_periods(velocity_period, pieces: Pieces, hubble_flow: str = "exact", aH_ref=None) -> jnp.ndarray | None:
    """(N_sk,) periods [km/s] of the option ``velocity_period``, or None (no wrap); True: ``velocity_span`` under
    ``hubble_flow``."""
    if velocity_period is None or velocity_period is False:
        return None
    if velocity_period is True:
        P = velocity_span(pieces, hubble_flow, aH_ref)
        what = "velocity_period=True: a skewer has no velocity span (a(first point) = a(last point)); give the period explicitly."
    else:
        P = jnp.broadcast_to(jnp.asarray(velocity_period, dtype=pieces.scale_factor.dtype), (pieces.scale_factor.shape[0],))
        what = f"velocity_period must be a finite period > 0 [km/s], got {velocity_period}."
    if not is_traced(P):
        with jax.ensure_compile_time_eval():
            if not bool(jnp.all(jnp.isfinite(P) & (P > 0))):
                raise ValueError(what)
    return P


_PIECE_RANGES = (
    ("scale_factor", 0.0, True),
    ("a_dagger", 0.0, True),
    ("length_m", 0.0, False),
    ("temperature", 0.0, False),
    ("turbulence", 0.0, False),
    ("collisional", 0.0, False),
    ("los_velocity", None, False),
    ("number_density", None, False),
    ("width_kms", 0.0, False),
)


def check_pieces(pieces: Pieces, line_bundle=None) -> None:
    """Shapes and values of ``pieces`` (eager; skipped under tracing): consistent shapes (N_lines, N_sk, N_p), N_lines =
    ``line_bundle.n_lines``, finite values, scale factors > 0 and lengths, temperatures, widths >= 0 (and the optional
    path fields, if given: finite, path lengths >= 0). ``ValueError``
    names the field and the first bad entry."""
    n_l, n_sk, n_p = pieces.shape
    if line_bundle is not None and n_l != line_bundle.n_lines:
        raise ValueError(
            f"the pieces hold {n_l} lines, the line bundle {line_bundle.n_lines} ({[l.id for l in line_bundle.lines]})."
        )
    for f in ("a_dagger", "scale_factor", "length_m"):
        check_shape(f"Pieces.{f}", getattr(pieces, f), (n_sk, n_p))
    for f in ("temperature", "turbulence", "collisional", "los_velocity", "number_density", "width_kms"):
        check_shape(f"Pieces.{f}", getattr(pieces, f), (n_l, n_sk, n_p))
    check_shape("Pieces.velocity_span_kms", pieces.velocity_span_kms, (n_sk,))
    for f, lo, strict in _PIECE_RANGES:
        check_values(f"Pieces.{f}", getattr(pieces, f), minimum=lo, strict=strict)
    for f, shape, lo in (
        ("path_cMpc", (n_sk, n_p), None),  # the chord points of a sphere at the seam lie beyond the ends of the path
        ("path_length_cMpc", (n_sk,), 0.0),
        ("aH_first", (n_sk,), None),
        ("vH_first", (n_sk,), None),
        ("cell", (n_sk, n_p), None),
        ("hubble_kms", (n_sk, n_p), None),
    ):
        x = getattr(pieces, f)
        if x is not None:
            check_shape(f"Pieces.{f}", x, shape)
            check_values(f"Pieces.{f}", x, minimum=lo)


def _cast_pieces(pieces: Pieces, dtype) -> Pieces:
    """``pieces`` with every floating field in ``dtype`` (the integer ``cell`` unchanged); no copy for fields already in it."""
    cast = lambda x: x if x is None or not jnp.issubdtype(x.dtype, jnp.floating) or x.dtype == dtype else jnp.asarray(x, dtype)
    return Pieces(**{f: cast(getattr(pieces, f)) for f in pieces.__dataclass_fields__})


def compute_optical_depth(
    source,
    velocities_kms: ArrayLike,
    line_bundle,
    cells=None,
    *,
    a_dagger: ArrayLike | None = None,
    ii_cosmology: int = 0,
    deposition: str = "voronoi",
    line_frame: str = "own",
    velocity_period=False,
    voigt: str = "harris",
    detuning: str = "exact",
    resolution_tolerance: float | None = 0.25,
    v_chunk: int | None = None,
    piece_chunk: int | None = None,
    batch_size: int | None = None,
    per_piece: bool = False,
    dtype: str | None = None,
    devices=None,
    a_eff=None,
    hubble_flow: str = "exact",
    aH_ref=None,
    b_constants: str = "codata",
    pixel_average: bool = False,
    tautail: float | None = None,
    chord_weight: float | None = None,
    return_info: bool = False,
) -> jnp.ndarray:
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
    deposition : see ``core.absorbers.deposit`` (bundle sources only; default "voronoi", the exact
        pieces of the Voronoi cells in every dual segment); chord_weight : ``deposition="tophat_chords"`` only, the weight
        of each chord point (``core.deposition.tophat_chord_segments``).
    line_frame : "own" (default): each line on its own line-centred axis; "reference": every line evaluated directly on
        the reference line's axis (``LineBundle.ref_line_id``), at v + delta_v_j: the lines share one axis exactly, with
        no shift, interpolation or wrap (review notes, relative line-centred velocity units).
    velocity_period : the only periodicity option of the package. False (default): no wrap, the spectrum is the direct
        integral along the path. True: each skewer is periodic over its own velocity span
        (``Pieces.velocity_span_kms``, set by the scale factors of its first and last points). A number: that period
        [km/s] for every skewer. With a period P, every detuning v - v_c is replaced by its nearest image in
        [-P/2, P/2): a piece whose line centre falls outside the period absorbs at the other end, and the wings that
        cross one end re-enter at the other (only the nearest image is kept).
    voigt : "harris" (default; ``core.voigt.voigt_harris``: fake_spectra's first-order expansion in the
        damping parameter a for a <= 1e-3, the fourth order up to a = 0.05, ExoJAX's exact function beyond; one batch per
        line) or "ExoJAX" (the exact function everywhere, ~1e-6 against wofz): ``line_profiles.PROFILE_MODES``. Any other
        value raises ``ValueError``; the FFT convolution and SciPy's wofz are standalone functions of ``core.voigt``.
    detuning : "exact" (default) or "linear" (fake_spectra's first-order mapping; matched mode).
    resolution_tolerance : warn when more than 1e-3 of the column sits in pieces wider than this fraction of the line
        width in velocity (``resolution_report``); None disables the check.
    v_chunk, batch_size : memory controls of ``optical_depth_from_pieces``; piece_chunk : pieces per kernel call (the
        pieces of each skewer are summed in rows of ``piece_chunk``, ``Pieces.rows``).
    per_piece : True returns every piece's contribution (N_lines, N_sk, N_p, N_v).
    dtype : the precision of the sum, "float32" or "float64"; None (default): the package's policy
        (``skewerjax.config.get_precision``, "float32" unless set). The pieces are cast to it once, and tau is returned in it.
    devices : None (the default device) or a list of devices, the same API on the CPU and on a GPU:
        ``skewerjax.config.devices("cpu")`` (the host devices of ``skewerjax.config.configure_devices``) or
        ``devices("gpu")``. The sum over pieces splits the skewers over them (``optical_depth_sharded``, ``shard_map``);
        the per-cell schemes (``pixel_average``, ``tautail``) dispatch their chunks of cells to them in turn. One device:
        everything runs on it. The result does not depend on the devices.
    a_eff : None (default) or a scalar > 0, the effective scale factor of the comoving-to-physical density conversion:
        None keeps a_p^-2 of every piece (tau_j = C_j f_j sum_p n_com,p l_p a_p^-2 phi_jp); a scalar uses a_eff^-2
        for every piece (e.g. 1 / (1 + z), a fixed conversion). The line centres v_c,p, the widths, the lengths and
        the detuning are untouched, so the Jacobian 1 / (a H) of the frame (module docstring) stays. Traced in the
        kernel (no recompilation per value) and differentiable. A value > 1 (a scale factor in the future, e.g. 1 + z
        typed for 1 / (1 + z)) issues a warning.
    hubble_flow : "exact" (default) or "linear": the Hubble part of the line centres (module docstring), V_H(y) =
        c ln(a_dagger / a(y)) (the frame) or (aH)_ref y (fake_spectra). Only the line centres change; the density factor
        (``a_eff``), the peculiar velocity, the widths, the lengths and the ``detuning`` do not. With "linear" the period
        of ``velocity_period=True`` is |(aH)_ref| L and the piece widths of the resolution check |(aH)_ref| l_p
        (``velocity_span``); needs the path fields of ``Pieces`` (``cell_pieces`` fills them).
    aH_ref : None (default: the frame's aH at the first point of each skewer, ``Pieces.aH_first``; for the toy boxes
        ``ToyBoxes.aH``) or a scalar [km/s per cMpc], "linear" only. Traced (no recompilation per value) and
        differentiable.
    b_constants : the constants of the thermal width b^2 = 2 k_B T / m. "codata" (default): k_B / (A m_u). "fake_spectra":
        its C engine's k_B / (A m_p) (``absorption.cpp`` L24, L158; ``absorption.h`` L4), 0.72 % lower in b^2, a factor of
        T inside the profile only (``line_profiles.B2_RATIOS``).
    pixel_average : False (default): the profile at the pixel centres. True: fake_spectra's average over the pixel
        (``singleabs.h:tau_kern_outer`` L133-L155): a piece of Doppler width b >= 2 dv keeps the centre, a narrower one
        takes the trapezoid on 2 ceil(dv / b) + 1 points of the pixel. Needs a uniform grid; eager (a host-side loop over
        the pieces, grouped by number of points: not under ``jit``).
    tautail : None (default): full profiles. A threshold (fake_spectra's is 1e-7, ``spectra.py`` L145): its early stop per
        cell (``absorption.cpp`` L253-L281): walking out from the pixel of the cell's line centre, forward and backward,
        each cell adds a pixel up to and including the first one that receives less than the threshold from that cell,
        at most half a period each way. Cuts the far wings of every cell, whatever their sum. Needs ``Pieces.cell`` and a
        uniform grid (over exactly the period, with ``velocity_period``); eager, as ``pixel_average``.
    return_info : also return dict(shape (N_lines, N_sk, N_p), n_v, line_frame, velocity_period (the (N_sk,) periods or
        None), resolution (``resolution_report``, None without the check), pieces_info (``cell_pieces``'s, bundle sources
        only), tau_max (N_lines, N_sk), tau_integral_kms (N_lines, N_sk) = sum_v tau dv over the grid, column_integral_kms
        (N_lines, N_sk) = sum_p C f n_p a_p^-2 l_p lambda_j (a_eff^-2 with ``a_eff``), its value from the pieces (every profile has unit area in
        frequency; exact for the linear detuning, O(b / c) otherwise), window_fraction = their ratio (what the grid holds
        of the absorption: 1 on one period with ``velocity_period``, up to the cut at P/2), timings {pieces, checks, tau}
        [s] (tau waits for the result), time_s). Off by default: no cost.

    Returns
    -------
    (N_lines, N_sk, N_v) optical depth ((N_lines, N_sk, N_p, N_v) with ``per_piece``), and the info.
    """
    t0 = time.perf_counter()
    if line_frame not in ("own", "reference"):
        raise ValueError(f"line_frame must be 'own' or 'reference', got {line_frame!r}.")
    check_profile_mode(voigt)
    b2_ratio = check_b_constants(b_constants)
    if detuning not in ("exact", "linear"):
        raise ValueError(f"detuning must be 'exact' or 'linear', got {detuning!r}.")
    if hubble_flow not in HUBBLE_FLOWS:
        raise ValueError(f"hubble_flow must be one of {HUBBLE_FLOWS}, got {hubble_flow!r}.")
    if tautail is not None and not tautail > 0:
        raise ValueError(f"tautail must be None or a threshold > 0, got {tautail!r}.")
    per_cell = bool(pixel_average) or tautail is not None
    if per_cell and per_piece:
        raise ValueError("pixel_average and tautail sum over the cells: per_piece does not apply.")
    if aH_ref is not None:
        if hubble_flow != "linear":
            raise ValueError('aH_ref applies to hubble_flow="linear" only.')
        if jnp.ndim(aH_ref) != 0:
            raise ValueError(f"aH_ref must be None or a scalar, got shape {jnp.shape(aH_ref)}.")
        if not is_traced(aH_ref):
            with jax.ensure_compile_time_eval():
                if not bool(jnp.isfinite(aH_ref) & (jnp.asarray(aH_ref) != 0)):
                    raise ValueError(f"aH_ref must be a finite, non-zero rate [km/s per cMpc], got {aH_ref}.")
    for name, x in (("v_chunk", v_chunk), ("piece_chunk", piece_chunk), ("batch_size", batch_size)):
        check_positive_int(name, x)
    v = jnp.asarray(velocities_kms)
    if (
        v.ndim not in (1, 2)
        or v.shape[-1] == 0
        or (v.ndim == 2 and (line_frame == "reference" or v.shape[0] != line_bundle.n_lines))
    ):
        raise ValueError(
            f"velocities_kms must be (N_v,) or, with line_frame='own', one grid per line (N_lines={line_bundle.n_lines}, N_v); "
            f"got shape {v.shape}."
        )
    check_values("velocities_kms", v)
    if a_eff is not None:
        if jnp.ndim(a_eff) != 0:
            raise ValueError(f"a_eff must be None or a scalar, got shape {jnp.shape(a_eff)}.")
        if not is_traced(a_eff):
            with jax.ensure_compile_time_eval():
                if not bool(jnp.isfinite(a_eff) & (jnp.asarray(a_eff) > 0)):
                    raise ValueError(f"a_eff must be a finite scale factor > 0, got {a_eff}.")
                if bool(jnp.asarray(a_eff) > 1):  # a scale factor of the past is < 1: most likely 1 + z for 1 / (1 + z)
                    warnings.warn(
                        f"a_eff = {float(a_eff):g} > 1: a scale factor above today's (a = 1); tau scales as a_eff^-2, "
                        f"so it is {float(a_eff) ** 2:.3g} times smaller than with a_eff = 1. Did you mean 1 / (1 + z)?",
                        stacklevel=2,
                    )
    work = real_dtype() if dtype is None else np.dtype(dtype)  # the precision policy (skewerjax.config), resolved outside jit
    pieces_info = None
    if isinstance(source, Pieces):
        if cells is not None or a_dagger is not None:
            raise ValueError("cells and a_dagger apply to a SkewerBundle source, not to Pieces.")
        pieces = source
    else:
        if cells is None:
            raise ValueError("a SkewerBundle source needs the gas: compute_optical_depth(bundle, v, lines, cells).")
        from skewerjax.core.absorbers import cell_pieces  # (absorbers imports this module)

        pieces = cell_pieces(
            cells,
            line_bundle,
            skewer_bundle=source,
            a_dagger=a_dagger,
            ii_cosmology=ii_cosmology,
            deposition=deposition,
            chord_weight=chord_weight,
            return_info=return_info,
        )
        if return_info:
            pieces, pieces_info = pieces
    t1 = time.perf_counter()
    check_pieces(pieces, line_bundle)
    if pieces.hubble_kms is None and work != np.float64 and not is_traced(pieces.a_dagger, pieces.scale_factor):
        # Pieces built by hand: V_H = c ln(a_dagger / a_p) in float64 on the host before the cast, as absorber_pieces
        # does (docs/precision.md, item 1); float64 keeps the in-kernel log, bit for bit
        a_p = np.asarray(pieces.scale_factor, dtype=np.float64)
        a_d = np.asarray(pieces.a_dagger, dtype=np.float64)
        a_d = a_d[:, None] if a_d.ndim == 1 else a_d  # (N_sk,) or per piece (N_sk, N_p)
        pieces = dataclasses.replace(pieces, hubble_kms=phys.c_kms * np.log(a_d / a_p))
    pieces = _cast_pieces(pieces, work)
    if per_cell and is_traced(pieces.temperature):
        raise ValueError("pixel_average and tautail are host-side loops over the cells: they cannot run under jit or vmap.")
    if b_constants != "codata":
        pieces = dataclasses.replace(pieces, temperature=pieces.temperature * b2_ratio)
    if hubble_flow == "linear":
        for f in ("path_cMpc", "path_length_cMpc"):
            if getattr(pieces, f) is None:
                raise ValueError(f'hubble_flow="linear" needs Pieces.{f} (core.absorbers.cell_pieces fills it).')
    rep = None
    if resolution_tolerance is not None:
        checked = pieces
        if hubble_flow == "linear":  # the line centre varies across a piece by |(aH)_ref| l_p
            w = jnp.abs(_hubble_rates(pieces, aH_ref))[:, None] * jnp.asarray(pieces.length_m) / phys.Mpc_to_m
            checked = dataclasses.replace(pieces, width_kms=jnp.broadcast_to(w[None], pieces.width_kms.shape))
        rep = resolution_report(checked, line_bundle, resolution_tolerance)
        if not rep["ok"]:
            warnings.warn(
                "compute_optical_depth: the pieces are too wide in velocity for the line widths: "
                + ", ".join(
                    f"line {j}: max W / b = {rep['max_ratio'][j]:.2f}, column fraction "
                    f"{rep['column_fraction'][j]:.1e} above tolerance {resolution_tolerance}"
                    for j in range(line_bundle.n_lines)
                )
                + ". Sample the skewers more densely.",
                stacklevel=2,
            )
    traced = is_traced(velocities_kms, *jax.tree_util.tree_leaves(line_bundle))
    v_host = None if traced else np.asarray(velocities_kms, dtype=np.float64)  # the grid, in float64 (concrete inputs)
    if line_frame == "reference":
        v = v[None, :] + jnp.asarray(line_bundle.delta_v)[:, None]
        v_host = None if v_host is None else v_host[None, :] + np.asarray(line_bundle.delta_v, dtype=np.float64)[:, None]
    v = (v if v_host is None else jnp.asarray(v_host)).astype(work)
    P = _velocity_periods(velocity_period, pieces, hubble_flow, aH_ref)
    n_sk, m = pieces.scale_factor.shape
    pieces_all = pieces
    if per_cell:
        t2 = time.perf_counter()
        from skewerjax.core.per_cell import per_cell_optical_depth

        tau = per_cell_optical_depth(
            pieces,
            v_host,
            line_bundle,
            P,
            voigt=voigt,
            detuning=detuning,
            a_eff=a_eff,
            hubble_flow=hubble_flow,
            aH_ref=aH_ref,
            dtype=work,
            tautail=tautail,
            pixel_average=pixel_average,
            devices=devices,
        )
        return (
            _finish_info(tau, pieces_all, v, line_bundle, source, line_frame, P, rep, a_eff, t0, t1, t2, False, pieces_info)
            if return_info
            else tau
        )
    if pieces.cell is not None:  # the sum over pieces does not read it
        pieces = dataclasses.replace(pieces, cell=None)
    if piece_chunk is not None:
        pieces = pieces.rows(int(piece_chunk))
        P = None if P is None else jnp.repeat(P, pieces.scale_factor.shape[0] // n_sk)
    kw = dict(
        voigt=voigt,
        detuning=detuning,
        velocity_period=P,
        v_chunk=v_chunk,
        batch_size=batch_size,
        per_piece=per_piece,
        dtype=None,
        a_eff=a_eff,
        hubble_flow=hubble_flow,
        aH_ref=aH_ref,
    )
    logger.debug(
        "compute_optical_depth: %d lines x %d skewers x %d pieces, %d velocities (%s frame, velocity_period=%s, voigt %s, "
        "detuning %s, dtype %s)",
        line_bundle.n_lines,
        n_sk,
        m,
        v.shape[-1],
        line_frame,
        velocity_period,
        voigt,
        detuning,
        work.name,
    )
    t2 = time.perf_counter()
    run = (
        (lambda p, vj, lb: optical_depth_sharded(p, vj, lb, devices=devices, **kw))
        if devices is not None
        else (lambda p, vj, lb: optical_depth_from_pieces(p, vj, lb, **kw))
    )
    # one line at a time: each has its own Voigt branch (voigt="harris": the first order for Lya, the fourth for Si III
    # near 1e4 K), and two one-line batches are faster than one two-line batch (measured 2.4x on CPU)
    n_l = line_bundle.n_lines
    tau = (
        run(pieces, v, line_bundle)
        if n_l == 1
        else jnp.concatenate(
            [run(pieces.lines(slice(j, j + 1)), v if v.ndim == 1 else v[j : j + 1], line_bundle.subset([j])) for j in range(n_l)]
        )
    )
    if piece_chunk is not None:
        n_l, n_v = tau.shape[0], tau.shape[-1]
        tau = tau.reshape(n_l, n_sk, -1, n_v)[:, :, :m] if per_piece else tau.reshape(n_l, n_sk, -1, n_v).sum(axis=2)
    if not return_info:
        return tau
    return _finish_info(tau, pieces_all, v, line_bundle, source, line_frame, P, rep, a_eff, t0, t1, t2, per_piece, pieces_info)


def _finish_info(tau, p, v, line_bundle, source, line_frame, P, rep, a_eff, t0, t1, t2, per_piece, pieces_info):
    """(tau, the dict of ``return_info``) of ``compute_optical_depth``."""
    tau = tau.block_until_ready()
    t3 = time.perf_counter()
    full = tau.sum(axis=2) if per_piece else tau
    n_l, n_sk, m = p.shape
    amp = (
        phys.cross_section_prefactor
        * np.asarray(line_bundle.f_osc)[:, None, None]
        * np.asarray(p.number_density)
        * (np.asarray(p.scale_factor)[None] ** -2.0 if a_eff is None else float(a_eff) ** -2.0)
        * 1e2
        * np.asarray(p.length_m)[None]
    )
    col_int = amp.sum(-1) * np.asarray(line_bundle.lambda0_A)[:, None] * 1e-4  # [km/s]: lambda [A] x 1e-4 = km/s per GHz
    vv = np.broadcast_to(np.asarray(v, dtype=np.float64), (n_l, v.shape[-1]))
    dv = np.stack([np.gradient(x) if x.size > 1 else np.ones(1) for x in vv])  # pixel widths (uniform grid: dv)
    tau_int = (np.asarray(full, dtype=np.float64) * dv[:, None, :]).sum(-1)
    info = dict(
        shape=(n_l, n_sk, m),
        n_v=int(v.shape[-1]),
        line_frame=line_frame,
        velocity_period=None if P is None else np.asarray(P),
        resolution=rep,
        tau_max=np.asarray(full).max(-1),
        tau_integral_kms=tau_int,
        column_integral_kms=col_int,
        window_fraction=tau_int / np.maximum(col_int, 1e-300),
        timings=dict(pieces=t1 - t0, checks=t2 - t1, tau=t3 - t2),
        time_s=t3 - t0,
    )
    if not isinstance(source, Pieces):
        info["pieces_info"] = pieces_info
    return tau, info


def wrap_to_ref_line(
    optical_depth: ArrayLike,
    velocities_kms: ArrayLike,
    line_bundle,
    *,
    velocity_period: float | None = None,
    fill_value: float = np.nan,
) -> jnp.ndarray:
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
            raise ValueError(
                f"velocity_period={P}: the grid must be uniform and span one period (N_v dv = {float(v.size * dv)})."
            )
        shift = lambda tau_1d, dv_j: jnp.interp(jnp.mod(v + dv_j - v0, P) + v0, v, tau_1d, period=P)
    return jax.vmap(lambda t, s: jax.vmap(lambda t1: shift(t1, s))(t), in_axes=(0, 0))(tau, jnp.asarray(line_bundle.delta_v))
