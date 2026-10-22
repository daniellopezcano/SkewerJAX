"""Line profiles and the optical depth tau(v) along skewers (notes ``sec_flux.tex``, Eqs. line_profile, tau_straight_los).

    tau_j(v) = C_j f_j sum_i w_i n_com,i a_i^-2 phi_ij(v),   C_j = q_e^2 / (4 eps_0 m_e c),   F = exp(-tau)

The sum is the quadrature of Eq. tau_straight_los on the sampling points i of a skewer, with line elements w_i
[comoving m] (trapezoid weights of the points by default, or explicit segment lengths), comoving densities
n_com [cm^-3] and the Voigt profile phi [ns] of the detuning nu (``spectral.frequency_offset_GHz``).

Layers (all pure; the last one only adds eager checks):

* ``optical_depth_kernel``: one skewer, arrays in, (N_lines, N_v) out (jit);
* ``optical_depth_from_samples``: every skewer of a ``LoSSamples`` pytree, ``lax.map`` over skewers (optionally
  vmapped in batches) and velocity chunks (jit);
* ``compute_optical_depth``: samples the ``Fields`` of a ``SkewerBundle`` (``los_samples``), checks the sampling
  resolution and runs the above.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from functools import partial
from math import prod
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec
from jax.typing import ArrayLike

from skewerjax.core.constants import phys
from skewerjax.core.skewers import line_elements as trapezoid_line_elements
from skewerjax.core.spectral import frequency_offset_GHz, line_centre_velocity_kms
from skewerjax.core.voigt import log10_voigt


# --------------------------------------------------------------------------- line widths and profiles
def voigt_widths_GHz(temperature, turbulence, collisional, lambda0_A, A21_GHz, mass_amu):
    """Gaussian sigma and Lorentzian gamma [GHz] per (line, point).

    b_th = sqrt(2 k_B T / m) [m/s], b = b_th + b_turb (a linear sum, as implemented so far; the notes add them in
    quadrature), sigma = b / (sqrt(2) lambda), gamma = (A_21 + A_coll) / (4 pi).
    temperature [K], turbulence [m/s], collisional [GHz] : (N_lines, N_points); the per-line arrays (N_lines,).
    """
    gamma = (A21_GHz[:, None] + collisional) / (4.0 * jnp.pi)
    v_th_ms = jnp.sqrt(2.0 * phys.kB * temperature / mass_amu[:, None])
    sigma = (v_th_ms + turbulence) * 10.0 / (jnp.sqrt(2.0) * lambda0_A[:, None])
    return sigma, gamma


def voigt_width_kms(sigma_GHz, gamma_GHz, lambda0_A):
    """Width of a Voigt profile as an equivalent Doppler parameter [km/s].

    f_G = 2 sqrt(ln 2) sqrt(2) sigma lambda, f_L = 2 gamma lambda, f_V = 0.5346 f_L + sqrt(0.2166 f_L^2 + f_G^2)
    (better than 0.02 %); returns f_V / (2 sqrt(ln 2)): the thermal b for a Doppler line, finite for T -> 0.
    sigma_GHz, gamma_GHz : (N_lines, ...); lambda0_A : (N_lines,) [A].
    """
    lam = jnp.reshape(jnp.asarray(lambda0_A), (-1,) + (1,) * (jnp.ndim(sigma_GHz) - 1))
    to_kms = lam * 1e-4                                  # GHz x A -> km/s
    f_G = 2.0 * jnp.sqrt(jnp.log(2.0)) * jnp.sqrt(2.0) * sigma_GHz * to_kms
    f_L = 2.0 * gamma_GHz * to_kms
    f_V = 0.5346 * f_L + jnp.sqrt(0.2166 * f_L ** 2 + f_G ** 2)
    return f_V / (2.0 * jnp.sqrt(jnp.log(2.0)))


def voigt_line_profiles(nu_GHz: ArrayLike, sigma_GHz: ArrayLike, gamma_GHz: ArrayLike, voigt: str = "ExoJAX") -> jnp.ndarray:
    """phi = V(nu; sigma, gamma) [ns] (Eq. line_profile), unit area in nu. nu (N, N_v); sigma, gamma (N,)."""
    log10_phi = jax.vmap(lambda x, s, g: log10_voigt(x, s, g, mode=voigt), in_axes=(0, 0, 0))(nu_GHz, sigma_GHz, gamma_GHz)
    return 10.0 ** log10_phi


def _line_fields(fields, line_bundle, qv_flat, los_flat, names) -> dict:
    """The fields of every species tiled to its lines: (N_lines, N_points) each, + ``LoS_velocity``."""
    species_fields, species_lines = set(fields.species_keys), set(line_bundle.species_keys)
    if species_fields != species_lines:
        raise ValueError(f"Species keys mismatch between fields and line_bundle. fields={species_fields}, line_bundle={species_lines}")
    for sp in line_bundle.species_keys:
        missing = [f for f in names + ("velocity",) if f not in fields.field_keys[sp]]
        if missing:
            raise ValueError(f"Missing fields for species '{sp}': {missing}. These are required for the line profiles.")
    idx = {sp: line_bundle.indices_for_species(sp) for sp in line_bundle.species_keys}
    out = {n: fields.tile_field_to_lines(n, qv_flat, indices_for_species=idx) for n in names}
    out["LoS_velocity"] = fields.tile_field_to_lines("LoS_velocity", qv_flat, indices_for_species=idx, LoS_directions=los_flat)
    return out


def line_profiles(quadri_vectors: jnp.ndarray, LoS_directions: jnp.ndarray, velocities_kms: jnp.ndarray, a_dagger: ArrayLike,
                  line_bundle, fields, voigt: str = "ExoJAX", velocity_period: Optional[float] = None,
                  detuning: str = "exact") -> jnp.ndarray:
    """phi_ij(x, v) [ns] of every line at events (..., 4) for the line-centred velocities (N_v,) [km/s].

    a_dagger : scalar, (*batch,) or (N_points,). ``velocity_period`` and ``detuning`` as in ``frequency_offset_GHz``.
    Returns (N_lines, *batch, N_v).
    """
    velocities_kms, a_dagger = jnp.asarray(velocities_kms), jnp.asarray(a_dagger)
    if quadri_vectors.ndim < 2 or quadri_vectors.shape[-1] != 4:
        raise ValueError(f"`quadri_vectors` must have shape (..., 4) with last axis [a,x,y,z]. Got shape {quadri_vectors.shape}.")
    batch = quadri_vectors.shape[:-1]
    n_pts, n_v, n_lines = int(prod(batch)), int(velocities_kms.size), int(line_bundle.n_lines)
    LoS_directions = jnp.asarray(LoS_directions)
    if LoS_directions.shape != batch + (3,):
        raise ValueError(f"`LoS_directions` shape {LoS_directions.shape} is incompatible with batch_shape={batch}; "
                         f"expected {batch + (3,)}")
    qv = quadri_vectors.reshape(n_pts, 4)
    if a_dagger.ndim == 0 or a_dagger.shape == (n_pts,):
        a_d = a_dagger
    elif a_dagger.shape == batch:
        a_d = a_dagger.reshape(n_pts)
    else:
        raise ValueError(f"`a_dagger` must be scalar, have shape {batch}, or have shape ({n_pts},). Got {a_dagger.shape}.")
    f = _line_fields(fields, line_bundle, qv, LoS_directions.reshape(n_pts, 3), ("collisional", "temperature", "turbulence"))
    lam = jnp.asarray(line_bundle.lambda0_A)
    sigma, gamma = voigt_widths_GHz(f["temperature"], f["turbulence"], f["collisional"], lam,
                                    jnp.asarray(line_bundle.A21_GHz), jnp.asarray(line_bundle.mass_amu))
    nu = frequency_offset_GHz(velocities_kms, lam, a_d, qv[:, 0], f["LoS_velocity"], velocity_period=velocity_period, detuning=detuning)
    n = n_lines * n_pts
    phi = voigt_line_profiles(nu.reshape(n, n_v), sigma.reshape(n), gamma.reshape(n), voigt)
    return phi.reshape(n_lines, *batch, n_v)


# --------------------------------------------------------------------------- the optical depth of one skewer
@partial(jax.jit, static_argnames=("voigt", "detuning", "velocity_period", "per_point"))
def optical_depth_kernel(velocities_kms, lambda0_A, A21_GHz, mass_amu, f_osc, a_dagger, scale_factor, los_velocity, temperature,
                         turbulence, collisional, number_density, weights_m, voigt: str = "ExoJAX", detuning: str = "exact",
                         velocity_period: Optional[float] = None, per_point: bool = False):
    """tau = sum_i w_i C f n_i a_i^-2 phi_i(v) for the points of one skewer (Eq. tau_straight_los, discretised).

    Parameters
    ----------
    velocities_kms : (N_v,); lambda0_A, A21_GHz [GHz], mass_amu, f_osc : (N_lines,)
    a_dagger : scalar or (N_points,); scale_factor : (N_points,)
    los_velocity [km/s], temperature [K], turbulence [m/s], collisional [GHz], number_density [comoving cm^-3] :
        (N_lines, N_points)
    weights_m : (N_points,) line elements [comoving m]
    per_point : True returns every point's contribution (N_lines, N_points, N_v) instead of the sum.

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
    amp = phys.cross_section_prefactor * f_osc[:, None] * number_density * scale_factor[None, :] ** -2 * 1e2 * weights_m[None, :]
    if per_point:
        return amp[..., None] * phi
    return jnp.einsum("lp,lpv->lv", amp, phi)


# --------------------------------------------------------------------------- every skewer
@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class LoSSamples:
    """What the optical depth needs at every sampling point (the boundary between field sources and tau).

    a_dagger, scale_factor : (N_sk, N_p) reference and local scale factors; weights_m : (N_sk, N_p) line elements
    [comoving m]; temperature [K], turbulence [m/s], collisional [GHz], los_velocity [km/s, along the photon
    direction], number_density [comoving cm^-3] : (N_lines, N_sk, N_p). JAX pytree (every field a leaf).
    """
    a_dagger: jnp.ndarray
    scale_factor: jnp.ndarray
    weights_m: jnp.ndarray
    temperature: jnp.ndarray
    turbulence: jnp.ndarray
    collisional: jnp.ndarray
    los_velocity: jnp.ndarray
    number_density: jnp.ndarray

    @property
    def shape(self) -> tuple:
        """(N_lines, N_sk, N_p)."""
        return self.temperature.shape


def los_samples(skewer_bundle, fields, line_bundle, *, a_dagger: Optional[ArrayLike] = None, ii_cosmology: int = 0,
                line_elements: Optional[ArrayLike] = None) -> LoSSamples:
    """Sample ``fields`` at the events of ``skewer_bundle`` for every line (eager: the fields may be any callables).

    Parameters
    ----------
    a_dagger : reference scale factor of v_dagger: None (the bundle's ``a0``), scalar, (N_sk,), (N_sk, N_ppsk) or
        (N_sk * N_ppsk,).
    ii_cosmology : which cosmology of the bundle's scale factors.
    line_elements : (N_sk, N_ppsk) [cMpc]; None takes the trapezoid weights of the points (point-sampled fields),
        explicit ones are used for segment averages (they conserve the column for any spacing).
    """
    qv = skewer_bundle.quadri_vectors[ii_cosmology]
    N_sk, N_ppsk = qv.shape[:2]
    n_pts = N_sk * N_ppsk
    if line_elements is not None:
        w = jnp.asarray(line_elements)
        if w.shape != (N_sk, N_ppsk):
            raise ValueError(f"line_elements must be ({N_sk}, {N_ppsk}), got {w.shape}.")
    else:
        w = trapezoid_line_elements(qv[..., 1:])
    w_m = (w * phys.Mpc_to_m).reshape(n_pts)
    a = jnp.asarray(skewer_bundle.a0 if a_dagger is None else a_dagger)
    if a.ndim == 0 or a.shape == (n_pts,):
        a_d = a
    elif a.shape == (N_sk,):
        a_d = jnp.repeat(a[:, None], N_ppsk, axis=1).reshape(n_pts)
    elif a.shape == (N_sk, N_ppsk):
        a_d = a.reshape(n_pts)
    else:
        raise ValueError(f"`a_dagger` must be None, scalar, ({N_sk},), ({N_sk}, {N_ppsk}), or ({n_pts},). Got {a.shape}.")
    qv_flat = qv.reshape(n_pts, 4)
    f = _line_fields(fields, line_bundle, qv_flat, skewer_bundle.los_directions.reshape(n_pts, 3),
                     ("temperature", "turbulence", "collisional", "number_density"))
    per_skewer = lambda x: x.reshape(x.shape[:-1] + (N_sk, N_ppsk))
    return LoSSamples(a_dagger=jnp.broadcast_to(a_d, (n_pts,)).reshape(N_sk, N_ppsk), scale_factor=per_skewer(qv_flat[:, 0]),
                      weights_m=per_skewer(w_m), temperature=per_skewer(f["temperature"]), turbulence=per_skewer(f["turbulence"]),
                      collisional=per_skewer(f["collisional"]), los_velocity=per_skewer(f["LoS_velocity"]),
                      number_density=per_skewer(f["number_density"]))


@partial(jax.jit, static_argnames=("voigt", "detuning", "velocity_period", "v_chunk", "batch_size", "per_point", "dtype"))
def optical_depth_from_samples(samples: LoSSamples, velocities_kms, line_bundle, *, voigt: str = "ExoJAX", detuning: str = "exact",
                               velocity_period: Optional[float] = None, v_chunk: Optional[int] = None,
                               batch_size: Optional[int] = None, per_point: bool = False,
                               dtype: Optional[str] = None) -> jnp.ndarray:
    """tau (N_lines, N_sk, N_v) of every skewer of ``samples``: ``optical_depth_kernel`` mapped over skewers.

    v_chunk : velocities per kernel call (memory ~ N_lines x N_p x v_chunk); batch_size : skewers vmapped together
    inside the sequential map (None: one at a time, the lowest memory). per_point : (N_lines, N_sk, N_p, N_v).
    dtype : None (default: the dtype of the inputs) or e.g. "float32": opt-in lower-precision kernel (2-3x faster on
        the CPU; max |dF| ~1e-3 on TNG50-4 skewers); the result is cast back to the input dtype.
    """
    out_dtype = samples.temperature.dtype
    if dtype is not None:
        samples, velocities_kms = jax.tree_util.tree_map(lambda x: x.astype(dtype), (samples, velocities_kms))
    lam, A21, mass, f_osc = (x if dtype is None else x.astype(dtype)
                             for x in (line_bundle.lambda0_A, line_bundle.A21_GHz, line_bundle.mass_amu, line_bundle.f_osc))
    n_v = velocities_kms.shape[0]
    nv = n_v if v_chunk is None else int(v_chunk)

    def one_skewer(s):
        a_d, a, w, T, tb, co, lv, n = s
        chunks = [optical_depth_kernel(velocities_kms[v0:v0 + nv], lam, A21, mass, f_osc, a_d, a, lv, T, tb, co, n, w, voigt=voigt,
                                       detuning=detuning, velocity_period=velocity_period, per_point=per_point)
                  for v0 in range(0, n_v, nv)]
        return jnp.concatenate(chunks, axis=-1)

    sk = lambda x: jnp.moveaxis(x, 1, 0)                          # (N_lines, N_sk, N_p) -> (N_sk, N_lines, N_p)
    xs = (samples.a_dagger, samples.scale_factor, samples.weights_m, sk(samples.temperature), sk(samples.turbulence),
          sk(samples.collisional), sk(samples.los_velocity), sk(samples.number_density))
    return jnp.moveaxis(jax.lax.map(one_skewer, xs, batch_size=batch_size), 0, 1).astype(out_dtype)


def optical_depth_sharded(samples: LoSSamples, velocities_kms, line_bundle, *, devices=None, **kw) -> jnp.ndarray:
    """``optical_depth_from_samples`` with the skewers split over ``devices`` (default: all ``jax.devices()``).

    The skewer axis is padded to a multiple of the number of devices (copies of the last skewer, dropped from the
    result), split with ``shard_map`` over a 1D mesh, and every device maps its own skewers; the velocity grid and the
    line data are replicated. Each skewer is computed exactly as on one device, so the result is the same; with one
    device this is ``optical_depth_from_samples``. kw: its options (``voigt``, ``detuning``, ``v_chunk``, ``dtype``, ...).
    CPU cores can be exposed as devices with ``XLA_FLAGS=--xla_force_host_platform_device_count=N`` (before importing JAX).
    """
    devices = list(jax.devices() if devices is None else devices)
    velocities_kms = jnp.asarray(velocities_kms)
    if len(devices) == 1:
        return optical_depth_from_samples(samples, velocities_kms, line_bundle, **kw)
    n_sk, n_dev = samples.scale_factor.shape[0], len(devices)
    pad = -n_sk % n_dev
    skewer_axis = lambda x: 0 if x.ndim == 2 else 1                 # (N_sk, N_p) or (N_lines, N_sk, N_p)
    padded = jax.tree_util.tree_map(
        lambda x: jnp.pad(x, [(0, pad) if i == skewer_axis(x) else (0, 0) for i in range(x.ndim)], mode="edge"), samples)
    spec = jax.tree_util.tree_map(lambda x: PartitionSpec("sk") if x.ndim == 2 else PartitionSpec(None, "sk"), padded)
    run = shard_map(lambda s, v, lb: optical_depth_from_samples(s, v, lb, **kw), mesh=Mesh(np.array(devices), ("sk",)),
                    in_specs=(spec, PartitionSpec(), PartitionSpec()), out_specs=PartitionSpec(None, "sk"), check_rep=False)
    return jax.jit(run)(padded, velocities_kms, line_bundle)[:, :n_sk]


# --------------------------------------------------------------------------- sampling resolution
def sampling_resolution_report(line_centre_kms, b_kms, weight, tolerance: float = 0.25) -> dict:
    """Are consecutive points closer in line-centre velocity than ``tolerance`` x the line width? (notes
    ``app_numerical_resolution.tex``; NumPy, diagnostics only).

    The quadrature puts point i's absorption at its line centre v_c,i; for a top hat of width W replaced by a delta
    the peak of a Gaussian of Doppler parameter b changes by W^2 / (12 b^2) (0.5 % for W = b / 4).
    line_centre_kms, b_kms (``voigt_width_kms``), weight (n w, the column of each point) : (N_lines, N_sk, N_ppsk).

    Returns
    -------
    dict(max_ratio (N_lines,), column_fraction (N_lines,) of the column in under-resolved intervals, ok, b_min (N_lines,)).
    """
    dv = np.abs(np.diff(np.asarray(line_centre_kms), axis=-1))
    b = np.asarray(b_kms)
    ratio = dv / np.maximum(np.minimum(b[..., 1:], b[..., :-1]), 1e-30)
    w = np.asarray(weight)
    wm = 0.5 * (w[..., 1:] + w[..., :-1])
    frac = (wm * (ratio > tolerance)).sum(axis=(-2, -1)) / np.maximum(wm.sum(axis=(-2, -1)), 1e-300)
    return dict(max_ratio=ratio.max(axis=(-2, -1)), column_fraction=frac, ok=bool(np.all(frac < 1e-3)), b_min=b.min(axis=(-2, -1)))


def resolution_report(samples: LoSSamples, line_bundle, tolerance: float = 0.25) -> dict:
    """``sampling_resolution_report`` of the points of ``samples``."""
    lam, shp = line_bundle.lambda0_A, samples.shape
    n_l = shp[0]
    flat = lambda x: x.reshape(n_l, -1)
    v_c = line_centre_velocity_kms(samples.a_dagger.reshape(-1), samples.scale_factor.reshape(-1), flat(samples.los_velocity))
    sigma, gamma = voigt_widths_GHz(flat(samples.temperature), flat(samples.turbulence), flat(samples.collisional), lam,
                                    line_bundle.A21_GHz, line_bundle.mass_amu)
    b = voigt_width_kms(sigma, gamma, lam)
    col = flat(samples.number_density) * samples.weights_m.reshape(-1)[None, :]
    return sampling_resolution_report(v_c.reshape(shp), b.reshape(shp), col.reshape(shp), tolerance)


# --------------------------------------------------------------------------- the eager entry point
def compute_optical_depth(skewer_bundle, velocities_kms: ArrayLike, line_bundle, fields, *, a_dagger: Optional[ArrayLike] = None,
                  ii_cosmology: int = 0, line_elements: Optional[ArrayLike] = None, voigt: str = "ExoJAX",
                  detuning: str = "exact", velocity_period: Optional[float] = None,
                  resolution_tolerance: Optional[float] = 0.25, v_chunk: Optional[int] = None,
                  batch_size: Optional[int] = None, dtype: Optional[str] = None, devices=None) -> jnp.ndarray:
    """tau (N_lines, N_sk, N_v) of every line along every skewer of ``skewer_bundle`` (F = exp(-tau)).

    Parameters
    ----------
    velocities_kms : (N_v,) line-centred velocity grid v_dagger [km/s], common to all lines.
    a_dagger, ii_cosmology, line_elements : see ``los_samples``.
    voigt : "ExoJAX" (default) or "convolution" ("scipy_wofz" cannot run inside the jitted kernel).
    detuning : "exact" (default) or "linear" (fake_spectra's first-order mapping; matched mode).
    velocity_period : period [km/s] of a periodic box: each point absorbs through the nearest image of its detuning.
    resolution_tolerance : warn when consecutive points are farther apart in line-centre velocity than this
        fraction of the line width (``resolution_report``); None disables the check.
    v_chunk, batch_size : memory controls of ``optical_depth_from_samples``.
    dtype : opt-in kernel precision, e.g. "float32" (``optical_depth_from_samples``); None keeps the input precision.
    devices : None (one device, the default) or a list of devices to split the skewers over (``optical_depth_sharded``).
    """
    samples = los_samples(skewer_bundle, fields, line_bundle, a_dagger=a_dagger, ii_cosmology=ii_cosmology,
                          line_elements=line_elements)
    if resolution_tolerance is not None:
        rep = resolution_report(samples, line_bundle, resolution_tolerance)
        if not rep["ok"]:
            warnings.warn(
                "compute_optical_depth: the points are too sparse for the line widths: "
                + ", ".join(f"line {j}: max dv_c / width = {rep['max_ratio'][j]:.2f}, column fraction "
                            f"{rep['column_fraction'][j]:.1e} above tolerance {resolution_tolerance}"
                            for j in range(line_bundle.n_lines))
                + ". Sample more densely (spacing <= tolerance x b_min / (dv/dx)).", stacklevel=2)
    kw = dict(voigt=voigt, detuning=detuning, velocity_period=None if velocity_period is None else float(velocity_period),
              v_chunk=v_chunk, batch_size=batch_size, dtype=dtype)
    if devices is not None:
        return optical_depth_sharded(samples, velocities_kms, line_bundle, devices=devices, **kw)
    return optical_depth_from_samples(samples, jnp.asarray(velocities_kms), line_bundle, **kw)


def wrap_to_ref_line(optical_depth: ArrayLike, velocities_kms: ArrayLike, line_bundle) -> jnp.ndarray:
    """Put every line's tau on the reference line's velocity frame: tau_j^ref(v) = tau_j(v + delta_v_j).

    A feature at v_j in line j's own frame appears at v_j - delta_v_j, delta_v_j = c ln(lambda_ref / lambda_j)
    (lines bluer than the reference move to lower velocities). The uniform grid is treated as periodic, period
    N_v dv (snapshot boxes). optical_depth : (N_lines, N_sk, N_v); returns the same shape.
    """
    tau, v = jnp.asarray(optical_depth), jnp.asarray(velocities_kms)
    dv, v0 = v[1] - v[0], v[0]
    period = v.size * dv

    def shift_profile(tau_1d, shift):
        vq = jnp.mod(v + shift - v0, period) + v0
        return jnp.interp(vq, v, tau_1d, period=period)

    return jax.vmap(lambda t, s: jax.vmap(lambda t1: shift_profile(t1, s))(t), in_axes=(0, 0))(tau, jnp.asarray(line_bundle.delta_v))
