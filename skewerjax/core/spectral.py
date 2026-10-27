"""Spectral coordinates: the observer's velocity v_dagger, local wavelengths lambda' and detunings nu (review notes, Sec. 2.2)."""

from __future__ import annotations

import jax.numpy as jnp
from jax.typing import ArrayLike

from skewerjax.core.constants import phys


def local_wavelength_A(
    velocities_kms: ArrayLike, lambda0_A: ArrayLike, a_dagger: ArrayLike, scale_factor: ArrayLike, LoS_velocity: ArrayLike
) -> jnp.ndarray:
    """lambda'_j = lambda_j exp(v_dagger / c) a(x) / a_dagger D^-1(x), D^-1 = sqrt((1 + beta) / (1 - beta)) (Eq. velocity_units_normalized).

    Parameters
    ----------
    velocities_kms : (N_v,) line-centred velocities v_dagger [km/s]; lambda0_A : (N_lines,) [A].
    a_dagger : scalar or (*batch,) reference scale factor; scale_factor : (*batch,) a(x).
    LoS_velocity : (N_lines, *batch) peculiar velocity along the photon direction [km/s], beta = v / c.

    Returns
    -------
    (N_lines, *batch, N_v) local wavelengths [A] (a full cube: the optical depth uses ``frequency_offset_GHz``).
    """
    velocities_kms, lambda0_A, a_dagger, scale_factor, LoS_velocity = (
        jnp.asarray(v) for v in (velocities_kms, lambda0_A, a_dagger, scale_factor, LoS_velocity)
    )
    if scale_factor.shape != LoS_velocity.shape[1:]:
        raise ValueError(
            f"scale_factor shape {scale_factor.shape} incompatible with LoS_velocity shape "
            f"{LoS_velocity.shape}. Expected scale_factor.shape == LoS_velocity.shape[1:]"
        )
    if a_dagger.ndim != 0 and a_dagger.shape != scale_factor.shape:
        raise ValueError(f"a_dagger must be scalar or have shape {scale_factor.shape}; got {a_dagger.shape}")
    nb = scale_factor.ndim
    lam = lambda0_A[(slice(None),) + (None,) * nb + (None,)]
    vel = jnp.exp(velocities_kms[(None,) * (nb + 1) + (slice(None),)] / phys.c_kms)
    ratio = scale_factor[None, ..., None] / (a_dagger if a_dagger.ndim == 0 else a_dagger[None, ..., None])
    beta = LoS_velocity[..., None] / phys.c_kms
    return lam * vel * ratio * jnp.sqrt((1.0 + beta) / (1.0 - beta))


def local_wavelength_from_observed_A(
    wavelengths_A: ArrayLike, scale_factor_obs: ArrayLike, scale_factor: ArrayLike, LoS_velocity: ArrayLike
) -> jnp.ndarray:
    """lambda' = lambda_obs a(x) / a_obs D^-1(x) for observed-wavelength grids (Eq. wavelength_cosmological_z_and_doppler).

    Parameters
    ----------
    wavelengths_A : (N_v,) or (N_lines, N_v) observed wavelengths [A] (e.g. from ``line_wavelength_grid_A``).
    scale_factor_obs : scalar, (N_lines,), (*batch,) or (N_lines, *batch) observer scale factor.
    scale_factor : (*batch,) a(x); LoS_velocity : (N_lines, *batch) [km/s].

    Returns
    -------
    (N_lines, *batch, N_v) local wavelengths [A]; equal to ``local_wavelength_A`` on the grid of
    ``line_wavelength_grid_A`` with the same a_obs.
    """
    wavelengths_A, scale_factor_obs, scale_factor, LoS_velocity = (
        jnp.asarray(v) for v in (wavelengths_A, scale_factor_obs, scale_factor, LoS_velocity)
    )
    if scale_factor.shape != LoS_velocity.shape[1:]:
        raise ValueError(
            f"`scale_factor` shape {scale_factor.shape} incompatible with `LoS_velocity` shape "
            f"{LoS_velocity.shape}. Expected scale_factor.shape == LoS_velocity.shape[1:]"
        )
    N_lines, batch, nb = LoS_velocity.shape[0], scale_factor.shape, scale_factor.ndim
    if wavelengths_A.ndim == 1:
        wl = wavelengths_A[(None,) + (None,) * nb + (slice(None),)]
    elif wavelengths_A.ndim == 2:
        if wavelengths_A.shape[0] != N_lines:
            raise ValueError(f"`wavelengths_A` has shape {wavelengths_A.shape}, but LoS_velocity implies N_lines={N_lines}.")
        wl = wavelengths_A[(slice(None),) + (None,) * nb + (slice(None),)]
    else:
        raise ValueError(f"`wavelengths_A` must have shape (N_v,) or (N_lines, N_v), got {wavelengths_A.shape}.")
    a = scale_factor[None, ..., None]
    if scale_factor_obs.ndim == 0:
        ratio = a / scale_factor_obs
    elif scale_factor_obs.shape == (N_lines,):
        ratio = a / scale_factor_obs[(slice(None),) + (None,) * nb + (None,)]
    elif scale_factor_obs.shape == batch:
        ratio = a / scale_factor_obs[None, ..., None]
    elif scale_factor_obs.shape == LoS_velocity.shape:
        ratio = a / scale_factor_obs[..., None]
    else:
        raise ValueError(
            f"`scale_factor_obs` must be scalar, ({N_lines},), {batch} or {LoS_velocity.shape}; got {scale_factor_obs.shape}."
        )
    beta = LoS_velocity[..., None] / phys.c_kms
    return wl * ratio * jnp.sqrt((1.0 + beta) / (1.0 - beta))


def line_centre_velocity_kms(a_dagger: ArrayLike, scale_factor: ArrayLike, LoS_velocity: ArrayLike) -> jnp.ndarray:
    """v_c = c [ln(a_dagger / a) + (ln(1 - beta) - ln(1 + beta)) / 2]: where each point absorbs at line centre [km/s].

    a_dagger : scalar or (N_points,); scale_factor : (N_points,); LoS_velocity : (N_lines, N_points) [km/s].
    Returns (N_lines, N_points).
    """
    beta = jnp.asarray(LoS_velocity) / phys.c_kms
    ratio = jnp.asarray(a_dagger) / jnp.asarray(scale_factor)
    return phys.c_kms * (jnp.log(ratio)[None, :] + 0.5 * (jnp.log1p(-beta) - jnp.log1p(beta)))


def frequency_offset_GHz(
    velocities_kms: ArrayLike,
    lambda0_A: ArrayLike,
    a_dagger: ArrayLike,
    scale_factor: ArrayLike,
    LoS_velocity: ArrayLike,
    velocity_period: ArrayLike | None = None,
    detuning: str = "exact",
) -> jnp.ndarray:
    """nu_j = c (1 / lambda' - 1 / lambda_j) [GHz] of every (line, piece, velocity), from the detuning dv = v - v_c.

    "exact" (default): nu = (c / lambda_j) expm1(-dv / c), the exact exponential mapping. "linear": nu = -(c / lambda_j)
    dv / c, the first-order mapping of fake_spectra (Sec. approx_velocity_units; matched mode only).
    ``velocity_period`` [km/s] (a number, possibly traced) replaces dv by its nearest periodic image in [-P/2, P/2):
    the opt-in periodic spectrum of ``optical_depth.compute_optical_depth``; None (default) does not wrap.

    Shapes: velocities (N_v,) (one grid for every line) or (N_lines, N_v) (one grid per line, e.g. the reference
    line's grid in every line's frame), lambda0_A (N_lines,), a_dagger scalar or (N_points,), scale_factor (N_points,),
    LoS_velocity (N_lines, N_points) -> (N_lines, N_points, N_v).
    """
    velocities_kms, lambda0_A, a_dagger, scale_factor, LoS_velocity = (
        jnp.asarray(v) for v in (velocities_kms, lambda0_A, a_dagger, scale_factor, LoS_velocity)
    )
    if scale_factor.ndim != 1:
        raise ValueError(f"`frequency_offset_GHz` expects flattened scale_factor (N_points,), got {scale_factor.shape}.")
    if LoS_velocity.ndim != 2:
        raise ValueError(f"`frequency_offset_GHz` expects flattened LoS_velocity (N_lines, N_points), got {LoS_velocity.shape}.")
    if LoS_velocity.shape[1] != scale_factor.shape[0]:
        raise ValueError(
            f"LoS_velocity.shape[1]={LoS_velocity.shape[1]} does not match scale_factor.shape[0]={scale_factor.shape[0]}."
        )
    if a_dagger.ndim != 0 and a_dagger.shape != scale_factor.shape:
        raise ValueError(f"a_dagger must be scalar or have shape {scale_factor.shape}; got {a_dagger.shape}.")
    if velocities_kms.ndim == 2 and velocities_kms.shape[0] != LoS_velocity.shape[0]:
        raise ValueError(f"per-line velocities must be (N_lines={LoS_velocity.shape[0]}, N_v), got {velocities_kms.shape}.")
    if velocities_kms.ndim not in (1, 2):
        raise ValueError(f"velocities must be (N_v,) or (N_lines, N_v), got {velocities_kms.shape}.")
    if detuning not in ("exact", "linear"):
        raise ValueError(f"detuning must be 'exact' or 'linear', got {detuning!r}.")
    v_c = line_centre_velocity_kms(a_dagger, scale_factor, LoS_velocity)
    v = velocities_kms[None, None, :] if velocities_kms.ndim == 1 else velocities_kms[:, None, :]
    delta = v - v_c[..., None]
    if velocity_period is not None:
        P = jnp.asarray(velocity_period, dtype=delta.dtype)
        delta = jnp.mod(delta + 0.5 * P, P) - 0.5 * P
    x = -delta / phys.c_kms
    factor = jnp.expm1(x) if detuning == "exact" else x
    return factor * (phys.c * 10.0) / lambda0_A[:, None, None]  # c / lambda [GHz] = 10 x (m/s) / A


def line_wavelength_grid_A(velocities_kms: ArrayLike, line_bundle, a_dagger: ArrayLike, a_obs: ArrayLike = 1.0) -> jnp.ndarray:
    """lambda_obs,j(v) = lambda_j (a_obs / a_dagger) exp(v / c): the observed wavelengths of each line's velocity grid.

    a_dagger, a_obs : scalars or (N_lines,). Returns (N_lines, N_v) [A].
    """
    velocities_kms, a_dagger, a_obs = jnp.asarray(velocities_kms), jnp.asarray(a_dagger), jnp.asarray(a_obs)
    lambda0_A = jnp.asarray(line_bundle.lambda0_A)
    N_lines = lambda0_A.shape[0]
    per_line = {}
    for name, a in (("a_obs", a_obs), ("a_dagger", a_dagger)):
        if a.ndim == 0:
            per_line[name] = jnp.full((N_lines,), a)
        elif a.shape == (N_lines,):
            per_line[name] = a
        else:
            raise ValueError(f"`{name}` must be scalar or have shape ({N_lines},), got {a.shape}.")
    return (
        lambda0_A[:, None]
        * (per_line["a_obs"][:, None] / per_line["a_dagger"][:, None])
        * jnp.exp(velocities_kms[None, :] / phys.c_kms)
    )


def velocities_in_line_frames(velocities_kms: ArrayLike, line_bundle, reference_line_idx: int = 0) -> jnp.ndarray:
    """v^(j) = v^(ref) + c ln(lambda_ref / lambda_j): the velocity grid of a reference line in every line's own frame.

    The same observed wavelength lambda_ref exp(v_ref / c) = lambda_j exp(v_j / c), independent of redshift
    (review notes). velocities_kms : (N_v,) [km/s]; returns (N_lines, N_v).
    """
    velocities_kms = jnp.asarray(velocities_kms)
    lambda0_A = jnp.asarray(line_bundle.lambda0_A)
    shifts = phys.c_kms * jnp.log(lambda0_A[reference_line_idx] / lambda0_A)
    if velocities_kms.ndim == 1:
        velocities_kms = velocities_kms[None, :]
    return velocities_kms + shifts[:, None]
