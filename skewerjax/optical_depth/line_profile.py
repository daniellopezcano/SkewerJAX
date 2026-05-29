from __future__ import annotations

from typing import Mapping
from math import prod
from jax.typing import ArrayLike

from jax import vmap
import jax.numpy as jnp

from skewerjax.optical_depth.voigt import log10_voigt
from skewerjax.constants.physical_constants import phys


def compute_lambda_prime(
    velocities_kms: ArrayLike,
    lambda0_A: ArrayLike,
    a_dagger: ArrayLike,
    scale_factor: ArrayLike,
    LoS_velocity: ArrayLike,
) -> jnp.ndarray:
    """
    Compute local wavelengths using exact line-centered velocity units.

    For each transition j,

        lambda'_j(x, v)
        =
        lambda_j exp(v / c) a(x) / a_dagger D^{-1}(x),

    with

        D^{-1} = sqrt((1 + beta) / (1 - beta)),
        beta = v_parallel / c.

    Parameters
    ----------
    velocities_kms : array, shape (N_v,)
        Common line-centered velocity grid in km/s.

    lambda0_A : array, shape (N_lines,)
        Rest wavelengths of each transition in Angstrom.

    a_dagger : scalar or array
        Reference scale factor.
        Must be scalar or have shape matching `scale_factor`.

    scale_factor : array, shape (*batch_shape,)
        Local scale factor a(x).

    LoS_velocity : array, shape (N_lines, *batch_shape)
        Line-of-sight peculiar velocity in km/s, tiled to lines.

    Returns
    -------
    lambdas_A_prime : array, shape (N_lines, *batch_shape, N_v)
        Local wavelength seen by each gas element, in Angstrom.

    Notes
    -----
    This function necessarily returns a heavy wavelength cube. For the
    line-profile calculation below, use `compute_nu_offset_GHz()` instead,
    which avoids materializing this wavelength cube.
    """
    velocities_kms = jnp.asarray(velocities_kms)
    lambda0_A = jnp.asarray(lambda0_A)
    a_dagger = jnp.asarray(a_dagger)
    scale_factor = jnp.asarray(scale_factor)
    LoS_velocity = jnp.asarray(LoS_velocity)

    if scale_factor.shape != LoS_velocity.shape[1:]:
        raise ValueError(
            f"scale_factor shape {scale_factor.shape} incompatible with "
            f"LoS_velocity shape {LoS_velocity.shape}. "
            f"Expected scale_factor.shape == LoS_velocity.shape[1:]"
        )

    if a_dagger.ndim != 0 and a_dagger.shape != scale_factor.shape:
        raise ValueError(
            f"a_dagger must be scalar or have shape {scale_factor.shape}; "
            f"got {a_dagger.shape}"
        )

    lambda0_expanded = lambda0_A[
        (slice(None),) + (None,) * scale_factor.ndim + (None,)
    ]

    velocity_factor = jnp.exp(
        velocities_kms[
            (None,) * (scale_factor.ndim + 1) + (slice(None),)
        ] / phys.c_kms
    )

    if a_dagger.ndim == 0:
        scale_factor_ratio = scale_factor[None, ..., None] / a_dagger
    else:
        scale_factor_ratio = (
            scale_factor[None, ..., None]
            / a_dagger[None, ..., None]
        )

    beta = LoS_velocity[..., None] / phys.c_kms
    doppler_inverse = jnp.sqrt((1.0 + beta) / (1.0 - beta))

    lambdas_A_prime = (
        lambda0_expanded
        * velocity_factor
        * scale_factor_ratio
        * doppler_inverse
    )

    return lambdas_A_prime


def compute_nu_offset_GHz(
    velocities_kms: ArrayLike,
    lambda0_A: ArrayLike,
    a_dagger: ArrayLike,
    scale_factor: ArrayLike,
    LoS_velocity: ArrayLike,
) -> jnp.ndarray:
    """
    Memory-lighter frequency-offset computation.

    Instead of materializing

        lambda_prime = lambda0 * exp(v/c) * a/a_dagger * D^{-1},

    and then computing

        nu = (1/lambda_prime - 1/lambda0) * c * 10,

    this computes the reciprocal directly:

        1/lambda_prime
        =
        1/lambda0
        * exp(-v/c)
        * a_dagger/a
        * sqrt((1 - beta)/(1 + beta)).

    Parameters
    ----------
    velocities_kms : array, shape (N_v,)
    lambda0_A : array, shape (N_lines,)
    a_dagger : scalar or shape (N_points,)
    scale_factor : array, shape (N_points,)
    LoS_velocity : array, shape (N_lines, N_points)

    Returns
    -------
    nu_GHz : array, shape (N_lines, N_points, N_v)
        Frequency offset in GHz.
    """
    velocities_kms = jnp.asarray(velocities_kms)
    lambda0_A = jnp.asarray(lambda0_A)
    a_dagger = jnp.asarray(a_dagger)
    scale_factor = jnp.asarray(scale_factor)
    LoS_velocity = jnp.asarray(LoS_velocity)

    if scale_factor.ndim != 1:
        raise ValueError(
            "`compute_nu_offset_GHz` expects flattened scale_factor "
            f"with shape (N_points,), got {scale_factor.shape}."
        )

    if LoS_velocity.ndim != 2:
        raise ValueError(
            "`compute_nu_offset_GHz` expects flattened LoS_velocity "
            f"with shape (N_lines, N_points), got {LoS_velocity.shape}."
        )

    if LoS_velocity.shape[1] != scale_factor.shape[0]:
        raise ValueError(
            f"LoS_velocity.shape[1]={LoS_velocity.shape[1]} does not match "
            f"scale_factor.shape[0]={scale_factor.shape[0]}."
        )

    if a_dagger.ndim != 0 and a_dagger.shape != scale_factor.shape:
        raise ValueError(
            f"a_dagger must be scalar or have shape {scale_factor.shape}; "
            f"got {a_dagger.shape}."
        )

    inv_lambda0 = 1.0 / lambda0_A[:, None, None]

    velocity_factor_inv = jnp.exp(
        -velocities_kms[None, None, :] / phys.c_kms
    )

    if a_dagger.ndim == 0:
        scale_factor_ratio_inv = a_dagger / scale_factor[None, :, None]
    else:
        scale_factor_ratio_inv = (
            a_dagger[None, :, None]
            / scale_factor[None, :, None]
        )

    beta = LoS_velocity[..., None] / phys.c_kms
    doppler_forward = jnp.sqrt((1.0 - beta) / (1.0 + beta))

    nu_GHz = (
        inv_lambda0
        * (
            velocity_factor_inv
            * scale_factor_ratio_inv
            * doppler_forward
            - 1.0
        )
        * phys.c
        * 10.0
    )

    return nu_GHz


def compute_line_profiles(
    quadri_vectors: jnp.ndarray,
    LoS_directions: jnp.ndarray,
    velocities_kms: jnp.ndarray,
    a_dagger: ArrayLike,
    line_bundle,
    fields,
    voigt_implementation_mode: str = "ExoJAX",
) -> jnp.ndarray:
    """
    Compute Voigt line profiles using line-centered velocity units.

    Parameters
    ----------
    quadri_vectors : array, shape (..., 4)
        Sampling events in the form [a, x, y, z].
        The leading dimensions define the spatial batch shape.

    LoS_directions : array, shape (..., 3)
        Line-of-sight direction vectors at each spatial point.

    velocities_kms : array, shape (N_v,)
        Common line-centered velocity grid in km/s.

    a_dagger : scalar or array
        Reference scale factor.
        Accepted shapes:
            - scalar
            - same as quadri_vectors.shape[:-1]
            - flattened shape (N_points,)

    line_bundle : LineBundle-like
        Object describing the set of lines.

    fields : Fields-like
        Field container.

    voigt_implementation_mode : str
        Passed through to `log10_voigt`.

    Returns
    -------
    phi : array, shape (N_lines, *batch_shape, N_v)
        Line profiles in time units [ns].
    """

    # ------------------------------------------------------------------
    # 0. Basic validation and bookkeeping
    # ------------------------------------------------------------------
    velocities_kms = jnp.asarray(velocities_kms)
    a_dagger = jnp.asarray(a_dagger)

    N_v = int(velocities_kms.size)

    if quadri_vectors.ndim < 2 or quadri_vectors.shape[-1] != 4:
        raise ValueError(
            "`quadri_vectors` must have shape (..., 4) with last axis [a,x,y,z]. "
            f"Got shape {quadri_vectors.shape}."
        )

    batch_shape = quadri_vectors.shape[:-1]
    N_points = int(prod(batch_shape))

    species_fields = set(fields.species_keys)
    species_lines = set(line_bundle.species_keys)

    if species_fields != species_lines:
        raise ValueError(
            "Species keys mismatch between fields and line_bundle. "
            f"fields={species_fields}, line_bundle={species_lines}"
        )

    N_lines = int(line_bundle.n_lines)
    N_species = int(line_bundle.n_species)

    if N_lines < N_species:
        raise ValueError(
            f"More species than lines: N_lines={N_lines}, N_species={N_species}"
        )

    LoS_directions = jnp.asarray(LoS_directions)

    if LoS_directions.shape != batch_shape + (3,):
        raise ValueError(
            f"`LoS_directions` shape {LoS_directions.shape} is incompatible with "
            f"batch_shape={batch_shape}; expected {batch_shape + (3,)}"
        )

    required_scalar_fields = ("temperature", "collisional", "turbulence")
    required_vector_field = "velocity"

    for sp in line_bundle.species_keys:
        available = set(fields.field_keys[sp])

        missing = [f for f in required_scalar_fields if f not in available]
        if missing:
            raise ValueError(
                f"Missing fields for species '{sp}': {missing}. "
                "These are required for line-profile computation."
            )

        if required_vector_field not in available:
            raise ValueError(
                f"Missing 'velocity' field for species '{sp}', "
                "required to build LoS_velocity."
            )

    # ------------------------------------------------------------------
    # 1. Flatten spatial dimensions
    # ------------------------------------------------------------------
    quadri_vectors_flat = quadri_vectors.reshape(N_points, 4)
    LoS_flat = LoS_directions.reshape(N_points, 3)
    scale_factors_flat = quadri_vectors_flat[:, 0]

    if a_dagger.ndim == 0:
        a_dagger_flat = a_dagger
    elif a_dagger.shape == batch_shape:
        a_dagger_flat = a_dagger.reshape(N_points)
    elif a_dagger.shape == (N_points,):
        a_dagger_flat = a_dagger
    else:
        raise ValueError(
            f"`a_dagger` must be scalar, have shape {batch_shape}, "
            f"or have shape ({N_points},). Got {a_dagger.shape}."
        )

    # ------------------------------------------------------------------
    # 2. Map species -> line indices, and tile fields to lines
    # ------------------------------------------------------------------
    indices_for_species: Mapping[str, jnp.ndarray] = {
        sp: line_bundle.indices_for_species(sp)
        for sp in line_bundle.species_keys
    }

    collisional = fields.tile_field_to_lines(
        "collisional",
        quadri_vectors_flat,
        indices_for_species=indices_for_species,
    )

    temperature = fields.tile_field_to_lines(
        "temperature",
        quadri_vectors_flat,
        indices_for_species=indices_for_species,
    )

    turbulence = fields.tile_field_to_lines(
        "turbulence",
        quadri_vectors_flat,
        indices_for_species=indices_for_species,
    )

    LoS_velocity = fields.tile_field_to_lines(
        "LoS_velocity",
        quadri_vectors_flat,
        indices_for_species=indices_for_species,
        LoS_directions=LoS_flat,
    )

    # LoS_flat is no longer needed after LoS_velocity is built.
    del LoS_flat

    # ------------------------------------------------------------------
    # 3. Extract line parameters
    # ------------------------------------------------------------------
    lambda0_A = jnp.asarray(line_bundle.lambda0_A)
    A21_GHz = jnp.asarray(line_bundle.A21_GHz)
    mass_amu = jnp.asarray(line_bundle.mass_amu)

    # ------------------------------------------------------------------
    # 4. Compute Voigt width parameters
    # ------------------------------------------------------------------
    gamma_GHz = (A21_GHz[:, None] + collisional) / (4.0 * jnp.pi)

    v_th_kms = jnp.sqrt(
        2.0 * phys.kB * temperature / mass_amu[:, None]
    )

    v_eff_A_GHz = (v_th_kms + turbulence) * 10.0

    sigma_GHz = (
        v_eff_A_GHz
        / (jnp.sqrt(2.0) * lambda0_A[:, None])
    )

    # These heavy tiled fields are no longer needed.
    del collisional, temperature, turbulence, v_th_kms, v_eff_A_GHz

    # ------------------------------------------------------------------
    # 5. Frequency offsets from line-centered velocity units
    # ------------------------------------------------------------------
    nu_GHz = compute_nu_offset_GHz(
        velocities_kms=velocities_kms,
        lambda0_A=lambda0_A,
        a_dagger=a_dagger_flat,
        scale_factor=scale_factors_flat,
        LoS_velocity=LoS_velocity,
    )

    # These are no longer needed after nu_GHz is built.
    del scale_factors_flat, LoS_velocity, a_dagger_flat

    # ------------------------------------------------------------------
    # 6. Flatten for Voigt evaluation
    # ------------------------------------------------------------------
    N_total = N_lines * N_points

    nu_flat = nu_GHz.reshape(N_total, N_v)
    sigma_flat = sigma_GHz.reshape(N_total)
    gamma_flat = gamma_GHz.reshape(N_total)

    # The unflattened arrays are no longer needed.
    del nu_GHz, sigma_GHz, gamma_GHz

    def _eval_voigt_batch(nu_batch, sigma_batch, gamma_batch):
        return vmap(
            lambda x, s, g: log10_voigt(
                x,
                s,
                g,
                mode=voigt_implementation_mode,
            ),
            in_axes=(0, 0, 0),
        )(nu_batch, sigma_batch, gamma_batch)

    log10_phi_flat = _eval_voigt_batch(
        nu_flat,
        sigma_flat,
        gamma_flat,
    )

    # Inputs to Voigt are no longer needed after log10_phi_flat is computed.
    del nu_flat, sigma_flat, gamma_flat

    # ------------------------------------------------------------------
    # 7. Convert back and reshape to the natural batch shape
    # ------------------------------------------------------------------
    phi_flat = 10.0 ** log10_phi_flat

    # Drop log version before keeping the linear profile.
    del log10_phi_flat

    phi = phi_flat.reshape(
        N_lines,
        *batch_shape,
        N_v,
    )

    del phi_flat, quadri_vectors_flat

    return phi


def velocities_to_line_centered_wavelengths(
    velocities_kms: ArrayLike,
    line_bundle,
    a_dagger: ArrayLike,
    a_obs: ArrayLike = 1.0,
) -> jnp.ndarray:
    """
    Convert a common line-centered velocity grid to wavelength grids.

    For each transition j,

        v^(j) = c ln[(lambda_obs / lambda_j) * (a_dagger / a_obs_j)],

    therefore

        lambda_obs_j(v)
        =
        lambda_j * (a_obs_j / a_dagger) * exp(v / c).

    Parameters
    ----------
    velocities_kms : array, shape (N_v,)
        Velocity grid in km/s.
    line_bundle : LineBundle-like
        Must provide lambda0_A with shape (N_lines,).
    a_dagger : scalar or array broadcastable to (N_lines,)
        Reference scale factor used to define the line-centered velocity coordinate.
    a_obs : scalar or array, shape (N_lines,), default 1.0
        Observer scale factor. Can be defined globally or per line.

    Returns
    -------
    wavelengths_A : jnp.ndarray, shape (N_lines, N_v)
        Wavelength grid in Angstrom for each transition.
    """
    velocities_kms = jnp.asarray(velocities_kms)
    lambda0_A = jnp.asarray(line_bundle.lambda0_A)
    a_dagger = jnp.asarray(a_dagger)
    a_obs = jnp.asarray(a_obs)

    N_lines = lambda0_A.shape[0]

    if a_obs.ndim == 0:
        a_obs_per_line = jnp.full((N_lines,), a_obs)
    elif a_obs.shape == (N_lines,):
        a_obs_per_line = a_obs
    else:
        raise ValueError(
            f"`a_obs` must be scalar or have shape ({N_lines},), got {a_obs.shape}."
        )

    if a_dagger.ndim == 0:
        a_dagger_per_line = jnp.full((N_lines,), a_dagger)
    elif a_dagger.shape == (N_lines,):
        a_dagger_per_line = a_dagger
    else:
        raise ValueError(
            f"`a_dagger` must be scalar or have shape ({N_lines},), got {a_dagger.shape}."
        )

    wavelengths_A = (
        lambda0_A[:, None]
        * (a_obs_per_line[:, None] / a_dagger_per_line[:, None])
        * jnp.exp(velocities_kms[None, :] / phys.c_kms)
    )

    return wavelengths_A


def compute_lambda_prime_wavelength_units(
    wavelengths_A: ArrayLike,        # (N_v,) or (N_lines, N_v)
    scale_factor_obs: ArrayLike,     # scalar, (N_lines,), (*batch_shape), or (N_lines, *batch_shape)
    scale_factor: ArrayLike,         # (*batch_shape,)
    LoS_velocity: ArrayLike,         # (N_lines, *batch_shape), [km/s]
) -> jnp.ndarray:
    """
    Compute local wavelengths from observer-frame wavelength grids.

    This is the wavelength-units counterpart of `compute_lambda_prime`.

    It applies
        lambda'_j(x, lambda_obs) = lambda_obs_j * a(x) / a_obs * D^{-1}(x),

    with
        D^{-1} = sqrt((1 + beta) / (1 - beta)),  beta = v_parallel / c.

    This function is designed to work directly with the output of
        wavelengths_per_line_A = velocities_to_line_centered_wavelengths(
                velocities_kms, line_bundle, a_dagger, a_obs=scale_factor_obs,
            )

    where `wavelengths_per_line_A` has shape
        (N_lines, N_v).

    If the same `scale_factor_obs` is used in both functions, then
        lambda_obs_j(v) = lambda_j * (a_obs / a_dagger) * exp(v / c)

    implies
        lambda'_j(x, v) = lambda_j * exp(v / c) * a(x) / a_dagger * D^{-1}(x),

    which matches the velocity-units implementation.

    Parameters
    ----------
    wavelengths_A : jnp.ndarray, shape (N_v,) or (N_lines, N_v)
        Observer-frame wavelength grid in Angstrom.
        If shape is (N_v,), the same wavelength grid is used for all lines.
        If shape is (N_lines, N_v), each line has its own wavelength grid.

    scale_factor_obs : scalar or array
        Observer/reference scale factor used to define `wavelengths_A`.

        Accepted shapes:
            - scalar
            - (N_lines,)
            - (*batch_shape,)
            - (N_lines, *batch_shape)

    scale_factor : jnp.ndarray, shape (*batch_shape,)
        Local scale factor a(x).

    LoS_velocity : jnp.ndarray, shape (N_lines, *batch_shape)
        Line-of-sight peculiar velocity in km/s, tiled to lines.

    Returns
    -------
    lambdas_A_prime : jnp.ndarray, shape (N_lines, *batch_shape, N_v)
        Local wavelength seen by each gas element, in Angstrom.
    """
    wavelengths_A = jnp.asarray(wavelengths_A)
    scale_factor_obs = jnp.asarray(scale_factor_obs)
    scale_factor = jnp.asarray(scale_factor)
    LoS_velocity = jnp.asarray(LoS_velocity)

    if scale_factor.shape != LoS_velocity.shape[1:]:
        raise ValueError(
            f"`scale_factor` shape {scale_factor.shape} incompatible with "
            f"`LoS_velocity` shape {LoS_velocity.shape}. Expected "
            f"scale_factor.shape == LoS_velocity.shape[1:]"
        )

    N_lines = LoS_velocity.shape[0]
    batch_shape = scale_factor.shape
    batch_ndim = scale_factor.ndim

    # ------------------------------------------------------------
    # 1. Expand wavelength grid to (N_lines, *batch_shape, N_v)
    # ------------------------------------------------------------
    if wavelengths_A.ndim == 1:
        # Same wavelength grid for every line.
        wavelengths_expanded = wavelengths_A[
            (None,) + (None,) * batch_ndim + (slice(None),)
        ]

    elif wavelengths_A.ndim == 2:
        if wavelengths_A.shape[0] != N_lines:
            raise ValueError(
                f"`wavelengths_A` has shape {wavelengths_A.shape}, but "
                f"LoS_velocity implies N_lines={N_lines}. Expected "
                f"`wavelengths_A.shape[0] == N_lines`."
            )

        # Line-dependent wavelength grid.
        wavelengths_expanded = wavelengths_A[
            (slice(None),) + (None,) * batch_ndim + (slice(None),)
        ]

    else:
        raise ValueError(
            f"`wavelengths_A` must have shape (N_v,) or (N_lines, N_v), "
            f"got {wavelengths_A.shape}."
        )

    # ------------------------------------------------------------
    # 2. Build a(x) / a_obs with flexible broadcasting
    # ------------------------------------------------------------
    if scale_factor_obs.ndim == 0:
        # Scalar observer scale factor.
        scale_factor_ratio = scale_factor[None, ..., None] / scale_factor_obs

    elif scale_factor_obs.shape == (N_lines,):
        # Per-line observer scale factor.
        scale_factor_ratio = (
            scale_factor[None, ..., None]
            / scale_factor_obs[(slice(None),) + (None,) * batch_ndim + (None,)]
        )

    elif scale_factor_obs.shape == batch_shape:
        # Per-point observer scale factor, shared by all lines.
        scale_factor_ratio = (
            scale_factor[None, ..., None]
            / scale_factor_obs[None, ..., None]
        )

    elif scale_factor_obs.shape == LoS_velocity.shape:
        # Per-line and per-point observer scale factor.
        scale_factor_ratio = (
            scale_factor[None, ..., None]
            / scale_factor_obs[..., None]
        )

    else:
        raise ValueError(
            "`scale_factor_obs` must be scalar, have shape "
            f"({N_lines},), have shape {batch_shape}, or have shape "
            f"{LoS_velocity.shape}. Got {scale_factor_obs.shape}."
        )

    # ------------------------------------------------------------
    # 3. Doppler inverse factor
    # ------------------------------------------------------------

    beta = LoS_velocity[..., None] / phys.c_kms
    doppler_inverse = jnp.sqrt((1.0 + beta) / (1.0 - beta))

    # ------------------------------------------------------------
    # 4. Local wavelength
    # ------------------------------------------------------------
    lambdas_A_prime = (
        wavelengths_expanded
        * scale_factor_ratio
        * doppler_inverse
    )

    return lambdas_A_prime