from __future__ import annotations

from functools import partial
from typing import Optional, Mapping
from jax.typing import ArrayLike

import logging
logger = logging.getLogger(__name__)

import jax
from jax import vmap, pmap
import jax.numpy as jnp

from skewerjax.optical_depth.voigt import log10_voigt
from skewerjax.constants.physical_constants import phys


def compute_lambda_prime(
    wavelengths: ArrayLike,           # (N_lambda,)
    scale_factor_obs: ArrayLike,      # (*batch_shape,)
    scale_factor: ArrayLike,          # (*batch_shape,)
    LoS_velocity: ArrayLike,          # (N_species, *batch_shape)
) -> jnp.ndarray:
    """
    Compute observer-frame wavelength shifted by cosmological expansion and peculiar velocities.
    
    λ' = λ_obs × (a / a_obs) × (1 + v_LoS / c)
    
    Args:
        wavelengths: Rest-frame wavelengths [Å], shape (N_lambda,)
        scale_factor_obs: Cosmological scale factor relative to observer position, shape (*batch_shape,)
        scale_factor: Scale factor a(z) for each point, shape (*batch_shape,)
        LoS_velocity: Line-of-sight velocity, shape (N_species, *batch_shape), UNITS [km/s]
    
    Returns:
        lambdas_A_prime: Shifted wavelengths [Å], shape (N_species, *batch_shape, N_lambda)
    """
    # Validate batch shapes match
    if scale_factor.shape != LoS_velocity.shape[1:]:
        raise ValueError(
            f"scale_factor shape {scale_factor.shape} incompatible with "
            f"LoS_velocity shape {LoS_velocity.shape}. "
            f"Expected scale_factor.shape == LoS_velocity.shape[1:]"
        )
        
    # Add leading dimensions to wavelengths (one for species + one per batch dim)
    wavelengths_expanded = wavelengths[(None,) * (LoS_velocity.ndim - 1) + (slice(None),)]
    
    # Add trailing dimension to scale_factor and LoS_velocity
    scale_factor_expanded = scale_factor[None, ..., None] / scale_factor_obs[None, ..., None]
    velocity_expanded = LoS_velocity[..., None] * 1e3 / phys.c # convert km/s to m/s, hence velocity_expanded has UNITS [adim]

    # Compute lambda prime
    lambdas_A_prime = wavelengths_expanded * scale_factor_expanded * (1.0 + velocity_expanded)
    
    return lambdas_A_prime


def compute_line_profiles(
    quadri_vectors: jnp.ndarray,
    LoS_directions: jnp.ndarray,
    scale_factor_obs: jnp.ndarray,
    wavelengths: jnp.ndarray,
    line_bundle,
    fields,
    compute_mode: str = "pmap",
    voigt_implementation_mode: str = "ExoJAX",
    num_gpus: Optional[int] = None,
) -> jnp.ndarray:
    """
    Compute Voigt line profiles for a set of quadri_vectors.

    Parameters
    ----------
    quadri_vectors : jnp.ndarray, shape (..., 4)
        Sampling events in the form [a, x, y, z]. The leading dimensions define the spatial batch shape.
    LoS_directions : jnp.ndarray, shape (..., 3),
        Line-of-sight direction vectors at each spatial point.
    scale_factor_obs : jnp.ndarray with same batch shape as quadri_vectors
        Cosmological scale factors relative to observer position.
    wavelengths : jnp.ndarray, shape (N_lambda,)
        Wavelength grid in Angstrom.
    line_bundle : LineBundle-like
        Object describing the set of lines. Must provide:
            - species_keys
            - n_lines, n_species
            - indices_for_species(species_name)
            - lambda0_A   : (N_lines,)  [Angstrom]
            - A21_GHz     : (N_lines,)  [GHz]
            - mass_amu    : (N_lines,)  [amu]
            - f_osc       : (N_lines,)  [adim]
    fields : Fields-like
        Field container. Must provide:
            - species_keys
            - field_keys  (mapping species -> tuple of field names)
            - tile_field_to_lines(field_name, quadri_vectors_flat, indices_for_species, ...)
              and support "LoS_velocity" with an extra LoS_directions= argument.

        Required per-species fields:
            - "temperature"                             # UNITS: [K]
            - "collisional"                             # UNITS: [GHz]
            - "turbulence"                              # UNITS: [km/s]
            - "velocity" (used to build LoS_velocity)   # UNITS: [km/s]  

    compute_mode : {"vmap", "pmap"}, default "pmap"
        How to parallelize Voigt evaluations:
            - "vmap": single-device vectorization
            - "pmap": multi-device sharding + per-device vmap

    voigt_implementation_mode : str, default "ExoJAX"
        Passed through to the low-level `log10_voigt(...)` implementation.

    num_gpus : int or None, default None
        Number of GPUs to use in "pmap" mode. If None, uses `jax.local_device_count()`.

    Returns
    -------
    phi : jnp.ndarray, shape (N_lines, *batch_shape, N_lambda)
        Line profiles in time units [ns] (same convention as your original script).
    """

    # ------------------------------------------------------------------
    # 0. Basic validation and bookkeeping
    # ------------------------------------------------------------------
    wavelengths = jnp.asarray(wavelengths)
    if wavelengths.ndim != 1:
        raise ValueError(f"`wavelengths` must be 1D, got shape {wavelengths.shape}")
    N_lambda = wavelengths.size

    if quadri_vectors.ndim < 2 or quadri_vectors.shape[-1] != 4:
        raise ValueError(
            f"`quadri_vectors` must have shape (..., 4) with last axis [a,x,y,z], "
            f"got shape {quadri_vectors.shape}"
        )

    batch_shape = quadri_vectors.shape[:-1]
    N_points = int(jnp.prod(jnp.array(batch_shape)))

    # Species consistency
    species_fields = set(fields.species_keys)
    species_lines = set(line_bundle.species_keys)
    if species_fields != species_lines:
        raise ValueError(
            f"Species keys mismatch between fields and line_bundle. "
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
            f"batch_shape={batch_shape} (expected {batch_shape + (3,)})"
        )

    # Required field names per species
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
                f"Missing 'velocity' field for species '{sp}', required to build LoS_velocity."
            )

    # ------------------------------------------------------------------
    # 1. Flatten spatial dimensions
    # ------------------------------------------------------------------
    quadri_vectors_flat = quadri_vectors.reshape(N_points, 4)        # (N_points, 4)
    LoS_flat = LoS_directions.reshape(N_points, 3)                   # (N_points, 3)
    scale_factors_flat = quadri_vectors_flat[:, 0]                   # (N_points,)
    scale_factor_obs_flat = scale_factor_obs.reshape(N_points,)      # (N_points,)

    # ------------------------------------------------------------------
    # 2. Map species → line indices, and tile fields to lines
    # ------------------------------------------------------------------

    # Precompute field indices for each species
    # Calling just fields.field_name(quadri_vectors) returns the fields as (N_species, *batch_shape)
    # but we need to extract the fields tiled to the different lines, not just the species,
    # so the field values have the dimension (N_lines, N_points), N_lines >= N_species

    indices_for_species: Mapping[str, jnp.ndarray] = {
        sp: line_bundle.indices_for_species(sp) for sp in line_bundle.species_keys
    }

    # Each of these returns shape (N_lines, N_points)
    collisional = fields.tile_field_to_lines(                                        # UNITS: [GHz]
        "collisional", quadri_vectors_flat, indices_for_species=indices_for_species,
    )
    temperature = fields.tile_field_to_lines(                                        # UNITS: [K]
        "temperature", quadri_vectors_flat, indices_for_species=indices_for_species,
    )
    turbulence = fields.tile_field_to_lines(                                         # UNITS: [km/s]
        "turbulence", quadri_vectors_flat, indices_for_species=indices_for_species,
    )
    LoS_velocity = fields.tile_field_to_lines(                                       # UNITS: [km/s]
        "LoS_velocity", quadri_vectors_flat, indices_for_species=indices_for_species, LoS_directions=LoS_flat,
    )

    # ------------------------------------------------------------------
    # 3. Extract line parameters
    # ------------------------------------------------------------------
    lambda0_A = jnp.asarray(line_bundle.lambda0_A)  # (N_lines,)
    A21_GHz   = jnp.asarray(line_bundle.A21_GHz)    # (N_lines,)
    mass_amu  = jnp.asarray(line_bundle.mass_amu)   # (N_lines,)

    # ------------------------------------------------------------------
    # 4. Compute line profile parameters (gamma, sigma, etc.)
    # ------------------------------------------------------------------
    # gamma [GHz]
    gamma_GHz = (A21_GHz[:, None] + collisional) / (4.0 * jnp.pi)  # (N_lines, N_points)

    # Thermal velocity [m/s]
    # phys.kB : Boltzmann constant in (amu km^2 s^-2 K^-1)
    v_th_m_per_s = jnp.sqrt(2.0 * phys.kB * temperature / mass_amu[:, None])

    # Effective Doppler velocity [Å/ns]
    v_eff_A_GHz = (v_th_m_per_s + turbulence) * 10.0

    # sigma [GHz]
    sigma_GHz = v_eff_A_GHz / (2.0 * lambda0_A[:, None])           # (N_lines, N_points)

    # ------------------------------------------------------------------
    # 5. Frequency offsets for each line / point / wavelength
    # ------------------------------------------------------------------

    # lambdas_A_prime: (N_lines, N_points, N_lambda)
    lambdas_A_prime = compute_lambda_prime(
        wavelengths, scale_factor_obs_flat, scale_factors_flat, LoS_velocity
    )

    # nu [GHz]: (N_lines, N_points, N_lambda)
    nu_GHz = ((1.0 / lambdas_A_prime) - (1.0 / lambda0_A[:, None, None])) * phys.c * 10.0

    # ------------------------------------------------------------------
    # 6. Flatten for Voigt evaluation: each (line,point) has its own 1D spectrum
    # ------------------------------------------------------------------
    N_total = N_lines * N_points

    nu_flat    = nu_GHz.reshape(N_total, N_lambda)   # (N_total, N_lambda)
    sigma_flat = sigma_GHz.reshape(N_total)          # (N_total,)
    gamma_flat = gamma_GHz.reshape(N_total)          # (N_total,)

    # ------------------------------------------------------------------
    # 7. Compute Voigt profiles (log10 space for numerical stability)
    # ------------------------------------------------------------------
    compute_mode = compute_mode.lower()
    if compute_mode not in ("vmap", "pmap"):
        raise ValueError(
            f"`compute_mode` must be 'vmap' or 'pmap', got '{compute_mode}'"
        )

    def _eval_voigt_batch(nu_batch, sigma_batch, gamma_batch):
        """
        nu_batch    : (B, N_lambda)
        sigma_batch : (B,)
        gamma_batch : (B,)
        returns log10(phi) with shape (B, N_lambda)
        """
        return vmap(
            lambda x, s, g: log10_voigt(x, s, g, mode=voigt_implementation_mode),
            in_axes=(0, 0, 0),
        )(nu_batch, sigma_batch, gamma_batch)

    if compute_mode == "vmap":
        # Single-device batched evaluation
        log10_phi_flat = _eval_voigt_batch(nu_flat, sigma_flat, gamma_flat)
        # (N_total, N_lambda)

    else:  # "pmap"
        # 1) Detect devices
        if num_gpus is None:
            num_gpus = jax.local_device_count()
            logger.info("Detected %d device(s) for pmap.", num_gpus)

        if num_gpus <= 0:
            raise RuntimeError("No devices available for pmap.")

        # 2) Handle non-divisible batch size via padding
        if N_total % num_gpus != 0:
            pad_size = num_gpus - (N_total % num_gpus)
            logger.info(
                "Padding %d elements for pmap (N_total=%d, num_gpus=%d)",
                pad_size, N_total, num_gpus,
            )
            nu_flat = jnp.pad(nu_flat, ((0, pad_size), (0, 0)), mode="constant")
            sigma_flat = jnp.pad(sigma_flat, (0, pad_size), mode="constant")
            gamma_flat = jnp.pad(gamma_flat, (0, pad_size), mode="constant")
            N_total_padded = N_total + pad_size
        else:
            N_total_padded = N_total

        N_per_gpu = N_total_padded // num_gpus

        # 3) Shard across devices: (num_gpus, N_per_gpu, ...)
        nu_sharded    = nu_flat.reshape(num_gpus, N_per_gpu, N_lambda)
        sigma_sharded = sigma_flat.reshape(num_gpus, N_per_gpu)
        gamma_sharded = gamma_flat.reshape(num_gpus, N_per_gpu)

        # 4) Define per-device computation
        @partial(pmap, in_axes=(0, 0, 0))
        def compute_on_device(nu_shard, sigma_shard, gamma_shard):
            return _eval_voigt_batch(nu_shard, sigma_shard, gamma_shard)
            # (N_per_gpu, N_lambda) per device

        # 5) Execute across devices
        log10_phi_padded = compute_on_device(
            nu_sharded, sigma_sharded, gamma_sharded
        )
        # (num_gpus, N_per_gpu, N_lambda)

        # 6) Flatten back and trim padding
        log10_phi_flat = log10_phi_padded.reshape(N_total_padded, N_lambda)[:N_total, :]

    # ------------------------------------------------------------------
    # 8. Convert back from log10 to phi, reshape to (N_lines, *batch_shape, N_lambda)
    # ------------------------------------------------------------------
    phi_flat = 10.0 ** log10_phi_flat # (N_total, N_lambda)
    phi = phi_flat.reshape(N_lines, *batch_shape, N_lambda)

    return phi
