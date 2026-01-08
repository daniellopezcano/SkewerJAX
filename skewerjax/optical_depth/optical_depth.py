from __future__ import annotations

from functools import partial
from typing import Optional, Mapping
from jax.typing import ArrayLike

import logging
logger = logging.getLogger(__name__)

import jax
from jax import vmap, pmap
import jax.numpy as jnp
from jax.scipy.integrate import trapezoid

from skewerjax.optical_depth.voigt import log10_voigt
from skewerjax.constants.physical_constants import phys

from skewerjax.optical_depth import line_profile




def compute_comoving_distance(points: jnp.ndarray) -> jnp.ndarray:
    """
    Compute cumulative comoving distance along each skewer.
    
    The distance is the Euclidean norm in the spatial (x,y,z) coordinates,
    accumulated from the first point along each line-of-sight.
    
    Args:
        points: Array of shape (N_skewers, N_ppsk, 4) where last dim is [a, x, y, z]
    
    Returns:
        distances: Array of shape (N_skewers, N_ppsk) with cumulative distance 
                   from the first point (in comoving units, same as x,y,z)
    """
    def _single_skewer_distances(skewer_points):
        # Extract spatial coordinates (x, y, z) -> shape (N_ppsk, 3)
        spatial_coords = skewer_points[:, 1:]
        
        # Compute differences between consecutive points
        diffs = spatial_coords[1:] - spatial_coords[:-1]
        
        # Euclidean distance for each step
        step_distances = jnp.linalg.norm(diffs, axis=-1)
        
        # Cumulative sum with initial 0
        cumulative = jnp.concatenate([
            jnp.zeros(1),
            jnp.cumsum(step_distances)
        ])
        
        return cumulative
    
    # Vectorize over skewers
    return vmap(_single_skewer_distances)(points)




def compute_optical_depth(
    skewer_bundle,
    wavelengths: jnp.ndarray,
    line_bundle,
    fields,
    ii_cosmology: int = 0,
    compute_mode: str = "pmap",
    voigt_implementation_mode: str = "ExoJAX",
    num_gpus: Optional[int] = None,
    mode_integration: str = "trapz",
) -> jnp.ndarray:
    """
    Compute integrand terms from Voigt line profiles for a bundle of lines and a 3D grid of points.

    Parameters
    ----------
    skewer_bundle : SkewerBundle-like
        Object describing the set of skewers.

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
            - tile_field_to_lines(field_name, points_flat, indices_for_species, ...)
              and support "LoS_velocity" with an extra LoS_directions= argument.

        Required per-species fields:
            - "number_density"
            - "temperature"
            - "collisional"
            - "turbulence"
            - "velocity" (used to build LoS_velocity)

    ii_cosmology : int, default 0
        Index of the cosmology to use. This applies only (for now) to skewers.
        Reasson explaining why this is as it is right now:
        When computing the scale factor along the skewer,
        the skewer class allows for multiple cosmologies, hence severa results of a(x) for different cosmologies.
        However, at the moment, the fields objects are assumed to be unique (i.e. only derived from a single cosmology).
        In the future, this should be fixed, in that case, first we should extend the fields object to support multiple cosmologies.

    compute_mode : {"vmap", "pmap"}, default "pmap"
        How to parallelize Voigt evaluations:
            - "vmap": single-device vectorization
            - "pmap": multi-device sharding + per-device vmap

    voigt_implementation_mode : str, default "ExoJAX"
        Passed through to the low-level `log10_voigt(...)` implementation.

    num_gpus : int or None, default None
        Number of GPUs to use in "pmap" mode. If None, uses `jax.local_device_count()`.

    mode_integration : {"trapz", "simps", ...?}, default "trapz"
        Integration mode for the Voigt line profiles.

    Returns
    -------
    optical_depth : jnp.ndarray, shape (N_lines, N_skewers, N_lambda)
        Integrand terms from Voigt line profiles.
    """

    logger.info("[OPTICAL_DEPTH] Starting computation for cosmology index %d, mode=%s, Voigt impl=%s", 
                ii_cosmology, compute_mode, voigt_implementation_mode)
    
    # ------------------------------------------------------------------
    # 0. Extract and process quadri-vectors from skewer_bundle
    # ------------------------------------------------------------------
    logger.info("[OPTICAL_DEPTH] Extracting %d skewers with %d points each", 
                skewer_bundle.N_sk, skewer_bundle.N_ppsk)
    
    quadri_vectors = skewer_bundle.quadri_vectors[ii_cosmology]                             # (N_sk, N_ppsk, 4)
    LoS_directions = skewer_bundle.los_directions                                           # (N_sk, N_ppsk, 3)
    scale_factor_obs_tiled = jnp.tile(skewer_bundle.a0[:, None], (1, skewer_bundle.N_ppsk)) # (N_sk, N_ppsk)

    # ------------------------------------------------------------------
    # 1. Compute line profiles
    # ------------------------------------------------------------------
    logger.info("[OPTICAL_DEPTH] Computing line profiles for %d lines across %d species using %s", 
                line_bundle.n_lines, line_bundle.n_species, compute_mode)

    phi_ns = line_profile.compute_line_profiles(
        quadri_vectors = quadri_vectors,
        LoS_directions = LoS_directions,
        scale_factor_obs = scale_factor_obs_tiled,
        wavelengths = wavelengths,
        line_bundle = line_bundle,
        fields = fields,
        compute_mode = compute_mode,
        voigt_implementation_mode = voigt_implementation_mode,
        num_gpus = num_gpus
    )

    logger.info("[OPTICAL_DEPTH] Line profiles computed with shape %s", phi_ns.shape)

    # ------------------------------------------------------------------
    # 2. Map species → line indices, and tile fields to lines
    # ------------------------------------------------------------------
    logger.info("[OPTICAL_DEPTH] Tiling fields for %d species to line-level", 
                len(line_bundle.species_keys))
    
    # Precompute field indices for each species
    # Calling just fields.field_name(points) returns the fields as (N_species, *batch_shape)
    # but we need to extract the fields tiled to the different lines, not just the species,
    # so the field values have the dimension (N_lines, N_points), N_lines >= N_species

    indices_for_species: Mapping[str, jnp.ndarray] = {
        sp: line_bundle.indices_for_species(sp) for sp in line_bundle.species_keys
    }

    # Each of these returns shape (N_lines, N_points)
    number_density = fields.tile_field_to_lines(
        "number_density", quadri_vectors, indices_for_species=indices_for_species,
    )

    logger.info("[OPTICAL_DEPTH] Tiled number_density with shape %s", number_density.shape)

    # ------------------------------------------------------------------
    # 3. Compute dtau_dx
    # ------------------------------------------------------------------
    logger.info("[OPTICAL_DEPTH] Computing dtau_dx with constant %e nm²/ns", 
                float((phys.qe**2 / (4 * phys.epsilon_0 * phys.me * phys.c))*1e3))
    
    f_osc = jnp.asarray(line_bundle.f_osc)  # (N_lines,)
    cte_nm2_per_ns = (phys.qe**2 / (4 * phys.epsilon_0 * phys.me * phys.c))*1e3                   # UNITS: [nm² / ns]
    sigma_nm2 = cte_nm2_per_ns * f_osc[:, None, None, None] * phi_ns                              # (N_lines, N_skewers, N_ppsk, N_lambda) — UNITS: [nm²]
    dtau_dx = sigma_nm2 * number_density[..., None] * quadri_vectors[..., 0][None, ..., None]**-2 # (N_lines, N_skewers, N_ppsk, N_lambda) — UNITS: [1/nm]  
    
    logger.info("[OPTICAL_DEPTH] dtau_dx computed with shape %s", dtau_dx.shape)

    # ------------------------------------------------------------------
    # 4. Reorder skewer array order to match the integration direction and
    #    compute comoving distances along skewers necessary for integration
    # ------------------------------------------------------------------
    logger.info("[OPTICAL_DEPTH] Reorienting %d skewers based on photon path direction mask", 
                skewer_bundle.N_sk)
    
    # Flip the skewers directions where necessary to facilitate integration along the photon path
    # Apply mask: where True pick reversed, else keep original. Broadcast mask to match array dimensions
    mask = skewer_bundle.array_direction == -1

    _original = quadri_vectors
    _reversed = jnp.flip(_original, axis=1) # Reverse along the N_ppsk dimension
    quadri_vectors_oriented = jnp.where(mask[:, None, None], _reversed, _original)
    comoving_distances_oriented = compute_comoving_distance(quadri_vectors_oriented)    

    # Reorder dtau_dx array order to match integration direction
    _original = dtau_dx
    _reversed = jnp.flip(dtau_dx, axis=2) # Reverse along the N_ppsk dimension
    dtau_dx_oriented = jnp.where(mask[None, :, None, None], _reversed, _original)

    logger.info("[OPTICAL_DEPTH] Arrays reoriented: quadri_vectors %s, dtau_dx %s", 
        quadri_vectors_oriented.shape, dtau_dx_oriented.shape)

    # ------------------------------------------------------------------
    # 5. perform numerical integration to compute optical_depth across the different skewers
    # ------------------------------------------------------------------

    logger.info("[OPTICAL_DEPTH] Performing %s integration along %d spatial points", 
                mode_integration, skewer_bundle.N_ppsk)

    # tile comoving distances to match dtau_dx shape
    comoving_distances_oriented_tiled = comoving_distances_oriented[None, :, :, None]  # (1, N_skewers, N_ppsk, 1)
    if mode_integration == "trapz":
        optical_depth = trapezoid(
            dtau_dx_oriented,
            x=comoving_distances_oriented_tiled,
            axis=2
        )  # (N_lines, N_skewers, N_lambda)
    else:
        raise NotImplementedError(f"Integration mode '{mode_integration}' not implemented.")
    
    logger.info("[OPTICAL_DEPTH] Computation complete. Result shape: %s", optical_depth.shape)

    return optical_depth
