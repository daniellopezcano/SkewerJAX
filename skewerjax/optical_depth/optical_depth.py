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
            - "temperature"                             # UNITS: [K]
            - "collisional"                             # UNITS: [GHz]
            - "turbulence"                              # UNITS: [km/s]
            - "velocity" (used to build LoS_velocity)   # UNITS: [km/s]  
            - "number_density"                          # UNITS: [cm^-3]

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
    number_density_cm3 = fields.tile_field_to_lines(
        "number_density", quadri_vectors, indices_for_species=indices_for_species,
    )

    logger.info("[OPTICAL_DEPTH] Tiled number_density with shape %s", number_density_cm3.shape)

    # ------------------------------------------------------------------
    # 3. Compute dtau_dx
    # ------------------------------------------------------------------
    logger.info("[OPTICAL_DEPTH] Computing dtau_dx with constant %e nm²/ns", 
                float((phys.qe**2 / (4 * phys.epsilon_0 * phys.me * phys.c))*1e3))
    
    sigma_cm2 = phys.cross_section_prefactor * line_bundle.f_osc[:, None, None, None] * phi_ns                # (N_lines, N_skewers, N_ppsk, N_lambda) — UNITS: [cm²]
    dtau_dx_m = sigma_cm2 * number_density_cm3[..., None] * quadri_vectors[..., 0][None, ..., None]**-2 * 1e2 # (N_lines, N_skewers, N_ppsk, N_lambda) — UNITS: [1/m]  
    
    logger.info("[OPTICAL_DEPTH] dtau_dx computed with shape %s", dtau_dx_m.shape)

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
    quadri_vectors = jnp.where(mask[:, None, None], _reversed, _original)
    comoving_distances_m = compute_comoving_distance(quadri_vectors) * phys.Mpc_to_m # (N_sk, N_ppsk), UNITS: [m]

    # Reorder dtau_dx array order to match integration direction
    _original = dtau_dx_m
    _reversed = jnp.flip(dtau_dx_m, axis=2) # Reverse along the N_ppsk dimension
    dtau_dx_m = jnp.where(mask[None, :, None, None], _reversed, _original)

    logger.info("[OPTICAL_DEPTH] Arrays reoriented: quadri_vectors %s, dtau_dx_m %s", 
        quadri_vectors.shape, dtau_dx_m.shape)

    # ------------------------------------------------------------------
    # 5. perform numerical integration to compute optical_depth across the different skewers
    # ------------------------------------------------------------------

    logger.info("[OPTICAL_DEPTH] Performing %s integration along %d spatial points", 
                mode_integration, skewer_bundle.N_ppsk)

    # tile comoving distances to match dtau_dx shape
    comoving_distances_m = comoving_distances_m[None, :, :, None]  # (1, N_skewers, N_ppsk, 1)
    if mode_integration == "trapz":
        optical_depth = trapezoid(
            dtau_dx_m,
            x=comoving_distances_m,
            axis=2
        )  # (N_lines, N_skewers, N_lambda)
    else:
        raise NotImplementedError(f"Integration mode '{mode_integration}' not implemented.")
    
    logger.info("[OPTICAL_DEPTH] Computation complete. Result shape: %s", optical_depth.shape)

    return optical_depth


def compute_optical_depth_analytical_radiation_domination_Natural(
    lambda_vals: jnp.ndarray,   # (N_lambda,) [Å]
    lambda_rest: jnp.ndarray,   # (N_lines,)  [Å]
    A21_GHz: jnp.ndarray,       # (N_lines,)  [GHz]
    f_osc: jnp.ndarray,         # (N_lines,)  dimensionless
    a0: jnp.ndarray,            # (N_sk,)     scale factor at observer
    x0_Mpc: jnp.ndarray,        # (N_sk,)     comoving distance to observer [Mpc]
    xem_Mpc: jnp.ndarray,       # (N_sk,)     comoving distance to emission [Mpc]
    xf_Mpc: jnp.ndarray,        # (N_sk,)     comoving distance to front [Mpc]
    n0_cm3: jnp.ndarray,        # (N_sk,)     number density at observer [cm^-3]
) -> jnp.ndarray:
    """
    Compute optical depth for natural (Lorentzian) line broadening in radiation-dominated era.
    
    Returns:
        Optical depth array of shape (N_lines, N_sk, N_lambda)
    """
    # --- Broadcast to common shape: (N_lines, N_sk, N_lambda) ---
    lambda_vals = lambda_vals[None, None, :]  # (1, 1, N_lambda)
    lambda_rest = lambda_rest[:, None, None]  # (N_lines, 1, 1)
    A21_GHz = A21_GHz[:, None, None]          # (N_lines, 1, 1)
    f_osc = f_osc[:, None, None]              # (N_lines, 1, 1)
    a0 = a0[None, :, None]                    # (1, N_sk, 1)
    x0_Mpc = x0_Mpc[None, :, None]            # (1, N_sk, 1)
    xem_Mpc = xem_Mpc[None, :, None]          # (1, N_sk, 1)
    xf_Mpc = xf_Mpc[None, :, None]            # (1, N_sk, 1)
    n0_cm3 = n0_cm3[None, :, None]            # (1, N_sk, 1)

    # --- Compute prefactor ----------------------------------------------------
    # cross_section_prefactor * Gyr_to_s * 1e9 = σ₀ in cm²·GHz
    # Divide by π for Lorentzian normalization
    constants_prefactor = (phys.cross_section_prefactor * phys.Gyr_to_s * 1e9) / (jnp.pi * phys.H0)  # [cm²]
    
    # Final prefactor: converts to dimensionless optical depth
    prefactor = constants_prefactor * 1e-8 * f_osc * n0_cm3 * lambda_vals / a0

    # --- Compute line profile terms -------------------------------------------
    gamma_GHz = A21_GHz / (4 * jnp.pi)  # Lorentzian HWHM [GHz]
    
    # Helper to compute λ' at given comoving distance
    def compute_shifted_lambda(x_Mpc):
        return lambda_vals * ((phys.H0_over_c / a0) * (x_Mpc - x0_Mpc) + 1.0)  # [Å]
    
    # Helper to compute arctan term at given comoving distance
    def compute_arctan_term(x_Mpc):
        lambda_prime = compute_shifted_lambda(x_Mpc)  # [Å]
        # Frequency offset from line center: ν - ν₀ [GHz]
        nu_offset_GHz = (1 / lambda_rest - 1 / lambda_prime) * phys.c * 10
        return jnp.arctan(nu_offset_GHz / gamma_GHz)
    
    # Evaluate at integration limits
    term_front = compute_arctan_term(xf_Mpc)
    term_emission = compute_arctan_term(xem_Mpc)
    
    return prefactor * (term_front - term_emission)


def compute_optical_depth_analytical_radiation_domination_Thermal(
    lambda_vals: jnp.ndarray,   # (N_lambda,) [Å]
    lambda_rest: jnp.ndarray,   # (N_lines,)  [Å]
    f_osc: jnp.ndarray,         # (N_lines,)  dimensionless
    mass_amu: jnp.ndarray,      # (N_lines,)  mass [amu]
    T0: jnp.ndarray,            # (N_sk,)     gas temperature [K]
    a0: jnp.ndarray,            # (N_sk,)     scale factor at observer
    x0_Mpc: jnp.ndarray,        # (N_sk,)     comoving distance to observer [Mpc]
    xem_Mpc: jnp.ndarray,       # (N_sk,)     comoving distance to emission [Mpc]
    xf_Mpc: jnp.ndarray,        # (N_sk,)     comoving distance to front [Mpc]
    n0_cm3: jnp.ndarray,        # (N_sk,)     number density at observer [cm^-3]
) -> jnp.ndarray:
    """
    Compute optical depth for thermal (Gaussian) line broadening in radiation-dominated era.
    
    Returns:
        Optical depth array of shape (N_lines, N_sk, N_lambda)
    """
    # --- Broadcast to common shape: (N_lines, N_sk, N_lambda) ---
    lambda_vals = lambda_vals[None, None, :]  # (1, 1, N_lambda)
    lambda_rest = lambda_rest[:, None, None]  # (N_lines, 1, 1)
    f_osc = f_osc[:, None, None]              # (N_lines, 1, 1)
    mass_amu = mass_amu[:, None, None]        # (N_lines, 1, 1)
    T0 = T0[None, :, None]                    # (1, N_sk, 1)
    a0 = a0[None, :, None]                    # (1, N_sk, 1)
    x0_Mpc = x0_Mpc[None, :, None]            # (1, N_sk, 1)
    xem_Mpc = xem_Mpc[None, :, None]          # (1, N_sk, 1)
    xf_Mpc = xf_Mpc[None, :, None]            # (1, N_sk, 1)
    n0_cm3 = n0_cm3[None, :, None]            # (1, N_sk, 1)

    # --- Compute prefactor ----------------------------------------------------
    # Same as natural but with factor 2 instead of π for Gaussian normalization
    constants_prefactor = (phys.cross_section_prefactor * phys.Gyr_to_s * 1e9) / (2.0 * phys.H0)  # [cm²]
    
    prefactor = constants_prefactor * 1e-8 * f_osc * n0_cm3 * lambda_vals / a0

    # --- Compute thermal broadening width -------------------------------------
    # Convert mass from amu to kg for proper thermal velocity calculation
    
    
    # Thermal velocity: v_th = √(2k_B T / m); σ_ν = (v_th / c) * ν₀
    v_eff_A_GHz = jnp.sqrt(2.0 * phys.kB * T0 / mass_amu) * 10.0
    sigma_GHz = v_eff_A_GHz / (2.0 * lambda_rest)

    # --- Compute error function terms -----------------------------------------
    # Helper to compute λ' at given comoving distance
    def compute_shifted_lambda(x_Mpc):
        return lambda_vals * ((phys.H0_over_c / a0) * (x_Mpc - x0_Mpc) + 1.0)  # [Å]
    
    # Helper to compute erf term at given comoving distance
    def compute_erf_term(x_Mpc):
        lambda_prime = compute_shifted_lambda(x_Mpc)  # [Å]
        # Frequency offset from line center: ν - ν₀ [GHz]
        nu_offset_GHz = (1 / lambda_rest - 1 / lambda_prime) * phys.c * 10
        # Argument of error function: (ν - ν₀) / (√2 σ_ν)
        arg = nu_offset_GHz / (jnp.sqrt(2.0) * sigma_GHz)
        return jax.scipy.special.erf(arg)
    
    # Evaluate at integration limits
    term_front = compute_erf_term(xf_Mpc)
    term_emission = compute_erf_term(xem_Mpc)
    
    return prefactor * (term_front - term_emission)


# def compute_optical_depth_analytical_radiation_domination_Thermal(
#     lambda_vals,                      # (N_lambda,) [Å]
#     lambda_rest,                      # (N_lines,)  [Å]
#     f_osc,                            # (N_lines,)  dimensionless oscillator strength
#     mass_amu,                         # (N_lines,)  mass [amu]
#     T0,                               # (N_sk,)     gas temperature [K]
#     a0,                               # (N_sk,)     scale factor at observer
#     x0_Mpc,                           # (N_sk,)     comoving dist to observer [Mpc]
#     xem_Mpc,                          # (N_sk,)     comoving dist to emission [Mpc]
#     xf_Mpc,                           # (N_sk,)     comoving dist to observer [Mpc]
#     n0_cm3,                           # (N_sk,)     number density at observer [cm^-3]
# ) -> jnp.ndarray:
    
#     # --- broadcast shapes (same as before) ---
#     lambda_vals = lambda_vals[None, None, :]
#     lambda_rest = lambda_rest[:, None, None]
#     f_osc = f_osc[:, None, None]
#     mass_amu = mass_amu[:, None, None]
#     T0 = T0[None, :, None]
#     a0 = a0[None, :, None]
#     x0_Mpc = x0_Mpc[None, :, None]
#     xem_Mpc = xem_Mpc[None, :, None]
#     xf_Mpc = xf_Mpc[None, :, None]
#     n0_cm3 = n0_cm3[None, :, None]

#     # --- prefactor (same structure, different constant) ---
#     _constants_prefactor = (phys.cross_section_prefactor * phys.Gyr_to_s * 1e9) / (2.0 * phys.H0)
#     _prefactor = _constants_prefactor * 1e-8 * f_osc * n0_cm3 * lambda_vals / a0

#     # --- thermal width in consistent units ---
#     v_eff_A_GHz = jnp.sqrt(2.0 * phys.kB * T0 / mass_amu) * 10.0
#     _sigma_GHz = v_eff_A_GHz / (2.0 * lambda_rest)

#     _lambda = lambda_vals * ((phys.H0_over_c / a0) * (xf_Mpc - x0_Mpc) + 1.0)   # UNITS: [Å]
#     _nu_GHz = (1 / lambda_rest - 1 / _lambda) * phys.c * 10                     # UNITS: [GHz]
#     termf = jax.scipy.special.erf(_nu_GHz / (jnp.sqrt(2.0) * _sigma_GHz))       # UNITS: [adim]                      

#     _lambda = lambda_vals * ((phys.H0_over_c / a0) * (xem_Mpc - x0_Mpc) + 1.0)  # UNITS: [Å]
#     _nu_GHz = (1 / lambda_rest - 1 / _lambda) * phys.c * 10                     # UNITS: [GHz]
#     termi = jax.scipy.special.erf(_nu_GHz / (jnp.sqrt(2.0) * _sigma_GHz))       # UNITS: [adim]

    
#     return _prefactor * (termf - termi)