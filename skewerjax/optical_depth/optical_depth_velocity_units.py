from __future__ import annotations

from typing import Optional, Mapping
from jax.typing import ArrayLike

import logging
logger = logging.getLogger(__name__)

from jax import vmap
import jax.numpy as jnp
from jax.scipy.integrate import trapezoid

from skewerjax.constants.physical_constants import phys
from skewerjax.optical_depth import line_profile_velocity_units


def compute_comoving_distance(points: jnp.ndarray) -> jnp.ndarray:
    """
    Compute cumulative comoving distance along each skewer.

    The distance is the Euclidean norm in the spatial (x,y,z) coordinates,
    accumulated from the first point along each line-of-sight.

    Args:
        points: Array of shape (N_skewers, N_ppsk, 4)
                where last dim is [a, x, y, z]

    Returns:
        distances: Array of shape (N_skewers, N_ppsk)
                   with cumulative distance from the first point.
    """

    def _single_skewer_distances(skewer_points):
        spatial_coords = skewer_points[:, 1:]

        diffs = spatial_coords[1:] - spatial_coords[:-1]
        step_distances = jnp.linalg.norm(diffs, axis=-1)

        cumulative = jnp.concatenate([
            jnp.zeros(1),
            jnp.cumsum(step_distances),
        ])

        return cumulative

    return vmap(_single_skewer_distances)(points)


def compute_optical_depth_velocity_units(
    skewer_bundle,
    velocities_kms: jnp.ndarray,
    line_bundle,
    fields,
    a_dagger: Optional[ArrayLike] = None,
    ii_cosmology: int = 0,
    voigt_implementation_mode: str = "ExoJAX",
    mode_integration: str = "trapz",
) -> jnp.ndarray:
    """
    Compute optical depth using line-centered velocity units.

    This is the velocity-units counterpart of the old wavelength-units
    `compute_optical_depth`.

    The line profile is evaluated using

        lambda'_j(x, v)
        =
        lambda_j * exp(v / c) * a(x) / a_dagger * D^{-1}(x)

    instead of

        lambda'
        =
        lambda_obs * a(x) / a_obs * D^{-1}(x).

    Parameters
    ----------
    skewer_bundle : SkewerBundle-like
        Object describing the set of skewers.

    velocities_kms : jnp.ndarray, shape (N_v,)
        Common line-centered velocity grid in km/s.

    line_bundle : LineBundle-like
        Object describing the set of lines. Must provide:
            - species_keys
            - n_lines, n_species
            - indices_for_species(species_name)
            - lambda0_A   : (N_lines,) [Angstrom]
            - A21_GHz     : (N_lines,) [GHz]
            - mass_amu    : (N_lines,) [amu]
            - f_osc       : (N_lines,) [adim]

    fields : Fields-like
        Field container. Must provide the fields required by
        `line_profile_velocity_units.compute_line_profiles_velocity_units`,
        plus:
            - "number_density" [cm^-3]

    a_dagger : scalar, shape (N_sk,), shape (N_sk, N_ppsk), or None
        Reference scale factor of the line-centered velocity coordinate.

        If None, use `skewer_bundle.a0`.

        If shape is (N_sk,), it is tiled along the skewer point axis,
        analogously to the previous `scale_factors_obs` logic.

    ii_cosmology : int, default 0
        Index of the cosmology to use.

    voigt_implementation_mode : str, default "ExoJAX"
        Passed through to the low-level `log10_voigt(...)` implementation.

    mode_integration : {"trapz"}, default "trapz"
        Integration mode.

    Returns
    -------
    optical_depth : jnp.ndarray, shape (N_lines, N_skewers, N_v)
        Optical depth for each line, skewer, and velocity pixel.
    """

    logger.info(
        "[OPTICAL_DEPTH_VELOCITY] Starting computation for cosmology index %d, Voigt impl=%s",
        ii_cosmology,
        voigt_implementation_mode,
    )

    # ------------------------------------------------------------------
    # 0. Extract and process quadri-vectors from skewer_bundle
    # ------------------------------------------------------------------
    logger.info(
        "[OPTICAL_DEPTH_VELOCITY] Extracting %d skewers with %d points each",
        skewer_bundle.N_sk,
        skewer_bundle.N_ppsk,
    )

    quadri_vectors = skewer_bundle.quadri_vectors[ii_cosmology]  # (N_sk, N_ppsk, 4)
    LoS_directions = skewer_bundle.los_directions                # (N_sk, N_ppsk, 3)

    N_sk = int(skewer_bundle.N_sk)
    N_ppsk = int(skewer_bundle.N_ppsk)
    N_points = N_sk * N_ppsk

    velocities_kms = jnp.asarray(velocities_kms)
    N_v = int(velocities_kms.size)

    # ------------------------------------------------------------------
    # 0.1 Process a_dagger similarly to old scale_factors_obs
    # ------------------------------------------------------------------
    if a_dagger is None:
        a_dagger = skewer_bundle.a0

    a_dagger = jnp.asarray(a_dagger)

    if a_dagger.ndim == 0:
        # Scalar reference scale factor, shared by all skewers and points.
        a_dagger_for_profiles = a_dagger

    elif a_dagger.shape == (N_sk,):
        # Per-skewer reference scale factor.
        # Tile along the N_ppsk axis and flatten because the line-profile
        # function internally flattens quadri_vectors to (N_sk * N_ppsk, 4).
        a_dagger_tiled = jnp.repeat(a_dagger[:, None], N_ppsk, axis=1)
        a_dagger_for_profiles = a_dagger_tiled.reshape(N_points)

    elif a_dagger.shape == (N_sk, N_ppsk):
        # Already provided per skewer and per point.
        a_dagger_for_profiles = a_dagger.reshape(N_points)

    elif a_dagger.shape == (N_points,):
        # Already flattened.
        a_dagger_for_profiles = a_dagger

    else:
        raise ValueError(
            "`a_dagger` must be None, scalar, have shape "
            f"({N_sk},), shape ({N_sk}, {N_ppsk}), or shape ({N_points},). "
            f"Got shape {a_dagger.shape}."
        )

    # ------------------------------------------------------------------
    # 1. Compute line profiles
    # ------------------------------------------------------------------
    logger.info(
        "[OPTICAL_DEPTH_VELOCITY] Computing line profiles for %d lines across %d species",
        line_bundle.n_lines,
        line_bundle.n_species,
    )

    phi_ns = line_profile_velocity_units.compute_line_profiles_velocity_units(
        quadri_vectors=quadri_vectors,
        LoS_directions=LoS_directions,
        velocities_kms=velocities_kms,
        a_dagger=a_dagger_for_profiles,
        line_bundle=line_bundle,
        fields=fields,
        voigt_implementation_mode=voigt_implementation_mode,
    )

    logger.info(
        "[OPTICAL_DEPTH_VELOCITY] Line profiles computed with raw shape %s",
        phi_ns.shape,
    )

    # Your current velocity-units line-profile function returns:
    #
    #     (N_lines, N_sk, N_ppsk * N_v)
    #
    # for quadri_vectors with shape:
    #
    #     (N_sk, N_ppsk, 4)
    #
    # For optical-depth integration we need:
    #
    #     (N_lines, N_sk, N_ppsk, N_v)
    #
    expected_flat_shape = (
        int(line_bundle.n_lines),
        N_sk,
        N_ppsk * N_v,
    )

    expected_full_shape = (
        int(line_bundle.n_lines),
        N_sk,
        N_ppsk,
        N_v,
    )

    if phi_ns.shape == expected_flat_shape:
        phi_ns = phi_ns.reshape(expected_full_shape)

    elif phi_ns.shape == expected_full_shape:
        pass

    else:
        raise ValueError(
            f"Unexpected phi_ns shape {phi_ns.shape}. Expected either "
            f"{expected_flat_shape} or {expected_full_shape}."
        )

    logger.info(
        "[OPTICAL_DEPTH_VELOCITY] Line profiles reshaped for integration with shape %s",
        phi_ns.shape,
    )

    # ------------------------------------------------------------------
    # 2. Map species → line indices, and tile fields to lines
    # ------------------------------------------------------------------
    logger.info(
        "[OPTICAL_DEPTH_VELOCITY] Tiling fields for %d species to line-level",
        len(line_bundle.species_keys),
    )

    indices_for_species: Mapping[str, jnp.ndarray] = {
        sp: line_bundle.indices_for_species(sp)
        for sp in line_bundle.species_keys
    }

    quadri_vectors_flat = quadri_vectors.reshape(N_points, 4)

    number_density_cm3 = fields.tile_field_to_lines(
        "number_density",
        quadri_vectors_flat,
        indices_for_species=indices_for_species,
    )  # expected: (N_lines, N_points)

    number_density_cm3 = number_density_cm3.reshape(
        int(line_bundle.n_lines),
        N_sk,
        N_ppsk,
    )

    logger.info(
        "[OPTICAL_DEPTH_VELOCITY] Tiled number_density with shape %s",
        number_density_cm3.shape,
    )

    # ------------------------------------------------------------------
    # 3. Compute dtau_dx
    # ------------------------------------------------------------------
    logger.info(
        "[OPTICAL_DEPTH_VELOCITY] Computing dtau_dx with cross-section prefactor"
    )

    sigma_cm2 = (
        phys.cross_section_prefactor
        * line_bundle.f_osc[:, None, None, None]
        * phi_ns
    )  # (N_lines, N_sk, N_ppsk, N_v)

    scale_factors = quadri_vectors[..., 0][None, ..., None]
    # shape: (1, N_sk, N_ppsk, 1)

    dtau_dx_m = (
        sigma_cm2
        * number_density_cm3[..., None]
        * scale_factors**-2
        * 1e2
    )  # (N_lines, N_sk, N_ppsk, N_v)

    logger.info(
        "[OPTICAL_DEPTH_VELOCITY] dtau_dx computed with shape %s",
        dtau_dx_m.shape,
    )

    # ------------------------------------------------------------------
    # 4. Reorder skewer array order to match the integration direction
    # ------------------------------------------------------------------
    logger.info(
        "[OPTICAL_DEPTH_VELOCITY] Reorienting %d skewers based on photon path direction mask",
        skewer_bundle.N_sk,
    )

    mask = skewer_bundle.array_direction == -1

    _original = quadri_vectors
    _reversed = jnp.flip(_original, axis=1)

    quadri_vectors_oriented = jnp.where(
        mask[:, None, None],
        _reversed,
        _original,
    )

    comoving_distances_m = (
        compute_comoving_distance(quadri_vectors_oriented)
        * phys.Mpc_to_m
    )  # (N_sk, N_ppsk)

    _original = dtau_dx_m
    _reversed = jnp.flip(dtau_dx_m, axis=2)

    dtau_dx_m = jnp.where(
        mask[None, :, None, None],
        _reversed,
        _original,
    )

    logger.info(
        "[OPTICAL_DEPTH_VELOCITY] Arrays reoriented: quadri_vectors %s, dtau_dx_m %s",
        quadri_vectors_oriented.shape,
        dtau_dx_m.shape,
    )

    # ------------------------------------------------------------------
    # 5. Perform numerical integration to compute optical depth
    # ------------------------------------------------------------------
    logger.info(
        "[OPTICAL_DEPTH_VELOCITY] Performing %s integration along %d spatial points",
        mode_integration,
        skewer_bundle.N_ppsk,
    )

    comoving_distances_m = comoving_distances_m[None, :, :, None]
    # shape: (1, N_sk, N_ppsk, 1)

    if mode_integration == "trapz":
        optical_depth = trapezoid(
            dtau_dx_m,
            x=comoving_distances_m,
            axis=2,
        )  # (N_lines, N_sk, N_v)

    else:
        raise NotImplementedError(
            f"Integration mode '{mode_integration}' not implemented."
        )

    logger.info(
        "[OPTICAL_DEPTH_VELOCITY] Computation complete. Result shape: %s",
        optical_depth.shape,
    )

    return optical_depth