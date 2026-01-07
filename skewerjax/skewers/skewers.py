from __future__ import annotations

from typing import Optional, Sequence, List, Union
import jax.numpy as jnp
from jax import vmap
from jax.typing import ArrayLike
import logging

from skewerjax.cosmology.cosmology import Cosmology

# Set up module-level logger
logger = logging.getLogger(__name__)

class SkewerBundle:
    """
    Collection of line-of-sight skewers through a cosmological volume.
    
    This class encapsulates the geometry of multiple
    line-of-sight (LoS) 'skewer' observations through a cosmology volume. Each skewer
    represents a sightline (e.g., to a quasar) sampling the intergalactic medium
    at discrete comoving positions.
    
    Key Concepts:
    - Initial conditions (ICs) can be defined either at the beginning or end of the skewer.
    - All skewers must have the same number of sampling points (N_ppsk) for batched operations
    
    Geometry:
    - spatial_points: comoving spatial coordinates in [Gpc]
    - x0: comoving distance of the IC point in [Gpc]
    - a0: scale factor at the IC point
    - The direction vector points from the IC point toward the observer
    
    Units:
    - Spatial coordinates: comoving 
    - Distances: comoving Gpc
    - Scale factors: dimensionless (a = 1/(1+z))
    """
    
    def __init__(
        self,
        spatial_points: ArrayLike,
        a0: ArrayLike = 1.0,
        x0: ArrayLike = 0.0,
        ICs_defined_at_index: ArrayLike = -1,
        ICs_at_emission_or_absorption: Union[str, Sequence[str]] = "absorption",
        los_directions: Optional[ArrayLike] = None,
        manual_scale_factors_in_the_skewer_direction: Optional[jnp.ndarray] = None,
        cosmologies: Optional[Cosmology] = None,
    ):
        """
        Initialize a SkewerBundle.
        
        Parameters
        ----------
        spatial_points : ArrayLike, shape (N_sk, N_ppsk, 3)
            Comoving spatial coordinates along each skewer in [Gpc].
            N_sk = number of skewers, N_ppsk = points per skewer.
            Last dimension contains [x, y, z] comoving positions.
            
        a0 : ArrayLike, broadcastable to (N_sk,), default 1.0
            Scale factor a = 1/(1+z) at the location where initial conditions
            are defined. Determines the reference redshift for each skewer.
            
        x0 : ArrayLike, broadcastable to (N_sk,), default 0.0
            Comoving distance [Gpc] from observer to the IC point along the
            LoS. Used as offset for distance calculations.
            
        ICs_defined_at_index : ArrayLike, broadcastable to (N_sk,), default -1
            Index along N_ppsk where ICs are defined. 
            -1 means ICs are at the far end (last point).
            0 means ICs are at the near end (first point).
            
        ICs_at_emission_or_absorption : str or Sequence[str], default "absorption"
            Whether each skewer has ICs defined at emission or absorption.
            It affects the direction of the skewer wrt the spatial points provided,
            hence the sign of distances and direction vectors.
            Can be a single string for all skewers or a sequence of length N_sk.
            
        los_directions : Optional[ArrayLike], shape (N_sk, N_ppsk, 3)
            Pre-computed line-of-sight unit vectors. If None, will be computed
            automatically from spatial coordinate differences.
            Must be normalized to unit length (‖d‖ = 1) within tolerance.
            
        manual_scale_factors_in_the_skewer_direction : Optional[jnp.ndarray], shape (N_sk, N_ppsk)
            Pre-computed scale factors a(x) at each point along the skewer.
            If None, will be computed from cosmology using the comoving distance grid.
            
        cosmologies : Optional[Cosmology]
            Cosmology object required for computing scale factors when
            manual_scale_factors_in_the_skewer_direction is not provided.
            Must implement solve_a_of_x(a0, distances) method.
        """
        logger.info(f"Initializing SkewerBundle with {spatial_points.shape[0]} skewers, "
                    f"{spatial_points.shape[1]} points per skewer")
        
        # Convert and validate spatial_points
        spatial_points = jnp.asarray(spatial_points)
        if spatial_points.ndim != 3 or spatial_points.shape[-1] != 3:
            raise ValueError(f"spatial_points must be (N_sk, N_ppsk, 3), got {spatial_points.shape}")
        N_sk, N_ppsk = spatial_points.shape[:2]
        
        # Enforce uniform N_ppsk across all skewers
        self.N_sk = N_sk
        self.N_ppsk = N_ppsk
        
        # Convert a0, x0 to arrays and broadcast to (N_sk,)
        a0 = jnp.broadcast_to(jnp.asarray(a0), (N_sk,))
        x0 = jnp.broadcast_to(jnp.asarray(x0), (N_sk,))
        
        # assert that ICs_defined_at_index is valid
        ICs_defined_at_index = jnp.asarray(ICs_defined_at_index)
        if jnp.any((ICs_defined_at_index != -1) & (ICs_defined_at_index != 0)):
            raise ValueError(f"ICs_defined_at_index must be -1 or 0, got {ICs_defined_at_index}")
        # Convert ICs_defined_at_index to array
        ICs_defined_at_index = jnp.asarray(ICs_defined_at_index)
        if ICs_defined_at_index.ndim == 0:
            logger.info(f"Broadcasting scalar ICs_defined_at_index={int(ICs_defined_at_index)} to all {N_sk} skewers")
            ICs_defined_at_index = jnp.full((N_sk,), ICs_defined_at_index, dtype=jnp.int32)
        else:
            logger.info(f"Using per-skewer ICs_defined_at_index: {ICs_defined_at_index}")
        
        # Convert ICs_at_emission_or_absorption to boolean array
        if isinstance(ICs_at_emission_or_absorption, str):
            logger.info(f"All skewers set to {ICs_at_emission_or_absorption} mode")
            is_absorption = jnp.full((N_sk,), ICs_at_emission_or_absorption == "absorption")
        else:
            logger.info("Using per-skewer emission/absorption flags")
            is_absorption = jnp.array([s == "absorption" for s in ICs_at_emission_or_absorption])
        
        # Compute direction per skewer: +1 or -1
        is_start = (ICs_defined_at_index == 0)
        direction = jnp.where(
            is_start, jnp.where(is_absorption, -1, 1), jnp.where(is_absorption, 1, -1)
        )

        # Reorganize spatial_points to match skewer orientation, inverting the order when the associated direction is -1
        oriented_spatial_points = jnp.where(
            direction[:, None, None] == 1,
            spatial_points,
            spatial_points[:, ::-1, :]
        )

        # Handle los_directions
        if los_directions is not None:
            logger.info("Using provided los_directions (bypassing auto-calculation)")
            los_directions = jnp.asarray(los_directions)
            if los_directions.shape != (N_sk, N_ppsk, 3): # Validate provided directions
                raise ValueError(
                    f"Provided los_directions shape {los_directions.shape} must match (N_sk, N_ppsk, 3) = {(N_sk, N_ppsk, 3)}"
                )
            # Check normalization (allow small tolerance)
            norms = jnp.linalg.norm(los_directions, axis=-1)
            max_dev = jnp.max(jnp.abs(norms - 1.0))
            if max_dev > 1e-5:
                raise ValueError(f"los_directions must be normalized. Max |‖d‖-1| = {max_dev:.3e}")
        else:
            logger.info("Computing los_directions automatically from spatial coordinate differences")
            los_directions = self._compute_los_directions(spatial_points, direction)

        # Store attributes
        self.spatial_points = spatial_points
        self.oriented_spatial_points = oriented_spatial_points
        self.a0 = a0
        self.x0 = x0
        self.ICs_defined_at_index = ICs_defined_at_index
        self.is_absorption = is_absorption
        self.direction = direction
        self.los_directions = los_directions  # Store computed/provided directions

        # compute comoving_distances_in_the_skewer_direction
        logger.info("Computing comoving distances along skewer direction")
        comoving_distances_in_the_skewer_direction = self._compute_comoving_distances_in_the_skewer_direction()
        self.comoving_distances_in_the_skewer_direction = comoving_distances_in_the_skewer_direction

        # Handle scale factors
        if manual_scale_factors_in_the_skewer_direction is not None:
            logger.info("Using provided manual_scale_factors_in_the_skewer_direction")
            self.scale_factors_in_the_skewer_direction = manual_scale_factors_in_the_skewer_direction
        else:
            logger.info("Computing scale factors from cosmology")
            if cosmologies is None: # Require cosmologies object to compute them
                logger.error("Cosmology object required but not provided")
                raise ValueError(
                    "When manual_scale_factors_in_the_skewer_direction is not provided, you must supply the 'cosmologies' argument."
                )
            self.scale_factors_in_the_skewer_direction = self._compute_scale_factors_in_the_skewer_direction(
                cosmo=cosmologies
            )
        logger.info("SkewerBundle initialization complete")

    @staticmethod
    def _compute_los_directions_single(
        spatial_points: jnp.ndarray,
        direction: int
    ) -> jnp.ndarray:
        """
        Compute LOS directions for a single skewer (JAX-safe).
        
        Parameters
        ----------
        spatial_points : jnp.ndarray, shape (N_ppsk, 3)
            Comoving coordinates [Gpc] for one skewer
        direction : int, ±1
            Integration direction: +1 from IC to observer, -1 from observer to IC
            
        Returns
        -------
        los_dir : jnp.ndarray, shape (N_ppsk, 3)
            Normalized unit vectors along the line-of-sight
        """
        # Compute differences for both directions
        # Forward: point[i+1] - point[i]
        diffs_forward = spatial_points[1:] - spatial_points[:-1]
        # Backward: point[i] - point[i+1] (reverse of forward)
        diffs_backward = spatial_points[:-1] - spatial_points[1:]
        
        # Pad each direction appropriately to maintain N_ppsk length
        # For forward: repeat last diff at the end (extrapolate)
        padded_forward = jnp.concatenate([diffs_forward, diffs_forward[-1][None, :]], axis=0)
        # For backward: repeat first diff at the beginning (extrapolate)
        padded_backward = jnp.concatenate([diffs_backward[0][None, :], diffs_backward], axis=0)
        
        # Select based on direction (JAX-safe, no Python if)
        diffs = jnp.where(direction == 1, padded_forward, padded_backward)
        
        # Normalize to unit vectors
        norms = jnp.linalg.norm(diffs, axis=-1, keepdims=True)
        # Prevent division by zero for degenerate cases
        norms = jnp.maximum(norms, 1e-20)
        return diffs / norms
    
    @staticmethod
    def _compute_los_directions(
        spatial_points: jnp.ndarray,
        direction: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Compute LOS directions for all skewers using vmap.
        
        Parameters
        ----------
        spatial_points : jnp.ndarray, shape (N_sk, N_ppsk, 3)
            Comoving coordinates for all skewers
        direction : jnp.ndarray, shape (N_sk,)
            Direction flag per skewer (±1)
            
        Returns
        -------
        los_dirs : jnp.ndarray, shape (N_sk, N_ppsk, 3)
            Normalized LoS unit vectors for all skewers
        """
        vmapped_fn = vmap(SkewerBundle._compute_los_directions_single, in_axes=(0, 0))
        return vmapped_fn(spatial_points, direction)
    
    @staticmethod
    def _compute_comoving_distances_single(
        spatial_points: jnp.ndarray,
        x0: float,
        ICs_defined_at_index: int,
        is_absorption: int,
    ) -> jnp.ndarray:
        """
        Compute comoving distances for a single skewer.
        
        Distances are measured from the observer (x=0) to each point along
        the LoS, with sign conventions for absorption vs emission.
        
        Parameters
        ----------
        spatial_points : jnp.ndarray, shape (N_ppsk, 3)
            Comoving coordinates for one skewer [Mpc/h]
        x0 : float
            Comoving distance to IC point [Mpc/h]
        ICs_defined_at_index : int
            Index where ICs are defined
        is_absorption : int (0 or 1)
            Whether this is absorption (1) or emission (0)
            
        Returns
        -------
        distances : jnp.ndarray, shape (N_ppsk,)
            Comoving distances from observer to each point [Mpc/h]
        """
        # Compute step-by-step Euclidean distances between consecutive points
        diffs = spatial_points[1:] - spatial_points[:-1]
        distances = jnp.linalg.norm(diffs, axis=-1)
        
        # If ICs are at the far end (index=-1), reverse distance accumulation
        distances = jnp.where(ICs_defined_at_index == -1, distances[::-1], distances) 
        
        # Cumulative sum from the IC point outward
        relative_distances = jnp.cumsum(distances)
        
        # For absorption, distances decrease toward observer (negative sign)
        # For emission, distances increase away from source (positive sign)
        relative_distances = jnp.where(is_absorption, -relative_distances, relative_distances)
        
        # Offset by x0 (distance to IC point) and add starting point (distance = x0)
        distances = x0 + jnp.concatenate([jnp.array([0.0]), relative_distances], axis=0)
        return distances
    
    def _compute_comoving_distances_in_the_skewer_direction(self) -> jnp.ndarray:
        """
        Compute comoving distances for all skewers.
        
        Returns
        -------
        distances : jnp.ndarray, shape (N_sk, N_ppsk)
            Comoving distances for all skewers [Mpc/h]
        """
        vmapped_fn = vmap(SkewerBundle._compute_comoving_distances_single, in_axes=(0, 0, 0, 0))
        return vmapped_fn(self.spatial_points, self.x0, self.ICs_defined_at_index, self.is_absorption)
    
    def _compute_scale_factors_in_the_skewer_direction(self, cosmo: Cosmology) -> jnp.ndarray:
        """
        Compute scale factors a(x) for all skewers using cosmology object.
        
        This solves the cosmological distance-redshift relation for each
        point along the skewer given the reference scale factor a0 at x0.
        
        Parameters
        ----------
        cosmo : Cosmology
            Cosmology object implementing solve_a_of_x(a0, distances)
            
        Returns
        -------
        scale_factors : jnp.ndarray, shape (N_sk, N_ppsk)
            Scale factor a(z) at each point along each skewer (dimensionless)
        """

        def _compute_skewer_scale_factors(a0_scalar, comoving_distances):
            # Expand a0 to match cosmology grid if needed
            a0_expanded = jnp.repeat(a0_scalar, cosmo.N_cosmologies)
            return cosmo.solve_a_of_x(a0_expanded, comoving_distances)

        scale_factors_in_the_skewer_direction = vmap(_compute_skewer_scale_factors)(
            self.a0, self.comoving_distances_in_the_skewer_direction
        )
        
        return scale_factors_in_the_skewer_direction
    
    # TODO: Store optical depth information (tau, transmitted flux) as attributes
    # These will be computed later using the SkewerJAX pipeline and attached
    # to this bundle for easy access to observables
    logger.info("SkewerBundle class definition complete")