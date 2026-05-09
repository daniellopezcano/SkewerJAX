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
        manual_scale_factors: Optional[jnp.ndarray] = None,
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
            
        manual_scale_factors : Optional[jnp.ndarray], shape (N_sk, N_cosmologies, N_ppsk)
            Pre-computed scale factors a(x) at each point along the skewer.
            If None, will be computed from cosmology using the comoving distance grid.
            
        cosmologies : Optional[Cosmology]
            Cosmology object required for computing scale factors when
            manual_scale_factors is not provided.
            Must implement solve_a_of_x(a0, distances) method.

        Attributes
        ----------
        quadri_vectors : jnp.ndarray, shape (N_cosmologies, N_sk, N_ppsk, 4)
            Quadratic vector coefficients for each skewer (a, x, y, z)
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
        
        # Compute array_direction per skewer: +1 or -1
        is_start = (ICs_defined_at_index == 0)
        array_direction = jnp.where(
            is_start, jnp.where(is_absorption, -1, 1), jnp.where(is_absorption, 1, -1)
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
            los_directions = self._compute_los_directions(spatial_points, array_direction)

        # Store attributes
        self.spatial_points = spatial_points
        self.a0 = a0
        self.x0 = x0
        self.ICs_defined_at_index = ICs_defined_at_index
        self.is_absorption = is_absorption
        self.array_direction = array_direction
        self.los_directions = los_directions  # Store computed/provided directions

        # compute comoving distances
        logger.info("Computing comoving distances along skewer direction")
        self.comoving_positions_from_ICs = self._compute_comoving_distances_starting_from_ICs()

        # Handle scale factors
        if manual_scale_factors is not None:
            logger.info("Using provided scale factors (bypassing auto-calculation)")
            # verify that the dimensions match the expected (N_cosmologies, N_sk, N_ppsk) -- length of shape == 3
            manual_scale_factors = jnp.asarray(manual_scale_factors)
            if len(manual_scale_factors.shape) != 3:
                raise ValueError(
                    f" must have shape (N_cosmologies, N_sk, N_ppsk), got {manual_scale_factors.shape}"
                )
            scale_factors = manual_scale_factors
        else:
            logger.info("Computing scale factors from cosmology")
            if cosmologies is None: # Require cosmologies object to compute them
                logger.error("Cosmology object required but not provided")
                raise ValueError(
                    "When manual_scale_factors is not provided, you must supply the 'cosmologies' argument."
                )
            scale_factors_from_ICs = self._compute_scale_factors(cosmo=cosmologies)
            # the output comes by default as (N_sk, N_cosmologies, N_ppsk), we need to transpose
            scale_factors_from_ICs = jnp.swapaxes(scale_factors_from_ICs, 0, 1)

            # Store scale factors ordered as spatial points
            mask = ICs_defined_at_index == -1
            reversed_scale_factors = jnp.flip(scale_factors_from_ICs, axis=-1) # Reverse the last axis for ALL rows
            # Apply mask: where True pick reversed, else keep original. Broadcast mask to match array dimensions
            scale_factors = jnp.where(mask[None, :, None], reversed_scale_factors, scale_factors_from_ICs)
        
        self.scale_factors = scale_factors

        # store quadri-vectors combining spatial points and scale factors
        # (N_cosmologies, N_sk, N_ppsk, 4)
        logger.info("Storing quadri-vectors")
        _spatial_points = jnp.tile(spatial_points[None, ...], (scale_factors.shape[0], 1, 1, 1))
        self.quadri_vectors = jnp.concatenate([scale_factors[..., None], _spatial_points], axis=-1)

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
        the LoS, from the IC point outward.
        
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
    
    def _compute_comoving_distances_starting_from_ICs(self) -> jnp.ndarray:
        """
        Compute comoving distances for all skewers.
        
        Returns
        -------
        distances : jnp.ndarray, shape (N_sk, N_ppsk)
            Comoving distances for all skewers [Mpc/h]
        """
        vmapped_fn = vmap(SkewerBundle._compute_comoving_distances_single, in_axes=(0, 0, 0, 0))
        return vmapped_fn(self.spatial_points, self.x0, self.ICs_defined_at_index, self.is_absorption)
    
    def _compute_scale_factors(self, cosmo: Cosmology) -> jnp.ndarray:
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
            Scale factor a(z) at each point (dimensionless)
        """

        def _compute_skewer_scale_factors(a0_scalar, comoving_distances):
            # Expand a0 to match cosmology grid if needed
            a0_expanded = jnp.repeat(a0_scalar, cosmo.N_cosmologies)
            return cosmo.solve_a_of_x(a0_expanded, comoving_distances)

        return vmap(_compute_skewer_scale_factors)(self.a0, self.comoving_positions_from_ICs)
    
    # TODO: Store optical depth information (tau, transmitted flux) as attributes
    # These will be computed later using the SkewerJAX pipeline and attached
    # to this bundle for easy access to observables
    logger.info("SkewerBundle class definition complete")


import logging
from itertools import combinations

logger = logging.getLogger(__name__)


class DeltaFieldBuilder:
    """
    Build single-field and composite overdensity fields from a dictionary
    of input fields with shape (N_samples, N_points).
    """

    def __init__(self, fields, labels=None, key_joiner="", dtype=float):
        self.fields = {k: jnp.asarray(v, dtype=dtype) for k, v in fields.items()}
        self.labels = list(fields) if labels is None else list(labels)
        self.key_joiner = key_joiner
        self.dtype = dtype

        self._validate_inputs()
        self._filter_bad_input_samples()

        self.N_samples = self.fields[self.labels[0]].shape[0]

        self.subset_keys = None
        self.composite_keys = None
        self.delta_terms = None
        self.C_per_sample = None
        self.field_total = None
        self.total = None

        logger.info(
            "Initialized DeltaFieldBuilder with %d fields, N_samples=%d, N_points=%d.",
            len(self.labels),
            self.N_samples,
            self.fields[self.labels[0]].shape[1],
        )

    def _validate_inputs(self):
        if not self.fields:
            raise ValueError("`fields` cannot be empty.")
        if set(self.labels) != set(self.fields):
            raise ValueError("`labels` must contain exactly the same keys as `fields`.")

        ref_shape = None
        for key in self.labels:
            arr = self.fields[key]
            if arr.ndim != 2:
                raise ValueError(
                    f"Field '{key}' must have shape (N_samples, N_points), got {arr.shape}."
                )
            if ref_shape is None:
                ref_shape = arr.shape
            elif arr.shape != ref_shape:
                raise ValueError(
                    f"All fields must have the same shape. Expected {ref_shape}, got {arr.shape} for field '{key}'."
                )

    def _filter_samples(self, bad_mask, reason):
        n_bad = int(jnp.sum(bad_mask))
        if n_bad == 0:
            return

        n_tot = self.fields[self.labels[0]].shape[0]
        keep_mask = ~bad_mask
        logger.warning("Removing %d/%d samples: %s", n_bad, n_tot, reason)

        for key in self.labels:
            self.fields[key] = self.fields[key][keep_mask]

        n_keep = self.fields[self.labels[0]].shape[0]
        if n_keep == 0:
            raise ValueError("All samples were removed during filtering.")
        logger.info("Remaining samples after filtering: %d", n_keep)

    def _filter_bad_input_samples(self):
        bad_mask = jnp.zeros(self.fields[self.labels[0]].shape[0], dtype=bool)

        for key in self.labels:
            arr = self.fields[key]
            nan_mask = jnp.any(jnp.isnan(arr), axis=1)
            n_nan = int(jnp.sum(jnp.isnan(arr)))
            n_bad = int(jnp.sum(nan_mask))
            if n_bad:
                logger.warning(
                    "NaNs detected in field %r: total_nan_count=%d, affected_samples=%d.",
                    key, n_nan, n_bad
                )
            bad_mask = bad_mask | nan_mask

        self._filter_samples(bad_mask, "input NaNs")

    def _make_key(self, members):
        return self.key_joiner.join(members)

    def build(self):
        logger.info("Starting build().")

        # Filter samples that would produce invalid deltas
        means = {k: jnp.mean(self.fields[k], axis=1) for k in self.labels}
        zero_mean_mask = jnp.zeros(self.fields[self.labels[0]].shape[0], dtype=bool)

        for key, mean in means.items():
            bad = mean == 0
            n_bad = int(jnp.sum(bad))
            if n_bad:
                logger.warning("Field %r has %d samples with zero mean.", key, n_bad)
            zero_mean_mask = zero_mean_mask | bad

        stacked_fields = jnp.stack([self.fields[k] for k in self.labels], axis=0)
        field_total = jnp.prod(stacked_fields, axis=0)
        total_mean = jnp.mean(field_total, axis=1)
        bad_total = total_mean == 0
        n_bad_total = int(jnp.sum(bad_total))
        if n_bad_total:
            logger.warning("Total field has %d samples with zero mean.", n_bad_total)
        zero_mean_mask = zero_mean_mask | bad_total

        self._filter_samples(zero_mean_mask, "zero mean in single field or total field")
        self.N_samples = self.fields[self.labels[0]].shape[0]

        self.delta_terms = {}
        self.subset_keys = []
        self.composite_keys = []
        self.C_per_sample = jnp.zeros((self.N_samples, 1), dtype=self.dtype)

        # Single-field deltas
        for key in self.labels:
            mean = jnp.mean(self.fields[key], axis=1, keepdims=True)
            term = self.fields[key] / mean - 1.0
            if int(jnp.sum(jnp.isnan(term))):
                logger.warning("NaNs detected in delta_terms[%r].", key)
            self.delta_terms[key] = term
            self.subset_keys.append(key)

        # Total field and total delta
        stacked_fields = jnp.stack([self.fields[k] for k in self.labels], axis=0)
        self.field_total = jnp.prod(stacked_fields, axis=0)
        field_total_mean = jnp.mean(self.field_total, axis=1, keepdims=True)
        self.total = self.field_total / field_total_mean - 1.0

        if int(jnp.sum(jnp.isnan(self.field_total))):
            logger.warning("NaNs detected in field_total.")
        if int(jnp.sum(jnp.isnan(self.total))):
            logger.warning("NaNs detected in total.")

        # Composite overdensity fields
        for r in range(2, len(self.labels) + 1):
            for members in combinations(self.labels, r):
                key = self._make_key(members)
                term = self.delta_terms[members[0]]
                for member in members[1:]:
                    term = term * self.delta_terms[member]

                if int(jnp.sum(jnp.isnan(term))):
                    logger.warning("NaNs detected in delta_terms[%r].", key)

                self.delta_terms[key] = term
                self.subset_keys.append(key)
                self.composite_keys.append(key)
                self.C_per_sample += jnp.mean(term, axis=1, keepdims=True)

        if int(jnp.sum(jnp.isnan(self.C_per_sample))):
            logger.warning("NaNs detected in C_per_sample.")

        logger.info(
            "Finished build(): N_samples=%d, subset_keys=%d, composite_keys=%d.",
            self.N_samples,
            len(self.subset_keys),
            len(self.composite_keys),
        )
        return self