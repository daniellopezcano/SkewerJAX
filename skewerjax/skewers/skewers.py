from __future__ import annotations

from typing import Optional, Sequence, List
import jax.numpy as jnp
from jax.typing import ArrayLike

from skewerjax.cosmology.cosmology import Cosmology

class Skewer:
    """
    Single line-of-sight skewer through a cosmological volume.
    
    Parameters
    ----------
    spatial_points : ArrayLike, shape (N_ppsk, 3)
        Comoving spatial coordinates [x, y, z] in Mpc/h
    a0 : float
        Scale factor at the first point (initial/final condition)
    a0_defined_at_index : int, optional
        Index where a0 is defined. Must be 0 (first point) or -1 (last point).
        Defaults to 0 (left-to-right direction).
    """
    
    def __init__(self, spatial_points: ArrayLike, a0: float,
                 a0_defined_at_index: int = 0):
        # Convert and validate spatial_points
        spatial_points = jnp.asarray(spatial_points)
        if spatial_points.ndim != 2 or spatial_points.shape[-1] != 3:
            raise ValueError(f"spatial_points must be (N_ppsk, 3), got {spatial_points.shape}")
        self.spatial_points = spatial_points
        
        # Store a0
        self.a0 = float(a0)
        
        # Validate and store a0_defined_at_index
        if a0_defined_at_index not in (0, -1):
            raise ValueError(f"a0_defined_at_index must be 0 or -1, got {a0_defined_at_index}")
        self.a0_defined_at_index = a0_defined_at_index
        
        # Set index_direction based on index
        if a0_defined_at_index == 0:
            # Left-to-right direction
            self.index_direction = +1
        else:  # -1
            # Right-to-left direction
            self.index_direction = -1
    
    @property
    def N_ppsk(self) -> int:
        """Number of points per skewer."""
        return self.spatial_points.shape[0]
    
    def compute_comoving_distances(self) -> jnp.ndarray:
        """
        Compute cumulative comoving distance along the skewer.
        
        Returns
        -------
        distances : jnp.ndarray, shape (N_ppsk,)
            Cumulative comoving distance from the first point [Mpc/h]
        """
        diffs = self.spatial_points[1:] - self.spatial_points[:-1]
        distances = jnp.linalg.norm(diffs, axis=-1)
        # Insert 0 at the beginning
        return jnp.concatenate([
            jnp.zeros(1, dtype=distances.dtype),
            jnp.cumsum(distances)
        ])
    
    def compute_scale_factors(self, cosmo: Cosmology) -> jnp.ndarray:
        """
        Compute scale factor along the skewer using cosmological ODE.
        
        Args
        ----
        cosmo : Cosmology
            Cosmology instance with solve_a_of_x method
            
        Returns
        -------
        scale_factors : jnp.ndarray, shape (N_ppsk,)
        """
        comoving_distances = self.compute_comoving_distances()

        if self.a0_defined_at_index == 0:
            # Standard case: a0 at first point
            a_of_x = cosmo.solve_a_of_x(self.a0, comoving_distances)
        else:
            # a0 at last point: reverse distances, solve, then reverse result back
            distances_reversed = comoving_distances[-1] - comoving_distances
            a_reversed = cosmo.solve_a_of_x(self.a0, distances_reversed)
            a_of_x = a_reversed[::-1]

        return a_of_x
    
    def compute_los_directions(self) -> jnp.ndarray:
        """
        Compute normalized line-of-sight direction vectors.
        
        Returns
        -------
        directions : jnp.ndarray, shape (N_ppsk, 3)
            Normalized direction vectors pointing toward the observer.
            The observer is at index 0 if a0_defined_at_index=0, otherwise at index -1.
        """
        # Compute raw segment directions (from i to i+1)
        diffs = self.spatial_points[1:] - self.spatial_points[:-1]
        norms = jnp.linalg.norm(diffs, axis=-1, keepdims=True)
        norms = jnp.maximum(norms, 1e-20)
        segment_dirs = diffs / norms
        
        if self.index_direction == +1:
            # Observer at first point: directions point toward decreasing index
            # So we negate segment directions (points from i+1 to i)
            dirs = -segment_dirs
            
            # First point is observer: repeat direction from point 1 to point 0
            observer_dir = dirs[0:1]
            return jnp.concatenate([observer_dir, dirs], axis=0)
        else:
            # Observer at last point: directions point toward increasing index
            # So we use segment directions as-is (points from i to i+1)
            
            # Last point is observer: repeat direction from point N-2 to point N-1
            observer_dir = segment_dirs[-1:]
            return jnp.concatenate([segment_dirs, observer_dir], axis=0)
    
    # TODO: Implement raytracing initialization method
    # TODO: Store optical depth information (e.g., tau, transmitted flux)