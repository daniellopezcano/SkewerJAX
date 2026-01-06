# from __future__ import annotations

# from typing import Optional, Sequence, List
# import jax.numpy as jnp
# from jax.typing import ArrayLike

# from skewerjax.cosmology.cosmology import Cosmology

# class Skewer:
#     """
#     Single line-of-sight skewer through a cosmological volume.
    
#     Parameters
#     ----------
#     spatial_points : ArrayLike, shape (N_ppsk, 3)
#         Comoving spatial coordinates [x, y, z] in [Gpc]
#     a0 : float
#         Scale factor at the 'initial conditions point'
#     x0 : float
#         Comoving distance at the 'initial conditions point' [Gpc].
#     ICs_at_emission_or_absorption : str, optional
#         Whether the initial conditions are defined at 'emission' or 'absorption'.
#         Defaults to 'absorption'.
#     ICs_defined_at_index : int, optional
#         Index where the initial conditions are defined. Must be 0 (first point) or -1 (last point).
#         Defaults to -1.
#     """
    
#     def __init__(
#             self,
#             spatial_points: ArrayLike,
#             a0: float = 1.0,
#             x0: float = 0.,
#             ICs_defined_at_index: int = -1,
#             ICs_at_emission_or_absorption: str = "absorption"
#         ):

#         # Store a0 and x0
#         self.a0 = float(a0)
#         self.x0 = float(x0)

#         # Convert and validate spatial_points
#         spatial_points = jnp.asarray(spatial_points)
#         if spatial_points.ndim != 2 or spatial_points.shape[-1] != 3:
#             raise ValueError(f"spatial_points must be (N_ppsk, 3), got {spatial_points.shape}")
#         self.spatial_points = spatial_points
        
#         # Validate and store ICs_defined_at_index
#         if ICs_defined_at_index not in (0, -1):
#             raise ValueError(f"ICs_defined_at_index must be 0 or -1, got {ICs_defined_at_index}")
#         self.ICs_defined_at_index = ICs_defined_at_index

#         # Validate ICs_at_emission_or_absorption
#         if ICs_at_emission_or_absorption not in ("emission", "absorption"):
#             raise ValueError(f"ICs_at_emission_or_absorption must be 'emission' or 'absorption', got {ICs_at_emission_or_absorption}")
#         self.ICs_at_emission_or_absorption = ICs_at_emission_or_absorption
        
#         # Set skewer_direction
#         if ICs_defined_at_index == 0:
#             if ICs_at_emission_or_absorption == "emission": 
#                 self.direction = +1 # Left-to-right direction
#             else: # "absorption"
#                 self.direction = -1 # Right-to-left direction
#         else:  # -1
#             if ICs_at_emission_or_absorption == "absorption":
#                 self.direction = +1 # Left-to-right direction 
#             else: # "emission"
#                 self.direction = -1 # Right-to-left direction

#     @property
#     def N_ppsk(self) -> int:
#         """Number of points per skewer."""
#         return self.spatial_points.shape[0]
    
#     def compute_comoving_distances(self, match_spatial_points_order: bool = False) -> jnp.ndarray:
#         """
#         Compute cumulative comoving distance along the skewer.
        
#         Returns
#         -------
#         distances : jnp.ndarray, shape (N_ppsk,)
#             Cumulative comoving distance from the first point [Mpc/h]
#         """
#         diffs = self.spatial_points[1:] - self.spatial_points[:-1]
#         distances = jnp.linalg.norm(diffs, axis=-1)

#         if self.ICs_defined_at_index == -1:
#             distances = distances[::-1]  # Reverse distances for right-to-left

#         relative_distances = jnp.cumsum(distances)

#         if self.ICs_at_emission_or_absorption == "absorption":
#             relative_distances = -relative_distances

#         distances = self.x0 + jnp.concatenate([jnp.array([0.]), relative_distances], axis=0)

#         if match_spatial_points_order and self.ICs_defined_at_index == -1:
#             distances = distances[::-1]  # Reverse for right-to-left

#         return distances
    
#     def compute_scale_factors(self, cosmo: Cosmology, match_spatial_points_order: bool = False) -> jnp.ndarray:
#         """
#         Compute scale factor along the skewer using a cosmology object.
        
#         Args
#         ----
#         cosmo : Cosmology
#             Cosmology instance with solve_a_of_x method
            
#         Returns
#         -------
#         scale_factors : jnp.ndarray, shape (N_ppsk,)
#         """
        
#         comoving_distances = self.compute_comoving_distances()
#         a_of_x = cosmo.solve_a_of_x(self.a0, comoving_distances)

#         if match_spatial_points_order and self.ICs_defined_at_index == -1:
#             a_of_x = a_of_x[:, ::-1]  # Reverse for right-to-left

#         return a_of_x
    
#     def compute_los_directions(self) -> jnp.ndarray:
#         """
#         Compute normalized line-of-sight direction vectors.
        
#         Returns
#         -------
#         directions : jnp.ndarray, shape (N_ppsk, 3)
#             Normalized direction vectors at each skewer point
#         """
#         if self.direction == +1: # Compute raw segment directions (from i to i+1)
#             diffs = self.spatial_points[1:] - self.spatial_points[:-1]    
#             diffs = jnp.concatenate([diffs, diffs[-1][None, :]], axis=0)  # Pad with zero vector at the end
#         else:  # -1, Compute raw segment directions (from i to i-1)
#             diffs = self.spatial_points[:-1] - self.spatial_points[1:]
#             diffs = jnp.concatenate([diffs[0][None, :], diffs], axis=0)  # Pad with zero vector at the beginning

#         norms = jnp.linalg.norm(diffs, axis=-1, keepdims=True)
#         norms = jnp.maximum(norms, 1e-20)
#         segment_dirs = diffs / norms

#         return segment_dirs
        
#     # TODO: Implement raytracing initialization method
#     # TODO: Store optical depth information (e.g., tau, transmitted flux)







# skewerjax/skewers.py
from __future__ import annotations

from typing import Optional, Sequence, List, Union
import jax.numpy as jnp
from jax import vmap
from jax.typing import ArrayLike

from skewerjax.cosmology.cosmology import Cosmology

class SkewerBundle:
    """
    Collection of line-of-sight skewers through a cosmological volume.
    
    All skewers must have the same number of points (N_ppsk).
    
    Parameters
    ----------
    spatial_points : ArrayLike, shape (N_skewers, N_ppsk, 3)
        Comoving spatial coordinates [x, y, z] in [Gpc] for all skewers
    a0 : ArrayLike, shape (N_skewers,)
        Scale factor at the IC point for each skewer
    x0 : ArrayLike, shape (N_skewers,)
        Comoving distance at the IC point for each skewer [Gpc]
    ICs_defined_at_index : ArrayLike, shape (N_skewers,)
        Index where ICs are defined: 0 (first point) or -1 (last point)
    ICs_at_emission_or_absorption : ArrayLike of str or str
        Whether ICs are at 'emission' or 'absorption'
    los_directions : ArrayLike, shape (N_skewers, N_ppsk, 3), optional
        Pre-computed normalized line-of-sight direction vectors.
        If provided, must be normalized (‖d‖ ≈ 1).
    """
    
    def __init__(
        self,
        spatial_points: ArrayLike,
        a0: ArrayLike,
        x0: ArrayLike,
        ICs_defined_at_index: ArrayLike = -1,
        ICs_at_emission_or_absorption: Union[str, Sequence[str]] = "absorption",
        los_directions: Optional[ArrayLike] = None
    ):
        # Convert and validate spatial_points
        spatial_points = jnp.asarray(spatial_points)
        if spatial_points.ndim != 3 or spatial_points.shape[-1] != 3:
            raise ValueError(f"spatial_points must be (N_sk, N_ppsk, 3), got {spatial_points.shape}")
        N_sk, N_ppsk = spatial_points.shape[:2]
        
        # Enforce uniform N_ppsk across all skewers
        self._N_sk = N_sk
        self._N_ppsk = N_ppsk
        
        # Convert a0, x0 to arrays and broadcast to (N_sk,)
        a0 = jnp.broadcast_to(jnp.asarray(a0), (N_sk,))
        x0 = jnp.broadcast_to(jnp.asarray(x0), (N_sk,))
        
        # Convert ICs_defined_at_index to array
        ICs_defined_at_index = jnp.asarray(ICs_defined_at_index)
        if ICs_defined_at_index.ndim == 0:
            ICs_defined_at_index = jnp.full((N_sk,), ICs_defined_at_index, dtype=jnp.int32)
        
        # Convert ICs_at_emission_or_absorption to boolean array
        if isinstance(ICs_at_emission_or_absorption, str):
            is_absorption = jnp.full((N_sk,), ICs_at_emission_or_absorption == "absorption")
        else:
            is_absorption = jnp.array([s == "absorption" for s in ICs_at_emission_or_absorption])
        
        # Compute direction per skewer: +1 or -1
        is_start = (ICs_defined_at_index == 0)
        direction = jnp.where(
            is_start,
            jnp.where(is_absorption, -1, 1),
            jnp.where(is_absorption, 1, -1)
        )
        
        # Handle los_directions
        if los_directions is not None:
            # Validate provided directions
            los_directions = jnp.asarray(los_directions)
            if los_directions.shape != (N_sk, N_ppsk, 3):
                raise ValueError(
                    f"Provided los_directions shape {los_directions.shape} must match "
                    f"(N_sk, N_ppsk, 3) = {(N_sk, N_ppsk, 3)}"
                )
            # Check normalization (allow small tolerance)
            norms = jnp.linalg.norm(los_directions, axis=-1)
            max_dev = jnp.max(jnp.abs(norms - 1.0))
            if max_dev > 1e-5:
                raise ValueError(f"los_directions must be normalized. Max |‖d‖-1| = {max_dev:.3e}")
        else:
            # Compute directions automatically
            los_directions = self._compute_los_directions(spatial_points, direction)
        
        # Store attributes
        self.spatial_points = spatial_points
        self.a0 = a0
        self.x0 = x0
        self.ICs_defined_at_index = ICs_defined_at_index
        self.is_absorption = is_absorption
        self.direction = direction
        self.los_directions = los_directions  # Store computed/provided directions
    
    @staticmethod
    def _compute_los_directions_single(
        spatial_points: jnp.ndarray,
        direction: int
    ) -> jnp.ndarray:
        """Compute LOS directions for a single skewer (JAX-safe)."""
        # Compute differences for both directions
        diffs_forward = spatial_points[1:] - spatial_points[:-1]
        diffs_backward = spatial_points[:-1] - spatial_points[1:]
        
        # Pad each direction appropriately
        padded_forward = jnp.concatenate([diffs_forward, diffs_forward[-1][None, :]], axis=0)
        padded_backward = jnp.concatenate([diffs_backward[0][None, :], diffs_backward], axis=0)
        
        # Select based on direction (JAX-safe, no Python if)
        diffs = jnp.where(direction == 1, padded_forward, padded_backward)
        
        # Normalize
        norms = jnp.linalg.norm(diffs, axis=-1, keepdims=True)
        norms = jnp.maximum(norms, 1e-20)
        return diffs / norms
    
    @staticmethod
    def _compute_los_directions(
        spatial_points: jnp.ndarray,
        direction: jnp.ndarray
    ) -> jnp.ndarray:
        """Compute LOS directions for all skewers using vmap."""
        vmapped_fn = vmap(SkewerBundle._compute_los_directions_single, in_axes=(0, 0))
        return vmapped_fn(spatial_points, direction)
    
    @property
    def N_skewers(self) -> int:
        """Number of skewers."""
        return self._N_sk
    
    @property
    def array_N_ppsk(self) -> jnp.ndarray:
        """Array of number of points per skewer (uniform)."""
        return jnp.full((self._N_sk,), self._N_ppsk, dtype=jnp.int32)
    
    @property
    def N_ppsk(self) -> int:
        """Number of points per skewer (uniform across all skewers)."""
        return self._N_ppsk
    
    @staticmethod
    def _compute_comoving_distances_single(
        spatial_points: jnp.ndarray,
        x0: float,
        direction: int
    ) -> jnp.ndarray:
        """Compute comoving distances for a single skewer."""
        diffs = spatial_points[1:] - spatial_points[:-1]
        distances = jnp.linalg.norm(diffs, axis=-1)
        
        relative_distances = jnp.cumsum(distances)
        
        # Apply direction sign and padding
        relative_distances = jnp.where(direction == -1, -relative_distances[::-1], -relative_distances)
        
        all_distances = jnp.concatenate([jnp.array([0.0]), relative_distances], axis=0)
        return x0 + all_distances
    
    def compute_comoving_distances(self) -> jnp.ndarray:
        """Compute comoving distances for all skewers."""
        vmapped_fn = vmap(SkewerBundle._compute_comoving_distances_single, in_axes=(0, 0, 0))
        return vmapped_fn(self.spatial_points, self.x0, self.direction)
    
    def compute_scale_factors(self, cosmo: Cosmology) -> jnp.ndarray:
        """Compute scale factors for all skewers using cosmology."""
        distances = self.compute_comoving_distances()
        vmapped_solve = vmap(cosmo.solve_a_of_x, in_axes=(0, 0))
        return vmapped_solve(self.a0, distances)
    
    # TODO: Implement method to pad skewers to uniform length (future: ragged arrays)
    # TODO: Store optical depth information (tau, transmitted flux) as attributes