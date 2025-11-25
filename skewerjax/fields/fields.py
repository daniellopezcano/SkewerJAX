import jax.numpy as jnp
from typing import Dict, Sequence, Union, Callable
from dataclasses import dataclass

# Type alias for clarity
Points = jnp.ndarray  # Shape: (N_points, 4) where columns are [scale_factor, x, y, z]

# ─────────────────────────────────────────────────────────────────────────────
# Base field classes
# ─────────────────────────────────────────────────────────────────────────────

class ScalarField:
    """Species-independent scalar field: f(points) → values."""
    def __call__(self, points: Points) -> jnp.ndarray:
        raise NotImplementedError

class VectorField:
    """Species-independent 3D vector field: f(points) → vectors."""
    def __call__(self, points: Points) -> jnp.ndarray:
        raise NotImplementedError

# ─────────────────────────────────────────────────────────────────────────────
# Concrete fields
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class UniformScalarField(ScalarField):
    """Constant scalar everywhere."""
    value: float
    
    def __call__(self, points: Points) -> jnp.ndarray:
        return jnp.full(points.shape[0], self.value)

@dataclass(frozen=True)
class UniformVectorField(VectorField):
    """Constant vector everywhere."""
    vx: float; vy: float; vz: float
    
    def __call__(self, points: Points) -> jnp.ndarray:
        vec = jnp.array([self.vx, self.vy, self.vz])
        return jnp.tile(vec, (points.shape[0], 1))

# ─────────────────────────────────────────────────────────────────────────────
# Species-dependent wrappers
# ─────────────────────────────────────────────────────────────────────────────

class SpeciesScalarField:
    """Species-dependent scalar: f(species_key, points) → values."""
    def __call__(self, species_key: str, points: Points) -> jnp.ndarray:
        raise NotImplementedError

class SpeciesVectorField:
    """Species-dependent vector: f(species_key, points) → vectors."""
    def __call__(self, species_key: str, points: Points) -> jnp.ndarray:
        raise NotImplementedError

@dataclass(frozen=True)
class SharedSpeciesScalarField(SpeciesScalarField):
    """Use same scalar field for all species."""
    base_field: ScalarField
    
    def __call__(self, species_key: str, points: Points) -> jnp.ndarray:
        return self.base_field(points)  # Species key ignored

@dataclass(frozen=True)
class SharedSpeciesVectorField(SpeciesVectorField):
    """Use same vector field for all species."""
    base_field: VectorField
    
    def __call__(self, species_key: str, points: Points) -> jnp.ndarray:
        return self.base_field(points)  # Species key ignored

@dataclass(frozen=True)
class SpeciesFieldMap(SpeciesScalarField):
    """Different scalar field per species."""
    fields: Dict[str, ScalarField]
    
    def __call__(self, species_key: str, points: Points) -> jnp.ndarray:
        return self.fields[species_key](points)

# ─────────────────────────────────────────────────────────────────────────────
# FieldBundle: container for all physics
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FieldBundle:
    """All fields needed to compute line profiles."""
    temperature: SpeciesScalarField
    collisional: SpeciesScalarField
    turbulence: SpeciesScalarField
    velocity: SpeciesVectorField
    
    @classmethod
    def create_uniform(
        cls,
        temp_dict: Dict[str, float],
        coll_dict: Dict[str, float],
        turb_dict: Dict[str, float],
        shared_velocity: tuple[float, float, float] = None,
        vel_dict: Dict[str, tuple[float, float, float]] = None,
    ) -> "FieldBundle":
        """Quick factory for uniform fields."""
        if shared_velocity is not None and vel_dict is not None:
            raise ValueError("Specify either shared_velocity or vel_dict, not both")
        
        # Build scalar maps
        temp_map = SpeciesFieldMap({
            sid: UniformScalarField(val) for sid, val in temp_dict.items()
        })
        coll_map = SpeciesFieldMap({
            sid: UniformScalarField(val) for sid, val in coll_dict.items()
        })
        turb_map = SpeciesFieldMap({
            sid: UniformScalarField(val) for sid, val in turb_dict.items()
        })
        
        # Build velocity field
        if vel_dict:
            vel_map = SpeciesFieldMap({
                sid: UniformVectorField(*vals) for sid, vals in vel_dict.items()
            })
        else:
            vx, vy, vz = shared_velocity or (0.0, 0.0, 0.0)
            vel_map = SharedSpeciesVectorField(UniformVectorField(vx, vy, vz))
        
        return cls(temp_map, coll_map, turb_map, vel_map)
    
    def project_velocity_los(
        self,
        species_keys: Sequence[str],
        points: Points,
        los_directions: jnp.ndarray,  # Shape: (N_points, 3)
    ) -> jnp.ndarray:
        """Compute LOS velocity for multiple species."""
        velocities = []
        for species in species_keys:
            v_3d = self.velocity(species, points)  # (N_points, 3)
            v_los = jnp.sum(v_3d * los_directions, axis=1)  # (N_points,)
            velocities.append(v_los)
        return jnp.stack(velocities)  # (N_sp, N_points)

# ─────────────────────────────────────────────────────────────────────────────
# Convenience function for your use case
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_fields_for_lines(
    bundle: FieldBundle,
    species_keys: Sequence[str],
    points: Points,
    los_directions: jnp.ndarray,
) -> dict:
    """
    Evaluate all fields for given species and points.
    
    Returns dictionary with keys:
      - 'temperature': (N_sp, N_points)
      - 'collisional': (N_sp, N_points)
      - 'turbulence': (N_sp, N_points)
      - 'velocity_los': (N_sp, N_points)
    """
    return {
        'temperature': jnp.stack([bundle.temperature(k, points) for k in species_keys]),
        'collisional': jnp.stack([bundle.collisional(k, points) for k in species_keys]),
        'turbulence': jnp.stack([bundle.turbulence(k, points) for k in species_keys]),
        'velocity_los': bundle.project_velocity_los(species_keys, points, los_directions),
    }