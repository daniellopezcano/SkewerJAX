# skewerjax/fields.py
from __future__ import annotations
from typing import Callable, Protocol, Any, Union
import jax.numpy as jnp
from jax.typing import ArrayLike

# =============================================================================
# Field Protocols & Base Classes
# =============================================================================

class Field(Protocol):
    """A callable that samples a field at points x."""
    def __call__(self, x: ArrayLike) -> ArrayLike:
        """x: (N_points, 4) → output: scalar or vector per point."""
        ...

class ScalarField:
    """
    Immutable wrapper for scalar-valued fields.
    Usage: field = ScalarField(lambda x: x[:,0])  # samples scale factor
    """
    def __init__(self, sampler: Callable[[ArrayLike], ArrayLike]):
        self._sampler = sampler
    
    def __call__(self, x: ArrayLike) -> ArrayLike:
        return self._sampler(x)  # Returns (N_points,)

class VectorField:
    """
    Immutable wrapper for vector-valued fields (e.g., velocity).
    Usage: field = VectorField(lambda x: jnp.stack([x[:,1], x[:,2], x[:,3]], axis=-1))
    """
    def __init__(self, sampler: Callable[[ArrayLike], ArrayLike]):
        self._sampler = sampler
    
    def __call__(self, x: ArrayLike) -> ArrayLike:
        return self._sampler(x)  # Returns (N_points, 3)
    
    def project(self, x: ArrayLike, directions: ArrayLike, tol: float = 1e-5) -> ArrayLike:
        """
        Project onto line-of-sight directions.

        Parameters
        ----------
        x : (..., 4)
            Sampling points.
        directions : (..., 3)
            Line-of-sight direction vectors. Must be normalized (‖d‖ = 1).
        tol : float
            Allowed deviation from unit norm. If max |‖d‖ - 1| > tol,
            a ValueError is raised.

        Returns
        -------
        projected : (...,)
            Velocity projected along each line-of-sight.
        """
        v = self(x)  # (..., 3)

        # Check normalization of LoS directions
        norms = jnp.linalg.norm(directions, axis=-1)          # (...)
        deviations = jnp.abs(norms - 1.0)                     # (...)
        max_dev = float(jnp.max(deviations))                  # scalar (for message)

        if max_dev > tol:
            raise ValueError(
                f"LoS directions must be normalized (‖d‖ ≈ 1). "
                f"Max |‖d‖ - 1| = {max_dev:.3e} > tol = {tol:.3e}."
            )

        # Actual projection
        return jnp.sum(v * directions, axis=-1)


class LoSProjector:
    """Descriptor that creates an LoS-projected field on-demand."""
    def __init__(self, vector_field: VectorField):
        self._vf = vector_field
    
    def sample(self, x: ArrayLike, directions: ArrayLike) -> ArrayLike:
        """Return LoS velocity at x along directions."""
        return self._vf.project(x, directions)

# =============================================================================
# Species & Master Container
# =============================================================================

class SpeciesFields:
    """
    Container for all fields of a single species.
    Access: species_field.temperature(x) -> (N_points,)
    """
    def __init__(self, fields: dict[str, Field]):
        self._fields = fields
    
    def __getattr__(self, name: str) -> Field:
        """Enable dot-access: species.temperature"""
        if name in self._fields:
            return self._fields[name]
        raise AttributeError(
            f"Field '{name}' not found. Available: {list(self._fields.keys())}"
        )
    
    @property
    def LoS_velocity(self) -> LoSProjector:
        """Special property: projects species velocity onto LoS."""
        if "velocity" not in self._fields:
            raise AttributeError("Cannot project LoS_velocity: no 'velocity' field defined.")
        return LoSProjector(self._fields["velocity"])


class Fields:
    """
    Master container for all species.

    Construction:
        Fields({"HI": {"temperature": ScalarField(...), ...}})

    Access:
        - Per-species:
            fields.HI.temperature(x)
        - All species, all fields:
            fields.sample(x, LoS_directions, as_array=False/True)
        - All species, single field:
            fields.temperature(x)                 -> (N_species, *batch_shape)
            fields.velocity(x)                    -> (N_species, *batch_shape, 3)
            fields.LoS_velocity(x, LoS_dirs)      -> (N_species, *batch_shape)
    """
    def __init__(self, species_fields: dict[str, dict[str, "Field"]]):
        self._species = {
            name: SpeciesFields(fields) for name, fields in species_fields.items()
        }
    
    def __getattr__(self, name: str) -> Any:
        """
        Enable:
          - dot-access to species: fields.HI
          - dot-access to cross-species field samplers: fields.temperature(x)
          - special-case LoS_velocity: fields.LoS_velocity(x, LoS_directions)
        """
        # 1) Species access: fields.HI
        if name in self._species:
            return self._species[name]

        # 2) Special case: LoS_velocity is *not* in species._fields,
        #    but is exposed via SpeciesFields.LoS_velocity.
        if name == "LoS_velocity":
            return self._make_LoS_velocity_sampler()

        # 3) Generic cross-species field access for fields that *are* in _fields
        if any(name in species._fields for species in self._species.values()):
            return self._make_field_sampler(name)

        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'. "
            f"Available species: {list(self._species.keys())}"
        )
    
    @property
    def species_keys(self) -> tuple[str, ...]:
        """Tuple of defined species names."""
        return tuple(self._species.keys())
    
    @property
    def field_keys(self) -> dict[str, tuple[str, ...]]:
        """Mapping species → tuple of field names."""
        return {
            name: tuple(species._fields.keys())
            for name, species in self._species.items()
        }

    def _make_field_sampler(self, field_name: str):
        """
        fields.<field_name>(x, *args, **kwargs) -> (N_species, *field_shape)
        """
        def _sampler(x: "ArrayLike", *args, **kwargs) -> jnp.ndarray:
            species_names = self.species_keys
            if not species_names:
                raise ValueError("No species defined in Fields.")

            values = []
            ref_shape = None

            for sname in species_names:
                species = self._species[sname]
                if field_name not in species._fields:
                    raise AttributeError(
                        f"Field '{field_name}' not found in species '{sname}'. "
                        "All species must carry this field for cross-species sampling."
                    )

                field = species._fields[field_name]
                v = jnp.asarray(field(x, *args, **kwargs))

                if ref_shape is None:
                    ref_shape = v.shape
                elif v.shape != ref_shape:
                    raise ValueError(
                        f"Inconsistent shape for field '{field_name}' across species. "
                        f"Species '{sname}' has shape {v.shape}, expected {ref_shape}."
                    )

                values.append(v)

            # (N_species, *ref_shape)
            return jnp.stack(values, axis=0)

        _sampler.__name__ = field_name
        return _sampler

    def _make_LoS_velocity_sampler(self):
        """
        fields.LoS_velocity(x, LoS_directions) -> (N_species, *batch_shape)
        """
        def _sampler(x: "ArrayLike", LoS_directions: "ArrayLike") -> jnp.ndarray:
            species_names = self.species_keys
            if not species_names:
                raise ValueError("No species defined in Fields.")

            values = []
            ref_shape = None

            for sname in species_names:
                species = self._species[sname]

                # Uses SpeciesFields.LoS_velocity property, which requires 'velocity'
                projector = species.LoS_velocity
                v = jnp.asarray(projector.sample(x, LoS_directions))

                if ref_shape is None:
                    ref_shape = v.shape
                elif v.shape != ref_shape:
                    raise ValueError(
                        "Inconsistent LoS_velocity shape across species. "
                        f"Species '{sname}' has shape {v.shape}, expected {ref_shape}."
                    )

                values.append(v)

            # (N_species, *batch_shape)
            return jnp.stack(values, axis=0)

        _sampler.__name__ = "LoS_velocity"
        return _sampler
    
    def sample(
        self,
        x: "ArrayLike",
        LoS_directions: "ArrayLike",
        as_array: bool = False,
    ) -> Any:
        """
        Sample ALL fields for ALL species at once.

        Default (as_array=False)
        ------------------------
        Returns a nested dict:
        {
            "HI": {
                "temperature": (batch_shape),
                "velocity": (batch_shape, 3),
                "LoS_velocity": (batch_shape,),  # projected
                "number_density": (batch_shape,),
                ...
            },
            "SiIII": {...}
        }

        Array mode (as_array=True)
        ---------------------------
        Returns a single jnp.ndarray with shape:
            (N_species, N_scalar_fields, *batch_shape)

        where:
        - batch_shape = x.shape[:-1]
        - Only scalar fields are included (shape == batch_shape).
        - Vector fields (e.g. "velocity" with trailing dim 3) are ignored.
        - Their LoS projections (e.g. "LoS_velocity") ARE included.
        - Axes:
            0 -> species (in self.species_keys order)
            1 -> scalar fields (sorted by name for determinism)
            2+ -> original point grid (same as each scalar field)
        """
        # --------- 1. Build the usual nested dict of sampled fields ---------
        results: dict[str, dict[str, jnp.ndarray]] = {}
        for name, species in self._species.items():
            species_results: dict[str, jnp.ndarray] = {}
            for field_name, field in species._fields.items():
                species_results[field_name] = field(x)
            
            # Add LoS projection if velocity exists
            if "velocity" in species._fields:
                species_results["LoS_velocity"] = species.LoS_velocity.sample(
                    x, LoS_directions
                )
            
            results[name] = species_results
        
        # If standard dict mode: we're done
        if not as_array:
            return results

        # --------- 2. Convert to a stacked jnp.ndarray without flattening ---------
        batch_shape = x.shape[:-1]  # e.g. (N_sk, N_ppsk)

        species_names = self.species_keys
        if not species_names:
            raise ValueError("No species defined in Fields.")

        # Use first species as reference to decide which fields are scalar
        ref_species_name = species_names[0]
        ref_fields = results[ref_species_name]

        # Scalar fields: those whose shape == batch_shape
        scalar_field_names = [
            fname for fname, value in ref_fields.items()
            if jnp.shape(value) == batch_shape
        ]
        if not scalar_field_names:
            raise ValueError(
                f"No scalar fields found for species '{ref_species_name}'. "
                "Cannot build stacked array."
            )

        # Optional: keep ordering deterministic (alphabetical)
        scalar_field_names = sorted(scalar_field_names)

        # Sanity-check that all species have these scalar fields
        for sname in species_names:
            for fname in scalar_field_names:
                if fname not in results[sname]:
                    raise ValueError(
                        f"Scalar field '{fname}' missing in species '{sname}'. "
                        "Cannot build a uniform stacked array."
                    )
                if jnp.shape(results[sname][fname]) != batch_shape:
                    raise ValueError(
                        f"Field '{fname}' in species '{sname}' does not match "
                        f"batch shape {batch_shape}: got {jnp.shape(results[sname][fname])}"
                    )

        # Stack: for each species → (N_scalar_fields, *batch_shape),
        # then stack species → (N_species, N_scalar_fields, *batch_shape)
        species_stacks = []
        for sname in species_names:
            field_values = []
            for fname in scalar_field_names:
                v = jnp.asarray(results[sname][fname])  # shape batch_shape
                field_values.append(v)
            # (N_scalar_fields, *batch_shape)
            species_stacks.append(jnp.stack(field_values, axis=0))

        # Final: (N_species, N_scalar_fields, *batch_shape)
        stacked = jnp.stack(species_stacks, axis=0)
        return stacked

    def tile_field_to_lines(
        self,
        field_name: str,
        points: ArrayLike,
        indices_for_species: dict[str, Union[list[int], jnp.ndarray]],
        *,
        LoS_directions: ArrayLike | None = None,
    ) -> ArrayLike:
        """
        Generic method to sample any species-level field and tile to line-level.
        Works for both regular fields and special LoS_velocity.
        
        Args:
            field_name: Name of the field to sample ("temperature", "collisional", etc.)
            points: Quadrivectors (a, x, y, z) of shape (*batch_shape, 4)
            indices_for_species: Mapping {species: [line_idx1, line_idx2, ...]}
            LoS_directions: Required ONLY for "LoS_velocity" field
        
        Returns:
            tiled_field: (N_lines, *batch_shape)
        """
        # 1) Sample the species-level field
        if field_name == "LoS_velocity":
            if LoS_directions is None:
                raise ValueError("LoS_directions required for field 'LoS_velocity'")
            species_field = self.LoS_velocity(points, LoS_directions)
        else:
            # Use the dynamic field sampler we already have
            field_sampler = self._make_field_sampler(field_name)
            species_field = field_sampler(points)
        
        # 2) Tile to line-level using the internal helper
        return self._tile_field_to_lines(species_field, indices_for_species)

    def _tile_field_to_lines(
        self,
        species_field: ArrayLike,
        indices_for_species: dict[str, Union[list[int], jnp.ndarray]]
    ) -> ArrayLike:
        """
        Core tiling logic: (N_species, *batch_shape) → (N_lines, *batch_shape)
        """
        # Determine total number of lines from mapping
        all_indices = []
        for species_name in self.species_keys:
            all_indices.extend(indices_for_species[species_name])
        N_lines = max(all_indices) + 1
        
        # Initialize tiled array
        batch_shape = species_field.shape[1:]
        tiled = jnp.zeros((N_lines,) + batch_shape, dtype=species_field.dtype)
        
        # Map species indices to line indices
        for i_species, species_name in enumerate(self.species_keys):
            line_indices = jnp.asarray(indices_for_species[species_name])
            tiled = tiled.at[line_indices].set(species_field[i_species])
        
        return tiled