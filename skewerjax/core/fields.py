"""Field samplers: the gas fields of every species, evaluated at events (a, x, y, z) of the skewers."""
from __future__ import annotations

from typing import Callable, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike


def _static(fn) -> bool:
    """True for a plain callable (not a pytree): it is carried as static data, so closures pass through jit."""
    return jax.tree_util.all_leaves([fn])


@jax.tree_util.register_pytree_node_class
class ScalarField:
    """A scalar field: ``sampler(x)`` maps events x (..., 4) = (a, x, y, z) to values (...)."""

    def __init__(self, sampler: Callable[[ArrayLike], ArrayLike]):
        self._sampler = sampler

    def __call__(self, x: ArrayLike) -> ArrayLike:
        return self._sampler(x)

    def tree_flatten(self):
        return ((), (self._sampler,)) if _static(self._sampler) else ((self._sampler,), None)

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        obj._sampler = aux[0] if aux is not None else children[0]
        return obj


@jax.tree_util.register_pytree_node_class
class VectorField(ScalarField):
    """A vector field (e.g. the peculiar velocity [km/s]): ``sampler(x)`` returns (..., 3)."""

    def project(self, x: ArrayLike, directions: ArrayLike, tol: float = 1e-5) -> ArrayLike:
        """Component along unit ``directions`` (..., 3): sum(v * d, -1). Checks |d| = 1 to ``tol`` (eager only)."""
        v = self(x)
        if not isinstance(directions, jax.core.Tracer):
            max_dev = float(jnp.max(jnp.abs(jnp.linalg.norm(directions, axis=-1) - 1.0)))
            if max_dev > tol:
                raise ValueError(f"LoS directions must be normalized (‖d‖ ≈ 1). Max |‖d‖ - 1| = {max_dev:.3e} > tol = {tol:.3e}.")
        return jnp.sum(v * directions, axis=-1)


@jax.tree_util.register_pytree_node_class
class SampledField:
    """Field given by its values at the points of a ``SkewerBundle`` (e.g. deposited from a simulation).

    values : (N_sk, N_ppsk) scalar or (N_sk, N_ppsk, 3) vector values. Called with the bundle's events
    (any batch shape with N_sk * N_ppsk points), it returns the values reshaped to that batch.
    """

    def __init__(self, values: ArrayLike):
        self.values = jnp.asarray(values)
        if self.values.ndim < 2:
            raise ValueError(f"values must be (N_sk, N_ppsk[, 3]), got {self.values.shape}.")

    def __call__(self, x: ArrayLike) -> ArrayLike:
        batch = tuple(x.shape[:-1])
        n_points = int(np.prod(self.values.shape[:2]))
        if int(np.prod(batch)) != n_points:
            raise ValueError(f"SampledField holds {n_points} points, called with batch {batch}.")
        return self.values.reshape(batch + tuple(self.values.shape[2:]))

    def tree_flatten(self):
        return (self.values,), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        obj.values = children[0]
        return obj


@jax.tree_util.register_pytree_node_class
class Fields:
    """The fields of every absorbing species.

    ``Fields({"HI": {"temperature": ScalarField(...), "velocity": VectorField(...), ...}, "SiIII": {...}})``

    The optical depth needs, per species: ``temperature`` [K], ``velocity`` [km/s] (vector),
    ``number_density`` [comoving cm^-3], ``turbulence`` [m/s] and ``collisional`` [GHz].

    Access: ``fields["HI"]["temperature"](x)``; across species ``fields.temperature(x)`` -> (N_species, ...),
    ``fields.LoS_velocity(x, directions)``; everything at once ``fields.sample(x, directions)``.
    JAX pytree (the samplers are children; species and field order are static).
    """

    def __init__(self, species_fields: dict):
        self._species = {sp: dict(f) for sp, f in species_fields.items()}

    def tree_flatten(self):
        keys = tuple((sp, tuple(f)) for sp, f in self._species.items())
        return tuple(self._species[sp][n] for sp, names in keys for n in names), keys

    @classmethod
    def tree_unflatten(cls, keys, children):
        obj = object.__new__(cls)
        it = iter(children)
        obj._species = {sp: {n: next(it) for n in names} for sp, names in keys}
        return obj

    def __getitem__(self, species: str) -> dict:
        return self._species[species]

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        if name == "LoS_velocity":
            return lambda x, directions: self._stack(lambda f: f["velocity"].project(x, directions), name)
        if any(name in f for f in self._species.values()):
            return lambda x: self._stack(lambda f: f[name](x), name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'. "
                             f"Available species: {list(self._species)}")

    @property
    def species_keys(self) -> tuple:
        return tuple(self._species)

    @property
    def field_keys(self) -> dict:
        return {sp: tuple(f) for sp, f in self._species.items()}

    def _stack(self, get, name) -> jnp.ndarray:
        """(N_species, ...) of ``get(species_fields)`` for every species (same shape required)."""
        if not self._species:
            raise ValueError("No species defined in Fields.")
        values = []
        for sp, f in self._species.items():
            if name != "LoS_velocity" and name not in f:
                raise AttributeError(f"Field '{name}' not found in species '{sp}'. "
                                     "All species must carry this field for cross-species sampling.")
            if name == "LoS_velocity" and "velocity" not in f:
                raise AttributeError("Cannot project LoS_velocity: no 'velocity' field defined.")
            v = jnp.asarray(get(f))
            if values and v.shape != values[0].shape:
                raise ValueError(f"Inconsistent shape for field '{name}' across species. "
                                 f"Species '{sp}' has shape {v.shape}, expected {values[0].shape}.")
            values.append(v)
        return jnp.stack(values, axis=0)

    def sample(self, x: ArrayLike, LoS_directions: ArrayLike, as_array: bool = False):
        """Every field of every species at events x (..., 4), plus ``LoS_velocity`` where there is a velocity.

        Returns {species: {field: array}}, or with ``as_array`` one (N_species, N_scalar_fields, ...) array of
        the scalar fields (sorted by name; vector fields left out, their LoS projection included).
        """
        res = {}
        for sp, f in self._species.items():
            d = {n: field(x) for n, field in f.items()}
            if "velocity" in f:
                d["LoS_velocity"] = f["velocity"].project(x, LoS_directions)
            res[sp] = d
        if not as_array:
            return res
        batch = x.shape[:-1]
        if not res:
            raise ValueError("No species defined in Fields.")
        ref = next(iter(res))
        names = sorted(n for n, v in res[ref].items() if jnp.shape(v) == batch)
        if not names:
            raise ValueError(f"No scalar fields found for species '{ref}'. Cannot build stacked array.")
        for sp, d in res.items():
            for n in names:
                if n not in d or jnp.shape(d[n]) != batch:
                    raise ValueError(f"Scalar field '{n}' missing in species '{sp}' or not of batch shape {batch}.")
        return jnp.stack([jnp.stack([jnp.asarray(d[n]) for n in names], axis=0) for d in res.values()], axis=0)

    def tile_field_to_lines(self, field_name: str, quadri_vectors: ArrayLike,
                            indices_for_species: dict[str, Union[list[int], jnp.ndarray]], *,
                            LoS_directions: ArrayLike | None = None) -> jnp.ndarray:
        """(N_lines, ...) a species-level field copied to every line of that species (``LineBundle.indices_for_species``)."""
        if field_name == "LoS_velocity":
            if LoS_directions is None:
                raise ValueError("LoS_directions required for field 'LoS_velocity'")
            per_species = self.LoS_velocity(quadri_vectors, LoS_directions)
        else:
            per_species = self._stack(lambda f: f[field_name](quadri_vectors), field_name)
        n_lines = max(i for sp in self.species_keys for i in indices_for_species[sp]) + 1
        tiled = jnp.zeros((n_lines,) + per_species.shape[1:], dtype=per_species.dtype)
        for i, sp in enumerate(self.species_keys):
            tiled = tiled.at[jnp.asarray(indices_for_species[sp])].set(per_species[i])
        return tiled
