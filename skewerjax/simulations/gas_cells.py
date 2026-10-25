"""Gas cells (layer 2): a format-independent pytree of gas elements in physical units.

:class:`GasCells` holds what every hydro (or FGPA-painted N-body) output reduces to: a position,
a volume (hence a size) and a dictionary of per-cell fields, in SkewerJAX's units:

* positions [comoving Mpc, no h], wrapped into the periodic box [0, box);
* volumes [comoving Mpc^3];
* fields: any arrays with leading dimension N (e.g. ``rho`` [g cm^-3, physical], ``u``
  [erg/g], ``velocity`` (N, 3) [km/s, physical peculiar], ``metals`` (N, 9), ...).

It is a JAX pytree (fields and positions are leaves; ``box`` and ``a`` are static metadata), so
it can be passed through ``jit``, ``vmap`` and ``tree_map``; it is format independent: readers
(layer 1) produce it and the physics (layer 3) consumes it. After the physics, ``SimulationSkewers.cells``
turns it into ``core.Cells``, the container of the gas shared with the toy models (per ion: comoving density,
temperature, peculiar velocity), which the deposition and the optical depth take.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from jax import tree_util


@tree_util.register_pytree_node_class
class GasCells:
    """Gas elements (cells or particles) with positions, volumes and fields.

    Parameters
    ----------
    positions : (N, 3) comoving Mpc (no h), in [0, box).
    volumes : (N,) comoving Mpc^3.
    fields : dict of arrays with leading dimension N, physical units (see module docstring).
    box : periodic box side [cMpc]; None for non-periodic data (light cones).
    a : scale factor of the data (snapshot epoch).
    ids : optional (N,) identifiers (e.g. file/row or ParticleIDs).
    """

    def __init__(self, positions, volumes, fields: dict, box: Optional[float], a: float, ids=None):
        self.positions = positions
        self.volumes = volumes
        self.fields = dict(fields)
        self.box = None if box is None else float(box)
        self.a = float(a)
        self.ids = ids
        n = np.shape(positions)[0]
        if np.shape(positions) != (n, 3):
            raise ValueError(f"positions must be (N, 3), got {np.shape(positions)}.")
        if np.shape(volumes) != (n,):
            raise ValueError(f"volumes must be (N,), got {np.shape(volumes)}.")
        for k, v in self.fields.items():
            if np.shape(v)[:1] != (n,):
                raise ValueError(f"field {k!r} has leading shape {np.shape(v)[:1]}, expected ({n},).")

    # -- pytree protocol
    def tree_flatten(self):
        keys = tuple(sorted(self.fields))
        return (self.positions, self.volumes, tuple(self.fields[k] for k in keys), self.ids), (keys, self.box, self.a)

    @classmethod
    def tree_unflatten(cls, aux, children):
        keys, box, a = aux
        obj = object.__new__(cls)
        obj.positions, obj.volumes, vals, obj.ids = children
        obj.fields = dict(zip(keys, vals))
        obj.box, obj.a = box, a
        return obj

    # -- basic API
    def __len__(self) -> int:
        return int(np.shape(self.positions)[0])

    @property
    def size_cMpc(self):
        """Cube side V^(1/3) [cMpc]; fake_spectra's top-hat radius h = (m/rho)^(1/3)
        (``abstractsnapshot.get_smooth_length`` L256-L284)."""
        return np.cbrt(np.asarray(self.volumes))

    def take(self, idx) -> "GasCells":
        """Subset (numpy fancy index)."""
        idx = np.asarray(idx)
        return GasCells(np.asarray(self.positions)[idx], np.asarray(self.volumes)[idx],
                        {k: np.asarray(v)[idx] for k, v in self.fields.items()}, self.box, self.a,
                        None if self.ids is None else np.asarray(self.ids)[idx])

    @staticmethod
    def concatenate(parts) -> "GasCells":
        """Concatenate containers with the same fields, box and epoch."""
        parts = [p for p in parts if len(p)]
        if not parts:
            raise ValueError("nothing to concatenate (all parts are empty).")
        keys = set(parts[0].fields)
        if any(set(p.fields) != keys or p.box != parts[0].box or p.a != parts[0].a for p in parts):
            raise ValueError("parts differ in fields, box or epoch.")
        ids = None if any(p.ids is None for p in parts) else np.concatenate([np.asarray(p.ids) for p in parts])
        return GasCells(np.concatenate([np.asarray(p.positions) for p in parts]),
                        np.concatenate([np.asarray(p.volumes) for p in parts]),
                        {k: np.concatenate([np.asarray(p.fields[k]) for p in parts]) for k in keys},
                        parts[0].box, parts[0].a, ids)

    def nbytes(self) -> int:
        """Host memory of the arrays [bytes]."""
        leaves = tree_util.tree_leaves(self)
        return int(sum(np.asarray(x).nbytes for x in leaves))
