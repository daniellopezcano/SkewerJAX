"""Block B, simulation extraction: read hydro (or FGPA-painted N-body) snapshots and turn them into the gas cells of the core.

Layers: ``readers`` (HDF5 I/O, NumPy) -> ``gas_cells`` (``GasCells``: the raw per-cell data of the snapshot) ->
``gas_physics`` (n, T, ion fractions; JAX) -> ``selection`` (the cells within reach of the skewers, streamed from disk)
-> ``core.Cells`` (``SimulationSkewers.cells``: positions, volumes and, per ion, the comoving density, temperature and
peculiar velocity: the same container as the toy models) -> the core's route (deposition, ``core.absorbers``,
``core.compute_optical_depth``). ``snapshot_skewers`` (``SimulationSkewers``) runs the chain for any options; ``maps``
projects slabs of the gas for figures. Depends on ``skewerjax.core`` only; the boundary with the core is
``core.Cells``.
"""
from skewerjax.simulations.readers import ArepoHDF5Reader, SnapshotHeader
from skewerjax.simulations.gas_cells import GasCells
from skewerjax.simulations.gas_physics import HI, CloudyIonTable, Ion, ion_from_species, ion_properties
from skewerjax.simulations.selection import select_cells_near_paths
from skewerjax.simulations.maps import slab_projection
from skewerjax.simulations.snapshot_skewers import ExtractionOptions, IntegrationOptions, SimulationSkewers, cached

__all__ = [n for n in dir() if not n.startswith("_")]
