"""Block B, simulation extraction: read hydro (or FGPA-painted N-body) snapshots and turn them into tau inputs.

Layers: ``readers`` (HDF5 I/O, NumPy) -> ``gas_cells`` (pytree) -> ``gas_physics`` (n, T, ion fractions; JAX) ->
``selection`` (k-d tree, NumPy) -> ``deposition`` (pieces of every dual segment) -> ``absorbers`` (``core.LoSSamples``
and tau through the core kernel) -> ``snapshot_skewers`` (``SimulationSkewers``: the whole chain for any options);
``maps`` projects slabs of the gas for figures. Depends on ``skewerjax.core`` only; the boundary with the core is
``core.LoSSamples`` / ``core.Fields``.
"""
from skewerjax.simulations.readers import ArepoHDF5Reader, SnapshotHeader
from skewerjax.simulations.gas_cells import GasCells
from skewerjax.simulations.gas_physics import HI, CloudyIonTable, Ion, ion_from_species, ion_properties
from skewerjax.simulations.selection import select_cells_near_paths
from skewerjax.simulations.deposition import CellCrossings, SegmentComponents, cell_crossings, tophat_segments, voronoi_segments
from skewerjax.simulations.absorbers import (Absorbers, ExactFrame, SnapshotFrame, absorber_samples, absorbers_from_crossings, sampling_check, tau_absorbers,
                                            tau_by_path_bins)
from skewerjax.simulations.maps import slab_projection
from skewerjax.simulations.snapshot_skewers import ExtractionOptions, IntegrationOptions, SimulationSkewers, cached

__all__ = [n for n in dir() if not n.startswith("_")]
