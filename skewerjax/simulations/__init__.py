"""Block B, the field sources: everything that produces the gas cells of the core (``core.Cells``).

Toy fields: ``grf`` (``PowerLawGRF``, Gaussian random fields with optional anisotropy and BAO) and ``toy_cells``
(``grf_cells``; uniform and density-weighted generating points).

Snapshots: read hydro (or FGPA-painted N-body) snapshots and turn them into the gas cells of the core. Layers:
``readers`` (HDF5 I/O, NumPy) -> ``gas_cells`` (``GasCells``: the raw per-cell data of the snapshot) -> ``gas_physics``
(n, T, ion fractions; JAX) -> ``core.Cells`` (positions, volumes and, per ion, the comoving density, temperature and
peculiar velocity: the same container as the toy fields), either for a region of the box (``snapshot_cells``, no
skewers) or for the cells within reach of a set of skewers (``selection``, streamed from disk;
``SimulationSkewers.to_cells``). ``snapshot_skewers`` (``SimulationSkewers``) then runs the core's route (deposition,
``core.absorbers``, ``core.compute_optical_depth``) for any options; ``maps`` projects slabs of the gas for figures.

Depends on ``skewerjax.core`` only; the boundary with the core is ``core.Cells``.
"""
from skewerjax.simulations.grf import PowerLawGRF
from skewerjax.simulations.toy_cells import density_weighted_generators, grf_cells, interpolate_periodic, uniform_generators
from skewerjax.simulations.readers import ArepoHDF5Reader, SnapshotHeader
from skewerjax.simulations.gas_cells import GasCells
from skewerjax.simulations.gas_physics import HI, CloudyIonTable, Ion, ion_from_species, ion_properties
from skewerjax.simulations.selection import select_cells_near_paths
from skewerjax.simulations.maps import slab_projection
from skewerjax.simulations.snapshot_skewers import ExtractionOptions, IntegrationOptions, SimulationSkewers, cached
from skewerjax.simulations.snapshot_cells import gas_to_cells, select_cells_in_region, snapshot_cells

__all__ = [n for n in dir() if not n.startswith("_")]
