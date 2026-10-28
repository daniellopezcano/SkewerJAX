"""Block B, the field sources: everything that produces the gas cells of the core (``core.Cells``).

Toy fields: ``grf`` (``PowerLawGRF``, Gaussian random fields with optional anisotropy and BAO), ``toy_cells``
(``grf_cells``: the realisation on the cells of its grid, cubes centred on the nodes) and ``toy_boxes`` (``ToyBoxes``: many
realisations regenerated from their seeds, their gas, skewers through them and their statistics, streamed and cached;
notebook 08).

Snapshots: read hydro (or FGPA-painted N-body) snapshots and turn them into the gas cells of the core. Layers:
``readers`` (HDF5 I/O, NumPy) -> ``gas_cells`` (``GasCells``: the raw per-cell data of the snapshot) -> ``gas_physics``
(n, T, ion fractions; JAX) -> ``core.Cells`` (positions, volumes and, per ion, the comoving density, temperature and
peculiar velocity: the same container as the toy fields), either for a region of the box (``snapshot_cells``, no
skewers) or for the cells within reach of a set of skewers (``selection``, streamed from disk;
``SimulationSkewers.to_cells``). ``snapshot_skewers`` (``SimulationSkewers``) then runs the core's route (deposition,
``core.absorbers``, ``core.compute_optical_depth``) for any options; ``maps`` projects slabs of the gas for figures.

Depends on ``skewerjax.core`` only; the boundary with the core is ``core.Cells``.
"""

from skewerjax.simulations.gas_cells import GasCells
from skewerjax.simulations.gas_physics import HI, CloudyIonTable, Ion, ion_from_species, ion_properties
from skewerjax.simulations.grf import PowerLawGRF
from skewerjax.simulations.maps import slab_projection
from skewerjax.simulations.readers import ArepoHDF5Reader, SnapshotHeader
from skewerjax.simulations.selection import select_cells_near_paths
from skewerjax.simulations.snapshot_cells import gas_to_cells, select_cells_in_region, snapshot_cells
from skewerjax.simulations.snapshot_skewers import ExtractionOptions, IntegrationOptions, SimulationSkewers, cached
from skewerjax.simulations.toy_boxes import (
    SkewerRun,
    ToyBoxes,
    field_statistics,
    hubble_velocity,
    load_skewer_run,
    run_skewers,
    skewer_box,
    xi_chi2,
    xi_subset_offsets,
)
from skewerjax.simulations.toy_cells import grf_cells, interpolate_periodic

__all__ = [n for n in dir() if not n.startswith("_")]
