"""The snapshot as gas cells, without skewers: the cells of a region of the box after the gas physics, as ``core.Cells``.

A field source like the toy fields of ``toy_cells``: ``snapshot_cells`` streams the snapshot (layer 1), keeps the cells
whose generating points lie in a periodic sub-box (``select_cells_in_region``), applies the gas physics (layer 3,
``gas_physics.ion_properties``) and returns ``core.Cells`` with the per-species fields of ``SimulationSkewers.to_cells``
(the same values for the same cells). No skewer is involved; the selection of the cells within reach of a set of skewers
(``selection.select_cells_near_paths``) belongs to the deposition.

Cost (TNG50-4, 19.3 M cells): one pass over the positions of the whole snapshot (24 bytes per cell, streamed in blocks
of ``max_rows``; the region test runs axis by axis on the cells still inside), then the other datasets of the kept rows
only; memory is one block plus the kept cells.
"""
from __future__ import annotations

import numpy as np

from skewerjax.core.cells import Cells
from skewerjax.simulations import gas_physics as gp
from skewerjax.simulations.gas_cells import GasCells
from skewerjax.simulations.selection import read_selected
from skewerjax.simulations.snapshot_skewers import ExtractionOptions, ion_cell_fields


def _region(lo, hi, margin: float, box: float):
    lo = np.broadcast_to(np.asarray(lo, dtype=np.float64), (3,))
    hi = np.broadcast_to(np.asarray(hi, dtype=np.float64), (3,))
    width = hi - lo + 2.0 * margin
    if np.any(hi <= lo) or margin < 0.0:
        raise ValueError(f"the region needs hi > lo and margin >= 0, got lo = {lo}, hi = {hi}, margin = {margin}.")
    if np.any(width > box):
        raise ValueError(f"the region with its margin ({width} cMpc) is wider than the box ({box:.4f} cMpc).")
    return lo - margin, width


def select_cells_in_region(reader, lo, hi, *, margin: float = 0.0, max_rows: int = 4_000_000) -> GasCells:
    """The cells of a snapshot whose generating points lie in the periodic sub-box [lo - margin, hi + margin) [cMpc].

    lo, hi : (3,) or scalars, the corners of the region [cMpc] (it may cross the box faces: positions are taken modulo
        the box, and a cell is kept if (x - lo + margin) mod box < hi - lo + 2 margin on every axis).
    margin : extra width around the region [cMpc], e.g. a few cell sizes, so that the nearest generating point of every
        point of the region is among the kept cells (the cells of the region are then complete).
    max_rows : cells per streamed block (memory bound).

    Returns ``GasCells`` in physical units (``reader.to_gas_cells``), ids = (file_index << 40) | row, as
    ``select_cells_near_paths``.
    """
    from skewerjax.simulations.readers import GAS_DATASETS
    box = reader.header.box_cMpc
    origin, width = _region(lo, hi, margin, box)
    parts, n_sel = [], 0
    rest = [k for k in GAS_DATASETS if k != "Coordinates"]
    order = np.argsort(width, kind="stable")                         # the narrowest axis first: most cells leave at once
    for fi, r0, base in reader.iter_blocks(("Coordinates",), max_rows=max_rows):
        c = base["Coordinates"]
        sel = np.arange(c.shape[0])
        for ax in order:                                              # the same test per axis, on the survivors only
            x = reader.positions_cMpc(c[sel, ax])
            sel = sel[np.mod(x - origin[ax], box) < width[ax]]
        if sel.size:
            parts.append(read_selected(reader, fi, r0, sel, base, rest))
            n_sel += sel.size
    if n_sel == 0:
        raise RuntimeError("no gas cell in the region.")
    return GasCells.concatenate(parts)


def gas_to_cells(gas: GasCells, species, ext: ExtractionOptions = ExtractionOptions()) -> Cells:
    """``core.Cells`` of raw snapshot cells after the gas physics, per species (as ``SimulationSkewers.to_cells``).

    species : a species key (e.g. "HI"), an ``Ion``, or a sequence of them. Per species: ``number_density`` = n_ion a^3
    [comoving cm^-3], ``temperature`` [K] and ``velocity`` [km/s, physical peculiar]; positions [cMpc], volumes
    [cMpc^3], the periodic box and the snapshot ids of the cells. ``ext``: the gas-physics options of
    ``ExtractionOptions`` (mass unit, T constants, x_HI model, CLOUDY density and interpolation).
    """
    items = [species] if isinstance(species, (str, gp.Ion)) else list(species)
    ions = [x if isinstance(x, gp.Ion) else gp.ion_from_species(x) for x in items]
    a = float(gas.a)
    per, cloudy = {}, None
    for ion in ions:
        if not ion.is_HI and cloudy is None:
            cloudy = gp.CloudyIonTable(1.0 / a - 1.0)
        P = gp.ion_properties(gas, ion, mass_unit=ext.mass_unit, T_constants=ext.T_constants, x_HI_model=ext.x_HI_model, cloudy=cloudy,
                              cloudy_density=ext.cloudy_density, cloudy_interpolation=ext.cloudy_interpolation)
        per[ion.species] = ion_cell_fields(P["n_ion"], P["T"], gas.fields["velocity"], a)
    return Cells(np.asarray(gas.positions, dtype=np.float64), np.asarray(gas.volumes, dtype=np.float64), per, box=gas.box, ids=gas.ids)


def snapshot_cells(reader, species, lo, hi, *, margin: float = 0.0, ext: ExtractionOptions = ExtractionOptions(),
                   max_rows: int = 4_000_000) -> Cells:
    """The gas cells of a sub-box of a snapshot as ``core.Cells``, for the given species, without skewers.

    ``select_cells_in_region(reader, lo, hi, margin=margin)`` then ``gas_to_cells(cells, species, ext)``; see both.
    """
    return gas_to_cells(select_cells_in_region(reader, lo, hi, margin=margin, max_rows=max_rows), species, ext)
