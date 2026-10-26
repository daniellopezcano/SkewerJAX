"""The snapshot side of notebook 05 on TNG50-4 snapshot 25 (read-only; skipped when ``$SKEWERJAX_DATA/TNG50-4`` is
missing): the reader (header, units, the streamed blocks), the selection streamed from disk
(``select_cells_near_paths``) against the generic in-memory ``core.select_cells`` on the cells of a region
(``select_cells_in_region``): the same cells, and complete; and the gas cells handed to the core
(``SimulationSkewers.to_cells`` = ``gas_to_cells``). The whole-snapshot checks (selection against a brute-force search
over 19.3 M cells, deposition, columns): ``scripts/validation/check_tng_fields.py`` (``SKEWERJAX_SLOW=1``)."""
from functools import lru_cache

import numpy as np
import pytest
from scipy.spatial import cKDTree

from skewerjax.core import select_cells
from skewerjax.simulations import (ArepoHDF5Reader, SimulationSkewers, gas_to_cells, select_cells_in_region,
                                   select_cells_near_paths)
from skewerjax.utils.paths import snapshot_dir

TNG = snapshot_dir("TNG50-4")
needs_tng = pytest.mark.skipif(not (TNG / "snapdir_025").is_dir(), reason=f"needs TNG50-4 snapshot 25 in {TNG}")


@lru_cache(maxsize=None)
def reader():
    return ArepoHDF5Reader(TNG, 25)


@lru_cache(maxsize=None)
def near():
    """The cells within reach of ``path()``, streamed from the snapshot (one pass, shared by the tests)."""
    return select_cells_near_paths(reader(), path())


def path():
    """(1, 120, 3) a curved 3 cMpc skewer through the halo region of notebook 05 [cMpc]."""
    t = np.linspace(0.0, 1.0, 120)
    d = np.array([1.0, 0.6, 0.4]) / np.linalg.norm([1.0, 0.6, 0.4])
    return (np.array([8.5, 36.0, 34.0]) + 3.0 * t[:, None] * d + 0.4 * np.sin(2 * np.pi * t)[:, None] * np.array([0, 0, 1.0]))[None]


@needs_tng
def test_reader_header_and_units():
    rd = reader()
    H = rd.header
    assert abs(H.redshift - 3.0) < 0.01 and np.isclose(H.a, 1 / (1 + H.redshift)) and H.n_gas > 19_000_000
    assert np.isclose(H.box_cMpc, H.box_code * H.code_length_cMpc) and np.isclose(H.box_cMpc, 35.0 / H.h, rtol=1e-3)
    assert np.isclose(H.vmax_kms, H.a * H.hubble_kms_mpc * H.box_cMpc, rtol=1e-12)
    raw = rd.read_rows(0, ["Coordinates", "Masses", "Density", "Velocities", "GFM_Metals", "InternalEnergy", "ElectronAbundance"],
                       start=0, stop=1000)
    g = rd.to_gas_cells(raw, ids=np.arange(1000))
    assert len(g) == 1000 and np.all((g.positions >= 0) & (g.positions < H.box_cMpc)) and g.a == H.a
    np.testing.assert_allclose(g.volumes, raw["Masses"] / raw["Density"] * H.code_length_cMpc ** 3, rtol=1e-6)
    np.testing.assert_allclose(g.fields["velocity"], raw["Velocities"] * np.sqrt(H.a), rtol=1e-6)    # peculiar [km/s]
    np.testing.assert_allclose(g.fields["rho"], raw["Density"] * H.density_cgs, rtol=1e-6)
    assert np.all(g.fields["metals"].sum(1) <= 1 + 1e-6) and 0.7 < np.median(g.fields["metals"][:, 0]) < 0.77
    with pytest.raises(KeyError):
        rd.read_rows(0, ["NoSuchDataset"], stop=10)
    rows = [(fi, r0, len(b["Masses"])) for fi, r0, b in rd.iter_blocks(["Masses"], max_rows=3_000_000)]
    assert sum(n for *_, n in rows) == H.n_gas and max(n for *_, n in rows) <= 3_000_000


@needs_tng
def test_streamed_selection_is_the_generic_rule():
    """Step 4 of notebook 05: the streamed selection keeps exactly the cells that ``core.select_cells`` keeps from the
    cells of a region around the path (in memory), and every cell the path crosses."""
    rd, p, sel = reader(), path(), near()
    lo, hi = p[0].min(0), p[0].max(0)
    region = select_cells_in_region(rd, lo, hi, margin=3.0)                 # 3 cMpc > 2.5 x the largest cells here
    assert np.isin(sel.ids, region.ids).all()
    generic = select_cells(region, p)
    assert np.array_equal(np.sort(generic.ids), np.sort(sel.ids))
    fine = (p[0, :-1, None] + np.linspace(0, 1, 5)[None, :, None] * np.diff(p[0], axis=0)[:, None]).reshape(-1, 3)
    owner = region.ids[cKDTree(region.positions, boxsize=rd.header.box_cMpc).query(np.mod(fine, rd.header.box_cMpc))[1]]
    assert np.isin(owner, sel.ids).all()                                    # every crossed cell is selected
    assert np.isin(select_cells(region, p, mode="tophat").ids, sel.ids).all()


@needs_tng
def test_skewers_to_cells_is_gas_to_cells():
    """Step 5: ``SimulationSkewers.to_cells`` (the cells near the skewers) and ``gas_to_cells`` (any raw cells) give the
    same ``core.Cells``, the container of the toy models."""
    sk = SimulationSkewers(reader(), path(), cells=near())
    a = sk.to_cells(["HI", "SiIII"])
    b = gas_to_cells(sk.cells, ["HI", "SiIII"])
    assert a.species_keys == b.species_keys == ("HI", "SiIII") and np.array_equal(a.ids, b.ids) and a.box == b.box
    np.testing.assert_array_equal(a.positions, b.positions)
    np.testing.assert_array_equal(a.volumes, b.volumes)
    for sp in a.species_keys:
        for f in ("number_density", "temperature", "velocity"):
            np.testing.assert_array_equal(a.species[sp][f], b.species[sp][f])
    assert np.all(a.species["HI"]["number_density"] > 0) and np.all(a.species["HI"]["temperature"] > 1e3)
