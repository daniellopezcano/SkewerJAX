"""The snapshot side of notebook 05 on TNG50-4 snapshot 25 (read-only; ``tng`` marker: skipped without
``$SKEWERJAX_DATA/TNG50-4``): the reader (header, units, the streamed blocks), the selection streamed from disk
(``select_cells_near_paths``) against the generic in-memory ``core.select_cells`` on the cells of a region
(``select_cells_in_region``): the same cells, and complete; the gas cells handed to the core
(``SimulationSkewers.to_cells`` = ``gas_to_cells``); and the field extraction over the whole snapshot (formerly
``scripts/validation/check_tng_fields.py``, same tolerances): per-cell H I physics, selection completeness against a
brute-force nearest-generator search over 19.3 M cells, the Voronoi deposition (exact pieces, a column independent of
the spacing), the pieces handed to the core and the cells of a sub-box without skewers."""

from functools import cache

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.spatial import cKDTree

from skewerjax.comparison import SKEWER_SETS
from skewerjax.core import absorber_pieces, axis_aligned_skewer_points, path_geometry, phys, select_cells
from skewerjax.core import deposition as dep
from skewerjax.simulations import (
    HI,
    ArepoHDF5Reader,
    Ion,
    SimulationSkewers,
    gas_to_cells,
    select_cells_in_region,
    select_cells_near_paths,
    snapshot_cells,
)
from skewerjax.simulations import gas_physics as gp
from skewerjax.utils.paths import snapshot_dir

TNG = snapshot_dir("TNG50-4")


@cache
def reader():
    return ArepoHDF5Reader(TNG, 25)


@cache
def near():
    """The cells within reach of ``path()``, streamed from the snapshot (one pass, shared by the tests)."""
    return select_cells_near_paths(reader(), path())


def path():
    """(1, 120, 3) a curved 3 cMpc skewer through the halo region of notebook 05 [cMpc]."""
    t = np.linspace(0.0, 1.0, 120)
    d = np.array([1.0, 0.6, 0.4]) / np.linalg.norm([1.0, 0.6, 0.4])
    return (np.array([8.5, 36.0, 34.0]) + 3.0 * t[:, None] * d + 0.4 * np.sin(2 * np.pi * t)[:, None] * np.array([0, 0, 1.0]))[
        None
    ]


@pytest.mark.tng
def test_reader_header_and_units():
    rd = reader()
    H = rd.header
    assert abs(H.redshift - 3.0) < 0.01 and np.isclose(H.a, 1 / (1 + H.redshift)) and H.n_gas > 19_000_000
    assert np.isclose(H.box_cMpc, H.box_code * H.code_length_cMpc) and np.isclose(H.box_cMpc, 35.0 / H.h, rtol=1e-3)
    assert np.isclose(H.vmax_kms, H.a * H.hubble_kms_mpc * H.box_cMpc, rtol=1e-12)
    raw = rd.read_rows(
        0,
        ["Coordinates", "Masses", "Density", "Velocities", "GFM_Metals", "InternalEnergy", "ElectronAbundance"],
        start=0,
        stop=1000,
    )
    g = rd.to_gas_cells(raw, ids=np.arange(1000))
    assert len(g) == 1000 and np.all((g.positions >= 0) & (g.positions < H.box_cMpc)) and g.a == H.a
    np.testing.assert_allclose(g.volumes, raw["Masses"] / raw["Density"] * H.code_length_cMpc**3, rtol=1e-6)
    np.testing.assert_allclose(g.fields["velocity"], raw["Velocities"] * np.sqrt(H.a), rtol=1e-6)  # peculiar [km/s]
    np.testing.assert_allclose(g.fields["rho"], raw["Density"] * H.density_cgs, rtol=1e-6)
    assert np.all(g.fields["metals"].sum(1) <= 1 + 1e-6) and 0.7 < np.median(g.fields["metals"][:, 0]) < 0.77
    with pytest.raises(KeyError):
        rd.read_rows(0, ["NoSuchDataset"], stop=10)
    rows = [(fi, r0, len(b["Masses"])) for fi, r0, b in rd.iter_blocks(["Masses"], max_rows=3_000_000)]
    assert sum(n for *_, n in rows) == H.n_gas and max(n for *_, n in rows) <= 3_000_000


@pytest.mark.tng
def test_streamed_selection_is_the_generic_rule():
    """Step 4 of notebook 05: the streamed selection keeps exactly the cells that ``core.select_cells`` keeps from the
    cells of a region around the path (in memory), and every cell the path crosses."""
    rd, p, sel = reader(), path(), near()
    lo, hi = p[0].min(0), p[0].max(0)
    region = select_cells_in_region(rd, lo, hi, margin=3.0)  # 3 cMpc > 2.5 x the largest cells here
    assert np.isin(sel.ids, region.ids).all()
    generic = select_cells(region, p)
    assert np.array_equal(np.sort(generic.ids), np.sort(sel.ids))
    fine = (p[0, :-1, None] + np.linspace(0, 1, 5)[None, :, None] * np.diff(p[0], axis=0)[:, None]).reshape(-1, 3)
    owner = region.ids[cKDTree(region.positions, boxsize=rd.header.box_cMpc).query(np.mod(fine, rd.header.box_cMpc))[1]]
    assert np.isin(owner, sel.ids).all()  # every crossed cell is selected
    assert np.isin(select_cells(region, p, mode="tophat").ids, sel.ids).all()


@pytest.mark.tng
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


# --------------------------------------------------------------------------- field extraction over the whole snapshot
def tng_field_checks(quick: bool):
    """Every check of ``test_tng_fields`` (quick: one skewer, coarser spacing scan, first 10^6 cells for the physics) [(name, value, tolerance, expect)]."""
    checks = []

    def check(name, value, tol, expect=False, detail=""):
        checks.append((name, float(value), tol, expect))

    rd = ArepoHDF5Reader(TNG, 25)
    H, L = rd.header, rd.header.box_cMpc
    # 1. per-cell physics: TNG's stored x_HI is its own ionisation equilibrium in diffuse gas
    raw = rd.read_rows(
        0,
        [
            "Coordinates",
            "Masses",
            "Density",
            "InternalEnergy",
            "ElectronAbundance",
            "NeutralHydrogenAbundance",
            "GFM_Metals",
            "StarFormationRate",
            "Velocities",
        ],
        start=0,
        stop=1_000_000 if quick else None,
    )
    C0 = rd.to_gas_cells(raw)
    P0 = gp.ion_properties(C0, HI)
    sf = np.asarray(C0.fields["sfr"]) > 0
    nH = np.asarray(C0.fields["rho"]) * np.asarray(C0.fields["metals"])[:, 0] / phys.mp
    x_eq = np.asarray(
        gp.arepo_equilibrium_neutral_fraction(
            nH, P0["T"], C0.fields["n_e"], gp.treecool_rates(H.redshift, gp.default_treecool_path())["Gamma_HI"]
        )
    )
    diffuse = ~sf & (nH < 1e-4)
    check(
        "stored x_HI == Arepo ionisation equilibrium, diffuse gas (n_H < 1e-4): median |ratio - 1|",
        np.median(np.abs(np.asarray(C0.fields["x_HI_stored"])[diffuse] / x_eq[diffuse] - 1)),
        2e-3,
        detail=f"({diffuse.sum():,} cells)",
    )
    check(
        "default x_HI = stored outside star-forming cells (max |diff|)",
        np.abs(np.asarray(P0["ion_fraction"])[~sf] - np.asarray(C0.fields["x_HI_stored"])[~sf]).max(),
        0,
    )
    # 2. skewers, selection and deposition
    names = ("single",) if quick else ("single", "nset")
    DS = 0.25
    SK = {n: SimulationSkewers.axis_aligned(rd, SKEWER_SETS[n]["cofm"], SKEWER_SETS[n]["axis"], DS) for n in names}
    S = SK["single"]
    check("line elements of a skewer sum to the box length (rel)", np.abs(S.line_elements.sum(-1) - L).max() / L, 1e-12)
    allpos = np.concatenate([rd.positions_cMpc(b["Coordinates"]) for _, _, b in rd.iter_blocks(["Coordinates"])])
    tree_all = cKDTree(allpos, boxsize=L)
    for n, Sk in SK.items():
        g = path_geometry(jnp.asarray(Sk.points))
        probe = np.mod(np.concatenate([Sk.points.reshape(-1, 3), np.asarray(g["mid_hi"]).reshape(-1, 3)]), L)
        _, i_all = tree_all.query(probe, workers=-1)
        _, i_sel = cKDTree(np.asarray(Sk.cells.positions), boxsize=L).query(probe, workers=-1)
        check(
            f"selection complete ({n}): fraction of probes whose nearest generator (over all {len(allpos):,}) is missed",
            np.mean(np.any(np.asarray(Sk.cells.positions)[i_sel] != allpos[i_all], axis=1)),
            0,
        )
    top = select_cells_near_paths(rd, S.points, mode="tophat")
    check(
        "tophat selection is a subset of the Voronoi selection (cells missing)", float(np.setdiff1d(top.ids, S.cells.ids).size), 0
    )
    for n, Sk in SK.items():
        seg = Sk.segments("voronoi")
        check(
            f"Voronoi pieces fill every dual segment exactly ({n}, max rel)",
            np.abs(seg.length.sum(-1) - Sk.line_elements).max() / Sk.line_elements.max(),
            1e-9,
        )
    Sn = SK[names[-1]]
    P = Sn.properties(HI)
    N_ref = Sn.segments("voronoi").column_lengths(P["n_ion"]).sum(-1)
    worst = 0.0
    for ds in (1.0, 16.0) if quick else (1.0, 4.0, 16.0, 64.0):
        pts = axis_aligned_skewer_points(
            SKEWER_SETS[names[-1]]["cofm"] * H.code_length_cMpc, SKEWER_SETS[names[-1]]["axis"], L, ds / H.velfac_kms_per_cMpc
        )
        seg_ds = dep.voronoi_segments(pts, Sn.cells.positions, L, gas_size=Sn.cells.size_cMpc)
        worst = max(worst, float(np.abs(seg_ds.column_lengths(P["n_ion"]).sum(-1) / N_ref - 1).max()))
    check(f"H I column independent of the spacing (0.25 to {ds:g} km/s, {Sn.n_skewers} skewers): max |dN/N|", worst, 2e-5)
    Ncol = Sn.column_per_pixel(HI)
    check(
        "column per position pixel sums to the total column (max rel)",
        np.abs(Ncol.sum(-1) / (N_ref * 3.085678e24 * H.a) - 1).max(),
        1e-10,
    )
    SI = Ion("Si", 3)
    N_si = S.segments("voronoi").column_lengths(S.properties(SI)["n_ion"]).sum(-1)
    pts4 = axis_aligned_skewer_points(
        SKEWER_SETS["single"]["cofm"] * H.code_length_cMpc, SKEWER_SETS["single"]["axis"], L, 4.0 / H.velfac_kms_per_cMpc
    )
    N_si4 = (
        dep.voronoi_segments(pts4, S.cells.positions, L, gas_size=S.cells.size_cMpc)
        .column_lengths(S.properties(SI)["n_ion"])
        .sum(-1)
    )
    check("Si III column independent of the spacing (0.25 vs 4 km/s): |dN/N|", np.abs(N_si4 / N_si - 1).max(), 2e-5)
    # 3. what the core receives: the cells (core.Cells), and the Pieces of their absorbers
    A1 = S.absorbers(HI)
    smp = absorber_pieces(A1, S.frame("exact"), 1, skewers=[0])
    M = A1.s.shape[1]
    check(
        "Pieces carry the cells' own temperatures (max |diff| [K])",
        np.abs(np.asarray(smp.temperature[0]).ravel()[:M] - A1.T[0]).max(),
        0,
    )
    live = A1.n[0] > 0
    check(
        "Pieces density = the absorbers' comoving density (max rel)",
        np.abs(np.asarray(smp.number_density[0]).ravel()[:M][live] / A1.n[0][live] - 1).max(),
        1e-12,
    )
    pv = S.point_values(HI)
    check(
        "segment averages x line elements = the deposited column (max rel)",
        np.abs(
            (pv["n"] * pv["line_elements"]).sum(-1) / S.segments("voronoi").column_lengths(S.properties(HI)["n_ion"]).sum(-1) - 1
        ).max(),
        1e-12,
    )
    cl = S.to_cells(["HI", "SiIII"])
    check(
        "core.Cells of the snapshot: the toy models' keys and shapes (mismatches)",
        float(
            sum(
                set(cl.species[sp]) != {"temperature", "velocity", "number_density"}
                or np.shape(cl.species[sp]["velocity"]) != (len(S.cells), 3)
                for sp in ("HI", "SiIII")
            )
        ),
        0,
    )
    check(
        "core.Cells density = n_ion a_snap^3, comoving (max rel)",
        np.abs(cl.species["HI"]["number_density"] / (np.asarray(S.properties(HI)["n_ion"]) * H.a**3) - 1).max(),
        1e-15,
    )
    # 4. the snapshot as gas cells without skewers: a sub-box (snapshot_cells), complete, and the same values as to_cells
    centre = S.points[0, S.points.shape[1] // 2]
    lo, hi, margin = centre - 3.0, centre + 3.0, 1.0
    sub = snapshot_cells(rd, ["HI", "SiIII"], lo, hi, margin=margin)
    inside = lambda x, a, b: np.all(np.mod(x - a, L) < b - a, axis=1)
    check(
        "sub-box cells: every cell of the snapshot in the region and its margin, and no other (count mismatch)",
        abs(int(inside(allpos, lo - margin, hi + margin).sum()) - len(sub))
        + int((~inside(sub.positions, lo - margin, hi + margin)).sum()),
        0,
    )
    common, ia, ib = np.intersect1d(sub.ids, cl.ids, return_indices=True)
    check(
        f"sub-box cells = to_cells of the skewer on the {common.size} common cells (max |diff|, every field and species)",
        max(
            np.abs(sub.species[sp][f][ia] - cl.species[sp][f][ib]).max()
            for sp in ("HI", "SiIII")
            for f in ("number_density", "temperature", "velocity")
        ),
        0,
    )
    probe = lo + np.random.default_rng(3).uniform(0.0, 1.0, (20000, 3)) * (hi - lo)
    k_sub = cKDTree(sub.positions, boxsize=L).query(np.mod(probe, L))[1]
    j_all = tree_all.query(np.mod(probe, L))[1]
    check(
        "sub-box: the nearest generator of every point of the region is a kept cell (probes whose cell differs)",
        float(np.any(allpos[j_all] != sub.positions[k_sub], axis=1).sum()),
        0,
    )
    del allpos, tree_all
    return checks


def _assert_all(checks):
    bad = [(n, v, t) for n, v, t, expect in checks if not (v > t if expect else v <= t)]
    assert not bad, bad


@pytest.mark.tng
def test_tng_fields():
    _assert_all(tng_field_checks(quick=True))


@pytest.mark.tng
@pytest.mark.slow
def test_tng_fields_full():
    _assert_all(tng_field_checks(quick=False))
