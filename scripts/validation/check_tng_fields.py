#!/usr/bin/env python
"""Field extraction from TNG50-4 (read-only, ``/home/STORAGE/TNG50-4`` snapshot 25): per-cell H I physics, selection
completeness, Voronoi deposition (exact pieces, conserved column for any spacing), the gas cells handed to the core
(``core.Cells``, the container of the toy models too) and the tau inputs."""
from _common import TNG, Checks

C = Checks("check_tng_fields", __doc__, quick_help="one skewer, coarser spacing scan, first 10^6 cells for the physics")
import numpy as np                                                        # noqa: E402
import jax.numpy as jnp                                                   # noqa: E402
from scipy.spatial import cKDTree                                         # noqa: E402
from skewerjax.core import absorber_pieces, axis_aligned_skewer_points, deposition as dep, path_geometry, phys  # noqa: E402
from skewerjax.simulations import HI, ArepoHDF5Reader, Ion, SimulationSkewers, gas_physics as gp, select_cells_near_paths  # noqa: E402
from skewerjax.comparison import SKEWER_SETS                              # noqa: E402

rd = ArepoHDF5Reader(*TNG)
H, L = rd.header, rd.header.box_cMpc
# 1. per-cell physics: TNG's stored x_HI is its own ionisation equilibrium in diffuse gas
raw = rd.read_rows(0, ["Coordinates", "Masses", "Density", "InternalEnergy", "ElectronAbundance", "NeutralHydrogenAbundance",
                       "GFM_Metals", "StarFormationRate", "Velocities"], start=0, stop=1_000_000 if C.quick else None)
C0 = rd.to_gas_cells(raw)
P0 = gp.ion_properties(C0, HI)
sf = np.asarray(C0.fields["sfr"]) > 0
nH = np.asarray(C0.fields["rho"]) * np.asarray(C0.fields["metals"])[:, 0] / phys.mp
x_eq = np.asarray(gp.arepo_equilibrium_neutral_fraction(nH, P0["T"], C0.fields["n_e"],
                                                        gp.treecool_rates(H.redshift, gp.default_treecool_path())["Gamma_HI"]))
diffuse = ~sf & (nH < 1e-4)
C.check("stored x_HI == Arepo ionisation equilibrium, diffuse gas (n_H < 1e-4): median |ratio - 1|",
        np.median(np.abs(np.asarray(C0.fields["x_HI_stored"])[diffuse] / x_eq[diffuse] - 1)), 2e-3, detail=f"({diffuse.sum():,} cells)")
C.check("default x_HI = stored outside star-forming cells (max |diff|)",
        np.abs(np.asarray(P0["ion_fraction"])[~sf] - np.asarray(C0.fields["x_HI_stored"])[~sf]).max(), 0)
# 2. skewers, selection and deposition
names = ("single",) if C.quick else ("single", "nset")
DS = 0.25
SK = {n: SimulationSkewers.axis_aligned(rd, SKEWER_SETS[n]["cofm"], SKEWER_SETS[n]["axis"], DS) for n in names}
S = SK["single"]
C.check("line elements of a skewer sum to the box length (rel)", np.abs(S.line_elements.sum(-1) - L).max() / L, 1e-12)
allpos = np.concatenate([rd.positions_cMpc(b["Coordinates"]) for _, _, b in rd.iter_blocks(["Coordinates"])])
tree_all = cKDTree(allpos, boxsize=L)
for n, Sk in SK.items():
    g = path_geometry(jnp.asarray(Sk.points))
    probe = np.mod(np.concatenate([Sk.points.reshape(-1, 3), np.asarray(g["mid_hi"]).reshape(-1, 3)]), L)
    _, i_all = tree_all.query(probe, workers=-1)
    _, i_sel = cKDTree(np.asarray(Sk.cells.positions), boxsize=L).query(probe, workers=-1)
    C.check(f"selection complete ({n}): fraction of probes whose nearest generator (over all {len(allpos):,}) is missed",
            np.mean(np.any(np.asarray(Sk.cells.positions)[i_sel] != allpos[i_all], axis=1)), 0)
del allpos, tree_all
top = select_cells_near_paths(rd, S.points, mode="tophat")
C.check("tophat selection is a subset of the Voronoi selection (cells missing)", float(np.setdiff1d(top.ids, S.cells.ids).size), 0)
for n, Sk in SK.items():
    seg = Sk.segments("voronoi")
    C.check(f"Voronoi pieces fill every dual segment exactly ({n}, max rel)", np.abs(seg.length.sum(-1) - Sk.line_elements).max()
            / Sk.line_elements.max(), 1e-9)
Sn = SK[names[-1]]
P = Sn.properties(HI)
N_ref = Sn.segments("voronoi").column_lengths(P["n_ion"]).sum(-1)
worst = 0.0
for ds in ((1.0, 16.0) if C.quick else (1.0, 4.0, 16.0, 64.0)):
    pts = axis_aligned_skewer_points(SKEWER_SETS[names[-1]]["cofm"] * H.code_length_cMpc, SKEWER_SETS[names[-1]]["axis"], L,
                                     ds / H.velfac_kms_per_cMpc)
    seg_ds = dep.voronoi_segments(pts, Sn.cells.positions, L, gas_size=Sn.cells.size_cMpc)
    worst = max(worst, float(np.abs(seg_ds.column_lengths(P["n_ion"]).sum(-1) / N_ref - 1).max()))
C.check(f"H I column independent of the spacing (0.25 to {ds:g} km/s, {Sn.n_skewers} skewers): max |dN/N|", worst, 2e-5)
Ncol = Sn.column_per_pixel(HI)
C.check("column per position pixel sums to the total column (max rel)", np.abs(Ncol.sum(-1) / (N_ref * 3.085678e24 * H.a) - 1).max(), 1e-10)
SI = Ion("Si", 3)
N_si = S.segments("voronoi").column_lengths(S.properties(SI)["n_ion"]).sum(-1)
pts4 = axis_aligned_skewer_points(SKEWER_SETS["single"]["cofm"] * H.code_length_cMpc, SKEWER_SETS["single"]["axis"], L, 4.0 / H.velfac_kms_per_cMpc)
N_si4 = dep.voronoi_segments(pts4, S.cells.positions, L, gas_size=S.cells.size_cMpc).column_lengths(S.properties(SI)["n_ion"]).sum(-1)
C.check("Si III column independent of the spacing (0.25 vs 4 km/s): |dN/N|", np.abs(N_si4 / N_si - 1).max(), 2e-5)
# 3. what the core receives: the cells (core.Cells), and the Pieces of their absorbers
A1 = S.absorbers(HI)
smp = absorber_pieces(A1, S.frame("exact"), 1, skewers=[0])
M = A1.s.shape[1]
C.check("Pieces carry the cells' own temperatures (max |diff| [K])", np.abs(np.asarray(smp.temperature[0]).ravel()[:M] - A1.T[0]).max(), 0)
live = A1.n[0] > 0
C.check("Pieces density = the absorbers' comoving density (max rel)",
        np.abs(np.asarray(smp.number_density[0]).ravel()[:M][live] / A1.n[0][live] - 1).max(), 1e-12)
pv = S.point_values(HI)
C.check("segment averages x line elements = the deposited column (max rel)",
        np.abs((pv["n"] * pv["line_elements"]).sum(-1) / S.segments("voronoi").column_lengths(S.properties(HI)["n_ion"]).sum(-1) - 1).max(), 1e-12)
cl = S.to_cells(["HI", "SiIII"])
C.check("core.Cells of the snapshot: the toy models' keys and shapes (mismatches)",
        float(sum(set(cl.species[sp]) != {"temperature", "velocity", "number_density"} or np.shape(cl.species[sp]["velocity"]) != (len(S.cells), 3)
                  for sp in ("HI", "SiIII"))), 0)
C.check("core.Cells density = n_ion a_snap^3, comoving (max rel)",
        np.abs(cl.species["HI"]["number_density"] / (np.asarray(S.properties(HI)["n_ion"]) * H.a ** 3) - 1).max(), 1e-15)

if C.outdir is not None:
    from _common import figure_backend
    figure_backend()
    import matplotlib.pyplot as plt
    from skewerjax.utils.plotting import ENTITY_COLORS as EC
    fig, axs = plt.subplots(2, 1, figsize=(14, 8), sharex=True, layout="constrained")
    x = (np.arange(Ncol.shape[1]) + 0.5) * L / Ncol.shape[1]
    axs[0].semilogy(x, np.maximum(Ncol[0], 1e8), color=EC["HI"]); axs[0].set_ylabel(r"$N_{\rm HI}$ per pixel [cm$^{-2}$]")
    axs[1].semilogy(S.s[0], np.maximum(pv["T"][0], 1), color=EC["temperature"]); axs[1].set(ylabel="T [K]", xlabel="s [cMpc]")
    C.savefig(fig, "tng_fields_single")
C.done()
