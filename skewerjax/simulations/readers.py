"""Snapshot readers (layer 1): format-specific HDF5 I/O and units, NumPy only.

A reader knows the file layout and the unit conventions of one simulation format, and nothing
else. It exposes:

* ``header``: a :class:`SnapshotHeader` (redshift, cosmology, box, code units);
* ``iter_blocks(datasets, max_rows)``: a stream over the gas, one block of at most ``max_rows``
  cells at a time, so memory is bounded by one block whatever the snapshot size;
* ``read_rows(file_index, datasets, rows)``: a partial read of selected rows;
* ``to_gas_cells(raw, ...)``: conversion of raw datasets to the generic, physical-unit container
  :class:`skewerjax.simulations.gas_cells.GasCells` (layer 2).

Only :class:`ArepoHDF5Reader` (Arepo / IllustrisTNG HDF5, ``snapdir_XXX/snap_XXX.N.hdf5``) is
implemented. A new format (Gadget SPH, Enzo/RAMSES grids, an N-body + FGPA box, a light cone)
needs only a new reader with the same three methods; layers 2-6 are unchanged.

TNG conventions (https://www.tng-project.org/data/docs/specifications/, PartType0):

* ``Coordinates`` [ckpc/h]: the Voronoi cell centre, i.e. the mesh-generating point
  (TNG forum topic 307; Springel 2010, Sec. 4). ``CenterOfMass`` is the mass centroid.
* ``Density`` [(1e10 Msun/h)/(ckpc/h)^3] = mass / volume (comoving); ``Masses`` [1e10 Msun/h].
* ``Velocities`` [km sqrt(a)/s]: "multiply by sqrt(a) to obtain peculiar velocity".
* ``InternalEnergy`` [(km/s)^2]; ``ElectronAbundance`` = n_e / n_H;
  ``NeutralHydrogenAbundance`` = n_HI / n_H; ``GFM_Metals`` mass fractions H, He, C, N, O, Ne,
  Mg, Si, Fe, other. For star-forming cells, T and x_HI follow the effective equation of state
  ("use with caution").
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Sequence

import h5py
import numpy as np

KPC_CM = 3.085678e21                     # 1 kpc [cm] (Arepo / fake_spectra value)
MPC_CM = 3.085678e24                     # 1 Mpc [cm]
GFM_METALS_ORDER = ("H", "He", "C", "N", "O", "Ne", "Mg", "Si", "Fe")
GAS_DATASETS = ("Coordinates", "Velocities", "Masses", "Density", "InternalEnergy",
                "ElectronAbundance", "NeutralHydrogenAbundance", "GFM_Metals", "StarFormationRate")


@dataclass(frozen=True)
class SnapshotHeader:
    """Snapshot epoch, cosmology, box and code units (format independent).

    Lengths in code units are ``unit_length_cm / h`` comoving cm (ckpc/h for TNG); the box is
    also given in comoving Mpc without h (SkewerJAX's length unit, ``SkewerBundle``).
    """
    redshift: float
    a: float
    h: float
    box_code: float                # [code length, ckpc/h]
    omega_m: float
    omega_l: float
    omega_b: float
    unit_length_cm: float
    unit_mass_g: float
    unit_velocity_cms: float
    n_files: int
    n_gas: int

    @property
    def box_cMpc(self) -> float:
        """Box side [comoving Mpc, no h]."""
        return self.box_code * self.unit_length_cm / self.h / MPC_CM

    @property
    def code_length_cMpc(self) -> float:
        """One code length in comoving Mpc (no h)."""
        return self.unit_length_cm / self.h / MPC_CM

    @property
    def hubble_kms_mpc(self) -> float:
        """H(z) [km/s/Mpc] for flat LCDM without radiation (the simulation's own expansion)."""
        return 100.0 * self.h * np.sqrt(self.omega_m / self.a ** 3 + self.omega_l)

    @property
    def velfac_kms_per_cMpc(self) -> float:
        """Hubble velocity per comoving Mpc at the snapshot epoch, a H(z) [km/s per cMpc]."""
        return self.a * self.hubble_kms_mpc

    @property
    def vmax_kms(self) -> float:
        """Box length in (linear) Hubble velocity [km/s] (fake_spectra ``spectra.py`` L228)."""
        return self.box_cMpc * self.velfac_kms_per_cMpc

    @property
    def density_cgs(self) -> float:
        """Code density -> physical g/cm^3 (h^2 / a^3 included)."""
        return self.unit_mass_g / self.unit_length_cm ** 3 * self.h ** 2 / self.a ** 3

    def pixel_grid(self, res_kms: float) -> tuple[int, float]:
        """fake_spectra's periodic pixel grid: nbins = int(v_max / res), dv = v_max / nbins [km/s]
        (``spectra.py`` L241-L243). Only used for comparisons; SkewerJAX takes any velocity grid."""
        if not res_kms > 0:
            raise ValueError(f"res_kms must be > 0, got {res_kms}.")
        nbins = int(self.vmax_kms / res_kms)
        if nbins < 2:
            raise ValueError(f"res = {res_kms} km/s leaves fewer than 2 pixels in v_max = {self.vmax_kms:.1f} km/s.")
        return nbins, self.vmax_kms / nbins


def _attr(d, key, default=None):
    if key not in d:
        if default is None:
            raise KeyError(f"header lacks {key!r}")
        return default
    return float(np.asarray(d[key]).ravel()[0])


class ArepoHDF5Reader:
    """Reader for Arepo / IllustrisTNG HDF5 snapshots (``<base>/snapdir_XXX/snap_XXX.N.hdf5``).

    Parameters
    ----------
    base : directory that contains ``snapdir_XXX`` (e.g. ``/home/STORAGE/TNG50-4``).
    snapnum : snapshot number.
    part_type : HDF5 group of the gas, default ``"PartType0"``.

    Notes
    -----
    Read-only: files are opened with ``h5py.File(..., "r")`` for each access and never written.
    """

    def __init__(self, base, snapnum: int, part_type: str = "PartType0"):
        self.base = Path(base)
        self.snapnum = int(snapnum)
        self.part_type = part_type
        snapdir = self.base / f"snapdir_{self.snapnum:03d}"
        if not snapdir.is_dir():
            raise FileNotFoundError(f"no snapshot directory {snapdir}")
        files = list(snapdir.glob(f"snap_{self.snapnum:03d}.*.hdf5"))
        if not files:
            raise FileNotFoundError(f"no snap_{self.snapnum:03d}.*.hdf5 in {snapdir}")
        key = lambda p: int(re.search(r"\.(\d+)\.hdf5$", p.name).group(1))
        self.files = sorted(files, key=key)
        with h5py.File(self.files[0], "r") as f:
            hdr = dict(f["Header"].attrs)
            par = dict(f["Parameters"].attrs) if "Parameters" in f else {}
            self.config = dict(f["Config"].attrs) if "Config" in f else {}
            self.parameters = par
            self.dataset_attrs = {k: dict(v.attrs) for k, v in f[part_type].items()}
        n_gas = int(np.asarray(hdr.get("NumPart_Total", [0]))[0]) + \
            (int(np.asarray(hdr.get("NumPart_Total_HighWord", [0]))[0]) << 32)
        a = _attr(hdr, "Time")
        if not 0.0 < a <= 1.0:
            raise ValueError(f"header Time (scale factor) must be in (0, 1], got {a}.")
        self.header = SnapshotHeader(
            redshift=1.0 / a - 1.0, a=a, h=_attr(hdr, "HubbleParam"), box_code=_attr(hdr, "BoxSize"),
            omega_m=_attr(hdr, "Omega0"), omega_l=_attr(hdr, "OmegaLambda"),
            omega_b=_attr(hdr, "OmegaBaryon", _attr(par, "OmegaBaryon", 0.0486)),
            unit_length_cm=_attr(hdr, "UnitLength_in_cm", _attr(par, "UnitLength_in_cm", KPC_CM)),
            unit_mass_g=_attr(hdr, "UnitMass_in_g", _attr(par, "UnitMass_in_g", 1.989e43)),
            unit_velocity_cms=_attr(hdr, "UnitVelocity_in_cm_per_s", _attr(par, "UnitVelocity_in_cm_per_s", 1e5)),
            n_files=len(self.files), n_gas=n_gas)

    # ------------------------------------------------------------------ raw access
    def n_rows(self, file_index: int) -> int:
        """Number of gas cells in one file."""
        with h5py.File(self.files[file_index], "r") as f:
            return int(f[self.part_type]["Density"].shape[0]) if self.part_type in f else 0

    def read_rows(self, file_index: int, datasets: Sequence[str], rows=None, start: int = 0,
                  stop: Optional[int] = None) -> dict:
        """Read ``datasets`` of one file: all rows in [start, stop), or the sorted ``rows`` (fancy index).

        Returns raw code-unit arrays with their native dtypes.
        """
        if not 0 <= file_index < len(self.files):
            raise IndexError(f"file_index {file_index} out of range [0, {len(self.files)}).")
        out = {}
        with h5py.File(self.files[file_index], "r") as f:
            g = f[self.part_type]
            for k in datasets:
                if k not in g:
                    raise KeyError(f"{self.files[file_index].name}: {self.part_type} has no dataset {k!r}")
                if rows is None:
                    out[k] = g[k][start:stop]
                else:
                    rows = np.asarray(rows)
                    if rows.size and (np.any(np.diff(rows) <= 0) or rows[0] < 0):
                        raise ValueError("rows must be strictly increasing and non-negative.")
                    out[k] = g[k][rows] if rows.size else np.empty((0,) + g[k].shape[1:], g[k].dtype)
        return out

    def iter_blocks(self, datasets: Sequence[str], max_rows: int = 4_000_000) -> Iterator[tuple[int, int, dict]]:
        """Stream the gas in blocks of at most ``max_rows`` cells.

        Yields ``(file_index, row_offset, raw)``. Peak memory ~ max_rows x (bytes per cell of
        ``datasets``), independent of the snapshot size.
        """
        if max_rows < 1:
            raise ValueError(f"max_rows must be >= 1, got {max_rows}.")
        for fi in range(len(self.files)):
            n = self.n_rows(fi)
            for r0 in range(0, n, max_rows):
                yield fi, r0, self.read_rows(fi, datasets, start=r0, stop=min(n, r0 + max_rows))

    # ------------------------------------------------------------------ conversion (layer 1 -> 2)
    def positions_cMpc(self, coordinates) -> np.ndarray:
        """Code coordinates [ckpc/h] -> comoving Mpc (no h), wrapped into [0, box)."""
        x = np.asarray(coordinates, dtype=np.float64) * self.header.code_length_cMpc
        return np.mod(x, self.header.box_cMpc)

    def volumes_cMpc3(self, masses, density) -> np.ndarray:
        """Cell volume m / rho (comoving) [cMpc^3]."""
        v_code = np.asarray(masses, dtype=np.float64) / np.asarray(density, dtype=np.float64)
        return v_code * self.header.code_length_cMpc ** 3

    def to_gas_cells(self, raw: dict, ids=None):
        """Convert raw TNG datasets to :class:`GasCells` in physical units.

        Fields of the container (all float64 except ``ids``):
        ``rho`` [physical g cm^-3], ``u`` [erg g^-1], ``n_e`` (n_e / n_H), ``x_HI_stored``,
        ``metals`` (N, 9) mass fractions in ``GFM_METALS_ORDER``, ``sfr`` [Msun/yr],
        ``velocity`` (N, 3) physical peculiar velocity [km/s] (= Velocities x sqrt(a)).
        """
        from skewerjax.simulations.gas_cells import GasCells
        hdr = self.header
        fields = {}
        if "Density" in raw:
            fields["rho"] = np.asarray(raw["Density"], dtype=np.float64) * hdr.density_cgs
        if "InternalEnergy" in raw:
            fields["u"] = np.asarray(raw["InternalEnergy"], dtype=np.float64) * hdr.unit_velocity_cms ** 2
        if "ElectronAbundance" in raw:
            fields["n_e"] = np.asarray(raw["ElectronAbundance"], dtype=np.float64)
        if "NeutralHydrogenAbundance" in raw:
            fields["x_HI_stored"] = np.asarray(raw["NeutralHydrogenAbundance"], dtype=np.float64)
        if "GFM_Metals" in raw:
            fields["metals"] = np.clip(np.asarray(raw["GFM_Metals"], dtype=np.float64)[:, :9], 0.0, None)
        if "StarFormationRate" in raw:
            fields["sfr"] = np.asarray(raw["StarFormationRate"], dtype=np.float64)
        if "Velocities" in raw:
            fields["velocity"] = np.asarray(raw["Velocities"], dtype=np.float64) * np.sqrt(hdr.a)
        return GasCells(positions=self.positions_cMpc(raw["Coordinates"]),
                        volumes=self.volumes_cMpc3(raw["Masses"], raw["Density"]),
                        fields=fields, box=hdr.box_cMpc, a=hdr.a,
                        ids=None if ids is None else np.asarray(ids))
