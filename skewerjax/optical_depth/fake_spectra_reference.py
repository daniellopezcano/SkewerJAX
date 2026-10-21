"""fake_spectra results for the comparison notebooks: live (driver) or from the HDF5 reference file.

``FakeSpectraSource`` hands the comparison notebooks fake_spectra's spectra and the per-cell values
it passes to its C engine, from one of two places:

* **live**: a fake_spectra build directory is given. Full runs go through the driver
  ``run_fake_spectra_skewers.py`` in its own process and are cached as ``<cache>/fs_<tag>.h5``;
  per-cell probes are cached in ``<cache>/fs_products.h5``.
* **reference**: no build. Every result is read from the reference file (one HDF5 group per
  result), which a live run writes with ``export``.

Nothing of fake_spectra is imported in reference mode. ``cell_inputs`` converts fake_spectra's
per-cell values to SkewerJAX units, in the order of SkewerJAX's selected cells, so that they can be
fed to ``SimulationSkewers`` (``override=``) to isolate the inputs from the integration.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional

import h5py
import numpy as np

from skewerjax.simulations.simulation_skewers import cached, h5_read, h5_write

# The skewer sets of the driver (axis 1/2/3 = x/y/z, cofm in ckpc/h).
SKEWER_SETS = {
    "single": dict(axis=np.array([3], dtype=np.int32), cofm=np.array([[17500.25, 17500.75, 0.0]])),
    "nset": dict(axis=np.tile(np.array([1, 2, 3], dtype=np.int32), 6)[:16],
                 cofm=np.random.default_rng(2155).uniform(0.0, 35000.0, size=(16, 3))),
    "face": dict(axis=np.array([3, 1], dtype=np.int32), cofm=np.array([[5.0, 17500.75, 0.0], [0.0, 34995.0, 12000.5]])),
}
SKEWER_SETS["sk8"] = dict(axis=SKEWER_SETS["nset"]["axis"][8:9].copy(), cofm=SKEWER_SETS["nset"]["cofm"][8:9].copy())


class FakeSpectraSource:
    """fake_spectra results, live or from the reference file.

    Parameters
    ----------
    reference : HDF5 reference file (read when there is no build; written by ``export``).
    build : fake_spectra build directory, or None for reference mode.
    driver : ``run_fake_spectra_skewers.py`` (live mode).
    cache : directory of the live results (``fs_<tag>.h5``, ``fs_products.h5``).
    snap_base, snapnum, res : snapshot and pixel width [km/s] of the runs (live mode).
    """

    def __init__(self, reference, build=None, driver=None, cache=None, snap_base=None, snapnum=None, res: float = 1.0):
        self.reference, self.cache = Path(reference), None if cache is None else Path(cache)
        self.snap_base, self.snapnum, self.res = snap_base, snapnum, res
        self.live = build is not None and Path(build).is_dir() and driver is not None
        self.build, self.driver = build, driver
        self.used, self._spec = {}, {}
        if self.live:
            s = importlib.util.spec_from_file_location("run_fake_spectra_skewers", driver)
            self.drv = importlib.util.module_from_spec(s)
            sys.modules["run_fake_spectra_skewers"] = self.drv
            s.loader.exec_module(self.drv)
            for k, v in SKEWER_SETS.items():
                if not (np.array_equal(v["axis"], self.drv.SKEWER_SETS[k]["axis"]) and np.allclose(v["cofm"], self.drv.SKEWER_SETS[k]["cofm"])):
                    raise ValueError(f"skewer set {k!r} differs from the driver's")
            self.spectra, self.line_data, self.PI = self.drv.import_fake_spectra(str(build))
        elif not self.reference.exists():
            raise FileNotFoundError(f"no fake_spectra build and no reference file {self.reference}")

    @property
    def mode(self) -> str:
        return f"live ({self.build})" if self.live else f"reference file ({self.reference})"

    def _ref(self, name):
        with h5py.File(self.reference, "r") as f:
            return h5_read(f[name])

    def run(self, tag: str, sset: str = "single", sf_neutral: bool = False, species: str = "H:1:1215",
            products: str = "tau,colden", kernel: str = "default") -> dict:
        """A full fake_spectra run (``fs_<tag>``): tau, colden, ... of the skewer set ``sset``."""
        name = f"fs_{tag}"
        if not self.live:
            d = self._ref(name)
        else:
            f = self.cache / f"{name}.h5"
            if not f.exists():
                subprocess.run([sys.executable, str(self.driver), "full", "--build", str(self.build), "--snap-base", str(self.snap_base),
                                "--snapnum", str(self.snapnum), "--cache", str(self.cache), "--tag", tag, "--set", sset,
                                "--res", str(self.res), "--sf-neutral", str(int(sf_neutral)), "--species", species,
                                "--products", products, "--kernel", kernel], check=True, capture_output=True)
            d = self.drv.load_h5(f)
        self.used[name] = d
        return d

    def get(self, name: str, build: Callable[[], dict]) -> dict:
        """Any other fake_spectra product, cached in ``fs_products.h5`` (live) or read (reference)."""
        d = cached(self.cache / "fs_products.h5", name, build) if self.live else self._ref(name)
        self.used[name] = d
        return d

    def spectra_object(self, sset: str, sf_neutral: bool = False):
        key = (sset, sf_neutral)
        if key not in self._spec:
            self._spec[key] = self.drv.make_spectra(self.spectra, self.snap_base, self.snapnum, str(self.cache), sset, self.res,
                                                    sf_neutral, f"obj_{sset}")
        return self._spec[key]

    def constants(self) -> dict:
        """fake_spectra's box constants (velfac, vmax, nbins, dvbin, rscale, ...)."""
        return self.get("fs_constants", lambda: (lambda s: dict(
            velfac=s.velfac, vmax=s.vmax, nbins=s.nbins, dvbin=s.dvbin, rscale=s.rscale, Hz=s.Hz, red=s.red, tautail=s.tautail,
            kernel_int=s.kernel_int, PhysDensThresh=s.gasprop.PhysDensThresh, gamma_UVB=s.gasprop.gamma_UVB))(self.spectra_object("single")))

    def cell_values(self, reader, cells, sset: str, element: str = "H", stage: int = 1, sf_neutral: bool = False) -> dict:
        """fake_spectra's per-cell values (``Spectra._read_particle_data``) for SkewerJAX's selected
        ``cells`` (same order; ``GasCells.ids`` = file index << 40 | row)."""
        ids = np.asarray(cells.ids)
        fi, row = (ids >> 40).astype(int), (ids & ((1 << 40) - 1)).astype(int)
        names = np.array([reader.files[i].name for i in fi])
        tag = f"cells_{sset}_{element}{stage}_{int(sf_neutral)}"
        return self.get(tag, lambda: self.drv.cell_values(self.spectra_object(sset, sf_neutral), names, row, element, stage))

    def cell_inputs(self, reader, cells, sset: str, ion, sf_neutral: bool = False) -> dict:
        """fake_spectra's per-cell inputs in SkewerJAX units: pos, size [cMpc], n [physical cm^-3],
        T [K], vel (N, 3) [km/s] (``SimulationSkewers.cell_inputs`` keys)."""
        H = reader.header
        f = self.cell_values(reader, cells, sset, ion.element, ion.stage, sf_neutral)
        rscale = float(self.constants()["rscale"])
        return dict(pos=np.mod(f["pos"] * H.code_length_cMpc, H.box_cMpc), size=f["hh"] * H.code_length_cMpc,
                    n=f["elem_den"] / rscale, T=f["temp"].astype(float), vel=f["vel"].astype(float))

    def export(self, path=None) -> int:
        """Add every result used so far to the reference file (groups already there are kept);
        returns the file size [bytes]."""
        path = Path(path or self.reference)
        with h5py.File(path, "a") as f:
            for n, d in self.used.items():
                if n not in f:
                    h5_write(f.create_group(n), {k: v for k, v in d.items() if np.asarray(v).dtype != object})
        return path.stat().st_size
