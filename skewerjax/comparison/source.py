"""fake_spectra results for the comparisons: live (``comparison.driver``) or read from an HDF5 reference file.

``FakeSpectraSource`` hands the comparison fake_spectra's spectra and the per-cell values it passes to its C engine,
from one of two places:

* **live** (fake_spectra importable and a ``cache`` directory given): full runs go through ``comparison.driver`` in
  their own process and are cached as ``<cache>/fs_<tag>.h5``; per-cell probes are cached in ``<cache>/fs_products.h5``;
* **reference**: every result is read from the reference file (one HDF5 group per result), written by ``export``.

Nothing of fake_spectra is imported in reference mode. ``cell_inputs`` converts fake_spectra's per-cell values to
SkewerJAX units, in the order of SkewerJAX's selected cells, so that they can be fed to ``SimulationSkewers``
(``override=``) to isolate the inputs from the integration. NumPy / h5py only.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from skewerjax.comparison import driver as drv
from skewerjax.comparison.driver import SKEWER_SETS  # noqa: F401  (re-exported by skewerjax.comparison)
from skewerjax.simulations.snapshot_skewers import cached, h5_read, h5_write


class FakeSpectraSource:
    """fake_spectra results, live or from the reference file.

    Parameters
    ----------
    reference : HDF5 reference file (read in reference mode; written by ``export``).
    cache : directory of the live results (``fs_<tag>.h5``, ``fs_products.h5``); None: reference mode.
    snap_base, snapnum, res : snapshot and pixel width [km/s] of the live runs.
    build : optional fake_spectra build directory (otherwise the installed package).
    live : force the mode (default: live when ``cache`` is given and fake_spectra is importable).
    """

    def __init__(
        self, reference=None, cache=None, snap_base=None, snapnum=None, res: float = 1.0, build=None, live: Optional[bool] = None
    ):
        self.reference = None if reference is None else Path(reference)
        self.cache = None if cache is None else Path(cache)
        self.snap_base, self.snapnum, self.res, self.build = snap_base, snapnum, res, build
        importable = build is not None or importlib.util.find_spec("fake_spectra") is not None
        self.live = (cache is not None and importable) if live is None else bool(live)
        self.used, self._spec = {}, {}
        if self.live:
            if not importable:
                raise ImportError("live mode needs fake_spectra (installed, or `build=`).")
            if self.cache is None:
                raise ValueError("live mode needs a `cache` directory.")
            self.spectra = drv.import_fake_spectra(build)
        elif self.reference is None or not self.reference.exists():
            raise FileNotFoundError(f"no live fake_spectra (cache={cache}) and no reference file {self.reference}")

    @property
    def mode(self) -> str:
        return f"live (cache {self.cache})" if self.live else f"reference file ({self.reference})"

    def _ref(self, name):
        import h5py  # optional dependency (the simulations extra): imported where it is used

        with h5py.File(self.reference, "r") as f:
            return h5_read(f[name])

    def run(
        self,
        tag: str,
        sset: str = "single",
        sf_neutral: bool = False,
        species: str = "H:1:1215",
        products: str = "tau,colden",
        kernel: str = "default",
    ) -> dict:
        """A full fake_spectra run (``fs_<tag>``): tau, colden, ... of the skewer set ``sset``."""
        name = f"fs_{tag}"
        if not self.live:
            d = self._ref(name)
        else:
            f = self.cache / f"{name}.h5"
            if not f.exists():
                cmd = [
                    sys.executable,
                    "-m",
                    "skewerjax.comparison.driver",
                    "full",
                    "--snap-base",
                    str(self.snap_base),
                    "--snapnum",
                    str(self.snapnum),
                    "--cache",
                    str(self.cache),
                    "--tag",
                    tag,
                    "--set",
                    sset,
                    "--res",
                    str(self.res),
                    "--sf-neutral",
                    str(int(sf_neutral)),
                    "--species",
                    species,
                    "--products",
                    products,
                    "--kernel",
                    kernel,
                ] + ([] if self.build is None else ["--build", str(self.build)])
                self.cache.mkdir(parents=True, exist_ok=True)
                subprocess.run(cmd, check=True, capture_output=True)
            d = drv.load_h5(f)
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
            self._spec[key] = drv.make_spectra(
                self.spectra,
                self.snap_base,
                self.snapnum,
                self.cache / "fs_work",
                sset,
                self.res,
                sf_neutral,
                f"obj_{sset}_{int(sf_neutral)}",
            )
        return self._spec[key]

    def constants(self) -> dict:
        """fake_spectra's box constants (velfac, vmax, nbins, dvbin, rscale, ...)."""
        return self.get("fs_constants", lambda: drv.constants(self.spectra_object("single")))

    def cell_values(self, reader, cells, sset: str, element: str = "H", stage: int = 1, sf_neutral: bool = False) -> dict:
        """fake_spectra's per-cell values for SkewerJAX's selected ``cells`` (same order; ``GasCells.ids`` = file << 40 | row)."""
        ids = np.asarray(cells.ids)
        fi, row = (ids >> 40).astype(int), (ids & ((1 << 40) - 1)).astype(int)
        names = np.array([reader.files[i].name for i in fi])
        tag = f"cells_{sset}_{element}{stage}_{int(sf_neutral)}"
        return self.get(tag, lambda: drv.cell_values(self.spectra_object(sset, sf_neutral), names, row, element, stage))

    def cell_inputs(self, reader, cells, sset: str, ion, sf_neutral: bool = False) -> dict:
        """fake_spectra's per-cell inputs in SkewerJAX units: pos, size [cMpc], n [physical cm^-3], T [K], vel (N, 3)
        [km/s] (``SimulationSkewers.cell_inputs`` keys)."""
        H = reader.header
        f = self.cell_values(reader, cells, sset, ion.element, ion.stage, sf_neutral)
        rscale = float(self.constants()["rscale"])
        return dict(
            pos=np.mod(f["pos"] * H.code_length_cMpc, H.box_cMpc),
            size=f["hh"] * H.code_length_cMpc,
            n=f["elem_den"] / rscale,
            T=f["temp"].astype(float),
            vel=f["vel"].astype(float),
        )

    def export(self, path=None) -> int:
        """Add every result used so far to the reference file (groups already there are kept); returns its size [bytes]."""
        path = Path(path or self.reference)
        import h5py  # optional dependency (the simulations extra): imported where it is used

        with h5py.File(path, "a") as f:
            for n, d in self.used.items():
                if n not in f:
                    h5_write(f.create_group(n), {k: v for k, v in d.items() if np.asarray(v).dtype != object})
        return path.stat().st_size
