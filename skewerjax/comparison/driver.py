"""Runs fake_spectra (NumPy / C, no JAX): full skewer runs to HDF5 and its per-cell inputs, for ``FakeSpectraSource``.

Command line (one run in its own process; fake_spectra's memory is released at exit):

    python -m skewerjax.comparison.driver full --snap-base DIR --snapnum N --cache DIR --tag TAG --set SET \
        [--res 1.0] [--sf-neutral 0|1] [--species H:1:1215] [--products tau,colden,temp,velocity] [--kernel default]

writes ``<cache>/fs_<TAG>.h5`` with the products of the skewer set ``SET`` (``SKEWER_SETS``) and the box constants.
fake_spectra must be importable (installed, or ``--build DIR`` with its build directory).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

# The skewer sets of the comparisons (axis 1/2/3 = x/y/z, cofm in ckpc/h, TNG50-4 box 35000 ckpc/h).
SKEWER_SETS = {
    "single": dict(axis=np.array([3], dtype=np.int32), cofm=np.array([[17500.25, 17500.75, 0.0]])),
    "nset": dict(axis=np.tile(np.array([1, 2, 3], dtype=np.int32), 6)[:16],
                 cofm=np.random.default_rng(2155).uniform(0.0, 35000.0, size=(16, 3))),
    "face": dict(axis=np.array([3, 1], dtype=np.int32), cofm=np.array([[5.0, 17500.75, 0.0], [0.0, 34995.0, 12000.5]])),
}
SKEWER_SETS["sk8"] = dict(axis=SKEWER_SETS["nset"]["axis"][8:9].copy(), cofm=SKEWER_SETS["nset"]["cofm"][8:9].copy())
CONSTANTS = ("velfac", "vmax", "nbins", "dvbin", "rscale", "Hz", "red", "tautail", "kernel_int")


def import_fake_spectra(build=None):
    """The ``fake_spectra.spectra`` module (from ``build`` if given, else the installed package)."""
    if build is not None:
        sys.path.insert(0, str(build))
    from fake_spectra import spectra
    return spectra


def make_spectra(spectra, snap_base, snapnum: int, savedir, sset: str, res: float = 1.0, sf_neutral: bool = False,
                 name: str = "spectra", kernel=None):
    """A fake_spectra ``Spectra`` object for the skewer set ``sset`` (its HDF5 save file goes to ``savedir``)."""
    Path(savedir).mkdir(parents=True, exist_ok=True)
    s = SKEWER_SETS[sset]
    return spectra.Spectra(int(snapnum), str(snap_base), s["cofm"].copy(), s["axis"].copy(), res=res, savefile=f"{name}.hdf5",
                           savedir=str(savedir), reload_file=True, sf_neutral=sf_neutral, kernel=kernel, quiet=True)


def constants(spec) -> dict:
    """fake_spectra's box constants of ``spec``."""
    d = {k: getattr(spec, k) for k in CONSTANTS}
    d.update(PhysDensThresh=spec.gasprop.PhysDensThresh, gamma_UVB=spec.gasprop.gamma_UVB)
    return d


def run_full(spec, species: str = "H:1:1215", products=("tau", "colden")) -> dict:
    """The ``products`` of ``spec`` for ``species`` = "element:stage:line" (e.g. "Si:3:1206"), plus the constants."""
    elem, stage, line = species.split(":")
    stage, line = int(stage), int(line)
    get = {"tau": lambda: spec.get_tau(elem, stage, line), "colden": lambda: spec.get_col_density(elem, stage),
           "temp": lambda: spec.get_temp(elem, stage), "velocity": lambda: spec.get_velocity(elem, stage)}
    out = {p: np.asarray(get[p]()) for p in products}
    out.update(axis=np.asarray(spec.axis), cofm=np.asarray(spec.cofm), **{k: np.asarray(v) for k, v in constants(spec).items()})
    return out


def cell_values(spec, file_names, rows, element: str = "H", stage: int = 1) -> dict:
    """fake_spectra's per-cell inputs (``Spectra._read_particle_data``) for the given (file name, row) cells, in that order.

    The particle filter of ``_read_particle_data`` is skipped (it would misalign the rows); the arithmetic is the same.
    Returns float32 arrays: pos [code], hh [code], elem_den [ion number density x rscale, code], temp [K], vel [km/s].
    """
    ss = spec.snapshot_set
    seg_of = {Path(f).name: i for i, f in enumerate(ss._files)}
    file_names, rows = np.asarray(file_names), np.asarray(rows)
    out = {k: np.zeros((rows.size, 3) if k in ("pos", "vel") else rows.size, np.float32) for k in ("pos", "hh", "elem_den", "temp", "vel")}
    amumass = spec.lines.get_mass(element)
    for name in np.unique(file_names):
        sel = np.nonzero(file_names == name)[0]
        fn, ind = seg_of[str(name)], rows[sel]
        pos = ss.get_data(0, "Position", segment=fn).astype(np.float32)[ind]
        hh = ss.get_smooth_length(0, segment=fn).astype(np.float32)[ind]
        vel = ss.get_peculiar_velocity(0, segment=fn).astype(np.float32)[ind]
        den = spec.gasprop.get_code_rhoH(0, segment=fn).astype(np.float32)[ind]
        temp = spec.gasprop.get_temp(0, segment=fn).astype(np.float32)[ind]
        temp[temp <= 0] = 1
        elem_den = (den * spec.rscale) * spec.get_mass_frac(element, fn, ind)
        if element == "H" and stage == 1:
            elem_den *= spec.gasprop.get_reproc_HI(0, segment=fn)[ind].astype(np.float32)
        else:
            elem_den = elem_den * spec._get_elem_den(element, stage, den, temp, ind, None)
        elem_den /= amumass
        for k, v in dict(pos=pos, hh=hh, elem_den=elem_den, temp=temp, vel=vel).items():
            out[k][sel] = v
    return out


def save_h5(path, d: dict):
    with h5py.File(path, "w") as f:
        for k, v in d.items():
            f.create_dataset(k, data=np.asarray(v))


def load_h5(path) -> dict:
    with h5py.File(path, "r") as f:
        return {k: f[k][()] for k in f}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=["full"])
    ap.add_argument("--build", default=None)
    ap.add_argument("--snap-base", required=True)
    ap.add_argument("--snapnum", type=int, required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--set", dest="sset", default="single", choices=sorted(SKEWER_SETS))
    ap.add_argument("--res", type=float, default=1.0)
    ap.add_argument("--sf-neutral", type=int, default=0)
    ap.add_argument("--species", default="H:1:1215")
    ap.add_argument("--products", default="tau,colden")
    ap.add_argument("--kernel", default="default")
    a = ap.parse_args(argv)
    spectra = import_fake_spectra(a.build)
    spec = make_spectra(spectra, a.snap_base, a.snapnum, Path(a.cache) / "fs_work", a.sset, a.res, bool(a.sf_neutral), f"run_{a.tag}",
                        kernel=None if a.kernel == "default" else a.kernel)
    save_h5(Path(a.cache) / f"fs_{a.tag}.h5", run_full(spec, a.species, tuple(a.products.split(","))))


if __name__ == "__main__":
    main()
