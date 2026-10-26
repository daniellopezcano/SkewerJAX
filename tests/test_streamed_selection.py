"""The streamed selections of a snapshot (``simulations.select_cells_near_paths``, ``select_cells_in_region``) on a small
synthetic Arepo HDF5 snapshot (two files, several blocks, uneven cells): they keep exactly the cells of their rules (brute
force), for paths through the periodic faces, many paths and regions across the faces; and the coarse prefilter of the
paths (``_PathOccupancy``) is conservative. No simulation data needed."""
import tempfile
from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial import cKDTree

from skewerjax.core import selection as sel
from skewerjax.simulations import ArepoHDF5Reader, select_cells_in_region, select_cells_near_paths
from skewerjax.simulations.selection import _PathOccupancy

BOX_CODE, HUBBLE = 10_000.0, 0.7                     # [ckpc/h]: a box of 14.29 cMpc


@lru_cache(maxsize=None)
def _reader(n: int = 30_000) -> ArepoHDF5Reader:
    """Two files of uneven cells: half in a dense slab, volumes from lognormal sizes."""
    rng = np.random.default_rng(5)
    pos = rng.uniform(0.0, BOX_CODE, (n, 3))
    pos[: n // 2, 2] = 0.45 * BOX_CODE + rng.uniform(0.0, 0.1 * BOX_CODE, n // 2)
    vol = (BOX_CODE ** 3 / n) * rng.lognormal(0.0, 1.0, n)
    mass = rng.uniform(0.5, 2.0, n)
    root = Path(tempfile.mkdtemp(prefix="sjx_snap_")) / "snapdir_000"
    root.mkdir()
    for k, rows in enumerate(np.array_split(np.arange(n), 2)):
        with h5py.File(root / f"snap_000.{k}.hdf5", "w") as f:
            h = f.create_group("Header")
            h.attrs.update(dict(Time=0.25, HubbleParam=HUBBLE, BoxSize=BOX_CODE, Omega0=0.3, OmegaLambda=0.7,
                                NumPart_Total=np.array([n, 0, 0, 0, 0, 0]), NumPart_ThisFile=np.array([rows.size, 0, 0, 0, 0, 0])))
            g = f.create_group("PartType0")
            m = rows.size
            for key, val in dict(Coordinates=pos[rows], Masses=mass[rows], Density=mass[rows] / vol[rows], InternalEnergy=np.ones(m),
                                 ElectronAbundance=np.ones(m), NeutralHydrogenAbundance=np.full(m, 1e-5),
                                 GFM_Metals=np.full((m, 10), 1e-3), StarFormationRate=np.zeros(m),
                                 Velocities=rng.normal(0.0, 50.0, (m, 3))).items():
                g.create_dataset(key, data=val)
    return ArepoHDF5Reader(root.parent, 0)


def _all_cells(rd):
    """Every cell's position, size and id (file << 40 | row), unfiltered."""
    pos, size, ids = [], [], []
    for fi in range(len(rd.files)):
        raw = rd.read_rows(fi, ("Coordinates", "Masses", "Density"))
        pos.append(rd.positions_cMpc(raw["Coordinates"]))
        size.append(np.cbrt(rd.volumes_cMpc3(raw["Masses"], raw["Density"])))
        ids.append((np.int64(fi) << 40) | np.arange(len(raw["Masses"]), dtype=np.int64))
    return np.concatenate(pos), np.concatenate(size), np.concatenate(ids)


def _paths(box, n_sk):
    """Curved paths through the periodic faces (unwrapped coordinates)."""
    t = np.linspace(0.0, 1.0, 400)[None, :, None]
    o = np.random.default_rng(n_sk).uniform(0.0, box, (n_sk, 1, 3))
    return o + box * np.concatenate([1.3 * t, 0.4 * t + 0.05 * np.sin(9 * t), 0.1 * np.cos(5 * t)], -1)


def test_streamed_selection_is_the_rule():
    rd = _reader()
    box = rd.header.box_cMpc
    pos, size, ids = _all_cells(rd)
    for n_sk in (1, 3, 40):                                   # one path, a few, enough to cover the whole box
        paths = _paths(box, n_sk)
        tree = cKDTree(sel.wrap_positions(paths.reshape(-1, 3), box), boxsize=box)
        d = tree.query(pos, k=1)[0]
        for mode in ("voronoi", "tophat"):
            brute = ids[d <= sel.selection_radius(size, mode, sel.max_path_spacing(paths))]
            got = select_cells_near_paths(rd, paths, mode=mode, max_rows=7_000)
            np.testing.assert_array_equal(got.ids, brute)       # the same cells, in snapshot order


def test_region_selection_is_the_periodic_box():
    rd = _reader()
    box = rd.header.box_cMpc
    pos, _, ids = _all_cells(rd)
    for lo, hi, margin in (([0.0, 6.0, 6.0], [box, 7.0, 8.0], 0.0), ([box - 1.0, -2.0, 3.0], [box + 2.0, 1.0, 5.0], 0.3),
                           ([2.0, 2.0, 2.0], [4.0, 4.0, 4.0], 1.0)):
        lo, hi = np.asarray(lo), np.asarray(hi)
        inside = np.all(np.mod(pos - (lo - margin), box) < hi - lo + 2 * margin, axis=1)
        got = select_cells_in_region(rd, lo, hi, margin=margin, max_rows=7_000)
        np.testing.assert_array_equal(got.ids, ids[inside])
        np.testing.assert_array_equal(got.positions, pos[inside])


def test_path_prefilter_is_conservative():
    box = 14.0
    rng = np.random.default_rng(2)
    pts = sel.wrap_positions(_paths(box, 2).reshape(-1, 3), box)
    cells = np.concatenate([rng.uniform(0.0, box, (20_000, 3)), pts[::7] + rng.normal(0.0, 0.3, (pts[::7].shape[0], 3))])
    cells = sel.wrap_positions(cells, box)
    d = cKDTree(pts, boxsize=box).query(cells)[0]
    occ = _PathOccupancy(pts, box, n=32)
    for r_max in (0.05, 0.3, 1.0, 2.5):
        cand = occ.candidates(cells, r_max)
        assert cand is not None and np.all(cand[d <= r_max]) and cand.mean() < 0.9   # every cell within r_max is kept
    assert occ.candidates(cells, 0.6 * box) is None            # a dilation over the whole box: no prefilter
    rad = rng.uniform(0.05, 1.0, cells.shape[0])
    full = sel._nearest_distance_cpu(cKDTree(pts, boxsize=box), cells, rad, workers=1)
    pre = sel._nearest_distance_cpu(cKDTree(pts, boxsize=box), cells, rad, workers=-1, candidates=occ.candidates(cells, rad.max()))
    np.testing.assert_array_equal(np.nonzero(pre <= rad)[0], np.nonzero(full <= rad)[0])


def test_header_cosmology_wrap_and_paths():
    """``SnapshotHeader.cosmology()`` is the snapshot's flat LCDM (H0 = 100 h km/s/Mpc in 1/Gyr); ``wrap_positions``
    without a box only casts; ``utils.paths`` follows its environment variables."""
    import os
    import subprocess
    import sys
    from skewerjax.core import phys
    H = _reader().header
    c = H.cosmology()
    assert np.isclose(float(np.ravel(c.H0)[0]), 100 * HUBBLE * 1e6 * phys.Gyr_to_s / phys.Gpc_to_m, rtol=1e-15)
    assert np.isclose(float(np.ravel(c.Omega_m_0)[0]), 0.3) and np.isclose(float(np.ravel(c.Omega_lambda_0)[0]), 0.7)
    x = np.array([[-1.0, 15.0, 3]], dtype=np.int64)
    w = sel.wrap_positions(x, None)
    assert w.dtype == np.float64 and np.array_equal(w, x)
    code = "from skewerjax.utils import paths as p; print(p.DATA_DIR, p.snapshot_dir('X'), p.BENCHMARK_RAW_DIR)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                         env=dict(os.environ, SKEWERJAX_DATA="/data/sims", SKEWERJAX_BENCHMARK_RAW="/tmp/raw")).stdout.split()
    assert out == ["/data/sims", "/data/sims/X", "/tmp/raw"]
