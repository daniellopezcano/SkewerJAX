"""The skewer store (``scripts/production/skewer_store.py``) and the extraction into it (``skewer_extraction.py``):
the lattices nest; doubling N recomputes nothing already stored and leaves the coarser lattice bit for bit; an interrupted
batch leaves no corrupt file and is redone; another option is another dataset; tau and the metadata round-trip. Everything
is written to a temporary directory. The toy tests compute tau on a toy box (``ToyBoxes``) through
``core.compute_optical_depth``; the ``tng`` test runs the streaming route on TNG50-4 and checks it against notebook 11's
pipeline for the same skewers and options."""

import json
import os
import types

import numpy as np
import pytest
import skewer_extraction as se  # scripts/production, on sys.path by conftest.py
import skewer_store as ss

from skewerjax.config import get_precision
from skewerjax.core import LineBundle, SkewerBundle, compute_optical_depth, lattice_columns
from skewerjax.simulations import ToyBoxes
from skewerjax.utils.paths import snapshot_dir

from helpers import raises, tol

N_REF = 16  # the finest lattice of the toy datasets (= the toy grid: the skewers through its nodes)
TOY = ToyBoxes(box_cMpc=20.0, n_cells=16, n_real=1, bao_ratio=0.0, velocity="linear")
LINE_IDS = ("HI-LyA", "SiIII-1206")


def _toy_header():
    return types.SimpleNamespace(
        a=1.0 / (1.0 + TOY.z),
        velfac_kms_per_cMpc=float(TOY.aH),
        box_cMpc=float(TOY.box_cMpc),
        cosmology=TOY.cosmology,
        redshift=TOY.z,
    )


def _toy_config(axis=2, spacing_kms=8.0, overrides=None, n_ref=N_REF):
    H = _toy_header()
    opts = se.dataset_options(
        H, mode="default", overrides=overrides, spacing_kms=spacing_kms, line_ids=LINE_IDS, precision=get_precision(), n_ref=n_ref
    )
    snap = dict(
        redshift=H.redshift, a=H.a, h=0.7, box_cMpc=H.box_cMpc, u_kms_per_cMpc=H.velfac_kms_per_cMpc, n_gas=16**3, n_files=1
    )
    return dict(simulation=dict(name="toy", path="toy", snapshot=0, axis=axis), preset="default", options=opts, snapshot=snap)


class ToyCompute:
    """``extract``'s ``compute`` on the toy box: tau of the requested skewers by ``compute_optical_depth``, one batch per
    ``tile_size`` skewers; counts the calls and the skewers; ``fail_after`` raises after that many batches."""

    def __init__(self, fail_after=None):
        self.calls, self.skewers, self.batches, self.fail_after = [], 0, 0, fail_after
        self.cells = TOY.cells(0)

    def tau(self, cfg, idx):
        o, s = cfg["options"], cfg["simulation"]
        L = cfg["snapshot"]["box_cMpc"]
        cofm = ss.lattice_positions(idx, s["axis"], L, o["lattice"]["n_ref"])
        b = SkewerBundle.axis_aligned(cofm, s["axis"], L, L / 64, TOY.cosmology(), a_ref=cfg["snapshot"]["a"])
        lines = LineBundle(list(o["lines"]["line_ids"]), ref_line_id=o["lines"]["ref_line_id"])
        v = (np.arange(o["pixels"]["n_pix"]) + 0.5) * o["pixels"]["dv_kms"]
        t = o["tau"]
        return np.asarray(
            compute_optical_depth(
                b, v, lines, self.cells, line_frame=t["line_frame"], velocity_period=t["velocity_period"], voigt=t["voigt"]
            )
        )

    def __call__(self, reader, cfg, idx, *, tile_size, devices, threads, on_batch, on_pass):
        self.calls.append(np.asarray(idx).copy())
        on_pass(dict(pass_s=0.0, n_cells_within_reach=0, store_GB=0.0, n_tiles=-(-len(idx) // tile_size)))
        tau = self.tau(cfg, idx)
        for a in range(0, len(idx), tile_size):
            if self.fail_after is not None and self.batches >= self.fail_after:
                raise KeyboardInterrupt("simulated interruption")
            on_batch(
                idx[a : a + tile_size],
                tau[:, a : a + tile_size],
                dict(resolution=None, geometry_s=0, contributions_s=0, tau_s={}),
            )
            self.batches += 1
            self.skewers += len(idx[a : a + tile_size])
        return tau, dict(pass_s=0.0, tiles_s=0.0)


def _extract(ds, ns, compute, tile=4):
    return se.extract(ds, None, ns, tile_size=tile, compute=compute, log=lambda *a: None, skewers_per_pass=10**6)


# --------------------------------------------------------------------------- the lattice
def test_the_lattices_nest():
    for n in (1, 2, 4, 8):
        coarse, fine = ss.lattice_indices(n, 16), ss.lattice_indices(2 * n, 16)
        assert set(map(tuple, coarse)) <= set(map(tuple, fine))
        assert np.array_equal(coarse, lattice_columns(n, (1, 1)) * (16 // n))  # the order of core.lattice_columns
    for bad in (3, 0, 6, 32, 2.5, "4"):
        raises(ValueError, "", ss.check_n, bad, 16)
    assert ss.check_n(4.0, 16) == 4
    p = ss.lattice_positions([[4, 8]], 2, 20.0, 16)  # axis y: (x, z) transverse
    assert np.allclose(p, [[5.0, 0.0, 10.0]])


# --------------------------------------------------------------------------- reuse
def test_doubling_reuses_every_skewer(tmp_path):
    ds = ss.SkewerDataset.create(tmp_path, _toy_config())
    comp = ToyCompute()
    rec2 = _extract(ds, [2], comp)
    assert comp.skewers == 4 and len(comp.calls) == 1 and rec2["status"] == "done"
    first = ds.load(2)
    rec = _extract(ds, [2, 4], comp)
    assert comp.skewers == 4 + 12 and len(comp.calls) == 2  # the 12 new skewers only
    assert not set(map(tuple, comp.calls[1])) & set(map(tuple, ss.lattice_indices(2, N_REF)))
    assert [(x["n"], x["reused"], x["computed"]) for x in rec["lattices"]] == [(2, 4, 0), (4, 4, 12)]
    _extract(ds, [4], comp)
    assert len(comp.calls) == 2  # nothing to compute
    again = ds.load(2)
    assert np.array_equal(again.tau, first.tau)  # the coarser lattice, bit for bit
    four = ds.load(4)
    assert np.array_equal(four.tau[:, np.isin(np.arange(16), [0, 2, 8, 10])], first.tau)  # its stride-2 subset
    assert ds.largest_complete() == 4 and ds.complete_ns() == [1, 2, 4]
    src = ds.sources(4)  # the levels of the 4 x 4 lattice: 1 + 3 + 12 skewers, the 2 x 2 ones in the first run's batches
    assert [(r["n"], r["skewers"], r["stored"]) for r in src] == [(1, 1, 1), (2, 3, 3), (4, 12, 12)]
    assert set(src[0]["files"]) | set(src[1]["files"]) <= {f.name for f in ds.batch_files()}
    assert not set(src[1]["files"]) & set(src[2]["files"])  # N = 2 and the 12 new skewers of N = 4: different runs


def test_the_stored_tau_is_the_computed_one(tmp_path):
    ds = ss.SkewerDataset.create(tmp_path, _toy_config())
    comp = ToyCompute()
    _extract(ds, [4], comp, tile=5)
    st = ds.load(4)
    ref = comp.tau(ds.config, ss.lattice_indices(4, N_REF))
    assert np.array_equal(st.tau, ref.astype(np.float32))  # stored as float32, nothing else
    assert st.tau.dtype == np.float32 and st.tau.shape == (2, 16, ds.n_pix)


# --------------------------------------------------------------------------- interruptions
def test_an_interrupted_run_leaves_no_corrupt_file(tmp_path, monkeypatch):
    ds = ss.SkewerDataset.create(tmp_path, _toy_config())
    with pytest.raises(KeyboardInterrupt):
        _extract(ds, [4], ToyCompute(fail_after=2), tile=4)  # two batches of four written, then the interruption
    assert len(ds.batch_files()) == 2 and len(ds.index(refresh=True)) == 8
    assert ds.runs()[-1]["status"] == "interrupted"
    # a batch interrupted between its temporary file and the rename: the temporary file is ignored, then removed
    real = os.replace
    monkeypatch.setattr(
        ss.os, "replace", lambda a, b: (_ for _ in ()).throw(KeyboardInterrupt()) if "batch_" in str(b) else real(a, b)
    )
    with pytest.raises(KeyboardInterrupt):
        _extract(ds, [4], ToyCompute(), tile=4)
    monkeypatch.setattr(ss.os, "replace", real)
    tmp = list((ds.path / "skewers").glob(".tmp_*"))
    assert len(tmp) == 1 and len(ds.index(refresh=True)) == 8
    # a corrupt batch (e.g. a file truncated by hand): ignored with a warning, its skewers recomputed
    (ds.path / "skewers" / "batch_zzz_corrupt.h5").write_bytes(b"not an HDF5 file")
    comp = ToyCompute()
    with pytest.warns(UserWarning, match="ignored"):
        _extract(ds, [4], comp, tile=4)
    assert comp.skewers == 8 and not list((ds.path / "skewers").glob(".tmp_*"))
    good = ToyCompute().tau(ds.config, ss.lattice_indices(4, N_REF)).astype(np.float32)
    with pytest.warns(UserWarning, match="ignored"):
        assert np.array_equal(ds.load(4).tau, good)


def test_one_writer_at_a_time(tmp_path):
    ds = ss.SkewerDataset.create(tmp_path, _toy_config())
    other = ss.SkewerDataset.open(tmp_path, "toy", 0, 2, ds.hash)
    with ds.lock():
        raises(RuntimeError, "another process", other.lock().__enter__)


# --------------------------------------------------------------------------- identity
def test_another_option_is_another_dataset(tmp_path):
    base = _toy_config()
    h = ss.config_hash(base["options"])
    assert ss.config_hash(_toy_config(axis=3)["options"]) == h  # the axis is the directory, not the options
    assert ss.config_hash(_toy_config()["options"]) == h  # deterministic
    others = [
        _toy_config(spacing_kms=4.0),
        _toy_config(overrides={"voigt": "ExoJAX"}),
        _toy_config(overrides={"physics.x_HI_model": "stored"}),
        _toy_config(overrides={"line_frame": "own"}),
        _toy_config(n_ref=32),
    ]
    hashes = {ss.config_hash(c["options"]) for c in others}
    assert len(hashes) == len(others) and h not in hashes
    a = ss.SkewerDataset.create(tmp_path, base)
    b = ss.SkewerDataset.create(tmp_path, others[0])
    assert a.path != b.path and a.path.parent == b.path.parent
    assert ss.SkewerDataset.create(tmp_path, base).path == a.path  # the same options: the same dataset
    raises(ValueError, "unknown option", se.mode_options, "default", {"not_an_option": 1})
    raises(ValueError, "spacing_kms or by n_pix", se.dataset_options, _toy_header(), spacing_kms=1.0, n_pix=100)
    assert se.parse_override("physics.x_HI_model=stored") == ("physics.x_HI_model", "stored")
    assert se.parse_override("tautail=1e-7") == ("tautail", 1e-7)


def test_the_presets_are_the_comparison_modes():
    from skewerjax.comparison import benchmark, scaling

    assert se.DEFAULT_CELLS == scaling.DEFAULT_CELLS and se.MATCHED_CELLS == benchmark.MATCHED_CELLS
    H = _toy_header()
    o = se.dataset_options(H, mode="matched", spacing_kms=8.0, precision=get_precision())
    mt = scaling.matched_tau(o["pixels"]["period_kms"], H.a, H.velfac_kms_per_cMpc)
    for k in (
        "voigt",
        "detuning",
        "a_eff",
        "hubble_flow",
        "aH_ref",
        "b_constants",
        "pixel_average",
        "tautail",
        "velocity_period",
    ):
        assert o["tau"][k] == pytest.approx(mt[k]) if isinstance(mt[k], float) else o["tau"][k] == mt[k], k
    assert o["tau"]["profile"] == "midpoint" and o["pieces"] == dict(pieces="chords", chord_weight=0.125)


# --------------------------------------------------------------------------- round trip
def test_the_round_trip(tmp_path):
    cfg = _toy_config()
    ds = ss.SkewerDataset.create(tmp_path, cfg)
    _extract(ds, [2, 4], ToyCompute(), tile=3)
    saved = json.loads((ds.path / "config.json").read_text())
    assert saved["options"] == json.loads(json.dumps(cfg["options"])) and saved["hash"] == ds.hash == ds.path.name
    st = ss.load_skewers(tmp_path, "toy", 0, 2, ds.hash[:6], 4)  # a prefix of the hash
    o = cfg["options"]
    assert st.line_ids == LINE_IDS and st.n_per_axis == 4 and st.hash == ds.hash
    assert np.array_equal(st.indices, ss.lattice_indices(4, N_REF)) and np.array_equal(st.columns, lattice_columns(4, (1, 1)))
    assert np.allclose(st.v, (np.arange(o["pixels"]["n_pix"]) + 0.5) * o["pixels"]["dv_kms"])
    assert st.period == pytest.approx(st.v.size * st.dv) and st.L == TOY.box_cMpc and st.u == pytest.approx(TOY.aH)
    assert np.allclose(st.positions[:, [0, 2]], st.columns * (TOY.box_cMpc / 4)) and np.all(st.positions[:, 1] == 0)
    st_cfg = ss.load_skewers(tmp_path, "toy", 0, 2, cfg, 2)  # by its config
    assert np.array_equal(st_cfg.tau, st.tau[:, [0, 2, 8, 10]])
    import h5py

    with h5py.File(ds.batch_files()[0], "r") as h:
        assert h.attrs["config_hash"] == ds.hash and json.loads(h.attrs["line_ids"]) == list(LINE_IDS)
        assert h.attrs["precision"] == get_precision() and h["tau/HI-LyA"].dtype == np.float32
    runs = ds.runs()
    assert len(runs) == 1 and runs[0]["status"] == "done" and runs[0]["computed"] == 16
    assert set(runs[0]["versions"]) >= {"skewerjax", "jax", "fake_spectra"} and "commit" in runs[0]["git"]
    rows = ss.list_datasets(tmp_path)
    assert len(rows) == 1 and rows[0]["n_max"] == 4 and rows[0]["n_stored"] == 16
    assert "toy" in ss.format_datasets(rows)
    raises(ValueError, "not complete", ss.load_skewers, tmp_path, "toy", 0, 2, ds.hash, 8)
    raises(FileNotFoundError, "no dataset", ss.load_skewers, tmp_path, "toy", 0, 3, ds.hash, 2)
    p = se.plan(tmp_path, cfg, [4, 8])
    assert p["exists"] and [r["new"] for r in p["rows"]] == [0, 48] and p["new_total"] == 48


# --------------------------------------------------------------------------- the snapshot
@pytest.mark.tng
def test_the_extraction_of_tng50_4(tmp_path):
    """N = 2 then 4 by the streaming route on TNG50-4: the 4 skewers of N = 2 reused; tau against notebook 11's pipeline
    (every cell read, ``gas_to_cells``, ``cell_pieces`` on ``SimulationSkewers.axis_aligned``, ``compute_optical_depth``)
    for the same skewers and options; and against a direct N = 4 run (other tiles)."""
    import warnings

    from skewerjax.config import devices
    from skewerjax.core.absorbers import cell_pieces
    from skewerjax.simulations import ArepoHDF5Reader, SimulationSkewers, gas_to_cells, select_cells_near_paths

    reader = ArepoHDF5Reader(snapshot_dir("TNG50-4"), 25)
    H = reader.header
    opts = se.dataset_options(H, spacing_kms=4.0, precision=get_precision())
    ds = ss.SkewerDataset.create(tmp_path / "a", se.dataset_config(reader, "TNG50-4", 25, 2, opts))
    log = []
    se.extract(ds, reader, [2], devices=devices("cpu"), log=log.append)
    rec = se.extract(ds, reader, [2, 4], devices=devices("cpu"), log=log.append)
    assert [(x["reused"], x["computed"]) for x in rec["lattices"]] == [(4, 0), (4, 12)]
    st = ds.load(4)
    direct = ss.SkewerDataset.create(tmp_path / "b", se.dataset_config(reader, "TNG50-4", 25, 2, opts))
    se.extract(direct, reader, [4], devices=devices("cpu"), log=log.append)
    F, Fd = np.exp(-st.tau.astype(float)), np.exp(-direct.load(4).tau.astype(float))
    assert np.max(np.abs(F - Fd)) < tol(1e-7, 1e-6)  # the tiles do not change tau beyond rounding
    t = opts["tau"]
    lines = LineBundle(list(st.line_ids), ref_line_id=st.line_ids[0])
    sk = SimulationSkewers.axis_aligned(reader, st.positions / H.code_length_cMpc, np.full(16, 2), 1.0)
    gas = select_cells_near_paths(reader, sk.points, mode="voronoi")
    cells = gas_to_cells(gas, lines.species_keys, **opts["physics"])
    pieces = cell_pieces(cells, lines, points=sk.points, frame=sk.frame("snapshot"), a_data=H.a)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = compute_optical_depth(
            pieces, st.v, lines, line_frame=t["line_frame"], velocity_period=True, voigt=t["voigt"], detuning=t["detuning"],
            hubble_flow=t["hubble_flow"], aH_ref=t["aH_ref"], a_eff=t["a_eff"], b_constants=t["b_constants"],
        )  # fmt: skip
    Fr = np.exp(-np.asarray(ref, float))
    assert np.max(np.abs(F - Fr)) < tol(2e-6, 5e-5)  # the stream store keeps n, T, v in float32 (test_stream.py)
