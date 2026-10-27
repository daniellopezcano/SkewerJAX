"""The bookkeeping of ``comparison.benchmark`` (no snapshot, no fake_spectra): exclusive stage timers and their patches,
nested skewer sets, configuration ids, the skip rule of a series, and the rows of a record."""

import json
import tempfile
import time
import types
from pathlib import Path

import numpy as np

from skewerjax.comparison import benchmark as B
from skewerjax.comparison.benchmark import StageTimer, benchmark_skewers, config_id, make_config


def test_stage_timer_is_exclusive():
    T = StageTimer()
    t0 = time.perf_counter()
    with T.stage("physics"):
        time.sleep(0.02)
        with T.stage("io"):
            time.sleep(0.03)
        time.sleep(0.01)
    wall = time.perf_counter() - t0
    assert T.t["io"] >= 0.03 and T.t["physics"] >= 0.03
    assert abs(sum(T.t.values()) - wall) < 5e-3  # stages tile the wall time, no double counting


def test_stage_timer_patches_and_restores():
    class Reader:
        def read(self, x):
            time.sleep(0.01)
            return x + 1

    mod = types.SimpleNamespace(build=lambda: time.sleep(0.01) or "built")
    rd, f = Reader(), mod.build
    T = StageTimer()
    with T.patched((rd, "read", "io"), (mod, "build", "pieces")), T.stage("tau"):
        assert rd.read(1) == 2 and mod.build() == "built"
    assert T.t["io"] >= 0.01 and T.t["pieces"] >= 0.01 and T.t["tau"] < 0.005
    assert "read" not in vars(rd) and mod.build is f  # the instance attribute is gone, the module's restored


def test_skewer_sets_are_nested():
    c16, a16 = benchmark_skewers(16)
    c4, a4 = benchmark_skewers(4)
    assert np.array_equal(c16[:4], c4) and np.array_equal(a16[:4], a4) and set(np.unique(a16)) <= {1, 2, 3}


def test_config_ids_are_unique():
    cs = [
        make_config("fake_spectra", "/x/TNG50-4", 25, 16),
        make_config("skewerjax", "/x/TNG50-4", 25, 16),
        make_config("skewerjax", "/x/TNG50-4", 25, 16, platform="gpu"),
        make_config("skewerjax", "/x/TNG50-4", 25, 16, modes=("float32",)),
        make_config("skewerjax", "/x/TNG50-4", 25, 16, threads=1),
    ]
    assert len({config_id(c) for c in cs}) == len(cs)
    for s in ("scaling", "threads", "smoke"):
        ids = [config_id(c) for c in B.benchmark_sets(s, "/x")]
        assert len(set(ids)) == len(ids)


def _fake_record(c, wall, rss):
    """A minimal successful record of configuration ``c`` (two runs)."""
    run = lambda w: (
        dict(stages=dict.fromkeys(B.STAGES, w / 6), wall=w)
        if c["code"] == "fake_spectra"
        else dict(
            shared=dict.fromkeys(B.SHARED_STAGES, w / 6),
            modes={m: dict(stages=dict(physics=w / 6, pieces=w / 6, tau=w / 6, io=0.0), wall=w / 2) for m in c["modes"]},
            wall=w,
            n_cells=100,
        )
    )
    rec = dict(
        id=config_id(c),
        config=c,
        status="ok",
        wall_s=2 * wall,
        runs=[run(wall), run(wall)],
        peak_rss_GB=rss,
        snapshot=dict(sim="TNG50-4", n_gas=1000, snapnum=25),
    )
    if c["code"] == "skewerjax":
        rec["modes"] = {m: dict(peak_rss_GB=rss, gpu_peak_GB=None, compile_s=0.5, n_absorbers=None) for m in c["modes"]}
    return rec


def test_series_skip_rule():
    """A point is skipped when a smaller N stopped, or when the two previous points extrapolate over the caps."""
    cs = [make_config("fake_spectra", "/x/TNG50-4", 25, n) for n in (1, 4, 16, 64)]
    with tempfile.TemporaryDirectory() as tmp:
        for c, w in zip(cs[:2], (100.0, 400.0)):  # 2 x 1300 s predicted at N = 16 > cap 2000 s
            (Path(tmp) / f"{config_id(c)}.json").write_text(json.dumps(_fake_record(c, w, 1.0)))
        recs = B.run_suite(cs, tmp, cap_s=2000.0, log=lambda m: None)
    assert [r["status"] for r in recs] == ["ok", "ok", "skipped", "skipped"]
    assert "predicted" in recs[2]["reason"] and "stopped" in recs[3]["reason"]
    assert B._predict([(1, 1.0), (4, 2.5)], 16) == 8.5 and B._predict([(1, 1.0)], 16) is None


def test_result_rows_add_shared_stages_to_each_mode():
    c = make_config("skewerjax", "/x/TNG50-4", 25, 4, modes=("matched", "default"))
    rows = B.result_rows(
        [
            _fake_record(c, 6.0, 2.0),
            _fake_record(make_config("fake_spectra", "/x/TNG50-4", 25, 4), 3.0, 0.5),
            dict(id="x", config=make_config("skewerjax", "/x/TNG50-4", 25, 16), status="skipped", reason="test"),
        ]
    )
    ok = B.select_rows(rows, code="skewerjax")
    assert {r["mode"] for r in ok} == {"matched", "default"} and len(rows) == 5
    for r in ok:  # 3 shared stages of 1 s + the mode's 3 s
        assert np.isclose(r["total"][0], 6.0) and np.isclose(sum(v[0] for v in r["stages"].values()), 6.0)
    assert np.isclose(B.value(rows, code="fake_spectra"), 3.0) and np.isnan(B.value(rows, code="skewerjax", n_skewers=16))


def test_snapshot_info_reads_only_the_snapshot_files():
    """Other HDF5 files in the snapshot folder (e.g. a saved fake_spectra grid) are not part of the snapshot."""
    import h5py

    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "TNG50-9" / "snapdir_025"
        d.mkdir(parents=True)
        with h5py.File(d / "grid_nspec16_axis-1_res10.0.hdf5", "w") as f:
            f.create_group("Header")
        for i in (0, 1, 10):
            with h5py.File(d / f"snap_025.{i}.hdf5", "w") as f:
                f.create_group("Header").attrs.update(
                    NumPart_Total=[7, 0], NumPart_Total_HighWord=[1, 0], BoxSize=35000.0, Redshift=3.0
                )
        s = B.snapshot_info(d.parent, 25)
    assert s["n_gas"] == 7 + 2**32 and s["n_files"] == 3 and s["sim"] == "TNG50-9"


def test_presets_budget_and_remeasure():
    """The quick preset: its own caps and estimates, unique ids, no point started beyond the total budget (and none
    written for it); the full preset re-measures a quick record that holds fewer timed runs, the quick one keeps a
    full record."""
    quick = [c for s in ("scaling", "threads") for c in B.benchmark_sets(s, "/x", preset="quick")]
    ids = [config_id(c) for c in quick]
    assert len(set(ids)) == len(ids) and all(c["preset"] == "quick" and c["cap_s"] and c["est_s"] for c in quick)
    full = make_config("fake_spectra", "/x/TNG50-2", 25, 1, repeats=2)
    q = make_config("fake_spectra", "/x/TNG50-2", 25, 1, repeats=1, preset="quick", cap_s=10.0, est_s=1.0)
    rec = _fake_record(q, 1.0, 1.0)
    assert B._remeasure(rec, full) and not B._remeasure(rec, q)
    assert not B._remeasure(_fake_record(full, 1.0, 1.0), q) and B._remeasure(dict(rec, status="error"), q)
    big = make_config("fake_spectra", "/x/TNG50-2", 25, 4, preset="quick", cap_s=10.0, est_s=100.0)
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / f"{config_id(q)}.json").write_text(json.dumps(rec))
        recs = B.run_suite([q, big], tmp, total_budget_s=50.0, log=lambda m: None)
        assert [r["status"] for r in recs] == ["ok", "budget"] and not (Path(tmp) / f"{config_id(big)}.json").exists()
