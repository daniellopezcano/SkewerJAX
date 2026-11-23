"""The production launcher (``scripts/production``: ``production_runs``, ``production_sweeps``, ``production_estimates``,
``run_production``): a run's configuration and identity; the campaign's sweeps vary one parameter around the fiducial run;
the estimates from the benchmark records and the memory models; the states of a run folder (interrupted, damaged, done);
the launcher resumes, skips finished runs, refuses to overwrite without --force, refuses when the disk would not fit,
waits for memory, lists the failures and goes on, writes nothing in a dry run. The computation is replaced by a fake child
(no snapshot); the real child runs on TNG50-4 in the ``slow`` + ``tng`` test (a lattice, its extension, its reuse)."""

import json
import os
import subprocess
import sys

import numpy as np
import production_estimates as pe
import production_runs as pr
import production_sweeps as ps
import pytest
import run_production as rp
import skewer_extraction as se

from helpers import raises

# --------------------------------------------------------------------------- the configuration


def test_a_run_configuration():
    d = pr.RunConfig()
    assert (d.sim, d.snap, d.lines, d.dv_kms, d.precision, d.axis, d.device) == (
        "TNG50-2",
        25,
        ("HI-LyA", "SiIII-1206"),
        1.0,
        "float32",
        2,
        "cpu",
    )  # the benchmark setup
    d.validate()
    assert pr.RunConfig.from_dict(d.to_dict()) == d
    assert d.key() == d.replace(name="other", threads=4, memory_fraction=0.5).key()  # not what is computed
    for change in (dict(dv_kms=2.0), dict(n=32), dict(sim="TNG50-3"), dict(device="gpu"), dict(options={"voigt": "ExoJAX"})):
        assert d.replace(**change).key() != d.key(), change
    raises(ValueError, "power of two", d.replace(n=48).validate)
    raises(ValueError, "one of them", d.replace(n_pix=100).validate)
    raises(ValueError, "points_kms", d.replace(options={"pieces": "segments"}).validate)
    raises(ValueError, "unknown option", d.replace(options={"nope": 1}).validate)
    raises(ValueError, "unknown run fields", pr.RunConfig.from_dict, {"bogus": 1})
    raises(ValueError, "not both", d.replace(points_kms=1.0, points_cMpc=0.05).validate)
    raises(ValueError, "points_cMpc", d.replace(points_cMpc=-1.0).validate)
    assert (
        d.replace(options={"pieces": "segments"}, points_cMpc=0.05).validate().route() == "sampled" and d.route() == "streaming"
    )
    assert "no effect" in d.replace(points_kms=0.25).points_effect()
    seg = d.replace(points_cMpc=0.01, options={"pieces": "segments", "frame": "exact"}).points_effect()
    assert "dual segments" in seg and "every 0.01 cMpc" in seg and "integrated profile" in seg and "a(s) table" in seg
    ex = d.replace(points_cMpc=0.05, options={"frame": "exact"}).points_effect()
    assert "a(s) table" in ex and "not the pieces" in ex
    assert "dx 0.05 cMpc" in ps.FIDUCIAL.summary()


# --------------------------------------------------------------------------- the sweeps
def _diff(a, b) -> set:
    da, db = a.to_dict(), b.to_dict()
    return {k for k in da if k != "name" and da[k] != db[k]}


def test_each_sweep_varies_one_parameter():
    F = ps.FIDUCIAL
    assert (F.sim, F.snap, F.n, F.axis, F.dv_kms, F.points_cMpc, F.precision, F.device) == (
        "TNG50-2",
        25,
        128,
        2,
        1.0,
        0.05,
        "float32",
        "cpu",
    )
    assert F.options == {"frame": "exact"} and F.route() == "streaming"  # the default route: crossings, integrated, exact frame
    for name in ("pixels", "skewers", "simulations"):
        _, varied, _ = ps.SWEEPS[name]
        runs = ps.sweep_runs(name)
        assert all(_diff(r, F) <= {varied} for r in runs), name
        assert sum(_diff(r, F) == set() for r in runs) == 1, name  # the fiducial is one of the runs
    assert [r.dv_kms for r in ps.sweep_runs("pixels")] == [10.0, 5.0, 1.0, 0.5, 0.1]
    assert [r.n for r in ps.sweep_runs("skewers")] == [32, 64, 128, 256, 512]
    assert [r.sim for r in ps.sweep_runs("simulations")] == ["TNG50-2", "TNG50-3", "TNG50-4"]
    pts = ps.sweep_runs("points")
    assert pts[0].name == "reference" and _diff(pts[0], F) == set()
    assert [r.points_cMpc for r in pts[1:]] == [1.0, 0.5, 0.1, 0.05, 0.01]  # coarse to fine
    for r in pts[1:]:
        assert _diff(r, F) <= {"points_cMpc", "options"} and r.route() == "sampled"
        assert r.options == {"frame": "exact", "pieces": "segments"}  # the integrated profile: the default, not midpoint
    gpu = ps.sweep_runs("pixels", {"device": "gpu"})
    assert all(r.device == "gpu" for r in gpu)
    raises(ValueError, "cannot be overridden", ps.sweep_runs, "pixels", {"dv_kms": 3.0})
    raises(ValueError, "unknown sweep", ps.sweep_runs, "bogus")
    assert ps.ALL_ORDER == ("fiducial", "simulations", "pixels", "points", "skewers")


# --------------------------------------------------------------------------- the estimates
def test_the_estimates():
    assert pe.disk_bytes(16, 2, 1000, 8) == 16 * (8000 + 32) + 2 * pe.HDF5_BATCH_OVERHEAD + 2e5
    t = [pe.streaming_time("TNG50-2", n * n, 1.0)["total_s"] for n in (4, 16, 64, 128, 256)]
    assert all(np.isfinite(t)) and all(b > a for a, b in zip(t, t[1:]))  # more skewers, more time
    dv = [pe.streaming_time("TNG50-2", 128**2, x)["total_s"] for x in (1.0, 4.0, 16.0)]
    assert dv[0] > dv[1] > dv[2]  # wider pixels, a cheaper tau
    sims = [pe.streaming_time(s, 128**2, 1.0)["compute_s"] for s in ("TNG50-2", "TNG50-3", "TNG50-4")]
    assert sims[0] > sims[1] > sims[2]
    rec = pe.load_records()["TNG50-2_n128_dv1-2-4-8-16"]  # the estimate at a measured point is the record
    assert pe.streaming_time("TNG50-2", 128**2, 1.0)["compute_s"] == pytest.approx(
        rec["entries"]["SkewerJAX default, CPU|1"]["total"], rel=0.05
    )
    assert pe.streaming_time("TNG50-2", 128**2, 1.0, device="gpu")["compute_s"] < sims[0]
    fine = [pe.streaming_time("TNG50-2", 128**2, x)["tau_s"] for x in (0.1, 0.5, 1.0)]
    assert fine[0] > fine[1] > fine[2]  # below the records' 1 km/s: extrapolated, still decreasing with dv
    sp = [pe.sampled_time("TNG50-2", 1024, p)["tau_s"] for p in (5168, 1034, 52)]
    assert sp[0] > sp[1] > sp[2]  # more points, more pieces
    n_gas = pe.N_GAS["TNG50-2"]
    c = [se.sampled_cells(4096, n_gas, p, 128) for p in (5168, 517, 52)]
    assert c[0] < c[1] < c[2] <= n_gas  # a wider spacing selects more cells (2.5 h + ds / 2 of every point)
    assert se.sampled_cells(10**6, n_gas, 52) == n_gas  # capped at every cell
    assert se.sampled_cells(4096, n_gas, 52, 128) < se.sampled_cells(4096, n_gas, 52)  # the patch of neighbours bounds it
    m = lambda per, **kw: se.pass_memory(per, n_gas, 2, 3961, **kw)  # noqa: E731
    assert m(1024, points_per_skewer=52) > m(1024, points_per_skewer=5168, n_lattice=128) > se.SAMPLED_BASE_BYTES
    per = se.pass_size(16384, n_gas, 2, 3961, budget=70e9, points_per_skewer=52, n_lattice=128)
    assert 1 <= per < 16384 and m(per, points_per_skewer=52, n_lattice=128) <= 70e9


# --------------------------------------------------------------------------- run folders
def _fake_done(d, cfg, sweep="t", nbytes=100):
    """A finished run folder: one batch file in the campaign's store and the record that points at it."""
    f = pr.store_root(d) / "fake" / cfg.name / "skewers" / "batch_0.h5"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(os.urandom(nbytes))
    rel = os.path.relpath(f, d)
    d.mkdir(parents=True, exist_ok=True)
    rec = dict(
        schema=pr.SCHEMA,
        name=cfg.name,
        sweep=sweep,
        config=cfg.to_dict(),
        key=cfg.key(),
        status="done",
        dataset=None,
        wall_s=1.0,
        timings=dict(computed=max(cfg.n * cfg.n, rp.CALIBRATION_MIN_SKEWERS)),
        outputs=dict(files={rel: nbytes}, bytes=nbytes),
    )
    pr.write_record(d, rec)
    return f


def test_no_pass_exceeds_its_size():
    """The skewers left by an interrupted run (uneven): ``tile_skewers``'s squares exceed the pass size, the passes of
    ``bounded_tiles`` do not, and every skewer is in exactly one pass."""
    from skewerjax.core import StraightPaths
    from skewerjax.simulations import tile_skewers

    n, L, per = 64, 51.668, 400
    ij = np.array([(i, j) for i in range(n) for j in range(n) if not (i < 35 and j < 45)])
    cofm = se.ss.lattice_positions(ij * (4096 // n), 2, L, 4096)
    paths = StraightPaths(cofm, np.eye(3)[np.full(len(ij), 1)], np.full(len(ij), L), L, 1)
    assert np.bincount(tile_skewers(paths, per)).max() > per
    g = se.bounded_tiles(paths, per)
    assert g.shape == (len(ij),) and np.bincount(g).max() <= per and np.bincount(g).min() > 0
    assert se.bounded_tiles(paths, len(ij)).max() == 0


def test_the_states_of_a_run(tmp_path):
    cfg = pr.RunConfig(name="a", n=2)
    d = tmp_path / "s" / "a"
    assert pr.run_state(d) == "absent" and pr.store_root(d) == (tmp_path / "store").resolve()
    pr.write_record(d, dict(status="running", pid=os.getpid(), host=None))
    assert pr.run_state(d) == "running"
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    pr.write_record(d, dict(status="running", pid=p.pid, host=None))  # its process is gone: killed
    assert pr.run_state(d) == "interrupted"
    pr.write_record(d, dict(status="failed"))
    assert pr.run_state(d) == "failed"
    pr.reset(d, tmp_path)
    f = _fake_done(d, cfg)
    assert pr.run_state(d) == "done"
    f.unlink()
    assert pr.run_state(d) == "damaged"  # a batch file of the manifest is missing
    pr.reset(d, tmp_path)
    assert not (d / pr.RUN_FILE).exists() and (tmp_path / "store").is_dir()  # the store is kept
    raises(RuntimeError, "not inside", pr.reset, tmp_path / "s", tmp_path / "s" / "a")
    (tmp_path / "loose").mkdir()
    (tmp_path / "loose" / "f").write_text("x")
    raises(RuntimeError, "no run.json", pr.reset, tmp_path / "loose", tmp_path)
    raises(RuntimeError, "not a dataset", pr.remove_dataset, tmp_path / "loose", tmp_path)
    assert pr.peak_rss_GB() > 0


# --------------------------------------------------------------------------- the launcher, with a fake child
@pytest.fixture
def fake(monkeypatch):
    calls = []

    def launch(self, d, template, action="compute"):
        calls.append(template["name"])
        cfg = pr.RunConfig.from_dict(template["config"])
        if cfg.name == "boom":  # a run that fails
            pr.write_record(d, dict(template, status="failed", error="RuntimeError: boom"))
            return 1
        pr.write_record(d, dict(template, status="launched", pid=os.getpid()))
        _fake_done(d, cfg, template["sweep"])
        rec = pr.read_record(d)
        pr.write_record(d, dict(rec, estimate=template["estimate"], wall_s=2.0 * template["estimate"]["wall_s_raw"]))
        return 0

    monkeypatch.setattr(rp.Launcher, "launch", launch)
    monkeypatch.setattr(rp.signal, "signal", lambda *a: None)  # the test process keeps its handlers
    return calls


def test_the_launcher_resumes_and_refuses(tmp_path, fake, capsys, monkeypatch):
    out = tmp_path / "prod"
    assert rp.main(["--sweep", "smoke", "--dry-run", "--out", str(out)]) == 0
    assert not out.exists()  # a dry run writes nothing
    assert "dry run:" in capsys.readouterr().out
    assert rp.main(["--sweep", "smoke", "--out", str(out)]) == 0
    assert fake == ["n_0002", "n_0004", "dx_0p5cMpc"]
    idx = json.loads((out / "smoke" / "index.json").read_text())
    assert [r["state"] for r in idx["runs"]] == ["done"] * 3 and idx["varied"] == "n" and idx["store"] == "../store"
    assert (out / "smoke" / "sweep.log").exists()
    assert rp.main(["--sweep", "smoke", "--out", str(out)]) == 0 and len(fake) == 3  # finished: skipped
    d = out / "smoke" / "n_0004"
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    pr.write_record(d, dict(pr.read_record(d), status="running", pid=p.pid))  # killed while running
    assert rp.main(["--sweep", "smoke", "--out", str(out)]) == 0
    assert fake[-1] == "n_0004" and len(fake) == 4  # redone, alone
    # the estimates are calibrated by the finished runs (the fake ones took twice their estimate)
    f, m = rp.calibration(out, "streaming")
    assert m >= 2 and f == pytest.approx(2.0)
    args = ["--run", "one", "--sim", "TNG50-4", "--n", "2", "--dv-kms", "16", "--out", str(out)]
    assert rp.main(args) == 0 and fake[-1] == "one"
    assert rp.main(args) == 3 and "REFUSED" in capsys.readouterr().out  # finished: not overwritten
    assert rp.main([*args, "--force"]) == 0 and fake[-1] == "one" and len(fake) == 6
    assert rp.main(["--run", "big", "--n", "8", "--reserve-gb", "1e9", "--out", str(out)]) == 3  # the disk would not fit
    assert "REFUSED" in capsys.readouterr().out and len(fake) == 6
    # a failed run is recorded, the next one starts, the failures are listed at the end
    monkeypatch.setattr(
        rp.ps, "SWEEPS", dict(rp.ps.SWEEPS, t=(lambda: [pr.RunConfig(name="boom", n=2), pr.RunConfig(name="ok", n=4)], "n", "t"))
    )
    assert rp.main(["--sweep", "t", "--out", str(out)]) == 1
    assert fake[-2:] == ["boom", "ok"] and pr.run_state(out / "t" / "ok") == "done"
    assert "FAILED runs (1): t/boom (failed" in capsys.readouterr().out
    raises(SystemExit, "inside the repository", rp.resolve_out, rp.REPO / "notebooks" / "x")


def test_the_launcher_waits_for_memory(tmp_path, monkeypatch):
    L = rp.Launcher(tmp_path, type("A", (), dict(force=False))())
    row = dict(
        cfg=pr.RunConfig(name="m", sim="TNG50-4", n=4),
        est=dict(min_pass_GB=10.0, n_new=16, n_pix=400, n_passes=1, points_per_skewer=None),
    )
    seen = iter([1e9, 2e9, 1e12])  # MemAvailable: too little twice, then enough
    monkeypatch.setattr(se, "memory_available", lambda: next(seen))
    monkeypatch.setattr(rp.time, "sleep", lambda s: None)
    logged = []
    monkeypatch.setattr(L, "log", logged.append)
    assert L.wait_for_memory(row)
    assert any("waiting for memory" in x for x in logged) and "1 pass(es)" in logged[-1]


# --------------------------------------------------------------------------- the real child, on TNG50-4
@pytest.mark.slow
@pytest.mark.tng
def test_runs_on_tng50_4(tmp_path):
    """The real child on TNG50-4: a 2^2 lattice, its extension to 4^2 (12 skewers computed, the 4 of 2^2 reused, the
    coarser lattice read back bit for bit), and a run of the 2^2 lattice again under another name (nothing computed)."""
    import skewer_store as ss

    out = tmp_path / "prod"

    def run(name, n):
        r = subprocess.run(
            [
                sys.executable,
                str(rp.HERE / "run_production.py"),
                "--run",
                name,
                "--sim",
                "TNG50-4",
                "--n",
                str(n),
                "--dv-kms",
                "16",
            ]
            + ["--points-cMpc", "0.05", "--opt", "frame=exact", "--tile", "4", "--out", str(out)],
            capture_output=True,
            text=True,
            timeout=900,
        )
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
        return pr.read_record(out / "single" / name)

    a = run("a", 2)
    assert a["status"] == "done" and a["software"]["git"]["commit"] and a["snapshot"]["n_gas"] > 0
    assert a["timings"]["computed"] == 4 and a["dataset"]["root"] == "../../store"
    s = a["sampling"]
    assert s["points_cMpc"] == 0.05 and s["points_kms"] == pytest.approx(0.05 * s["u_kms_per_cMpc"]) and s["frame"] == "exact"
    assert s["dv_kms"] * s["n_pix"] == pytest.approx(s["period_kms"]) and abs(s["dv_kms"] - 16) < 0.5  # nominal 16 km/s
    b = run("b", 4)
    assert b["dataset"]["hash"] == a["dataset"]["hash"] and b["timings"]["computed"] == 12 and b["timings"]["reused"] == 4
    c = run("c", 2)
    assert c["timings"]["computed"] == 0
    st2, st4 = ss.load_run(out / "single" / "a"), ss.load_run(out / "single" / "b")
    assert st4.tau.shape == (2, 16, b["outputs"]["n_pix"]) and np.isfinite(st4.tau).all() and (st4.tau >= 0).all()
    assert np.array_equal(ss.load_run(out / "single" / "b", 2).tau, st2.tau)  # the coarser lattice, the same skewers
