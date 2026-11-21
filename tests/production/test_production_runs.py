"""The production launcher (``scripts/production``: ``production_runs``, ``production_sweeps``, ``production_estimates``,
``run_production``): a run's configuration and identity; the sweeps vary one parameter around their baseline; the
estimates from the benchmark records; the states of a run folder (interrupted, damaged, done); sharing by hard links;
the launcher resumes, skips finished runs, refuses to overwrite without --force, refuses when the disk would not fit,
writes nothing in a dry run. The computation is replaced by a fake child (no snapshot); the real child runs on TNG50-4 in
the ``slow`` + ``tng`` test."""

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
    assert "no effect" in d.replace(points_kms=0.25).points_effect()
    assert "dual segments" in d.replace(points_kms=0.25, options={"pieces": "segments", "profile": "midpoint"}).points_effect()
    assert "a(s) table" in d.replace(points_kms=0.25, options={"frame": "exact"}).points_effect()


# --------------------------------------------------------------------------- the sweeps
def _diff(a, b) -> set:
    da, db = a.to_dict(), b.to_dict()
    return {k for k in da if k != "name" and da[k] != db[k]}


def test_each_sweep_varies_one_parameter():
    B = ps.BASELINE
    for name in ("pixels", "skewers", "simulations"):
        _, varied, _ = ps.SWEEPS[name]
        runs = ps.sweep_runs(name)
        assert all(_diff(r, B) <= {varied} for r in runs), name
        assert sum(_diff(r, B) == set() for r in runs) == 1, name  # the baseline is one of the runs
    assert [r.dv_kms for r in ps.sweep_runs("pixels")] == [1.0, 2.0, 4.0, 8.0, 16.0]
    ns = [r.n for r in ps.sweep_runs("skewers")]
    assert ns == [4, 8, 16, 32, 64, 128, 256] and all(b == 2 * a for a, b in zip(ns, ns[1:]))  # nested by doubling
    assert [r.sim for r in ps.sweep_runs("simulations")] == ["TNG50-2", "TNG50-3", "TNG50-4"]
    pts = ps.sweep_runs("points")
    ref = pts[0]
    assert _diff(ref, B) == {"n"} and ref.n == 8
    for fam in ("segments", "exact"):
        fr = [r for r in pts if r.name.startswith(fam)]
        assert [r.points_kms for r in fr] == list(ps.POINTS_KMS) and 0.25 in ps.POINTS_KMS
        assert all(_diff(r, fr[0]) <= {"points_kms"} for r in fr) and _diff(fr[0], ref) == {"points_kms", "options"}
    assert ref.key() == ps.sweep_runs("skewers")[1].key()  # the points reference is the 8^2 run of skewers: shared in "all"
    gpu = ps.sweep_runs("pixels", {"device": "gpu"})
    assert all(r.device == "gpu" for r in gpu)
    raises(ValueError, "cannot be overridden", ps.sweep_runs, "pixels", {"dv_kms": 3.0})
    raises(ValueError, "unknown sweep", ps.sweep_runs, "bogus")
    assert set(ps.ALL_ORDER) == {"points", "pixels", "skewers", "simulations"}


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
    sp = [pe.sampled_time(64, s)["tau_s"] for s in (0.125, 0.5, 4.0)]
    assert sp[0] > sp[1] > sp[2]


# --------------------------------------------------------------------------- run folders
def _fake_done(d, cfg, sweep="t", nbytes=100):
    """A finished run folder: one store file and its record."""
    f = d / pr.STORE_DIR / "x" / "batch_0.h5"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(os.urandom(nbytes))
    rec = dict(
        schema=pr.SCHEMA,
        name=cfg.name,
        sweep=sweep,
        config=cfg.to_dict(),
        key=cfg.key(),
        status="done",
        dataset=None,
        outputs=dict(files=pr.manifest(d), bytes=nbytes),
    )
    pr.write_record(d, rec)
    return rec


def test_the_states_of_a_run_and_sharing(tmp_path):
    cfg = pr.RunConfig(name="a", n=2)
    d = tmp_path / "s" / "a"
    assert pr.run_state(d) == "absent"
    pr.write_record(d, dict(status="running", pid=os.getpid(), host=None))
    assert pr.run_state(d) == "running"
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    pr.write_record(d, dict(status="running", pid=p.pid, host=None))  # its process is gone: killed
    assert pr.run_state(d) == "interrupted"
    pr.write_record(d, dict(status="failed"))
    assert pr.run_state(d) == "failed"
    pr.reset(d, tmp_path)
    _fake_done(d, cfg)
    assert pr.run_state(d) == "done"
    other = tmp_path / "u" / "b"
    pr.reset(other, tmp_path)
    assert pr.find_shareable(tmp_path, cfg.key(), exclude=other) == d
    pr.share(d, other, dict(name="b", sweep="u", config=cfg.replace(name="b").to_dict()))
    rec = pr.read_record(other)
    assert pr.run_state(other) == "done" and rec["shared_from"] == str(d) and rec["name"] == "b"
    fa, fb = d / pr.STORE_DIR / "x" / "batch_0.h5", other / pr.STORE_DIR / "x" / "batch_0.h5"
    assert os.stat(fa).st_ino == os.stat(fb).st_ino  # a hard link: no extra disk
    fa.unlink()
    assert pr.run_state(d) == "damaged"  # a file of the manifest is missing
    raises(RuntimeError, "not inside", pr.reset, tmp_path / "s", tmp_path / "s" / "a")
    (tmp_path / "loose").mkdir()
    (tmp_path / "loose" / "f").write_text("x")
    raises(RuntimeError, "no run.json", pr.reset, tmp_path / "loose", tmp_path)


# --------------------------------------------------------------------------- the launcher, with a fake child
@pytest.fixture
def fake(monkeypatch):
    calls = []

    def launch(self, d, template):
        calls.append(template["name"])
        pr.write_record(d, dict(template, status="launched", pid=os.getpid()))
        _fake_done(d, pr.RunConfig.from_dict(template["config"]), template["sweep"])
        return 0

    monkeypatch.setattr(rp.Launcher, "launch", launch)
    monkeypatch.setattr(rp.signal, "signal", lambda *a: None)  # the test process keeps its handlers
    return calls


def test_the_launcher_resumes_and_refuses(tmp_path, fake, capsys):
    out = tmp_path / "prod"
    assert rp.main(["--sweep", "smoke", "--dry-run", "--out", str(out)]) == 0
    assert not out.exists()  # a dry run writes nothing
    assert rp.main(["--sweep", "smoke", "--out", str(out)]) == 0
    assert fake == ["n_0002", "n_0004"]
    idx = json.loads((out / "smoke" / "index.json").read_text())
    assert [r["state"] for r in idx["runs"]] == ["done", "done"] and idx["varied"] == "n"
    assert (out / "smoke" / "sweep.log").exists()
    assert rp.main(["--sweep", "smoke", "--out", str(out)]) == 0 and fake == ["n_0002", "n_0004"]  # finished: skipped
    d = out / "smoke" / "n_0004"
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    pr.write_record(d, dict(pr.read_record(d), status="running", pid=p.pid))  # killed while running
    assert rp.main(["--sweep", "smoke", "--out", str(out)]) == 0
    assert fake[-1] == "n_0004" and len(fake) == 3  # redone from scratch, alone
    args = ["--run", "one", "--sim", "TNG50-4", "--n", "2", "--dv-kms", "16", "--out", str(out)]
    assert rp.main(args) == 0 and fake[-1] == "one"
    assert rp.main(args) == 3 and "REFUSED" in capsys.readouterr().out  # finished: not overwritten
    assert rp.main([*args, "--force"]) == 0 and fake[-1] == "one" and len(fake) == 5
    assert rp.main(["--run", "two", "--sim", "TNG50-4", "--n", "2", "--dv-kms", "8", "--tile", "8", "--out", str(out)]) == 0
    assert len(fake) == 5 and pr.read_record(out / "single" / "two")["shared_from"].endswith("n_0002")  # linked, not computed
    assert rp.main(["--run", "big", "--n", "8", "--reserve-gb", "1e9", "--out", str(out)]) == 3  # the disk would not fit
    assert "REFUSED" in capsys.readouterr().out and len(fake) == 5
    raises(SystemExit, "inside the repository", rp.resolve_out, rp.REPO / "notebooks" / "x")


# --------------------------------------------------------------------------- the real child, on TNG50-4
@pytest.mark.slow
@pytest.mark.tng
def test_one_run_on_tng50_4(tmp_path):
    import skewer_store as ss

    out = tmp_path / "prod"
    r = subprocess.run(
        [
            sys.executable,
            str(rp.HERE / "run_production.py"),
            "--run",
            "t",
            "--sim",
            "TNG50-4",
            "--n",
            "2",
            "--dv-kms",
            "16",
            "--tile",
            "2",
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    d = out / "single" / "t"
    rec = pr.read_record(d)
    assert rec["status"] == "done" and rec["software"]["git"]["commit"] and rec["snapshot"]["n_gas"] > 0
    st = ss.load_run(d)
    assert st.tau.shape == (2, 4, rec["outputs"]["n_pix"]) and np.isfinite(st.tau).all() and (st.tau >= 0).all()
