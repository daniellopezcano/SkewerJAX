"""The memory benchmark's tools (``comparison.memory``): the /proc sampler, the stage peaks and busy cores of a timeline,
the configurations, the accounting of a peak and the gap between two codes, on small synthetic records."""

import time

import numpy as np

from skewerjax.comparison import memory as M
from skewerjax.comparison import scaling as S


def test_proc_readers_and_sampler():
    st = M.read_status()
    assert st["RssAnon"] > 0 and st["VmHWM"] >= st["VmRSS"] > 0
    assert M.mem_available() > 0 and M.cpu_seconds() > 0
    s = M.ProcessSampler(dt=0.01)
    s.mark("work")
    x = np.ones((2000, 2000))  # 32 MB anonymous
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 0.15:
        x = x * 1.0001
    s.mark("done")
    tl = s.stop()
    assert len(tl["t"]) >= 5 and max(tl["anon_GB"]) > 0.03
    pk = M.stage_peaks(tl, s.markers)
    assert set(pk) == {"work", "done"} and pk["work"]["anon_GB"] > 0.0
    t, c = M.busy_cores(tl, window_s=0.05)
    assert np.nanmax(c) > 0.3  # the loop kept a core busy


def test_points_ids_and_fits():
    pts = M.memory_points()
    ids = [p["id"] for p in pts]
    assert len(ids) == len(set(ids)) and "TNG50-2_n128_dv1_fs_search" in ids and "TNG50-2_n128_dv1_cpu_tile256" in ids
    assert next(p for p in pts if p["id"] == "TNG50-2_n128_dv1_cpu")["blocks"] == ["B", "A", "C1", "D"]
    a, b = M.fit_linear([1, 2, 4, 8], [3.0, 5.0, 9.0, 17.0])
    assert abs(a - 1) < 1e-9 and abs(b - 2) < 1e-9
    a, b = S.fit_cost_model([16, 256, 4096, 65536], [300 + 0.01 * n for n in (16, 256, 4096, 65536)])
    assert abs(a - 300) < 1e-6 and abs(b - 0.01) < 1e-9


def _record(entry, n_gas=1000, sjx=False, rate=1.0):
    t = np.linspace(0.0, 10.0, 101)
    tl = dict(
        t=t.tolist(),
        anon_GB=(1 + 0.1 * t).tolist(),
        file_GB=[0.1] * 101,
        shmem_GB=[0.0] * 101,
        hwm_GB=[2.1] * 101,
        avail_GB=[100.0] * 101,
        cpu_s=(rate * t).tolist(),
        read_GB=(t / 10).tolist(),
        gpu_GB=[None] * 101,
    )
    markers = (
        [[0.0, "start"], [1.0, "setup"]]
        + ([[2.0, "pass"], [6.0, "assemble"], [6.5, "tiles"]] if sjx else [[2.0, "io"], [5.0, "physics"], [7.0, "tau"]])
        + [[10.0, "done"]]
    )
    rec = dict(
        id=entry,
        config=dict(entry=entry),
        timeline=tl,
        markers=markers,
        snapshot=dict(n_gas=n_gas, n_files=4),
        read_GB=1.0,
        peak=dict(anon_GB=2.0),
        tau_bytes=8_000_000,
    )
    rec["peaks"] = M.stage_peaks(tl, [tuple(m) for m in markers])
    if sjx:
        rec.update(
            store_bytes={"positions": 24_000, "fields:matched": 16_000},
            run_s=8.0,
            pass_info=dict(
                n_cells=n_gas,
                n_needed=250,
                threads=4,
                timings={"read": 2.0, "read_rest": 1.0, "reach": 0.5, "tiles": 0.1, "physics:matched": 0.4},
                blocks=dict(total=dict(tests=500.0, tests_h=400.0, pairs=300)),
                store_bytes={"positions": 24_000, "fields:matched": 16_000},
            ),
            tiles=[
                dict(
                    n_contributions=[100, 100],
                    n_cells=50,
                    wait_s=0.2,
                    tau_s=3.0,
                    contributions_bytes=1000,
                    engine=[dict(evaluations=10_000, walk_evaluations=40_000), dict(evaluations=1000, walk_evaluations=2000)],
                )
            ],
        )
    else:
        rec["stages"] = dict(setup=0.1, io=3.0, physics=2.0, tau=3.0, selection=0.0, pieces=0.0)
    return rec


def test_accounting_and_gap_table():
    fs, sjx, srch = _record(S.FS), _record(S.CPU_MAT, sjx=True, rate=8.0), _record(M.FS_SEARCH)
    srch["stages"]["tau"] = 1.0
    for r in (fs, sjx):
        pc = M.peak_components(r)
        assert abs(sum(pc["host"].values()) - r["peak"]["anon_GB"]) < 1e-9 and pc["gpu"] is None
    W = M.work_counts(fs, sjx, srch)
    assert W["cells whose physics is computed"] == (2000, 250) and W["chord points prepared on the host"] == (1400, 200)
    rows = M.gap_table(fs, sjx, srch)
    assert [r["stage"] for r in rows] == ["read", "gas physics", "search", "chords", "tau accumulation"]
    for r in rows:
        if np.isfinite(r["fs_s"]):  # the shares split the stage's gap (hardware use: only where the cores are separable)
            assert abs(sum(v for v in r["shares"].values() if np.isfinite(v)) - (r["fs_s"] - r["sjx_s"])) < 1e-9
    assert rows[2]["fs_s"] == 1.0 and rows[4]["fs_s"] == 2.0  # search: the search-only run; tau: the rest of the C call
