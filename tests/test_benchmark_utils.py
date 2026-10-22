"""The bookkeeping of ``comparison.benchmark`` (no snapshot, no fake_spectra): exclusive stage timers, nested skewer
sets, configuration ids."""
import time

import numpy as np

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
    assert abs(sum(T.t.values()) - wall) < 5e-3                         # stages tile the wall time, no double counting


def test_skewer_sets_are_nested():
    c16, a16 = benchmark_skewers(16)
    c4, a4 = benchmark_skewers(4)
    assert np.array_equal(c16[:4], c4) and np.array_equal(a16[:4], a4) and set(np.unique(a16)) <= {1, 2, 3}


def test_config_ids_are_unique():
    cs = [make_config("fake_spectra", "/x/TNG50-4", 25, 16), make_config("skewerjax", "/x/TNG50-4", 25, 16),
          make_config("skewerjax", "/x/TNG50-4", 25, 16, platform="gpu"), make_config("skewerjax", "/x/TNG50-4", 25, 16, dtype="float32")]
    assert len({config_id(c) for c in cs}) == len(cs)
