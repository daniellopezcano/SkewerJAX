"""The scaling benchmark of notebook 14 (``comparison.scaling``) without running it: the nested grids and the reference
skewer, the points of the blocks (each once), the modes, the cold protocol's helpers, the rebinning of the precision
check, the slopes, the records, and its three figures on synthetic records."""

import json

import matplotlib.pyplot as plt
import numpy as np
import pytest

from skewerjax.comparison import scaling as S
from skewerjax.comparison.benchmark import STAGES
from skewerjax.utils import plotting as pl


def test_nested_grids_hold_the_reference_skewer():
    L = 51.7
    for n in (4, 8, 16, 64):
        X, Z = S.nested_grid(n, L)
        i = S.reference_index(n)
        assert X.size == n * n and (X[i], Z[i]) == pytest.approx((L / 4, L / 4))
        Xc, Zc = S.nested_grid(n // 2, L) if n > 4 else (X, Z)
        assert set(zip(np.round(Xc, 9), np.round(Zc, 9))) <= set(zip(np.round(X, 9), np.round(Z, 9))), "nested"
    with pytest.raises(ValueError):
        S.reference_index(6)


def test_points():
    pts = S.scaling_points(128, ("A", "B", "C1"))
    ids = [c["id"] for c in pts]
    assert len(ids) == len(set(ids)), "every point once"
    target = next(c for c in pts if c["sim"] == "TNG50-2" and c["n"] == 128)
    assert set(target["blocks"]) == {"A", "B", "C1"}, "the target belongs to every block"
    assert target["dv_list"] == list(S.PIXELS_KMS), "C1's grids share the target's pass"
    assert sorted(c["n"] for c in pts if "B" in c["blocks"]) == list(S.B_SIDES)
    assert sorted(c["sim"] for c in pts if "A" in c["blocks"]) == sorted(S.SIMULATIONS)
    assert all(set(c["entries"]) == set(S.ENTRIES) for c in pts)
    assert all(c["repeats"] == (2 if c["n"] <= 16 else 1) for c in pts if c["sim"] == "TNG50-2" and c["n"] != 128)
    alone = S.scaling_points(128, ("B",))
    assert [c["n"] for c in alone] == list(S.B_SIDES) and alone[-2]["dv_list"] == [1.0], "without C1: one grid"


def test_mode_options():
    class H:
        a, velfac_kms_per_cMpc = 0.25, 100.0

    d = S.mode_options("default", 100, 1.0, H)
    assert d["pieces"] == "crossings" and d["frame"] == "exact"
    m = S.mode_options("matched", 100, 1.0, H)
    assert m["pieces"] == "chords" and m["frame"] == "snapshot" and m["tau"]["velocity_period"] == 100.0
    assert m["tau"]["tautail"] == 1e-7 and m["tau"]["pixel_average"] and m["tau"]["profile"] == "midpoint"


def test_cold_protocol_helpers(tmp_path):
    (tmp_path / "snap.0.hdf5").write_bytes(b"0" * 4096)
    assert S.evict_page_cache(tmp_path) == 1  # a hint: the file is unchanged
    assert (tmp_path / "snap.0.hdf5").read_bytes() == b"0" * 4096
    m = S.MemorySampler(dt=0.01)
    r = m.read()
    assert r["rss_GB"] > 0 and 0 < r["anon_GB"] <= r["rss_GB"] + 1e-3


def test_rebinning_is_the_pixel_average():
    ef = np.arange(11) * 1.0
    F = np.linspace(0.1, 1.0, 10)
    np.testing.assert_allclose(S.rebin_average(F, ef, ef[::2]), F.reshape(5, 2).mean(1))
    ec = np.array([0.0, 2.5, 5.0])  # not nested: overlap weights
    np.testing.assert_allclose(S.rebin_average(F, ef, ec), [(F[0] + F[1] + 0.5 * F[2]) / 2.5, (0.5 * F[2] + F[3] + F[4]) / 2.5])
    x = np.array([1.0, 2.0, 4.0, 8.0])
    assert S.fit_slope(x, 3 * x**1.5) == pytest.approx(1.5)


def _record(n, scale=1.0, dv_list=(1.0,)):
    st = lambda f: {s: f * (k + 1) for k, s in enumerate(STAGES)}  # noqa: E731
    v = np.linspace(0, 1, 20)
    F = np.exp(-np.sin(5 * v) ** 2)
    c = S.point_config("TNG50-2", n, dv_list=dv_list)
    grids = [f"{d:g}" for d in dv_list]
    ents = [(e, g) for g in grids for e in S.ENTRIES if e != S.CPU_MAT or g == grids[0]]
    return dict(
        id=c["id"],
        config=c,
        snapshot=dict(n_gas=1e9),
        grids={g: dict(nbins=20, dv_kms=float(g)) for g in grids},
        jit_s=0.5,
        entries={f"{e}|{g}": dict(stages=st(scale * (k + 1)), total=0.0) for k, (e, g) in enumerate(ents)},
        reference=dict(flux={f"{e}|{g}": {ln: (F + 1e-7 * k).tolist() for ln in S.LINE_IDS} for k, (e, g) in enumerate(ents)}),
        agreement={f"matched_vs_fs|{g}:HI-LyA": dict(rms_dF=1e-7, max_dF=1e-6) for g in grids},
    )


def test_block_rows():
    R = {}
    for n in S.B_SIDES:
        r = _record(n, n / 4, dv_list=S.PIXELS_KMS if n == 128 else (1.0,))
        R[r["id"]] = r
    rows = S.block_rows(R, "B", 128)
    assert [r["x"] for r in rows] == [n * n for n in S.B_SIDES], "the shared target is found under its C1 id"
    c1 = S.block_rows(R, "C1", 128)
    assert [r["x"] for r in c1] == list(S.PIXELS_KMS)
    assert S.CPU_MAT in c1[0]["stages"] and S.CPU_MAT not in c1[1]["stages"], "the matched entry: first grid only"
    assert set(c1[2]["agreement"]) == {"matched_vs_fs:HI-LyA"} and c1[0]["jit"][S.FS] == 0.0 and c1[0]["jit"][S.CPU_DEF] == 0.5
    assert set(c1[0]["flux"]) == set(S.ENTRIES)


def test_records_round_trip(tmp_path):
    rec = _record(8)
    rec["x"] = 1.234567890123
    p = S.write_record(rec, tmp_path)
    back = S.load_records(tmp_path)[rec["id"]]
    assert back["x"] == pytest.approx(1.23456789, abs=1e-9) and json.loads(p.read_text())["id"] == rec["id"]


def test_scaling_figures():
    pl.use_style()
    R = {}
    for n in (4, 16, 32):
        r = _record(n, scale=n / 4)
        R[r["id"]] = r
    rows = [r["stages"] for r in S.block_rows(R, "B", 128)]
    fig, axs = pl.plot_stage_grid(rows, labels=["a", "b", "c"], jit=[{e: 0.5 for e in S.ENTRIES}] * 3)
    fig.canvas.draw()
    assert axs.shape == (3, 3) and len(fig.legends) == 1 and pl.check_fonts(fig) == []
    plt.close(fig)
    fig, axs = pl.plot_scaling_block([16, 256, 1024], rows, xlabel="skewers")
    fig.canvas.draw()
    assert axs[0, 0].get_xscale() == "log" and pl.check_fonts(fig) == []
    plt.close(fig)
    v = np.linspace(0, 1, 20)
    fl = S.block_rows(R, "B", 128)[0]["flux"]
    cols = [dict(title="t", v=v, style=pl.SCALING_ENTRY_STYLES, F={e: fl[e]["HI-LyA"] for e in S.ENTRIES},
                 dF={e: (np.subtract(fl[e]["HI-LyA"], fl[S.FS]["HI-LyA"]), np.zeros(20)) for e in S.ENTRIES[1:]})]
    fig, axs, stats = pl.plot_flux_checks(cols)
    fig.canvas.draw()
    assert axs.shape == (3, 1) and axs[1, 0].get_yscale() == "log"
    for e in S.ENTRIES[1:]:
        d = np.subtract(fl[e]["HI-LyA"], fl[S.FS]["HI-LyA"])
        rms, mx = stats[("t", e, "Lya")]
        assert np.isclose(rms, np.sqrt(np.mean(d**2))) and np.isclose(mx, np.abs(d).max())
    plt.close(fig)
