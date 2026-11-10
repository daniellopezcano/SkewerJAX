"""The scaling benchmark of notebook 14 (``comparison.scaling``) without running it: the nested grids and the reference
skewer, the batches, the points of the blocks (each once), the rebinning of the precision check, the slopes, the records,
and its three figures on synthetic records."""

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


def test_batches_and_points():
    b = S.batches(16384)
    assert len(b) == 16 and b[0] == slice(0, S.BATCH) and b[-1].stop == 16384
    assert len(S.batches(16384, 0.125)) == 32 and len(S.batches(100)) == 1
    pts = S.scaling_points(64)
    ids = [c["id"] for c in pts]
    assert len(ids) == len(set(ids)), "every point once"
    shared = next(c for c in pts if c["id"] == S.point_config("TNG50-2", 64)["id"])
    assert set(shared["blocks"]) == {"A", "B", "C1", "C2"}, "the target belongs to every block"
    c2 = [c for c in pts if "C2" in c["blocks"]]
    assert sorted(c["spacing_kms"] for c in c2) == sorted(S.SPACINGS_KMS)
    assert all(S.FS not in c["entries"] for c in c2 if c["spacing_kms"] != S.DEFAULT_SPACING_KMS)
    assert sorted(c["n"] for c in pts if "B" in c["blocks"]) == [4, 8, 16, 32, 64]


def test_rebinning_is_the_pixel_average():
    ef = np.arange(11) * 1.0
    F = np.linspace(0.1, 1.0, 10)
    np.testing.assert_allclose(S.rebin_average(F, ef, ef[::2]), F.reshape(5, 2).mean(1))
    ec = np.array([0.0, 2.5, 5.0])  # not nested: overlap weights
    np.testing.assert_allclose(S.rebin_average(F, ef, ec), [(F[0] + F[1] + 0.5 * F[2]) / 2.5, (0.5 * F[2] + F[3] + F[4]) / 2.5])
    x = np.array([1.0, 2.0, 4.0, 8.0])
    assert S.fit_slope(x, 3 * x**1.5) == pytest.approx(1.5)


def _record(n, scale=1.0):
    st = lambda f: {s: f * (k + 1) for k, s in enumerate(STAGES)}  # noqa: E731
    v = np.linspace(0, 1, 20)
    F = np.exp(-np.sin(5 * v) ** 2)
    return dict(
        id=f"TNG50-2_n{n}",
        config=S.point_config("TNG50-2", n),
        snapshot=dict(n_gas=1e9),
        pixels=dict(nbins=20, dv_kms=1.0),
        entries={e: dict(stages=st(scale * (k + 1)), total=0.0, jit_s=0.5) for k, e in enumerate(S.ENTRIES)},
        reference=dict(flux={e: {ln: (F + 1e-7 * k).tolist() for ln in S.LINE_IDS} for k, e in enumerate(S.ENTRIES)}),
    )


def test_records_round_trip(tmp_path):
    rec = _record(8)
    rec["x"] = 1.234567890123
    p = S.write_record(rec, tmp_path)
    back = S.load_records(tmp_path)[rec["id"]]
    assert back["x"] == pytest.approx(1.23456789, abs=1e-9) and json.loads(p.read_text())["id"] == rec["id"]


def test_scaling_figures():
    pl.use_style()
    recs = [_record(n, scale=n / 4) for n in (4, 8, 16)]
    rows = [{e: r["entries"][e]["stages"] for e in S.ENTRIES} for r in recs]
    fig, axs = pl.plot_stage_grid(rows, labels=["a", "b", "c"], jit=[{e: 0.5 for e in S.ENTRIES}] * 3)
    fig.canvas.draw()
    assert axs.shape == (3, 3) and len(fig.legends) == 1 and pl.check_fonts(fig) == []
    plt.close(fig)
    fig, axs = pl.plot_scaling_block([4, 8, 16], rows, xlabel="skewers")
    fig.canvas.draw()
    assert axs[0, 0].get_xscale() == "log" and "slope 1.00" in axs[0, 0].get_legend().get_texts()[-1].get_text()
    assert pl.check_fonts(fig) == []
    plt.close(fig)
    fig, axs = pl.plot_scaling_block([4, 8, 16], rows, xlabel="skewers", reference_total=10.0, entries=list(S.SKEWERJAX_ENTRIES))
    assert not axs[1, 0].get_visible() and any(ln.get_linestyle() == ":" for ln in axs[1, 1].lines)
    plt.close(fig)
    v = np.linspace(0, 1, 20)
    fl = recs[0]["reference"]["flux"]
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
