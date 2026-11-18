"""Smoke tests of the figure functions of notebooks 07, 15 and 16 (Agg): every one draws from small synthetic inputs,
within the font floor, and returns what its docstring says."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from skewerjax.utils import plotting as pl  # noqa: E402


def setup_module():
    pl.use_style()


def teardown_function():
    plt.close("all")


def _draw(fig):
    fig.canvas.draw()
    assert not pl.check_fonts(fig)


def test_engine_figures():
    rng = np.random.default_rng(0)
    v = np.arange(400) * 1.0
    cpix = rng.integers(50, 350, 12)
    D = np.full(12, 16)
    dtau = np.exp(-0.5 * ((v[None, :] - cpix[:, None]) / 5.0) ** 2) + 1e-4 / (1 + (v[None, :] - cpix[:, None]) ** 2)
    far = 1e-5 * np.ones(400)
    plan = dict(cpix=cpix, D=D, D_x=16)
    tot = dtau.sum(0)
    fig, axs, info = pl.plot_engine_anatomy(v, dtau, plan, far, tot + 1e-9, tot, show=[0, 1], tau_full=tot, wing_tolerance=1e-9)
    _draw(fig)
    assert info["max_dtau"] < 2e-9 and info["trunc_max_dtau"] > 0 and "full_max_dtau" in info
    m = rng.random((4, 12))
    offs = np.arange(1, 200) * 1.0
    kern = np.stack([offs**-2.0, 2 * offs**-3.0, 3 * offs**-4.0, offs**-4.0])
    fig, axs, info = pl.plot_far_convolution(v, cpix, m, offs, kern, far, far * (1 + 1e-12), D_x=16)
    _draw(fig)
    fig, axs = pl.plot_window_levels(
        {"HI-LyA": {64: 10, 128: 30}, "SiIII-1206": {4: 30, 8: 5}},
        {"dense": {"HI-LyA": 1e6, "SiIII-1206": 1e6}, "windows": {"HI-LyA": 1e5, "SiIII-1206": 1e4}},
    )
    _draw(fig)
    fig, axs = pl.plot_truncation(rng.random(100) * 1e-5, rng.random(100) * 1e-5, rng.random(100) * 1e-9, rng.random(100) * 1e-9)
    _draw(fig)


def test_pass_and_tile_figures():
    n = np.array([16, 256, 4096, 65536])
    totals = {"fake_spectra": 300 + 0.01 * n, "SkewerJAX default, CPU": 20 + 0.004 * n}
    fits = {k: (300.0, 0.01) if k == "fake_spectra" else (20.0, 0.004) for k in totals}
    fig, axs = pl.plot_cost_model(n, totals, fits, fixed_measured={"SkewerJAX default, CPU": 20 + 0.001 * n})
    _draw(fig)
    X, Z = np.meshgrid(np.arange(8) + 0.5, np.arange(8) + 0.5)
    sk = np.column_stack([X.ravel(), Z.ravel()])
    fig, ax, info = pl.plot_reach_test(
        sk, 8.0, 8, np.array([[2.2, 2.3], [5.0, 5.6]]), np.array([0.4, 1.3]), region=((0, 8), (0, 8))
    )
    _draw(fig)
    assert info["tests"][0] >= 1 and info["reached"][1] >= 1
    lat = {
        k: dict(within=w, physics=w, pages=[w, 2 * w], contiguous=[w > 0.3, w > 0.3], bytes_GB=10 * w + 1, total_GB=12.0)
        for k, w in (("4²", 0.01), ("64²", 0.4))
    }
    _draw(pl.plot_pass_reads(lat)[0])
    tile = (np.arange(64) % 4).astype(int)
    _draw(
        pl.plot_store_tiles(
            sk,
            tile,
            8.0,
            sk[:5] + 0.1,
            {"positions": 2400, "sizes": 400, "fields:default": 1600, "tile_pairs": 900},
            n_cells=100,
            n_pairs=110,
        )[0]
    )
    rows = [
        dict(prep_start=k * 0.1, prep_end=k * 0.1 + 0.15, prep_thread=k % 3, tau_start=k * 0.2 + 0.15, tau_end=k * 0.2 + 0.3)
        for k in range(6)
    ]
    fig, ax, info = pl.plot_tile_timeline(rows)
    _draw(fig)
    assert info["overlap"] > 0
    _draw(pl.plot_fixed_shapes([900, 900, 625], {"compiled": 10, "unpadded": 30}, {"16": 17.0, "65,536": 31.0})[0])


def test_crossing_and_profile_figures():
    u = np.array([0.1, 0.5, 0.9, 1.4])
    r2 = np.array([0.01, 0.02, 0.005, 0.03])
    lo, hi, who = np.array([0.0, 0.3, 0.7]), np.array([0.3, 0.7, 1.2]), np.array([0, 1, 2])
    _draw(
        pl.plot_envelope(
            u, r2, (0.0, 1.2), (lo, hi, who), sampled_edges=np.linspace(0, 1.2, 7), counts=dict(crossings=300, sampled=900)
        )[0]
    )
    fig, axs, info = pl.plot_integrated_profile(12.0, 1e-4, 25.0, tilt_kms=1e-5, n_sub=(1, 2, 4, 8))
    _draw(fig)
    errs = list(info["midpoint_errors"].values())
    assert errs[-1] < errs[0] / 10  # the midpoint rule converges
    c2 = dict(
        integrated=dict(tau_s_per_skewer=0.0095),
        midpoint={
            k: {
                "pieces_per_skewer": p,
                "tau_s_per_skewer": t,
                "HI-LyA:rms_dF": e,
                "HI-LyA:max_dF": 10 * e,
                "SiIII-1206:rms_dF": e,
                "SiIII-1206:max_dF": 10 * e,
            }
            for k, p, t, e in (("0.25", 17000, 0.016, 3e-6), ("1", 5200, 0.006, 4e-5))
        },
    )
    _draw(pl.plot_accuracy_cost(c2)[0])
    _draw(pl.plot_chord(0.3, 0.15, 12.0, 1e-4, 3.0, tau0=2.0)[0])
    o = np.arange(-300, 300)
    prof = 2.0 * np.exp(-0.5 * (o / 10.0) ** 2) + 1e-6 / (1 + (o / 50.0) ** 2)
    walk = np.where(prof > 1e-7, prof, 0.0)
    fig, axs, info = pl.plot_cell_walk(
        o,
        prof,
        walk,
        walk,
        tautail=1e-7,
        levels=np.array([4, 8, 16, 32, 64, 128, 256]),
        level_start=4,
        level_end=6,
        core=64,
        b_px=10.0,
    )
    _draw(fig)
    assert info["max_diff"] == 0.0


def test_profile_and_memory_figures():
    t = np.linspace(0, 10, 21)
    tl = {
        "fake_spectra": (t, np.full(21, 1.0), [(0, 4, "io"), (4, 7, "physics"), (7, 10, "tau")]),
        "SkewerJAX matched, CPU": (t, np.full(21, 20.0), [(0, 5, "pass"), (5, 10, "tiles")]),
    }
    stg = {
        "read": {"fake_spectra": 4.0, "SkewerJAX matched, CPU": 1.0},
        "tau": {"fake_spectra": 3.0, "SkewerJAX matched, CPU": 1.5},
    }
    fig, axs = pl.plot_cpu_profile(tl, stg)
    fig.canvas.draw()
    rec = dict(
        timeline=dict(t=t.tolist(), anon_GB=(1 + t).tolist(), file_GB=[0.2] * 21, gpu_GB=[0.5] * 21),
        markers=[[0.0, "start"], [1.0, "setup"], [2.0, "pass"], [6.0, "tiles"], [10.0, "done"]],
    )
    _draw(pl.plot_memory_timelines({"SkewerJAX default, GPU": rec})[0])
    bars = [
        dict(
            label="a", host={"baseline": 0.5, "store": 2.0, "unaccounted": 0.3}, gpu={"CUDA context": 0.1, "device buffers": 1.0}
        ),
        dict(label="b", host={"baseline": 0.3, "per-file arrays": 0.9}, gpu=None),
    ]
    _draw(pl.plot_memory_peaks(bars)[0])
    x = np.array([16.0, 256.0, 4096.0])
    ser = {
        "fake_spectra": dict(anon=[1.6, 1.7, 2.0], file=[0.3] * 3, gpu=[np.nan] * 3),
        "SkewerJAX default, GPU": dict(anon=[7.0, 7.1, 15.0], file=[0.4] * 3, gpu=[0.4, 0.5, 1.3]),
    }
    _draw(
        pl.plot_memory_scaling(
            x,
            ser,
            xlabel="skewers",
            fits={"fake_spectra": {"anon": (1.6, 1e-4)}, "SkewerJAX default, GPU": {"anon": lambda z: 7 + 2e-3 * z}},
            extrapolate=65536,
        )[0]
    )
    rows = [
        dict(label="default", anon_GB=10.0, gpu_GB=None, time_s=100.0),
        dict(label="tile 256", anon_GB=9.5, gpu_GB=None, time_s=110.0),
    ]
    _draw(pl.plot_memory_knobs(rows, reference="default")[0])
