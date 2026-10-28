"""SkewerJAX against fake_spectra on TNG50-4 snapshot 25: the matched mode reproduces fake_spectra (per-cell inputs,
columns, tau), the default differs from it, and the effect chains telescope from one to the other (formerly
``scripts/validation/check_fake_spectra.py``, same tolerances). fake_spectra runs live in its own process (results in a
temporary cache): markers ``tng``, ``fake_spectra`` and ``slow`` (several minutes)."""

import tempfile

import numpy as np
import pytest

from skewerjax.comparison import (
    DEFAULT_INTEGRATION,
    EXTRACTION_EFFECTS,
    INTEGRATION_EFFECTS,
    SKEWER_SETS,
    FakeSpectraSource,
    effect_rows,
    matched_tau,
    run_effects,
    spectral_metrics,
)
from skewerjax.comparison import FAKE_SPECTRA_EXTRACTION as EF
from skewerjax.comparison import FAKE_SPECTRA_INTEGRATION as IF
from skewerjax.core import LineBundle
from skewerjax.simulations import HI, ArepoHDF5Reader, ExtractionOptions, Ion, SimulationSkewers
from skewerjax.summary_stats import p1d_from_tau
from skewerjax.utils.paths import snapshot_dir

TNG = snapshot_dir("TNG50-4")


def fake_spectra_checks(quick: bool):
    """Every check of the module docstring (quick: single skewer and Si III skewer only, no 16-skewer P1D, a shorter
    effect chain): [(name, value, tolerance, expect)]."""
    checks = []

    def check(name, value, tol, expect=False, detail=""):
        checks.append((name, float(value), tol, expect))

    FS = FakeSpectraSource(cache=tempfile.mkdtemp(prefix="fs_cache_"), snap_base=str(TNG), snapnum=25)
    rd = ArepoHDF5Reader(TNG, 25)
    LYA, LSI, SI = (
        LineBundle(["HI-LyA"], ref_line_id="HI-LyA"),
        LineBundle(["SiIII-1206"], ref_line_id="SiIII-1206"),
        Ion("Si", 3),
    )
    SK = {
        n: SimulationSkewers.axis_aligned(rd, SKEWER_SETS[n]["cofm"], SKEWER_SETS[n]["axis"], 0.25)
        for n in (("single", "sk8") if quick else ("single", "sk8", "nset"))
    }
    S = SK["single"]
    v, dv = S.spectral_grid(1.0)  # fake_spectra's periodic pixel grid
    fs1 = FS.run("single_matched", "single", products="tau,colden,temp,velocity")
    check(
        "same pixel grid as fake_spectra (|nbins - N_v| + |dv - dvbin| / dv)",
        abs(int(fs1["nbins"]) - v.size) + abs(float(fs1["dvbin"]) - dv) / dv,
        1e-9,
    )
    # 1. inputs: fake_spectra's per-cell values == SkewerJAX's matched extraction (float32 floor)
    fi = FS.cell_inputs(rd, S.cells, "single", HI)
    P = S.cell_inputs(HI, EF)
    for k in ("n", "T"):
        a_, b_ = np.asarray(P[k]), np.asarray(fi[k])
        ok = b_ > 0
        check(
            f"matched extraction, per-cell {k} == fake_spectra's (max rel, float32 floor)",
            np.max(np.abs(a_[ok] / b_[ok] - 1)),
            1e-6,
        )
    # 2. integration: fake_spectra's own cells through the emulated scheme, then SkewerJAX's matched mode end to end
    t_same = matched_tau(S, HI, LYA, v, EF, IF, override=fi, override_key="fake_spectra_cells")
    check(
        "fake_spectra's cells + emulated integration == fake_spectra (max |dF|)",
        np.abs(np.exp(-t_same) - np.exp(-fs1["tau"])).max(),
        1e-5,
    )
    t_m = matched_tau(S, HI, LYA, v, EF, IF)
    m = spectral_metrics(t_m, fs1["tau"], dv)
    check(
        "matched mode == fake_spectra, single skewer (max |dF|)",
        m["max_dF"],
        5e-5,
        detail=f"rms {m['rms_dF']:.1e}, d<F> {m['d_meanF']:.1e}",
    )
    N_th = S.column_per_pixel(HI, EF, nbins=v.size)
    has = fs1["colden"] > 0
    r_pix = np.abs(N_th[has] / fs1["colden"][has] - 1)
    check(
        "top-hat deposition: N_HI per position pixel == fake_spectra's get_col_density (median rel)",
        np.median(r_pix),
        1e-6,
        detail=f"max {r_pix.max():.1e}",
    )
    check(
        "top-hat deposition: total N_HI == fake_spectra's (rel; a few pixels differ at <= 1e-3)",
        abs(N_th.sum() / fs1["colden"].sum() - 1),
        1e-5,
    )
    t_def = matched_tau(S, HI, LYA, v, ExtractionOptions(), DEFAULT_INTEGRATION.replace(frame="snapshot"))
    check(
        "the default extraction differs from fake_spectra (|d<F>|, expected > 1e-4)",
        abs(spectral_metrics(t_def, fs1["tau"], dv)["d_meanF"]),
        1e-4,
        expect=True,
    )
    # 3. Si III on skewer 8
    S8 = SK["sk8"]
    fs8 = FS.run("sk8_SiIII", "sk8", species="Si:3:1206")
    check(
        "Si III, skewer 8: matched mode == fake_spectra (max |dF|)",
        np.abs(np.exp(-matched_tau(S8, SI, LSI, v, EF, IF)) - np.exp(-fs8["tau"])).max(),
        5e-5,
    )
    # 4. the effect chains: from fake_spectra's options to SkewerJAX's, one switch at a time (integration on fake_spectra's
    # fields, then extraction with SkewerJAX's integration; together they telescope from fake_spectra to the default)
    res_i = run_effects(S, HI, LYA, v, INTEGRATION_EFFECTS[:3] if quick else INTEGRATION_EFFECTS, EF, IF, verbose=False)
    rows_i, tot_i = effect_rows(res_i, dv)
    check(
        "integration chain telescopes: end - start = sum of the steps (d<F>)",
        abs(tot_i["dF"] - sum(r["dF_chain"] for r in rows_i)),
        1e-12,
    )
    if not quick:
        t_end_i = res_i["links"][list(res_i["links"])[-1]]
        check(
            "integration chain ends at [matched fields, SkewerJAX integration] (max |dtau|)",
            np.abs(t_end_i - matched_tau(S, HI, LYA, v, EF, DEFAULT_INTEGRATION)).max(),
            1e-12,
        )
        check(
            "integration: isolated differences add up to the total (|total - sum| / sum |isolated|, d<F>)",
            abs(tot_i["dF"] - tot_i["dF_sum"]) / max(tot_i["dF_abs"], 1e-300),
            0.1,
        )
        res_e = run_effects(S, HI, LYA, v, EXTRACTION_EFFECTS, EF, DEFAULT_INTEGRATION, verbose=False)
        rows_e, tot_e = effect_rows(res_e, dv)
        t_end_e = res_e["links"][list(res_e["links"])[-1]]
        check(
            "extraction chain ends at SkewerJAX's default, periodic as the comparison (max |dtau| against the tau)",
            np.abs(t_end_e - S.tau(HI, LYA, v, velocity_period=True)).max(),
            1e-12,
        )
        check(
            "fake_spectra -> default: d<F> = integration + extraction chains (telescoping, abs)",
            abs(np.exp(-t_end_e).mean() - np.exp(-res_i["links"]["start"]).mean() - tot_i["dF"] - tot_e["dF"]),
            1e-12,
        )
        Sn = SK["nset"]
        fsn = FS.run("nset_matched", "nset")
        k, Pm = p1d_from_tau(matched_tau(Sn, HI, LYA, v, EF, IF), dv)
        _, Pf = p1d_from_tau(fsn["tau"], dv)
        check("16 skewers: matched P1D == fake_spectra's (k < 1 s/km, max |dP / P|)", np.abs(Pm / Pf - 1)[k < 1.0].max(), 1e-3)
    return checks


def _assert_all(checks):
    bad = [(n, v, t) for n, v, t, expect in checks if not (v > t if expect else v <= t)]
    assert not bad, bad


@pytest.mark.tng
@pytest.mark.fake_spectra
@pytest.mark.slow
def test_matched_mode_reproduces_fake_spectra():
    _assert_all(fake_spectra_checks(quick=True))


@pytest.mark.tng
@pytest.mark.fake_spectra
@pytest.mark.slow
def test_matched_mode_reproduces_fake_spectra_full():
    _assert_all(fake_spectra_checks(quick=False))
