#!/usr/bin/env python
"""SkewerJAX against fake_spectra on TNG50-4: the matched mode reproduces fake_spectra (per-cell inputs, columns, tau),
and the default differs from it by the sum of the named differences (effect chain). fake_spectra runs live
(installed; results cached in ``--cache``) or is read from ``--reference``; ``--export`` writes such a file."""
import argparse
import sys
import tempfile

from _common import TNG, Checks

extra = argparse.ArgumentParser(add_help=False)
extra.add_argument("--cache", default=None)
extra.add_argument("--reference", default=None)
extra.add_argument("--export", default=None)
opts, rest = extra.parse_known_args()
sys.argv = sys.argv[:1] + rest
C = Checks("check_fake_spectra", __doc__ + " Extra options: --cache DIR, --reference FILE, --export FILE.",
           quick_help="single skewer and Si III skewer only (no 16-skewer P1D, shorter effect chain)")
import numpy as np                                                        # noqa: E402
from skewerjax.core import LineBundle                                     # noqa: E402
from skewerjax.core.stats import p1d_from_tau                             # noqa: E402
from skewerjax.simulations import HI, ArepoHDF5Reader, ExtractionOptions, Ion, SimulationSkewers  # noqa: E402
from skewerjax.comparison import (DEFAULT_INTEGRATION, EXTRACTION_EFFECTS, FAKE_SPECTRA_EXTRACTION as EF,
                                  FAKE_SPECTRA_INTEGRATION as IF, INTEGRATION_EFFECTS, SKEWER_SETS, FakeSpectraSource,
                                  effect_rows, matched_tau, run_effects, spectral_metrics)  # noqa: E402

cache = opts.cache or (str(C.outdir / "fs_cache") if C.outdir is not None else tempfile.mkdtemp(prefix="fs_cache_"))
FS = FakeSpectraSource(reference=opts.reference, cache=None if opts.reference else cache, snap_base=TNG[0], snapnum=TNG[1])
print("fake_spectra:", FS.mode)
rd = ArepoHDF5Reader(*TNG)
LYA, LSI, SI = LineBundle(["HI-LyA"], ref_line_id="HI-LyA"), LineBundle(["SiIII-1206"], ref_line_id="SiIII-1206"), Ion("Si", 3)
SK = {n: SimulationSkewers.axis_aligned(rd, SKEWER_SETS[n]["cofm"], SKEWER_SETS[n]["axis"], 0.25)
      for n in (("single", "sk8") if C.quick else ("single", "sk8", "nset"))}
S = SK["single"]
v, dv = S.spectral_grid(1.0)                                              # fake_spectra's periodic pixel grid
fs1 = FS.run("single_matched", "single", products="tau,colden,temp,velocity")
C.check("same pixel grid as fake_spectra (|nbins - N_v| + |dv - dvbin| / dv)", abs(int(fs1["nbins"]) - v.size) + abs(float(fs1["dvbin"]) - dv) / dv, 1e-9)
# 1. inputs: fake_spectra's per-cell values == SkewerJAX's matched extraction (float32 floor)
fi = FS.cell_inputs(rd, S.cells, "single", HI)
P = S.cell_inputs(HI, EF)
for k in ("n", "T"):
    a_, b_ = np.asarray(P[k]), np.asarray(fi[k])
    ok = b_ > 0
    C.check(f"matched extraction, per-cell {k} == fake_spectra's (max rel, float32 floor)", np.max(np.abs(a_[ok] / b_[ok] - 1)), 1e-6)
# 2. integration: fake_spectra's own cells through the emulated scheme, then SkewerJAX's matched mode end to end
t_same = matched_tau(S, HI, LYA, v, EF, IF, override=fi, override_key="fake_spectra_cells")
C.check("fake_spectra's cells + emulated integration == fake_spectra (max |dF|)", np.abs(np.exp(-t_same) - np.exp(-fs1["tau"])).max(), 1e-5)
t_m = matched_tau(S, HI, LYA, v, EF, IF)
m = spectral_metrics(t_m, fs1["tau"], dv)
C.check("matched mode == fake_spectra, single skewer (max |dF|)", m["max_dF"], 5e-5, detail=f"rms {m['rms_dF']:.1e}, d<F> {m['d_meanF']:.1e}")
N_th = S.column_per_pixel(HI, EF, nbins=v.size)
has = fs1["colden"] > 0
r_pix = np.abs(N_th[has] / fs1["colden"][has] - 1)
C.check("top-hat deposition: N_HI per position pixel == fake_spectra's get_col_density (median rel)", np.median(r_pix), 1e-6,
        detail=f"max {r_pix.max():.1e}")
C.check("top-hat deposition: total N_HI == fake_spectra's (rel; a few pixels differ at <= 1e-3)", abs(N_th.sum() / fs1["colden"].sum() - 1), 1e-5)
t_def = matched_tau(S, HI, LYA, v, ExtractionOptions(), DEFAULT_INTEGRATION.replace(frame="snapshot"))
C.check("the default extraction differs from fake_spectra (|d<F>|, expected > 1e-4)",
        abs(spectral_metrics(t_def, fs1["tau"], dv)["d_meanF"]), 1e-4, expect=True)
# 3. Si III on skewer 8
S8 = SK["sk8"]
fs8 = FS.run("sk8_SiIII", "sk8", species="Si:3:1206")
C.check("Si III, skewer 8: matched mode == fake_spectra (max |dF|)",
        np.abs(np.exp(-matched_tau(S8, SI, LSI, v, EF, IF)) - np.exp(-fs8["tau"])).max(), 5e-5)
# 4. the effect chains: from fake_spectra's options to SkewerJAX's, one switch at a time (integration on fake_spectra's
# fields, then extraction with SkewerJAX's integration; together they telescope from fake_spectra to the default)
res_i = run_effects(S, HI, LYA, v, INTEGRATION_EFFECTS[:3] if C.quick else INTEGRATION_EFFECTS, EF, IF, verbose=False)
rows_i, tot_i = effect_rows(res_i, dv)
C.check("integration chain telescopes: end - start = sum of the steps (d<F>)", abs(tot_i["dF"] - sum(r["dF_chain"] for r in rows_i)), 1e-12)
if not C.quick:
    t_end_i = res_i["links"][list(res_i["links"])[-1]]
    C.check("integration chain ends at [matched fields, SkewerJAX integration] (max |dtau|)",
            np.abs(t_end_i - matched_tau(S, HI, LYA, v, EF, DEFAULT_INTEGRATION)).max(), 1e-12)
    C.check("integration: isolated differences add up to the total (|total - sum| / sum |isolated|, d<F>)",
            abs(tot_i["dF"] - tot_i["dF_sum"]) / max(tot_i["dF_abs"], 1e-300), 0.1)
    res_e = run_effects(S, HI, LYA, v, EXTRACTION_EFFECTS, EF, DEFAULT_INTEGRATION, verbose=False)
    rows_e, tot_e = effect_rows(res_e, dv)
    t_end_e = res_e["links"][list(res_e["links"])[-1]]
    C.check("extraction chain ends at SkewerJAX's default (max |dtau| against the default tau)",
            np.abs(t_end_e - S.tau(HI, LYA, v)).max(), 1e-12)
    C.check("fake_spectra -> default: d<F> = integration + extraction chains (telescoping, abs)",
            abs(np.exp(-t_end_e).mean() - np.exp(-res_i["links"]["start"]).mean() - tot_i["dF"] - tot_e["dF"]), 1e-12)
    Sn = SK["nset"]
    fsn = FS.run("nset_matched", "nset")
    k, Pm = p1d_from_tau(matched_tau(Sn, HI, LYA, v, EF, IF), dv)
    _, Pf = p1d_from_tau(fsn["tau"], dv)
    C.check("16 skewers: matched P1D == fake_spectra's (k < 1 s/km, max |dP / P|)", np.abs(Pm / Pf - 1)[k < 1.0].max(), 1e-3)
if opts.export:
    print(f"reference file {opts.export}: {FS.export(opts.export) / 1e6:.1f} MB")

if C.outdir is not None:
    from _common import figure_backend
    figure_backend()
    import matplotlib.pyplot as plt
    from skewerjax.utils.plotting import ENTITY_COLORS as EC, MODE_STYLES as MS, add_panel_letters
    fig, axs = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]}, layout="constrained")
    axs[0].plot(v, np.exp(-fs1["tau"][0]), color=EC["fake_spectra"], **MS["reference"], label="fake_spectra")
    axs[0].plot(v, np.exp(-t_m[0]), color=EC["HI"], **MS["matched"], label="SkewerJAX matched")
    axs[0].plot(v, np.exp(-t_def[0]), color=EC["HI"], **MS["default"], alpha=0.6, label="SkewerJAX default (snapshot frame)")
    axs[1].plot(v, np.exp(-t_m[0]) - np.exp(-fs1["tau"][0]), color=EC["HI"])
    axs[0].set_ylabel("F"); axs[1].set(xlabel="u [km/s]", ylabel=r"$F_{\rm matched} - F_{\rm fs}$"); axs[0].legend()
    add_panel_letters(axs)
    C.savefig(fig, "fake_spectra_single")
C.done()
