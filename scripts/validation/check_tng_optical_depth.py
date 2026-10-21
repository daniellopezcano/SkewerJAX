#!/usr/bin/env python
"""tau through TNG50-4 (read-only): the exact frame a(x), the periodic wrap, the integral of tau against the column,
the sampling criterion and its convergence, P1D, curved and non-periodic paths, and Si III."""
from _common import TNG, Checks

C = Checks("check_tng_optical_depth", __doc__, quick_help="fewer spacings, no 16-skewer set, no curved skewer")
import numpy as np                                                        # noqa: E402
from skewerjax.core import LineBundle, phys                               # noqa: E402
from skewerjax.core.stats import p1d_from_tau                             # noqa: E402
from skewerjax.simulations import HI, ArepoHDF5Reader, Ion, IntegrationOptions, SimulationSkewers, sampling_check, tau_absorbers  # noqa: E402
from skewerjax.comparison import SKEWER_SETS                              # noqa: E402

rd = ArepoHDF5Reader(*TNG)
H, L = rd.header, rd.header.box_cMpc
LYA = LineBundle(["HI-LyA"], ref_line_id="HI-LyA")
DS = 0.25
S = SimulationSkewers.axis_aligned(rd, SKEWER_SETS["single"]["cofm"], SKEWER_SETS["single"]["axis"], DS)
fr = S.frame("exact")
s, a = S.s[0], fr.a_grid[0]
v_stat = phys.c_kms * np.log(fr.a_dagger / a)
C.check("exact frame: a(L/2) = a_snap (rel)", abs(np.interp(L / 2, s, a) / H.a - 1), 1e-10)
C.check("exact frame: the period is the line-centre span of the box (rel)", abs(v_stat[-1] - v_stat[0] - fr.period_kms) / fr.period_kms, 1e-10)
v, dv = S.spectral_grid(1.0, fr)
tau1 = S.tau(HI, LYA, v)[0]
A1 = S.absorbers(HI)
win = (A1.n[0] > 0) & (A1.s[0] > 30.0) & (A1.s[0] < 31.0)
sub = A1.with_values(**{f: getattr(A1, f)[:, win] for f in ("s", "n", "length", "T", "v_los", "cell", "point")})
vw = v[(v > 2150) & (v < 2500)]
t_pp = tau_absorbers(sub, LYA, vw, fr, per_absorber=True, check=None)[0, 0, :win.sum()]
t_sum = tau_absorbers(sub, LYA, vw, fr, check=None)[0, 0]
C.check("sum of the per-piece profiles = tau of the pieces (rel)", np.abs(t_pp.sum(0) - t_sum).max() / t_sum.max(), 1e-10)
live = A1.n[0] > 0
N_p = A1.n[0][live] * H.a ** 3 * fr.scale_factors(0, A1.s[0][live]) ** -2 * A1.length[0][live] * 3.085678e24
pred = np.pi * 2.8179403262e-13 * 2.99792458e10 * float(LYA.f_osc[0]) * float(LYA.lambda0_A[0]) * 1e-8 * N_p.sum() / 1e5
C.check("integral of tau over one period = (pi e^2 / m_e c) f lambda N (rel)", abs(tau1.sum() * dv / pred - 1), 1e-3)
# sampling criterion and convergence
hw = float(LYA.lambda0_A[0] * LYA.A21_GHz[0] / (4 * np.pi)) * 1e-4
rows, Fs = {}, {}
for ds in ((0.125, 0.25, 16.0) if C.quick else (0.125, 0.25, 1.0, 4.0, 16.0)):
    Sd = SimulationSkewers.axis_aligned(rd, SKEWER_SETS["single"]["cofm"], SKEWER_SETS["single"]["axis"], ds, cells=S.cells)
    rows[ds] = sampling_check(Sd.absorbers(HI), Sd.frame("exact"), warn=False, lorentz_hwhm_kms=hw)["column_fraction"]
    Fs[ds] = np.exp(-Sd.tau(HI, LYA, v, check=None)[0])
C.check("0.25 km/s spacing passes the sampling criterion (column fraction above W > b/4)", rows[0.25], 1e-3)
C.check("0.25 km/s spacing: max |dF| from 0.125 km/s", np.abs(Fs[0.25] - Fs[0.125]).max(), 2e-3)
C.check("16 km/s spacing is flagged by the criterion (column fraction)", rows[16.0], 1e-3, expect=True)
F1 = np.exp(-tau1)
C.check("0 <= F <= 1 and tau finite (violations)", float(np.sum(~np.isfinite(tau1)) + np.sum((F1 < 0) | (F1 > 1))), 0)
# P1D of 16 skewers
if not C.quick:
    Sn = SimulationSkewers.axis_aligned(rd, SKEWER_SETS["nset"]["cofm"], SKEWER_SETS["nset"]["axis"], DS)
    vn, dvn = Sn.spectral_grid(1.0, Sn.frame("exact"))
    k, P = p1d_from_tau(Sn.tau(HI, LYA, vn), dvn)
    C.check("P1D of 16 skewers: finite and non-negative (violations)", float(np.sum(~np.isfinite(P)) + np.sum(P < 0)), 0)
# non-periodic path: the direct integral misses the gas across the face near the ends
DIRECT = IntegrationOptions(periodic=False)
frd = S.frame("exact", periodic=False)
v_d = np.arange(-100.0, phys.c_kms * np.log(frd.a_grid[0, 0] / frd.a_grid[0, -1]) + 100.0, 1.0)
t_d = S.tau(HI, LYA, v_d, integ=DIRECT)[0]
mid = (v_d > 300) & (v_d < fr.period_kms - 300)
C.check("away from the ends, direct = periodic integral (max |dF|)",
        np.abs(np.exp(-t_d[mid]) - np.exp(-np.interp(v_d[mid], v, tau1, period=fr.period_kms))).max(), 1e-3)
if not C.quick:
    x0 = SKEWER_SETS["single"]["cofm"][0] * H.code_length_cMpc

    def curved(A, ds):
        t_ = np.linspace(0, L, 200001)
        xyz = np.stack([x0[0] + A * np.sin(4 * np.pi * t_ / L), np.full_like(t_, x0[1]), t_], 1)
        arc = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))])
        sq = np.linspace(0, arc[-1], int(round(arc[-1] * H.velfac_kms_per_cMpc / ds)) + 1)
        return np.stack([np.interp(sq, arc, xyz[:, i]) for i in range(3)], 1)[None]

    def tau_curved(pts):
        Sc = SimulationSkewers(rd, pts)
        f_ = Sc.frame("exact", periodic=False)
        vc = np.arange(-100.0, phys.c_kms * np.log(f_.a_grid[0, 0] / f_.a_grid[0, -1]) + 100.0, 1.0)
        return Sc.tau(HI, LYA, vc, integ=DIRECT)[0]

    t0 = tau_curved(curved(0.0, DS))
    C.check("zero-amplitude curve == straight skewer (max |dtau|)", np.abs(t0 - t_d[:t0.size]).max(), 1e-12)
    tc, th = tau_curved(curved(1.5, DS)), tau_curved(curved(1.5, DS / 2))
    n_ = min(tc.size, th.size)
    C.check("curved skewer converged when the sampling is halved (max |dF|)", np.abs(np.exp(-tc[:n_]) - np.exp(-th[:n_])).max(), 2e-3)
# Si III on skewer 8
SI, LSI = Ion("Si", 3), LineBundle(["SiIII-1206"], ref_line_id="SiIII-1206")
sk8 = SimulationSkewers.axis_aligned(rd, SKEWER_SETS["sk8"]["cofm"], SKEWER_SETS["sk8"]["axis"], DS)
r = sampling_check(sk8.absorbers(SI), sk8.frame("exact"), warn=False, lorentz_hwhm_kms=float(LSI.lambda0_A[0] * LSI.A21_GHz[0] / (4 * np.pi)) * 1e-4)
C.check("Si III (skewer 8): 0.25 km/s spacing passes the criterion (column fraction)", r["column_fraction"], 1e-3)

if C.outdir is not None:
    from _common import figure_backend
    figure_backend()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(14, 5), layout="constrained")
    ax.plot(v, F1, lw=1.2); ax.set(xlabel="v [km/s]", ylabel=r"$F$ (Ly$\alpha$)", title="TNG50-4 single skewer, SkewerJAX default")
    C.savefig(fig, "tng_flux_single")
C.done()
