"""tau through TNG50-4 snapshot 25 (read-only; ``tng`` marker): the exact frame a(x), the opt-in periodic spectrum
(``velocity_period=True``), the integral of tau against the column, the sampling criterion and its convergence, P1D,
the non-periodic default, curved paths, and Si III (formerly ``scripts/validation/check_tng_optical_depth.py``, same
tolerances; the quick variant by default, the full one with the ``slow`` marker)."""

import numpy as np
import pytest

from skewerjax.comparison import SKEWER_SETS
from skewerjax.core import LineBundle, phys, sampling_check, tau_absorbers
from skewerjax.simulations import HI, ArepoHDF5Reader, Ion, SimulationSkewers
from skewerjax.summary_stats import p1d_from_tau
from skewerjax.utils.paths import snapshot_dir

TNG = snapshot_dir("TNG50-4")


def tng_tau_checks(quick: bool):
    """Every check of this module (quick: fewer spacings, no 16-skewer set, no curved skewer) [(name, value, tolerance, expect)]."""
    checks = []

    def check(name, value, tol, expect=False, detail=""):
        checks.append((name, float(value), tol, expect))

    rd = ArepoHDF5Reader(TNG, 25)
    H, L = rd.header, rd.header.box_cMpc
    LYA = LineBundle(["HI-LyA"], ref_line_id="HI-LyA")
    DS = 0.25
    S = SimulationSkewers.axis_aligned(rd, SKEWER_SETS["single"]["cofm"], SKEWER_SETS["single"]["axis"], DS)
    fr = S.frame("exact")
    s, a = S.s[0], fr.a_grid[0]
    v_stat = phys.c_kms * np.log(fr.a_dagger / a)
    check("exact frame: a(L/2) = a_snap (rel)", abs(np.interp(L / 2, s, a) / H.a - 1), 1e-10)
    P_box = fr.velocity_span(0)
    check(
        "exact frame: the velocity span is the line-centre span of the box (rel)",
        abs(v_stat[-1] - v_stat[0] - P_box) / P_box,
        1e-10,
    )
    v, dv = S.spectral_grid(1.0, fr)
    tau1 = S.tau(HI, LYA, v, velocity_period=True)[0]  # periodic over the box: the integral below
    A1 = S.absorbers(HI)
    win = (A1.n[0] > 0) & (A1.s[0] > 30.0) & (A1.s[0] < 31.0)
    sub = A1.with_values(**{f: getattr(A1, f)[:, win] for f in ("s", "n", "length", "T", "v_los", "cell", "point")})
    vw = v[(v > 2150) & (v < 2500)]
    t_pp = tau_absorbers(sub, LYA, vw, fr, per_absorber=True, check=None, velocity_period=True)[0, 0, : win.sum()]
    t_sum = tau_absorbers(sub, LYA, vw, fr, check=None, velocity_period=True)[0, 0]
    check("sum of the per-piece profiles = tau of the pieces (rel)", np.abs(t_pp.sum(0) - t_sum).max() / t_sum.max(), 1e-10)
    live = A1.n[0] > 0
    N_p = A1.n[0][live] * fr.scale_factors(0, A1.s[0][live]) ** -2 * A1.length[0][live] * 3.085678e24
    pred = np.pi * 2.8179403262e-13 * 2.99792458e10 * float(LYA.f_osc[0]) * float(LYA.lambda0_A[0]) * 1e-8 * N_p.sum() / 1e5
    check("integral of tau over one period = (pi e^2 / m_e c) f lambda N (rel)", abs(tau1.sum() * dv / pred - 1), 1e-3)
    # sampling criterion and convergence
    hw = float(LYA.lambda0_A[0] * LYA.A21_GHz[0] / (4 * np.pi)) * 1e-4
    rows, Fs = {}, {}
    for ds in (0.125, 0.25, 16.0) if quick else (0.125, 0.25, 1.0, 4.0, 16.0):
        Sd = SimulationSkewers.axis_aligned(rd, SKEWER_SETS["single"]["cofm"], SKEWER_SETS["single"]["axis"], ds, cells=S.cells)
        rows[ds] = sampling_check(Sd.absorbers(HI), Sd.frame("exact"), warn=False, lorentz_hwhm_kms=hw)["column_fraction"]
        Fs[ds] = np.exp(-Sd.tau(HI, LYA, v, check=None, velocity_period=True)[0])
    check("0.25 km/s spacing passes the sampling criterion (column fraction above W > b/4)", rows[0.25], 1e-3)
    check("0.25 km/s spacing: max |dF| from 0.125 km/s", np.abs(Fs[0.25] - Fs[0.125]).max(), 2e-3)
    check("16 km/s spacing is flagged by the criterion (column fraction)", rows[16.0], 1e-3, expect=True)
    F1 = np.exp(-tau1)
    check("0 <= F <= 1 and tau finite (violations)", float(np.sum(~np.isfinite(tau1)) + np.sum((F1 < 0) | (F1 > 1))), 0)
    # P1D of 16 skewers
    if not quick:
        Sn = SimulationSkewers.axis_aligned(rd, SKEWER_SETS["nset"]["cofm"], SKEWER_SETS["nset"]["axis"], DS)
        vn, dvn = Sn.spectral_grid(1.0, Sn.frame("exact"))
        k, P = p1d_from_tau(Sn.tau(HI, LYA, vn, velocity_period=True), dvn)
        check("P1D of 16 skewers: finite and non-negative (violations)", float(np.sum(~np.isfinite(P)) + np.sum(P < 0)), 0)
    # the default (no wrap): the direct integral misses the gas across the face near the ends
    v_d = np.arange(-100.0, P_box + 100.0, 1.0)
    t_d = S.tau(HI, LYA, v_d)[0]
    mid = (v_d > 300) & (v_d < P_box - 300)
    check(
        "away from the ends, direct (default) = periodic integral (max |dF|)",
        np.abs(np.exp(-t_d[mid]) - np.exp(-np.interp(v_d[mid], v, tau1, period=P_box))).max(),
        1e-3,
    )
    check(
        "velocity_period = the span given as a number == velocity_period=True (max |dtau|)",
        np.abs(S.tau(HI, LYA, v, velocity_period=P_box)[0] - tau1).max(),
        1e-9,
    )
    if not quick:
        x0 = SKEWER_SETS["single"]["cofm"][0] * H.code_length_cMpc

        def curved(A, ds):
            t_ = np.linspace(0, L, 200001)
            xyz = np.stack([x0[0] + A * np.sin(4 * np.pi * t_ / L), np.full_like(t_, x0[1]), t_], 1)
            arc = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))])
            sq = np.linspace(0, arc[-1], int(round(arc[-1] * H.velfac_kms_per_cMpc / ds)) + 1)
            return np.stack([np.interp(sq, arc, xyz[:, i]) for i in range(3)], 1)[None]

        def tau_curved(pts):
            Sc = SimulationSkewers(rd, pts)
            f_ = Sc.frame("exact")
            vc = np.arange(-100.0, f_.velocity_span(0) + 100.0, 1.0)
            return Sc.tau(HI, LYA, vc)[0]

        t0 = tau_curved(curved(0.0, DS))
        check("zero-amplitude curve == straight skewer (max |dtau|)", np.abs(t0 - t_d[: t0.size]).max(), 1e-12)
        tc, th = tau_curved(curved(1.5, DS)), tau_curved(curved(1.5, DS / 2))
        n_ = min(tc.size, th.size)
        check(
            "curved skewer converged when the sampling is halved (max |dF|)",
            np.abs(np.exp(-tc[:n_]) - np.exp(-th[:n_])).max(),
            2e-3,
        )
    # Si III on skewer 8
    SI, LSI = Ion("Si", 3), LineBundle(["SiIII-1206"], ref_line_id="SiIII-1206")
    sk8 = SimulationSkewers.axis_aligned(rd, SKEWER_SETS["sk8"]["cofm"], SKEWER_SETS["sk8"]["axis"], DS)
    r = sampling_check(
        sk8.absorbers(SI),
        sk8.frame("exact"),
        warn=False,
        lorentz_hwhm_kms=float(LSI.lambda0_A[0] * LSI.A21_GHz[0] / (4 * np.pi)) * 1e-4,
    )
    check("Si III (skewer 8): 0.25 km/s spacing passes the criterion (column fraction)", r["column_fraction"], 1e-3)
    return checks


def _assert_all(checks):
    bad = [(n, v, t) for n, v, t, expect in checks if not (v > t if expect else v <= t)]
    assert not bad, bad


@pytest.mark.tng
def test_tng_optical_depth():
    _assert_all(tng_tau_checks(quick=True))


@pytest.mark.tng
@pytest.mark.slow
def test_tng_optical_depth_full():
    _assert_all(tng_tau_checks(quick=False))
