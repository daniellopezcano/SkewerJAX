"""fake_spectra vs SkewerJAX, one difference at a time: the matched option sets, the effect chain and its metrics.

``FAKE_SPECTRA_EXTRACTION`` / ``FAKE_SPECTRA_INTEGRATION`` reproduce fake_spectra; ``ExtractionOptions()`` /
``DEFAULT_INTEGRATION`` are SkewerJAX's defaults (the physically most accurate choices). ``EXTRACTION_EFFECTS`` and
``INTEGRATION_EFFECTS`` list every difference as one switch; ``run_effects`` evaluates them as a cumulative chain
(which telescopes exactly from fake_spectra's options to SkewerJAX's) and one at a time.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from skewerjax.comparison.emulation import MatchedIntegration, matched_tau
from skewerjax.simulations import gas_physics as gp
from skewerjax.simulations.snapshot_skewers import ExtractionOptions, SimulationSkewers
from skewerjax.summary_stats.overflux import equivalent_width_kms
from skewerjax.summary_stats.stats_1d import p1d_from_tau

FAKE_SPECTRA_EXTRACTION = ExtractionOptions(
    deposition="tophat",
    mass_unit="proton",
    T_constants="fake_spectra",
    b_constants="fake_spectra",
    x_HI_model="fake_sf",
    cloudy_density="fake_spectra",
    cloudy_interpolation="cubic",
)
# fake_spectra's spectra are periodic by construction, so both ends of the comparison use a periodic spectrum
# (velocity_period=True: each skewer over its own velocity span); SkewerJAX's own default is not periodic.
FAKE_SPECTRA_INTEGRATION = MatchedIntegration(
    frame="snapshot", chord="fake_7pt", detuning="linear", tautail=True, pixel_average=True, velocity_period=True
)
DEFAULT_INTEGRATION = MatchedIntegration(velocity_period=True)


# --------------------------------------------------------------------------- the differences, one switch each
@dataclass(frozen=True)
class Effect:
    """One difference between fake_spectra and SkewerJAX.

    change : options switched from fake_spectra's value to SkewerJAX's in the cumulative chain.
    isolated : the switch applied alone, if it differs from ``change`` (default: ``change``).
    isolated_from : "start" (fake_spectra's options) or the label of an earlier link of the chain
        whose options are the base of the isolated switch (when the switch needs an earlier one).
    ions : "all", "HI" or "metals" (a switch that does not act on the ion is skipped).
    """

    label: str
    kind: str
    change: dict
    isolated: Optional[dict] = None
    isolated_from: str = "start"
    ions: str = "all"
    what: str = ""

    def applies(self, ion: gp.Ion) -> bool:
        return self.ions == "all" or (self.ions == "HI") == ion.is_HI


EXTRACTION_EFFECTS = (
    Effect("T constants", "extraction", dict(T_constants="codata"), what="m_p / k_B of fake_spectra -> CODATA in T"),
    Effect("b constants", "extraction", dict(b_constants="codata"), what="b with fake_spectra's C constants -> CODATA, m_u"),
    Effect("mass constant", "extraction", dict(mass_unit="amu"), what="n = rho X / (A m_p) -> rho X / (A m_u)"),
    Effect(
        "neutral fraction",
        "extraction",
        dict(x_HI_model="stored_sf"),
        ions="HI",
        what="fake_spectra's star-forming rule (rho / m_p threshold) -> stored, SF cells recomputed",
    ),
    Effect("CLOUDY density", "extraction", dict(cloudy_density="hydrogen"), ions="metals", what="hden = 0.774 rho / m_p -> n_H"),
    Effect(
        "CLOUDY interpolation",
        "extraction",
        dict(cloudy_interpolation="linear"),
        ions="metals",
        what="cubic -> bilinear in log f",
    ),
    Effect("deposition", "extraction", dict(deposition="voronoi"), what="top-hat spheres V^(1/3) -> Voronoi cells"),
)
INTEGRATION_EFFECTS = (
    Effect("tautail", "integration", dict(tautail=False), what="per-cell early stop at tau < 1e-7 -> full profiles"),
    Effect(
        "pixel average", "integration", dict(pixel_average=False), what="per-cell pixel average for b < 2 dv -> pixel centres"
    ),
    Effect("detuning", "integration", dict(detuning="exact"), what="H(a, dv / b) linear -> nu = (c / lambda) expm1(-dv / c)"),
    Effect("7/8 normalisation", "integration", dict(normalisation=8 / 7), what="7 chord points of weight 1/8 -> weights 1/7"),
    Effect(
        "chord quadrature",
        "integration",
        dict(chord="segments", normalisation=1.0),
        isolated=dict(chord="segments", normalisation=7 / 8),
        what="7 points per chord -> the exact chord (pieces of the dual segments)",
    ),
    Effect(
        "frame",
        "integration",
        dict(frame="exact"),
        isolated_from="chord quadrature",
        what="one a, u = a H s + v_pec -> a(x) along the path, v_dagger",
    ),
)


def effect_chain(effects: Sequence[Effect], ext0: ExtractionOptions, int0: MatchedIntegration, ion: gp.Ion):
    """The cumulative chain and the isolated switches of ``effects`` applicable to ``ion``.

    Returns
    -------
    links : list of (label, ExtractionOptions, IntegrationOptions); links[0] is ("start", ext0, int0).
    isolated : {label: (ExtractionOptions, IntegrationOptions, base label)}.
    """

    def apply(ext, integ, kind, d):
        return (ext.replace(**d), integ) if kind == "extraction" else (ext, integ.replace(**d))

    for e in effects:  # a switch that does not act on this ion is set from the start
        if not e.applies(ion):
            ext0, int0 = apply(ext0, int0, e.kind, e.change)
    links, ext, integ = [("start", ext0, int0)], ext0, int0
    for e in effects:
        if e.applies(ion):
            ext, integ = apply(ext, integ, e.kind, e.change)
            links.append((e.label, ext, integ))
    opts = {lab: (x, i) for lab, x, i in links}
    isolated = {}
    for e in effects:
        if e.applies(ion):
            bx, bi = opts[e.isolated_from]
            x, i = apply(bx, bi, e.kind, e.isolated if e.isolated is not None else e.change)
            isolated[e.label] = (x, i, e.isolated_from)
    return links, isolated


# --------------------------------------------------------------------------- the chain and its metrics
def run_effects(
    sk: SimulationSkewers,
    ion: gp.Ion,
    line_bundle,
    velocities_kms,
    effects: Sequence[Effect],
    ext0: ExtractionOptions,
    int0: MatchedIntegration,
    verbose: bool = True,
) -> dict:
    """tau of every link of the chain of ``effects`` from (ext0, int0) and of every isolated switch.

    Returns dict(links={label: tau}, isolated={label: tau}, base={label: base label},
    options={label: (ext, integ)}).
    """
    t = time.perf_counter()
    links, iso = effect_chain(effects, ext0, int0, ion)
    L = {lab: matched_tau(sk, ion, line_bundle, velocities_kms, x, i) for lab, x, i in links}
    I = {lab: matched_tau(sk, ion, line_bundle, velocities_kms, x, i) for lab, (x, i, _) in iso.items()}
    if verbose:
        print(f"{sk.name}/{ion.species}: {len(L)} links + {len(I)} isolated switches in {time.perf_counter() - t:.0f} s")
    return dict(
        links=L, isolated=I, base={lab: b for lab, (_, _, b) in iso.items()}, options={lab: (x, i) for lab, x, i in links}
    )


def spectral_metrics(tau, tau_ref, dv: float) -> dict:
    """Differences of tau against tau_ref (both (N_sk, N_v)): max |dF|, rms dF, d<F>, dEW / EW (summed
    over skewers), median |dtau| / tau for 1e-3 < tau_ref < 5, mean velocity lag [km/s]."""
    tau, tau_ref = np.atleast_2d(tau), np.atleast_2d(tau_ref)
    F, Fr = np.exp(-tau), np.exp(-tau_ref)
    m = (tau_ref > 1e-3) & (tau_ref < 5)
    EW, EWr = equivalent_width_kms(tau, dv).sum(), equivalent_width_kms(tau_ref, dv).sum()
    return dict(
        max_dF=float(np.abs(F - Fr).max()),
        rms_dF=float(np.sqrt(np.mean((F - Fr) ** 2))),
        d_meanF=float(F.mean() - Fr.mean()),
        dEW=float((EW - EWr) / max(EWr, 1e-300)),
        med_dtau=float(np.median(np.abs(tau[m] - tau_ref[m]) / tau_ref[m])) if m.any() else 0.0,
        lag=float(np.mean([velocity_lag_kms(F[k], Fr[k], dv) for k in range(F.shape[0])])),
    )


def velocity_lag_kms(F1, F2, dv: float) -> float:
    """Shift [km/s] maximising the periodic cross-correlation of two spectra (frame/sign errors)."""
    F1, F2 = np.asarray(F1, float), np.asarray(F2, float)
    c = np.fft.irfft(np.fft.rfft(F1 - F1.mean()) * np.conj(np.fft.rfft(F2 - F2.mean())), n=F1.size)
    k = int(np.argmax(c))
    return (k if k <= F1.size // 2 else k - F1.size) * dv


def p1d_change(tau, tau_ref, dv: float, bands=((0.0, 0.1), (0.1, 1.0))) -> list:
    """Median |P / P_ref - 1| of the P1D in k bands [s/km]."""
    k, P = p1d_from_tau(tau, dv)
    _, Pr = p1d_from_tau(tau_ref, dv)
    return [float(np.median(np.abs(P / Pr - 1)[(k > lo) & (k <= hi)])) for lo, hi in bands]


def effect_rows(result: dict, dv: float, tau_ref=None) -> tuple:
    """One row per effect: the change along the chain and alone (d<F>, dEW / EW, max |dF|, rms dF, P1D
    change at k <= 0.1 and 0.1 < k <= 1 s/km). EW changes are relative to ``tau_ref`` (default: the
    chain's start). Also returns the totals: (end - start), the sum of the isolated changes and of their
    magnitudes, for d<F> and dEW / EW."""
    L, I, base = result["links"], result["isolated"], result["base"]
    labels = list(L)
    ref = L["start"] if tau_ref is None else tau_ref
    EW = lambda t: equivalent_width_kms(t, dv).sum()
    EWr = EW(ref)
    rows = []
    for prev, lab in zip(labels[:-1], labels[1:]):
        b = L[base[lab]]
        mc, mi = spectral_metrics(L[lab], L[prev], dv), spectral_metrics(I[lab], b, dv)
        rows.append(
            dict(
                effect=lab,
                dF_chain=mc["d_meanF"],
                dEW_chain=(EW(L[lab]) - EW(L[prev])) / EWr,
                max_chain=mc["max_dF"],
                rms_chain=mc["rms_dF"],
                dF_iso=mi["d_meanF"],
                dEW_iso=(EW(I[lab]) - EW(b)) / EWr,
                max_iso=mi["max_dF"],
                rms_iso=mi["rms_dF"],
                p1d_iso=p1d_change(I[lab], b, dv),
            )
        )
    tot = dict(
        dF=float(np.exp(-L[labels[-1]]).mean() - np.exp(-L["start"]).mean()),
        dEW=float((EW(L[labels[-1]]) - EW(L["start"])) / EWr),
        dF_sum=float(sum(r["dF_iso"] for r in rows)),
        dEW_sum=float(sum(r["dEW_iso"] for r in rows)),
        dF_abs=float(sum(abs(r["dF_iso"]) for r in rows)),
        dEW_abs=float(sum(abs(r["dEW_iso"]) for r in rows)),
    )
    return rows, tot


def print_effect_rows(rows: list, tot: dict, title: str = ""):
    """The table of ``effect_rows``, ranked by the rms dF of the isolated switch."""
    if title:
        print(title)
    print(
        f"{'effect':22s} {'rms dF':>9s} {'max |dF|':>9s} {'d<F>':>10s} {'dEW/EW':>10s} {'P1D k<0.1':>10s} {'0.1-1':>9s} | "
        f"{'chain d<F>':>10s} {'chain dEW/EW':>12s}"
    )
    for r in sorted(rows, key=lambda r: -r["rms_iso"]):
        print(
            f"{r['effect']:22s} {r['rms_iso']:9.2e} {r['max_iso']:9.2e} {r['dF_iso']:10.2e} {r['dEW_iso']:10.2e} "
            f"{r['p1d_iso'][0]:10.2e} {r['p1d_iso'][1]:9.2e} | {r['dF_chain']:10.2e} {r['dEW_chain']:12.2e}"
        )
    print(
        f"{'total (end - start)':22s} d<F> = {tot['dF']:+.4e}, dEW/EW = {tot['dEW']:+.4e} | sum of the isolated switches: "
        f"d<F> = {tot['dF_sum']:+.4e}, dEW/EW = {tot['dEW_sum']:+.4e}"
    )
