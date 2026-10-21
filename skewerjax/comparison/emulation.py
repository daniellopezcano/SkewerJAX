"""fake_spectra's per-cell integration scheme on SkewerJAX's absorbers, and the matched integration options.

fake_spectra (``absorption.cpp:LineAbsorption::add_tau_particle`` L215-L282,
``singleabs.h:SingleAbsorber`` L86-L155) computes, for every cell whose top-hat sphere touches an
axis-aligned sightline:

1. **Chord quadrature** (``singleabs.h`` L100-L118, NGRID = 8): the chord of half-length
   z_r = sqrt(h^2 - dr^2) is sampled at its 7 *interior* points z_c + z_r (k/4 - 1), k = 1..7,
   each with weight (2 z_r / 8) (3 / 4 pi) n. The two end points, which carry zero weight for an
   SPH kernel, are dropped; for a top hat this integrates 7/8 of the column.
2. **Pixel average** (``tau_kern_outer`` L133-L155): if the pixel is wider than b/2, the profile is
   averaged over the pixel with a trapezoid on npoints = 2 ceil(dv / b) + 1 points, else taken at
   the pixel centre.
3. **Linear detuning**: H(a, (u - u_c)/b) (L182) instead of the exp mapping.
4. **One image within half a box** (L249-L281) and the **early stop** ``tautail`` = 1e-7
   (``spectra.py`` L145): walking out from the pixel of the line centre, stop after the first pixel
   that receives less than tautail from this cell.

``fake_chord_absorbers`` builds item 1 as SkewerJAX absorbers; ``fake_emulated_tau`` adds 2-4 to
any absorbers grouped by cell (also those of an exact deposition, to isolate each item).
Each piece can be switched off (``MatchedIntegration``), which is how the effect chain (``comparison.effects``)
isolates them; ``matched_tau`` evaluates any combination on a ``SimulationSkewers``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from skewerjax.simulations.absorbers import Absorbers, tau_absorbers
from skewerjax.simulations.snapshot_skewers import IntegrationOptions, cached

TOPHAT_NORM = 3.0 / (4.0 * np.pi)
FAKE_TAUTAIL = 1e-7


def fake_chord_absorbers(origin, direction, path_length: float, gas_positions, gas_size, n_cell, T_cell,
                         velocity_cell, box: float, a_data: float, species: str, mass_amu: float, *,
                         n_grid: int = 8, interior_only: bool = True, cells=None) -> Absorbers:
    """fake_spectra's chord sampling of the top-hat spheres touching one straight periodic skewer.

    Parameters
    ----------
    origin, direction : (3,) start [cMpc] and unit direction of the skewer; path_length [cMpc]
        (= box for fake_spectra's sightlines).
    gas_* : the cells (positions [cMpc] in [0, box), size V^(1/3) [cMpc], ion density [cm^-3],
        T [K], velocity (M, 3) [km/s]).
    n_grid : intervals per chord (fake_spectra NGRID = 8).
    interior_only : True (fake_spectra) keeps the n_grid - 1 interior points, each 1/n_grid of the
        chord: 7/8 of the column. False uses the n_grid + 1 trapezoid points (ends weight 1/2): the
        whole column, i.e. fake_spectra's points with the normalisation restored.
    cells : optional subset of cell indices.

    Returns
    -------
    Absorbers for one skewer (N_sk = 1), with ``cell`` the cell of each chord point and
    ``meta["zmax_s"]`` the chord centre of each cell [cMpc].
    """
    e = np.asarray(direction, float) / np.linalg.norm(direction)
    idx = np.arange(len(gas_positions)) if cells is None else np.asarray(cells)
    d = np.asarray(gas_positions, float)[idx] - np.asarray(origin, float)
    d -= box * np.round(d / box)
    s_c = np.mod(d @ e, path_length)
    dr2 = np.sum(d * d, axis=1) - (d @ e) ** 2
    h = np.asarray(gas_size, float)[idx]
    touch = h * h - dr2 > 0                      # absorption.cpp L235
    idx, s_c, zr = idx[touch], s_c[touch], np.sqrt(h[touch] ** 2 - dr2[touch])
    if interior_only:
        k = np.arange(1, n_grid)
        wk = np.full(k.size, 1.0 / n_grid)
    else:
        k = np.arange(0, n_grid + 1)
        wk = np.full(k.size, 1.0 / n_grid); wk[[0, -1]] *= 0.5
    pos = s_c[:, None] + zr[:, None] * (2.0 * k[None, :] / n_grid - 1.0)
    ln = 2.0 * zr[:, None] * wk[None, :]
    n = TOPHAT_NORM * np.asarray(n_cell, float)[idx]
    vl = np.asarray(velocity_cell, float)[idx] @ e
    rep = lambda x: np.repeat(x, k.size)
    ab = Absorbers(s=pos.ravel()[None], n=rep(n)[None], length=ln.ravel()[None], T=rep(np.asarray(T_cell, float)[idx])[None],
                   v_los=rep(vl)[None], cell=rep(idx)[None], point=-np.ones((1, pos.size), int), a_data=float(a_data),
                   species=species, mass_amu=float(mass_amu),
                   meta=dict(kind="fake_chord", zmax_s=s_c, cells=idx, v_los_cell=vl, T_cell=np.asarray(T_cell, float)[idx],
                             n_per_cell=k.size))
    return ab


def fake_pixel_points(b_kms, dv: float):
    """fake_spectra's sub-pixel trapezoid for one cell (``tau_kern_outer``): (offsets, weights)."""
    if dv < b_kms / 2.0:
        return np.zeros(1), np.ones(1)
    npts = int(2 * np.ceil(dv / b_kms) + 1)
    off = (np.linspace(0.0, 1.0, npts) - 0.5) * dv
    w = np.ones(npts); w[[0, -1]] = 0.5
    return off, w / w.sum()


def apply_tautail(tau_cells, centre_pixel, tautail: float = FAKE_TAUTAIL) -> np.ndarray:
    """fake_spectra's early stop (``absorption.cpp`` L253-L281) applied to per-cell tau.

    tau_cells : (N_cells, nbins) periodic per-cell optical depth; centre_pixel : (N_cells,) zmax.
    Walking forward from zmax and backward from zmax - 1 (up to nbins/2 pixels each, indices mod
    nbins), every pixel is kept up to and including the first one whose value is < tautail.
    """
    tau_cells = np.asarray(tau_cells, float)
    nc, nb = tau_cells.shape
    half = nb // 2
    out = np.zeros_like(tau_cells)
    rows = np.arange(nc)[:, None]
    for start, step in ((np.asarray(centre_pixel), 1), (np.asarray(centre_pixel) - 1, -1)):
        j = np.mod(start[:, None] + step * np.arange(half)[None, :], nb)
        vals = tau_cells[rows, j]
        above = vals >= tautail
        keep = np.concatenate([np.ones((nc, 1), bool), np.cumprod(above, axis=1)[:, :-1].astype(bool)], axis=1)
        np.add.at(out, (np.broadcast_to(rows, j.shape)[keep], j[keep]), vals[keep])
    return out


def fake_emulated_tau(ab: Absorbers, line_bundle, frame, nbins: int, dv: float, *, T_scale: float,
                      cell_centre_s, cell_v_los, cell_T, pixel_average: bool = True, tautail: bool = True,
                      detuning: str = "linear", n_scale: float = 1.0, group: int = 100, **kw) -> np.ndarray:
    """tau (N_lines, nbins) of one skewer (``ab.n_skewers == 1``) with fake_spectra's per-cell pixel
    average, detuning and early stop, each optional.

    The absorbers are grouped by ``ab.cell`` (all the pieces or chord points of one cell form one
    fake_spectra "particle"), so this works for ``fake_chord_absorbers`` and for any deposition.

    Parameters
    ----------
    frame : ``SnapshotFrame`` with ``period_kms = nbins dv``.
    group : cells per batch; memory ~ group x (pieces per cell) x nbins x 8 bytes.
    cell_centre_s, cell_v_los, cell_T : per-cell arrays indexed by ``ab.cell``: the cell's position
        along the skewer [cMpc] (fake_spectra's line centre is velfac * that + v_pec,
        ``absorption.cpp`` L237), LoS peculiar velocity [km/s] and temperature [K].
    """
    if ab.n_skewers != 1:
        raise ValueError("fake_emulated_tau works on one skewer at a time.")
    v = (np.arange(nbins) + 0.5) * dv
    live = (ab.n[0] > 0) & (ab.cell[0] >= 0)
    cells = np.unique(ab.cell[0][live])
    T_c = np.asarray(cell_T, float)[cells]
    b = np.sqrt(2.0 * 8314.46262102654 * T_c * T_scale / ab.mass_amu) / 1e3      # phys.kB, as gas_physics.doppler_b_kms
    rules = [fake_pixel_points(bb, dv) if pixel_average else (np.zeros(1), np.ones(1)) for bb in b]
    key = np.array([len(r[0]) for r in rules])
    u_c = frame.velfac * (np.asarray(cell_centre_s, float)[cells] - frame.s_origin) + np.asarray(cell_v_los, float)[cells]
    zmax = np.floor(u_c / dv).astype(int)
    order = np.argsort(ab.cell[0], kind="stable")
    sorted_cells = ab.cell[0][order]
    total = np.zeros((line_bundle.n_lines, nbins))
    for npts in np.unique(key):
        sel = np.nonzero(key == npts)[0]
        off, wts = rules[sel[0]]
        for g0 in range(0, sel.size, group):
            cs = sel[g0:g0 + group]
            lo, hi = np.searchsorted(sorted_cells, cells[cs], "left"), np.searchsorted(sorted_cells, cells[cs], "right")
            pick = np.concatenate([order[a:b_] for a, b_ in zip(lo, hi)])
            owner = np.repeat(np.arange(cs.size), hi - lo)
            sub = ab.with_values(**{k: getattr(ab, k)[:, pick] for k in ("s", "n", "length", "T", "v_los", "cell", "point")})
            t = tau_absorbers(sub, line_bundle, v, frame, T_scale=T_scale, n_scale=n_scale, detuning=detuning,
                              pixel_offsets=off, pixel_weights=wts, per_absorber=True, check=None, **kw)[:, 0, :pick.size]
            for li in range(line_bundle.n_lines):
                per_cell = np.zeros((cs.size, nbins))
                np.add.at(per_cell, owner, t[li])
                total[li] += (apply_tautail(per_cell, zmax[cs]) if tautail else per_cell).sum(axis=0)
    return total


# --------------------------------------------------------------------------- matched options and tau
@dataclass(frozen=True)
class MatchedIntegration(IntegrationOptions):
    """``IntegrationOptions`` plus fake_spectra's per-cell scheme, one switch per difference. Defaults: SkewerJAX.

    chord : "segments" (the exact pieces of the dual segments) or "fake_7pt" (7 interior points of every top-hat
        chord, weight 1/8: 7/8 of the column; needs ``deposition="tophat"`` and the snapshot frame).
    tautail : the per-cell early stop at tau < 1e-7; pixel_average : the per-cell pixel average for b < 2 dv.
    The per-cell options need the snapshot frame and fake_spectra's periodic pixel grid (``spectral_grid()``).
    """
    chord: str = "segments"
    tautail: bool = False
    pixel_average: bool = False

    @property
    def per_cell(self) -> bool:
        return self.chord == "fake_7pt" or self.tautail or self.pixel_average


def matched_tau(sk, ion, line_bundle, velocities_kms, ext, integ: IntegrationOptions, override: Optional[dict] = None,
                override_key: str = "", check: Optional[float] = 0.25) -> np.ndarray:
    """(N_sk, N_v) tau of ``SimulationSkewers`` ``sk`` for any extraction and (matched) integration options.

    Without per-cell options this is ``sk.tau``; with them, fake_spectra's per-cell scheme on straight skewers
    (cached in ``sk.cache`` under the same key format).
    """
    if not getattr(integ, "per_cell", False):
        return sk.tau(ion, line_bundle, velocities_kms, ext, integ, override=override, override_key=override_key, check=check)
    if override is not None and not override_key:
        override_key = "override"
    v = np.asarray(velocities_kms, dtype=np.float64)
    build = lambda: dict(tau=_per_cell_tau(sk, ion, line_bundle, v, ext, integ, override))
    return cached(sk.cache, sk.tau_key(ion, line_bundle, v, ext, integ, override_key), build, enabled=sk.name is not None)["tau"]


def _per_cell_tau(sk, ion, line_bundle, v, ext, integ: MatchedIntegration, override):
    from skewerjax.simulations import gas_physics as gp
    if integ.frame != "snapshot":
        raise ValueError("fake_spectra's per-cell scheme (fake_7pt / tautail / pixel_average) is defined in its snapshot frame.")
    if integ.chord == "fake_7pt" and ext.deposition != "tophat":
        raise ValueError("chord='fake_7pt' is fake_spectra's top-hat chord quadrature: use deposition='tophat'.")
    dv, nbins = float(v[1] - v[0]), v.size
    if not (np.allclose(v, (np.arange(nbins) + 0.5) * dv) and np.isclose(nbins * dv, sk.header.vmax_kms)):
        raise ValueError("fake_spectra's per-cell scheme needs its periodic pixel grid (spectral_grid()).")
    T_scale = gp.FAKE_SPECTRA_B2_RATIO if ext.b_constants == "fake_spectra" else 1.0
    fr = sk.frame(integ.frame, integ.periodic)
    ci = sk.cell_inputs(ion, ext, override)
    A = None if integ.chord == "fake_7pt" else sk.absorbers(ion, ext, override)
    out = []
    for k in range(sk.n_skewers):
        e = sk.points[k, -1] - sk.points[k, 0]
        e = e / np.linalg.norm(e)
        if integ.chord == "fake_7pt":
            Ak = fake_chord_absorbers(sk.points[k, 0], e, sk.box, ci["pos"], ci["size"], ci["n"], ci["T"], ci["vel"],
                                      sk.box, sk.header.a, ci["species"], ci["mass"])
        else:
            Ak = A.with_values(**{f: getattr(A, f)[k:k + 1] for f in ("s", "n", "length", "T", "v_los", "cell", "point")})
        d = ci["pos"] - sk.points[k, 0]
        d -= sk.box * np.round(d / sk.box)
        out.append(fake_emulated_tau(Ak, line_bundle, fr, nbins, dv, T_scale=T_scale, cell_centre_s=np.mod(d @ e, sk.box),
                                     cell_v_los=ci["vel"] @ e, cell_T=ci["T"], pixel_average=integ.pixel_average,
                                     tautail=integ.tautail, detuning=integ.detuning, n_scale=integ.normalisation)[0])
    return np.stack(out)
