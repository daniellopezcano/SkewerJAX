"""Skewers through a simulation snapshot: one object that runs layers 1-6 for any options.

``SimulationSkewers`` holds the sampling points of a set of skewers (a ``SkewerBundle``'s points),
the gas cells selected near them, the deposited segments and the frames. It returns per-cell
properties, deposited absorbers, columns along the skewers and tau(v) for any combination of

* ``ExtractionOptions``: how the fields are extracted from the cells (deposition geometry, mass and
  temperature constants, ionisation model): layers 3 and 5;
* ``IntegrationOptions``: how the optical-depth integral is evaluated (frame, chord quadrature,
  normalisation, detuning, per-cell early stop, pixel treatment, periodicity): layer 6.

The defaults of both are SkewerJAX's. ``FAKE_SPECTRA_EXTRACTION`` and ``FAKE_SPECTRA_INTEGRATION``
reproduce fake_spectra (validation only). ``EXTRACTION_EFFECTS`` and ``INTEGRATION_EFFECTS`` list
every difference between the two as one switch each; ``run_effects`` evaluates them as a cumulative
chain (which telescopes exactly from fake_spectra's options to SkewerJAX's) and one at a time.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, replace, asdict
from pathlib import Path
from typing import Callable, Optional, Sequence

import h5py
import numpy as np
import jax.numpy as jnp

from skewerjax.constants.physical_constants import phys
from skewerjax.skewers.skewers import path_geometry, axis_aligned_skewer_points
from skewerjax.simulations import cell_selection as sel
from skewerjax.simulations import gas_physics as gp
from skewerjax.fields import deposition as dep
from skewerjax.optical_depth import absorbers as ab_
from skewerjax.optical_depth import fake_spectra_emulation as fse
from skewerjax.fields.fields import Fields, ScalarField, VectorField, SampledField


# --------------------------------------------------------------------------- options
@dataclass(frozen=True)
class ExtractionOptions:
    """Field extraction (layers 3 and 5). Defaults: SkewerJAX.

    deposition : "voronoi" (cells of the mesh-generating points), "tophat" (spheres of radius
        V^(1/3), fake_spectra) or "points" (summed sphere values at the points; previous default).
    mass_unit : "amu" (A m_u) or "proton" (A m_p, fake_spectra) in n_X = rho X / (A m).
    T_constants : "codata" or "fake_spectra" (its m_p, k_B) in T = (gamma - 1) mu m_p u / k_B.
    b_constants : "codata" (b^2 = 2 k_B T / (A m_u)) or "fake_spectra" (2 k T / (A m_p) with the C
        engine's constants; applied as T x FAKE_SPECTRA_B2_RATIO inside the profile only).
    x_HI_model : "stored_sf" (stored; star-forming cells recomputed), "stored", "fake_sf"
        (fake_spectra's default, ``sf_neutral=True``) or "rahmati_all" (``gas_physics.neutral_fraction``).
    cloudy_density : "hydrogen" (hden = n_H) or "fake_spectra" (0.774132 rho / m_p); metals only.
    cloudy_interpolation : "linear" (bilinear in log f) or "cubic" (fake_spectra); metals only.
    """
    deposition: str = "voronoi"
    mass_unit: str = "amu"
    T_constants: str = "codata"
    b_constants: str = "codata"
    x_HI_model: str = "stored_sf"
    cloudy_density: str = "hydrogen"
    cloudy_interpolation: str = "linear"

    def replace(self, **kw) -> "ExtractionOptions":
        return replace(self, **kw)


@dataclass(frozen=True)
class IntegrationOptions:
    """Evaluation of the optical-depth integral (layer 6). Defaults: SkewerJAX.

    frame : "exact" (a(x) along the path from the cosmology) or "snapshot" (one a, linear Hubble flow).
    chord : "segments" (every deposited piece with its exact length) or "fake_7pt" (fake_spectra's 7
        interior points of each top-hat chord, weight 1/8 each: 7/8 of the column; needs
        ``deposition="tophat"``, straight skewers and the snapshot frame).
    normalisation : multiplies every density (8/7 restores the full column of "fake_7pt").
    detuning : "exact" (nu = (c / lambda) expm1(-dv / c)) or "linear" (fake_spectra).
    tautail : fake_spectra's per-cell early stop at 1e-7 (snapshot frame and pixel grid only).
    pixel_average : fake_spectra's per-cell pixel average when the pixel is wider than b/2 (idem).
    periodic : True wraps every detuning to its nearest periodic image (period: the box length in
        velocity); False is the direct integral.
    """
    frame: str = "exact"
    chord: str = "segments"
    normalisation: float = 1.0
    detuning: str = "exact"
    tautail: bool = False
    pixel_average: bool = False
    periodic: bool = True

    def replace(self, **kw) -> "IntegrationOptions":
        return replace(self, **kw)


FAKE_SPECTRA_EXTRACTION = ExtractionOptions(deposition="tophat", mass_unit="proton", T_constants="fake_spectra",
                                            b_constants="fake_spectra", x_HI_model="fake_sf", cloudy_density="fake_spectra",
                                            cloudy_interpolation="cubic")
FAKE_SPECTRA_INTEGRATION = IntegrationOptions(frame="snapshot", chord="fake_7pt", detuning="linear", tautail=True,
                                              pixel_average=True)


def relevant_options(ext: ExtractionOptions, ion: gp.Ion) -> ExtractionOptions:
    """``ext`` with the options that do not act on ``ion`` set to their defaults (x_HI_model for
    metals, the CLOUDY options for H I): equal results compare equal."""
    d = ExtractionOptions()
    if ion.is_HI:
        return ext.replace(cloudy_density=d.cloudy_density, cloudy_interpolation=d.cloudy_interpolation)
    return ext.replace(x_HI_model=d.x_HI_model)


_ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10}


def ion_from_species(species: str) -> gp.Ion:
    """The ``Ion`` of a ``LineBundle`` species key ("HI" -> H I, "SiIII" -> Si III, "CIV" -> C IV)."""
    import re
    m = re.fullmatch(r"([A-Z][a-z]?)([IVX]+)", species)
    if m is None or m.group(2) not in _ROMAN:
        raise ValueError(f"cannot read an element and an ionisation stage from species {species!r}.")
    return gp.Ion(m.group(1), _ROMAN[m.group(2)])


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
    Effect("neutral fraction", "extraction", dict(x_HI_model="stored_sf"), ions="HI",
           what="fake_spectra's star-forming rule (rho / m_p threshold) -> stored, SF cells recomputed"),
    Effect("CLOUDY density", "extraction", dict(cloudy_density="hydrogen"), ions="metals", what="hden = 0.774 rho / m_p -> n_H"),
    Effect("CLOUDY interpolation", "extraction", dict(cloudy_interpolation="linear"), ions="metals", what="cubic -> bilinear in log f"),
    Effect("deposition", "extraction", dict(deposition="voronoi"), what="top-hat spheres V^(1/3) -> Voronoi cells"),
)
INTEGRATION_EFFECTS = (
    Effect("tautail", "integration", dict(tautail=False), what="per-cell early stop at tau < 1e-7 -> full profiles"),
    Effect("pixel average", "integration", dict(pixel_average=False), what="per-cell pixel average for b < 2 dv -> pixel centres"),
    Effect("detuning", "integration", dict(detuning="exact"), what="H(a, dv / b) linear -> nu = (c / lambda) expm1(-dv / c)"),
    Effect("7/8 normalisation", "integration", dict(normalisation=8 / 7), what="7 chord points of weight 1/8 -> weights 1/7"),
    Effect("chord quadrature", "integration", dict(chord="segments", normalisation=1.0), isolated=dict(chord="segments", normalisation=7 / 8),
           what="7 points per chord -> the exact chord (pieces of the dual segments)"),
    Effect("frame", "integration", dict(frame="exact"), isolated_from="chord quadrature",
           what="one a, u = a H s + v_pec -> a(x) along the path, v_dagger"),
)


def effect_chain(effects: Sequence[Effect], ext0: ExtractionOptions, int0: IntegrationOptions, ion: gp.Ion):
    """The cumulative chain and the isolated switches of ``effects`` applicable to ``ion``.

    Returns
    -------
    links : list of (label, ExtractionOptions, IntegrationOptions); links[0] is ("start", ext0, int0).
    isolated : {label: (ExtractionOptions, IntegrationOptions, base label)}.
    """
    def apply(ext, integ, kind, d):
        return (ext.replace(**d), integ) if kind == "extraction" else (ext, integ.replace(**d))
    for e in effects:                     # a switch that does not act on this ion is set from the start
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


# --------------------------------------------------------------------------- HDF5 result cache
def h5_write(group, d: dict):
    """Arrays and scalars as datasets of an HDF5 group (unicode stored as bytes, gzip above 4096)."""
    for k, v in d.items():
        a = np.asarray(v)
        if a.dtype.kind == "U":
            a = a.astype("S")
        group.create_dataset(k, data=a, compression="gzip" if a.size > 4096 else None)


def h5_read(group) -> dict:
    out = {}
    for k in group:
        v = group[k][()]
        out[k] = np.asarray(v).astype(str) if np.asarray(v).dtype.kind == "S" else v
    return out


def cached(path, name: str, build: Callable[[], dict], enabled: bool = True, attrs: Optional[dict] = None) -> dict:
    """Dict ``name`` from the HDF5 file ``path`` (one group per result), built and stored if missing."""
    if path is None or not enabled:
        return build()
    path = Path(path)
    if path.exists():
        with h5py.File(path, "r") as f:
            if name in f:
                return h5_read(f[name])
    d = build()
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "a") as f:
        g = f.create_group(name)
        h5_write(g, d)
        g.attrs["created"] = time.ctime()
        for k, v in (attrs or {}).items():
            g.attrs[k] = v
    return d


# --------------------------------------------------------------------------- skewers through a snapshot
class SimulationSkewers:
    """Skewers through one snapshot (layers 1-6 for any options).

    Parameters
    ----------
    reader : layer-1 reader (e.g. ``ArepoHDF5Reader``).
    points : (N_sk, N, 3) sampling points [cMpc], unwrapped along each path (a ``SkewerBundle``'s points).
    cosmology : ``Cosmology`` for the exact frame (default: flat LCDM of the snapshot header).
    cells : ``GasCells`` already selected for these points (default: ``select_cells_near_paths``).
    s_origin : path coordinate [cMpc] where the box starts (extended sampling; 0 otherwise).
    name, cache : a name for this set of skewers and an HDF5 file where tau results are cached
        (both needed for caching).
    scale_factors : optional (N_sk, N) a(x) at the points (e.g. a ``SkewerBundle``'s); then the exact frame
        uses them instead of placing the snapshot's a at the centre of the box (``from_bundle``).
    """

    def __init__(self, reader, points, *, cosmology=None, cells=None, s_origin: float = 0.0, max_rows: int = 4_000_000,
                 name: Optional[str] = None, cache=None, scale_factors=None):
        self.reader, self.header = reader, reader.header
        self.box = self.header.box_cMpc
        self.points = np.asarray(points, dtype=np.float64)
        t = time.perf_counter()
        self.cells = cells if cells is not None else sel.select_cells_near_paths(reader, self.points, mode="voronoi", max_rows=max_rows)
        self.selection_s = time.perf_counter() - t
        g = path_geometry(jnp.asarray(self.points))
        self.s = np.asarray(g["s"])
        self.line_elements = np.asarray(g["line_elements"])
        self.tangents = np.asarray(g["tangents"])
        self.s_origin = float(s_origin)
        if cosmology is None:
            from skewerjax.cosmology.cosmology import Cosmology
            H = self.header
            cosmology = Cosmology(100 * H.h * 1e6 * phys.Gyr_to_s / phys.Gpc_to_m, 0.0, H.omega_m, 0.0, H.omega_l)
        self.cosmology = cosmology
        self.name, self.cache = name, cache
        self.scale_factors = None if scale_factors is None else np.asarray(scale_factors, dtype=np.float64)
        if self.scale_factors is not None and self.scale_factors.shape != self.points.shape[:2]:
            raise ValueError(f"scale_factors must be (N_sk, N) = {self.points.shape[:2]}, got {self.scale_factors.shape}.")
        self._seg, self._props, self._frames = {}, {}, {}

    @classmethod
    def from_bundle(cls, reader, skewer_bundle, ii_cosmology: int = 0, **kw):
        """Skewers through the snapshot along the points of a ``SkewerBundle`` (any polylines) [cMpc].

        The bundle's scale factors (``quadri_vectors[ii_cosmology][..., 0]``, from its cosmology or
        ``manual_scale_factors``) are used by the exact frame. SkewerJAX's convention for simulations is
        that the observer is before the first point: a(x) must not increase along the points."""
        qv = np.asarray(skewer_bundle.quadri_vectors[ii_cosmology])
        sk = cls(reader, qv[..., 1:], scale_factors=qv[..., 0], **kw)
        sk.skewer_bundle = skewer_bundle
        return sk

    @classmethod
    def axis_aligned(cls, reader, cofm_code, axis, spacing_kms: float, *, extend_cMpc: float = 0.0, **kw):
        """Straight skewers along box axes (``axis`` 1/2/3 = x/y/z) through the code-unit points
        ``cofm_code`` [ckpc/h], sampled every ``spacing_kms`` of Hubble flow (a H(z) ds);
        ``extend_cMpc`` adds path before and after the box (periodic gas)."""
        H = reader.header
        pts = axis_aligned_skewer_points(np.atleast_2d(cofm_code) * H.code_length_cMpc, axis, H.box_cMpc,
                                         spacing_kms / H.velfac_kms_per_cMpc, extend=extend_cMpc)
        sk = cls(reader, pts, s_origin=extend_cMpc, **kw)
        sk.axis = np.atleast_1d(np.asarray(axis))
        return sk

    @property
    def n_skewers(self) -> int:
        return self.points.shape[0]

    def spectral_grid(self, res_kms: float = 1.0, frame=None):
        """A periodic pixel grid over one period of the box: pixel centres [km/s] and width.

        The period is that of ``frame`` (``period_kms``; the exact frame's is c ln(a_0 / a_L)), or by
        default the linear v_max = a H L of fake_spectra's grid: n_bins = int(P / res), dv = P / n_bins.
        A non-periodic integral takes any grid instead."""
        P = self.header.vmax_kms if frame is None else frame.period_kms
        if P is None:
            raise ValueError("a non-periodic frame has no period: choose the velocity grid freely.")
        nbins = int(P / res_kms)
        if nbins < 2:
            raise ValueError(f"res = {res_kms} km/s leaves fewer than 2 pixels in P = {P:.1f} km/s.")
        dv = P / nbins
        return (np.arange(nbins) + 0.5) * dv, dv

    # ------------------------------------------------------------------ layers 3 and 5
    def properties(self, ion: gp.Ion, ext: ExtractionOptions = ExtractionOptions()) -> dict:
        """Per-cell n_H, T, ion fraction and ion density of the selected cells (``ion_properties``)."""
        key = (ion.species, ext.mass_unit, ext.T_constants, ext.x_HI_model, ext.cloudy_density, ext.cloudy_interpolation)
        if key not in self._props:
            if not ion.is_HI and not hasattr(self, "_cloudy"):
                self._cloudy = gp.CloudyIonTable(self.header.redshift)
            P = gp.ion_properties(self.cells, ion, mass_unit=ext.mass_unit, T_constants=ext.T_constants, x_HI_model=ext.x_HI_model,
                                  cloudy=getattr(self, "_cloudy", None), cloudy_density=ext.cloudy_density,
                                  cloudy_interpolation=ext.cloudy_interpolation)
            self._props[key] = {k: (v if isinstance(v, (str, float)) else np.asarray(v)) for k, v in P.items()}
        return self._props[key]

    def cell_inputs(self, ion: gp.Ion, ext: ExtractionOptions = ExtractionOptions(), override: Optional[dict] = None) -> dict:
        """What the deposition needs per cell: pos, size [cMpc], n [physical cm^-3], T [K], vel (N, 3)
        [km/s], mass [u], species. ``override`` replaces any of them (e.g. another code's values)."""
        need = override is None or not {"n", "T"} <= set(override)
        P = self.properties(ion, ext) if need else None
        d = dict(pos=np.asarray(self.cells.positions), size=np.asarray(self.cells.size_cMpc),
                 n=None if P is None else P["n_ion"], T=None if P is None else P["T"],
                 vel=np.asarray(self.cells.fields["velocity"]), mass=ion.mass_amu, species=ion.species)
        if override:
            d.update(override)
        return d

    def segments(self, kind: str, inputs: Optional[dict] = None) -> dep.SegmentComponents:
        """Deposited pieces of every dual segment ("voronoi" or "tophat"); cached for the selected cells."""
        if kind not in ("voronoi", "tophat"):
            raise ValueError(f"segments kind must be 'voronoi' or 'tophat', got {kind!r}.")
        build = lambda pos, size: (dep.voronoi_segments(self.points, pos, self.box, gas_size=size) if kind == "voronoi"
                                   else dep.tophat_segments(self.points, pos, size, self.box))
        if inputs is not None:                                   # other per-cell positions / sizes: not cached
            return build(inputs["pos"], inputs["size"])
        if kind not in self._seg:
            self._seg[kind] = build(np.asarray(self.cells.positions), np.asarray(self.cells.size_cMpc))
        return self._seg[kind]

    def absorbers(self, ion: gp.Ion, ext: ExtractionOptions = ExtractionOptions(), override: Optional[dict] = None) -> ab_.Absorbers:
        """Deposited absorbers of every skewer (one per piece of cell in each dual segment)."""
        ci = self.cell_inputs(ion, ext, override)
        if ext.deposition == "points":
            tan = self.tangents.reshape(-1, 3)
            vel = ci["vel"]
            pv = dep.tophat_point_values(self.points, ci["pos"], ci["size"], self.box, ci["n"], ci["T"],
                                         lambda c, p: np.sum(vel[c] * tan[p], -1))
            return ab_.absorbers_from_point_values(self.points, pv, self.header.a, ci["species"], ci["mass"])
        seg = self.segments(ext.deposition, ci if override and "pos" in override else None)
        return ab_.absorbers_from_components(seg, self.points, ci["n"], ci["T"], ci["vel"], self.header.a, ci["species"], ci["mass"])

    def column_per_pixel(self, ion: gp.Ion, ext: ExtractionOptions = ExtractionOptions(), nbins: Optional[int] = None,
                         override: Optional[dict] = None, weight: Optional[str] = None) -> np.ndarray:
        """(N_sk, nbins) physical column [cm^-2] in equal position pixels of the box; with
        ``weight="T"`` or ``"v_los"``, the column-weighted mean of that quantity per pixel instead."""
        nbins = nbins or self.header.pixel_grid(1.0)[0]
        A = self.absorbers(ion, ext, override)
        N = ab_.column_per_pixel(A, self.box, nbins, s_origin=self.s_origin)
        if weight is None:
            return N
        q = getattr(A, weight)
        NQ = ab_.column_per_pixel(A.with_values(n=A.n * q), self.box, nbins, s_origin=self.s_origin)
        return np.where(N > 0, NQ / np.maximum(N, 1e-300), 0.0)

    def point_values(self, ion: gp.Ion, ext: ExtractionOptions = ExtractionOptions()) -> dict:
        """The deposited fields at the sampling points: segment averages over each point's dual segment.

        n = sum_p n_p l_p / w_i (the exact column of the segment over its line element); T and the peculiar
        velocity are column-weighted over the pieces (length-weighted where the segment holds no ion).

        Returns
        -------
        dict of (N_sk, N) arrays: ``n`` [physical cm^-3 at the snapshot], ``n_com`` [comoving cm^-3],
        ``T`` [K], ``v_los`` [km/s, along the path], ``column`` [cm^-2, physical, per segment]; ``velocity``
        (N_sk, N, 3) [km/s]; ``line_elements`` (N_sk, N) [cMpc].
        """
        if ext.deposition not in ("voronoi", "tophat"):
            raise ValueError("point_values needs a segment deposition ('voronoi' or 'tophat').")
        seg, P = self.segments(ext.deposition), self.properties(ion, ext)
        n_c, T_c, v_c = np.asarray(P["n_ion"]), np.asarray(P["T"]), np.asarray(self.cells.fields["velocity"])
        live = seg.cell >= 0
        cc = np.where(live, seg.cell, 0)
        nl = np.where(live, seg.weight * n_c[cc] * seg.length, 0.0)                # n l of every piece
        ll = np.where(live, seg.length, 0.0)
        col = nl.sum(-1)
        wt = np.where((col > 0)[..., None], nl, ll)
        norm = np.maximum(wt.sum(-1), 1e-300)
        T = (wt * T_c[cc]).sum(-1) / norm
        vel = (wt[..., None] * v_c[cc]).sum(-2) / norm[..., None]
        n = col / np.maximum(self.line_elements, 1e-300)
        a = self.header.a
        return dict(n=n, n_com=n * a ** 3, T=T, velocity=vel, v_los=np.sum(vel * self.tangents, -1),
                    column=col * ab_.MPC_CM * a, line_elements=self.line_elements)

    def fields(self, species, ext: ExtractionOptions = ExtractionOptions()) -> Fields:
        """A ``Fields`` of the simulation at the sampling points, for the species keys ``species`` (e.g.
        ``["HI", "SiIII"]``, as in ``LineBundle.species_keys``).

        Every sampler is a ``SampledField`` of ``point_values``: ``temperature`` [K], ``velocity`` [km/s],
        ``number_density`` [comoving cm^-3, what ``OpticalDepthCalculator`` takes], ``turbulence`` and
        ``collisional`` (0). Call them with this bundle's quadri-vectors, e.g. ``fields.sample(qv, los)``.
        With ``line_elements`` = the bundle's, ``OpticalDepthCalculator`` integrates the segment averages;
        ``optical_depth`` integrates the pieces of every segment instead (the default, exact for any spacing).
        """
        out = {}
        for sp in ([species] if isinstance(species, str) else list(species)):
            pv = self.point_values(ion_from_species(sp), ext)
            zeros = np.zeros_like(pv["T"])
            out[sp] = {"temperature": ScalarField(SampledField(pv["T"])), "velocity": VectorField(SampledField(pv["velocity"])),
                       "number_density": ScalarField(SampledField(pv["n_com"])),
                       "turbulence": ScalarField(SampledField(zeros)), "collisional": ScalarField(SampledField(zeros))}
        return Fields(out)

    def optical_depth(self, line_bundle, velocities_kms=None, ext: ExtractionOptions = ExtractionOptions(),
                      integ: IntegrationOptions = IntegrationOptions(), check: Optional[float] = 0.25):
        """tau of every line of ``line_bundle`` (any species): (velocities [km/s], tau (N_lines, N_sk, N_v)).

        Each line is integrated in its own line-centred frame, as ``OpticalDepthCalculator`` does. Without
        ``velocities_kms`` the grid is one period of the box in 1 km/s pixels (``spectral_grid``).
        """
        from skewerjax.constants.lines import LineBundle
        if velocities_kms is None:
            velocities_kms = self.spectral_grid(1.0, self.frame(integ.frame, integ.periodic))[0]
        v = np.asarray(velocities_kms, dtype=np.float64)
        taus = [self.tau(ion_from_species(line.species), LineBundle([line.id], ref_line_id=line.id), v, ext, integ, check=check)
                for line in line_bundle.lines]
        return v, np.stack(taus)

    # ------------------------------------------------------------------ layer 6
    def frame(self, kind: str = "exact", periodic: bool = True):
        """``ExactFrame`` (a(s) with a_snap at the centre of the box, a_dagger = a at the box start) or
        ``SnapshotFrame``; ``periodic`` sets the velocity period to the box length in velocity."""
        key = (kind, periodic)
        if key in self._frames:
            return self._frames[key]
        H = self.header
        if kind == "snapshot":
            fr = ab_.SnapshotFrame(H.a, H.velfac_kms_per_cMpc, period_kms=H.vmax_kms if periodic else None, s_origin=self.s_origin)
        elif kind == "exact" and self.scale_factors is not None:
            a = self.scale_factors
            if np.any(a[:, 0] < a[:, -1]):
                raise ValueError("SimulationSkewers puts the observer before the first point, so a(x) must decrease along the "
                                 "points; reverse the points of the SkewerBundle or define its ICs at the observer's end.")
            period = None
            if periodic:
                closed = (np.allclose(self.s[:, -1], self.box) and
                          np.allclose(np.mod(self.points[:, -1] - self.points[:, 0] + self.box / 2, self.box) - self.box / 2, 0, atol=1e-9))
                if not closed:
                    raise ValueError("periodic=True needs skewers that cross the box once and close on themselves "
                                     "(first and last points the same place of the box); use periodic=False.")
                period = float(np.mean(phys.c_kms * np.log(a[:, 0] / a[:, -1])))
            fr = ab_.ExactFrame(self.s, a, a_dagger_value=float(a[0, 0]), period_kms=period)
        elif kind == "exact":
            s_ref = np.full(self.n_skewers, self.s_origin + 0.5 * self.box)
            a = ab_.scale_factors_along_paths(self.s, H.a, self.cosmology, s_ref=s_ref)
            a0 = np.exp([np.interp(self.s_origin, self.s[k], np.log(a[k])) for k in range(self.n_skewers)])
            aL = np.exp([np.interp(self.s_origin + self.box, self.s[k], np.log(a[k])) for k in range(self.n_skewers)])
            period = float(np.mean(phys.c_kms * np.log(a0 / aL))) if periodic else None
            fr = ab_.ExactFrame(self.s, a, a_dagger_value=float(a0[0]), period_kms=period)
        else:
            raise ValueError(f"frame must be 'exact' or 'snapshot', got {kind!r}.")
        self._frames[key] = fr
        return fr

    def tau(self, ion: gp.Ion, line_bundle, velocities_kms, ext: ExtractionOptions = ExtractionOptions(),
            integ: IntegrationOptions = IntegrationOptions(), override: Optional[dict] = None, override_key: str = "",
            check: Optional[float] = 0.25) -> np.ndarray:
        """(N_sk, N_v) optical depth of the first line of ``line_bundle`` (one species) for ``ext`` and
        ``integ``; cached when the object has a ``name`` and a ``cache`` (``override_key`` names an override)."""
        if override is not None and not override_key:
            override_key = "override"
        v = np.asarray(velocities_kms, dtype=np.float64)
        opts = {**asdict(relevant_options(ext, ion)), **asdict(integ)}
        geo = hashlib.md5(self.points.tobytes() + (b"" if self.scale_factors is None else self.scale_factors.tobytes())).hexdigest()[:10]
        key = "/".join(["tau", str(self.name), geo, ion.species, ",".join(map(str, line_bundle.lambda0_A)),
                        hashlib.md5(v.tobytes()).hexdigest()[:10], override_key,
                        ",".join(f"{k}={w}" for k, w in sorted(opts.items()))])
        build = lambda: dict(tau=self._tau(ion, line_bundle, v, ext, integ, override, check))
        return cached(self.cache, key, build, enabled=self.name is not None)["tau"]

    def _tau(self, ion, line_bundle, v, ext, integ, override, check):
        T_scale = gp.FAKE_SPECTRA_B2_RATIO if ext.b_constants == "fake_spectra" else 1.0
        fr = self.frame(integ.frame, integ.periodic)
        per_cell = integ.chord == "fake_7pt" or integ.tautail or integ.pixel_average
        if not per_cell:
            A = self.absorbers(ion, ext, override)
            return ab_.tau_absorbers(A, line_bundle, v, fr, T_scale=T_scale, n_scale=integ.normalisation,
                                     detuning=integ.detuning, check=check)[0]
        # fake_spectra's per-cell scheme: snapshot frame, its periodic pixel grid, straight skewers
        if integ.frame != "snapshot":
            raise ValueError("fake_spectra's per-cell scheme (fake_7pt / tautail / pixel_average) is defined in its snapshot frame.")
        if integ.chord == "fake_7pt" and ext.deposition != "tophat":
            raise ValueError("chord='fake_7pt' is fake_spectra's top-hat chord quadrature: use deposition='tophat'.")
        dv, nbins = float(v[1] - v[0]), v.size
        if not (np.allclose(v, (np.arange(nbins) + 0.5) * dv) and np.isclose(nbins * dv, self.header.vmax_kms)):
            raise ValueError("fake_spectra's per-cell scheme needs its periodic pixel grid (spectral_grid()).")
        ci = self.cell_inputs(ion, ext, override)
        A = None if integ.chord == "fake_7pt" else self.absorbers(ion, ext, override)
        out = []
        for k in range(self.n_skewers):
            e = self.points[k, -1] - self.points[k, 0]
            e = e / np.linalg.norm(e)
            if integ.chord == "fake_7pt":
                Ak = fse.fake_chord_absorbers(self.points[k, 0], e, self.box, ci["pos"], ci["size"], ci["n"], ci["T"], ci["vel"],
                                              self.box, self.header.a, ci["species"], ci["mass"])
            else:
                Ak = A.with_values(**{f: getattr(A, f)[k:k + 1] for f in ("s", "n", "length", "T", "v_los", "cell", "point")})
            d = ci["pos"] - self.points[k, 0]
            d -= self.box * np.round(d / self.box)
            out.append(fse.fake_emulated_tau(Ak, line_bundle, fr, nbins, dv, T_scale=T_scale, cell_centre_s=np.mod(d @ e, self.box),
                                             cell_v_los=ci["vel"] @ e, cell_T=ci["T"], pixel_average=integ.pixel_average,
                                             tautail=integ.tautail, detuning=integ.detuning, n_scale=integ.normalisation)[0])
        return np.stack(out)


# --------------------------------------------------------------------------- effects and metrics
def run_effects(sk: SimulationSkewers, ion: gp.Ion, line_bundle, velocities_kms, effects: Sequence[Effect],
                ext0: ExtractionOptions, int0: IntegrationOptions, verbose: bool = True) -> dict:
    """tau of every link of the chain of ``effects`` from (ext0, int0) and of every isolated switch.

    Returns dict(links={label: tau}, isolated={label: tau}, base={label: base label},
    options={label: (ext, integ)}).
    """
    t = time.perf_counter()
    links, iso = effect_chain(effects, ext0, int0, ion)
    L = {lab: sk.tau(ion, line_bundle, velocities_kms, x, i) for lab, x, i in links}
    I = {lab: sk.tau(ion, line_bundle, velocities_kms, x, i) for lab, (x, i, _) in iso.items()}
    if verbose:
        print(f"{sk.name}/{ion.species}: {len(L)} links + {len(I)} isolated switches in {time.perf_counter() - t:.0f} s")
    return dict(links=L, isolated=I, base={lab: b for lab, (_, _, b) in iso.items()},
                options={lab: (x, i) for lab, x, i in links})


def spectral_metrics(tau, tau_ref, dv: float) -> dict:
    """Differences of tau against tau_ref (both (N_sk, N_v)): max |dF|, rms dF, d<F>, dEW / EW (summed
    over skewers), median |dtau| / tau for 1e-3 < tau_ref < 5, mean velocity lag [km/s]."""
    tau, tau_ref = np.atleast_2d(tau), np.atleast_2d(tau_ref)
    F, Fr = np.exp(-tau), np.exp(-tau_ref)
    m = (tau_ref > 1e-3) & (tau_ref < 5)
    EW, EWr = ab_.equivalent_width_kms(tau, dv).sum(), ab_.equivalent_width_kms(tau_ref, dv).sum()
    return dict(max_dF=float(np.abs(F - Fr).max()), rms_dF=float(np.sqrt(np.mean((F - Fr) ** 2))), d_meanF=float(F.mean() - Fr.mean()),
                dEW=float((EW - EWr) / max(EWr, 1e-300)),
                med_dtau=float(np.median(np.abs(tau[m] - tau_ref[m]) / tau_ref[m])) if m.any() else 0.0,
                lag=float(np.mean([ab_.velocity_lag_kms(F[k], Fr[k], dv) for k in range(F.shape[0])])))


def p1d(tau, dv: float):
    """P1D of delta_F = F / <F> - 1 over all skewers (``DeltaFieldBuilder`` -> ``OverfluxStatistics1D``):
    (k [s/km], P [km/s]) without the k = 0 mode."""
    from skewerjax.optical_depth.overflux import DeltaFieldBuilder
    from skewerjax.summary_stats.P1D_xi1D import OverfluxStatistics1D
    st = OverfluxStatistics1D(DeltaFieldBuilder(np.asarray(tau)[None]).build(), dv).compute_p1d()
    return np.asarray(st.k)[1:], np.asarray(st.Pk_total)[1:]


def p1d_change(tau, tau_ref, dv: float, bands=((0.0, 0.1), (0.1, 1.0))) -> list:
    """Median |P / P_ref - 1| of the P1D in k bands [s/km]."""
    k, P = p1d(tau, dv)
    _, Pr = p1d(tau_ref, dv)
    return [float(np.median(np.abs(P / Pr - 1)[(k > lo) & (k <= hi)])) for lo, hi in bands]


def effect_rows(result: dict, dv: float, tau_ref=None) -> tuple:
    """One row per effect: the change along the chain and alone (d<F>, dEW / EW, max |dF|, rms dF, P1D
    change at k <= 0.1 and 0.1 < k <= 1 s/km). EW changes are relative to ``tau_ref`` (default: the
    chain's start). Also returns the totals: (end - start), the sum of the isolated changes and of their
    magnitudes, for d<F> and dEW / EW."""
    L, I, base = result["links"], result["isolated"], result["base"]
    labels = list(L)
    ref = L["start"] if tau_ref is None else tau_ref
    EW = lambda t: ab_.equivalent_width_kms(t, dv).sum()
    EWr = EW(ref)
    rows = []
    for prev, lab in zip(labels[:-1], labels[1:]):
        b = L[base[lab]]
        mc, mi = spectral_metrics(L[lab], L[prev], dv), spectral_metrics(I[lab], b, dv)
        rows.append(dict(effect=lab, dF_chain=mc["d_meanF"], dEW_chain=(EW(L[lab]) - EW(L[prev])) / EWr, max_chain=mc["max_dF"],
                         rms_chain=mc["rms_dF"], dF_iso=mi["d_meanF"], dEW_iso=(EW(I[lab]) - EW(b)) / EWr, max_iso=mi["max_dF"],
                         rms_iso=mi["rms_dF"], p1d_iso=p1d_change(I[lab], b, dv)))
    tot = dict(dF=float(np.exp(-L[labels[-1]]).mean() - np.exp(-L["start"]).mean()), dEW=float((EW(L[labels[-1]]) - EW(L["start"])) / EWr),
               dF_sum=float(sum(r["dF_iso"] for r in rows)), dEW_sum=float(sum(r["dEW_iso"] for r in rows)),
               dF_abs=float(sum(abs(r["dF_iso"]) for r in rows)), dEW_abs=float(sum(abs(r["dEW_iso"]) for r in rows)))
    return rows, tot


def print_effect_rows(rows: list, tot: dict, title: str = ""):
    """The table of ``effect_rows``, ranked by the rms dF of the isolated switch."""
    if title:
        print(title)
    print(f"{'effect':22s} {'rms dF':>9s} {'max |dF|':>9s} {'d<F>':>10s} {'dEW/EW':>10s} {'P1D k<0.1':>10s} {'0.1-1':>9s} | "
          f"{'chain d<F>':>10s} {'chain dEW/EW':>12s}")
    for r in sorted(rows, key=lambda r: -r["rms_iso"]):
        print(f"{r['effect']:22s} {r['rms_iso']:9.2e} {r['max_iso']:9.2e} {r['dF_iso']:10.2e} {r['dEW_iso']:10.2e} "
              f"{r['p1d_iso'][0]:10.2e} {r['p1d_iso'][1]:9.2e} | {r['dF_chain']:10.2e} {r['dEW_chain']:12.2e}")
    print(f"{'total (end - start)':22s} d<F> = {tot['dF']:+.4e}, dEW/EW = {tot['dEW']:+.4e} | sum of the isolated switches: "
          f"d<F> = {tot['dF_sum']:+.4e}, dEW/EW = {tot['dEW_sum']:+.4e}")
