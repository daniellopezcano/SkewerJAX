"""Skewers through a simulation snapshot (layers 1-6): fields, columns and tau for any extraction / integration options.

``SimulationSkewers`` holds the sampling points of a set of skewers (a ``SkewerBundle``'s points), the gas cells
selected near them, the deposited segments and the frames. It returns per-cell properties, deposited absorbers,
columns along the skewers, ``core.Fields`` at the points, and tau(v), for

* ``ExtractionOptions``: how the fields are extracted from the cells (deposition geometry, mass and temperature
  constants, ionisation model): layers 3 and 5;
* ``IntegrationOptions``: how the optical-depth integral is evaluated (frame, normalisation, detuning,
  periodicity): layer 6.

The defaults of both are SkewerJAX's (the physically most accurate choices); every other value is an isolated,
named option. fake_spectra's per-cell integration scheme and the option sets that reproduce it are in
``skewerjax.comparison`` (which depends on this module, not the other way round).
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable, Optional

import h5py
import jax.numpy as jnp
import numpy as np

from skewerjax.core.constants import phys
from skewerjax.core.cosmology import Cosmology
from skewerjax.core.fields import Fields, SampledField, ScalarField, VectorField
from skewerjax.core.lines import LineBundle
from skewerjax.core.skewers import axis_aligned_skewer_points, path_geometry
from skewerjax.simulations import absorbers as ab_
from skewerjax.simulations import deposition as dep
from skewerjax.simulations import gas_physics as gp
from skewerjax.simulations import selection as sel


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

    frame : "exact" (a(x) along the path from the cosmology) or "snapshot" (one a, linear Hubble flow; matched).
    normalisation : multiplies every density (matched mode: 8/7 or 7/8 of fake_spectra's chords).
    detuning : "exact" (nu = (c / lambda) expm1(-dv / c)) or "linear" (fake_spectra).
    periodic : True wraps every detuning to its nearest periodic image (period: the box length in velocity);
        False is the direct integral.
    """
    frame: str = "exact"
    normalisation: float = 1.0
    detuning: str = "exact"
    periodic: bool = True

    def replace(self, **kw) -> "IntegrationOptions":
        return replace(self, **kw)


def relevant_options(ext: ExtractionOptions, ion: gp.Ion) -> ExtractionOptions:
    """``ext`` with the options that do not act on ``ion`` set to their defaults (x_HI_model for
    metals, the CLOUDY options for H I): equal results compare equal."""
    d = ExtractionOptions()
    if ion.is_HI:
        return ext.replace(cloudy_density=d.cloudy_density, cloudy_interpolation=d.cloudy_interpolation)
    return ext.replace(x_HI_model=d.x_HI_model)


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
            pv = self.point_values(gp.ion_from_species(sp), ext)
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
        if velocities_kms is None:
            velocities_kms = self.spectral_grid(1.0, self.frame(integ.frame, integ.periodic))[0]
        v = np.asarray(velocities_kms, dtype=np.float64)
        taus = [self.tau(gp.ion_from_species(line.species), LineBundle([line.id], ref_line_id=line.id), v, ext, integ, check=check)
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

    def tau_key(self, ion: gp.Ion, line_bundle, velocities_kms, ext: ExtractionOptions, integ, override_key: str = "") -> str:
        """Cache key of a tau result: skewers, ion, lines, velocity grid, override name and every option."""
        v = np.asarray(velocities_kms, dtype=np.float64)
        opts = {**asdict(relevant_options(ext, ion)), **asdict(integ)}
        geo = hashlib.md5(self.points.tobytes() + (b"" if self.scale_factors is None else self.scale_factors.tobytes())).hexdigest()[:10]
        return "/".join(["tau", str(self.name), geo, ion.species, ",".join(map(str, line_bundle.lambda0_A)),
                         hashlib.md5(v.tobytes()).hexdigest()[:10], override_key,
                         ",".join(f"{k}={w}" for k, w in sorted(opts.items()))])

    def tau(self, ion: gp.Ion, line_bundle, velocities_kms, ext: ExtractionOptions = ExtractionOptions(),
            integ: IntegrationOptions = IntegrationOptions(), override: Optional[dict] = None, override_key: str = "",
            check: Optional[float] = 0.25) -> np.ndarray:
        """(N_sk, N_v) optical depth of the first line of ``line_bundle`` (one species) for ``ext`` and ``integ``;
        cached when the object has a ``name`` and a ``cache`` (``override_key`` names an override)."""
        if override is not None and not override_key:
            override_key = "override"
        v = np.asarray(velocities_kms, dtype=np.float64)
        build = lambda: dict(tau=self.compute_tau(ion, line_bundle, v, ext, integ, override, check))
        return cached(self.cache, self.tau_key(ion, line_bundle, v, ext, integ, override_key), build, enabled=self.name is not None)["tau"]

    def compute_tau(self, ion: gp.Ion, line_bundle, velocities_kms, ext: ExtractionOptions = ExtractionOptions(),
                    integ: IntegrationOptions = IntegrationOptions(), override: Optional[dict] = None,
                    check: Optional[float] = 0.25) -> np.ndarray:
        """``tau`` without the cache: the deposited absorbers through ``absorbers.tau_absorbers``."""
        T_scale = gp.FAKE_SPECTRA_B2_RATIO if ext.b_constants == "fake_spectra" else 1.0
        fr = self.frame(integ.frame, integ.periodic)
        A = self.absorbers(ion, ext, override)
        return ab_.tau_absorbers(A, line_bundle, np.asarray(velocities_kms, dtype=np.float64), fr, T_scale=T_scale,
                                 n_scale=integ.normalisation, detuning=integ.detuning, check=check)[0]
