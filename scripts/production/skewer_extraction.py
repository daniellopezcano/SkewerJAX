"""Filling the skewer store: the options of a dataset, written out in full, and the resumable computation of the
skewers of nested lattices missing from it (``skewer_store``; ``extract_skewers.py``, ``run_production.py``;
``STORE.md``). Production code, outside the package: it composes the public API of ``skewerjax.core``,
``skewerjax.simulations`` and ``skewerjax.config`` only.

**Options** (``dataset_options``): a preset (``MODES``: "default", "matched") with any option overridden, resolved to
values (no None standing for "the package's default", no "snapshot" standing for a number), so that the hash of the
dataset names what was computed whatever the package's defaults become. The presets:

* "default": notebook 11's tau of a snapshot (the snapshot frame, linear Hubble flow and detuning, periodic over one
  box, every line on the reference line's axis) with the package's pieces and integration: the exact Voronoi cell
  crossings, the integrated profile, Harris' Voigt (as notebook 11; ``voigt="harris"`` an option), CODATA
  constants, SkewerJAX's gas physics;
* "matched": fake_spectra's scheme as in ``comparison.scaling.mode_options`` (its gas physics, the top-hat chords of
  weight 1/8, the midpoint profile, ``pixel_average``, ``tautail`` 1e-7, a_eff and (aH)_ref of the snapshot, its
  constants), on the same frame and axes as "default".

**The sampling points** (``points_kms``, the spacing of points along a skewer in km/s of Hubble flow). The default
pieces, the exact cell crossings with the integrated profile, have no sampling points: in the snapshot frame nothing
depends on a spacing, and in the exact frame (``frame="exact"``) only the table of a(s) does (``stream.box_skewers``'s
``frame_spacing``, default 0.05 cMpc; ``options["tau"]["frame_spacing_cMpc"]`` = points_kms / u when given). With
``pieces="segments"`` the cells are deposited into the dual segments of points every ``points_kms`` (``cell_pieces``
with ``deposition="voronoi_segments"``, the gather route ``select_cells_near_paths``; ``compute_skewers_sampled``),
usually with ``profile="midpoint"``: then the spacing sets the pieces and the accuracy (benchmark block C2,
``benchmarks/results/scaling/C2.json``). Both keys enter the options (and the hash) only when they apply, so the
datasets of the other configurations keep their hashes.

**Computation** (``compute_skewers``): the streaming route of ``simulations.stream`` on the requested skewers only: one
pass over the snapshot (the cells within reach, the gas physics on those), then the tiles, each tile written to the
store as one batch as soon as its tau is computed (``tiles_optical_depth(on_tile=)``). ``extract`` runs the lattices
of ``n_list`` in increasing order, each time on the skewers not stored yet (the coarser lattices are subsets: doubling N
reuses them all), in passes small enough for the memory (``pass_size``), and records every run in ``runs/``. An
interrupted run loses the tiles in progress only; the same command resumes.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import subprocess
import time
import uuid
from pathlib import Path

import numpy as np
import skewer_store as ss  # the sibling module of scripts/production (the directory is on sys.path: run_production.py, conftest)

from skewerjax.core import VORONOI_FACTOR

# The gas physics of the presets (``gas_to_cells``): SkewerJAX's and fake_spectra's. The same values as
# ``comparison.scaling.DEFAULT_CELLS`` and ``comparison.benchmark.MATCHED_CELLS`` (kept here as they were when this module
# was in ``simulations``, which does not import comparison; tests/production/test_skewer_store.py checks that they agree).
DEFAULT_CELLS = dict(
    mass_unit="amu", T_constants="codata", x_HI_model="stored_sf", cloudy_density="hydrogen", cloudy_interpolation="linear"
)
MATCHED_CELLS = dict(
    mass_unit="proton",
    T_constants="fake_spectra",
    x_HI_model="fake_sf",
    cloudy_density="fake_spectra",
    cloudy_interpolation="cubic",
)
FRAMES = ("snapshot", "exact")
PIECES = ("crossings", "chords", "segments")  # segments: the dual segments of sampling points (``compute_skewers_sampled``)

MODES = {
    "default": dict(
        physics=DEFAULT_CELLS,
        pieces="crossings",
        chord_weight=None,
        frame="snapshot",
        line_frame="reference",
        profile="integrated",
        voigt="harris",
        detuning="linear",
        hubble_flow="linear",
        aH_ref=None,
        a_eff=None,
        b_constants="codata",
        pixel_average=False,
        tautail=None,
        wing_tolerance=None,
        velocity_period=True,
    ),
    "matched": dict(
        physics=MATCHED_CELLS,
        pieces="chords",
        chord_weight=1.0 / 8.0,
        frame="snapshot",
        line_frame="reference",
        profile="midpoint",
        voigt="harris",
        detuning="linear",
        hubble_flow="linear",
        aH_ref="snapshot",
        a_eff="snapshot",
        b_constants="fake_spectra",
        pixel_average=True,
        tautail=1e-7,
        wing_tolerance=None,
        velocity_period=True,
    ),
}
LINE_IDS = ("HI-LyA", "SiIII-1206")  # notebook 11's lines
RESOLUTION_TOLERANCE = 0.25  # the resolution check of compute_optical_depth, recorded per batch (notebook 11's value)
# Memory model of one pass (notebook 16; ``docs/performance.md`` section 4): 5.7 GB + 92 B per cell within reach, the
# cells within reach of n skewers at most min(n_gas, n x 2.5^2 pi n_gas^(1/3)) (Hölder: sum h^2 <= n^(1/3) L^2 for cells
# of sizes h filling the box; equality for equal cells), plus the tau of the pass.
PASS_BASE_BYTES = 5.7e9
PASS_BYTES_PER_CELL = 92.0
MEMORY_FRACTION = 0.6  # of MemAvailable for one pass
# Memory model of one pass of the sampled route (``compute_skewers_sampled``: the gather selection, the gas physics of the
# selected cells held for the pass, the points, one tile's pieces), measured on TNG50-2 (anonymous memory; stem 20261121_1641)
# (one tile of 1,024 skewers at 0.01 cMpc: 8.6 GB anonymous for 18.8 M cells and 5.3 M points; the memory maps of the
# snapshot are file-backed, reclaimable, and not counted). The campaign's passes at 1 cMpc took ~500 B per selected cell
# (MemAvailable fell by ~73 GB for a pass of 1,806 skewers and 143 M cells: the candidate pairs of the long dual
# segments), so the model uses that bound for every spacing.
SAMPLED_BASE_BYTES = 4.0e9
SAMPLED_BYTES_PER_CELL = 500.0
SAMPLED_BYTES_PER_POINT = 50.0
VORONOI_REACH_AREA = np.pi * VORONOI_FACTOR**2  # the area within reach of a skewer, in units of h^2 (the reach 2.5 h)


# --------------------------------------------------------------------------- options
def parse_override(text: str) -> tuple:
    """Parse "key=value" -> (key, value), the value read as JSON (numbers, true/false, null, lists), else as a string; dotted
    keys reach into the gas physics ("physics.x_HI_model=stored")."""
    if "=" not in text:
        raise ValueError(f"an override is key=value, got {text!r}.")
    k, v = text.split("=", 1)
    try:
        return k.strip(), json.loads(v)
    except json.JSONDecodeError:
        return k.strip(), v.strip()


def mode_options(mode: str, overrides: dict | None = None) -> dict:
    """The options of a preset (``MODES``) with ``overrides`` ({key: value}; "physics.<key>" for one gas-physics
    option); an unknown key is a ValueError."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {tuple(MODES)}, got {mode!r}.")
    o = json.loads(json.dumps(MODES[mode]))  # a deep copy
    for k, v in (overrides or {}).items():
        if k.startswith("physics."):
            sub = k.split(".", 1)[1]
            if sub not in o["physics"]:
                raise ValueError(f"unknown gas-physics option {sub!r} (one of {sorted(o['physics'])}).")
            o["physics"][sub] = v
        elif k == "physics" and isinstance(v, dict):
            unknown = set(v) - set(o["physics"])
            if unknown:
                raise ValueError(f"unknown gas-physics options {sorted(unknown)}.")
            o["physics"].update(v)
        elif k in o:
            o[k] = v
        else:
            raise ValueError(f"unknown option {k!r} (one of {sorted(o)}).")
    if o["frame"] not in FRAMES:
        raise ValueError(f"frame must be one of {FRAMES}, got {o['frame']!r}.")
    if o["pieces"] not in PIECES:
        raise ValueError(f"pieces must be one of {PIECES}, got {o['pieces']!r}.")
    return o


def _frame_span(header, frame: str, hubble_flow: str, aH_ref) -> float:
    """The velocity span of one box along a skewer [km/s]: the period of ``velocity_period=True`` and the length of the
    pixel axis (snapshot frame: (aH)_snap L; the exact frame: ``core.absorbers.velocity_span`` of the box-mode frame)."""
    from skewerjax.core import SnapshotFrame, velocity_span
    from skewerjax.simulations import box_skewers

    L = header.box_cMpc
    if frame == "snapshot":
        fr = SnapshotFrame(header.a, header.velfac_kms_per_cMpc, path_length=np.full(1, L))
    else:
        fr = box_skewers(np.zeros((1, 3)), 1, L, header.cosmology(), header.a)[1]
    return float(np.asarray(velocity_span(fr, hubble_flow, aH_ref), dtype=np.float64).ravel()[0])


def dataset_options(
    header,
    *,
    mode: str = "default",
    overrides: dict | None = None,
    spacing_kms: float | None = 1.0,
    n_pix: int | None = None,
    line_ids=LINE_IDS,
    precision: str = "float32",
    n_ref: int = ss.N_REF,
    points_kms: float | None = None,
    points_cMpc: float | None = None,
) -> dict:
    """The options of a dataset (``config["options"]``, hashed), resolved to values from the snapshot's ``header``:

    lattice (n_ref, the offset 0, the order); pixels (``spacing_kms`` the target width: n_pix = int(span / spacing),
    dv = span / n_pix, as fake_spectra's grid; or ``n_pix`` itself; the centres (i + 1/2) dv, the period = n_pix dv =
    the span of one box); lines (the ids, the first the reference); physics (``gas_to_cells``); pieces; tau (every
    option of ``tiles_optical_depth`` / ``compute_optical_depth``: velocity_period True -> the span, a_eff and aH_ref
    "snapshot" -> the snapshot's a and aH, wing_tolerance None -> ``core.WING_TOLERANCE``); precision. ``points_cMpc``
    or ``points_kms`` (one of them): the spacing of the sampling points [cMpc], or [km/s] of Hubble flow (module
    docstring; ds = points_kms / u, u = aH the snapshot's km/s per cMpc): required with ``pieces="segments"``
    (``pieces["points_kms"]``, and ``pieces["points_cMpc"]`` when given in cMpc), the a(s) table in the exact frame
    (``tau["frame_spacing_cMpc"]``); elsewhere it has no effect and is not part of the options."""
    from skewerjax.core import WING_TOLERANCE

    o = mode_options(mode, overrides)
    if precision not in ("float32", "float64"):
        raise ValueError(f"precision must be 'float32' or 'float64', got {precision!r}.")
    ss.check_n(1, n_ref)
    a_snap, aH = float(header.a), float(header.velfac_kms_per_cMpc)
    resolve = lambda x, val: val if x == "snapshot" else (None if x is None else float(x))
    aH_ref = resolve(o["aH_ref"], aH)
    a_eff = resolve(o["a_eff"], a_snap)
    span = _frame_span(header, o["frame"], o["hubble_flow"], aH_ref)
    if (spacing_kms is None) == (n_pix is None):
        raise ValueError("give the pixel grid by spacing_kms or by n_pix (one of them).")
    if n_pix is None:
        if not spacing_kms > 0:
            raise ValueError(f"spacing_kms must be > 0, got {spacing_kms}.")
        n_pix = int(span / float(spacing_kms))
    n_pix = int(n_pix)
    if n_pix < 2:
        raise ValueError(f"{n_pix} pixels over the span {span:.1f} km/s: too few.")
    dv = span / n_pix
    vp = o["velocity_period"]
    if vp is True:
        vp = span
    elif vp in (False, None):
        vp = False
    else:
        vp = float(vp)
    ids = [str(x) for x in line_ids]
    tau = {
        k: o[k] for k in ("frame", "line_frame", "profile", "voigt", "detuning", "hubble_flow", "b_constants", "pixel_average")
    }
    tau.update(
        aH_ref=aH_ref,
        a_eff=a_eff,
        tautail=None if o["tautail"] is None else float(o["tautail"]),
        wing_tolerance=float(WING_TOLERANCE if o["wing_tolerance"] is None else o["wing_tolerance"]),
        velocity_period=vp,
    )
    pieces = dict(pieces=o["pieces"], chord_weight=None if o["chord_weight"] is None else float(o["chord_weight"]))
    if points_kms is not None and points_cMpc is not None:
        raise ValueError("give the sampling spacing by points_cMpc or by points_kms, not both.")
    for name, x in (("points_kms", points_kms), ("points_cMpc", points_cMpc)):
        if x is not None and not float(x) > 0:
            raise ValueError(f"{name} must be > 0, got {x}.")
    ds_cMpc = float(points_cMpc) if points_cMpc is not None else (None if points_kms is None else float(points_kms) / aH)
    if o["pieces"] == "segments":
        if ds_cMpc is None:
            raise ValueError("pieces='segments' needs points_cMpc or points_kms (the spacing of the sampling points).")
        pieces["points_kms"] = float(points_kms) if points_kms is not None else ds_cMpc * aH
        if points_cMpc is not None:
            pieces["points_cMpc"] = ds_cMpc  # the spacing as given: it sets the points (points_kms = ds u, recorded)
    if o["frame"] == "exact" and ds_cMpc is not None:
        tau["frame_spacing_cMpc"] = ds_cMpc
    return dict(
        lattice=dict(n_ref=int(n_ref), offset=0.0, order="a-major (i_a outer), (a, b) the transverse axes in increasing order"),
        pixels=dict(
            spacing_kms=None if spacing_kms is None else float(spacing_kms),
            n_pix=n_pix,
            dv_kms=dv,
            period_kms=n_pix * dv,
            centres="(i + 1/2) dv, i = 0 .. n_pix - 1, on the reference line's axis",
        ),
        lines=dict(line_ids=ids, ref_line_id=ids[0]),
        physics=dict(o["physics"]),
        pieces=pieces,
        tau=tau,
        precision=precision,
    )


def dataset_config(reader, sim: str, snap: int, axis: int, options: dict, *, preset: str | None = None) -> dict:
    """config.json of a dataset: its location ("simulation": name, path, snapshot, axis), ``options`` (hashed) and the
    snapshot it reads (not hashed: the location names it)."""
    if int(axis) not in ss.TRANSVERSE:
        raise ValueError(f"axis must be 1, 2 or 3 (x, y, z), got {axis!r}.")
    H = reader.header
    path = Path(reader.files[0]).resolve().parent.parent
    return dict(
        layout_version=ss.LAYOUT_VERSION,
        simulation=dict(name=str(sim), path=str(path), snapshot=int(snap), axis=int(axis)),
        preset=preset,
        options=options,
        snapshot=dict(
            redshift=float(H.redshift),
            a=float(H.a),
            h=float(H.h),
            omega_m=float(H.omega_m),
            box_cMpc=float(H.box_cMpc),
            u_kms_per_cMpc=float(H.velfac_kms_per_cMpc),
            n_gas=int(H.n_gas),
            n_files=int(H.n_files),
        ),
    )


# --------------------------------------------------------------------------- the computation
def compute_skewers(
    reader,
    config: dict,
    indices,
    *,
    tile_size: int = 1024,
    devices=None,
    threads: int | None = None,
    on_batch=None,
    on_pass=None,
    resolution_tolerance: float | None = RESOLUTION_TOLERANCE,
) -> tuple:
    """tau of the skewers ``indices`` (m, 2) (on N_ref) of the dataset ``config`` by the streaming route: one pass over
    the snapshot, then the tiles of about ``tile_size`` skewers; ``on_pass(info)`` when the pass is over (its timings
    and memory), ``on_batch(indices_of_tile, tau (n_lines, n, N_pix), row)`` after each tile (``tiles_optical_depth``'s
    row: timings, resolution). Returns (tau (n_lines, m, N_pix), info with the pass's and the tiles' timings)."""
    from skewerjax import simulations as st
    from skewerjax.config import get_precision
    from skewerjax.core import LineBundle, SnapshotFrame

    o, s = config["options"], config["simulation"]
    if get_precision() != o["precision"]:
        raise RuntimeError(
            f"the dataset is {o['precision']}, the package runs in {get_precision()} (skewerjax.config.set_precision)."
        )
    H = reader.header
    L = float(H.box_cMpc)
    idx = np.asarray(indices, dtype=np.int64).reshape(-1, 2)
    lines = LineBundle(list(o["lines"]["line_ids"]), ref_line_id=o["lines"]["ref_line_id"])
    t0 = time.perf_counter()
    cofm = ss.lattice_positions(idx, s["axis"], L, o["lattice"]["n_ref"])
    t = o["tau"]
    fs = {"frame_spacing": t["frame_spacing_cMpc"]} if t.get("frame_spacing_cMpc") is not None else {}
    paths, exact = st.box_skewers(cofm, s["axis"], L, H.cosmology(), H.a, **fs)
    frame = SnapshotFrame(H.a, H.velfac_kms_per_cMpc, path_length=np.full(len(paths), L)) if t["frame"] == "snapshot" else exact
    tile_of = st.tile_skewers(paths, tile_size)
    store, pinfo = st.stream_cells(reader, paths, tile_of, list(lines.species_keys), {"run": o["physics"]}, threads=threads)
    t1 = time.perf_counter()
    info = dict(
        n_skewers=int(idx.shape[0]),
        n_tiles=int(tile_of.max()) + 1,
        n_cells_within_reach=int(pinfo["n_needed"]),
        store_GB=float(pinfo["store_GB"]),
        pass_s=float(t1 - t0),
        pass_stages={k: float(v) for k, v in pinfo["timings"].items()},
    )
    if on_pass is not None:
        on_pass(dict(info))
    p = o["pixels"]
    v = (np.arange(p["n_pix"]) + 0.5) * p["dv_kms"]

    def tile_done(k, sk, taus, row):
        if on_batch is not None:
            on_batch(idx[sk], taus["tau"], row)

    taus, tinfo = st.tiles_optical_depth(
        store,
        "run",
        paths,
        tile_of,
        frame,
        v,
        lines,
        pieces=o["pieces"]["pieces"],
        chord_weight=o["pieces"]["chord_weight"],
        profile=t["profile"],
        detuning=t["detuning"],
        voigt=t["voigt"],
        wing_tolerance=t["wing_tolerance"],
        a_eff=t["a_eff"],
        b_constants=t["b_constants"],
        hubble_flow=t["hubble_flow"],
        aH_ref=t["aH_ref"],
        velocity_period=t["velocity_period"] or None,
        tautail=t["tautail"],
        pixel_average=t["pixel_average"],
        line_frame=t["line_frame"],
        resolution_tolerance=resolution_tolerance,
        engines={"tau": list(devices) if devices else [None]},
        on_tile=tile_done,
    )
    info["tiles_s"] = float(time.perf_counter() - t1)
    del store, tinfo
    return taus["tau"], info


def compute_skewers_sampled(
    reader,
    config: dict,
    indices,
    *,
    tile_size: int = 1024,
    devices=None,
    threads: int | None = None,
    on_batch=None,
    on_pass=None,
    resolution_tolerance: float | None = RESOLUTION_TOLERANCE,
) -> tuple:
    """tau of the skewers ``indices`` (m, 2) of a ``pieces="segments"`` dataset (module docstring), with the interface of
    ``compute_skewers``: the points of every skewer every ``points_kms`` (``SkewerBundle.axis_aligned``), the cells within
    reach of them (``select_cells_near_paths``, one read of the snapshot) and their gas (``gas_to_cells``), then per tile
    of ``tile_size`` skewers the dual-segment pieces (``cell_pieces``, ``deposition="voronoi_segments"``, in the
    dataset's frame) and tau (``compute_optical_depth`` with every option of the dataset). The route of benchmark block
    C2 (``comparison.scaling.run_c2``) with the options of the store."""
    from skewerjax.config import get_precision
    from skewerjax.core import LineBundle, SkewerBundle, SnapshotFrame, cell_pieces, compute_optical_depth
    from skewerjax.simulations import gas_to_cells, select_cells_near_paths

    o, s = config["options"], config["simulation"]
    if get_precision() != o["precision"]:
        raise RuntimeError(
            f"the dataset is {o['precision']}, the package runs in {get_precision()} (skewerjax.config.set_precision)."
        )
    if o["pieces"]["pieces"] != "segments":
        raise ValueError("compute_skewers_sampled computes pieces='segments' datasets only (compute_skewers the others).")
    H = reader.header
    L, aH = float(H.box_cMpc), float(H.velfac_kms_per_cMpc)
    idx = np.asarray(indices, dtype=np.int64).reshape(-1, 2)
    lines = LineBundle(list(o["lines"]["line_ids"]), ref_line_id=o["lines"]["ref_line_id"])
    t = o["tau"]
    ds_cMpc = sampling_cMpc(o, aH)
    t0 = time.perf_counter()
    cofm = ss.lattice_positions(idx, s["axis"], L, o["lattice"]["n_ref"])
    bundle = lambda c: SkewerBundle.axis_aligned(c, s["axis"], L, ds_cMpc, H.cosmology(), a_ref=H.a)  # noqa: E731
    points = bundle(cofm).host["points"]
    gas = select_cells_near_paths(reader, points, mode="voronoi", workers=threads or -1)
    cells = gas_to_cells(gas, list(lines.species_keys), **o["physics"])
    n_cells = int(np.asarray(cells.positions).shape[0]) if hasattr(cells, "positions") else -1
    del gas
    t1 = time.perf_counter()
    n_tiles = -(-len(idx) // int(tile_size))
    info = dict(
        n_skewers=int(idx.shape[0]),
        n_tiles=int(n_tiles),
        n_cells_within_reach=n_cells,
        store_GB=0.0,
        pass_s=float(t1 - t0),
        pass_stages={"select_and_physics": float(t1 - t0)},
        points_per_skewer=int(points.shape[1]),
    )
    if on_pass is not None:
        on_pass(dict(info))
    p = o["pixels"]
    v = (np.arange(p["n_pix"]) + 0.5) * p["dv_kms"]
    out = []
    for k in range(n_tiles):
        sk = slice(k * int(tile_size), min((k + 1) * int(tile_size), len(idx)))
        tg = time.perf_counter()
        if t["frame"] == "snapshot":
            m = sk.stop - sk.start
            frame = SnapshotFrame(H.a, aH, path_length=np.full(m, L))
            pieces = cell_pieces(cells, lines, points=points[sk], frame=frame, deposition="voronoi_segments", a_data=H.a)
        else:
            pieces = cell_pieces(cells, lines, skewer_bundle=bundle(cofm[sk]), deposition="voronoi_segments", a_data=H.a)
        tp = time.perf_counter()
        tau, tinfo = compute_optical_depth(
            pieces,
            v,
            lines,
            line_frame=t["line_frame"],
            velocity_period=t["velocity_period"] or False,
            voigt=t["voigt"],
            detuning=t["detuning"],
            profile_integration=t["profile"],
            wing_tolerance=t["wing_tolerance"],
            resolution_tolerance=resolution_tolerance,
            devices=list(devices) if devices else None,
            a_eff=t["a_eff"],
            hubble_flow=t["hubble_flow"],
            aH_ref=t["aH_ref"],
            b_constants=t["b_constants"],
            pixel_average=t["pixel_average"],
            tautail=t["tautail"],
            chord_weight=o["pieces"]["chord_weight"],
            return_info=True,
        )
        tau = np.asarray(tau)
        row = dict(
            resolution=tinfo.get("resolution"),
            geometry_s=float(tp - tg),
            contributions_s=0.0,
            tau_s={"tau": float(time.perf_counter() - tp)},
        )
        if on_batch is not None:
            on_batch(idx[sk], tau, row)
        out.append(tau)
    info["tiles_s"] = float(time.perf_counter() - t1)
    return np.concatenate(out, axis=1), info


def sampling_cMpc(options: dict, aH: float) -> float | None:
    """The spacing of the sampling points of a dataset's ``options`` [cMpc]: ``pieces["points_cMpc"]`` when it was given in
    cMpc, else ``pieces["points_kms"]`` / aH; None for pieces without sampling points."""
    p = options["pieces"]
    if p.get("points_cMpc") is not None:
        return float(p["points_cMpc"])
    return None if p.get("points_kms") is None else float(p["points_kms"]) / float(aH)


def compute_for(config: dict):
    """The function that computes a pass of the dataset ``config``: ``compute_skewers_sampled`` for
    ``pieces="segments"``, ``compute_skewers`` (the streaming route) otherwise."""
    return compute_skewers_sampled if config["options"]["pieces"]["pieces"] == "segments" else compute_skewers


# --------------------------------------------------------------------------- planning
def memory_available() -> float:
    """MemAvailable [bytes] (/proc/meminfo; the total RAM elsewhere, or 8 GB)."""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return float(line.split()[1]) * 1024
    except OSError:
        pass
    try:
        return float(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (ValueError, OSError, AttributeError):
        return 8e9


def sampled_cells(n_skewers: int, n_gas: int, points_per_skewer: int, n_lattice: int | None = None) -> float:
    """The cells the gather route selects for ``n_skewers`` skewers sampled every L / ``points_per_skewer`` (an upper
    bound, in units of the mean cell size h = L / n_gas^(1/3)): every cell within r = 2.5 h + ds / 2 of a point
    (``select_cells_near_paths``), at most the sum of the skewers' cross sections n pi r^2 (no overlap) and at most the
    square patch of the neighbouring skewers of one pass (side sqrt(n) d + 2 r, d = L / ``n_lattice`` the lattice spacing:
    the passes group neighbours, ``stream.tile_skewers``), as fractions of the box's cross section L^2."""
    g = float(n_gas) ** (1 / 3)  # L / h
    r = VORONOI_FACTOR / g + 0.5 / float(points_per_skewer)  # r / L
    area = n_skewers * np.pi * r * r
    if n_lattice:
        area = min(area, (np.sqrt(n_skewers) / float(n_lattice) + 2 * r) ** 2)
    return float(n_gas) * min(1.0, area)


def pass_memory(
    n_skewers: int,
    n_gas: int,
    n_lines: int,
    n_pix: int,
    precision: str = "float32",
    *,
    points_per_skewer: int | None = None,
    n_lattice: int | None = None,
) -> float:
    """The peak memory of one pass over ``n_skewers`` skewers [bytes]: the streaming route by the model of
    ``PASS_BASE_BYTES``; with ``points_per_skewer`` the sampled route (``compute_skewers_sampled``) by the model of
    ``SAMPLED_BASE_BYTES`` (``sampled_cells``, the points, one tile's pieces)."""
    tau = n_skewers * n_lines * n_pix * (8 if precision == "float64" else 4)
    if points_per_skewer:
        cells = sampled_cells(n_skewers, n_gas, points_per_skewer, n_lattice)
        points = float(n_skewers) * points_per_skewer
        return SAMPLED_BASE_BYTES + SAMPLED_BYTES_PER_CELL * cells + SAMPLED_BYTES_PER_POINT * points + tau
    cells = min(float(n_gas), n_skewers * VORONOI_REACH_AREA * float(n_gas) ** (1 / 3))
    return PASS_BASE_BYTES + PASS_BYTES_PER_CELL * cells + tau


def pass_size(
    n_missing: int,
    n_gas: int,
    n_lines: int,
    n_pix: int,
    precision: str = "float32",
    budget: float | None = None,
    *,
    points_per_skewer: int | None = None,
    n_lattice: int | None = None,
) -> int:
    """The largest number of skewers per pass whose ``pass_memory`` fits ``budget`` (default ``MEMORY_FRACTION`` of
    MemAvailable), at most ``n_missing`` and at least 1."""
    budget = MEMORY_FRACTION * memory_available() if budget is None else float(budget)
    mem = lambda m: pass_memory(m, n_gas, n_lines, n_pix, precision, points_per_skewer=points_per_skewer, n_lattice=n_lattice)  # noqa: E731
    lo, hi = 1, max(int(n_missing), 1)
    if mem(hi) <= budget:
        return hi
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if mem(mid) <= budget:
            lo = mid
        else:
            hi = mid - 1
    return lo


def bounded_tiles(paths, per: int) -> np.ndarray:
    """(N_sk,) the pass of every skewer: the squares of ``stream.tile_skewers`` (neighbours together), a square holding
    more than ``per`` skewers cut into consecutive chunks of at most ``per`` (strips of the square, the skewers being in
    lattice order), so that no pass exceeds ``per``. ``tile_skewers`` sizes its grid for skewers spread evenly; the
    skewers left by an interrupted run are not, and its squares then held up to ~2 x ``per`` (2,688 of a planned 1,626
    at 1 cMpc: a pass twice the memory budget)."""
    from skewerjax import simulations as st

    per = max(int(per), 1)
    group = np.asarray(st.tile_skewers(paths, per), dtype=np.int64)
    out, nxt = np.empty(len(group), np.int64), 0
    for g in range(int(group.max()) + 1 if len(group) else 0):
        idx = np.flatnonzero(group == g)
        for c in range(0, len(idx), per):
            out[idx[c : c + per]], nxt = nxt, nxt + 1
    return out.astype(np.int32)


def points_per_skewer(config: dict) -> int | None:
    """The sampling points per skewer of a ``pieces="segments"`` dataset (L / ds), None for the other pieces."""
    ds = sampling_cMpc(config["options"], config["snapshot"]["u_kms_per_cMpc"])
    if config["options"]["pieces"]["pieces"] != "segments" or not ds:
        return None
    return int(round(float(config["snapshot"]["box_cMpc"]) / ds))


def plan(root, config: dict, n_list) -> dict:
    """What a run of ``n_list`` would do on the dataset of ``config`` (nothing written): per N (increasing) the
    skewers, those stored, those new (not stored and not in a smaller N of the list), and the path and hash."""
    chash = ss.config_hash(config["options"])
    s = config["simulation"]
    path = ss.dataset_path(root, s["name"], s["snapshot"], s["axis"], chash)
    n_ref = config["options"]["lattice"]["n_ref"]
    ns = sorted({ss.check_n(int(n), n_ref) for n in n_list})
    ds = ss.SkewerDataset(path, dict(config, hash=chash)) if (path / "config.json").exists() else None
    rows, seen = [], set()
    for n in ns:
        ind = ss.lattice_indices(n, n_ref)
        keys = ss._keys(ind, n_ref)
        stored = ds.stored(ind) if ds is not None else np.zeros(len(ind), bool)
        new = [k for k, st in zip(keys.tolist(), stored) if not st and k not in seen]
        seen.update(new)
        rows.append(dict(n=n, skewers=n * n, stored=int(stored.sum()), new=len(new)))
    return dict(path=str(path), hash=chash, exists=ds is not None, rows=rows, new_total=len(seen))


# --------------------------------------------------------------------------- the run
def git_commit(repo=None) -> dict:
    """The commit of the package's repository (``git rev-parse HEAD``, read only) and whether its tree has changes."""
    repo = Path(__file__).resolve().parents[2] if repo is None else Path(repo)
    try:
        sha = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, timeout=10
        ).stdout
        return dict(commit=sha or None, dirty=bool(dirty.strip()))
    except (OSError, subprocess.SubprocessError):
        return dict(commit=None, dirty=None)


def versions() -> dict:
    """The versions of skewerjax, jax, jaxlib, numpy, h5py and fake_spectra (None if absent)."""
    from importlib import metadata

    out = {}
    import skewerjax

    out["skewerjax"] = skewerjax.__version__
    for name in ("jax", "jaxlib", "numpy", "h5py", "numba", "fake_spectra"):
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def _fmt_s(x: float) -> str:
    x = float(x)
    return f"{x:.0f} s" if x < 120 else (f"{x / 60:.1f} min" if x < 7200 else f"{x / 3600:.1f} h")


def extract(
    dataset: ss.SkewerDataset,
    reader,
    n_list,
    *,
    tile_size: int = 1024,
    devices=None,
    device_name: str = "cpu",
    threads: int | None = None,
    skewers_per_pass: int | None = None,
    memory_budget: float | None = None,
    log=print,
    argv=None,
    compute=None,
) -> dict:
    """Fill the lattices ``n_list`` of ``dataset`` (module docstring) and return the run's record (also written to
    ``runs/``). ``compute``: the function computing a pass (default ``compute_for(dataset.config)``: ``compute_skewers``,
    or ``compute_skewers_sampled`` for ``pieces="segments"``; the tests count its calls).

    For each N, increasing: the skewers not stored are split into passes of at most ``skewers_per_pass`` (default
    ``pass_size``: the memory model within ``memory_budget``), neighbouring skewers together (squares of the plane,
    ``stream.tile_skewers``); every tile of a pass is written as one batch as soon as it is computed. The log gives the
    progress and the ETA after every batch."""
    from skewerjax.config import get_precision
    from skewerjax.core import StraightPaths

    compute = compute_for(dataset.config) if compute is None else compute
    cfg = dataset.config
    o = cfg["options"]
    n_ref = dataset.n_ref
    ns = sorted({ss.check_n(int(n), n_ref) for n in n_list})
    run_id = _dt.datetime.now().strftime("%Y%m%dT%H%M%S") + f"_{os.getpid()}_{uuid.uuid4().hex[:6]}"  # unique: names the batches
    gitinfo = git_commit()
    rec = dict(
        run_id=run_id,
        started=ss._now(),
        status="running",
        argv=list(argv) if argv is not None else None,
        n_requested=ns,
        device=device_name,
        devices=[str(d) for d in (devices or [])],
        precision=get_precision(),
        threads=threads or os.cpu_count(),
        tile_size=int(tile_size),
        git=gitinfo,
        versions=versions(),
        host=os.uname().nodename,
        lattices=[],
        batches=[],
    )
    L = float(cfg["snapshot"]["box_cMpc"])
    n_lines, n_pix = len(dataset.line_ids), dataset.n_pix
    t_run = time.perf_counter()
    with dataset.lock():
        dataset.index(refresh=True)
        dataset.write_run(rec)
        todo = {n: dataset.missing(n) for n in ns}
        total_new = len({k for m in todo.values() for k in ss._keys(m, n_ref).tolist()}) if todo else 0
        done_new, tau_s, pass_s, n_pass_done = 0, 0.0, 0.0, 0
        log(f"dataset {dataset.path}")
        log(
            f"lattices {ns}: {total_new} skewers to compute ({n_lines} lines x {n_pix} pixels each; device {device_name}, "
            f"{rec['precision']}, tiles of {tile_size})"
        )
        try:
            for n in ns:
                miss = dataset.missing(n)  # the smaller N of this run are stored by now
                n_all = n * n
                model = dict(points_per_skewer=points_per_skewer(cfg), n_lattice=n)  # the route's memory model
                lat = dict(n=n, skewers=n_all, reused=n_all - len(miss), computed=len(miss), passes=[])
                rec["lattices"].append(lat)
                if not len(miss):
                    log(f"N = {n}: {n_all} skewers, all stored (reused {n_all})")
                    dataset.write_run(rec)
                    continue
                per = (
                    int(skewers_per_pass)
                    if skewers_per_pass
                    else pass_size(len(miss), cfg["snapshot"]["n_gas"], n_lines, n_pix, o["precision"], memory_budget, **model)
                )
                cofm = ss.lattice_positions(miss, dataset.axis, L, n_ref)
                ax = dataset.axis - 1
                paths = StraightPaths(cofm, np.eye(3)[np.full(len(miss), ax)], np.full(len(miss), L), L, ax)
                group = bounded_tiles(paths, per) if per < len(miss) else np.zeros(len(miss), np.int32)
                n_groups = int(group.max()) + 1
                est = pass_memory(
                    int(np.bincount(group).max()), cfg["snapshot"]["n_gas"], n_lines, n_pix, o["precision"], **model
                )
                log(
                    f"N = {n}: {n_all} skewers, {n_all - len(miss)} stored (reused), {len(miss)} to compute in {n_groups} pass(es) "
                    f"of <= {int(np.bincount(group).max())} skewers (estimated peak {est / 1e9:.1f} GB of "
                    f"{memory_available() / 1e9:.0f} GB available)"
                )
                for g in range(n_groups):
                    sel = miss[group == g]
                    t_g = time.perf_counter()
                    clock = dict(last=t_g)
                    prow = dict(n_skewers=int(len(sel)), batches=0)

                    def on_pass(info, prow=prow, clock=clock):
                        nonlocal pass_s, n_pass_done
                        pass_s += info["pass_s"]
                        n_pass_done += 1
                        clock["last"] = time.perf_counter()
                        log(
                            f"  the pass over the snapshot: {info['pass_s']:.1f} s, {info['n_cells_within_reach']:,} cells within "
                            f"reach, store {info['store_GB']:.2f} GB; {info['n_tiles']} batch(es) to compute"
                        )

                    def on_batch(idx_b, tau_b, row, n=n, lat=lat, prow=prow, clock=clock, g=g, n_groups=n_groups):
                        nonlocal done_new, tau_s
                        name = f"{run_id}_{len(rec['batches']):05d}"
                        res = row.get("resolution")
                        resd = (
                            None
                            if res is None
                            else {k: np.asarray(res[k]).tolist() for k in ("max_ratio", "column_fraction", "ok")}
                        )
                        dataset.write_batch(
                            idx_b,
                            tau_b,
                            name=name,
                            attrs=dict(
                                run_id=run_id,
                                git_commit=gitinfo["commit"] or "",
                                git_dirty=bool(gitinfo["dirty"]),
                                precision=rec["precision"],
                                device=device_name,
                                resolution=resd or {},
                            ),
                        )
                        now = time.perf_counter()
                        dt, clock["last"] = now - clock["last"], now
                        tau_s += dt
                        done_new += len(idx_b)
                        lat["done"] = lat.get("done", lat["reused"]) + len(idx_b)
                        prow["batches"] += 1
                        rec["batches"].append(
                            dict(
                                name=name,
                                n=n,
                                n_skewers=int(len(idx_b)),
                                seconds=dt,
                                resolution=resd,
                                geometry_s=row["geometry_s"],
                                contributions_s=row["contributions_s"],
                                tau_s=row["tau_s"].get("tau"),
                            )
                        )
                        rate = done_new / max(tau_s, 1e-9)
                        passes_left = (n_groups - g - 1) + sum(1 for m in ns if m > n and len(todo[m]))
                        eta = (total_new - done_new) / rate + passes_left * pass_s / max(n_pass_done, 1)
                        log(
                            f"  batch {name}: {len(idx_b)} skewers in {dt:.1f} s; N = {n}: {lat['done']}/{n * n}; new {done_new}/"
                            f"{total_new} ({100 * done_new / max(total_new, 1):.1f} %), {rate:.1f} skewers/s, ETA {_fmt_s(eta)}"
                            + ("" if resd is None else f"; max W/b per line {np.round(resd['max_ratio'], 3).tolist()}")
                        )
                        dataset.write_run(rec)

                    log(f" pass {g + 1}/{n_groups}: {len(sel)} skewers ...")
                    _, info = compute(
                        reader,
                        cfg,
                        sel,
                        tile_size=tile_size,
                        devices=devices,
                        threads=threads,
                        on_batch=on_batch,
                        on_pass=on_pass,
                    )
                    prow.update(info, total_s=time.perf_counter() - t_g)
                    lat["passes"].append(prow)
                    log(
                        f" pass {g + 1}/{n_groups} done in {_fmt_s(prow['total_s'])} (pass {info.get('pass_s', 0):.1f} s, tiles "
                        f"{info.get('tiles_s', 0):.1f} s)"
                    )
                    dataset.write_run(rec)
                log(f"N = {n}: complete ({n_all} skewers: {lat['reused']} reused, {lat['computed']} computed)")
            rec["status"] = "done"
        except BaseException as e:
            rec["status"] = "interrupted" if isinstance(e, KeyboardInterrupt) else "failed"
            rec["error"] = f"{type(e).__name__}: {e}"
            raise
        finally:
            rec["finished"] = ss._now()
            rec["wall_s"] = time.perf_counter() - t_run
            rec["computed"] = done_new
            dataset.write_run(rec)
            log(
                f"run {run_id}: {rec['status']}; {done_new} skewers computed and stored in {_fmt_s(rec['wall_s'])}; record "
                f"{dataset.path / 'runs' / f'run_{run_id}.json'}"
            )
    return rec
