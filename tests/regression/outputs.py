"""The arrays of the regression tests (``test_regression.py``) and of ``scripts/dev/update_regression_references.py``,
which writes them to ``data/`` (only for an intended change of the results, with the reason in ``CHANGELOG.md``).

* ``route_outputs`` -> ``data/route_toy.npz``: the route cells -> selection -> pieces -> tau and every optional mode the
  comparison material relies on (velocity_period, the reference frame, top hats, cell crossings, Harris's Voigt, the
  linear detuning), on uneven Voronoi cells and a grid of cells, skewers through the periodic faces;
* ``modules_outputs`` -> ``data/modules_toy.npz``: lines, a(x), bundles, the GRF and its optional terms, the Voigt
  modes, line profiles, spectral coordinates, the closed-form tau, every deposition back end, frames, the absorber
  integrals;
* ``tng_outputs`` -> ``data/modules_tng.npz``: the TNG50-4 reader, the gas physics of every ion and model, the streamed
  selections and tau along a skewer (needs the snapshot).

Every function imports SkewerJAX lazily: x64 must be enabled first (as ``tests/conftest.py`` does).
"""

import importlib

import numpy as np

REFERENCES = {"route_toy": "route_outputs", "modules_toy": "modules_outputs", "modules_tng": "tng_outputs"}


def route_outputs() -> dict:
    """The route cells -> selection -> pieces -> tau in every mode, on uneven Voronoi cells and a grid of cells."""
    import jax
    import jax.numpy as jnp

    from skewerjax.core import (
        Cosmology,
        LineBundle,
        SkewerBundle,
        cell_pieces,
        cells_from_values,
        compute_optical_depth,
        phys,
        select_cells,
        voronoi_volumes,
    )
    from skewerjax.simulations import PowerLawGRF, grf_cells

    box = 12.0
    rng = np.random.default_rng(11)
    gen = rng.uniform(0.0, box, (2500, 3))
    gen[:900, 1] = 5.0 + rng.uniform(0.0, 1.5, 900)  # a dense filament around the skewers
    nH = 1e-10 * rng.lognormal(0.0, 1.2, gen.shape[0])
    cells = cells_from_values(
        gen,
        box,
        {"HI": nH, "SiIII": 1e-4 * nH},
        temperature=10 ** rng.uniform(3.8, 5.5, gen.shape[0]),
        velocity=rng.normal(0.0, 60.0, gen.shape),
        volumes=voronoi_volumes(gen, box, samples_per_cell=32),
        ids=np.arange(gen.shape[0]),
    )
    cosmo = Cosmology(H0=phys.H0, Omega_r_0=0.0, Omega_m_0=0.3, Omega_k_0=0.0, Omega_lambda_0=0.7)
    t = np.linspace(0.0, 1.0, 1201)[:, None]
    paths = np.stack(
        [
            np.concatenate([1.2 * box * t, 5.6 + 0 * t, 9.0 + 0 * t], -1),  # along x, through the x faces
            np.concatenate([2.0 + box * t, 5.2 + 0.08 * box * t, 11.5 + 0.1 * box * t + 0.3 * np.sin(6 * t)], -1),
        ]
    )
    bundle = SkewerBundle(
        spatial_points=jnp.asarray(paths),
        a0=0.25,
        x0=0.0,
        ICs_defined_at_index=0,
        ICs_at_emission_or_absorption="absorption",
        cosmologies=cosmo,
    )
    lines = LineBundle(["HI-LyA", "SiIII-1206"], ref_line_id="HI-LyA")
    out = {
        "select_ids": np.asarray(select_cells(cells, paths).ids),
        "select_ids_tophat": np.asarray(select_cells(cells, paths, mode="tophat").ids),
    }
    pieces = cell_pieces(cells, lines, skewer_bundle=bundle)
    for f in ("scale_factor", "length_m", "temperature", "los_velocity", "number_density", "width_kms", "velocity_span_kms"):
        out[f"pieces_{f}"] = np.asarray(getattr(pieces, f))
    v = np.arange(-60.0, float(np.max(pieces.velocity_span_kms)) + 60.0, 1.0)
    run = lambda src, **kw: np.asarray(
        compute_optical_depth(src, v, lines, cells if src is bundle else None, resolution_tolerance=None, **kw)
    )
    out["tau_default"] = run(bundle)
    out["tau_period"] = run(pieces, velocity_period=True)
    out["tau_period_reference"] = run(pieces, velocity_period=True, line_frame="reference")
    out["tau_period_number"] = run(pieces, velocity_period=500.0)
    out["tau_tophat"] = run(bundle, deposition="tophat")
    out["tau_crossings"] = run(bundle, deposition="voronoi_crossings")
    out["tau_matched_kernel"] = run(pieces, voigt="harris", detuning="linear")
    out["tau_per_piece_sum"] = run(pieces, per_piece=True, velocity_period=True).sum(2)
    grf = PowerLawGRF(A_s=0.05, k_0=1.0, n_s=-1.0, L=(box,) * 3, N=(16,) * 3)
    grid = grf_cells(
        grf.sample(jax.random.PRNGKey(3)), grf, {"HI": 1e-10, "SiIII": 1e-14}, temperature=2e4, velocity=(30.0, 0.0, 0.0)
    )
    gpaths = np.stack([np.concatenate([box * t, 0.3 * box + 0 * t, 0.6 * box + 0 * t], -1)])
    gbundle = SkewerBundle(
        spatial_points=jnp.asarray(gpaths),
        a0=0.25,
        x0=0.0,
        ICs_defined_at_index=0,
        ICs_at_emission_or_absorption="absorption",
        cosmologies=cosmo,
    )
    out["tau_grid"] = np.asarray(compute_optical_depth(gbundle, v, lines, grid, velocity_period=True, resolution_tolerance=None))
    return out


def _flat(out: dict, name: str, x) -> None:
    if isinstance(x, dict):
        for k, v in x.items():
            _flat(out, f"{name}.{k}", v)
    elif x is not None and not isinstance(x, str):
        out[name] = np.asarray(x)


def modules_outputs() -> dict:
    import jax
    import jax.numpy as jnp

    from skewerjax.core import (
        Cosmology,
        LineBundle,
        SkewerBundle,
        cell_pieces,
        cells_from_values,
        compute_optical_depth,
        phys,
        select_cells,
        voronoi_volumes,
        wrap_to_ref_line,
    )
    from skewerjax.simulations import PowerLawGRF, grf_cells, interpolate_periodic

    dep, ab, spc, vg, lp, an = (
        importlib.import_module(f"skewerjax.core.{m}")
        for m in ("deposition", "absorbers", "spectral", "voigt", "line_profiles", "analytic")
    )
    out = {}
    put = lambda n, x: _flat(out, n, x)
    lines = LineBundle(["HI-LyA", "HI-LyB", "SiIII-1206", "SiII-1260"])
    put("lines", dict(lam=lines.lambda0_A, f=lines.f_osc, A=lines.A21_GHz, m=lines.mass_amu, dv=lines.delta_v))
    cosmo = Cosmology(H0=phys.H0, Omega_r_0=0.0, Omega_m_0=0.3, Omega_k_0=0.0, Omega_lambda_0=0.7)
    cosmo2 = Cosmology(
        H0=jnp.array([phys.H0, 1.1 * phys.H0]),
        Omega_r_0=jnp.array([1e-4, 0.0]),
        Omega_m_0=jnp.array([0.3, 0.25]),
        Omega_k_0=jnp.array([-1e-4, 0.05]),
        Omega_lambda_0=jnp.array([0.7, 0.7]),
    )
    put("a_of_x", cosmo2.solve_a_of_x(jnp.array([0.25, 0.3]), jnp.linspace(0.0, 400.0, 21)))
    t = np.linspace(0.0, 1.0, 61)[:, None]
    pts = np.stack(
        [np.concatenate([30 * t, 2 + 0 * t, 3 + 0 * t], -1), np.concatenate([5 + 20 * t, 4 + 3 * t, 1 + np.sin(5 * t)], -1)]
    )
    for ic in (0, -1):
        for mode in ("absorption", "emission"):
            b = SkewerBundle(
                spatial_points=jnp.asarray(pts),
                a0=0.25,
                ICs_defined_at_index=ic,
                ICs_at_emission_or_absorption=mode,
                cosmologies=cosmo2,
            )
            put(f"bundle_{ic}_{mode}", dict(a=b.scale_factors, d=b.los_directions, x=b.comoving_positions_from_ICs))
    x = jnp.linspace(-5.0, 5.0, 401)
    for s_, g_ in ((1.0, 1e-3), (0.3, 0.5), (0.05, 2.0)):
        for m in ("ExoJAX", "convolution", "harris", "scipy_wofz"):
            put(f"voigt_{m}_{s_}_{g_}", vg.log10_voigt(x, s_, g_, mode=m))
    v = jnp.linspace(-300.0, 300.0, 121)
    for m in ("ExoJAX", "harris"):
        put(
            f"profiles_{m}",
            lp.line_profiles(
                v,
                lines,
                temperature=jnp.array([1e3, 2e4, 1e6]),
                los_velocity=jnp.array([0.0, 30.0, -80.0]),
                scale_factor=0.25,
                a_dagger=0.25,
                turbulence=jnp.array([0.0, 1e3, 0.0]),
                voigt=m,
            ),
        )
    aa, u = jnp.array([0.25, 0.249, 0.248]), jnp.array([[0.0, 10.0, -5.0]] * 4)
    put(
        "spectral",
        dict(
            lw=spc.local_wavelength_A(v, lines.lambda0_A, 0.25, aa, u),
            vc=spc.line_centre_velocity_kms(0.25, aa, u),
            nul=spc.frequency_offset_GHz(v, lines.lambda0_A, 0.25, aa, u, velocity_period=400.0, detuning="linear"),
            frames=spc.velocities_in_line_frames(v, lines),
        ),
    )
    lam, one = jnp.linspace(4800.0, 4900.0, 200), lambda z: jnp.atleast_1d(jnp.asarray(z, dtype=float))
    put(
        "tau_static_thermal",
        an.tau_static_thermal(lam, lines.lambda0_A, lines.mass_amu, lines.f_osc, one(0.25), one(1.0), one(1e-12), one(1e4)),
    )
    for nd, shape in ((1, 128), (2, (32, 24)), (3, (12, 12, 12))):
        g = PowerLawGRF(A_s=0.05, k_0=1.0, n_s=-1.0, L=12.0 if nd == 1 else (12.0,) * nd, N=shape)
        d, dk = g.sample(jax.random.PRNGKey(nd), return_modes=True)
        put(f"grf{nd}", dict(d=d, dk=dk, var=g.variance()))
    ga = PowerLawGRF(
        A_s=0.05,
        k_0=1.0,
        n_s=-1.0,
        L=(12.0,) * 3,
        N=(12,) * 3,
        beta=1.3,
        sigma_fog=0.4,
        A_bao=1e-4,
        r_bao=3.0,
        sigma_bao=0.8,
        los_axis=1,
    )
    put("grf_aniso", dict(d=ga.sample(jax.random.PRNGKey(5)), pm=ga.power_monopole(jnp.linspace(0.1, 3, 10))))
    put(
        "grf_bao2d",
        PowerLawGRF(A_s=0.05, k_0=1.0, n_s=-1.0, L=(12.0, 12.0), N=(16, 16), A_bao=1e-4, r_bao=3.0, sigma_bao=0.8).power_grid(),
    )
    box, rng = 8.0, np.random.default_rng(11)
    gen = rng.uniform(0.0, box, (700, 3))
    nH = 1e-10 * rng.lognormal(0.0, 1.2, 700)
    cells = cells_from_values(
        gen,
        box,
        {"HI": nH, "SiIII": 1e-4 * nH},
        temperature=10 ** rng.uniform(3.8, 5.5, 700),
        velocity=rng.normal(0.0, 60.0, (700, 3)),
        volumes=voronoi_volumes(gen, box, samples_per_cell=16),
        turbulence={"HI": 0.0, "SiIII": 500.0},
        collisional=1e-3,
    )
    t = np.linspace(0.0, 1.0, 401)[:, None]
    paths = np.stack(
        [
            np.concatenate([1.2 * box * t, 3.6 + 0 * t, 5.0 + 0 * t], -1),
            np.concatenate([2.0 + box * t, 3.2 + 0.08 * box * t, 7.5 + 0.1 * box * t + 0.3 * np.sin(6 * t)], -1),
        ]
    )
    bundle = SkewerBundle(spatial_points=jnp.asarray(paths), a0=0.25, ICs_defined_at_index=0, cosmologies=cosmo)
    use = select_cells(cells, paths)
    for nm, s in (
        ("seg_numpy", dep.voronoi_segments(paths, use.positions, box, gas_size=use.size_cMpc)),
        ("seg_jax", dep.voronoi_segments(paths, use.positions, box, gas_size=use.size_cMpc, backend="jax")),
        ("seg_tophat", dep.tophat_segments(paths, use.positions, use.size_cMpc, box)),
    ):
        put(nm, dict(cell=s.cell, length=s.length, offset=s.offset))
    two = LineBundle(["HI-LyA", "SiIII-1206"])
    pieces, parts = cell_pieces(cells, two, skewer_bundle=bundle, return_parts=True)
    A, fr = parts["absorbers"]["HI"], parts["frame"]
    put("frame", dict(a=fr.scale_factors(1, np.linspace(0, 8, 20)), h=fr.hubble_rate(1, np.linspace(0, 8, 20))))
    vv = np.arange(-60.0, float(np.max(pieces.velocity_span_kms)) + 60.0, 2.0)
    put("tau_float32", compute_optical_depth(pieces, vv, two, dtype="float32", resolution_tolerance=None))
    put(
        "tau_chunks", compute_optical_depth(pieces, vv, two, piece_chunk=111, v_chunk=97, batch_size=2, resolution_tolerance=None)
    )
    P = float(pieces.velocity_span_kms[0])
    vw = (np.arange(500) + 0.5) * P / 500
    put(
        "wrap_to_ref",
        wrap_to_ref_line(
            compute_optical_depth(pieces, vw, two, velocity_period=P, resolution_tolerance=None), vw, two, velocity_period=P
        ),
    )
    sframe = ab.SnapshotFrame(a_snap=0.25, velfac=100.0, path_length=np.array([9.6, 8.2]))
    hi = LineBundle(["HI-LyA", "HI-LyB"])
    put(
        "tau_absorbers_snapshot",
        ab.tau_absorbers(
            A,
            hi,
            np.arange(0, 1000.0, 2.0),
            sframe,
            check=None,
            detuning="linear",
            pixel_offsets=np.array([-0.5, 0.5]),
            pixel_weights=np.array([0.5, 0.5]),
            velocity_period=960.0,
        ),
    )
    put("tau_by_path_bins", ab.tau_by_path_bins(A, hi, vv, fr, [0.0, 3.0, 9.0], skewer=1))
    put("crossings_tau", compute_optical_depth(bundle, vv, two, cells, deposition="voronoi_crossings", resolution_tolerance=None))
    grf = PowerLawGRF(A_s=0.05, k_0=1.0, n_s=-1.0, L=(box,) * 3, N=(12,) * 3)
    delta = grf.sample(jax.random.PRNGKey(3))
    put("interp", interpolate_periodic(delta, jnp.asarray(rng.uniform(0, box, (50, 3))), box))
    grid = grf_cells(delta, grf, {"HI": 1e-10, "SiIII": 1e-14}, temperature=2e4, velocity=(30.0, 0.0, 0.0))
    gp = cell_pieces(grid, two, skewer_bundle=bundle)
    put("grid_pieces", dict(n=gp.number_density, l=gp.length_m, a=gp.scale_factor, w=gp.width_kms))
    return out


def tng_outputs() -> dict:
    import jax.numpy as jnp

    from skewerjax.core import LineBundle, SkewerBundle, cell_pieces, compute_optical_depth
    from skewerjax.simulations import (
        HI,
        ArepoHDF5Reader,
        Ion,
        SimulationSkewers,
        ion_properties,
        select_cells_near_paths,
        snapshot_cells,
    )
    from skewerjax.utils.paths import snapshot_dir

    out = {}
    put = lambda n, x: _flat(out, n, x)
    r = ArepoHDF5Reader(snapshot_dir("TNG50-4"), 33)
    H = r.header
    put("header", dict(z=H.redshift, h=H.h, box=H.box_cMpc, vmax=H.vmax_kms, dens=H.density_cgs, n=H.n_gas))
    gas = r.to_gas_cells(
        r.read_rows(
            0,
            [
                "Coordinates",
                "Masses",
                "Density",
                "InternalEnergy",
                "ElectronAbundance",
                "NeutralHydrogenAbundance",
                "GFM_Metals",
                "StarFormationRate",
                "Velocities",
            ],
            stop=4000,
        )
    )
    for ion in (HI, Ion("Si", 3), Ion("C", 4)):
        put(f"ion_{ion.species}", {k: v for k, v in ion_properties(gas, ion).items() if not isinstance(v, (str, float))})
    for model in ("stored", "fake_sf", "rahmati_all"):
        put(f"ion_HI_{model}", ion_properties(gas, HI, x_HI_model=model, mass_unit="proton", T_constants="fake_spectra")["n_ion"])
    put("ion_SiIII_fake", ion_properties(gas, Ion("Si", 3), cloudy_density="fake_spectra", cloudy_interpolation="cubic")["n_ion"])
    reg = snapshot_cells(r, ["HI", "SiIII"], [10.0, 30.0, 30.0], [11.0, 31.0, 31.0], margin=0.3)
    put("region", dict(ids=reg.ids, n=reg.species["SiIII"]["number_density"]))
    L = H.box_cMpc
    t = np.linspace(0.0, 1.0, 1001)[:, None]
    pts = np.stack([np.concatenate([L * t, 0 * t + L / 4, 0 * t + L / 2], -1)])
    b = SkewerBundle(spatial_points=jnp.asarray(pts), a0=H.a, ICs_defined_at_index=0, cosmologies=H.cosmology())
    near = select_cells_near_paths(r, pts)
    put("near_ids", near.ids)
    cells = SimulationSkewers.from_bundle(r, b, cells=near).to_cells(["HI", "SiIII"])
    lines = LineBundle(["HI-LyA", "SiIII-1206"])
    p = cell_pieces(cells, lines, skewer_bundle=b)
    put("tau", compute_optical_depth(p, np.arange(0.0, float(p.velocity_span_kms[0]), 2.0), lines, resolution_tolerance=None))
    return out
