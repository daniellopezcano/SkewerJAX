"""Small helpers shared by the test modules (importable: ``pythonpath = ["tests"]`` in ``pyproject.toml``)."""

import numpy as np

# Skewer sets of the snapshot tests (axis 1/2/3 = x/y/z, cofm in ckpc/h, TNG50-4 box 35000 ckpc/h).
SKEWER_SETS = {
    "single": dict(axis=np.array([3], dtype=np.int32), cofm=np.array([[17500.25, 17500.75, 0.0]])),
    "nset": dict(
        axis=np.tile(np.array([1, 2, 3], dtype=np.int32), 6)[:16],
        cofm=np.random.default_rng(2155).uniform(0.0, 35000.0, size=(16, 3)),
    ),
    "face": dict(axis=np.array([3, 1], dtype=np.int32), cofm=np.array([[5.0, 17500.75, 0.0], [0.0, 34995.0, 12000.5]])),
}
SKEWER_SETS["sk8"] = dict(axis=SKEWER_SETS["nset"]["axis"][8:9].copy(), cofm=SKEWER_SETS["nset"]["cofm"][8:9].copy())


def float64_only(fn):
    """Run the test in float64 whatever the precision of the run (``SKEWERJAX_PRECISION``): validations whose
    tolerances are float64 ones (closed forms to 1e-10, bit-for-bit equalities, the regression references). Works
    with pytest and with the shim (a plain decorator, no fixture)."""
    import functools

    @functools.wraps(fn)
    def run(*a, **kw):
        from skewerjax.config import precision

        with precision("float64"):
            return fn(*a, **kw)

    run.float64_only = True
    return run


def is_float32() -> bool:
    """True when the run's precision (``skewerjax.config.get_precision``) is float32."""
    from skewerjax.config import get_precision

    return get_precision() == "float32"


def tol(f64: float, f32: float) -> float:
    """A tolerance for the run's precision: ``f64`` in float64, ``f32`` in float32 (stated per test, from the float32
    audit of docs/precision.md)."""
    return f32 if is_float32() else f64


def raises(exc, match, fn, *a, **kw):
    """``fn(*a, **kw)`` raises ``exc`` whose message holds ``match``; returns the exception."""
    try:
        fn(*a, **kw)
    except exc as e:
        assert match in str(e), f"{match!r} not in {str(e)!r}"
        return e
    raise AssertionError(f"{getattr(fn, '__name__', fn)} did not raise {exc.__name__}")


def rel(a, b) -> float:
    """max |a - b| / max |b| (relative to the scale of b)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.max(np.abs(a - b)) / max(float(np.max(np.abs(b))), 1e-300))


def uniform_skewer(n_pt=201, L=2.0, n=1e-12, T=1e4):
    """A straight skewer of manual a(x) (log-linear, a from 0.30 to 0.302), unevenly sampled, through uniform H I gas:
    (SkewerBundle, Cells, points (1, n_pt, 3), a (1, 1, n_pt))."""
    import jax.numpy as jnp

    from skewerjax.core import SkewerBundle, cells_from_values

    pos = np.zeros((1, n_pt, 3))
    pos[0, :, 0] = np.linspace(0.0, L, n_pt) ** 1.3 / L**0.3  # uneven spacing
    a = np.exp(np.linspace(np.log(0.30), np.log(0.302), n_pt))[None, None]
    sb = SkewerBundle(spatial_points=jnp.asarray(pos), manual_scale_factors=jnp.asarray(a), a0=0.302, ICs_defined_at_index=-1)
    # cells at the points: the Voronoi cell of a point of the straight skewer is its dual segment
    cells = cells_from_values(pos[0], None, {"HI": np.full(n_pt, n)}, temperature=T, volumes=np.full(n_pt, (L / n_pt) ** 3))
    return sb, cells, pos, a


def ramp_pieces(n_skewers: int = 7, L: float = 1e-3):
    """Pieces of cells at the points of ``n_skewers`` parallel skewers (T a log ramp along z, n along y, v_x a ramp
    along x; H I and Si III): (Pieces, velocity grid (300,), LineBundle)."""
    import jax.numpy as jnp

    from skewerjax.core import LineBundle, SkewerBundle, cell_pieces, cells_from_values, plane_points

    lb = LineBundle(["HI-LyA", "SiIII-1206"])
    pts = plane_points(L, "x", L / 2, n_skewers, 64)
    gen = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    ramp = lambda lo, hi, ax: lambda x: 10 ** (np.log10(lo) + (np.log10(hi) - np.log10(lo)) * x[:, ax] / L)
    vel = lambda x: np.stack([-50.0 + 130.0 * x[:, 0] / L, 0 * x[:, 0], 0 * x[:, 0]], -1)
    cells = cells_from_values(
        gen,
        None,
        {"HI": ramp(1e-9, 1e-6, 1), "SiIII": ramp(1e-9, 1e-6, 1)},
        temperature=ramp(1e4, 1e6, 2),
        velocity=vel,
        collisional=0.1,
        volumes=np.full(gen.shape[0], (L / 63) ** 3),
    )
    sb = SkewerBundle(spatial_points=pts, manual_scale_factors=jnp.full((1,) + pts.shape[:2], 0.9))
    return cell_pieces(cells, lb, skewer_bundle=sb, a_dagger=0.88), jnp.linspace(-400, 400, 300), lb


# ---------------------------------------------------------------------- toy boxes along skewers (test-only fixtures)
def toy_tau(boxes, i, columns, points_per_cell: int, n_pix: int, *, fields=None, batch: int = 256, line_ids=None, **opts) -> dict:
    """tau (N_lines, N_sk, n_pix) of the lines ``line_ids`` (default ``simulations.LINE_IDS``) along skewers through the cell ``columns`` of realisation i of a
    ``ToyBoxes``, by the explicit pipeline of notebook 10: ``SkewerBundle(lattice_points)`` in the exact frame (z at
    y = 0) -> ``cell_pieces`` -> ``compute_optical_depth`` (periodic over one box, both lines on the Lya axis);
    ``opts``: a_eff, detuning, hubble_flow, aH_ref. ``fields``: the gas instead of ``boxes.fields(i)``. Returns
    dict(tau, v, period, dv, a_pix (the scale factor of the pixels), bundle (of the first batch), resolution
    (n_batches, 2, 2), warnings (list of str), fields)."""
    import warnings

    import jax.numpy as jnp

    from skewerjax.core import (
        LineBundle,
        SkewerBundle,
        bundle_frame,
        cell_pieces,
        compute_optical_depth,
        lattice_points,
        phys,
        velocity_span,
    )
    from skewerjax.simulations import LINE_IDS

    ids = list(LINE_IDS if line_ids is None else line_ids)
    lines = LineBundle(ids, ref_line_id=ids[0])
    f = boxes.fields(i) if fields is None else fields
    cells = boxes.cells(i, f)
    cols = np.atleast_2d(columns)
    frame_kw = dict(cosmologies=boxes.cosmology(), a0=1 / (1 + boxes.z), x0=0.0, ICs_defined_at_index=0)
    bundles = [
        SkewerBundle(
            spatial_points=jnp.asarray(
                lattice_points(cols[b : b + batch], boxes.dx, boxes.box_cMpc, boxes.n_cells, points_per_cell)
            ),
            **frame_kw,
        )
        for b in range(0, len(cols), batch)
    ]
    period = float(velocity_span(bundles[0], opts.get("hubble_flow", "exact"), opts.get("aH_ref"))[0])
    v = (np.arange(n_pix) + 0.5) * period / n_pix
    taus, res = [], []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for sb in bundles:
            pieces = cell_pieces(cells, lines, skewer_bundle=sb)
            tau, info = compute_optical_depth(
                pieces, v, lines, velocity_period=True, line_frame="reference", return_info=True, **opts
            )
            taus.append(np.asarray(tau))
            r = info["resolution"]
            res.append(np.stack([r["max_ratio"], r["column_fraction"]], -1))
    a_dagger = float(np.asarray(bundle_frame(bundles[0]).a_dagger))
    return dict(
        tau=np.concatenate(taus, axis=1),
        v=v,
        period=period,
        dv=period / n_pix,
        a_pix=a_dagger * np.exp(-v / phys.c_kms),
        bundle=bundles[0],
        resolution=np.stack(res),
        warnings=sorted({str(w.message) for w in caught}),
        fields=f,
    )


def toy_lattice_statistics(
    boxes, idx, stride, points_per_cell: int, n_pix: int, *, mean="redshift", decompose=False, line_ids=None
) -> dict:
    """The statistics of notebook 10 on a lattice of skewers through realisations ``idx`` of a ``ToyBoxes``, composed of
    the public functions (``toy_tau``; ``ensemble_mean_transmissions``, delta = T / T_bar - 1, ``overflux_check``;
    ``skewer_p1d``, ``skewer_xi1d``, ``lattice_grid`` + ``measure_lattice``; ``decompose_1d`` with ``decompose``; the
    density at the points, ``skewer_density_statistics``, against ``lattice_density_theory``). |k| bins: ``bao_k_edges``
    up to the smallest Nyquist frequency of the lattice. Returns dict(tau (n, N_lines, N_sk, n_pix), T (n, N_lines, N_sk,
    n_pix), T_bar and delta (n, N_lines + 1, N_sk, n_pix), check, v, period, dv, a_pix, mean_transmission (N_lines + 1, n_pix),
    resolution, warnings, p1d, xi1d, flux3d, axes3d, grid, flux (None without ``decompose``), and the density entries of
    ``skewer_density_statistics`` with density_theory and density_p1d["theory"])."""
    from skewerjax.core import lattice_columns
    from skewerjax.simulations import LINE_IDS
    from skewerjax.summary_stats import (
        GRID_KEYS,
        bao_k_edges,
        decompose_1d,
        ensemble_1d,
        ensemble_mean,
        ensemble_mean_transmissions,
        lattice_density_theory,
        lattice_grid,
        measure_lattice,
        overflux_check,
        skewer_density_statistics,
        skewer_p1d,
        skewer_xi1d,
    )

    sx, sz = stride
    N, L, p = boxes.n_cells, float(boxes.box_cMpc), int(points_per_cell)
    cols = lattice_columns(N, stride)
    ids = tuple(LINE_IDS if line_ids is None else line_ids)
    runs = [toy_tau(boxes, int(i), cols, p, n_pix, line_ids=ids) for i in idx]
    tau = np.stack([r["tau"] for r in runs])
    r0 = runs[0]
    T = np.exp(-tau)
    T_bar = ensemble_mean_transmissions(T, mean, grid=r0["a_pix"])
    delta = np.concatenate([T, T.prod(axis=1, keepdims=True)], axis=1) / T_bar - 1
    d_perp = np.array(stride, dtype=float) * boxes.dx
    k_edges = bao_k_edges(2 * np.pi / L, float(min(np.pi * n_pix / L, *(np.pi / d_perp))), 3.0 / boxes.sigma_bao)
    gs = lattice_grid(N // sx, N // sz, n_pix, L, d_perp, k_edges=k_edges)
    fields = (*ids, "total")
    p1d, xi1d, f3d = [], [], []
    for d in delta:
        k1d, P = skewer_p1d(d, r0["dv"])
        lags, X = skewer_xi1d(d, r0["dv"])
        p1d.append(dict(p1d=P))
        xi1d.append(dict(xi1d=X))
        f3d.append({f: measure_lattice(x, gs) for f, x in zip(fields, d)})
    out = dict(
        tau=tau,
        T=T,
        T_bar=T_bar,
        delta=delta,
        check=overflux_check(delta, T_bar, mean),
        v=r0["v"],
        period=r0["period"],
        dv=r0["dv"],
        a_pix=r0["a_pix"],
        mean_transmission=np.asarray(T_bar[0, :, 0]),
        resolution=np.stack([r["resolution"] for r in runs]),
        warnings=[r["warnings"] for r in runs],
        p1d=dict(ensemble_mean(p1d, ("p1d",)), k=k1d),
        xi1d=dict(ensemble_mean(xi1d, ("xi1d",)), lags=lags),
        flux3d={f: ensemble_mean([s[f] for s in f3d], GRID_KEYS) for f in fields},
        axes3d=gs.axes(),
        grid=gs,
        flux=ensemble_1d([decompose_1d(t, r0["dv"], tb, mean=mean) for t, tb in zip(tau, T_bar)]) if decompose else None,
    )
    dens = [{sp: n[::sx, :, ::sz] for sp, n in r["fields"]["n"].items()} for r in runs]
    gs_rho = lattice_grid(N // sx, N // sz, N * p, L, d_perp, k_edges=k_edges)
    ds = skewer_density_statistics(dens, p, gs_rho, L, n_bar=boxes.n_bar)
    th, p1d_th = lattice_density_theory(
        boxes.grf().power, gs_rho, N, stride, points_per_cell=p, p1d=True, scale=ds["density_scale"]
    )
    ds["density_p1d"]["theory"] = p1d_th
    out.update(ds, density_theory=th, grid_rho=gs_rho)
    return out
