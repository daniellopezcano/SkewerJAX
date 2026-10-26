"""Regression of the route cells -> selection -> pieces -> tau on small toy inputs, against reference arrays computed with
the code of the checkpoint of 2026-10-26 before its cleanup (``tests/data/regression_toy.npz``; regenerate with
``python tests/test_regression.py --write`` only after a deliberate change of the results). Every array must agree to
1e-10 of its largest value: uneven Voronoi cells and a grid of cells, skewers through the periodic faces, the default
options and every optional mode the comparison material relies on (velocity_period, the reference frame, top hats,
cell crossings, Harris's Voigt and the linear detuning)."""
import sys
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent / "data" / "regression_toy.npz"


def toy_outputs() -> dict:
    """The arrays of the regression (each a NumPy array)."""
    import jax
    import jax.numpy as jnp
    from skewerjax.core import (Cosmology, LineBundle, SkewerBundle, cell_pieces, cells_from_values, compute_optical_depth, phys,
                                select_cells, voronoi_volumes)
    from skewerjax.simulations import PowerLawGRF, grf_cells
    box = 12.0
    rng = np.random.default_rng(11)
    gen = rng.uniform(0.0, box, (2500, 3))
    gen[:900, 1] = 5.0 + rng.uniform(0.0, 1.5, 900)                          # a dense filament around the skewers
    nH = 1e-10 * rng.lognormal(0.0, 1.2, gen.shape[0])
    cells = cells_from_values(gen, box, {"HI": nH, "SiIII": 1e-4 * nH}, temperature=10 ** rng.uniform(3.8, 5.5, gen.shape[0]),
                              velocity=rng.normal(0.0, 60.0, gen.shape), volumes=voronoi_volumes(gen, box, samples_per_cell=32),
                              ids=np.arange(gen.shape[0]))
    cosmo = Cosmology(H0=phys.H0, Omega_r_0=0.0, Omega_m_0=0.3, Omega_k_0=0.0, Omega_lambda_0=0.7)
    t = np.linspace(0.0, 1.0, 1201)[:, None]
    paths = np.stack([np.concatenate([1.2 * box * t, 5.6 + 0 * t, 9.0 + 0 * t], -1),          # along x, through the x faces
                      np.concatenate([2.0 + box * t, 5.2 + 0.08 * box * t, 11.5 + 0.1 * box * t + 0.3 * np.sin(6 * t)], -1)])
    bundle = SkewerBundle(spatial_points=jnp.asarray(paths), a0=0.25, x0=0.0, ICs_defined_at_index=0,
                          ICs_at_emission_or_absorption="absorption", cosmologies=cosmo)
    lines = LineBundle(["HI-LyA", "SiIII-1206"], ref_line_id="HI-LyA")
    out = {"select_ids": np.asarray(select_cells(cells, paths).ids), "select_ids_tophat": np.asarray(select_cells(cells, paths, mode="tophat").ids)}
    pieces = cell_pieces(cells, lines, skewer_bundle=bundle)
    for f in ("scale_factor", "length_m", "temperature", "los_velocity", "number_density", "width_kms", "velocity_span_kms"):
        out[f"pieces_{f}"] = np.asarray(getattr(pieces, f))
    v = np.arange(-60.0, float(np.max(pieces.velocity_span_kms)) + 60.0, 1.0)
    run = lambda src, **kw: np.asarray(compute_optical_depth(src, v, lines, cells if src is bundle else None, resolution_tolerance=None, **kw))
    out["tau_default"] = run(bundle)
    out["tau_period"] = run(pieces, velocity_period=True)
    out["tau_period_reference"] = run(pieces, velocity_period=True, line_frame="reference")
    out["tau_period_number"] = run(pieces, velocity_period=500.0)
    out["tau_tophat"] = run(bundle, deposition="tophat")
    out["tau_crossings"] = run(bundle, deposition="voronoi_crossings")
    out["tau_matched_kernel"] = run(pieces, voigt="harris", detuning="linear")
    out["tau_per_piece_sum"] = run(pieces, per_piece=True, velocity_period=True).sum(2)
    grf = PowerLawGRF(A_s=0.05, k_0=1.0, n_s=-1.0, L=(box,) * 3, N=(16,) * 3)
    grid = grf_cells(grf.sample(jax.random.PRNGKey(3)), grf, {"HI": 1e-10, "SiIII": 1e-14}, temperature=2e4, velocity=(30.0, 0.0, 0.0))
    gpaths = np.stack([np.concatenate([box * t, 0.3 * box + 0 * t, 0.6 * box + 0 * t], -1)])
    gbundle = SkewerBundle(spatial_points=jnp.asarray(gpaths), a0=0.25, x0=0.0, ICs_defined_at_index=0,
                           ICs_at_emission_or_absorption="absorption", cosmologies=cosmo)
    out["tau_grid"] = np.asarray(compute_optical_depth(gbundle, v, lines, grid, velocity_period=True, resolution_tolerance=None))
    return out


def test_route_matches_the_reference():
    ref = np.load(DATA)
    out = toy_outputs()
    assert set(ref.files) == set(out), set(ref.files) ^ set(out)
    for k in ref.files:
        a, b = ref[k], out[k]
        assert a.shape == b.shape, (k, a.shape, b.shape)
        if a.dtype.kind in "iu":
            np.testing.assert_array_equal(b, a, err_msg=k)
        else:
            assert np.max(np.abs(b - a)) <= 1e-10 * max(np.max(np.abs(a)), 1e-300), (k, np.max(np.abs(b - a)), np.max(np.abs(a)))
    assert ref["tau_default"].max() > 1.0 and ref["tau_period"].max() > 1.0          # the toy absorbs: the test has teeth


if __name__ == "__main__" and "--write" in sys.argv:
    import jax
    jax.config.update("jax_enable_x64", True)
    DATA.parent.mkdir(exist_ok=True)
    np.savez_compressed(DATA, **toy_outputs())
    print("wrote", DATA)
