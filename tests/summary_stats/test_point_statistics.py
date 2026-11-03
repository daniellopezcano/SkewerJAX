"""The two paths of the 3D statistics: grid-native inputs (a field on its own grid, the lattice of skewer samples) take
the exact FFT and refuse a mass-assignment scheme; fields known at points go through a mesh (``mesh.fourier_mesh``:
assignment, interlacing, the window divided; Sefusatti et al. 2016) in ``GridStatistics3D.measure_points`` and
``point_statistics``, against the expectation with its alias images (``mesh.aliased_power``)."""

import inspect

import numpy as np
import pytest

from skewerjax.simulations import ToyBoxes
from skewerjax.summary_stats import (
    GRID_KEYS,
    GridStatistics3D,
    box_grid,
    field_statistics,
    fourier_mesh,
    grid_limits,
    lattice_grid,
    measure_lattice,
    paint,
    point_statistics,
    skewer_density_statistics,
)

from helpers import raises

BOXES = ToyBoxes(box_cMpc=64.0, n_cells=64, n_real=4, r_bao=20.0, sigma_bao=2.0, bao_ratio=0.01)


def _grid(n=16, L=64.0):
    lim = grid_limits(n, L)
    return box_grid(
        n, L, k_edges=np.linspace(0.5 * lim["k_min"], lim["k_max"], 9), r_edges=np.arange(0, L / 2 + 0.1, 4.0), log=None
    )


def test_the_point_path_defaults_to_interlaced_pcs():
    for fn in (paint, fourier_mesh, point_statistics, GridStatistics3D.measure_points):
        par = inspect.signature(fn).parameters
        assert par["mas"].default == "pcs", fn.__name__
        assert "interlace" not in par or par["interlace"].default is True, fn.__name__


def test_grid_native_inputs_refuse_a_scheme():
    gs = _grid()
    d = np.random.default_rng(0).normal(size=gs.shape)
    for fn, args in (
        (gs.measure, (d,)),
        (field_statistics, ([d], gs)),
        (measure_lattice, (d.transpose(0, 2, 1).reshape(-1, 16), gs)),
    ):
        raises(ValueError, "grid-native", fn, *args, mas="tsc")
    lat = lattice_grid(4, 4, 32, 64.0, 16.0, k_edges=np.linspace(0.05, 0.2, 4))
    cols = [{"HI": np.ones((4, 16, 4)), "SiIII": np.ones((4, 16, 4))}]
    raises(ValueError, "grid-native", skewer_density_statistics, cols, 2, lat, 64.0, mas="cic")
    a, b = gs.measure(d), gs.measure(d, mas=None)
    assert all(np.array_equal(a[k], b[k], equal_nan=True) for k in GRID_KEYS), "mas=None is the exact path, unchanged"
    raises(ValueError, "mas must be one of", gs.measure_points, np.zeros((5, 3)), np.ones(5), mas="sph")
    raises(ValueError, "(N, 3)", gs.measure_points, np.zeros((5, 2)), np.ones(5))


def test_a_plane_wave_through_the_mesh_and_the_shot_noise():
    """A plane wave sampled by jittered random points, measured through the mesh, equals its measurement on the grid's
    nodes (1 %, the sampling noise) in the bins that hold it; ``shot_noise`` lowers every P bin by exactly its value."""
    gs, rng = _grid(), np.random.default_rng(1)
    q, n, L = 4, 16, 64.0
    g = np.arange(n * q) * L / (n * q)
    X = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    X = X + rng.uniform(0, L / (n * q), X.shape)  # one random point in every sub-cell (jittered)
    k0 = 2 * np.pi * np.array([2, 3, 1]) / L
    on_nodes = np.cos(np.stack(np.meshgrid(*[np.arange(n) * L / n] * 3, indexing="ij"), -1) @ k0 + 0.2)
    ref = gs.measure(on_nodes)
    j = int(np.nanargmax(ref["P_iso"]))
    for mas in ("cic", "tsc", "pcs"):
        got = gs.measure_points(X, np.cos(X @ k0 + 0.2), mas=mas, interlace=True)
        assert got["P_iso"][j] == pytest.approx(ref["P_iso"][j], rel=1e-2), mas
        np.testing.assert_allclose(got["xi_iso"], ref["xi_iso"], atol=5e-3)  # xi = cos(k0.r) / 2 averaged
        less = gs.measure_points(X, np.cos(X @ k0 + 0.2), mas=mas, interlace=True, shot_noise=0.75)
        for key in ("P_iso", "P_map", "P_mu"):
            ok = np.isfinite(got[key])
            np.testing.assert_allclose(less[key][ok], got[key][ok] - 0.75, rtol=0, atol=1e-9 * np.nanmax(np.abs(got[key])))


def test_point_statistics_against_the_aliased_expectation():
    """Fields of a toy GRF known at the 64^3 nodes of their grid, measured through a 16^3 mesh: with every scheme and
    with or without interlacing, P(|k|) follows the alias-sum expectation within its errors (|pull| < 3.5 in every
    bin for TSC and PCS, below k_Nyq / 2 for NGP and CIC), while the same fields decimated to the 16^3 nodes (grid-native, no assignment) carry the aliased power."""
    b, g = BOXES, BOXES.grf()
    gs = _grid()
    x1 = np.arange(b.n_cells) * b.dx
    X = np.stack(np.meshgrid(x1, x1, x1, indexing="ij"), -1).reshape(-1, 3)
    for mas in ("ngp", "cic", "tsc", "pcs"):
        for interlace in (False, True):
            ps = point_statistics(
                ((X, b.delta(i).ravel()) for i in range(b.n_real)), gs, mas=mas, interlace=interlace, power=g.power, log=None
            )
            assert ps["mas"] == mas and ps["interlace"] is interlace and ps["n_real"] == b.n_real
            pull = (ps["P_iso"] - ps["theory"]["P_iso"]) / ps["P_iso_err"]
            # The points sit at 4 fixed offsets in every node's cell: their kernel is W aliased at 4 k_s, W(k) +
            # W(k -+ 4 k_s) + ..., a few % above W near k_Nyq for the kinked NGP and CIC (|W(k_Nyq - 4 k_s) / W(k_Nyq)|
            # = 1/7 per order), < 0.5 % for TSC and PCS: NGP and CIC are checked below k_Nyq / 2 only
            low = ps["axes"]["k_iso"] < (np.inf if mas in ("tsc", "pcs") else 0.5 * np.pi / gs.dx[0])
            assert np.all(np.abs(pull[low]) < 3.5), (mas, interlace, pull)
            assert np.all(np.isfinite(ps["cov_xi_iso"][1:, 1:]))
    dec = field_statistics((b.delta(i)[::4, ::4, ::4] for i in range(b.n_real)), gs, power=g.power, log=None)
    assert np.max(np.abs(dec["P_iso"] - dec["theory"]["P_iso"]) / dec["P_iso_err"]) > 5, "decimation aliases"
