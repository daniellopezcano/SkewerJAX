"""fake_spectra's choices as options of the core, one at a time: the default is unchanged (bit for bit) and the
alternative value reproduces the rule it cites, on toy input (``absorption.cpp``, ``singleabs.h`` written out in Python
here). The last test compares the whole set with fake_spectra's C engine itself, when it is importable."""

import importlib.util

import numpy as np
import pytest

from skewerjax.core import (
    Cells,
    LineBundle,
    SnapshotFrame,
    cell_pieces,
    column_per_pixel,
    compute_optical_depth,
    density_per_pixel,
    phys,
    tophat_chord_segments,
)
from skewerjax.core.line_profiles import B2_RATIOS

from helpers import float64_only, raises, tol

LYA = LineBundle(["HI-LyA"], ref_line_id="HI-LyA")
SI3 = LineBundle(["SiIII-1206"], ref_line_id="SiIII-1206")
BOX, A, VELFAC = 20.0, 0.25, 77.0  # [cMpc], scale factor, aH [km/s per cMpc]
V_MAX = BOX * VELFAC


def toy_gas(n_cells=120, seed=3, species=("HI",), log_n=(-15.0, -11.5), T=(3e3, 3e4)):
    """Random cells around the x axis of a periodic box: comoving densities spanning decades, as a skewer does."""
    r = np.random.default_rng(seed)
    pos = np.column_stack(
        [r.uniform(0, BOX, n_cells), BOX / 2 + r.normal(0, 0.25, n_cells), BOX / 2 + r.normal(0, 0.25, n_cells)]
    )
    vol = r.uniform(0.3, 1.0, n_cells) ** 3
    per = {
        sp: dict(
            number_density=10 ** r.uniform(*log_n, n_cells),
            temperature=r.uniform(*T, n_cells),
            velocity=r.normal(0, 60.0, (n_cells, 3)),
        )
        for sp in species
    }
    return Cells(pos, vol, per, box=BOX)


def toy_points(dv=0.25):
    x = np.linspace(0.0, BOX, int(round(V_MAX / dv)) + 1)
    return np.stack([x, np.full_like(x, BOX / 2), np.full_like(x, BOX / 2)], -1)[None]


def frame():
    return SnapshotFrame(A, VELFAC, path_length=np.array([BOX]))


def grid(nbins=400):
    dv = V_MAX / nbins
    return (np.arange(nbins) + 0.5) * dv, dv


def test_the_defaults_change_nothing():
    gas, pts, (v, _) = toy_gas(), toy_points(4.0), grid(200)
    p = cell_pieces(gas, LYA, points=pts, frame=frame(), a_data=A)
    assert p.cell is not None and p.cell.shape == p.scale_factor.shape
    t0 = np.asarray(compute_optical_depth(p, v, LYA, resolution_tolerance=None))
    t1 = np.asarray(
        compute_optical_depth(p, v, LYA, resolution_tolerance=None, b_constants="codata", pixel_average=False, tautail=None)
    )
    import dataclasses

    t2 = np.asarray(compute_optical_depth(dataclasses.replace(p, cell=None), v, LYA, resolution_tolerance=None))
    assert np.array_equal(t0, t1) and np.array_equal(t0, t2)
    for bad, msg in (
        (dict(b_constants="x"), "b_constants must be"),
        (dict(tautail=-1.0), "tautail must be"),
        (dict(tautail=1e-7, per_piece=True), "per_piece"),
    ):
        raises(ValueError, msg, compute_optical_depth, p, v, LYA, resolution_tolerance=None, **bad)
    raises(ValueError, "chord_weight applies", cell_pieces, gas, LYA, points=pts, frame=frame(), chord_weight=0.125)


def test_chord_points_of_a_sphere():
    """singleabs.h L100-L118: 7 interior points s_c + z_r (k/4 - 1), each of length 2 z_r w; the column is 7/7 or 7/8."""
    gas = Cells(
        np.array([[3.0, 5.0 + 0.3, 5.0]]),
        np.array([0.5**3]),
        dict(HI=dict(number_density=np.array([2e-6]), temperature=np.array([1e4]), velocity=np.zeros((1, 3)))),
        box=10.0,
    )
    x = np.linspace(0.0, 10.0, 41)
    pts = np.stack([x, np.full_like(x, 5.0), np.full_like(x, 5.0)], -1)[None]
    z_r = np.sqrt(0.5**2 - 0.3**2)
    for w, frac in ((None, 1.0), (1 / 8, 7 / 8)):
        seg = tophat_chord_segments(pts, gas.positions, gas.size_cMpc, gas.box, chord_weight=w)
        live = seg.cell >= 0
        s = (x[None, :, None] + seg.offset)[live]
        assert seg.kind == "tophat_chords" and seg.weight == 3.0 / (4.0 * np.pi)
        np.testing.assert_allclose(np.sort(s), 3.0 + z_r * (np.arange(1, 8) / 4.0 - 1.0), rtol=1e-12)
        np.testing.assert_allclose(seg.length[live].sum(), frac * 2.0 * z_r, rtol=1e-12)
    raises(
        ValueError,
        "straight",
        tophat_chord_segments,
        pts + np.array([0, 0, 1.0]) * np.sin(x)[None, :, None],
        gas.positions,
        gas.size_cMpc,
        10.0,
    )


def test_chords_wrap_at_the_seam_and_a_short_path_drops_them():
    gas = Cells(
        np.array([[0.1, 5.0, 5.0]]),
        np.array([0.5**3]),
        dict(HI=dict(number_density=np.array([2e-6]), temperature=np.array([1e4]), velocity=np.zeros((1, 3)))),
        box=10.0,
    )
    x = np.linspace(0.0, 10.0, 41)
    pts = np.stack([x, np.full_like(x, 5.0), np.full_like(x, 5.0)], -1)[None]
    s = x[None, :, None] + tophat_chord_segments(pts, gas.positions, gas.size_cMpc, 10.0).offset
    live = tophat_chord_segments(pts, gas.positions, gas.size_cMpc, 10.0).cell >= 0
    np.testing.assert_allclose(np.sort(s[live]), 0.1 + 0.5 * (np.arange(1, 8) / 4.0 - 1.0), rtol=1e-12)  # beyond the first point
    short = np.stack([x[:21], np.full(21, 5.0), np.full(21, 5.0)], -1)[None]
    seg = tophat_chord_segments(short, gas.positions, gas.size_cMpc, 10.0)
    assert (seg.cell >= 0).sum() == 4  # only the points with s >= 0 of the sphere at s = 0.1


@float64_only
def test_chord_weight_is_seven_eighths_of_the_column():
    gas, pts = toy_gas(), toy_points(4.0)
    v, _ = grid(200)
    full = cell_pieces(gas, LYA, points=pts, frame=frame(), a_data=A, deposition="tophat_chords")
    low = cell_pieces(gas, LYA, points=pts, frame=frame(), a_data=A, deposition="tophat_chords", chord_weight=1 / 8)
    np.testing.assert_allclose(np.asarray(low.length_m), 7 / 8 * np.asarray(full.length_m), rtol=1e-14)
    t_full = compute_optical_depth(full, v, LYA, resolution_tolerance=None, velocity_period=V_MAX)
    t_low = compute_optical_depth(low, v, LYA, resolution_tolerance=None, velocity_period=V_MAX)
    # linear in the amplitudes, up to the windows they select (the wing tolerance, 1e-9 of tau)
    np.testing.assert_allclose(np.asarray(t_low), 7 / 8 * np.asarray(t_full), rtol=tol(1e-7, 2e-6), atol=1e-8)
    exact = cell_pieces(gas, LYA, points=pts, frame=frame(), a_data=A, deposition="tophat")
    col = lambda p: float((np.asarray(p.number_density[0]) * np.asarray(p.length_m)).sum())
    np.testing.assert_allclose(col(full), col(exact), rtol=1e-2)  # the same chords; the seam and the touch rule at most differ


def test_density_per_pixel_is_the_column_over_the_physical_pixel():
    gas = toy_gas()
    _, parts = cell_pieces(gas, LYA, points=toy_points(4.0), frame=frame(), a_data=A, deposition="tophat", return_parts=True)
    ab = parts["absorbers"]["HI"]
    n = density_per_pixel(ab, BOX, 200)
    np.testing.assert_allclose(n, column_per_pixel(ab, BOX, 200) / (BOX / 200 * A * phys.Mpc_to_cm), rtol=1e-14)
    # the volume-weighted mean physical density along the path: sum n_com l / a^3 over the path length
    expected = (ab.n * ab.length).sum() / A**3 / BOX
    np.testing.assert_allclose(n.mean(), expected, rtol=1e-12)


def test_b_constants():
    """absorption.cpp L158: b^2 = 2 k T / (A m_p) with its constants, a factor of T inside the profile only."""
    ratio = (1.3806504e-16 / 1.67262178e-24) / (phys.kB_erg_per_K * phys.g_to_amu)
    np.testing.assert_allclose(B2_RATIOS["fake_spectra"], ratio, rtol=1e-15)
    assert 0.9927 < ratio < 0.9929
    gas, (v, _) = toy_gas(), grid(200)
    p = cell_pieces(gas, LYA, points=toy_points(4.0), frame=frame(), a_data=A)
    import dataclasses

    scaled = dataclasses.replace(p, temperature=p.temperature * ratio)
    # the same sums up to the rounding of T x ratio (in float64 on the host, or in the policy's dtype in the pieces) and the
    # windows it selects (the wing tolerance, 1e-9 of tau)
    np.testing.assert_allclose(
        np.asarray(compute_optical_depth(p, v, LYA, b_constants="fake_spectra", resolution_tolerance=None)),
        np.asarray(compute_optical_depth(scaled, v, LYA, resolution_tolerance=None)),
        rtol=tol(1e-7, 2e-6),
        atol=1e-8,
    )


def _pixel_rule_of_fake_spectra(b, dv):
    """singleabs.h L139-L154: the sub-pixel points of one absorber, as written there."""
    if dv < b / 2.0:
        return np.array([0.0]), np.array([1.0])
    npoints = int(2 * np.ceil(dv / (b / 2.0) / 2) + 1.0)
    w = np.ones(npoints)
    w[[0, -1]] = 0.5
    return np.linspace(-dv / 2, dv / 2, npoints), w / (npoints - 1)


@float64_only
def test_pixel_average_reproduces_the_trapezoid():
    """Si III (b ~ 2.4 km/s at 1e4 K) on 1.5 and 8 km/s pixels: each cell on its own number of points, against one
    compute_optical_depth per cell at those points."""
    gas = toy_gas(n_cells=40, species=("SiIII",), T=(5e2, 2e4), log_n=(-13.5, -11.5))
    p = cell_pieces(gas, SI3, points=toy_points(2.0), frame=frame(), a_data=A)
    for nbins in (1026, 192):
        v, dv = grid(nbins)
        got = np.asarray(compute_optical_depth(p, v, SI3, pixel_average=True, velocity_period=V_MAX, resolution_tolerance=None))[
            0, 0
        ]
        T = np.asarray(p.temperature[0, 0])
        m = float(np.asarray(SI3.mass_amu)[0])
        expected = np.zeros(nbins)
        for i in np.nonzero(np.asarray(p.length_m[0]) > 0)[0]:
            off, w = _pixel_rule_of_fake_spectra(np.sqrt(2.0 * phys.kB * T[i] / m) / 1e3, dv)
            one = p.skewers(np.array([0]))
            import dataclasses

            mask = np.zeros(p.scale_factor.shape[1])
            mask[i] = 1.0
            one = dataclasses.replace(one, number_density=one.number_density * mask, cell=None)
            t = np.asarray(
                compute_optical_depth(one, (v[:, None] + off).ravel(), SI3, velocity_period=V_MAX, resolution_tolerance=None)
            )
            expected += (t[0, 0].reshape(nbins, off.size) * w).sum(-1)
        np.testing.assert_allclose(got, expected, rtol=tol(1e-9, 2e-6), atol=tol(1e-8, 1e-7))  # the windows' wing tolerance
        if nbins == 1026:  # dv = 1.5 km/s: narrow cells only; the wider ones stay at the centre
            centre = np.asarray(compute_optical_depth(p, v, SI3, velocity_period=V_MAX, resolution_tolerance=None))[0, 0]
            assert np.abs(got - centre).max() > 0.0


def _walk_of_fake_spectra(tau_cell, zmax, tautail):
    """absorption.cpp L253-L281 for one cell, as written there (periodic indices, half a box each way)."""
    nbins = tau_cell.size
    out = np.zeros(nbins)
    for z in range(zmax, zmax + nbins // 2):
        out[z % nbins] += tau_cell[z % nbins]
        if tau_cell[z % nbins] < tautail:
            break
    for z in range(zmax - 1, zmax - nbins // 2 - 1, -1):
        out[z % nbins] += tau_cell[z % nbins]
        if tau_cell[z % nbins] < tautail:
            break
    return out


@pytest.mark.parametrize("tautail", [1e-7, 1e-3])
@float64_only
def test_tautail_reproduces_the_walk(tautail):
    gas = toy_gas(n_cells=60)
    p, parts = cell_pieces(
        gas,
        LYA,
        points=toy_points(4.0),
        frame=frame(),
        a_data=A,
        deposition="tophat_chords",
        chord_weight=1 / 8,
        return_parts=True,
    )
    v, dv = grid(300)
    got = np.asarray(compute_optical_depth(p, v, LYA, tautail=tautail, velocity_period=V_MAX, resolution_tolerance=None))[0, 0]
    pp = np.asarray(compute_optical_depth(p, v, LYA, per_piece=True, velocity_period=V_MAX, resolution_tolerance=None))[0, 0]
    cell = np.asarray(p.cell[0])
    live = cell >= 0
    expected = np.zeros(v.size)
    for c in np.unique(cell[live]):
        m = cell == c
        s_c = np.asarray(parts["cells"].positions)[c, 0]
        vel = VELFAC * s_c + np.asarray(parts["cells"].species["HI"]["velocity"])[c, 0]
        expected += _walk_of_fake_spectra(pp[m].sum(0), int(np.floor(vel / dv)), tautail)
    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-13)
    full = pp.sum(0)
    assert np.abs(got - full).max() > 0.0 and np.all(got <= full * (1 + 1e-12))  # only wings are cut


@float64_only
def test_tautail_needs_the_cells_and_a_grid_over_the_period():
    import dataclasses

    gas, (v, _) = toy_gas(n_cells=30), grid(100)
    p = cell_pieces(gas, LYA, points=toy_points(8.0), frame=frame(), a_data=A)
    raises(
        ValueError,
        "Pieces.cell",
        compute_optical_depth,
        dataclasses.replace(p, cell=None),
        v,
        LYA,
        tautail=1e-7,
        resolution_tolerance=None,
    )
    raises(
        ValueError,
        "its span",
        compute_optical_depth,
        p,
        v,
        LYA,
        tautail=1e-7,
        velocity_period=2 * V_MAX,
        resolution_tolerance=None,
    )
    raises(ValueError, "uniform", compute_optical_depth, p, v**1.01, LYA, tautail=1e-7, resolution_tolerance=None)
    # without a period the walk stops at the ends of the grid, and with a threshold of 0+ it keeps every pixel
    # (the per-cell route against the windowed engine: equal up to the wing tolerance, 1e-9 of tau, and float32 rounding)
    t = np.asarray(compute_optical_depth(p, v, LYA, tautail=1e-300, resolution_tolerance=None))
    np.testing.assert_allclose(
        t, np.asarray(compute_optical_depth(p, v, LYA, resolution_tolerance=None)), rtol=tol(1e-9, 2e-6), atol=1e-8
    )
    # no pixel average and no threshold reachable: the per-cell route equals the sum over pieces, whatever the deposition
    tp = np.asarray(compute_optical_depth(p, v, LYA, tautail=1e-300, velocity_period=V_MAX, resolution_tolerance=None))
    np.testing.assert_allclose(
        tp,
        np.asarray(compute_optical_depth(p, v, LYA, velocity_period=V_MAX, resolution_tolerance=None)),
        rtol=tol(1e-9, 2e-6),
        atol=1e-8,
    )
    _, info = compute_optical_depth(p, v, LYA, tautail=1e-7, velocity_period=V_MAX, resolution_tolerance=None, return_info=True)
    assert info["n_v"] == v.size and info["timings"]["tau"] > 0.0


@pytest.mark.fake_spectra
@pytest.mark.skipif(importlib.util.find_spec("fake_spectra") is None, reason="fake_spectra is not installed")
def test_the_matched_options_reproduce_the_c_engine():
    """Every option at fake_spectra's value against ``_Particle_Interpolate`` on toy particles (float64, same units:
    the code length 1 kpc/h with h = 1, velfac from a H)."""
    from fake_spectra._spectra_priv import _Particle_Interpolate
    from fake_spectra.line_data import LineData

    gas = toy_gas(n_cells=300)
    f = gas.species["HI"]
    line = LineData()[("H", 1)][1215]
    amu = LineData().get_mass("H")
    box = BOX * 1e3  # [kpc/h], h = 1
    velfac_code = VELFAC / 1e3  # [km/s per kpc/h]
    nbins = 1540
    pos = np.asarray(gas.positions) * 1e3
    elem_den = np.asarray(f["number_density"]) / A**3 * (phys.Mpc_to_cm / 1e3 * A)  # n_phys rscale: atoms cm^-3 x cm / (kpc/h)
    tau_c = _Particle_Interpolate(
        1,
        nbins,
        0,
        box,
        velfac_code,
        A,
        line.lambda_X * 1e-8,
        line.gamma_X,
        line.fosc_X,
        amu,
        1e-7,
        pos,
        np.asarray(f["velocity"], dtype=np.float64),
        elem_den,
        np.asarray(f["temperature"], dtype=np.float64),
        np.asarray(gas.size_cMpc) * 1e3,
        np.array([1], dtype=np.int32),
        np.array([[0.0, BOX / 2 * 1e3, BOX / 2 * 1e3]]),
    )
    p = cell_pieces(gas, LYA, points=toy_points(1.0), frame=frame(), a_data=A, deposition="tophat_chords", chord_weight=1 / 8)
    v, dv = grid(nbins)
    t = np.asarray(
        compute_optical_depth(
            p,
            v,
            LYA,
            velocity_period=nbins * dv,
            voigt="harris",
            detuning="linear",
            b_constants="fake_spectra",
            pixel_average=True,
            tautail=1e-7,
            resolution_tolerance=None,
        )
    )[0, 0]
    F_c, F = np.exp(-np.asarray(tau_c)[0]), np.exp(-t)
    assert 0.1 < F.mean() < 0.95, F.mean()  # neither empty nor saturated
    assert np.abs(F_c - F).max() < tol(1e-6, 1e-5), np.abs(F_c - F).max()  # float32: 1.2e-6
