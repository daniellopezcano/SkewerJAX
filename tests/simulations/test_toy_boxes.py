"""The toy boxes of notebooks 08 and 10 (``simulations.ToyBoxes``) on a small grid: realisations regenerated from their
seeds, the gas (n_X = r_X n_HI per metal species, T = T0 (1 + delta)^(gamma - 1), zero or linear-theory velocities, no clipping), the
continuous xi(r); and the boxes along skewers by the explicit pipeline of notebook 10 (``helpers.toy_tau``,
``helpers.toy_lattice_statistics``: ``core.lattice_points`` -> ``core.SkewerBundle`` -> ``cell_pieces`` ->
``compute_optical_depth`` -> ``summary_stats``): one realisation (tau, the resolution check, the 1D statistics by order,
the 3D statistics and P1D of the overfluxes), the ensemble normalisation of the overflux in every mode of the mean
transmission (<delta> = 0 over the sets of the mode, C = sum_{|A| >= 2} <delta_A>, the exact decomposition), the density
conversion a_eff and the Jacobian of the exact frame on a uniform gas, and the linear Hubble flow."""

from dataclasses import replace

import numpy as np
import pytest

from skewerjax.core import hubble_velocity, lattice_columns
from skewerjax.simulations import LINE_IDS, ToyBoxes

from helpers import float64_only, tol, toy_lattice_statistics, toy_tau

BOXES = ToyBoxes(box_cMpc=8.0, n_cells=16, n_real=3, bao_ratio=1e-3)  # below R_max of this grid
STRIDE, P, N_PIX = (4, 4), 80, 64  # 16 skewers; cells of 40 km/s: pieces of W = 0.5 km/s


def _uniform(boxes):
    """The gas of realisation 0 made uniform: n = n_bar, T = T0, v = 0."""
    f = boxes.fields(0)
    n_bar = boxes.n_bar
    return dict(
        delta=np.zeros_like(f["delta"]),
        n={sp: np.full_like(f["delta"], n) for sp, n in n_bar.items()},
        T=np.full_like(f["T"], boxes.T0),
        v=np.zeros_like(f["v"]),
    )


@float64_only
def test_realisations_and_gas():
    b = BOXES
    assert np.array_equal(b.seeds, b.base_seed + np.arange(b.n_real))
    np.testing.assert_array_equal(b.delta(1), b.delta(1))  # regenerated, identical
    assert not np.allclose(b.delta(0), b.delta(1))
    assert np.isclose(float(b.grf().variance()), b.sigma_delta**2, rtol=1e-12)
    assert not b.grf().anisotropic and b.grf().A_bao == pytest.approx(b.bao_ratio * b.A_s)
    f = b.fields(0)
    np.testing.assert_allclose(f["n"]["HI"], b.n_bar_HI * (1 + f["delta"]), rtol=1e-14)
    assert b.species == ("HI", "SiIII", "SiII") and b.metal_to_hi == {"SiIII": 0.2, "SiII": 0.1}, "the toy ratios"
    for x, r in b.metal_to_hi.items():
        np.testing.assert_allclose(f["n"][x], r * f["n"]["HI"], rtol=1e-14)
    with pytest.raises(ValueError, match="metal_to_hi"):
        ToyBoxes(box_cMpc=8.0, bao_ratio=1e-3, metal_to_hi={"SiIII": 0.0})
    np.testing.assert_allclose(f["T"], b.T0 * (1 + f["delta"]) ** b.gamma_minus_1, rtol=1e-14)
    assert b.velocity == "zero" and f["v"].shape == f["delta"].shape + (3,) and not np.any(f["v"]), "zero velocities by default"
    with pytest.raises(ValueError, match="velocity"):
        ToyBoxes(box_cMpc=8.0, velocity="kaiser", bao_ratio=1e-3)
    # linear theory: div v = -a H f delta, mode by mode (except on the Nyquist planes, where k_i v_i is a sine of the grid)
    lin = replace(b, velocity="linear")
    f = lin.fields(0)
    np.testing.assert_array_equal(f["delta"], b.fields(0)["delta"])  # the same density
    g = b.grf()
    k_vec, _ = g.k_grid(real=True)
    div = sum(1j * np.asarray(kv) * np.fft.rfftn(f["v"][..., a]) for a, kv in enumerate(k_vec))
    k_nyq = np.pi / b.dx
    inner = ~np.any([np.isclose(np.abs(np.broadcast_to(kv, div.shape)), k_nyq) for kv in k_vec], axis=0)
    inner.ravel()[0] = False
    np.testing.assert_allclose(div[inner], -b.aHf * np.fft.rfftn(f["delta"])[inner], rtol=1e-8, atol=1e-8)
    assert b.aH == pytest.approx(70.0 * np.sqrt(0.31 * 64 + 0.69) / 4) and b.aHf == b.aH * b.f_growth  # notebook 04, z = 3
    cells = b.cells(0, f)
    assert len(cells) == b.n_cells**3 and set(cells.species_keys) == {"HI", "SiIII", "SiII"}
    with pytest.raises(ValueError, match="1 \\+ delta"):  # no clipping: a negative density is an error
        ToyBoxes(box_cMpc=8.0, n_cells=16, sigma_delta=0.8, bao_ratio=1e-3).fields(0)


@float64_only
def test_one_box_along_its_skewers():
    """One realisation along its lattice of skewers, normalised by the mean of its own skewers at each pixel (the
    default mean="redshift"): tau, the resolution check, the exact decomposition by order, Parseval for P1D, and the 3D
    statistics of each overflux on the lattice."""
    from skewerjax.core import velocity_span

    r = toy_lattice_statistics(BOXES, [0], STRIDE, P, N_PIX, decompose=True)  # the order-by-order decomposition: opt-in
    n_sk = lattice_columns(BOXES.n_cells, STRIDE).shape[0]
    m = len(LINE_IDS)
    assert m == 4 and r["tau"].shape == (1, m, n_sk, N_PIX)
    # the pieces are the cells' crossings, wide in velocity, integrated exactly: no warning (the report only informs)
    assert r["warnings"] == [[]] and np.all(r["resolution"][..., 0] > 0), "no resolution warning with the integrated profile"
    F = r["T"][0]
    assert 0 < F.prod(0).mean() < F.mean(axis=(1, 2)).min() < 1
    s = r["flux"]
    assert np.max(np.abs(s["P_order"].sum(0) - s["P_total"])) < 1e-10 * np.max(np.abs(s["P_total"]))
    sb = toy_tau(BOXES, 0, [[0, 0]], P, N_PIX)["bundle"]
    assert r["period"] == pytest.approx(float(velocity_span(sb, "exact")[0]), rel=1e-12)
    assert set(r["flux3d"]) == {*LINE_IDS, "total"} and r["p1d"]["p1d"].shape == (m + 1, N_PIX // 2 + 1)
    assert s["P_order"].shape[0] == 2 * m + 1, "orders 0 .. 2 N_lines"
    delta = F.prod(0) / F.prod(0).mean(axis=0) - 1
    np.testing.assert_allclose(r["delta"][0, -1], delta, atol=1e-12)
    assert abs(delta.mean()) < 1e-14 and r["check"]["decomposition_error"] < 1e-14
    w = np.r_[1.0, np.full(N_PIX // 2 - 1, 2.0), 1.0]  # rfft modes -> full grid
    assert np.sum(w * r["p1d"]["p1d"][-1]) / (N_PIX * r["dv"]) == pytest.approx(np.mean(delta**2), rel=1e-10)
    xi = r["flux3d"]["total"]["xi_iso"]
    assert np.isfinite(xi[0]) and xi[0] <= np.mean(delta**2) * (1 + 1e-12)  # bin 0 holds lag 0 and its neighbours


@pytest.mark.parametrize("mean", ["redshift", "global", "box", "skewer"])
@float64_only
def test_ensemble_normalisation_of_the_overflux(mean):
    """The explicit pipeline of notebook 10 in every mode of the mean transmission: T_bar over the pixels at the same
    redshift of every skewer of every box ("redshift", the default: the per-pixel ensemble mean on the shared grid of the
    lattice), over every pixel of every skewer of every box ("global"), of each box ("box") or along each skewer
    ("skewer"), so <delta> = 0 over the sets of the mode (``mean_delta_level``); C = T_bar / prod_j T_bar_j - 1 =
    sum_{|A| >= 2} <delta_A> over the same sets (``C_error``), the decomposition delta_tot = [sum_A delta_A - C] / (1 + C) is
    exact and order 1 vanishes; only "redshift" removes the pattern shared by every skewer (``mean_delta``)."""
    r = toy_lattice_statistics(BOXES, [0, 1, 2], STRIDE, P, N_PIX, mean=mean, decompose=True)
    T = np.moveaxis(r["T"], 1, 0)  # (N_lines, 3 boxes, N_sk, n_pix)
    T = np.concatenate([T, T.prod(0)[None]])  # the lines and the total
    axes = {"redshift": (1, 2), "global": (1, 2, 3), "box": (2, 3), "skewer": 3}[mean]
    T_bar = np.broadcast_to(T.mean(axis=axes, keepdims=True), T.shape)  # the known means of the mode
    np.testing.assert_allclose(np.moveaxis(r["T_bar"], 1, 0), T_bar, rtol=1e-13)
    assert np.all(np.diff(r["a_pix"]) < 0) and r["a_pix"][0] < 1 / (1 + BOXES.z), "a decreases with v"
    delta = T / T_bar - 1
    chk = r["check"]
    np.testing.assert_allclose(chk["mean_delta"], delta.mean(axis=(1, 2)), atol=1e-14)
    assert np.max(chk["mean_delta_level"]) < 1e-14, "<delta> = 0 over the sets of the mode"
    assert np.max(np.abs(delta.mean(axis=axes))) < 1e-14
    shared = np.max(np.abs(chk["mean_delta"]))
    assert (shared < 1e-14) if mean == "redshift" else (shared > 1e-4), "only 'redshift' removes the shared pattern"
    assert chk["C_error"] < 1e-14 and chk["decomposition_error"] < 1e-14, "C = sum <delta_A>, the decomposition exact"
    C = T_bar[-1] / np.prod(T_bar[:-1], axis=0) - 1
    assert chk["C"] == pytest.approx(C.mean(), rel=1e-10)
    e = r["flux"]
    for kind in ("P", "xi"):
        sc = np.max(np.abs(e[f"{kind}_total"]))
        assert np.max(np.abs(e[f"{kind}_order"].sum(0) - e[f"{kind}_total"])) < 1e-12 * sc
        assert np.max(np.abs(e[f"{kind}_order"][1])) < 1e-12 * sc, "order 1 vanishes over the sets of the mode"


@float64_only
def test_a_eff_and_the_jacobian_of_the_exact_frame():
    """A uniform gas in the exact frame of the toy skewers (z at y = 0, a(y) decreasing): tau per pixel follows
    n a(y)^-2 / (a H(y)) (the comoving-to-physical factor and the Jacobian dy / dv of the line centres), about +6.4 % from
    y = 0 to y = L at z = 3, L = 150 cMpc; with ``a_eff`` only the Jacobian 1 / (a H(y)) remains, predicted from a(y)
    of the bundle (about -1.8 %: a H grows with z), each to 1e-4; a_eff = None is the per-piece a_p."""
    from skewerjax.core import phys

    boxes = ToyBoxes(box_cMpc=150.0, n_cells=16, n_real=1, bao_ratio=0.0)
    p, n_pix = 128, 64  # W = 5.9 km/s << b = 15.7 km/s (Lya)
    uniform = _uniform(boxes)
    col = np.array([[0, 0]])
    sb = toy_tau(boxes, 0, col, p, n_pix, fields=uniform)["bundle"]
    vH = hubble_velocity(sb)
    y = np.linspace(0.0, boxes.box_cMpc, vH.size)
    a = np.asarray(sb.scale_factors[0, 0], dtype=np.float64)
    assert a[0] == pytest.approx(1 / (1 + boxes.z), rel=1e-12) and a[-1] < a[0], "z at y = 0, a decreasing along y"
    aH = np.gradient(vH, y)  # dv / dy [km/s per cMpc]
    res, taus = {}, {}
    for name, a_eff in (("None", None), ("a_eff", 1 / (1 + boxes.z))):
        r = toy_tau(boxes, 0, col, p, n_pix, fields=uniform, a_eff=a_eff)
        tau = r["tau"][0, 0]  # Lya
        y_pix = np.interp(r["v"], vH, y)
        conv = np.interp(y_pix, y, a) ** -2 if a_eff is None else np.full_like(y_pix, a_eff**-2)
        pred = conv / np.interp(y_pix, y, aH)
        np.testing.assert_allclose(tau / tau.mean(), pred / pred.mean(), rtol=1e-4)
        res[name] = tau[-1] / tau[0] - 1
        taus[name] = tau
    # a_eff = a(0): every pixel scaled by (a_p / a_eff)^2 against the per-piece conversion
    np.testing.assert_allclose(taus["a_eff"] / taus["None"], (np.interp(y_pix, y, a) * (1 + boxes.z)) ** 2, rtol=1e-4)
    full = (a[0] / a[-1]) ** 2 * aH[0] / aH[-1] - 1  # from y = 0 to L
    jac = aH[0] / aH[-1] - 1
    assert 0.06 < full < 0.07 and -0.02 < jac < -0.015, (full, jac)
    span = r["v"][-1] - r["v"][0]  # the first to the last pixel centre: (n_pix - 1) / n_pix of the box
    assert res["None"] == pytest.approx(full * span / r["period"], rel=0.05)
    assert res["a_eff"] == pytest.approx(jac * span / r["period"], rel=0.05)
    assert phys.c_kms * np.log(a[0] / a[-1]) == pytest.approx(r["period"], rel=1e-12)


def test_linear_hubble_flow_flattens_a_uniform_gas():
    """``hubble_flow="linear"`` with ``a_eff = 1/(1+z)`` and ``detuning="linear"`` (fake_spectra's model of one snapshot):
    a uniform gas in the toy boxes gives a flat tau(v), apart from the wrap pixel; the period is the frame's aH at y = 0
    times L, = ``ToyBoxes.aH`` L (``core.velocity_span``). With "exact" the Jacobian 1/(aH) remains (-1.8 %,
    ``test_a_eff_and_the_jacobian_of_the_exact_frame``)."""
    from skewerjax.core import velocity_span

    boxes = ToyBoxes(box_cMpc=150.0, n_cells=16, n_real=1, bao_ratio=0.0)
    uniform = _uniform(boxes)
    flat = {}
    for hf in ("exact", "linear"):
        r = toy_tau(boxes, 0, [[0, 0]], 128, 64, fields=uniform, a_eff=1 / (1 + boxes.z), detuning="linear", hubble_flow=hf)
        tau = r["tau"][0, 0]  # Lya
        inner = tau[1:-1]  # without the pixels next to the wrap (y = 0 = L, the split cell)
        flat[hf] = float(np.max(np.abs(inner / inner.mean() - 1)))
        if hf == "linear":
            assert r["period"] == pytest.approx(boxes.aH * boxes.box_cMpc, rel=1e-6)
            assert float(velocity_span(r["bundle"], "linear")[0]) == pytest.approx(r["period"], rel=1e-12)
    assert flat["linear"] < tol(1e-8, 1e-4) and flat["exact"] > 5e-3, (
        flat
    )  # float32: 2.4e-5  # 5.7e-10 and 8.6e-3 (the Jacobian 1/(aH))


def test_continuous_correlation():
    """``PowerLawGRF.correlation``, the closed form of the continuous xi(r) (xi_pl = A_s k_0^(1 - n_s) Gamma(1 + n_s)
    sin(pi (1 + n_s) / 2) / (2 pi^2 r^(n_s + 2)) and the bump with its mirror), against the numerical Hankel transform
    of the full spectrum P_pl + A_bao W (``correlation_hankel``) across -2 < n_s < 0 (n_s = -1 the limit pi / 2);
    outside that range, one line and the transform over the grid's k range."""
    from dataclasses import replace

    from scipy.special import gamma

    b = ToyBoxes(box_cMpc=260.0, n_cells=64, bao_ratio=1e-4, sigma_bao=5.0, r_bao=100.0)  # below R_max for every n_s here
    g = b.grf()
    r = np.linspace(0.5, 260.0, 300)
    A, k0, ns = g.A_s, g.k_0, g.n_s
    c = np.pi / 2 if ns == -1 else gamma(1 + ns) * np.sin(np.pi * (1 + ns) / 2)  # the limit pi / 2 at n_s = -1
    xi_pl = A * k0 ** (1 - ns) * c / (2 * np.pi**2 * r ** (ns + 2))
    Ab, r0, sg = g.A_bao, g.r_bao, g.sigma_bao
    bump = lambda x: np.exp(-(x**2) / (2 * sg**2))
    np.testing.assert_allclose(g.correlation(r), xi_pl + Ab * (bump(r - r0) + bump(r + r0)), rtol=1e-12)
    for n_s in (-1.9, -1.25, -1.0, -0.5, -0.05):
        gn = replace(b, n_s=n_s).grf()
        np.testing.assert_allclose(gn.correlation(r), gn.correlation_hankel(r, k_tail=400), rtol=1e-7)
    with pytest.raises(ValueError, match="r > 0"):
        g.correlation([0.0, 1.0])
    logs = []
    g_out = replace(b, n_s=0.5, bao_ratio=1e-5).grf()
    xi = g_out.correlation(r, log=logs.append)
    assert len(logs) == 1 and "-2 < n_s < 0" in logs[0] and np.all(np.isfinite(xi))
    np.testing.assert_allclose(xi, g_out.correlation_hankel(r, k_range=(2 * np.pi / 260.0, np.sqrt(3) * np.pi / b.dx)))
    with pytest.raises(ValueError, match="converges"):
        g_out.correlation_hankel(r)
