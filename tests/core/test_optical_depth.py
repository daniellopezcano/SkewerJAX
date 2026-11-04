"""The optical depth tau = sum over pieces (``core.optical_depth``; review notes, Eq. tau_straight_los).

* the one periodicity option, ``velocity_period`` (False / True / a number), and its two effects; the reference line
  frame as the shifted evaluation;
* the closed forms of uniform gas (review notes, Sec. analytic_no_expansion and radiation domination): static natural
  and thermal lines to 1e-6 in F, radiation domination with thermal lines converging as N^-2, and the resolution check
  that flags unresolved natural lines (formerly ``scripts/validation/check_optical_depth_analytic.py``; the figure:
  ``scripts/figures/optical_depth_vs_analytic.py``; the cases: ``analytic_cases.py``);
* the effective scale factor ``a_eff`` of the density conversion (None: a_p per piece; traced, differentiable);
* the opt-in float32 kernel, the sharded tau (one and four CPU devices, bit for bit), jit of the pieces path;
* on a GPU (``gpu`` marker): the same tau as on the CPU.
"""

import os
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from skewerjax.core import (
    LineBundle,
    Pieces,
    cell_pieces,
    cells_from_values,
    compute_optical_depth,
    optical_depth_from_pieces,
    optical_depth_sharded,
    phys,
)

from analytic_cases import radiation_natural, radiation_thermal, static_natural, static_thermal
from helpers import ramp_pieces
from helpers import uniform_skewer as uniform

LYA = LineBundle(["HI-LyA"], ref_line_id="HI-LyA")


# --------------------------------------------------------------------------- periodicity and frames
def test_velocity_period_options():
    sb, f, _, _ = uniform()
    p = cell_pieces(f, LYA, skewer_bundle=sb, a_dagger=0.302)
    P = float(p.velocity_span_kms[0])
    v = (np.arange(800) + 0.5) * P / 800
    t_no = np.asarray(compute_optical_depth(p, v, LYA))
    t_T = np.asarray(compute_optical_depth(p, v, LYA, velocity_period=True))
    t_P = np.asarray(compute_optical_depth(p, v, LYA, velocity_period=P))
    assert np.array_equal(t_T, t_P), "True = the skewer's own span"
    assert np.array_equal(t_no, np.asarray(compute_optical_depth(p, v, LYA, velocity_period=False)))
    assert t_T[0, 0, 0] > t_no[0, 0, 0] and t_T[0, 0, -1] > t_no[0, 0, -1], "the wings re-enter at both ends"
    with pytest.raises(ValueError):
        compute_optical_depth(p, v, LYA, velocity_period=-1.0)


def one_piece(v_c, P, log_N=17.0):
    """One static piece whose line centre is v_c (a = a_dagger, the velocity sets v_c)."""
    a, L_m = 0.3, 1e-3 * phys.Mpc_to_m
    n_com = 10**log_N / (1e-3 * 3.085678e24 * a) * a**3
    u = -phys.c_kms * np.tanh(v_c / phys.c_kms)
    one = lambda x: jnp.full((1, 1, 1), x)
    return Pieces(
        a_dagger=jnp.full((1, 1), a),
        scale_factor=jnp.full((1, 1), a),
        length_m=jnp.full((1, 1), L_m),
        temperature=one(1e4),
        turbulence=one(0.0),
        collisional=one(0.0),
        los_velocity=one(u),
        number_density=one(n_com),
        width_kms=one(0.0),
        velocity_span_kms=jnp.full(1, P),
    )


def test_the_two_periodic_effects():
    P = 1000.0
    v = (np.arange(2000) + 0.5) * P / 2000
    # (i) a line centre outside the period absorbs at v_c - P
    t = np.asarray(compute_optical_depth(one_piece(P + 40.0, P), v, LYA, velocity_period=True))[0, 0]
    assert abs(v[np.argmax(t)] - 40.0) < 1.0
    assert np.asarray(compute_optical_depth(one_piece(P + 40.0, P), v, LYA))[0, 0].max() < 1e-3 * t.max(), "no wrap: off the grid"
    # (ii) a line centre inside, at the edge: its wing re-enters at small v
    t2 = np.asarray(compute_optical_depth(one_piece(P - 30.0, P), v, LYA, velocity_period=True))[0, 0]
    t2n = np.asarray(compute_optical_depth(one_piece(P - 30.0, P), v, LYA))[0, 0]
    assert t2[v < 20].max() > 1e3 * t2n[v < 20].max()


def test_reference_frame_is_the_shifted_evaluation():
    two = LineBundle(["HI-LyA", "SiIII-1206"], ref_line_id="HI-LyA")
    sb, f1, _, _ = uniform()
    f = f1.merge_species(
        [
            f1,
            cells_from_values(
                f1.positions, None, {"SiIII": f1.species["HI"]["number_density"]}, temperature=1e4, volumes=f1.volumes
            ),
        ]
    )
    v = jnp.linspace(-3000.0, 1000.0, 801)
    t_ref = np.asarray(compute_optical_depth(sb, v, two, f, a_dagger=0.302, line_frame="reference", resolution_tolerance=None))
    t_si = np.asarray(compute_optical_depth(sb, v + float(two.delta_v[1]), two, f, a_dagger=0.302, resolution_tolerance=None))[1]
    t_ly = np.asarray(compute_optical_depth(sb, v, two, f, a_dagger=0.302, resolution_tolerance=None))[0]
    assert np.allclose(t_ref[1], t_si, rtol=1e-12, atol=0), "line j on the reference axis = line j at v + delta_v_j"
    assert np.array_equal(t_ref[0], t_ly), "the reference line is unchanged"


def test_reference_frame_wraps_a_shift_beyond_the_period():
    """``line_frame="reference"`` with ``velocity_period``: every line is evaluated at v + delta_v_j and the detuning
    takes its nearest periodic image, so a line whose offset exceeds the period (Lya-Si II 1193 and 1190, 5571 and 6294
    km/s, against a period of 1000 km/s here, 3962 km/s in TNG50-4) lands at delta_v_j mod P: delta_v_j and delta_v_j + n
    P give the same tau (to rounding), for every line of the default bundle (``simulations.LINE_IDS``)."""
    from skewerjax.simulations import LINE_IDS

    lb = LineBundle(list(LINE_IDS), ref_line_id=LINE_IDS[0])
    P = 1000.0
    one = one_piece(40.0, P)
    tile = lambda x: jnp.repeat(x, lb.n_lines, axis=0)  # the same piece for every line
    pieces = Pieces(
        **{
            k: (
                tile(getattr(one, k))
                if k in ("temperature", "turbulence", "collisional", "los_velocity", "number_density", "width_kms")
                else getattr(one, k)
            )
            for k in (
                "a_dagger",
                "scale_factor",
                "length_m",
                "temperature",
                "turbulence",
                "collisional",
                "los_velocity",
                "number_density",
                "width_kms",
                "velocity_span_kms",
            )
        }
    )
    v = (np.arange(2000) + 0.5) * P / 2000
    dv = np.asarray(lb.delta_v)
    assert np.all(np.abs(dv[2:]) > P), "the Si II offsets exceed the period"
    ref = np.asarray(
        compute_optical_depth(pieces, v, lb, line_frame="reference", velocity_period=True, resolution_tolerance=None)
    )
    for n in (-3, 0, 2, 7):  # each line on its own axis, shifted by delta_v_j + n P
        grids = jnp.asarray(v[None, :] + dv[:, None] + n * P)
        own = np.asarray(compute_optical_depth(pieces, grids, lb, velocity_period=True, resolution_tolerance=None))
        np.testing.assert_allclose(own, ref, rtol=1e-9, atol=1e-12 * ref.max())
    for j in range(lb.n_lines):  # the centre at v_c - delta_v_j folded into the period
        want = np.mod(40.0 - dv[j], P)
        assert abs(v[np.argmax(ref[j, 0])] - want) < 1.0, (j, want)


# --------------------------------------------------------------------------- closed forms of uniform gas
def _max_dF(tau, ref):
    return float(np.max(np.abs(np.exp(-np.asarray(tau)) - np.exp(-np.asarray(ref)))))


def test_static_gas_matches_the_closed_forms():
    for case in (static_natural, static_thermal):
        _, tau, ref = case()
        assert _max_dF(tau, ref) <= 1e-6, case.__name__  # over 3 lines


def test_radiation_domination_natural_lines_are_flagged():
    *_, rep = radiation_natural()
    assert float(np.min(rep["max_ratio"])) > 0.25  # the resolution check flags them


def _thermal_errors(points):
    return np.array(
        [
            np.max(np.abs(np.exp(-np.asarray(t)) - np.exp(-np.asarray(r))), axis=(1, 2))
            for _, t, r in (radiation_thermal(n) for n in points)
        ]
    )  # (n_resolutions, n_lines)


def test_radiation_domination_thermal_converges():
    """Second order: the error falls ~N^-2 when the points double (from 84 points on)."""
    errs = _thermal_errors((84, 168))
    assert float(np.min(errs[:-1] / errs[1:])) > 2.5, errs
    assert float(errs[-1].max()) <= 1e-1, errs


@pytest.mark.slow
def test_radiation_domination_thermal_converges_full():
    """The full scan (42 to 336 points): from 84 points every step is in the N^-2 regime (42 points is pre-asymptotic
    for the hot line), and 336 points reach max |dF| <= 1e-2."""
    errs = _thermal_errors((42, 84, 168, 336))[1:]
    assert float(np.min(errs[:-1] / errs[1:])) > 2.5, errs
    assert float(errs[-1].max()) <= 1e-2, errs


# --------------------------------------------------------------------------- the density conversion a_eff
def test_a_eff_density_conversion():
    """``a_eff``: None is the per-piece a_p^-2 (the same tau as without the option, bit for bit); a scalar is a_eff^-2
    for every piece, i.e. the densities rescaled by (a_p / a_eff)^2 with the line centres, widths and detuning untouched;
    traced in the kernel (no recompilation per value) and differentiable, d tau / d a_eff = -2 tau / a_eff; a value that
    is not a finite positive scalar raises, a value > 1 warns."""
    import warnings
    from dataclasses import replace

    smp, v, lb = ramp_pieces()
    t0 = np.asarray(compute_optical_depth(smp, v, lb, resolution_tolerance=None))
    assert np.array_equal(np.asarray(compute_optical_depth(smp, v, lb, a_eff=None, resolution_tolerance=None)), t0)
    a_eff = 0.27
    scaled = replace(smp, number_density=smp.number_density * (smp.scale_factor[None] / a_eff) ** 2)
    t_ref = np.asarray(compute_optical_depth(scaled, v, lb, resolution_tolerance=None))
    t_a = np.asarray(compute_optical_depth(smp, v, lb, a_eff=a_eff, resolution_tolerance=None))
    np.testing.assert_allclose(t_a, t_ref, rtol=1e-12, atol=1e-14 * t_ref.max())
    assert np.max(np.abs(t_a - t0)) > 1e-3 * t0.max(), "the a_p of the pieces differ from a_eff"
    n0 = optical_depth_from_pieces._cache_size()
    for a in (0.25, 0.3, 0.35):
        optical_depth_from_pieces(smp, v, lb, a_eff=a)
    assert optical_depth_from_pieces._cache_size() - n0 <= 1, "a_eff is traced: one compilation for every value"
    g = jax.grad(lambda a: jnp.sum(optical_depth_from_pieces(smp, v, lb, a_eff=a)))(a_eff)
    np.testing.assert_allclose(float(g), -2.0 * t_a.sum() / a_eff, rtol=1e-10)
    for bad in (-1.0, np.inf, np.ones(2)):
        with pytest.raises(ValueError, match="a_eff"):
            compute_optical_depth(smp, v, lb, a_eff=bad, resolution_tolerance=None)
    with warnings.catch_warnings(record=True) as caught:  # 1 + z typed for 1 / (1 + z): a warning, not an error
        warnings.simplefilter("always")
        t4 = np.asarray(compute_optical_depth(smp, v, lb, a_eff=4.0, resolution_tolerance=None))
    assert any("a_eff = 4 > 1" in str(w.message) for w in caught)
    np.testing.assert_allclose(t4, t_a * (a_eff / 4.0) ** 2, rtol=1e-12)


# --------------------------------------------------------------------------- precision, sharding, devices
def test_float32_is_opt_in_and_close():
    smp, v, lb = ramp_pieces()
    t64 = np.asarray(optical_depth_from_pieces(smp, v, lb))
    t32 = np.asarray(optical_depth_from_pieces(smp, v, lb, dtype="float32"))
    assert t32.dtype == t64.dtype == np.float64  # cast back to the input precision
    assert np.max(np.abs(np.exp(-t32) - np.exp(-t64))) < 1e-4


def test_sharded_one_device_is_identical():
    smp, v, lb = ramp_pieces()
    assert np.array_equal(np.asarray(optical_depth_sharded(smp, v, lb)), np.asarray(optical_depth_from_pieces(smp, v, lb)))


SCRIPT = """
import os; os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"; os.environ["JAX_PLATFORMS"] = "cpu"
import sys; sys.path.insert(0, {tests!r})
import numpy as np, jax; jax.config.update("jax_enable_x64", True)
from helpers import ramp_pieces
from skewerjax.core import optical_depth_sharded, optical_depth_from_pieces
assert len(jax.devices()) == 4
smp, v, lb = ramp_pieces(7)                                          # 7 skewers on 4 devices: padded
assert np.array_equal(np.asarray(optical_depth_sharded(smp, v, lb)), np.asarray(optical_depth_from_pieces(smp, v, lb)))
P = np.linspace(300.0, 360.0, 7)                                     # one period per skewer, sharded with the skewers
assert np.array_equal(np.asarray(optical_depth_sharded(smp, v, lb, velocity_period=P)),
                      np.asarray(optical_depth_from_pieces(smp, v, lb, velocity_period=P)))
assert np.array_equal(np.asarray(optical_depth_sharded(smp, v, lb, a_eff=0.3)),        # a_eff replicated on every device
                      np.asarray(optical_depth_from_pieces(smp, v, lb, a_eff=0.3)))
"""


def test_sharded_four_cpu_devices_is_identical():
    """In a separate process: the device count is fixed when JAX starts."""
    tests = str(Path(__file__).resolve().parents[1])
    p = subprocess.run(
        [sys.executable, "-c", SCRIPT.format(tests=tests)],
        capture_output=True,
        text=True,
        env={k: v for k, v in os.environ.items() if k != "XLA_FLAGS"},
    )
    assert p.returncode == 0, p.stderr[-2000:]


def test_jit_of_the_pieces_path():
    smp, v, lb = ramp_pieces(3)
    f = jax.jit(lambda s, vv: optical_depth_from_pieces(s, vv, lb))
    assert np.allclose(np.asarray(f(smp, v)), np.asarray(optical_depth_from_pieces(smp, v, lb)), rtol=1e-12, atol=0)


@pytest.mark.gpu
def test_gpu_matches_cpu():
    smp, v, lb = ramp_pieces(3)
    gpu, cpu = jax.devices("gpu")[0], jax.devices("cpu")[0]
    on = lambda dev: np.asarray(optical_depth_from_pieces(jax.device_put(smp, dev), jax.device_put(v, dev), lb))
    t_cpu, t_gpu = on(cpu), on(gpu)
    assert np.max(np.abs(t_gpu - t_cpu)) <= 1e-10 * np.max(t_cpu)


# --------------------------------------------------------------------------- the Hubble flow of the line centres
def test_hubble_flow_option():
    """``hubble_flow``: "exact" (default) is the current tau, bit for bit; "linear" puts the line centres at V_H(0) +
    (aH)_ref y (the frame's aH at the first point by default; ``Pieces.path_cMpc``, ``aH_first``, ``vH_first``), which
    departs from the exact c ln(a_dagger / a(y)) at second order only; aH_ref is traced (one compilation for every value)
    and differentiable; the period of ``velocity_period=True`` is |(aH)_ref| L and the widths of the resolution check
    |(aH)_ref| l_p, so the window fraction over one period stays 1; in a ``SnapshotFrame`` (its Hubble flow is in the
    injected velocity) the two options give the same tau."""
    import warnings
    from dataclasses import replace

    from skewerjax.core import SnapshotFrame, velocity_span

    sb, f, pos, a = uniform()
    p = cell_pieces(f, LYA, skewer_bundle=sb, a_dagger=0.302)
    P = float(p.velocity_span_kms[0])
    v = (np.arange(400) + 0.5) * P / 400
    t0 = np.asarray(compute_optical_depth(p, v, LYA, velocity_period=True))
    assert np.array_equal(t0, np.asarray(compute_optical_depth(p, v, LYA, velocity_period=True, hubble_flow="exact")))
    # the path fields: y from the first point, aH and V_H there (a_dagger = a at the last point here: V_H(0) != 0)
    s = np.asarray(p.path_cMpc)[0]
    assert s.min() >= 0 and float(p.path_length_cMpc[0]) == pytest.approx(pos[0, -1, 0] - pos[0, 0, 0], rel=1e-12)
    lna = np.log(a[0, 0])
    aH0 = -phys.c_kms * np.gradient(lna[:3], pos[0, :3, 0], edge_order=2)[0]  # second order, one-sided
    assert float(p.aH_first[0]) == pytest.approx(aH0, rel=1e-6) and float(p.vH_first[0]) == pytest.approx(
        phys.c_kms * np.log(0.302 / 0.30), rel=1e-9
    )
    # a regular skewer, a log-linear in y and a_dagger at the first point: the period is |aH| L, the window holds the
    # whole column, and the line centres differ from the exact ones at second order only
    from skewerjax.core import SkewerBundle, cells_from_values

    n_pt, L = 801, 2.0
    xq = np.zeros((1, n_pt, 3))
    xq[0, :, 0] = np.linspace(0.0, L, n_pt)
    aq = np.exp(np.linspace(np.log(0.302), np.log(0.300), n_pt))[None, None]
    sbq = SkewerBundle(spatial_points=jnp.asarray(xq), manual_scale_factors=jnp.asarray(aq), a0=0.302, ICs_defined_at_index=0)
    cq = cells_from_values(xq[0], None, {"HI": np.full(n_pt, 1e-12)}, temperature=1e4, volumes=np.full(n_pt, (L / n_pt) ** 3))
    q = cell_pieces(cq, LYA, skewer_bundle=sbq)
    aHq = phys.c_kms * np.log(0.302 / 0.300) / L
    assert float(q.aH_first[0]) == pytest.approx(aHq, rel=1e-9) and abs(float(q.vH_first[0])) < 1e-9
    P_lin = float(velocity_span(q, "linear")[0])
    assert P_lin == pytest.approx(aHq * L, rel=1e-9) and P_lin == pytest.approx(float(q.velocity_span_kms[0]), rel=1e-9)
    vq = (np.arange(400) + 0.5) * P_lin / 400
    te, info_e = compute_optical_depth(q, vq, LYA, velocity_period=True, return_info=True)
    tl, info_l = compute_optical_depth(q, vq, LYA, hubble_flow="linear", velocity_period=True, return_info=True)
    np.testing.assert_allclose(info_l["velocity_period"], P_lin, rtol=1e-12)
    np.testing.assert_allclose(info_l["window_fraction"], 1.0, rtol=1e-4)  # one period holds the whole column
    np.testing.assert_allclose(np.asarray(tl), np.asarray(te), rtol=1e-6), "a linear in y: the same line centres"
    # the resolution check uses |aH_ref| l_p: a large (aH)_ref widens the pieces
    for h, warned in ((aHq, False), (1e5 * aHq, True)):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            compute_optical_depth(q, vq, LYA, hubble_flow="linear", aH_ref=h, resolution_tolerance=0.25)
        assert any("too wide" in str(w.message) for w in caught) == warned
    aH0 = float(q.aH_first[0])
    p = q
    v = vq
    with pytest.raises(ValueError, match="aH_ref"):
        compute_optical_depth(p, v, LYA, aH_ref=aH0)  # exact has no aH_ref
    with pytest.raises(ValueError, match="hubble_flow"):
        compute_optical_depth(p, v, LYA, hubble_flow="quadratic")
    with pytest.raises(ValueError, match="path"):
        compute_optical_depth(replace(p, path_cMpc=None), v, LYA, hubble_flow="linear")
    # SnapshotFrame: one a, the Hubble flow in the velocity: linear == exact, bit for bit
    L = float(pos[0, -1, 0])
    fr = SnapshotFrame(a_snap=0.3, velfac=80.0, path_length=np.array([L]))
    ps = cell_pieces(f, LYA, skewer_bundle=sb, frame=fr)
    vs = (np.arange(300) + 0.5) * 80.0 * L / 300
    te = np.asarray(compute_optical_depth(ps, vs, LYA, velocity_period=True))
    tl = np.asarray(compute_optical_depth(ps, vs, LYA, velocity_period=True, hubble_flow="linear"))
    assert np.array_equal(te, tl) and float(velocity_span(ps, "linear")[0]) == pytest.approx(80.0 * L, rel=1e-12)
