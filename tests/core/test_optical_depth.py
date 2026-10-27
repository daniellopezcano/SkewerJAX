"""The optical depth tau = sum over pieces (``core.optical_depth``; review notes, Eq. tau_straight_los).

* the one periodicity option, ``velocity_period`` (False / True / a number), and its two effects; the reference line
  frame as the shifted evaluation;
* the closed forms of uniform gas (review notes, Sec. analytic_no_expansion and radiation domination): static natural
  and thermal lines to 1e-6 in F, radiation domination with thermal lines converging as N^-2, and the resolution check
  that flags unresolved natural lines (formerly ``scripts/validation/check_optical_depth_analytic.py``; the figure:
  ``scripts/figures/optical_depth_vs_analytic.py``; the cases: ``analytic_cases.py``);
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
