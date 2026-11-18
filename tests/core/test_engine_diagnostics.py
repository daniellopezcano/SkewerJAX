"""The read-only diagnostics of the engine (notebooks 07 and 15) are what the engine computes: ``window_plan`` holds the
windows ``windowed_line_tau`` uses, ``far_wing_term`` is the direct sum of the far series, the evaluation counts follow
from the plan, and ``per_cell_line_tau``'s info leaves its result unchanged."""

import jax.numpy as jnp
import numpy as np

from skewerjax.core import tau_engine as te
from skewerjax.core.lines import LineBundle

from helpers import float64_only

LINES = LineBundle(["HI-LyA", "SiIII-1206"])


def _contributions(j, n_sk, n_v, rng, n_per=80, width=True):
    lam, A21, mass = float(LINES.lambda0_A[j]), float(LINES.A21_GHz[j]), float(LINES.mass_amu[j])
    n = n_sk * n_per
    T = 10 ** rng.uniform(3.8, 5.0, n)
    sig, gam = te.line_widths(T, np.zeros(n), np.zeros(n), lam, A21, mass)
    tau0 = 10 ** rng.uniform(-3, 1.0, n)
    c = te.LineContributions(
        np.repeat(np.arange(n_sk), n_per),
        rng.uniform(0.1 * n_v, 0.9 * n_v, n),
        rng.uniform(0, 8, n) if width else np.zeros(n),
        tau0 * np.sqrt(2 * np.pi) * sig,
        sig,
        gam,
        None,
        rng.normal(0, 1e-5, n),
    )
    return c, lam


@float64_only
def test_plan_far_term_and_counts_are_the_engines():
    rng = np.random.default_rng(7)
    n_sk, n_v = 3, 900
    v = (np.arange(n_v) + 0.5) * 1.0
    c, lam = _contributions(0, n_sk, n_v, rng)
    plan = te.window_plan(c, v, n_sk, lam, small_problem=0)
    tau, info = te.windowed_line_tau(c, v, n_sk, lam, small_problem=0, return_info=True)
    full = np.asarray(te.windowed_line_tau(c, v, n_sk, lam))  # a small problem: every contribution on every pixel
    assert not plan["small"] and te.window_plan(c, v, n_sk, lam)["small"]
    win = plan["win"]
    assert {int(D): int(n) for D, n in zip(*np.unique(plan["D"][win], return_counts=True))} == info["levels"]
    assert plan["D_x"] == info["D_x"] and int(plan["full"].sum()) == info["n_full"]
    assert info["evaluations"] == int(np.sum(2 * plan["D"][win] + 1) + plan["full"].sum() * n_v)
    assert np.max(np.abs(np.asarray(tau) - full)) < 1e-8  # the windows and the FFT: exact to the tolerance
    far = te.far_wing_term(plan, n_sk)
    k = np.arange(-2 * n_v, 2 * n_v + 1)
    kern = np.stack([np.asarray(x, np.float64) for x in te.far_kernels(jnp.asarray(k * 1.0), "exact")])
    kern[:, np.abs(k) <= plan["D_x"]] = 0.0
    for s in range(n_sk):
        w = win[plan["c"].row[win] == s]
        direct = sum(plan["m"][q][w] @ kern[q][np.arange(n_v)[None, :] - plan["cpix"][w][:, None] + 2 * n_v] for q in range(4))
        assert np.max(np.abs(far[s] - direct)) < 1e-9 * np.abs(direct).max()


def test_per_cell_info_leaves_the_result_unchanged():
    rng = np.random.default_rng(3)
    n_sk, n_v = 2, 600
    v = (np.arange(n_v) + 0.5) * 1.0
    c, lam = _contributions(0, n_sk, n_v, rng, n_per=30, width=False)
    P = np.full(n_sk, n_v * 1.0)
    kw = dict(period=P, profile="midpoint", detuning="linear", tautail=1e-7, pixel_average=True)
    a = np.asarray(te.per_cell_line_tau(c, v, n_sk, lam, **kw))
    b, info = te.per_cell_line_tau(c, v, n_sk, lam, return_info=True, **kw)
    assert np.array_equal(a, np.asarray(b))
    n_cells = len(c)
    assert all(np.size(info[k]) == n_cells for k in ("row", "centre", "level_start", "level_end", "half_width", "core"))
    assert np.all(info["level_end"] >= info["level_start"]) and info["evaluations"] > 0 and info["walk_evaluations"] > 0
    whole = np.asarray(te.per_cell_line_tau(c, v, n_sk, lam, levels=(n_v,), **kw))  # one window: the whole grid
    assert np.max(np.abs(a - whole)) <= 1e-6 * np.abs(whole).max()
