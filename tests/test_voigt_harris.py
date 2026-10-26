"""Accuracy of the first-order Voigt expansion (``voigt="harris"``, fake_spectra's fastvoigt.h; a matched fast mode)
against the exact evaluations (scipy's wofz; the default ExoJAX mode): the expansion, its table, the fallback above a_max,
shapes, jit / vmap / grad, and tau through ``compute_optical_depth``."""
import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import wofz

from skewerjax.core import LineBundle, Pieces, compute_optical_depth, phys
from skewerjax.core import voigt as V


def _pieces(lb, a, length_cMpc, T, u_away_kms, n_com, velocity_span_kms):
    """``Pieces`` of one skewer from (N_p,) arrays (a_dagger = 0.25; the photon travels towards the observer)."""
    rows = lambda x: jnp.broadcast_to(jnp.asarray(x, dtype=float)[None, None], (lb.n_lines, 1, np.size(x)))
    one = lambda x: jnp.asarray(x, dtype=float)[None]
    z = np.zeros(np.size(a))
    return Pieces(a_dagger=one(np.full(np.size(a), 0.25)), scale_factor=one(a), length_m=one(np.asarray(length_cMpc) * phys.Mpc_to_m),
                  temperature=rows(T), turbulence=rows(z), collisional=rows(z), los_velocity=rows(-np.asarray(u_away_kms)),
                  number_density=rows(n_com), width_kms=rows(z), velocity_span_kms=jnp.asarray([float(velocity_span_kms)]))


U = np.linspace(0.0, 60.0, 120001)


def H_exact(u, a):
    return np.real(wofz(u + 1j * a))


def test_expansion_accuracy():
    """|H_harris / H - 1| over u in [0, 60] for a <= 1e-3: the a^2 term in the core, the 5-term series at u = 6."""
    for a, tol in ((1e-5, 1e-5), (1e-4, 1e-5), (5e-4, 1e-5), (1e-3, 2e-5)):
        r = np.abs(np.asarray(V.harris_H(jnp.asarray(U), a)) / H_exact(U, a) - 1)
        assert r.max() < tol, (a, r.max())
        core = U < 3
        assert r[core].max() < max(20 * a * a, 1e-6), (a, r[core].max())    # the a^2 term, or the table interpolation


def test_matches_fake_spectra_table():
    """The same table as fastvoigt.h: H1(u_i) = 2 u_i Im w(u_i) - 2/sqrt(pi) at u_i = i 6/1024."""
    u = V._HARRIS_U
    h1_ref = 2 * u * np.imag(wofz(u)) - 2 / np.sqrt(np.pi)
    assert np.abs(V._HARRIS_H1 - h1_ref).max() < 1e-14


def test_fallback_and_shapes():
    """a > a_max takes the exact path; x (N, N_v) with sigma, gamma (N, 1); unit area; scalar inputs."""
    x = jnp.linspace(-40.0, 40.0, 8001)
    sig = jnp.array([1.0, 1.0, 0.05, 1e-3])[:, None]
    gam = jnp.array([1e-4, 5e-3, 1.0, 1.0])[:, None]               # a = 7e-5 (expansion), 3.5e-3, 14, 707 (exact)
    Vh = np.asarray(V.voigt_harris(x[None], sig, gam))
    ref = np.stack([V.voigt_scipy(np.asarray(x), float(s), float(g)) for s, g in zip(sig[:, 0], gam[:, 0])])
    assert Vh.shape == (4, 8001)
    assert np.abs(Vh / ref - 1).max() < 2e-5
    assert np.abs(Vh[1:] - np.stack([np.asarray(V.voigt_exojax(x, float(s), float(g))) for s, g in zip(sig[1:, 0], gam[1:, 0])])).max() == 0
    area = np.asarray(V.voigt_harris(jnp.linspace(-2e3, 2e3, 400001), 1.0, 1e-4)).sum() * 0.01
    assert abs(area - 1) < 2e-3                                    # the Lorentzian tails beyond 2000 sigma: 2 a / (pi 2000) ...
    assert np.ndim(V.voigt_harris(0.3, 1.0, 1e-4)) == 0


def test_jit_vmap_grad():
    f = lambda x, s, g: V.voigt_harris(x, s, g)
    x = jnp.linspace(-8.0, 8.0, 1001)
    eager = np.asarray(f(x, 1.0, 5e-4))
    assert np.allclose(np.asarray(jax.jit(f)(x, 1.0, 5e-4)), eager, rtol=1e-13, atol=0)        # XLA fusion: last-bit rounding only
    vm = np.asarray(jax.vmap(f, in_axes=(None, 0, 0))(x, jnp.array([1.0, 2.0]), jnp.array([5e-4, 5e-3])))
    assert np.abs(vm[0] - eager).max() == 0 and np.abs(vm[1] / V.voigt_scipy(np.asarray(x), 2.0, 5e-3) - 1).max() < 1e-5
    for argnum, h in ((1, 1e-6), (2, 1e-9)):                       # d/dsigma, d/dgamma against central differences
        g = jax.grad(lambda *p: jnp.sum(f(*p)), argnums=argnum)(x, 1.0, 5e-4)
        p0 = [x, 1.0, 5e-4]
        lo, hi = list(p0), list(p0)
        lo[argnum] -= h; hi[argnum] += h
        fd = (jnp.sum(f(*hi)) - jnp.sum(f(*lo))) / (2 * h)
        assert abs(float(g) / float(fd) - 1) < 1e-4, (argnum, float(g), float(fd))
    gx = jax.grad(lambda xx: f(xx, 1.0, 5e-4))(0.7)
    assert np.isfinite(float(gx))


def test_line_profiles_and_tau():
    """The mode through voigt_line_profiles and compute_optical_depth: tau within 1e-5 (relative) of the default."""
    lb = LineBundle(["HI-LyA", "SiIII-1206"], ref_line_id="HI-LyA")
    rng = np.random.default_rng(1)
    n = 200
    pc = _pieces(lb, np.full(n, 0.25), np.full(n, 0.02), rng.uniform(1e3, 1e5, n), rng.normal(0, 50, n), 10 ** rng.uniform(-13, -9, n),
                 400.0)
    v = np.linspace(-500.0, 900.0, 2801)
    t_ex = np.asarray(compute_optical_depth(pc, v, lb, resolution_tolerance=None))
    t_h = np.asarray(compute_optical_depth(pc, v, lb, voigt="harris", resolution_tolerance=None))
    live = t_ex > 1e-12
    assert np.abs(t_h[live] / t_ex[live] - 1).max() < 3e-5
