"""Points on a periodic mesh (``summary_stats.mesh``; Sefusatti et al. 2016, "S16"): mass assignment, interlacing and
the window correction, in 1D, 2D and 3D.

a. mass conservation of every scheme, and the periodic wrap at the box edges;
b. points exactly on the nodes, NGP, no interlacing, no deconvolution: the FFT of the grid field, bit for bit;
c. the window of one deposited point, averaged over its sub-cell positions (a midpoint rule), equals ``mas_window``
   (S16 eq. 27);
d. a plane wave below k_Nyq sampled by random points: its amplitude recovered after the deconvolution (tolerance
   5e-3 for every scheme: the sampling noise of 65 536 jittered points; the deconvolution is exact in expectation);
e. a plane wave at 1.31 k_Nyq: without interlacing it aliases to k_s - k with the predicted amplitude W(k) / W(k_s - k),
   with interlacing that odd image cancels (< 1e-12);
f. the sign of the interlacing phase: + (numpy's forward FFT) matches the direct sum, the opposite sign fails;
g. jit (no recompilation for new values) and the gradient in the weights against finite differences;
plus ``aliased_power`` (S16 eqs. 29, 37) and ``fourier_grid``."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from skewerjax.summary_stats import mesh as M
from skewerjax.summary_stats.mesh import (
    MAS_ORDERS,
    MAS_SCHEMES,
    aliased_power,
    fourier_grid,
    fourier_mesh,
    mas_window,
    mesh_k_axes,
    paint,
)

from helpers import raises


@pytest.mark.parametrize("D", [1, 2, 3])
def test_mass_conservation_and_periodic_wrap(D):
    rng = np.random.default_rng(D)
    n, L = (8, 6, 4)[:D], (10.0, 7.5, 4.0)[:D]
    x = rng.uniform(-L[0], 2 * L[0], (400, D))  # many points outside the box
    w = rng.normal(size=400)
    for mas in MAS_SCHEMES:
        m = np.asarray(paint(x, w, n, L, mas))
        assert m.shape == n
        assert abs(m.sum() - w.sum()) <= 1e-13 * np.abs(w).sum(), mas
        shifted = np.asarray(paint(x + np.asarray(L) * rng.integers(-2, 3, (400, D)), w, n, L, mas))
        np.testing.assert_allclose(shifted, m, rtol=0, atol=1e-12 * np.abs(w).sum())  # the box is periodic
    # the box edges: a point just below L goes (CIC) to the first node; one at 0 (TSC) also to the last
    pos = np.zeros((1, D))
    pos[0, 0] = L[0] * (1 - 1e-9)
    edge = np.asarray(paint(pos, 1.0, n, L, "cic")).reshape(n[0], -1)
    assert edge[0].sum() > 0.999 and edge.sum() == pytest.approx(1.0, abs=1e-15)
    zero = np.asarray(paint(np.zeros((1, D)), 1.0, n, L, "tsc")).reshape(n[0], -1)
    assert zero[0].sum() == pytest.approx(0.75) and zero[-1].sum() == pytest.approx(0.125)
    raises(ValueError, "mas must be one of", paint, x, w, n, L, "sph")
    raises(ValueError, "n_mesh", paint, x, w, (8,) * (D + 1), 1.0, "cic")


@pytest.mark.parametrize("D", [1, 2, 3])
def test_ngp_on_the_nodes_is_the_fft_of_the_grid(D):
    rng = np.random.default_rng(10 + D)
    n, L = (16, 8, 12)[:D], (3.0, 2.0, 5.0)[:D]
    field = rng.normal(size=n)
    H = np.asarray(L) / np.asarray(n)
    nodes = np.stack(np.meshgrid(*[np.arange(m) * h for m, h in zip(n, H)], indexing="ij"), -1).reshape(-1, D)
    F, meta = fourier_mesh(nodes, field.ravel(), n, L, mas="ngp", interlace=False, deconvolve=False)
    G, gmeta = fourier_grid(field, L)
    assert np.array_equal(np.asarray(F), np.asarray(G)), "bit for bit"
    assert np.array_equal(np.asarray(paint(nodes, field.ravel(), n, L, "ngp")), field)
    assert meta["mas"] == "ngp" and gmeta["mas"] is None and not gmeta["interlace"]
    np.testing.assert_allclose(meta["k_nyq"], np.pi / H)


def _point_window(k, D, n, L, mas, m_sub):
    """The mean over m_sub^D sub-cell positions x of F_x(k) e^{i k.x}, F_x the rfft of one point at x (unit weight): by
    linearity, two paints with the weights cos(k.x) and sin(k.x)."""
    H = L / n
    g = (np.arange(m_sub) + 0.5) * H / m_sub + 2 * H
    x = np.stack(np.meshgrid(*[g] * D, indexing="ij"), -1).reshape(-1, D)
    idx = tuple(int(round(k_i * L / (2 * np.pi))) % n for k_i in k)
    ph = x @ np.asarray(k)
    Fc, _ = fourier_mesh(x, np.cos(ph), n, L, mas=mas, interlace=False, deconvolve=False)
    Fs, _ = fourier_mesh(x, np.sin(ph), n, L, mas=mas, interlace=False, deconvolve=False)
    return complex(Fc[idx] + 1j * Fs[idx]) / x.shape[0]


def test_measured_window_of_one_point():
    n, L = 16, 4.0
    kf = 2 * np.pi / L
    for mas in MAS_SCHEMES:
        for j in (1, 3, 6, 8):  # up to the Nyquist mode
            got = _point_window((j * kf,), 1, n, L, mas, 512)
            assert abs(got - float(mas_window(np.array([[j * kf]]), L / n, mas)[0])) < 2e-6, (mas, j)
        k = np.array([3, 5, 7]) * kf  # 3D: the separable product
        got = _point_window(k, 3, n, L, mas, 24)
        tol = 1e-3 if MAS_ORDERS[mas] <= 2 else 1e-6  # the midpoint rule over 24^3 positions: O(m^-2) for kinked kernels
        assert abs(got - float(mas_window(k[None], L / n, mas)[0])) < tol, mas
    axes = mesh_k_axes((8, 6), (2.0, 3.0))
    Wg = np.asarray(mas_window(axes, (0.25, 0.5), "pcs"))
    assert Wg.shape == (8, 4) and Wg[0, 0] == 1.0
    assert Wg[2, 3] == pytest.approx(float(mas_window(np.array([[float(axes[0][2]), float(axes[1][3])]]), (0.25, 0.5), "pcs")[0]))


@pytest.mark.parametrize("D", [2, 3])
def test_plane_wave_below_nyquist(D):
    rng = np.random.default_rng(20 + D)
    n, L, q = (32, 16)[D - 2], 10.0, (8, 4)[D - 2]
    g = np.arange(n * q) * L / (n * q)
    grid = np.stack(np.meshgrid(*[g] * D, indexing="ij"), -1).reshape(-1, D)
    x = grid + rng.uniform(0, L / (n * q), grid.shape)  # one random point in every sub-cell (jittered)
    kk = (11, 7, 3)[:D] if D == 2 else (5, 3, 2)
    k0 = 2 * np.pi * np.asarray(kk) / L
    w = np.cos(x @ k0 + 0.3)
    per_node = x.shape[0] / n**D
    for mas in MAS_SCHEMES:
        for interlace in (False, True):
            F, _ = fourier_mesh(x, w, n, L, mas=mas, interlace=interlace)
            amp = complex(F[kk]) / per_node / (n**D / 2) * np.exp(-0.3j)
            assert abs(amp - 1) < 5e-3, (mas, interlace, amp)
            if not interlace:
                raw, _ = fourier_mesh(x, w, n, L, mas=mas, interlace=False, deconvolve=False)
                Wk = float(mas_window(k0[None], L / n, mas)[0])
                assert abs(complex(raw[kk]) / per_node / (n**D / 2) * np.exp(-0.3j) - Wk) < 5e-3  # the window itself


def test_plane_wave_above_nyquist_aliases_and_interlacing_cancels_it():
    n, L, m = 64, 1.0, 42  # k = 42 k_f = 1.31 k_Nyq; its image k_s - k = 22 k_f
    x = (np.arange(64 * n) + 0.37) * L / (64 * n)  # a fine regular set of points (no sampling noise)
    w = np.cos(2 * np.pi * m * x / L)
    for mas in MAS_SCHEMES:
        p = MAS_ORDERS[mas]
        pred = (np.sinc(m / n) / np.sinc((n - m) / n)) ** p  # |w_1(k_s - k)| = W(k) / W(k_s - k)
        amp = {}
        for interlace in (False, True):
            F, _ = fourier_mesh(x, w, n, L, mas=mas, interlace=interlace)
            amp[interlace] = abs(complex(F[n - m])) / (x.size / n) / (n / 2)
        assert amp[False] == pytest.approx(pred, rel=2e-3), mas
        assert amp[True] < 1e-12, (mas, amp[True])


def test_interlacing_phase_sign():
    n, L = 32, 1.0
    x = (np.arange(4096) + 0.3) * L / 4096
    xs = jnp.asarray(x)[:, None]
    for j in (5, 12, 15):
        k = 2 * np.pi * j / L
        w = np.cos(k * x + 0.4)
        ws = jnp.asarray(w)
        direct = np.sum(w * np.exp(-1j * k * x)) / (x.size / n)  # the alias-free coefficient
        for mas in ("cic", "tsc", "pcs"):
            good = M._fourier_mesh(xs, ws, (n,), jnp.asarray([L]), mas, True, True, 1.0)
            bad = M._fourier_mesh(xs, ws, (n,), jnp.asarray([L]), mas, True, True, -1.0)
            assert abs(complex(good[j]) / (x.size / n) - direct) < 1e-4 * abs(direct), (mas, j)  # CIC: the 4096-point images
            assert abs(complex(bad[j]) / (x.size / n) - direct) > 0.4 * abs(direct), "a wrong sign must fail"
        F, _ = fourier_mesh(x, w, n, L, mas="tsc")
        np.testing.assert_allclose(complex(F[j]) / (x.size / n), direct, rtol=1e-6)


def test_jit_and_gradients():
    rng = np.random.default_rng(5)
    n, L = (8, 8, 8), 5.0
    x = rng.uniform(0, L, (300, 3))
    w = rng.normal(size=300)
    fourier_mesh(x, w, n, L, mas="tsc")
    sizes = (M._paint._cache_size(), M._fourier_mesh._cache_size())
    for s in range(3):  # new positions, weights and box side: no new compilation
        fourier_mesh(rng.uniform(0, L, (300, 3)), rng.normal(size=300), n, L + s, mas="tsc")
    assert (M._paint._cache_size(), M._fourier_mesh._cache_size()) == sizes

    def loss(weights, pos):
        F, _ = fourier_mesh(pos, weights, n, L, mas="pcs", interlace=True)
        return jnp.sum(jnp.abs(F[1:3, 1:3, 1:3]) ** 2)

    g = np.asarray(jax.grad(loss)(jnp.asarray(w), jnp.asarray(x)))
    for i in (0, 7, 123):
        e = np.zeros_like(w)
        e[i] = 1e-6
        fd = (float(loss(jnp.asarray(w + e), x)) - float(loss(jnp.asarray(w - e), x))) / 2e-6
        assert g[i] == pytest.approx(fd, rel=1e-6, abs=1e-9)
    gx = np.asarray(jax.grad(loss, argnums=1)(jnp.asarray(w), jnp.asarray(x)))  # the positions: PCS is C^2
    e = np.zeros_like(x)
    e[4, 1] = 1e-6
    fd = (float(loss(w, x + e)) - float(loss(w, x - e))) / 2e-6
    assert gx[4, 1] == pytest.approx(fd, rel=1e-5, abs=1e-8)


def test_aliased_power():
    """``aliased_power`` = sum_n theta_n |W(k - n k_s) / W(k)|^2 P(|k - n k_s|): 1 at low k for a steep spectrum, the
    interlaced sum closer to P than the plain one, and a white spectrum folded exactly like the window."""
    n, L = 16, 8.0
    steep = lambda k: k**-3.0  # noqa: E731
    for mas in ("cic", "pcs"):
        Pa = np.asarray(aliased_power(steep, (n,) * 3, L, mas=mas, interlace=False))
        Pi = np.asarray(aliased_power(steep, (n,) * 3, L, mas=mas, interlace=True))
        kv = M._outer(mesh_k_axes((n,) * 3, L))
        kmod = np.sqrt(sum(np.asarray(c) ** 2 for c in kv))
        live = kmod > 0
        r_a, r_i = Pa[live] / steep(kmod[live]), Pi[live] / steep(kmod[live])
        assert np.all(r_a >= 1 - 1e-12) and np.all(r_i >= 1 - 1e-12), "images only add power"
        assert np.all(r_i <= r_a + 1e-12) and r_i.max() < r_a.max()
        low = kmod[live] < 0.25 * np.pi * n / L
        assert np.max(r_a[low] - 1) < 2e-3 and np.max(r_i[low] - 1) < 1e-5
    # 1D white noise: sum_n |W(k - n k_s)|^2 / |W(k)|^2 with NGP = 1 / sinc^2 (C09, footnote to eq. 28)
    P1 = np.asarray(aliased_power(lambda k: jnp.ones_like(k), 32, 1.0, mas="ngp", interlace=False, n_images=2000))
    k = np.asarray(mesh_k_axes(32, 1.0, 1)[0])
    np.testing.assert_allclose(P1[1:], 1 / np.sinc(k[1:] / (2 * np.pi * 32)) ** 2, rtol=1e-3)
