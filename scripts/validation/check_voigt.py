#!/usr/bin/env python
"""The Voigt function: ExoJAX and FFT-convolution implementations against scipy's Faddeeva ``wofz``, unit area, the
Gaussian and Lorentzian limits, and finite gradients (notes ``sec_flux.tex``, Eq. line_profile)."""
from _common import Checks, figure_backend

C = Checks("check_voigt", __doc__)
import numpy as np                                                        # noqa: E402
import jax                                                                # noqa: E402
import jax.numpy as jnp                                                   # noqa: E402
from skewerjax.core import voigt as V                                     # noqa: E402

x = jnp.linspace(-60.0, 60.0, 1201 if C.quick else 4801)
pairs = [(1.0, 0.01), (1.0, 1.0), (3.0, 0.5), (0.5, 3.0)] if C.quick else \
    [(s, g) for s in (0.3, 1.0, 3.0) for g in (0.003, 0.1, 1.0, 3.0)]
worst = {"ExoJAX": 0.0, "convolution": 0.0, "unresolved": 0.0}
dx_fine = float(x[1] - x[0]) / 4
xn = np.asarray(x)
for s, g in pairs:
    ref = V.voigt_scipy(xn, np.asarray(s), np.asarray(g))
    core = ref > 1e-3 * ref.max()                                          # the line, not the far wings
    got = np.asarray(V.voigt_exojax(x, jnp.asarray(s), jnp.asarray(g)))
    worst["ExoJAX"] = max(worst["ExoJAX"], float(np.max(np.abs(got[core] / ref[core] - 1))))
    if g <= 1.0:            # the FFT convolution truncates the Lorentzian at the grid ends: compare |x| < 10 only
        got = np.asarray(V.voigt_convolution(x, jnp.asarray(s), jnp.asarray(g), oversample=4))
        mid = core & (np.abs(xn) < 10)
        key = "convolution" if g >= 2 * dx_fine else "unresolved"         # it needs a grid that resolves gamma
        worst[key] = max(worst[key], float(np.max(np.abs(got[mid] / ref[mid] - 1))))
C.check(f"ExoJAX vs wofz, {len(pairs)} (sigma, gamma), where V > 1e-3 V_max: max relative error", worst["ExoJAX"], 1e-4)
C.check("convolution (oversample 4) vs wofz, gamma resolved (>= 2 fine cells), |x| < 10: max relative error", worst["convolution"], 1e-10)
C.check("convolution with gamma below 2 fine cells is inaccurate (the documented resolution requirement)", worst["unresolved"], 1e-3,
        expect=True)
dx = float(x[1] - x[0])
area = float(np.sum(np.asarray(V.voigt_exojax(x, jnp.asarray(1.0), jnp.asarray(1e-7)))) * dx)
C.check("unit area of a Doppler-dominated profile, gamma = 1e-7 sigma (|sum V dx - 1|)", abs(area - 1), 1e-6)
xc = x[np.abs(xn) < 6.0]                                                  # within 3 sigma: the Lorentzian wings are negligible
C.check("gamma -> 0: Voigt -> Gaussian (max |V/G - 1|, |x| < 3 sigma)",
        np.max(np.abs(np.asarray(V.voigt_exojax(xc, jnp.asarray(2.0), jnp.asarray(1e-8))) / np.asarray(V.gaussian(xc, 2.0)) - 1)), 1e-4)
C.check("sigma -> 0: Voigt -> Lorentzian (max |V/L - 1|, |x| > 1)",
        np.max(np.abs(np.asarray(V.voigt_exojax(x, jnp.asarray(1e-4), jnp.asarray(1.0)))[np.abs(np.asarray(x)) > 1]
                      / np.asarray(V.lorentzian(x, 1.0))[np.abs(np.asarray(x)) > 1] - 1)), 1e-3)
grad = jax.grad(lambda s, g: jnp.sum(V.voigt_exojax(x, s, g)), argnums=(0, 1))(jnp.asarray(2.0), jnp.asarray(0.5))
C.check("d/dsigma, d/dgamma of sum V are finite (non-finite count)", float(sum(not np.isfinite(float(v)) for v in grad)), 0)
C.check("log10_voigt ExoJAX == log10 voigt_exojax", np.max(np.abs(np.asarray(V.log10_voigt(x, 2.0, 0.5))
                                                                  - np.log10(np.asarray(V.voigt_exojax(x, 2.0, 0.5))))), 1e-12)

if C.outdir is not None:
    figure_backend()
    import matplotlib.pyplot as plt
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}, layout="constrained")
    ref = V.voigt_scipy(np.asarray(x), np.asarray(1.0), np.asarray(1.0))
    ax.semilogy(x, ref, color="0.7", lw=5, label="wofz")
    for m, f in (("ExoJAX", V.voigt_exojax), ("convolution", lambda a, b, c: V.voigt_convolution(a, b, c, oversample=4))):
        got = np.asarray(f(x, jnp.asarray(1.0), jnp.asarray(1.0)))
        ax.semilogy(x, got, label=m); axr.plot(x, got / ref - 1)
    ax.set(ylabel=r"$V(x;\sigma=1,\gamma=1)$"); ax.legend()
    axr.set(xlabel="x", ylabel="rel. error", yscale="symlog", ylim=(-1, 1))
    C.savefig(fig, "voigt_vs_wofz")
C.done()
