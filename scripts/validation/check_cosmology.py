#!/usr/bin/env python
"""a(x) along a photon path: the diffrax solution against the closed forms of radiation, matter and Lambda domination
(notes ``sec_cosmology.tex``, Eq. dadx_general, Fig. a_of_x)."""
from _common import Checks, figure_backend

C = Checks("check_cosmology", __doc__)
import numpy as np                                                        # noqa: E402
import jax.numpy as jnp                                                   # noqa: E402
from skewerjax.core import Cosmology, phys                                # noqa: E402

h = phys.H0_over_c                                                        # [1/Mpc]
cosmo = Cosmology(H0=jnp.full(3, phys.H0), Omega_r_0=jnp.array([1.0, 0.0, 0.0]), Omega_m_0=jnp.array([0.0, 1.0, 0.0]),
                  Omega_k_0=jnp.zeros(3), Omega_lambda_0=jnp.array([0.0, 0.0, 1.0]))
exact = {"radiation": lambda x: 1.0 + h * x, "matter": lambda x: (1.0 + 0.5 * h * x) ** 2, "Lambda": lambda x: 1.0 / (1.0 - h * x)}
n = 200 if C.quick else 1000
x_f = np.linspace(0.0, 0.8 / h, n)                                        # towards the future (a grows)
x_b = np.linspace(0.0, -0.8 / h, n)                                       # towards the past
for tol_name, kw, tol in (("default rtol=atol=1e-4", {}, 2e-3), ("rtol=1e-10, atol=1e-12", dict(rtol=1e-10, atol=1e-12), 1e-8)):
    for x in (x_f, x_b):
        a = np.asarray(cosmo.solve_a_of_x(jnp.ones(3), jnp.asarray(x), **kw))
        for i, name in enumerate(exact):
            r = np.max(np.abs(a[i] / exact[name](x) - 1))
            C.check(f"{name}, {'forward' if x[-1] > 0 else 'backward'} to |x| = 0.8 c/H0, {tol_name}: max |a / a_exact - 1|", r, tol)

planck = Cosmology(H0=phys.H0, Omega_r_0=9.1436e-5, Omega_m_0=0.3089, Omega_k_0=0.0, Omega_lambda_0=0.6911)
a_p = np.asarray(planck.solve_a_of_x(jnp.ones(1), jnp.asarray(x_b)))[0]
C.check("Planck-2018: a(x) monotonic and bracketed by the matter and radiation solutions (violations)",
        float(np.sum(np.diff(a_p) > 0)) + float(np.sum(a_p < exact["matter"](x_b) - 1e-6)), 0.5)
try:
    Cosmology(H0=phys.H0, Omega_r_0=0.0, Omega_m_0=0.5, Omega_k_0=0.0, Omega_lambda_0=0.4)
    C.check("densities that do not sum to 1 raise ValueError", 1, 0)
except ValueError:
    C.check("densities that do not sum to 1 raise ValueError", 0, 0)

if C.outdir is not None:
    figure_backend()
    import matplotlib.pyplot as plt
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}, layout="constrained")
    x = np.concatenate([x_b[::-1], x_f])
    a = np.concatenate([np.asarray(cosmo.solve_a_of_x(jnp.ones(3), jnp.asarray(x_b)))[:, ::-1],
                        np.asarray(cosmo.solve_a_of_x(jnp.ones(3), jnp.asarray(x_f)))], axis=1)
    for i, name in enumerate(exact):
        l, = ax.plot(x, a[i], label=name)
        ax.plot(x, exact[name](x), ls="--", lw=3, color=l.get_color(), alpha=0.5)
        axr.plot(x, a[i] / exact[name](x) - 1, color=l.get_color())
    ax.set(yscale="log", ylabel="$a(x)$"); ax.legend()
    axr.set(xlabel="comoving distance $x$ [Mpc]", ylabel="residual")
    C.savefig(fig, "cosmology_a_of_x")
C.done()
