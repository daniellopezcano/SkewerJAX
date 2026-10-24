#!/usr/bin/env python
"""tau of uniform gas against the closed forms of the review notes: static natural and
thermal lines (Sec. analytic_no_expansion) and radiation domination with natural and thermal broadening."""
from _common import Checks, figure_backend

C = Checks("check_optical_depth_analytic", __doc__)
import numpy as np                                                        # noqa: E402
import jax.numpy as jnp                                                   # noqa: E402
from skewerjax.core import (Constant, Cosmology, Fields, LineBundle, LogRamp, ScalarField, SkewerBundle, VectorField,
                            VelocityRamp, analytic as an, compute_optical_depth, line_wavelength_grid_A, phys,
                            plane_points)                                 # noqa: E402

lines = LineBundle(["HI-LyA", "HI-LyB", "SiIII-1206"])
null = LineBundle(["nullHI-LyA", "nullHI-LyB", "nullSiIII1206"])         # A_21 = 0: purely thermal lines
T_tiny = float(np.finfo(np.float32).tiny)
rad = Cosmology(H0=phys.H0, Omega_r_0=1.0, Omega_m_0=0.0, Omega_k_0=0.0, Omega_lambda_0=0.0)


def fields(T, n, L):
    sp = lambda: {"temperature": ScalarField(LogRamp(*T, L, 3)), "velocity": VectorField(VelocityRamp(0.0, 0.0, L)),
                  "number_density": ScalarField(LogRamp(*n, L, 2)), "turbulence": ScalarField(Constant(0.0)),
                  "collisional": ScalarField(Constant(0.0))}
    return Fields({"HI": sp(), "SiIII": sp()})


def per_line(f, lb, v, a_dagger=1.0):
    wl = line_wavelength_grid_A(v, lb, a_dagger=a_dagger, a_obs=1.0)
    return jnp.stack([f(j, wl[j]) for j in range(lb.n_lines)])


n_pp = 32 if C.quick else 64
cases = {}
# (i) static gas, T = 0: a Lorentzian of the natural width (points on the plane x = L/2, densities along y)
Lk = 1e-3
pts = plane_points(Lk, "x", Lk / 2, 3, n_pp)
sb = SkewerBundle(spatial_points=pts, manual_scale_factors=jnp.ones_like(pts[None, ..., 0]))
v = jnp.linspace(-200, 200, 512)
F_i = fields((T_tiny, T_tiny), (1e-9, 1e-3), Lk)
tau = compute_optical_depth(sb, v, lines, F_i, a_dagger=jnp.ones_like(sb.scale_factors[0]))
n_sk = F_i["HI"]["number_density"](jnp.stack([jnp.ones(3), jnp.full(3, Lk / 2), jnp.linspace(0, Lk, 3), jnp.full(3, Lk / 2)], -1))
ref = per_line(lambda j, wl: an.tau_static_natural(wl, lines.lambda0_A[j:j + 1], lines.A21_GHz[j:j + 1], lines.f_osc[j:j + 1],
                                                   sb.a0, Lk * jnp.ones(3), n_sk)[0], lines, v)
cases["static, T = 0 (natural)"] = (v, tau, ref, 1e-6)
# (ii) static gas, A_21 = 0: Gaussians of the thermal width (temperatures along z)
pts = plane_points(Lk, "y", Lk / 2, 3, n_pp)
sb = SkewerBundle(spatial_points=pts, manual_scale_factors=jnp.ones_like(pts[None, ..., 0]))
v = jnp.linspace(-3000.0, 3000.0, 1024)
F_ii = fields((1e4, 1e8), (1e-7, 1e-7), Lk)
tau = compute_optical_depth(sb, v, null, F_ii, a_dagger=jnp.ones_like(sb.scale_factors[0]))
T_sk = F_ii["HI"]["temperature"](sb.quadri_vectors[0])[:, 0]
ref = per_line(lambda j, wl: an.tau_static_thermal(wl, null.lambda0_A[j:j + 1], null.mass_amu[j:j + 1], null.f_osc[j:j + 1], sb.a0,
                                                   Lk * jnp.ones(3), 1e-7 * jnp.ones(3), T_sk)[0], null, v)
cases["static, A_21 = 0 (thermal)"] = (v, tau, ref, 1e-6)
# (iii) radiation domination, T = 0: one skewer with the ICs at absorption, one at emission. The natural width
# (~0.006 km/s for Lya) is never resolved by the points (spacing ~3 km/s), so tau is a comb of Lorentzians, not the
# analytic arctan: the resolution check must flag the sampling (the comparison is drawn in the figure only).
L5 = 5.0
v_rad = jnp.linspace(-250.0, 250.0, 768)


def rad_bundle(n, plane=False):
    pts = (plane_points(L5, "y", L5 / 2, 2, n) if plane else
           jnp.tile(jnp.array([jnp.linspace(0.0, L5, n), jnp.full(n, L5 / 2), jnp.full(n, L5 / 2)]).T[None], (2, 1, 1)))
    return SkewerBundle(spatial_points=pts, a0=[1.0, 0.99999976], x0=[0.0, 0.0], ICs_defined_at_index=[-1, 0],
                        ICs_at_emission_or_absorption=["absorption", "emission"], cosmologies=rad)


sb = rad_bundle(128)
F_iii = fields((T_tiny, T_tiny), (1e-7, 1e-7), L5)
a_d = jnp.repeat(jnp.asarray(sb.a0)[:, None], 128, axis=1)
tau = compute_optical_depth(sb, v_rad, lines, F_iii, a_dagger=a_d, resolution_tolerance=None)
ref = per_line(lambda j, wl: an.tau_radiation_natural(wl, lines.lambda0_A[j:j + 1], lines.A21_GHz[j:j + 1], lines.f_osc[j:j + 1], sb.a0,
                                                      sb.x0, jnp.array([-L5, 0.0]), jnp.array([0.0, L5]), jnp.array([1e-7, 1e-7]))[0], lines, v_rad)
from skewerjax.core import pieces_from_fields, resolution_report           # noqa: E402
rep = resolution_report(pieces_from_fields(sb, F_iii, lines, a_dagger=a_d), lines, 0.25)
C.check("radiation domination, T = 0: the resolution check flags the unresolved natural lines (max W / width)",
        float(np.min(rep["max_ratio"])), 0.25, expect=True)
cases["radiation domination, T = 0 (natural, unresolved comb)"] = (v_rad, tau, ref, None)
# (iv) radiation domination, A_21 = 0, two temperatures: second-order convergence to the analytic F
v = jnp.linspace(-4000.0, 4000.0, 1024)
F_iv = fields((5e4, 5e6), (5e-11, 5e-11), L5)
errs = []
for n in ((84, 168) if C.quick else (42, 84, 168, 336)):
    sb = rad_bundle(n, plane=True)
    tau = compute_optical_depth(sb, v, null, F_iv, a_dagger=jnp.repeat(jnp.asarray(sb.a0)[:, None], n, axis=1), resolution_tolerance=None)
    ref = per_line(lambda j, wl: an.tau_radiation_thermal(wl, null.lambda0_A[j:j + 1], null.f_osc[j:j + 1], null.mass_amu[j:j + 1],
                                                          jnp.array([5e4, 5e6]), sb.a0, sb.x0, jnp.array([-L5, 0.0]), jnp.array([0.0, L5]),
                                                          jnp.array([5e-11, 5e-11]))[0], null, v)
    errs.append(np.max(np.abs(np.exp(-np.asarray(tau)) - np.exp(-np.asarray(ref))), axis=(1, 2)))
errs = np.array(errs)                                      # (n_resolutions, n_lines)
# the ratio from 42 points is pre-asymptotic for the hot line (the pieces at their centroids, 2.1); from 84 points on
# every step is in the N^-2 regime
asym = errs if C.quick else errs[1:]
C.check("radiation domination, A_21 = 0: error falls ~N^-2 when the points double (min ratio over lines, steps from 84 points)",
        float(np.min(asym[:-1] / asym[1:])), 2.5, expect=True, detail="max |dF| per line and resolution: "
        + "; ".join(" ".join(f"{e:.1e}" for e in row) for row in errs))
C.check(f"radiation domination, A_21 = 0, {(168 if C.quick else 336)} points: max |F - F_analytic| over 3 lines",
        float(errs[-1].max()), 1e-1 if C.quick else 1e-2)
cases["radiation domination, A_21 = 0 (thermal)"] = (v, tau, ref, None)

for name, (v, tau, ref, tol) in cases.items():
    if tol is not None:
        C.check(f"{name}: max |F - F_analytic| over 3 lines", np.max(np.abs(np.exp(-np.asarray(tau)) - np.exp(-np.asarray(ref)))), tol)

if C.outdir is not None:
    figure_backend()
    import matplotlib.pyplot as plt
    from skewerjax.utils.plotting import MODE_STYLES as MS, add_panel_letters
    fig, axs = plt.subplots(2, len(cases), figsize=(22, 8), sharex="col", gridspec_kw={"height_ratios": [2, 1]}, layout="constrained")
    for k, (name, (v, tau, ref, tol)) in enumerate(cases.items()):
        F, Fa = np.exp(-np.asarray(tau)), np.exp(-np.asarray(ref))
        for s in range(F.shape[1]):
            l, = axs[0, k].plot(v, F[0, s], **MS["default"])
            axs[0, k].plot(v, Fa[0, s], color=l.get_color(), **MS["reference"])
            axs[1, k].plot(v, F[0, s] - Fa[0, s], color=l.get_color())
        axs[0, k].set_title(name, fontsize=15); axs[1, k].set_xlabel(r"$v$ [km/s]")
    axs[0, 0].set_ylabel(r"$F$ (Ly$\alpha$)"); axs[1, 0].set_ylabel(r"$F - F_{\rm analytic}$")
    add_panel_letters(axs)
    C.savefig(fig, "optical_depth_vs_analytic")
C.done()
