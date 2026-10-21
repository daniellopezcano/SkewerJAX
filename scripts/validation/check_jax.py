#!/usr/bin/env python
"""JAX-readiness of Block A: every public numerical function of ``skewerjax.core`` gives the eager result under
``jax.jit`` and, batched over skewers (or keys, parameters), under ``jax.vmap``; the containers pass through jit as
pytrees. Tolerance 1e-12 of the array scale (float64)."""
from _common import Checks

C = Checks("check_jax", __doc__)
import numpy as np                                                        # noqa: E402
import jax
import jax.numpy as jnp
from skewerjax.core import (phys, LineBundle, Cosmology, SkewerBundle, Fields, ScalarField, VectorField, SampledField,
                            PowerLawGRF, GaussianDensity, LogRamp, VelocityRamp, Constant, plane_points, path_geometry,
                            line_elements, overflux_terms, equivalent_width_kms)
from skewerjax.core import skewers as skw, voigt as V, spectral as sp_, optical_depth as od, analytic as an
from skewerjax.core.stats import two_point as tp, contaminants as cm
tol = 1e-12

def rel(a, b):
    fa = jax.tree_util.tree_leaves(a)
    fb = jax.tree_util.tree_leaves(b)
    m = 0.0
    for x, y in zip(fa, fb):
        x, y = np.asarray(x, dtype=np.complex128 if np.iscomplexobj(x) else np.float64), np.asarray(y, dtype=np.complex128 if np.iscomplexobj(y) else np.float64)
        if x.shape != y.shape:
            return np.inf
        fin = np.isfinite(y)
        if not np.array_equal(np.isfinite(x), fin):
            return np.inf
        sc = max(float(np.max(np.abs(y[fin]), initial=0.0)), 1e-300)
        m = max(m, float(np.max(np.abs(x[fin] - y[fin]), initial=0.0)) / sc)
    return m

def check(name, kind, fn):
    try:
        got, ref = fn()
        C.check(f"{kind:4s} {name}: max |x - x_eager| / max |x_eager|", rel(got, ref), tol)
    except Exception as e:                          # noqa: BLE001
        C.check(f"{kind:4s} {name}: raised {type(e).__name__}: {e}"[:200], float("inf"), tol)

def jit_check(name, f, *args, static=()):
    check(name, "jit", lambda: (jax.jit(f, static_argnums=static)(*args), f(*args)))

def vmap_check(name, f, batch, in_axes=0):
    def run():
        got = jax.vmap(f, in_axes=in_axes)(*batch)
        n = jax.tree_util.tree_leaves(batch[0] if in_axes == 0 or in_axes[0] == 0 else batch[1])[0].shape[0]
        ax = in_axes if isinstance(in_axes, tuple) else (in_axes,) * len(batch)
        ref = [f(*[jax.tree_util.tree_map(lambda x: x[i], b) if a == 0 else b for b, a in zip(batch, ax)]) for i in range(n)]
        return got, jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *ref)
    check(name, "vmap", run)

rng = np.random.default_rng(11)
lb = LineBundle(["HI-LyA", "HI-LyB", "SiIII-1206"])
# cosmology and skewers
cos = Cosmology(H0=jnp.array([phys.H0] * 2), Omega_r_0=jnp.array([0.0, 9e-5]), Omega_m_0=jnp.array([1.0, 0.3089]),
                Omega_k_0=jnp.zeros(2), Omega_lambda_0=jnp.array([0.0, 0.69101]))
x = jnp.linspace(0.0, -400.0, 30)
jit_check("Cosmology.solve_a_of_x", lambda c, a, x_: c.solve_a_of_x(a, x_), cos, jnp.ones(2), x)
vmap_check("Cosmology.solve_a_of_x (vmap over a0)", lambda a: cos.solve_a_of_x(a, x), (jnp.array([[1.0, 1.0], [0.5, 0.4], [0.3, 0.3]]),))
pts = jnp.asarray(rng.normal(size=(4, 12, 3)) * 10)
jit_check("photon_directions", skw.photon_directions, pts[0], 1)
vmap_check("photon_directions (skewers)", skw.photon_directions, (pts, jnp.array([1, -1, 1, -1])))
jit_check("comoving_distances", skw.comoving_distances, pts[0], 3.0, -1, True)
vmap_check("comoving_distances (skewers)", skw.comoving_distances, (pts, jnp.arange(4.0), jnp.array([0, -1, 0, -1]),
                                                                    jnp.array([True, False, True, False])))
jit_check("path_geometry", path_geometry, pts)
vmap_check("path_geometry (skewers)", path_geometry, (pts,))
jit_check("line_elements", line_elements, pts)
sb = SkewerBundle(spatial_points=pts, a0=0.3, x0=0.0, ICs_defined_at_index=0, cosmologies=cos)
jit_check("SkewerBundle (pytree through jit)", lambda b: (b.line_elements, b.tangents, b.quadri_vectors), sb)
# fields
Lk = 1e-3
pp = plane_points(Lk, "x", Lk / 2, 3, 16)
qv = jnp.concatenate([jnp.full(pp.shape[:-1] + (1,), 0.9), pp], -1)
los = jnp.broadcast_to(jnp.array([0.0, 0.6, 0.8]), pp.shape)
spf = lambda: {"temperature": ScalarField(LogRamp(1e4, 1e6, Lk, 3)), "velocity": VectorField(VelocityRamp(-50.0, 80.0, Lk)),
               "number_density": ScalarField(LogRamp(1e-9, 1e-3, Lk, 2)), "turbulence": ScalarField(Constant(1.5)),
               "collisional": ScalarField(Constant(0.2))}
F = Fields({"HI": spf(), "SiIII": spf()})
jit_check("Fields.sample (pytree samplers)", lambda f, q, l: f.sample(q, l), F, qv, los)
jit_check("Fields.sample (closure sampler)", lambda q: Fields({"HI": {"temperature": ScalarField(lambda y: y[..., 0] * 2)}}).sample(q, los), qv)
vmap_check("Fields.sample (skewers)", lambda q, l: F.sample(q, l), (qv, los))
jit_check("Fields.tile_field_to_lines", lambda f, q, l: f.tile_field_to_lines("LoS_velocity", q, {"HI": [0, 1], "SiIII": [2]}, LoS_directions=l),
          F, qv.reshape(-1, 4), los.reshape(-1, 3))
jit_check("SampledField", lambda s, q: s(q), SampledField(jnp.arange(48.0).reshape(3, 16)), qv)
key = jax.random.PRNGKey(0)
g3 = PowerLawGRF(A_s=2.0, k_0=1.0, n_s=-0.5, L=(10.0, 10.0, 20.0), N=(8, 8, 16))
jit_check("PowerLawGRF.sample", lambda g, k: g.sample(k, return_modes=True), g3, key)
vmap_check("PowerLawGRF.sample (keys)", lambda k: g3.sample(k), (jax.random.split(key, 3),))
jit_check("PowerLawGRF.variance", lambda g: g.variance(), g3)
vmap_check("PowerLawGRF.power (n_s batch)", lambda n: PowerLawGRF(A_s=2.0, k_0=1.0, n_s=n, L=10.0, N=32).power(jnp.linspace(0, 3, 9)),
           (jnp.array([-1.0, -0.5, 0.5]),))
gd = GaussianDensity(key, PowerLawGRF(A_s=4.0, k_0=2.0, n_s=-1.25, L=(50.0,) * 3, N=(16,) * 3), n0_min=1e-16, n0_max=1e-11, contrast=1.2)
ev = jnp.asarray(rng.uniform(-5, 60, size=(4, 9, 4)))
jit_check("GaussianDensity", lambda s, e: s(e), gd, ev)
vmap_check("GaussianDensity (skewers)", gd, (ev,))
# Voigt
xv = jnp.linspace(-50.0, 50.0, 201)
for nm, f in (("gaussian", V.gaussian), ("lorentzian", V.lorentzian), ("voigt_exojax", V.voigt_exojax), ("voigt_convolution", V.voigt_convolution)):
    jit_check(nm, (lambda f_: (lambda a, b: f_(xv, a, b)))(f) if nm.startswith("voigt") else (lambda f_: (lambda a: f_(xv, a)))(f),
              *((jnp.asarray(3.0), jnp.asarray(0.5)) if nm.startswith("voigt") else (jnp.asarray(3.0),)))
for m in ("ExoJAX", "convolution"):
    jit_check(f"log10_voigt ({m})", lambda s, g: V.log10_voigt(xv, s, g, mode=m), jnp.asarray(3.0), jnp.asarray(0.5))
    vmap_check(f"log10_voigt ({m}, sigma batch)", lambda s: V.log10_voigt(xv, s, jnp.asarray(0.5), mode=m), (jnp.array([1.0, 3.0, 8.0]),))
# spectral coordinates (vmap over skewers: batches of points)
vel = jnp.linspace(-300.0, 300.0, 64)
a_pts = jnp.asarray(rng.uniform(0.30, 0.32, size=(4, 10)))
vlos = jnp.asarray(rng.normal(size=(4, 3, 10)) * 100.0)
lam = lb.lambda0_A
jit_check("local_wavelength_A", sp_.local_wavelength_A, vel, lam, 0.31, a_pts[0], vlos[0])
vmap_check("local_wavelength_A (skewers)", lambda a, v: sp_.local_wavelength_A(vel, lam, 0.31, a, v), (a_pts, vlos))
jit_check("local_wavelength_from_observed_A", sp_.local_wavelength_from_observed_A, sp_.line_wavelength_grid_A(vel, lb, 0.31, 0.5), 0.5, a_pts[0], vlos[0])
jit_check("line_centre_velocity_kms", sp_.line_centre_velocity_kms, 0.31, a_pts[0], vlos[0])
vmap_check("line_centre_velocity_kms (skewers)", lambda a, v: sp_.line_centre_velocity_kms(0.31, a, v), (a_pts, vlos))
for det in ("exact", "linear"):
    for P in (None, 250.0):
        jit_check(f"frequency_offset_GHz ({det}, P={P})", lambda a, v: sp_.frequency_offset_GHz(vel, lam, 0.31, a, v, velocity_period=P, detuning=det),
                  a_pts[0], vlos[0])
    vmap_check(f"frequency_offset_GHz ({det}, skewers)", lambda a, v: sp_.frequency_offset_GHz(vel, lam, 0.31, a, v, detuning=det), (a_pts, vlos))
jit_check("line_wavelength_grid_A", lambda l, a: sp_.line_wavelength_grid_A(vel, l, a, 0.9), lb, 0.31)
jit_check("velocities_in_line_frames", lambda l: sp_.velocities_in_line_frames(vel, l), lb)
# widths, profiles, tau
T3 = jnp.asarray(rng.uniform(1e3, 1e6, size=(4, 3, 10))); tb3 = jnp.asarray(rng.uniform(0, 5e3, size=(4, 3, 10)))
co3 = jnp.asarray(rng.uniform(0, 1.0, size=(4, 3, 10)))
W = lambda T, tb, co: od.voigt_widths_GHz(T, tb, co, lam, lb.A21_GHz, lb.mass_amu)
jit_check("voigt_widths_GHz", W, T3[0], tb3[0], co3[0])
vmap_check("voigt_widths_GHz (skewers)", W, (T3, tb3, co3))
sg, gm = W(T3[0], tb3[0], co3[0])
jit_check("voigt_width_kms", od.voigt_width_kms, sg, gm, lam)
nu = sp_.frequency_offset_GHz(vel, lam, 0.31, a_pts[0], vlos[0]).reshape(30, 64)
jit_check("voigt_line_profiles (ExoJAX)", lambda n, s, g: od.voigt_line_profiles(n, s, g, "ExoJAX"), nu, sg.reshape(-1), gm.reshape(-1))
# the convolution needs a uniform grid that resolves the line (on ``nu`` above it underflows to 0 / NaN)
nu_u = jnp.tile(jnp.linspace(-50.0, 50.0, 201)[None], (3, 1))
jit_check("voigt_line_profiles (convolution)", lambda n, s, g: od.voigt_line_profiles(n, s, g, "convolution"), nu_u,
          jnp.array([1.0, 3.0, 8.0]), jnp.array([0.1, 0.5, 2.0]))
jit_check("line_profiles", lambda q, l, f, b: od.line_profiles(q, l, vel, 0.88, b, f), qv, los, F, lb)
vmap_check("line_profiles (skewers)", lambda q, l: od.line_profiles(q, l, vel, 0.88, lb, F), (qv, los))
w_m = jnp.asarray(rng.uniform(0.5, 1.5, size=(4, 10))) * 1e20
n3 = jnp.asarray(rng.uniform(0, 1e-9, size=(4, 3, 10)))
K = lambda ad, a, lv, T, tb, co, n, w: od.optical_depth_kernel(vel, lam, lb.A21_GHz, lb.mass_amu, lb.f_osc, ad, a, lv, T, tb, co, n, w)
args = (jnp.full((4, 10), 0.31), a_pts, vlos, T3, tb3, co3, n3, w_m)
check("optical_depth_kernel (eager == jitted kernel)", "jit", lambda: (K(*[x[0] for x in args]),
      od.optical_depth_kernel.__wrapped__(vel, lam, lb.A21_GHz, lb.mass_amu, lb.f_osc, *[x[0] for x in args])))
vmap_check("optical_depth_kernel (skewers)", K, args)
S_ = od.LoSSamples(a_dagger=args[0], scale_factor=a_pts, weights_m=w_m, temperature=jnp.moveaxis(T3, 0, 1),
                   turbulence=jnp.moveaxis(tb3, 0, 1), collisional=jnp.moveaxis(co3, 0, 1), los_velocity=jnp.moveaxis(vlos, 0, 1),
                   number_density=jnp.moveaxis(n3, 0, 1))
ref_tau = jnp.stack([K(*[x[i] for x in args]) for i in range(4)], axis=1)
check("optical_depth_from_samples (lax.map == per-skewer kernel)", "vmap", lambda: (od.optical_depth_from_samples(S_, vel, lb), ref_tau))
check("optical_depth_from_samples (batch_size=2 vmap)", "vmap", lambda: (od.optical_depth_from_samples(S_, vel, lb, batch_size=2), ref_tau))
check("optical_depth_from_samples (v_chunk=20)", "jit", lambda: (od.optical_depth_from_samples(S_, vel, lb, v_chunk=20), ref_tau))
sbq = SkewerBundle(spatial_points=pp, manual_scale_factors=jnp.full((1,) + pp.shape[:2], 0.9))
jit_check("los_samples (bundle, fields, lines as pytrees)", lambda b, f, l: od.los_samples(b, f, l, a_dagger=0.88), sbq, F, lb)
jit_check("optical_depth (whole pipeline under jit)", lambda b, f, l: od.compute_optical_depth(b, vel, l, f, a_dagger=0.88, resolution_tolerance=None),
          sbq, F, lb)
tau_b = jnp.asarray(rng.uniform(0, 2, size=(3, 4, 64)))
jit_check("wrap_to_ref_line", lambda t, l: od.wrap_to_ref_line(t, vel, l), tau_b, lb)
vmap_check("wrap_to_ref_line (skewer batches)", lambda t: od.wrap_to_ref_line(t, vel, lb), (tau_b[None].repeat(2, 0),))
# analytic
wl = sp_.line_wavelength_grid_A(vel, lb, 1.0)
one = jnp.ones(2)
jit_check("tau_static_natural", lambda w: an.tau_static_natural(w, lam, lb.A21_GHz, lb.f_osc, one, one * 1e-3, one * 1e-5), wl[0])
jit_check("tau_static_thermal", lambda w: an.tau_static_thermal(w, lam, lb.mass_amu, lb.f_osc, one, one * 1e-3, one * 1e-5, one * 1e4), wl[0])
jit_check("tau_radiation_natural", lambda w: an.tau_radiation_natural(w, lam, lb.A21_GHz, lb.f_osc, one, one * 0, -one * 5, one * 0, one * 1e-7), wl[0])
jit_check("tau_radiation_thermal", lambda w: an.tau_radiation_thermal(w, lam, lb.f_osc, lb.mass_amu, one * 1e5, one, one * 0, -one * 5, one * 0, one * 1e-7), wl[0])
jit_check("tau_mid_point", lambda l: an.tau_mid_point(l, jnp.array([1e4, 1e5]), jnp.array([1e-10, 1e-9]), jnp.array([0.1, 1.0]), 1.0), lb)
jit_check("tau_mid_point_radiation_thermal", lambda l: an.tau_mid_point_radiation_thermal(l, jnp.array([1e4]), jnp.array([1e-10]), jnp.array([0.1, 1.0]), 1.0), lb)
# flux, overflux, statistics, models
Fx = jnp.exp(-tau_b)
check("overflux_terms", "jit", lambda: (jax.jit(overflux_terms, static_argnums=(1,))(Fx, ("a", "b", "c")), overflux_terms(Fx, ("a", "b", "c"))))
jit_check("equivalent_width_kms", lambda t: equivalent_width_kms(t, 2.0), tau_b)
vmap_check("equivalent_width_kms (skewers)", lambda t: equivalent_width_kms(t, 2.0), (jnp.moveaxis(tau_b, 1, 0),))
jit_check("xi_from_power", lambda P: tp.xi_from_power(P, 64, 2.0), jnp.asarray(rng.normal(size=(4, 33))))
vmap_check("xi_from_power (skewers)", lambda P: tp.xi_from_power(P, 64, 2.0), (jnp.asarray(rng.normal(size=(4, 33))),))
jit_check("binned_mean", lambda v, i, w: tp.binned_mean(v, i, w, 5), jnp.asarray(rng.normal(size=50)), jnp.asarray(rng.integers(-1, 5, 50)), jnp.ones(50))
kk = jnp.linspace(1e-3, 0.08, 40)
jit_check("mcdonald_factor", lambda k, a: cm.mcdonald_factor(k, a), kk, 0.05)
vmap_check("mcdonald_factor (a batch)", lambda a: cm.mcdonald_factor(kk, a), (jnp.array([0.01, 0.05, -0.02]),))
jit_check("ma_factor", lambda k, a: cm.ma_factor(k, a, 0.1, 0.05, -0.02), kk, 0.01)
vmap_check("ma_factor (k_auto batch, incl. inf)", lambda ka: cm.ma_factor(kk, 0.01, 0.1, ka, -0.02), (jnp.array([0.05, 0.1, jnp.inf]),))
C.done()
