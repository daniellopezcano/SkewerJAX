"""The closed-form cases of uniform gas (review notes, Sec. analytic_no_expansion and radiation domination), shared by
``test_optical_depth.py`` and ``scripts/figures/optical_depth_vs_analytic.py``: each returns (v [km/s], tau, the analytic
tau) for three lines. The gas is a set of cells whose generating points are the sampling points of the skewers: the
Voronoi cell of a point of a straight skewer is its dual segment, so each skewer is cut into its dual segments, with the
values of the closed forms (uniform density; one temperature per skewer). Imports nothing from pytest."""

import jax.numpy as jnp
import numpy as np

from skewerjax.core import (
    Cosmology,
    LineBundle,
    SkewerBundle,
    cell_pieces,
    cells_from_values,
    compute_optical_depth,
    line_wavelength_grid_A,
    phys,
    plane_points,
    resolution_report,
)
from skewerjax.core import analytic as an

LINES3 = LineBundle(["HI-LyA", "HI-LyB", "SiIII-1206"])
NULL3 = LineBundle(["nullHI-LyA", "nullHI-LyB", "nullSiIII1206"])  # A_21 = 0: purely thermal lines
T_TINY = float(np.finfo(np.float32).tiny)
RAD = Cosmology(H0=phys.H0, Omega_r_0=1.0, Omega_m_0=0.0, Omega_k_0=0.0, Omega_lambda_0=0.0)
L_STATIC, L_RAD = 1e-3, 5.0


def _log_ramp(lo, hi, L, axis):
    """10^(log lo + (log hi - log lo) x_axis / L) at the generating points (axis 1 = y, 2 = z)."""
    return lambda x: 10 ** (np.log10(lo) + (np.log10(hi) - np.log10(lo)) * x[:, axis] / L)


def _gas(pts, T, n, L):
    """Cells at the sampling points (static gas): temperature a log ramp along z, density a log ramp along y. The
    Voronoi cell of a point of a straight skewer is its dual segment."""
    gen = np.unique(np.asarray(pts, dtype=np.float64).reshape(-1, 3), axis=0)
    dx = float(np.min(np.linalg.norm(np.diff(np.asarray(pts, dtype=np.float64), axis=-2), axis=-1)))
    dens = _log_ramp(*n, L, 1)
    return cells_from_values(
        gen, None, {"HI": dens, "SiIII": dens}, temperature=_log_ramp(*T, L, 2), volumes=np.full(gen.shape[0], dx**3)
    )


def _per_line(f, lb, v, a_dagger=1.0):
    wl = line_wavelength_grid_A(v, lb, a_dagger=a_dagger, a_obs=1.0)
    return jnp.stack([f(j, wl[j]) for j in range(lb.n_lines)])


def _rad_bundle(n, plane=False):
    pts = (
        plane_points(L_RAD, "y", L_RAD / 2, 2, n)
        if plane
        else jnp.tile(jnp.array([jnp.linspace(0.0, L_RAD, n), jnp.full(n, L_RAD / 2), jnp.full(n, L_RAD / 2)]).T[None], (2, 1, 1))
    )
    return SkewerBundle(
        spatial_points=pts,
        a0=[1.0, 0.99999976],
        x0=[0.0, 0.0],
        ICs_defined_at_index=[-1, 0],
        ICs_at_emission_or_absorption=["absorption", "emission"],
        cosmologies=RAD,
    )


def static_natural(n_pp=64):
    """Static gas, T = 0: a Lorentzian of the natural width (points on the plane x = L/2, densities along y): (v, tau, ref)."""
    pts = plane_points(L_STATIC, "x", L_STATIC / 2, 3, n_pp)
    sb = SkewerBundle(spatial_points=pts, manual_scale_factors=jnp.ones_like(pts[None, ..., 0]))
    v = jnp.linspace(-200, 200, 512)
    tau = compute_optical_depth(sb, v, LINES3, _gas(pts, (T_TINY, T_TINY), (1e-9, 1e-3), L_STATIC), a_dagger=1.0)
    n_sk = jnp.asarray(_log_ramp(1e-9, 1e-3, L_STATIC, 1)(np.asarray(pts)[:, 0]))
    ref = _per_line(
        lambda j, wl: an.tau_static_natural(
            wl,
            LINES3.lambda0_A[j : j + 1],
            LINES3.A21_GHz[j : j + 1],
            LINES3.f_osc[j : j + 1],
            sb.a0,
            L_STATIC * jnp.ones(3),
            n_sk,
        )[0],
        LINES3,
        v,
    )
    return v, tau, ref


def static_thermal(n_pp=64):
    """Static gas, A_21 = 0: Gaussians of the thermal width (temperatures along z): (v, tau, ref)."""
    pts = plane_points(L_STATIC, "y", L_STATIC / 2, 3, n_pp)
    sb = SkewerBundle(spatial_points=pts, manual_scale_factors=jnp.ones_like(pts[None, ..., 0]))
    v = jnp.linspace(-3000.0, 3000.0, 1024)
    tau = compute_optical_depth(sb, v, NULL3, _gas(pts, (1e4, 1e8), (1e-7, 1e-7), L_STATIC), a_dagger=1.0)
    T_sk = jnp.asarray(_log_ramp(1e4, 1e8, L_STATIC, 2)(np.asarray(pts)[:, 0]))
    ref = _per_line(
        lambda j, wl: an.tau_static_thermal(
            wl,
            NULL3.lambda0_A[j : j + 1],
            NULL3.mass_amu[j : j + 1],
            NULL3.f_osc[j : j + 1],
            sb.a0,
            L_STATIC * jnp.ones(3),
            1e-7 * jnp.ones(3),
            T_sk,
        )[0],
        NULL3,
        v,
    )
    return v, tau, ref


def radiation_natural():
    """Radiation domination, T = 0: one skewer with the ICs at absorption, one at emission. The natural width (~0.006
    km/s for Lya) is never resolved by the points (~3 km/s), so tau is a comb of Lorentzians, not the analytic arctan:
    (v, tau, ref, the resolution report)."""
    sb = _rad_bundle(128)
    cells = _gas(sb.spatial_points, (T_TINY, T_TINY), (1e-7, 1e-7), L_RAD)
    a_d = jnp.asarray(sb.a0)
    v = jnp.linspace(-250.0, 250.0, 768)
    tau = compute_optical_depth(sb, v, LINES3, cells, a_dagger=a_d, resolution_tolerance=None)
    ref = _per_line(
        lambda j, wl: an.tau_radiation_natural(
            wl,
            LINES3.lambda0_A[j : j + 1],
            LINES3.A21_GHz[j : j + 1],
            LINES3.f_osc[j : j + 1],
            sb.a0,
            sb.x0,
            jnp.array([-L_RAD, 0.0]),
            jnp.array([0.0, L_RAD]),
            jnp.array([1e-7, 1e-7]),
        )[0],
        LINES3,
        v,
    )
    return v, tau, ref, resolution_report(cell_pieces(cells, LINES3, skewer_bundle=sb, a_dagger=a_d), LINES3, 0.25)


def radiation_thermal(n):
    """Radiation domination, A_21 = 0, two temperatures, ``n`` points per skewer: (v, tau, ref)."""
    v = jnp.linspace(-4000.0, 4000.0, 1024)
    sb = _rad_bundle(n, plane=True)
    tau = compute_optical_depth(
        sb,
        v,
        NULL3,
        _gas(sb.spatial_points, (5e4, 5e6), (5e-11, 5e-11), L_RAD),
        a_dagger=jnp.asarray(sb.a0),
        resolution_tolerance=None,
    )
    ref = _per_line(
        lambda j, wl: an.tau_radiation_thermal(
            wl,
            NULL3.lambda0_A[j : j + 1],
            NULL3.f_osc[j : j + 1],
            NULL3.mass_amu[j : j + 1],
            jnp.array([5e4, 5e6]),
            sb.a0,
            sb.x0,
            jnp.array([-L_RAD, 0.0]),
            jnp.array([0.0, L_RAD]),
            jnp.array([5e-11, 5e-11]),
        )[0],
        NULL3,
        v,
    )
    return v, tau, ref
