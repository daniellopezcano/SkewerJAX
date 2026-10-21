"""Closed-form optical depths of uniform gas (notes ``app_analytical_solutions.tex``): the validation targets of tau."""
from __future__ import annotations

import jax
import jax.numpy as jnp

from skewerjax.core.constants import phys
from skewerjax.core.voigt import gaussian, log10_voigt, lorentzian

# Shapes: lambda_vals (N_lambda,) [A]; per-line arrays lambda_rest [A], A21_GHz [GHz], f_osc, mass_amu [amu] (N_lines,);
# per-skewer arrays (N_sk,). Every function returns (N_lines, N_sk, N_lambda) unless stated.


def tau_static_natural(lambda_vals, lambda_rest, A21_GHz, f_osc, a0, Delta_x_Mpc, n0_cm3):
    """No expansion (a = a0), T = 0, uniform n0 over Delta_x: a Lorentzian of the natural width (Sec. analytic_no_expansion)."""
    lambda_vals, lambda_rest = lambda_vals[None, None, :], lambda_rest[:, None, None]
    A21_GHz, f_osc = A21_GHz[:, None, None], f_osc[:, None, None]
    a0, Delta_x_Mpc, n0_cm3 = a0[None, :, None], Delta_x_Mpc[None, :, None], n0_cm3[None, :, None]
    prefactor = phys.cross_section_prefactor * f_osc * n0_cm3 * Delta_x_Mpc * 1e2 * phys.Mpc_to_m / a0**2    # [GHz]
    nu_GHz = (1 / lambda_rest - a0 / lambda_vals) * phys.c * 10
    return prefactor * lorentzian(nu_GHz, A21_GHz / (4 * jnp.pi))


def tau_static_thermal(lambda_vals, lambda_rest, mass_amu, f_osc, a0, Delta_x_Mpc, n0_cm3, T0_K):
    """No expansion, A_21 = 0, uniform n0 and T0 over Delta_x: a Gaussian of the thermal width (Sec. analytic_no_expansion)."""
    lambda_vals, lambda_rest = lambda_vals[None, None, :], lambda_rest[:, None, None]
    mass_amu, f_osc = mass_amu[:, None, None], f_osc[:, None, None]
    a0, Delta_x_Mpc, n0_cm3, T0_K = a0[None, :, None], Delta_x_Mpc[None, :, None], n0_cm3[None, :, None], T0_K[None, :, None]
    prefactor = phys.cross_section_prefactor * f_osc * n0_cm3 * Delta_x_Mpc * 1e2 * phys.Mpc_to_m / a0**2    # [GHz]
    nu_GHz = (1 / lambda_rest - a0 / lambda_vals) * phys.c * 10
    v_eff_A_GHz = jnp.sqrt(2.0 * phys.kB * T0_K / mass_amu) * 10.0                                        # [A / ns]
    return prefactor * gaussian(nu_GHz, v_eff_A_GHz / (jnp.sqrt(2.0) * lambda_rest))


def tau_radiation_natural(lambda_vals, lambda_rest, A21_GHz, f_osc, a0, x0_Mpc, xem_Mpc, xf_Mpc, n0_cm3):
    """Radiation domination, uniform comoving n0, T = 0: arctan of the detuning at the path ends (Sec. radiation-dominated).

    x0_Mpc, xem_Mpc, xf_Mpc : (N_sk,) comoving distances of the observer, the emitter and the far end [Mpc].
    """
    lambda_vals, lambda_rest = lambda_vals[None, None, :], lambda_rest[:, None, None]
    A21_GHz, f_osc = A21_GHz[:, None, None], f_osc[:, None, None]
    a0, x0_Mpc, xem_Mpc, xf_Mpc = a0[None, :, None], x0_Mpc[None, :, None], xem_Mpc[None, :, None], xf_Mpc[None, :, None]
    n0_cm3 = n0_cm3[None, :, None]
    # C Gyr_to_s 1e9 = sigma_0 [cm^2 GHz]; 1 / pi of the Lorentzian normalisation
    prefactor = (phys.cross_section_prefactor * phys.Gyr_to_s * 1e9) / (jnp.pi * phys.H0) * 1e-8 * f_osc * n0_cm3 * lambda_vals / a0
    gamma_GHz = A21_GHz / (4 * jnp.pi)

    def arctan_term(x_Mpc):
        lambda_prime = lambda_vals * ((phys.H0_over_c / a0) * (x_Mpc - x0_Mpc) + 1.0)
        return jnp.arctan((1 / lambda_rest - 1 / lambda_prime) * phys.c * 10 / gamma_GHz)

    return prefactor * (arctan_term(xf_Mpc) - arctan_term(xem_Mpc))


def tau_radiation_thermal(lambda_vals, lambda_rest, f_osc, mass_amu, T0, a0, x0_Mpc, xem_Mpc, xf_Mpc, n0_cm3):
    """Radiation domination, uniform comoving n0 and T0 (A_21 = 0): erf of the detuning at the path ends.

    T0 : (N_sk,) [K]; the distances as in ``tau_radiation_natural``.
    """
    lambda_vals, lambda_rest = lambda_vals[None, None, :], lambda_rest[:, None, None]
    f_osc, mass_amu = f_osc[:, None, None], mass_amu[:, None, None]
    T0, a0, x0_Mpc = T0[None, :, None], a0[None, :, None], x0_Mpc[None, :, None]
    xem_Mpc, xf_Mpc, n0_cm3 = xem_Mpc[None, :, None], xf_Mpc[None, :, None], n0_cm3[None, :, None]
    # as the natural case, with 2 instead of pi (Gaussian normalisation)
    prefactor = (phys.cross_section_prefactor * phys.Gyr_to_s * 1e9) / (2.0 * phys.H0) * 1e-8 * f_osc * n0_cm3 * lambda_vals / a0
    v_eff_A_GHz = jnp.sqrt(2.0 * phys.kB * T0 / mass_amu) * 10.0
    sigma_GHz = v_eff_A_GHz / (jnp.sqrt(2.0) * lambda_rest)

    def erf_term(x_Mpc):
        lambda_prime = lambda_vals * ((phys.H0_over_c / a0) * (x_Mpc - x0_Mpc) + 1.0)
        nu = (1 / lambda_rest - 1 / lambda_prime) * phys.c * 10
        return jax.scipy.special.erf(nu / (jnp.sqrt(2.0) * sigma_GHz))

    return prefactor * (erf_term(xf_Mpc) - erf_term(xem_Mpc))


def tau_mid_point(line_bundle, T0_K, n0_cm3, Delta_x_Mpc, a0, method: str = "trapezoidal"):
    """tau at the line centre of a slab of width Delta_x seen by one trapezoid interval (convergence tests).

    T0_K (N_T,), n0_cm3 (N_n,), Delta_x_Mpc (N_x,); method "trapezoidal" (weight Delta_x / 2) or "analytical_deltas"
    (weight 1). Returns (N_lines, N_T, N_n, N_x). Note: sigma = v_eff / (2 lambda) here, not / (sqrt(2) lambda).
    """
    if method == "trapezoidal":
        term = Delta_x_Mpc / 2
    elif method == "analytical_deltas":
        term = 1.0
    else:
        raise NotImplementedError
    f_osc, lambda0_A, A21_GHz, mass_amu = line_bundle.f_osc, line_bundle.lambda0_A, line_bundle.A21_GHz, line_bundle.mass_amu
    N_lines, N_T0, N_xx = f_osc.shape[0], T0_K.shape[0], Delta_x_Mpc.shape[0]
    T0_K, n0_cm3 = T0_K[None, :, None, None], n0_cm3[None, None, :, None]
    f_osc, lambda0_A = f_osc[:, None, None, None], lambda0_A[:, None, None, None]
    A21_GHz, mass_amu = A21_GHz[:, None, None, None], mass_amu[:, None, None, None]
    Delta_x_Mpc = Delta_x_Mpc[None, None, None, :]
    v_eff_A_GHz = jnp.sqrt(2.0 * phys.kB * T0_K / mass_amu) * 10.0
    sigma_flat = (v_eff_A_GHz / (2.0 * lambda0_A)).reshape(-1)
    gamma_flat = jnp.tile(A21_GHz / (4.0 * jnp.pi), (1, N_T0, 1, 1)).reshape(-1)
    nu_GHz = (phys.c * 10.0 / lambda0_A) * (1 - a0 / (a0 + 0.5 * phys.H0_over_c * Delta_x_Mpc))
    nu_flat = jnp.reshape(jnp.tile(nu_GHz, (1, N_T0, 1, 1)), (N_lines * N_T0, N_xx))
    phi = jax.vmap(lambda x, s, g: 10**log10_voigt(x, s, g), in_axes=(0, 0, 0))(nu_flat, sigma_flat, gamma_flat)
    C_j = phys.cross_section_prefactor * f_osc
    rhs = (2 * n0_cm3 * C_j * term) * (1e2 * phys.Mpc_to_m)
    rhs *= (a0 + 0.5 * phys.H0_over_c * Delta_x_Mpc)**-2
    return rhs * phi.reshape(N_lines, N_T0, 1, N_xx)


def tau_mid_point_radiation_thermal(line_bundle, T0_K, n0_cm3, Delta_x_Mpc, a0):
    """Radiation-domination counterpart of ``tau_mid_point`` for thermal lines (erf form). Returns (N_lines, N_T, N_n, N_x)."""
    f_osc, lambda0_A, mass_amu = line_bundle.f_osc, line_bundle.lambda0_A, line_bundle.mass_amu
    T0_K, n0_cm3 = T0_K[None, :, None, None], n0_cm3[None, None, :, None]
    f_osc, lambda0_A, mass_amu = f_osc[:, None, None, None], lambda0_A[:, None, None, None], mass_amu[:, None, None, None]
    Delta_x_Mpc = Delta_x_Mpc[None, None, None, :]
    C_j = phys.cross_section_prefactor * f_osc
    prefactor = n0_cm3 * C_j * lambda0_A / (a0 * phys.H0) * 10 * phys.Gyr_to_s
    sigma_GHz = jnp.sqrt(2.0 * phys.kB * T0_K / mass_amu) * 10.0 / (2.0 * lambda0_A)
    arg = phys.c * 10 / (sigma_GHz * lambda0_A * jnp.sqrt(2.0))
    arg *= (1 - a0 / (a0 + 0.5 * phys.H0_over_c * Delta_x_Mpc))
    return prefactor * jax.scipy.special.erf(arg)
