from __future__ import annotations

import logging
logger = logging.getLogger(__name__)

import jax
from jax import vmap
import jax.numpy as jnp

from skewerjax.optical_depth.voigt import Lorentzian
from skewerjax.optical_depth.voigt import Gaussian
from skewerjax.constants.physical_constants import phys

def compute_optical_depth_analytical_static_Natural(
    lambda_vals: jnp.ndarray,   # (N_lambda,) [Å]
    lambda_rest: jnp.ndarray,   # (N_lines,)  [Å]
    A21_GHz: jnp.ndarray,       # (N_lines,)  [GHz]
    f_osc: jnp.ndarray,         # (N_lines,)  dimensionless
    a0: jnp.ndarray,            # (N_sk,)     scale factor at observer
    Delta_x_Mpc: jnp.ndarray,   # (N_sk,)     comoving distance to emission [Mpc]
    n0_cm3: jnp.ndarray,        # (N_sk,)     number density at observer [cm^-3]
) -> jnp.ndarray:
    """
    Compute optical depth for natural (Lorentzian) line broadening in radiation-dominated era.
    
    Returns:
        Optical depth array of shape (N_lines, N_sk, N_lambda)
    """
    # --- Broadcast to common shape: (N_lines, N_sk, N_lambda) ---
    lambda_vals = lambda_vals[None, None, :]  # (1, 1, N_lambda)
    lambda_rest = lambda_rest[:, None, None]  # (N_lines, 1, 1)
    A21_GHz = A21_GHz[:, None, None]          # (N_lines, 1, 1)
    f_osc = f_osc[:, None, None]              # (N_lines, 1, 1)
    a0 = a0[None, :, None]                    # (1, N_sk, 1)
    Delta_x_Mpc = Delta_x_Mpc[None, :, None]  # (1, N_sk, 1)
    n0_cm3 = n0_cm3[None, :, None]            # (1, N_sk, 1)

    # --- Compute prefactor ----------------------------------------------------
    prefactor = phys.cross_section_prefactor * f_osc * n0_cm3 * Delta_x_Mpc * 1e2 * phys.Mpc_to_m / a0**2 # UNITS [GHz]

    # --- Compute nu_GHz -------------------------------------------
    nu_GHz = (1 / lambda_rest - a0 / lambda_vals) * phys.c * 10

    # --- Compute line profile terms -------------------------------------------
    gamma_GHz = A21_GHz / (4 * jnp.pi)  # Lorentzian HWHM [GHz]
    
    # --- Compute line profile -------------------------------------------
    phi_ns = Lorentzian(nu_GHz, gamma_GHz)

    return prefactor * phi_ns


def compute_optical_depth_analytical_static_Thermal(
    lambda_vals: jnp.ndarray,   # (N_lambda,) [Å]
    lambda_rest: jnp.ndarray,   # (N_lines,)  [Å]
    mass_amu: jnp.ndarray,      # (N_lines,)  [amu]
    f_osc: jnp.ndarray,         # (N_lines,)  dimensionless
    a0: jnp.ndarray,            # (N_sk,)     scale factor at observer
    Delta_x_Mpc: jnp.ndarray,   # (N_sk,)     comoving distance to emission [Mpc]
    n0_cm3: jnp.ndarray,        # (N_sk,)     number density at observer [cm^-3]
    T0_K: jnp.ndarray,          # (N_sk,)     Temperature at observer [K]
) -> jnp.ndarray:
    """
    Compute optical depth for natural (Lorentzian) line broadening in radiation-dominated era.
    
    Returns:
        Optical depth array of shape (N_lines, N_sk, N_lambda)
    """
    # --- Broadcast to common shape: (N_lines, N_sk, N_lambda) ---
    lambda_vals = lambda_vals[None, None, :]  # (1, 1, N_lambda)
    lambda_rest = lambda_rest[:, None, None]  # (N_lines, 1, 1)
    mass_amu = mass_amu[:, None, None]        # (N_lines, 1, 1)
    f_osc = f_osc[:, None, None]              # (N_lines, 1, 1)
    a0 = a0[None, :, None]                    # (1, N_sk, 1)
    Delta_x_Mpc = Delta_x_Mpc[None, :, None]  # (1, N_sk, 1)
    n0_cm3 = n0_cm3[None, :, None]            # (1, N_sk, 1)
    T0_K = T0_K[None, :, None]                # (1, N_sk, 1)

    # --- Compute prefactor ----------------------------------------------------
    prefactor = phys.cross_section_prefactor * f_osc * n0_cm3 * Delta_x_Mpc * 1e2 * phys.Mpc_to_m / a0**2 # UNITS [GHz]

    # --- Compute nu_GHz -------------------------------------------
    nu_GHz = (1 / lambda_rest - a0 / lambda_vals) * phys.c * 10

    # --- Compute line profile terms -------------------------------------------
    v_eff_A_GHz = jnp.sqrt(2.0 * phys.kB * T0_K / mass_amu) * 10.0 # UNITS: [Å/ns]
    sigma_GHz = v_eff_A_GHz / (jnp.sqrt(2.0) * lambda_rest)                  # UNITS: [GHz]
    
    # --- Compute line profile -------------------------------------------
    phi_ns = Gaussian(nu_GHz, sigma_GHz)

    return prefactor * phi_ns


def compute_optical_depth_analytical_radiation_domination_Natural(
    lambda_vals: jnp.ndarray,   # (N_lambda,) [Å]
    lambda_rest: jnp.ndarray,   # (N_lines,)  [Å]
    A21_GHz: jnp.ndarray,       # (N_lines,)  [GHz]
    f_osc: jnp.ndarray,         # (N_lines,)  dimensionless
    a0: jnp.ndarray,            # (N_sk,)     scale factor at observer
    x0_Mpc: jnp.ndarray,        # (N_sk,)     comoving distance to observer [Mpc]
    xem_Mpc: jnp.ndarray,       # (N_sk,)     comoving distance to emission [Mpc]
    xf_Mpc: jnp.ndarray,        # (N_sk,)     comoving distance to front [Mpc]
    n0_cm3: jnp.ndarray,        # (N_sk,)     number density at observer [cm^-3]
) -> jnp.ndarray:
    """
    Compute optical depth for natural (Lorentzian) line broadening in radiation-dominated era.
    
    Returns:
        Optical depth array of shape (N_lines, N_sk, N_lambda)
    """
    # --- Broadcast to common shape: (N_lines, N_sk, N_lambda) ---
    lambda_vals = lambda_vals[None, None, :]  # (1, 1, N_lambda)
    lambda_rest = lambda_rest[:, None, None]  # (N_lines, 1, 1)
    A21_GHz = A21_GHz[:, None, None]          # (N_lines, 1, 1)
    f_osc = f_osc[:, None, None]              # (N_lines, 1, 1)
    a0 = a0[None, :, None]                    # (1, N_sk, 1)
    x0_Mpc = x0_Mpc[None, :, None]            # (1, N_sk, 1)
    xem_Mpc = xem_Mpc[None, :, None]          # (1, N_sk, 1)
    xf_Mpc = xf_Mpc[None, :, None]            # (1, N_sk, 1)
    n0_cm3 = n0_cm3[None, :, None]            # (1, N_sk, 1)

    # --- Compute prefactor ----------------------------------------------------
    # cross_section_prefactor * Gyr_to_s * 1e9 = σ₀ in cm²·GHz
    # Divide by π for Lorentzian normalization
    constants_prefactor = (phys.cross_section_prefactor * phys.Gyr_to_s * 1e9) / (jnp.pi * phys.H0)  # [cm²]
    
    # Final prefactor: converts to dimensionless optical depth
    prefactor = constants_prefactor * 1e-8 * f_osc * n0_cm3 * lambda_vals / a0

    # --- Compute line profile terms -------------------------------------------
    gamma_GHz = A21_GHz / (4 * jnp.pi)  # Lorentzian HWHM [GHz]
    
    # Helper to compute λ' at given comoving distance
    def compute_shifted_lambda(x_Mpc):
        return lambda_vals * ((phys.H0_over_c / a0) * (x_Mpc - x0_Mpc) + 1.0)  # [Å]
    
    # Helper to compute arctan term at given comoving distance
    def compute_arctan_term(x_Mpc):
        lambda_prime = compute_shifted_lambda(x_Mpc)  # [Å]
        # Frequency offset from line center: ν - ν₀ [GHz]
        nu_offset_GHz = (1 / lambda_rest - 1 / lambda_prime) * phys.c * 10
        return jnp.arctan(nu_offset_GHz / gamma_GHz)
    
    # Evaluate at integration limits
    term_front = compute_arctan_term(xf_Mpc)
    term_emission = compute_arctan_term(xem_Mpc)
    
    return prefactor * (term_front - term_emission)


def compute_optical_depth_analytical_radiation_domination_Thermal(
    lambda_vals: jnp.ndarray,   # (N_lambda,) [Å]
    lambda_rest: jnp.ndarray,   # (N_lines,)  [Å]
    f_osc: jnp.ndarray,         # (N_lines,)  dimensionless
    mass_amu: jnp.ndarray,      # (N_lines,)  mass [amu]
    T0: jnp.ndarray,            # (N_sk,)     gas temperature [K]
    a0: jnp.ndarray,            # (N_sk,)     scale factor at observer
    x0_Mpc: jnp.ndarray,        # (N_sk,)     comoving distance to observer [Mpc]
    xem_Mpc: jnp.ndarray,       # (N_sk,)     comoving distance to emission [Mpc]
    xf_Mpc: jnp.ndarray,        # (N_sk,)     comoving distance to front [Mpc]
    n0_cm3: jnp.ndarray,        # (N_sk,)     number density at observer [cm^-3]
) -> jnp.ndarray:
    """
    Compute optical depth for thermal (Gaussian) line broadening in radiation-dominated era.
    
    Returns:
        Optical depth array of shape (N_lines, N_sk, N_lambda)
    """
    # --- Broadcast to common shape: (N_lines, N_sk, N_lambda) ---
    lambda_vals = lambda_vals[None, None, :]  # (1, 1, N_lambda)
    lambda_rest = lambda_rest[:, None, None]  # (N_lines, 1, 1)
    f_osc = f_osc[:, None, None]              # (N_lines, 1, 1)
    mass_amu = mass_amu[:, None, None]        # (N_lines, 1, 1)
    T0 = T0[None, :, None]                    # (1, N_sk, 1)
    a0 = a0[None, :, None]                    # (1, N_sk, 1)
    x0_Mpc = x0_Mpc[None, :, None]            # (1, N_sk, 1)
    xem_Mpc = xem_Mpc[None, :, None]          # (1, N_sk, 1)
    xf_Mpc = xf_Mpc[None, :, None]            # (1, N_sk, 1)
    n0_cm3 = n0_cm3[None, :, None]            # (1, N_sk, 1)

    # --- Compute prefactor ----------------------------------------------------
    # Same as natural but with factor 2 instead of π for Gaussian normalization
    constants_prefactor = (phys.cross_section_prefactor * phys.Gyr_to_s * 1e9) / (2.0 * phys.H0)  # [cm²]
    
    prefactor = constants_prefactor * 1e-8 * f_osc * n0_cm3 * lambda_vals / a0

    # --- Compute thermal broadening width -------------------------------------
    # Convert mass from amu to kg for proper thermal velocity calculation
    
    
    # Thermal velocity: v_th = √(2k_B T / m); σ_ν = (v_th / c) * ν₀
    v_eff_A_GHz = jnp.sqrt(2.0 * phys.kB * T0 / mass_amu) * 10.0
    sigma_GHz = v_eff_A_GHz / (jnp.sqrt(2.0) * lambda_rest)

    # --- Compute error function terms -----------------------------------------
    # Helper to compute λ' at given comoving distance
    def compute_shifted_lambda(x_Mpc):
        return lambda_vals * ((phys.H0_over_c / a0) * (x_Mpc - x0_Mpc) + 1.0)  # [Å]
    
    # Helper to compute erf term at given comoving distance
    def compute_erf_term(x_Mpc):
        lambda_prime = compute_shifted_lambda(x_Mpc)  # [Å]
        # Frequency offset from line center: ν - ν₀ [GHz]
        nu_offset_GHz = (1 / lambda_rest - 1 / lambda_prime) * phys.c * 10
        # Argument of error function: (ν - ν₀) / (√2 σ_ν)
        arg = nu_offset_GHz / (jnp.sqrt(2.0) * sigma_GHz)
        return jax.scipy.special.erf(arg)
    
    # Evaluate at integration limits
    term_front = compute_erf_term(xf_Mpc)
    term_emission = compute_erf_term(xem_Mpc)
    
    return prefactor * (term_front - term_emission)


def compute_mid_point_tau_trapezoidal(
    line_bundle,
    T0_K,
    n0_cm3,
    Delta_x_Mpc,
    a0,
    method: str = "trapezoidal"
):    
    
    from skewerjax.optical_depth.voigt import log10_voigt

    f_osc = line_bundle.f_osc           # N_lines
    lambda0_A = line_bundle.lambda0_A   # N_lines
    A21_GHz = line_bundle.A21_GHz       # N_lines
    mass_amu = line_bundle.mass_amu     # N_lines

    if method == "trapezoidal":
        _term = Delta_x_Mpc / 2
    elif method == "analytical_deltas":
        _term = 1.0
    else:
        raise NotImplementedError
    
    N_lines = f_osc.shape[0]
    N_T0 = T0_K.shape[0]
    N_xx = Delta_x_Mpc.shape[0]

    # --- Broadcast to common shape: (N_lines, N_T0, N_n0, N_xx) ---
    T0_K = T0_K[None, :, None, None]
    n0_cm3 = n0_cm3[None, None, :, None]
    f_osc = f_osc[:, None, None, None]
    lambda0_A = lambda0_A[:, None, None, None]
    A21_GHz = A21_GHz[:, None, None, None]
    mass_amu = mass_amu[:, None, None, None]
    Delta_x_Mpc = Delta_x_Mpc[None, None, None, :]

    v_eff_A_GHz = jnp.sqrt(2.0 * phys.kB * T0_K / mass_amu) * 10.0
    sigma_GHz = v_eff_A_GHz / (2.0 * lambda0_A) # (N_lines, N_T0, 1, 1)
    gamma_GHz = A21_GHz / (4.0 * jnp.pi)        # (N_lines, 1, 1, 1)

    # Tile gamma_GHz to (N_lines * N_T0), then flatten to (N_lines * N_T0)
    gamma_GHz_tiled = jnp.tile(gamma_GHz, (1, N_T0, 1, 1))
    gamma_GHz_flat = gamma_GHz_tiled.reshape(-1)
    # Flatten sigma_GHz to (N_lines * N_T0)
    sigma_GHz_flat = sigma_GHz.reshape(-1)
    # flatten Delta_xx_Mpc
    nu_GHz = (phys.c * 10.0 / lambda0_A) * (1 - a0 / (a0 + 0.5 * phys.H0_over_c * Delta_x_Mpc)) # GHz, (N_lines, 1, 1, N_xx)
    nu_GHz_tiled = jnp.tile(nu_GHz, (1, N_T0, 1, 1))
    nu_flat_GHz = jnp.reshape(nu_GHz_tiled, (N_lines*N_T0, N_xx))
    # compute phi using Voigt profile
    voigt_batched = vmap(lambda x, s, g: 10**log10_voigt(x, s, g), in_axes=(0, 0, 0))
    phi_flat_ns = voigt_batched(nu_flat_GHz, sigma_GHz_flat, gamma_GHz_flat) # (N_lines * N_T0, N_xx)
    # Reshape to (N_lines, N_T0, 1, N_xx)
    phi_ns = phi_flat_ns.reshape(N_lines, N_T0, 1, N_xx)

    C_j_cm2_GHz = phys.cross_section_prefactor * f_osc
    rhs = (2 * n0_cm3 * C_j_cm2_GHz * _term) * (1e2 * phys.Mpc_to_m) # GHz
    rhs *= (a0 + 0.5 * phys.H0_over_c * Delta_x_Mpc)**-2             # GHz
    mid_point_tau = rhs * phi_ns                                     # adim

    return mid_point_tau


def compute_mid_point_tau_radiation_domination_Thermal(
    line_bundle,
    T0_K,
    n0_cm3,
    Delta_x_Mpc,
    a0
):    

    f_osc = line_bundle.f_osc           # N_lines
    lambda0_A = line_bundle.lambda0_A   # N_lines
    mass_amu = line_bundle.mass_amu     # N_lines
    
    N_lines = f_osc.shape[0]
    N_T0 = T0_K.shape[0]
    N_xx = Delta_x_Mpc.shape[0]

    # --- Broadcast to common shape: (N_lines, N_T0, N_n0, N_xx) ---
    T0_K = T0_K[None, :, None, None]
    n0_cm3 = n0_cm3[None, None, :, None]
    f_osc = f_osc[:, None, None, None]
    lambda0_A = lambda0_A[:, None, None, None]
    mass_amu = mass_amu[:, None, None, None]
    Delta_x_Mpc = Delta_x_Mpc[None, None, None, :]

    C_j_cm2_GHz = phys.cross_section_prefactor * f_osc

    prefactor = n0_cm3 * C_j_cm2_GHz * lambda0_A / (a0 * phys.H0) * 10 * phys.Gyr_to_s # [adim]

    v_eff_A_GHz = jnp.sqrt(2.0 * phys.kB * T0_K / mass_amu) * 10.0
    sigma_GHz = v_eff_A_GHz / (2.0 * lambda0_A) # (N_lines, N_T0, 1, 1)

    arg = phys.c * 10 / (sigma_GHz * lambda0_A * jnp.sqrt(2.0)) # [adim]
    arg *= (1 - a0 / (a0 + 0.5 * phys.H0_over_c * Delta_x_Mpc)) # [adim]
    
    return prefactor * jax.scipy.special.erf(arg)