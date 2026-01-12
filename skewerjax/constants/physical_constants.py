from dataclasses import dataclass

@dataclass(frozen=True)
class PhysicalConstants:
    """
    Immutable container for physical constants with cosmological unit conversions.
    
    Provides both fundamental physical constants and derived cosmological
    quantities calculated consistently from base units.
    """
    
    # Fundamental physical constants
    c: float = 2.99792458e8                         # speed of light      — UNITS: [m / s]
    kB: float = 8314.46262102654                    # Boltzmann constant  — UNITS: [amu ⋅ m^2 ⋅ s^-2 ⋅ K^-1]
    qe: float = 1.60217663e-19                      # electron charge     — UNITS: [C]
    me: float = 9.1093837e-31                       # electron mass       — UNITS: [kg]
    epsilon_0: float = 8.85418782e-12               # vacuum permittivity — UNITS: [C^2 ⋅ kg^−1 ⋅ m^−3 ⋅ s^2]
    
    # Unit conversions for cosmology
    Gpc_to_m: float = 3.085678e25                   # 1 Gpc in meters     — UNITS: [m / Gpc]
    Mpc_to_m: float = 3.085678e22                   # 1 Mpc in meters     — UNITS: [m / Mpc]
    Gyr_to_s: float = 3.1536e16                     # 1 Gyr in seconds    — UNITS: [s / Gyr]
    
    # Cosmological parameters (default ΛCDM values)
    H0_km_s_mpc: float = 70.0                       # Hubble constant     — UNITS: [km / s / Mpc]
    
    @property
    def H0(self) -> float:
        """
        Hubble constant in 1/Gyr units.
        """
        return self.H0_km_s_mpc * 1e6 * self.Gyr_to_s / self.Gpc_to_m
    
    @property
    def cc_Gpc_per_Gyr(self) -> float:
        """
        Speed of light in Gpc/Gyr units.
        """
        return self.c * self.Gyr_to_s / self.Gpc_to_m
    
    @property
    def H0_over_c(self) -> float:
        """
        Hubble constant over speed of light in 1/Mpc units.
        """
        return self.H0 / self.cc_Gpc_per_Gyr * 1e-3
    
    @property
    def cross_section_prefactor(self) -> float:
        """
        units of [cm² / ns]
        """
        return (self.qe**2 / (4 * self.epsilon_0 * self.me * self.c))*1e-5

# single shared instance (immutable)
phys = PhysicalConstants()