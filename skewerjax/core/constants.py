"""Physical constants and unit conversions (``phys``)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PhysicalConstants:
    """Immutable physical constants and the unit conversions used by SkewerJAX (SI unless stated)."""

    c: float = 2.99792458e8  # speed of light [m/s]
    c_kms: float = 2.99792458e5  # speed of light [km/s]
    kB: float = 8314.46262102654  # Boltzmann constant / atomic mass unit [m^2 s^-2 K^-1]
    kB_erg_per_K: float = 1.380649e-16  # Boltzmann constant [erg/K]
    qe: float = 1.60217663e-19  # electron charge [C]
    me: float = 9.1093837e-31  # electron mass [kg]
    mp: float = 1.67262192369e-24  # proton mass [g]
    epsilon_0: float = 8.85418782e-12  # vacuum permittivity [C^2 kg^-1 m^-3 s^2]

    Gpc_to_m: float = 3.085678e25  # [m / Gpc]
    Mpc_to_m: float = 3.085678e22  # [m / Mpc]
    Mpc_to_cm: float = 3.085678e24  # [cm / Mpc] (the Arepo / fake_spectra value)
    Gyr_to_s: float = 3.1536e16  # [s / Gyr]
    g_to_amu: float = 1 / 1.66053906660e-24  # [amu / g]

    H0_km_s_mpc: float = 70.0  # default Hubble constant [km/s/Mpc]

    @property
    def H0(self) -> float:
        """Hubble constant [1/Gyr]."""
        return self.H0_km_s_mpc * 1e6 * self.Gyr_to_s / self.Gpc_to_m

    @property
    def cc_Gpc_per_Gyr(self) -> float:
        """Speed of light [Gpc/Gyr]."""
        return self.c * self.Gyr_to_s / self.Gpc_to_m

    @property
    def H0_over_c(self) -> float:
        """H0 / c [1/Mpc]."""
        return self.H0 / self.cc_Gpc_per_Gyr * 1e-3

    @property
    def cross_section_prefactor(self) -> float:
        """q_e^2 / (4 epsilon_0 m_e c) [cm^2 / ns]: the C_j of notes Eq. tau_straight_los without f_osc."""
        return (self.qe**2 / (4 * self.epsilon_0 * self.me * self.c)) * 1e-5


phys = PhysicalConstants()
