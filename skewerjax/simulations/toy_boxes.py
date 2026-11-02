"""Toy boxes (notebooks 08 and 09): periodic 3D Gaussian fields of known spectrum and their gas, regenerated from their
seeds whenever they are needed: nothing is stored, every call recomputes.

``ToyBoxes``: the configuration and realisation i, regenerated from its seed ``base_seed + i`` (``delta``, ``fields``,
``cells``). The spectrum is a power law plus a BAO term (a Gaussian bump in xi, its exact transform A_bao W(k) in P,
``PowerLawGRF``; valid for bao_ratio <= R_max), isotropic (no Kaiser or Fingers-of-God term), normalised to
``sigma_delta``; the gas of every cell, both species being the same gas (one temperature, one velocity):

    n_HI = n_bar_HI (1 + delta),   n_SiIII = r n_HI,   T = T0 (1 + delta)^(gamma - 1),
    v = 0 (``velocity="zero"``, the default)  or  v_k = i a H f (k / k^2) delta_k (``"linear"``: linear theory, notebook 04).

The gas only: the skewers through the boxes are ``skewerjax.skewers`` (``lattice_columns``, ``lattice_points``, then a
``core.SkewerBundle`` with ``ToyBoxes.cosmology`` and a0 = 1 / (1 + z)), and their statistics ``skewerjax.summary_stats``
(``field_statistics``, ``skewer_density_statistics``, ``lattice_density_theory`` with ``grf().power`` as the
expectation). Notebooks compose the three.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import jax
import jax.numpy as jnp
import numpy as np

from skewerjax.core import Cosmology, phys
from skewerjax.simulations.grf import PowerLawGRF, _bao_ratio_max
from skewerjax.simulations.toy_cells import grf_cells

LINE_IDS = ("HI-LyA", "SiIII-1206")  # the lines of the two species (HI, SiIII): line 1, line 2 of the terms "1", "2", "12"
VELOCITY_MODES = ("zero", "linear")  # ToyBoxes.velocity


@dataclass(frozen=True)
class ToyBoxes:
    """The toy boxes of notebooks 08 and 09: ``n_real`` periodic cubes of side ``box_cMpc`` on ``n_cells``^3 grids.

    Spectrum: P(k) = A_s (k / k_0)^(n_s - 1) + A_bao W(k), isotropic (``PowerLawGRF`` with A_bao = bao_ratio A_s,
    r_bao, sigma_bao [cMpc]): ``bao_ratio`` = R_bao is the height of the BAO bump in xi in units of A_s [cMpc^-3], and
    W the exact transform of the bump, which changes sign: the spectrum is valid (P >= 0 at every mode of the grid) for
    bao_ratio <= R_max (``bao_ratio_max``, independent of sigma_delta); a larger ratio raises a ``ValueError`` quoting
    R_max. A_s is set by ``sigma_delta`` = sqrt(variance on the grid). Gas: the module
    docstring, with n_bar_HI [comoving cm^-3], ``siiii_to_hi`` = r, T0 [K], ``gamma_minus_1``. ``z`` and ``Omega_m``
    (a flat cosmology, H0 = ``phys.H0_km_s_mpc``: ``cosmology``, ``aH``) set a(z) H(z), the Hubble flow of skewers through
    the boxes (``skewerjax.skewers``): in notebook 09 they are in the exact frame, ``z`` at y = 0 (their first point,
    a0 = 1 / (1 + z)), a(y) decreasing along y from the cosmology (Delta ln a = v_per / c over one box), wrapped as
    periodic by ``velocity_period=True``.
    Realisation i is ``grf().sample(jax.random.PRNGKey(base_seed + i))``.

    velocity : "zero" (default) or "linear"
        The peculiar velocities of the gas: "zero" sets them to 0 in every cell (the flux then follows the density
        and the temperature only); "linear" uses linear theory, v_k = i a H f (k / k^2) delta_k [km/s] (notebook 04),
        with the growth rate ``f_growth``, which matters for this option only.
    """

    box_cMpc: float
    n_cells: int = 256
    n_real: int = 96
    base_seed: int = 1000
    sigma_delta: float = 0.15
    n_s: float = -1.0
    k_0: float = 2.0
    r_bao: float = 147.1
    sigma_bao: float = 1.0
    bao_ratio: float = 0.002
    n_bar_HI: float = 4e-13
    siiii_to_hi: float = 0.2
    T0: float = 1.5e4
    gamma_minus_1: float = 0.6
    z: float = 3.0
    Omega_m: float = 0.31
    f_growth: float = 1.0
    velocity: str = "zero"

    def __post_init__(self):
        if self.velocity not in VELOCITY_MODES:
            raise ValueError(f"velocity must be one of {VELOCITY_MODES}, got {self.velocity!r}.")
        r_max = self.bao_ratio_max
        if self.bao_ratio > r_max * (1.0 + 1e-12):
            raise ValueError(
                f"bao_ratio = {self.bao_ratio:g} exceeds R_max = {r_max:.6g}, the largest BAO ratio with P(k) >= 0 at every "
                f"mode of this grid (L = {self.box_cMpc:g} cMpc, N = {self.n_cells}): lower bao_ratio."
            )

    # ------------------------------------------------------------------ the spectrum and the realisations
    @property
    def bao_ratio_max(self) -> float:
        """R_max [cMpc^-3]: the largest bao_ratio with P(k) = A_s [(k / k_0)^(n_s - 1) + R_bao W(k)] >= 0 at every mode
        of the grid (``PowerLawGRF.bao_ratio_max``); independent of A_s, so of sigma_delta."""
        L, N = (float(self.box_cMpc),) * 3, (int(self.n_cells),) * 3
        return _bao_ratio_max(N, L, float(self.k_0), float(self.n_s), float(self.r_bao), float(self.sigma_bao))

    @cached_property
    def A_s(self) -> float:
        """The amplitude that gives sigma_delta on the grid (the variance is linear in A_s at fixed bao_ratio)."""
        return self.sigma_delta**2 / float(self._grf(1.0).variance())

    def _grf(self, A_s: float) -> PowerLawGRF:
        L, N = (float(self.box_cMpc),) * 3, (int(self.n_cells),) * 3
        return PowerLawGRF(
            A_s, self.k_0, self.n_s, L=L, N=N, A_bao=self.bao_ratio * A_s, r_bao=self.r_bao, sigma_bao=self.sigma_bao, los_axis=1
        )

    def grf(self) -> PowerLawGRF:
        """The ``PowerLawGRF`` of every realisation (line of sight: axis 1, y)."""
        return self._grf(self.A_s)

    @property
    def seeds(self) -> np.ndarray:
        """The seed of every realisation: base_seed + i, i = 0 .. n_real - 1."""
        return self.base_seed + np.arange(self.n_real)

    def key(self, i: int):
        return jax.random.PRNGKey(int(self.base_seed + i))

    def delta(self, i: int) -> np.ndarray:
        """Realisation i (n_cells^3), regenerated from its seed."""
        return np.asarray(self.grf().sample(self.key(i)))

    @property
    def dx(self) -> float:
        """Cell side [cMpc]."""
        return float(self.box_cMpc) / self.n_cells

    @property
    def aH(self) -> float:
        """a H(z) [km/s per cMpc] (flat cosmology, H0 = phys.H0_km_s_mpc): the Hubble flow at the start of a skewer."""
        E = np.sqrt(self.Omega_m * (1 + self.z) ** 3 + 1 - self.Omega_m)
        return phys.H0_km_s_mpc * E / (1 + self.z)

    @property
    def aHf(self) -> float:
        """a H(z) f [km/s per cMpc]: the factor of the linear-theory velocities (``velocity="linear"``)."""
        return self.aH * self.f_growth

    def cosmology(self) -> Cosmology:
        return Cosmology(H0=phys.H0, Omega_r_0=0.0, Omega_m_0=self.Omega_m, Omega_k_0=0.0, Omega_lambda_0=1.0 - self.Omega_m)

    def fields(self, i: int) -> dict:
        """The gas of realisation i on the grid (NumPy): delta, n {species: comoving cm^-3}, T [K], v (N, N, N, 3) [km/s]
        (zeros, or linear theory: ``velocity``). Both species are the same gas: one T and one v."""
        g = self.grf()
        d = g.sample(self.key(i))
        if self.velocity == "linear":
            k_vec, k_mod = g.k_grid(real=True)
            k2 = k_mod**2
            fk = jnp.fft.rfftn(d)
            v = jnp.stack([jnp.fft.irfftn(1j * self.aHf * kv / jnp.where(k2 > 0, k2, 1.0) * fk, s=g.shape) for kv in k_vec], -1)
        else:
            v = np.zeros(g.shape + (3,), dtype=np.asarray(d).dtype)
        d = np.asarray(d)
        if d.min() <= -1.0:  # no clipping: a negative density has no temperature; lower sigma_delta instead
            raise ValueError(
                f"realisation {i}: min(1 + delta) = {1 + d.min():.3f} <= 0 (sigma_delta = {self.sigma_delta}); lower sigma_delta."
            )
        n_HI = self.n_bar_HI * (1.0 + d)
        return dict(
            delta=d,
            n={"HI": n_HI, "SiIII": self.siiii_to_hi * n_HI},
            T=self.T0 * (1.0 + d) ** self.gamma_minus_1,
            v=np.asarray(v),
        )

    def cells(self, i: int, fields: dict | None = None):
        """``core.Cells`` of realisation i: the grid cells (``grf_cells``), n of both species, the shared T and v."""
        f = self.fields(i) if fields is None else fields
        n_bar = {"HI": self.n_bar_HI, "SiIII": self.siiii_to_hi * self.n_bar_HI}
        return grf_cells(f["delta"], self.grf(), n_bar, temperature=f["T"], velocity=f["v"])
