from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Protocol, runtime_checkable
import numpy as np

def min_image_1d(dx: np.ndarray, L: float) -> np.ndarray:
    """Minimum-image displacement on a 1D periodic domain of length L."""
    return (dx + 0.5 * L) % L - 0.5 * L


@dataclass(frozen=True)
class SimplePeriodicFluxSampler:
    """
    Schematic flux sampler producing a quasi-periodic comb of absorption dips,
    with per-realization jitter in dip positions and widths.

    Core model (flux-space):
        F(x) = 1 - depth * max_k exp( -0.5 * ((x - c_k)/sigma_k)^2 )

    Parameters
    ----------
    x_a, x_b : float
        Domain boundaries.
    n_x : int
        Number of sample points.
    periodic : bool
        If True, treat the domain as periodic (wrap distances).
    period : float
        Mean spacing between dips.
    width : float
        Mean Gaussian width of each dip.
    depth : float
        Dip depth in flux-space. Must satisfy 0 < depth < 1.
    phase : float or None
        Global shift of the pattern. If None, drawn randomly per realization
        as Uniform(0, period).
    jitter_pos : float
        Std-dev of per-dip position jitter (in x units).
    jitter_width_frac : float
        Std-dev of per-dip *fractional* width jitter. Widths are:
            sigma_k = width * (1 + jitter_width_frac * N(0,1))
        clipped to be positive.
    """
    x_a: float
    x_b: float
    n_x: int = 2000
    periodic: bool = True

    period: float = 20.0
    width: float = 2.0
    depth: float = 0.95
    phase: Optional[float] = None

    jitter_pos: float = 0.0
    jitter_width_frac: float = 0.0

    def __post_init__(self):
        if not (0.0 < self.depth < 1.0):
            raise ValueError(f"`depth` must be in (0,1). Got depth={self.depth}.")
        if self.period <= 0:
            raise ValueError(f"`period` must be > 0. Got period={self.period}.")
        if self.width <= 0:
            raise ValueError(f"`width` must be > 0. Got width={self.width}.")
        if self.jitter_width_frac < 0:
            raise ValueError("`jitter_width_frac` must be >= 0.")
        if self.jitter_pos < 0:
            raise ValueError("`jitter_pos` must be >= 0.")

    def x_grid(self) -> np.ndarray:
        if self.periodic:
            return np.linspace(self.x_a, self.x_b, self.n_x, endpoint=False)
        return np.linspace(self.x_a, self.x_b, self.n_x)

    def sample(self, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (x, F) for realization indexed by seed.
        """
        rng = np.random.default_rng(seed)
        x = self.x_grid()
        L = self.x_b - self.x_a

        # Phase: fixed if provided, otherwise sampled randomly in [0, period)
        phase_s = self.phase if (self.phase is not None) else rng.uniform(0.0, self.period)

        # Base dip centers (comb)
        k_max = int(np.ceil(L / self.period)) + 4
        k = np.arange(-2, k_max - 2)
        centers = self.x_a + phase_s + k * self.period  # not wrapped on purpose

        # Add per-dip position jitter (same seed -> same jitter pattern)
        if self.jitter_pos > 0:
            centers = centers + self.jitter_pos * rng.normal(size=centers.shape)

        # Per-dip widths
        if self.jitter_width_frac > 0:
            sigmas = self.width * (1.0 + self.jitter_width_frac * rng.normal(size=centers.shape))
            sigmas = np.clip(sigmas, 1e-6, None)  # ensure positive
        else:
            sigmas = np.full_like(centers, self.width, dtype=float)

        # Evaluate dips
        dx = x[None, :] - centers[:, None]  # (n_centers, n_x)
        if self.periodic:
            dx = min_image_1d(dx, L)

        dips = np.exp(-0.5 * (dx / sigmas[:, None]) ** 2)  # each in [0,1]
        profile = np.max(dips, axis=0)                     # nearest/strongest dip
        F = 1.0 - self.depth * profile
        F = np.clip(F, 0.0, 1.0)

        return x, F

@runtime_checkable
class FluxSampler1D(Protocol):
    x_a: float
    x_b: float
    n_x: int
    periodic: bool

    def x_grid(self) -> np.ndarray: ...
    def sample(self, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]: ...


@dataclass(frozen=True)
class FractionalFluctuationFieldSampler:
    """
    Wrap a flux sampler F(x) -> produce fractional fluctuation field:
        δ(x) = F(x) / <F> - 1

    Mean definition follows the manuscript:
        <F> = (1/L) ∫_{x_a}^{x_b} F(x) dx,  L = x_b - x_a

    Notes
    -----
    - Uses trapezoidal integration on the provided (possibly non-uniform) x-grid.
    - If `periodic_integration=True` and the underlying sampler is periodic and the grid
      does not include x_b, we "close the loop" by appending (x_b, F(x_a)) to better
      approximate the integral over a full period.
    """
    flux_sampler: FluxSampler1D
    periodic_integration: bool = True
    eps_mean: float = 1e-12  # safety in case <F> gets extremely small (shouldn't happen for flux)

    def x_grid(self) -> np.ndarray:
        return self.flux_sampler.x_grid()

    def mean_flux(self, x: np.ndarray, F: np.ndarray) -> float:
        """Compute <F> = (1/L) ∫ F dx using trapezoidal rule (robust to non-uniform grids)."""
        x = np.asarray(x)
        F = np.asarray(F)
        if x.ndim != 1 or F.ndim != 1 or x.shape != F.shape:
            raise ValueError(f"x and F must be 1D with same shape. Got x={x.shape}, F={F.shape}")

        # Ensure sorted x for integration
        if not np.all(np.diff(x) >= 0):
            idx = np.argsort(x)
            x = x[idx]
            F = F[idx]

        L = float(self.flux_sampler.x_b - self.flux_sampler.x_a)
        if L <= 0:
            raise ValueError(f"Invalid domain length L={L}. Check x_a, x_b.")

        # Optionally close the integral for periodic sampling if endpoint x_b is not included
        if self.periodic_integration and getattr(self.flux_sampler, "periodic", False):
            # If the grid doesn't include x_b (common with endpoint=False), append endpoint
            if x[-1] < self.flux_sampler.x_b:
                x_ext = np.concatenate([x, [self.flux_sampler.x_b]])
                F_ext = np.concatenate([F, [F[0]]])  # periodic wrap value
                integral = np.trapz(F_ext, x_ext)
            else:
                integral = np.trapz(F, x)
        else:
            integral = np.trapz(F, x)

        return float(integral / L)

    def sample(
        self,
        seed: int = 0,
        *,
        return_flux: bool = False,
        return_mean: bool = False,
    ):
        """
        Sample one realization and return δ(x). Optionally also return F and <F>.

        Returns
        -------
        (x, delta) by default.
        If return_flux: also returns F.
        If return_mean: also returns mean_F.
        """
        x, F = self.flux_sampler.sample(seed=seed)
        mean_F = self.mean_flux(x, F)
        mean_F = mean_F if abs(mean_F) > self.eps_mean else (self.eps_mean * np.sign(mean_F) if mean_F != 0 else self.eps_mean)

        delta = F / mean_F - 1.0

        out = (x, delta)
        if return_flux:
            out = out + (F,)
        if return_mean:
            out = out + (mean_F,)
        return out
