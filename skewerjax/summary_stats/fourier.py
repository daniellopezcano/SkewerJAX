"""The Fourier conventions of the two-point statistics: lag and mode axes, and the Wiener-Khinchin pair P <-> xi.

Conventions (review notes, Eqs. P1D_delta_tilde_sample, P1D_sample_definition,
P1D_ensemble_translation_invariant, xi1D_from_P1D_ensemble; generalised to d FFT axes)
-----------------------------------------------------------------------------------
Grid: N = prod(N_i) cells of size dV = prod(d_i) per realisation, V = N dV; forward FFT
sign e^{-i k.x} (``jnp.fft``), angular wavenumbers k = 2 pi fftfreq(N_i, d_i).

    P_ab(k)  = (dV / N) F_a(k) F_b(k)^*,       F = rfftn(field)   (P_ab(-k) = P_ab(k)^*)
    xi_ab(r) = (1/N) sum_x a(x) b(x + r)        (periodic estimator)
             = irfftn( P_ab(k)^* ) / dV         (exact discrete Fourier pair)

so that P_ab(k) = dV sum_r xi_ab(r) e^{+i k.r}, the notes' sign (P = int xi e^{+i w Delta}).
Cross spectra are complex; P_ab + P_ba = 2 Re P_ab is the notes' ``sym`` combination.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np


def lag_axis(n: int, d: float) -> np.ndarray:
    """Lags of an fftshift-ed correlation function: (j - n//2) d, j = 0..n-1."""
    return (np.arange(n) - n // 2) * d


def rfft_multiplicity(n: int) -> np.ndarray:
    """Full-grid modes represented by each rfft column: 2, except k = 0 and Nyquist (1)."""
    m = np.full(n // 2 + 1, 2.0)
    m[0] = 1.0
    if n % 2 == 0:
        m[-1] = 1.0
    return m


@partial(jax.jit, static_argnames=("n",))
def xi_from_power(P, n: int, d: float):
    """xi = irfft(P^*, n) / d along the last axis (raw FFT lag order)."""
    return jnp.fft.irfft(jnp.conj(P), n=n, axis=-1) / d


@partial(jax.jit, static_argnames=("shape",))
def xi3d_from_power(P, shape: tuple, dV: float):
    """xi(r) = fftshift(irfftn(P^*)) / dV on the full (Nx, Ny, Nz) lag grid, centred."""
    return jnp.fft.fftshift(jnp.fft.irfftn(jnp.conj(P), s=shape, axes=(-3, -2, -1)), axes=(-3, -2, -1)) / dV
