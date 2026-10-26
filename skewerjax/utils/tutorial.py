"""Small toy set-ups of the tutorial notebooks (NumPy), kept here so the notebooks stay readable.

* ``p1d_expected``, ``p3d_expected_on_skewer_grid``: the expected spectra of a GRF along skewers (discrete transverse
  projection or aliasing, and the window of the representation along the line of sight: piecewise-constant cells,
  ``cell_window``, or linear interpolation, ``linear_interp_window``).
* ``binned_input_power``: the input power of a GRF averaged in |k| bins.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


# --------------------------------------------------------------------------- expectations for the GRF study of notebook 08
def linear_interp_window(k, dx):
    """|T(k)|^2 / dx^2 = sinc^4(k dx / 2) of linear interpolation between nodes dx apart (k [1/length])."""
    return np.sinc(np.asarray(k, dtype=float) * dx / (2 * np.pi)) ** 4


def cell_window(k, dx):
    """|T(k)|^2 / dx^2 = sinc^2(k dx / 2) of piecewise-constant cells of size dx along the line (k [1/length]): a top
    hat of width dx around every node (the grid cells of ``simulations.grf_cells``)."""
    return np.sinc(np.asarray(k, dtype=float) * dx / (2 * np.pi)) ** 2


WINDOWS = {None: None, "cells": cell_window, "linear": linear_interp_window}


def fold(k, dx):
    """k folded into the band [-pi/dx, pi/dx) of a grid of spacing dx (the periodic spectrum of node samples)."""
    kn = np.pi / dx
    return np.mod(np.asarray(k, dtype=float) + kn, 2 * kn) - kn


def p1d_expected(grf, k_par, axis: int = 1, window: Optional[str] = None, return_sigma: bool = False, step: int = 1):
    """Expected P1D of a 3D ``PowerLawGRF`` delta along grid lines parallel to ``axis`` [length], averaged over the lines
    at every ``step``-th grid column (``step=1``: all of them).

    The mean is the discrete projection over the grid's transverse modes (``plotting.grf_p1d_along_axis``) at the nodes
    (``window=None``). Between the nodes the node spectrum, periodic in k_par with period 2 pi / dx, is multiplied by
    the window of the representation: ``"cells"``, piecewise-constant grid cells (sinc^2(k dx / 2), ``cell_window``),
    or ``"linear"``, linear interpolation (sinc^4(k dx / 2)).
    ``return_sigma``: also the standard deviation of that average for one realisation. Every 3D mode is an independent
    complex Gaussian; the lines of a coarse transverse grid see the sum of the step^2 fine modes that alias onto each
    coarse mode j, so sigma = sqrt(sum_j (sum_aliases P)^2) / L_perp^2 (the mean does not depend on ``step``)."""
    tr = [i for i in range(3) if i != axis]
    N = [grf.shape[i] for i in tr]
    ka, kb = (2 * np.pi * np.fft.fftfreq(n, d=grf.dx[i]) for n, i in zip(N, tr))
    dx = grf.dx[axis]
    k = np.asarray(k_par, dtype=float)
    kp = np.abs(fold(k, dx))
    comps = {axis: kp[:, None, None], tr[0]: ka[None, :, None], tr[1]: kb[None, None, :]}
    P = np.asarray(grf.power_at(*(comps[i] for i in range(3))))                  # P(k, mu): mu along grf.los_axis
    area = grf.L[tr[0]] * grf.L[tr[1]]
    mean = P.sum((1, 2)) / area
    Nc = [n // step for n in N]
    Pc = P.reshape(-1, step, Nc[0], N[1]).sum(1).reshape(-1, Nc[0], step, Nc[1]).sum(2)
    sig = np.sqrt((Pc ** 2).sum((1, 2))) / area
    W = 1.0 if WINDOWS[window] is None else WINDOWS[window](k, dx)
    return (mean * W, sig * W) if return_sigma else mean * W


def p3d_expected_on_skewer_grid(grf, step: int, k_par, axis: int = 1, window: Optional[str] = "cells"):
    """Expected P3D (Nx, Ny, N_kpar) of delta on skewers along ``axis`` placed at every ``step``-th grid column, in the
    layout of ``OverfluxStatistics3D`` (transverse fftfreq of the coarse grid, k_par >= 0 [1/length]).

    Sampling the transverse plane every ``step`` nodes aliases the fine-grid modes onto the coarse ones: P_coarse(j) =
    sum over the fine modes i = j (mod N / step). Along the line of sight: the node spectrum folded into the fine grid's
    band and the window of the representation (``p1d_expected``; default ``"cells"``)."""
    tr = [i for i in range(3) if i != axis]
    N = [grf.shape[i] for i in tr]
    ka, kb = (2 * np.pi * np.fft.fftfreq(n, d=grf.dx[i]) for n, i in zip(N, tr))
    dpar = grf.dx[axis]
    kp = np.abs(fold(np.asarray(k_par, dtype=float), dpar))
    comps = {tr[0]: ka[:, None, None], tr[1]: kb[None, :, None], axis: kp[None, None, :]}
    P = np.asarray(grf.power_at(*(comps[i] for i in range(3))))
    Nc = [n // step for n in N]
    Pc = P.reshape(step, Nc[0], N[1], -1).sum(0)                       # fold axis a: fine index i = j + m Nc (fftfreq order)
    Pc = Pc.reshape(Nc[0], step, Nc[1], -1).sum(1)
    if WINDOWS[window] is not None:
        Pc = Pc * WINDOWS[window](np.asarray(k_par, dtype=float), dpar)[None, None, :]
    return Pc


def binned_input_power(grf, edges):
    """The input P of a ``PowerLawGRF`` (P(k, mu) if anisotropic) averaged over the grid's modes in each bin of |k| (the
    same weights as ``plotting.binned_power``): the exact expectation of a binned measurement, free of the curvature of
    P inside a bin."""
    _, kmod = grf.k_grid(real=True)
    kmod = np.asarray(kmod)
    n_last = grf.shape[-1]
    w1 = np.full(n_last // 2 + 1, 2.0)
    w1[0] = 1.0
    if n_last % 2 == 0:
        w1[-1] = 1.0
    w = np.broadcast_to(w1, kmod.shape).ravel()
    k = kmod.ravel()
    P = np.asarray(grf.power_grid()).ravel()
    j = np.digitize(k, edges) - 1
    ok = (j >= 0) & (j < len(edges) - 1) & (k > 0)
    W = np.bincount(j[ok], weights=w[ok], minlength=len(edges) - 1)
    Pb = np.bincount(j[ok], weights=(w * P)[ok], minlength=len(edges) - 1) / np.maximum(W, 1e-300)
    return Pb[W > 0]
