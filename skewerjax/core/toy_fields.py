"""Toy models: power-law Gaussian random fields, and the gas cells they define (``grf_cells``).

A toy model enters the optical depth like a simulation does: as ``core.cells.Cells``. ``grf_cells`` turns one
``PowerLawGRF`` realisation delta into the densities n = n_bar (1 + delta) of a set of Voronoi cells, with temperatures
and velocities given as arrays or as functions of the generating points.
"""
from __future__ import annotations

import warnings
from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np


# --------------------------------------------------------------------------- Gaussian random fields
def _as_axes(x, name: str):
    """Scalar -> (x,), sequence of length 1-3 -> tuple; anything else raises."""
    if x is None:
        return None
    t = tuple(np.atleast_1d(np.asarray(x, dtype=float)).tolist())
    if not 1 <= len(t) <= 3:
        raise ValueError(f"`{name}` must be a scalar or a tuple of length 2 or 3, got {x!r}.")
    return t


def _broadcast_axes(named: dict) -> dict:
    """Broadcast scalar entries to the common tuple length; mismatched lengths raise."""
    given = {k: v for k, v in named.items() if v is not None}
    ndims = {len(v) for v in given.values()}
    ndim = max(ndims)
    if ndims - {1, ndim}:
        raise ValueError(f"Inconsistent dimensionality between inputs: "
                         f"{ {k: len(v) for k, v in given.items()} }.")
    return {k: (None if v is None else (v * ndim if len(v) == 1 else v)) for k, v in named.items()}


def _as_int(x: float, name: str) -> int:
    n = int(round(x))
    if not np.isclose(x, n, rtol=0.0, atol=1e-6 * max(1.0, abs(x))):
        raise ValueError(f"`{name}` must be an integer, got {x}.")
    return n


@jax.tree_util.register_pytree_node_class
class PowerLawGRF:
    r"""Real, periodic Gaussian random field with a power-law spectrum in 1D, 2D or 3D.

        P(k) = A_s (k / k_0)^(n_s - 1),   P(0) := 0.

    Grid
    ----
    The dimensionality d is inferred from the grid inputs: scalars give d = 1, tuples of
    length 2 or 3 give d = 2 or 3 (a scalar mixed with a tuple is broadcast to every axis).
    Boxes and cells may be anisotropic (L_i and N_i differ per axis). Exactly one mode:

    (a) ``L`` and ``N``: box side(s) [length units] and number of cells per axis.
    (b) any two (or all three) of ``k_min``, ``k_max``, ``N_k``, per axis, with

            k_min = 2 pi / L                (fundamental mode)
            k_max = pi N / L = N_k k_min    (Nyquist frequency of the axis)
            N_k   = N / 2                   (number of positive Fourier modes on the axis)

        so mode (b) always gives an even N. If k_max / k_min is not an integer, or if all
        three values are given and k_max != N_k k_min, a ``ValueError`` is raised.

    Positions are x_j = j L / N (j = 0..N-1), and wavenumbers are angular,
    k = 2 pi fftfreq(N, d=L/N) [1 / length units], as in ``P1D_xi1D.py``.

    FFT and normalisation (review notes, Eqs. P1D_delta_tilde_sample,
    P1D_sample_definition and xi1D_from_P1D_ensemble, generalised to d dimensions with
    dV = prod(L_i / N_i), V = prod(L_i), N_pix = prod(N_i))
    ------------------------------------------------------------------------------------
    Finite-box transform (forward sign e^{-i k.x}, as ``jnp.fft``):

        delta_k = \int_V d^dx e^{-i k.x} delta(x)  ~=  dV * rfftn(delta),

    which is the notes' tilde-delta times (2 pi)^{d/2}. Periodogram:

        P_hat(k) = |delta_k|^2 / V = (dV / N_pix) |FFT(delta)|^2,

    (the d-dimensional version of the ``dv / N_v`` prefactor in ``core.stats.p1d``). The
    sampler gives <P_hat(k)> = P(k) for every k != 0, and the real-space variance is

        sigma^2 = xi(0) = (1 / V) sum_{k != 0} P(k)   (discrete form of
                  \int d^dk / (2 pi)^d P(k); see ``variance``).

    Algorithm: real unit white noise w(x) -> rfftn -> multiply by sqrt(P(|k|) / dV)
    -> irfftn. The filter is real and even in k, so the result is real; the k = 0 mode is
    set to zero, so the field has zero mean and no divergence when n_s < 1.

    Parameters
    ----------
    A_s : float
        Amplitude, P(k_0) = A_s [length^d].
    k_0 : float
        Pivot wavenumber [1 / length].
    n_s : float
        Spectral index; the slope of P(k) is n_s - 1.
    L, N : float or tuple, int or tuple
        Grid mode (a).
    k_min, k_max, N_k : float or tuple, float or tuple, int or tuple
        Grid mode (b).

    Notes
    -----
    * JAX pytree: ``A_s, k_0, n_s`` are leaves (traceable, differentiable); the grid
      (shape, L) is static. Changing the spectrum does not recompile ``sample``; changing
      the grid does.
    * dtype: float64 when ``jax_enable_x64`` is on (recommended: P(k) spans many decades
      for |n_s - 1| >~ 2), else float32 with a warning.
    * Memory: one real field plus one rfft half-spectrum (complex, ~N_pix/2 entries).
    """

    def __init__(self, A_s, k_0, n_s, *, L=None, N=None, k_min=None, k_max=None, N_k=None):
        mode_a = L is not None or N is not None
        mode_b = any(v is not None for v in (k_min, k_max, N_k))
        if mode_a == mode_b:
            raise ValueError("Specify the grid either with (L, N) or with two or three of "
                             "(k_min, k_max, N_k), not both and not neither.")

        if mode_a:
            if L is None or N is None:
                raise ValueError("Mode (a) needs both `L` and `N`.")
            ax = _broadcast_axes({"L": _as_axes(L, "L"), "N": _as_axes(N, "N")})
            L_ax = ax["L"]
            N_ax = tuple(_as_int(n, "N") for n in ax["N"])
        else:
            ax = _broadcast_axes({"k_min": _as_axes(k_min, "k_min"),
                                  "k_max": _as_axes(k_max, "k_max"),
                                  "N_k": _as_axes(N_k, "N_k")})
            if sum(v is not None for v in ax.values()) < 2:
                raise ValueError("Mode (b) needs at least two of (k_min, k_max, N_k).")
            for name in ("k_min", "k_max", "N_k"):
                if ax[name] is not None and not all(np.isfinite(v) and v > 0 for v in ax[name]):
                    raise ValueError(f"`{name}` must be finite and > 0, got {ax[name]}.")
            ndim = len(next(v for v in ax.values() if v is not None))
            L_ax, N_ax = [], []
            for i in range(ndim):
                kmin, kmax, nk = (None if ax[n] is None else ax[n][i] for n in ("k_min", "k_max", "N_k"))
                if nk is None:
                    nk = _as_int(kmax / kmin, f"k_max / k_min (axis {i})")
                nk = _as_int(nk, f"N_k (axis {i})")
                if kmin is None:
                    kmin = kmax / nk
                if kmax is not None and not np.isclose(kmax, nk * kmin, rtol=1e-8, atol=0.0):
                    raise ValueError(f"Over-determined grid on axis {i} is inconsistent: "
                                     f"k_max = {kmax} but N_k * k_min = {nk} * {kmin} = {nk * kmin} "
                                     "(convention: k_max = Nyquist = N_k * k_min).")
                L_ax.append(2.0 * np.pi / kmin)
                N_ax.append(2 * nk)
            L_ax, N_ax = tuple(L_ax), tuple(N_ax)

        if any(n < 2 for n in N_ax):
            raise ValueError(f"Need N >= 2 cells per axis (N_k >= 1), got N = {N_ax}.")
        if any(not (np.isfinite(l) and l > 0) for l in L_ax):
            raise ValueError(f"Box sides must be finite and > 0 (k_min > 0), got L = {L_ax}.")
        if not isinstance(A_s, jax.core.Tracer) and not float(A_s) >= 0.0:
            raise ValueError(f"`A_s` must be >= 0, got {A_s}.")
        if not isinstance(k_0, jax.core.Tracer) and not float(k_0) > 0.0:
            raise ValueError(f"`k_0` must be > 0, got {k_0}.")
        if not jax.config.jax_enable_x64:
            warnings.warn("jax_enable_x64 is off: PowerLawGRF runs in float32.", stacklevel=2)

        self.A_s, self.k_0, self.n_s = A_s, k_0, n_s
        self.shape: Tuple[int, ...] = N_ax
        self.L: Tuple[float, ...] = L_ax

    # -- pytree -------------------------------------------------------------
    def tree_flatten(self):
        return (self.A_s, self.k_0, self.n_s), (self.shape, self.L)

    @classmethod
    def tree_unflatten(cls, aux, leaves):
        obj = object.__new__(cls)
        obj.A_s, obj.k_0, obj.n_s = leaves
        obj.shape, obj.L = aux
        return obj

    def __repr__(self):
        return (f"PowerLawGRF(A_s={self.A_s}, k_0={self.k_0}, n_s={self.n_s}, "
                f"L={self.L}, N={self.shape})")

    # -- grid ---------------------------------------------------------------
    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def dx(self) -> Tuple[float, ...]:
        """Cell sizes L_i / N_i [length]."""
        return tuple(l / n for l, n in zip(self.L, self.shape))

    @property
    def cell_volume(self) -> float:
        """dV = prod(dx_i) [length^d]."""
        return float(np.prod(self.dx))

    @property
    def volume(self) -> float:
        """V = prod(L_i) [length^d]."""
        return float(np.prod(self.L))

    @property
    def k_min(self) -> Tuple[float, ...]:
        """Fundamental mode per axis, 2 pi / L_i [1 / length]."""
        return tuple(2.0 * np.pi / l for l in self.L)

    @property
    def k_max(self) -> Tuple[float, ...]:
        """Largest |k_i| on each axis, 2 pi (N_i // 2) / L_i (= Nyquist for even N_i)."""
        return tuple(2.0 * np.pi * (n // 2) / l for l, n in zip(self.L, self.shape))

    def x_grid(self) -> Tuple[np.ndarray, ...]:
        """1D position axes x_j = j * dx_i, one per dimension [length]."""
        return tuple(np.arange(n) * d for n, d in zip(self.shape, self.dx))

    def k_grid(self, real: bool = True):
        """Wavevector components and |k| on the FFT grid [1 / length].

        Parameters
        ----------
        real : bool
            True: ``rfftn`` layout (last axis has N_last // 2 + 1 entries), matching the
            modes returned by ``sample``. False: full ``fftn`` layout.

        Returns
        -------
        k_vec : tuple of d arrays
            Sparse (broadcastable) components k_i, standard FFT order (not shifted).
        k_mod : array, shape = ``self.shape`` (or its rfft version)
            |k| = sqrt(sum_i k_i^2).
        """
        axes = [2.0 * jnp.pi * jnp.fft.fftfreq(n, d=d) for n, d in zip(self.shape, self.dx)]
        if real:
            axes[-1] = 2.0 * jnp.pi * jnp.fft.rfftfreq(self.shape[-1], d=self.dx[-1])
        k_vec = tuple(jnp.meshgrid(*axes, indexing="ij", sparse=True))
        return k_vec, jnp.sqrt(sum(k ** 2 for k in k_vec))

    # -- spectrum -----------------------------------------------------------
    def power(self, k):
        """P(k) = A_s (k / k_0)^(n_s - 1) for k > 0, and 0 at k = 0 [length^d].

        Uses the double-``where`` pattern, so no inf / NaN appears for n_s < 1, in the
        value or in its gradient.
        """
        k = jnp.asarray(k)
        pos = k > 0
        k_safe = jnp.where(pos, k, self.k_0)
        return jnp.where(pos, self.A_s * (k_safe / self.k_0) ** (self.n_s - 1.0), 0.0)

    def variance(self):
        """Expected real-space variance sigma^2 = (1/V) sum_{k != 0} P(|k|) on this grid."""
        _, k_mod = self.k_grid(real=True)
        n_last = self.shape[-1]
        w = jnp.full(n_last // 2 + 1, 2.0).at[0].set(1.0)   # rfft half-axis -> full-grid weights
        if n_last % 2 == 0:
            w = w.at[-1].set(1.0)
        return jnp.sum(w * self.power(k_mod)) / self.volume

    # -- sampling -----------------------------------------------------------
    def sample(self, key: jax.Array, return_modes: bool = False):
        """Draw one realisation (jitted; ``jax.vmap`` over keys for a batch).

        Parameters
        ----------
        key : jax PRNG key
        return_modes : bool
            Also return the Fourier modes delta_k = dV * rfftn(delta) [field x length^d],
            so that |delta_k|^2 / V is the periodogram (see class docstring).

        Returns
        -------
        delta : real array, shape ``self.shape``, zero mean.
        delta_k : complex array, rfft shape (only if ``return_modes``), layout of
            ``k_grid(real=True)``.
        """
        return _sample_power_law_grf(self, key, return_modes)


@partial(jax.jit, static_argnames=("return_modes",))
def _sample_power_law_grf(grf: PowerLawGRF, key: jax.Array, return_modes: bool):
    dtype = jnp.result_type(float)                      # float64 under x64
    white_k = jnp.fft.rfftn(jax.random.normal(key, grf.shape, dtype=dtype))
    _, k_mod = grf.k_grid(real=True)
    # <|white_k|^2> = N_pix  ->  <|FFT(delta)|^2> = N_pix P / dV  ->  <P_hat> = P
    fft_delta = white_k * jnp.sqrt(grf.power(k_mod) / grf.cell_volume)
    delta = jnp.fft.irfftn(fft_delta, s=grf.shape)
    if return_modes:
        return delta, fft_delta * grf.cell_volume
    return delta


def interpolate_periodic(field: jnp.ndarray, pos_xyz: jnp.ndarray, L_box) -> jnp.ndarray:
    """Periodic trilinear interpolation of a grid ``field`` (values at the nodes x_j = j L / N) at world positions.

    field   : (Nx, Ny, Nz) real
    pos_xyz : (M, 3) positions, periodic in [0, L_box)
    L_box   : float or (3,) array, box side(s) (anisotropic boxes allowed)
    returns : (M,) sampled values
    """
    Nx, Ny, Nz = field.shape
    N = jnp.array([Nx, Ny, Nz], dtype=pos_xyz.dtype)
    u = (jnp.mod(pos_xyz, L_box) / L_box) * N
    i0 = jnp.floor(u).astype(jnp.int32)
    f = u - i0.astype(u.dtype)

    ix0 = jnp.mod(i0[:, 0], Nx); ix1 = jnp.mod(ix0 + 1, Nx)
    iy0 = jnp.mod(i0[:, 1], Ny); iy1 = jnp.mod(iy0 + 1, Ny)
    iz0 = jnp.mod(i0[:, 2], Nz); iz1 = jnp.mod(iz0 + 1, Nz)
    fx, fy, fz = f[:, 0], f[:, 1], f[:, 2]

    c000 = field[ix0, iy0, iz0]; c100 = field[ix1, iy0, iz0]
    c010 = field[ix0, iy1, iz0]; c110 = field[ix1, iy1, iz0]
    c001 = field[ix0, iy0, iz1]; c101 = field[ix1, iy0, iz1]
    c011 = field[ix0, iy1, iz1]; c111 = field[ix1, iy1, iz1]

    c00 = c000 * (1 - fx) + c100 * fx
    c10 = c010 * (1 - fx) + c110 * fx
    c01 = c001 * (1 - fx) + c101 * fx
    c11 = c011 * (1 - fx) + c111 * fx
    c0 = c00 * (1 - fy) + c10 * fy
    c1 = c01 * (1 - fy) + c11 * fy
    return c0 * (1 - fz) + c1 * fz


# --------------------------------------------------------------------------- toy gas cells
def grf_cells(delta, grf: PowerLawGRF, n_bar: dict, *, generators=None, temperature=1e4, velocity=0.0, turbulence=None,
              collisional=None, volumes: str = "voronoi", samples_per_cell: int = 32, seed: int = 0):
    """``core.Cells`` of the gas of one 3D ``PowerLawGRF`` realisation: n = n_bar (1 + delta) for every species.

    No transform, clipping or threshold: delta is the realisation as drawn (its variance is ``grf.variance()``). The
    amplitude must keep 1 + delta > 0; a warning is issued when the minimum over the cells is <= 0.

    Parameters
    ----------
    delta : (N_x, N_y, N_z) one realisation (``grf.sample(key)``), on the nodes x_j = j L / N_j.
    grf : the cubic 3D ``PowerLawGRF`` of ``delta`` (its box ``grf.L`` is the periodic box of the cells).
    n_bar : {species: mean comoving density [cm^-3]} (e.g. {"HI": 1e-12, "SiIII": 2e-14}).
    generators : None (default): the nodes of the grid, whose Voronoi cells are exactly the grid cells (cubes of side
        L / N centred on the nodes): the gas is the realisation itself, piecewise constant, an exact GRF at grid
        resolution (``cells.cells_on_grid``). Or (M, 3) generating points [same length unit as the box], e.g. random
        ones (a moving mesh): each takes delta interpolated trilinearly at its position (``interpolate_periodic``), so
        the gas is the realisation smoothed by the interpolation and by its own cells.
    temperature [K], velocity [km/s], turbulence [m/s], collisional [GHz] : per cell, as in ``cells.cells_from_values``:
        a scalar, an array (on the grid (N_x, N_y, N_z)[, 3] for the default generators, else (M,)[, 3]), a callable
        of the generating points (M, 3), or a dict {species: one of those}.
    volumes : generators given only: "voronoi" (default, ``cells.voronoi_volumes`` with ``samples_per_cell`` and
        ``seed``: each cell's own volume, so its size V^(1/3) is the simulation's definition) or "mean" (L^3 / M).
    """
    from skewerjax.core.cells import cells_from_values, cells_on_grid, voronoi_volumes
    if grf.ndim != 3 or len(set(grf.L)) != 1:
        raise ValueError(f"grf_cells needs a cubic 3D PowerLawGRF, got L = {grf.L}.")
    d = np.asarray(delta, dtype=np.float64)
    if d.shape != grf.shape:
        raise ValueError(f"delta must have the grid shape {grf.shape}, got {d.shape}.")
    box = float(grf.L[0])
    if generators is None:
        one = 1.0 + d
        n = {sp: nb * one for sp, nb in n_bar.items()}
        cells = cells_on_grid(box, n, temperature=temperature, velocity=velocity, turbulence=turbulence, collisional=collisional)
    else:
        pos = np.mod(np.asarray(generators, dtype=np.float64), box)
        one = 1.0 + np.asarray(interpolate_periodic(jnp.asarray(d), jnp.asarray(pos), box))
        if volumes == "voronoi":
            vol = voronoi_volumes(pos, box, samples_per_cell=samples_per_cell, seed=seed)
        elif volumes == "mean":
            vol = np.full(pos.shape[0], box ** 3 / pos.shape[0])
        else:
            raise ValueError(f"volumes must be 'voronoi' or 'mean', got {volumes!r}.")
        n = {sp: nb * one for sp, nb in n_bar.items()}
        cells = cells_from_values(pos, box, n, temperature=temperature, velocity=velocity, turbulence=turbulence,
                                  collisional=collisional, volumes=vol)
    if float(one.min()) <= 0.0:
        warnings.warn(f"grf_cells: min(1 + delta) = {float(one.min()):.3f} <= 0, so the density is negative somewhere; "
                      "lower A_s.", stacklevel=2)
    return cells
