"""Toy models: power-law Gaussian random fields and simple analytic field samplers (all JAX pytrees)."""
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

    FFT and normalisation (notes, ``subsec_P1D.tex``, Eqs. P1D_delta_tilde_sample,
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


def _trilinear_periodic_sample(field: jnp.ndarray, pos_xyz: jnp.ndarray,
                               L_box) -> jnp.ndarray:
    """Periodic trilinear interpolation of ``field`` at world positions.

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


# --------------------------------------------------------------------------- samplers of events (a, x, y, z)
@jax.tree_util.register_pytree_node_class
class GaussianDensity:
    """n(x, y, z) from one 3D ``PowerLawGRF`` realisation, trilinearly interpolated in the periodic box.

        s = tanh(delta / contrast),   log10 n = log_mid + log_amp s,   so n in [n0_min, n0_max] [cm^-3].

    Parameters
    ----------
    key : PRNG key of the realisation; grf : 3D ``PowerLawGRF`` (its box ``grf.L`` is the periodic domain).
    n0_min, n0_max : density range [cm^-3]; contrast : tanh saturation scale, in units of delta (of sigma if
    ``normalize``); normalize : divide delta by the expected sigma = sqrt(grf.variance()).

    ``delta_grid`` holds the (normalised) field for diagnostics. JAX pytree (the grid and the mapping
    constants are leaves).
    """

    def __init__(self, key: jax.Array, grf: PowerLawGRF, *, n0_min: float = 1e0, n0_max: float = 1e6,
                 contrast: float = 1.0, normalize: bool = True):
        if grf.ndim != 3:
            raise ValueError(f"`grf` must be 3D, got ndim={grf.ndim}.")
        delta_grid = grf.sample(key)
        if normalize:
            delta_grid = delta_grid / jnp.sqrt(grf.variance())
        log_min, log_max = jnp.log10(n0_min), jnp.log10(n0_max)
        self.delta_grid, self.L_box, self.contrast = delta_grid, jnp.asarray(grf.L), contrast
        self.log_mid, self.log_amp = 0.5 * (log_min + log_max), 0.5 * (log_max - log_min)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x_flat = x.reshape(-1, x.shape[-1])
        pos = jnp.mod(x_flat[:, 1:4], self.L_box)
        delta = _trilinear_periodic_sample(self.delta_grid, pos, self.L_box)
        n_flat = 10.0 ** (self.log_mid + self.log_amp * jnp.tanh(delta / self.contrast))
        return n_flat.reshape(x.shape[:-1])

    def tree_flatten(self):
        return (self.delta_grid, self.L_box, self.contrast, self.log_mid, self.log_amp), None

    @classmethod
    def tree_unflatten(cls, aux, leaves):
        obj = object.__new__(cls)
        obj.delta_grid, obj.L_box, obj.contrast, obj.log_mid, obj.log_amp = leaves
        return obj


@jax.tree_util.register_pytree_node_class
class LogRamp:
    """10^(log v_min + (log v_max - log v_min) x_axis / L_box): a log-linear ramp along one coordinate.

    axis : index of the event column (1, 2, 3 = x, y, z). E.g. temperatures along z ``LogRamp(T_min, T_max, L, 3)``,
    densities along y ``LogRamp(n_min, n_max, L, 2)``; v_min = v_max gives a constant.
    """

    def __init__(self, v_min: float, v_max: float, L_box: float = 1.0, axis: int = 3):
        self.v_min, self.v_max, self.L_box, self.axis = v_min, v_max, L_box, int(axis)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x_flat = x.reshape(-1, x.shape[-1])
        lo, hi = jnp.log10(self.v_min), jnp.log10(self.v_max)
        return (10 ** (lo + (hi - lo) * (x_flat[:, self.axis] / self.L_box))).reshape(x.shape[:-1])

    def tree_flatten(self):
        return (self.v_min, self.v_max, self.L_box), self.axis

    @classmethod
    def tree_unflatten(cls, axis, leaves):
        obj = object.__new__(cls)
        obj.v_min, obj.v_max, obj.L_box = leaves
        obj.axis = axis
        return obj


@jax.tree_util.register_pytree_node_class
class VelocityRamp:
    """Velocity (v_x, 0, 0) [km/s] with v_x = v_min + (v_max - v_min) x / L_box; v_min = v_max: uniform."""

    def __init__(self, v_min: float = -0.001, v_max: float = 0.001, L_box: float = 1.0):
        self.v_min, self.v_max, self.L_box = v_min, v_max, L_box

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x_flat = x.reshape(-1, x.shape[-1])
        vx = self.v_min + (self.v_max - self.v_min) * x_flat[:, 1] / self.L_box
        return jnp.stack([vx, jnp.zeros_like(vx), jnp.zeros_like(vx)], axis=-1).reshape(*x.shape[:-1], 3)

    def tree_flatten(self):
        return (self.v_min, self.v_max, self.L_box), None

    @classmethod
    def tree_unflatten(cls, aux, leaves):
        obj = object.__new__(cls)
        obj.v_min, obj.v_max, obj.L_box = leaves
        return obj


@jax.tree_util.register_pytree_node_class
class Constant:
    """A uniform scalar field (e.g. ``turbulence`` b_turb [m/s] or ``collisional`` A_coll [GHz])."""

    def __init__(self, value: float = 0.0):
        self.value = value

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.full(int(np.prod(x.shape[:-1])), self.value).reshape(x.shape[:-1])

    def tree_flatten(self):
        return (self.value,), None

    @classmethod
    def tree_unflatten(cls, aux, leaves):
        obj = object.__new__(cls)
        obj.value = leaves[0]
        return obj
