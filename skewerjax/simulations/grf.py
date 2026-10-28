"""Toy field source: power-law Gaussian random fields (``PowerLawGRF``) in 1D, 2D and 3D.

A field source of ``skewerjax.simulations``: the realisation becomes gas cells with ``toy_cells.grf_cells``, the same
container (``core.Cells``) as a snapshot (``snapshot_skewers``).
"""

from __future__ import annotations

import warnings
from functools import partial

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
        raise ValueError(f"Inconsistent dimensionality between inputs: { {k: len(v) for k, v in given.items()} }.")
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

    Optional terms (all off by default; review notes, Sec. xi3D, for the anisotropic Lya picture)
    ----------------------------------------------------------------------------------------------
    The spectrum becomes

        P(k, mu) = [P_pl(k) + P_BAO(k)] K(mu) D(k mu),   P_pl = A_s (k / k_0)^(n_s - 1),

    with mu = |k_par| / |k| the cosine of the angle to the line of sight, the grid axis ``los_axis`` (default: the last
    axis, the k_par of ``utils.plotting``; give the axis of the skewers, e.g. 1 for skewers along y). In 1D, mu = 1.

    * Kaiser-like anisotropy (``beta``): K(mu) = (1 + beta mu^2)^2 (Kaiser 1987), >= 0 for every beta.
    * Fingers of God (``sigma_fog`` [length], ``fog``): a damping along the line of sight with one dispersion,
      D = 1 / (1 + (k mu sigma)^2) (``"lorentzian"``, an exponential pairwise-velocity distribution) or
      exp(-(k mu sigma)^2) (``"gaussian"``); 0 < D <= 1. ``sigma_fog`` must be >= 0.
    * BAO-like feature (``A_bao``, ``r_bao``, ``sigma_bao`` [length], all three or none): a Gaussian bump in the
      correlation function, xi_w(r) = A_bao [g(r - r_bao) + g(r + r_bao)], g(u) = exp(-u^2 / (2 sigma_bao^2)) (the
      mirror term is the even extension; it is below A_bao exp(-r_bao^2 / 2 sigma_bao^2) for r >= 0), whose exact
      d-dimensional Fourier transform A_bao W(k) oscillates in sign, carried by its envelope E(k) >= |W(k)|:

          P_BAO(k) = A_bao [E(k) + W(k)] = A_bao E(k) [1 + f(k)],   f = W / E,  |f| <= 1,

          1D: W = 2 G cos(k r_bao),                                  E = 2 G,
          2D: W = 2 pi int_0^inf r [g(r - r_bao) + g(r + r_bao)] J_0(k r) dr,   E = the same with J_0(k r) -> m(k r),
          3D: W = 4 pi G [sigma_bao^2 cos(k r_bao) + r_bao sin(k r_bao) / k],     E = 4 pi G [sigma_bao^2 + r_bao / k],

      with G(k) = sigma_bao sqrt(2 pi) exp(-k^2 sigma_bao^2 / 2) and, in 2D, m(x) = min(1, sqrt(2 / (pi x))) >= |J_0(x)|
      (Sonin-Polya: the maxima of sqrt(x) |J_0(x)| rise towards sqrt(2 / pi)); Gauss-Legendre, 256 nodes on
      r_bao +- 10 sigma_bao, J_0 from Abramowitz & Stegun 9.4.1 / 9.4.3, |error| < 5e-8. So P_BAO >= 0 for every
      A_bao >= 0, by construction: the wiggle f multiplies its own envelope, and the smooth spectrum P_pl + A_bao E
      carries them. In xi the term is the bump plus the inverse transform of A_bao E, a positive-definite function
      peaked at r = 0 (3D: 2 A_bao exp(-r^2 / 2 sigma_bao^2) + 4 A_bao r_bao F(r / (sqrt(2) sigma_bao)) / (sqrt(pi) r),
      F the Dawson function: ~ 2 sqrt(2 / pi) A_bao r_bao sigma_bao / r^2 for r >> sigma_bao, a fraction ~ 1.6
      sigma_bao / r_bao of the bump at r_bao; 1D: 2 A_bao g(r)). A bare bump, P_pl + A_bao W, would need
      A_bao |W| <= P_pl wherever P_pl is small, which caps A_bao; the envelope lifts that cap.

    A Gaussian field needs P >= 0 at every mode: P_pl > 0, P_BAO >= 0, and K and D are never negative. The k = 0 mode
    stays zero.

    Positions are x_j = j L / N (j = 0..N-1), and wavenumbers are angular,
    k = 2 pi fftfreq(N, d=L/N) [1 / length units], as in ``core.stats.p1d``.

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
    beta, sigma_fog, fog : the Kaiser and Fingers-of-God terms (None: off).
    A_bao, r_bao, sigma_bao : the BAO bump: its height in xi (>= 0) [field^2], position and width [length]
        (None: off).
    los_axis : the grid axis of the line of sight (default: the last one).

    Notes
    -----
    * JAX pytree: ``A_s, k_0, n_s`` are leaves (traceable, differentiable); the grid
      (shape, L) is static. Changing the spectrum does not recompile ``sample``; changing
      the grid does.
    * dtype: float64 when ``jax_enable_x64`` is on (recommended: P(k) spans many decades
      for |n_s - 1| >~ 2), else float32 with a warning.
    * Memory: one real field plus one rfft half-spectrum (complex, ~N_pix/2 entries).
    """

    def __init__(
        self,
        A_s,
        k_0,
        n_s,
        *,
        L=None,
        N=None,
        k_min=None,
        k_max=None,
        N_k=None,
        beta=None,
        sigma_fog=None,
        fog: str = "lorentzian",
        A_bao=None,
        r_bao=None,
        sigma_bao=None,
        los_axis: int = -1,
    ):
        mode_a = L is not None or N is not None
        mode_b = any(v is not None for v in (k_min, k_max, N_k))
        if mode_a == mode_b:
            raise ValueError(
                "Specify the grid either with (L, N) or with two or three of (k_min, k_max, N_k), not both and not neither."
            )

        if mode_a:
            if L is None or N is None:
                raise ValueError("Mode (a) needs both `L` and `N`.")
            ax = _broadcast_axes({"L": _as_axes(L, "L"), "N": _as_axes(N, "N")})
            L_ax = ax["L"]
            N_ax = tuple(_as_int(n, "N") for n in ax["N"])
        else:
            ax = _broadcast_axes(
                {"k_min": _as_axes(k_min, "k_min"), "k_max": _as_axes(k_max, "k_max"), "N_k": _as_axes(N_k, "N_k")}
            )
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
                    raise ValueError(
                        f"Over-determined grid on axis {i} is inconsistent: "
                        f"k_max = {kmax} but N_k * k_min = {nk} * {kmin} = {nk * kmin} "
                        "(convention: k_max = Nyquist = N_k * k_min)."
                    )
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
        self.shape: tuple[int, ...] = N_ax
        self.L: tuple[float, ...] = L_ax
        if not -len(N_ax) <= int(los_axis) < len(N_ax):
            raise ValueError(f"los_axis = {los_axis} is not an axis of a {len(N_ax)}D grid.")
        self.los_axis = int(los_axis) % len(N_ax)
        if fog not in ("lorentzian", "gaussian"):
            raise ValueError(f"fog must be 'lorentzian' or 'gaussian', got {fog!r}.")
        self.fog = fog
        bao = (A_bao, r_bao, sigma_bao)
        if any(x is not None for x in bao) and any(x is None for x in bao):
            raise ValueError("the BAO term needs all of A_bao, r_bao and sigma_bao (or none).")
        self.beta, self.sigma_fog, self.A_bao, self.r_bao, self.sigma_bao = beta, sigma_fog, A_bao, r_bao, sigma_bao
        concrete = lambda x: x is not None and not isinstance(x, jax.core.Tracer)
        if concrete(sigma_fog) and float(sigma_fog) < 0.0:
            raise ValueError(f"sigma_fog must be >= 0, got {sigma_fog}.")
        if concrete(sigma_bao) and not float(sigma_bao) > 0.0:
            raise ValueError(f"sigma_bao must be > 0, got {sigma_bao}.")
        if concrete(r_bao) and float(r_bao) < 0.0:
            raise ValueError(f"r_bao must be >= 0, got {r_bao}.")
        if concrete(A_bao) and float(A_bao) < 0.0:
            raise ValueError(f"A_bao must be >= 0 (the height of the bump in xi), got {A_bao}.")

    # -- pytree -------------------------------------------------------------
    _OPTIONAL = ("beta", "sigma_fog", "A_bao", "r_bao", "sigma_bao")

    def tree_flatten(self):
        opt = tuple(n for n in self._OPTIONAL if getattr(self, n) is not None)  # the terms switched on are leaves
        return (self.A_s, self.k_0, self.n_s) + tuple(getattr(self, n) for n in opt), (
            self.shape,
            self.L,
            opt,
            self.fog,
            self.los_axis,
        )

    @classmethod
    def tree_unflatten(cls, aux, leaves):
        obj = object.__new__(cls)
        obj.shape, obj.L, opt, obj.fog, obj.los_axis = aux
        obj.A_s, obj.k_0, obj.n_s = leaves[:3]
        for n in cls._OPTIONAL:
            setattr(obj, n, None)
        for n, x in zip(opt, leaves[3:]):
            setattr(obj, n, x)
        return obj

    def __repr__(self):
        extra = "".join(f", {n}={getattr(self, n)}" for n in self._OPTIONAL if getattr(self, n) is not None)
        if self.sigma_fog is not None:
            extra += f", fog={self.fog!r}"
        if self.anisotropic and self.ndim > 1:
            extra += f", los_axis={self.los_axis}"
        return f"PowerLawGRF(A_s={self.A_s}, k_0={self.k_0}, n_s={self.n_s}, L={self.L}, N={self.shape}{extra})"

    @property
    def anisotropic(self) -> bool:
        """True if a Kaiser or Fingers-of-God term is on (then P depends on mu)."""
        return self.beta is not None or self.sigma_fog is not None

    # -- grid ---------------------------------------------------------------
    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def dx(self) -> tuple[float, ...]:
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
    def k_min(self) -> tuple[float, ...]:
        """Fundamental mode per axis, 2 pi / L_i [1 / length]."""
        return tuple(2.0 * np.pi / l for l in self.L)

    @property
    def k_max(self) -> tuple[float, ...]:
        """Largest |k_i| on each axis, 2 pi (N_i // 2) / L_i (= Nyquist for even N_i)."""
        return tuple(2.0 * np.pi * (n // 2) / l for l, n in zip(self.L, self.shape))

    def x_grid(self) -> tuple[np.ndarray, ...]:
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
        return k_vec, jnp.sqrt(sum(k**2 for k in k_vec))

    # -- spectrum -----------------------------------------------------------
    def power(self, k, mu=None, bao: str = "total"):
        """P(k, mu) for k > 0, and 0 at k = 0 [length^d] (class docstring).

        Without the optional terms, P(k) = A_s (k / k_0)^(n_s - 1). ``mu`` = |k_par| / |k| (broadcastable to k) is
        needed when a Kaiser or Fingers-of-God term is on, except in 1D where mu = 1. ``bao``: "total" (the spectrum
        of ``sample``) or "envelope", the no-wiggle spectrum P_pl + A_bao E, whose xi lacks only the bump (``bao_power``).
        Uses the double-``where`` pattern, so no inf / NaN appears for n_s < 1, in the value or in its gradient.
        """
        if bao not in ("total", "envelope"):
            raise ValueError(f"bao must be 'total' or 'envelope', got {bao!r}.")
        k = jnp.asarray(k)
        pos = k > 0
        k_safe = jnp.where(pos, k, self.k_0)
        p = self.A_s * (k_safe / self.k_0) ** (self.n_s - 1.0)
        if self.A_bao is not None:
            p = p + self.bao_power(k_safe, bao)
        if self.anisotropic:
            if self.ndim == 1:
                mu = 1.0
            elif mu is None:
                raise ValueError("this PowerLawGRF is anisotropic: give mu = |k_par| / |k| (power(k, mu)).")
            mu = jnp.asarray(mu)
            if self.beta is not None:
                p = p * (1.0 + self.beta * mu**2) ** 2
            if self.sigma_fog is not None:
                x2 = (k_safe * mu * self.sigma_fog) ** 2
                p = p / (1.0 + x2) if self.fog == "lorentzian" else p * jnp.exp(-x2)
        return jnp.where(pos, p, 0.0)

    def mu_grid(self, k_vec, k_mod):
        """mu = |k_par| / |k| on a ``k_grid`` (k_par along ``los_axis``; 0 at k = 0)."""
        return jnp.abs(k_vec[self.los_axis]) / jnp.where(k_mod > 0, k_mod, 1.0)

    def power_grid(self, real: bool = True, bao: str = "total"):
        """P of every mode of the FFT grid (``k_grid(real)`` layout): what ``sample`` and ``variance`` use (``bao``:
        ``power``)."""
        k_vec, k_mod = self.k_grid(real=real)
        if not self.anisotropic:
            return self.power(k_mod, bao=bao)
        return self.power(k_mod, self.mu_grid(k_vec, k_mod), bao=bao)

    def power_at(self, *k_components):
        """P at wavevectors given by their components along the d grid axes (broadcastable arrays [1 / length]):
        |k| and mu = |k_los| / |k| along ``los_axis``. The input spectrum at any set of modes (theory curves)."""
        if len(k_components) != self.ndim:
            raise ValueError(f"power_at needs {self.ndim} components, got {len(k_components)}.")
        k_components = [jnp.asarray(c) for c in k_components]
        k_mod = jnp.sqrt(sum(c**2 for c in k_components))
        if not self.anisotropic:
            return self.power(k_mod)
        return self.power(k_mod, jnp.abs(k_components[self.los_axis]) / jnp.where(k_mod > 0, k_mod, 1.0))

    def power_monopole(self, k, n_mu: int = 64):
        """P(k) averaged over the directions of k [length^d]: mu uniform on [0, 1] in 3D, the polar angle uniform in 2D
        (Gauss-Legendre, ``n_mu`` nodes); P(k) itself in 1D or without a Kaiser / Fingers-of-God term."""
        k = jnp.asarray(k)
        if not self.anisotropic or self.ndim == 1:
            return self.power(k)
        t, w = np.polynomial.legendre.leggauss(n_mu)
        if self.ndim == 3:
            mu, wt = 0.5 * (t + 1.0), 0.5 * w
        else:
            mu, wt = np.cos(0.25 * np.pi * (t + 1.0)), 0.5 * w
        return jnp.sum(wt * self.power(k[..., None], mu), axis=-1)

    def bao_power(self, k, part: str = "total"):
        """P_BAO(k) [length^d] for k > 0 (class docstring): ``part`` "total" = A_bao (E + W) >= 0, "wiggle" = A_bao W
        (the exact transform of the bump in xi), "envelope" = A_bao E (the smooth part)."""
        k = jnp.asarray(k)
        A, r0, sg = self.A_bao, self.r_bao, self.sigma_bao
        G = sg * jnp.sqrt(2.0 * jnp.pi) * jnp.exp(-0.5 * (k * sg) ** 2)
        if self.ndim == 1:
            W, E = 2.0 * G * jnp.cos(k * r0), 2.0 * G
        elif self.ndim == 3:
            W = 4.0 * jnp.pi * G * (sg**2 * jnp.cos(k * r0) + r0 * jnp.sin(k * r0) / k)
            E = 4.0 * jnp.pi * G * (sg**2 + r0 / k)
        else:
            t, w = np.polynomial.legendre.leggauss(256)  # 2D: the Hankel transform, by quadrature
            lo = jnp.maximum(0.0, r0 - 10.0 * sg)
            hi = r0 + 10.0 * sg
            r = lo + 0.5 * (hi - lo) * (t + 1.0)
            wr = 0.5 * (hi - lo) * w
            f = 2.0 * jnp.pi * wr * r * (jnp.exp(-0.5 * ((r - r0) / sg) ** 2) + jnp.exp(-0.5 * ((r + r0) / sg) ** 2))
            kr = k[..., None] * r
            W = jnp.sum(f * bessel_j0(kr), axis=-1)
            E = jnp.sum(f * jnp.minimum(1.0, jnp.sqrt(2.0 / (jnp.pi * jnp.maximum(kr, 1e-300)))), axis=-1)
        if part == "wiggle":
            return A * W
        if part == "envelope":
            return A * E
        if part != "total":
            raise ValueError(f"part must be 'total', 'wiggle' or 'envelope', got {part!r}.")
        return A * jnp.maximum(E + W, 0.0)  # E >= |W| analytically; the floor only absorbs rounding

    def variance(self):
        """Expected real-space variance sigma^2 = (1/V) sum_{k != 0} P(k) on this grid."""
        n_last = self.shape[-1]
        w = jnp.full(n_last // 2 + 1, 2.0).at[0].set(1.0)  # rfft half-axis -> full-grid weights
        if n_last % 2 == 0:
            w = w.at[-1].set(1.0)
        return jnp.sum(w * self.power_grid()) / self.volume

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
    dtype = jnp.result_type(float)  # float64 under x64
    white_k = jnp.fft.rfftn(jax.random.normal(key, grf.shape, dtype=dtype))
    # <|white_k|^2> = N_pix  ->  <|FFT(delta)|^2> = N_pix P / dV  ->  <P_hat> = P
    fft_delta = white_k * jnp.sqrt(grf.power_grid() / grf.cell_volume)
    delta = jnp.fft.irfftn(fft_delta, s=grf.shape)
    if return_modes:
        return delta, fft_delta * grf.cell_volume
    return delta


def bessel_j0(x):
    """J_0(x) (jax.numpy): Abramowitz & Stegun 9.4.1 for |x| <= 3 and 9.4.3 for |x| > 3, |error| < 5e-8."""
    x = jnp.abs(jnp.asarray(x))
    small = x <= 3.0
    y = (jnp.where(small, x, 3.0) / 3.0) ** 2
    j_small = 1.0 + y * (-2.2499997 + y * (1.2656208 + y * (-0.3163866 + y * (0.0444479 + y * (-0.0039444 + y * 0.0002100)))))
    xl = jnp.where(small, 3.0, x)
    z = 3.0 / xl
    f0 = 0.79788456 + z * (
        -0.00000077 + z * (-0.00552740 + z * (-0.00009512 + z * (0.00137237 + z * (-0.00072805 + z * 0.00014476))))
    )
    th = (
        xl
        - 0.78539816
        + z * (-0.04166397 + z * (-0.00003954 + z * (0.00262573 + z * (-0.00054125 + z * (-0.00029333 + z * 0.00013558)))))
    )
    return jnp.where(small, j_small, f0 * jnp.cos(th) / jnp.sqrt(xl))
