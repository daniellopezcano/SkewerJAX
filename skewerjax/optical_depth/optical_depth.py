from __future__ import annotations

from functools import partial
from typing import Optional, Mapping
from jax.typing import ArrayLike

import os
import logging
import warnings

logger = logging.getLogger(__name__)

try:
    import psutil
except ImportError:
    psutil = None

import numpy as np
from jax import vmap, jit
import jax.numpy as jnp

from skewerjax.constants.physical_constants import phys
from skewerjax.optical_depth import line_profile


# ----------------------------------------------------------------------------------------------
# Quadrature along the skewer, and the pure optical-depth kernel
# ----------------------------------------------------------------------------------------------
def trapezoid_line_elements(points: jnp.ndarray) -> jnp.ndarray:
    """Line element of every point of a polyline: half the distance to each neighbour.

    For points x_0 ... x_{N-1} with segment lengths d_k = |x_{k+1} - x_k|, the element of point i
    is w_i = (d_{i-1} + d_i) / 2 (w_0 = d_0 / 2, w_{N-1} = d_{N-2} / 2): the length of the
    *dual segment* between the midpoints that surround x_i. sum_i w_i f_i is exactly the
    trapezoid rule on the path length, and sum_i w_i = path length.

    Parameters
    ----------
    points : (..., N, 3) positions (any unit).

    Returns
    -------
    (..., N) line elements, same unit.
    """
    d = jnp.linalg.norm(jnp.diff(points, axis=-2), axis=-1)
    z = jnp.zeros(d.shape[:-1] + (1,), d.dtype)
    return 0.5 * (jnp.concatenate([z, d], axis=-1) + jnp.concatenate([d, z], axis=-1))


@partial(jit, static_argnames=("voigt_implementation_mode", "detuning", "velocity_period"))
def optical_depth_kernel(velocities_kms, lambda0_A, A21_GHz, mass_amu, f_osc, a_dagger, scale_factor,
                         los_velocity, temperature, turbulence, collisional, number_density, weights_m,
                         voigt_implementation_mode: str = "ExoJAX", detuning: str = "exact",
                         velocity_period: Optional[float] = None):
    """tau = sum_i w_i C f n_i a_i^-2 phi_i(v): the discretised notes' Eq. ``tau_straight_los``.

    Pure function of arrays (jit-compiled, differentiable in every float input).

    Parameters
    ----------
    velocities_kms : (N_v,)
    lambda0_A, A21_GHz, mass_amu, f_osc : (N_lines,)
    a_dagger : scalar or (N_points,)
    scale_factor : (N_points,)
    los_velocity, temperature [K], turbulence [m/s], collisional [GHz], number_density
        [comoving cm^-3] : (N_lines, N_points)
    weights_m : (N_points,) line element of each point [comoving m]

    Returns
    -------
    (N_lines, N_v) optical depth, summed over the points.
    """
    sigma, gamma = line_profile.voigt_widths_GHz(temperature, turbulence, collisional, lambda0_A, A21_GHz, mass_amu)
    nu = line_profile.compute_nu_offset_GHz(velocities_kms, lambda0_A, a_dagger, scale_factor, los_velocity,
                                            velocity_period=velocity_period, detuning=detuning)
    n_l, n_p, n_v = nu.shape
    phi = line_profile.voigt_line_profiles(nu.reshape(n_l * n_p, n_v), sigma.reshape(-1), gamma.reshape(-1),
                                           voigt_implementation_mode).reshape(n_l, n_p, n_v)
    # dtau/dx [1/m] = sigma_0 f phi n a^-2 (x comoving); 1e2 converts the cm^2 cm^-3 m to a number
    amp = phys.cross_section_prefactor * f_osc[:, None] * number_density * scale_factor[None, :] ** -2 * 1e2 * weights_m[None, :]
    return jnp.einsum("lp,lpv->lv", amp, phi)


def voigt_width_kms(sigma_GHz, gamma_GHz, lambda0_A):
    """Width of a Voigt profile in velocity, expressed as an equivalent Doppler parameter [km/s].

    The Gaussian part has Doppler parameter b_G = sqrt(2) sigma lambda and FWHM f_G = 2 sqrt(ln 2) b_G;
    the Lorentzian part has FWHM f_L = 2 gamma lambda. The Voigt FWHM is approximated by
    f_V = 0.5346 f_L + sqrt(0.2166 f_L^2 + f_G^2) (better than 0.02 % for any f_L / f_G), and the
    returned width is f_V / (2 sqrt(ln 2)): b_G for a thermal line, and a finite width for a
    natural line (T -> 0), where b_G alone vanishes.

    Parameters
    ----------
    sigma_GHz, gamma_GHz : (N_lines, ...) Gaussian standard deviation and Lorentzian half width
        (``line_profile.voigt_widths_GHz``); lambda0_A : (N_lines,) rest wavelength [A].
    """
    lam = jnp.reshape(jnp.asarray(lambda0_A), (-1,) + (1,) * (jnp.ndim(sigma_GHz) - 1))
    to_kms = lam * 1e-4                                  # GHz x A -> km/s
    f_G = 2.0 * jnp.sqrt(jnp.log(2.0)) * jnp.sqrt(2.0) * sigma_GHz * to_kms
    f_L = 2.0 * gamma_GHz * to_kms
    f_V = 0.5346 * f_L + jnp.sqrt(0.2166 * f_L ** 2 + f_G ** 2)
    return f_V / (2.0 * jnp.sqrt(jnp.log(2.0)))


def sampling_resolution_report(line_centre_kms, b_kms, weight, tolerance: float = 0.25) -> dict:
    """Is the spacing of the points fine enough to resolve the lines? (notes
    ``app_numerical_resolution.tex``).

    The quadrature places point i's absorption at its line centre v_c,i. Consecutive points of a
    skewer must therefore be closer in v_c than a fraction of the line width:
    |v_c,i+1 - v_c,i| <= tolerance x min(b_i, b_i+1). For a top hat of width W replaced by a delta,
    the peak of a Gaussian of Doppler parameter b changes by W^2 / (12 b^2): 0.5 % for W = b/4.

    Parameters
    ----------
    line_centre_kms, b_kms, weight : (N_lines, N_sk, N_ppsk) arrays (weight = n w, the column of
        each point, to report the fraction of absorbers affected). ``b_kms`` is the width of the
        whole profile (``voigt_width_kms``): the thermal b for a Doppler-dominated line, the
        Lorentzian width for a natural one.

    Returns
    -------
    dict(max_ratio (N_lines,), column_fraction (N_lines,) of the column in under-resolved intervals,
    ok (bool), b_min (N_lines,)).
    """
    dv = np.abs(np.diff(np.asarray(line_centre_kms), axis=-1))
    b = np.asarray(b_kms)
    bmin = np.maximum(np.minimum(b[..., 1:], b[..., :-1]), 1e-30)
    ratio = dv / bmin
    w = np.asarray(weight)
    wm = 0.5 * (w[..., 1:] + w[..., :-1])
    bad = ratio > tolerance
    tot = np.maximum(wm.sum(axis=(-2, -1)), 1e-300)
    frac = (wm * bad).sum(axis=(-2, -1)) / tot
    return dict(max_ratio=ratio.max(axis=(-2, -1)), column_fraction=frac, ok=bool(np.all(frac < 1e-3)),
                b_min=b.min(axis=(-2, -1)))


class OpticalDepthCalculator:
    """
    Memory-light optical-depth calculator for line-centered velocity coordinates.

    Main stored result
    ------------------
    self.optical_depth : array, shape (N_lines, N_sk, N_v)
        Optical depth for each line, skewer, and velocity pixel.

    Memory policy
    -------------
    By default, after compute() finishes, this class releases references to the
    heavy input containers:

        self.skewer_bundle = None
        self.fields = None

    but keeps

        self.line_bundle
        self.a_dagger
        self.velocities_kms
        self.optical_depth

    because these may be useful afterwards, especially for wrap_to_ref_line()
    and metadata inspection.

    If you need to call compute() repeatedly on the same instance, set

        release_inputs_after_compute=False
    """

    def __init__(
        self,
        skewer_bundle=None,
        velocities_kms: Optional[jnp.ndarray] = None,
        line_bundle=None,
        fields=None,
        optical_depth: Optional[ArrayLike] = None,
        a_dagger: Optional[ArrayLike] = None,
        ii_cosmology: int = 0,
        voigt_implementation_mode: str = "ExoJAX",
        mode_integration: str = "trapz",
        debug_memory: bool = False,
        sync_memory: bool = True,
        *,
        release_inputs_after_compute: bool = True,
        line_elements: Optional[ArrayLike] = None,
        velocity_period: Optional[float] = None,
        detuning: str = "exact",
        resolution_tolerance: Optional[float] = 0.25,
        v_chunk: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        skewer_bundle :
            Skewer container. Required only in compute mode. Expected attributes:
            `quadri_vectors`, `los_directions`, `array_direction`, `N_sk`,
            `N_ppsk`, and `a0`.

        velocities_kms : array, shape (N_v,)
            Common line-centered velocity grid in km/s. Required in both modes.

        line_bundle :
            Line container. Required in both modes. Expected attributes:
            `n_lines`, `n_species`, `species_keys`, `f_osc`, `delta_v`,
            and `indices_for_species`.

        fields :
            Field container. Required only in compute mode. Must provide
            `tile_field_to_lines(...)` and all fields required by
            `line_profile.compute_line_profiles`.

        optical_depth : array, optional, shape (N_lines, N_sk, N_v)
            If provided, activate loading mode.

        a_dagger : scalar or array, optional
            Reference scale factor for the velocity coordinate. Accepted forms:
            scalar, shape (N_sk,), shape (N_sk, N_ppsk), or shape
            (N_sk * N_ppsk,). If None, uses `skewer_bundle.a0`.

        debug_memory : bool
            If True, log process-memory checkpoints.

        sync_memory : bool
            If True, call `.block_until_ready()` before memory checkpoints.

        release_inputs_after_compute : bool
            If True, release `skewer_bundle` and `fields` after compute().
            `line_bundle` and `a_dagger` are always kept.

        line_elements : array, optional, shape (N_sk, N_ppsk)
            Comoving length [Mpc] that each point represents in the quadrature
            tau = sum_i w_i dtau/dx_i. None (default, ``mode_integration="trapz"``) takes the
            trapezoid weights of the points, ``SkewerBundle.line_elements``: identical to the
            previous ``jax.scipy.integrate.trapezoid`` for point-sampled fields. Explicit elements
            are used when the fields are segment averages (e.g. the column of a cell inside the
            segment, ``skewerjax.fields.deposition``), which conserves the column whatever the
            sampling.

        velocity_period : float, optional
            Period [km/s] of a periodic box. Each point then absorbs only through the nearest
            periodic image of its detuning (``line_profile.compute_nu_offset_GHz``).

        detuning : "exact" (default) or "linear" (fake_spectra's first-order mapping; validation).

        resolution_tolerance : float or None
            Warn when consecutive points are farther apart in line-centre velocity than this
            fraction of the local width of the Voigt profile (``voigt_width_kms``: the thermal b,
            or the natural width for T -> 0; ``sampling_resolution_report``). None disables.

        v_chunk : int or None
            Velocities per kernel call (bounds memory by N_lines x N_points x v_chunk).
        """
        if mode_integration not in ("trapz", "line_elements"):
            raise NotImplementedError(f"Integration mode {mode_integration!r} not implemented.")
        if mode_integration == "line_elements" and line_elements is None and optical_depth is None:
            raise ValueError("mode_integration='line_elements' needs explicit `line_elements` (N_sk, N_ppsk); "
                             "with line_elements=None the trapezoid weights of the points are used ('trapz').")
        if detuning not in ("exact", "linear"):
            raise ValueError(f"detuning must be 'exact' or 'linear', got {detuning!r}.")
        self.line_elements = None if line_elements is None else jnp.asarray(line_elements)
        self.velocity_period = None if velocity_period is None else float(velocity_period)
        self.detuning = detuning
        self.resolution_tolerance = resolution_tolerance
        self.v_chunk = v_chunk
        self.resolution_report = None
        if velocities_kms is None:
            raise ValueError("`velocities_kms` is required in both compute and loading modes.")

        if line_bundle is None:
            raise ValueError("`line_bundle` is required in both compute and loading modes.")

        # --------------------------------------------------------------
        # Mode flag
        # --------------------------------------------------------------
        self.loaded_mode = optical_depth is not None

        # --------------------------------------------------------------
        # Persistent lightweight / needed metadata
        # --------------------------------------------------------------
        self.line_bundle = line_bundle
        self.velocities_kms = jnp.asarray(velocities_kms)
        self.a_dagger = a_dagger

        self.ii_cosmology = ii_cosmology
        self.voigt_implementation_mode = voigt_implementation_mode
        self.mode_integration = mode_integration

        self.release_inputs_after_compute = release_inputs_after_compute
        self._inputs_released = False
        self._computed_once = False

        # --------------------------------------------------------------
        # Memory-debugging state
        # --------------------------------------------------------------
        self.debug_memory = debug_memory
        self.sync_memory = sync_memory
        self._proc = psutil.Process(os.getpid()) if psutil is not None else None
        self._last_rss_bytes = None

        # --------------------------------------------------------------
        # Dimensions and cached small arrays from line_bundle
        # --------------------------------------------------------------
        self.N_lines = int(line_bundle.n_lines)
        self.N_species = int(getattr(line_bundle, "n_species", 0))
        self.N_v = int(self.velocities_kms.size)

        self.delta_v = jnp.asarray(line_bundle.delta_v)
        self.f_osc = jnp.asarray(line_bundle.f_osc)
        self.species_keys = tuple(getattr(line_bundle, "species_keys", ()))

        if self.delta_v.shape != (self.N_lines,):
            raise ValueError(
                f"`line_bundle.delta_v` has shape {self.delta_v.shape}; "
                f"expected {(self.N_lines,)}."
            )

        if self.f_osc.shape[0] != self.N_lines:
            raise ValueError(
                "`line_bundle.f_osc` must have one entry per line. "
                f"Got shape {self.f_osc.shape}, N_lines={self.N_lines}."
            )

        self.indices_for_species: Mapping[str, jnp.ndarray] = {
            sp: jnp.asarray(line_bundle.indices_for_species(sp))
            for sp in self.species_keys
        }

        # --------------------------------------------------------------
        # Main result
        # --------------------------------------------------------------
        self.optical_depth = None

        if self.loaded_mode:
            tau = jnp.asarray(optical_depth)

            if tau.ndim != 3:
                raise ValueError(
                    "`optical_depth` must have shape (N_lines, N_sk, N_v). "
                    f"Got {tau.shape}."
                )

            if tau.shape[0] != self.N_lines:
                raise ValueError(
                    "`optical_depth.shape[0]` must match `line_bundle.n_lines`. "
                    f"Got optical_depth.shape[0]={tau.shape[0]} and "
                    f"line_bundle.n_lines={self.N_lines}."
                )

            if tau.shape[2] != self.N_v:
                raise ValueError(
                    "`optical_depth.shape[2]` must match len(velocities_kms). "
                    f"Got optical_depth.shape[2]={tau.shape[2]} and "
                    f"len(velocities_kms)={self.N_v}."
                )

            self.N_sk = int(tau.shape[1])
            self.N_ppsk = None
            self.N_points = None

            self.optical_depth = tau

            # No compute-mode heavy inputs needed in loading mode.
            self.skewer_bundle = None
            self.fields = None
            self._inputs_released = True

            logger.info(
                "[OPTICAL_DEPTH] initialized in loading mode: optical_depth.shape=%s",
                self.optical_depth.shape,
            )
            self._log_mem("initialized from precomputed optical_depth", self.optical_depth)

        else:
            if skewer_bundle is None:
                raise ValueError("`skewer_bundle` is required when `optical_depth` is not provided.")

            if fields is None:
                raise ValueError("`fields` is required when `optical_depth` is not provided.")

            self.skewer_bundle = skewer_bundle
            self.fields = fields

            self.N_sk = int(skewer_bundle.N_sk)
            self.N_ppsk = int(skewer_bundle.N_ppsk)
            self.N_points = self.N_sk * self.N_ppsk

            logger.info(
                "[OPTICAL_DEPTH] initialized in compute mode: "
                "N_lines=%d, N_species=%d, N_sk=%d, N_ppsk=%d, N_v=%d.",
                self.N_lines,
                self.N_species,
                self.N_sk,
                self.N_ppsk,
                self.N_v,
            )
            self._log_mem("initialized")

    # ------------------------------------------------------------------
    # Memory cleanup helpers
    # ------------------------------------------------------------------

    def release_inputs(self):
        """
        Release references to heavy compute-mode input containers.

        This intentionally keeps:
            - self.line_bundle
            - self.a_dagger
            - self.velocities_kms
            - self.optical_depth
        """
        self.skewer_bundle = None
        self.fields = None
        self._inputs_released = True

        logger.info(
            "[OPTICAL_DEPTH] released heavy input containers: skewer_bundle and fields."
        )

    def clear_optical_depth(self):
        """
        Explicitly release the stored optical-depth result.

        Use this only after downstream objects no longer need it.
        """
        self.optical_depth = None
        logger.info("[OPTICAL_DEPTH] cleared stored optical_depth.")

    # ------------------------------------------------------------------
    # Small utilities
    # ------------------------------------------------------------------

    @staticmethod
    def compute_comoving_distance(points: jnp.ndarray) -> jnp.ndarray:
        """
        Compute cumulative comoving distance along each skewer.

        Parameters
        ----------
        points : array, shape (N_sk, N_ppsk, 4)
            Last axis is [a, x, y, z].

        Returns
        -------
        distances : array, shape (N_sk, N_ppsk)
            Cumulative Euclidean comoving distance along each skewer.
        """
        def one(skewer_points):
            dx = skewer_points[1:, 1:] - skewer_points[:-1, 1:]
            step = jnp.linalg.norm(dx, axis=-1)
            return jnp.concatenate([jnp.zeros(1), jnp.cumsum(step)])

        return vmap(one)(points)

    def _gb(self, nbytes) -> float:
        """Convert bytes to GiB."""
        return float(nbytes) / 1024**3

    def _mem_bar(self, fraction: float, width: int = 28) -> str:
        """Return a visual percentage bar."""
        fraction = max(0.0, min(1.0, float(fraction)))
        n_fill = int(round(width * fraction))
        return f"[{'█' * n_fill}{'░' * (width - n_fill)}] {100.0 * fraction:5.1f}%"

    def _log_mem(self, label: str, arr=None):
        """
        Log visual memory diagnostics.

        Reports:
        - process RSS memory;
        - process RSS as a fraction of total system RAM;
        - total system RAM usage;
        - change in process RSS since previous checkpoint;
        - optional array shape, dtype, and nominal nbytes.
        """
        if not self.debug_memory:
            return

        if self.sync_memory and hasattr(arr, "block_until_ready"):
            arr.block_until_ready()

        if self._proc is None:
            logger.info("[OPTICAL_DEPTH][MEM] %s | psutil unavailable", label)
            return

        mem, vm = self._proc.memory_info(), psutil.virtual_memory()

        rss_bytes = mem.rss
        rss_gb = self._gb(mem.rss)
        total_gb = self._gb(vm.total)
        rss_fraction = rss_bytes / vm.total
        system_used_fraction = vm.percent / 100.0

        if self._last_rss_bytes is None:
            delta_txt = "ΔRSS=first checkpoint"
        else:
            delta_bytes = rss_bytes - self._last_rss_bytes
            sign = "+" if delta_bytes >= 0 else "-"
            delta_txt = f"ΔRSS={sign}{self._gb(abs(delta_bytes)):.3f} GB"

        self._last_rss_bytes = rss_bytes

        msg = (
            "\n"
            f"[OPTICAL_DEPTH][MEM] {label}\n"
            f"  Process RSS : {rss_gb:.3f} / {total_gb:.3f} GB  {self._mem_bar(rss_fraction)}\n"
            f"  System RAM  : used={self._gb(vm.used):.3f} GB, "
            f"available={self._gb(vm.available):.3f} GB  {self._mem_bar(system_used_fraction)}\n"
            f"  Change      : {delta_txt}"
        )

        if arr is not None:
            shape = getattr(arr, "shape", None)
            dtype = getattr(arr, "dtype", None)
            nbytes = getattr(arr, "nbytes", None)

            if nbytes is None:
                msg += f"\n  Array       : shape={shape}, dtype={dtype}, nbytes=?"
            else:
                msg += (
                    f"\n  Array       : shape={shape}, dtype={dtype}, "
                    f"nbytes={self._gb(nbytes):.3f} GB  {self._mem_bar(nbytes / vm.total)}"
                )

        logger.info(msg)

    def _prepare_a_dagger(self):
        """
        Convert `a_dagger` to the format expected by `compute_line_profiles`.

        Returns
        -------
        scalar or array, shape (N_sk * N_ppsk,)
        """
        if self.loaded_mode:
            raise RuntimeError("`_prepare_a_dagger()` is unavailable in loading mode.")

        if self.skewer_bundle is None:
            raise RuntimeError(
                "Cannot prepare `a_dagger`: skewer inputs have been released."
            )

        a = jnp.asarray(
            self.skewer_bundle.a0 if self.a_dagger is None else self.a_dagger
        )

        if a.ndim == 0:
            return a

        if a.shape == (self.N_sk,):
            return jnp.repeat(a[:, None], self.N_ppsk, axis=1).reshape(self.N_points)

        if a.shape == (self.N_sk, self.N_ppsk):
            return a.reshape(self.N_points)

        if a.shape == (self.N_points,):
            return a

        raise ValueError(
            f"`a_dagger` must be None, scalar, ({self.N_sk},), "
            f"({self.N_sk}, {self.N_ppsk}), or ({self.N_points},). Got {a.shape}."
        )

    # ------------------------------------------------------------------
    # Main optical-depth calculation
    # ------------------------------------------------------------------

    def compute(self) -> jnp.ndarray:
        """
        Compute or return optical depth.

        The fields are sampled at the points (any Python callables), then the pure kernel
        ``optical_depth_kernel`` evaluates tau = sum_i w_i dtau/dx_i, in velocity chunks.

        Returns
        -------
        optical_depth : array, shape (N_lines, N_sk, N_v)
        """
        if self.loaded_mode:
            if self.optical_depth is None:
                raise ValueError("Loaded mode is active, but `self.optical_depth` is None.")
            return self.optical_depth

        if self.optical_depth is not None and self._inputs_released:
            return self.optical_depth

        if self.skewer_bundle is None or self.fields is None:
            raise RuntimeError(
                "Cannot compute optical depth because compute-mode inputs are missing. "
                "This usually means `release_inputs_after_compute=True` and compute() "
                "has already been called. Create a new calculator or set "
                "`release_inputs_after_compute=False`."
            )

        logger.info("[OPTICAL_DEPTH] start: ii_cosmology=%d, voigt=%s", self.ii_cosmology, self.voigt_implementation_mode)

        # 1. Geometry: quadri-vectors, photon directions, line elements [comoving m].
        qv = self.skewer_bundle.quadri_vectors[self.ii_cosmology]
        los = self.skewer_bundle.los_directions
        if self.line_elements is not None:
            w = self.line_elements
            if w.shape != (self.N_sk, self.N_ppsk):
                raise ValueError(f"line_elements must be ({self.N_sk}, {self.N_ppsk}), got {w.shape}.")
        else:
            w = trapezoid_line_elements(qv[..., 1:])
        w_m = (w * phys.Mpc_to_m).reshape(self.N_points)

        # 2. Sample the fields once (tiled to lines).
        qv_flat = qv.reshape(self.N_points, 4)
        los_flat = los.reshape(self.N_points, 3)
        tile = lambda name, **kw: self.fields.tile_field_to_lines(
            name, qv_flat, indices_for_species=self.indices_for_species, **kw)
        temperature, turbulence, collisional = tile("temperature"), tile("turbulence"), tile("collisional")
        los_velocity = tile("LoS_velocity", LoS_directions=los_flat)
        n_cm3 = tile("number_density")
        a = qv_flat[:, 0]
        a_dagger = self._prepare_a_dagger()
        lb = self.line_bundle
        lam, A21, mass = (jnp.asarray(lb.lambda0_A), jnp.asarray(lb.A21_GHz), jnp.asarray(lb.mass_amu))
        self._log_mem("sampled fields", n_cm3)

        # 3. Optional sampling check (line-centre spacing vs the width of the Voigt profile).
        if self.resolution_tolerance is not None:
            v_c = line_profile.line_centre_velocity_kms(a_dagger, a, los_velocity)
            sigma, gamma = line_profile.voigt_widths_GHz(temperature, turbulence, collisional, lam, A21, mass)
            b = voigt_width_kms(sigma, gamma, lam)
            shp = (self.N_lines, self.N_sk, self.N_ppsk)
            rep = sampling_resolution_report(v_c.reshape(shp), b.reshape(shp),
                                             (n_cm3 * w_m[None, :]).reshape(shp), self.resolution_tolerance)
            self.resolution_report = rep
            if not rep["ok"]:
                warnings.warn(
                    "OpticalDepthCalculator: the points are too sparse for the line widths: "
                    + ", ".join(f"line {j}: max dv_c / width = {rep['max_ratio'][j]:.2f}, column fraction "
                                f"{rep['column_fraction'][j]:.1e} above tolerance {self.resolution_tolerance}"
                                for j in range(self.N_lines))
                    + ". Sample more densely (spacing <= tolerance x b_min / (dv/dx)).", stacklevel=2)

        # 4. Kernel, per skewer and velocity chunk (memory ~ N_lines x N_ppsk x v_chunk).
        nv = self.N_v if self.v_chunk is None else int(self.v_chunk)
        per_point = lambda arr: arr.reshape(arr.shape[:-1] + (self.N_sk, self.N_ppsk))
        T_s, tb_s, co_s, lv_s, n_s = (per_point(x) for x in (temperature, turbulence, collisional, los_velocity, n_cm3))
        a_s, w_s = a.reshape(self.N_sk, self.N_ppsk), w_m.reshape(self.N_sk, self.N_ppsk)
        a_d = jnp.broadcast_to(jnp.asarray(a_dagger), (self.N_points,)).reshape(self.N_sk, self.N_ppsk)
        taus = []
        for k in range(self.N_sk):
            chunks = []
            for v0 in range(0, self.N_v, nv):
                chunks.append(optical_depth_kernel(
                    self.velocities_kms[v0:v0 + nv], lam, A21, mass, self.f_osc, a_d[k], a_s[k], lv_s[:, k], T_s[:, k],
                    tb_s[:, k], co_s[:, k], n_s[:, k], w_s[k], voigt_implementation_mode=self.voigt_implementation_mode,
                    detuning=self.detuning, velocity_period=self.velocity_period))
            taus.append(jnp.concatenate(chunks, axis=-1))
        self.optical_depth = jnp.stack(taus, axis=1)                      # (N_lines, N_sk, N_v)
        self._computed_once = True
        self._log_mem("after optical_depth_kernel", self.optical_depth)

        if self.release_inputs_after_compute:
            self.release_inputs()
        return self.optical_depth

    # ------------------------------------------------------------------
    # Velocity-frame wrapping
    # ------------------------------------------------------------------

    def wrap_to_ref_line(self) -> jnp.ndarray:
        """
        Put each line's optical depth on the reference line's velocity frame (periodic shift).

        A feature at v_j in line j's own frame appears at v_ref = v_j - delta_v_j, with
        delta_v_j = c ln(lambda_ref / lambda_j): tau_j^ref(v) = tau_j(v + delta_v_j). Lines bluer than
        the reference (delta_v_j > 0, e.g. Si III 1206.5 for Lya) move to lower velocities. The
        velocity grid is treated as periodic with period N_v dv (snapshot boxes).

        This method overwrites `self.optical_depth`.

        Returns
        -------
        optical_depth : array, shape (N_lines, N_sk, N_v)
            Optical depth wrapped onto the reference-line velocity frame.
        """
        if self.optical_depth is None:
            raise ValueError(
                "No optical depth stored. Run `.compute()` first or initialize "
                "with `optical_depth=`."
            )

        tau = jnp.asarray(self.optical_depth)
        delta_v = jnp.asarray(self.line_bundle.delta_v)

        expected_shape = (self.N_lines, self.N_sk, self.N_v)
        if tau.shape != expected_shape:
            raise ValueError(
                f"`optical_depth` has shape {tau.shape}; expected {expected_shape}."
            )
        if delta_v.shape != (self.N_lines,):
            raise ValueError(
                f"`line_bundle.delta_v` has shape {delta_v.shape}; "
                f"expected {(self.N_lines,)}."
            )

        # Uniform periodic velocity grid.
        dv = self.velocities_kms[1] - self.velocities_kms[0]
        v0 = self.velocities_kms[0]
        period = self.N_v * dv

        def shift_profile(tau_1d, shift):
            # Evaluate tau(v + shift) periodically on the original velocity grid.
            vq = jnp.mod(self.velocities_kms + shift - v0, period) + v0
            return jnp.interp(vq, self.velocities_kms, tau_1d, period=period)

        wrapped = vmap(
            lambda tau_line, shift: vmap(lambda tau_sk: shift_profile(tau_sk, shift))(tau_line),
            in_axes=(0, 0),
        )(tau, delta_v)

        # Overwrite the unwrapped result. Do not keep both.
        self.optical_depth = wrapped
        self._log_mem("after wrap_to_ref_line", self.optical_depth)
        return self.optical_depth
