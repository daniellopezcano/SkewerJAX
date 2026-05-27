from __future__ import annotations

from typing import Optional, Mapping
from jax.typing import ArrayLike

import os
import logging
logger = logging.getLogger(__name__)

try:
    import psutil
except ImportError:
    psutil = None

from jax import vmap
import jax.numpy as jnp
from jax.scipy.integrate import trapezoid

from skewerjax.constants.physical_constants import phys
from skewerjax.optical_depth import line_profile


class OpticalDepthCalculator:
    """
    Simple optical-depth calculator for line-centered velocity coordinates.

    Main stored result
    ------------------
    self.optical_depth : array, shape (N_lines, N_sk, N_v)
        Optical depth for each line, skewer, and velocity pixel.

    Typical usage
    -------------
    od = OpticalDepthCalculator(
        skewer_bundle=skewer_bundle,
        velocities_kms=velocities_kms,
        line_bundle=line_bundle,
        fields=fields,
        debug_memory=True,
    )

    tau = od.compute()
    tau = od.wrap_to_ref_line()
    """

    def __init__(
        self,
        skewer_bundle,
        velocities_kms: jnp.ndarray,
        line_bundle,
        fields,
        a_dagger: Optional[ArrayLike] = None,
        ii_cosmology: int = 0,
        voigt_implementation_mode: str = "ExoJAX",
        mode_integration: str = "trapz",
        debug_memory: bool = False,
        sync_memory: bool = True,
    ):
        """
        Parameters
        ----------
        skewer_bundle :
            Skewer container. Expected attributes:
            `quadri_vectors`, `los_directions`, `array_direction`, `N_sk`,
            `N_ppsk`, and `a0`.

        velocities_kms : array, shape (N_v,)
            Common line-centered velocity grid in km/s.

        line_bundle :
            Line container. Expected attributes:
            `n_lines`, `n_species`, `species_keys`, `f_osc`, `delta_v`,
            and `indices_for_species`.

        fields :
            Field container. Must provide `tile_field_to_lines(...)` and
            all fields required by `line_profile.compute_line_profiles`.

        a_dagger : scalar or array, optional
            Reference scale factor for the velocity coordinate. Accepted forms:
            scalar, shape (N_sk,), shape (N_sk, N_ppsk), or shape
            (N_sk * N_ppsk,). If None, uses `skewer_bundle.a0`.

        debug_memory : bool
            If True, log visual process-memory checkpoints.

        sync_memory : bool
            If True, call `.block_until_ready()` before memory checkpoints.
            This is useful because JAX execution is often asynchronous.
        """
        # Store external containers and numerical options.
        self.skewer_bundle, self.line_bundle, self.fields = skewer_bundle, line_bundle, fields
        self.velocities_kms = jnp.asarray(velocities_kms)
        self.a_dagger, self.ii_cosmology = a_dagger, ii_cosmology
        self.voigt_implementation_mode, self.mode_integration = voigt_implementation_mode, mode_integration

        # Memory-debugging state.
        self.debug_memory, self.sync_memory = debug_memory, sync_memory
        self._proc = psutil.Process(os.getpid()) if psutil is not None else None
        self._last_rss_bytes = None

        # Basic dimensions stored for later validation and reshaping.
        self.N_lines, self.N_species = int(line_bundle.n_lines), int(line_bundle.n_species)
        self.N_sk, self.N_ppsk = int(skewer_bundle.N_sk), int(skewer_bundle.N_ppsk)
        self.N_points, self.N_v = self.N_sk * self.N_ppsk, int(self.velocities_kms.size)

        # Filled after compute().
        self.optical_depth = None

        self._log_mem("initialized")

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

        # Force previous JAX work to finish before measuring memory.
        if self.sync_memory and hasattr(arr, "block_until_ready"):
            arr.block_until_ready()

        if self._proc is None:
            logger.info("[OPTICAL_DEPTH][MEM] %s | psutil unavailable", label)
            return

        mem, vm = self._proc.memory_info(), psutil.virtual_memory()
        rss_bytes, rss_gb, total_gb = mem.rss, self._gb(mem.rss), self._gb(vm.total)
        rss_fraction, system_used_fraction = rss_bytes / vm.total, vm.percent / 100.0

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
            shape, dtype, nbytes = getattr(arr, "shape", None), getattr(arr, "dtype", None), getattr(arr, "nbytes", None)
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
        a = jnp.asarray(self.skewer_bundle.a0 if self.a_dagger is None else self.a_dagger)

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
        Compute optical depth and store it in `self.optical_depth`.

        Returns
        -------
        optical_depth : array, shape (N_lines, N_sk, N_v)
        """
        logger.info("[OPTICAL_DEPTH] start: ii_cosmology=%d, voigt=%s", self.ii_cosmology, self.voigt_implementation_mode)

        # 1. Load skewer geometry for the requested cosmology.
        qv = self.skewer_bundle.quadri_vectors[self.ii_cosmology]  # (N_sk, N_ppsk, 4)
        los = self.skewer_bundle.los_directions                    # (N_sk, N_ppsk, 3)
        self._log_mem("loaded quadri_vectors", qv)

        # 2. Compute line profiles phi(line, skewer, point, velocity).
        phi = line_profile.compute_line_profiles(
            quadri_vectors=qv,
            LoS_directions=los,
            velocities_kms=self.velocities_kms,
            a_dagger=self._prepare_a_dagger(),
            line_bundle=self.line_bundle,
            fields=self.fields,
            voigt_implementation_mode=self.voigt_implementation_mode,
        )
        self._log_mem("after compute_line_profiles", phi)

        # 3. Accept either flattened or expanded profile shapes.
        flat_shape = (self.N_lines, self.N_sk, self.N_ppsk * self.N_v)
        full_shape = (self.N_lines, self.N_sk, self.N_ppsk, self.N_v)

        if phi.shape == flat_shape:
            phi = phi.reshape(full_shape)
        elif phi.shape != full_shape:
            raise ValueError(f"Unexpected phi shape {phi.shape}. Expected {flat_shape} or {full_shape}.")

        self._log_mem("after phi reshape", phi)

        # 4. Tile species-level number-density fields onto line-level arrays.
        indices: Mapping[str, jnp.ndarray] = {
            sp: self.line_bundle.indices_for_species(sp)
            for sp in self.line_bundle.species_keys
        }

        n_cm3 = self.fields.tile_field_to_lines(
            "number_density",
            qv.reshape(self.N_points, 4),
            indices_for_species=indices,
        ).reshape(self.N_lines, self.N_sk, self.N_ppsk)

        self._log_mem("after number_density tiling", n_cm3)

        # 5. Build d tau / dx.
        #
        # sigma = prefactor * f_osc * phi
        #
        # Shapes:
        #   sigma  : (N_lines, N_sk, N_ppsk, N_v)
        #   n_cm3  : (N_lines, N_sk, N_ppsk)
        #   a      : (N_sk, N_ppsk)
        #
        # The 1e2 factor keeps the same unit convention as the original function.
        sigma = phys.cross_section_prefactor * self.line_bundle.f_osc[:, None, None, None] * phi
        dtau_dx = sigma * n_cm3[..., None] * qv[..., 0][None, ..., None] ** -2 * 1e2

        self._log_mem("after dtau_dx", dtau_dx)

        # 6. Reorient skewers so integration follows the photon path direction.
        mask = self.skewer_bundle.array_direction == -1

        qv_oriented = jnp.where(mask[:, None, None], jnp.flip(qv, axis=1), qv)
        x_m = self.compute_comoving_distance(qv_oriented) * phys.Mpc_to_m

        dtau_dx = jnp.where(mask[None, :, None, None], jnp.flip(dtau_dx, axis=2), dtau_dx)

        self._log_mem("after orientation", dtau_dx)

        # 7. Integrate along the skewer-point axis.
        if self.mode_integration != "trapz":
            raise NotImplementedError(f"Integration mode {self.mode_integration!r} not implemented.")

        self.optical_depth = trapezoid(dtau_dx, x=x_m[None, :, :, None], axis=2)

        self._log_mem("after trapz integration / optical_depth", self.optical_depth)
        logger.info("[OPTICAL_DEPTH] done: optical_depth.shape=%s", self.optical_depth.shape)

        return self.optical_depth

    # ------------------------------------------------------------------
    # Velocity-frame wrapping
    # ------------------------------------------------------------------

    def wrap_to_ref_line(self) -> jnp.ndarray:
        """
        Periodically shift each line optical-depth field by `line_bundle.delta_v`.

        This method overwrites `self.optical_depth`.

        Returns
        -------
        optical_depth : array, shape (N_lines, N_sk, N_v)
            Optical depth wrapped onto the reference-line velocity frame.
        """
        if self.optical_depth is None:
            raise ValueError("No optical depth stored. Run `.compute()` first.")

        tau = jnp.asarray(self.optical_depth)
        delta_v = jnp.asarray(self.line_bundle.delta_v)

        # Shape validation before interpolation.
        if tau.shape != (self.N_lines, self.N_sk, self.N_v):
            raise ValueError(f"`optical_depth` has shape {tau.shape}; expected {(self.N_lines, self.N_sk, self.N_v)}.")

        if delta_v.shape != (self.N_lines,):
            raise ValueError(f"`line_bundle.delta_v` has shape {delta_v.shape}; expected {(self.N_lines,)}.")

        # Uniform periodic velocity grid.
        dv = self.velocities_kms[1] - self.velocities_kms[0]
        v0 = self.velocities_kms[0]
        period = self.N_v * dv

        def shift_profile(tau_1d, shift):
            # Evaluate tau(v - shift) periodically on the original velocity grid.
            vq = jnp.mod(self.velocities_kms - shift - v0, period) + v0
            return jnp.interp(vq, self.velocities_kms, tau_1d, period=period)

        # vmap over skewers inside each line, then over lines.
        self.optical_depth = vmap(
            lambda tau_line, shift: vmap(lambda tau_sk: shift_profile(tau_sk, shift))(tau_line),
            in_axes=(0, 0),
        )(tau, delta_v)

        self._log_mem("after wrap_to_ref_line", self.optical_depth)
        return self.optical_depth