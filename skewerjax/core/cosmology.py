"""Background cosmology: the scale factor along a photon path, a(x) (review notes)."""

from __future__ import annotations

import diffrax
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from skewerjax.core.constants import phys
from skewerjax.core.validation import check_values


def da_dx(xx, aa, params, cc=phys.cc_Gpc_per_Gyr):
    """Photon geodesic da/dx = (H0 / c) sqrt(Om_r + Om_m a + Om_k a^2 + Om_L a^4) / sqrt(1 - k x^2) (notes Eq. dadx_general).

    Parameters
    ----------
    xx : comoving distance [Gpc]; aa : scale factor; params : (H0 [1/Gyr], Om_r, Om_m, Om_k, Om_L);
    cc : speed of light [Gpc/Gyr].
    """
    H0, omega_r, omega_m, omega_k, omega_lambda = params
    kk = -omega_k / (cc**2)
    denominator = jnp.maximum(1e-32, 1 - kk * xx**2)
    numerator = omega_r + omega_m * aa + omega_k * aa**2 + omega_lambda * aa**4
    return (H0 / cc) * jnp.sqrt(numerator / denominator)


@jax.tree_util.register_pytree_node_class
class Cosmology:
    """A batch of FLRW backgrounds (one per entry of the parameter arrays).

    Parameters
    ----------
    H0 : Hubble constant [1/Gyr] (``phys.H0`` for 70 km/s/Mpc); scalar or (N_cosmologies,).
    Omega_r_0, Omega_m_0, Omega_k_0, Omega_lambda_0 : density parameters today, broadcast-compatible with H0.

    Notes
    -----
    JAX pytree with the five parameter arrays as leaves (traceable, differentiable); the closure check
    Om_r + Om_m + Om_k + Om_L = 1 (to 1e-4) is only done on concrete values.
    """

    def __init__(self, H0, Omega_r_0, Omega_m_0, Omega_k_0, Omega_lambda_0):
        p = [jnp.atleast_1d(v) for v in (H0, Omega_r_0, Omega_m_0, Omega_k_0, Omega_lambda_0)]
        try:
            jnp.broadcast_shapes(*(v.shape for v in p))
        except ValueError as e:
            raise ValueError(
                "Incompatible shapes for cosmological parameters: " + ", ".join(f"{n}: {v.shape}" for n, v in zip(self._names, p))
            ) from e
        self.H0, self.Omega_r_0, self.Omega_m_0, self.Omega_k_0, self.Omega_lambda_0 = p
        if not any(isinstance(v, jax.core.Tracer) for v in p):
            for n, v in zip(self._names, p):
                check_values(f"Cosmology: {n}", v, minimum=0.0 if n == "H0" else None, strict=True)
            density_sum = p[1] + p[2] + p[3] + p[4]
            if not jnp.allclose(density_sum, 1.0, atol=1e-4):
                raise ValueError(f"Cosmological densities do not sum to 1: {density_sum}")

    _names = ("H0", "Omega_r_0", "Omega_m_0", "Omega_k_0", "Omega_lambda_0")

    def tree_flatten(self):
        return tuple(getattr(self, n) for n in self._names), None

    @classmethod
    def tree_unflatten(cls, aux, leaves):
        obj = object.__new__(cls)
        for n, v in zip(cls._names, leaves):
            setattr(obj, n, v)
        return obj

    @property
    def N_cosmologies(self) -> int:
        return self.H0.shape[0]

    @property
    def params(self) -> jnp.ndarray:
        """(N_cosmologies, 5) stacked (H0, Om_r, Om_m, Om_k, Om_L)."""
        return jnp.stack([self.H0, self.Omega_r_0, self.Omega_m_0, self.Omega_k_0, self.Omega_lambda_0], axis=-1)

    def solve_a_of_x(
        self,
        ICs: ArrayLike,
        x_eval_Mpc: ArrayLike,
        solver: diffrax.AbstractSolver = diffrax.Tsit5(),
        rtol: float = 1e-4,
        atol: float = 1e-4,
        max_steps: int = 1000,
    ) -> jnp.ndarray:
        """Scale factor a(x) by integrating Eq. dadx_general with ``diffrax`` (one solve per cosmology, vmapped).

        Parameters
        ----------
        ICs : (N_cosmologies,) a(x_eval[0]).
        x_eval_Mpc : (N_eval,) comoving distances [Mpc] (monotonic; the first one is the initial condition).
        solver, rtol, atol, max_steps : ``diffrax`` options.

        Returns
        -------
        (N_cosmologies, N_eval) scale factors.
        """

        def solve_one(H0, Omega_r_0, Omega_m_0, Omega_k_0, Omega_lambda_0, a0, x_eval_Gpc):
            sol = diffrax.diffeqsolve(
                diffrax.ODETerm(da_dx),
                solver=solver,
                t0=x_eval_Gpc[0],
                t1=x_eval_Gpc[-1],
                dt0=None,
                y0=a0,
                args=jnp.array([H0, Omega_r_0, Omega_m_0, Omega_k_0, Omega_lambda_0]),
                saveat=diffrax.SaveAt(ts=x_eval_Gpc),
                stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
                max_steps=max_steps,
            )
            return sol.ys

        ICs = jnp.atleast_1d(ICs)
        if ICs.shape[0] != self.N_cosmologies:
            raise ValueError(f"ICs shape {ICs.shape} incompatible with number of cosmologies {self.N_cosmologies}.")
        ys = jax.vmap(solve_one, in_axes=(0, 0, 0, 0, 0, 0, None))(
            self.H0, self.Omega_r_0, self.Omega_m_0, self.Omega_k_0, self.Omega_lambda_0, ICs[:, None], x_eval_Mpc * 1e-3
        )  # Mpc -> Gpc
        return ys[..., 0]
