"""Background cosmology: the scale factor along a photon path, a(x) (review notes)."""

from __future__ import annotations

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from skewerjax.core.constants import phys
from skewerjax.core.validation import check_values


def sound_horizon_drag(omega_b, omega_cb, omega_nu=0.0):
    """The sound horizon at the drag epoch r_d [Mpc] from the physical densities omega_b = Omega_b h^2, omega_cb =
    (Omega_c + Omega_b) h^2 and omega_nu = Omega_nu h^2 (default 0, massless neutrinos), by the fit of Aubourg et al.
    2015 (PRD 92, 123516, eq. 16; 0.02 % against CAMB near Planck):

        r_d = 55.154 exp[-72.3 (omega_nu + 0.0006)^2] / (omega_cb^0.25351 omega_b^0.12807) Mpc.

    Planck 2015 (IllustrisTNG: Omega_b = 0.0486, Omega_m = 0.3089, h = 0.6774): r_d = 147.3 Mpc = 99.8 h^-1 Mpc. Works
    on arrays and under jit."""
    wb, wcb, wnu = jnp.asarray(omega_b), jnp.asarray(omega_cb), jnp.asarray(omega_nu)
    return 55.154 * jnp.exp(-72.3 * (wnu + 0.0006) ** 2) / (wcb**0.25351 * wb**0.12807)


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

    def solve_a_of_x_host(self, ICs, x_eval_Mpc, rtol: float = 1e-12, atol: float = 1e-14) -> np.ndarray:
        """a(x) of Eq. dadx_general in float64 on the host (SciPy's DOP853, 8th order, rtol 1e-12), whatever the precision
        policy: the scale factors of a ``SkewerBundle`` with concrete inputs. The Hubble velocities of the line
        centres, c ln(a_dagger / a), are differences of nearly equal values times c, so a needs ~1e-12 accuracy there
        (in float32 it would err by ~0.02 km/s).

        ICs : (N_cosmologies,) a(x_eval[0]); x_eval_Mpc : (N_eval,) comoving distances [Mpc], monotonic (the first one
        is the initial condition). Returns (N_cosmologies, N_eval) float64 NumPy. Not traceable (``solve_a_of_x`` is)."""
        from scipy.integrate import solve_ivp

        x = np.asarray(x_eval_Mpc, dtype=np.float64) * 1e-3  # Mpc -> Gpc
        a0 = np.atleast_1d(np.asarray(ICs, dtype=np.float64))
        pars = np.asarray(self.params, dtype=np.float64)
        if a0.shape[0] != pars.shape[0]:
            raise ValueError(f"ICs shape {a0.shape} incompatible with number of cosmologies {pars.shape[0]}.")
        out = np.empty((pars.shape[0], x.size))
        for i, (H0, om_r, om_m, om_k, om_l) in enumerate(pars):
            kk = -om_k / phys.cc_Gpc_per_Gyr**2

            def rhs(xx, aa, H0=H0, om_r=om_r, om_m=om_m, om_k=om_k, om_l=om_l, kk=kk):
                num = om_r + om_m * aa + om_k * aa**2 + om_l * aa**4
                return (H0 / phys.cc_Gpc_per_Gyr) * np.sqrt(num / max(1e-32, 1.0 - kk * xx**2))

            if x.size == 1 or x[-1] == x[0]:
                out[i] = a0[i]
                continue
            # SciPy wants strictly monotonic evaluation points: solve on the distinct ones (ties: the same a)
            xu, inv = np.unique(x, return_inverse=True)
            if x[-1] < x[0]:
                xu, inv = xu[::-1], xu.size - 1 - inv
            sol = solve_ivp(rhs, (xu[0], xu[-1]), [a0[i]], method="DOP853", t_eval=xu, rtol=rtol, atol=atol)
            if not sol.success:
                raise RuntimeError(f"solve_a_of_x_host: {sol.message}")
            out[i] = sol.y[0][np.ravel(inv)]
        return out
