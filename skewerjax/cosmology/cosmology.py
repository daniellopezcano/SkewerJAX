import jax.numpy as jnp
from jax import vmap
from dataclasses import dataclass
from jax.typing import ArrayLike
import diffrax

from skewerjax.constants.physical_constants import phys

def da_dx(xx, aa, params, cc=phys.cc_Gpc_per_Gyr):
    """
    ODE for photon geodesic: da/dx = (H0/cc) * sqrt(Ω_r + Ω_m a + Ω_k a² + Ω_Λ a⁴) / sqrt(1 - kk x²)

    Parameters
    ----------
    xx : ArrayLike
        Comoving distance [Gpc]
    aa : ArrayLike
        Scale factor
    params : ArrayLike
        Cosmological parameters
    cc : float, optional
        Speed of light by default phys.cc_Gpc_per_Gyr
    """
    # Unpack parameters
    H0, omega_r, omega_m, omega_k, omega_lambda = params
            
    # Curvature term
    kk = -omega_k / (cc**2)
    
    # Safe terms
    denominator = jnp.maximum(1e-32, 1 - kk * xx**2)
    numerator = omega_r + omega_m * aa + omega_k * aa**2 + omega_lambda * aa**4
    
    # Compute derivative
    da_dx = (H0 / cc) * jnp.sqrt(numerator / denominator)
    
    return da_dx

@dataclass(frozen=True)
class Cosmology:
    """
    Immutable cosmology container for batched FLRW parameters with JAX-differentiable ODE solving.
    
    All parameters can be scalar or array-valued. Array parameters must be broadcast-compatible.
    
    Parameters
    ----------
    H0 : ArrayLike
        Hubble constant [1/Gyr]
    Omega_r_0 : ArrayLike
        Radiation density today, Ω_r
    Omega_m_0 : ArrayLike
        Matter density today, Ω_m
    Omega_k_0 : ArrayLike
        Curvature density today, Ω_k
    Omega_lambda_0 : ArrayLike
        Dark energy density today, Ω_Λ
    
    Raises
    ------
    ValueError
        If parameters have incompatible shapes or densities don't sum to 1 within tolerance.
    """
    
    def __init__(
            self, H0, Omega_r_0, Omega_m_0, Omega_k_0, Omega_lambda_0):
        
        # Convert inputs to arrays. If the inputs are floats, convert them to 1D arrays with a single element
        H0 = jnp.atleast_1d(H0)
        Omega_r_0 = jnp.atleast_1d(Omega_r_0)
        Omega_m_0 = jnp.atleast_1d(Omega_m_0)
        Omega_k_0 = jnp.atleast_1d(Omega_k_0)
        Omega_lambda_0 = jnp.atleast_1d(Omega_lambda_0)
        
        # verify that all input parameters have equivalent lengths (1D arrays or scalars)
        try:
            jnp.broadcast_shapes(
                H0.shape,
                Omega_r_0.shape,
                Omega_m_0.shape,
                Omega_k_0.shape,
                Omega_lambda_0.shape,
            )
            
            object.__setattr__(self, 'N_cosmologies', H0.shape[0])

            object.__setattr__(self, 'H0', H0)
            object.__setattr__(self, 'Omega_r_0', Omega_r_0)
            object.__setattr__(self, 'Omega_m_0', Omega_m_0)
            object.__setattr__(self, 'Omega_k_0', Omega_k_0)
            object.__setattr__(self, 'Omega_lambda_0', Omega_lambda_0)

            object.__setattr__(self, 'params',
                jnp.stack([self.H0, self.Omega_r_0, self.Omega_m_0, self.Omega_k_0, self.Omega_lambda_0], axis=-1)
            )

        except ValueError as e:
            raise ValueError(f"Incompatible shapes for cosmological parameters: "
                             f"H0: {H0.shape}, "
                             f"Omega_r_0: {Omega_r_0.shape}, "
                             f"Omega_m_0: {Omega_m_0.shape}, "
                             f"Omega_k_0: {Omega_k_0.shape}, "
                             f"Omega_lambda_0: {Omega_lambda_0.shape}") from e
        
        # Check closure condition
        density_sum = Omega_r_0 + Omega_m_0 + Omega_k_0 + Omega_lambda_0
        if not jnp.allclose(density_sum, 1.0, atol=1e-4):
            raise ValueError(f"Cosmological densities do not sum to 1: {density_sum}")
        
    def solve_a_of_x(
        self,
        ICs: ArrayLike,
        x_eval_Mpc: ArrayLike,
        solver: diffrax.AbstractSolver = diffrax.Tsit5(),
        rtol: float = 1e-4,
        atol: float = 1e-4,
        max_steps: int = 1000
        ) -> jnp.ndarray:
        
        """
        Solve for the scale factor a(x) over specified comoving distances employing diffrax.diffeqsolve for ODE integration.
        diffrax.diffeqsolve handles systems of ODEs. In this case we have a single ODE for the scale factor.
        
        Parameters
        ----------
        ICs : ArrayLike
            Initial conditions for the scale factor, a(x0). shape (N_cosmologies,).
        x_eval : ArrayLike
            Comoving distances to evaluate at. Units [Mpc]. shape (N_eval,).
        solver : diffrax.AbstractSolver, optional
            Integration solver, by default diffrax.Tsit5()
        rtol : float, optional
            Relative tolerance, by default 1e-4
        atol : float, optional
            Absolute tolerance, by default 1e-4
        max_steps : int, optional
            Maximum number of integration steps, by default 1000
        """

        def _solve_single_cosmology(H0, Omega_r_0, Omega_m_0, Omega_k_0, Omega_lambda_0, ICs, x_eval_Gpc, solver=solver, rtol=rtol, atol=atol, max_steps=max_steps):
            """
            Solve for the scale factor a(x) over specified comoving distances employing diffrax.diffeqsolve for ODE integration.
            """

            params = jnp.array([H0, Omega_r_0, Omega_m_0, Omega_k_0, Omega_lambda_0])

            solution_obj = diffrax.diffeqsolve(
                diffrax.ODETerm(da_dx),
                solver=solver,
                t0=x_eval_Gpc[0], t1=x_eval_Gpc[-1], dt0=None,
                y0=ICs,
                args=params,
                saveat=diffrax.SaveAt(ts=x_eval_Gpc),
                stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
                max_steps=max_steps,
            )
            return solution_obj.ys

        # assert that ICs has shape (N_cosmologies,)
        ICs = jnp.atleast_1d(ICs)
        if ICs.shape[0] != self.N_cosmologies:
            raise ValueError(f"ICs shape {ICs.shape} incompatible with number of cosmologies {self.N_cosmologies}.")
        # convert ICs to shape (N_cosmologies, 1) because diffrax expects a single initial condition value
        # for each of the variables in the system of ODEs, however, in this case we have a single ODE.
        ICs = jnp.atleast_1d(ICs)[:, None]  

        # shape (N_cosmologies, N_eval, N_ODEs)
        _solution_obj_ys = vmap(_solve_single_cosmology, in_axes=(0, 0, 0, 0, 0, 0, None))(
            self.H0, self.Omega_r_0, self.Omega_m_0, self.Omega_k_0, self.Omega_lambda_0, ICs, x_eval_Mpc*1e-3 # convert Mpc to Gpc
        )

        # Extract only the entry [..., 0] corresponding to the scale factor as a function of x
        # solution_obj.ys has shape (N_cosmologies, N_eval, N_ODEs), in this case N_ODEs = 1.
        # shape (N_cosmologies, N_eval)
        a_of_x = _solution_obj_ys[..., 0]

        return a_of_x
        