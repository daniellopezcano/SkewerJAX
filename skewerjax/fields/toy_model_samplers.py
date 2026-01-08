import jax.numpy as jnp

def temperature_sampler(T0_min: float = 1e0, T0_max: float = 1e6, L_box: float = 1.0):
    def _sample(x: jnp.ndarray) -> jnp.ndarray:
        # Capture batch dimensions (e.g., (7,5) from (7,5,4))
        *batch_dims, _ = x.shape
        # Flatten all but last dimension: (..., 4) -> (N_points, 4)
        x_flat = x.reshape(-1, x.shape[-1])
        # Compute on flat array
        z = x_flat[:, -1]  # z-position
        log_T0_min = jnp.log10(T0_min)
        log_T0_max = jnp.log10(T0_max)
        T_flat = 10**(log_T0_min + (log_T0_max - log_T0_min) * (z / L_box))
        # Restore original batch shape
        return T_flat.reshape(batch_dims)
    return _sample


def velocity_sampler(v0_min: float = -0.001, v0_max: float = 0.001, L_box: float = 1.0):
    def _sample(x: jnp.ndarray) -> jnp.ndarray:
        *batch_dims, _ = x.shape
        x_flat = x.reshape(-1, x.shape[-1])
        # Extract x-position and compute vx
        x_pos = x_flat[:, 1]
        vx_flat = v0_min + (v0_max - v0_min) * x_pos / L_box
        # Build 3D vector: (N_points, 3)
        v_flat = jnp.stack([vx_flat, jnp.zeros_like(vx_flat), jnp.zeros_like(vx_flat)], axis=-1)
        # Restore batch shape + vector dimension: (..., 3)
        return v_flat.reshape(*batch_dims, 3)
    return _sample


def turbulence_sampler(b_turb: float = 0.0):
    def _sample(x: jnp.ndarray) -> jnp.ndarray:
        *batch_dims, _ = x.shape
        x_flat = x.reshape(-1, x.shape[-1])
        result_flat = jnp.full(x_flat.shape[0], b_turb)
        return result_flat.reshape(batch_dims)
    return _sample


def collisional_sampler(A_coll: float = 0.0):
    def _sample(x: jnp.ndarray) -> jnp.ndarray:
        *batch_dims, _ = x.shape
        x_flat = x.reshape(-1, x.shape[-1])
        result_flat = jnp.full(x_flat.shape[0], A_coll)
        return result_flat.reshape(batch_dims)
    return _sample


def number_density_sampler(n0_min: float = 1e0, n0_max: float = 1e6, L_box: float = 1.0):
    def _sample(x: jnp.ndarray) -> jnp.ndarray:
        *batch_dims, _ = x.shape
        x_flat = x.reshape(-1, x.shape[-1])
        y = x_flat[:, 2]  # y-position
        log_n0_min = jnp.log10(n0_min)
        log_n0_max = jnp.log10(n0_max)
        n_flat = 10**(log_n0_min + (log_n0_max - log_n0_min) * (y / L_box))
        return n_flat.reshape(batch_dims)
    return _sample
