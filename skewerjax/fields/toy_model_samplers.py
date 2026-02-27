import jax
import jax.numpy as jnp
from typing import Callable, Optional, Tuple

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

def _trilinear_periodic_sample(field: jnp.ndarray, pos_xyz: jnp.ndarray, L_box: float) -> jnp.ndarray:
    """
    field: (Nx, Ny, Nz) real
    pos_xyz: (M, 3) positions in same units as L_box, assumed periodic in [0, L_box)
    returns: (M,) sampled values
    """
    Nx, Ny, Nz = field.shape
    N = jnp.array([Nx, Ny, Nz], dtype=jnp.float32)

    # convert to grid coordinates in [0, N)
    u = (pos_xyz / L_box) * N
    i0 = jnp.floor(u).astype(jnp.int32)
    f = u - i0.astype(jnp.float32)

    # periodic wrap
    ix0 = jnp.mod(i0[:, 0], Nx); ix1 = jnp.mod(ix0 + 1, Nx)
    iy0 = jnp.mod(i0[:, 1], Ny); iy1 = jnp.mod(iy0 + 1, Ny)
    iz0 = jnp.mod(i0[:, 2], Nz); iz1 = jnp.mod(iz0 + 1, Nz)

    fx, fy, fz = f[:, 0], f[:, 1], f[:, 2]

    # gather the 8 corners
    c000 = field[ix0, iy0, iz0]
    c100 = field[ix1, iy0, iz0]
    c010 = field[ix0, iy1, iz0]
    c110 = field[ix1, iy1, iz0]
    c001 = field[ix0, iy0, iz1]
    c101 = field[ix1, iy0, iz1]
    c011 = field[ix0, iy1, iz1]
    c111 = field[ix1, iy1, iz1]

    # interpolate
    c00 = c000 * (1 - fx) + c100 * fx
    c10 = c010 * (1 - fx) + c110 * fx
    c01 = c001 * (1 - fx) + c101 * fx
    c11 = c011 * (1 - fx) + c111 * fx

    c0 = c00 * (1 - fy) + c10 * fy
    c1 = c01 * (1 - fy) + c11 * fy

    return c0 * (1 - fz) + c1 * fz


def _gaussian_field_from_pk(
    key: jax.random.PRNGKey,
    shape: Tuple[int, int, int],
    L_box: float,
    pk_of_k: Callable[[jnp.ndarray], jnp.ndarray],
) -> jnp.ndarray:
    """
    Build a real 3D Gaussian random field delta(x) with target isotropic P(k)
    using Fourier synthesis with rfftn/irfftn.
    """
    Nx, Ny, Nz = shape

    # k-grids (cycles per unit) -> angular k = 2π * freq
    kx = 2.0 * jnp.pi * jnp.fft.fftfreq(Nx, d=L_box / Nx)
    ky = 2.0 * jnp.pi * jnp.fft.fftfreq(Ny, d=L_box / Ny)
    kz = 2.0 * jnp.pi * jnp.fft.rfftfreq(Nz, d=L_box / Nz)

    KX, KY, KZ = jnp.meshgrid(kx, ky, kz, indexing="ij")
    K = jnp.sqrt(KX**2 + KY**2 + KZ**2)

    Pk = pk_of_k(K)
    Pk = jnp.where(K == 0.0, 0.0, Pk)  # remove DC mode

    # Draw complex Fourier modes with variance ~ P(k)
    # For rfft half-spectrum, a standard choice is:
    #   delta_k = sqrt(Pk/2) * (a + i b) with a,b ~ N(0,1)
    key_a, key_b = jax.random.split(key)
    a = jax.random.normal(key_a, (Nx, Ny, Nz // 2 + 1))
    b = jax.random.normal(key_b, (Nx, Ny, Nz // 2 + 1))

    amp = jnp.sqrt(jnp.maximum(Pk, 0.0) / 2.0)
    delta_k = amp * (a + 1j * b)

    # Inverse transform to real space
    delta_x = jnp.fft.irfftn(delta_k, s=(Nx, Ny, Nz)).real

    # Normalize to zero mean, unit std (handy for controlled mapping later)
    delta_x = delta_x - jnp.mean(delta_x)
    delta_x = delta_x / (jnp.std(delta_x) + 1e-12)
    return delta_x


def number_density_sampler_gaussian_3d(
    key: jax.random.PRNGKey,
    pk_of_k: Callable[[jnp.ndarray], jnp.ndarray],
    n0_min: float = 1e0,
    n0_max: float = 1e6,
    L_box: float = 1.0,
    grid_shape: Tuple[int, int, int] = (64, 64, 64),
    contrast: float = 1.0,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """
    Creates a sampler n(x,y,z) based on a 3D Gaussian random field with power spectrum P(k).

    Mapping to density:
      s = tanh(delta/contrast)  in [-1,1]
      log10 n = log_mid + log_amp * s   => n in [n0_min, n0_max]
    """
    delta_grid = _gaussian_field_from_pk(key, grid_shape, L_box, pk_of_k)

    log_min = jnp.log10(n0_min)
    log_max = jnp.log10(n0_max)
    log_mid = 0.5 * (log_min + log_max)
    log_amp = 0.5 * (log_max - log_min)

    def _sample(x: jnp.ndarray) -> jnp.ndarray:
        *batch_dims, _ = x.shape
        x_flat = x.reshape(-1, x.shape[-1])

        # spatial positions (x,y,z), periodic in [0,L_box)
        pos = x_flat[:, 1:4]
        pos = jnp.mod(pos, L_box)

        delta = _trilinear_periodic_sample(delta_grid, pos, L_box)  # (M,)

        s = jnp.tanh(delta / contrast)  # bounded [-1,1]
        logn = log_mid + log_amp * s
        n_flat = 10.0 ** logn

        return n_flat.reshape(batch_dims)

    return _sample