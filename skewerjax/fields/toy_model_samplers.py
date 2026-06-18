from __future__ import annotations

from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Core: resolution-consistent GRF
# ---------------------------------------------------------------------------
def _band_limited_white_noise(u_ref: jnp.ndarray, n_grid: int) -> jnp.ndarray:
    """Keep the low-|k| central block of a reference Fourier-space field.

    ``u_ref`` is a cubic ``(n_ref, n_ref, n_ref)`` complex array in standard FFT
    order.  Returns the ``(n_grid, n_grid, n_grid)`` block of modes with integer
    frequencies in ``[-n_grid/2, n_grid/2)`` -- i.e. an ideal low-pass filter.
    """
    n_ref = u_ref.shape[0]
    if n_grid == n_ref:
        return u_ref
    shifted = jnp.fft.fftshift(u_ref)           # DC -> centre
    c = n_ref // 2
    lo = c - n_grid // 2
    block = shifted[lo:lo + n_grid, lo:lo + n_grid, lo:lo + n_grid]
    return jnp.fft.ifftshift(block)             # DC -> corner (FFT order)


def _isotropic_pk_on_grid(pk_of_k, n_grid: int, L_box: float):
    """Evaluate the (clipped, DC-removed) power spectrum on a size-n_grid grid."""
    k1d = 2.0 * jnp.pi * jnp.fft.fftfreq(n_grid, d=L_box / n_grid)
    KX, KY, KZ = jnp.meshgrid(k1d, k1d, k1d, indexing="ij")
    K = jnp.sqrt(KX ** 2 + KY ** 2 + KZ ** 2)
    Pk = jnp.clip(pk_of_k(K), 0.0, None)
    Pk = jnp.where(K == 0.0, 0.0, Pk)           # no DC / mean mode
    return K, Pk


def generate_gaussian_random_field(
    key: jax.Array,
    n_grid: int,
    L_box: float,
    pk_of_k: Callable[[jnp.ndarray], jnp.ndarray],
    *,
    n_ref: Optional[int] = None,
    normalize: str = "analytic",
) -> jnp.ndarray:
    """Generate a real 3D Gaussian random field delta(x) with isotropic P(k).

    Parameters
    ----------
    key : jax PRNG key
        Fix this (with ``n_ref``) to get fields that converge as ``n_grid`` grows.
    n_grid : int (even)
        Resolution of the returned field.  Must be even and ``<= n_ref``.
    L_box : float
        Box side length (sets the physical meaning of k = 2*pi*freq).
    pk_of_k : callable
        ``P(|k|)`` evaluated on angular wavenumbers (2*pi/L units).
    n_ref : int (even), optional
        Master resolution of the reference white-noise field.  Defaults to
        ``n_grid``.  Set it once to the FINEST resolution you intend to study and
        keep it fixed; all coarser ``n_grid`` then refine into the finer ones.
    normalize : {"analytic", "none", "realized"}
        "analytic"  -> divide by the resolution-INDEPENDENT expected std computed
                       from the full reference k-grid (recommended: delta has
                       ~unit variance, controlled, and converges with n_grid).
        "none"      -> physical normalization only (variance grows with n_grid as
                       small-scale power is added; large scales still converge).
        "realized"  -> the old (NOT recommended) divide-by-sample-std behaviour;
                       this rescaling is resolution- and realization-dependent.

    Returns
    -------
    delta : (n_grid, n_grid, n_grid) real array
    """
    if n_ref is None:
        n_ref = int(n_grid)
    n_grid = int(n_grid)
    n_ref = int(n_ref)
    if n_grid > n_ref:
        raise ValueError(f"n_grid ({n_grid}) must be <= n_ref ({n_ref}).")
    if (n_grid % 2) or (n_ref % 2):
        raise ValueError("n_grid and n_ref must both be even.")

    # 1) ONE reference white-noise field, fixed by (key, n_ref).
    #    FFT of real white noise -> Hermitian complex spectrum automatically.
    g = jax.random.normal(key, (n_ref, n_ref, n_ref))
    u_ref = jnp.fft.fftn(g) / (n_ref ** 1.5)            # unit-variance per mode

    # 2) Band-limit to the requested resolution (shared low-k modes identical).
    u = _band_limited_white_noise(u_ref, n_grid)

    # 3) Colour by sqrt(P(k)):  delta(x) = sum_k u_k sqrt(P(k)) e^{i k.x}
    _, Pk = _isotropic_pk_on_grid(pk_of_k, n_grid, L_box)
    delta = jnp.fft.ifftn(u * jnp.sqrt(Pk)).real * (n_grid ** 3)

    # 4) Controlled normalization.
    if normalize == "analytic":
        _, Pk_ref = _isotropic_pk_on_grid(pk_of_k, n_ref, L_box)
        sigma_ref = jnp.sqrt(jnp.sum(Pk_ref))          # resolution-independent
        delta = delta / (sigma_ref + 1e-30)
    elif normalize == "realized":
        delta = (delta - jnp.mean(delta)) / (jnp.std(delta) + 1e-12)
    elif normalize != "none":
        raise ValueError("normalize must be 'analytic', 'none', or 'realized'.")

    return delta


# ---------------------------------------------------------------------------
# Trilinear periodic sampling (kept; correct as-is)
# ---------------------------------------------------------------------------
def _trilinear_periodic_sample(field: jnp.ndarray, pos_xyz: jnp.ndarray,
                               L_box: float) -> jnp.ndarray:
    """Periodic trilinear interpolation of ``field`` at world positions.

    field   : (Nx, Ny, Nz) real
    pos_xyz : (M, 3) positions, periodic in [0, L_box)
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


# ---------------------------------------------------------------------------
# Number-density sampler built on the robust GRF
# ---------------------------------------------------------------------------
def number_density_sampler_gaussian_3d(
    key: jax.Array,
    pk_of_k: Callable[[jnp.ndarray], jnp.ndarray],
    *,
    n0_min: float = 1e0,
    n0_max: float = 1e6,
    L_box: float = 1.0,
    n_grid: int = 64,
    n_ref: Optional[int] = None,
    contrast: float = 1.0,
    normalize: str = "analytic",
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Build a callable n(x,y,z) from a resolution-consistent 3D GRF.

    Mapping (unchanged from your version):
        s = tanh(delta / contrast)            in [-1, 1]
        log10 n = log_mid + log_amp * s       => n in [n0_min, n0_max]

    Because ``delta`` now uses the ``"analytic"`` normalization, its variance is
    controlled and resolution-independent, so the tanh saturation -- and hence the
    density statistics -- behave consistently as you vary ``n_grid``.

    Convergence contract: fix ``key`` and ``n_ref`` (the finest resolution you
    will test) and vary ``n_grid <= n_ref``.

    Parameters mirror your original signature; ``grid_shape`` is replaced by the
    scalar ``n_grid`` (cubic) plus the master resolution ``n_ref``.
    """
    if n_ref is None:
        n_ref = int(n_grid)

    delta_grid = generate_gaussian_random_field(
        key, n_grid, L_box, pk_of_k, n_ref=n_ref, normalize=normalize)

    log_min = jnp.log10(n0_min)
    log_max = jnp.log10(n0_max)
    log_mid = 0.5 * (log_min + log_max)
    log_amp = 0.5 * (log_max - log_min)

    def _sample(x: jnp.ndarray) -> jnp.ndarray:
        *batch_dims, _ = x.shape
        x_flat = x.reshape(-1, x.shape[-1])
        pos = jnp.mod(x_flat[:, 1:4], L_box)               # (M, 3) spatial cols
        delta = _trilinear_periodic_sample(delta_grid, pos, L_box)
        s = jnp.tanh(delta / contrast)
        n_flat = 10.0 ** (log_mid + log_amp * s)
        return n_flat.reshape(batch_dims)

    # expose the underlying field for diagnostics / convergence checks
    _sample.delta_grid = delta_grid
    return _sample


# =====================================================================
# ====================== simple field generators ======================
# ===================================================================== 


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

# =====================================================================
# ============== top hat absorber samplers (periodic) ================
# ===================================================================== 

def _mix_uint32(x: jnp.ndarray) -> jnp.ndarray:
    """Small deterministic uint32 mixer for stable coordinate hashing."""
    x = jnp.asarray(x, dtype=jnp.uint32)
    x = x ^ (x >> jnp.uint32(16))
    x = x * jnp.uint32(0x7FEB352D)
    x = x ^ (x >> jnp.uint32(15))
    x = x * jnp.uint32(0x846CA68B)
    x = x ^ (x >> jnp.uint32(16))
    return x


def _hash_xz_to_seed(
    ix: jnp.ndarray,
    iz: jnp.ndarray,
    base_seed: int,
) -> jnp.ndarray:
    """Combine integerized transverse coordinates into one uint32 seed."""
    h = jnp.asarray(base_seed, dtype=jnp.uint32)
    h = h ^ _mix_uint32(ix + jnp.uint32(0x9E3779B9))
    h = h ^ _mix_uint32(iz + jnp.uint32(0x85EBCA6B))
    return _mix_uint32(h)


def periodic_top_hat_absorber_sampler(
    N_absorvers: int = 1,
    Delta: float = 0.01,
    C0: float = 1.0,
    C_low: float = 0.0,
    L_box: float = 1.0,
    xz_seed_shape: Tuple[int, int] = (4096, 4096),
    base_seed: int = 0,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """
    Toy scalar sampler with independent random y-absorber realizations per (x,z).

    For each transverse seed cell (x,z), draw N_absorvers absorber centers y_i
    uniformly in [0, L_box). The returned field is C0 whenever y lies within
    periodic distance Delta/2 of at least one center, and C_low otherwise.

    Coordinate convention:
        x[..., 1] -> x-position
        x[..., 2] -> y-position
        x[..., 3] -> z-position
    """
    if N_absorvers < 0:
        raise ValueError("N_absorvers must be >= 0.")
    if Delta < 0.0:
        raise ValueError("Delta must be >= 0.")
    if L_box <= 0.0:
        raise ValueError("L_box must be > 0.")
    if len(xz_seed_shape) != 2 or xz_seed_shape[0] <= 0 or xz_seed_shape[1] <= 0:
        raise ValueError("xz_seed_shape must be a pair of positive integers.")

    Nx_seed, Nz_seed = int(xz_seed_shape[0]), int(xz_seed_shape[1])
    base_seed_uint32 = int(base_seed) & 0xFFFFFFFF

    def _centers_from_seed(seed: jnp.ndarray, dtype: jnp.dtype) -> jnp.ndarray:
        key = jax.random.PRNGKey(seed)
        return jax.random.uniform(
            key,
            shape=(N_absorvers,),
            minval=jnp.asarray(0.0, dtype=dtype),
            maxval=jnp.asarray(L_box, dtype=dtype),
            dtype=dtype,
        )

    def _sample(x: jnp.ndarray) -> jnp.ndarray:
        *batch_dims, _ = x.shape
        x_flat = x.reshape(-1, x.shape[-1])

        dtype = x_flat.dtype

        x_pos = jnp.mod(x_flat[:, 1], L_box)
        y_pos = jnp.mod(x_flat[:, 2], L_box)
        z_pos = jnp.mod(x_flat[:, 3], L_box)

        ix = jnp.floor(x_pos / L_box * Nx_seed).astype(jnp.int32)
        iz = jnp.floor(z_pos / L_box * Nz_seed).astype(jnp.int32)

        ix = jnp.clip(ix, 0, Nx_seed - 1).astype(jnp.uint32)
        iz = jnp.clip(iz, 0, Nz_seed - 1).astype(jnp.uint32)

        seeds = _hash_xz_to_seed(ix, iz, base_seed_uint32)

        centers = jax.vmap(lambda s: _centers_from_seed(s, dtype))(seeds)

        dy = jnp.abs(
            jnp.mod(y_pos[:, None] - centers + 0.5 * L_box, L_box)
            - 0.5 * L_box
        )

        inside_any_absorber = jnp.any(dy <= 0.5 * Delta, axis=-1)

        values = jnp.where(
            inside_any_absorber,
            jnp.asarray(C0, dtype=dtype),
            jnp.asarray(C_low, dtype=dtype),
        )

        return values.reshape(batch_dims)

    return _sample