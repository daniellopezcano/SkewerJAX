import jax.numpy as jnp

def make_plane_points(L_box, fixed_axis, fixed_value, scale_factor, n_var1, n_var2):
    """Points on plane fixed_axis='x'/'y'/'z'. Shape: (n_var1, n_var2, 4)."""
    # Map: fixed → (fixed_idx, var1_idx, var2_idx)
    f, v1, v2 = {'x': (0,1,2), 'y': (1,2,0), 'z': (2,0,1)}[fixed_axis]
    
    c1 = jnp.linspace(0.0, L_box, n_var1)
    c2 = jnp.linspace(0.0, L_box, n_var2)
    g1, g2 = jnp.meshgrid(c1, c2, indexing="ij")
    
    # Initialize coordinates with dummy arrays, then overwrite
    coords = [jnp.zeros_like(g1) for _ in range(3)]
    coords[f] = jnp.full_like(g1, fixed_value)
    coords[v1] = g1
    coords[v2] = g2
    
    return jnp.stack([jnp.full_like(g1, scale_factor)] + coords, axis=-1)


def make_LoS(batch_shape, direction):
    """Normalized LoS directions: batch_shape + (3,)."""
    base = jnp.asarray(direction, dtype=float)
    loS = jnp.broadcast_to(base, batch_shape + (3,))
    return loS / jnp.linalg.norm(loS, axis=-1, keepdims=True)