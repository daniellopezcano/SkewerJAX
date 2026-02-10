import jax.numpy as jnp
from pathlib import Path

def print_tree(start_path=".", ignore=None, max_depth=None):

    if ignore is None:
        ignore = {'.git', '__pycache__', '.ipynb_checkpoints', 'build', 'dist', '*.egg-info'}
    
    start = Path(start_path).resolve()
    print(f"{start}/")
    
    def _walk(path, prefix="", depth=0):
        if max_depth is not None and depth >= max_depth:
            return
        
        try:
            contents = sorted(path.iterdir())
        except PermissionError:
            print(f"{prefix}└── [Permission Denied]")
            return
        
        files = [p for p in contents if p.is_file()]
        dirs = [p for p in contents if p.is_dir() and p.name not in ignore]
        
        for i, directory in enumerate(dirs):
            is_last_dir = (i == len(dirs) - 1) and (len(files) == 0)
            connector = "└── " if is_last_dir else "├── "
            print(f"{prefix}{connector}{directory.name}/")
            
            new_prefix = prefix + ("    " if is_last_dir else "│   ")
            _walk(directory, new_prefix, depth + 1)
        
        for i, file in enumerate(files):
            is_last = (i == len(files) - 1)
            connector = "└── " if is_last else "├── "
            print(f"{prefix}{connector}{file.name}")
    
    _walk(start)

def make_plane_points(L_box, fixed_axis, fixed_value, n_var1, n_var2):
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
    
    return jnp.stack(coords, axis=-1)


def make_LoS(batch_shape, direction):
    """Normalized LoS directions: batch_shape + (3,)."""
    base = jnp.asarray(direction, dtype=float)
    loS = jnp.broadcast_to(base, batch_shape + (3,))
    return loS / jnp.linalg.norm(loS, axis=-1, keepdims=True)


def make_skewer_grid_points(L_box, N_grid, N_ppsk, direction="y"):
    """
    Build a regular grid of skewers through a periodic box.

    Returns an array of spatial comoving positions (x, y, z) with shape:
        (N_grid, N_grid, N_ppsk, 3)

    Interpretation:
      - There are N_grid**2 skewers.
      - For direction="y", each skewer starts at y=0 and goes to y=L_box,
        at fixed (x,z) set by a N_grid x N_grid transverse grid.
      - Analogously for direction="x" (vary x) or direction="z" (vary z).

    Parameters
    ----------
    L_box : float
        Box size (same units you want in the output positions).
    N_grid : int
        Number of skewers per transverse dimension (total skewers = N_grid**2).
    N_ppsk : int
        Number of points per skewer (samples along the LOS direction).
    direction : {"x","y","z"}
        Skewer direction / LOS axis.

    Notes
    -----
    Uses linspace(0, L_box, ...) including endpoints. If you prefer to avoid
    duplicating the periodic boundary (0 and L_box), change to endpoint=False.
    """
    dir_idx = {"x": 0, "y": 1, "z": 2}[direction]
    # transverse axes for the 2D grid
    t1_idx, t2_idx = {0: (1, 2), 1: (0, 2), 2: (0, 1)}[dir_idx]

    # Transverse grid (N_grid x N_grid)
    c1 = jnp.linspace(0.0, L_box, N_grid)
    c2 = jnp.linspace(0.0, L_box, N_grid)
    g1, g2 = jnp.meshgrid(c1, c2, indexing="ij")  # (N_grid, N_grid)

    # Coordinate along the skewer direction (N_ppsk)
    s = jnp.linspace(0.0, L_box, N_ppsk)  # (N_ppsk,)

    # Broadcast to (N_grid, N_grid, N_ppsk)
    g1b = g1[..., None]
    g2b = g2[..., None]
    sb  = s[None, None, :]

    # Fill xyz
    xyz = jnp.zeros((N_grid, N_grid, N_ppsk, 3), dtype=jnp.float32)
    xyz = xyz.at[..., dir_idx].set(sb)
    xyz = xyz.at[..., t1_idx].set(g1b)
    xyz = xyz.at[..., t2_idx].set(g2b)

    return xyz
