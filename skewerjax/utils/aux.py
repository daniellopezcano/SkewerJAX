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
