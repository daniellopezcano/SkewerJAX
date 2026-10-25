"""Cell selection: the cells within reach of a set of paths (periodic k-d tree), for any ``Cells``-like container.

The paths are the sampling points of a ``SkewerBundle`` (any shape: straight, curved, crossing the box faces;
positions in comoving Mpc, unwrapped along the path). A periodic k-d tree is built on the *path points*
(``scipy.spatial.cKDTree(boxsize=box)``, minimum-image distances) and every cell is tested against it.

Selection radius of a cell c with cube side h_c = V_c^(1/3), for a path sampled with spacing <= ds (the nearest
*point* is at most ds/2 farther than the nearest point of the polyline):

* ``"tophat"``: d <= h_c + ds/2 (a sphere of radius h_c can touch the path);
* ``"voronoi"``: d <= max(k h_c, r_min) + ds/2. A point of the path can belong to the Voronoi cell of generator c
  only if it is within the cell's circumradius, which ``k = 2.5`` bounds for all but pathological cells; the
  deposition checks this a posteriori.

Back ends (``backend``): ``"cpu"`` (default) queries a scipy ``cKDTree``; ``"jztree"`` computes the same
nearest-path-point distance on a CUDA GPU with JZ-TREE (``jztree.knn.knn`` with ``boxsize``, periodic;
https://jstuecker.github.io/jztree/api.html), re-checking in float64 on the host the cells whose distance is
within a relative ``1e-6`` of their radius, so that the selected set does not depend on the back end;
``"auto"`` takes ``"jztree"`` when jztree imports and JAX sees a CUDA device, otherwise ``"cpu"``. jztree is
optional: CPU-only installations never import it.

``select_cells`` works on cells in memory (``core.Cells``, or any container with ``positions``, ``size_cMpc``,
``box`` and ``take``); ``skewerjax.simulations.select_cells_near_paths`` applies the same rule to a snapshot streamed
from disk.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def wrap_positions(x, box: float) -> np.ndarray:
    """Wrap positions into [0, box) (guarding against x mod box == box in floating point)."""
    y = np.mod(np.asarray(x, dtype=np.float64), box)
    return np.where(y >= box, 0.0, y)


def max_path_spacing(paths) -> float:
    """Largest distance between consecutive points of the (unwrapped) paths [cMpc]."""
    p = np.asarray(paths, dtype=np.float64)
    if p.ndim != 3 or p.shape[-1] != 3 or p.shape[1] < 2:
        raise ValueError(f"paths must be (N_sk, N_pts >= 2, 3), got {p.shape}.")
    return float(np.max(np.linalg.norm(np.diff(p, axis=1), axis=-1)))


def selection_radius(size, mode: str, ds: float, voronoi_factor: float = 2.5, voronoi_min: float = 0.0):
    """Per-cell selection radius [cMpc] (see module docstring)."""
    size = np.asarray(size, dtype=np.float64)
    if mode == "tophat":
        return size + 0.5 * ds
    if mode == "voronoi":
        return np.maximum(voronoi_factor * size, voronoi_min) + 0.5 * ds
    raise ValueError(f"mode must be 'tophat' or 'voronoi', got {mode!r}.")


def jztree_available() -> bool:
    """True if jztree imports and JAX sees a CUDA device (the conditions of ``backend="jztree"``)."""
    try:
        import jztree  # noqa: F401
        import jax
        return any(d.platform == "gpu" for d in jax.devices())
    except Exception:                                   # ImportError, or no GPU platform initialised
        return False


def resolve_backend(backend: str) -> str:
    """"cpu", "jztree" or "auto" -> "cpu" / "jztree"; "jztree" without jztree or a CUDA device raises."""
    if backend not in ("cpu", "jztree", "auto"):
        raise ValueError(f"backend must be 'cpu', 'jztree' or 'auto', got {backend!r}.")
    if backend == "auto":
        return "jztree" if jztree_available() else "cpu"
    if backend == "jztree" and not jztree_available():
        raise RuntimeError("backend='jztree' needs the optional package jztree (CUDA wheels, `pip install \"jztree[cuda12]\"`) "
                           "and a CUDA device visible to JAX; use backend='cpu' or 'auto'.")
    return backend


def _nearest_distance_cpu(tree: cKDTree, pos: np.ndarray, rad: np.ndarray, workers: int) -> np.ndarray:
    """Distance of every cell to its nearest path point, inf beyond the cell's radius (periodic cKDTree)."""
    d = np.full(rad.size, np.inf)
    edges = np.unique(np.geomspace(rad.min(), rad.max(), 33))          # radius bins: a tight bound per bin
    which = np.clip(np.searchsorted(edges, rad, side="left"), 0, edges.size - 1)
    for b in np.unique(which):
        m = which == b
        d[m] = tree.query(pos[m], k=1, distance_upper_bound=float(rad[m].max()), workers=workers)[0]
    return d


def _nearest_distance_jztree(pts: np.ndarray, pos: np.ndarray, box: float, tree: cKDTree, rad: np.ndarray,
                             margin: float = 1e-6) -> np.ndarray:
    """The same distance from JZ-TREE on the GPU (k = 1 nearest path point, periodic ``boxsize``).

    jztree's working precision is not documented, so the cells within ``margin`` (relative) of their radius are
    re-measured in float64 with the host tree: the decision d <= r is then the CPU back end's for every cell.
    """
    import jax.numpy as jnp
    import jztree
    rnn, _ = jztree.knn.knn(jnp.asarray(pts), 1, boxsize=box, part_query=jnp.asarray(pos))
    d = np.asarray(rnn, dtype=np.float64).reshape(-1)
    border = np.abs(d - rad) <= margin * np.maximum(rad, 1.0)
    if border.any():
        d[border] = tree.query(pos[border], k=1)[0]
    return d


def select_cells(cells, paths, *, mode: str = "voronoi", voronoi_factor: float = 2.5, voronoi_min: float = 0.0,
                 workers: int = -1, backend: str = "cpu"):
    """The cells of ``cells`` that can reach any of ``paths`` (selection radius of the module docstring).

    cells : ``core.Cells`` (or any container with ``positions``, ``size_cMpc``, ``box`` and ``take``).
    paths : (N_sk, N_pts, 3) sampling points [cMpc], e.g. ``SkewerBundle.spatial_points``.
    mode : "voronoi" (default deposition) or "tophat" (spheres of radius V^(1/3)).
    backend : "cpu" (default), "jztree" or "auto" (``resolve_backend``); every back end selects the same cells.
    Returns ``cells.take(selected)``.
    """
    box = cells.box
    ds = max_path_spacing(paths)
    pts = np.asarray(paths, dtype=np.float64).reshape(-1, 3)
    pts = pts if box is None else wrap_positions(pts, box)
    tree = cKDTree(pts) if box is None else cKDTree(pts, boxsize=box)
    pos = np.asarray(cells.positions, dtype=np.float64)
    pos = pos if box is None else wrap_positions(pos, box)
    rad = selection_radius(cells.size_cMpc, mode, ds, voronoi_factor, voronoi_min)
    backend = resolve_backend(backend)
    d = _nearest_distance_jztree(pts, pos, box, tree, rad) if backend == "jztree" else _nearest_distance_cpu(tree, pos, rad, workers)
    sel = np.nonzero(d <= rad)[0]
    if sel.size == 0:
        raise RuntimeError("no gas cell is within reach of the paths.")
    return cells.take(sel)
