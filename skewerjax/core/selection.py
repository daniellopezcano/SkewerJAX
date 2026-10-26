"""Cell selection: the cells within reach of a set of paths (periodic k-d tree), for any ``Cells``-like container.

The paths are the sampling points of a ``SkewerBundle`` (any shape: straight, curved, crossing the box faces;
positions in comoving Mpc, unwrapped along the path). A periodic k-d tree is built on the *path points*
(``scipy.spatial.cKDTree(boxsize=box)``, minimum-image distances) and every cell is tested against it.

Selection radius of a cell c with cube side h_c = V_c^(1/3), for a path sampled with spacing <= ds (the nearest
*point* is at most ds/2 farther than the nearest point of the polyline):

* ``"tophat"``: d <= h_c + ds/2 (a sphere of radius h_c can touch the path);
* ``"voronoi"``: d <= k h_c + ds/2. A point of the path can belong to the Voronoi cell of generator c only if it is
  within the cell's circumradius, which ``k = VORONOI_FACTOR = 2.5`` bounds for all but pathological cells; the
  deposition checks this a posteriori, and ``scripts/validation/check_tng_fields.py`` checks the selection on TNG50-4
  against a brute-force nearest-generator search over the whole snapshot.

Back ends (``backend``): ``"cpu"`` (default) queries a scipy ``cKDTree``; ``"jztree"`` computes the same
nearest-path-point distance on a CUDA GPU with JZ-TREE (``jztree.knn.knn`` with ``boxsize``, periodic;
https://jstuecker.github.io/jztree/api.html), re-checking in float64 on the host the cells whose distance is
within a relative ``1e-6`` of their radius, so that the selected set does not depend on the back end;
``"auto"`` takes ``"jztree"`` when jztree imports and JAX sees a CUDA device, otherwise ``"cpu"``. jztree is
optional: CPU-only installations never import it.

``select_cells`` works on cells in memory (``core.Cells``, or any container with ``positions``, ``size_cMpc``,
``box`` and ``take``), whatever their source; ``skewerjax.simulations.select_cells_near_paths`` applies the same rule
(``selection_radius`` and the back ends below) to a snapshot streamed from disk.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree

from skewerjax.core.validation import check_values

VORONOI_FACTOR = 2.5      # k of the "voronoi" selection radius k h_c + ds/2
logger = logging.getLogger(__name__)


def wrap_positions(x, box: Optional[float]) -> np.ndarray:
    """Wrap positions into [0, box) (guarding against x mod box == box in floating point); ``box=None``: unchanged
    (as float64)."""
    if box is None:
        return np.asarray(x, dtype=np.float64)
    y = np.mod(np.asarray(x, dtype=np.float64), box)
    return np.where(y >= box, 0.0, y)


def max_path_spacing(paths) -> float:
    """Largest distance between consecutive points of the (unwrapped) paths [cMpc]."""
    p = np.asarray(paths, dtype=np.float64)
    if p.ndim != 3 or p.shape[-1] != 3 or p.shape[1] < 2:
        raise ValueError(f"paths must be (N_sk, N_pts >= 2, 3), got {p.shape}.")
    check_values("paths [cMpc]", p)
    return float(np.max(np.linalg.norm(np.diff(p, axis=1), axis=-1)))


def selection_radius(size, mode: str, ds: float):
    """Per-cell selection radius [cMpc] of cells of size ``size`` = V^(1/3) for paths sampled every <= ``ds`` (see module
    docstring)."""
    size = np.asarray(size, dtype=np.float64)
    if mode == "tophat":
        return size + 0.5 * ds
    if mode == "voronoi":
        return VORONOI_FACTOR * size + 0.5 * ds
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


def _query_workers(n: int, workers: int) -> int:
    """Threads of one k-d tree query of ``n`` points: ``workers``, except that -1 (all cores) is capped at one thread
    per 20 000 points (starting threads costs more than small queries)."""
    if workers != -1:
        return workers
    return int(max(1, min(os.cpu_count() or 1, n // 20_000)))


def _nearest_distance_cpu(tree: cKDTree, pos: np.ndarray, rad: np.ndarray, workers: int,
                          candidates: Optional[np.ndarray] = None) -> np.ndarray:
    """Distance of every cell to its nearest path point, inf beyond the cell's radius (periodic cKDTree).

    The cells are queried in bins of radius (33 geometric edges above 10^6 cells, 5 below), each with the bin's largest
    radius as the distance bound; ``candidates`` (bool mask): only these cells are queried, the others get inf (the
    caller has shown they are beyond their radius). Neither the bins nor the mask change which cells have d <= rad.
    """
    d = np.full(rad.size, np.inf)
    if rad.size == 0:
        return d
    edges = np.unique(np.geomspace(rad.min(), rad.max(), 33 if rad.size > 1_000_000 else 5))   # a tight bound per bin
    which = np.clip(np.searchsorted(edges, rad, side="left"), 0, edges.size - 1)
    bound = np.full(edges.size, -np.inf)
    np.maximum.at(bound, which, rad)                                    # the largest radius of every bin (of all cells)
    idx = np.arange(rad.size) if candidates is None else np.flatnonzero(candidates)
    w = which[idx]
    for b in np.flatnonzero(np.bincount(w, minlength=edges.size)):
        sub = idx[w == b]
        d[sub] = tree.query(pos[sub], k=1, distance_upper_bound=float(bound[b]), workers=_query_workers(sub.size, workers))[0]
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


def path_tree(paths, box: Optional[float]):
    """The path points of ``paths`` (N_sk, N_pts, 3) [cMpc] wrapped into the box, as (points (N_sk N_pts, 3), periodic
    ``cKDTree`` on them, the largest spacing ds of the paths [cMpc]): what every selection queries."""
    ds = max_path_spacing(paths)
    pts = wrap_positions(np.asarray(paths, dtype=np.float64).reshape(-1, 3), box)
    return pts, (cKDTree(pts) if box is None else cKDTree(pts, boxsize=box)), ds


def select_cells(cells, paths, *, mode: str = "voronoi", workers: int = -1, backend: str = "cpu", return_info: bool = False):
    """The cells of ``cells`` that can reach any of ``paths`` (selection radius of the module docstring).

    cells : ``core.Cells`` (or any container with ``positions``, ``size_cMpc``, ``box`` and ``take``).
    paths : (N_sk, N_pts, 3) sampling points [cMpc], e.g. ``SkewerBundle.spatial_points``.
    mode : "voronoi" (default deposition) or "tophat" (spheres of radius V^(1/3)).
    workers : threads of the k-d tree queries (-1: all cores).
    backend : "cpu" (default), "jztree" or "auto" (``resolve_backend``); every back end selects the same cells.
    return_info : also return dict(n_cells, n_selected, indices (the selected cells in ``cells``), mode, backend,
        ds_cMpc (path spacing), radius_cMpc (min, max), box, time_s).

    Returns ``cells.take(selected)`` (and the info). ``RuntimeError`` if no cell is within reach (with the distance and
    radius of the closest cell).
    """
    t0 = time.perf_counter()
    box = cells.box
    if len(cells) == 0:
        raise ValueError("select_cells: the container holds no cells.")
    pts, tree, ds = path_tree(paths, box)
    pos = wrap_positions(cells.positions, box)
    rad = selection_radius(cells.size_cMpc, mode, ds)
    backend = resolve_backend(backend)
    d = _nearest_distance_jztree(pts, pos, box, tree, rad) if backend == "jztree" else _nearest_distance_cpu(tree, pos, rad, workers)
    sel = np.nonzero(d <= rad)[0]
    if sel.size == 0:
        raise RuntimeError(no_cell_message(tree, pos, rad, box, ds))
    out = cells.take(sel)
    if not (return_info or logger.isEnabledFor(logging.DEBUG)):
        return out
    info = dict(n_cells=len(cells), n_selected=int(sel.size), indices=sel, mode=mode, backend=backend, ds_cMpc=ds,
                radius_cMpc=(float(rad.min()), float(rad.max())), box=box, time_s=time.perf_counter() - t0)
    logger.debug("select_cells: %d of %d cells within reach (%s radius %.3g-%.3g cMpc, ds = %.3g cMpc, %s back end), %.3f s",
                 info["n_selected"], info["n_cells"], mode, *info["radius_cMpc"], ds, backend, info["time_s"])
    return (out, info) if return_info else out


def no_cell_message(tree: cKDTree, pos: np.ndarray, rad: np.ndarray, box: Optional[float], ds: float) -> str:
    """Why a selection is empty: the cell nearest to the paths and its selection radius (computed on failure only)."""
    d = tree.query(pos, k=1)[0]
    i = int(np.argmin(d - rad))
    return (f"no gas cell is within reach of the paths: of {pos.shape[0]} cells, the closest one (relative to its radius) is "
            f"{d[i]:.4g} cMpc from the nearest path point, beyond its selection radius {rad[i]:.4g} cMpc (box = {box}, path "
            f"spacing ds = {ds:.4g} cMpc). Are the paths and the cells in the same units (cMpc) and the same box?")
