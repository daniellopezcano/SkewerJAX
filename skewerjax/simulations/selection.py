"""Cell selection (layer 4): streaming k-d tree selection of the gas cells that can reach a set of paths.

The paths are the sampling points of a ``SkewerBundle`` (any shape: straight, curved, crossing
the box faces; positions in comoving Mpc, unwrapped along the path). A periodic k-d tree is built
on the *path points* (``scipy.spatial.cKDTree(boxsize=box)``, minimum-image distances), and each
block of cells streamed by the reader (layer 1) is queried against it. Memory is bounded by one
block of cells (``max_rows``) plus the tree of the current skewer batch.

Selection radius of a cell c with cube side h_c = V_c^(1/3), for a path sampled with spacing
<= ds (the nearest *point* is at most ds/2 farther than the nearest point of the polyline):

* ``"tophat"``: d <= h_c + ds/2 (the fake_spectra sphere of radius h_c can touch the path);
* ``"voronoi"``: d <= max(k h_c, r_min) + ds/2. A point of the path can belong to the Voronoi cell
  of generator c only if it is within the cell's circumradius, which ``k = 2.5`` bounds for all
  but pathological cells; the deposition (layer 5) checks this a posteriori and the notebook
  verifies it against a brute-force nearest-generator search over the whole snapshot.

Back ends (``backend``): ``"cpu"`` (default) queries a scipy ``cKDTree``; ``"jztree"`` computes the same
nearest-path-point distance on a CUDA GPU with JZ-TREE (``jztree.knn.knn`` with ``boxsize``, periodic;
https://jstuecker.github.io/jztree/api.html), re-checking in float64 on the host the cells whose distance is
within a relative ``1e-6`` of their radius, so that the selected set does not depend on the back end;
``"auto"`` takes ``"jztree"`` when jztree imports and JAX sees a CUDA device, otherwise ``"cpu"``. jztree is
optional: CPU-only installations never import it.

Spatial-index note: the tree is on the paths, not on the cells, because the paths are few
(N_sk x N_pts points) and the cells are many and streamed. For multi-device runs, skewer batches
are independent and can be distributed across devices or hosts; each worker streams the same
files.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np
from scipy.spatial import cKDTree

from skewerjax.simulations.gas_cells import GasCells

GEOMETRY_DATASETS = ("Coordinates", "Masses", "Density")


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


def select_cells_near_paths(reader, paths, *, mode: str = "voronoi", voronoi_factor: float = 2.5,
                            voronoi_min: float = 0.0, datasets: Sequence[str] = None,
                            max_rows: int = 4_000_000, workers: int = -1, backend: str = "cpu",
                            progress: Optional[Callable[[int, int, int], None]] = None) -> GasCells:
    """Stream the snapshot and keep the cells that can contribute to any of ``paths``.

    Parameters
    ----------
    reader : layer-1 reader (``iter_blocks``, ``read_rows``, ``to_gas_cells``, ``header``).
    paths : (N_sk, N_pts, 3) sampling points [cMpc], e.g. ``SkewerBundle.spatial_points``.
    mode : "voronoi" (default deposition) or "tophat" (fake_spectra's kernel); "voronoi" selects
        a superset of "tophat" for voronoi_factor >= 1.
    datasets : raw datasets to read for the selected cells (default: the GAS_DATASETS of the reader).
    max_rows : cells per streamed block (memory bound).
    workers : threads of the k-d tree queries (-1: all cores).
    backend : "cpu" (default, scipy cKDTree), "jztree" (CUDA GPU; ImportError / RuntimeError if jztree or a CUDA
        device is missing) or "auto" (``resolve_backend``). Every back end selects the same cells.

    Returns
    -------
    GasCells of the selected cells, physical units (``reader.to_gas_cells``), with ``ids`` =
    (file_index << 40) | row, so that every cell can be traced back to the snapshot.
    """
    from skewerjax.simulations.readers import GAS_DATASETS
    datasets = tuple(GAS_DATASETS if datasets is None else datasets)
    missing = set(GEOMETRY_DATASETS) - set(datasets)
    if missing:
        raise ValueError(f"datasets must include {sorted(missing)}.")
    hdr = reader.header
    box = hdr.box_cMpc
    ds = max_path_spacing(paths)
    pts = wrap_positions(np.asarray(paths, dtype=np.float64).reshape(-1, 3), box)
    tree = cKDTree(pts, boxsize=box)
    backend = resolve_backend(backend)
    parts = []
    n_sel = 0
    n_blocks = None
    for ib, (fi, r0, base) in enumerate(reader.iter_blocks(GEOMETRY_DATASETS, max_rows=max_rows)):
        pos = reader.positions_cMpc(base["Coordinates"])
        size = np.cbrt(reader.volumes_cMpc3(base["Masses"], base["Density"]))
        rad = selection_radius(size, mode, ds, voronoi_factor, voronoi_min)
        if backend == "jztree":
            d = _nearest_distance_jztree(pts, pos, box, tree, rad)
        else:
            d = _nearest_distance_cpu(tree, pos, rad, workers)
        sel = np.nonzero(d <= rad)[0]
        if sel.size:
            rest = [k for k in datasets if k not in GEOMETRY_DATASETS]
            raw = {k: base[k][sel] for k in GEOMETRY_DATASETS}
            if rest:                         # one contiguous slab of the block (bounded by max_rows), indexed in memory:
                lo, hi = r0 + int(sel[0]), r0 + int(sel[-1]) + 1     # h5py's scattered-row reads are much slower
                slab = reader.read_rows(fi, rest, start=lo, stop=hi)
                raw.update({k: v[r0 + sel - lo] for k, v in slab.items()})
            parts.append(reader.to_gas_cells(raw, ids=(np.int64(fi) << 40) | (r0 + sel).astype(np.int64)))
            n_sel += sel.size
        if progress is not None:
            progress(ib, fi, n_sel)
    if n_sel == 0:
        raise RuntimeError("no gas cell is within reach of the paths.")
    return GasCells.concatenate(parts)
