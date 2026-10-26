"""Skewers: the sampling points of a set of sightlines, their geometry and their scale factors a(x)."""
from __future__ import annotations

from typing import Optional, Sequence, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from skewerjax.core.cosmology import Cosmology


# --------------------------------------------------------------------------- polyline geometry
def line_elements(points) -> jnp.ndarray:
    """Trapezoid line element of every point of a polyline: w_i = (d_{i-1} + d_i) / 2.

    w_i is the length of the dual segment between the midpoints around x_i, so sum_i w_i f_i is the trapezoid
    rule on the path length and sum_i w_i = path length. points : (..., N, 3); returns (..., N), same unit.
    """
    d = jnp.linalg.norm(jnp.diff(points, axis=-2), axis=-1)
    z = jnp.zeros(d.shape[:-1] + (1,), d.dtype)
    return 0.5 * (jnp.concatenate([z, d], axis=-1) + jnp.concatenate([d, z], axis=-1))


def path_geometry(points) -> dict:
    """Path length, line elements, tangents and segment midpoints of polylines.

    Parameters
    ----------
    points : (..., N, 3) positions [cMpc], N >= 2, unwrapped along the path.

    Returns
    -------
    dict: ``s`` (..., N) path length from the first point; ``line_elements`` (..., N) (``line_elements``);
    ``tangents`` (..., N, 3) unit tangent (central difference, one-sided at the ends); ``mid_lo``, ``mid_hi``
    (..., N, 3) the midpoints m_{i-1/2}, m_{i+1/2} (= x_i at the two ends).
    """
    x = jnp.asarray(points)
    if x.ndim < 2 or x.shape[-1] != 3 or x.shape[-2] < 2:
        raise ValueError(f"points must be (..., N >= 2, 3), got {x.shape}.")
    d = jnp.linalg.norm(jnp.diff(x, axis=-2), axis=-1)
    z = jnp.zeros(d.shape[:-1] + (1,), d.dtype)
    s = jnp.concatenate([z, jnp.cumsum(d, axis=-1)], axis=-1)
    mid = 0.5 * (x[..., 1:, :] + x[..., :-1, :])
    mid_lo = jnp.concatenate([x[..., :1, :], mid], axis=-2)
    mid_hi = jnp.concatenate([mid, x[..., -1:, :]], axis=-2)
    t = mid_hi - mid_lo
    t = t / jnp.maximum(jnp.linalg.norm(t, axis=-1, keepdims=True), 1e-300)
    return dict(s=s, line_elements=line_elements(x), tangents=t, mid_lo=mid_lo, mid_hi=mid_hi)


# --------------------------------------------------------------------------- point layouts
def plane_points(L_box, fixed_axis: str, fixed_value, n_var1: int, n_var2: int) -> jnp.ndarray:
    """(n_var1, n_var2, 3) grid on the plane ``fixed_axis`` = fixed_value ('x', 'y' or 'z'), linspace(0, L_box)."""
    f, v1, v2 = {"x": (0, 1, 2), "y": (1, 2, 0), "z": (2, 0, 1)}[fixed_axis]
    g1, g2 = jnp.meshgrid(jnp.linspace(0.0, L_box, n_var1), jnp.linspace(0.0, L_box, n_var2), indexing="ij")
    coords = [jnp.zeros_like(g1) for _ in range(3)]
    coords[f], coords[v1], coords[v2] = jnp.full_like(g1, fixed_value), g1, g2
    return jnp.stack(coords, axis=-1)


def grid_skewer_points(L_box, N_grid: int, N_ppsk: int, direction: str = "y", half_shift: bool = True) -> jnp.ndarray:
    """(N_grid, N_grid, N_ppsk, 3) float32 points of N_grid^2 parallel skewers along ``direction``.

    Each skewer runs linspace(0, L_box, N_ppsk) along ``direction``; the transverse grid is linspace(0, L_box,
    N_grid) (``half_shift=False``) or the cell centres (j + 1/2) L_box / N_grid (``half_shift=True``, default).
    """
    d = {"x": 0, "y": 1, "z": 2}[direction]
    t1, t2 = {0: (1, 2), 1: (0, 2), 2: (0, 1)}[d]
    if half_shift:
        c = jnp.linspace(0.0, L_box, N_grid, endpoint=False) + 0.5 * (L_box / N_grid)
    else:
        c = jnp.linspace(0.0, L_box, N_grid)
    g1, g2 = jnp.meshgrid(c, c, indexing="ij")
    xyz = jnp.zeros((N_grid, N_grid, N_ppsk, 3), dtype=jnp.float32)
    xyz = xyz.at[..., d].set(jnp.linspace(0.0, L_box, N_ppsk)[None, None, :])
    xyz = xyz.at[..., t1].set(g1[..., None])
    return xyz.at[..., t2].set(g2[..., None])


def axis_aligned_skewer_points(cofm, axis, box: float, spacing: float) -> np.ndarray:
    """Straight skewers parallel to box axes, as fake_spectra's sightlines [cMpc].

    Parameters
    ----------
    cofm : (N_sk, 3) a point of each skewer [cMpc] (its coordinate along ``axis`` is ignored).
    axis : (N_sk,) 1, 2 or 3 (x, y, z).
    box : box side [cMpc]; spacing : target spacing [cMpc] (rounded to divide the box).

    Returns
    -------
    (N_sk, N + 1, 3) points from 0 to box; the first and last point of a skewer are the same place of the periodic box.
    """
    cofm = np.atleast_2d(np.asarray(cofm, dtype=np.float64))
    axis = np.atleast_1d(np.asarray(axis, dtype=int))
    n = int(round(box / spacing))
    ds = box / n
    pts = np.repeat(cofm[:, None, :], n + 1, axis=1)
    for k, ax in enumerate(axis):
        pts[k, :, ax - 1] = np.arange(n + 1) * ds
    return pts


# --------------------------------------------------------------------------- single-skewer kernels
def photon_directions(points: jnp.ndarray, direction) -> jnp.ndarray:
    """(N, 3) unit photon directions of one skewer from consecutive point differences.

    direction = +1: x_{i+1} - x_i (last one repeated); -1: x_i - x_{i+1} (first one repeated).
    """
    fwd = points[1:] - points[:-1]
    bwd = points[:-1] - points[1:]
    diffs = jnp.where(direction == 1, jnp.concatenate([fwd, fwd[-1][None, :]], axis=0),
                      jnp.concatenate([bwd[0][None, :], bwd], axis=0))
    return diffs / jnp.maximum(jnp.linalg.norm(diffs, axis=-1, keepdims=True), 1e-20)


def comoving_distances(points: jnp.ndarray, x0, ic_index, is_absorption) -> jnp.ndarray:
    """(N,) comoving distance [Mpc] of the points of one skewer, accumulated from the IC point.

    The steps are taken from the IC end (``ic_index`` 0 or -1), counted negative for absorption (towards the
    observer) and positive for emission, and offset by ``x0``.
    """
    steps = jnp.linalg.norm(points[1:] - points[:-1], axis=-1)
    steps = jnp.where(ic_index == -1, steps[::-1], steps)
    rel = jnp.cumsum(steps)
    rel = jnp.where(is_absorption, -rel, rel)
    return x0 + jnp.concatenate([jnp.array([0.0]), rel], axis=0)


def _scale_factors(cosmology: Cosmology, a0, distances) -> jnp.ndarray:
    """(N_cosmologies, N) a(x) of one skewer, from a(x0) = a0 along ``distances``."""
    return cosmology.solve_a_of_x(jnp.repeat(a0, cosmology.N_cosmologies), distances)


# --------------------------------------------------------------------------- SkewerBundle
@jax.tree_util.register_pytree_node_class
class SkewerBundle:
    """N_sk skewers of N_ppsk comoving points each, with their photon directions and scale factors.

    Parameters
    ----------
    spatial_points : (N_sk, N_ppsk, 3) comoving positions [Mpc].
    a0 : scale factor at the IC point, broadcastable to (N_sk,).
    x0 : comoving distance [Mpc] of the IC point, broadcastable to (N_sk,).
    ICs_defined_at_index : 0 (first point) or -1 (last point), broadcastable to (N_sk,).
    ICs_at_emission_or_absorption : "absorption" or "emission" (or one per skewer): sets the direction of the
        photons with respect to the order of the points.
    los_directions : optional (N_sk, N_ppsk, 3) unit photon directions (default: from the points).
    manual_scale_factors : optional (N_cosmologies, N_sk, N_ppsk) a(x); otherwise solved with ``cosmologies``.
    cosmologies : ``Cosmology`` (required without ``manual_scale_factors``).

    Attributes
    ----------
    quadri_vectors : (N_cosmologies, N_sk, N_ppsk, 4) events (a, x, y, z) of the sampling points.
    scale_factors, los_directions, comoving_positions_from_ICs, array_direction (+1 / -1), is_absorption.

    Notes
    -----
    JAX pytree (every array is a leaf). The constructor validates and solves a(x) eagerly; the per-skewer
    kernels ``photon_directions``, ``comoving_distances`` are pure and vmapped over skewers.
    """

    _leaves = ("spatial_points", "a0", "x0", "ICs_defined_at_index", "is_absorption", "array_direction",
               "los_directions", "comoving_positions_from_ICs", "scale_factors", "quadri_vectors")

    def __init__(self, spatial_points: ArrayLike, a0: ArrayLike = 1.0, x0: ArrayLike = 0.0,
                 ICs_defined_at_index: ArrayLike = -1,
                 ICs_at_emission_or_absorption: Union[str, Sequence[str]] = "absorption",
                 los_directions: Optional[ArrayLike] = None, manual_scale_factors: Optional[ArrayLike] = None,
                 cosmologies: Optional[Cosmology] = None):
        spatial_points = jnp.asarray(spatial_points)
        if spatial_points.ndim != 3 or spatial_points.shape[-1] != 3:
            raise ValueError(f"spatial_points must be (N_sk, N_ppsk, 3), got {spatial_points.shape}")
        N_sk, N_ppsk = spatial_points.shape[:2]
        a0 = jnp.broadcast_to(jnp.asarray(a0), (N_sk,))
        x0 = jnp.broadcast_to(jnp.asarray(x0), (N_sk,))
        ic = jnp.asarray(ICs_defined_at_index)
        if jnp.any((ic != -1) & (ic != 0)):
            raise ValueError(f"ICs_defined_at_index must be -1 or 0, got {ic}")
        if ic.ndim == 0:
            ic = jnp.full((N_sk,), ic, dtype=jnp.int32)
        if isinstance(ICs_at_emission_or_absorption, str):
            is_absorption = jnp.full((N_sk,), ICs_at_emission_or_absorption == "absorption")
        else:
            is_absorption = jnp.array([s == "absorption" for s in ICs_at_emission_or_absorption])
        is_start = ic == 0
        direction = jnp.where(is_start, jnp.where(is_absorption, -1, 1), jnp.where(is_absorption, 1, -1))

        if los_directions is not None:
            los_directions = jnp.asarray(los_directions)
            if los_directions.shape != (N_sk, N_ppsk, 3):
                raise ValueError(f"Provided los_directions shape {los_directions.shape} must match (N_sk, N_ppsk, 3) = "
                                 f"{(N_sk, N_ppsk, 3)}")
            max_dev = jnp.max(jnp.abs(jnp.linalg.norm(los_directions, axis=-1) - 1.0))
            if max_dev > 1e-5:
                raise ValueError(f"los_directions must be normalized. Max |‖d‖-1| = {max_dev:.3e}")
        else:
            los_directions = jax.vmap(photon_directions)(spatial_points, direction)

        distances = jax.vmap(comoving_distances)(spatial_points, x0, ic, is_absorption)
        if manual_scale_factors is not None:
            scale_factors = jnp.asarray(manual_scale_factors)
            if scale_factors.ndim != 3:
                raise ValueError(f"manual_scale_factors must have shape (N_cosmologies, N_sk, N_ppsk), got {scale_factors.shape}")
        else:
            if cosmologies is None:
                raise ValueError("When manual_scale_factors is not provided, you must supply the 'cosmologies' argument.")
            a_from_ics = jnp.swapaxes(jax.vmap(lambda a, d: _scale_factors(cosmologies, a, d))(a0, distances), 0, 1)
            scale_factors = jnp.where((ic == -1)[None, :, None], jnp.flip(a_from_ics, axis=-1), a_from_ics)

        self.spatial_points, self.a0, self.x0 = spatial_points, a0, x0
        self.ICs_defined_at_index, self.is_absorption, self.array_direction = ic, is_absorption, direction
        self.los_directions, self.comoving_positions_from_ICs = los_directions, distances
        self.scale_factors = scale_factors
        sp = jnp.tile(spatial_points[None, ...], (scale_factors.shape[0], 1, 1, 1))
        self.quadri_vectors = jnp.concatenate([scale_factors[..., None], sp], axis=-1)

    def tree_flatten(self):
        return tuple(getattr(self, n) for n in self._leaves), None

    @classmethod
    def tree_unflatten(cls, aux, leaves):
        obj = object.__new__(cls)
        for n, v in zip(cls._leaves, leaves):
            setattr(obj, n, v)
        return obj

    @property
    def N_sk(self) -> int:
        return self.spatial_points.shape[0]

    @property
    def N_ppsk(self) -> int:
        return self.spatial_points.shape[1]

    @property
    def path_lengths(self) -> jnp.ndarray:
        """(N_sk, N_ppsk) comoving path length [Mpc] from the first point."""
        return path_geometry(self.spatial_points)["s"]

    @property
    def line_elements(self) -> jnp.ndarray:
        """(N_sk, N_ppsk) line elements [cMpc]: the lengths of the dual segments (from midpoint to
        midpoint), which the deposition cuts into pieces (``absorbers.cell_pieces``)."""
        return line_elements(self.spatial_points)

    @property
    def tangents(self) -> jnp.ndarray:
        """(N_sk, N_ppsk, 3) unit tangent of the path, in the order of the points."""
        return path_geometry(self.spatial_points)["tangents"]

    @property
    def segment_midpoints(self) -> tuple:
        """(m_{i-1/2}, m_{i+1/2}), each (N_sk, N_ppsk, 3) [cMpc]: the dual segment of every point."""
        g = path_geometry(self.spatial_points)
        return g["mid_lo"], g["mid_hi"]
