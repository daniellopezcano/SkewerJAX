"""Small helpers shared by the test modules (importable: ``pythonpath = ["tests"]`` in ``pyproject.toml``)."""

import numpy as np


def raises(exc, match, fn, *a, **kw):
    """``fn(*a, **kw)`` raises ``exc`` whose message holds ``match``; returns the exception."""
    try:
        fn(*a, **kw)
    except exc as e:
        assert match in str(e), f"{match!r} not in {str(e)!r}"
        return e
    raise AssertionError(f"{getattr(fn, '__name__', fn)} did not raise {exc.__name__}")


def rel(a, b) -> float:
    """max |a - b| / max |b| (relative to the scale of b)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.max(np.abs(a - b)) / max(float(np.max(np.abs(b))), 1e-300))


def uniform_skewer(n_pt=201, L=2.0, n=1e-12, T=1e4):
    """A straight skewer of manual a(x) (log-linear, a from 0.30 to 0.302), unevenly sampled, through uniform H I gas:
    (SkewerBundle, Cells, points (1, n_pt, 3), a (1, 1, n_pt))."""
    import jax.numpy as jnp

    from skewerjax.core import SkewerBundle, cells_from_values

    pos = np.zeros((1, n_pt, 3))
    pos[0, :, 0] = np.linspace(0.0, L, n_pt) ** 1.3 / L**0.3  # uneven spacing
    a = np.exp(np.linspace(np.log(0.30), np.log(0.302), n_pt))[None, None]
    sb = SkewerBundle(spatial_points=jnp.asarray(pos), manual_scale_factors=jnp.asarray(a), a0=0.302, ICs_defined_at_index=-1)
    # cells at the points: the Voronoi cell of a point of the straight skewer is its dual segment
    cells = cells_from_values(pos[0], None, {"HI": np.full(n_pt, n)}, temperature=T, volumes=np.full(n_pt, (L / n_pt) ** 3))
    return sb, cells, pos, a


def ramp_pieces(n_skewers: int = 7, L: float = 1e-3):
    """Pieces of cells at the points of ``n_skewers`` parallel skewers (T a log ramp along z, n along y, v_x a ramp
    along x; H I and Si III): (Pieces, velocity grid (300,), LineBundle)."""
    import jax.numpy as jnp

    from skewerjax.core import LineBundle, SkewerBundle, cell_pieces, cells_from_values, plane_points

    lb = LineBundle(["HI-LyA", "SiIII-1206"])
    pts = plane_points(L, "x", L / 2, n_skewers, 64)
    gen = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    ramp = lambda lo, hi, ax: lambda x: 10 ** (np.log10(lo) + (np.log10(hi) - np.log10(lo)) * x[:, ax] / L)
    vel = lambda x: np.stack([-50.0 + 130.0 * x[:, 0] / L, 0 * x[:, 0], 0 * x[:, 0]], -1)
    cells = cells_from_values(
        gen,
        None,
        {"HI": ramp(1e-9, 1e-6, 1), "SiIII": ramp(1e-9, 1e-6, 1)},
        temperature=ramp(1e4, 1e6, 2),
        velocity=vel,
        collisional=0.1,
        volumes=np.full(gen.shape[0], (L / 63) ** 3),
    )
    sb = SkewerBundle(spatial_points=pts, manual_scale_factors=jnp.full((1,) + pts.shape[:2], 0.9))
    return cell_pieces(cells, lb, skewer_bundle=sb, a_dagger=0.88), jnp.linspace(-400, 400, 300), lb
