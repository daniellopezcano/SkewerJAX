"""Selection back ends: "auto" falls back to the CPU without jztree, "jztree" fails clearly, and the CPU distance is the
periodic minimum-image distance (toy points; the TNG50-4 equivalence of the GPU back end needs jztree and a GPU)."""
import numpy as np
import pytest
from scipy.spatial import cKDTree

from skewerjax.simulations import selection as sel

BOX = 10.0


def test_auto_resolves_to_cpu_without_jztree():
    expected = "jztree" if sel.jztree_available() else "cpu"
    assert sel.resolve_backend("auto") == expected
    assert sel.resolve_backend("cpu") == "cpu"
    with pytest.raises(ValueError):
        sel.resolve_backend("gpu")


def test_jztree_backend_fails_clearly_when_missing():
    if sel.jztree_available():
        pytest.skip("jztree and a CUDA device are available")
    with pytest.raises(RuntimeError, match="jztree"):
        sel.resolve_backend("jztree")


def test_cpu_distance_is_the_periodic_minimum_image():
    rng = np.random.default_rng(3)
    t = np.linspace(0.0, 1.0, 300)
    path = np.stack([1.0 + 1.7 * BOX * t, 9.0 + 0.6 * BOX * t, np.full_like(t, 5.0)], -1)      # crosses the x and y faces
    pts = sel.wrap_positions(path, BOX)
    cells = rng.uniform(0.0, BOX, (2000, 3))
    rad = np.full(cells.shape[0], 0.8)
    d = sel._nearest_distance_cpu(cKDTree(pts, boxsize=BOX), cells, rad, workers=1)
    diff = cells[:, None, :] - pts[None, :, :]
    diff -= BOX * np.round(diff / BOX)
    brute = np.sqrt((diff ** 2).sum(-1)).min(1)
    inside = brute <= rad
    np.testing.assert_allclose(d[inside], brute[inside], rtol=0, atol=1e-12)
    assert np.all(np.isinf(d[~inside]) | (d[~inside] > rad[~inside]))
