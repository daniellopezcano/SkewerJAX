"""Regression of everything the notebooks build on, against reference arrays (``data/*.npz``, see ``outputs.py``): every
array must agree to 1e-10 of its largest value (integers exactly). The references are regenerated only for an intended
change of the results (``scripts/dev/update_regression_references.py``; the reason goes to ``CHANGELOG.md``)."""

from pathlib import Path

import numpy as np
import pytest

import outputs

DATA = Path(__file__).resolve().parent / "data"


def _compare(name, out):
    ref = np.load(DATA / f"{name}.npz")
    assert set(ref.files) == set(out), set(ref.files) ^ set(out)
    for k in ref.files:
        a, b = ref[k], out[k]
        assert a.shape == b.shape, (k, a.shape, b.shape)
        if a.dtype.kind in "iub":
            np.testing.assert_array_equal(b, a, err_msg=k)
        else:
            assert np.max(np.abs(b - a)) <= 1e-10 * max(np.max(np.abs(a)), 1e-300), (k, np.max(np.abs(b - a)), np.max(np.abs(a)))


def test_route():
    out = outputs.route_outputs()
    _compare("route_toy", out)
    assert out["tau_default"].max() > 1.0 and out["tau_period"].max() > 1.0  # the toy absorbs: the test has teeth


def test_modules():
    _compare("modules_toy", outputs.modules_outputs())


@pytest.mark.tng
def test_modules_tng():
    _compare("modules_tng", outputs.tng_outputs())
