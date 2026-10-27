"""Run tests/ without pytest (``pytest_shim/``), for environments where pytest is not installed; with pytest, use
``pytest`` (``tests/README.md``): the same tests, markers and default.

    python scripts/dev/run_tests.py [-m EXPR] [--all] [path or module ...]

* ``-m EXPR``: a marker expression as pytest's (``tng``, ``not slow``, ``tng and not slow``); default ``not slow``, the
  default of ``pyproject.toml``; ``--all``: every test (``-m "slow or not slow"``);
* paths (``tests/core``, ``tests/core/test_voigt.py``) or module names (``test_voigt``) select the files.

Tests whose markers need what this machine lacks (``conftest.MARKER_REQUIREMENTS``) are skipped with the reason.
"""

import argparse
import importlib.util
import inspect
import itertools
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
TESTS = HERE.parents[1] / "tests"
sys.path.insert(0, str(HERE / "pytest_shim"))
sys.path.insert(0, str(TESTS))  # pyproject: pythonpath = ["tests"]
import pytest  # noqa: E402  (the shim)

import conftest  # noqa: E402  (CPU, x64, marker requirements)

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("-m", dest="markexpr", default="not slow")
ap.add_argument("--all", action="store_true")
ap.add_argument("select", nargs="*")
args = ap.parse_args()
expr = "slow or not slow" if args.all else args.markexpr
MARKERS = ("slow", "tng", "gpu", "fake_spectra")


def selected(f: Path) -> bool:
    if not args.select:
        return True
    return any(s == f.stem or Path(s).resolve() == f or Path(s).resolve() in f.parents for s in args.select)


def wanted(marks) -> bool:
    return bool(eval(expr, {}, {m: m in marks for m in MARKERS}))


res = []
for f in sorted(TESTS.rglob("test_*.py")):
    if not selected(f.resolve()):
        continue
    sys.path.insert(0, str(f.parent))
    spec = importlib.util.spec_from_file_location(f.stem, f)
    m = importlib.util.module_from_spec(spec)
    sys.modules[f.stem] = m
    spec.loader.exec_module(m)
    rel = f.relative_to(TESTS)
    if hasattr(m, "setup_module"):
        m.setup_module()
    for name, fn in inspect.getmembers(m, inspect.isfunction):
        if not name.startswith("test_") or fn.__module__ != m.__name__:
            continue
        marks = getattr(fn, "_marks", set())
        if not wanted(marks):
            continue
        skip = getattr(fn, "_skip", (False, ""))
        reason = conftest.skip_reason(marks) or (skip[1] if skip[0] else "")
        params = getattr(fn, "_params", [])
        combos = [dict(zip([p[0] for p in params], c)) for c in itertools.product(*[p[1] for p in params])] if params else [{}]
        for i, kw in enumerate(combos):
            tag = f"{rel}::{name}" + (f"[{i}]" if kw else "")
            if reason:
                res.append((tag, "SKIP", reason))
                continue
            t = time.time()
            try:
                fn(**kw)
                st, det = "PASS", ""
            except pytest._Skipped as e:
                st, det = "SKIP", str(e)
            except Exception:
                st, det = "FAIL", traceback.format_exc()[-1500:]
            if hasattr(m, "teardown_function"):
                m.teardown_function()
            res.append((tag, st, det))
            print(f"{st} {tag} ({time.time() - t:.1f} s)", flush=True)
            if st == "FAIL":
                print(det)
for tag, st, det in res:
    if st == "SKIP":
        print(f"SKIP {tag}: {det}")
print(
    f"\nSUMMARY ({expr}): {sum(r[1] == 'PASS' for r in res)} pass, {sum(r[1] == 'FAIL' for r in res)} fail, "
    f"{sum(r[1] == 'SKIP' for r in res)} skip"
)
sys.exit(1 if any(r[1] == "FAIL" for r in res) else 0)
