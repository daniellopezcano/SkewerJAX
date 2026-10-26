"""Run tests/ without pytest: import each test module and call its test_* functions, expanding parametrize and honouring
skipif (the ``pytest`` of ``pytest_shim/``: mark.parametrize, mark.skipif, raises, approx, skip; no fixtures).

    python scripts/dev/run_tests.py [module ...]        (e.g. test_regression test_plotting; default: every module)

With pytest installed (the ``dev`` extra), ``pytest`` runs the same suite."""
import importlib.util, inspect, itertools, sys, time, traceback
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "pytest_shim"))
TESTS = HERE.parents[1] / "tests"
sys.path.insert(0, str(TESTS))
import conftest  # noqa: F401  (CPU, x64)
import pytest  # noqa: E402  (the shim)
res = []
for f in sorted(TESTS.glob("test_*.py")):
    if len(sys.argv) > 1 and f.stem not in sys.argv[1:]:
        continue
    spec = importlib.util.spec_from_file_location(f.stem, f); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    if hasattr(m, "setup_module"): m.setup_module()
    for name, fn in inspect.getmembers(m, inspect.isfunction):
        if not name.startswith("test_") or fn.__module__ != m.__name__: continue
        skip = getattr(fn, "_skip", (False, ""))
        params = getattr(fn, "_params", [])
        combos = [dict(zip([p[0] for p in params], c)) for c in itertools.product(*[p[1] for p in params])] if params else [{}]
        for kw in combos:
            tag = f"{f.stem}::{name}" + (f"[{','.join(map(str, kw.values()))}]" if kw else "")
            if skip[0]:
                res.append((tag, "SKIP", skip[1])); continue
            t = time.time()
            try:
                fn(**kw); st, det = "PASS", ""
            except sys.modules["pytest"]._Skipped as e:
                st, det = "SKIP", str(e)
            except Exception as e:
                st, det = "FAIL", traceback.format_exc()[-1500:]
            if hasattr(m, "teardown_function"): m.teardown_function()
            res.append((tag, st, det)); print(f"{st} {tag} ({time.time() - t:.1f} s)", flush=True)
            if st == "FAIL": print(det)
print(f"\nSUMMARY: {sum(r[1] == 'PASS' for r in res)} pass, {sum(r[1] == 'FAIL' for r in res)} fail, {sum(r[1] == 'SKIP' for r in res)} skip")
sys.exit(1 if any(r[1] == "FAIL" for r in res) else 0)
