"""Minimal stand-in for pytest, for environments without it (``scripts/dev/run_tests.py``): ``mark.parametrize``,
``mark.skipif``, any other ``mark.<name>`` (recorded for ``-m``), ``raises`` (with ``match``), ``approx``, ``skip``. The
test suite uses nothing else (no fixtures)."""

import re


class _Mark:
    def parametrize(self, name, values, ids=None):
        def deco(f):
            f.__dict__.setdefault("_params", []).append((name, list(values), ids))
            return f

        return deco

    def skipif(self, cond, reason=""):
        def deco(f):
            f._skip = (bool(cond), reason)
            return f

        return deco

    def __getattr__(self, name):  # slow, tng, gpu, fake_spectra, ...
        def deco(f):
            f.__dict__.setdefault("_marks", set()).add(name)
            return f

        return deco


mark = _Mark()


class raises:
    def __init__(self, exc, match=None):
        self.exc, self.match = exc, match

    def __enter__(self):
        return self

    def __exit__(self, t, v, tb):
        if t is None:
            raise AssertionError(f"did not raise {self.exc}")
        if not issubclass(t, self.exc):
            return False
        if self.match is not None and not re.search(self.match, str(v)):
            raise AssertionError(f"{self.match!r} does not match {str(v)!r}")
        return True


def approx(x, rel=1e-6, abs=1e-12):
    class A:
        def __eq__(s, o):
            return __builtins__["abs"](o - x) <= max(rel * __builtins__["abs"](x), abs)

    return A()


class _Skipped(Exception):
    pass


def skip(reason=""):
    raise _Skipped(reason)
