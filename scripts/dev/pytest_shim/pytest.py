"""Minimal stand-in for pytest (not installed): mark.parametrize, mark.skipif, raises, approx."""
class _Mark:
    def parametrize(self, name, values):
        def deco(f):
            f.__dict__.setdefault("_params", []).append((name, list(values)))
            return f
        return deco
    def skipif(self, cond, reason=""):
        def deco(f):
            f._skip = (bool(cond), reason)
            return f
        return deco
mark = _Mark()
class raises:
    def __init__(self, exc, match=None): self.exc = exc
    def __enter__(self): return self
    def __exit__(self, t, v, tb):
        if t is None: raise AssertionError(f"did not raise {self.exc}")
        return issubclass(t, self.exc)
def approx(x, rel=1e-6, abs=1e-12):
    class A:
        def __eq__(s, o): return __builtins__["abs"](o - x) <= max(rel * __builtins__["abs"](x), abs)
    return A()
class _Skipped(Exception):
    pass
def skip(reason=""):
    raise _Skipped(reason)
