"""The import graph of the package: the core at the bottom and two realms on it, with dependencies one way only.

* ``core`` (lines, cosmology, the paths, cells, tau) imports nothing from the other subpackages;
* ``simulations`` (the gas) imports ``core`` only: not ``summary_stats`` or ``comparison``;
* ``summary_stats`` (statistics from arrays, geometry and ``core.Cells``) imports ``core`` at most: never
  ``simulations``, ``comparison`` or ``utils``.

Checked twice: statically, on the source of every module (every ``import`` and ``from ... import``, also those inside
functions), and at run time, in a fresh interpreter: importing ``skewerjax.summary_stats``
loads no module of ``skewerjax.simulations`` (the subpackages of ``skewerjax/__init__.py`` are lazy, PEP 562)."""

import ast
import subprocess
import sys
from pathlib import Path

import skewerjax

PKG = Path(skewerjax.__file__).resolve().parent
REALMS = ("core", "simulations", "summary_stats", "comparison", "utils")
ALLOWED = {  # the subpackages each realm may import (besides itself)
    "core": set(),
    "simulations": {"core"},
    "summary_stats": {"core"},
}


def _imported_subpackages(source: str, parts: tuple, realm: str) -> set:
    """The other ``skewerjax`` subpackages imported anywhere (also inside functions) in a module of source ``source`` at
    ``skewerjax/<parts>`` (e.g. ("summary_stats", "lattice.py")), absolute and relative imports."""
    package = ["skewerjax", *parts[:-1]]
    found = set()
    for node in ast.walk(ast.parse(source)):
        names = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = package[: len(package) - (node.level - 1)] if node.level else []
            mod = ".".join(base + ([node.module] if node.module else []))
            names = [mod] + ([f"{mod}.{a.name}" for a in node.names] if mod == "skewerjax" else [])
        for n in names:
            q = n.split(".")
            if q[0] == "skewerjax" and len(q) > 1 and q[1] in REALMS and q[1] != realm:
                found.add(q[1])
    return found


def test_static_import_graph():
    bad = []
    for realm, allowed in ALLOWED.items():
        files = sorted((PKG / realm).rglob("*.py"))
        assert files, realm
        for f in files:
            extra = _imported_subpackages(f.read_text(), f.relative_to(PKG).parts, realm) - allowed
            if extra:
                bad.append(f"{f.relative_to(PKG)} imports skewerjax.{sorted(extra)}")
    assert not bad, "\n".join(bad)


def test_the_checker_sees_lazy_imports():
    """The static check catches an import inside a function, absolute or relative, and ``from skewerjax import x``."""
    src = (
        "def f():\n    from skewerjax.simulations import ToyBoxes\n    import skewerjax.comparison.source\n"
        "from ..comparison import x\nfrom skewerjax import utils\nfrom . import grid3d\n"
    )
    assert _imported_subpackages(src, ("summary_stats", "probe.py"), "summary_stats") == {"simulations", "comparison", "utils"}


def test_runtime_imports():
    code = (
        "import sys; import skewerjax.summary_stats; "
        "print(sorted({m.split('.')[1] for m in sys.modules if m.startswith('skewerjax.')}))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    loaded = set(eval(out))  # noqa: S307 (our own print of a list of names)
    assert "summary_stats" in loaded and not loaded & {"simulations", "comparison"}, loaded
    code = "import skewerjax; from skewerjax import simulations, comparison, utils; print(skewerjax.core.__name__, simulations.ToyBoxes.__name__)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.split()
    assert out == ["skewerjax.core", "ToyBoxes"], "the lazy subpackages still import as before"
