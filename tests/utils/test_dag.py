"""The architecture as data (``skewerjax.utils.dag``) against the code and the README.

* every node's modules exist and define its key functions and classes (``api``); every edge names the functions that
  implement it, and each "module:name" exists (a static scan: the module's file and a top-level def, class or
  assignment of that name); the graph is acyclic and every key, realm and notebook it names exists;
* the layers (``LAYER_EDGES``) are the import graph of the subpackages (``tests/test_import_graph.py``'s scan);
* the README's generated blocks (layers, pipeline, tree, concepts) are those of the specification
  (``scripts/dev/make_diagrams.py``);
* ``plot_dag`` draws every edge from outline to outline and refuses an unknown ingredient."""

import ast
import subprocess
import sys

import numpy as np
import pytest
from test_import_graph import PKG, REALMS, _imported_subpackages

from skewerjax.utils import dag
from skewerjax.utils import plotting as pl

ROOT = PKG.parent


def _defined(module: str) -> set:
    """The top-level names defined in ``skewerjax.<module>`` (def, class, assignment, import), from its source."""
    path = PKG.joinpath(*module.split(".")).with_suffix(".py")
    assert path.exists(), f"skewerjax.{module}: no file {path}"
    names = set()
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names |= {t.id for t in node.targets if isinstance(t, ast.Name)}
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names |= {(a.asname or a.name).split(".")[0] for a in node.names}
    return names


def test_nodes_and_edges_name_existing_code():
    for key, n in dag.DAG_NODES.items():
        mods = [m.strip() for m in n.module.split(",")]
        defined = set().union(*(_defined(m) for m in mods))
        missing = [a.strip() for a in n.api.split(",") if a.strip() not in defined]
        assert not missing, f"{key}: {missing} not in {mods}"
        assert n.realm in dag.REALMS or n.realm == "external", key
    for e in dag.DAG_EDGES + dag.DAG_EXTERNAL:
        assert e.a in dag.DAG_NODES and e.b in dag.DAG_NODES, e
        assert e.kind in ("solid", "optional", "external") and e.impl, e
        for item in e.impl:
            module, name = item.split(":")
            assert name in _defined(module), f"{e.a} -> {e.b}: {name} not defined in skewerjax.{module}"


def test_the_graph_is_acyclic_and_the_notebooks_exist():
    succ = {k: [e.b for e in dag.DAG_EDGES + dag.DAG_EXTERNAL if e.a == k] for k in dag.DAG_NODES}
    state = {}

    def visit(k):
        assert state.get(k) != 1, f"cycle through {k}"
        if state.get(k) == 2:
            return
        state[k] = 1
        for j in succ[k]:
            visit(j)
        state[k] = 2

    for k in dag.DAG_NODES:
        visit(k)
    names = {p.name[:2] for p in (ROOT / "notebooks").glob("[0-9][0-9]_*.ipynb")}
    assert set(dag.DAG_NOTEBOOKS) <= set(dag.DAG_NODES)
    for k, nbs in dag.DAG_NOTEBOOKS.items():
        assert {s.strip() for s in nbs.split(",")} <= names, (k, nbs)


def test_layers_are_the_import_graph():
    found = set()
    for realm in REALMS:
        for f in (PKG / realm).rglob("*.py"):
            found |= {(realm, b) for b in _imported_subpackages(f.read_text(), f.relative_to(PKG).parts, realm)}
    spec = {(a, b) for a, b in dag.LAYER_EDGES if a != "notebooks"}
    assert spec == found, f"in the spec only: {spec - found}; in the code only: {found - spec}"
    assert set(dag.LAYERS) == set(REALMS) | {"notebooks"}
    assert {b for a, b in dag.LAYER_EDGES if a == "notebooks"} == set(REALMS)


def test_mermaid_and_the_readme():
    pipe, layers = dag.dag_to_mermaid("pipeline"), dag.dag_to_mermaid("layers")
    assert pipe.startswith("flowchart LR") and layers.startswith("flowchart TB")
    assert all(f"subgraph {r}_band" in pipe for r in dag.REALMS) and "fake_spectra" not in pipe
    assert "fake_spectra" in dag.dag_to_mermaid("pipeline", external=True)
    assert pipe.count("-.->") == sum(e.kind == "optional" for e in dag.DAG_EDGES)
    with pytest.raises(ValueError, match="kind"):
        dag.dag_to_mermaid("tree")
    tree = dag.module_tree()
    assert "mesh.py" in tree and "dag.py" in tree and "cloudy_tables" not in tree and "__init__.py" not in tree
    out = subprocess.run(
        [sys.executable, str(ROOT / "scripts/dev/make_diagrams.py"), "--check"], capture_output=True, text=True, cwd=ROOT
    )
    assert out.returncode == 0, "README out of date: run python scripts/dev/make_diagrams.py\n" + out.stdout + out.stderr


def test_plot_dag_routes_and_keys():
    nodes = dag.DAG_NODES
    ew, eh = pl._DAG_W / 2 + pl._DAG_PAD, pl._DAG_H / 2 + pl._DAG_PAD

    def on_outline(p, k):
        x, y = nodes[k][:2]
        return (np.isclose(abs(p[0] - x), ew) and abs(p[1] - y) <= eh + 1e-9) or (
            np.isclose(abs(p[1] - y), eh) and abs(p[0] - x) <= ew + 1e-9
        )

    for e in dag.DAG_EDGES + dag.DAG_EXTERNAL:  # every edge starts and ends on the drawn outline of its boxes
        pts = pl._dag_route(e.a, e.b, e.route, nodes, pl._DAG_LANES.get((e.a, e.b), 0.0), pl._DAG_END_DX.get((e.a, e.b), 0.0))
        assert on_outline(pts[0], e.a) and on_outline(pts[-1], e.b), (e.a, e.b, pts)
    with pytest.raises(ValueError, match="unknown ingredients"):
        pl.plot_dag(("stats",))
    fig, ax = pl.plot_dag(("samplers", "field_stats"))
    texts = {t.get_text() for t in ax.texts}
    assert {"core", "simulations", "summary_stats"} <= texts, "the bands, labelled"
    assert "overflux.py" in texts and "skewers.py" in texts, "the module under every name"
    assert pl.check_fonts(fig) == []
