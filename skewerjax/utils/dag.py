"""The architecture of SkewerJAX as data: one specification of the dependency graph (DAG) of the ingredients and of the
layers of the package, rendered by ``utils.plotting.plot_dag`` (matplotlib, the notebooks) and by ``dag_to_mermaid``
(the README, written by ``scripts/dev/make_diagrams.py``); ``tests/utils/test_dag.py`` checks the specification
against the code (every module, function and import it names exists; the layers are the import graph) and the README
against the specification.

* ``DAG_NODES`` {key: ``DagNode``}: an ingredient, its place in the figure, its realm (``REALMS``: the background band),
  its module(s) and key functions or classes (``api``);
* ``DAG_EDGES`` (``DagEdge``): a dependency of the code, from an ingredient to what needs it, with the functions that
  implement it (``impl``, "module:name") and the array it carries (``shape``); ``DAG_EXTERNAL``: fake_spectra, the
  external reference (comparison notebooks only); ``DAG_NOTEBOOKS`` {key: the notebooks of an ingredient};
* ``LAYERS``, ``LAYER_EDGES``: the subpackages and who imports whom (core at the bottom; notebooks on top);
* ``dag_to_mermaid("pipeline" | "layers")``, ``module_tree``, ``concept_table``, ``readme_blocks``: the README texts.

Imports nothing from the other subpackages (standard library only).
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import NamedTuple

PACKAGE = Path(__file__).resolve().parents[1]  # skewerjax/


class DagNode(NamedTuple):
    x: float  # column of the figure
    y: float  # row of the figure
    name: str
    detail: str  # mathtext (the figure)
    text: str  # plain text (Mermaid, tables)
    module: str  # the module(s) under skewerjax, comma separated
    realm: str  # REALMS key: the background band; "external" for none
    api: str  # the key functions and classes, comma separated (each defined in one of ``module``)


class DagEdge(NamedTuple):
    a: str
    b: str
    kind: str  # "solid": a required input | "optional": optional input or data only | "external"
    label: str | None  # the label of the figure (mathtext)
    route: str | None  # the route of the figure (``plotting._dag_route``)
    impl: tuple  # "module:name": the functions or classes that implement the dependency
    shape: str | None = None  # the array (or object) the edge carries (Mermaid)


REALMS = {  # realm: (label, PALETTE key of its band)
    "core": ("core", "grey"),
    "simulations": ("simulations", "green"),
    "skewers": ("skewers", "yellow"),
    "summary_stats": ("summary_stats", "blue"),
}

DAG_NODES = {
    "lines": DagNode(0.0, 0.85, "lines", "LineBundle", "LineBundle", "core.lines", "core", "LineBundle, get_line, get_mass_amu"),
    "cosmology": DagNode(0.0, -0.3, "cosmology", "Cosmology, a(x)", "Cosmology, a(x)", "core.cosmology", "core", "Cosmology"),
    "skewers": DagNode(
        1.2,
        -1.4,
        "skewers",
        "SkewerBundle, lattice",
        "SkewerBundle; lattice of paths",
        "core.skewers, skewers.lattice",
        "skewers",
        "SkewerBundle, lattice_columns, lattice_points",
    ),
    "samplers": DagNode(
        1.6,
        1.85,
        "toy fields",
        "PowerLawGRF → cells",
        "PowerLawGRF → cells",
        "simulations.grf, simulations.toy_cells",
        "simulations",
        "PowerLawGRF, grf_cells",
    ),
    "simulation": DagNode(
        2.8,
        1.85,
        "simulation outputs",
        "snapshot → cells",
        "TNG snapshot → cells",
        "simulations.snapshot_cells",
        "simulations",
        "snapshot_cells, select_cells_in_region",
    ),
    "fields": DagNode(2.4, 0.15, "gas cells", "Cells: n, T, v", "Cells: n, T, v", "core.cells", "core", "Cells, cells_on_grid"),
    "deposition": DagNode(
        3.6,
        0.15,
        "deposition",
        "cells → pieces",
        "cells → pieces",
        "core.absorbers, core.deposition",
        "core",
        "cell_pieces, voronoi_segments, absorber_pieces",
    ),
    "profiles": DagNode(
        4.8,
        0.15,
        "line profiles",
        r"Voigt $\phi_j(v)$ per piece",
        "Voigt φ_j(v) per piece",
        "core.line_profiles",
        "core",
        "line_profiles, voigt_widths_GHz",
    ),
    "tau": DagNode(
        6.0,
        0.15,
        "optical depth",
        r"$\tau(v)$, $T=e^{-\tau}$",
        "τ(v), T = e^−τ",
        "core.optical_depth",
        "core",
        "compute_optical_depth, Pieces",
    ),
    "normalisation": DagNode(
        7.2,
        0.15,
        "mean transmission",
        r"$\bar T = \langle T\rangle$",
        "T̄ = ⟨T⟩ over a set of pixels",
        "summary_stats.overflux",
        "summary_stats",
        "ensemble_mean_transmissions, mean_transmissions",
    ),
    "overflux": DagNode(
        8.4,
        0.15,
        "overflux",
        r"$\delta = T/\bar T - 1$",
        "δ = T/T̄ − 1, per line and total",
        "summary_stats.overflux",
        "summary_stats",
        "DeltaFieldBuilder, overflux_check",
    ),
    "stats_1d": DagNode(
        9.6,
        0.7,
        "1D statistics",
        "P1D, ξ1D, by order",
        "P1D, ξ1D (by order and term)",
        "summary_stats.lattice, summary_stats.stats_1d",
        "summary_stats",
        "skewer_p1d, skewer_xi1d, decompose_1d",
    ),
    "stats_3d": DagNode(
        9.6,
        -0.4,
        "3D statistics",
        "P3D, ξ3D on the lattice",
        "P3D, ξ3D on the lattice of skewers",
        "summary_stats.lattice, summary_stats.grid3d",
        "summary_stats",
        "lattice_grid, measure_lattice, GridStatistics3D",
    ),
    "field_stats": DagNode(
        9.6,
        1.85,
        "field statistics",
        "P, ξ: grid or mesh",
        "P, ξ of 3D fields: grid, or mesh (MAS + interlacing)",
        "summary_stats.field_stats, summary_stats.mesh",
        "summary_stats",
        "field_statistics, point_statistics, fourier_mesh",
    ),
    "fake_spectra": DagNode(
        4.0,
        1.85,
        "fake_spectra",
        "external reference",
        "external reference",
        "comparison.source",
        "external",
        "FakeSpectraSource",
    ),
}

DAG_EDGES = (
    DagEdge("cosmology", "skewers", "optional", "a(x) (or manual)", "d", ("core.skewers:SkewerBundle",), "a(x)"),
    DagEdge("lines", "simulation", "optional", "ion masses", "over", ("simulations.gas_physics:Ion", "core.lines:get_mass_amu")),
    DagEdge("samplers", "fields", "solid", None, "d", ("simulations.toy_cells:grf_cells",), "Cells (N³)"),
    DagEdge("simulation", "fields", "solid", None, "d", ("simulations.snapshot_cells:snapshot_cells",), "Cells (N_cells)"),
    DagEdge("fields", "deposition", "solid", None, None, ("core.absorbers:cell_pieces", "core.deposition:voronoi_segments")),
    DagEdge(
        "skewers",
        "deposition",
        "solid",
        "points, a(x)",
        "h",
        ("core.absorbers:bundle_frame", "skewers.lattice:lattice_points"),
        "points (N_sk, N_pt, 3)",
    ),
    DagEdge("deposition", "profiles", "solid", r"T, $v_\parallel$, a", None, ("core.optical_depth:optical_depth_kernel",)),
    DagEdge(
        "deposition",
        "tau",
        "solid",
        r"$n\,\ell/a^2$ of the pieces",
        "u",
        ("core.optical_depth:compute_optical_depth",),
        "Pieces (N_lines, N_sk, N_p)",
    ),
    DagEdge("lines", "profiles", "solid", "atomic data: λ, A₂₁, m", "h", ("core.line_profiles:voigt_widths_GHz",)),
    DagEdge(
        "lines",
        "tau",
        "solid",
        "f, λ, line offsets Δv",
        "h",
        ("core.spectral:frequency_offset_GHz", "core.optical_depth:wrap_to_ref_line"),
    ),
    DagEdge("profiles", "tau", "solid", None, None, ("core.line_profiles:line_profiles",)),
    DagEdge(
        "tau",
        "normalisation",
        "solid",
        None,
        None,
        ("summary_stats.overflux:ensemble_mean_transmissions",),
        "τ (N_lines, N_sk, N_v)",
    ),
    DagEdge(
        "normalisation",
        "overflux",
        "solid",
        None,
        None,
        ("summary_stats.overflux:DeltaFieldBuilder",),
        "T̄ (N_lines + 1, N_sk, N_v)",
    ),
    DagEdge(
        "overflux",
        "stats_1d",
        "solid",
        None,
        "s",
        ("summary_stats.lattice:skewer_p1d", "summary_stats.lattice:skewer_xi1d", "summary_stats.stats_1d:decompose_1d"),
        "δ (N_lines + 1, N_sk, N_v)",
    ),
    DagEdge(
        "overflux",
        "stats_3d",
        "solid",
        None,
        "s",
        ("summary_stats.lattice:measure_lattice",),
        "δ on (n_x, N_v, n_z)",
    ),
    DagEdge(
        "samplers",
        "field_stats",
        "solid",
        "on its grid (exact)",
        "over",
        ("summary_stats.field_stats:field_statistics", "summary_stats.grid3d:GridStatistics3D"),
        "δ (N, N, N)",
    ),
)
DAG_EXTERNAL = (
    DagEdge("simulation", "fake_spectra", "external", None, None, ("comparison.source:FakeSpectraSource",)),
    DagEdge("fake_spectra", "tau", "external", "compared", "h", ("comparison.emulation:matched_tau",)),
)
DAG_NOTEBOOKS = {
    "lines": "01",
    "cosmology": "02",
    "skewers": "03, 09, 10",
    "samplers": "04, 08",
    "fields": "04, 05, 10",
    "simulation": "05, 10",
    "profiles": "06",
    "deposition": "07",
    "tau": "07, 09, 10",
    "normalisation": "09, 10",
    "overflux": "09, 10",
    "stats_1d": "09, 10",
    "stats_3d": "09, 10",
    "field_stats": "08, 10",
    "fake_spectra": "11, 12",
}

# The layers: the subpackages and the notebooks, and who imports whom (``tests/test_import_graph.py``)
LAYERS = {
    "notebooks": "notebooks/: compose the realms",
    "utils": "utils: plotting, dag, paths",
    "comparison": "comparison: fake_spectra",
    "simulations": "simulations: the gas",
    "skewers": "skewers: the paths",
    "summary_stats": "summary_stats: statistics from arrays",
    "core": "core: lines, cosmology, cells, τ",
}
LAYER_EDGES = (  # (importer, imported)
    ("notebooks", "utils"),
    ("notebooks", "comparison"),
    ("notebooks", "simulations"),
    ("notebooks", "skewers"),
    ("notebooks", "summary_stats"),
    ("notebooks", "core"),
    ("utils", "comparison"),
    ("utils", "simulations"),
    ("utils", "summary_stats"),
    ("utils", "core"),
    ("comparison", "simulations"),
    ("comparison", "summary_stats"),
    ("comparison", "core"),
    ("simulations", "core"),
    ("skewers", "core"),
)


# ---------------------------------------------------------------------------------------------------- Mermaid
def _mm(text: str) -> str:
    """Text for a Mermaid label (quoted: no double quotes, no backticks inside)."""
    return text.replace('"', "'").replace("`", "")


def dag_to_mermaid(kind: str = "pipeline", *, external: bool = False) -> str:
    """A Mermaid flowchart of the specification: ``"pipeline"`` (the ingredients as boxes, name, plain detail and module,
    grouped in the bands of their realm; the edges labelled with the functions that implement them and the array they
    carry; optional edges dotted; ``external`` adds fake_spectra) or ``"layers"`` (the subpackages, core at the bottom,
    an arrow from importer to imported)."""
    if kind == "layers":
        out = ["flowchart TB"]
        out += [f'    {k}["{_mm(v)}"]' for k, v in LAYERS.items()]
        out += [f"    {a} --> {b}" for a, b in LAYER_EDGES]
        return "\n".join(out)
    if kind != "pipeline":
        raise ValueError(f"kind must be 'pipeline' or 'layers', got {kind!r}.")
    nodes = {k: v for k, v in DAG_NODES.items() if external or v.realm != "external"}
    out = ["flowchart LR"]

    def box(k, n):
        return f'{k}["<b>{_mm(n.name)}</b><br/>{_mm(n.text)}<br/><code>{n.module}</code>"]'

    for realm, (label, _) in REALMS.items():
        members = [k for k, n in nodes.items() if n.realm == realm]
        out.append(f'    subgraph {realm}_band["{label}"]')
        out += [f"        {box(k, nodes[k])}" for k in members]
        out.append("    end")
    out += [f"    {box(k, n)}" for k, n in nodes.items() if n.realm not in REALMS]
    edges = list(DAG_EDGES) + (list(DAG_EXTERNAL) if external else [])
    for e in edges:
        if e.a not in nodes or e.b not in nodes:
            continue
        funcs = ", ".join(i.split(":")[1] for i in e.impl)
        lab = _mm(funcs + (f"<br/>{e.shape}" if e.shape else ""))
        arrow = "-.->" if e.kind in ("optional", "external") else "-->"
        out.append(f'    {e.a} {arrow}|"{lab}"| {e.b}')
    return "\n".join(out)


# ---------------------------------------------------------------------------------------------------- the module tree
_SKIP = {"__pycache__", "cloudy_tables"}  # not modules of the package (cloudy_tables: a git submodule of data)


def _first_line(path: Path, width: int = 84) -> str:
    """The first sentence of a module's docstring (its first line; a sentence ends at ". " before a capital, not after
    an abbreviation), cut at a word to ``width`` characters (empty if none)."""
    import re

    try:
        doc = ast.get_docstring(ast.parse(path.read_text()))
    except SyntaxError:
        return ""
    line = doc.strip().splitlines()[0] if doc else ""
    m = re.search(r"(?<!\bal)(?<!\bEqs)(?<!\bEq)(?<!\be\.g)\.\s+(?=[A-Z])", line)
    line = line[: m.start()] if m else line.rstrip(".")
    return line if len(line) <= width else line[: width - 1].rsplit(" ", 1)[0].rstrip(",;:(") + " …"


def module_tree(root: Path = PACKAGE) -> str:
    """The tree of the package's modules (``root``: skewerjax/), each with the first sentence of its docstring (a folder:
    that of its ``__init__.py``)."""
    rows = [f"{root.name}/".ljust(30) + f"# {_first_line(root / '__init__.py')}"]

    def walk(d: Path, indent: str):
        kids = sorted(
            p
            for p in d.iterdir()
            if p.name not in _SKIP and p.name != "__init__.py" and (p.suffix == ".py" or (p / "__init__.py").exists())
        )
        for i, p in enumerate(kids):
            last = i == len(kids) - 1
            branch = "└── " if last else "├── "
            if p.is_dir():
                rows.append(f"{indent}{branch}{p.name}/".ljust(30) + f"# {_first_line(p / '__init__.py')}")
                walk(p, indent + ("    " if last else "│   "))
            else:
                rows.append(f"{indent}{branch}{p.name}".ljust(30) + f"# {_first_line(p)}")

    walk(root, "")
    return "\n".join(r.rstrip() for r in rows)


def concept_table() -> str:
    """A Markdown table: concept (an ingredient of the DAG) → module(s) → key functions and classes → notebooks."""
    rows = ["| Concept | Module | Key functions and classes | Notebooks |", "|---|---|---|---|"]
    for k, n in DAG_NODES.items():
        mods = ", ".join(f"`{m.strip()}`" for m in n.module.split(","))
        api = ", ".join(f"`{a.strip()}`" for a in n.api.split(","))
        rows.append(f"| {n.name}: {n.text} | {mods} | {api} | {DAG_NOTEBOOKS.get(k, '—')} |")
    return "\n".join(rows)


def readme_blocks() -> dict:
    """The generated blocks of the README "Architecture" section, {marker: text} (``scripts/dev/make_diagrams.py``)."""
    return {
        "layers": "```mermaid\n" + dag_to_mermaid("layers") + "\n```",
        "pipeline": "```mermaid\n" + dag_to_mermaid("pipeline") + "\n```",
        "tree": "```\n" + module_tree() + "\n```",
        "concepts": concept_table(),
    }
