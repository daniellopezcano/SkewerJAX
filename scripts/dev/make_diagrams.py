"""Write the generated blocks of the README's "Architecture" section from ``skewerjax.utils.dag`` (the one specification
of the DAG and of the layers): the layer and pipeline diagrams (Mermaid), the module tree and the table of concepts.

Each block sits between ``<!-- BEGIN generated:<name> -->`` and ``<!-- END generated:<name> -->``; the text between the
markers is replaced, nothing else. ``--check`` only reports whether the README is up to date (exit 1 if not), as
``tests/utils/test_dag.py`` does.

    python scripts/dev/make_diagrams.py [--check] [README.md]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from skewerjax.utils.dag import readme_blocks

ROOT = Path(__file__).resolve().parents[2]


def render(text: str) -> str:
    """``text`` with every generated block replaced by the current one (a missing marker pair raises)."""
    for name, block in readme_blocks().items():
        pat = re.compile(rf"(<!-- BEGIN generated:{name} -->\n)(.*?)(<!-- END generated:{name} -->)", re.S)
        if not pat.search(text):
            raise ValueError(f"the README has no markers for the generated block {name!r}.")
        text = pat.sub(lambda m, b=block: m.group(1) + b + "\n" + m.group(3), text)
    return text


def main(argv) -> int:
    check = "--check" in argv
    paths = [a for a in argv if not a.startswith("--")]
    path = Path(paths[0]) if paths else ROOT / "README.md"
    old = path.read_text()
    new = render(old)
    if check:
        print("README up to date" if new == old else "README out of date: run scripts/dev/make_diagrams.py")
        return int(new != old)
    if new != old:
        path.write_text(new)
    print(f"{path}: {'updated' if new != old else 'unchanged'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
