"""Markdown-math checker for the tutorial notebooks (Jupyter / MathJax rendering).

    python check_markdown_math.py NB.ipynb [...]     (exit status 1 if any issue; one line per issue)

Checks every markdown cell, outside code spans (`...`) and fenced code:
  1. balanced $ and $$ (escaped \\$ ignored);
  2. no blank line inside a $$ ... $$ block;
  3. display math on its own lines: the opening $$ starts a line, the closing $$ ends one, with a blank line (or the
     cell boundary) before and after the block;
  4. macros: every \\macro in math must be in SUPPORTED (MathJax 3 default TeX + AMS, as in JupyterLab / Notebook 7);
     known package-only macros (physics, siunitx, mathtools, ...) are reported by name;
  5. no '|' inside $...$ in a markdown table row (it splits the cell): use \\vert / \\lvert ... \\rvert;
  6. no formula written in Unicode or plain text outside math: Greek letters and math symbols (≈ ∝ ≤ ⟨ ⟩ √ ∑ ...),
     superscript / subscript digits, combining accents (F̄), the Unicode minus, and plain-text notation such as x_i,
     e^{...}, 10^5 or "N = 16";
  7. no LaTeX macro outside math (a bare \\alpha, \\mathrm{...}, \\frac, ...: it renders as text);
  8. no Greek letter or symbol spelled out where the symbol is meant (alpha, sigma, Delta, ... as words outside math
     and code), and one form for the Lyman lines: $\\mathrm{Ly}\\alpha$, $\\mathrm{Ly}\\beta$ (not Ly$\\alpha$, Lyα,
     Lya or Ly-alpha in the text).
"""
from __future__ import annotations

import json
import re
import sys

SUPPORTED = set("""
alpha beta gamma delta epsilon varepsilon zeta eta theta vartheta iota kappa lambda mu nu xi pi varpi rho varrho sigma
varsigma tau upsilon phi varphi chi psi omega Gamma Delta Theta Lambda Xi Pi Sigma Upsilon Phi Psi Omega
frac tfrac dfrac sqrt left right big Big bigg Bigg bigl bigr Bigl Bigr biggl biggr middle sum prod int iint oint lim limsup liminf max min
sup inf exp log ln sin cos tan sinh cosh tanh arcsin arccos arctan det dim mod bmod pmod
cdot cdots ldots dots vdots ddots times div pm mp ast star circ bullet oplus otimes
le leq ge geq ll gg ne neq approx simeq sim cong equiv propto in notin subset subseteq supset cup cap setminus
to rightarrow leftarrow Rightarrow Leftarrow leftrightarrow Leftrightarrow mapsto longrightarrow longmapsto uparrow downarrow
infty partial nabla prime dagger ddagger ell hbar emptyset forall exists neg land lor parallel perp mid vert Vert lvert rvert lVert rVert
langle rangle lfloor rfloor lceil rceil lbrace rbrace
hat bar tilde vec dot ddot widehat widetilde overline underline overbrace underbrace overrightarrow
mathrm mathbf mathit mathcal mathbb mathsf mathtt boldsymbol rm bf it text textrm textit textbf operatorname
quad qquad enspace thinspace negthinspace
begin end array aligned align cases matrix pmatrix bmatrix gathered split
displaystyle textstyle scriptstyle limits nolimits
underset overset stackrel substack color boxed tag nonumber notag
Re Im backslash setminus
""".split())
PACKAGE_ONLY = {"bm", "si", "SI", "num", "unit", "qty", "dd", "dv", "pdv", "abs", "norm", "expval", "coloneqq", "eqqcolon",
                "mathbbm", "textsc", "label", "eqref", "ref", "cite", "ce", "vb", "vu", "order", "tr", "Tr"}

GREEK = "αβγδεζηθικλμνξοπρστυφχψωϕϑϵΓΔΘΛΞΠΣΦΨΩ"
SYMBOLS = "≈≃≅∝≤≥≪≫±∓×·⟨⟩√∑∏∫∂∇∞→←↔⇒∈∉⊂⊥∥‖†′″ℓ−⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺₀₁₂₃₄₅₆₇₈₉₋ᵢⱼₖₚ"
PLAIN = [
    (re.compile(r"(?<![\w\\`$])[A-Za-z]{1,3}_\{?[A-Za-z0-9†]+"), "plain subscript"),
    (re.compile(r"\^\{|\b\d+\^-?\d|\b[A-Za-z)]\^-?[\w(]"), "plain superscript"),
    (re.compile(r"(?<![\w$`])[A-Za-z](?:\([^)\s]*\))?\s?[=<>]\s?[-+(]?[\w.]"), "plain equation"),
]


SPELLED = re.compile(r"(?<![\w\\$`-])(alpha|beta|gamma|delta|epsilon|zeta|theta|kappa|lambda|mu|nu|xi|pi|rho|sigma|tau|"
                     r"phi|chi|psi|omega|Gamma|Delta|Theta|Lambda|Sigma|Phi|Psi|Omega|dagger)(?![\w$`-])")
LYMAN = re.compile(r"Ly\$\\(alpha|beta)\$|Ly[αβ]|\bLy-?(?:a|b|alpha|beta)\b|\bLyman-?(?:alpha|beta)\b")


def _strip_code(text: str) -> str:
    text = re.sub(r"```.*?```", lambda m: " " * len(m.group()), text, flags=re.S)
    return re.sub(r"`[^`\n]*`", lambda m: " " * len(m.group()), text)


def _math_spans(text: str):
    """[(start, end, is_display, body)] of $$...$$ and $...$ (escaped dollars removed), plus balance problems."""
    t = text.replace(r"\$", "  ")
    spans, issues, i = [], [], 0
    while i < len(t):
        if t.startswith("$$", i):
            j = t.find("$$", i + 2)
            if j < 0:
                issues.append("unbalanced $$")
                break
            spans.append((i, j + 2, True, t[i + 2:j]))
            i = j + 2
        elif t[i] == "$":
            j = t.find("$", i + 1)
            if j < 0 or "\n\n" in t[i:j]:
                issues.append(f"unbalanced $ near {t[i:i + 40]!r}")
                break
            spans.append((i, j + 1, False, t[i + 1:j]))
            i = j + 1
        else:
            i += 1
    return spans, issues


def check_markdown(src: str) -> list:
    text = _strip_code(src)
    spans, issues = _math_spans(text)
    for a, b, disp, body in spans:
        for m in re.finditer(r"\\([A-Za-z]+)", body):
            name = m.group(1)
            if name in PACKAGE_ONLY:
                issues.append(f"unsupported macro \\{name} (package only)")
            elif name not in SUPPORTED:
                issues.append(f"unknown macro \\{name}")
        if disp:
            if re.search(r"\n[ \t]*\n", body):
                issues.append("blank line inside $$...$$")
            line_start = text.rfind("\n", 0, a) + 1
            line_end = text.find("\n", b)
            line_end = len(text) if line_end < 0 else line_end
            if text[line_start:a].strip() or text[b:line_end].strip():
                issues.append(f"display math not on its own lines: {text[line_start:min(line_end, line_start + 60)]!r}")
            before, after = text[:line_start], text[line_end:]
            if before.strip() and not re.search(r"\n[ \t]*\n$", before):
                issues.append("display math not separated from the text before it (blank line)")
            if after.strip() and not re.match(r"\n[ \t]*\n", after):
                issues.append("display math not separated from the text after it (blank line)")
    for line in text.splitlines():
        if line.lstrip().startswith("|"):
            ls, _ = _math_spans(line)
            for a, b, disp, body in ls:
                if "|" in body:
                    issues.append(f"'|' inside math in a table row: {body[:40]!r}")
    # outside math
    out = list(text)
    for a, b, *_ in spans:
        out[a:b] = " " * (b - a)
    plain = re.sub(r"(?<=\S) · (?=\S)", "   ", "".join(out))        # " · " is the heading separator, not a product
    for ch in sorted({c for c in plain if c in GREEK or c in SYMBOLS or 0x0300 <= ord(c) <= 0x036F}):
        k = plain.index(ch)
        issues.append(f"Unicode math {ch!r} outside LaTeX: {plain[max(0, k - 25):k + 15].strip()!r}")
    for rx, what in PLAIN:
        for m in rx.finditer(plain):
            issues.append(f"{what} outside LaTeX: {plain[max(0, m.start() - 15):m.end() + 10].strip()!r}")
    plain_links = re.sub(r"\]\([^)]*\)", "]", plain)                 # link targets are URLs, not text
    for m in re.finditer(r"\\[A-Za-z]+", plain_links):
        issues.append(f"LaTeX macro outside math: {plain_links[max(0, m.start() - 15):m.end() + 10].strip()!r}")
    for m in SPELLED.finditer(plain_links):
        issues.append(f"spelled-out symbol {m.group(1)!r} (write it in LaTeX): {plain_links[max(0, m.start() - 20):m.end() + 10].strip()!r}")
    for m in LYMAN.finditer(text.replace(r"$\mathrm{Ly}\alpha$", "").replace(r"$\mathrm{Ly}\beta$", "")):
        issues.append(f"Lyman line not written as $\\mathrm{{Ly}}\\alpha$: {m.group()!r}")
    return issues


def check_notebook(path: str) -> list:
    nb = json.load(open(path))
    out = []
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] == "markdown":
            out += [(i, s) for s in check_markdown("".join(c["source"]))]
    return out


if __name__ == "__main__":
    bad = 0
    for f in sys.argv[1:]:
        issues = check_notebook(f)
        bad += len(issues)
        print(f"{f.split('/')[-1]}: {len(issues)} issue(s)")
        for i, s in issues:
            print(f"  cell {i}: {s}")
    sys.exit(1 if bad else 0)
