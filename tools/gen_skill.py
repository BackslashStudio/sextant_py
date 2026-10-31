"""Fill the signature blocks of the agent skill from this package's type stub.

    python tools/gen_skill.py            # rewrite the blocks
    python tools/gen_skill.py --check    # exit 1 if one is out of date

The skill (plugins/sextant/skills/sextant/) lives in the Agent-Skills repo: $AGENT_SKILLS, else
../Agent-Skills next to this checkout. Run it at every release, so the skill's
calls match the stub that ships.

A block in a skill markdown file names the calls it lists; everything between
the markers is generated, the prose around it is written by hand:

    <!-- sig: Axes.line Axes.scatter -->
    ```
    line(x, y=None, *, err=None, **opts)    color, linewidth, linestyle, ...
    ```
    <!-- /sig -->

A line is the call as its docstring's first line writes it (else as the stub
declares it, without annotations), `-> <type>` when it returns something other
than its own class (the chaining calls return self), then the keywords of its
options struct. The keywords nearly every kind takes (COMMON) are left out:
each file says so once in its prose. An overloaded call gets a line per
overload. The stub is read with `ast`, not imported, so this runs without a
built module.
"""

import ast
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUB = ROOT / "src" / "sextant" / "_sextant.pyi"
AGENT_SKILLS = Path(os.environ.get("AGENT_SKILLS", ROOT.parent / "Agent-Skills")).resolve()
SKILL = AGENT_SKILLS / "plugins" / "sextant" / "skills" / "sextant"

COMMON = ("name", "show_legend", "visible", "zorder", "hint_labels")
WIDE = 44              # the widest call that keeps its keywords on its own line
BLOCK = re.compile(r"(<!-- sig: (?P<names>[^>]*?) -->\n)```\n.*?```\n(<!-- /sig -->)", re.S)


def load_stub(path=STUB):
    """{class: {method: [FunctionDef, ...]}} and {TypedDict: [field, ...]}, bases resolved."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    classes, typed, bases = {}, {}, {}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        base_names = [ast.unparse(b) for b in node.bases]
        if "TypedDict" in base_names or any(b in typed for b in base_names):
            typed[node.name] = [s.target.id for s in node.body if isinstance(s, ast.AnnAssign)]
            bases[node.name] = [b for b in base_names if b != "TypedDict"]
            continue
        methods = classes.setdefault(node.name, {})
        for f in node.body:
            if isinstance(f, ast.FunctionDef):
                methods.setdefault(f.name, []).append(f)

    def fields(name):
        out = []
        for b in bases.get(name, []):
            out += [f for f in fields(b) if f not in out]
        return out + [f for f in typed.get(name, []) if f not in out]

    return classes, {name: fields(name) for name in typed}


def _default(node):
    # An enum default (sextant._enums.Coords.DATA) is written as the string it stands for.
    text = ast.unparse(node)
    m = re.fullmatch(r"sextant\._enums\.\w+\.(\w+)", text)
    return repr(m.group(1).lower()) if m else text


def _declared(f):
    a = f.args
    params = a.posonlyargs + a.args
    defaults = [None] * (len(params) - len(a.defaults)) + list(a.defaults)
    out = [p.arg if d is None else f"{p.arg}={_default(d)}" for p, d in zip(params, defaults)]
    out = out[1:]                                            # self
    if a.vararg:
        out.append("*" + a.vararg.arg)
    elif a.kwonlyargs:
        out.append("*")
    out += [p.arg if d is None else f"{p.arg}={_default(d)}" for p, d in zip(a.kwonlyargs, a.kw_defaults)]
    if a.kwarg:
        out.append("**" + a.kwarg.arg)
    return f"{f.name}({', '.join(out)})"


def call_text(f):
    doc = ast.get_docstring(f) or ""
    first = doc.splitlines()[0].strip() if doc else ""
    return first if first.startswith(f.name + "(") and first.endswith(")") else _declared(f)


def keywords(f, typed):
    kw = f.args.kwarg
    if kw is None or kw.annotation is None:
        return []
    m = re.fullmatch(r"Unpack\[(\w+)\]", ast.unparse(kw.annotation))
    if not m:
        return []
    return [k for k in typed[m.group(1)] if k not in COMMON]


def lines_for(qualname, classes, typed):
    cls, _, method = qualname.partition(".")
    overloads = classes.get(cls, {}).get(method)
    if not overloads:
        raise KeyError(f"{qualname}: not in the stub")
    rows = []
    for f in overloads:
        left = call_text(f)
        ret = ast.unparse(f.returns) if f.returns is not None else "None"
        if ret not in (cls, "None"):
            left += f" -> {ret}"
        rows.append((left, ", ".join(keywords(f, typed))))
    return rows


def render(names, classes, typed):
    rows = [r for q in names.split() for r in lines_for(q, classes, typed)]
    # A keyword list already given in the block is named, not repeated.
    seen, shown = {}, []
    for left, kw in rows:
        call = left.partition("(")[0]
        if kw and kw in seen:
            kw = "(as above)" if seen[kw] == call else f"(as {seen[kw]})"
        elif kw:
            seen[kw] = call
        shown.append((left, kw))
    # Keywords line up in one column; a call too long for it puts them on the next line.
    width = min(max(len(left) for left, _ in shown), WIDE) + 2
    out = []
    for left, kw in shown:
        if kw and len(left) + 2 > width:
            out += [left, " " * width + kw]
        else:
            out.append((left.ljust(width) + kw).rstrip())
    return "".join(line + "\n" for line in out)


def regenerate(text, classes, typed):
    def sub(m):
        return f"{m.group(1)}```\n{render(m.group('names'), classes, typed)}```\n{m.group(3)}"
    return BLOCK.sub(sub, text)


def skill_files(root=SKILL):
    return sorted(root.rglob("*.md"))


def main(argv):
    check = "--check" in argv
    if not SKILL.is_dir():
        print(f"no agent skill at {SKILL} (set AGENT_SKILLS)")
        return 2
    classes, typed = load_stub()
    stale = []
    for path in skill_files():
        old = path.read_text(encoding="utf-8")
        new = regenerate(old, classes, typed)
        if new == old:
            continue
        stale.append(path.relative_to(AGENT_SKILLS).as_posix())
        if not check:
            path.write_text(new, encoding="utf-8", newline="\n")
    if check and stale:
        print("out of date (run tools/gen_skill.py):", *stale, sep="\n  ")
        return 1
    for p in stale:
        print("wrote", p)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
