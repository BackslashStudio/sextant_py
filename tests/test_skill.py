"""The agent skill (skills/sextant/): current with the stub, and right about the API.

An agent copies what the skill shows, so a renamed call or a broken example
teaches it the wrong thing. These need the source tree (skills/, tools/):
skipped when testing an installed wheel.
"""

import ast
import builtins
import json
import pathlib
import re
import tomllib

import numpy as np
import pytest

import sextant
from _subproc import run_python

ROOT = pathlib.Path(__file__).resolve().parents[1]
SKILL = ROOT / "skills" / "sextant"
GEN = ROOT / "tools" / "gen_skill.py"

pytestmark = pytest.mark.skipif(not SKILL.exists() or not GEN.exists(), reason="needs the source tree")

# What an agent loads: SKILL.md always, a reference file when the task needs it.
SKILL_MAX = 10_000
REFERENCE_MAX = 12_000

FENCE = re.compile(r"^```python\n(.*?)^```", re.S | re.M)
CALL = re.compile(r"\b([A-Za-z_]\w*)\.(\w+)\(")
BARE = re.compile(r"(?<![\w.])([A-Za-z_]\w*)\(")
STRING = re.compile(r"r?\"[^\"]*\"|r?'[^']*'")
# Receivers that are not sextant objects: numpy, the standard library, matplotlib (named to contrast).
FOREIGN = {"np", "numpy", "io", "os", "m", "math", "random", "rng", "pathlib", "Path", "sys", "plt"}


def markdown_files():
    return sorted(SKILL.rglob("*.md"))


def examples():
    return sorted((SKILL / "examples").glob("*.py"))


def frontmatter(text):
    m = re.match(r"---\n(.*?)\n---\n", text, re.S)
    assert m, "SKILL.md starts with a --- frontmatter block"
    return dict(line.split(": ", 1) for line in m.group(1).splitlines())


def test_signature_blocks_are_current():
    # A stub change (signature, option field) without tools/gen_skill.py fails here.
    r = run_python([str(GEN), "--check"])
    assert r.returncode == 0, r.stdout + r.stderr


def test_frontmatter():
    meta = frontmatter((SKILL / "SKILL.md").read_text(encoding="utf-8"))
    assert meta.get("name") == SKILL.name
    # The Agent Skills format: name lowercase letters, digits and '-' (<= 64), description <= 1024.
    assert re.fullmatch(r"[a-z0-9-]{1,64}", meta["name"])
    assert 0 < len(meta.get("description", "")) <= 1024


def test_states_the_package_version():
    version = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    minor = ".".join(version.split(".")[:2])
    text = (SKILL / "SKILL.md").read_text(encoding="utf-8")
    assert f"**pysextant {minor}**" in text, f"SKILL.md should say it is written for pysextant {minor}"


def test_plugin_manifests():
    # Claude Code installs the plugin at its manifest version, so it moves with the package.
    version = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    plugin_root = SKILL.parent
    plugin = json.loads((plugin_root / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8"))
    assert plugin["name"] == SKILL.name
    assert plugin["version"] == version
    assert plugin["skills"] == ["./"]
    market = json.loads((ROOT / ".claude-plugin" / "marketplace.json").read_text(encoding="utf-8"))
    (entry,) = [p for p in market["plugins"] if p["name"] == plugin["name"]]
    assert (ROOT / entry["source"]).resolve() == plugin_root
    assert "version" not in entry, "the version lives in plugin.json only"


def test_sizes():
    assert len((SKILL / "SKILL.md").read_bytes()) <= SKILL_MAX
    for p in (SKILL / "references").glob("*.md"):
        assert len(p.read_bytes()) <= REFERENCE_MAX, p.name


def test_every_file_is_named_in_skill_md():
    # A reference or example SKILL.md does not name is never found.
    text = (SKILL / "SKILL.md").read_text(encoding="utf-8")
    for p in (SKILL / "references").glob("*.md"):
        assert f"](references/{p.name})" in text, p.name
    for p in examples():
        assert f"`{p.name}`" in text, p.name
    for link in re.findall(r"\]\(([^)#]+)\)", "".join(p.read_text(encoding="utf-8") for p in markdown_files())):
        if "://" not in link:
            assert (SKILL / link).exists() or any((p.parent / link).exists() for p in markdown_files()), link


def _shown(path):
    """(code, inline): what a file shows. An example is all code; markdown has its python
    blocks as code and its `inline code` as inline, less the first column of a table headed
    'matplotlib' (calls named to contrast) and the string literals (math, labels)."""
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".py":
        return text, ""
    code = "\n".join(FENCE.findall(text))
    inline, contrast = [], False
    for line in FENCE.sub("", text).splitlines():
        if not line.startswith("|"):
            contrast = False
            inline += re.findall(r"`([^`]+)`", line)
            continue
        if re.fullmatch(r"[-:| ]+", line):
            continue
        cells = line.strip("|").split("|")
        if "matplotlib" in cells[0] and not contrast:
            contrast = True
            continue
        inline += re.findall(r"`([^`]+)`", "|".join(cells[1:] if contrast else cells))
    return code, STRING.sub('""', "\n".join(inline))


def test_named_calls_exist():
    # receiver.name( in what the skill shows: a sextant method of some kind of axes or the
    # figure, or a sextant attribute; numpy array methods (Z.min()) pass. A bare name( in
    # inline code (`fill(x, y1, y2)`) is a method, a sextant attribute or a builtin. So a call
    # the skill names as absent goes in plain text or in a matplotlib column, not in backticks.
    kinds = (sextant.Figure, sextant.Axes, sextant.Axes3D, sextant.PolarAxes, sextant.Plane2D)
    methods = set().union(*(dir(k) for k in kinds))
    missing = []
    for path in markdown_files() + examples():
        code, inline = _shown(path)
        where = path.relative_to(SKILL).as_posix()
        for recv, name in CALL.findall(code + "\n" + inline):
            if recv in FOREIGN:
                continue
            if recv == "sextant":
                ok = hasattr(sextant, name)
            elif hasattr(sextant, recv):
                ok = hasattr(getattr(sextant, recv), name)
            else:
                ok = name in methods or hasattr(np.ndarray, name)
            if not ok:
                missing.append(f"{where}: {recv}.{name}(")
        for name in BARE.findall(inline):
            if name not in methods and not hasattr(sextant, name) and not hasattr(builtins, name):
                missing.append(f"{where}: {name}(")
    assert not missing, "\n".join(missing)


def test_skill_md_examples_run(tmp_path):
    # SKILL.md's python blocks are whole scripts; each writes a PNG.
    blocks = FENCE.findall((SKILL / "SKILL.md").read_text(encoding="utf-8"))
    assert blocks
    for i, code in enumerate(blocks):
        ast.parse(code)
        script = tmp_path / f"skill_{i}.py"
        script.write_text(code, encoding="utf-8")
        before = set(tmp_path.glob("*.png"))
        r = run_python(["-W", "error", str(script)], cwd=tmp_path)
        assert r.returncode == 0, f"block {i}:\n{r.stdout}{r.stderr}"
        new = set(tmp_path.glob("*.png")) - before
        assert new and all(p.stat().st_size > 0 for p in new), f"block {i} wrote no PNG"


@pytest.mark.parametrize("script", examples(), ids=lambda p: p.stem)
def test_example_runs(script, tmp_path):
    r = run_python(["-W", "error", str(script)], cwd=tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    png = tmp_path / f"{script.stem}.png"
    assert png.exists() and png.stat().st_size > 0
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
