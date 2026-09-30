"""The type stubs: current with the binding, and right about what they accept.

Both need the source tree (tools/, src/): skipped when testing an installed wheel.
"""

import os
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
GEN = ROOT / "tools" / "gen_stubs.py"
SRC = ROOT / "src"

pytestmark = pytest.mark.skipif(not GEN.exists() or not (SRC / "sextant").exists(),
                                reason="needs the source tree")


def test_stub_is_current():
    # A binding change (signature, option field, enum name) without a
    # regenerated stub fails here.
    r = subprocess.run([sys.executable, str(GEN), "--check"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_stub_accepts_and_rejects(tmp_path):
    # typing_usage.py marks every call the stubs must reject with
    # `# type: ignore[code]`; --warn-unused-ignores fails on one that passes.
    pytest.importorskip("mypy")
    env = dict(os.environ, MYPYPATH=str(SRC))
    r = subprocess.run(
        [sys.executable, "-m", "mypy", "--strict", "--warn-unused-ignores", "--follow-imports=silent",
         "--cache-dir", str(tmp_path / "mypy_cache"), str(ROOT / "tests" / "typing_usage.py")],
        capture_output=True, text=True, env=env, cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr
