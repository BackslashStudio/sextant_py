"""Keeping windows live at an interactive prompt: the REPL's PyOS_InputHook and
IPython's inputhook."""

import subprocess
import sys
import textwrap
import types

import pytest

import sextant
from sextant import _interactive, _sextant

window = pytest.mark.window


def test_input_hook_install_and_uninstall():
    assert not _sextant._input_hook_installed()
    try:
        assert _sextant._install_input_hook()
        assert _sextant._install_input_hook()  # idempotent
        assert _sextant._input_hook_installed()
    finally:
        _sextant._uninstall_input_hook()
    assert not _sextant._input_hook_installed()


@window
def test_repl_delivers_events_between_lines():
    """`python -i` with piped stdin: the hook runs before each line is read."""
    lines = textwrap.dedent("""
        import sextant
        from sextant import _sextant
        got = []
        fig = sextant.Figure(width=120, height=90)
        fig.connect("close", lambda e: got.append(str(e.kind)))
        fig.show()
        print("hook", _sextant._input_hook_installed(), "open", fig.is_open())
        fig.close()
        print("got", got)
        print("hook", _sextant._input_hook_installed())
    """)
    p = subprocess.run([sys.executable, "-i", "-q"], input=lines, capture_output=True, text=True, timeout=60)
    out = [line.replace(">>> ", "").strip() for line in p.stdout.splitlines()]
    out = [line for line in out if line]
    assert p.returncode == 0, p.stderr
    assert "hook True open True" in out  # show() did not block, and installed the hook
    assert "got ['close']" in out  # delivered while the next line was read
    assert out[-1] == "hook False"  # uninstalled itself once nothing was open


class FakeContext:
    def __init__(self, polls):
        self.polls = polls

    def input_is_ready(self):
        self.polls -= 1
        return self.polls < 0


@window
def test_ipython_inputhook_runs_until_input():
    fig = sextant.Figure(width=100, height=80)
    fig.show(block=False)
    got = []
    fig.connect("close", got.append)
    fig.close()
    ctx = FakeContext(3)
    _interactive.inputhook(ctx)  # delivers, then sees nothing open and returns
    assert got and ctx.polls >= 0


def test_ipython_inputhook_returns_when_input_is_ready():
    _interactive.inputhook(FakeContext(0))


def test_enable_ipython_without_ipython(monkeypatch):
    monkeypatch.delitem(sys.modules, "IPython", raising=False)
    assert _interactive.enable_ipython() is False


@pytest.fixture
def fake_ipython(monkeypatch):
    registered, enabled = {}, []

    class TerminalInteractiveShell:
        active_eventloop = None

        def enable_gui(self, name):
            enabled.append(name)
            self.active_eventloop = name

    shell = TerminalInteractiveShell()
    ipython = types.ModuleType("IPython")
    ipython.get_ipython = lambda: shell
    hooks = types.ModuleType("IPython.terminal.pt_inputhooks")
    hooks.register = lambda name, fn: registered.setdefault(name, fn)
    monkeypatch.setitem(sys.modules, "IPython", ipython)
    monkeypatch.setitem(sys.modules, "IPython.terminal", types.ModuleType("IPython.terminal"))
    monkeypatch.setitem(sys.modules, "IPython.terminal.pt_inputhooks", hooks)
    monkeypatch.setattr(_interactive, "_registered", False)
    return shell, registered, enabled


def test_enable_ipython_registers_and_enables(fake_ipython):
    shell, registered, enabled = fake_ipython
    assert _interactive.enable_ipython() is True
    assert registered == {"sextant": _interactive.inputhook} and enabled == ["sextant"]
    assert _interactive.enable_ipython() is True  # already on: nothing more
    assert enabled == ["sextant"]


def test_enable_ipython_leaves_another_gui_alone(fake_ipython):
    shell, registered, enabled = fake_ipython
    shell.active_eventloop = "qt"
    assert _interactive.enable_ipython() is True
    assert enabled == []


def test_enable_ipython_in_a_kernel_installs_nothing(monkeypatch):
    class ZMQInteractiveShell:
        pass

    ipython = types.ModuleType("IPython")
    ipython.get_ipython = ZMQInteractiveShell
    monkeypatch.setitem(sys.modules, "IPython", ipython)
    assert _interactive.enable_ipython() is True
    assert not _sextant._input_hook_installed()
