"""Ownership, locking, blocking waits and shutdown."""

import _thread
import gc
import os
import sys
import textwrap
import threading
import time

import numpy as np
import pytest
from _subproc import run_python

import sextant

window = pytest.mark.window


def run_script(code):
    """Run code in a fresh interpreter; return (returncode, stdout, stderr, seconds)."""
    start = time.monotonic()
    p = run_python(["-c", textwrap.dedent(code)])
    return p.returncode, p.stdout, p.stderr, time.monotonic() - start


def close_later(fig, delay=0.3):
    t = threading.Timer(delay, fig.close)
    t.start()
    return t


# --- ownership ---------------------------------------------------------------

def test_axes_wrapper_is_reused():
    fig = sextant.Figure()
    assert fig.axes() is fig.axes()


def test_axes_keeps_its_figure_alive():
    ax = sextant.Figure().axes()
    gc.collect()
    ax.line(np.arange(3.0))
    assert ax.line_count() == 1


def test_concurrent_calls_on_one_figure_are_serialised():
    fig = sextant.Figure()
    ax = fig.axes()
    y = np.arange(100.0)

    def plot():
        for _ in range(200):
            ax.line(y)

    threads = [threading.Thread(target=plot) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert ax.line_count() == 800


# --- errors ------------------------------------------------------------------

def test_file_error_is_oserror_subclass(tmp_path):
    fig = sextant.Figure()
    fig.axes().line(np.arange(3.0))
    with pytest.raises(FileNotFoundError):
        fig.savefig(str(tmp_path / "no_such_dir" / "out.png"))


# --- diagnostics -------------------------------------------------------------

@pytest.mark.parametrize("other_thread", [False, True])
def test_messages_become_runtime_warnings(other_thread):
    with pytest.warns(RuntimeWarning, match="hello from sextant"):
        sextant._sextant._emit_message("hello from sextant", other_thread)


@pytest.mark.parametrize("other_thread", [False, True])
def test_custom_message_handler(other_thread):
    got = []
    handler = got.append
    prev = sextant.set_message_handler(handler)
    try:
        assert prev is None
        sextant._sextant._emit_message("to the handler", other_thread)
    finally:
        assert sextant.set_message_handler(prev) is handler
    assert got == ["to the handler"]


def test_set_message_handler_returns_previous():
    def h(msg):
        pass

    assert sextant.set_message_handler(h) is None
    assert sextant.set_message_handler(None) is h


def test_raising_handler_goes_to_unraisablehook(monkeypatch):
    seen = []
    monkeypatch.setattr(sys, "unraisablehook", seen.append)

    def bad(msg):
        raise ValueError("boom")

    sextant.set_message_handler(bad)
    try:
        sextant._sextant._emit_message("x", True)
    finally:
        sextant.set_message_handler(None)
    assert len(seen) == 1 and isinstance(seen[0].exc_value, ValueError)


def test_handler_must_be_callable():
    with pytest.raises(TypeError):
        sextant.set_message_handler(42)


# --- windows -----------------------------------------------------------------

@window
def test_wait_closed_times_out_then_sees_close():
    fig = sextant.Figure(width=200, height=150)
    fig.show(block=False)
    assert fig.is_open()
    assert fig.wait_closed(0.2) is False
    t = close_later(fig)
    assert fig.wait_closed() is True
    t.join()
    assert not fig.is_open()


def test_wait_closed_on_a_figure_never_shown():
    assert sextant.Figure().wait_closed() is True


@window
def test_ctrl_c_interrupts_wait_closed():
    fig = sextant.Figure(width=200, height=150)
    fig.show(block=False)
    threading.Timer(0.3, _thread.interrupt_main).start()
    try:
        with pytest.raises(KeyboardInterrupt):
            fig.wait_closed()
    finally:
        fig.close()


@window
def test_graph_calls_proceed_while_another_thread_waits():
    fig = sextant.Figure(width=200, height=150)
    fig.show(block=False)
    done = threading.Event()

    def plot_then_close():
        fig.axes().line(np.arange(5.0))
        done.set()
        fig.close()

    t = threading.Thread(target=plot_then_close)
    t.start()
    assert fig.wait_closed(10) is True
    t.join()
    assert done.is_set()


@window
def test_context_manager_closes():
    with sextant.Figure(width=200, height=150) as fig:
        fig.show(block=False)
        assert fig.is_open()
    assert not fig.is_open()


@window
def test_run_returns_when_every_figure_closed():
    figs = [sextant.Figure(width=200, height=150) for _ in range(2)]
    for f in figs:
        f.show(block=False)
    timers = [close_later(f, 0.2 + 0.2 * i) for i, f in enumerate(figs)]
    sextant.run()
    for t in timers:
        t.join()
    assert not any(f.is_open() for f in figs)


@window
def test_show_does_not_block_in_an_interactive_session(monkeypatch):
    monkeypatch.setattr(sys, "ps1", ">>> ", raising=False)
    fig = sextant.Figure(width=200, height=150)
    try:
        fig.show()
        assert fig.is_open()
    finally:
        fig.close()


@window
def test_show_blocks_in_a_script():
    rc, out, err, _ = run_script("""
        import threading, sextant
        fig = sextant.Figure(width=200, height=150)
        threading.Timer(0.5, fig.close).start()
        fig.show()
        print("closed" if not fig.is_open() else "still open")
    """)
    assert rc == 0, err
    assert out.strip() == "closed"


@window
def test_dropping_a_shown_figure_closes_it():
    rc, out, err, secs = run_script("""
        import gc, sextant
        fig = sextant.Figure(width=200, height=150)
        fig.show(block=False)
        del fig
        gc.collect()
        print("ok")
    """)
    assert rc == 0, err
    assert out.strip() == "ok"


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="a display is optional on Linux only")
def test_show_without_a_display_raises():
    env = {k: v for k, v in os.environ.items() if k not in ("DISPLAY", "WAYLAND_DISPLAY")}
    code = textwrap.dedent("""
        import numpy as np, sextant, tempfile, os
        fig = sextant.Figure(width=200, height=150)
        fig.axes().line(np.arange(4.0))
        try:
            fig.show(block=False)
            print("no error")
        except RuntimeError as e:
            print("RuntimeError", "DISPLAY" in str(e))
        print("open", fig.is_open())
        fig.savefig(os.path.join(tempfile.mkdtemp(), "x.png"))
        print("saved")
    """)
    p = run_python(["-c", code], env=env)
    assert p.returncode == 0, p.stderr
    assert p.stdout.split("\n")[:3] == ["RuntimeError True", "open False", "saved"]


@window
def test_exit_with_open_windows_is_clean():
    rc, out, err, secs = run_script("""
        import sextant, numpy as np
        figs = [sextant.Figure(width=200, height=150) for _ in range(3)]
        for f in figs:
            f.axes().line(np.arange(4.0))
            f.show(block=False)
        keep = figs[0].axes()   # an Axes still holding a figure at exit
        print("exiting")
    """)
    assert rc == 0, err
    assert out.strip() == "exiting"
    assert err == ""
