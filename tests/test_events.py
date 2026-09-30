"""Figure.connect() and event delivery.

Only Close can be caused by a program (closing a shown figure queues one), so
it carries the delivery tests; mouse, key, scroll, pick and resize events
need a person at the window and are checked by hand.
"""

import gc
import subprocess
import sys
import textwrap
import threading
import weakref

import numpy as np
import pytest

import sextant
from sextant import EventKind

window = pytest.mark.window


def shown():
    fig = sextant.Figure(width=200, height=150)
    fig.show(block=False)
    return fig


def test_connect_rejects_bad_arguments():
    fig = sextant.Figure()
    with pytest.raises(TypeError):
        fig.connect("mouse_wiggle", print)
    with pytest.raises(TypeError):
        fig.connect("close", 42)


def test_nothing_is_delivered_without_a_window():
    fig = sextant.Figure()
    got = []
    fig.connect("close", got.append)
    fig.close()
    assert fig.dispatch_events() == 0 and got == []


@window
@pytest.mark.parametrize("kind", [EventKind.CLOSE, "close", "Close", "close_event"])
def test_close_event(kind):
    fig = shown()
    got = []
    fig.connect(kind, got.append)
    fig.close()
    assert got == []  # queued, not yet delivered
    assert fig.dispatch_events() == 1
    (e,) = got
    assert e.kind is EventKind.CLOSE
    assert e.axes == -1 and e.inaxes is None and not e.has_data and np.isnan(e.xdata)
    assert e.modifiers == () and e.pick_kind == "none" and e.consumed == "none"
    assert repr(e) == "Event(kind='close')"


@window
def test_wait_closed_delivers_on_the_waiting_thread():
    fig = shown()
    threads = []
    fig.connect("close", lambda e: threads.append(threading.get_ident()))
    threading.Timer(0.3, fig.close).start()
    assert fig.wait_closed(10)
    assert threads == [threading.get_ident()]


@window
def test_disconnect():
    fig = shown()
    got = []
    cid = fig.connect("close", got.append)
    fig.disconnect(cid)
    fig.disconnect(12345)  # unknown ids are ignored
    fig.close()
    fig.dispatch_events()
    assert got == []


@window
def test_a_raising_callback_is_reported_and_delivery_goes_on(monkeypatch):
    seen = []
    monkeypatch.setattr(sys, "unraisablehook", seen.append)
    fig = shown()
    got = []

    def bad(e):
        raise ValueError("boom")

    fig.connect("close", bad)
    fig.connect("close", got.append)
    fig.close()
    fig.dispatch_events()
    assert len(got) == 1
    assert len(seen) == 1 and isinstance(seen[0].exc_value, ValueError)
    seen.clear()  # its traceback reaches this frame: a cycle nanobind's exit check would report


@window
def test_keyboard_interrupt_in_a_callback_reaches_the_wait():
    fig = shown()

    def stop(e):
        raise KeyboardInterrupt

    fig.connect("close", stop)
    threading.Timer(0.2, fig.close).start()
    with pytest.raises(KeyboardInterrupt):
        fig.wait_closed(10)


@window
def test_callback_may_use_the_figure():
    fig = shown()
    out = []

    def cb(e):
        fig.axes().line([0.0, 1.0])  # the graph lock is free during delivery
        out.append(len(fig.render_png()))

    fig.connect("close", cb)
    fig.close()
    fig.dispatch_events()
    assert out and out[0] > 0 and fig.axes().line_count() == 1


@window
def test_callback_may_drop_the_last_reference_to_its_figure():
    holder = {"fig": shown()}
    fig = holder["fig"]
    fig.connect("close", lambda e: holder.clear())
    fig.close()
    del fig
    sextant.poll_events()  # delivers; the figure dies after the dispatch
    assert holder == {}


def test_a_callback_referring_to_its_figure_is_collected():
    fig = sextant.Figure()
    # fig -> callback -> fig. A default argument holds the object itself; a
    # closure would hold the variable, which `del` empties.
    fig.connect("close", lambda e, f=fig: f.close())
    ref = weakref.ref(fig)
    del fig
    gc.collect()
    assert ref() is None


@window
def test_a_shown_figure_in_a_cycle_is_collected_and_closed():
    code = textwrap.dedent("""
        import gc, weakref, sextant
        fig = sextant.Figure(width=100, height=80)
        fig.connect("key_down", lambda e, f=fig: f.close())
        fig.show(block=False)
        ax = fig.axes()
        ref = weakref.ref(fig)
        del fig
        gc.collect()
        print(ref() is None)
        print(sextant._sextant._any_open())  # ax keeps the figure, so the window stays
        del ax
        print(sextant._sextant._any_open())
    """)
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr
    assert p.stdout.split() == ["True", "True", "False"]


@window
def test_run_delivers_every_figures_events():
    figs = [shown(), shown()]
    got = []
    for i, f in enumerate(figs):
        f.connect("close", lambda e, i=i: got.append(i))
        threading.Timer(0.2 + 0.2 * i, f.close).start()
    sextant.run()
    assert sorted(got) == [0, 1]


@window
def test_exit_with_callbacks_connected_is_clean():
    code = textwrap.dedent("""
        import sextant
        figs = [sextant.Figure(width=100, height=80) for _ in range(2)]
        for f in figs:
            f.connect("close", lambda e: print("closed"))
            f.connect("mouse_move", print)
            f.show(block=False)
        print("exiting")
    """)
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr
    assert p.stdout.strip() == "exiting" and p.stderr == ""
