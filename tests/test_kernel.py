"""The ipykernel event loop (sextant/_interactive.py), in a real kernel.

Each test starts a kernel with this interpreter (jupyter_client's native
"python3" spec) and runs cells in it. An idle kernel delivers events only
through its event loop, so a Close callback that fires between cells shows the
loop at work.
"""

import time

import pytest

pytest.importorskip("ipykernel")
manager = pytest.importorskip("jupyter_client.manager")

window = pytest.mark.window

SETUP = """
import threading, sextant
k = get_ipython().kernel
def loop_name():
    return getattr(k.eventloop, "__name__", k.eventloop)
def shown(close_after=None):
    fig = sextant.Figure(width=200, height=150)
    fig.axes().line([0, 1, 2], [0, 1, 4])
    got = []
    fig.connect("close", lambda e: got.append("close"))
    fig.show()
    if close_after is not None:
        threading.Timer(close_after, fig.close).start()
    return fig, got
"""


@pytest.fixture
def kernel():
    km, kc = manager.start_new_kernel(kernel_name="python3", startup_timeout=60)
    try:
        run(kc, SETUP)
        yield km, kc
    finally:
        kc.stop_channels()
        km.shutdown_kernel(now=True)


def run(kc, code, timeout=20):
    """Run a cell; its stdout. A cell that raises fails the test."""
    msg_id = kc.execute(code)
    out = []
    while True:
        msg = kc.get_iopub_msg(timeout=timeout)
        if msg["parent_header"].get("msg_id") != msg_id:
            continue
        kind, content = msg["msg_type"], msg["content"]
        if kind == "stream" and content["name"] == "stdout":
            out.append(content["text"])
        elif kind == "error":
            pytest.fail("cell raised:\n" + "\n".join(content["traceback"]))
        elif kind == "status" and content["execution_state"] == "idle":
            return "".join(out).strip()


@window
def test_show_enables_the_loop_which_delivers_and_then_turns_off(kernel):
    _, kc = kernel
    assert run(kc, "print(loop_name())") == "None"
    assert run(kc, "fig, got = shown(close_after=0.5); print(loop_name())") == "kernel_loop"
    time.sleep(1.5)  # the kernel is idle: only its event loop can deliver the Close
    assert run(kc, "print(got, loop_name(), fig.is_open())") == "['close'] None False"


@window
def test_the_kernel_answers_while_a_window_is_open(kernel):
    _, kc = kernel
    run(kc, "fig, got = shown()")
    time.sleep(0.5)  # into the loop
    for i in range(5):
        start = time.monotonic()
        assert run(kc, f"print({i})") == str(i)
        assert time.monotonic() - start < 1.0
    assert run(kc, "print(loop_name(), fig.is_open())") == "kernel_loop True"
    run(kc, "fig.close()")


@window
def test_gui_sextant_by_hand_stays_on(kernel):
    _, kc = kernel
    run(kc, "get_ipython().run_line_magic('gui', 'sextant')")
    assert run(kc, "print(loop_name())") == "kernel_loop"
    run(kc, "fig, got = shown(close_after=0.3)")
    time.sleep(1.0)
    assert run(kc, "print(got, loop_name())") == "['close'] kernel_loop"
    run(kc, "get_ipython().run_line_magic('gui', '')")
    assert run(kc, "print(loop_name())") == "None"


@window
def test_another_event_loop_is_left_alone(kernel):
    _, kc = kernel
    run(kc, """
from ipykernel.eventloops import register_integration
@register_integration("other")
def other_loop(kernel):
    pass
get_ipython().run_line_magic('gui', 'other')
""")
    assert run(kc, "fig, got = shown(); print(loop_name())") == "other_loop"
    run(kc, "fig.close()")


@window
def test_an_interrupt_in_the_loop_leaves_the_kernel_working(kernel):
    km, kc = kernel
    run(kc, "fig, got = shown()")
    time.sleep(0.5)
    km.interrupt_kernel()
    time.sleep(0.5)
    assert run(kc, "print(loop_name(), fig.is_open())") == "kernel_loop True"
    run(kc, "fig.close()")
