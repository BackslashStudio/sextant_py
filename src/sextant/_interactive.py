"""Keeping windows live in an interactive session.

Figure.show() in an interactive session returns at once. For the window to
keep working between statements -- events delivered, and on macOS the window
pumped at all -- something has to run sextant while the prompt waits:

* the plain REPL (and ``python -i``): a PyOS_InputHook, installed by the C++
  module (src/ext/lifecycle.cpp) on the first show();
* IPython in a terminal: prompt_toolkit's inputhook, registered here as the
  GUI integration "sextant" and enabled on the first show() unless another
  GUI integration is active (``%gui sextant`` enables it by hand);
* a Jupyter kernel (ipykernel): a kernel event loop, registered here as
  "sextant" too, which pumps sextant while the kernel waits for a message.
  The first show() enables it unless another event loop is active, and it
  turns itself off when the last window closes; ``%gui sextant`` enables it
  by hand, and then it stays until ``%gui`` turns it off.
"""

import sys
import time

from . import _sextant

_registered = False
_kernel_registered = False
# The kernel loop was enabled by show(), not by %gui: it ends with the last window.
_kernel_auto = False
# Longest stretch kernel_loop() keeps the kernel's own event loop waiting, in seconds.
_KERNEL_SLICE = 0.05


def _any_open():
    return _sextant._any_open()


def inputhook(context):
    """prompt_toolkit inputhook: pump sextant until a line is being typed."""
    while not context.input_is_ready():
        _sextant.poll_events()
        if not _any_open():
            return
        time.sleep(0.01)


def _shell_stream(kernel):
    try:
        from ipykernel.eventloops import get_shell_stream
    except ImportError:  # ipykernel 6
        return kernel.shell_stream
    return get_shell_stream(kernel)


def kernel_loop(kernel):
    """ipykernel event loop: pump sextant until the shell has a message.

    ipykernel calls it on the kernel's main thread whenever it is idle, and
    again after handling the message this returns for (as its macOS loop
    does). Events are delivered here, so callbacks run between cells.
    """
    global _kernel_auto
    stream = _shell_stream(kernel)
    # Back to the kernel at least this often even with nothing on the stream:
    # a message the kernel has already read off it (ipykernel 7 queues its
    # handler as an asyncio task) shows no event to flush() and is handled only
    # once this returns.
    deadline = time.monotonic() + _KERNEL_SLICE
    while True:
        _sextant.poll_events()
        if stream.flush(limit=1) or time.monotonic() >= deadline:
            return
        if _kernel_auto and not _any_open():
            # Off with the last window, as the REPL hook: an idle kernel then
            # waits for messages as it did before show(). A window closed since
            # the poll above has queued its Close before reading as closed, and
            # nothing pumps after this: deliver it now, as run() does.
            _sextant.poll_events()
            _kernel_auto = False
            kernel.shell.enable_gui(None)
            return
        time.sleep(kernel._poll_interval)


def register_kernel_loop():
    """Make "sextant" a kernel event loop (``%gui sextant``). True once done;
    False without ipykernel."""
    global _kernel_registered
    if _kernel_registered:
        return True
    try:
        from ipykernel.eventloops import register_integration
    except ImportError:
        return False
    register_integration("sextant")(kernel_loop)
    _kernel_registered = True
    return True


def _ipython():
    mod = sys.modules.get("IPython")
    get = getattr(mod, "get_ipython", None) if mod else None
    return get() if get else None


def _enable_kernel(ip):
    global _kernel_auto
    kernel = getattr(ip, "kernel", None)
    if kernel is None or not register_kernel_loop():
        return
    if getattr(kernel, "eventloop", None) is None:
        ip.enable_gui("sextant")
        _kernel_auto = True


def enable_ipython():
    """Called by Figure.show() in an interactive session. True when this is
    IPython (terminal: the inputhook is on; kernel: the kernel loop is on),
    False for the plain REPL, which the caller handles."""
    global _registered
    ip = _ipython()
    if ip is None:
        return False
    if type(ip).__name__ != "TerminalInteractiveShell":
        _enable_kernel(ip)
        return True
    if not _registered:
        from IPython.terminal.pt_inputhooks import register

        register("sextant", inputhook)
        _registered = True
    if getattr(ip, "active_eventloop", None) is None:
        ip.enable_gui("sextant")
    return True


# Imported in a kernel: "sextant" is there for %gui before the first show().
if "ipykernel" in sys.modules:
    register_kernel_loop()
