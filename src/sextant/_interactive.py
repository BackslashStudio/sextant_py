"""Keeping windows live in an interactive session.

Figure.show() in an interactive session returns at once. For the window to
keep working between statements -- events delivered, and on macOS the window
pumped at all -- something has to run sextant while the prompt waits:

* the plain REPL (and ``python -i``): a PyOS_InputHook, installed by the C++
  module (src/ext/lifecycle.cpp) on the first show();
* IPython in a terminal: prompt_toolkit's inputhook, registered here as the
  GUI integration "sextant" and enabled on the first show() unless another
  GUI integration is active (``%gui sextant`` enables it by hand);
* a Jupyter kernel: nothing is installed. On Windows and Linux the window
  works regardless and events arrive during wait_closed()/poll_events(); on
  macOS the window needs those calls to respond at all.
"""

import sys
import time

from . import _sextant

_registered = False


def _any_open():
    return _sextant._any_open()


def inputhook(context):
    """prompt_toolkit inputhook: pump sextant until a line is being typed."""
    while not context.input_is_ready():
        _sextant.poll_events()
        if not _any_open():
            return
        time.sleep(0.01)


def _ipython():
    mod = sys.modules.get("IPython")
    get = getattr(mod, "get_ipython", None) if mod else None
    return get() if get else None


def enable_ipython():
    """Called by Figure.show() in an interactive session. True when this is
    IPython (terminal: the inputhook is on; kernel: nothing to do), False for
    the plain REPL, which the caller handles."""
    global _registered
    ip = _ipython()
    if ip is None:
        return False
    if type(ip).__name__ != "TerminalInteractiveShell":
        return True
    if not _registered:
        from IPython.terminal.pt_inputhooks import register

        register("sextant", inputhook)
        _registered = True
    if getattr(ip, "active_eventloop", None) is None:
        ip.enable_gui("sextant")
    return True
