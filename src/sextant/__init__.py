"""sextant: scientific plotting from a C++20 core."""

import atexit as _atexit

from . import _sextant
from ._sextant import Axes, Figure, poll_events, run, set_message_handler

# Close every window while the interpreter is still whole: a figure left to
# finalization would join its window thread against a half torn-down runtime.
_atexit.register(_sextant._shutdown)

__all__ = ["Axes", "Figure", "poll_events", "run", "set_message_handler"]
