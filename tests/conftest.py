import os
import sys

import pytest


def _can_open_windows():
    if os.environ.get("SEXTANT_TEST_NO_WINDOW"):
        return False
    if sys.platform.startswith("linux"):
        return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return True


def pytest_configure(config):
    config.addinivalue_line("markers", "window: opens a real window (needs a display)")


def pytest_collection_modifyitems(config, items):
    if _can_open_windows():
        return
    skip = pytest.mark.skip(reason="no display (or SEXTANT_TEST_NO_WINDOW set)")
    for item in items:
        if "window" in item.keywords:
            item.add_marker(skip)
