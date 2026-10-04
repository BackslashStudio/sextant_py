"""Manual check of the window icon on Windows: title bar, taskbar, Alt+Tab.

    uv run python tools/icon_check.py            # as the binding ships
    uv run python tools/icon_check.py --aumid    # with an explicit AppUserModelID

Opens a figure, prints which icons the window holds (WM_GETICON), and waits for
you to close it. Look at the title bar, the taskbar button and Alt+Tab. Python
windows on the taskbar are grouped under python.exe's identity, which may show
Python's icon there whatever the window holds; --aumid gives the process its own
identity first (what matplotlib does) so the two runs can be compared.
"""

import argparse
import ctypes
import sys
import time

import numpy as np

import sextant

TITLE = "sextant icon check"


def window_icons(title):
    """(hwnd, big icon handle, small icon handle); 0 for a missing one."""
    user32 = ctypes.windll.user32
    user32.FindWindowW.restype = ctypes.c_void_p
    user32.SendMessageW.restype = ctypes.c_void_p
    user32.SendMessageW.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_size_t, ctypes.c_ssize_t]
    hwnd = user32.FindWindowW(None, title) or 0
    if not hwnd:
        return 0, 0, 0
    WM_GETICON, ICON_SMALL, ICON_BIG = 0x7F, 0, 1
    big = user32.SendMessageW(hwnd, WM_GETICON, ICON_BIG, 0) or 0
    small = user32.SendMessageW(hwnd, WM_GETICON, ICON_SMALL, 0) or 0
    return hwnd, big, small


def main():
    if sys.platform != "win32":
        sys.exit("icon_check.py is for Windows (macOS has no per-window icon; on Linux look at the title bar)")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--aumid", nargs="?", const="BackslashStudio.sextant", default=None,
                    help="set this AppUserModelID before the window opens (default: %(const)s)")
    args = ap.parse_args()

    if args.aumid:
        hr = ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(ctypes.c_wchar_p(args.aumid))
        print(f"AppUserModelID = {args.aumid!r} (HRESULT {hr:#x})")
    else:
        print("AppUserModelID: not set (grouped as python.exe)")
    print(f"sextant {sextant.__file__}")

    fig = sextant.Figure(title=TITLE)
    x = np.linspace(0, 2 * np.pi, 200)
    fig.axes().line(x, np.sin(x))
    fig.show(block=False)

    hwnd = big = small = 0
    deadline = time.monotonic() + 10
    while not hwnd and time.monotonic() < deadline:
        sextant.poll_events()
        time.sleep(0.05)
        hwnd, big, small = window_icons(TITLE)
    if not hwnd:
        fig.close()
        sys.exit("window not found")
    print(f"window {hwnd:#x}: big icon {'set' if big else 'MISSING'}, small icon {'set' if small else 'MISSING'}")
    print("Check the title bar, the taskbar button and Alt+Tab, then close the window.")
    fig.wait_closed()


if __name__ == "__main__":
    main()
