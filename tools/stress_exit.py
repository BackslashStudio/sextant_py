"""Close windows on their first frames, many times, and fail on any hang.

A rare hang at exit (a window's render thread inside its first GL draw, the
main thread joining it) showed up only on CI's macOS VMs, in about one wheel
test run in twenty. One pytest run cannot tell a fix from luck; this repeats
the two ways a window is torn down early:

  exit   three windows shown without blocking, then the interpreter exits
         (the atexit sweep closes them), as test_exit_with_open_windows_is_clean
  close  show(block=False) followed at once by close(), three times

Each run is a fresh Python that must exit 0, print "exiting" and write nothing
to stderr within 30 s. On a hang the native stacks are sampled (macOS:
`sample`), the child gets SIGABRT so faulthandler prints its Python stacks, and
the script stops with all of it.

    python tools/stress_exit.py [runs per scenario, default 100]

Needs sextant installed (a wheel or the dev build) and a session where windows
can open.
"""

import os
import signal
import subprocess
import sys
import tempfile
import time

TIMEOUT = 30

SCENARIOS = {
    "exit": """
import sextant, numpy as np
figs = [sextant.Figure(width=200, height=150) for _ in range(3)]
for f in figs:
    f.axes().line(np.arange(4.0))
    f.show(block=False)
keep = figs[0].axes()   # an Axes still holding a figure at exit
print("exiting")
""",
    "close": """
import sextant, numpy as np
for _ in range(3):
    f = sextant.Figure(width=200, height=150)
    f.axes().line(np.arange(4.0))
    f.show(block=False)
    f.close()
print("exiting")
""",
}


def native_stacks(pid):
    if sys.platform != "darwin":
        return "(native stacks are only sampled on macOS)"
    fd, path = tempfile.mkstemp(suffix=".txt")
    os.close(fd)
    try:
        subprocess.run(["sample", str(pid), "2", "-file", path], capture_output=True, timeout=20,
                       check=False)
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()
    except (OSError, subprocess.SubprocessError) as e:
        return f"(sample failed: {e!r})"
    finally:
        os.unlink(path)


def run_once(src):
    """None if the child exited cleanly, else a report."""
    env = dict(os.environ, PYTHONFAULTHANDLER="1")
    p = subprocess.Popen([sys.executable, "-c", src], env=env, text=True,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        out, err = p.communicate(timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        native = native_stacks(p.pid)
        if sys.platform == "win32":
            p.kill()
        else:
            p.send_signal(signal.SIGABRT)
        try:
            out, err = p.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()
            out, err = p.communicate()
        return (f"HANG: still running after {TIMEOUT} s\n--- stdout\n{out}\n"
                f"--- stderr (with faulthandler's Python stacks after SIGABRT)\n{err}\n"
                f"--- native stacks\n{native}")
    if p.returncode != 0 or out.strip() != "exiting" or err.strip():
        return f"exit code {p.returncode}\n--- stdout\n{out}\n--- stderr\n{err}"
    return None


def main():
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    for name, src in SCENARIOS.items():
        worst = 0.0
        for i in range(runs):
            t = time.monotonic()
            report = run_once(src)
            worst = max(worst, time.monotonic() - t)
            if report is not None:
                print(f"{name}: run {i + 1} of {runs} failed\n{report}", flush=True)
                return 1
        print(f"{name}: {runs} runs clean, slowest {worst:.2f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
