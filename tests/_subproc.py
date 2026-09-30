"""A Python child for a test, run so that a hang can be diagnosed.

A child given as long as pytest-timeout (60 s) times out at the same moment:
pytest's alarm then fires while the failure is being reported, and both the
failure and the child's state are lost. So a child gets 30 s. If it is still
running then, its native stacks are sampled (macOS: `sample`), it is sent
SIGABRT so that faulthandler (enabled in it through PYTHONFAULTHANDLER) prints
every thread's Python stack (not on Windows), and the test fails with all of it.
"""

import os
import signal
import subprocess
import sys
import tempfile

import pytest

TIMEOUT = 30


def _native_stacks(pid):
    if sys.platform != "darwin":
        return "(native stacks are only sampled on macOS)"
    fd, path = tempfile.mkstemp(suffix=".txt")
    os.close(fd)
    try:
        subprocess.run(["sample", str(pid), "2", "-file", path], capture_output=True, timeout=20)
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()
    except (OSError, subprocess.SubprocessError) as e:
        return f"(sample failed: {e!r})"
    finally:
        os.unlink(path)


def run_python(args, *, input=None, env=None, timeout=TIMEOUT):
    """`python *args` -> CompletedProcess (text); fails the test if it hangs."""
    env = dict(os.environ if env is None else env, PYTHONFAULTHANDLER="1")
    p = subprocess.Popen([sys.executable, *args], env=env, text=True,
                         stdin=None if input is None else subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        out, err = p.communicate(input, timeout=timeout)
        return subprocess.CompletedProcess(p.args, p.returncode, out, err)
    except subprocess.TimeoutExpired:
        pass
    native = _native_stacks(p.pid)
    if sys.platform == "win32":
        p.kill()
        out, err = p.communicate()
    else:
        p.send_signal(signal.SIGABRT)
        try:
            out, err = p.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()
            out, err = p.communicate()
    pytest.fail(f"child Python still running after {timeout} s: {args!r}\n"
                f"--- stdout\n{out}\n"
                f"--- stderr (with faulthandler's Python stacks after SIGABRT)\n{err}\n"
                f"--- native stacks\n{native}", pytrace=False)
