"""Narrow down the macOS teardown hang: one variable per scenario.

On GitHub's GPU-less macOS VMs, a window's render thread can block for good
inside Apple's software GL renderer (its first draw, in
cvmRequestFunctionPointerArrayWrite) while windows are being torn down. Each
scenario below changes one thing against the baseline ("exit3", what
stress_exit.py's "exit" does), so the hang counts per scenario say what the
trigger needs:

  exit3                three windows shown without blocking, then exit (baseline)
  exit1                one window: is another window needed at all?
  exit3_framed         every window has drawn a frame before exit: is the
                       first frame the vulnerable one?
  exit3_staggered      windows shown one at a time, each framed before the next;
                       only the last is on its first frame at exit: are
                       concurrent first frames needed?
  close3               three shown, then close() each at once, no exit: is the
                       atexit sweep (interpreter shutdown) needed?
  close_other_first    A framed, B shown, A closed at once, then B must draw:
                       does destroying one context stall another's first draw?
  close_self_first     show then close at once, three times in a row: one
                       window's own teardown during its first frame

    python tools/teardown_bisect.py <scenario> [runs=40] [--max-hangs N] [--report-dir DIR]

Runs a fresh Python per run (same checks and hang diagnosis as
stress_exit.py) and keeps going after a hang, up to --max-hangs (default 3),
to count rather than stop. Prints one summary line, appends a table row to
$GITHUB_STEP_SUMMARY when set, writes each failure report to --report-dir, and
exits 0: the counts are the result.
"""

import argparse
import os
import sys
import time

from stress_exit import run_once

PRELUDE = """
import sextant, numpy as np

def fig():
    f = sextant.Figure(width=200, height=150)
    f.axes().line(np.arange(4.0))
    return f

def wait_framed(f):
    # wait_closed() pumps the main thread on macOS while it waits; a frame
    # that never comes is a hang, caught by the run's own timeout.
    while f.frame_stats().frames < 1:
        f.wait_closed(0.005)
"""

SCENARIOS = {
    "exit3": """
figs = [fig() for _ in range(3)]
for f in figs:
    f.show(block=False)
""",
    "exit1": """
f = fig()
f.show(block=False)
""",
    "exit3_framed": """
figs = [fig() for _ in range(3)]
for f in figs:
    f.show(block=False)
for f in figs:
    wait_framed(f)
""",
    "exit3_staggered": """
figs = [fig() for _ in range(3)]
for f in figs[:-1]:
    f.show(block=False)
    wait_framed(f)
figs[-1].show(block=False)
""",
    "close3": """
figs = [fig() for _ in range(3)]
for f in figs:
    f.show(block=False)
for f in figs:
    f.close()
""",
    "close_other_first": """
a, b = fig(), fig()
a.show(block=False)
wait_framed(a)
b.show(block=False)
a.close()
wait_framed(b)
b.close()
""",
    "close_self_first": """
for _ in range(3):
    f = fig()
    f.show(block=False)
    f.close()
""",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario", choices=sorted(SCENARIOS))
    ap.add_argument("runs", nargs="?", type=int, default=40)
    ap.add_argument("--max-hangs", type=int, default=3)
    ap.add_argument("--report-dir")
    args = ap.parse_args()

    src = PRELUDE + SCENARIOS[args.scenario] + '\nprint("exiting")\n'
    if args.report_dir:
        os.makedirs(args.report_dir, exist_ok=True)

    done = hangs = errors = 0
    first_hang = None
    worst = 0.0
    for i in range(args.runs):
        t = time.monotonic()
        report = run_once(src)
        worst = max(worst, time.monotonic() - t)
        done += 1
        if report is None:
            continue
        if report.startswith("HANG"):
            hangs += 1
            first_hang = first_hang or i + 1
        else:
            errors += 1
        print(f"run {i + 1}: {report.splitlines()[0]}", flush=True)
        if args.report_dir:
            path = os.path.join(args.report_dir, f"{args.scenario}-run{i + 1}.txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write(report)
        if hangs >= args.max_hangs:
            break

    line = (f"{args.scenario}: {hangs} hang(s), {errors} other failure(s) in {done} of {args.runs} runs"
            f"{f', first hang at run {first_hang}' if first_hang else ''}; slowest {worst:.2f} s")
    print(line, flush=True)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write(f"| `{args.scenario}` | {hangs} | {errors} | {done} | {first_hang or '-'} | {worst:.2f} s |\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
