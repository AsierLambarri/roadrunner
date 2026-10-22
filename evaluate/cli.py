"""What every evaluation script shares: its command line, where it writes, and its log.

Run one point with ``python3 evaluate/<point>.py --path RUN [...]``, or all of them with
``python3 evaluate/run_all.py --path RUN [...]``. Everything goes to ``RUN/evaluation/``: the report
``<point>.txt``, the log ``<point>.log`` and the PDFs under ``plots/``. Nothing else is written; a point
that needed scratch data would put it under ``evaluation/tmp/``.
"""
import argparse
import os
import sys
from pathlib import Path

from evaluate.load import Run

WORKERS = min(8, os.cpu_count() or 1)


def arguments(doc, every=False, light_every=False, flags=None):
    """Parse the command line of an evaluation script (``every``: it samples every N-th snapshot;
    ``light_every``: its extra pass samples its own; ``flags``: {name: help} of extra store-true options)."""
    p = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--path", required=True,
                   help="the roadrunner run directory; results go to PATH/evaluation/")
    p.add_argument("--particles", help="directory holding particle_data/ (default: the run's own)")
    p.add_argument("--workers", type=int, default=WORKERS,
                   help=f"processes for the per-snapshot work (default {WORKERS}; 1: serial)")
    p.add_argument("--snaps", help="first:last snapshot, both included (default: all)")
    if every:
        p.add_argument("--every", type=int, default=10, help="every N-th snapshot (default 10)")
    if light_every:
        p.add_argument("--light-every", type=int, default=3,
                       help="the extra pass (model: 3.2b, 3.4): every N-th snapshot (default 3; 1: all)")
    for name, help_ in (flags or {}).items():
        p.add_argument(f"--{name}", action="store_true", help=help_)
    return p.parse_args()


def start(point, args):
    """Open the run and its ``evaluation/`` directory, and copy the script's output to ``<point>.log``."""
    run = Run(args.path, particles=args.particles)
    if args.snaps:
        lo, hi = (int(x) for x in args.snaps.split(":"))
        run.snaps = run.snaps[(run.snaps >= lo) & (run.snaps <= hi)]
    out = Path(args.path) / "evaluation"
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / f"{point}.log", "w")
    sys.stdout, sys.stderr = Tee(sys.stdout, log), Tee(sys.stderr, log)
    return run, out


class Tee:
    """A stream that also writes to a log file, leaving out progress-bar redraws (carriage returns)."""

    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, text):
        self.stream.write(text)
        if "\r" not in text:
            self.log.write(text)
        return len(text)

    def flush(self):
        self.stream.flush()
        self.log.flush()

    def isatty(self):
        return self.stream.isatty()
