"""What every evaluation script shares: its command line, where it writes, and its log.

Run one point with ``python3 evaluate/<point>.py --path RUN [...]``, or all of them with
``python3 evaluate/run_all.py --path RUN [...]``. Everything goes to ``RUN/evaluation/``: the report
``<point>.txt``, the log ``<point>.log`` and the PDFs under ``plots/``. The comparisons of catalogues
(science, compare) take ``--path REF --vs OTHER [...]`` and write the same layout to ``--out``, by
default the first roadrunner run's ``evaluation/<REF>_vs_<OTHERS>/``. Nothing else is written; a point
that needed scratch data would put it under ``evaluation/tmp/``.
"""
import argparse
import os
import sys
from pathlib import Path

from evaluate.load import Run, open_catalogue

WORKERS = min(8, os.cpu_count() or 1)


def arguments(doc, every=None, light_every=False, flags=None, options=None, compare=False):
    """Parse the command line of an evaluation script. ``every``: the default N of its ``--every`` (it
    samples every N-th snapshot; None: no such option); ``light_every``: its extra pass samples its own;
    ``flags``: {name: help} of extra store-true options; ``options``: {name: add_argument keywords} of
    other extra options; ``compare``: it compares catalogues (``--path`` the reference, ``--vs`` the
    others, ``--out`` where the results go)."""
    p = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    if compare:
        p.add_argument("--path", required=True,
                       help="the reference catalogue: a roadrunner run or an ACCME result directory")
        p.add_argument("--vs", nargs="+", required=True, help="the catalogues compared with the reference")
        p.add_argument("--out", help="output directory (default: the first roadrunner run's "
                                     "evaluation/<REF>_vs_<OTHERS>/)")
    else:
        p.add_argument("--path", required=True,
                       help="the roadrunner run directory; results go to PATH/evaluation/")
    p.add_argument("--particles", help="directory holding particle_data/ (default: each run's own)")
    p.add_argument("--workers", type=int, default=WORKERS,
                   help=f"processes for the per-snapshot work (default {WORKERS}; 1: serial)")
    p.add_argument("--snaps", help="first:last snapshot, both included (default: all)")
    if every is not None:
        p.add_argument("--every", type=int, default=every, help=f"every N-th snapshot (default {every})")
    if light_every:
        p.add_argument("--light-every", type=int, default=3,
                       help="the extra pass (model: 3.2b, 3.4): every N-th snapshot (default 3; 1: all)")
    for name, help_ in (flags or {}).items():
        p.add_argument(f"--{name}", action="store_true", help=help_)
    for name, kw in (options or {}).items():
        p.add_argument(f"--{name}", **kw)
    return p.parse_args()


def start(point, args):
    """Open the run and its ``evaluation/`` directory, and copy the script's output to ``<point>.log``."""
    run = Run(args.path, particles=args.particles)
    if args.snaps:
        lo, hi = (int(x) for x in args.snaps.split(":"))
        run.snaps = run.snaps[(run.snaps >= lo) & (run.snaps <= hi)]
        run.planned = run.planned[(run.planned >= lo) & (run.planned <= hi)]
    out = Path(args.path) / "evaluation"
    log_to(out, point)
    return run, out


def start_compare(point, args):
    """Open the catalogues (the reference first) and the roadrunner run whose particle data they are read
    on (the first one given), make the output directory, and copy the script's output to ``<point>.log``."""
    cats = [open_catalogue(p, particles=args.particles) for p in [args.path, *args.vs]]
    particles = next((c for c in cats if isinstance(c, Run)), None)
    if particles is None:
        raise SystemExit("a comparison reads the particles through a roadrunner run: give at least one")
    if args.snaps:
        lo, hi = (int(x) for x in args.snaps.split(":"))
        particles.snaps = particles.snaps[(particles.snaps >= lo) & (particles.snaps <= hi)]
    names = "_".join(c.name for c in cats[1:])
    out = Path(args.out) if args.out else particles.path / "evaluation" / f"{cats[0].name}_vs_{names}"
    log_to(out, point)
    return cats, particles, out


def log_to(out, point):
    """Make ``out`` and copy the script's output to ``out/<point>.log``."""
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / f"{point}.log", "w")
    sys.stdout, sys.stderr = Tee(sys.stdout, log), Tee(sys.stderr, log)


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
