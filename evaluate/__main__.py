"""Command line of the multi-catalogue points, from the repository root:

    python -m evaluate {science,compare} CATALOGUE CATALOGUE [...] --out DIR [--particles DIR] [--every N]
    python -m evaluate sweep RUN_DIR [...] --csv FILE [--reports science.txt compare.txt ...] [--out DIR]

A CATALOGUE is a roadrunner run directory or an ACCME result directory (with catalogues.hdf5); the
first one is the reference. --particles is the directory holding particle_data/ (default: each run's
own); these points read the particles through the first roadrunner run given. --workers processes share
the per-snapshot work (default: min(8, cores)).

The per-run points are standalone scripts: python3 evaluate/run_all.py --path RUN (or one of
evaluate/soundness.py, stability.py, model.py).
"""
import argparse
import os
from pathlib import Path

from evaluate import compare, science, sweep
from evaluate.load import Run, open_catalogue

MULTI = {"science": science, "compare": compare}


def main():
    parser = argparse.ArgumentParser(prog="python -m evaluate", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("point", choices=[*MULTI, "sweep"])
    parser.add_argument("runs", nargs="+", help="run (or catalogue) directories")
    parser.add_argument("--particles", help="directory holding particle_data/ (default: each run's own)")
    parser.add_argument("--out", help="output directory; sweep: where the per-run reports are, as "
                                      "DIR/<run name>/ (default: RUN/evaluation)")
    parser.add_argument("--every", type=int, default=10, help="every N-th snapshot (default 10)")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1),
                        help="processes for the per-snapshot work (default: min(8, cores); 1: serial)")
    parser.add_argument("--csv", help="sweep: the table to append to")
    parser.add_argument("--reports", nargs="*", default=[], help="sweep: multi-catalogue reports to read")
    args = parser.parse_args()

    if args.point == "sweep":
        sweep.evaluate(args.runs, args.csv, args.reports, root=args.out)
        return
    cats = [open_catalogue(p, particles=args.particles) for p in args.runs]
    particles = next(c for c in cats if isinstance(c, Run))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    MULTI[args.point].evaluate(cats, particles, out, every=args.every, workers=args.workers)


if __name__ == "__main__":                  # workers are spawned: they import this module, never run it
    main()
