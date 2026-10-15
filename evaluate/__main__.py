"""Command line, from the repository root:

    python -m evaluate {soundness,stability,model} RUN_DIR [--particles DIR] [--out DIR] [--snaps A:B] [--every N]
    python -m evaluate {science,compare} CATALOGUE CATALOGUE [...] --out DIR [--particles DIR] [--every N]
    python -m evaluate sweep RUN_DIR [...] --csv FILE [--reports science.txt compare.txt ...] [--out DIR]

A CATALOGUE is a roadrunner run directory or an ACCME result directory (with catalogues.hdf5); the
first one is the reference. --particles is the run whose particle_data/ holds the particles (default:
the first roadrunner run given).
"""
import argparse
from pathlib import Path

from evaluate import compare, model, science, soundness, stability, sweep
from evaluate.load import Run, open_catalogue

PER_RUN = {"soundness": soundness, "stability": stability, "model": model}
MULTI = {"science": science, "compare": compare}

parser = argparse.ArgumentParser(prog="python -m evaluate", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("point", choices=[*PER_RUN, *MULTI, "sweep"])
parser.add_argument("runs", nargs="+", help="run (or catalogue) directories")
parser.add_argument("--particles", help="run directory whose particle_data/ holds the particles")
parser.add_argument("--out", help="output directory (default for one run: RUN/evaluate); sweep: where the "
                                  "per-run reports are, as DIR/<run name>/")
parser.add_argument("--snaps", help="one run: first:last snapshot, both included (default: all)")
parser.add_argument("--every", type=int, default=10, help="model, science, compare: every N-th snapshot (default 10)")
parser.add_argument("--csv", help="sweep: the table to append to")
parser.add_argument("--reports", nargs="*", default=[], help="sweep: multi-catalogue reports to read")
args = parser.parse_args()

if args.point == "sweep":
    sweep.evaluate(args.runs, args.csv, args.reports, root=args.out)
elif args.point in PER_RUN:
    run = Run(args.runs[0], particles=args.particles)
    if args.snaps:
        lo, hi = (int(x) for x in args.snaps.split(":"))
        run.snaps = run.snaps[(run.snaps >= lo) & (run.snaps <= hi)]
    out = Path(args.out or Path(args.runs[0]) / "evaluate")
    out.mkdir(parents=True, exist_ok=True)
    PER_RUN[args.point].evaluate(run, out, **({"every": args.every} if args.point == "model" else {}))
else:
    cats = [open_catalogue(p, particles=args.particles) for p in args.runs]
    particles = Run(args.particles) if args.particles else next(c for c in cats if isinstance(c, Run))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    MULTI[args.point].evaluate(cats, particles, out, every=args.every)
