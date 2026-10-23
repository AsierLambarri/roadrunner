"""Command line of the sweep (goal 7), from the repository root:

    python -m evaluate sweep RUN_DIR [...] --csv FILE [--reports science.txt compare.txt ...] [--out DIR]

The other points are standalone scripts: ``python3 evaluate/run_all.py --path RUN`` (soundness, stability,
model; or one of them), and ``python3 evaluate/science.py`` / ``compare.py --path REF --vs OTHER [...]``
for two or more catalogues.
"""
import argparse

from evaluate import sweep


def main():
    parser = argparse.ArgumentParser(prog="python -m evaluate", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("point", choices=["sweep"])
    parser.add_argument("runs", nargs="+", help="run directories")
    parser.add_argument("--out", help="where the per-run reports are, as DIR/<run name>/ "
                                      "(default: RUN/evaluation)")
    parser.add_argument("--csv", help="the table to append to")
    parser.add_argument("--reports", nargs="*", default=[], help="multi-catalogue reports to read")
    args = parser.parse_args()
    sweep.evaluate(args.runs, args.csv, args.reports, root=args.out)


if __name__ == "__main__":                  # workers are spawned: they import this module, never run it
    main()
