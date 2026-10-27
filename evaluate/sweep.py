"""Goal 7: one row per run for tuning: its configuration and every SCORE of its reports.

    python3 evaluate/sweep.py --path RUN [RUN ...] --csv FILE [--reports science.txt compare.txt ...]

A run's row holds its name (``vanilla/rank``, load.run_name), its path and its configuration (the
catalogue's header/config), then every SCORE of its own reports in RUN/evaluation/ (soundness,
stability, model; column "<point> | <score>"; soundness and stability start with their FAIL and flag
counts) and of the comparison reports given with --reports (column "<report directory>/<report> |
<score>"). A comparison report's SCOREs named "<catalogue>: ..." belong to that catalogue (the one
compared with the reference) and go to its row without the prefix; the others describe the comparison
as a whole and go to the row of every catalogue in it (its first line lists them). The rows replace
those of the same runs (by path) in --csv, the others are kept; new columns are added, older rows keep
them empty. Nothing is computed or judged here.
"""
import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):                   # run as a file: make the repository root importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h5py
import pandas as pd

from evaluate.load import run_name

PER_RUN = ("soundness", "stability", "model")


def read_scores(path):
    """``{name: value}`` of a report's SCORES block."""
    scores, inside = {}, False
    for line in Path(path).read_text().splitlines():
        if line == "SCORES":
            inside = True
        elif inside and not line.strip():
            break
        elif inside:
            name, value = line.strip().rsplit(maxsplit=1)
            scores[name] = pd.to_numeric(value, errors="coerce")
    return scores


def catalogues(path):
    """The catalogues a comparison report covers, from its first line ``<title>: <A> / <B> / ...``."""
    return Path(path).read_text().splitlines()[0].split(": ", 1)[1].split(" / ")


def evaluate(runs, csv, reports=()):
    """The rows of ``runs`` (see the module docstring), written into ``csv``; returns the whole table."""
    shared = {f"{r.parent.name}/{r.stem}": (catalogues(r), read_scores(r)) for r in map(Path, reports)}
    rows = []
    for path in (Path(p).resolve() for p in runs):
        name = run_name(path)
        with h5py.File(path / "catalogue.hdf5", "r") as f:
            row = {"run": name, "path": str(path), **json.loads(f["header/config"][()])}
        for point in PER_RUN:
            report = path / "evaluation" / f"{point}.txt"
            if report.exists():
                row.update({f"{point} | {k}": v for k, v in read_scores(report).items()})
            else:
                print(f"[sweep] {name}: no {report}")
        for key, (names, scores) in shared.items():
            if name not in names:
                continue
            for k, v in scores.items():
                owner = next((n for n in names if k.startswith(f"{n}: ")), None)
                if owner is None:
                    row[f"{key} | {k}"] = v
                elif owner == name:
                    row[f"{key} | {k[len(name) + 2:]}"] = v
        rows.append(row)
    new = pd.DataFrame(rows)
    table = new
    if Path(csv).exists():
        old = pd.read_csv(csv)
        table = pd.concat([old[~old["path"].isin(new["path"])], new], ignore_index=True)
    table.to_csv(csv, index=False)
    print(new.T.to_string())
    return table


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--path", nargs="+", required=True, help="the roadrunner run directories")
    p.add_argument("--csv", required=True, help="the table (rows of the same runs are replaced)")
    p.add_argument("--reports", nargs="*", default=[], help="comparison reports (science.txt, compare.txt)")
    args = p.parse_args()
    evaluate(args.path, args.csv, args.reports)


if __name__ == "__main__":
    main()
