"""Goal 7: one row per run for tuning: its configuration and every SCORE of its reports.

Reads RUN/evaluate/{soundness,stability,model}.txt (or <--out>/<run name>/ when the per-run points were
written there), and the SCOREs of the multi-catalogue reports given
with --reports (science.txt, compare.txt) whose names start with "<run name>: "; appends one row per
run to --csv (new columns are added, existing rows keep theirs empty).
"""
import json
from pathlib import Path

import h5py
import pandas as pd

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


def evaluate(runs, csv, reports=(), root=None):
    shared = {f"{Path(r).parent.name}/{Path(r).stem}": read_scores(r) for r in reports}
    rows = []
    for path in map(Path, runs):
        with h5py.File(path / "catalogue.hdf5", "r") as f:
            row = {"run": path.name, "path": str(path), **json.loads(f["header/config"][()])}
        for point in PER_RUN:
            report = (Path(root) / path.name if root else path / "evaluate") / f"{point}.txt"
            if report.exists():
                row.update({f"{point} | {k}": v for k, v in read_scores(report).items()})
        prefix = f"{path.name}: "
        for point, scores in shared.items():
            row.update({f"{point} | {k[len(prefix):]}": v for k, v in scores.items() if k.startswith(prefix)})
        rows.append(row)
    new = pd.DataFrame(rows)
    table = pd.concat([pd.read_csv(csv), new], ignore_index=True) if Path(csv).exists() else new
    table.to_csv(csv, index=False)
    print(new.T.to_string())
    return table
