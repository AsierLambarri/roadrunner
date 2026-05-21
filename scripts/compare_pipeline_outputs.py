#!/usr/bin/env python3
"""Compare outputs from SnapshotProcessor and old pipeline.

Reads:  test_data/mock_comparison_spherical/{sp_processor,old_pipeline}/
Writes: test_data/mock_comparison_spherical/comparison_report.txt
"""

import argparse
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from roadrunner.io.hdf5_reader import HDF5CatalogueReader
from roadrunner.physics.scaler import StandardScaler


def compare_arrays(name, a, b, atol=1e-4):
    """Compare two arrays, return list of discrepancy lines."""
    lines = []
    if a.shape != b.shape:
        lines.append(f"  SHAPE MISMATCH: {a.shape} vs {b.shape}")
    elif not np.allclose(a, b, atol=atol, equal_nan=True):
        diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
        max_diff = diff.max()
        mean_diff = diff.mean()
        lines.append(f"  VALUES DIFFER: max_diff={max_diff:.6e}, "
                     f"mean_diff={mean_diff:.6e}, "
                     f"n_diff={(diff > atol).sum()}/{diff.size}")
    else:
        lines.append(f"  OK (max_diff={np.max(np.abs(a-b)):.2e}, "
                     f"shape={a.shape})")
    return lines


def compare_hdf5_files(path_a, path_b, label, atol=1e-4):
    """Compare all datasets in two HDF5 files."""
    lines = [f"\n{'='*70}", f"Comparing {label}", f"{'='*70}"]
    lines.append(f"  File A: {path_a}")
    lines.append(f"  File B: {path_b}")

    if not os.path.exists(path_a):
        lines.append("  MISSING: File A not found")
        return lines
    if not os.path.exists(path_b):
        lines.append("  MISSING: File B not found")
        return lines

    with h5py.File(path_a, "r") as fa, h5py.File(path_b, "r") as fb:
        # Collect all dataset paths
        def visit_items(f, prefix=""):
            items = {}
            for key in f:
                path = f"{prefix}/{key}" if prefix else key
                obj = f[key]
                if isinstance(obj, h5py.Dataset):
                    items[path] = obj
                elif isinstance(obj, h5py.Group):
                    items.update(visit_items(obj, path))
            return items

        items_a = visit_items(fa)
        items_b = visit_items(fb)

        keys_a = set(items_a.keys())
        keys_b = set(items_b.keys())

        only_a = keys_a - keys_b
        only_b = keys_b - keys_a
        common = keys_a & keys_b

        if only_a:
            lines.append(f"  Only in A ({len(only_a)}): {sorted(only_a)[:10]}")
        if only_b:
            lines.append(f"  Only in B ({len(only_b)}): {sorted(only_b)[:10]}")

        for key in sorted(common):
            da, db = items_a[key], items_b[key]
            val_a = da[()]
            val_b = db[()]

            # Skip metadata/time attrs
            if isinstance(val_a, str) or isinstance(val_b, str):
                if val_a == val_b:
                    lines.append(f"  {key}: OK (string match)")
                else:
                    lines.append(f"  {key}: STRING MISMATCH: "
                                 f"'{str(val_a)[:50]}' vs '{str(val_b)[:50]}'")
                continue

            if isinstance(val_a, bytes) or isinstance(val_b, bytes):
                lines.append(f"  {key}: bytes (skipped)")
                continue

            # Try as structured array (record)
            if da.dtype.names is not None:
                common_names = set(da.dtype.names) & set(db.dtype.names)
                only_a = set(da.dtype.names) - common_names
                only_b = set(db.dtype.names) - common_names
                if only_a:
                    lines.append(f"  {key}: fields only in A: {only_a}")
                if only_b:
                    lines.append(f"  {key}: fields only in B: {only_b}")
                for name in sorted(common_names):
                    sub_key = f"{key}/{name}"
                    sub_a = val_a[name]
                    sub_b = val_b[name]
                    sub_lines = compare_arrays(sub_key, sub_a, sub_b, atol)
                    lines.extend(sub_lines)
            elif da.shape == () and db.shape == ():
                # Scalars
                if np.allclose(np.atleast_1d(val_a), np.atleast_1d(val_b),
                               atol=atol, equal_nan=True):
                    lines.append(f"  {key}: OK ({val_a})")
                else:
                    lines.append(f"  {key}: MISMATCH ({val_a} vs {val_b})")
            else:
                sub_lines = compare_arrays(key, val_a, val_b, atol)
                lines.extend(sub_lines)

    return lines


def compare_particle_data(base_dir, atol=1e-4):
    """Compare particles.hdf5 — scaled data with inverse transform."""
    lines = [f"\n{'='*70}", "Particle data comparison", f"{'='*70}"]

    for snap in ["0"]:
        for prefix, label in [("sp_processor", "A (SnapshotProcessor)"),
                               ("old_pipeline", "B (Old pipeline)")]:
            p = os.path.join(base_dir, prefix, "particles.hdf5")
            if not os.path.exists(p):
                lines.append(f"  MISSING: {label} — {p}")
                continue
            with h5py.File(p, "r") as hf:
                grp = hf[f"/snapshots/{snap}"]
                mean = grp["scaler/mean"][:]
                scale = grp["scaler/scale"][:]
                pos = grp["positions"][:]
                vel = grp["velocities"][:]
                masses = grp["masses"][:]
                indices = grp["indices"][:]
                lines.append(f"  {label}: {len(masses)} particles, "
                             f"pos={pos.shape}, vel={vel.shape}")

    # Compare
    for prefix_a, prefix_b in [("sp_processor", "old_pipeline")]:
        fa = os.path.join(base_dir, prefix_a, "particles.hdf5")
        fb = os.path.join(base_dir, prefix_b, "particles.hdf5")
        if not os.path.exists(fa) or not os.path.exists(fb):
            continue
        with h5py.File(fa, "r") as ha, h5py.File(fb, "r") as hb:
            ga = ha[f"/snapshots/{snap}"]
            gb = hb[f"/snapshots/{snap}"]
            for key in ["indices", "masses", "positions", "velocities",
                         "scaler/mean", "scaler/scale"]:
                sub_lines = compare_arrays(
                    f"particles/{snap}/{key}",
                    ga[key][:], gb[key][:], atol,
                )
                lines.extend(sub_lines)

    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir",
                        default="test_data/mock_comparison_spherical")
    parser.add_argument("--atol", type=float, default=1e-4)
    args = parser.parse_args()

    base = args.base_dir
    atol = args.atol
    report_path = os.path.join(base, "comparison_report.txt")

    all_lines = []
    all_lines.append("Pipeline Output Comparison Report")
    all_lines.append(f"SnapshotProcessor vs Old Pipeline")
    all_lines.append(f"Tolerance: atol={atol}")
    all_lines.append(f"Date: {__import__('time').strftime('%Y-%m-%d %H:%M:%S')}")
    all_lines.append("")

    # 1. catalogue.hdf5
    all_lines.extend(compare_hdf5_files(
        os.path.join(base, "sp_processor", "catalogue.hdf5"),
        os.path.join(base, "old_pipeline", "catalogue.hdf5"),
        "catalogue.hdf5", atol,
    ))

    # 2. assignment.hdf5
    all_lines.extend(compare_hdf5_files(
        os.path.join(base, "sp_processor", "assignment.hdf5"),
        os.path.join(base, "old_pipeline", "assignment.hdf5"),
        "assignment.hdf5", atol,
    ))

    # 3. particles.hdf5
    all_lines.extend(compare_particle_data(base, atol))

    # 4. Assignment statistics summary
    all_lines.append(f"\n{'='*70}")
    all_lines.append("Assignment statistics (from run.log)")
    all_lines.append(f"{'='*70}")
    for prefix, label in [("sp_processor", "SnapshotProcessor"),
                           ("old_pipeline", "Old pipeline")]:
        log_path = os.path.join(base, prefix, "run.log")
        if os.path.exists(log_path):
            with open(log_path) as f:
                content = f.read()
            all_lines.append(f"\n  --- {label} ---")
            # Extract stats line
            for line in content.split("\n"):
                if "conf=" in line or "cond=" in line or "unassigned" in line:
                    all_lines.append(f"  {line.strip()}")

    report = "\n".join(all_lines)
    with open(report_path, "w") as f:
        f.write(report + "\n")
    print(f"Report saved to {report_path}")
    print(report)


if __name__ == "__main__":
    main()
