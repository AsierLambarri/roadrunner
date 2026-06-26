#!/usr/bin/env python3
"""Compare outputs from SnapshotProcessor and old pipeline against each
other and against ground truth.

Reads:  test_data/mock_comparison_spherical/{sp_processor,old_pipeline}/
        test_data/mock_snap_spherical/  (ground truth)
Writes: test_data/mock_comparison_spherical/comparison_report.txt
"""

import argparse
import os
import sys

import h5py
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from roadrunner.physics.scaler import StandardScaler
from roadrunner.postprocessing.properties import half_mass_radius, velocity_dispersion, find_center


def compare_arrays(name, a, b, atol=1e-4):
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
            if isinstance(val_a, str) or isinstance(val_b, str):
                if val_a == val_b:
                    lines.append(f"  {key}: OK (string match)")
                else:
                    lines.append(f"  {key}: STRING MISMATCH")
                continue
            if isinstance(val_a, bytes) or isinstance(val_b, bytes):
                lines.append(f"  {key}: bytes (skipped)")
                continue
            if da.dtype.names is not None:
                common_names = set(da.dtype.names) & set(db.dtype.names)
                only_a_f = set(da.dtype.names) - common_names
                only_b_f = set(db.dtype.names) - common_names
                if only_a_f:
                    lines.append(f"  {key}: fields only in A: {only_a_f}")
                if only_b_f:
                    lines.append(f"  {key}: fields only in B: {only_b_f}")
                for name in sorted(common_names):
                    sub_lines = compare_arrays(f"{key}/{name}", val_a[name], val_b[name], atol)
                    lines.extend(sub_lines)
            elif da.shape == () and db.shape == ():
                if np.allclose(np.atleast_1d(val_a), np.atleast_1d(val_b), atol=atol, equal_nan=True):
                    lines.append(f"  {key}: OK ({val_a})")
                else:
                    lines.append(f"  {key}: MISMATCH ({val_a} vs {val_b})")
            else:
                sub_lines = compare_arrays(key, val_a, val_b, atol)
                lines.extend(sub_lines)
    return lines


def compare_particle_data(base_dir, atol=1e-4):
    lines = [f"\n{'='*70}", "Particle data comparison", f"{'='*70}"]
    snap = "0"
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
                sub_lines = compare_arrays(f"particles/{snap}/{key}", ga[key][:], gb[key][:], atol)
                lines.extend(sub_lines)
    return lines


def compare_against_ground_truth(base_dir, data_dir, atol=1e-4):
    """Compare both pipelines against ground truth data."""
    lines = [f"\n{'='*70}", "Comparison against ground truth", f"{'='*70}"]

    # Load ground truth
    particles = np.load(os.path.join(data_dir, "particles.npz"))
    tree = pd.read_csv(os.path.join(data_dir, "merger_tree.csv"))
    coords = particles["coords"]
    masses = particles["masses"]
    true_id = particles["galaxy_id"]
    N = coords.shape[0]
    u_true = np.unique(true_id)

    lines.append(f"  Ground truth: {N} particles, {len(u_true)} galaxies")

    for prefix, label in [("sp_processor", "SnapshotProcessor"),
                           ("old_pipeline", "Old pipeline")]:
        lines.append(f"\n  --- {label} ---")
        assign_path = os.path.join(base_dir, prefix, "assignment.hdf5")
        if not os.path.exists(assign_path):
            lines.append(f"    No assignment.hdf5 found, skipping")
            continue

        with h5py.File(assign_path, "r") as hf:
            hard_rec = hf["/snapshots/0/hard_assignment"][()]
            if isinstance(hard_rec, np.void):
                pred_id = np.array([hard_rec["Sub_tree_id"]])
            else:
                pred_id = hard_rec["Sub_tree_id"]

        pred_id = np.asarray(pred_id, dtype=np.int64)

        # ── 1. Assignment accuracy ──────────────────────────────
        u_pred = np.unique(pred_id)
        cm = np.zeros((len(u_true), len(u_pred)), dtype=int)
        t_map = {t: i for i, t in enumerate(u_true)}
        p_map = {p: j for j, p in enumerate(u_pred)}
        for t, p in zip(true_id, pred_id):
            cm[t_map[t], p_map[p]] += 1
        row_ind, col_ind = linear_sum_assignment(-cm)
        matched = cm[row_ind, col_ind].sum()
        accuracy = matched / N
        lines.append(f"    Accuracy: {accuracy*100:.2f}% ({matched}/{N})")
        lines.append(f"    Unassigned (pred == -1): {(pred_id == -1).sum()}")

        # Per-class top 3
        true_counts = pd.Series(true_id).value_counts()
        lines.append("    Top 3 galaxies:")
        for tid in true_counts.index[:3]:
            i = t_map[tid]
            j = col_ind[i]
            tp = cm[i, j]
            fn = cm[i, :].sum() - tp
            lines.append(f"      G{tid:>2d} ({cm[i,:].sum():>5d} particles): "
                         f"TP={tp:>5d}  FN={fn:>5d}  "
                         f"recall={tp/max(tp+fn,1):.3f}")

        # ── 2. Galaxy properties vs ground truth ────────────────
        lines.append("\n    Galaxy properties (rh, sigma) vs ground truth:")

        # Ground-truth properties
        gt_props = {}
        for gid in u_true:
            mask = true_id == gid
            pos = coords[mask, :3]
            vel = coords[mask, 3:6]
            m = masses[mask]
            cp, cv = find_center(pos, vel, m)
            rh = half_mass_radius(pos - cp, m, np.zeros(3), mass_fraction=0.5)
            sigma = velocity_dispersion(vel - cv)
            gt_props[gid] = {"rh": rh, "sigma": sigma}

        # Predicted properties (from SP pipeline's properties dataframe)
        cat_path = os.path.join(base_dir, prefix, "catalogue.hdf5")
        if os.path.exists(cat_path):
            try:
                with h5py.File(cat_path, "r") as hf:
                    if "snapshots/0/galaxy_properties" in hf:
                        props_rec = hf["snapshots/0/galaxy_properties"][()]
                        if props_rec.dtype.names is not None:
                            pred_sub = props_rec["Sub_tree_id"]
                            idx_rh = list(props_rec.dtype.names).index("rh") if "rh" in props_rec.dtype.names else None
                            idx_sig = list(props_rec.dtype.names).index("sigma") if "sigma" in props_rec.dtype.names else None
                            pred_rh = props_rec["rh"] if idx_rh is not None else (props_rec["Rhp"] if "Rhp" in props_rec.dtype.names else None)
                            pred_sigma = props_rec["sigma"] if idx_sig is not None else None
                            if pred_rh is not None:
                                for gid in u_true:
                                    j = np.where(pred_sub == gid)[0]
                                    if len(j) == 0:
                                        continue
                                    j = j[0]
                                    drh = abs(pred_rh[j] - gt_props[gid]["rh"])
                                    dsig = abs(pred_sigma[j] - gt_props[gid]["sigma"]) if pred_sigma is not None else -1
                                    lines.append(
                                        f"      G{gid:>2d}: rh_pred={pred_rh[j]:.3f} "
                                        f"rh_true={gt_props[gid]['rh']:.3f} "
                                        f"Δ={drh:.4f}  "
                                        + (f"σ_pred={pred_sigma[j]:.1f} "
                                           f"σ_true={gt_props[gid]['sigma']:.1f} "
                                           f"Δ={dsig:.1f}" if pred_sigma is not None else ""))
                            else:
                                lines.append("      No rh column in galaxy_properties")
                        else:
                            lines.append("      galaxy_properties is not a record array")
                    else:
                        lines.append("      No galaxy_properties dataset found in catalogue")
            except Exception as e:
                lines.append(f"      Error reading properties: {e}")
        else:
            lines.append("      No catalogue.hdf5 found, skipping properties")

        # ── 3. GMM means vs true centers ────────────────────────
        lines.append("\n    GMM mean vs true center (first 5 matched galaxies):")

        # Match via Hungarian
        matched_rows = {}
        for i, j in zip(row_ind, col_ind):
            matched_rows[int(u_true[i])] = int(u_pred[j])

        with h5py.File(assign_path, "r") as hf:
            gal_grp = hf[f"/snapshots/0/galaxies"]
            count = 0
            for true_gid in list(matched_rows.keys())[:5]:
                pred_gid = matched_rows[true_gid]
                grp_path = str(pred_gid)
                if grp_path not in gal_grp:
                    continue
                grp = gal_grp[grp_path]
                if "mean" not in grp:
                    continue
                mean_pred = grp["mean"][:]
                true_center = np.concatenate([
                    tree[tree["Sub_tree_id"] == true_gid][
                        ["position_x", "position_y", "position_z"]
                    ].values[0],
                    tree[tree["Sub_tree_id"] == true_gid][
                        ["velocity_x", "velocity_y", "velocity_z"]
                    ].values[0],
                ])
                diff = np.linalg.norm(mean_pred - true_center)
                lines.append(f"      G{true_gid:>2d}: |Δ|={diff:.2f} "
                             f"(mean_pred={mean_pred[:3].round(1)} "
                             f"center_true={true_center[:3].round(1)})")
                count += 1

    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default="test_data/mock_comparison_spherical")
    parser.add_argument("--data-dir", default="test_data/mock_snap_spherical")
    parser.add_argument("--atol", type=float, default=1e-4)
    args = parser.parse_args()

    base, data_dir, atol = args.base_dir, args.data_dir, args.atol
    report_path = os.path.join(base, "comparison_report.txt")

    all_lines = [
        "Pipeline Output Comparison Report",
        f"SnapshotProcessor vs Old Pipeline  |  Ground truth: {data_dir}",
        f"Tolerance: atol={atol}",
        f"Date: {__import__('time').strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]

    # 1. HDF5 file comparison (SP vs old)
    all_lines.extend(compare_hdf5_files(
        os.path.join(base, "sp_processor", "catalogue.hdf5"),
        os.path.join(base, "old_pipeline", "catalogue.hdf5"),
        "catalogue.hdf5 (SP vs Old)", atol,
    ))
    all_lines.extend(compare_hdf5_files(
        os.path.join(base, "sp_processor", "assignment.hdf5"),
        os.path.join(base, "old_pipeline", "assignment.hdf5"),
        "assignment.hdf5 (SP vs Old)", atol,
    ))
    all_lines.extend(compare_particle_data(base, atol))

    # 2. Both pipelines vs ground truth
    all_lines.extend(compare_against_ground_truth(base, data_dir, atol))

    # 3. Statistics summary
    all_lines.append(f"\n{'='*70}")
    all_lines.append("Pipeline statistics (from run.log)")
    all_lines.append(f"{'='*70}")
    for prefix, label in [("sp_processor", "SnapshotProcessor"),
                           ("old_pipeline", "Old pipeline")]:
        log_path = os.path.join(base, prefix, "run.log")
        if os.path.exists(log_path):
            with open(log_path) as f:
                content = f.read()
            all_lines.append(f"\n  --- {label} ---")
            for line in content.split("\n"):
                if "conf=" in line or "cond=" in line or "unassigned" in line:
                    all_lines.append(f"  {line.strip()}")

    report = "\n".join(all_lines)
    with open(report_path, "w") as f:
        f.write(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
