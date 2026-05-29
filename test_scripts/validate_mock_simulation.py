#!/usr/bin/env python3
"""Validate mock simulation pipeline output.

Checks:
  1) Properties convergence: galaxy centers, velocities, masses, Rh, sigma
     are stable and physically consistent across snapshots.
  2) Particle tagging: ground truth vs pipeline assignment accuracy,
     including newborn particles.

Usage:
  python test_scripts/validate_mock_simulation.py
      --data-dir test_data/mock_simulation
      --output-dir test_output/mock_sim
"""

import argparse
import os

import h5py
import numpy as np


def _load_equiv(data_dir):
    path = os.path.join(data_dir, "equivalence.csv")
    eq = np.genfromtxt(path, delimiter=",", names=True)
    return eq, len(eq)


def _load_ground_truth(data_dir, snap_k):
    p = np.load(os.path.join(data_dir, f"particles_{snap_k:03d}.npz"))
    return p["coords"], p["galaxy_id"], p["born_snap"]


def _load_pipeline(output_dir, snap_k):
    with h5py.File(os.path.join(output_dir, "assignment.hdf5"), "r") as f:
        ha = f[f"snapshots/{snap_k}/hard_assignment"][:]
    pred_id = ha["Sub_tree_id"]
    with h5py.File(os.path.join(output_dir, "catalogue.hdf5"), "r") as f:
        gp = f[f"snapshots/{snap_k}/galaxy_properties"][:]
    with h5py.File(os.path.join(output_dir, "catalogue.hdf5"), "r") as f:
        dc = f[f"snapshots/{snap_k}/riley_criterion"][:]
    return pred_id, gp, dc


def _check_tagging(data_dir, output_dir, snap_k, snap_ids):
    coords, true_id, born_snap = _load_ground_truth(data_dir, snap_k)
    pred_id, gp, dc = _load_pipeline(output_dir, snap_k)

    n_total = len(true_id)
    n_pred = len(pred_id)
    assert n_total == n_pred, f"Snap {snap_k}: size mismatch {n_total} vs {n_pred}"

    correct = np.sum(pred_id == true_id)
    accuracy = correct / n_total

    newborn_mask = (born_snap == snap_k) & (snap_k > 0)
    n_newborn = newborn_mask.sum()
    newborn_correct = np.sum(pred_id[newborn_mask] == true_id[newborn_mask])
    newborn_acc = newborn_correct / n_newborn if n_newborn > 0 else 0.0

    initial_mask = born_snap == 0
    n_initial = initial_mask.sum()
    initial_correct = np.sum(pred_id[initial_mask] == true_id[initial_mask])
    initial_acc = initial_correct / n_initial if n_initial > 0 else 0.0

    return {
        "snap": snap_k,
        "n_total": n_total,
        "n_newborn": n_newborn,
        "accuracy": accuracy,
        "newborn_accuracy": newborn_acc,
        "initial_accuracy": initial_acc,
        "n_predicted_galaxies": len(np.unique(pred_id)),
        "n_true_galaxies": len(np.unique(true_id)),
    }


def _check_properties(data_dir, output_dir, snap_k, snap_ids, equiv):
    _, gp, dc = _load_pipeline(output_dir, snap_k)
    coords, true_id, born_snap = _load_ground_truth(data_dir, snap_k)

    # Expected displacement for this snapshot
    # physical pos = base + v_sys * t_raw, then * (1+z) for comoving
    # But we don't have t_raw here — compute from equiv time
    # t_k = equiv[snap_k]["time"] (cosmic time)
    # t_raw = t_k - (equiv[-1]["time"] - raw_cumulative_time[-1])

    stat = {
        "snap": snap_k,
        "n_galaxies": len(gp),
        "n_dynstate": len(dc),
    }

    if len(gp) == 0:
        return stat

    # Mass range
    stat["mass_min"] = float(gp["Mtot"].min())
    stat["mass_max"] = float(gp["Mtot"].max())
    stat["mass_med"] = float(np.median(gp["Mtot"]))

    # Half-mass radius range
    stat["rh_min"] = float(gp["rh"].min())
    stat["rh_max"] = float(gp["rh"].max())
    stat["rh_med"] = float(np.median(gp["rh"]))

    # Velocity dispersion range
    stat["sigma_min"] = float(gp["sigma"].min())
    stat["sigma_max"] = float(gp["sigma"].max())
    stat["sigma_med"] = float(np.median(gp["sigma"]))

    # Dynstate distribution
    dyn_counts = {}
    for sid in dc["dynstate"]:
        dyn_counts[int(sid)] = dyn_counts.get(int(sid), 0) + 1
    stat["dynstates"] = dyn_counts

    # Ground truth galaxy positions vs pipeline: mean offset
    # For the host galaxy (accretion_id = 1 usually)
    # Compute mean particle position per true galaxy, compare to pipeline center
    try:
        acc_id_idx = list(gp["Sub_tree_id"]).index(1)
        center_x = gp[acc_id_idx]["position_x"]
        center_y = gp[acc_id_idx]["position_y"]
        center_z = gp[acc_id_idx]["position_z"]
        # Compute ground truth mean position for galaxy 1
        true_mask = true_id == 1
        true_mean = coords[true_mask, :3].mean(axis=0)
        pos_err = np.sqrt((center_x - true_mean[0])**2 + (center_y - true_mean[1])**2 + (center_z - true_mean[2])**2)
        stat["host_pos_error"] = float(pos_err)
    except (ValueError, IndexError):
        stat["host_pos_error"] = -1.0

    return stat


def main():
    parser = argparse.ArgumentParser(description="Validate mock simulation pipeline")
    parser.add_argument("--data-dir", default="test_data/mock_simulation",
                        help="Mock simulation data directory")
    parser.add_argument("--output-dir", default="test_output/mock_sim",
                        help="Pipeline output directory")
    parser.add_argument("--n-snap", type=int, default=None,
                        help="Process first N snapshots (default: all)")
    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = args.output_dir
    equiv, n_snap = _load_equiv(data_dir)
    snap_ids = list(range(args.n_snap or n_snap))

    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("MOCK SIMULATION VALIDATION REPORT")
    report_lines.append("=" * 70)

    # ── 1. Particle tagging ────────────────────────────────────────
    report_lines.append("\n" + "-" * 70)
    report_lines.append("1. PARTICLE TAGGING ACCURACY")
    report_lines.append("-" * 70)
    report_lines.append(
        f"{'Snap':>4}  {'Total':>8} {'Newborn':>8} "
        f"{'Acc':>8} {'NewbornAcc':>10} {'InitAcc':>8} "
        f"{'NGalPred':>9} {'NGalTrue':>9}"
    )

    all_tag = []
    for snap_k in snap_ids:
        tag = _check_tagging(data_dir, output_dir, snap_k, snap_ids)
        all_tag.append(tag)
        report_lines.append(
            f"{tag['snap']:>4}  {tag['n_total']:>8} {tag['n_newborn']:>8} "
            f"{tag['accuracy']:>8.4f} {tag['newborn_accuracy']:>10.4f} "
            f"{tag['initial_accuracy']:>8.4f} "
            f"{tag['n_predicted_galaxies']:>9} {tag['n_true_galaxies']:>9}"
        )

    report_lines.append(f"\n  Overall accuracy range: "
                         f"[{min(t['accuracy'] for t in all_tag):.4f}, "
                         f"{max(t['accuracy'] for t in all_tag):.4f}]")

    # ── 2. Properties convergence ──────────────────────────────────
    report_lines.append("\n" + "-" * 70)
    report_lines.append("2. PROPERTIES CONVERGENCE")
    report_lines.append("-" * 70)
    report_lines.append(
        f"{'Snap':>4}  {'NGal':>5}  {'NDyn':>5}  "
        f"{'Mmin':>10}  {'Mmax':>10}  {'Mmed':>10}  "
        f"{'Rhmed':>7}  {'Sigmed':>7}  {'HostErr':>8}"
    )

    all_prop = []
    for snap_k in snap_ids:
        prop = _check_properties(data_dir, output_dir, snap_k, snap_ids, equiv)
        all_prop.append(prop)
        report_lines.append(
            f"{prop['snap']:>4}  {prop['n_galaxies']:>5}  {prop['n_dynstate']:>5}  "
            f"{prop['mass_min']:>10.2e}  {prop['mass_max']:>10.2e}  {prop['mass_med']:>10.2e}  "
            f"{prop['rh_med']:>7.2f}  {prop['sigma_med']:>7.2f}  "
            f"{prop['host_pos_error']:>8.2f}"
        )

    # ── 3. Stability checks ────────────────────────────────────────
    report_lines.append("\n" + "-" * 70)
    report_lines.append("3. STABILITY CHECKS")
    report_lines.append("-" * 70)

    n_gals = [p["n_galaxies"] for p in all_prop]
    if max(n_gals) == min(n_gals):
        report_lines.append(f"  Galaxy count: stable at {n_gals[0]} across all snapshots ✓")
    else:
        report_lines.append(f"  Galaxy count: VARIES ({min(n_gals)}-{max(n_gals)}) ✗")

    rh_meds = [p["rh_med"] for p in all_prop]
    rh_range = max(rh_meds) - min(rh_meds)
    rh_rel = rh_range / (np.median(rh_meds) + 1e-10)
    report_lines.append(f"  Median Rh: range={rh_range:.2f} kpccm, rel={rh_rel:.4f} "
                        f"{'✓' if rh_rel < 0.5 else '✗—may be unstable'}")

    sigma_meds = [p["sigma_med"] for p in all_prop]
    sig_range = max(sigma_meds) - min(sigma_meds)
    sig_rel = sig_range / (np.median(sigma_meds) + 1e-10)
    report_lines.append(f"  Median sigma: range={sig_range:.2f} km/s, rel={sig_rel:.4f} "
                        f"{'✓' if sig_rel < 0.5 else '✗—may be unstable'}")

    mass_meds = [p["mass_med"] for p in all_prop]
    m_rel = (max(mass_meds) - min(mass_meds)) / (np.median(mass_meds) + 1e-10)
    report_lines.append(f"  Median Mtot: rel_range={m_rel:.4f} "
                        f"{'✓' if m_rel < 0.3 else '✗—may be unstable'}")

    # Dynstate stability
    dyn_keys = set()
    for p in all_prop:
        dyn_keys.update(p["dynstates"].keys())
    report_lines.append(f"  Unique dynstate values observed: {sorted(dyn_keys)}")
    for dk in sorted(dyn_keys):
        counts = [p["dynstates"].get(dk, 0) for p in all_prop]
        if max(counts) == min(counts):
            report_lines.append(f"    Dynstate {dk}: stable at {counts[0]} ✓")
        else:
            report_lines.append(f"    Dynstate {dk}: ranges {min(counts)}-{max(counts)}")

    # ── 4. Summary ─────────────────────────────────────────────────
    report_lines.append("\n" + "=" * 70)
    avg_acc = np.mean([t["accuracy"] for t in all_tag])
    avg_newborn_acc = np.mean([t["newborn_accuracy"] for t in all_tag if t["newborn_accuracy"] > 0])
    report_lines.append("SUMMARY")
    report_lines.append(f"  Snapshots processed: {len(snap_ids)}")
    report_lines.append(f"  Mean overall accuracy: {avg_acc:.4f}")
    report_lines.append(f"  Mean newborn accuracy: {avg_newborn_acc:.4f}")
    report_lines.append(f"  Total particles at last snap: {all_tag[-1]['n_total']}")
    report_lines.append(f"  Median Mass range: {min(mass_meds):.2e} - {max(mass_meds):.2e}")
    report_lines.append(f"  Median Rh range: {min(rh_meds):.2f} - {max(rh_meds):.2f} kpccm")
    report_lines.append(f"  Median sigma range: {min(sigma_meds):.2f} - {max(sigma_meds):.2f} km/s")
    report_lines.append("=" * 70)

    report = "\n".join(report_lines)
    print(report)

    report_path = os.path.join(output_dir, "validation_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
