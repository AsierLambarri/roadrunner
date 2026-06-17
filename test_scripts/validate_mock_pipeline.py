#!/usr/bin/env python3
"""Validate pipeline output against mock ground truth.

Compares:
  - Assignment accuracy (confusion after Hungarian matching)
  - Galaxy properties (half-mass radii, centre recovery)
Uses the ground-truth galaxy_id from the mock dataset.

Usage:
  python scripts/validate_mock_pipeline.py
"""

import sys
import os
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

warnings.filterwarnings("ignore")

import argparse

# Import what we need for ground-truth property computation
from roadrunner.postprocessing.properties import (
    half_mass_radius,
    find_center,
    velocity_dispersion,
)
from roadrunner.physics.halo_model import HaloModel
from roadrunner.physics.boundness import compute_halo_bound_particles
from roadrunner.physics.halo_ensemble import HaloEnsemble
from roadrunner.clustering.segmentation import HaloSegmenter
from roadrunner.clustering.assignment.gmm import XGMMAssigner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="test_data/mock_snap",
                        help="Path to mock dataset directory")
    args = parser.parse_args()
    data_dir = args.data_dir

    print("=" * 60)
    print(f"Mock Pipeline Validation — {data_dir}")
    print("=" * 60)

    print("\n1. Loading data...")
    tree = pd.read_csv(os.path.join(data_dir, "merger_tree.csv"))
    particles = np.load(os.path.join(data_dir, "particles.npz"))
    coords = particles["coords"]
    masses = particles["masses"]
    true_id = particles["galaxy_id"]
    N = coords.shape[0]
    print(f"   {N} particles, {len(tree)} galaxies")

    print("\n2. Running boundness → segment → assign...")
    halos = []
    for _, row in tree.iterrows():
        h = HaloModel.from_snapshot_row(row, model="kepler", comoving=False)
        halos.append(h)

    halos = compute_halo_bound_particles(halos, coords, search_factor=2.0)
    ensemble = HaloEnsemble(halos)
    pop_idx = ensemble.populated_indices()
    csc_b, _ = ensemble.get_particles()
    candidates = [csc_b.column_indices[i] for i in pop_idx]

    seg = HaloSegmenter(
        ensemble.positions[pop_idx],
        ensemble.virial_radii[pop_idx],
    )
    seg.overlap_groups().prune(candidates, min_particles=10, discard=False)
    groups_sorted = sorted(
        [pop_idx[g] for g in seg.pruned_groups],
        key=len, reverse=True,
    ) if seg.pruned_groups else []

    assigner = XGMMAssigner(
        cov_type="full", max_iter=10, tol=1e-2,
        min_particles=10, reg_covar=1e-6, prior_type="", verbose=0,
    )
    newborn = np.arange(N, dtype=np.uint64)
    result = assigner.assign(halos, coords, newborn, groups_sorted)
    pred_id = result.particle_df["Sub_tree_id"].values

    # ── 3. Assignment accuracy ─────────────────────────────────────
    print("\n3. Assignment accuracy:")

    u_true = np.unique(true_id)
    u_pred = np.unique(pred_id)
    cm = np.zeros((len(u_true), len(u_pred)), dtype=int)
    t_map = {t: i for i, t in enumerate(u_true)}
    p_map = {p: j for j, p in enumerate(u_pred)}
    for t, p in zip(true_id, pred_id):
        cm[t_map[t], p_map[p]] += 1

    row_ind, col_ind = linear_sum_assignment(-cm)
    matched = cm[row_ind, col_ind].sum()
    accuracy = matched / N
    print(f"   Hungarian-matched accuracy: {accuracy*100:.2f}% ({matched}/{N})")
    print(f"   Unassigned (pred == -1):    {(pred_id == -1).sum()}")

    # Per-class accuracy for the 5 largest true galaxies
    true_counts = pd.Series(true_id).value_counts()
    print("   Top 5 galaxies by size:")
    for tid in true_counts.index[:5]:
        i = t_map[tid]
        j = col_ind[i]
        tp = cm[i, j]
        fn = cm[i, :].sum() - tp
        print(f"     True G{tid:>2d} ({cm[i,:].sum():>5d} particles): "
              f"TP={tp:>5d} FN={fn:>5d} recall={tp/max(tp+fn,1):.3f}")

    # ── 4. Galaxy properties comparison (top 5) ────────────────────
    print("\n4. Galaxy properties (computed from pipeline assignment):")

    # Compute ground-truth properties per galaxy
    stats_rows = []
    for gid in np.unique(true_id):
        mask = true_id == gid
        gal_pos = coords[mask, :3]
        gal_vel = coords[mask, 3:6]
        gal_masses = masses[mask]
        center_pos, center_vel = find_center(gal_pos, gal_vel, gal_masses)
        centered = gal_pos - center_pos
        rh = half_mass_radius(centered, gal_masses, np.zeros(3), mass_fraction=0.5)
        sigma = velocity_dispersion(gal_vel - center_vel)
        stats_rows.append({
            "Sub_tree_id": gid,
            "N": len(gal_pos),
            "rh_kpc": rh,
            "sigma_kms": sigma,
        })
    ground_truth = pd.DataFrame(stats_rows)

    # Also compute from pipeline assignment
    pipe_rows = []
    for gid in np.unique(pred_id):
        if gid == -1:
            continue
        mask = pred_id == gid
        gal_pos = coords[mask, :3]
        gal_vel = coords[mask, 3:6]
        gal_masses = masses[mask]
        center_pos, center_vel = find_center(gal_pos, gal_vel, gal_masses)
        centered = gal_pos - center_pos
        rh = half_mass_radius(centered, gal_masses, np.zeros(3), mass_fraction=0.5)
        sigma = velocity_dispersion(gal_vel - center_vel)
        pipe_rows.append({
            "Sub_tree_id": gid,
            "N": len(gal_pos),
            "rh_kpc": rh,
            "sigma_kms": sigma,
        })
    pipeline = pd.DataFrame(pipe_rows)

    # Match via Hungarian
    for i in range(min(5, len(row_ind))):
        tid = u_true[row_ind[i]]
        pid = u_pred[col_ind[i]]

        gt = ground_truth[ground_truth["Sub_tree_id"] == tid].iloc[0]
        pp = pipeline[pipeline["Sub_tree_id"] == pid]
        if pp.empty:
            continue
        pp = pp.iloc[0]

        drh = abs(gt["rh_kpc"] - pp["rh_kpc"])
        dsig = abs(gt["sigma_kms"] - pp["sigma_kms"])
        print(f"   True G{tid:>2d} ↔ Pred G{pid:>2d}: "
              f"rh={pp['rh_kpc']:.2f} kpc (truth={gt['rh_kpc']:.2f}, "
              f"Δ={drh:.2f})  "
              f"σ={pp['sigma_kms']:.0f} km/s (truth={gt['sigma_kms']:.0f}, "
              f"Δ={dsig:.0f})")

    # ── 5. Summary statistics ──────────────────────────────────────
    print("\n5. Summary:")
    print(f"   {result.statistics}")
    print("\n   Validation complete.")


if __name__ == "__main__":
    main()
