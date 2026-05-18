#!/usr/bin/env python3
"""Run the roadrunner assignment pipeline on mock data.

Reads:  test_data/mock_snap/merger_tree.csv
        test_data/mock_snap/particles.npz
Output: test_data/mock_snap/assignment.csv
        (columns: array_index, Sub_tree_id)
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from roadrunner.physics.halo_model import HaloModel
from roadrunner.physics.boundness import compute_halo_bound_particles
from roadrunner.physics.halo_ensemble import HaloEnsemble
from roadrunner.clustering.segmentation import HaloSegmenter
from roadrunner.clustering.assignment.gmm import GMMAssigner


def main():
    data_dir = "test_data/mock_snap"

    print("Loading merger tree...")
    tree = pd.read_csv(os.path.join(data_dir, "merger_tree.csv"))

    print("Loading particles...")
    particles = np.load(os.path.join(data_dir, "particles.npz"))
    coords = particles["coords"]  # (N, 6) = [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z]
    masses = particles["masses"]
    N = coords.shape[0]
    print(f"  {N} particles")

    print("Building HaloModel list...")
    halos = []
    for _, row in tree.iterrows():
        h = HaloModel.from_snapshot_row(row, model="kepler", comoving=False)
        halos.append(h)
    print(f"  {len(halos)} halos")

    print("Computing boundness...")
    halos = compute_halo_bound_particles(halos, coords, search_factor=2.0)

    print("Building ensemble...")
    ensemble = HaloEnsemble(halos)
    pop_idx = ensemble.populated_indices()
    print(f"  {len(pop_idx)} populated halos")

    csc_b, _ = ensemble.get_particles()
    candidates = [csc_b.column_indices[i] for i in pop_idx]

    print("Segmenting...")
    seg = HaloSegmenter(
        ensemble.positions[pop_idx],
        ensemble.virial_radii[pop_idx],
    )
    seg.overlap_groups().prune(candidates, min_particles=10, discard=False)

    if seg.pruned_groups is None or len(seg.pruned_groups) == 0:
        print("No groups found after pruning! Saving empty assignment.")
        assignment = pd.DataFrame({
            "array_index": np.arange(N, dtype=np.uint64),
            "Sub_tree_id": -1,
        })
        out_path = os.path.join(data_dir, "assignment.csv")
        assignment.to_csv(out_path, index=False)
        print(f"Saved {out_path}")
        return

    # Remap group indices from populated-halo space to full halo space
    groups_sorted = sorted(
        [pop_idx[g] for g in seg.pruned_groups],
        key=len, reverse=True,
    )
    print(f"  {len(groups_sorted)} groups")

    print("Assigning particles (diagonal covariances)...")
    assigner = GMMAssigner(
        cov_type="diagonal",
        max_iter=10,
        tol=5e-2,
        min_particles=10,
        reg_covar=1e-6,
        prior_type="",
        verbose=1,
    )
    newborn = np.arange(N, dtype=np.uint64)
    result = assigner.assign(
        halos, coords, newborn, groups_sorted,
    )

    out_path = os.path.join(data_dir, "assignment.csv")
    result.particle_df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")
    print(f"  {len(result.particle_df)} particles assigned")
    print(f"  Unique Sub_tree_id assigned: {result.particle_df['Sub_tree_id'].nunique()}")
    print(f"  Particles assigned to -1: {(result.particle_df['Sub_tree_id'] == -1).sum()}")


if __name__ == "__main__":
    main()
