#!/usr/bin/env python3
"""Run the roadrunner assignment pipeline on mock data.

Usage:
  python test_scripts/run_mock_pipeline.py --data-dir test_data/mock_snap --cov-type diagonal
"""

import argparse
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
from roadrunner.clustering.assignment.gmm import XGMMAssigner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="test_data/mock_snap")
    parser.add_argument("--cov-type", default="diagonal", choices=["diagonal", "full", "spherical"])
    parser.add_argument("--tol", type=float, default=5e-2)
    parser.add_argument("--search-factor", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = args.data_dir
    print(f"Data dir: {data_dir}")
    print(f"cov_type={args.cov_type}, tol={args.tol}, search_factor={args.search_factor}")

    print("Loading merger tree...")
    tree = pd.read_csv(os.path.join(data_dir, "merger_tree.csv"))

    print("Loading particles...")
    particles = np.load(os.path.join(data_dir, "particles.npz"))
    coords = particles["coords"]
    N = coords.shape[0]
    print(f"  {N} particles")

    print("Building HaloModel list...")
    halos = []
    for _, row in tree.iterrows():
        h = HaloModel.from_snapshot_row(row, model="kepler", comoving=False)
        halos.append(h)
    print(f"  {len(halos)} halos")

    print("Computing boundness...")
    halos = compute_halo_bound_particles(halos, coords, search_factor=args.search_factor)

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

    groups_sorted = sorted(
        [pop_idx[g] for g in seg.pruned_groups],
        key=len, reverse=True,
    )
    print(f"  {len(groups_sorted)} groups")

    print(f"Assigning particles ({args.cov_type} covariances)...")
    assigner = XGMMAssigner(
        cov_type=args.cov_type,
        max_iter=10,
        tol=args.tol,
        min_particles=10,
        reg_covar=1e-6,
        prior_type="",
        verbose=1,
    )
    newborn = np.arange(N, dtype=np.uint64)
    result = assigner.assign(halos, coords, newborn, groups_sorted)

    out_path = os.path.join(data_dir, "assignment.csv")
    result.particle_df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")
    print(f"  {len(result.particle_df)} particles assigned")
    print(f"  Unique Sub_tree_id assigned: {result.particle_df['Sub_tree_id'].nunique()}")
    print(f"  Particles assigned to -1: {(result.particle_df['Sub_tree_id'] == -1).sum()}")


if __name__ == "__main__":
    main()
