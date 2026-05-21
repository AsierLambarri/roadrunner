#!/usr/bin/env python3
"""Process a single mock snapshot using SnapshotProcessor.

Compares output with the old pipeline for validation.

Usage:
  python scripts/run_mock_pipeline_sp.py --data-dir test_data/mock_snap_spherical \
      --output-dir test_data/mock_comparison_spherical/sp_processor \
      --cov-type spherical
"""

import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from roadrunner._mcf_types import SnapshotData
from roadrunner.clustering.assignment.gmm import GMMAssigner
from roadrunner.io.hdf5_catalogue import HDF5CatalogueWriter
from roadrunner.io.hdf5_particles import HDF5ParticleWriter
from roadrunner.io.hdf5_assignment import HDF5AssignmentWriter
from roadrunner.io.logging import RunLogger
from roadrunner.pipeline.snapshot_processor import SnapshotProcessor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="test_data/mock_snap_spherical")
    parser.add_argument("--output-dir",
                        default="test_data/mock_comparison_spherical/sp_processor")
    parser.add_argument("--cov-type", default="spherical",
                        choices=["full", "diagonal", "spherical"])
    parser.add_argument("--accretion-id", type=int, default=1)
    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print(f"Data dir: {data_dir}")
    print(f"Output dir: {output_dir}")
    print(f"cov_type: {args.cov_type}")

    # ── Load data ────────────────────────────────────────────────
    tree = pd.read_csv(os.path.join(data_dir, "merger_tree.csv"))
    particles = np.load(os.path.join(data_dir, "particles.npz"))
    coords = particles["coords"]
    masses = particles["masses"]
    N = coords.shape[0]

    snap_data = SnapshotData(
        indices=np.arange(N, dtype=np.uint64),
        masses=masses,
        positions=coords[:, :3],
        velocities=coords[:, 3:6],
        redshift=0.0,
        time=13.8,
    )

    if "host_id" not in tree.columns:
        tree["host_id"] = -1
    if "distance_to_acc_id" not in tree.columns:
        tree["distance_to_acc_id"] = 0.0

    # ── I/O writers ──────────────────────────────────────────────
    cat_w = HDF5CatalogueWriter(output_dir, mode="w-")
    part_w = HDF5ParticleWriter(output_dir, mode="w-", float_atol=1e-4)
    assign_w = HDF5AssignmentWriter(output_dir, mode="w-", float_atol=1e-4)
    logger = RunLogger(os.path.join(output_dir, "run.log"))

    # ── Header ───────────────────────────────────────────────────
    cat_w.write_header(
        accretion_id=args.accretion_id,
        snapshots=[0],
        config_dict={"halo_model": "kepler", "cov_type": args.cov_type},
        merger_tree_df=tree,
        equivalence_df=pd.read_csv(os.path.join(data_dir, "equivalence.csv")),
    )
    logger.write_header({
        "output_dir": output_dir,
        "halo_model": "kepler",
        "cov_type": args.cov_type,
    })

    # ── SnapshotProcessor ────────────────────────────────────────
    assigner = GMMAssigner(
        cov_type=args.cov_type, max_iter=10, tol=1e-2,
        min_particles=10, reg_covar=1e-6, prior_type="", verbose=0,
    )
    processor = SnapshotProcessor(
        assigner=assigner,
        halo_model="kepler",
        accretion_id=args.accretion_id,
        n_los=3,
        cat_writer=cat_w,
        part_writer=part_w,
        assign_writer=assign_w,
    )

    t0 = time.time()
    newborn = np.arange(N, dtype=np.uint64)

    halos, ensemble, result = processor.process(
        tree, coords, masses, newborn,
    )
    properties, dynstate = processor.reduce(
        tree, snap_data, ensemble, result, {},
    )

    total = time.time() - t0

    # ── Finalize ─────────────────────────────────────────────────
    cat_w.write_finalize(
        pd.DataFrame(), pd.DataFrame(),
    )
    logger.write_snapshot({
        "snap": 0,
        "runtime": f"{total:.3f}s",
        "z": 0.0,
        "load": 0.0,
        "process": total,
        "reduction": 0.0,
        "bound": ensemble.nstars,
        "groups": result.statistics.get("groups", 0),
        **result.statistics,
    })
    logger.write_summary()

    print(f"\nPipeline complete in {total:.3f}s")
    print(f"  unassigned={result.statistics.get('unassigned', '?')}")
    print(f"  conf={result.statistics.get('avg_conf', '?'):.4f}")
    print(f"  cond={result.statistics.get('avg_cond', '?'):.2f}")
    print(f"  retention={result.statistics.get('avg_retention', '?'):.2f}")


if __name__ == "__main__":
    main()
