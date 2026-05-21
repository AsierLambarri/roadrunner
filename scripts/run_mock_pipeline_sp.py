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
from roadrunner.postprocessing.properties import compute_galaxy_properties, find_center
from roadrunner.postprocessing.mixing import compute_riley_criterion


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
    cat_w = HDF5CatalogueWriter(output_dir)
    part_w = HDF5ParticleWriter(output_dir, float_atol=1e-4)
    assign_w = HDF5AssignmentWriter(output_dir, float_atol=1e-4)
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
    )

    t0 = time.time()
    newborn = np.arange(N, dtype=np.uint64)

    ensemble, result = processor.process(
        tree, snap_data, newborn,
    )

    # ── Inline reduction (processor.reduce was removed in Part 1) ─
    particle_coords = np.column_stack([snap_data.positions, snap_data.velocities])
    df = result.particle_df

    galaxy_particles = {}
    for sid in df["Sub_tree_id"].unique():
        if sid == -1:
            continue
        mask = df["Sub_tree_id"] == sid
        galaxy_particles[int(sid)] = df.loc[mask, "array_index"].values

    galaxy_table = tree[["Sub_tree_id", "host_id", "mass",
                         "distance_to_acc_id"]].copy()
    galaxy_table.set_index("Sub_tree_id", inplace=True)
    host_row = tree[tree["Sub_tree_id"] == args.accretion_id]
    host_props = host_row.iloc[0] if not host_row.empty else tree.iloc[0]

    galaxy_centers = {}
    for sid, params in result.fitted_parameters.items():
        mean = params.get("mean")
        if mean is not None:
            galaxy_centers[int(sid)] = np.asarray(mean)

    properties = compute_galaxy_properties(
        accretion_id=args.accretion_id,
        particle_masses=masses,
        particle_coords=particle_coords,
        galaxy_particles=galaxy_particles,
        galaxy_table=galaxy_table,
        host_props=host_props,
        halo_model="kepler",
        n_los=3,
        galaxy_centers=galaxy_centers if galaxy_centers else None,
    )

    bound_csc, _ = ensemble.get_particles()
    galaxy_bound = {}
    bound_sid_to_idx = {sid: i for i, sid in enumerate(bound_csc.column_id)}
    for j, gid in enumerate(result.responsibilities.column_id):
        col = bound_sid_to_idx.get(gid)
        if col is not None:
            galaxy_bound[int(gid)] = bound_csc.column_indices[col]

    dynstate = compute_riley_criterion(
        main_id=args.accretion_id,
        particle_masses=masses,
        particle_coords=particle_coords,
        galaxy_allowed=galaxy_particles,
        galaxy_bound=galaxy_bound,
        redshift=0.0,
    )

    # ── I/O ──────────────────────────────────────────────────────
    cat_w.write_snapshot(0, 13.8, properties, dynstate, {})
    part_w.write_snapshot(0, 13.8, 0.0, snap_data)
    assign_w.write_snapshot(0, 13.8, result, bound_csc)

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
