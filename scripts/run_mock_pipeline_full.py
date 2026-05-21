#!/usr/bin/env python3
"""End-to-end pipeline on a mock dataset with checkpoint+restart.

Flow per snapshot:
  1. Save checkpoint (BEFORE processing)
  2. Boundness → segment → GMM assign → I/O
  3. On crash, restart re-does the failed snapshot.

Usage:
  python scripts/run_mock_pipeline_full.py                     # fresh run
  python scripts/run_mock_pipeline_full.py --resume            # resume
  python scripts/run_mock_pipeline_full.py --fail-on 1         # simulate crash
"""

import argparse
import os
import shutil
import sys
import time
import warnings

import numpy as np
import pandas as pd

from roadrunner.clustering.sparse import SparseCSC

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from roadrunner.physics.halo_model import HaloModel
from roadrunner.physics.boundness import compute_halo_bound_particles
from roadrunner.physics.halo_ensemble import HaloEnsemble
from roadrunner.clustering.segmentation import HaloSegmenter
from roadrunner.clustering.assignment.gmm import GMMAssigner
from roadrunner.io.hdf5_catalogue import HDF5CatalogueWriter
from roadrunner.io.hdf5_particles import HDF5ParticleWriter
from roadrunner.io.hdf5_assignment import HDF5AssignmentWriter
from roadrunner.io.serialization import save_checkpoint, load_checkpoint
from roadrunner.io.logging import RunLogger, format_runtime
from roadrunner._exceptions import RestartError

def load_mock_snapshot(data_dir, snapshot_index):
    """Load the same data for both snapshots (duplicate)."""
    tree = pd.read_csv(os.path.join(data_dir, "merger_tree.csv"))
    particles = np.load(os.path.join(data_dir, "particles.npz"))
    return tree, particles


def build_halos(tree):
    return [
        HaloModel.from_snapshot_row(row, model="kepler", comoving=False)
        for _, row in tree.iterrows()
    ]


def build_snapshot_data(coords, masses):
    from roadrunner._mcf_types import SnapshotData
    return SnapshotData(
        indices=np.arange(coords.shape[0], dtype=np.uint64),
        masses=masses,
        positions=coords[:, :3],
        velocities=coords[:, 3:6],
        redshift=0.0,
        time=13.8,
    )


def process_snapshot(snapshot_id, tree, coords, masses, cat_writer,
                     part_writer, assign_writer, logger, previous_resp,
                     t_start, cov_type="full"):
    """Run boundness → segment → assign → I/O for one snapshot.
    Returns updated previous_resp.
    """
    t_snap = time.time()
    halos = build_halos(tree)
    halos = compute_halo_bound_particles(halos, coords, search_factor=2.0)
    ensemble = HaloEnsemble(halos)
    csc_b, _ = ensemble.get_particles()
    pop_idx = ensemble.populated_indices()
    candidates = [csc_b.column_indices[i] for i in pop_idx]
    process_time = time.time() - t_snap

    seg = HaloSegmenter(
        ensemble.positions[pop_idx],
        ensemble.virial_radii[pop_idx],
    )
    seg.overlap_groups().prune(candidates, min_particles=10, discard=False)
    groups_sorted = sorted(
        [pop_idx[g] for g in seg.pruned_groups],
        key=len, reverse=True,
    ) if seg.pruned_groups else []

    assigner = GMMAssigner(
        cov_type=cov_type, max_iter=10, tol=1e-2,
        min_particles=10, reg_covar=1e-6, prior_type="", verbose=0,
    )
    newborn = np.arange(coords.shape[0], dtype=np.uint64)
    result = assigner.assign(
        halos, coords, newborn, groups_sorted,
        previous_resp=previous_resp,
    )
    assign_time = time.time() - t_snap

    # ── Catalogue ───
    snap_data = build_snapshot_data(coords, masses)
    cat_writer.write_snapshot(
        snapshot_id=snapshot_id, time=13.8,
        properties_df=pd.DataFrame(),
        dynstate_df=pd.DataFrame(),
        satellites_map={},
    )
    part_writer.write_snapshot(
        snapshot_id, 13.8, 0.0, snap_data,
    )
    assign_writer.write_snapshot(
        snapshot_id, 13.8, result, csc_b,
    )
    io_time = time.time() - t_snap

    # ── Log ───
    total_elapsed = time.time() - t_start
    stats = {
        "snap": snapshot_id,
        "runtime": format_runtime(total_elapsed),
        "z": 0.0,
        "load": 0.0,
        "process": assign_time,
        "reduction": 0.0,
        "bound": ensemble.nstars,
        "groups": len(groups_sorted),
    }
    stats.update(result.statistics)
    logger.write_snapshot(stats)

    print(f"  Snapshot {snapshot_id}: "
          f"unassigned={stats.get('unassigned','?')}, "
          f"conf={stats.get('avg_conf','?'):.4f}, "
          f"cond={stats.get('avg_cond','?'):.1f}, "
          f"time={format_runtime(total_elapsed)}")

    # result.responsibilities is already a SparseCSC — use directly
    return result.responsibilities


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="test_data/mock_snap_tight",
                        help="Path to mock dataset directory")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-on", type=int, default=None,
                        help="Simulate crash on this snapshot")
    parser.add_argument("--n-snapshots", type=int, default=2,
                        help="Number of snapshots to process")
    parser.add_argument("--cov-type", default="full",
                        choices=["full", "diagonal", "spherical"],
                        help="GMM covariance type")
    args = parser.parse_args()

    DATA_DIR = args.data_dir
    N_SNAPSHOTS = args.n_snapshots
    fail_on = args.fail_on

    OUTPUT_DIR = os.path.join(DATA_DIR, "output")
    CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "checkpoint.zst")
    LOG_PATH = os.path.join(OUTPUT_DIR, "run.log")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Determine starting point ─────────────────────────────────
    resume = args.resume
    start_snapshot = 0

    if resume:
        try:
            ckpt = load_checkpoint(CHECKPOINT_PATH)
            start_snapshot = ckpt["last_snapshot"]  # re-do this snapshot
            print(f"Resumed from checkpoint. Re-processing snapshot {start_snapshot}")
        except RestartError:
            print("No valid checkpoint, starting fresh.")
            resume = False

    # ── Clean output dir on fresh run or resume ──────────────────
    # Both cases: we rebuild output files from scratch.
    for f in os.listdir(OUTPUT_DIR):
        fp = os.path.join(OUTPUT_DIR, f)
        if os.path.isfile(fp):
            os.remove(fp)

    # ── Init writers ─────────────────────────────────────────────
    cat_writer = HDF5CatalogueWriter(OUTPUT_DIR, mode="w-")
    part_writer = HDF5ParticleWriter(OUTPUT_DIR, mode="w-", float_atol=1e-4)
    assign_writer = HDF5AssignmentWriter(OUTPUT_DIR, mode="w-", float_atol=1e-4)
    logger = RunLogger(LOG_PATH)
    logger.write_header({
        "output_dir": OUTPUT_DIR,
        "halo_model": "kepler",
        "cov_type": args.cov_type,
    })

    # ── Load data (same for both snapshots) ──────────────────────
    tree, particles = load_mock_snapshot(DATA_DIR, 0)
    coords = particles["coords"]
    masses = particles["masses"]

    # ── Write header once ────────────────────────────────────────
    cat_writer.write_header(
        accretion_id=1,
        snapshots=list(range(N_SNAPSHOTS)),
        config_dict={"halo_model": "kepler", "cov_type": args.cov_type},
        merger_tree_df=tree,
        equivalence_df=pd.read_csv(os.path.join(DATA_DIR, "equivalence.csv")),
    )

    # ── Snapshot loop ────────────────────────────────────────────
    t_start = time.time()
    previous_resp = None

    for snap_id in range(start_snapshot, N_SNAPSHOTS):
        # Save checkpoint BEFORE processing
        ckpt = {"last_snapshot": snap_id, "previous_resp": previous_resp}
        save_checkpoint(CHECKPOINT_PATH, ckpt)

        print(f"Processing snapshot {snap_id}...")

        # Simulate crash (for testing restart)
        if fail_on is not None and snap_id >= fail_on:
            raise RuntimeError(f"Simulated crash on snapshot {snap_id}")

        previous_resp = process_snapshot(
            snap_id, tree, coords, masses,
            cat_writer, part_writer, assign_writer, logger,
            previous_resp, t_start, cov_type=args.cov_type,
        )

    # ── Finalize ─────────────────────────────────────────────────
    births = pd.DataFrame({"particle_index": [], "birth_id": []})
    assembly = pd.DataFrame({"particle_index": [], "galaxy_id": []})
    cat_writer.write_finalize(births, assembly)
    logger.write_summary()
    print(f"Pipeline complete in {format_runtime(time.time() - t_start)}")


if __name__ == "__main__":
    main()
