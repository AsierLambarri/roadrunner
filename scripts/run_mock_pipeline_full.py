#!/usr/bin/env python3
"""End-to-end pipeline on a mock dataset: boundness → segment → GMM assign
→ catalogue I/O → logging → checkpoint save/restart.

Usage:
  python scripts/run_mock_pipeline_full.py
  python scripts/run_mock_pipeline_full.py --resume  (restart from checkpoint)
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


DATA_DIR = "test_data/mock_snap_tight"
OUTPUT_DIR = os.path.join(DATA_DIR, "output")
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "checkpoint.zst")
LOG_PATH = os.path.join(OUTPUT_DIR, "run.log")


def load_mock_data(data_dir):
    tree = pd.read_csv(os.path.join(data_dir, "merger_tree.csv"))
    particles = np.load(os.path.join(data_dir, "particles.npz"))
    return tree, particles


def build_halos(tree):
    halos = []
    for _, row in tree.iterrows():
        h = HaloModel.from_snapshot_row(row, model="kepler", comoving=False)
        halos.append(h)
    return halos


def build_snapshot_data(coords, masses):
    N = coords.shape[0]
    return SnapshotData(
        indices=np.arange(N, dtype=np.uint64),
        masses=masses,
        positions=coords[:, :3],
        velocities=coords[:, 3:6],
        redshift=0.0,
        time=13.8,
    )


def run_pipeline(resume=False):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Clean up any previous run files (except when resuming)
    if not resume:
        for f in os.listdir(OUTPUT_DIR):
            fp = os.path.join(OUTPUT_DIR, f)
            if os.path.isfile(fp):
                os.remove(fp)

    # ── Initialise writers and logger ──────────────────────────────
    cat_writer = HDF5CatalogueWriter(OUTPUT_DIR, mode="w-")
    part_writer = HDF5ParticleWriter(OUTPUT_DIR, mode="w-", float_atol=1e-4)
    assign_writer = HDF5AssignmentWriter(OUTPUT_DIR, mode="w-", float_atol=1e-4)
    logger = RunLogger(LOG_PATH)
    logger.write_header({
        "output_dir": OUTPUT_DIR,
        "halo_model": "kepler",
        "cov_type": "full",
    })

    previous_resp = None
    start_snapshot = 0

    # ── Resume from checkpoint ─────────────────────────────────────
    if resume:
        try:
            ckpt = load_checkpoint(CHECKPOINT_PATH)
            previous_resp = ckpt.get("previous_resp")
            start_snapshot = ckpt.get("last_snapshot", 0) + 1
            cat_writer = HDF5CatalogueWriter(OUTPUT_DIR, mode="r+")
            assign_writer = HDF5AssignmentWriter(OUTPUT_DIR, mode="r+",
                                                  float_atol=1e-4)
            print(f"Resumed from checkpoint. Starting at snapshot {start_snapshot}")
        except RestartError as e:
            print(f"No valid checkpoint found, starting fresh: {e}")
            start_snapshot = 0

    # ── Load data ──────────────────────────────────────────────────
    t0 = time.time()
    tree, particles = load_mock_data(DATA_DIR)
    coords = particles["coords"]
    masses = particles["masses"]
    N = coords.shape[0]
    snap_data = build_snapshot_data(coords, masses)

    # Write header once (only on fresh run)
    if not resume:
        cat_writer.write_header(
            accretion_id=1,
            snapshots=[0],
            config_dict={"halo_model": "kepler", "cov_type": "full"},
            merger_tree_df=tree,
            equivalence_df=pd.read_csv(os.path.join(DATA_DIR, "equivalence.csv")),
        )
    load_time = time.time() - t0

    # ── Pipeline per snapshot ──────────────────────────────────────
    # Single snapshot: snapshot_id = 0
    snapshot_id = 0
    if snapshot_id < start_snapshot:
        print(f"Snapshot {snapshot_id} already processed, skipping.")
        return

    t_snap = time.time()

    # 1. Build halos
    halos = build_halos(tree)
    print(f"Built {len(halos)} halos")

    # 2. Boundness
    halos = compute_halo_bound_particles(halos, coords, search_factor=2.0)
    bound_time = time.time() - t_snap

    # 3. Ensemble
    ensemble = HaloEnsemble(halos)
    pop_idx = ensemble.populated_indices()
    csc_b, _ = ensemble.get_particles()
    candidates = [csc_b.column_indices[i] for i in pop_idx]
    print(f"  Populated halos: {len(pop_idx)}")

    # 4. Segmentation
    seg = HaloSegmenter(
        ensemble.positions[pop_idx],
        ensemble.virial_radii[pop_idx],
    )
    seg.overlap_groups().prune(candidates, min_particles=10, discard=False)
    if seg.pruned_groups and len(seg.pruned_groups) > 0:
        groups_sorted = sorted(
            [pop_idx[g] for g in seg.pruned_groups],
            key=len, reverse=True,
        )
    else:
        groups_sorted = []
    print(f"  Groups: {len(groups_sorted)}")
    seg_time = time.time() - t_snap

    # 5. GMM Assignment
    assigner = GMMAssigner(
        cov_type="full", max_iter=10, tol=1e-2,
        min_particles=10, reg_covar=1e-6, prior_type="", verbose=0,
    )
    newborn = np.arange(N, dtype=np.uint64)
    result = assigner.assign(
        halos, coords, newborn, groups_sorted,
        previous_resp=previous_resp,
    )
    assign_time = time.time() - t_snap

    # 6. Write outputs
    cat_writer.write_snapshot(
        snapshot_id=snapshot_id,
        time=13.8,
        properties_df=pd.DataFrame(),
        dynstate_df=pd.DataFrame(),
        satellites_map={},
    )
    part_writer.write_snapshot(snapshot_id, 13.8, 0.0, snap_data)
    assign_writer.write_snapshot(
        snapshot_id, 13.8, result, csc_b,
    )

    # 7. Log snapshot
    stats = {
        "snap": snapshot_id,
        "runtime": format_runtime(time.time() - t0),
        "z": 0.0,
        "load": load_time,
        "process": assign_time - bound_time,
        "reduction": 0.0,
        "bound": ensemble.nstars,
        "groups": len(groups_sorted),
        **result.statistics,
    }
    logger.write_snapshot(stats)
    io_time = time.time() - t_snap

    total = time.time() - t0
    print(f"  Runtime: {format_runtime(total)}")
    print(f"  Unassigned: {result.statistics.get('unassigned', '?')}")
    print(f"  Avg conf:   {result.statistics.get('avg_conf', '?'):.4f}")
    print(f"  Avg entropy:{result.statistics.get('avg_entropy', '?'):.4f}")
    print(f"  Avg cond:   {result.statistics.get('avg_cond', '?'):.2f}")

    # 8. Save checkpoint (for restart demonstration)
    ckpt = {
        "last_snapshot": snapshot_id,
        "previous_resp": result.responsibilities,
    }
    save_checkpoint(CHECKPOINT_PATH, ckpt)
    print(f"Checkpoint saved to {CHECKPOINT_PATH}")

    logger.write_summary()
    print("Pipeline complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()
    run_pipeline(resume=args.resume)
