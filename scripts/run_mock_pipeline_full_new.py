#!/usr/bin/env python3
"""End-to-end mock pipeline using AccretionPipeline.

Fresh run: processes all snapshots.
Crash+resume: a crash on snapshot N leaves earlier snapshots intact;
  resume re-processes only snapshot N onward, keeping previous data.

Usage:
  python scripts/run_mock_pipeline_full_new.py
  python scripts/run_mock_pipeline_full_new.py --resume
  python scripts/run_mock_pipeline_full_new.py --fail-on 1
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from roadrunner._mcf_types import SnapshotData
from roadrunner.clustering.assignment.gmm import GMMAssigner
from roadrunner.pipeline.snapshot_processor import SnapshotProcessor
from roadrunner.pipeline.accretion_pipeline import AccretionPipeline
from roadrunner.readers.snapshot import SnapshotReader as RawSnapshotReader
from roadrunner.readers.equivalence import EquivalenceTable
from roadrunner.physics.merger_tree import MergerTreeHandlerCSV
from roadrunner.io.hdf5_catalogue import HDF5CatalogueWriter
from roadrunner.io.hdf5_particles import HDF5ParticleWriter
from roadrunner.io.hdf5_assignment import HDF5AssignmentWriter
from roadrunner.io.logging import RunLogger
from roadrunner.io.serialization import save_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="test_data/mock_snap_tight")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-on", type=int, default=None,
                        help="Simulate crash on this snapshot number")
    parser.add_argument("--n-snapshots", type=int, default=2)
    parser.add_argument("--cov-type", default="full",
                        choices=["full", "diagonal", "spherical"])
    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = args.output_dir or os.path.join(data_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    if not args.resume:
        for f in os.listdir(output_dir):
            fp = os.path.join(output_dir, f)
            if os.path.isfile(fp):
                os.remove(fp)

    # ── Load and duplicate mock data across snapshots ────────────
    tree = pd.read_csv(os.path.join(data_dir, "merger_tree.csv"))
    frames = []
    for i in range(args.n_snapshots):
        dup = tree.copy()
        dup["Snapshot"] = i
        frames.append(dup)
    tree = pd.concat(frames, ignore_index=True)
    if "host_id" not in tree.columns:
        tree["host_id"] = -1
    if "distance_to_acc_id" not in tree.columns:
        tree["distance_to_acc_id"] = 0.0

    particles = np.load(os.path.join(data_dir, "particles.npz"))
    coords = particles["coords"]
    masses = particles["masses"]
    N = coords.shape[0]

    class MockSnapshotReader:
        def __init__(self):
            self.call_count = 0
        def load(self, path):
            self.call_count += 1
            if args.fail_on is not None and self.call_count - 1 >= args.fail_on:
                raise RuntimeError(
                    f"Simulated crash on snapshot {self.call_count - 1}"
                )
            return SnapshotData(
                indices=np.arange(N, dtype=np.uint64),
                masses=masses,
                positions=coords[:, :3],
                velocities=coords[:, 3:6],
                redshift=0.0, time=13.8,
            )

    merger_handler = MergerTreeHandlerCSV(tree.copy())
    assigner = GMMAssigner(
        cov_type=args.cov_type, max_iter=3, tol=1e-2,
        min_particles=10, reg_covar=1e-6, prior_type="", verbose=0,
    )
    processor = SnapshotProcessor(
        assigner=assigner,
        halo_model="kepler",
        accretion_id=1,
        n_los=3,
    )
    cat_w = HDF5CatalogueWriter(output_dir)
    part_w = HDF5ParticleWriter(output_dir, float_atol=1e-4)
    assign_w = HDF5AssignmentWriter(output_dir, float_atol=1e-4)
    logger = RunLogger(os.path.join(output_dir, "run.log"))

    equiv = EquivalenceTable(pd.DataFrame({
        "snapshot": list(range(args.n_snapshots)),
        "snapname": ["mock"] * args.n_snapshots,
        "time": [13.8 - i * 0.5 for i in range(args.n_snapshots)],
        "redshift": [i * 0.1 for i in range(args.n_snapshots)],
    }))

    pipeline = AccretionPipeline(
        merger_handler=merger_handler,
        snapshot_reader=MockSnapshotReader(),
        equiv_table=equiv,
        snapshot_processor=processor,
        cat_writer=cat_w,
        part_writer=part_w,
        assign_writer=assign_w,
        logger=logger,
    )

    # If --fail-on is set, save a pre-checkpoint so that a crash inside
    # the pipeline doesn't lose the crash snapshot's previous_resp.
    if args.fail_on is not None and not args.resume:
        ckpt_path = os.path.join(output_dir, "checkpoint.zst")
        save_checkpoint(ckpt_path, {
            "last_snapshot": args.fail_on,
            "previous_resp": None,
        })

    try:
        pipeline.run(output_dir, resume=args.resume)
    except RuntimeError as e:
        if "Simulated crash" in str(e):
            print(f"\nSimulated crash on snapshot "
                  f"{args.fail_on}. Checkpoint saved.")
            print("Run with --resume to continue.")
            sys.exit(1)
        raise


if __name__ == "__main__":
    main()
