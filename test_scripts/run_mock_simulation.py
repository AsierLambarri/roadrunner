#!/usr/bin/env python3
"""Run the accretion pipeline on mock simulation data.

Mirrors roadrunner.pipeline.entry.run_accretion_history() exactly,
except it replaces the YT-based SnapshotReader with a
MockSimulationReader that loads per-snapshot .npz files.

Usage:
  python test_scripts/run_mock_simulation.py
      --data-dir test_data/mock_simulation
      --output-dir test_output/mock_sim
      --accretion-id 1
"""

import sys
sys.path.insert(0, "src")

import argparse
import os

import numpy as np

from roadrunner._mcf_types import SnapshotData
from roadrunner.readers.merger_tree import MergerTreeReaderCSV
from roadrunner.readers.equivalence import EquivalenceTable
from roadrunner.physics.merger_tree import MergerTreeHandlerCSV
from roadrunner.clustering.assignment.gmm import XGMMAssigner
from roadrunner.pipeline.processing import ProcessingConfig
from roadrunner.pipeline.reduction import ReductionConfig
from roadrunner.pipeline.snapshot_orchestrator import SnapshotOrchestrator
from roadrunner.pipeline.accretion_pipeline import AccretionPipeline
from roadrunner.io.hdf5_catalogue import HDF5CatalogueWriter
from roadrunner.io.hdf5_particles import HDF5ParticleWriter
from roadrunner.io.hdf5_assignment import HDF5AssignmentWriter
from roadrunner.io.logging import RunLogger
from roadrunner.postprocessing.tracking.birth import BirthTracker
from roadrunner.postprocessing.tracking.assembly import AssemblyTracker


class MockSimulationReader:
    """Replacement for SnapshotReader that loads mock .npz files.

    Parses the snapshot ID from the filename and looks up time/redshift
    from the equivalence table.
    """
    def __init__(self, equiv_table):
        self._equiv = equiv_table

    def load(self, path):
        p = np.load(path)
        basename = os.path.basename(path)
        snap_id = int(basename.split("_")[1].split(".")[0])
        return SnapshotData(
            indices=p["indices"],
            masses=p["masses"],
            positions=p["coords"][:, :3],
            velocities=p["coords"][:, 3:6],
            redshift=self._equiv.snapshot_redshift(snap_id),
            time=self._equiv.snapshot_time(snap_id),
        )


def main():
    parser = argparse.ArgumentParser(description="Run pipeline on mock simulation")
    # Data
    parser.add_argument("--data-dir", default="test_data/mock_simulation",
                        help="Mock simulation data directory")
    parser.add_argument("--output-dir", default="test_output/mock_sim",
                        help="Output directory for pipeline results")
    parser.add_argument("--accretion-id", type=int, default=1,
                        help="Accretion host Sub_tree_id")
    parser.add_argument("--n-snap", type=int, default=None,
                        help="Only process the first N snapshots (default: all)")

    # GMM
    parser.add_argument("--method", default="gmm",
                        choices=["gmm", "bgmm"])
    parser.add_argument("--cov-type", default="full",
                        choices=["full", "diagonal", "spherical"])
    parser.add_argument("--max-iter", type=int, default=10)
    parser.add_argument("--tol", type=float, default=1e-2)
    parser.add_argument("--min-particles", type=int, default=10)

    # Physics
    parser.add_argument("--halo-model", default="kepler",
                        choices=["kepler", "nfw"])
    parser.add_argument("--n-los", type=int, default=15)
    parser.add_argument("--search-factor", type=float, default=1.0)

    # Tracking
    parser.add_argument("--birth-window", type=float, default=5.0)
    parser.add_argument("--no-birth", action="store_true",
                        help="Disable birth tracker")

    # Output
    parser.add_argument("--save-particles", action="store_true",
                        help="Save particle positions/velocities")
    parser.add_argument("--no-assignment", action="store_true",
                        help="Disable assignment writer")

    # Resume
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint")
    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = args.output_dir

    # ── 1. Merger tree ─────────────────────────────────────────────
    reader = MergerTreeReaderCSV(os.path.join(data_dir, "merger_tree.csv"))
    merger_handler = MergerTreeHandlerCSV(reader.dataframe)

    # ── 2. Snapshot reader (mock .npz instead of YT) ───────────────
    equiv_table = EquivalenceTable(
        os.path.join(data_dir, "equivalence.csv"), base_dir=data_dir,
    )
    snapshot_reader = MockSimulationReader(equiv_table)

    # ── 3. Accretion ID auto-detect ────────────────────────────────
    accretion_id = args.accretion_id
    if accretion_id is None:
        last_snap = merger_handler.snapshots[-1]
        host_df = merger_handler.select_accretion_host(
            last_snap, criterion="max_mass",
        )
        accretion_id = int(host_df["Sub_tree_id"].iloc[0])

    # ── 4. Assigner ────────────────────────────────────────────────
    assigner = XGMMAssigner(
        cov_type=args.cov_type,
        max_iter=args.max_iter,
        tol=args.tol,
        min_particles=args.min_particles,
        reg_covar=1e-6,
        prior_type="",
        verbose=1,
        method=args.method,
    )

    # ── 5. Configs ─────────────────────────────────────────────────
    processing_config = ProcessingConfig(
        halo_model=args.halo_model,
        search_factor=args.search_factor,
        min_particles=args.min_particles,
    )
    reduction_config = ReductionConfig(
        accretion_id=accretion_id,
        halo_model=args.halo_model,
        n_los=args.n_los,
    )

    # ── 6. Trackers ────────────────────────────────────────────────
    birth_tracker = (
        BirthTracker(
            factor=args.birth_window,
            enforce_initial_hosts=False,
        ) if not args.no_birth else None
    )
    assembly_tracker = (
        AssemblyTracker() if birth_tracker is not None else None
    )

    # ── 7. Orchestrator ────────────────────────────────────────────
    orchestrator = SnapshotOrchestrator(
        processing_config=processing_config,
        reduction_config=reduction_config,
        assigner=assigner,
        birth_tracker=birth_tracker,
        assembly_tracker=assembly_tracker,
    )

    # ── 8. Writers ─────────────────────────────────────────────────
    cat_w = HDF5CatalogueWriter(output_dir)
    part_w = (
        HDF5ParticleWriter(output_dir, float_atol=1e-4)
        if args.save_particles else None
    )
    assign_w = (
        HDF5AssignmentWriter(output_dir, float_atol=1e-4)
        if not args.no_assignment else None
    )

    # ── 9. Logger ──────────────────────────────────────────────────
    logger = RunLogger(os.path.join(output_dir, "run.log"))

    # ── 10. Pipeline ───────────────────────────────────────────────
    pipeline = AccretionPipeline(
        merger_handler=merger_handler,
        snapshot_reader=snapshot_reader,
        equiv_table=equiv_table,
        orchestrator=orchestrator,
        cat_writer=cat_w,
        part_writer=part_w,
        assign_writer=assign_w,
        logger=logger,
    )

    # ── 11. Run ────────────────────────────────────────────────────
    pipeline.run(
        output_dir,
        start_snapshot=0,
        end_snapshot=(args.n_snap - 1) if args.n_snap is not None else None,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
