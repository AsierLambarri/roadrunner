#!/usr/bin/env python3
"""End-to-end mock pipeline using SnapshotOrchestrator (new Level 1 code).

Runs on the mock_snap_tight dataset with full covariance.
Produces: catalogue, assignment, and particle HDF5 files plus a
comparison report saved alongside this script.

Usage:
    python test_scripts/run_mock_orchestrator.py
"""

import os
import sys
import warnings
import time

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from roadrunner._mcf_types import SnapshotData
from roadrunner.clustering.assignment.gmm import GMMAssigner
from roadrunner.pipeline.processing import ProcessingConfig
from roadrunner.pipeline.reduction import ReductionConfig
from roadrunner.pipeline.snapshot_orchestrator import SnapshotOrchestrator
from roadrunner.physics.merger_tree import MergerTreeHandlerCSV
from roadrunner.postprocessing.tracking.assembly import AssemblyTracker
from roadrunner.postprocessing.tracking.birth import BirthTracker
from roadrunner.io.hdf5_catalogue import HDF5CatalogueWriter
from roadrunner.io.hdf5_particles import HDF5ParticleWriter
from roadrunner.io.hdf5_assignment import HDF5AssignmentWriter


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "test_data", "mock_snap_tight")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "test_output", "run_mock_orchestrator")

N_SNAPSHOTS = 2
ACCRETION_ID = 1
COV_TYPE = "full"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Load merger tree and duplicate across snapshots ────────────
    tree_single = pd.read_csv(os.path.join(DATA_DIR, "merger_tree.csv"))
    frames = []
    for i in range(N_SNAPSHOTS):
        dup = tree_single.copy()
        dup["Snapshot"] = i
        frames.append(dup)
    tree = pd.concat(frames, ignore_index=True)
    if "host_id" not in tree.columns:
        tree["host_id"] = -1
    if "distance_to_acc_id" not in tree.columns:
        tree["distance_to_acc_id"] = 0.0

    # ── Compute merger tree physics ────────────────────────────────
    merger_handler = MergerTreeHandlerCSV(tree.copy())
    merger_handler.compute_scale_radii()
    merger_handler.compute_most_bound_satellite()
    merger_handler.compute_distance_to_host()

    # ── Load particle data ─────────────────────────────────────────
    particles = np.load(os.path.join(DATA_DIR, "particles.npz"))
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

    # ── Build orchestrator ─────────────────────────────────────────
    assigner = GMMAssigner(
        cov_type=COV_TYPE, max_iter=3, tol=1e-2,
        min_particles=10, reg_covar=1e-6, prior_type="", verbose=0,
    )

    bt = BirthTracker(factor=5, enforce_initial_hosts=False)
    at = AssemblyTracker(n_sat_history=2)

    orchestrator = SnapshotOrchestrator(
        processing_config=ProcessingConfig(
            halo_model="kepler",
            search_factor=1.0,
            min_particles=10,
        ),
        reduction_config=ReductionConfig(
            accretion_id=ACCRETION_ID,
            halo_model="kepler",
            n_los=3,
        ),
        assigner=assigner,
        birth_tracker=bt,
        assembly_tracker=at,
    )

    # ── Run snapshots ──────────────────────────────────────────────
    t_start = time.time()
    results = []
    snapshot_ids = merger_handler.snapshots[:N_SNAPSHOTS]

    for i, snap_id in enumerate(snapshot_ids):
        snap_df = merger_handler.select_snapshots(snap_id)
        satellites = merger_handler.compute_satellites(snap_df)

        prev_sim = results[-1].previous_resp_sim if results else None

        result = orchestrator.process(
            snap_id=snap_id,
            snap_df=snap_df,
            snap_data=snap_data,
            satellites=satellites,
            previous_resp_sim=prev_sim,
        )
        results.append(result)

        print(f"Snapshot {snap_id} ({i+1}/{N_SNAPSHOTS}): "
              f"{result.result.particle_df.shape[0]} particles, "
              f"{result.result.statistics.get('groups', 0)} groups, "
              f"{len(result.properties)} properties, "
              f"{len(result.dynstate)} dynstate rows")

    elapsed = time.time() - t_start

    # ── Write outputs ──────────────────────────────────────────────
    cat_w = HDF5CatalogueWriter(OUTPUT_DIR)
    cat_w.write_header(
        accretion_id=ACCRETION_ID,
        snapshots=snapshot_ids,
        config_dict={"halo_model": "kepler", "n_los": 3, "search_factor": 1.0},
        merger_tree_df=merger_handler.dataframe,
        equivalence_df=pd.DataFrame({
            "snapshot": snapshot_ids,
            "snapname": ["mock"] * N_SNAPSHOTS,
            "time": [13.8 - i * 0.5 for i in range(N_SNAPSHOTS)],
            "redshift": [i * 0.1 for i in range(N_SNAPSHOTS)],
        }),
    )

    part_w = HDF5ParticleWriter(OUTPUT_DIR, float_atol=1e-4)
    assign_w = HDF5AssignmentWriter(OUTPUT_DIR, float_atol=1e-4)

    for i, (snap_id, result) in enumerate(zip(snapshot_ids, results)):
        snap_time = 13.8 - i * 0.5
        snap_z = i * 0.1

        cat_w.write_snapshot(
            snap_id, snap_time,
            result.properties, result.dynstate,
            {k: list(v) for k, v in merger_handler.compute_satellites(
                merger_handler.select_snapshots(snap_id)
            ).items()},
        )

        part_w.write_snapshot(snap_id, snap_time, snap_z, snap_data)

        bound_csc, _ = result.ensemble.get_particles()
        assign_w.write_snapshot(
            snap_id, snap_time, result.result, bound_csc,
        )

    # ── Finalize ───────────────────────────────────────────────────
    birth_df = bt.finalize()
    assembly_dict = at.current()
    assembly_records = [
        (pid, g) for g, pset in assembly_dict.items() for pid in pset
    ]
    assembly_df = pd.DataFrame(
        assembly_records, columns=["particle_index", "galaxy_id"],
    ) if assembly_records else pd.DataFrame()

    cat_w.write_finalize(birth_df, assembly_df)

    timescales = np.array([
        bt._finalized.get(i, {}).get("timescale", -1)
        for i in range(len(bt._finalized))
    ]) if bt._finalized else np.array([], dtype=np.float32)

    if len(timescales) > 0:
        assign_w.write_timescales(timescales)

    print(f"\nPipeline complete in {elapsed:.1f}s")
    print(f"Output written to {OUTPUT_DIR}")

    # ── Compare with reference (if available) ──────────────────────
    ref_dir = os.path.join(DATA_DIR, "output")
    if os.path.isdir(ref_dir):
        compare_results(OUTPUT_DIR, ref_dir)
    else:
        print(f"\nNo reference output found at {ref_dir}, skipping comparison.")


def compare_results(new_dir, ref_dir):
    """Compare new output against reference output."""
    import h5py

    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("COMPARISON REPORT: New Orchestrator vs Reference")
    report_lines.append("=" * 60)

    all_ok = True

    for fname in ["catalogue.hdf5", "assignment.hdf5", "particles.hdf5"]:
        new_path = os.path.join(new_dir, fname)
        ref_path = os.path.join(ref_dir, fname)

        if not os.path.exists(new_path):
            report_lines.append(f"\n[MISSING] {fname} not found in new output")
            all_ok = False
            continue
        if not os.path.exists(ref_path):
            report_lines.append(f"\n[MISSING] {fname} not found in reference")
            all_ok = False
            continue

        report_lines.append(f"\n--- {fname} ---")

        with h5py.File(new_path, "r") as f_new, h5py.File(ref_path, "r") as f_ref:
            # Compare snapshots
            if "snapshots" in f_new and "snapshots" in f_ref:
                new_snaps = sorted([int(k) for k in f_new["snapshots"].keys()])
                ref_snaps = sorted([int(k) for k in f_ref["snapshots"].keys()])
                report_lines.append(f"  Snapshots: new={new_snaps}, ref={ref_snaps}")
                if new_snaps != ref_snaps:
                    report_lines.append("  [DIFF] Snapshot IDs differ!")
                    all_ok = False

                for snap_id in new_snaps:
                    snap_str = str(snap_id)
                    if snap_str not in f_ref["snapshots"]:
                        report_lines.append(f"  [SKIP] Snapshot {snap_id} not in reference")
                        continue

                    new_snap = f_new["snapshots"][snap_str]
                    ref_snap = f_ref["snapshots"][snap_str]

                    for ds_name in new_snap.keys():
                        if ds_name == "satellite_relations":
                            continue
                        if isinstance(new_snap[ds_name], h5py.Dataset):
                            new_data = new_snap[ds_name][:]
                            ref_data = ref_snap[ds_name][:]
                            if new_data.shape != ref_data.shape:
                                report_lines.append(
                                    f"  [DIFF] Snap {snap_id}/{ds_name}: "
                                    f"shape new={new_data.shape} ref={ref_data.shape}"
                                )
                                all_ok = False
                            elif new_data.dtype.names:
                                for field in new_data.dtype.names:
                                    n = new_data[field]
                                    r = ref_data[field]
                                    max_diff = np.max(np.abs(n.astype(float) - r.astype(float)))
                                    report_lines.append(
                                        f"  Snap {snap_id}/{ds_name}/{field}: "
                                        f"max_diff={max_diff:.6e}"
                                    )
                            else:
                                max_diff = np.max(np.abs(
                                    new_data.astype(float) - ref_data.astype(float)
                                ))
                                report_lines.append(
                                    f"  Snap {snap_id}/{ds_name}: "
                                    f"max_diff={max_diff:.6e}"
                                )

    report_lines.append(f"\n{'=' * 60}")
    report_lines.append(f"ALL COMPARISONS OK: {all_ok}")
    report_lines.append(f"{'=' * 60}")

    report = "\n".join(report_lines)
    print(report)

    report_path = os.path.join(OUTPUT_DIR, "comparison_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()