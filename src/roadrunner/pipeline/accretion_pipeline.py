from __future__ import annotations

import os
import shutil
import time
import traceback

import numpy as np
import pandas as pd

from roadrunner._exceptions import RestartError
from roadrunner.io.hdf5_assignment import HDF5AssignmentWriter
from roadrunner.io.hdf5_catalogue import HDF5CatalogueWriter
from roadrunner.io.hdf5_particles import HDF5ParticleWriter
from roadrunner.io.logging import format_runtime
from roadrunner.io.serialization import load_checkpoint, save_checkpoint
from roadrunner.physics.merger_tree import MergerTreeHandlerCSV
from roadrunner.readers.equivalence import EquivalenceTable
from roadrunner.readers.snapshot import SnapshotReader
from roadrunner.pipeline.snapshot_orchestrator import SnapshotOrchestrator


class AccretionPipeline:
    def __init__(
        self,
        merger_handler: MergerTreeHandlerCSV,
        snapshot_reader: SnapshotReader,
        equiv_table: EquivalenceTable,
        orchestrator: SnapshotOrchestrator,
        cat_writer: HDF5CatalogueWriter,
        part_writer: HDF5ParticleWriter | None,
        assign_writer: HDF5AssignmentWriter | None,
        logger,
    ):
        self.merger_handler = merger_handler
        self.snapshot_reader = snapshot_reader
        self.equiv_table = equiv_table
        self.orchestrator = orchestrator
        self.cat_writer = cat_writer
        self.part_writer = part_writer
        self.assign_writer = assign_writer
        self.logger = logger

    def run(self, output_dir: str, *, start_snapshot: int | None = None,
            end_snapshot: int | None = None, resume: bool = False) -> None:
        snapshot_ids = self._filter_snapshots(start_snapshot, end_snapshot)
        if not snapshot_ids:
            print("No snapshots to process.")
            return

        self._checkpoint_path = os.path.join(output_dir, "checkpoint.zst")
        self._log_path = self.logger._log_path
        self._error_log_path = os.path.join(output_dir, "error.log")

        if resume:
            previous_resp_sim, start_idx = self._try_resume(snapshot_ids)
        else:
            previous_resp_sim, start_idx = None, 0

        if start_idx == 0:
            if os.path.isdir(output_dir):
                shutil.rmtree(output_dir)
            os.makedirs(output_dir, exist_ok=True)
            self._initialize_run(snapshot_ids)

        t_start = time.time()
        for idx in range(start_idx, len(snapshot_ids)):
            snap_id = snapshot_ids[idx]
            print(f"\nSnapshot {snap_id}  ({idx + 1}/{len(snapshot_ids)})")
            snap_result = self._process_snapshot(
                snap_id, idx, len(snapshot_ids), previous_resp_sim, t_start,
            )
            previous_resp_sim = snap_result.previous_resp_sim
            save_checkpoint(self._checkpoint_path, {
                "last_snapshot": snap_id,
                "previous_resp": previous_resp_sim,
            })

        self._finalize(t_start)

    def _filter_snapshots(self, start_snapshot, end_snapshot):
        ids = self.merger_handler.snapshots
        if start_snapshot is not None:
            ids = [s for s in ids if s >= start_snapshot]
        if end_snapshot is not None:
            ids = [s for s in ids if s <= end_snapshot]
        return ids

    def _try_resume(self, snapshot_ids):
        try:
            ckpt = load_checkpoint(self._checkpoint_path)
        except (RestartError, Exception):
            raise RestartError(
                f"Checkpoint at '{self._checkpoint_path}' is corrupt or unreadable. "
                f"Delete it and start a fresh run."
            )
        if "last_snapshot" not in ckpt:
            raise RestartError(
                f"Checkpoint at '{self._checkpoint_path}' is missing 'last_snapshot'. "
                f"Delete it and start a fresh run."
            )
        last_completed = ckpt["last_snapshot"]
        if last_completed not in snapshot_ids:
            raise RestartError(
                f"Checkpoint snapshot {last_completed} not in snapshot range. "
                f"Delete the checkpoint and start a fresh run."
            )
        next_idx = snapshot_ids.index(last_completed) + 1
        previous_resp_sim = ckpt.get("previous_resp")
        if next_idx >= len(snapshot_ids):
            print(f"Snapshot {last_completed} was the last snapshot. Nothing to resume.")
            return previous_resp_sim, len(snapshot_ids)
        print(f"Resuming after snapshot {last_completed}, starting at snapshot {snapshot_ids[next_idx]}")
        return previous_resp_sim, next_idx

    def _initialize_run(self, snapshot_ids):
        self.merger_handler.compute_scale_radii()
        self.merger_handler.compute_most_bound_satellite()
        self.merger_handler.compute_distance_to_host()
        acc_id = self.orchestrator.reduction_config.accretion_id
        self.merger_handler.set_constant_column("acc_id", acc_id)
        self.merger_handler.compute_distance_to_host(column="acc_id")
        self.logger.write_header({
            "output_dir": os.path.dirname(self._log_path) or ".",
            "halo_model": self.orchestrator.processing_config.halo_model,
            "cov_type": self.orchestrator.assigner.cov_type,
        })
        self.cat_writer.write_header(
            accretion_id=self.orchestrator.reduction_config.accretion_id,
            snapshots=snapshot_ids,
            config_dict={
                "halo_model": self.orchestrator.processing_config.halo_model,
                "n_los": self.orchestrator.reduction_config.n_los,
                "search_factor": self.orchestrator.processing_config.search_factor,
            },
            merger_tree_df=self.merger_handler.dataframe,
            equivalence_df=self.equiv_table.dataframe,
        )

    def _process_snapshot(self, snap_id, idx, total, previous_resp_sim, t_start):
        try:
            t0 = time.time()
            snap_df = self.merger_handler.select_snapshots(snap_id)
            if snap_df.empty:
                raise RuntimeError(f"Empty merger tree for snapshot {snap_id}")

            file_path = self.equiv_table.snapshot_path(snap_id)
            snap_data = self.snapshot_reader.load(file_path)
            t1 = time.time()

            satellites = self.merger_handler.compute_satellites(snap_df)

            snap_result = self.orchestrator.process(
                snap_id, snap_df, snap_data, satellites, previous_resp_sim,
            )
            t2 = time.time()

            time_stats = {"load": t1 - t0, "process": t2 - t1}

            if self.cat_writer:
                self.cat_writer.write_snapshot(
                    snap_id, snap_data.time,
                    snap_result.properties, snap_result.dynstate, satellites,
                )
            if self.part_writer:
                self.part_writer.write_snapshot(
                    snap_id, snap_data.time, snap_data.redshift, snap_data,
                )
            if self.assign_writer:
                bound_csc, _ = snap_result.ensemble.get_particles()
                self.assign_writer.write_snapshot(
                    snap_id, snap_data.time, snap_result.result, bound_csc,
                )

            elapsed = time.time() - t_start
            self._log_snapshot(snap_id, elapsed, snap_result, snap_data, time_stats)

            return snap_result

        except Exception as exc:
            self._log_error(snap_id, exc)
            print(f"FATAL: {type(exc).__name__} on snapshot {snap_id}: {exc}")
            print(f"Checkpoint saved. See {self._error_log_path} for traceback.")
            raise

    def _finalize(self, t_start):
        print("\nFinalizing...")
        bt = self.orchestrator.birth_tracker
        at = self.orchestrator.assembly_tracker

        if bt is not None:
            birth_df = bt.finalize()
            timescales = np.array([
                bt._finalized.get(i, {}).get("timescale", -1)
                for i in range(len(bt._finalized))
            ]) if bt._finalized else np.array([], dtype=np.float32)
        else:
            birth_df = pd.DataFrame()
            timescales = np.array([], dtype=np.float32)

        assembly_dict = at.current() if at is not None else {}
        assembly_records = [
            (pid, g) for g, pset in assembly_dict.items() for pid in pset
        ]
        assembly_df = pd.DataFrame(
            assembly_records, columns=["particle_index", "galaxy_id"],
        ) if assembly_records else pd.DataFrame()

        self.cat_writer.write_finalize(birth_df, assembly_df)

        if self.assign_writer is not None and len(timescales) > 0:
            self.assign_writer.write_timescales(timescales)

        self.logger.write_summary()
        print(f"Pipeline complete in {format_runtime(time.time() - t_start)}")

    def _log_snapshot(self, snap_id, elapsed, snap_result, snap_data, time_stats):
        stats = {
            "snap": snap_id,
            "runtime": format_runtime(elapsed),
            "z": snap_data.redshift,
            "load": time_stats["load"],
            "process": time_stats["process"],
            "bound": snap_result.ensemble.nstars,
            "groups": snap_result.result.statistics.get("groups", 0),
        }
        stats.update(snap_result.result.statistics)
        self.logger.write_snapshot(stats)

    def _log_error(self, snap_id, exc):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        tb_str = traceback.format_exc()
        with open(self._error_log_path, "a") as f:
            f.write(f"[{timestamp}] Snapshot {snap_id}: {type(exc).__name__}\n")
            f.write(tb_str)
            f.write("---\n")