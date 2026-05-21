import os
import time
import traceback

import numpy as np
import pandas as pd

from roadrunner._exceptions import RestartError
from roadrunner.clustering.sparse import SparseCSC
from roadrunner.io.hdf5_catalogue import HDF5CatalogueWriter
from roadrunner.io.hdf5_particles import HDF5ParticleWriter
from roadrunner.io.hdf5_assignment import HDF5AssignmentWriter
from roadrunner.io.serialization import save_checkpoint, load_checkpoint
from roadrunner.io.logging import RunLogger, format_runtime
from roadrunner.readers.snapshot import SnapshotReader
from roadrunner.readers.equivalence import EquivalenceTable
from roadrunner.physics.merger_tree import MergerTreeHandlerCSV
from roadrunner.pipeline.snapshot_processor import SnapshotProcessor


class AccretionPipeline:
    def __init__(
        self,
        merger_handler: MergerTreeHandlerCSV,
        snapshot_reader: SnapshotReader,
        equiv_table: EquivalenceTable,
        snapshot_processor: SnapshotProcessor,
        cat_writer: HDF5CatalogueWriter,
        part_writer: HDF5ParticleWriter | None,
        assign_writer: HDF5AssignmentWriter | None,
        logger: RunLogger,
    ):
        self.merger_handler = merger_handler
        self.snapshot_reader = snapshot_reader
        self.equiv_table = equiv_table
        self.processor = snapshot_processor
        self.cat_writer = cat_writer
        self.part_writer = part_writer
        self.assign_writer = assign_writer
        self.logger = logger
        # Propagate writers to processor so reduce() writes per-snapshot data
        self.processor.cat_writer = cat_writer
        self.processor.part_writer = part_writer
        self.processor.assign_writer = assign_writer

    # ── ID translation helpers ────────────────────────────────────

    @staticmethod
    def _to_sim_space(csc, snap_data):
        if csc is None or len(csc) == 0:
            return None
        src, dst = snap_data.index_to_id_map()
        return csc.remap_rows(src, dst)

    @staticmethod
    def _from_sim_space(csc, snap_data):
        if csc is None or len(csc) == 0:
            return None
        src, dst = snap_data.id_to_index_map()
        return csc.remap_rows(src, dst)

    # ── Logging helper ────────────────────────────────────────────

    def _log_snapshot(self, snap_id, elapsed, result, ensemble, **extra):
        stats = {
            "snap": snap_id,
            "runtime": format_runtime(elapsed),
            "z": 0.0,
            "load": 0.0,
            "process": elapsed,
            "reduction": 0.0,
            "bound": ensemble.nstars,
            "groups": result.statistics.get("groups", 0),
        }
        stats.update(result.statistics)
        stats.update(extra)
        self.logger.write_snapshot(stats)

    # ── Run ───────────────────────────────────────────────────────

    def run(
        self,
        output_dir: str,
        *,
        start_snapshot: int | None = None,
        end_snapshot: int | None = None,
        resume: bool = False,
    ) -> None:
        snapshot_ids = self.merger_handler.snapshots
        if start_snapshot is not None:
            snapshot_ids = [s for s in snapshot_ids if s >= start_snapshot]
        if end_snapshot is not None:
            snapshot_ids = [s for s in snapshot_ids if s <= end_snapshot]

        if not snapshot_ids:
            print("No snapshots to process.")
            return

        checkpoint_path = os.path.join(output_dir, "checkpoint.zst")
        log_path = self.logger._log_path

        previous_resp_sim = None
        start_idx = 0

        # ── Resume ────────────────────────────────────────────────
        if resume:
            try:
                ckpt = load_checkpoint(checkpoint_path)
                failed_snap = ckpt["last_snapshot"]
                start_idx = snapshot_ids.index(failed_snap)
                previous_resp_sim = ckpt.get("previous_resp")
                print(f"Resumed. Re-processing snapshot {failed_snap}")
            except (RestartError, Exception) as e:
                print(f"Resume failed, starting fresh: {e}")
                resume = False

        # ── Resume cleanup: remove per-snapshot output files ─────
        # Keep catalogue.hdf5 (header + previously written snapshots)
        # and checkpoint.zst. Recreate particles + assignment.
        if resume:
            for f in ("particles.hdf5", "assignment.hdf5"):
                fp = os.path.join(output_dir, f)
                if os.path.isfile(fp):
                    os.remove(fp)

        # ── Pre-loop (once, skipped on resume) ────────────────────
        if not resume:
            self.merger_handler.compute_scale_radii()
            self.merger_handler.compute_most_bound_satellite()
            self.merger_handler.compute_distance_to_host()

            self.cat_writer.write_header(
                accretion_id=self.processor.accretion_id,
                snapshots=snapshot_ids,
                config_dict={
                    "halo_model": self.processor.halo_model,
                    "n_los": self.processor.n_los,
                    "search_factor": self.processor.search_factor,
                },
                merger_tree_df=self.merger_handler.dataframe,
                equivalence_df=self.equiv_table._df,
            )

        t_start = time.time()

        # ── Snapshot loop ─────────────────────────────────────────
        for idx in range(start_idx, len(snapshot_ids)):
            snap_id = snapshot_ids[idx]
            print(f"\nSnapshot {snap_id}  ({idx + 1}/{len(snapshot_ids)})")

            # Checkpoint BEFORE processing (in sim ID space)
            ckpt_data = {
                "last_snapshot": snap_id,
                "previous_resp": previous_resp_sim,
            }
            save_checkpoint(checkpoint_path, ckpt_data)

            try:
                snap_df = self.merger_handler.select_snapshots(snap_id)
                if snap_df.empty:
                    raise RuntimeError(
                        f"Empty merger tree for snapshot {snap_id}"
                    )

                file_path = self.equiv_table.snapshot_path(snap_id)
                snap_data = self.snapshot_reader.load(file_path)

                coords = np.column_stack([
                    snap_data.positions, snap_data.velocities,
                ])
                masses = snap_data.masses
                N = coords.shape[0]

                # Translate previous_resp: sim ID → array_index
                previous_resp = self._from_sim_space(
                    previous_resp_sim, snap_data,
                )

                # Newborn detection
                if previous_resp is None:
                    newborn = np.arange(N, dtype=np.uint64)
                else:
                    existing = set(previous_resp.row_id)
                    newborn = np.array(
                        [i for i in range(N) if i not in existing],
                        dtype=np.uint64,
                    )

                # Satellites for this snapshot
                satellites = self.merger_handler.compute_satellites(
                    snap_df,
                )

                # Process + reduce
                halos, ensemble, result = self.processor.process(
                    snap_df, coords, masses, newborn,
                    previous_resp=previous_resp,
                )
                self.processor.reduce(
                    snap_df, snap_data, ensemble, result, satellites,
                )

                # Convert for next snapshot: array_index → sim ID
                previous_resp_sim = self._to_sim_space(
                    result.responsibilities, snap_data,
                )

                # Log
                elapsed = time.time() - t_start
                self._log_snapshot(snap_id, elapsed, result, ensemble)

            except Exception as exc:
                with open(log_path, "a") as f:
                    f.write(f"\n--- ERROR on snapshot {snap_id} ---\n")
                    f.write(f"{type(exc).__name__}: {exc}\n")
                    traceback.print_exc(file=f)
                    f.write("---\n\n")
                print(f"FATAL: {type(exc).__name__} on snapshot "
                      f"{snap_id}: {exc}")
                print(f"Checkpoint saved. See {log_path} for traceback.")
                raise

        # ── Post-loop: finalize ───────────────────────────────────
        print("\nFinalizing...")
        bt = self.processor.birth_tracker
        at = self.processor.assembly_tracker

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
