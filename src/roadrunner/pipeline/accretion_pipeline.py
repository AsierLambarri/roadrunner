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
from roadrunner.postprocessing.properties import compute_galaxy_properties
from roadrunner.postprocessing.mixing import compute_riley_criterion
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
        birth_tracker=None,
        assembly_tracker=None,
    ):
        self.merger_handler = merger_handler
        self.snapshot_reader = snapshot_reader
        self.equiv_table = equiv_table
        self.processor = snapshot_processor
        self.cat_writer = cat_writer
        self.part_writer = part_writer
        self.assign_writer = assign_writer
        self.logger = logger
        self.birth_tracker = birth_tracker
        self.assembly_tracker = assembly_tracker

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

    # ── Tracker helpers ──────────────────────────────────────────

    def _update_trackers(self, snap_id, snap_data, result, satellites,
                         birth_tracker, assembly_tracker):
        if birth_tracker is not None:
            df = result.particle_df
            arr_idx = df["array_index"].values
            sim_ids = snap_data.indices[arr_idx]
            birth_tracker.update(
                t_snap=snap_data.time, snapshot_id=snap_id,
                particle_ids=sim_ids,
                host_ids=df["Sub_tree_id"].values,
                timescales=df.get("timescale", np.full(len(df), 0.1)).values,
            )
        if assembly_tracker is not None:
            assignment_map = {}
            for sid, group in result.particle_df.groupby("Sub_tree_id"):
                arr_idx = group["array_index"].values
                assignment_map[int(sid)] = set(
                    snap_data.indices[arr_idx].tolist(),
                )
            birth_map = (
                birth_tracker.current_birth_map()
                if birth_tracker else {}
            )
            assembly_tracker.update(
                snap_id, assignment_map, birth_map, satellites,
            )

    def _track_and_reduce(self, snap_df, snap_data, ensemble, result,
                          assembly_tracker):
        bound_csc, _ = ensemble.get_particles()
        coords = np.column_stack([snap_data.positions, snap_data.velocities])

        assembly_map = assembly_tracker.current()
        galaxy_particles = {}
        galaxy_bound = {}
        sid_to_col = {sid: i for i, sid in enumerate(bound_csc.column_id)}
        for gid, sim_set in assembly_map.items():
            col = sid_to_col.get(gid)
            if col is None:
                continue
            bound_idx = bound_csc.column_indices[col]
            sim_arr = np.array(list(sim_set), dtype=np.uint64)
            allowed_idx = snap_data.array_index(sim_arr)
            allowed_idx = allowed_idx[allowed_idx >= 0]
            intersection = np.intersect1d(allowed_idx, bound_idx)
            if len(intersection) > 0:
                galaxy_particles[int(gid)] = intersection
                galaxy_bound[int(gid)] = bound_idx

        galaxy_table = snap_df[["Sub_tree_id", "host_id", "mass",
                                "distance_to_acc_id"]].copy()
        galaxy_table.set_index("Sub_tree_id", inplace=True)
        host_row = snap_df[
            snap_df["Sub_tree_id"] == self.processor.accretion_id
        ]
        host_props = host_row.iloc[0] if not host_row.empty else snap_df.iloc[0]

        galaxy_centers = {}
        for sid, params in result.fitted_parameters.items():
            mean = params.get("mean")
            if mean is not None:
                galaxy_centers[int(sid)] = np.asarray(mean)

        properties = compute_galaxy_properties(
            accretion_id=self.processor.accretion_id,
            particle_masses=snap_data.masses,
            particle_coords=coords,
            galaxy_particles=galaxy_particles,
            galaxy_table=galaxy_table,
            host_props=host_props,
            halo_model=self.processor.halo_model,
            n_los=self.processor.n_los,
            galaxy_centers=galaxy_centers if galaxy_centers else None,
        )

        dynstate = compute_riley_criterion(
            main_id=self.processor.accretion_id,
            particle_masses=snap_data.masses,
            particle_coords=coords,
            galaxy_allowed=galaxy_particles,
            galaxy_bound=galaxy_bound,
            redshift=snap_data.redshift,
        )

        return properties, dynstate

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

                # Process (boundness → segment → assign → timescales)
                ensemble, result = self.processor.process(
                    snap_df, snap_data, newborn,
                    previous_resp=previous_resp,
                )

                # Trackers (persistent sim ID space)
                self._update_trackers(
                    snap_id, snap_data, result, satellites,
                    self.birth_tracker, self.assembly_tracker,
                )

                # Track + reduce (sim ID → array index for bound ∩ allowed)
                properties, dynstate = self._track_and_reduce(
                    snap_df, snap_data, ensemble, result,
                    self.assembly_tracker,
                ) if self.assembly_tracker is not None else (
                    pd.DataFrame(), pd.DataFrame(),
                )

                # Convert for next snapshot: array_index → sim ID
                previous_resp_sim = self._to_sim_space(
                    result.responsibilities, snap_data,
                )

                # I/O
                if self.cat_writer:
                    self.cat_writer.write_snapshot(
                        snap_id, snap_data.time,
                        properties, dynstate, satellites,
                    )
                if self.part_writer:
                    self.part_writer.write_snapshot(
                        snap_id, snap_data.time, snap_data.redshift, snap_data,
                    )
                if self.assign_writer:
                    bound_csc, _ = ensemble.get_particles()
                    self.assign_writer.write_snapshot(
                        snap_id, snap_data.time, result, bound_csc,
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
        bt = self.birth_tracker
        at = self.assembly_tracker

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
