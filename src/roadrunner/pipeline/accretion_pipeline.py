"""Full accretion-history pipeline orchestration with error-checkpoint/resume."""

from __future__ import annotations

import json
import os
import shutil
import signal
import time
import traceback
import warnings

import numpy as np
import pandas as pd

from roadrunner._exceptions import RestartError
from roadrunner._defaults import SIM_ID, data_dtype, math_dtype
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
    """Orchestrates the full accretion-history pipeline over multiple snapshots.

    Manages snapshot iteration, error-checkpoint/resume, logging, and I/O
    delegation to catalogue, particle, and assignment writers.

    Parameters
    ----------
    merger_handler : MergerTreeHandlerCSV
    snapshot_reader : SnapshotReader
    equiv_table : EquivalenceTable
    orchestrator : SnapshotOrchestrator
    cat_writer : HDF5CatalogueWriter
    part_writer : HDF5ParticleWriter or None
    assign_writer : HDF5AssignmentWriter or None
    logger : RunLogger
    """

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
        """Run the pipeline over the specified snapshot range.

        Parameters
        ----------
        output_dir : str
            Directory for all output files.
        start_snapshot : int or None, optional
            First snapshot to process (defaults to the earliest known).
        end_snapshot : int or None, optional
            Last snapshot to process (defaults to the latest known).
        resume : bool, default=False
            If ``True``, try to resume from a previously saved checkpoint.
        """
        snapshot_ids = self._filter_snapshots(start_snapshot, end_snapshot)
        if not snapshot_ids:
            print("No snapshots to process.")
            return

        self._checkpoint_path = os.path.join(output_dir, "checkpoint.zst")
        self._progress_path = os.path.join(output_dir, "progress.json")
        self._log_path = self.logger._log_path
        self._error_log_path = os.path.join(output_dir, "error.log")
        self._warning_log_path = os.path.join(output_dir, "warnings.log")

        dyn_snaps = self._resolve_snap_indices(
            self.orchestrator.reduction_config.dynstate_snapshots,
            snapshot_ids,
        )

        if resume:
            previous_resp_sim, start_idx = self._try_resume(snapshot_ids)
        else:
            previous_resp_sim, start_idx = None, 0

        self._ensure_merger_columns()

        if start_idx == 0:
            if os.path.isdir(output_dir):
                shutil.rmtree(output_dir)
            os.makedirs(output_dir, exist_ok=True)
            self._initialize_run(snapshot_ids)

        self._write_progress(
            snapshot_ids,
            snapshot_ids[start_idx - 1] if start_idx > 0 else None,
            False,
        )

        t_start = time.time()
        last_good = None  # (snap_id, previous_resp_sim, fitted_parameters)
        for idx in range(start_idx, len(snapshot_ids)):
            snap_id = snapshot_ids[idx]
            print(f"\nSnapshot {snap_id}  ({idx + 1}/{len(snapshot_ids)})")
            showwarning = warnings.showwarning
            with warnings.catch_warnings(record=True) as records:
                try:
                    snap_result = self._process_snapshot(
                        snap_id, idx, len(snapshot_ids), previous_resp_sim, t_start, dyn_snaps,
                    )
                except (Exception, KeyboardInterrupt):
                    self._save_error_checkpoint(last_good, snapshot_ids)
                    raise
                finally:
                    self._flush_warnings(snap_id, records, showwarning)
            previous_resp_sim = snap_result.previous_resp_sim
            last_good = (
                snap_id, previous_resp_sim, snap_result.result.fitted_parameters,
            )
            self._write_progress(snapshot_ids, snap_id, False)

        self._finalize(t_start)
        self._write_progress(snapshot_ids, snapshot_ids[-1], True)

    def _save_error_checkpoint(self, last_good, snapshot_ids) -> None:
        """Persist resume state after a snapshot failure (best effort).

        Builds the checkpoint payload exactly as the former per-snapshot
        save did (same keys plus the run's ``snapshots`` list, same
        :func:`save_checkpoint` call) for the last fully completed
        snapshot, using live tracker state. Never raises from the save
        itself: a failing error-save (e.g. ``MemoryError`` while pickling)
        must not mask the original failure; it only prints a warning.

        ``SIGINT`` is deferred while the checkpoint is written (re-raised
        afterwards), so a second Ctrl-C cannot skip or tear the save.
        Requires the main thread, like all signal handling.

        Note: both trackers guard reprocessing with ``_last_snapshot``
        and apply idempotent set operations, so resuming is exact unless
        the failure struck inside ``birth._add_update_particles``'s
        per-particle loop (double-added weights on resume) or inside
        ``assembly._update``'s galaxy loop (partial infall lists stand).
        """
        if last_good is None:
            return  # nothing completed; a fresh rerun is equivalent
        snap_id, previous_resp_sim, fitted_parameters = last_good
        bt = self.orchestrator.birth_tracker
        at = self.orchestrator.assembly_tracker
        deferred = {}

        def _defer(signum, frame):
            deferred["hit"] = True

        try:
            try:
                old_handler = signal.signal(signal.SIGINT, _defer)
            except ValueError:
                old_handler = None  # non-main thread: proceed unguarded
            try:
                save_checkpoint(self._checkpoint_path, {
                    "last_snapshot": snap_id,
                    "snapshots": list(snapshot_ids),
                    "precision": {
                        "data": np.dtype(data_dtype()).name,
                        "math": np.dtype(math_dtype()).name,
                    },
                    "previous_resp": previous_resp_sim,
                    "previous_parameters": fitted_parameters,
                    "birth_tracker": bt._get_state() if bt is not None else None,
                    "assembly_tracker": at._get_state() if at is not None else None,
                })
            finally:
                if old_handler is not None:
                    signal.signal(signal.SIGINT, old_handler)
            if deferred.get("hit"):
                raise KeyboardInterrupt
        except Exception as exc:
            print(f"WARNING: error checkpoint could not be saved: {exc}")

    def _write_progress(self, snapshot_ids, last_completed, is_finished) -> None:
        """Record run progress in a small human-readable JSON file.

        Parameters
        ----------
        snapshot_ids : list of int
            All snapshot IDs in this run.
        last_completed : int or None
            Last fully completed snapshot ID (``None`` before the first).
        is_finished : bool
            ``True`` once the run (including finalisation) completed.
        """
        with open(self._progress_path, "w") as f:
            json.dump(
                {
                    "snapshots": list(snapshot_ids),
                    "last_completed_snapshot": last_completed,
                    "is_finished": bool(is_finished),
                },
                f,
                indent=2,
            )

    def _read_progress(self):
        """Load the progress file, or ``None`` when absent (legacy runs)."""
        try:
            with open(self._progress_path) as f:
                return json.load(f)
        except (OSError, ValueError):
            return None

    def _filter_snapshots(self, start_snapshot, end_snapshot):
        """Filter the full snapshot list to the requested range.

        Parameters
        ----------
        start_snapshot : int or None
        end_snapshot : int or None

        Returns
        -------
        ids : list of int
            Filtered snapshot IDs.
        """
        ids = self.merger_handler.snapshots
        start = ids[start_snapshot] if start_snapshot is not None and start_snapshot < 0 else start_snapshot
        end = ids[end_snapshot] if end_snapshot is not None and end_snapshot < 0 else end_snapshot
        if start is not None:
            ids = [s for s in ids if s >= start]
        if end is not None:
            ids = [s for s in ids if s <= end]
        return ids

    def _resolve_snap_indices(self, raw, all_snaps):
        """Resolve snapshot specifications (int, list, str, None) to a set of IDs.

        Parameters
        ----------
        raw : int, list, str, or None
            Snapshot specification.
        all_snaps : list of int
            All available snapshot IDs.

        Returns
        -------
        resolved : set of int or None
        """
        if raw is None:
            return None
        if isinstance(raw, str):
            snap_data = np.atleast_1d(np.loadtxt(raw))
            return {int(s) for s in snap_data}
        snaps = [raw] if isinstance(raw, int) else list(raw)
        resolved = set()
        for s in snaps:
            try:
                resolved.add(all_snaps[s] if s < 0 else s)
            except IndexError:
                continue
        return resolved

    def _try_resume(self, snapshot_ids):
        """Attempt to load a checkpoint and determine the resume start index.

        Parameters
        ----------
        snapshot_ids : list of int

        Returns
        -------
        previous_resp_sim : SparseCSC or None
        start_idx : int
            Index in ``snapshot_ids`` to resume from.
        """
        try:
            ckpt = load_checkpoint(self._checkpoint_path)
        except (RestartError, Exception):
            progress = self._read_progress()
            if progress is not None and progress.get("is_finished", False):
                raise RestartError(
                    f"Run already finished successfully; nothing to resume. "
                    f"Start a fresh run without 'resume' to recompute."
                )
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
        if "snapshots" in ckpt and list(ckpt["snapshots"]) != list(snapshot_ids):
            raise RestartError(
                f"Checkpoint snapshots {list(ckpt['snapshots'])} do not match "
                f"requested {list(snapshot_ids)}. "
                f"Delete the checkpoint and start a fresh run."
            )
        progress = self._read_progress()
        if progress is not None:
            if progress.get("is_finished", False):
                raise RestartError(
                    f"Run already finished successfully; checkpoint is stale. "
                    f"Start a fresh run without 'resume' to recompute."
                )
            if progress.get("last_completed_snapshot") != last_completed:
                raise RestartError(
                    f"Checkpoint snapshot {last_completed} does not match "
                    f"progress file snapshot {progress.get('last_completed_snapshot')}; "
                    f"the checkpoint is stale. "
                    f"Delete the checkpoint and start a fresh run."
                )
        next_idx = snapshot_ids.index(last_completed) + 1
        ckpt_precision = ckpt.get("precision")
        if ckpt_precision is not None:
            current = {
                "data": np.dtype(data_dtype()).name,
                "math": np.dtype(math_dtype()).name,
            }
            if dict(ckpt_precision) != current:
                raise RestartError(
                    f"Checkpoint precision {dict(ckpt_precision)} does not match "
                    f"requested {current}. "
                    f"Delete the checkpoint and start a fresh run."
                )
        previous_resp_sim = ckpt.get("previous_resp")
        previous_parameters = ckpt.get("previous_parameters")
        if previous_parameters is not None:
            self.orchestrator.assigner.parameters = previous_parameters

        bt_state = ckpt.get("birth_tracker")
        if bt_state is not None and self.orchestrator.birth_tracker is not None:
            self.orchestrator.birth_tracker._set_state(bt_state)
        at_state = ckpt.get("assembly_tracker")
        if at_state is not None and self.orchestrator.assembly_tracker is not None:
            self.orchestrator.assembly_tracker._set_state(at_state)

        if next_idx >= len(snapshot_ids):
            print(f"Snapshot {last_completed} was the last snapshot. Nothing to resume.")
            return previous_resp_sim, len(snapshot_ids)
        print(f"Resuming after snapshot {last_completed}, starting at snapshot {snapshot_ids[next_idx]}")
        return previous_resp_sim, next_idx

    def _ensure_merger_columns(self):
        """Ensure merger tree columns (scale radius, host, distance) are computed.

        Calls the merger handler to compute ``scale_radius`` via the Duffy
        relation, finds the most bound satellite host for each subhalo,
        and computes distances both to the ``host_id`` and to the
        accretion host (``acc_id``).
        """
        self.merger_handler.compute_scale_radii()
        self.merger_handler.compute_most_bound_satellite()
        self.merger_handler.compute_distance_to_host()
        acc_id = self.orchestrator.reduction_config.accretion_id
        self.merger_handler.set_constant_column("acc_id", acc_id)
        self.merger_handler.compute_distance_to_host(column="acc_id")

    def _initialize_run(self, snapshot_ids):
        """Write the catalogue header and initialise the run log.

        Parameters
        ----------
        snapshot_ids : list of int
            All snapshot IDs to be processed.
        """
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

    def _process_snapshot(self, snap_id, idx, total, previous_resp_sim, t_start, dyn_snaps):
        """Process a single snapshot end-to-end.

        Parameters
        ----------
        snap_id : int
        idx : int
            Index in the snapshot list.
        total : int
            Total number of snapshots.
        previous_resp_sim : SparseCSC or None
        t_start : float
            Wall-clock start of the run.
        dyn_snaps : set of int or None
            Snapshots on which to compute dynamical state.

        Returns
        -------
        snap_result : SnapshotResult

        Raises
        ------
        Exception
            Propagated after logging the error; the caller persists an
            error checkpoint before it reaches the user.
        """
        try:
            t0 = time.time()
            snap_df = self.merger_handler.select_snapshots(snap_id)
            if snap_df.empty:
                raise RuntimeError(f"Empty merger tree for snapshot {snap_id}")

            file_path = self.equiv_table.snapshot_path(snap_id)
            snap_data = self.snapshot_reader.load(file_path)
            t1 = time.time()

            satellites = self.merger_handler.compute_satellites(snap_df)

            compute_dynstate = dyn_snaps is None or snap_id in dyn_snaps
            snap_result = self.orchestrator.process(
                snap_id, snap_df, snap_data, satellites, previous_resp_sim,
                compute_dynstate=compute_dynstate,
            )
            t2 = time.time()

            time_stats = {"load": t1 - t0, "process": t2 - t1}

            if self.cat_writer:
                self.cat_writer.write_snapshot(
                    snap_id, snap_data.time,
                    snap_result.properties, snap_result.dynstate,
                    snap_result.satellites,
                )
            if self.part_writer:
                self.part_writer.write_snapshot(
                    snap_id, snap_data.time, snap_data.redshift, snap_data,
                )
            if self.assign_writer:
                if "timescale" in snap_result.result.particle_df.columns:
                    sim_ids = snap_data.index[snap_result.result.particle_df["array_index"].values]
                    recs = np.array(
                        list(zip(sim_ids, snap_result.result.particle_df["timescale"].values)),
                        dtype=[("particle_index", SIM_ID), ("timescale", np.float32)],
                    )
                    self.assign_writer.write_timescales(recs)

                snap_result.result.particle_df["particle_index"] = snap_data.index[
                    snap_result.result.particle_df["array_index"].values
                ]
                snap_result.result.particle_df.drop(
                    columns=["array_index", "timescale"], inplace=True, errors="ignore",
                )

                bound_csc, _ = snap_result.ensemble.get_particles()
                self.assign_writer.write_snapshot(
                    snap_id, snap_data.time, snap_result.result, bound_csc,
                )

            elapsed = time.time() - t_start
            self._log_snapshot(snap_id, elapsed, snap_result, snap_data, time_stats)

            return snap_result

        except (Exception, KeyboardInterrupt) as exc:
            self._log_error(snap_id, exc)
            print(f"FATAL: {type(exc).__name__} on snapshot {snap_id}: {exc}")
            print(f"Error checkpoint will be saved. See {self._error_log_path} for traceback.")
            raise

    def _finalize(self, t_start):
        """Finalise the pipeline: finalise trackers and write the final catalogue.

        Parameters
        ----------
        t_start : float
            Wall-clock start of the run (for runtime reporting).
        """
        print("\nFinalizing...")
        bt = self.orchestrator.birth_tracker
        at = self.orchestrator.assembly_tracker

        birth_df = bt.finalize() if bt is not None else pd.DataFrame()

        assembly_dict = at.current() if at is not None else {}
        assembly_records = [
            (pid, g) for g, pset in assembly_dict.items() for pid in pset
        ]
        assembly_df = pd.DataFrame(
            assembly_records, columns=["particle_index", "galaxy_id"],
        ) if assembly_records else pd.DataFrame()

        self.cat_writer.write_finalize(birth_df, assembly_df)

        self.logger.write_summary()
        print(f"Pipeline complete in {format_runtime(time.time() - t_start)}")

    def _log_snapshot(self, snap_id, elapsed, snap_result, snap_data, time_stats):
        """Write snapshot log entry.

        Parameters
        ----------
        snap_id : int
        elapsed : float
        snap_result : SnapshotResult
        snap_data : SnapshotData
        time_stats : dict
        """
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
        """Write error details to the error log file.

        Parameters
        ----------
        snap_id : int
        exc : Exception
        """
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        tb_str = traceback.format_exc()
        with open(self._error_log_path, "a") as f:
            f.write(f"[{timestamp}] Snapshot {snap_id}: {type(exc).__name__}\n")
            f.write(tb_str)
            f.write("---\n")

    def _log_warning(self, snap_id, category, message, filename, lineno):
        """Write one non-fatal warning to the warnings log.

        Mirrors :meth:`_log_error` (timestamp, snapshot scope,
        ``---`` separator, append mode), but is best effort: a
        failing warning save must never mask real work.

        Parameters
        ----------
        snap_id : int
        category : type
        message : Warning
        filename : str
        lineno : int
        """
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        try:
            with open(self._warning_log_path, "a", encoding="utf-8") as f:
                f.write(
                    f"[{timestamp}] Snapshot {snap_id}: "
                    f"{category.__name__}: {message} "
                    f"({filename}:{lineno})\n---\n"
                )
        except OSError as exc:
            print(f"WARNING: could not append to warnings log: {exc}")

    def _flush_warnings(self, snap_id, records, showwarning):
        """Persist captured warnings, then replay them to the console.

        Parameters
        ----------
        snap_id : int
        records : list of warnings.WarningMessage
        showwarning : callable
            The original ``warnings.showwarning`` saved before
            entering the ``catch_warnings`` context.
        """
        for record in records:
            self._log_warning(
                snap_id, record.category, record.message,
                record.filename, record.lineno,
            )
            showwarning(
                record.message, record.category,
                record.filename, record.lineno,
                record.file, record.line,
            )