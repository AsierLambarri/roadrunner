import os

import numpy as np
import pandas as pd
import pytest

from roadrunner._exceptions import RestartError
from roadrunner._mcf_types import SnapshotData
from roadrunner.clustering.assignment.gmm import XGMMAssigner
from roadrunner.io.hdf5_assignment import HDF5AssignmentWriter
from roadrunner.io.hdf5_catalogue import HDF5CatalogueWriter
from roadrunner.io.hdf5_reader import HDF5CatalogueReader
from roadrunner.io.hdf5_particles import HDF5ParticleWriter
from roadrunner.io.logging import RunLogger
from roadrunner.physics.merger_tree import MergerTreeHandlerCSV
from roadrunner.pipeline.accretion_pipeline import AccretionPipeline
from roadrunner.pipeline.processing import ProcessingConfig
from roadrunner.pipeline.reduction import ReductionConfig
from roadrunner.pipeline.snapshot_orchestrator import SnapshotOrchestrator
from roadrunner.postprocessing.tracking.assembly import AssemblyTracker
from roadrunner.postprocessing.tracking.birth import BirthTracker
from roadrunner.readers.equivalence import EquivalenceTable

import warnings

warnings.filterwarnings("ignore")

DATA_DIR = "test_data/mock_snap_tight"


def _build_mock_pipeline(tmp_path, cov_type="full", n_duplicate=2, with_trackers=True):
    tree = pd.read_csv(os.path.join(DATA_DIR, "merger_tree.csv"))
    frames = []
    for i in range(n_duplicate):
        dup = tree.copy()
        dup["Snapshot"] = i
        frames.append(dup)
    tree = pd.concat(frames, ignore_index=True)

    if "host_id" not in tree.columns:
        tree["host_id"] = -1
    if "distance_to_acc_id" not in tree.columns:
        tree["distance_to_acc_id"] = 0.0

    merger_handler = MergerTreeHandlerCSV(tree.copy())

    assigner = XGMMAssigner(
        cov_type=cov_type, max_iter=3, tol=1e-2,
        min_particles=10, reg_covar=1e-6, prior_type="", verbose=0,
    )

    bt = BirthTracker(factor=5, enforce_initial_hosts=False) if with_trackers else None
    at = AssemblyTracker(n_sat_history=2) if with_trackers else None

    orchestrator = SnapshotOrchestrator(
        processing_config=ProcessingConfig(
            halo_model="kepler", search_factor=1.0, min_particles=10,
        ),
        reduction_config=ReductionConfig(
            accretion_id=1, halo_model="kepler", n_los=3,
        ),
        assigner=assigner,
        birth_tracker=bt,
        assembly_tracker=at,
    )

    cat_w = HDF5CatalogueWriter(str(tmp_path))
    part_w = HDF5ParticleWriter(str(tmp_path), float_atol=1e-4)
    assign_w = HDF5AssignmentWriter(str(tmp_path), float_atol=1e-4)
    logger = RunLogger(os.path.join(str(tmp_path), "run.log"))

    class MockSnapshotReader:
        def load(self, path):
            p = np.load(os.path.join(DATA_DIR, "particles.npz"))
            return SnapshotData(
                index=np.arange(p["coords"].shape[0], dtype=np.uint64),
                mass=p["masses"],
                position=p["coords"][:, :3],
                velocity=p["coords"][:, 3:6],
                redshift=0.0, time=13.8,
            )

    equiv = EquivalenceTable(pd.DataFrame({
        "snapshot": list(range(n_duplicate)),
        "snapname": ["mock"] * n_duplicate,
        "time": [13.8 - i * 0.5 for i in range(n_duplicate)],
        "redshift": [i * 0.1 for i in range(n_duplicate)],
    }))

    mock_reader = MockSnapshotReader()

    pipeline = AccretionPipeline(
        merger_handler=merger_handler,
        snapshot_reader=mock_reader,
        equiv_table=equiv,
        orchestrator=orchestrator,
        cat_writer=cat_w,
        part_writer=part_w,
        assign_writer=assign_w,
        logger=logger,
    )
    return pipeline, merger_handler, mock_reader, equiv


class TestAccretionPipeline:
    def test_full_run_two_snapshots(self, tmp_path):
        pipeline, _, _, _ = _build_mock_pipeline(tmp_path, n_duplicate=2)
        pipeline.run(str(tmp_path))

        cat_path = os.path.join(str(tmp_path), "catalogue.hdf5")
        assert os.path.exists(cat_path)

        cat = HDF5CatalogueReader(cat_path)
        hdr = cat.read_header()
        assert hdr["accretion_id"] == 1
        assert len(hdr["snapshots"]) == 2
        assert cat.read_last_snapshot() is not None

    def test_previous_resp_roundtrip(self, tmp_path):
        pipeline, merger_handler, _, _ = _build_mock_pipeline(tmp_path, n_duplicate=2)
        pipeline.run(str(tmp_path))

        bt = pipeline.orchestrator.birth_tracker
        at = pipeline.orchestrator.assembly_tracker
        assert bt is not None
        assert at is not None
        assert bt._last_snapshot > 0
        assert len(at.current()) > 0

    def test_no_trackers_pipeline(self, tmp_path):
        pipeline, _, _, _ = _build_mock_pipeline(tmp_path, n_duplicate=1, with_trackers=False)
        pipeline.run(str(tmp_path))

        cat_path = os.path.join(str(tmp_path), "catalogue.hdf5")
        assert os.path.exists(cat_path)

        cat = HDF5CatalogueReader(cat_path)
        hdr = cat.read_header()
        assert hdr["accretion_id"] == 1

    def test_error_log_created_on_failure(self, tmp_path):
        pipeline, _, mock_reader, _ = _build_mock_pipeline(tmp_path, n_duplicate=2)

        call_count = [0]
        original_load = mock_reader.load

        def failing_load(path):
            call_count[0] += 1
            if call_count[0] == 2:
                raise RuntimeError("Simulated crash on snapshot 1")
            return original_load(path)

        mock_reader.load = failing_load

        with pytest.raises(RuntimeError, match="Simulated crash"):
            pipeline.run(str(tmp_path))

        error_log_path = os.path.join(str(tmp_path), "error.log")
        assert os.path.exists(error_log_path)
        with open(error_log_path) as f:
            content = f.read()
        assert "RuntimeError" in content
        assert "Snapshot 1" in content

    def test_warnings_log_created_on_warning(self, tmp_path):
        pipeline, _, mock_reader, _ = _build_mock_pipeline(tmp_path, n_duplicate=2)
        original_showwarning = warnings.showwarning
        original_load = mock_reader.load
        call_count = [0]

        def warning_load(path):
            call_count[0] += 1
            if call_count[0] == 1:
                warnings.warn("Simulated unresolved group", UserWarning)
            return original_load(path)

        mock_reader.load = warning_load
        old_filters = warnings.filters[:]
        warnings.simplefilter("always")
        try:
            pipeline.run(str(tmp_path))
        finally:
            warnings.filters[:] = old_filters

        assert warnings.showwarning is original_showwarning

        warnings_log_path = os.path.join(str(tmp_path), "warnings.log")
        assert os.path.exists(warnings_log_path)
        with open(warnings_log_path) as f:
            content = f.read()
        assert "Snapshot 0" in content
        assert "UserWarning" in content
        assert "Simulated unresolved group" in content

    def test_checkpoint_restart(self, tmp_path):
        from roadrunner.io.serialization import load_checkpoint

        pipeline, _, mock_reader, _ = _build_mock_pipeline(tmp_path, n_duplicate=2)
        original_load = mock_reader.load

        call_count = [0]

        def failing_load(path):
            call_count[0] += 1
            if call_count[0] == 2:
                raise RuntimeError("Simulated crash on snapshot 1")
            return original_load(path)

        mock_reader.load = failing_load

        with pytest.raises(RuntimeError, match="Simulated crash"):
            pipeline.run(str(tmp_path))

        ckpt_path = os.path.join(str(tmp_path), "checkpoint.zst")
        assert os.path.exists(ckpt_path)
        ckpt = load_checkpoint(ckpt_path)
        assert ckpt["last_snapshot"] == 0

        call_count[0] = 0
        resume_pipeline, _, _, _ = _build_mock_pipeline(tmp_path, n_duplicate=2)
        resume_pipeline.run(str(tmp_path), resume=True)

        cat_reader = HDF5CatalogueReader(os.path.join(str(tmp_path), "catalogue.hdf5"))
        last_snap = cat_reader.read_last_snapshot()
        assert last_snap is not None

    def test_successful_run_writes_no_checkpoint(self, tmp_path):
        pipeline, _, _, _ = _build_mock_pipeline(tmp_path, n_duplicate=2)
        pipeline.run(str(tmp_path))

        assert not os.path.exists(os.path.join(str(tmp_path), "checkpoint.zst"))

    def test_keyboard_interrupt_saves_checkpoint(self, tmp_path):
        from roadrunner.io.serialization import load_checkpoint

        pipeline, _, mock_reader, _ = _build_mock_pipeline(tmp_path, n_duplicate=2)
        original_load = mock_reader.load

        call_count = [0]

        def interrupting_load(path):
            call_count[0] += 1
            if call_count[0] == 2:
                raise KeyboardInterrupt("Simulated user interrupt on snapshot 1")
            return original_load(path)

        mock_reader.load = interrupting_load

        with pytest.raises(KeyboardInterrupt, match="Simulated user interrupt"):
            pipeline.run(str(tmp_path))

        error_log_path = os.path.join(str(tmp_path), "error.log")
        assert os.path.exists(error_log_path)
        with open(error_log_path) as f:
            content = f.read()
        assert "KeyboardInterrupt" in content
        assert "Snapshot 1" in content

        ckpt_path = os.path.join(str(tmp_path), "checkpoint.zst")
        assert os.path.exists(ckpt_path)
        ckpt = load_checkpoint(ckpt_path)
        assert ckpt["last_snapshot"] == 0

        call_count[0] = 0
        resume_pipeline, _, _, _ = _build_mock_pipeline(tmp_path, n_duplicate=2)
        resume_pipeline.run(str(tmp_path), resume=True)

        cat_reader = HDF5CatalogueReader(os.path.join(str(tmp_path), "catalogue.hdf5"))
        last_snap = cat_reader.read_last_snapshot()
        assert last_snap is not None

    def test_resume_snapshot_mismatch_raises(self, tmp_path):
        pipeline, _, mock_reader, _ = _build_mock_pipeline(tmp_path, n_duplicate=2)
        original_load = mock_reader.load

        call_count = [0]

        def failing_load(path):
            call_count[0] += 1
            if call_count[0] == 2:
                raise RuntimeError("Simulated crash on snapshot 1")
            return original_load(path)

        mock_reader.load = failing_load

        with pytest.raises(RuntimeError, match="Simulated crash"):
            pipeline.run(str(tmp_path))

        wider_pipeline, _, _, _ = _build_mock_pipeline(tmp_path, n_duplicate=3)
        with pytest.raises(RestartError, match="do not match"):
            wider_pipeline.run(str(tmp_path), resume=True)

    def test_progress_file_tracks_run(self, tmp_path):
        import json

        pipeline, _, _, _ = _build_mock_pipeline(tmp_path, n_duplicate=2)
        pipeline.run(str(tmp_path))

        progress_path = os.path.join(str(tmp_path), "progress.json")
        assert os.path.exists(progress_path)
        with open(progress_path) as f:
            progress = json.load(f)
        assert progress["snapshots"] == [0, 1]
        assert progress["last_completed_snapshot"] == 1
        assert progress["is_finished"] is True

    def test_stale_checkpoint_after_success_refuses(self, tmp_path):
        pipeline, _, mock_reader, _ = _build_mock_pipeline(tmp_path, n_duplicate=2)
        original_load = mock_reader.load

        call_count = [0]

        def failing_load(path):
            call_count[0] += 1
            if call_count[0] == 2:
                raise RuntimeError("Simulated crash on snapshot 1")
            return original_load(path)

        mock_reader.load = failing_load

        with pytest.raises(RuntimeError, match="Simulated crash"):
            pipeline.run(str(tmp_path))

        call_count[0] = 0
        resume_pipeline, _, _, _ = _build_mock_pipeline(tmp_path, n_duplicate=2)
        resume_pipeline.run(str(tmp_path), resume=True)

        stale_pipeline, _, _, _ = _build_mock_pipeline(tmp_path, n_duplicate=2)
        with pytest.raises(RestartError, match="finished successfully"):
            stale_pipeline.run(str(tmp_path), resume=True)

    def test_sigint_guard_installed_during_save(self, tmp_path, monkeypatch):
        import signal as signal_mod

        import roadrunner.pipeline.accretion_pipeline as ap_mod

        pipeline, _, _, _ = _build_mock_pipeline(tmp_path, n_duplicate=1)
        pipeline._checkpoint_path = os.path.join(str(tmp_path), "checkpoint.zst")
        pipeline._progress_path = os.path.join(str(tmp_path), "progress.json")

        before = signal_mod.getsignal(signal_mod.SIGINT)
        seen = {}
        real_save = ap_mod.save_checkpoint

        def spy_save(path, data, *a, **k):
            seen["during"] = signal_mod.getsignal(signal_mod.SIGINT)
            return real_save(path, data, *a, **k)

        monkeypatch.setattr(ap_mod, "save_checkpoint", spy_save)
        pipeline._save_error_checkpoint((0, None, {}), [0])

        assert seen["during"] is not before
        assert signal_mod.getsignal(signal_mod.SIGINT) is before
        assert os.path.exists(os.path.join(str(tmp_path), "checkpoint.zst"))

    def test_corrupt_checkpoint_raises(self, tmp_path):
        output_dir = str(tmp_path)
        os.makedirs(output_dir, exist_ok=True)

        corrupt_path = os.path.join(output_dir, "checkpoint.zst")
        with open(corrupt_path, "wb") as f:
            f.write(b"this is not a valid checkpoint")

        pipeline, _, _, _ = _build_mock_pipeline(tmp_path, n_duplicate=2)
        with pytest.raises(RestartError, match="corrupt or unreadable"):
            pipeline.run(output_dir, resume=True)

    def test_fresh_run_wipes_output_dir(self, tmp_path):
        output_dir = str(tmp_path)
        os.makedirs(output_dir, exist_ok=True)

        stale_file = os.path.join(output_dir, "stale_data.txt")
        with open(stale_file, "w") as f:
            f.write("stale")

        pipeline, _, _, _ = _build_mock_pipeline(tmp_path, n_duplicate=1)
        pipeline.run(output_dir)

        assert not os.path.exists(stale_file)
        assert os.path.exists(os.path.join(output_dir, "catalogue.hdf5"))