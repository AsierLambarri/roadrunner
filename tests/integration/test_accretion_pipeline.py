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


def _build_mock_pipeline(tmp_path, cov_type="full", n_duplicate=2, with_trackers=True,
                         halo_model="kepler", plausibility="rank"):
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
        plausibility=plausibility,
    )

    bt = BirthTracker(factor=5, enforce_initial_hosts=False) if with_trackers else None
    at = AssemblyTracker(n_sat_history=2) if with_trackers else None

    orchestrator = SnapshotOrchestrator(
        processing_config=ProcessingConfig(
            halo_model=halo_model, search_factor=1.0, min_particles=10,
        ),
        reduction_config=ReductionConfig(
            accretion_id=1, halo_model=halo_model, n_los=3,
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

        bt = pipeline.orchestrator.birth_tracker
        at = pipeline.orchestrator.assembly_tracker
        assert bt is not None
        assert at is not None
        assert bt._last_snapshot > 0
        assert len(at.current()) > 0

        assert not os.path.exists(os.path.join(str(tmp_path), "checkpoint.zst"))

        import json

        progress_path = os.path.join(str(tmp_path), "progress.json")
        assert os.path.exists(progress_path)
        with open(progress_path) as f:
            progress = json.load(f)
        assert progress["snapshots"] == [0, 1]
        assert progress["last_completed_snapshot"] == 1
        assert progress["is_finished"] is True

    def test_no_trackers_pipeline(self, tmp_path):
        output_dir = str(tmp_path)
        os.makedirs(output_dir, exist_ok=True)

        stale_file = os.path.join(output_dir, "stale_data.txt")
        with open(stale_file, "w") as f:
            f.write("stale")

        pipeline, _, _, _ = _build_mock_pipeline(tmp_path, n_duplicate=1, with_trackers=False)
        pipeline.run(str(tmp_path))

        assert not os.path.exists(stale_file)

        cat_path = os.path.join(str(tmp_path), "catalogue.hdf5")
        assert os.path.exists(cat_path)

        cat = HDF5CatalogueReader(cat_path)
        hdr = cat.read_header()
        assert hdr["accretion_id"] == 1

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
        # C01 regression: replaying a captured warning must not re-trigger
        # the recorder and grow the log without bound -- the injected
        # warning was raised exactly once, so it must be logged exactly
        # once (the old bug replayed it into the log forever).
        assert content.count("Simulated unresolved group") == 1

    def test_whole_tree_empty_raises(self, tmp_path):
        pipeline, merger_handler, _, _ = _build_mock_pipeline(tmp_path, n_duplicate=1)
        merger_handler._df = merger_handler._df.iloc[0:0]

        with pytest.raises(RuntimeError, match="zero halos"):
            pipeline.run(str(tmp_path))

        assert not os.path.exists(os.path.join(str(tmp_path), "catalogue.hdf5"))

    def test_empty_particles_snapshot_skipped(self, tmp_path):
        pipeline, _, mock_reader, _ = _build_mock_pipeline(tmp_path, n_duplicate=3)
        original_load = mock_reader.load
        call_count = [0]

        def sparse_load(path):
            call_count[0] += 1
            if call_count[0] == 2:
                empty = original_load(path)
                return SnapshotData(
                    index=np.array([], dtype=np.uint64),
                    mass=np.array([], dtype=empty.mass.dtype),
                    position=np.empty((0, 3), dtype=empty.position.dtype),
                    velocity=np.empty((0, 3), dtype=empty.velocity.dtype),
                    redshift=empty.redshift, time=empty.time,
                )
            return original_load(path)

        mock_reader.load = sparse_load

        old_filters = warnings.filters[:]
        warnings.simplefilter("always")
        try:
            with pytest.warns(UserWarning, match="Snapshot 1 has zero particles"):
                pipeline.run(str(tmp_path))
        finally:
            warnings.filters[:] = old_filters

        cat_path = os.path.join(str(tmp_path), "catalogue.hdf5")
        cat = HDF5CatalogueReader(cat_path)
        assert cat.read_last_snapshot() == 2

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

        error_log_path = os.path.join(str(tmp_path), "error.log")
        assert os.path.exists(error_log_path)
        with open(error_log_path) as f:
            content = f.read()
        assert "RuntimeError" in content
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

        stale_pipeline, _, _, _ = _build_mock_pipeline(tmp_path, n_duplicate=2)
        with pytest.raises(RestartError, match="finished successfully"):
            stale_pipeline.run(str(tmp_path), resume=True)

    def test_seed_persisted_in_checkpoint_and_restored_on_resume(self, tmp_path):
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
            pipeline.run(str(tmp_path), seed=777)

        ckpt = load_checkpoint(os.path.join(str(tmp_path), "checkpoint.zst"))
        assert ckpt["seed"] == 777

        resume_pipeline, _, _, _ = _build_mock_pipeline(tmp_path, n_duplicate=2)
        resume_pipeline.run(str(tmp_path), resume=True)
        assert resume_pipeline._base_seed == 777

    def test_resume_with_conflicting_seed_raises(self, tmp_path):
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
            pipeline.run(str(tmp_path), seed=777)

        resume_pipeline, _, _, _ = _build_mock_pipeline(tmp_path, n_duplicate=2)
        with pytest.raises(RestartError, match="conflict"):
            resume_pipeline.run(str(tmp_path), resume=True, seed=999)

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
        pipeline._save_error_checkpoint((0, None, {}, None, None, None), [0])

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

    def test_writer_failure_after_tracker_mutation_then_resume(self, tmp_path):
        """C02/C03: a failure inside a writer, AFTER orchestrator.process()
        has already fully mutated the trackers for that snapshot, must
        checkpoint tracker state as of the PREVIOUS snapshot -- not the
        just-mutated one -- and a resume must cleanly reprocess the
        failed snapshot exactly once (neither skipped nor double-run).
        This is the exact gap the prior checkpoint mechanism had no
        coverage for: every other failure-injection test raises before
        orchestrator.process() ever runs."""
        from roadrunner.io.serialization import load_checkpoint

        # Relies on cat_writer.write_snapshot running BEFORE
        # part_writer.write_snapshot within _process_snapshot (verified
        # directly in accretion_pipeline.py) -- otherwise this wouldn't
        # actually exercise the C03 retry path at all.
        pipeline, _, _, _ = _build_mock_pipeline(tmp_path, n_duplicate=2)
        original_write = pipeline.part_writer.write_snapshot
        call_count = [0]

        def failing_write(snapshot_id, time, redshift, snapshot_data):
            call_count[0] += 1
            if call_count[0] == 2:
                raise RuntimeError("Simulated crash inside part_writer on snapshot 1")
            return original_write(snapshot_id, time, redshift, snapshot_data)

        pipeline.part_writer.write_snapshot = failing_write

        with pytest.raises(RuntimeError, match="Simulated crash"):
            pipeline.run(str(tmp_path))

        ckpt_path = os.path.join(str(tmp_path), "checkpoint.zst")
        ckpt = load_checkpoint(ckpt_path)
        assert ckpt["last_snapshot"] == 0
        # The checkpoint's tracker payload must reflect snapshot 0 only --
        # not snapshot 1, whose orchestrator.process() already ran
        # (fully mutating the trackers) before part_writer raised.
        assert ckpt["assembly_tracker"]["last_snapshot"] == 0
        assert ckpt["birth_tracker"]["last_snapshot"] == 0

        resume_pipeline, _, _, _ = _build_mock_pipeline(tmp_path, n_duplicate=2)
        resume_pipeline.run(str(tmp_path), resume=True)

        # Snapshot 1 must have been fully, cleanly reprocessed: the
        # trackers now report it as done (not stuck at 0, which is what
        # AssemblyTracker's old set-_last_snapshot-before-_update bug
        # would silently cause on a retry).
        assert resume_pipeline.orchestrator.assembly_tracker._last_snapshot == 1
        assert resume_pipeline.orchestrator.birth_tracker._last_snapshot == 1

        # The retried cat_writer.write_snapshot call for snapshot 1 (which
        # already succeeded once, before part_writer raised) must not
        # crash on "name already exists" (C03).
        cat_reader = HDF5CatalogueReader(os.path.join(str(tmp_path), "catalogue.hdf5"))
        assert cat_reader.read_last_snapshot() == 1

    def test_resume_from_initial_state_checkpoint(self, tmp_path):
        """C02/C03: a failure on the very first snapshot must still write
        a usable checkpoint (last_snapshot: None) instead of skipping the
        save entirely, and resuming from it must reprocess from snapshot
        0 -- not raise RestartError -- while preserving the original
        run's seed rather than silently regenerating a new random one."""
        from roadrunner.io.serialization import load_checkpoint

        pipeline, _, mock_reader, _ = _build_mock_pipeline(tmp_path, n_duplicate=2)

        def failing_load(path):
            raise RuntimeError("Simulated crash on the very first snapshot")

        mock_reader.load = failing_load

        with pytest.raises(RuntimeError, match="Simulated crash"):
            pipeline.run(str(tmp_path), seed=123)

        ckpt_path = os.path.join(str(tmp_path), "checkpoint.zst")
        assert os.path.exists(ckpt_path)
        ckpt = load_checkpoint(ckpt_path)
        assert ckpt["last_snapshot"] is None
        assert ckpt["seed"] == 123

        error_log_path = os.path.join(str(tmp_path), "error.log")
        assert os.path.exists(error_log_path)
        with open(error_log_path) as f:
            error_log_before = f.read()

        resume_pipeline, _, _, _ = _build_mock_pipeline(tmp_path, n_duplicate=2)
        # No explicit seed=: must adopt the checkpointed one, not
        # regenerate a fresh random one (the run()'s `if not resume:`
        # seed-regeneration block must not fire here).
        resume_pipeline.run(str(tmp_path), resume=True)

        assert resume_pipeline._base_seed == 123
        cat_reader = HDF5CatalogueReader(os.path.join(str(tmp_path), "catalogue.hdf5"))
        assert cat_reader.read_last_snapshot() == 1  # both snapshots completed

        # output_dir must not have been wiped: the crashed attempt's own
        # error.log survives the resume (proving `elif start_idx == 0:`
        # took the non-destructive branch, not `if not resume:`'s wipe).
        assert os.path.exists(error_log_path)
        with open(error_log_path) as f:
            assert f.read().startswith(error_log_before)

    def test_first_snapshot_of_resumed_run_fails_preserves_prior_checkpoint(self, tmp_path):
        """C02: the specific gap the multi-lens design review caught in an
        earlier draft -- if snapshot 0 succeeds, the run crashes on
        snapshot 1 (checkpointing snapshot 0's state), and then the FIRST
        snapshot processed during the resumed run (snapshot 1 again) ALSO
        fails, the second checkpoint must still correctly reflect
        snapshot 0's tracker state -- not get clobbered with a bogus
        `last_snapshot: None` "nothing succeeded" checkpoint, which is
        what happens if `last_good` is left hardcoded to `None` at the
        top of a resumed run instead of being seeded from the checkpoint
        `_try_resume` just loaded."""
        from roadrunner.io.serialization import load_checkpoint

        pipeline, _, mock_reader, _ = _build_mock_pipeline(tmp_path, n_duplicate=2)
        original_load = mock_reader.load
        call_count = [0]

        def failing_load(path):
            call_count[0] += 1
            if call_count[0] == 2:
                raise RuntimeError("Simulated crash on snapshot 1, first attempt")
            return original_load(path)

        mock_reader.load = failing_load

        with pytest.raises(RuntimeError, match="first attempt"):
            pipeline.run(str(tmp_path))

        ckpt = load_checkpoint(os.path.join(str(tmp_path), "checkpoint.zst"))
        assert ckpt["last_snapshot"] == 0

        # Resume, but make the FIRST snapshot processed this resumed run
        # (snapshot 1 again) fail too -- e.g. inside part_writer, well
        # after orchestrator.process() has mutated the trackers.
        resume_pipeline, _, _, _ = _build_mock_pipeline(tmp_path, n_duplicate=2)

        def always_failing_write(snapshot_id, time, redshift, snapshot_data):
            raise RuntimeError("Simulated crash on snapshot 1, resumed attempt")

        resume_pipeline.part_writer.write_snapshot = always_failing_write

        with pytest.raises(RuntimeError, match="resumed attempt"):
            resume_pipeline.run(str(tmp_path), resume=True)

        ckpt2 = load_checkpoint(os.path.join(str(tmp_path), "checkpoint.zst"))
        assert ckpt2["last_snapshot"] == 0  # must NOT have become None
        assert ckpt2["assembly_tracker"]["last_snapshot"] == 0
        assert ckpt2["birth_tracker"]["last_snapshot"] == 0

        # A further resume (this time letting it succeed) must still work.
        final_pipeline, _, _, _ = _build_mock_pipeline(tmp_path, n_duplicate=2)
        final_pipeline.run(str(tmp_path), resume=True)
        cat_reader = HDF5CatalogueReader(os.path.join(str(tmp_path), "catalogue.hdf5"))
        assert cat_reader.read_last_snapshot() == 1

    def test_first_snapshot_writer_failure_checkpoints_pristine_trackers(self, tmp_path):
        """C02: on a fresh run, snapshot 0's orchestrator.process() mutates
        the trackers before its writers run. If a writer then fails,
        nothing has completed yet -- but the checkpoint must still carry
        the untouched pre-run tracker state, not whatever snapshot 0
        already applied. Otherwise it would claim `last_snapshot: None`
        while its trackers say snapshot 0 is done, and a retry could
        apply snapshot 0 twice."""
        from roadrunner.io.serialization import load_checkpoint

        pipeline, _, _, _ = _build_mock_pipeline(tmp_path, n_duplicate=2)

        def failing_write(snapshot_id, time, redshift, snapshot_data):
            raise RuntimeError("Simulated crash inside part_writer on snapshot 0")

        pipeline.part_writer.write_snapshot = failing_write

        with pytest.raises(RuntimeError, match="Simulated crash"):
            pipeline.run(str(tmp_path), seed=7)

        # Sanity: snapshot 0 really did mutate the live trackers first.
        assert pipeline.orchestrator.birth_tracker._last_snapshot == 0
        assert pipeline.orchestrator.assembly_tracker._last_snapshot == 0

        ckpt = load_checkpoint(os.path.join(str(tmp_path), "checkpoint.zst"))
        assert ckpt["last_snapshot"] is None
        assert ckpt["birth_tracker"]["last_snapshot"] == -1
        assert ckpt["assembly_tracker"]["last_snapshot"] == -1
        assert ckpt["birth_tracker"]["active"]["pid"].size == 0
        assert len(ckpt["assembly_tracker"]["infall_lists"]) == 0

        resume_pipeline, _, _, _ = _build_mock_pipeline(tmp_path, n_duplicate=2)
        resume_pipeline.run(str(tmp_path), resume=True)
        assert resume_pipeline._base_seed == 7
        assert resume_pipeline.orchestrator.birth_tracker._last_snapshot == 1
        assert resume_pipeline.orchestrator.assembly_tracker._last_snapshot == 1
        cat_reader = HDF5CatalogueReader(os.path.join(str(tmp_path), "catalogue.hdf5"))
        assert cat_reader.read_last_snapshot() == 1

def test_energy_plausibility_checkpoint_resume(tmp_path):
    """NFW + energy plausibility: the refit histogram survives a crash and resume."""
    import h5py
    from roadrunner.io.serialization import load_checkpoint

    pipeline, _, mock_reader, _ = _build_mock_pipeline(
        tmp_path, n_duplicate=2, halo_model="nfw", plausibility="energy")
    original_load, calls = mock_reader.load, [0]

    def failing_load(path):
        calls[0] += 1
        if calls[0] == 2:
            raise RuntimeError("Simulated crash on snapshot 1")
        return original_load(path)

    mock_reader.load = failing_load
    with pytest.raises(RuntimeError, match="Simulated crash"):
        pipeline.run(str(tmp_path))
    state = load_checkpoint(os.path.join(str(tmp_path), "checkpoint.zst"))["plausibility"]
    assert state is not None
    np.testing.assert_array_equal(state["edges"], pipeline.orchestrator.assigner.plausibility.state["edges"])

    with h5py.File(os.path.join(str(tmp_path), "assignment", "snapshot0000.hdf5"), "r") as hf:
        for gid, grp in hf["galaxies"].items():
            b = grp["boundness"][:][grp["boundness_valid"][:]]
            assert grp.attrs["energy_scale"] > 0
            assert np.all((b > 0) & (b < 1))          # NFW boundness is E / Phi_0

    resumed, _, _, _ = _build_mock_pipeline(
        tmp_path, n_duplicate=2, halo_model="nfw", plausibility="energy")
    restored = []
    plaus = resumed.orchestrator.assigner.plausibility
    original_set = plaus.set_state
    plaus.set_state = lambda s: (restored.append(s), original_set(s))
    resumed.run(str(tmp_path), resume=True)
    np.testing.assert_array_equal(restored[0]["edges"], state["edges"])
    assert HDF5CatalogueReader(os.path.join(str(tmp_path), "catalogue.hdf5")).read_last_snapshot() == 1
