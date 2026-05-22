import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from roadrunner._mcf_types import SnapshotData
from roadrunner.clustering.assignment.gmm import GMMAssigner
from roadrunner.pipeline.processing import ProcessingConfig
from roadrunner.pipeline.reduction import ReductionConfig
from roadrunner.pipeline.snapshot_orchestrator import (
    SnapshotOrchestrator,
    SnapshotResult,
)
from roadrunner.postprocessing.tracking.assembly import AssemblyTracker
from roadrunner.postprocessing.tracking.birth import BirthTracker


DATA_DIR = "test_data/mock_snap_tight"


def _load_mock():
    tree = pd.read_csv(os.path.join(DATA_DIR, "merger_tree.csv"))
    particles = np.load(os.path.join(DATA_DIR, "particles.npz"))
    coords = particles["coords"]
    masses = particles["masses"]
    snap_data = SnapshotData(
        indices=np.arange(coords.shape[0], dtype=np.uint64),
        masses=masses,
        positions=coords[:, :3],
        velocities=coords[:, 3:6],
        redshift=0.0,
        time=13.8,
    )
    if "host_id" not in tree.columns:
        tree["host_id"] = -1
    if "distance_to_acc_id" not in tree.columns:
        tree["distance_to_acc_id"] = 0.0
    return tree, coords, masses, snap_data


def _make_assigner():
    return GMMAssigner(
        cov_type="full", max_iter=5, tol=1e-2,
        min_particles=10, reg_covar=1e-6, prior_type="", verbose=0,
    )


class TestSnapshotOrchestratorWithoutTrackers:
    def test_process_returns_snapshot_result(self):
        tree, coords, masses, snap_data = _load_mock()
        orchestrator = SnapshotOrchestrator(
            processing_config=ProcessingConfig(halo_model="kepler"),
            reduction_config=ReductionConfig(accretion_id=1),
            assigner=_make_assigner(),
        )
        result = orchestrator.process(
            snap_id=0, snap_df=tree, snap_data=snap_data,
            satellites={},
        )
        assert isinstance(result, SnapshotResult)
        assert result.ensemble is not None
        assert result.result.particle_df is not None
        assert result.properties is not None
        assert result.dynstate is not None
        assert "Sub_tree_id" in result.result.particle_df.columns
        assert "timescale" in result.result.particle_df.columns

    def test_process_without_trackers_still_produces_properties(self):
        tree, coords, masses, snap_data = _load_mock()
        orchestrator = SnapshotOrchestrator(
            processing_config=ProcessingConfig(halo_model="kepler"),
            reduction_config=ReductionConfig(accretion_id=1),
            assigner=_make_assigner(),
        )
        result = orchestrator.process(
            snap_id=0, snap_df=tree, snap_data=snap_data,
            satellites={},
        )
        assert result.properties is not None
        assert len(result.properties) > 0
        assert "Sub_tree_id" in result.properties.columns
        assert "Mtot" in result.properties.columns

    def test_process_twice_accumulates_no_tracker_state(self):
        tree, coords, masses, snap_data = _load_mock()
        orchestrator = SnapshotOrchestrator(
            processing_config=ProcessingConfig(halo_model="kepler"),
            reduction_config=ReductionConfig(accretion_id=1),
            assigner=_make_assigner(),
        )
        result1 = orchestrator.process(
            snap_id=0, snap_df=tree, snap_data=snap_data,
            satellites={},
        )
        result2 = orchestrator.process(
            snap_id=1, snap_df=tree, snap_data=snap_data,
            satellites={},
            previous_resp_sim=result1.previous_resp_sim,
        )
        assert len(result2.result.particle_df) == len(result1.result.particle_df)


class TestSnapshotOrchestratorWithTrackers:
    def test_process_with_birth_tracker(self):
        tree, coords, masses, snap_data = _load_mock()
        bt = BirthTracker(factor=5, enforce_initial_hosts=False)
        orchestrator = SnapshotOrchestrator(
            processing_config=ProcessingConfig(halo_model="kepler"),
            reduction_config=ReductionConfig(accretion_id=1),
            assigner=_make_assigner(),
            birth_tracker=bt,
        )
        result = orchestrator.process(
            snap_id=0, snap_df=tree, snap_data=snap_data,
            satellites={},
        )
        assert isinstance(result, SnapshotResult)
        assert result.properties is not None

    def test_trackers_accumulate_state_across_snapshots(self):
        tree, coords, masses, snap_data = _load_mock()
        bt = BirthTracker(factor=5, enforce_initial_hosts=False)
        at = AssemblyTracker(n_sat_history=2)
        orchestrator = SnapshotOrchestrator(
            processing_config=ProcessingConfig(halo_model="kepler"),
            reduction_config=ReductionConfig(accretion_id=1),
            assigner=_make_assigner(),
            birth_tracker=bt,
            assembly_tracker=at,
        )
        result1 = orchestrator.process(
            snap_id=0, snap_df=tree, snap_data=snap_data,
            satellites={},
        )
        result2 = orchestrator.process(
            snap_id=1, snap_df=tree, snap_data=snap_data,
            satellites={},
            previous_resp_sim=result1.previous_resp_sim,
        )
        assert bt._last_snapshot == 1
        assert at._last_snapshot == 1
        assert len(at.current()) > 0

    def test_previous_resp_roundtrip_sim_space(self):
        tree, coords, masses, snap_data = _load_mock()
        orchestrator = SnapshotOrchestrator(
            processing_config=ProcessingConfig(halo_model="kepler"),
            reduction_config=ReductionConfig(accretion_id=1),
            assigner=_make_assigner(),
        )
        result1 = orchestrator.process(
            snap_id=0, snap_df=tree, snap_data=snap_data,
            satellites={},
        )
        assert result1.previous_resp_sim is not None
        assert len(result1.previous_resp_sim) > 0

    def test_dynstate_columns(self):
        tree, coords, masses, snap_data = _load_mock()
        orchestrator = SnapshotOrchestrator(
            processing_config=ProcessingConfig(halo_model="kepler"),
            reduction_config=ReductionConfig(accretion_id=1),
            assigner=_make_assigner(),
        )
        result = orchestrator.process(
            snap_id=0, snap_df=tree, snap_data=snap_data,
            satellites={},
        )
        assert result.dynstate is not None
        expected = ["Sub_tree_id", "mstar", "f_bound", "sigma50", "dynstate"]
        assert list(result.dynstate.columns) == expected