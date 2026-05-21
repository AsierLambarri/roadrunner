import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from roadrunner.pipeline.snapshot_processor import SnapshotProcessor
from roadrunner.clustering.assignment.gmm import GMMAssigner
from roadrunner._mcf_types import SnapshotData


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


class TestSnapshotProcessor:
    def test_process_returns_assignment(self):
        tree, coords, masses, snap_data = _load_mock()
        assigner = GMMAssigner(
            cov_type="full", max_iter=5, tol=1e-2,
            min_particles=10, reg_covar=1e-6, prior_type="", verbose=0,
        )
        sp = SnapshotProcessor(
            assigner=assigner,
            halo_model="kepler",
            accretion_id=1,
        )
        newborn = np.arange(coords.shape[0], dtype=np.uint64)
        ensemble, result = sp.process(
            tree, snap_data, newborn,
        )
        assert isinstance(ensemble, object)
        assert result.particle_df is not None
        assert "Sub_tree_id" in result.particle_df.columns
        assert "timescale" in result.particle_df.columns
        assert (result.particle_df["Sub_tree_id"] != -1).all()

    def test_process_twice_with_previous_resp(self):
        tree, coords, masses, snap_data = _load_mock()
        assigner = GMMAssigner(
            cov_type="full", max_iter=3, tol=1e-2,
            min_particles=10, reg_covar=1e-6, prior_type="", verbose=0,
        )
        sp = SnapshotProcessor(
            assigner=assigner,
            halo_model="kepler",
            accretion_id=1,
        )
        newborn = np.arange(coords.shape[0], dtype=np.uint64)

        ensemble, r1 = sp.process(tree, snap_data, newborn)
        prev_csc = r1.responsibilities

        ensemble, r2 = sp.process(
            tree, snap_data, newborn, previous_resp=prev_csc,
        )
        assert len(r2.particle_df) == len(r1.particle_df)
        assert r2.statistics.get("unassigned", 0) == 0

    def test_no_trackers_or_writers_in_processor(self):
        sp = SnapshotProcessor(
            assigner=GMMAssigner(verbose=0, max_iter=1),
            halo_model="kepler",
            accretion_id=1,
        )
        assert not hasattr(sp, "birth_tracker")
        assert not hasattr(sp, "cat_writer")
