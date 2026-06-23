import os
import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")

from roadrunner.pipeline.processing import ProcessingConfig, process_snapshot
from roadrunner._mcf_types import SnapshotData
from roadrunner.clustering.assignment.gmm import XGMMAssigner
from roadrunner.physics.halo_ensemble import HaloEnsemble


DATA_DIR = "test_data/mock_snap_tight"


def _load_mock():
    tree = pd.read_csv(os.path.join(DATA_DIR, "merger_tree.csv"))
    particles = np.load(os.path.join(DATA_DIR, "particles.npz"))
    coords = particles["coords"]
    masses = particles["masses"]
    snap_data = SnapshotData(
        index=np.arange(coords.shape[0], dtype=np.uint64),
        mass=masses,
        position=coords[:, :3],
        velocity=coords[:, 3:6],
        redshift=0.0,
        time=13.8,
    )
    if "host_id" not in tree.columns:
        tree["host_id"] = -1
    if "distance_to_acc_id" not in tree.columns:
        tree["distance_to_acc_id"] = 0.0
    return tree, coords, masses, snap_data


_DEFAULT_CONFIG = ProcessingConfig()


class TestProcessingConfig:
    def test_defaults(self):
        cfg = ProcessingConfig()
        assert cfg.halo_model == "kepler"
        assert cfg.search_factor == 1.0
        assert cfg.min_particles == 10

    def test_custom(self):
        cfg = ProcessingConfig(halo_model="nfw", search_factor=2.0, min_particles=5)
        assert cfg.halo_model == "nfw"
        assert cfg.search_factor == 2.0
        assert cfg.min_particles == 5

    def test_frozen(self):
        cfg = ProcessingConfig()
        with pytest.raises(AttributeError):
            cfg.halo_model = "nfw"


class TestProcessSnapshot:
    def test_returns_ensemble_and_result(self):
        tree, coords, masses, snap_data = _load_mock()
        assigner = XGMMAssigner(
            cov_type="full", max_iter=5, tol=1e-2,
            min_particles=10, reg_covar=1e-6, prior_type="", verbose=0,
        )
        newborn = np.arange(coords.shape[0], dtype=np.uint64)
        ensemble, result = process_snapshot(
            snap_data, tree, newborn, None, assigner, _DEFAULT_CONFIG,
        )
        assert isinstance(ensemble, HaloEnsemble)
        assert result.particle_df is not None
        assert "Sub_tree_id" in result.particle_df.columns
        assert "timescale" in result.particle_df.columns

    def test_with_previous_resp(self):
        tree, coords, masses, snap_data = _load_mock()
        assigner = XGMMAssigner(
            cov_type="full", max_iter=3, tol=1e-2,
            min_particles=10, reg_covar=1e-6, prior_type="", verbose=0,
        )
        newborn = np.arange(coords.shape[0], dtype=np.uint64)
        _, r1 = process_snapshot(
            snap_data, tree, newborn, None, assigner, _DEFAULT_CONFIG,
        )
        prev_csc = r1.responsibilities

        _, r2 = process_snapshot(
            snap_data, tree, newborn, prev_csc, assigner, _DEFAULT_CONFIG,
        )
        assert len(r2.particle_df) == len(r1.particle_df)

    def test_search_factor_affects_boundness(self):
        tree, coords, masses, snap_data = _load_mock()
        assigner = XGMMAssigner(
            cov_type="full", max_iter=5, tol=1e-2,
            min_particles=10, reg_covar=1e-6, prior_type="", verbose=0,
        )
        newborn = np.arange(coords.shape[0], dtype=np.uint64)

        cfg_small = ProcessingConfig(search_factor=0.5)
        cfg_large = ProcessingConfig(search_factor=4.0)

        ens_small, _ = process_snapshot(
            snap_data, tree, newborn, None, assigner, cfg_small,
        )
        ens_large, _ = process_snapshot(
            snap_data, tree, newborn, None, assigner, cfg_large,
        )
        assert ens_large.nstars >= ens_small.nstars

    def test_min_particles_affects_groups(self):
        tree, coords, masses, snap_data = _load_mock()
        assigner = XGMMAssigner(
            cov_type="full", max_iter=5, tol=1e-2,
            min_particles=5, reg_covar=1e-6, prior_type="", verbose=0,
        )
        newborn = np.arange(coords.shape[0], dtype=np.uint64)

        _, result = process_snapshot(
            snap_data, tree, newborn, None, assigner,
            ProcessingConfig(min_particles=5),
        )
        assert result.particle_df is not None
        assert len(result.particle_df) > 0