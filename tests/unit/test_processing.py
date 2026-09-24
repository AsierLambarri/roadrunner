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


class TestExactIntegerSubTreeId:
    """C09: halo identities above 2**53 must survive process_snapshot exactly."""

    def test_synthetic_two_halo_exact_ids(self):
        id_a = 2**53 + 1
        id_b = 2**53 + 2
        centre_a = np.array([0.0, 0.0, 0.0])
        centre_b = np.array([500.0, 0.0, 0.0])

        rng = np.random.default_rng(0)
        n_per_halo = 50
        pos_a = centre_a + rng.normal(scale=1.0, size=(n_per_halo, 3))
        pos_b = centre_b + rng.normal(scale=1.0, size=(n_per_halo, 3))
        vel_a = rng.normal(scale=5.0, size=(n_per_halo, 3))
        vel_b = rng.normal(scale=5.0, size=(n_per_halo, 3))

        positions = np.vstack([pos_a, pos_b])
        velocities = np.vstack([vel_a, vel_b])
        n = positions.shape[0]
        masses = np.ones(n)

        snap_data = SnapshotData(
            index=np.arange(n, dtype=np.uint64),
            mass=masses,
            position=positions,
            velocity=velocities,
            redshift=0.0,
            time=13.8,
        )

        snap_df = pd.DataFrame({
            "Sub_tree_id": np.array([id_a, id_b], dtype=np.int64),
            "Redshift": [0.0, 0.0],
            "position_x": [centre_a[0], centre_b[0]],
            "position_y": [centre_a[1], centre_b[1]],
            "position_z": [centre_a[2], centre_b[2]],
            "velocity_x": [0.0, 0.0],
            "velocity_y": [0.0, 0.0],
            "velocity_z": [0.0, 0.0],
            "mass": [1e10, 1e10],
            "virial_radius": [50.0, 50.0],
            "scale_radius": [5.0, 5.0],
        })

        assigner = XGMMAssigner(
            method="gmm", cov_type="full", max_iter=10, tol=1e-2,
            min_particles=10, reg_covar=1e-6, prior_type="", verbose=0,
        )
        newborn = np.arange(n, dtype=np.uint64)

        ensemble, result = process_snapshot(
            snap_data, snap_df, newborn, None, assigner, ProcessingConfig(),
        )

        assert ensemble.sub_tree_ids.dtype == np.int64
        assert ensemble.sub_tree_ids.tolist() == [id_a, id_b]

        valid_ids = {id_a, id_b, -1}
        assert set(result.particle_df["Sub_tree_id"].unique()).issubset(valid_ids)