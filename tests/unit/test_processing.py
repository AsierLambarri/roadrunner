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


class TestSmallHaloOwnership:
    """C05: a small halo's particles must not leak into (or bias) the fit
    of the large group it was split out of."""

    def test_large_group_fit_excludes_small_halos_particles(self):
        id_big, id_small = 1, 2
        rng = np.random.default_rng(0)

        n_big, n_small = 6, 3
        pos_big = rng.normal(scale=0.05, size=(n_big, 3))
        pos_small = np.array([15.0, 0.0, 0.0]) + rng.normal(scale=0.05, size=(n_small, 3))
        positions = np.vstack([pos_big, pos_small])
        velocities = np.zeros_like(positions)  # E = phi(r) < 0 everywhere: unconditionally bound
        n = positions.shape[0]

        snap_data = SnapshotData(
            index=np.arange(n, dtype=np.uint64),
            mass=np.ones(n),
            position=positions,
            velocity=velocities,
            redshift=0.0,
            time=13.8,
        )
        snap_df = pd.DataFrame({
            "Sub_tree_id": [id_big, id_small],
            "Redshift": [0.0, 0.0],
            "position_x": [0.0, 15.0], "position_y": [0.0, 0.0], "position_z": [0.0, 0.0],
            "velocity_x": [0.0, 0.0], "velocity_y": [0.0, 0.0], "velocity_z": [0.0, 0.0],
            "mass": [1e12, 1e9],
            "virial_radius": [30.0, 5.0],
            "scale_radius": [5.0, 1.0],
        })

        assigner = XGMMAssigner(
            method="gmm", cov_type="full", max_iter=10, tol=1e-2,
            min_particles=5, reg_covar=1e-6, prior_type="", verbose=0,
        )
        config = ProcessingConfig(min_particles=5)
        newborn = np.arange(n, dtype=np.uint64)

        ensemble, result = process_snapshot(
            snap_data, snap_df, newborn, None, assigner, config,
        )

        # The 3 small-halo particles must be exclusively the small halo's:
        # responsibility 1 under id_small, 0 under id_big.
        dense = result.responsibilities.to_dense(columns=np.array([id_big, id_small]))
        row_id = result.responsibilities.row_id
        small_rows = np.arange(n_big, n)
        pos_in_dense = np.searchsorted(row_id, small_rows)
        np.testing.assert_allclose(dense[pos_in_dense, 0], 0.0, atol=1e-8)
        np.testing.assert_allclose(dense[pos_in_dense, 1], 1.0, atol=1e-8)

        # The big halo's fitted mean must reflect ONLY its own 6 particles
        # (mean_x ~ 0), not the 3 contested ones at x=15 (which would pull
        # it to x ~ 5.0 if they had leaked into the fit).
        mean_big = np.asarray(result.fitted_parameters[id_big]["mean"])
        assert mean_big[0] < 1.0, (
            f"big halo's fitted mean_x={mean_big[0]} suggests the small "
            "halo's particles leaked into its fit (expected ~0, not ~5)"
        )


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