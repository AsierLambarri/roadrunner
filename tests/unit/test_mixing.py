import numpy as np
import pandas as pd
import pytest

from roadrunner.physics.constants import (
    RILEY_BOUND_THRESHOLD,
    RILEY_SVM_SLOPE,
    RILEY_SVM_INTERCEPT,
)
from roadrunner.postprocessing.mixing import (
    _local_velocity_dispersion,
    _riley_criterion_single,
    compute_riley_criterion,
)


class TestLocalVelocityDispersion:
    def test_enough_particles_returns_finite(self):
        rng = np.random.default_rng(42)
        pos = rng.normal(0, 10, (200, 3))
        vel = rng.normal(0, 100, (200, 3))
        result = _local_velocity_dispersion(pos, vel)
        assert result.shape == (200,)
        assert np.all(np.isfinite(result))

    def test_too_few_particles_returns_nan(self):
        pos = np.random.default_rng(42).normal(0, 10, (5, 3))
        vel = np.random.default_rng(42).normal(0, 100, (5, 3))
        result = _local_velocity_dispersion(pos, vel, nmin=10)
        assert result.shape == (5,)
        assert np.all(np.isnan(result))

    def test_single_particle_returns_nan(self):
        pos = np.array([[0.0, 0.0, 0.0]])
        vel = np.array([[10.0, 20.0, 30.0]])
        result = _local_velocity_dispersion(pos, vel)
        assert np.isnan(result[0])


class TestRileyCriterionSingle:
    def test_intact_high_f_bound(self):
        rng = np.random.default_rng(42)
        pos = rng.normal(0, 1, (50, 3))
        vel = rng.normal(0, 10, (50, 3))
        result = _riley_criterion_single(
            10, mstar=1e5, f_bound=0.98,
            subset_positions=pos, subset_velocities=vel,
        )
        assert result["dynstate"] == 0
        assert result["Sub_tree_id"] == 10
        assert result["mstar"] == 1e5

    def test_stream_low_f_bound_low_sigma(self):
        rng = np.random.default_rng(42)
        pos = rng.normal(0, 1, (200, 3))
        # very low velocity dispersion → sigma50 < SVM
        vel = rng.normal(0, 0.1, (200, 3))
        result = _riley_criterion_single(
            20, mstar=1e8, f_bound=0.5,
            subset_positions=pos, subset_velocities=vel,
        )
        assert result["dynstate"] == 1
        assert result["sigma50"] < RILEY_SVM_SLOPE * np.log10(1e8) + RILEY_SVM_INTERCEPT

    def test_phase_mixed_low_f_bound_high_sigma(self):
        rng = np.random.default_rng(42)
        pos = rng.normal(0, 1, (200, 3))
        # very high velocity dispersion → sigma50 ≥ SVM
        vel = rng.normal(0, 500, (200, 3))
        result = _riley_criterion_single(
            30, mstar=1e6, f_bound=0.5,
            subset_positions=pos, subset_velocities=vel,
        )
        assert result["dynstate"] == 2
        svm = RILEY_SVM_SLOPE * np.log10(1e6) + RILEY_SVM_INTERCEPT
        assert result["sigma50"] >= svm

    def test_zero_mstar(self):
        pos = np.empty((0, 3))
        vel = np.empty((0, 3))
        result = _riley_criterion_single(
            40, mstar=0.0, f_bound=0.0,
            subset_positions=pos, subset_velocities=vel,
        )
        assert result["mstar"] == 0.0
        assert result["dynstate"] in (0, 1, 2)  # should handle gracefully


class TestComputeRileyCriterion:
    def test_output_columns(self):
        rng = np.random.default_rng(42)
        N = 300
        masses = rng.uniform(0.1, 1.0, N)
        coords = np.column_stack([
            rng.normal(0, 10, N), rng.normal(0, 10, N), rng.normal(0, 10, N),
            rng.normal(0, 50, N), rng.normal(0, 50, N), rng.normal(0, 50, N),
        ])
        galaxy_allowed = {10: np.arange(0, 100), 20: np.arange(100, 200)}
        galaxy_bound = {10: np.arange(0, 80), 20: np.arange(100, 150)}
        result = compute_riley_criterion(
            main_id=1,
            particle_masses=masses,
            particle_coords=coords,
            galaxy_allowed=galaxy_allowed,
            galaxy_bound=galaxy_bound,
            redshift=0.1,
        )
        assert list(result.columns) == [
            "Sub_tree_id", "mstar", "f_bound", "sigma50", "dynstate",
        ]
        assert len(result) == 2
        assert set(result["Sub_tree_id"]) == {10, 20}

    def test_f_bound_computed_from_intersection(self):
        rng = np.random.default_rng(42)
        masses = np.ones(100)
        coords = np.column_stack([
            rng.normal(0, 10, 100), rng.normal(0, 10, 100), rng.normal(0, 10, 100),
            rng.normal(0, 50, 100), rng.normal(0, 50, 100), rng.normal(0, 50, 100),
        ])
        # allowed = [0..99], bound = [0..49] → f_bound = 0.5
        galaxy_allowed = {5: np.arange(100)}
        galaxy_bound = {5: np.arange(50)}
        result = compute_riley_criterion(
            main_id=1,
            particle_masses=masses,
            particle_coords=coords,
            galaxy_allowed=galaxy_allowed,
            galaxy_bound=galaxy_bound,
            redshift=0.0,
        )
        assert np.isclose(result.iloc[0]["f_bound"], 0.5)

    def test_main_id_and_minus_one_skipped(self):
        rng = np.random.default_rng(42)
        masses = np.ones(100)
        coords = np.column_stack([
            rng.normal(0, 10, 100), rng.normal(0, 10, 100), rng.normal(0, 10, 100),
            rng.normal(0, 50, 100), rng.normal(0, 50, 100), rng.normal(0, 50, 100),
        ])
        galaxy_allowed = {1: np.arange(30), -1: np.arange(30, 60), 5: np.arange(60, 100)}
        galaxy_bound = {}
        result = compute_riley_criterion(
            main_id=1,
            particle_masses=masses,
            particle_coords=coords,
            galaxy_allowed=galaxy_allowed,
            galaxy_bound=galaxy_bound,
            redshift=0.0,
        )
        assert len(result) == 1
        assert result.iloc[0]["Sub_tree_id"] == 5

    def test_empty_allowed_returns_empty_dataframe(self):
        result = compute_riley_criterion(
            main_id=1,
            particle_masses=np.array([]),
            particle_coords=np.empty((0, 6)),
            galaxy_allowed={},
            galaxy_bound={},
            redshift=0.0,
        )
        assert len(result) == 0
        assert list(result.columns) == [
            "Sub_tree_id", "mstar", "f_bound", "sigma50", "dynstate",
        ]

    def test_empty_allowed_indices_skipped(self):
        rng = np.random.default_rng(42)
        masses = np.ones(50)
        coords = np.column_stack([
            rng.normal(0, 10, 50), rng.normal(0, 10, 50), rng.normal(0, 10, 50),
            rng.normal(0, 50, 50), rng.normal(0, 50, 50), rng.normal(0, 50, 50),
        ])
        galaxy_allowed = {5: np.array([], dtype=int), 10: np.arange(50)}
        galaxy_bound = {}
        result = compute_riley_criterion(
            main_id=1,
            particle_masses=masses,
            particle_coords=coords,
            galaxy_allowed=galaxy_allowed,
            galaxy_bound=galaxy_bound,
            redshift=0.0,
        )
        assert len(result) == 1
        assert result.iloc[0]["Sub_tree_id"] == 10
