import numpy as np
import pytest

from roadrunner.postprocessing.properties import (
    random_lines_of_sight,
    rotation_matrix_from_los,
)


class TestRandomLinesOfSight:
    def test_returns_unit_vectors(self):
        N = 100
        vectors = random_lines_of_sight(N, seed=0)
        norms = np.linalg.norm(vectors, axis=1)
        assert norms.shape == (N,)
        assert np.allclose(norms, 1.0)

    def test_half_sphere_default(self):
        vectors = random_lines_of_sight(1000, seed=0)
        # all z should be >= 0 (cos_theta in [0, 1])
        assert np.all(vectors[:, 2] >= 0)

    def test_full_sphere(self):
        vectors = random_lines_of_sight(1000, half_sphere=False, seed=0)
        # z should span [-1, 1]
        assert np.any(vectors[:, 2] < 0)
        assert np.any(vectors[:, 2] > 0)

    def test_reproducible_seed(self):
        v1 = random_lines_of_sight(50, seed=42)
        v2 = random_lines_of_sight(50, seed=42)
        assert np.array_equal(v1, v2)

    def test_different_seeds_different(self):
        v1 = random_lines_of_sight(50, seed=0)
        v2 = random_lines_of_sight(50, seed=1)
        assert not np.array_equal(v1, v2)

    def test_N_zero_returns_empty(self):
        vectors = random_lines_of_sight(0, seed=0)
        assert vectors.shape == (0, 3)

    def test_N_one_returns_single_vector(self):
        vectors = random_lines_of_sight(1, seed=0)
        assert vectors.shape == (1, 3)
        assert np.isclose(np.linalg.norm(vectors[0]), 1.0)

    def test_coverage_uniform(self):
        vectors = random_lines_of_sight(5000, half_sphere=False, seed=0)
        # mean of unit vectors on sphere should be approx zero
        mean_pos = np.mean(vectors, axis=0)
        assert np.allclose(mean_pos, 0.0, atol=0.1)


class TestRotationMatrixFromLos:
    def test_aligns_los_with_z(self):
        los = np.array([1.0, 2.0, 3.0])
        R = rotation_matrix_from_los(los)
        rot_los = R @ (los / np.linalg.norm(los))
        assert np.allclose(rot_los, [0, 0, 1], atol=1e-10)

    def test_orthogonal(self):
        los = np.array([0.5, -1.0, 0.3])
        R = rotation_matrix_from_los(los)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-10)

    def test_determinant_one(self):
        los = np.array([-2.0, 1.0, 0.5])
        R = rotation_matrix_from_los(los)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-10)

    def test_x_axis_orthogonal_to_z(self):
        los = np.array([1.0, 1.0, 1.0])
        R = rotation_matrix_from_los(los)
        # e_x should be perpendicular to e_z
        assert np.isclose(R[0] @ R[2], 0.0, atol=1e-10)
        # e_y should be perpendicular to e_z
        assert np.isclose(R[1] @ R[2], 0.0, atol=1e-10)

    def test_z_aligned_los_preserved(self):
        los = np.array([0.0, 0.0, 1.0])
        R = rotation_matrix_from_los(los)
        # any rotation that keeps +z fixed is valid
        assert np.allclose(R @ los, los, atol=1e-10)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)

    def test_negative_z(self):
        los = np.array([0.0, 0.0, -1.0])
        R = rotation_matrix_from_los(los)
        rot_los = R @ (los / np.linalg.norm(los))
        assert np.allclose(rot_los, [0, 0, 1], atol=1e-10)

    def test_raises_on_wrong_shape(self):
        with pytest.raises(ValueError, match="3-element"):
            rotation_matrix_from_los(np.array([1.0, 2.0]))
        with pytest.raises(ValueError, match="3-element"):
            rotation_matrix_from_los(np.array([[1.0, 2.0, 3.0]]))

    def test_row_vector_convention(self):
        los = np.array([1.0, 2.0, 3.0])
        R = rotation_matrix_from_los(los)
        points = np.random.default_rng(0).uniform(-1, 1, (5, 3))
        # (R @ pᵢᵀ)ᵀ = pᵢ @ Rᵀ  (matrix identity)
        rot1 = (R @ points.T).T
        rot2 = points @ R.T
        assert np.allclose(rot1, rot2, atol=1e-10)
