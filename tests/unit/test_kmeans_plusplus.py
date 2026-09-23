import numpy as np
import pytest

from roadrunner.mixture._kmeans_plusplus import kmeans_plusplus_prior


class TestKMeansPlusPlus:
    def test_returns_centers_and_indices(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(100, 2))
        centers, indices = kmeans_plusplus_prior(X, 5, random_state=42)
        assert centers.shape == (5, 2)
        assert indices.shape == (5,)
        assert indices.dtype.kind in ("i", "u")

    def test_n_clusters_1(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        centers, indices = kmeans_plusplus_prior(X, 1, random_state=42)
        assert centers.shape == (1, 2)

    def test_custom_weights(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(50, 3))
        weights = np.ones((50, 3))
        weights[:10] = 10.0  # bias toward first 10
        centers, indices = kmeans_plusplus_prior(
            X, 3, cluster_weights=weights, random_state=42
        )
        assert centers.shape == (3, 3)

    def test_negative_weights_raises(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        weights = np.array([[1.0, -1.0], [1.0, 1.0]])
        with pytest.raises(ValueError, match="non-negative"):
            kmeans_plusplus_prior(X, 2, cluster_weights=weights)

    def test_zero_column_weight_raises(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        weights = np.array([[1.0, 0.0], [1.0, 0.0]])
        with pytest.raises(ValueError, match="zero total"):
            kmeans_plusplus_prior(X, 2, cluster_weights=weights)

    def test_all_identical_points(self):
        X = np.ones((10, 3))
        centers, indices = kmeans_plusplus_prior(X, 2, random_state=42)
        assert np.allclose(centers, 1.0)

    def test_reproducible_seed(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(50, 2))
        a, _ = kmeans_plusplus_prior(X, 3, random_state=42)
        b, _ = kmeans_plusplus_prior(X, 3, random_state=42)
        assert np.array_equal(a, b)

    def test_every_separated_blob_gets_a_center(self):
        rng = np.random.default_rng(42)
        n_blobs, per_blob = 10, 200
        centres = 100.0 * np.array([(i % 5, i // 5) for i in range(n_blobs)])
        X = np.repeat(centres, per_blob, axis=0) + 0.1 * rng.normal(size=(n_blobs * per_blob, 2))
        labels = np.repeat(np.arange(n_blobs), per_blob)
        for seed in range(10):
            _, indices = kmeans_plusplus_prior(X, n_blobs, random_state=seed)
            assert len(set(labels[indices])) == n_blobs
