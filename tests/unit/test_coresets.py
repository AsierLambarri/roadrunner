import numpy as np
import pytest

from roadrunner.mixture.coresets import (
    GaussianCoreset,
    _estimate_mahalanobis_squared,
    _exact_fudge,
)


@pytest.fixture
def sample_data():
    rng = np.random.default_rng(42)
    X = np.vstack([
        rng.normal(loc=[-3, -3], scale=0.5, size=(100, 2)),
        rng.normal(loc=[3, 3], scale=0.5, size=(100, 2)),
    ]).astype(np.float64)
    return X


class TestFitAndGenerate:
    def test_fit_returns_self(self, sample_data):
        coreset = GaussianCoreset(n_components=2, random_state=42)
        result = coreset.fit(sample_data)
        assert result is coreset

    def test_generate_coreset_shape(self, sample_data):
        coreset = GaussianCoreset(n_components=2, random_state=42)
        coreset.fit(sample_data)
        C, w, idx = coreset.generate_coreset(50)
        assert C.shape[1] == 2
        assert w.shape[0] == C.shape[0]
        assert idx.shape[0] == C.shape[0]

    def test_weights_positive(self, sample_data):
        coreset = GaussianCoreset(n_components=2, random_state=42)
        coreset.fit(sample_data)
        C, w, idx = coreset.generate_coreset(50)
        assert np.all(w > 0)

    def test_estimate_error_finite(self, sample_data):
        coreset = GaussianCoreset(n_components=2, random_state=42)
        coreset.fit(sample_data)
        C, w, idx = coreset.generate_coreset(50)
        eps_sum, eps_mean = coreset.estimate_error(idx, w)
        assert np.isfinite(eps_sum)
        assert np.isfinite(eps_mean)


class TestInitialize:
    def test_initialize_centroids(self, sample_data):
        coreset = GaussianCoreset(n_components=2, random_state=42)
        coreset._initialize(sample_data)
        assert coreset.centers.shape == (2, 2)
        assert coreset.covariances is not None


class TestMahalanobis:
    def test_returns_correct_shape(self, sample_data):
        coreset = GaussianCoreset(n_components=2, random_state=42)
        coreset.fit(sample_data)
        quad = _estimate_mahalanobis_squared(
            sample_data, coreset.centers, coreset.covariances, coreset.cov_type
        )
        assert quad.shape == (200, 2)

    def test_nonnegative(self, sample_data):
        coreset = GaussianCoreset(n_components=2, cov_type="full", random_state=42)
        coreset.fit(sample_data)
        quad = _estimate_mahalanobis_squared(
            sample_data, coreset.centers, coreset.covariances, coreset.cov_type
        )
        assert np.all(quad >= -1e-6)


class TestExactFudge:
    def test_uniform_probabilities(self):
        q = np.ones(1000) / 1000
        fudge = _exact_fudge(q, M=100, tol=1e-2)
        assert np.isfinite(fudge)
        assert fudge > 0

    def test_raises_on_nonzero_negative(self):
        q = np.array([0.5, -0.1, 0.6])
        with pytest.raises(ValueError, match="nonnegative"):
            _exact_fudge(q, M=10)

    def test_raises_on_bad_sum(self):
        q = np.array([0.3, 0.3])
        with pytest.raises(ValueError, match="sum to 1"):
            _exact_fudge(q, M=5)
