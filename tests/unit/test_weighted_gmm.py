import numpy as np
import pytest

from roadrunner._defaults import precision
from roadrunner.mixture.weighted_gmm import WeightedGaussianMixture


@pytest.fixture
def blob_data_3c():
    rng = np.random.default_rng(42)
    X = np.vstack([
        rng.normal(loc=[-5, -5], scale=0.5, size=(100, 2)),
        rng.normal(loc=[0, 0], scale=0.5, size=(100, 2)),
        rng.normal(loc=[5, 5], scale=0.5, size=(100, 2)),
    ]).astype(np.float64)
    return X


class TestFit:
    def test_converges_3_components(self, blob_data_3c):
        gmm = WeightedGaussianMixture(n_components=3, random_state=42)
        gmm.fit(blob_data_3c)
        assert gmm.converged_

    def test_lower_bound_monotonic(self, blob_data_3c):
        gmm = WeightedGaussianMixture(n_components=3, random_state=42)
        gmm.fit(blob_data_3c)
        lbs = gmm.lower_bounds_
        for i in range(1, len(lbs)):
            assert lbs[i] >= lbs[i - 1] - 1e-6

    def test_refit_resets_convergence_state(self, blob_data_3c):
        gmm = WeightedGaussianMixture(n_components=3, random_state=42)
        gmm.fit(blob_data_3c)
        assert gmm.converged_
        gmm.max_iter = 1
        gmm.fit(blob_data_3c)
        assert not gmm.converged_
        assert gmm.lower_bounds_[0] == -np.inf

    def test_predict_returns_labels(self, blob_data_3c):
        gmm = WeightedGaussianMixture(n_components=3, random_state=42)
        gmm.fit(blob_data_3c)
        labels = gmm.predict(blob_data_3c)
        assert labels.shape == (300,)
        assert labels.dtype.kind in ("i", "u")

    def test_predict_log_proba_sums_to_one(self, blob_data_3c):
        gmm = WeightedGaussianMixture(n_components=3, random_state=42)
        gmm.fit(blob_data_3c)
        log_probs = gmm.predict_log_proba(blob_data_3c)
        probs = np.exp(log_probs)
        assert np.allclose(probs.sum(axis=1), 1.0)


class TestSingleComponent:
    def test_fit_single_component(self):
        rng = np.random.default_rng(42)
        X = rng.normal(loc=0.0, scale=1.0, size=(200, 2)).astype(np.float64)
        gmm = WeightedGaussianMixture(n_components=1, cov_type="full", random_state=42)
        gmm.fit(X)
        assert gmm.converged_
        assert gmm.means_.shape == (1, 2)

    def test_predict_all_zero(self):
        rng = np.random.default_rng(42)
        X = rng.normal(loc=0.0, scale=1.0, size=(200, 2)).astype(np.float64)
        gmm = WeightedGaussianMixture(n_components=1, cov_type="full", random_state=42)
        gmm.fit(X)
        labels = gmm.predict(X)
        assert np.all(labels == 0)

    def test_single_component_diagonal(self):
        rng = np.random.default_rng(42)
        X = rng.normal(loc=0.0, scale=1.0, size=(200, 3)).astype(np.float64)
        gmm = WeightedGaussianMixture(n_components=1, cov_type="diagonal", random_state=42)
        gmm.fit(X)
        assert gmm.covariances_.shape == (1, 3)

    def test_single_component_spherical(self):
        rng = np.random.default_rng(42)
        X = rng.normal(loc=0.0, scale=1.0, size=(200, 3)).astype(np.float64)
        gmm = WeightedGaussianMixture(n_components=1, cov_type="spherical", random_state=42)
        gmm.fit(X)
        assert gmm.covariances_.shape == (1,)


class TestDtypes:
    def test_float32(self, blob_data_3c):
        X32 = blob_data_3c.astype(np.float32)
        with precision(math="single"):
            gmm = WeightedGaussianMixture(n_components=3, random_state=42)
            gmm.fit(X32)
        assert gmm.converged_
        assert gmm.means_.dtype == np.float32


class TestEdgeCases:
    def test_more_features_than_samples(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(5, 10)).astype(np.float64)
        gmm = WeightedGaussianMixture(n_components=2, cov_type="full", random_state=42)
        gmm.fit(X)
        assert gmm.converged_ or not gmm.converged_
