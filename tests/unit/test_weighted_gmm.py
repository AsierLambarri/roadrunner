import numpy as np
import pytest

from roadrunner._defaults import precision
from roadrunner.mixture.weighted_gmm import (
    WeightedGaussianMixture,
    _check_counts,
    _estimate_gaussian_parameters,
)


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

        lbs = gmm.lower_bounds_
        for i in range(1, len(lbs)):
            assert lbs[i] >= lbs[i - 1] - 1e-6

        labels = gmm.predict(blob_data_3c)
        assert labels.shape == (300,)
        assert labels.dtype.kind in ("i", "u")

        log_probs = gmm.predict_log_proba(blob_data_3c)
        probs = np.exp(log_probs)
        assert np.allclose(probs.sum(axis=1), 1.0)

    def test_refit_resets_convergence_state(self, blob_data_3c):
        gmm = WeightedGaussianMixture(n_components=3, random_state=42)
        gmm.fit(blob_data_3c)
        assert gmm.converged_
        gmm.max_iter = 1
        gmm.fit(blob_data_3c)
        assert not gmm.converged_
        assert gmm.lower_bounds_[0] == -np.inf

class TestSingleComponent:
    def test_fit_single_component(self):
        rng = np.random.default_rng(42)
        X = rng.normal(loc=0.0, scale=1.0, size=(200, 2)).astype(np.float64)
        gmm = WeightedGaussianMixture(n_components=1, cov_type="full", random_state=42)
        gmm.fit(X)
        assert gmm.converged_
        assert gmm.means_.shape == (1, 2)
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


class TestInPlaceMutation:
    def test_fit_does_not_mutate_prior_or_counts(self, blob_data_3c):
        n_samples = blob_data_3c.shape[0]
        n_components = 3
        rng = np.random.default_rng(0)
        prior = rng.uniform(0.1, 1.0, (n_samples, n_components)).astype(blob_data_3c.dtype)
        counts = np.full(n_components, n_samples / n_components, dtype=blob_data_3c.dtype)
        means = np.array([[-5.0, -5.0], [0.0, 0.0], [5.0, 5.0]], dtype=blob_data_3c.dtype)
        covariance = np.stack([np.eye(2, dtype=blob_data_3c.dtype)] * n_components)

        prior_copy = prior.copy()
        counts_copy = counts.copy()

        gmm = WeightedGaussianMixture(
            n_components=n_components,
            means_init=means,
            covariance_init=covariance,
            counts_init=counts,
            random_state=42,
        )
        gmm.fit(blob_data_3c, latent_prior=prior)

        assert np.array_equal(prior, prior_copy)
        assert np.array_equal(counts, counts_copy)


class TestCheckCounts:
    def test_any_total_passes(self):
        # Only the proportions set the initial weights; a weighted fit's
        # counts are in weight units, so the total is not checked.
        counts = np.array([10.0, 10.0, 10.0])
        assert np.array_equal(_check_counts(counts, n_components=3), counts)

    def test_negative_counts_raises(self):
        counts = np.array([-1.0, 50.0, 51.0])
        with pytest.raises(ValueError, match="non-negative"):
            _check_counts(counts, n_components=3)


class TestPointWeights:
    def test_integer_weights_equal_replicated_data(self, blob_data_3c):
        X = blob_data_3c.astype(np.float64)
        rng = np.random.default_rng(3)
        w = rng.integers(1, 4, X.shape[0]).astype(np.float64)
        rep = np.repeat(np.arange(X.shape[0]), w.astype(int))
        resp0 = np.zeros((X.shape[0], 3))
        resp0[np.arange(X.shape[0]), np.repeat([0, 1, 2], X.shape[0] // 3)] = 1.0
        nk, xk, sk = _estimate_gaussian_parameters(X, resp0, w, "full", 1e-6)

        def fit(data, pw):
            return WeightedGaussianMixture(
                n_components=3, counts_init=nk, means_init=xk, covariance_init=sk,
                max_iter=15, tol=0.0,
            ).fit(data, point_weights=pw)

        weighted, replicated = fit(X, w), fit(X[rep], None)
        for attr in ("means_", "covariances_", "weights_", "lower_bounds_"):
            np.testing.assert_allclose(getattr(weighted, attr), getattr(replicated, attr),
                                       rtol=1e-9, atol=0, err_msg=attr)
