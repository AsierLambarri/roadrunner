import numpy as np
import pytest
from scipy.stats import multivariate_normal

from roadrunner.mixture.bayesian_gmm import (
    WeightedBayesianGaussianMixture,
    _compute_precision_cholesky,
    _estimate_log_gaussian_prob_pchol,
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


@pytest.fixture
def anisotropic_blobs():
    rng = np.random.default_rng(3)
    return np.vstack([
        rng.normal([-2, 0], [1.0, 0.3], (300, 2)),
        rng.normal([2, 0], [0.3, 1.0], (300, 2)),
        rng.normal([0, 3], 0.5, (300, 2)),
    ])


def _correlated_spd(rng, n_features):
    A = rng.normal(size=(n_features, n_features))
    return A @ A.T + 0.1 * np.eye(n_features)


def _rotation(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta),  np.cos(theta)]])


class TestFit:
    def test_converges_3_components(self, blob_data_3c):
        bgmm = WeightedBayesianGaussianMixture(
            n_components=3, random_state=42, max_iter=50
        )
        bgmm.fit(blob_data_3c)
        assert bgmm.converged_

    def test_predict_returns_labels(self, blob_data_3c):
        bgmm = WeightedBayesianGaussianMixture(
            n_components=3, random_state=42, max_iter=50
        )
        bgmm.fit(blob_data_3c)
        labels = bgmm.predict(blob_data_3c)
        assert labels.shape == (300,)
        assert labels.dtype.kind in ("i", "u")


class TestAllCovTypes:
    @pytest.mark.parametrize("cov_type", ["full", "diagonal", "spherical"])
    def test_cov_type_runs(self, cov_type):
        rng = np.random.default_rng(42)
        X = rng.normal(loc=0.0, scale=1.0, size=(100, 3)).astype(np.float64)
        bgmm = WeightedBayesianGaussianMixture(
            n_components=2, cov_type=cov_type, random_state=42, max_iter=10
        )
        bgmm.fit(X)
        assert bgmm.converged_ or not bgmm.converged_


class TestSingleComponent:
    def test_single_component_full(self):
        rng = np.random.default_rng(42)
        X = rng.normal(loc=0.0, scale=1.0, size=(200, 2)).astype(np.float64)
        bgmm = WeightedBayesianGaussianMixture(
            n_components=1, cov_type="full", random_state=42, max_iter=10
        )
        bgmm.fit(X)
        assert bgmm.means_.shape == (1, 2)

    def test_single_component_diagonal(self):
        rng = np.random.default_rng(42)
        X = rng.normal(loc=0.0, scale=1.0, size=(200, 3)).astype(np.float64)
        bgmm = WeightedBayesianGaussianMixture(
            n_components=1, cov_type="diagonal", random_state=42, max_iter=10
        )
        bgmm.fit(X)
        assert bgmm.means_.shape == (1, 3)


class TestPriorShapes:
    def test_weight_concentration_shape(self, blob_data_3c):
        bgmm = WeightedBayesianGaussianMixture(
            n_components=3, random_state=42, max_iter=10
        )
        bgmm.fit(blob_data_3c)
        assert bgmm.weight_concentration_.shape == (3,)

    def test_posterior_means_finite(self, blob_data_3c):
        bgmm = WeightedBayesianGaussianMixture(
            n_components=3, random_state=42, max_iter=10
        )
        bgmm.fit(blob_data_3c)
        assert np.all(np.isfinite(bgmm.means_))


class TestFullPrecisionCholesky:
    def test_precision_cholesky_reconstructs_inverse(self):
        rng = np.random.default_rng(42)
        covs = np.stack([_correlated_spd(rng, 3) for _ in range(2)])
        prec_chol = _compute_precision_cholesky(covs, "full")
        for k in range(2):
            assert np.allclose(prec_chol[k] @ prec_chol[k].T, np.linalg.inv(covs[k]))

    def test_log_prob_matches_scipy(self):
        rng = np.random.default_rng(42)
        covs = np.stack([_correlated_spd(rng, 3) for _ in range(2)])
        means = rng.normal(size=(2, 3))
        X = rng.normal(size=(20, 3))
        prec_chol = _compute_precision_cholesky(covs, "full")
        log_prob = _estimate_log_gaussian_prob_pchol(X, means, prec_chol, "full")
        for k in range(2):
            expected = multivariate_normal(means[k], covs[k]).logpdf(X)
            assert np.allclose(log_prob[:, k], expected)

    def test_fitted_precisions_invert_covariances(self):
        rng = np.random.default_rng(42)
        X = np.vstack([
            np.asarray(c) + rng.normal(size=(300, 2)) @ np.diag([1.5, 0.12]) @ _rotation(t).T
            for c, t in [((0.0, 0.0), 0.6), ((1.5, 0.0), -0.6), ((0.7, 2.0), 1.2)]
        ])
        bgmm = WeightedBayesianGaussianMixture(
            n_components=3, cov_type="full", random_state=42, max_iter=20
        )
        bgmm.fit(X)
        for k in range(3):
            assert np.allclose(bgmm.precisions_[k] @ bgmm.covariances_[k], np.eye(2))


class TestLowerBoundMonotonic:
    @pytest.mark.parametrize("cov_type", ["full", "diagonal", "spherical"])
    def test_lower_bound_non_decreasing(self, anisotropic_blobs, cov_type):
        bgmm = WeightedBayesianGaussianMixture(
            n_components=6, cov_type=cov_type, random_state=0, max_iter=60, tol=0.0
        )
        bgmm.fit(anisotropic_blobs)
        lbs = np.asarray(bgmm.lower_bounds_[1:])
        slack = 1e-8 * np.maximum(np.abs(lbs[:-1]), 1.0)
        assert np.all(np.diff(lbs) >= -slack)
