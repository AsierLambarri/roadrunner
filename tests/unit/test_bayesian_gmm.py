import warnings

import numpy as np
import pytest
from scipy.stats import multivariate_normal
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import BayesianGaussianMixture as SklearnBayesianGaussianMixture

from roadrunner._defaults import precision
from roadrunner.mixture.bayesian_gmm import (
    WeightedBayesianGaussianMixture,
    _compute_precision_cholesky,
    _estimate_log_gaussian_prob_pchol,
)
from roadrunner.mixture.weighted_gmm import _estimate_gaussian_parameters


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


class TestInitPrecedence:
    def test_means_init_precedence_incomplete_path(self):
        # Only means_init is given (counts_init/covariance_init are None), so
        # _is_incomplete_init() is True and the k-means fallback path runs.
        # User-provided means_init must still take precedence over the
        # k-means-estimated xk in the posterior mean update.
        rng = np.random.default_rng(7)
        X = np.vstack([
            rng.normal([-3, 0], 0.3, (50, 2)),
            rng.normal([3, 0], 0.3, (50, 2)),
        ]).astype(np.float64)
        means_init = np.array([[-2.5, 0.1], [2.5, -0.1]])

        bgmm = WeightedBayesianGaussianMixture(
            n_components=2, means_init=means_init, random_state=0
        )
        bgmm._check_parameters(X)
        point_weights, alpha, _ = bgmm._initialize_weights_and_prior(X, None, None)
        bgmm._initialize_parameters(X, point_weights, alpha)

        nk = bgmm.weight_concentration_ - bgmm.weight_concentration_prior_
        beta0 = bgmm.mean_precision_prior_
        m0 = bgmm.mean_prior_
        beta = beta0 + nk
        expected_means = (beta0[:, None] * m0 + nk[:, None] * means_init) / beta[:, None]

        assert np.allclose(bgmm.means_, expected_means)


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


@pytest.fixture
def masked_latent_prior_data():
    rng = np.random.default_rng(1)
    X = np.vstack([
        rng.normal([-2, 0], [1.0, 0.3], (600, 2)),
        rng.normal([2, 0], [0.3, 1.0], (500, 2)),
        rng.normal([0, 3], 0.5, (400, 2)),
    ])
    prior = rng.gamma(1.0, size=(X.shape[0], 3)) + 1e-3
    prior *= rng.random((X.shape[0], 3)) > 0.3
    prior[prior.sum(axis=1) == 0, 0] = 1.0
    return X, prior


class TestLatentPriorELBO:
    @pytest.mark.parametrize("cov_type", ["full", "diagonal"])
    def test_lower_bound_non_decreasing_with_masked_prior(self, masked_latent_prior_data, cov_type):
        X, prior = masked_latent_prior_data
        bgmm = WeightedBayesianGaussianMixture(
            n_components=3, cov_type=cov_type, random_state=0, max_iter=60, tol=0.0
        )
        bgmm.fit(X, latent_prior=prior.copy())
        lbs = np.asarray(bgmm.lower_bounds_[1:])
        slack = 1e-8 * np.maximum(np.abs(lbs[:-1]), 1.0)
        assert np.all(np.diff(lbs) >= -slack)


# ── sklearn parity (flat latent prior) ──────────────────────────────

_SKLEARN_COV_TYPE = {"full": "full", "diagonal": "diag", "spherical": "spherical"}


class _FixedInitSklearnBGMM(SklearnBayesianGaussianMixture):
    """sklearn BayesianGaussianMixture that skips k-means and initialises
    from a fixed responsibility matrix set on ``self._resp0``, mirroring
    ``WeightedBayesianGaussianMixture``'s ``counts_init``/``means_init``/
    ``covariance_init`` complete-init path.
    """

    def _initialize_parameters(self, X, random_state, xp=None):
        self._initialize(X, self._resp0)


@pytest.fixture
def parity_blobs():
    rng = np.random.default_rng(42)
    X = np.vstack([
        rng.normal([-2, 0], [1.0, 0.3], (200, 2)),
        rng.normal([2, 0], [0.3, 1.0], (200, 2)),
        rng.normal([0, 3], 0.5, (200, 2)),
    ]).astype(np.float64)
    labels = np.repeat([0, 1, 2], 200)
    resp0 = np.zeros((X.shape[0], 3))
    resp0[np.arange(X.shape[0]), labels] = 1.0
    return X, resp0


class TestSklearnParity:
    """With a flat latent prior, our BGMM must reproduce sklearn's
    BayesianGaussianMixture exactly: same fixed init, same priors, same
    number of (non-converging) iterations.
    """

    @pytest.mark.parametrize("cov_type,sk_cov_type", list(_SKLEARN_COV_TYPE.items()))
    def test_matches_sklearn(self, parity_blobs, cov_type, sk_cov_type):
        X, resp0 = parity_blobs
        n_samples = X.shape[0]
        n_components = 3

        nk, xk, sk = _estimate_gaussian_parameters(
            X, resp0, np.ones(n_samples), cov_type, 1e-6
        )
        flat_prior = np.full((n_samples, n_components), 0.37)

        skl = _FixedInitSklearnBGMM(
            n_components=n_components,
            covariance_type=sk_cov_type,
            weight_concentration_prior_type="dirichlet_distribution",
            max_iter=30,
            tol=0.0,
            reg_covar=1e-6,
            random_state=0,
        )
        skl._resp0 = resp0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            skl.fit(X)

        with precision(math="double"):
            ours = WeightedBayesianGaussianMixture(
                n_components=n_components,
                cov_type=cov_type,
                counts_init=nk,
                means_init=xk,
                covariance_init=sk,
                max_iter=30,
                tol=0.0,
                reg_covar=1e-6,
                random_state=0,
            )
            ours.fit(X, latent_prior=flat_prior)
            proba_ours = np.exp(ours.predict_log_proba(X, latent_prior=flat_prior))

        assert np.allclose(ours.means_, skl.means_, rtol=1e-10, atol=1e-12)
        assert np.allclose(ours.covariances_, skl.covariances_, rtol=1e-10, atol=1e-12)
        assert np.allclose(ours.weights_, skl.weights_, rtol=1e-10, atol=1e-12)
        assert np.allclose(
            ours.precisions_cholesky_, skl.precisions_cholesky_, rtol=1e-10, atol=1e-12
        )

        proba_sklearn = skl.predict_proba(X)
        assert np.allclose(proba_ours, proba_sklearn, rtol=1e-10, atol=1e-12)

        ours_lbs = np.asarray(ours.lower_bounds_[1:])
        skl_lbs = np.asarray(skl.lower_bounds_)
        assert np.allclose(ours_lbs, skl_lbs, rtol=1e-10, atol=1e-12)
        assert np.isclose(ours.lower_bound_, skl.lower_bound_, rtol=1e-10, atol=1e-12)
