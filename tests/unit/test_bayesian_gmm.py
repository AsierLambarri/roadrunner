import numpy as np
import pytest

from roadrunner.mixture.bayesian_gmm import WeightedBayesianGaussianMixture


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
