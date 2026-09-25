import numpy as np
import pytest
from sklearn.cluster import KMeans as SklearnKMeans

from roadrunner.mixture.bayesian_gmm import WeightedBayesianGaussianMixture
from roadrunner.mixture.kmeans import WeightedKMeans
from roadrunner.mixture.weighted_gmm import WeightedGaussianMixture


@pytest.fixture
def separated_blobs():
    rng = np.random.default_rng(0)
    X = np.vstack([
        rng.normal([-10, -10], 0.3, (60, 2)),
        rng.normal([10, -10], 0.3, (60, 2)),
        rng.normal([0, 10], 0.3, (60, 2)),
    ]).astype(np.float64)
    centers = np.array([[-9.0, -9.0], [9.0, -9.0], [1.0, 9.0]])
    return X, centers


class TestSklearnEquivalence:
    """With a flat mask, WeightedKMeans must match sklearn.cluster.KMeans
    given the same fixed init and sample weights (labels_, cluster_centers_,
    inertia_, n_iter_)."""

    def test_matches_sklearn_uniform_weights(self, separated_blobs):
        X, centers = separated_blobs

        skl = SklearnKMeans(
            n_clusters=3, init=centers.copy(), n_init=1, algorithm="lloyd",
            max_iter=50, tol=1e-8,
        )
        skl.fit(X)

        ours = WeightedKMeans(n_clusters=3, init=centers.copy(), max_iter=50, tol=1e-8)
        ours.fit(X)

        assert np.array_equal(ours.labels_, skl.labels_)
        assert np.allclose(ours.cluster_centers_, skl.cluster_centers_)
        assert np.isclose(ours.inertia_, skl.inertia_)
        assert ours.n_iter_ == skl.n_iter_

    def test_matches_sklearn_with_point_weights(self, separated_blobs):
        X, centers = separated_blobs
        rng = np.random.default_rng(1)
        weights = rng.uniform(0.5, 2.0, size=X.shape[0])

        skl = SklearnKMeans(
            n_clusters=3, init=centers.copy(), n_init=1, algorithm="lloyd",
            max_iter=50, tol=1e-8,
        )
        skl.fit(X, sample_weight=weights)

        ours = WeightedKMeans(n_clusters=3, init=centers.copy(), max_iter=50, tol=1e-8)
        ours.fit(X, point_weights=weights)

        assert np.array_equal(ours.labels_, skl.labels_)
        assert np.allclose(ours.cluster_centers_, skl.cluster_centers_)
        assert np.isclose(ours.inertia_, skl.inertia_)
        assert ours.n_iter_ == skl.n_iter_


class TestMasking:
    def test_masked_points_never_assigned_forbidden_cluster(self):
        rng = np.random.default_rng(2)
        X = rng.normal(0.0, 1.0, size=(100, 2))
        cluster_weights = np.ones((100, 2))
        cluster_weights[:50, 1] = 0.0  # first 50 points cannot join cluster 1

        km = WeightedKMeans(n_clusters=2, init=np.array([[-1.0, 0.0], [1.0, 0.0]]), random_state=0)
        km.fit(X, cluster_weights=cluster_weights)

        assert np.all(km.labels_[:50] != 1)

    def test_raises_when_row_allows_no_cluster(self):
        X = np.random.default_rng(3).normal(size=(10, 2))
        cw = np.ones((10, 2))
        cw[3] = 0.0
        km = WeightedKMeans(n_clusters=2, init=np.array([[-1.0, 0.0], [1.0, 0.0]]))
        with pytest.raises(ValueError, match="allow"):
            km.fit(X, cluster_weights=cw)


class TestWeightedCentres:
    def test_centres_are_weighted_means(self):
        rng = np.random.default_rng(4)
        X = np.vstack([
            rng.normal([-5, 0], 0.2, (40, 2)),
            rng.normal([5, 0], 0.2, (40, 2)),
        ])
        w = rng.uniform(0.5, 3.0, size=X.shape[0])
        km = WeightedKMeans(n_clusters=2, init=np.array([[-4.0, 0.0], [4.0, 0.0]]), max_iter=50, tol=1e-10)
        km.fit(X, point_weights=w)

        assert km.n_iter_ < 50  # should have hit strict convergence, not max_iter

        for k in range(2):
            m = km.labels_ == k
            expected = np.average(X[m], axis=0, weights=w[m])
            assert np.allclose(km.cluster_centers_[k], expected)


class TestEmptyClusterRelocation:
    def test_relocation_respects_mask(self):
        rng = np.random.default_rng(5)
        X = rng.normal([0, 0], 0.1, (30, 2))
        X[0] = [3.0, 3.0]  # outlier disallowed for cluster 2: a mask-ignoring relocation would pick it
        # Cluster 2's init centre is far from all data, so it starts empty;
        # only the last 15 points are allowed to fill it.
        init_centers = np.array([[-0.1, 0.0], [0.1, 0.0], [100.0, 100.0]])
        cluster_weights = np.ones((30, 3))
        cluster_weights[:15, 2] = 0.0

        km = WeightedKMeans(n_clusters=3, init=init_centers, max_iter=1)
        km.fit(X, cluster_weights=cluster_weights)

        assert np.any(km.labels_ == 2)  # relocation actually happened
        filled = km.labels_ == 2
        assert np.all(cluster_weights[filled, 2] > 0)

        # The relocated centre must actually be one of the points allowed
        # into cluster 2 (this fails if relocation ignored the mask, since
        # the farthest-point choice would then land on the disallowed
        # outlier X[0] instead).
        k_empty = 2
        centre = km.cluster_centers_[k_empty]
        allowed_rows = cluster_weights[:, k_empty] > 0
        assert np.any(np.all(np.isclose(X[allowed_rows], centre), axis=1))

    def test_cluster_with_no_allowed_points_keeps_its_centre(self):
        rng = np.random.default_rng(5)
        X = rng.normal([0, 0], 0.1, (30, 2))
        init_centers = np.array([[-0.1, 0.0], [0.1, 0.0], [100.0, 100.0]])
        cluster_weights = np.ones((30, 3))
        cluster_weights[:, 2] = 0.0  # no point may ever join cluster 2

        km = WeightedKMeans(n_clusters=3, init=init_centers, max_iter=5)
        km.fit(X, cluster_weights=cluster_weights)

        assert np.array_equal(km.cluster_centers_[2], init_centers[2])
        assert not np.any(km.labels_ == 2)


class TestMaxIterZero:
    def test_no_iterations_run(self):
        rng = np.random.default_rng(7)
        X = rng.normal(0.0, 1.0, size=(20, 2))
        init_centers = np.array([[-1.0, 0.0], [1.0, 0.0]])
        cluster_weights = np.ones((20, 2))
        cluster_weights[:5, 1] = 0.0  # first 5 points cannot join cluster 1

        km = WeightedKMeans(n_clusters=2, init=init_centers.copy(), max_iter=0)
        km.fit(X, cluster_weights=cluster_weights)

        assert km.n_iter_ == 0
        assert np.array_equal(km.cluster_centers_, init_centers)

        # labels_ must be the masked nearest-centre labels of the initial
        # centres (no Lloyd step should have run).
        allowed = cluster_weights > 0
        d2 = np.full((20, 2), np.inf)
        for k in range(2):
            d2[allowed[:, k], k] = np.sum((X[allowed[:, k]] - init_centers[k]) ** 2, axis=1)
        expected_labels = np.argmin(d2, axis=1)
        assert np.array_equal(km.labels_, expected_labels)


class TestPredict:
    def test_predict_matches_labels_after_fit(self, separated_blobs):
        X, centers = separated_blobs
        km = WeightedKMeans(n_clusters=3, init=centers.copy(), max_iter=50, tol=1e-10)
        km.fit(X)
        pred = km.predict(X)
        assert np.array_equal(pred, km.labels_)


class TestKMeansPlusPlusInit:
    def test_reproducible_with_seed(self):
        rng = np.random.default_rng(6)
        X = np.vstack([
            rng.normal([-5, 0], 0.3, (30, 2)),
            rng.normal([5, 0], 0.3, (30, 2)),
            rng.normal([0, 5], 0.3, (30, 2)),
        ])
        km1 = WeightedKMeans(n_clusters=3, init="k-means++", random_state=42, max_iter=50)
        km1.fit(X)
        km2 = WeightedKMeans(n_clusters=3, init="k-means++", random_state=42, max_iter=50)
        km2.fit(X)

        assert np.array_equal(km1.labels_, km2.labels_)
        assert np.allclose(km1.cluster_centers_, km2.cluster_centers_)


class TestGMMIntegration:
    @pytest.mark.parametrize("init_params", ["kmeans", "kmeans++"])
    def test_incomplete_init_with_hard_mask_no_nan(self, init_params):
        rng = np.random.default_rng(8)
        X = np.vstack([
            rng.normal([-5, 0], 0.3, (40, 2)),
            rng.normal([5, 0], 0.3, (40, 2)),
            rng.normal([0, 5], 0.3, (40, 2)),
        ])
        n = X.shape[0]
        latent_prior = np.ones((n, 3))
        latent_prior[:40, 1:] = 0.0
        latent_prior[40:80, [0, 2]] = 0.0
        latent_prior[80:, :2] = 0.0

        gmm = WeightedGaussianMixture(n_components=3, init_params=init_params, random_state=0, max_iter=5)
        gmm.fit(X, latent_prior=latent_prior)
        assert np.all(np.isfinite(gmm.means_))
        assert np.all(np.isfinite(gmm.covariances_))

        bgmm = WeightedBayesianGaussianMixture(n_components=3, init_params=init_params, random_state=0, max_iter=5)
        bgmm.fit(X, latent_prior=latent_prior)
        assert np.all(np.isfinite(bgmm.means_))


class TestInputValidation:
    """Bad inputs must raise clear ValueErrors, not corrupt results silently
    (NaN in X) or fail with confusing errors from the numba kernel's
    unchecked out-of-bounds reads (wrong-shaped init/cluster_weights/point_weights)."""

    def test_fit_raises_on_nan_in_X(self):
        rng = np.random.default_rng(10)
        X = rng.normal(size=(10, 2))
        X[3, 0] = np.nan
        km = WeightedKMeans(n_clusters=2)
        with pytest.raises(ValueError, match="NaN or infinity"):
            km.fit(X)

    def test_fit_raises_on_inf_in_X(self):
        rng = np.random.default_rng(11)
        X = rng.normal(size=(10, 2))
        X[3, 0] = np.inf
        km = WeightedKMeans(n_clusters=2)
        with pytest.raises(ValueError, match="NaN or infinity"):
            km.fit(X)

    @pytest.mark.parametrize("point_weights,cluster_weights,match", [
        (np.array([1.0, np.nan, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), None, "finite"),
        (np.array([1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), None, "non-negative"),
        (np.zeros(10), None, "positive value"),
        (None, np.full((10, 2), np.inf), "finite"),
    ])
    def test_fit_raises_on_invalid_weights(self, point_weights, cluster_weights, match):
        rng = np.random.default_rng(15)
        X = rng.normal(size=(10, 2))
        km = WeightedKMeans(n_clusters=2)
        with pytest.raises(ValueError, match=match):
            km.fit(X, point_weights=point_weights, cluster_weights=cluster_weights)

    def test_int_input_preserves_float_precision(self):
        # True mean of [0, 1] is 0.5 -- an int-dtype X must not truncate it.
        X = np.array([[0], [1]], dtype=np.int64)
        km = WeightedKMeans(n_clusters=1, init=np.array([[0.0]]))
        km.fit(X)
        assert np.isclose(km.cluster_centers_[0, 0], 0.5)

        # predict() must not downcast the already-correct fitted centers to
        # match a later int-dtype query either. Centers 0.9/1.9 (kept exact
        # via max_iter=0) truncated to 0/1 would flip an integer query at 1
        # from cluster 0 (correct: |1-0.9|=0.1 < |1-1.9|=0.9) to cluster 1.
        km2 = WeightedKMeans(n_clusters=2, init=np.array([[0.9], [1.9]]), max_iter=0)
        km2.fit(np.array([[0.9], [1.9]]))
        assert np.array_equal(km2.cluster_centers_, [[0.9], [1.9]])
        labels = km2.predict(np.array([[1]], dtype=np.int64))
        assert labels[0] == 0

    def test_fit_raises_on_init_with_too_many_rows(self):
        rng = np.random.default_rng(12)
        X = rng.normal(size=(10, 2))
        init = np.zeros((4, 2))  # n_clusters=3, one row too many
        km = WeightedKMeans(n_clusters=3, init=init)
        with pytest.raises(ValueError, match="initial centers"):
            km.fit(X)

    def test_fit_raises_on_init_with_too_few_columns(self):
        rng = np.random.default_rng(13)
        X = rng.normal(size=(10, 2))
        init = np.zeros((3, 1))  # n_features=2, one column short
        km = WeightedKMeans(n_clusters=3, init=init)
        with pytest.raises(ValueError, match="initial centers"):
            km.fit(X)

    def test_fit_raises_on_cluster_weights_with_wrong_n_columns(self):
        rng = np.random.default_rng(14)
        X = rng.normal(size=(10, 2))
        cluster_weights = np.ones((10, 2))  # n_clusters=3, one column short
        km = WeightedKMeans(n_clusters=3, init=np.zeros((3, 2)))
        with pytest.raises(ValueError, match="cluster_weights must have shape"):
            km.fit(X, cluster_weights=cluster_weights)

    def test_predict_raises_on_cluster_weights_with_wrong_n_columns(self):
        rng = np.random.default_rng(15)
        X = rng.normal(size=(10, 2))
        km = WeightedKMeans(n_clusters=3, init=np.zeros((3, 2)), max_iter=1)
        km.fit(X)

        cluster_weights = np.ones((10, 2))  # n_clusters=3, one column short
        with pytest.raises(ValueError, match="cluster_weights must have shape"):
            km.predict(X, cluster_weights=cluster_weights)

    def test_fit_raises_on_wrong_length_point_weights(self):
        rng = np.random.default_rng(16)
        X = rng.normal(size=(10, 2))
        point_weights = np.ones(9)  # n_samples=10, one short
        km = WeightedKMeans(n_clusters=2)
        with pytest.raises(ValueError, match="point_weights must have shape"):
            km.fit(X, point_weights=point_weights)

    def test_predict_raises_on_wrong_n_features(self):
        rng = np.random.default_rng(17)
        X = rng.normal(size=(10, 2))
        km = WeightedKMeans(n_clusters=2, init=np.zeros((2, 2)), max_iter=1)
        km.fit(X)

        X_bad = rng.normal(size=(10, 3))
        with pytest.raises(ValueError, match="features"):
            km.predict(X_bad)

    def test_predict_raises_on_nan_in_X(self):
        rng = np.random.default_rng(18)
        X = rng.normal(size=(10, 2))
        km = WeightedKMeans(n_clusters=2, init=np.zeros((2, 2)), max_iter=1)
        km.fit(X)

        X_bad = X.copy()
        X_bad[0, 0] = np.nan
        with pytest.raises(ValueError, match="NaN or infinity"):
            km.predict(X_bad)
