import numpy as np
import pytest

from roadrunner.mixture.base import (
    BaseMixture,
    _check_parameter_values,
    _nonpositive_definite,
)


class _ConcreteMixture(BaseMixture):
    def _set_parameters(self): pass
    def _check_parameters(self, X): pass
    def _initialize_complete(self, X, resp, point_weights): pass
    def _is_incomplete_init(self): return True
    def _initialize_means(self, X, point_weights, alpha): pass
    def _estimate_log_weights(self): pass
    def _m_step(self, X, resp, point_weights): pass
    def _estimate_log_gaussian_prob(self, X): pass


class TestAbstractMethods:
    def test_cannot_instantiate_base(self):
        with pytest.raises(TypeError):
            BaseMixture()


class TestInitializeWeightsAndPrior:
    def test_none_inputs(self):
        bm = _ConcreteMixture(n_components=3)
        X = np.ones((10, 2), dtype=np.float32)
        pw, alpha, log_a = bm._initialize_weights_and_prior(X, None, None)
        assert pw.shape == (10,)
        assert alpha.shape == (10, 3)
        assert log_a.shape == (10, 3)
        assert np.allclose(pw, 1.0)
        assert np.allclose(alpha.sum(axis=1), 3.0)

    def test_custom_latent_prior(self):
        bm = _ConcreteMixture(n_components=3)
        X = np.ones((10, 2), dtype=np.float32)
        custom_prior = np.random.uniform(0.1, 1.0, (10, 3)).astype(np.float32)
        pw, alpha, log_a = bm._initialize_weights_and_prior(X, None, custom_prior)
        assert alpha.shape == (10, 3)
        assert np.allclose(alpha.sum(axis=1), 3.0)

    def test_custom_point_weights(self):
        bm = _ConcreteMixture(n_components=2)
        X = np.ones((5, 3), dtype=np.float32)
        pw, _, _ = bm._initialize_weights_and_prior(
            X, np.array([0.5, 1.0, 1.5, 2.0, 2.5]), None
        )
        assert np.allclose(pw, [0.5, 1.0, 1.5, 2.0, 2.5])

    def test_wrong_prior_shape_raises(self):
        bm = _ConcreteMixture(n_components=2)
        X = np.ones((5, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="latent_prior must have shape"):
            bm._initialize_weights_and_prior(X, None, np.ones((5, 3)))

    def test_negative_prior_raises(self):
        bm = _ConcreteMixture(n_components=3)
        X = np.ones((5, 3), dtype=np.float32)
        prior = np.ones((5, 3), dtype=np.float32)
        prior[0, 0] = -1.0  # row still sums to 1 > 0, isolating the negativity check
        with pytest.raises(ValueError, match="non-negative"):
            bm._initialize_weights_and_prior(X, None, prior)

    def test_latent_prior_not_mutated(self):
        bm = _ConcreteMixture(n_components=3)
        X = np.ones((10, 2), dtype=np.float32)
        prior = np.random.uniform(0.1, 1.0, (10, 3)).astype(X.dtype)
        prior_copy = prior.copy()
        bm._initialize_weights_and_prior(X, None, prior)
        assert np.array_equal(prior, prior_copy)


class TestNonpositiveDefinite:
    def test_spherical_all_positive(self):
        result = _nonpositive_definite(np.array([1.0, 2.0]), "spherical")
        assert not np.any(result)

    def test_spherical_some_nonpositive(self):
        result = _nonpositive_definite(np.array([1.0, -1.0]), "spherical")
        assert np.any(result)

    def test_diagonal_all_positive(self):
        cov = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = _nonpositive_definite(cov, "diagonal")
        assert not np.any(result)

    def test_diagonal_some_nonpositive(self):
        cov = np.array([[1.0, 2.0], [3.0, -4.0]])
        result = _nonpositive_definite(cov, "diagonal")
        assert np.any(result)


class TestCheckParameterValues:
    def test_valid_params(self):
        w = np.array([0.5, 0.5])
        m = np.array([[0.0], [1.0]])
        c = np.array([1.0, 1.0])
        result = _check_parameter_values(w, m, c, "spherical")
        assert result == (False, False, False)

    def test_nan_weights(self):
        w = np.array([0.5, np.nan])
        m = np.array([[0.0], [1.0]])
        c = np.array([1.0, 1.0])
        result = _check_parameter_values(w, m, c, "spherical")
        assert result[0]
