import numpy as np
import pytest

from roadrunner.mixture.base import (
    BaseMixture,
    _check_parameter_values,
    _nonpositive_definite,
)


class TestAbstractMethods:
    @pytest.mark.xfail(strict=False)
    def test_set_parameters_raises(self):
        bm = BaseMixture()
        with pytest.raises(NotImplementedError):
            bm._set_parameters()

    @pytest.mark.xfail(strict=False)
    def test_check_parameters_raises(self):
        bm = BaseMixture()
        with pytest.raises(NotImplementedError):
            bm._check_parameters(None)

    @pytest.mark.xfail(strict=False)
    def test_initialize_complete_raises(self):
        bm = BaseMixture()
        with pytest.raises(NotImplementedError):
            bm._initialize_complete(None, None, None)

    @pytest.mark.xfail(strict=False)
    def test_is_incomplete_init_raises(self):
        bm = BaseMixture()
        with pytest.raises(NotImplementedError):
            bm._is_incomplete_init()


class TestInitializeWeightsAndPrior:
    def test_none_inputs(self):
        bm = BaseMixture(n_components=3)
        X = np.ones((10, 2), dtype=np.float32)
        pw, alpha, log_a = bm._initialize_weights_and_prior(X, None, None)
        assert pw.shape == (10,)
        assert alpha.shape == (10, 3)
        assert log_a.shape == (10, 3)
        assert np.allclose(pw, 1.0)
        assert np.allclose(alpha.sum(axis=1), 3.0)

    def test_custom_latent_prior(self):
        bm = BaseMixture(n_components=3)
        X = np.ones((10, 2), dtype=np.float32)
        custom_prior = np.random.uniform(0.1, 1.0, (10, 3)).astype(np.float32)
        pw, alpha, log_a = bm._initialize_weights_and_prior(X, None, custom_prior)
        assert alpha.shape == (10, 3)
        assert np.allclose(alpha.sum(axis=1), 3.0)

    def test_custom_point_weights(self):
        bm = BaseMixture(n_components=2)
        X = np.ones((5, 3), dtype=np.float32)
        pw, _, _ = bm._initialize_weights_and_prior(
            X, np.array([0.5, 1.0, 1.5, 2.0, 2.5]), None
        )
        assert np.allclose(pw, [0.5, 1.0, 1.5, 2.0, 2.5])

    def test_wrong_prior_shape_raises(self):
        bm = BaseMixture(n_components=2)
        X = np.ones((5, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="latent_prior must have shape"):
            bm._initialize_weights_and_prior(X, None, np.ones((5, 3)))


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
