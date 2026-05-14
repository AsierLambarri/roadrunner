import numpy as np
from scipy.special import logsumexp as scipy_logsumexp

from roadrunner.mixture._math import logsumexp, row_l1_normalize, row_squared_norms


class TestLogSumExp:
    def test_matches_scipy(self):
        rng = np.random.default_rng(42)
        X = rng.uniform(-10, 0, (100, 5)).astype(np.float64)
        result = logsumexp(X)
        expected = scipy_logsumexp(X, axis=1, keepdims=True)
        assert np.allclose(result, expected)

    def test_extreme_values(self):
        X = np.array([[-1e3, -1e2, -5e1], [-2e2, -1e4, -3e3]], dtype=np.float64)
        result = logsumexp(X)
        expected = scipy_logsumexp(X, axis=1, keepdims=True)
        assert np.allclose(result, expected)

    def test_single_component(self):
        X = np.array([[-1.0], [-2.0], [-3.0]], dtype=np.float64)
        result = logsumexp(X)
        assert np.allclose(result, X.reshape(-1, 1))


class TestRowL1Normalize:
    def test_rows_sum_to_one(self):
        X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        result = row_l1_normalize(X.copy())
        assert np.allclose(result.sum(axis=1), 1.0)

    def test_zero_row_unchanged(self):
        X = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 0.0]], dtype=np.float32)
        result = row_l1_normalize(X.copy())
        assert result[0, 0] == 0.0 and result[0, 1] == 0.0 and result[0, 2] == 0.0
        assert np.isclose(result[1, 0], 1.0 / 3.0)

    def test_matches_python(self):
        X = np.random.default_rng(42).uniform(0, 10, (50, 5)).astype(np.float32)
        numba_result = row_l1_normalize(X.copy())
        py_result = X / X.sum(axis=1, keepdims=True)
        assert np.allclose(numba_result, py_result)

    def test_single_row(self):
        X = np.array([[3.0, 4.0]], dtype=np.float32)
        result = row_l1_normalize(X.copy())
        assert np.isclose(result[0, 0], 3.0 / 7.0)
        assert np.isclose(result[0, 1], 4.0 / 7.0)


class TestRowSquaredNorms:
    def test_matches_numpy(self):
        rng = np.random.default_rng(42)
        X = rng.uniform(-5, 5, (50, 3)).astype(np.float64)
        result = row_squared_norms(X)
        expected = np.sum(X**2, axis=1)
        assert np.allclose(result, expected)

    def test_float32(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        result = row_squared_norms(X)
        assert result.dtype == np.float32
        assert np.allclose(result, np.sum(X**2, axis=1))

    def test_single_row(self):
        X = np.array([[3.0, 4.0]], dtype=np.float64)
        result = row_squared_norms(X)
        assert np.isclose(result[0], 25.0)

    def test_zeros(self):
        X = np.zeros((2, 3), dtype=np.float64)
        result = row_squared_norms(X)
        assert np.all(result == 0.0)
