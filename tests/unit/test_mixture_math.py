import numpy as np
from scipy.special import logsumexp as scipy_logsumexp

from roadrunner.mixture._math import (
    centered_squared_sums,
    logsumexp,
    row_l1_normalize,
    row_squared_norms,
)


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

    def test_all_neg_inf_row_gives_neg_inf(self):
        X = np.array([[-np.inf, -np.inf, -np.inf], [-1.0, -2.0, -3.0]], dtype=np.float64)
        result = logsumexp(X)
        assert result[0, 0] == -np.inf
        assert np.isfinite(result[1, 0])

    def test_row_with_pos_inf_gives_pos_inf(self):
        X = np.array([[1.0, np.inf, 2.0], [-1.0, -2.0, -3.0]], dtype=np.float64)
        result = logsumexp(X)
        assert result[0, 0] == np.inf
        assert np.isfinite(result[1, 0])


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


def _centered_squared_sums_reference(X, resp, means):
    X, resp, means = (np.asarray(a, dtype=np.float64) for a in (X, resp, means))
    return np.stack([resp[:, k] @ (X - means[k]) ** 2 for k in range(means.shape[0])])


class TestCenteredSquaredSums:
    def test_matches_numpy_float64(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(1000, 4))
        resp = rng.random((1000, 3))
        means = rng.normal(size=(3, 4))
        result = centered_squared_sums(X, resp, means)
        assert result.dtype == np.float64
        assert np.allclose(result, _centered_squared_sums_reference(X, resp, means), rtol=1e-10)

    def test_float32_keeps_dtype(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(1000, 4)).astype(np.float32)
        resp = rng.random((1000, 3)).astype(np.float32)
        means = rng.normal(size=(3, 4)).astype(np.float32)
        result = centered_squared_sums(X, resp, means)
        assert result.dtype == np.float32
        assert np.allclose(result, _centered_squared_sums_reference(X, resp, means), rtol=1e-5)

    def test_single_component(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(500, 2))
        resp = rng.random((500, 1))
        means = rng.normal(size=(1, 2))
        assert np.allclose(centered_squared_sums(X, resp, means),
                           _centered_squared_sums_reference(X, resp, means), rtol=1e-10)

    def test_fewer_samples_than_threads(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        resp = np.array([[1.0], [0.5], [0.0]])
        means = np.array([[2.0, 3.0]])
        assert np.allclose(centered_squared_sums(X, resp, means), [[1.5, 1.5]])

    def test_no_cancellation_float32_offset_component(self):
        rng = np.random.default_rng(42)
        X = (5.0 + 1e-3 * rng.normal(size=(20_000, 3))).astype(np.float32)
        resp = np.ones((X.shape[0], 1), dtype=np.float32)
        means = (resp.T @ X) / resp.sum(axis=0)[:, None]
        var = centered_squared_sums(X, resp, means)[0] / X.shape[0]
        assert np.allclose(var, X.astype(np.float64).var(axis=0), rtol=1e-3)
