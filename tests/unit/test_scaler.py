import numpy as np
import pytest

from roadrunner.physics.scaler import StandardScaler


class TestStandardScaler:
    def test_fit_zero_samples_raises_with_clear_message(self):
        X = np.empty((0, 6), dtype=np.float64)
        with pytest.raises(ValueError, match="zero samples"):
            StandardScaler().fit(X)

    def test_fit_transform_centered(self):
        rng = np.random.default_rng(42)
        X = rng.uniform(-10, 10, (1000, 4)).astype(np.float64)
        Xt = StandardScaler().fit_transform(X)
        assert np.allclose(Xt.mean(axis=0), 0.0, atol=1e-10)
        assert np.all(Xt.max(axis=0) <= 5.5)
        assert np.all(Xt.min(axis=0) >= -5.5)

    def test_single_row(self):
        X = np.array([[5.0, 10.0, -3.0]], dtype=np.float64)
        scaler = StandardScaler()
        Xt = scaler.fit_transform(X)
        Xr = scaler.inverse_transform(Xt)
        assert np.allclose(X, Xr)

    def test_constant_column(self):
        X = np.array([[1.0, 5.0], [1.0, 10.0]], dtype=np.float64)
        scaler = StandardScaler()
        Xt = scaler.fit_transform(X)
        assert scaler.scale_[0] == 10.0  # 10.0 / 1.0 (diff=0 → clamped to 1)
        assert np.allclose(Xt[:, 0], 0.0)

    def test_preserves_dtype_float32(self):
        X = np.random.default_rng(42).uniform(-5, 5, (50, 2)).astype(np.float32)
        scaler = StandardScaler()
        Xt = scaler.fit_transform(X)
        Xr = scaler.inverse_transform(Xt)
        assert Xt.dtype == np.float32
        assert Xr.dtype == np.float32
        assert np.allclose(X, Xr, atol=1e-5)

    def test_preserves_dtype_float64(self):
        X = np.random.default_rng(42).uniform(-5, 5, (50, 2)).astype(np.float64)
        scaler = StandardScaler()
        Xt = scaler.fit_transform(X)
        Xr = scaler.inverse_transform(Xt)
        assert Xt.dtype == np.float64
        assert Xr.dtype == np.float64
        assert np.allclose(X, Xr)
