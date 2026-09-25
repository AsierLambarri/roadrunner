"""StandardScaler for 6-D phase-space coordinates.

Transforms positions and velocities to a fixed range for numerical
stability in the mixture models.
"""

import numpy as np

from roadrunner.physics.constants import SCALER_RANGE


class StandardScaler:
    """Standardise 6-D phase-space coordinates to a fixed range.

    Transforms each feature dimension independently so that the
    range of the data maps to the interval ``[0, SCALER_RANGE]``.
    The transformation is
    ``z = (x - mean) * scale`` where ``scale = RANGE / (max - min)``.

    Attributes
    ----------
    mean_ : ndarray of shape (n_features,)
        Per-feature mean of the training data.
    scale_ : ndarray of shape (n_features,)
        Per-feature scaling factor.
    """

    def __init__(self):
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, X: np.ndarray):
        """Compute the mean and scale from training data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        """
        if X.shape[0] == 0:
            raise ValueError(
                f"StandardScaler.fit received zero samples (X.shape={X.shape}); "
                "cannot compute a scale from an empty snapshot or group."
            )
        self.mean_ = np.mean(X, axis=0).astype(X.dtype, copy=False)
        diff = X.max(axis=0) - X.min(axis=0)
        diff[diff == 0] = 1.0
        self.scale_ = (SCALER_RANGE / diff).astype(X.dtype, copy=False)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the scaling transformation.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        Z : ndarray of shape (n_samples, n_features)
            Scaled data.
        """
        return ((X - self.mean_) * self.scale_).astype(X.dtype, copy=False)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit to data, then transform it.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        Z : ndarray of shape (n_samples, n_features)
            Scaled data.
        """
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Reverse the scaling transformation.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Scaled data.

        Returns
        -------
        X_orig : ndarray of shape (n_samples, n_features)
            Data in the original coordinate space.
        """
        inv_s = 1.0 / self.scale_
        return (X * inv_s + self.mean_).astype(X.dtype, copy=False)
