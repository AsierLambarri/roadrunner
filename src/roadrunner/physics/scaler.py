#############################################################################
#
# package:   roadrunner.physics
# file:      scaler.py
# brief:     <TODO>
#
# copyright: GPLv3
# author:    Asier Lambarri Martinez
# changes:   15 may 2026 - Created
#            15 may 2026 - Last edit
#
#############################################################################

import numpy as np

from roadrunner.physics.constants import SCALER_RANGE


class StandardScaler:
    def __init__(self):
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, X: np.ndarray):
        self.mean_ = np.mean(X, axis=0).astype(np.float64, copy=False)
        diff = X.max(axis=0) - X.min(axis=0)
        diff[diff == 0] = 1.0
        self.scale_ = (SCALER_RANGE / diff).astype(np.float64, copy=False)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean_) * self.scale_).astype(X.dtype, copy=False)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        inv_s = 1.0 / self.scale_
        return (X * inv_s + self.mean_).astype(X.dtype, copy=False)
