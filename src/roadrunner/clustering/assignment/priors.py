import numpy as np

from roadrunner._defaults import (
    PRIOR_COVARIANCE_SCALE,
    PRIOR_WEIGHT_DIVISOR,
    PRIOR_MEAN_DIVISOR,
)


def degrade_covariance(cov, n_features):
    """Reduce stored covariance to 1D per-dimension variances."""
    if cov.ndim == 0:
        return np.full(n_features, cov)
    if cov.ndim == 1:
        return cov
    if cov.ndim == 2:
        return np.diag(cov)


def covariance_prior(diag_vars, nk_n1, n_bound, inv_s, dof, cov_type):
    """Scale per-dimension variances to BGMM-expected shape."""
    scale = dof * n_bound / max(nk_n1, 1.0) * PRIOR_COVARIANCE_SCALE
    scaled = diag_vars * inv_s**2 * scale

    if cov_type == "spherical":
        return float(scaled.mean())
    if "diag" in cov_type:
        return scaled
    return np.diag(scaled)


def weight_concentration_prior(nk_n1, n_bound, n_comp):
    return min(nk_n1 / PRIOR_WEIGHT_DIVISOR, n_bound / PRIOR_WEIGHT_DIVISOR)


def mean_precision_prior(nk_n1, n_bound):
    return min(nk_n1 / PRIOR_MEAN_DIVISOR, n_bound / PRIOR_MEAN_DIVISOR)
