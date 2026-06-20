#############################################################################
#
# package:   roadrunner.clustering.assignment
# file:      priors.py
# brief:     Prior-parameter helpers for Bayesian Gaussian mixture assigners.
#
# copyright: GPLv3
# author:    Asier Lambarri Martinez
# changes:   17 Jun 2026 - Created
#            18 Jun 2026 - Last edit
#
#############################################################################

"""Prior parameter computation for Bayesian Gaussian mixtures.

Provides pure helper functions (no assigner state) for computing
per-component prior kwargs from previous snapshot parameters.
"""

import numpy as np

from roadrunner._defaults import (
    PRIOR_COVARIANCE_SCALE,
    PRIOR_WEIGHT_DIVISOR,
    PRIOR_MEAN_DIVISOR,
)


def degrade_covariance(cov, n_features):
    """Reduce a stored covariance to 1D per-dimension variances.

    Parameters
    ----------
    cov : ndarray
        Stored covariance (scalar for spherical, 1-D for diagonal,
        2-D for full).
    n_features : int
        Dimensionality.

    Returns
    -------
    diag_vars : ndarray of shape (n_features,)
        Per-dimension variances.
    """
    if cov.ndim == 0:
        return np.full(n_features, cov)
    if cov.ndim == 1:
        return cov
    if cov.ndim == 2:
        return np.diag(cov)


def covariance_prior(diag_vars, nk_n1, n_bound, s, dof, cov_type):
    """Scale per-dimension variances to the BGMM-expected covariance shape.

    The scaling factor is ``dof * n_bound / nk_n1 * PRIOR_COVARIANCE_SCALE``,
    and the result is reshaped according to ``cov_type``.

    Parameters
    ----------
    diag_vars : ndarray of shape (n_features,)
        Per-dimension variances in natural coordinates.
    nk_n1 : float
        Previous snapshot effective count.
    n_bound : float
        Current snapshot bound count.
    s : ndarray of shape (n_features,)
        StandardScaler ``scale_`` values.
    dof : float
        Degrees of freedom (``n_features + PRIOR_DOF_OFFSET``).
    cov_type : str
        Covariance type.

    Returns
    -------
    cov_prior : float or ndarray
        Covariance prior in the shape expected by the BGMM constructor.
    """
    scale = dof * n_bound / max(nk_n1, 1.0) * PRIOR_COVARIANCE_SCALE
    scaled = diag_vars * s**2 * scale

    if cov_type == "spherical":
        return float(scaled.mean())
    if "diag" in cov_type:
        return scaled
    return np.diag(scaled)


def weight_concentration_prior(nk_n1, n_bound, n_comp):
    """Compute the weight-concentration Dirichlet prior.

    ``wp = min(nk_n1 / divisor, n_bound / divisor)``

    Parameters
    ----------
    nk_n1 : float
        Previous snapshot effective count.
    n_bound : float
        Current snapshot bound count.
    n_comp : int
        Number of components (unused, kept for signature consistency).

    Returns
    -------
    wp : float
        Weight concentration prior value.
    """
    return min(nk_n1 / PRIOR_WEIGHT_DIVISOR, n_bound / PRIOR_WEIGHT_DIVISOR)


def mean_precision_prior(nk_n1, n_bound):
    """Compute the mean-precision prior.

    ``pp = min(nk_n1 / divisor, n_bound / divisor)``

    Parameters
    ----------
    nk_n1 : float
        Previous snapshot effective count.
    n_bound : float
        Current snapshot bound count.

    Returns
    -------
    pp : float
        Mean precision prior value.
    """
    return min(nk_n1 / PRIOR_MEAN_DIVISOR, n_bound / PRIOR_MEAN_DIVISOR)
