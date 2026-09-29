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
    PRIOR_COVARIANCE_SCALE_LARGE,
    PRIOR_COVARIANCE_SCALE_SMALL,
    PRIOR_GROWTH_MAX,
    PRIOR_GROWTH_MIN,
    PRIOR_SCALE_N_LARGE,
    PRIOR_SCALE_N_SMALL,
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


def covariance_scale(n):
    """Covariance-prior scale ``f(n)`` for a halo with ``n`` particles.

    Linear in ``n`` from ``PRIOR_COVARIANCE_SCALE_SMALL`` at
    ``PRIOR_SCALE_N_SMALL`` to ``PRIOR_COVARIANCE_SCALE_LARGE`` at
    ``PRIOR_SCALE_N_LARGE``, and flat outside that range. The prior's
    ``nu0`` pseudo-points are centred on ``f(n)`` times the reference
    variance, so a small scale shrinks small halos, whose own particles
    barely outweigh the pseudo-points.

    Parameters
    ----------
    n : float
        The halo's (expected) particle count, in the fit's weight units.

    Returns
    -------
    scale : float
    """
    t = (n - PRIOR_SCALE_N_SMALL) / (PRIOR_SCALE_N_LARGE - PRIOR_SCALE_N_SMALL)
    t = min(max(t, 0.0), 1.0)
    return (PRIOR_COVARIANCE_SCALE_SMALL
            + (PRIOR_COVARIANCE_SCALE_LARGE - PRIOR_COVARIANCE_SCALE_SMALL) * t)


def covariance_prior_count_ratio(diag_vars, nk_n1, n_current, s, dof, cov_type):
    """Former count-ratio covariance prior, kept for reference, unused.

    Scale per-dimension variances to the BGMM-expected covariance shape.

    The scaling factor is ``dof * n_current / nk_n1 * covariance_scale(n_current)``:
    the halo's growth since the reference times ``f(n)``. The result is
    reshaped according to ``cov_type``.

    Parameters
    ----------
    diag_vars : ndarray of shape (n_features,)
        Reference per-dimension variances in natural coordinates.
    nk_n1 : float
        Reference effective count (previous snapshot's fitted count).
    n_current : float
        The halo's expected count this snapshot: its pre-fit
        responsibilities summed, so particles shared with a neighbour
        count only fractionally.
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
    scale = dof * n_current / max(nk_n1, 1.0) * covariance_scale(n_current)
    scaled = diag_vars * s**2 * scale

    if cov_type == "spherical":
        return float(scaled.mean())
    if "diag" in cov_type:
        return scaled
    return np.diag(scaled)


def covariance_growth(mass_ratio):
    """Covariance growth factor from the tree-mass ratio.

    ``mass_ratio ** (2/3)`` (virial scaling at fixed mean density), clipped
    to ``[PRIOR_GROWTH_MIN, PRIOR_GROWTH_MAX]``: the stellar cloud is
    detached from the dark mass, so it cannot grow or shrink as fast. A
    clipped step moves the covariance less than the real one would, so
    the telescoping bound is kept.

    Parameters
    ----------
    mass_ratio : float
        Current tree mass over the tree mass at the reference snapshot.

    Returns
    -------
    growth : float
    """
    return float(np.clip(mass_ratio ** (2.0 / 3.0), PRIOR_GROWTH_MIN, PRIOR_GROWTH_MAX))


def covariance_prior(diag_vars, mass_ratio, n_current, s, dof, cov_type):
    """Scale per-dimension variances to the BGMM-expected covariance shape.

    The scaling factor is ``dof * covariance_growth(mass_ratio) * covariance_scale(n_current)``.
    ``mass_ratio`` is the halo's current tree mass over the tree mass at the
    reference snapshot; the ``2/3`` exponent is virial scaling at fixed mean
    density (``R**2`` and ``sigma**2`` both scale as ``M**(2/3)``). Because
    the ratio is of the same quantity (tree mass) at two snapshots, it
    telescopes: a prior-only halo (never refit) can only see its covariance
    change as much as its tree mass did, unlike the count ratio it replaces,
    which compounded for starved halos. The result is reshaped according to
    ``cov_type``.

    Parameters
    ----------
    diag_vars : ndarray of shape (n_features,)
        Reference per-dimension variances in natural coordinates.
    mass_ratio : float
        Current tree mass over the tree mass at the reference snapshot.
    n_current : float
        The halo's expected count this snapshot: its pre-fit
        responsibilities summed, so particles shared with a neighbour
        count only fractionally.
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
    scale = dof * covariance_growth(mass_ratio) * covariance_scale(n_current)
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
