#############################################################################
#
# package:   roadrunner.mixture
# file:      weighted_gmm.py
# brief:     Standard (non-Bayesian) weighted Gaussian mixture with EM.
#
# Provides :class:`WeightedGaussianMixture` together with module-level
# helpers for covariance estimation, Gaussian log-probability, and
# parameter validation that are shared with the Bayesian and SVI
# mixture types.
#
# copyright: GPLv3
# author:    Asier Lambarri Martinez
# changes:   13 May 2026 - Created
#            17 Jun 2026 - Last edit
#
#############################################################################

"""Standard (non-Bayesian) Gaussian mixture with weighted responsibilities.

.. highlight:: python

The :class:`WeightedGaussianMixture` class fits a Gaussian mixture
via the EM algorithm, supporting per-particle ``point_weights`` and
a ``latent_prior`` responsibility mask.  The module-level covariances
estimation functions are also used by the Bayesian and SVI variants.
"""

import numpy as np
from scipy.linalg import solve_triangular

from ._kmeans_plusplus import kmeans_plusplus_prior
from ._math import row_squared_norms
from .base import BaseMixture




def _check_counts(counts, n_components, n_samples):
    """Validate that *counts* has the expected shape and non-negative values.

    Parameters
    ----------
    counts : ndarray or None
    n_components : int
    n_samples : int

    Returns
    -------
    counts : ndarray of shape (n_components,)
    """
    if counts is None:
        raise ValueError("counts should not be None")

    counts = np.asarray(counts)

    if counts.shape != (n_components,):
        raise ValueError(
            f"counts must have shape ({n_components},), got {counts.shape}"
        )

    if np.any(counts < 0):
        raise ValueError("counts must be non-negative")
        if not np.allclose(counts.sum(), n_samples, atol=1E-1):
            raise ValueError(f"counts must sum to {n_samples}, got {counts.sum()}")

    return counts


def _check_weights(weights, n_components):
    """Validate that *weights* has the expected shape and non-negative values.

    Parameters
    ----------
    weights : ndarray or None
    n_components : int

    Returns
    -------
    weights : ndarray of shape (n_components,)
    """
    if weights is None:
        raise ValueError("weights should not be None")

    weights = np.asarray(weights)

    if weights.shape != (n_components,):
        raise ValueError(
            f"weights must have shape ({n_components},), got {weights.shape}"
        )

    if np.any(weights < 0):
        raise ValueError("weights must be non-negative")

    return weights


def _check_means(means, n_components, n_features):
    """Validate that *means* has the expected shape.

    Parameters
    ----------
    means : ndarray or None
    n_components : int
    n_features : int

    Returns
    -------
    means : ndarray of shape (n_components, n_features)
    """
    if means is None:
        raise ValueError("means should not be None")

    means = np.asarray(means)

    if means.shape != (n_components, n_features):
        raise ValueError(
            f"means must have shape ({n_components}, {n_features}), got {means.shape}"
        )

    return means


def _check_covariances(covariances, cov_type, n_components, n_features):
    """Validate that *covariances* has the correct shape for *cov_type*.

    Parameters
    ----------
    covariances : ndarray or None
    cov_type : str
        ``"spherical"``, ``"diagonal"``, or ``"full"``.
    n_components : int
    n_features : int

    Returns
    -------
    covariances : ndarray
    """
    if covariances is None:
        raise ValueError("covariances should not be None")

    covariances = np.asarray(covariances)

    if cov_type == "spherical":
        expected_shape = (n_components,)
    elif cov_type == "diagonal":
        expected_shape = (n_components, n_features)
    elif cov_type == "full":
        expected_shape = (n_components, n_features, n_features)
    else:
        raise ValueError(f"Unknown cov_type {cov_type}")

    if covariances.shape != expected_shape:
        raise ValueError(
            f"{cov_type} covariances must have shape {expected_shape}, "
            f"got {covariances.shape}"
        )

    return covariances


def _check_parameter_shapes(weights, means, covariances,
                            cov_type, n_components, n_features):
    """Validate shapes of all parameters (weights, means, covariances).

    Parameters
    ----------
    weights : ndarray
    means : ndarray
    covariances : ndarray
    cov_type : str
    n_components : int
    n_features : int

    Returns
    -------
    weights, means, covariances : tuple of ndarray
    """
    weights = _check_weights(weights, n_components)
    means = _check_means(means, n_components, n_features)
    covariances = _check_covariances(
        covariances, cov_type, n_components, n_features
    )
    return weights, means, covariances





###########################################################################################
#                                                                                         #
#-------------------------------- COVARIANCE FUNCTIONS -----------------------------------#

def _estimate_covariances_diagonal(resp, X, nk, means, reg_covar):
    """Estimates the diagonal covariance of a cloud of points.

    Parameters(
    ----------
    resp : array-like of shape (n_samples,  n_components)
        Responsibilities of each data point.
    X : array-like of shape (n_samples, n_features)
        Input data array.
    nk :  array-like of shape (n_components,)
        Effective number of points.
    means : array-like of shape (n_components, n_features)
        Component mean.
    reg_covar : float
        Covariance regularization.

    Returns
    -------
    covariance : array, shape (n_components, n_features,)
        The covariance matrix of the current components, only the diagonal is returned.
    """
    avg_X2 = ( resp.T @ X**2 ) / nk[:, np.newaxis]
    avg_means2 = means**2
    return avg_X2 - avg_means2 + reg_covar

def _estimate_covariances_spherical(resp, X, nk, means, reg_covar):
    """Estimates the spherical covariance of a cloud of points.

    Parameters
    ----------
    resp : array-like of shape (n_samples,  n_components)
        Responsibilities of each data point.
    X : array-like of shape (n_samples, n_features)
        Input data array.
    nk :  array-like of shape (n_components,)
        Effective number of points.
    means : array-like of shape (n_components, n_features)
        Component mean.
    reg_covar : float
        Covariance regularization.

    Returns
    -------
    covariance : array-like of shape (n_components, )
        The covariance matrix of the current component, only sigma**2 is returned.
    """
    return _estimate_covariances_diagonal(resp, X, nk, means, reg_covar).mean(axis=1)

def _estimate_covariances_full(resp, X, nk, means, reg_covar):
    """Estimates the full covariance of a cloud of points over multiple components.

    Parameters
    ----------
    resp : array-like of shape (n_samples,  n_components)
        Responsibilities of each data point.
    X : array-like of shape (n_samples, n_features)
        Input data array.
    nk :  array-like of shape (n_components,)
        Effective number of points.
    means : array-like of shape (n_components, n_features)
        Component mean.
    reg_covar : float
        Covariance regularization.

    Returns
    -------
    covariance : array-like of shape (n_components, n_features, n_features)
        The covariance matrix of the current component, the full matrix is returned.
    """
    n_components, n_features = means.shape
    covariances = np.empty((n_components, n_features, n_features), dtype=X.dtype)
    for k in range(n_components):
        diff = X - means[k]
        covariances[k] = ( (resp[:, k] * diff.T) @ diff ) / nk[k]
        covariances[k] += reg_covar * np.eye(n_features, dtype=covariances.dtype)
    return covariances

def _estimate_gaussian_parameters(X, resp, point_weights, cov_type, reg_covar):
    """Estimate the Gaussian distribution parameters, changing covariance type
    adaptativelly depending on each gaussian's effective number of points.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data array.
    resp : array-like of shape (n_samples, n_components)
        Responsibilities for each data sample in X.
    cov_type : str
        Covariance type.
    point_weights : array-like of shape (n_samples,)
        Weights of individual points.
    reg_covar : float
        Regularization added to the diagonal of the covariance matrices.

    Returns
    -------
    nk : array-like of shape (n_components,)
        Numbers of effectivge data samples in the current components.
    means : array-like of shape (n_components, n_features)
        Centers of the current components.
    covariances : array-like of shape (n_components, n_features, n_features)
        Covariance matrices of the current components.
    """
    n_samples, _ = X.shape
    _, n_components = resp.shape

    resp_weighted = resp * point_weights[:, np.newaxis]
    nk = resp_weighted.sum(axis=0) + 10 * np.finfo(resp_weighted.dtype).eps
    means = (resp_weighted.T @ X) / nk[:, np.newaxis]

    covariances = {
        "full": _estimate_covariances_full,
        "diagonal": _estimate_covariances_diagonal,
        "diag": _estimate_covariances_diagonal,
        "spherical": _estimate_covariances_spherical,
    }[cov_type](resp_weighted, X, nk, means, reg_covar)

    return nk, means, np.asarray(covariances)

###########################################################################################
#                                                                                         #
#--------------------------------- LOG PROB FUNCTIONS ------------------------------------#
def _estimate_log_gaussian_prob(X, means, covs, cov_type):
    """log N(x | mean, cov) for all components in a Gaussian mixture.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data array.
    means : array-like of shape (n_components, n_features)
        Component means.
    covs : array-like of shape (n_components,) / (n_components, n_features) or (n_components, n_features, n_features)
        Covariance matrices per component. Shape dependent on covariance type.
    cov_type : str
        Covariance type.

    Returns
    -------
    log_prob : array-like of shape (n_samples, n_components)
        Log probability of each sample under each component.
    """
    n_components = means.shape[0]
    n_samples, n_features = X.shape

    const = n_features * np.log(2 * np.pi)

    if cov_type == "spherical":
        covariances = covs.reshape(-1)
        log_dets    = n_features * np.log(covariances)
        precisions  = 1.0 / covariances
        log_prob = (
            np.sum(means**2, axis=1) * precisions
            - 2.0 * (X @ means.T * precisions)
            + np.outer(row_squared_norms(X), precisions)  # SUBSTITUTE row_squared_norms for np.sum(X * X, axis=1)
        )

    if cov_type == "diagonal":
        covariances = covs.reshape(n_components, n_features)
        log_dets    = np.sum(
            np.log(covariances),
            axis=1
        )
        precisions = 1.0 / covariances
        log_prob = (
            np.sum(( (means**2) * precisions), axis=1)
            - 2.0 * (X @ (means * precisions).T)
            + ( X**2 @ precisions.T)
        )

    if cov_type == "full":
        log_prob = np.empty((n_samples, n_components), dtype=X.dtype)

        choleskys = np.linalg.cholesky(covs)
        log_dets = 2 * np.sum(
            np.log(np.diagonal(choleskys, axis1=1, axis2=2)),
            axis=1
        )
        for k in range(n_components):
            sol = solve_triangular(choleskys[k], (X - means[k]).T, lower=True, overwrite_b=True)
            log_prob[:, k] = row_squared_norms(sol.T)

    return -0.5 * (const + log_dets + log_prob)



class WeightedGaussianMixture(BaseMixture):
    """Standard (non-Bayesian) Gaussian mixture with weighted responsibilities.

    Fits a Gaussian mixture model via the Expectation-Maximisation (EM)
    algorithm, supporting per-particle point weights and a
    ``latent_prior`` that acts as a responsibility mask.  The model
    can be initialised from pre-computed sufficient statistics
    via ``counts_init``, ``means_init``, and ``covariance_init``.

    Parameters
    ----------
    n_components : int, default=2
        Number of mixture components.
    means_init : ndarray of shape (n_components, n_features), optional
        Initial means.
    covariance_init : ndarray, optional
        Initial covariances. Shape depends on ``cov_type``.
    counts_init : ndarray of shape (n_components,), optional
        Initial effective counts (sum of responsibilities).
    cov_type : str, default='full'
        Covariance type.
    init_params : str, default='kmeans'
        Initialisation method.
    max_iter : int, default=10
        Maximum EM iterations.
    tol : float, default=1e-3
        Convergence threshold (absolute change in lower bound).
    verbose : bool or int, default=False
        Verbosity flag.
    random_state : int or RandomState, optional
        Random state.
    reg_covar : float, default=1e-6
        Regularisation added to covariance diagonal.
    **kwargs
        Additional keyword arguments.
    """
    def __init__(self, n_components=2, means_init=None, covariance_init=None, counts_init=None, cov_type="full",
                 init_params='kmeans', max_iter=10, tol=1e-3, verbose=False, random_state=None,  reg_covar=1E-6,
                 **kwargs):

        super().__init__(
            n_components=n_components,
            init_params=init_params,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            random_state=random_state,
            reg_covar=reg_covar,
            **kwargs
        )

        self.means_init   = means_init
        self.covariance_init = covariance_init
        self.counts_init = counts_init 
        self.means_       = None         # (K, D)
        self.covariances_ = None         # (K, D, D)
        self.weights_     = None         # (K,)
        self.cov_type     = cov_type

    ####################################### Private API #######################################
    #                                                                                         #
    #---------------------------------- Initialization API -----------------------------------#
    def _set_parameters(self):
        """Set fitted parameters of the model.

        No-op for ``WeightedGaussianMixture`` — all parameters are
        stored directly on the instance after the M-step.
        """
        pass

    def _check_parameters(self, X):
        """Check the values and shapes of weights, covariances, and means.

        Validates user-provided ``counts_init``, ``means_init``, and
        ``covariance_init`` against the expected shapes. Raises
        descriptive errors on mismatch.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data (used only for shape inference).
        """
        n_samples, n_features = X.shape
        if self.cov_type not in ["spherical", "diagonal", "full"]:
            raise ValueError("provided covariance type is not valid.")

        if self.counts_init is not None:
            self.counts_init = _check_counts(
                self.counts_init,
                self.n_components,
                n_samples
            )
        if self.means_init is not None:
            self.means_init = _check_means(
                self.means_init,
                self.n_components,
                n_features
            )
        if self.covariance_init is not None:
            self.covariance_init = _check_covariances(
                self.covariance_init,
                self.cov_type,
                self.n_components,
                n_features
            )


    def _is_incomplete_init(self):
        """Check whether initialization is incomplete.

        Returns ``True`` if any of ``means_init``, ``counts_init``,
        or ``covariance_init`` is ``None``, requiring a fallback to
        k-means++ initialisation.

        Returns
        -------
        incomplete : bool
        """
        return (
            self.means_init is None
            or self.counts_init is None
            or self.covariance_init is None
        )

    def _initialize_means(self, X, point_weights, alpha):
        """Initializes the means of the clusters through kmeans++ algorithm and taking into account
        the provided prior's information and ordering. If means_init is provided, those are used directly.
        """
        n_samples, n_features = X.shape
        weighted_prior = alpha * point_weights[:, None]

        if self.means_init is None:
            means, _ = kmeans_plusplus_prior(X, self.n_components, cluster_weights=weighted_prior, random_state=self.random_state)
        else:
            means = self.means_init

        return np.asarray(means, dtype=X.dtype)

    def _initialize_complete(self, X, resp, point_weights):
        """Initialise Gaussian mixture parameters from a complete set of responsibilities.

        When ``resp`` is provided, estimates weights, means and covariances.
        User-provided ``counts_init``, ``means_init``, ``covariance_init``
        take precedence over estimated values.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        resp : ndarray of shape (n_samples, n_components) or None
            Responsibility matrix.
        point_weights : ndarray of shape (n_samples,)
            Per-point sample weights.
        """
        n_samples, _ = X.shape

        weights, means, covariances = None, None, None
        if resp is not None:
            weights, means, covariances = _estimate_gaussian_parameters(
                X, resp, point_weights, self.cov_type, self.reg_covar
            )
            weights /= weights.sum()

        self.weights_  = weights if self.counts_init is None else np.asarray(self.counts_init, dtype=X.dtype)
        self.weights_ /= self.weights_.sum()
        self.means_    = means if self.means_init is None else np.asarray(self.means_init, dtype=X.dtype)
        self.covariances_ = covariances if self.covariance_init is None else np.asarray(self.covariance_init, dtype=X.dtype)



    #                                                                                         #
    #---------------------------------- Model Fitting API ------------------------------------#
    def _estimate_log_weights(self):
        """Compute log-weights from the current ``weights_``.

        Returns
        -------
        log_weights : ndarray of shape (n_components,)
        """
        return np.log(self.weights_)

    def _m_step(self, X, resp, point_weights):
        """M-step: update weights, means, and covariances.

        Delegates to ``_estimate_gaussian_parameters`` for the actual
        computation, then normalises the weights.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        resp : ndarray of shape (n_samples, n_components)
        point_weights : ndarray of shape (n_samples,)
        """
        self.weights_, self.means_, self.covariances_ = _estimate_gaussian_parameters(
            X, resp, point_weights, self.cov_type, self.reg_covar
        )
        self.weights_ /= self.weights_.sum()

    def _estimate_log_gaussian_prob(self, X):
        """Compute log N(x | mean, cov) for all components.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        log_prob : ndarray of shape (n_samples, n_components)
        """
        return _estimate_log_gaussian_prob(X, self.means_, self.covariances_, self.cov_type)


    def _compute_lower_bound(self, log_resp, log_prob_norm, point_weights):
        """Compute the log-likelihood lower bound.

        Parameters
        ----------
        log_resp : ndarray of shape (n_samples, n_components)
            Log-responsibilities.
        log_prob_norm : ndarray of shape (n_samples,)
            Log-probability normalisation term.
        point_weights : ndarray of shape (n_samples,)

        Returns
        -------
        lower_bound : float
        """
        ll = np.average(log_prob_norm.ravel(), weights=point_weights)
        return ll
