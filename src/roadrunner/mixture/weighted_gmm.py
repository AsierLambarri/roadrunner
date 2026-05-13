
import numpy as np
from scipy.linalg import solve_triangular

from ._kmeans_plusplus import kmeans_plusplus_prior
from ._math import row_squared_norms
from .base import BaseMixture






def _check_weights(weights, n_components):
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
    if means is None:
        raise ValueError("means should not be None")

    means = np.asarray(means)

    if means.shape != (n_components, n_features):
        raise ValueError(
            f"means must have shape ({n_components}, {n_features}), got {means.shape}"
        )

    return means


def _check_covariances(covariances, cov_type, n_components, n_features):
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
        covariances[k] += reg_covar * np.eye(n_features)
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
    def __init__(self, n_components=2, means_init=None, covariance_init=None, weights_init=None, cov_type="full",
                 init_params='kmeans', max_iter=10, tol=1e-3, verbose=False, random_state=None,  reg_covar=1E-6, cast_dtype=np.float64,
                 **kwargs):

        super().__init__(
            n_components=n_components,
            init_params=init_params,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            random_state=random_state,
            reg_covar=reg_covar,
            cast_dtype=cast_dtype,
            **kwargs
        )

        self.means_init   = means_init
        self.covariance_init = covariance_init
        self.weights_init = weights_init if weights_init is None else np.maximum(weights_init, np.finfo(self.cast_dtype).eps)
        self.means_       = None         # (K, D)
        self.covariances_ = None         # (K, D, D)
        self.weights_     = None         # (K,)
        self.cov_type     = cov_type

    ####################################### Private API #######################################
    #                                                                                         #
    #---------------------------------- Initialization API -----------------------------------#
    def _set_parameters(self):
        """Set fitted parameters of the model
        """
        pass

    def _check_parameters(self, X):
        """Check the values and shapes of weights, covariances, and means.
        """
        _, n_features = X.shape
        if self.cov_type not in ["spherical", "diagonal", "full"]:
            raise ValueError("provided covariance type is not valid.")

        if self.weights_init is not None:
            self.weights_init = _check_weights(
                self.weights_init,
                self.n_components
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
        """Checks wether initialization is incomplete or not.
        """
        return (
            self.means_init is None
            or self.weights_init is None
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
        """Initialization of the Gaussian mixture parameters from complete set of
        means, covars and weights.
        """
        n_samples, _ = X.shape

        weights, means, covariances = None, None, None
        if resp is not None:
            weights, means, covariances = _estimate_gaussian_parameters(
                X, resp, point_weights, self.cov_type, self.reg_covar
            )
            weights /= weights.sum()

        self.weights_  = np.maximum(weights, np.finfo(X.dtype).eps) if self.weights_init is None else np.asarray(self.weights_init, dtype=X.dtype)
        self.weights_ /= self.weights_.sum()
        self.means_    = means if self.means_init is None else np.asarray(self.means_init, dtype=X.dtype)
        self.covariances_ = covariances if self.covariance_init is None else np.asarray(self.covariance_init, dtype=X.dtype)



    #                                                                                         #
    #---------------------------------- Model Fitting API ------------------------------------#
    def _estimate_log_weights(self):
        """M-step: update weights, means, and covariances."""
        return np.log(self.weights_)

    def _m_step(self, X, resp, point_weights):
        """M-step: update weights, means, and covariances."""
        self.weights_, self.means_, self.covariances_ = _estimate_gaussian_parameters(
            X, resp, point_weights, self.cov_type, self.reg_covar
        )
        self.weights_ /= self.weights_.sum()

    def _estimate_log_gaussian_prob(self, X):
        """Compute log N(x | mean, cov) for all components."""
        return _estimate_log_gaussian_prob(X, self.means_, self.covariances_, self.cov_type)


    def _compute_lower_bound(self, log_resp, log_prob_norm, point_weights):
        """Compute the log-likelihood lower bound."""
        ll = np.average(log_prob_norm.ravel(), weights=point_weights)
        return ll
