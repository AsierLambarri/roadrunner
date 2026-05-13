import abc
from time import time

import numpy as np
from sklearn.cluster import KMeans

from ._math import logsumexp


def _nonpositive_definite(covariances, cov_type):
    """Check if any covariance matrix is non-positive definite."""
    if cov_type == "spherical":
        return covariances <= 0
    elif cov_type == "diagonal":
        return np.any(covariances <= 0, axis=1)
    elif cov_type == "full":
        symm_cov = 0.5 * (covariances + np.transpose(covariances, (0, 2, 1)))
        eigvals = np.linalg.eigvalsh(symm_cov)
        return np.any(eigvals <= 0, axis=1)

def _check_parameter_values(weights, means, covariances, cov_type):
    """Check that GMM parameters do not have NaN nor Inf values."""
    return (
        np.any(np.isnan(weights)) or np.any(np.isinf(weights)) or np.any(weights <= 0),
        np.any(np.isnan(means)) or np.any(np.isinf(means)),
        np.any(np.isnan(covariances)) or np.any(np.isinf(covariances)) or np.any(_nonpositive_definite(covariances, cov_type))
    )





class BaseMixture:
    def __init__(self, n_components=2, init_params='kmeans', max_iter=10, tol=1e-3, verbose=0,
                 random_state=None, reg_covar=1E-6, cast_dtype=np.float32, **kwargs):

        self.cast_dtype  = cast_dtype
        self.reg_covar   = self.cast_dtype(reg_covar)

        self.n_components = n_components

        self.init_params  = init_params
        self.max_iter     = max_iter
        self.tol          = tol
        self.random_state = random_state if isinstance(random_state, np.random.RandomState) else np.random.RandomState(random_state)
        self.converged_   = False
        self.n_iter_      = 0
        self.lower_bound_ = -np.inf
        self.verbose      = verbose


    ####################################### Private API #######################################
    #                                                                                         #
    #---------------------------------- Initialization API -----------------------------------#
    @abc.abstractmethod
    def _set_parameters(self):
        """Set fitted parameters of the model
        """
        pass

    @abc.abstractmethod
    def _check_parameters(self, X):
        """Check the values and shapes of parameter models.
        """
        pass

    @abc.abstractmethod
    def _initialize_complete(self, X, resp, point_weights):
        """Initialization of the Gaussian mixture parameters from complete set of
        means, covars and weights.
        """
        pass

    @abc.abstractmethod
    def _is_incomplete_init(self):
        """Checks wether initialization is incomplete or not.
        """
        pass

    def _initialize_weights_and_prior(self, X, point_weights, latent_prior):
        """Initializes latent variable prior. If latent_prior is None, a flat
        prior is assumed (standard GMM).
        """
        n_samples, n_features = X.shape

        if latent_prior is None:
            latent_prior = np.ones((n_samples, self.n_components), dtype=X.dtype)
        else:
            latent_prior = np.asarray(latent_prior, dtype=X.dtype)
        if latent_prior.shape != (n_samples, self.n_components):
            raise ValueError(f"latent_prior must have shape {(n_samples, self.n_components)}")

        if point_weights is None:
            point_weights = np.ones((n_samples, ), dtype=X.dtype)
        else:
            point_weights = np.asarray(point_weights, dtype=X.dtype)
        if point_weights.shape != (n_samples,):
            raise ValueError(f"point_weights must have shape {(n_samples, )}")



        row_sums = latent_prior.sum(axis=1, keepdims=True)
        if np.any(row_sums == 0):
            raise ValueError("Each row of latent_prior must sum to > 0")
        latent_prior /= row_sums
        latent_prior *= self.n_components
        with np.errstate(divide='ignore'):
            log_alpha = np.log(latent_prior)

        return point_weights, latent_prior, log_alpha

    def _initialize_incomplete(self, X, point_weights, alpha):
        """Initialization of the Gaussian mixture parameters using kmeans++, where weights and
        covariance are unknown. means may or may not be known.
        """
        n_samples, _ = X.shape

        means_init = self._initialize_means(X, point_weights, alpha)
        resp = np.zeros((n_samples, self.n_components), dtype=X.dtype)

        if self.init_params == "kmeans":
            max_iters = 300
        elif self.init_params == "kmeans++":
            max_iters = 1
        else:
            raise ValueError("provided ini_params is not valid.")

        label = (
            KMeans(
                n_clusters=self.n_components, n_init=1, init=means_init, max_iter=max_iters, random_state=self.random_state
            )
            .fit(X)
            .labels_
        )
        resp[np.arange(n_samples), label] = 1

        self._initialize_complete(X, resp, point_weights)

    @abc.abstractmethod
    def _initialize_means(self, X, point_weights, alpha):
        """Initializes the means of the clusters through kmeans++ algorithm and taking into account
        the provided prior's information and ordering. If means_init is provided, those are used directly.
        """
        pass

    def _initialize_parameters(self, X, point_weights, alpha):
        """Initialize the model parameters using kmeans++ and information from provided parameters
        and latent variable prior.
        """
        self._incomplete = self._is_incomplete_init()
        if self._incomplete:
            self._initialize_incomplete(X, point_weights, alpha)
        else:
            self._initialize_complete(X, None, point_weights)

        del alpha


    #                                                                                         #
    #---------------------------------- Model Fitting API ------------------------------------#
    def _e_step(self, X, log_alpha):
        """E-step: compute responsibilities γ_ik with masking."""
        with np.errstate(divide='ignore'):
            log_gauss   = self._estimate_log_gaussian_prob(X)
            log_weights = self._estimate_log_weights()
            w_log_gauss = log_gauss + log_weights + log_alpha
            log_norm  = logsumexp(w_log_gauss)
            log_resp = w_log_gauss - log_norm

        return log_resp, log_norm

    @abc.abstractmethod
    def _estimate_log_weights(self):
        """M-step: update weights, means, and covariances."""
        pass

    @abc.abstractmethod
    def _m_step(self, X, resp, point_weights):
        """M-step: update weights, means, and covariances."""
        pass

    @abc.abstractmethod
    def _estimate_log_gaussian_prob(self, X):
        """Compute log N(x | mean, cov) for all components."""
        pass

    ####################################### Public API #######################################
    #                                                                                        #
    #----------------------------------------------------------------------------------------#
    def fit(self, X, point_weights=None, latent_prior=None):
        """
        X             : array (N, D)
        latent_prior  : array (N, K) with prior weights 1>= α_{n,k} >= 0
        """
        t0 = time()

        X = np.ascontiguousarray(X, dtype=self.cast_dtype)
        n_samples, n_features = X.shape

        if self.verbose > 0:
            print(f"Initialization 0; n_samples={n_samples}, n_features={n_features}, n_components={self.n_components}")

        self._check_parameters(X)
        point_weights, alpha, log_alpha = self._initialize_weights_and_prior(X, point_weights, latent_prior)
        self._initialize_parameters(X, point_weights, alpha)

        t1 = time()
        dt = t1 - t0
        if self.verbose > 0:
            if self._incomplete:
                print(f"Initialized 0 (incomplete params.): time lapse={dt:.6f}\n")
            else:
                print(f"Initialized 0 (complete params.):   time lapse={dt:.6f}\n")

        lower_bound = self.lower_bound_
        lower_bounds = [lower_bound]
        for i in range(1, self.max_iter + 1):
            tx = time()

            log_resp, log_norm = self._e_step(X, log_alpha)
            ll = self._compute_lower_bound(log_resp, log_norm, point_weights)

            self._m_step(X, np.exp(log_resp), point_weights)

            change = ll - lower_bound
            lower_bound = ll
            lower_bounds.append(lower_bound)

            dt = time() - tx
            if self.verbose > 1:
                print(f"Iter {i}: time lapse {dt:.6f}, lower bound={ll:.6f}, change={change:.6f}")
            if abs(change) < self.tol:
                self.converged_ = True
                break

        if self.verbose > 0:
            dt = time() - t1
            print(f"Initialization converged {self.converged_}: iters={i}, time lapse={dt:.6f}, lower bound={ll:.6f}")

        self.n_iter_ = i
        self.lower_bound_ = lower_bound
        self.lower_bounds_ = lower_bounds
        self._set_parameters()

        del log_alpha, point_weights, log_norm

        return self

    def predict_log_proba(self, X, latent_prior=None):
        """Posterior p(z=k|x), using the same α stored at fit."""
        X = np.ascontiguousarray(X, dtype=self.cast_dtype)
        _, _, log_alpha = self._initialize_weights_and_prior(
            X,
            np.ones(X.shape[0]),
            latent_prior
        )
        log_resp, _ = self._e_step(X, log_alpha=log_alpha)
        return log_resp

    def predict(self, X, latent_prior=None):
        """Hard labels from the masked posterior."""
        return np.argmax(
            self.predict_log_proba(X, latent_prior=latent_prior),
            axis=1
        )
