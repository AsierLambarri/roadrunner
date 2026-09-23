#############################################################################
#
# package:   roadrunner.mixture
# file:      base.py
# brief:     Abstract base class for weighted Gaussian mixture models.
#
# Implements the common EM loop, E-step, convergence check, and the
# abstract methods that concrete mixture classes (standard, Bayesian,
# SVI) must implement.
#
# copyright: GPLv3
# author:    Asier Lambarri Martinez
# changes:   13 May 2026 - Created
#            16 Jun 2026 - Last edit
#
#############################################################################

"""Abstract base class for Gaussian mixture models with weighted responsibilities.

:class:`BaseMixture` provides the skeleton EM loop and the
:meth:`_initialize_weights_and_prior` helper that normalises a
``latent_prior`` into ``log_alpha`` for the E-step.  Concrete
subclasses implement :meth:`_check_parameters`,
:meth:`_initialize_complete`, :meth:`_is_incomplete_init`,
:meth:`_initialize_means`, :meth:`_m_step`,
:meth:`_estimate_log_weights`, and
:meth:`_estimate_log_gaussian_prob`.
"""

import abc
from time import time

import numpy as np

from ._math import logsumexp
from .kmeans import WeightedKMeans
from roadrunner._defaults import KMEANS_MAX_ITER, KMEANS_PP_MAX_ITER


def _nonpositive_definite(covariances, cov_type):
    """Check if any covariance matrix is non-positive definite.

    Parameters
    ----------
    covariances : ndarray
        Covariance matrices. Shape depends on ``cov_type``.
    cov_type : str
        One of ``'spherical'``, ``'diagonal'``, or ``'full'``.

    Returns
    -------
    bad : bool or ndarray of bool
        ``True`` for components whose covariance is non-positive definite.
    """
    if cov_type == "spherical":
        return covariances <= 0
    elif cov_type == "diagonal":
        return np.any(covariances <= 0, axis=1)
    elif cov_type == "full":
        symm_cov = 0.5 * (covariances + np.transpose(covariances, (0, 2, 1)))
        eigvals = np.linalg.eigvalsh(symm_cov)
        return np.any(eigvals <= 0, axis=1)

def _check_parameter_values(weights, means, covariances, cov_type):
    """Check mixture parameters for NaN, Inf, or non-positive-definite values.

    Parameters
    ----------
    weights : ndarray of shape (n_components,)
        Component weights.
    means : ndarray of shape (n_components, n_features)
        Component means.
    covariances : ndarray
        Component covariances. Shape depends on ``cov_type``.
    cov_type : str
        Covariance type.

    Returns
    -------
    has_nan : tuple of bool
        Three-element tuple ``(weights_bad, means_bad, covariances_bad)``.
    """
    return (
        np.any(np.isnan(weights)) or np.any(np.isinf(weights)) or np.any(weights <= 0),
        np.any(np.isnan(means)) or np.any(np.isinf(means)),
        np.any(np.isnan(covariances)) or np.any(np.isinf(covariances)) or np.any(_nonpositive_definite(covariances, cov_type))
    )





class BaseMixture(abc.ABC):
    """Abstract base class for Gaussian mixture models with weighted responsibilities.

    Implements the common EM loop, ``fit()``, ``predict()``,
    and ``predict_log_proba()``.  Subclasses must implement the
    abstract methods to define their own parameterisation (standard
    EM, variational Bayes, or stochastic variational inference).

    Parameters
    ----------
    n_components : int, default=2
        Number of mixture components.
    init_params : str, default='kmeans'
        Initialisation method.
    max_iter : int, default=10
        Maximum number of EM iterations.
    tol : float, default=1e-3
        Convergence threshold on the change in lower bound, relative
        to ``max(|lower_bound|, 1)``.
    verbose : int, default=0
        Verbosity level.
    random_state : int or RandomState, optional
        Random state for reproducibility.
    reg_covar : float, default=1e-6
        Regularisation added to the diagonal of covariance matrices.
        Stored raw (a Python float never upcasts an array); applied to
        working-precision arrays at the use sites.
    **kwargs
        Additional keyword arguments consumed by subclasses.
    """

    def __init__(self, n_components=2, init_params='kmeans', max_iter=10, tol=1e-3, verbose=0,
                 random_state=None, reg_covar=1E-6, **kwargs):

        self.reg_covar   = reg_covar

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
        """Update derived quantities after a parameter update.

        Called at the end of ``fit()`` to compute ``weights_``,
        ``precisions_``, etc. from the current parameter state.

        Subclasses must override to populate any cached attributes
        that depend on the raw fitted parameters.
        """

    @abc.abstractmethod
    def _check_parameters(self, X):
        """Validate and initialise all prior/hyper-parameters.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data (used to infer default priors).
        """

    @abc.abstractmethod
    def _initialize_complete(self, X, resp, point_weights):
        """Initialise model parameters from a complete set of responsibilities.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        resp : ndarray of shape (n_samples, n_components) or None
            Responsibilities (``None`` when using externally-provided
            ``counts_init``, ``means_init``, ``covariance_init``).
        point_weights : ndarray of shape (n_samples,)
            Per-point weights.
        """

    @abc.abstractmethod
    def _is_incomplete_init(self):
        """Check whether initialisation requires k-means.

        Returns
        -------
        incomplete : bool
            ``True`` if the model needs k-means to obtain initial
            responsibilities.
        """

    def _initialize_weights_and_prior(self, X, point_weights, latent_prior):
        """Normalise the latent prior and compute ``log_alpha`` for the E-step.

        If ``latent_prior`` is ``None`` a flat (uniform) prior is
        assumed.  The returned ``log_alpha`` is ``log(alpha)`` where
        ``alpha`` has been row-normalised and scaled by
        ``n_components`` so that ``sum_k alpha_{n,k} = n_components``
        for every row.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data (used for shape validation only).
        point_weights : ndarray or None
            Per-point weights. ``None`` means uniform weights.
        latent_prior : ndarray of shape (n_samples, n_components) or None
            Per-point, per-component prior weights, used as soft-label
            evidence in the E-step; zero entries act as hard masks.

        Returns
        -------
        point_weights : ndarray of shape (n_samples,)
            Per-point weights (always an array).
        alpha : ndarray of shape (n_samples, n_components)
            Normalised latent prior.
        log_alpha : ndarray of shape (n_samples, n_components)
            ``log(alpha)`` for use in the E-step.
        """
        n_samples, n_features = X.shape

        if latent_prior is None:
            latent_prior = np.ones((n_samples, self.n_components), dtype=X.dtype)
        else:
            latent_prior = np.asarray(latent_prior, dtype=X.dtype).copy()   # never rescale the caller's array
        if latent_prior.shape != (n_samples, self.n_components):
            raise ValueError(f"latent_prior must have shape {(n_samples, self.n_components)}")
        if np.any(latent_prior < 0):
            raise ValueError("latent_prior must be non-negative")

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
        """Initialise parameters using k-means hard assignment.

        Means are obtained from ``_initialize_means()``, hard labels
        are assigned via scikit-learn ``KMeans``, and then
        ``_initialize_complete()`` is called with the resulting
        one-hot responsibilities.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        point_weights : ndarray of shape (n_samples,)
            Per-point weights.
        alpha : ndarray of shape (n_samples, n_components)
            Latent prior (used to weight the k-means sampling).
        """
        n_samples, _ = X.shape

        means_init = self._initialize_means(X, point_weights, alpha)
        resp = np.zeros((n_samples, self.n_components), dtype=X.dtype)

        if self.init_params == "kmeans":
            max_iters = KMEANS_MAX_ITER
        elif self.init_params == "kmeans++":
            max_iters = KMEANS_PP_MAX_ITER
        else:
            raise ValueError("provided ini_params is not valid.")

        label = (
            WeightedKMeans(n_clusters=self.n_components, init=means_init, max_iter=max_iters)
            .fit(X, point_weights=point_weights, cluster_weights=alpha)
            .labels_
        )
        resp[np.arange(n_samples), label] = 1

        self._initialize_complete(X, resp, point_weights)

    @abc.abstractmethod
    def _initialize_means(self, X, point_weights, alpha):
        """Initialise component means (k-means++ with prior weighting).

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        point_weights : ndarray of shape (n_samples,)
            Per-point weights.
        alpha : ndarray of shape (n_samples, n_components)
            Latent prior.

        Returns
        -------
        means : ndarray of shape (n_components, n_features)
            Initial component means.
        """

    def _initialize_parameters(self, X, point_weights, alpha):
        """Dispatch initialisation to complete or incomplete path.

        If :meth:`_is_incomplete_init` returns ``True`` the
        k-means path is taken; otherwise the complete path is used,
        which assumes ``counts_init``, ``means_init``, and
        ``covariance_init`` have been provided externally.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        point_weights : ndarray of shape (n_samples,)
            Per-point weights.
        alpha : ndarray of shape (n_samples, n_components)
            Latent prior.
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
        """E-step: compute posterior responsibilities.

        ``log_resp[i, k] = log_gauss[i, k] + log_weights[k] + log_alpha[i, k] - log_norm[i]``

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Data.
        log_alpha : ndarray of shape (n_samples, n_components)
            Normalised log prior.

        Returns
        -------
        log_resp : ndarray of shape (n_samples, n_components)
            Log of the posterior responsibilities.
        log_norm : ndarray of shape (n_samples, 1)
            Log normalisation constants.
        """
        with np.errstate(divide='ignore'):
            log_gauss   = self._estimate_log_gaussian_prob(X)
            log_weights = self._estimate_log_weights()
            w_log_gauss = log_gauss + log_weights + log_alpha
            log_norm  = logsumexp(w_log_gauss)
            log_resp = w_log_gauss - log_norm

        return log_resp, log_norm

    @abc.abstractmethod
    def _estimate_log_weights(self):
        """Compute log of the component weights.

        Returns
        -------
        log_weights : ndarray of shape (n_components,)
            ``log(weights_[k])`` for each component.
        """

    @abc.abstractmethod
    def _m_step(self, X, resp, point_weights):
        """M-step: update component parameters.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Data.
        resp : ndarray of shape (n_samples, n_components)
            Posterior responsibilities.
        point_weights : ndarray of shape (n_samples,)
            Per-point weights.
        """

    @abc.abstractmethod
    def _estimate_log_gaussian_prob(self, X):
        """Compute log Gaussian density for all components.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Data.

        Returns
        -------
        log_prob : ndarray of shape (n_samples, n_components)
            ``log N(x_n | mu_k, Sigma_k)``.
        """

    ####################################### Public API #######################################
    #                                                                                        #
    #----------------------------------------------------------------------------------------#
    def fit(self, X, point_weights=None, latent_prior=None):
        """Fit the mixture model to the data via EM.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        point_weights : ndarray of shape (n_samples,), optional
            Per-point weights (defaults to uniform).
        latent_prior : ndarray of shape (n_samples, n_components), optional
            Per-point, per-component prior weights (``None`` = flat).

        Returns
        -------
        self : BaseMixture
            The fitted model.
        """
        t0 = time()

        X = np.ascontiguousarray(X)
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

        self.converged_ = False
        lower_bound = -np.inf
        lower_bounds = [lower_bound]
        for i in range(1, self.max_iter + 1):
            tx = time()

            log_resp, log_norm = self._e_step(X, log_alpha)
            self._m_step(X, np.exp(log_resp), point_weights)
            # Bound after the M-step (sklearn order): the simplified VB bound is only
            # valid when q(theta) has just been updated from these responsibilities.
            ll = self._compute_lower_bound(log_resp, log_norm, point_weights, log_alpha)

            change = ll - lower_bound
            lower_bound = ll
            lower_bounds.append(lower_bound)

            dt = time() - tx
            if self.verbose > 1:
                print(f"Iter {i}: time lapse {dt:.6f}, lower bound={ll:.6f}, change={change:.6f}")
            if abs(change) < self.tol * max(abs(lower_bound), 1.0):
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
        """Compute log posterior probabilities for the fitted model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Data.
        latent_prior : ndarray of shape (n_samples, n_components), optional
            Per-point mask / prior.

        Returns
        -------
        log_resp : ndarray of shape (n_samples, n_components)
            ``log p(z_n = k | x_n)``.
        """
        X = np.ascontiguousarray(X)
        _, _, log_alpha = self._initialize_weights_and_prior(
            X,
            np.ones(X.shape[0], dtype=X.dtype),
            latent_prior
        )
        log_resp, _ = self._e_step(X, log_alpha=log_alpha)
        return log_resp

    def predict(self, X, latent_prior=None):
        """Return hard cluster assignments.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Data.
        latent_prior : ndarray of shape (n_samples, n_components), optional
            Per-point mask / prior.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the most-likely component for each point.
        """
        return np.argmax(
            self.predict_log_proba(X, latent_prior=latent_prior),
            axis=1
        )
