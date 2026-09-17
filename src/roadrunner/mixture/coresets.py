#############################################################################
#
# package:   roadrunner.mixture
# file:      coresets.py
# brief:     Gaussian coreset construction via Mahalanobis importance sampling.
#
# Implements importance-sampled coresets for Gaussian mixture models
# using the Mahalanobis distance (following Lucic et al. 2018) instead
# of the Euclidean distance, making the method suitable for
# dimensionally heterogeneous data.
#
# copyright: GPLv3
# author:    Asier Lambarri Martinez
# changes:   13 May 2026 - Created
#            16 Jun 2026 - Last edit
#
#############################################################################

"""Gaussian coreset construction via Mahalanobis importance sampling.

The :class:`GaussianCoreset` builds a weighted subsample (coreset)
that preserves the log-likelihood of a Gaussian mixture to within a
user-specified tolerance, enabling fast approximate fitting on large
datasets.
"""

import numpy as np
from scipy.optimize import root_scalar
from sklearn.cluster import kmeans_plusplus

from ._math import logsumexp
from .weighted_gmm import _estimate_log_gaussian_prob
from roadrunner._defaults import CORESET_ALPHA_BASE, CORESET_ALPHA_OFFSET


def _estimate_mahalanobis_squared(X, means, covs, cov_type):
    """Squared Mahalanobis distance (X-mu)*COV_INV*(x-mu)^T for each sample and component.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data points.
    means : array-like of shape (n_components, n_features)
        Component means.
    covs : array-like of shape (n_components,) / (n_components, n_features) or (n_components, n_features, n_features)
        Covariance matrices per component. Shape dependent on covariance type.
    cov_type : str
        Covariance type.

    Returns
    -------
    qud : array-like of shape (n_samples, n_components)
        Squared Mahalanobis distance of each sample under each component.
    """
    n_components, _ = means.shape
    n_samples, n_features = X.shape
    
    const = n_features * np.log(2 * np.pi)
    log_prob = _estimate_log_gaussian_prob(X, means, covs, cov_type)
    
    if cov_type == "spherical":
        assert np.all(covs > 0), "Spherical covariance matrices need to be possitive definite."
        log_dets = n_features * np.log(covs.reshape(-1))    
    if cov_type == "diagonal":
        assert np.all(covs > 0), "Diagonal covariance matrices need to be possitive definite."
        log_dets = np.sum(np.log(covs.reshape(n_components, n_features)), axis=1)
    if cov_type == "full":
        sign, log_dets = np.linalg.slogdet(covs)
        assert np.all(sign > 0), "Full covariance matrices need to be possitive definite."
    
    return -2 * log_prob - const - log_dets[None, :]


def _estimate_log_likelihood(X, means, covs, weights, quad, cov_type):
    """Estimates the per-point log-likelihood, for fixed parameter values.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data array.
    means : array-like of shape (n_components, n_features)
        Component means.
    covs : array-like of shape (n_components,) / (n_components, n_features) or (n_components, n_features, n_features)
        Covariance matrices per component. Shape dependent on covariance type.
    weights : array-like of shape (n_components,)
        Component Mixing Weights.
    quad : array-like of shape (n_samples, n_components)
        Precomputed Squared Mahalanobis distance.
    cov_type : str
        Covariance type.
        
    Returns
    -------
    loglik : array[float] of shape (n_samples, )
    """   
    n_components, _ = means.shape
    n_samples, n_features = X.shape
    
    const = n_features * np.log(2 * np.pi)
    
    if cov_type == "spherical":
        assert np.all(covs > 0), "Spherical covariance matrices need to be possitive definite."
        log_dets = n_features * np.log(covs.reshape(-1))    
    if cov_type == "diagonal":
        assert np.all(covs > 0), "Diagonal covariance matrices need to be possitive definite."
        log_dets = np.sum(np.log(covs.reshape(n_components, n_features)), axis=1)
    if cov_type == "full":
        sign, log_dets = np.linalg.slogdet(covs)
        assert np.all(sign > 0), "Full covariance matrices need to be possitive definite."
    
        
    log_prob  = -0.5 * (
        const + log_dets[None, :] + quad
    )
    log_lik = logsumexp(log_prob + np.log(weights)[None, :]).flatten()
    return log_lik



def _exact_fudge(q, M, tol=5e-2, max_eta=1000):
    """Exact oversampling factor such that drawing fudge * M samples with replacement yields, 
    in expectation, M unique points. Given by the expression


               E[U(fudge*M)] = M    where    E[U(T)] = sum_i 1 - (1-q_i)^T 

    where E[U(T)] is the expected number of unique points when drawing a total of T points with
    replacements. Fulfills constraints such that E[U(T)] is strictly increasing, E[U(0)] = 0 and 
    E[U(inf)] = n.

    
    Parameters
    ----------
    q : array-like, shape (n,)
        Sampling probabilities. MUST sum to 1.
    M : int
        Desired number of unique points.
    tol : float
        Root-finding tolerance.
    max_eta : float
        Upper bound for the root-bracketing.

    Returns
    -------
    fudge : float
        Oversampling factor.
    """
    q = np.asarray(q, dtype=np.float64)
    if np.any(q < 0):
        raise ValueError("q_i must be nonnegative.")

    s = q.sum()
    if not np.isclose(s, 1.0, atol=1e-5):
        raise ValueError(f"q_i must sum to 1. Currently sum={s}")

    log1mq = np.log1p(-q)

    def expected_unique(T):
        """Expected number of unique points after T draws.

        ``sum_i (1 - (1 - q_i)^T)`` for a categorical distribution
        with probabilities ``q_i``.

        Parameters
        ----------
        T : float
            Number of draws.

        Returns
        -------
        n_unique : float
        """
        return np.sum(1.0 - np.exp(T * log1mq))

    def f(eta):
        """Equation U(eta*M) - M = 0, solved to find the coreset size.

        Parameters
        ----------
        eta : float
            Ratio of coreset size to full data size.

        Returns
        -------
        residual : float
        """
        return expected_unique(eta * M) - M

    sol = root_scalar(f, bracket=[1e-1, max_eta], method='brentq', xtol=tol, x0=2)
    if not sol.converged:
        return -1.5
        
    return sol.root




class GaussianCoreset:
    """Handles coreset construction for Gaussian Mixture Models following M. Lucic et al. 2016, O. Bachem et al. 2017 and M. Lucic et al. 2018.The
    CORESETs are sampled using importance sampling as in M. Lucic et al. 2018, but the EUCLIDEAN distance is changed for the MAHALANOBIS distance,
    in order to accomodate dimensionally heterogeneous data and accomodate the significance to the specific GAUSSIAN model.
    """
    def __init__(self, n_components=2, centers=None, covariances=None, cov_type="full", random_state=None, **kwargs):
        """Init importance sampling and coreset construction.

        Parameters
        ----------
        n_components : int
            Number of components or clusters. Default to 2.
        centers : array of shape (n_components, n_features)
            Cluster centers. Estimated using kmeans++ if not provided.
        covariances : array of shape (n_features, n_features, n_components)
            Gaussian Covariance of each component. Initialized to IDENTITY if not provided.
        random_state : int or numpy.random
            Random State used in sampling.
        """
        self.n_components = n_components
        self.centers      = centers
        self.covariances  = covariances
        self.cov_type     = cov_type
        self.random_state = random_state if isinstance(random_state, np.random.RandomState) else np.random.RandomState(random_state)
        
        self.alpha        = CORESET_ALPHA_BASE * (np.log2(self.n_components) + CORESET_ALPHA_OFFSET)

        
    def _initialize_centroids(self, X):
        """Initialise centroids and covariances using k-means++.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        centroids : ndarray of shape (n_components, n_features)
        covariances : list
            Initial covariance estimates (identity for each component).
        """
        _, n_features = X.shape
        centroids, _ = kmeans_plusplus(
            X, n_clusters=self.n_components
        )
        if self.cov_type == "spherical":
            covariances = [1] * self.n_components
            
        if self.cov_type == "diagonal":
            covariances = [np.ones(n_features)] * self.n_components
            
        if self.cov_type == "full":
            covariances = [np.eye(n_features)] * self.n_components

        return centroids, covariances 

    def _initialize(self, X):
        """Initialise the centers and covariances.

        Falls back to :meth:`_initialize_centroids` if either
        ``self.centers`` or ``self.covariances`` is ``None``.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        """
        if self.centers is None or self.covariances is None:
            centers, covariances = self._initialize_centroids(X)

        self.centers     = centers if self.centers is None else self.centers
        self.covariances = covariances if self.covariances is None else self.covariances
        
        self.centers = np.asarray(self.centers, dtype=X.dtype)
        self.covariances = np.asarray(self.covariances, dtype=X.dtype)
        
    def _estimate_importances(self, X, quad):
        """Estimate the importance q(x) of each sample.

        Uses the per-cluster quadratic distances to compute a sampling
        weight for coreset construction, following Lucic et al. (2018).

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        quad : ndarray of shape (n_samples, n_components)
            Quadratic distances to each cluster centre.

        Returns
        -------
        importances : ndarray of shape (n_samples,)
            Positive importance weights.
        """
        n_samples, n_features = X.shape

        cluster_membership = np.argmin(quad, axis=1)
        min_quad = quad[np.arange(n_samples), cluster_membership]
        total_cost = np.sum(min_quad)
        
        importances        = -1 * np.ones(n_samples, dtype=X.dtype)
        for k in range(self.n_components):
            mask = cluster_membership == k
            cluster_size = np.sum(mask)
            if cluster_size == 0:
                continue

            cluster_cost = np.sum(min_quad[mask])
            
            importances[mask] = (
                self.alpha * min_quad[mask] 
                + (2 * self.alpha / cluster_size) * cluster_cost
                + (2 / cluster_size) * total_cost
            )

        assert np.all(importances > 0), ""

        return importances, cluster_membership
    
    
    def fit(self, X):
        """Fits the Gaussian Coreset model to the data.
        
        Parameters
        ----------
        X : array[float] of shape (n_samples, n_features)
            Data points.

        Returns
        -------
        self : class <GaussianCoreset>
        """
        self.X = np.ascontiguousarray(X)
        n_samples, n_features = self.X.shape
        
        self._initialize(self.X)

        self.quad = _estimate_mahalanobis_squared(self.X, self.centers, self.covariances, self.cov_type)
            
        self.importance_estimates, self.membership = self._estimate_importances(X, self.quad)
        self.sampling_weights = self.importance_estimates / np.sum(self.importance_estimates)

        return self

    
    def generate_coreset(self, Ncore):
        """Draws coreset samples and generates weights with or without replacement. When drawn 
        with replacement, the weights are computed as c_i / (N_core * q_i) where c_i is the 
        multiplicity of the data point. When drawn with replacement, the unique number of core samples
        is samller than N_core.

        Parameters
        ----------
        X : array[float] of shape (n_samples, n_features)
            Data points.
        N_core : int
            Number of draws to sample the coreset.

        Returns
        -------
        C : array[float] of shape (n_core, n_features)
        weights : array[float] of shape (n_core)
        """
        n_samples, n_features = self.X.shape
        
        sampled_indices = self.random_state.choice(
            np.arange(n_samples), 
            size=int(Ncore), 
            p=self.sampling_weights, 
            replace=True
        )
        unique_sampling_indices, ocurrences = np.unique(sampled_indices, return_counts=True)
        weights = np.array([c_i / (self.sampling_weights[i] * Ncore) for i, c_i in zip(unique_sampling_indices, ocurrences)])    
        
        return self.X[unique_sampling_indices, :], weights, unique_sampling_indices


    def estimate_error(self, sampled_indices, weights):
        """Estimates the accuracy of the coreset epsilon, defined as 

               | L(X|theta) - L(C|theta) | < epsilon * L(X|theta)

        where L(X|theta) is the usual GM log-likelihood and L(C|theta) is the
        weighted GM log-likelihood as in M. Lucin et al. 2018

        For the sklearn metric, one should use weights / weights.sum(): since the
        loglik for each point is defined as w_i*log[p(x_i)], the mean would be

                            sum( w_i*log[p(x_i)] )
                            ----------------------
                                    sum(w_i)

        Parameters
        ----------
        X : array[float] of shape (n_samples, n_features)
            Data points.
        C : array[float] of shape (n_core, n_features)
            Core sample.
        weights : array[float] of shape (n_core)
            Core weights.

        Returns
        -------
        eps_sum : float
        eps_mean : float
        """
        n_samples, n_features = self.X.shape
                
        mixing_weights = np.array([self.membership[self.membership == k].size for k in range(self.n_components)])
        mixing_weights = mixing_weights / mixing_weights.sum()
        base_loglik = _estimate_log_likelihood(
            self.X,
            self.centers, self.covariances,
            mixing_weights, self.quad,
            self.cov_type
        )       
        coreset_loglik = base_loglik[sampled_indices] * weights
        
        base_sum, base_mean = np.sum(base_loglik), np.mean(base_loglik)
        core_sum, core_mean = np.sum(coreset_loglik), np.mean(coreset_loglik) * (weights.size / weights.sum())
        

        epsilon_sum  = np.abs( (core_sum - base_sum) / base_sum )
        epsilon_mean = np.abs( (core_mean - base_mean) / base_mean )
        return epsilon_sum, epsilon_mean      
