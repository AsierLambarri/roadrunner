import numpy as np
from scipy.special import digamma, gammaln
from scipy.linalg import solve_triangular

from ._kmeans_plusplus import kmeans_plusplus_prior
from ._math import row_squared_norms
from .base import BaseMixture
from .weighted_gmm import _estimate_gaussian_parameters


def _log_dirichlet_norm(alpha):
    """Log normalization of Dirichlet distribution.

    Parameters
    ----------
    alpha : array-like of shape (n_components, )
        Concentrations parameters of Dirichlet distribution.
        
    Returns
    -------
    log_dirichlet_norm : float
        Log normalization of Dirichlet.
    """
    return gammaln(np.sum(alpha)) - np.sum(gammaln(alpha))

def _log_wishart_norm(degrees_of_freedom, log_det_prec_chol, n_features):
    """Log normalization of Wishart distribution.

    Parameters
    ----------
    degrees_of_freedom : array-like of shape (n_components, )
        Degrees of freedom on the covariance Wishart distribution.
    log_det_prec_chol : array-like of shape (n_components, )
        Log determinants of precision matrices.
    n_features : float
        Number of data features

    Returns
    -------
    log_wishart_norm : float
        Log normalization of Wishart distribution.
    """
    return (
        - degrees_of_freedom * log_det_prec_chol
        - degrees_of_freedom * n_features * 0.5 * np.log(2)
        - np.sum(
            gammaln(0.5 * (degrees_of_freedom - np.arange(n_features)[:, None])),
            axis=0
        )
    )

def _compute_precision_cholesky(covariances, cov_type):
    """Computes the cholesky decomposition of precision matrices from covariance matrices.
    
    Parameters
    ----------
    covariances : array-like of shape (n_components,) / (n_components, n_features) or (n_components, n_features, n_features)
        Covariance matrices per component. Shape dependent on covariance type.    
    cov_type : str {'full', 'diagonal', 'spherical'}
        Covariance type.

    Returns
    -------
    precisions_chol : array-like of shape covariances.shape
        Cholesky decomposition of precision matrices from current covariances.
        Shape dependent on covariance type. 
    """
    if cov_type == "full":
        n_components, n_features, _ = covariances.shape
        precisions_chol = np.empty(
            (n_components, n_features, n_features), dtype=covariances.dtype
        )
        choleskys = np.linalg.cholesky(covariances)
        for k in range(n_components): 
            precisions_chol[k, :, :] = solve_triangular(
                choleskys[k, :, :], np.eye(n_features), lower=True, overwrite_b=True
            )
    else: 
        # diagonal or shperical
        precisions_chol = 1 / np.sqrt(covariances)

    return precisions_chol

    
def _compute_log_det_cholesky(choleskys, cov_type, n_features):
    """Computes log determinant from Cholesky decomposition of matrices.

    Parameters
    ----------
    choleskys : array-like of shape (n_components,) / (n_components, n_features) or (n_components, n_features, n_features)
        Cholesky matrices. Shape dependent on covariance type.    
    cov_type : str {'full', 'diagonal', 'spherical'}
        Covariance type.
    n_features : float
        Data dimensionality.

    Returns
    -------
    log_det_chol : array-like of shape (n_components, )
        Log determinant from Choleskys.
    """
    if cov_type == "full":
        diagonals = np.diagonal(choleskys, axis1=1, axis2=2)
        log_det_chol = np.sum(
            np.log(diagonals), axis=1
        )
    elif cov_type == "diag" or cov_type == "diagonal":
        log_det_chol = np.sum(
            np.log(choleskys), axis=1
        )        
    elif cov_type == "spherical":
        log_det_chol = np.log(choleskys) * n_features
        
    return log_det_chol

def _estimate_log_gaussian_prob_pchol(X, means, precisions_chol, cov_type):
    """log N(x | mean, cov) for all components in a Gaussian mixture.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data array.
    means : array-like of shape (n_components, n_features)
        Component means.
    precisions_chol : array-like of shape (n_components,) / (n_components, n_features) or (n_components, n_features, n_features)
        Cholesky decomposition of precision matrices per component. Shape dependent on covariance type.
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
    log_det_pchol = _compute_log_det_cholesky(precisions_chol, cov_type, n_features)
    
    if cov_type == "spherical": 
        precisions  = precisions_chol**2
        log_prob = (
            np.sum(means**2, axis=1) * precisions
            - 2.0 * (X @ means.T * precisions)
            + np.outer(row_squared_norms(X), precisions)
        ) 
        
    if cov_type == "diagonal":
        precisions = precisions_chol**2
        log_prob = (
            np.sum(( (means**2) * precisions), axis=1)
            - 2.0 * (X @ (means * precisions).T)
            + ( X**2 @ precisions.T)
        )
        
    if cov_type == "full":
        log_prob = np.empty((n_samples, n_components), dtype=X.dtype)
        for k in range(n_components):
            mu = means[k, :]
            prec_chol = precisions_chol[k, :, :]
            y = (X @ prec_chol) - (mu @ prec_chol)
            log_prob[:, k] = np.sum(np.square(y), axis=1)

    return -0.5 * (const + log_prob) + log_det_pchol






class WeightedBayesianGaussianMixture(BaseMixture):
    def __init__(self, n_components=2, means_init=None, resp_init=None, cov_type="full", 
                 weight_concentration_prior=None, mean_precision_prior=None, mean_prior=None, degrees_of_freedom_prior=None, covariance_prior=None,
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

        self.weight_concentration_prior = weight_concentration_prior
        self.mean_precision_prior       = mean_precision_prior
        self.mean_prior                 = mean_prior
        self.degrees_of_freedom_prior   = degrees_of_freedom_prior
        self.covariance_prior           = covariance_prior

        self.means_init   = means_init
        self.resp_init    = resp_init
        self.means_       = None         
        self.covariances_ = None         
        self.weight_concentration_ = None         
        self.cov_type     = cov_type    

    ####################################### Private API #######################################
    #                                                                                         #
    #---------------------------------- Initialization API -----------------------------------#        
    def _set_parameters(self):
        """Set fitted parameters of the model
        """
        self.weights_ = self.weight_concentration_ / np.sum(
            self.weight_concentration_
        )
        
        if self.cov_type == "full":
            self.precisions_ = np.array([
                np.dot(prec_chol, prec_chol.T) for prec_chol in self.precisions_cholesky_
            ])

        elif self.cov_type == "tied":
            self.precisions_ = np.dot(
                self.precisions_cholesky_, self.precisions_cholesky_.T
            )
        else:
            self.precisions_ = self.precisions_cholesky_**2        
            
    def _check_parameters(self, X):
        """Check the values and shapes of parameter models.
        """        
        self._check_weights_prior()
        self._check_means_prior(X)
        self._check_precisions_prior(X.shape[1])
        self._check_covariance_prior(X)

    def _check_weights_prior(self):
        """Checks and initializes weight concentration prior.
        """
        if self.weight_concentration_prior is None:
            self.weight_concentration_prior_ = np.full(
                self.n_components,
                1.0 / self.n_components,
                dtype=self.cast_dtype,
            )
        else:
            alpha = np.asarray(
                self.weight_concentration_prior,
                dtype=self.cast_dtype
            )
    
            if alpha.ndim == 0:
                alpha = np.full(self.n_components, alpha, dtype=self.cast_dtype)
    
            if alpha.shape != (self.n_components,):
                raise ValueError(
                    "weight_concentration_prior must be scalar or "
                    f"shape ({self.n_components},)"
                )
    
            self.weight_concentration_prior_ = alpha


    def _check_means_prior(self, X):
        """Checks and initializes mean priors.
        """
        _, n_features = X.shape
        
        if self.mean_prior is None:
            mean_prior = np.tile(
                X.mean(axis=0),
                (self.n_components, 1)
            )
        else:
            mean_prior = np.asarray(
                self.mean_prior,
                dtype=self.cast_dtype
            )
            if mean_prior.ndim == 1:
                mean_prior = np.tile(
                    mean_prior,
                    (self.n_components, 1)
                )
        
            expected_shape = (self.n_components, n_features)
            if mean_prior.shape != expected_shape:
                raise ValueError(
                    f"mean_prior must have shape "
                    f"({n_features},) or {expected_shape}"
                )

        
        if self.mean_precision_prior is None:
            beta = np.ones(
                self.n_components,
                dtype=self.cast_dtype
            )
        else:
            beta = np.asarray(
                self.mean_precision_prior,
                dtype=self.cast_dtype
            )
            if beta.ndim == 0:
                beta = np.full(
                    self.n_components,
                    beta,
                    dtype=self.cast_dtype
                )
        
            if beta.shape != (self.n_components,):
                raise ValueError(
                    f"mean_precision_prior must be scalar or "
                    f"shape ({self.n_components},)"
                )

        
        self.mean_prior_ = mean_prior        
        self.mean_precision_prior_ = beta
        
    def _check_covariance_prior(self, X):
        """Checks and initializes covariance prior.
        """
        _, n_features = X.shape        
        
        if self.covariance_prior is None:
            base = {
                "full": lambda x: np.cov(x.T),
                "diag": lambda x: np.var(x, axis=0, ddof=1),
                "diagonal": lambda x: np.var(x, axis=0, ddof=1),
                "spherical": lambda x: np.var(x, axis=0, ddof=1).mean(),
            }[self.cov_type](X)
    
            if self.cov_type == "full":
                cov_prior = np.tile(
                    base[None, :, :],
                    (self.n_components, 1, 1)
                )
            elif self.cov_type in ("diag", "diagonal"):
                cov_prior = np.tile(
                    base[None, :],
                    (self.n_components, 1)
                )
            else:
                cov_prior = np.full(
                    self.n_components,
                    base,
                    dtype=self.cast_dtype
                )
        else:
            cov_prior = np.asarray(
                self.covariance_prior,
                dtype=self.cast_dtype
            )
    
            if self.cov_type == "full" and cov_prior.ndim == 2:
                cov_prior = np.tile(
                    cov_prior[None, :, :],
                    (self.n_components, 1, 1)
                )
            elif self.cov_type in ("diag", "diagonal") and cov_prior.ndim == 1:
                cov_prior = np.tile(
                    cov_prior[None, :],
                    (self.n_components, 1)
                )
            elif self.cov_type == "spherical" and cov_prior.ndim == 0:
                cov_prior = np.full(
                    self.n_components,
                    cov_prior,
                    dtype=self.cast_dtype
                )               
    
        self.covariance_prior_ = cov_prior       

    def _check_precisions_prior(self, n_features):
        """Checks and initializes precision prior.
        """    
        if self.degrees_of_freedom_prior is None:
            self.degrees_of_freedom_prior_ = n_features
        elif self.degrees_of_freedom_prior > n_features - 1.0:
            self.degrees_of_freedom_prior_ = self.degrees_of_freedom_prior
        else:
            raise ValueError(
                f"degrees_of_freedom should be greater than {n_features-1} but for {self.degrees_of_freedom_prior}"
            )

            
    def _is_incomplete_init(self):
        """Checks wether initialization is incomplete or not.
        """
        return (
            self.resp_init is None
        )
        
    def _initialize_complete(self, X, resp, _):
        """Initialization of the Gaussian mixture parameters from complete set of
        means, covars and weights.
        """
        n_samples, _ = X.shape
        
        nk, xk, sk = _estimate_gaussian_parameters(
            X, 
            resp if self._incomplete else self.resp_init, 
            np.ones(X.shape[0]), self.cov_type, self.reg_covar
        )

        self._estimate_weights(nk)
        self._estimate_means(nk, xk)
        self._estimate_covariances(nk, xk, sk)
        del self.resp_init

    def _initialize_means(self, X, _, alpha):
        """Initializes the means of the clusters through kmeans++ algorithm and taking into account
        the provided prior's information and ordering. If means_init is provided, those are used directly.
        """
        n_samples, n_features = X.shape

        if self.means_init is None:
            means, _ = kmeans_plusplus_prior(X, self.n_components, cluster_weights=alpha, random_state=self.random_state)
        else:
            means = self.means_init

        return np.asarray(means, dtype=X.dtype)

    
    #                                                                                         #
    #---------------------------------- Model Fitting API ------------------------------------#
    def _estimate_weights(self, nk):
        """Estimate the weights of the gaussian distributions.
        
        Parameters
        ----------
        nk : array-like of shape (n_components,)
            Effective number of points.        
        """
        self.weight_concentration_ = self.weight_concentration_prior_ + nk
        
    def _estimate_means(self, nk, xk):
        """Estimate the centers of the gaussian distributions.

        Parameters
        ----------
        nk : array-like of shape (n_components,)
            Effective number of points.
        xk : array-like of shape (n_components, n_features)
            Component means.
        """
        self.mean_precision_ = self.mean_precision_prior_ + nk
        self.means_ = (
            self.mean_precision_prior_[:, None] * self.mean_prior_ + nk[:, None] * xk
        ) / self.mean_precision_[:, None]

    def _estimate_covariances(self, nk, xk, sk):
        """Estimate the covariances of the gaussian distributions.

        Parameters
        ----------
        nk : array-like of shape (n_components,)
            Effective number of points.
        xk : array-like of shape (n_components, n_features)
            Component means.
        sk : array-like 
            Covariance matrix of the current components.
                spherical -> (n_components)
                diagonal  -> (n_components, n_features)
                full      -> (n_components, n_features, n_features)
        """
        {
            "full": self._estimate_wishart_full,
            "diagonal": self._estimate_wishart_diagonal,
            "diag": self._estimate_wishart_diagonal,
            "spherical": self._estimate_wishart_spherical,
        }[self.cov_type](nk, xk, sk)

        self.precisions_cholesky_ = _compute_precision_cholesky(
            self.covariances_, self.cov_type
        )

    def _estimate_wishart_full(self, nk, xk, sk):
        """Estimate the full wishart distribution parameters
        """
        _, n_features = xk.shape
        self.degrees_of_freedom_ = self.degrees_of_freedom_prior_ + nk

        self.covariances_ = np.empty((self.n_components, n_features, n_features))
        for k in range(self.n_components):
            diff = xk[k] - self.mean_prior_[k]
            self.covariances_[k] = (
                self.covariance_prior_ [k]
                + nk[k] * sk[k]
                + nk[k] * self.mean_precision_prior_[k]
                / self.mean_precision_[k]
                * np.outer(diff, diff)
            )

        self.covariances_ /= self.degrees_of_freedom_[:, None, None]
                               
    def _estimate_wishart_diagonal(self, nk, xk, sk):
        """Estimate the diagonal wishart distribution parameters
        """
        _, n_features = xk.shape
        self.degrees_of_freedom_ = self.degrees_of_freedom_prior_ + nk

        diff = xk - self.mean_prior_
        self.covariances_ = (
            self.covariance_prior_
            + nk[:, None] * sk
            + nk[:, None] 
            * (self.mean_precision_prior_ / self.mean_precision_)[:, None] 
            * np.square(diff)
        )
        
        self.covariances_ /= self.degrees_of_freedom_[:, None]
                          
    def _estimate_wishart_spherical(self, nk, xk, sk):
        """Estimate the spherical wishart distribution parameters
        """
        _, n_features = xk.shape
        self.degrees_of_freedom_ = self.degrees_of_freedom_prior_ + nk

        diff = xk - self.mean_prior_        
        self.covariances_ = (
            self.covariance_prior_
            + nk * sk
            + nk 
            * (self.mean_precision_prior_ / self.mean_precision_)
            * np.square(diff).mean(axis=1)
        )
        
        self.covariances_ /= self.degrees_of_freedom_

    def _m_step(self, X, resp, _):
        """M-step: update weights, means, and covariances."""
        n_samples, _ = X.shape

        nk, xk, sk = _estimate_gaussian_parameters(
            X, resp, np.ones(X.shape[0]), self.cov_type, self.reg_covar
        )
        self._estimate_weights(nk)
        self._estimate_means(nk, xk)
        self._estimate_covariances(nk, xk, sk)

    def _estimate_log_weights(self):
        """Estimate log concentrations
        """
        return digamma(self.weight_concentration_) - digamma(
            np.sum(self.weight_concentration_)
        )

    def _estimate_log_gaussian_prob(self, X):
        """Compute log N(x | mean, cov) for all components."""
        _, n_features = X.shape

        log_gauss = _estimate_log_gaussian_prob_pchol(
            X, self.means_, self.precisions_cholesky_, self.cov_type
        ) - 0.5 * n_features * np.log(self.degrees_of_freedom_)
        
        log_lambda = (
            n_features * np.log(2)
            + np.sum(digamma(
                0.5 * (self.degrees_of_freedom_ - np.arange(0, n_features)[:, None])
            ), axis=0)
        )

        return log_gauss + 0.5 * (log_lambda - n_features / self.mean_precision_)

    def _compute_lower_bound(self, log_resp, log_prob_norm, _):
        """Compute the log-likelihood lower bound."""
        _, n_features = self.mean_prior_.shape

        log_norm_weight = _log_dirichlet_norm(self.weight_concentration_)

        log_det_pchol = _compute_log_det_cholesky(
            self.precisions_cholesky_, self.cov_type, n_features
        ) - 0.5 * n_features * np.log(self.degrees_of_freedom_)
        
        log_wishart = _log_wishart_norm(
            self.degrees_of_freedom_, 
            log_det_pchol, 
            n_features
        ).sum()

        return (
            - np.sum(np.exp(log_resp) * log_resp)
            - log_norm_weight
            - log_wishart
            - 0.5 * n_features * np.sum(np.log(self.mean_precision_))
        )



        
    def fit(self, X, latent_prior=None):
        return super().fit(
            X, 
            latent_prior=latent_prior,
            point_weights=np.ones(X.shape[0])
        )
