#############################################################################
#
# package:   roadrunner.mixture
# file:      svi_bayesian_gmm.py
# brief:     Stochastic variational Bayesian Gaussian mixture (SVI-BGMM).
#
# Implements mini-batch natural-gradient SVI for the variational
# Bayesian Gaussian mixture model.  Convergence is assessed via a
# rolling window of relative lower-bound changes (Stan-style
# mean/median delta-ELBO criterion).
#
# copyright: GPLv3
# author:    Asier Lambarri Martinez
# changes:   20 Jun 2026 - Created
#
#############################################################################

"""Stochastic variational Bayesian Gaussian mixture (SVI-BGMM).

The :class:`SVIBayesianGaussianMixture` class extends
:class:`WeightedBayesianGaussianMixture` by replacing the full-data
VB M-step with a mini-batch natural-gradient update.  The learning
rate follows a Robbins-Monro schedule ``rho = (t + tau)^(-kappa)``.
Convergence is detected when the mean or median relative change of
the lower bound over a rolling window falls below ``tol``.
"""

import warnings

import numpy as np

from .bayesian_gmm import (
    WeightedBayesianGaussianMixture,
    _compute_precision_cholesky,
)
from .weighted_gmm import _estimate_gaussian_parameters


class SVIBayesianGaussianMixture(WeightedBayesianGaussianMixture):
    """Stochastic variational Bayesian Gaussian mixture.

    Inherits all parameter handling, priors, E-step, and Wishart
    updates from WeightedBayesianGaussianMixture. Overrides fit()
    to use mini-batch natural gradient updates instead of full-data EM.

    Convergence is assessed by monitoring the relative change in the
    variational lower bound over a rolling window of evaluations.
    When the windowed mean or median relative change falls below
    ``tol``, the algorithm is considered converged.

    Parameters
    ----------
    n_svi_iters : int, default=1000
        Maximum number of SVI iterations (hard cap).

    batch_size : int, default=10000
        Number of data points sampled per mini-batch.

    tau : float, default=1.0
        Learning rate schedule offset: rho = (t + tau)^(-kappa).

    kappa : float, default=0.5
        Learning rate schedule exponent.

    eval_every : int, default=5
        Number of SVI iterations between lower-bound evaluations.

    window_size : int, default=5
        Number of relative-change deltas in the rolling buffer.

    tol : float, default=1e-3
        Convergence threshold. The algorithm stops when
        mean(delta_buffer) < tol or median(delta_buffer) < tol.

    **kwargs
        All parameters of WeightedBayesianGaussianMixture.
    """

    def __init__(self, n_svi_iters=1000, batch_size=10000,
                 tau=1.0, kappa=0.5, eval_every=5, window_size=5,
                 tol=1e-3, **kwargs):
        super().__init__(**kwargs)
        self.n_svi_iters = n_svi_iters
        self.batch_size = batch_size
        self.tau = tau
        self.kappa = kappa
        self.eval_every = eval_every
        self.window_size = window_size
        self.tol = tol

    def _sample_batch(self, n_samples):
        """Draw a uniform random mini-batch index array."""
        M = min(self.batch_size, n_samples)
        return self.random_state.choice(n_samples, M, replace=False)

    def _svi_m_step(self, X_b, resp_b, pw_b, rho, n_total):
        """Natural gradient M-step on mini-batch statistics."""
        nk, xk, sk = _estimate_gaussian_parameters(
            X_b, resp_b, pw_b, self.cov_type, self.reg_covar)
        nk_s = nk * (n_total / X_b.shape[0])

        old = {
            k: getattr(self, k).copy()
            for k in (
                'weight_concentration_',
                'mean_precision_',
                'means_',
                'degrees_of_freedom_',
                'covariances_',
            )
        }

        self._estimate_weights(nk_s)
        self._estimate_means(nk_s, xk)
        self._estimate_covariances(nk_s, xk, sk)

        for k in old:
            setattr(self, k, (1 - rho) * old[k] + rho * getattr(self, k))

        self.precisions_cholesky_ = _compute_precision_cholesky(
            self.covariances_, self.cov_type)

    # ── Convergence methods ──────────────────────────────────────────

    def _store_lower_bound(self, lb):
        """Update the relative-change rolling buffer and track best LB."""
        if self._prev_lb is not None:
            delta = abs(lb - self._prev_lb) / max(abs(self._prev_lb), 1.0)
            self._delta_buffer.append(delta)
            if len(self._delta_buffer) > self.window_size:
                self._delta_buffer.pop(0)
        self._prev_lb = lb
        self._best_lb = max(self._best_lb, lb)
        self.lower_bound_ = lb

    def _is_converged(self):
        """Check if the rolling-window ELBO change has stabilised."""
        if len(self._delta_buffer) < self.window_size:
            return False
        return (
            np.mean(self._delta_buffer) < self.tol
            or np.median(self._delta_buffer) < self.tol
        )

    def _is_diverging(self):
        """Check if the optimisation is numerically unstable."""
        if len(self._delta_buffer) < self.window_size:
            return False
        return np.median(self._delta_buffer) > 0.5

    def _check_convergence_quality(self):
        """Warn if the final LB is substantially worse than the best seen."""
        if self._best_lb > -np.inf and self.lower_bound_ > -np.inf:
            ratio = abs(self.lower_bound_ - self._best_lb) / max(abs(self._best_lb), 1.0)
            if ratio > 0.05 and self.verbose > 0:
                warnings.warn(
                    f"Best lower bound >5% better than final "
                    f"({self._best_lb:.1f} vs {self.lower_bound_:.1f})",
                    UserWarning,
                )

    # ── fit() override ──────────────────────────────────────────────

    def fit(self, X, latent_prior=None):
        """Fit the SVI Bayesian GMM using stochastic variational inference.

        The method uses mini-batch natural-gradient updates with a
        Robbins–Monro learning-rate schedule and rolling-window
        convergence detection.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        latent_prior : SparseCSC or None, optional
            Per-point-per-component responsibility prior.

        Returns
        -------
        self : SVIBayesianGaussianMixture
        """
        from time import time

        t0 = time()
        X = np.ascontiguousarray(X, dtype=self.cast_dtype)
        n_samples, n_features = X.shape

        if self.verbose > 0:
            print(f"Initialization 0; n_samples={n_samples}, "
                  f"n_features={n_features}, "
                  f"n_components={self.n_components}")

        self._check_parameters(X)
        _, alpha, log_alpha = self._initialize_weights_and_prior(
            X, np.ones(n_samples, dtype=X.dtype), latent_prior)
        self._initialize_parameters(
            X, np.ones(n_samples, dtype=X.dtype), alpha)

        t1 = time()
        dt = t1 - t0
        if self.verbose > 0:
            if self._incomplete:
                print(f"Initialized 0 (incomplete params.): "
                      f"time lapse={dt:.6f}\n")
            else:
                print(f"Initialized 0 (complete params.):   "
                      f"time lapse={dt:.6f}\n")

        # ── Convergence state ───────────────────────────────────────
        self._delta_buffer = []
        self._prev_lb = None
        self._best_lb = -np.inf
        self.converged_ = False
        n_iters = self.n_svi_iters
        last_log_resp, last_log_norm = None, None

        # ── SVI loop ────────────────────────────────────────────────
        report_every = max(1, self.n_svi_iters // 10)

        for t in range(1, self.n_svi_iters + 1):
            tx = time()

            idx_b = self._sample_batch(n_samples)
            X_b = X[idx_b]
            log_alpha_b = log_alpha[idx_b]

            log_resp_b, log_norm_b = self._e_step(X_b, log_alpha_b)
            last_log_resp, last_log_norm = log_resp_b, log_norm_b

            rho = (t + self.tau) ** (-self.kappa)
            self._svi_m_step(
                X_b, np.exp(log_resp_b),
                np.ones(len(idx_b), dtype=X.dtype),
                rho, n_samples)

            # ── Convergence check ───────────────────────────────────
            if t % self.eval_every == 0:
                pw_b = np.ones(len(idx_b), dtype=X.dtype)
                lb = self._compute_lower_bound(
                    log_resp_b, log_norm_b, pw_b)
                self._store_lower_bound(lb)

                if self._is_diverging() and self.verbose > 0:
                    warnings.warn(
                        f"Iter {t}: SVI may be diverging (median Δ>0.5)",
                        UserWarning,
                    )

                if self._is_converged():
                    self.converged_ = True
                    n_iters = t
                    break

            # ── Verbose reporting ───────────────────────────────────
            if self.verbose > 1 and t % report_every == 0:
                if t % self.eval_every == 0:
                    lb_report = lb
                else:
                    lb_report = self.lower_bound_
                dt_iter = time() - tx
                print(f"Iter {t}: time lapse {dt_iter:.6f}, "
                      f"lower bound={lb_report:.6f}, rho={rho:.6f}")

        # ── Finalise ────────────────────────────────────────────────
        self.n_iter_ = n_iters

        if self.lower_bound_ == -np.inf:
            pw_b = np.ones(len(idx_b), dtype=X.dtype)
            self.lower_bound_ = self._compute_lower_bound(
                last_log_resp, last_log_norm, pw_b)

        self._set_parameters()

        dt = time() - t1
        if self.verbose > 0:
            status = "converged" if self.converged_ else "max iterations"
            print(f"Initialization {status}: iters={n_iters}, "
                  f"time lapse={dt:.6f}, "
                  f"lower bound={self.lower_bound_:.6f}")

        self._check_convergence_quality()

        return self
