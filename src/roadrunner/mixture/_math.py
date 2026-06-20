#############################################################################
#
# package:   roadrunner.mixture
# file:      _math.py
# brief:     Numba-accelerated numerical kernels for mixture models.
#
# copyright: GPLv3
# author:    Asier Lambarri Martinez
# changes:   13 May 2026 - Created
#            19 Jun 2026 - Last edit
#
#############################################################################

"""Numba-accelerated math primitives for Gaussian mixtures.

Provides numerically stable log-space operations (logsumexp, entropy)
and row-normalisation for responsibility matrices, all implemented
as parallel numba kernels.
"""

import numpy as np
from numba import njit, prange


@njit(parallel=True, fastmath=False, inline="always", cache=True)
def logsumexp(log_probs):
    """Stable log-sum-exp reduction over components (parallel).

    ``out[i] = log(sum_j exp(log_probs[i, j]))``

    Parameters
    ----------
    log_probs : ndarray of shape (N, M)
        Input log-probabilities.

    Returns
    -------
    out : ndarray of shape (N, 1)
        ``log(sum_j exp(...))`` for each row.
    """
    N, M = log_probs.shape
    out = np.empty((N, 1), dtype=log_probs.dtype)

    for i in prange(N):
        a_max = log_probs[i, 0]
        for j in range(1, M):
            v = log_probs[i, j]
            if v > a_max:
                a_max = v

        s = 0.0
        for j in range(M):
            s += np.exp(log_probs[i, j] - a_max)

        out[i, 0] = a_max + np.log(s)

    return out


@njit(parallel=True, cache=True)
def entropy_sum(log_resp, resp):
    """Compute the negative entropy :math:`-\\sum r \\log r` over all entries.

    Only entries where ``resp[i, j] > 0`` contribute, avoiding the
    undefined ``0 * log(0)`` case.

    Parameters
    ----------
    log_resp : ndarray of shape (N, K)
        Log-responsibilities.
    resp : ndarray of shape (N, K)
        Responsibilities (``exp(log_resp)``).

    Returns
    -------
    s : float
        ``sum_{i,j, resp>0} resp[i, j] * log_resp[i, j]``.
    """
    s = 0.0
    for i in prange(log_resp.shape[0]):
        for j in range(log_resp.shape[1]):
            if resp[i, j] > 0:
                s += resp[i, j] * log_resp[i, j]
    return s


@njit(parallel=True, cache=True)
def row_l1_normalize(resp):
    """Normalise each row of a matrix to unit L1 norm (parallel).

    Rows that sum to zero are left unchanged. Very small row sums
    are scaled up before the division to avoid underflow.

    Parameters
    ----------
    resp : ndarray of shape (N, K)
        Input matrix, modified in place.

    Returns
    -------
    resp : ndarray of shape (N, K)
        Row-normalised matrix (each row sums to 1).
    """
    n_samples, n_components = resp.shape
    eps = 10 * np.finfo(resp.dtype).tiny
    SCALE = 1e16
    for n in prange(n_samples):
        row = resp[n]
        row_sum = 0.0
        for k in range(n_components):
            row_sum += float(row[k])
        if row_sum > 0:
            if row_sum < eps:
                for k in range(n_components):
                    row[k] *= SCALE
                row_sum *= SCALE
            inv_sum = 1.0 / row_sum
            for k in range(n_components):
                row[k] *= inv_sum
    return resp


@njit(fastmath=False, cache=True)
def row_squared_norms(X):
    """Compute the squared L2 norm of each row.

    ``result[i] = sum_j X[i, j] ** 2``

    Parameters
    ----------
    X : ndarray of shape (N, D)
        Input matrix.

    Returns
    -------
    result : ndarray of shape (N,)
        Row-wise squared norms.
    """
    n_samples, n_features = X.shape
    result = np.empty(n_samples, dtype=X.dtype)
    for i in range(n_samples):
        s = 0.0
        for j in range(n_features):
            s += X[i, j] * X[i, j]
        result[i] = s
    return result
