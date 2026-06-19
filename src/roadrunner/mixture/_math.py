import numpy as np
from numba import njit, prange


@njit(parallel=True, fastmath=False, inline="always", cache=True)
def logsumexp(log_probs):
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
    s = 0.0
    for i in prange(log_resp.shape[0]):
        for j in range(log_resp.shape[1]):
            if resp[i, j] > 0:
                s += resp[i, j] * log_resp[i, j]
    return s


@njit(parallel=True, cache=True)
def row_l1_normalize(resp):
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
    n_samples, n_features = X.shape
    result = np.empty(n_samples, dtype=X.dtype)
    for i in range(n_samples):
        s = 0.0
        for j in range(n_features):
            s += X[i, j] * X[i, j]
        result[i] = s
    return result
