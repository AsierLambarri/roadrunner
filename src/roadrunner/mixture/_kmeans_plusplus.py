#############################################################################
#
# package:   roadrunner.mixture
# file:      _kmeans_plusplus.py
# brief:     K-means++ initialisation with cluster-dependent per-sample weights.
#
# copyright: GPLv3
# author:    Asier Lambarri Martinez
# changes:   15 Apr 2025 - Created
#            13 May 2026 - Last edit
#
#############################################################################

"""K-means++ seeding with cluster-dependent per-sample weights.

Extends the standard k-means++ algorithm by allowing the sampling
weights of each data point to depend on the cluster (component)
being seeded.  This is required for initialising mixture models
with a non-uniform latent prior.
"""

import numpy as np

from ._math import row_squared_norms
from roadrunner._defaults import LOCAL_IDX, math_dtype
    
def kmeans_plusplus_prior(X, n_clusters, *, cluster_weights=None, random_state=None):
    """K-means++ seeding with cluster-dependent per-point weights.

    The probability of selecting point ``i`` as the next center for
    cluster ``k`` is proportional to
    ``cluster_weights[i, k] * distance_to_nearest_center[i]``.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input data.
    n_clusters : int
        Number of clusters to seed.
    cluster_weights : ndarray of shape (n_samples, n_clusters), optional
        Per-point, per-cluster sampling weights.  If ``None``,
        standard uniform k-means++ is used.
    random_state : int or RandomState, optional
        Random state for reproducibility.

    Returns
    -------
    centers : ndarray of shape (n_clusters, n_features)
        Seeded cluster centers.
    indices : ndarray of shape (n_clusters,)
        Indices of the selected centers in ``X``.
    """
    X = np.asarray(X, dtype=math_dtype())
    n_samples, n_features = X.shape

    if isinstance(random_state, (int, np.integer)):
        rng = np.random.default_rng(random_state)
    elif random_state is None:
        rng = np.random.default_rng()
    else:
        rng = random_state
        
    n_local_trials = 2 + int(np.log(n_clusters))
    
    centers        = np.empty((n_clusters, n_features), dtype=X.dtype)
    center_indices = np.empty(n_clusters, dtype=LOCAL_IDX)
    
    candidate_dist_sq_all = np.empty((n_samples, n_local_trials), dtype=X.dtype)
    
    cumsum_buf = np.empty(n_samples, dtype=X.dtype)
    X_norm_sq  = row_squared_norms(X)
    
    if cluster_weights is None:
        cluster_weights = np.ones((n_samples, n_clusters), dtype=X.dtype)
        
    cluster_weights = np.asarray(cluster_weights, dtype=X.dtype)
    if cluster_weights.shape != (n_samples, n_clusters):
        raise ValueError("cluster_weights must have shape (n_samples, n_clusters)")
    if np.any(cluster_weights < 0):
        raise ValueError("cluster_weights must be non-negative")
    if np.any(cluster_weights.sum(axis=0) == 0):
        raise ValueError("Each cluster (column) must have non-zero total sample weights")
    
    first_idx = rng.choice(
        n_samples, 
        size=1, 
        p=cluster_weights[:, 0] / cluster_weights[:, 0].sum() 
    )[0]
    centers[0]        = X[first_idx]
    center_indices[0] = first_idx
    
    closest_dist_sq = row_squared_norms(X - centers[0])
    for c in range(1, n_clusters):
        weights = cluster_weights[:, c]
        
        np.multiply(closest_dist_sq, weights, out=cumsum_buf)
        np.cumsum(cumsum_buf, out=cumsum_buf)
        if not np.isfinite(cumsum_buf[-1]) or cumsum_buf[-1] <= 0:
            probs = weights / weights.sum()
            candidate_ids = rng.choice(n_samples, size=n_local_trials, p=probs)
        else:
            rand_vals = rng.uniform(0, cumsum_buf[-1], size=n_local_trials)
            candidate_ids = np.searchsorted(cumsum_buf, rand_vals)    

        np.clip(candidate_ids, 0, n_samples - 1, out=candidate_ids)

        Y = X[candidate_ids]
        Y_norm_sq = row_squared_norms(Y)

        np.add.outer(X_norm_sq, Y_norm_sq, out=candidate_dist_sq_all)
        candidate_dist_sq_all -= 2 * (X @ Y.T)
        np.maximum(candidate_dist_sq_all, 0.0, out=candidate_dist_sq_all)
        
        np.minimum(closest_dist_sq[:, None], candidate_dist_sq_all, out=candidate_dist_sq_all)
        potentials = candidate_dist_sq_all.T @ weights
        
        best_idx = np.argmin(potentials)
        best_candidate = candidate_ids[best_idx]
        centers[c] = X[best_candidate]
        center_indices[c] = best_candidate
        closest_dist_sq = candidate_dist_sq_all[:, best_idx].copy()

    return centers, center_indices
