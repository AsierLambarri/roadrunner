#############################################################################
#
# package:   roadrunner.mixture
# file:      kmeans.py
# brief:     Weighted, mask-aware K-means (Lloyd) for GMM initialisation.
#
# Provides :class:`WeightedKMeans`, a K-means implementation mirroring
# ``sklearn.cluster.KMeans`` that additionally supports per-point
# sample weights and a per-point, per-cluster mask (``cluster_weights``)
# restricting which clusters a point may join.  Used by
# :class:`roadrunner.mixture.base.BaseMixture` for the k-means /
# k-means++ hard-label initialisation step.
#
# copyright: GPLv3
# author:    Asier Lambarri Martinez
# changes:   23 Sep 2026 - Created
#            23 Sep 2026 - Rewrote around a single fused numba Lloyd kernel
#
#############################################################################

"""Weighted, mask-aware K-means (Lloyd) for GMM initialisation.

:class:`WeightedKMeans` mirrors the ``sklearn.cluster.KMeans`` API
(:meth:`fit`, :meth:`predict`, :meth:`fit_predict`,
``cluster_centers_``, ``labels_``, ``inertia_``, ``n_iter_``), while
adding per-point ``point_weights`` and a ``cluster_weights`` mask:
point ``i`` may only be assigned to cluster ``k`` when
``cluster_weights[i, k] > 0``.

Each Lloyd iteration is a single fused, parallel numba pass
(:func:`_lloyd_step`) that assigns points to their nearest allowed
centre and accumulates the weighted centre sums in the same loop,
avoiding N x D masked-distance temporaries and per-feature bincount
passes.
"""

import numpy as np
from numba import njit, prange, get_num_threads

from ._kmeans_plusplus import kmeans_plusplus_prior
from ._math import row_squared_norms
from roadrunner._defaults import LOCAL_IDX


def _check_finite(X):
    """Raise a clear error for non-2-D or non-finite X.

    The numba kernel has no bounds checking, so NaN/inf in X would
    otherwise silently corrupt labels/centres instead of raising.
    """
    if X.ndim != 2:
        raise ValueError(f"Expected a 2D array for X, got {X.ndim}D.")
    if not np.all(np.isfinite(X)):
        raise ValueError("Input X contains NaN or infinity.")


def _check_cluster_weights_shape(cluster_weights, n_samples, n_clusters):
    """Raise a clear error for a mis-shaped cluster_weights mask.

    The numba kernel indexes ``allowed`` with no bounds checking, so a
    wrong number of columns/rows would otherwise be read out of bounds
    (silent wrong labels) instead of raising.
    """
    if cluster_weights.shape != (n_samples, n_clusters):
        raise ValueError(
            f"cluster_weights must have shape ({n_samples}, {n_clusters}), "
            f"got {cluster_weights.shape}"
        )


@njit(parallel=True, cache=True)
def _lloyd_step(X, centers, allowed, point_weights, n_chunks):
    """Numba kernel: masked nearest-centre assignment plus weighted centre sums.

    Returns labels, squared distance to the assigned centre, per-cluster
    weighted coordinate sums and per-cluster weight totals.
    """
    n_samples, n_features = X.shape
    n_clusters = centers.shape[0]
    labels = np.empty(n_samples, dtype=np.int64)
    min_d2 = np.empty(n_samples, dtype=X.dtype)
    step = (n_samples + n_chunks - 1) // n_chunks
    sums = np.zeros((n_chunks, n_clusters, n_features), dtype=X.dtype)
    weights = np.zeros((n_chunks, n_clusters), dtype=X.dtype)
    for c in prange(n_chunks):
        acc = np.zeros((n_clusters, n_features), dtype=X.dtype)   # thread-local: no false sharing
        wacc = np.zeros(n_clusters, dtype=X.dtype)
        for i in range(c * step, min(n_samples, (c + 1) * step)):
            best = -1
            best_d2 = np.inf
            for k in range(n_clusters):
                if allowed[i, k]:
                    d2 = 0.0
                    for j in range(n_features):
                        t = X[i, j] - centers[k, j]
                        d2 += t * t
                    if d2 < best_d2:          # strict: first minimum, like np.argmin
                        best_d2 = d2
                        best = k
            labels[i] = best
            min_d2[i] = best_d2
            wi = point_weights[i]
            wacc[best] += wi
            for j in range(n_features):
                acc[best, j] += wi * X[i, j]
        sums[c] = acc
        weights[c] = wacc
    return labels, min_d2, sums.sum(axis=0), weights.sum(axis=0)


class WeightedKMeans:
    """K-means (Lloyd) with per-point weights and per-point cluster masks.

    Mirrors the ``sklearn.cluster.KMeans`` API. Point ``i`` may only join
    clusters with ``cluster_weights[i, k] > 0``; centres are the
    ``point_weights``-weighted means of their members.

    Parameters
    ----------
    n_clusters : int, default=8
    init : 'k-means++' or ndarray of shape (n_clusters, n_features), default='k-means++'
        Initial centres, or prior-aware k-means++ seeding (:func:`kmeans_plusplus_prior`).
    max_iter : int, default=300
    tol : float, default=1e-4
        Tolerance on the squared centre shift, relative to the mean feature variance.
    random_state : int or RandomState, optional

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
    labels_ : ndarray of shape (n_samples,)
    inertia_ : float
        Weighted sum of squared distances to the assigned centres.
    n_iter_ : int
    """

    def __init__(self, n_clusters=8, init="k-means++", max_iter=300, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X, point_weights=None, cluster_weights=None):
        """Fit the model, running Lloyd's algorithm to convergence.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        point_weights : ndarray of shape (n_samples,), optional
            Per-point weights (defaults to uniform).
        cluster_weights : ndarray of shape (n_samples, n_clusters), optional
            Per-point, per-cluster mask; ``cluster_weights[i, k] <= 0``
            forbids point ``i`` from joining cluster ``k``.  ``None``
            means every point may join every cluster.

        Returns
        -------
        self : WeightedKMeans
        """
        X = np.ascontiguousarray(X)
        if not np.issubdtype(X.dtype, np.floating):
            X = X.astype(np.float64)
        _check_finite(X)
        n_samples, n_features = X.shape
        n_clusters = self.n_clusters

        if point_weights is None:
            w = np.ones(n_samples, dtype=X.dtype)
        else:
            w = np.asarray(point_weights, dtype=X.dtype)
            if w.shape != (n_samples,):
                raise ValueError(
                    f"point_weights must have shape ({n_samples},), got {w.shape}"
                )
            if not np.all(np.isfinite(w)):
                raise ValueError("point_weights must be finite")
            if np.any(w < 0):
                raise ValueError("point_weights must be non-negative")
            if not np.any(w > 0):
                raise ValueError("point_weights must contain at least one positive value")

        if cluster_weights is None:
            allowed = np.ones((n_samples, n_clusters), dtype=bool)
            weighted_prior = np.ones((n_samples, n_clusters), dtype=X.dtype) * w[:, None]
        else:
            cluster_weights = np.asarray(cluster_weights, dtype=X.dtype)
            _check_cluster_weights_shape(cluster_weights, n_samples, n_clusters)
            if not np.all(np.isfinite(cluster_weights)):
                raise ValueError("cluster_weights must be finite")
            allowed = np.ascontiguousarray(cluster_weights > 0)
            weighted_prior = cluster_weights * w[:, None]

        if np.any(~allowed.any(axis=1)):
            raise ValueError("each row of cluster_weights must allow at least one cluster")

        if isinstance(self.init, str):
            if self.init != "k-means++":
                raise ValueError(f"init must be 'k-means++' or an ndarray, got {self.init!r}.")
            centers, _ = kmeans_plusplus_prior(
                X, n_clusters, cluster_weights=weighted_prior, random_state=self.random_state
            )
            centers = np.asarray(centers, dtype=X.dtype)
        else:
            centers = np.array(self.init, dtype=X.dtype, copy=True)
            if centers.shape != (n_clusters, n_features):
                raise ValueError(
                    f"The shape of the initial centers {centers.shape} does not match "
                    f"(n_clusters, n_features) = ({n_clusters}, {n_features})."
                )

        tol_abs = self.tol * X.var(axis=0).mean()
        n_chunks = max(1, min(get_num_threads(), n_samples))

        labels_old = np.full(n_samples, -1, dtype=LOCAL_IDX)
        strict_convergence = False
        labels = min_d2 = None

        n_iter = 0
        for it in range(self.max_iter):
            labels, min_d2, sums, weight_in_clusters = _lloyd_step(X, centers, allowed, w, n_chunks)

            # Relocate clusters left empty by this assignment, as sklearn does:
            # each takes the allowed point farthest from its current centre.
            empty = np.flatnonzero(weight_in_clusters == 0)
            if empty.size:
                used = np.zeros(n_samples, dtype=bool)
                for k in empty:
                    candidates = allowed[:, k] & ~used
                    if not np.any(candidates):
                        continue  # no allowed point at all: cluster keeps its centre
                    far_idx = np.argmax(np.where(candidates, min_d2, -np.inf))
                    old_k, wi = labels[far_idx], w[far_idx]
                    sums[old_k] -= wi * X[far_idx]
                    weight_in_clusters[old_k] -= wi
                    sums[k] = wi * X[far_idx]
                    weight_in_clusters[k] = wi
                    used[far_idx] = True

            nonempty = weight_in_clusters > 0
            centers_new = centers.copy()
            centers_new[nonempty] = sums[nonempty] / weight_in_clusters[nonempty, None]

            shift = row_squared_norms(centers_new - centers).sum()
            n_iter = it + 1
            labels = labels.astype(LOCAL_IDX)

            if np.array_equal(labels, labels_old):
                strict_convergence = True
                centers = centers_new
                break

            centers = centers_new
            labels_old = labels
            if shift <= tol_abs:
                break

        if not strict_convergence:
            # Rerun the E-step so the reported labels match the final centres.
            labels, min_d2, _, _ = _lloyd_step(X, centers, allowed, w, n_chunks)
            labels = labels.astype(LOCAL_IDX)

        self.cluster_centers_ = centers
        self.labels_ = labels
        self.inertia_ = float(w @ min_d2)
        self.n_iter_ = n_iter

        return self

    def predict(self, X, cluster_weights=None):
        """Assign each point to its nearest allowed centre.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        cluster_weights : ndarray of shape (n_samples, n_clusters), optional
            Per-point, per-cluster mask (see :meth:`fit`).

        Returns
        -------
        labels : ndarray of shape (n_samples,)
        """
        X = np.ascontiguousarray(X, dtype=self.cluster_centers_.dtype)
        _check_finite(X)
        n_samples, n_features = X.shape

        n_clusters_fit, n_features_fit = self.cluster_centers_.shape
        if n_features != n_features_fit:
            raise ValueError(
                f"X has {n_features} features, but WeightedKMeans was fitted with "
                f"{n_features_fit} features."
            )

        if cluster_weights is None:
            allowed = np.ones((n_samples, n_clusters_fit), dtype=bool)
        else:
            cluster_weights = np.asarray(cluster_weights, dtype=X.dtype)
            _check_cluster_weights_shape(cluster_weights, n_samples, n_clusters_fit)
            allowed = np.ascontiguousarray(cluster_weights > 0)
            if np.any(~allowed.any(axis=1)):
                raise ValueError("each row of cluster_weights must allow at least one cluster")

        centers = np.ascontiguousarray(self.cluster_centers_)
        w = np.ones(n_samples, dtype=X.dtype)
        n_chunks = max(1, min(get_num_threads(), n_samples))

        labels, _, _, _ = _lloyd_step(X, centers, allowed, w, n_chunks)
        return labels.astype(LOCAL_IDX)

    def fit_predict(self, X, point_weights=None, cluster_weights=None):
        """Fit the model and return the resulting labels.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        point_weights : ndarray of shape (n_samples,), optional
        cluster_weights : ndarray of shape (n_samples, n_clusters), optional

        Returns
        -------
        labels : ndarray of shape (n_samples,)
        """
        return self.fit(X, point_weights=point_weights, cluster_weights=cluster_weights).labels_
