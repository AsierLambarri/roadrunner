"""Shrink-sphere centre (SSC) and GMM-based centre estimation.

Provides :func:`ssc_center` which iteratively shrinks a sphere around
the weighted centre of mass, and numba-accelerated helper kernels.
"""

import numpy as np
from numba import njit


@njit(cache=True)
def _weighted_com(positions, masses):
    """Weighted centre of mass of a set of particles (numba kernel).

    Parameters
    ----------
    positions : ndarray of shape (n, ndim)
    masses : ndarray of shape (n,)

    Returns
    -------
    center : ndarray of shape (ndim,)
    """
    ndim = positions.shape[1]
    center = np.zeros(ndim, dtype=positions.dtype)
    mtot = 0.0
    for i in range(positions.shape[0]):
        m = masses[i]
        mtot += m
        for j in range(ndim):
            center[j] += m * positions[i, j]
    for j in range(ndim):
        center[j] /= mtot
    return center


@njit(cache=True)
def _max_radius(positions, center):
    """Maximum Euclidean distance from *center* (numba kernel).

    Parameters
    ----------
    positions : ndarray of shape (n, ndim)
    center : ndarray of shape (ndim,)

    Returns
    -------
    rmax : float
    """
    rmax2 = 0.0
    for i in range(positions.shape[0]):
        r2 = 0.0
        for j in range(positions.shape[1]):
            dx = positions[i, j] - center[j]
            r2 += dx * dx
        if r2 > rmax2:
            rmax2 = r2
    return np.sqrt(rmax2)


@njit(fastmath=True, cache=True)
def centering_statistic_SSC_numba(
    positions, velocities, masses,
    alpha=0.9, nmin=100,
):
    """Iterative shrink-sphere centre (numba kernel).

    Repeatedly shrinks the search sphere by factor *alpha*, stopping once
    fewer than *nmin* particles remain inside it (or after the iteration
    cap).  The returned centres are the mass-weighted position and
    velocity means of the valid selection (particle count >= *nmin*)
    whose particle count is closest to *nmin*; if the very first
    selection already has fewer than *nmin* particles, the whole set is
    used, so the result always contains particles.

    Parameters
    ----------
    positions : ndarray of shape (n, 3)
    velocities : ndarray of shape (n, 3)
    masses : ndarray of shape (n,)
    alpha : float, default=0.9
        Sphere-shrinking factor.
    nmin : int, default=100
        Minimum particle count to continue.

    Returns
    -------
    center : ndarray of shape (3,)
        Position centre.
    vcenter : ndarray of shape (3,)
        Velocity centre.
    """
    n = positions.shape[0]
    ndim = positions.shape[1]

    best_mask = np.ones(n, dtype=np.bool_)
    best_npart = n

    center = _weighted_com(positions, masses)
    rsphere = _max_radius(positions, center)

    for _ in range(100):
        rsphere2 = rsphere * rsphere
        mask = np.zeros(n, dtype=np.bool_)
        npart = 0
        for i in range(n):
            r2 = 0.0
            for j in range(ndim):
                dx = positions[i, j] - center[j]
                r2 += dx * dx
            inside = r2 <= rsphere2
            mask[i] = inside
            if inside:
                npart += 1

        if npart < nmin:
            break

        if npart <= best_npart:
            best_mask[:] = mask
            best_npart = npart

        msum = 0.0
        new_center = np.zeros(ndim, dtype=positions.dtype)
        for i in range(n):
            if mask[i]:
                m = masses[i]
                msum += m
                for j in range(ndim):
                    new_center[j] += m * positions[i, j]
        for j in range(ndim):
            new_center[j] /= msum

        center[:] = new_center
        rsphere *= alpha

    center_out = np.zeros(ndim, dtype=positions.dtype)
    vcenter = np.zeros(ndim, dtype=velocities.dtype)
    msum = 0.0
    for i in range(n):
        if best_mask[i]:
            m = masses[i]
            msum += m
            for j in range(ndim):
                center_out[j] += m * positions[i, j]
                vcenter[j] += m * velocities[i, j]
    for j in range(ndim):
        center_out[j] /= msum
        vcenter[j] /= msum

    return center_out, vcenter


def ssc_center(positions, velocities, masses, alpha=0.9, nmin=100):
    """Shrink-sphere centre (python wrapper).

    Delegates to :func:`centering_statistic_SSC_numba`.  The underlying
    loop stops once the search sphere holds fewer than *nmin* particles
    (or after the iteration cap); the returned centres are the
    mass-weighted means of the valid selection closest to *nmin*.

    Parameters
    ----------
    positions : ndarray of shape (n, 3)
    velocities : ndarray of shape (n, 3)
    masses : ndarray of shape (n,)
    alpha : float, default=0.9
    nmin : int, default=100

    Returns
    -------
    center : ndarray of shape (3,)
    vcenter : ndarray of shape (3,)
    """
    return centering_statistic_SSC_numba(positions, velocities, masses, alpha, nmin)
