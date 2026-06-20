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
    alpha=0.9, nmin=100, atol=1e-5,
):
    """Iterative shrink-sphere centre (numba kernel).

    Repeatedly shrinks the search sphere by factor *alpha* until the
    centre converges (delta < *atol*) or fewer than *nmin* particles
    remain.  Returns both position and velocity centres.

    Parameters
    ----------
    positions : ndarray of shape (n, 3)
    velocities : ndarray of shape (n, 3)
    masses : ndarray of shape (n,)
    alpha : float, default=0.9
        Sphere-shrinking factor.
    nmin : int, default=100
        Minimum particle count to continue.
    atol : float, default=1e-5
        Convergence tolerance on centre displacement.

    Returns
    -------
    center : ndarray of shape (3,)
        Position centre.
    vcenter : ndarray of shape (3,)
        Velocity centre.
    """
    n = positions.shape[0]
    ndim = positions.shape[1]
    atol2 = atol ** 2

    center = _weighted_com(positions, masses)
    rsphere = _max_radius(positions, center)

    last_valid_mask = np.zeros(n, dtype=np.bool_)
    have_valid_mask = False
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

        if npart == 0:
            break

        last_valid_mask[:] = mask
        have_valid_mask = True
        if npart < nmin:
            break

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

        delta2 = 0.0
        for j in range(ndim):
            dx = new_center[j] - center[j]
            delta2 += dx * dx

        center[:] = new_center
        if delta2 <= atol2:
            break

        rsphere *= alpha

    vcenter = np.zeros(ndim, dtype=velocities.dtype)
    if have_valid_mask:
        msum = 0.0
        for i in range(n):
            if last_valid_mask[i]:
                m = masses[i]
                msum += m
                for j in range(ndim):
                    vcenter[j] += m * velocities[i, j]
    else:
        msum = 0.0
        for i in range(n):
            m = masses[i]
            msum += m
            for j in range(ndim):
                vcenter[j] += m * velocities[i, j]

    for j in range(ndim):
        vcenter[j] /= msum

    return center, vcenter


def ssc_center(positions, velocities, masses, alpha=0.9, nmin=100, atol=1e-5):
    """Shrink-sphere centre (python wrapper).

    Delegates to :func:`centering_statistic_SSC_numba`.

    Parameters
    ----------
    positions : ndarray of shape (n, 3)
    velocities : ndarray of shape (n, 3)
    masses : ndarray of shape (n,)
    alpha : float, default=0.9
    nmin : int, default=100
    atol : float, default=1e-5

    Returns
    -------
    center : ndarray of shape (3,)
    vcenter : ndarray of shape (3,)
    """
    return centering_statistic_SSC_numba(positions, velocities, masses, alpha, nmin, atol)
