#############################################################################
#
# package:   roadrunner.postprocessing
# file:      centering.py
# brief:     <TODO>
#
# copyright: GPLv3
# author:    Asier Lambarri Martinez
# changes:   16 jun 2026 - Created
#            16 jun 2026 - Last edit
#
#############################################################################

import numpy as np
from numba import njit


@njit(cache=True)
def _weighted_com(positions, masses):
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
    return centering_statistic_SSC_numba(positions, velocities, masses, alpha, nmin, atol)
