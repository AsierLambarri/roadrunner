#############################################################################
#
# package:   roadrunner.physics
# file:      boundness.py
# brief:     Gravitational boundness computation for halo particles.
#
# copyright: GPLv3
# author:    Asier Lambarri Martinez
# changes:   13 may 2026 - Created
#            13 may 2026 - Last edit
#
#############################################################################

"""Gravitational boundness computation for halo particles."""

import numpy as np
from scipy.spatial import KDTree

from roadrunner._defaults import LOCAL_IDX, math_dtype

from roadrunner.threads import tree_workers


def compute_halo_bound_particles(
    halos: list,
    particle_coordinates: np.ndarray,
) -> list:
    """Compute gravitational boundness for a list of halos.

    Uses a KD-tree to efficiently find particles within each halo's search
    sphere (:attr:`HaloModel.search_factor` virial radii), then evaluates
    boundness via the total specific orbital energy ``E = Φ + ½v²``. The
    stored boundness is ``-E / HaloModel.binding_energy_scale()``.
    Bound particles are stored on each halo via
    :meth:`HaloModel.set_boundness`.

    For Keplerian potentials the dynamical time is computed from the
    orbital semi-major axis derived from the orbital energy, giving
    the correct timescale for elliptical orbits.  For NFW potentials
    the instantaneous radius is used.

    Parameters
    ----------
    halos : list of HaloModel
        Halos to evaluate.
    particle_coordinates : ndarray of shape (n_particles, 6)
        6-D phase-space coordinates.

    Returns
    -------
    halos : list of HaloModel
        The same list with boundness data set on each halo.
    """
    positions = particle_coordinates[:, :3]
    velocities = particle_coordinates[:, 3:6]
    tree = KDTree(positions)

    empty_indices = np.array([], dtype=LOCAL_IDX)
    empty_values = np.array([], dtype=math_dtype())

    for halo in halos:
        local = np.asarray(
            tree.query_ball_point(
                halo.xcen, r=halo.search_factor * halo.virial_radius, workers=tree_workers()
            )
        )
        if local.size == 0:
            halo.set_boundness(empty_indices, empty_values, empty_values)
            continue

        rel_pos = positions[local] - halo.xcen
        rel_vel = velocities[local] - halo.velocity
        dist = np.linalg.norm(rel_pos, axis=1)

        E = halo.compute_energy(rel_pos, rel_vel, relative=True)
        # -E over the halo's energy scale (physical virial radius, C08).
        boundness = -E / halo.binding_energy_scale()

        # Kepler: the period of the orbit with energy E (bound particles only, I03);
        # otherwise the dynamical time at the instantaneous radius.
        tdyns = halo.orbital_time(E, dist).astype(math_dtype(), copy=False)

        bound = E < 0
        valid = local[bound]
        if valid.size == 0:
            halo.set_boundness(empty_indices, empty_values, empty_values)
        else:
            # Sorted by particle index (C06): query_ball_point's own
            # traversal order is unspecified, and everything downstream
            # that consumes boundness by position (rather than by ID,
            # like SparseCSC.to_dense or a set intersection) implicitly
            # relies on a stable, predictable order. Sorting here once
            # establishes that as a real invariant instead of leaving it
            # to chance.
            order = np.argsort(valid)
            halo.set_boundness(
                valid[order].astype(LOCAL_IDX),
                boundness[bound][order].astype(math_dtype(), copy=False),
                tdyns[bound][order],
            )

    return halos
