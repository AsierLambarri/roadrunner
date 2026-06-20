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

import numpy as np
from scipy.spatial import KDTree

from roadrunner.physics.constants import G_KM


def compute_halo_bound_particles(
    halos: list,
    particle_coordinates: np.ndarray,
    search_factor: float = 1.0,
) -> list:
    positions = particle_coordinates[:, :3]
    velocities = particle_coordinates[:, 3:6]
    tree = KDTree(positions)

    empty_indices = np.array([], dtype=np.uint64)
    empty_values = np.array([], dtype=np.float32)

    for halo in halos:
        local = np.asarray(
            tree.query_ball_point(
                halo.xcen, r=search_factor * halo.virial_radius, workers=-1
            )
        )
        if local.size == 0:
            halo.set_boundness(empty_indices, empty_values, empty_values)
            continue

        rel_pos = positions[local] - halo.xcen
        rel_vel = velocities[local] - halo.velocity
        dist = np.linalg.norm(rel_pos, axis=1)
        vel_mags = np.linalg.norm(rel_vel, axis=1)

        v_vir_sq = G_KM * halo._inner.M / halo.virial_radius * (1 + halo.redshift)
        phi = halo.potential(dist)
        v_esc = np.sqrt(2 * np.abs(phi))
        boundness = 0.5 * (v_esc**2 - vel_mags**2) / v_vir_sq
        tdyns = halo.dynamical_time(dist)

        bound = vel_mags <= v_esc
        valid = local[bound]
        if valid.size == 0:
            halo.set_boundness(empty_indices, empty_values, empty_values)
        else:
            halo.set_boundness(
                valid.astype(np.uint64),
                boundness[bound].astype(np.float32),
                tdyns[bound].astype(np.float32),
            )

    return halos
