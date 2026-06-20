#############################################################################
#
# package:   roadrunner.physics
# file:      timescales.py
# brief:     <TODO>
#
# copyright: GPLv3
# author:    Asier Lambarri Martinez
# changes:   15 may 2026 - Created
#            15 may 2026 - Last edit
#
#############################################################################

import numpy as np

from roadrunner.physics.constants import MAX_DYN_TIMESCALE


def compute_particle_dynamical_timescales(particle_df, ensemble, groups, td_factor=1.0):
    particle_df["timescale"] = 0.0

    for group in groups:
        sub = ensemble.select(group)
        _, csc_t = sub.get_particles()

        tdyn_dense = csc_t.to_dense()

        max_tdyn = np.minimum(np.nanmax(tdyn_dense, axis=1) * td_factor, MAX_DYN_TIMESCALE)
        gp_idx = csc_t.row_id
        particle_df.loc[gp_idx, "timescale"] = max_tdyn

    return particle_df


def compute_tidal_radius(halo, satellite_mass, distance):
    denom = np.asarray(halo.tidal_denominator(np.array([distance]))).flat[0]
    return distance * (satellite_mass / denom) ** (1.0 / 3.0)
