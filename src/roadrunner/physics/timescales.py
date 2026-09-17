#############################################################################
#
# package:   roadrunner.physics
# file:      timescales.py
# brief:     Dynamical time-scale computation for bound particles.
#
# copyright: GPLv3
# author:    Asier Lambarri Martinez
# changes:   15 may 2026 - Created
#            15 may 2026 - Last edit
#
#############################################################################

"""Dynamical time-scale computation for bound particles."""

import numpy as np

from roadrunner.physics.constants import MAX_DYN_TIMESCALE
from roadrunner._defaults import math_dtype


def compute_particle_dynamical_timescales(particle_df, ensemble, groups, td_factor=1.0):
    """Compute dynamical timescales for bound particles and add them to the DataFrame.

    For each overlapping group, the per-particle dynamical time is
    taken from the boundness computation, multiplied by
    ``td_factor``, and capped at ``MAX_DYN_TIMESCALE``.

    Parameters
    ----------
    particle_df : DataFrame
        Particle-level data with an ``array_index`` index.
    ensemble : HaloEnsemble
        The full halo ensemble with boundness computed.
    groups : list of list of int
        Overlapping group definitions.
    td_factor : float, default=1.0
        Multiplicative factor applied to the raw dynamical time.

    Returns
    -------
    particle_df : DataFrame
        The same DataFrame with a ``timescale`` column added.
    """
    particle_df["timescale"] = np.zeros(len(particle_df), dtype=math_dtype())

    for group in groups:
        sub = ensemble.select(group)
        _, csc_t = sub.get_particles()

        tdyn_dense = csc_t.to_dense()

        max_tdyn = np.minimum(np.nanmax(tdyn_dense, axis=1) * td_factor, MAX_DYN_TIMESCALE)
        gp_idx = csc_t.row_id
        particle_df.loc[gp_idx, "timescale"] = max_tdyn

    return particle_df


def compute_tidal_radius(halo, satellite_mass, distance):
    """Compute the tidal radius of a satellite in a host halo potential.

    Uses the tidal denominator from :meth:`HaloModel.tidal_denominator`
    and the formula ``r_t = d * (M_sat / M_denom)^(1/3)``.

    Parameters
    ----------
    halo : HaloModel
        The host halo.
    satellite_mass : float
        Mass of the satellite.
    distance : float
        Distance from the satellite to the host centre.

    Returns
    -------
    rt : float
        Tidal radius in the same units as ``distance``.
    """
    denom = np.asarray(halo.tidal_denominator(np.array([distance]))).flat[0]
    return distance * (satellite_mass / denom) ** (1.0 / 3.0)
