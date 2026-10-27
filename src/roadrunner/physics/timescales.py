#############################################################################
#
# package:   roadrunner.physics
# file:      timescales.py
# brief:     Dynamical time-scales of bound particles and the tidal radius.
#
# copyright: GPLv3
# author:    Asier Lambarri Martinez
# changes:   15 may 2026 - Created
#            15 may 2026 - Last edit
#
#############################################################################

"""Dynamical time-scales of bound particles and the tidal radius."""

import numpy as np

from roadrunner.mixture._math import row_squared_norms


def pair_dynamical_timescales(ensemble, responsibilities, td_factor=1.0):
    """Find the dynamical time of every entry of a responsibility matrix.

    An entry is a (particle, galaxy) pair. Its time is the one the boundness
    stored for the particle in that galaxy.

    Parameters
    ----------
    ensemble : HaloEnsemble
        The halos the responsibilities were fit to, with boundness computed.
    responsibilities : SparseCSC
        Responsibility matrix keyed by array index; its columns are halos of the
        ensemble.
    td_factor : float, default=1.0
        Multiplicative factor applied to the raw dynamical time.

    Returns
    -------
    dynamical_times : ndarray of float64, shape (n_entries,)
        One value per entry, in the order of
        ``np.concatenate(responsibilities.column_indices)``; ``NaN`` for an
        entry the halo holds no bound particle for.
    """
    _, stored_times = ensemble.get_particles()
    column_of_galaxy = {
        int(galaxy_id): column
        for column, galaxy_id in enumerate(stored_times.column_id)
    }
    times_per_column = []
    entries = zip(responsibilities.column_id, responsibilities.column_indices)
    for galaxy_id, rows in entries:
        column = column_of_galaxy[int(galaxy_id)]
        # Sorted by particle index (C06).
        bound_rows = stored_times.column_indices[column]
        # A trailing NaN and -1 are what a row that is not bound lands on.
        bound_times = np.append(
            stored_times.column_values[column].astype(np.float64) * td_factor, np.nan
        )
        position = np.searchsorted(bound_rows, rows)
        is_bound = np.append(bound_rows, -1)[position] == rows
        times_per_column.append(np.where(is_bound, bound_times[position], np.nan))
    if not times_per_column:
        return np.empty(0)
    return np.concatenate(times_per_column)


def compute_tidal_radius(potential, satellite_mass, xyz):
    """Compute the tidal radius of a satellite in a host potential.

    ``r_t = d * (M_sat / M_denom)^(1/3)``, with ``d`` the satellite's distance
    from the potential's centre and ``M_denom`` the potential's tidal
    denominator there: 0 at ``d = 0`` (the main host itself), where an
    extended host's denominator vanishes.

    Parameters
    ----------
    potential : PotentialModel
        The host's potential.
    satellite_mass : float
        Mass of the satellite.
    xyz : ndarray of shape (1, 3)
        Absolute physical position of the satellite.

    Returns
    -------
    rt : float
        Tidal radius (physical).
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    distance = np.sqrt(row_squared_norms(xyz - potential.centre))[0]
    denom = potential.tidal_denominator(xyz)[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        rt = distance * (satellite_mass / denom) ** (1.0 / 3.0)
    return np.nan_to_num(rt, nan=0.0, posinf=np.inf)
