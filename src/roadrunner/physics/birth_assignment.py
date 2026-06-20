#############################################################################
#
# package:   roadrunner.physics
# file:      birth_assignment.py
# brief:     Birth-tag assignment for star particles.
#
# copyright: GPLv3
# author:    Asier Lambarri Martinez
# changes:   13 may 2026 - Created
#            13 may 2026 - Last edit
#
#############################################################################

import numpy as np

from roadrunner.helpers import check_particle_uniqueness, remove_duplicates


def assign_birth_snapshots(
    creation_times: np.ndarray,
    snapshot_times: np.ndarray,
    snapshot_ids: np.ndarray,
) -> np.ndarray:
    """Assign each particle to its birth snapshot based on creation time.

    Uses ``np.searchsorted`` to find the snapshot whose time is closest
    to each particle's creation time.

    Parameters
    ----------
    creation_times : ndarray of float
        Per-particle cosmic time of formation.
    snapshot_times : ndarray of float
        Cosmic time of each snapshot.
    snapshot_ids : ndarray of int
        Snapshot IDs corresponding to ``snapshot_times``.

    Returns
    -------
    birth_snapshots : ndarray of int64
        Birth snapshot ID for each particle.
    """
    if len(snapshot_times) == 0:
        return np.array([], dtype=np.int64)
    indices = np.searchsorted(snapshot_times, creation_times, side="right")
    indices = np.clip(indices, 0, len(snapshot_times) - 1)
    return snapshot_ids[indices]


def build_birth_dict(
    particle_indices: np.ndarray,
    birth_snapshot_ids: np.ndarray,
) -> dict[int, np.ndarray]:
    """Build a birth dictionary mapping snapshot IDs to arrays of particle indices.

    Parameters
    ----------
    particle_indices : ndarray of int64
        All particle indices.
    birth_snapshot_ids : ndarray of int64
        Birth snapshot ID for each particle.

    Returns
    -------
    birth_dict : dict of int → ndarray
        Keys are snapshot IDs, values are arrays of particle indices
        created in that snapshot.
    """
    result: dict[int, np.ndarray] = {}
    for snap in np.unique(birth_snapshot_ids):
        mask = birth_snapshot_ids == snap
        result[int(snap)] = particle_indices[mask].astype(np.uint64)
    return result


def sanitize_birth_dict(
    birth_dict: dict[int, np.ndarray],
) -> dict[int, np.ndarray]:
    """Ensure all particles in the birth dictionary are unique.

    Parameters
    ----------
    birth_dict : dict of int → ndarray
        Raw birth dictionary.

    Returns
    -------
    sanitized : dict of int → ndarray
        Birth dictionary with duplicate particles removed.
    """
    if not check_particle_uniqueness(birth_dict):
        birth_dict = remove_duplicates(birth_dict)
    return birth_dict
