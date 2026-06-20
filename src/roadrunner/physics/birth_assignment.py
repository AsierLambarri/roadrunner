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
    if len(snapshot_times) == 0:
        return np.array([], dtype=np.int64)
    indices = np.searchsorted(snapshot_times, creation_times, side="right")
    indices = np.clip(indices, 0, len(snapshot_times) - 1)
    return snapshot_ids[indices]


def build_birth_dict(
    particle_indices: np.ndarray,
    birth_snapshot_ids: np.ndarray,
) -> dict[int, np.ndarray]:
    result: dict[int, np.ndarray] = {}
    for snap in np.unique(birth_snapshot_ids):
        mask = birth_snapshot_ids == snap
        result[int(snap)] = particle_indices[mask].astype(np.uint64)
    return result


def sanitize_birth_dict(
    birth_dict: dict[int, np.ndarray],
) -> dict[int, np.ndarray]:
    if not check_particle_uniqueness(birth_dict):
        birth_dict = remove_duplicates(birth_dict)
    return birth_dict
