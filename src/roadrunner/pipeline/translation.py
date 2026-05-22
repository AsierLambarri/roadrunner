from __future__ import annotations

import numpy as np

from roadrunner._mcf_types import SnapshotData


def responsibilities_to_sim(csc, snap_data: SnapshotData):
    if csc is None or len(csc) == 0:
        return None
    src, dst = snap_data.index_to_id_map()
    return csc.remap_rows(src, dst)


def responsibilities_from_sim(csc, snap_data: SnapshotData):
    if csc is None or len(csc) == 0:
        return None
    src, dst = snap_data.id_to_index_map()
    return csc.remap_rows(src, dst)


def detect_newborns(previous_resp, n_particles: int) -> np.ndarray:
    if previous_resp is None:
        return np.arange(n_particles, dtype=np.uint64)
    existing = set(previous_resp.row_id.tolist())
    return np.array(
        [i for i in range(n_particles) if i not in existing],
        dtype=np.uint64,
    )