import numpy as np
import pandas as pd
from scipy.spatial import KDTree

from roadrunner.physics.constants import (
    NN_FRACTION,
    RILEY_BOUND_THRESHOLD,
    RILEY_SVM_SLOPE,
    RILEY_SVM_INTERCEPT,
)


def _local_velocity_dispersion(pos, vel, nmin=10):
    N = pos.shape[0]
    n = int(max(NN_FRACTION * N, nmin))
    if N <= nmin + 1:
        return np.full(N, np.nan)

    wi = np.std(vel, axis=0) / np.std(pos, axis=0)
    data = np.empty((N, 6), dtype=np.float32)
    data[:, :3] = pos
    data[:, 3:] = vel / wi

    tree = KDTree(data)
    _, idx = tree.query(data, k=n, workers=-1)

    local_disp = np.empty(N, dtype=np.float64)
    for k in range(N):
        local_disp[k] = np.linalg.norm(np.std(vel[idx[k]], axis=0))

    return local_disp


def _riley_criterion_single(
    subtree_id,
    mstar,
    f_bound,
    subset_positions,
    subset_velocities,
):
    if f_bound >= RILEY_BOUND_THRESHOLD:
        sigma50 = np.linalg.norm(np.std(subset_velocities, axis=0))
        dynstate = 0
    else:
        local_disp = _local_velocity_dispersion(
            subset_positions, subset_velocities,
        )
        sigma50 = np.nanmedian(local_disp)
        svm = RILEY_SVM_SLOPE * np.log10(mstar) + RILEY_SVM_INTERCEPT
        dynstate = 2 if (sigma50 >= svm or np.isnan(sigma50)) else 1

    return {
        "Sub_tree_id": int(subtree_id),
        "mstar": mstar,
        "f_bound": f_bound,
        "sigma50": sigma50,
        "dynstate": dynstate,
    }


def compute_riley_criterion(
    main_id,
    particle_masses,
    particle_coords,
    galaxy_allowed,
    galaxy_bound,
    redshift,
):
    positions = particle_coords[:, :3] / (1 + redshift)
    velocities = particle_coords[:, 3:6]

    skip_ids = {main_id, -1}
    empty = np.array([], dtype=np.int64)

    records = []
    for sid in galaxy_allowed:
        if sid in skip_ids:
            continue

        allowed_idx = galaxy_allowed[sid]
        if allowed_idx.size == 0:
            continue

        bound_idx = galaxy_bound.get(sid, empty)
        mstar = particle_masses[allowed_idx].sum()
        if mstar > 0:
            bound_overlap = np.intersect1d(
                allowed_idx.astype(np.int64), bound_idx.astype(np.int64),
            )
            f_bound = particle_masses[bound_overlap].sum() / mstar
        else:
            f_bound = 0.0

        result = _riley_criterion_single(
            sid, mstar, f_bound,
            positions[allowed_idx], velocities[allowed_idx],
        )
        records.append(result)

    columns = [
        "Sub_tree_id", "mstar", "f_bound", "sigma50", "dynstate",
    ]
    if not records:
        return pd.DataFrame(columns=columns, dtype=np.float32)
    return pd.DataFrame.from_records(records)[columns]
