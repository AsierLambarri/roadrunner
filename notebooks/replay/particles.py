"""Reading a run's particle_data/snapshotNNNN.hdf5 and merger-tree rows (from lvjeans' greedy.sim_extract)."""
import numpy as np


def load_particles(path):
    """(ids i8, pos f4 scaled, vel f4 scaled, mean f8(6), scale f8(6), masses f8, redshift) of one
    particle_data snapshot. Positions and velocities are stored standardised: unscale them as
    pos / scale[:3] + mean[:3] (catalogue units), vel / scale[3:] + mean[3:] (km/s)."""
    import h5py
    with h5py.File(path, 'r') as f:
        ids = f['data/indices'][()].astype(np.int64)
        pos = f['data/positions'][()]
        vel = f['data/velocities'][()]
        mean = f['data/scaler/mean'][()].astype(np.float64)
        scale = f['data/scaler/scale'][()].astype(np.float64)
        masses = f['data/masses'][()].astype(np.float64)
        redshift = float(f['header'].attrs['redshift'])
    return ids, pos, vel, mean, scale, masses, redshift


def row_index_of_ids(pd_ids, sim_ids):
    """Row positions of sim_ids (simulation ids) within pd_ids. Identity fast path if
    array_equal; else argsort + searchsorted, asserting all are found."""
    pd_ids = np.asarray(pd_ids)
    sim_ids = np.asarray(sim_ids)
    if sim_ids.shape == pd_ids.shape and np.array_equal(sim_ids, pd_ids):
        return np.arange(pd_ids.shape[0])
    order = np.argsort(pd_ids)
    sorted_pd = pd_ids[order]
    pos = np.searchsorted(sorted_pd, sim_ids)
    pos = np.clip(pos, 0, len(sorted_pd) - 1)
    rows = order[pos]
    found = pd_ids[rows] == sim_ids
    if not np.all(found):
        raise AssertionError(
            f'{int((~found).sum())} of {sim_ids.size} sim IDs not found in particle_data')
    return rows


def xyz(row):
    """A merger-tree row's position (catalogue units)."""
    return np.array([row['position_x'], row['position_y'], row['position_z']], dtype=np.float64)


def vxyz(row):
    """A merger-tree row's velocity (km/s)."""
    return np.array([row['velocity_x'], row['velocity_y'], row['velocity_z']], dtype=np.float64)
