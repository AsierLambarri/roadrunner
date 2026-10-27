"""Coordinate and ID translation between simulation and array space."""

from __future__ import annotations

import numpy as np

from roadrunner._mcf_types import AssignmentResult, SnapshotData
from roadrunner._defaults import GALAXY_ID, LOCAL_IDX, SIM_ID, UNBOUND, math_dtype
from roadrunner.clustering.sparse import SparseCSC
from roadrunner.physics.constants import DYN_TIME_FACTOR
from roadrunner.physics.timescales import pair_dynamical_timescales
from roadrunner.postprocessing.tracking.records import Memberships


def responsibilities_to_sim(csc, snap_data: SnapshotData):
    """Remap responsibility row IDs from array indices to simulation particle IDs.

    Parameters
    ----------
    csc : SparseCSC or None
        Responsibility matrix keyed by array indices.
    snap_data : SnapshotData
        Snapshot data providing the index-to-ID mapping.

    Returns
    -------
    remapped : SparseCSC or None
    """
    if csc is None or len(csc) == 0:
        return None
    src, dst = snap_data.index_to_id_map()
    return csc.remap_rows(src, dst)


def responsibilities_from_sim(csc, snap_data: SnapshotData):
    """Remap responsibility row IDs from simulation particle IDs to array indices.

    Parameters
    ----------
    csc : SparseCSC or None
        Responsibility matrix keyed by simulation particle IDs.
    snap_data : SnapshotData
        Snapshot data providing the ID-to-index mapping.

    Returns
    -------
    remapped : SparseCSC or None
    """
    if csc is None or len(csc) == 0:
        return None
    src, dst = snap_data.id_to_index_map()
    return csc.remap_rows(src, dst)


def update_birth_tracker(birth_tracker, snap_id: int, snap_data: SnapshotData, result: AssignmentResult, ensemble):
    """Update the birth tracker with the current snapshot assignment.

    The evidence is soft: one (particle, host) pair per non-zero
    responsibility, weighted by it and carrying the particle's dynamical
    time in that host. Particles without responsibilities (unbound) count
    fully for ``UNBOUND``, without a dynamical time. The galaxies that exist
    at this snapshot are the columns of the responsibility matrix. Without a
    responsibility matrix the hard assignment is used, one pair per particle
    with weight 1 and the particle's timescale.

    Only the particles whose birth is not settled are passed on: a finalised
    particle can no longer change, and the vast majority of a snapshot is
    finalised.

    Parameters
    ----------
    birth_tracker : BirthTracker
    snap_id : int
    snap_data : SnapshotData
    result : AssignmentResult
        Assigner result containing particle assignment, responsibilities
        and timescales.
    ensemble : HaloEnsemble
        The halos the assignment was fit to, with boundness computed: the
        source of the dynamical time of each pair.

    Returns
    -------
    changes : BirthChanges
        What the update changed in the birth hosts.
    """
    todo = ~birth_tracker.settled(snap_data.index)                      # by array index
    df = result.particle_df
    arr_idx = df["array_index"].values

    resp = result.responsibilities
    if resp is None or len(resp) == 0:
        keep = todo[arr_idx]
        alive = np.unique(df["Sub_tree_id"].values)
        timescales = df["timescale"].values[keep] if "timescale" in df.columns else np.full(int(keep.sum()), 0.1)
        rows, hosts, weights = arr_idx[keep], df["Sub_tree_id"].values[keep], np.ones(int(keep.sum()))
    else:
        alive = resp.column_id
        mask = [todo[rows_c] & (w_c > 0) for rows_c, w_c in zip(resp.column_indices, resp.column_values)]
        kept = SparseCSC([rows_c[m] for rows_c, m in zip(resp.column_indices, mask)],
                         [w_c[m] for w_c, m in zip(resp.column_values, mask)], column_id=resp.column_id)
        rows = np.concatenate(kept.column_indices).astype(np.int64) if len(kept) else np.empty(0, np.int64)
        hosts = np.repeat(np.asarray(kept.column_id, dtype=GALAXY_ID), [len(c) for c in kept.column_indices])
        weights = np.concatenate(kept.column_values).astype(math_dtype(), copy=False) if len(kept) else np.empty(0, math_dtype())
        timescales = pair_dynamical_timescales(ensemble, kept, DYN_TIME_FACTOR)
        paired = np.zeros(todo.size, dtype=bool)
        paired[rows] = True
        unbound = np.flatnonzero(todo & ~paired)
        rows = np.concatenate([rows, unbound])
        hosts = np.concatenate([hosts, np.full(unbound.size, UNBOUND, dtype=GALAXY_ID)])
        weights = np.concatenate([weights, np.ones(unbound.size, dtype=weights.dtype)])
        timescales = np.concatenate([timescales, np.full(unbound.size, np.nan)])

    return birth_tracker.update(Memberships(
        t_snap=snap_data.time,
        snapshot_id=snap_id,
        particle_ids=snap_data.index[rows],
        host_ids=hosts,
        weights=weights,
        t_dyn=timescales,
        alive=alive,
    ))


def update_assembly_tracker(assembly_tracker, snap_id: int, snap_data: SnapshotData, result: AssignmentResult, satellites, changes) -> None:
    """Update the assembly tracker with current assignment and satellite data.

    Parameters
    ----------
    assembly_tracker : AssemblyTracker
    snap_id : int
    snap_data : SnapshotData
    result : AssignmentResult
    satellites : dict of {int: set of int}
        Satellite map from the merger tree.
    changes : BirthChanges
        What the birth tracker changed in this snapshot.
    """
    assignment_map = {}
    for sid, group in result.particle_df.groupby("Sub_tree_id"):
        arr_idx = group["array_index"].values
        assignment_map[int(sid)] = set(snap_data.index[arr_idx].tolist())
    assembly_tracker.update(snap_id, assignment_map, changes, satellites)


def build_reduction_input(snap_data: SnapshotData, ensemble, assembly_tracker) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Build allowed and bound particle sets for each galaxy.

    Combines the ensemble's boundness matrix with the assembly tracker's
    infall lists to produce the input dictionaries expected by the
    reduction step.

    Parameters
    ----------
    snap_data : SnapshotData
    ensemble : HaloEnsemble
    assembly_tracker : AssemblyTracker or None
        If ``None``, all bound particles are used directly.

    Returns
    -------
    galaxy_particles : dict of {int: ndarray}
        Allowed particle array indices per galaxy.
    galaxy_bound : dict of {int: ndarray}
        Bound particle array indices per galaxy.
    """
    bound_csc, _ = ensemble.get_particles()
    sid_to_col = {int(sid): i for i, sid in enumerate(bound_csc.column_id)}

    if assembly_tracker is None:
        galaxy_particles = {}
        galaxy_bound = {}
        for i, sid in enumerate(bound_csc.column_id):
            idx = bound_csc.column_indices[i]
            if len(idx) > 0:
                galaxy_particles[int(sid)] = idx.astype(LOCAL_IDX)
                galaxy_bound[int(sid)] = idx.astype(LOCAL_IDX)
        return galaxy_particles, galaxy_bound

    assembly_map = assembly_tracker.current()

    galaxy_particles = {}
    galaxy_bound = {}

    for gid, sim_set in assembly_map.items():
        col = sid_to_col.get(int(gid))
        if col is None:
            continue
        bound_idx = bound_csc.column_indices[col]
        sim_arr = np.array(list(sim_set), dtype=SIM_ID)
        allowed_idx = snap_data.array_index(sim_arr)
        allowed_idx = allowed_idx[allowed_idx >= 0]
        intersection = np.intersect1d(
            allowed_idx.astype(LOCAL_IDX), bound_idx.astype(LOCAL_IDX),
        )
        if len(allowed_idx) > 0:
            galaxy_particles[int(gid)] = allowed_idx
            galaxy_bound[int(gid)] = intersection

    return galaxy_particles, galaxy_bound