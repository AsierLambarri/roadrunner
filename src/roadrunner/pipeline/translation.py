"""Coordinate and ID translation between simulation and array space."""

from __future__ import annotations

import numpy as np

from roadrunner._mcf_types import AssignmentResult, SnapshotData


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


def detect_newborns(previous_resp, n_particles: int) -> np.ndarray:
    """Detect newborn particles not present in the previous snapshot.

    Parameters
    ----------
    previous_resp : SparseCSC or None
        Responsibility matrix from the previous snapshot.
    n_particles : int
        Total number of particles in the current snapshot.

    Returns
    -------
    newborns : ndarray of uint64
        Array indices of particles that are new.
    """
    if previous_resp is None:
        return np.arange(n_particles, dtype=np.uint64)
    existing = set(previous_resp.row_id.tolist())
    return np.array(
        [i for i in range(n_particles) if i not in existing],
        dtype=np.uint64,
    )


def update_birth_tracker(birth_tracker, snap_id: int, snap_data: SnapshotData, result: AssignmentResult) -> None:
    """Update the birth tracker with the current snapshot assignment.

    Parameters
    ----------
    birth_tracker : BirthTracker
    snap_id : int
    snap_data : SnapshotData
    result : AssignmentResult
        Assigner result containing particle assignment and timescales.
    """
    df = result.particle_df
    arr_idx = df["array_index"].values
    sim_ids = snap_data.indices[arr_idx]
    if "timescale" in df.columns:
        timescales = df["timescale"].values
    else:
        timescales = np.full(len(df), 0.1)
    birth_tracker.update(
        t_snap=snap_data.time,
        snapshot_id=snap_id,
        particle_ids=sim_ids,
        host_ids=df["Sub_tree_id"].values,
        timescales=timescales,
    )


def update_assembly_tracker(assembly_tracker, snap_id: int, snap_data: SnapshotData, result: AssignmentResult, satellites, birth_tracker=None) -> None:
    """Update the assembly tracker with current assignment and satellite data.

    Parameters
    ----------
    assembly_tracker : AssemblyTracker
    snap_id : int
    snap_data : SnapshotData
    result : AssignmentResult
    satellites : dict of {int: set of int}
        Satellite map from the merger tree.
    birth_tracker : BirthTracker or None, optional
        If provided, uses its current birth map.
    """
    assignment_map = {}
    for sid, group in result.particle_df.groupby("Sub_tree_id"):
        arr_idx = group["array_index"].values
        assignment_map[int(sid)] = set(snap_data.indices[arr_idx].tolist())
    birth_map = birth_tracker.current_birth_map() if birth_tracker is not None else {}
    assembly_tracker.update(snap_id, assignment_map, birth_map, satellites)


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
                galaxy_particles[int(sid)] = idx.astype(np.int64)
                galaxy_bound[int(sid)] = idx.astype(np.int64)
        return galaxy_particles, galaxy_bound

    assembly_map = assembly_tracker.current()

    galaxy_particles = {}
    galaxy_bound = {}

    for gid, sim_set in assembly_map.items():
        col = sid_to_col.get(int(gid))
        if col is None:
            continue
        bound_idx = bound_csc.column_indices[col]
        sim_arr = np.array(list(sim_set), dtype=np.uint64)
        allowed_idx = snap_data.array_index(sim_arr)
        allowed_idx = allowed_idx[allowed_idx >= 0]
        intersection = np.intersect1d(
            allowed_idx.astype(np.int64), bound_idx.astype(np.int64),
        )
        if len(allowed_idx) > 0:
            galaxy_particles[int(gid)] = allowed_idx
            galaxy_bound[int(gid)] = intersection

    return galaxy_particles, galaxy_bound