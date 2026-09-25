"""Per-snapshot processing: boundness, segmentation, and assignment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from roadrunner._mcf_types import AssignmentResult, ParticleAssigner, SnapshotData
from roadrunner.randomness import current_seed
from roadrunner.clustering.segmentation import HaloSegmenter
from roadrunner.physics.boundness import compute_halo_bound_particles
from roadrunner.physics.constants import DYN_TIME_FACTOR
from roadrunner.physics.halo_ensemble import HaloEnsemble
from roadrunner.physics.halo_model import HaloModel
from roadrunner.physics.timescales import compute_particle_dynamical_timescales


@dataclass(frozen=True)
class ProcessingConfig:
    """Configuration for per-snapshot processing.

    Parameters
    ----------
    halo_model : str, default='kepler'
        Potential model name.
    search_factor : float, default=1.0
        Virial radius multiplier for boundness search.
    min_particles : int, default=10
        Minimum particles for a group to be kept.
    comoving : bool, default=True
        Whether coordinates are in comoving kpc.
    """
    halo_model: str = "kepler"
    search_factor: float = 1.0
    min_particles: int = 10
    comoving: bool = True


def process_snapshot(
    snap_data: SnapshotData,
    snap_df: pd.DataFrame,
    newborn: np.ndarray,
    previous_resp,
    assigner: ParticleAssigner,
    config: ProcessingConfig,
) -> tuple[HaloEnsemble, AssignmentResult]:
    """Process a single snapshot through boundness, segmentation, and assignment.

    Parameters
    ----------
    snap_data : SnapshotData
        Particle data for this snapshot.
    snap_df : DataFrame
        Merger-tree data for this snapshot.
    newborn : ndarray
        Indices of newborn particles.
    previous_resp : SparseCSC or None
        Responsibilities from the previous snapshot.
    assigner : ParticleAssigner
        The assigner to use for this snapshot.
    config : ProcessingConfig
        Processing configuration.

    Returns
    -------
    ensemble : HaloEnsemble
        The ensemble with boundness computed.
    result : AssignmentResult
        The assignment result from the assigner.
    """
    particle_coords = snap_data.data

    ensemble = HaloEnsemble([
        HaloModel.from_snapshot_row(row, model=config.halo_model, comoving=config.comoving)
        for row in snap_df.to_dict("records")
    ])
    compute_halo_bound_particles(
        ensemble, particle_coords, search_factor=config.search_factor,
    )

    csc_b, _ = ensemble.get_particles()
    pop_idx = ensemble.populated_indices()
    candidates = [csc_b.column_indices[i] for i in pop_idx]
    values = [csc_b.column_values[i] for i in pop_idx]

    seg = HaloSegmenter(
        ensemble.positions[pop_idx],
        ensemble.virial_radii[pop_idx],
    )
    seg.overlap_groups().prune(candidates, config.min_particles, discard=False)

    # Resolve contested particle ownership before any group is fit (C05):
    # small halos always keep their own particles over the group they were
    # split from, and ties between two small halos go to whichever has the
    # higher boundness. HaloEnsemble.select()/__getitem__ return the same
    # HaloModel references, so mutating boundness here is what makes every
    # downstream consumer (the assigner, timescales) see corrected data.
    removals = seg.resolve_ownership(candidates, values, ensemble.sub_tree_ids[pop_idx])
    for local_h, rows in removals.items():
        halo = ensemble[pop_idx[local_h]]
        idx, ener, tdyn = halo.get_boundness()
        keep = ~np.isin(idx, rows)
        halo.set_boundness(idx[keep], ener[keep], tdyn[keep])

    groups = (
        sorted(
            [pop_idx[g] for g in seg.pruned_groups],
            key=len, reverse=True,
        )
        if seg.pruned_groups
        else []
    )

    result = assigner.assign(
        ensemble, particle_coords, newborn, groups,
        previous_resp=previous_resp, seed=current_seed("fit"),
    )

    result.particle_df = compute_particle_dynamical_timescales(
        result.particle_df, ensemble, groups,
        td_factor=DYN_TIME_FACTOR,
    )

    return ensemble, result