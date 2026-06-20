#############################################################################
#
# package:   roadrunner.pipeline
# file:      processing.py
# brief:     <TODO>
#
# copyright: GPLv3
# author:    Asier Lambarri Martinez
# changes:   22 may 2026 - Created
#            22 may 2026 - Last edit
#
#############################################################################

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from roadrunner._mcf_types import AssignmentResult, ParticleAssigner, SnapshotData
from roadrunner.clustering.segmentation import HaloSegmenter
from roadrunner.physics.boundness import compute_halo_bound_particles
from roadrunner.physics.constants import DYN_TIME_FACTOR
from roadrunner.physics.halo_ensemble import HaloEnsemble
from roadrunner.physics.halo_model import HaloModel
from roadrunner.physics.timescales import compute_particle_dynamical_timescales


@dataclass(frozen=True)
class ProcessingConfig:
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
    particle_coords = np.column_stack([
        snap_data.positions, snap_data.velocities,
    ])

    ensemble = HaloEnsemble([
        HaloModel.from_snapshot_row(row, model=config.halo_model, comoving=config.comoving)
        for _, row in snap_df.iterrows()
    ])
    compute_halo_bound_particles(
        ensemble, particle_coords, search_factor=config.search_factor,
    )

    csc_b, _ = ensemble.get_particles()
    pop_idx = ensemble.populated_indices()
    candidates = [csc_b.column_indices[i] for i in pop_idx]

    seg = HaloSegmenter(
        ensemble.positions[pop_idx],
        ensemble.virial_radii[pop_idx],
    )
    seg.overlap_groups().prune(candidates, config.min_particles, discard=False)

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
        previous_resp=previous_resp,
    )

    result.particle_df = compute_particle_dynamical_timescales(
        result.particle_df, ensemble, groups,
        td_factor=DYN_TIME_FACTOR,
    )

    return ensemble, result