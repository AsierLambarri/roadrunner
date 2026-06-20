#############################################################################
#
# package:   roadrunner.pipeline
# file:      snapshot_orchestrator.py
# brief:     Snapshot-level orchestrator coordinating processing and reduction.
#
# copyright: GPLv3
# author:    Asier Lambarri Martinez
# changes:   22 may 2026 - Created
#            22 may 2026 - Last edit
#
#############################################################################

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from roadrunner._mcf_types import AssignmentResult, ParticleAssigner, SnapshotData
from roadrunner.clustering.sparse import SparseCSC
from roadrunner.physics.halo_ensemble import HaloEnsemble
from roadrunner.pipeline.processing import ProcessingConfig, process_snapshot
from roadrunner.pipeline.reduction import ReductionConfig, reduce_snapshot
from roadrunner.pipeline.translation import (
    build_reduction_input,
    detect_newborns,
    responsibilities_from_sim,
    responsibilities_to_sim,
    update_assembly_tracker,
    update_birth_tracker,
)
from roadrunner.postprocessing.tracking.assembly import AssemblyTracker
from roadrunner.postprocessing.tracking.birth import BirthTracker


@dataclass
class SnapshotResult:
    """Result container for a single snapshot's processing and reduction."""
    ensemble: HaloEnsemble
    result: AssignmentResult
    previous_resp_sim: SparseCSC | None
    properties: pd.DataFrame | None
    dynstate: pd.DataFrame | None
    satellites: dict | None = None


class SnapshotOrchestrator:
    """Coordinates per-snapshot processing and reduction.

    Manages the assigner, birth tracker, assembly tracker, and the
    pipeline stages for a single snapshot.

    Parameters
    ----------
    processing_config : ProcessingConfig
        Boundness and segmentation configuration.
    reduction_config : ReductionConfig
        Galaxy property and dynamical-state configuration.
    assigner : ParticleAssigner
        The assigner (GMM, BGMM, or SVI-BGMM).
    birth_tracker : BirthTracker or None, optional
        Tracks birth events of particles.
    assembly_tracker : AssemblyTracker or None, optional
        Tracks galaxy assembly histories.
    """

    def __init__(
        self,
        processing_config: ProcessingConfig,
        reduction_config: ReductionConfig,
        assigner: ParticleAssigner,
        birth_tracker: BirthTracker | None = None,
        assembly_tracker: AssemblyTracker | None = None,
    ):
        self.processing_config = processing_config
        self.reduction_config = reduction_config
        self.assigner = assigner
        self.birth_tracker = birth_tracker
        self.assembly_tracker = assembly_tracker

    def process(
        self,
        snap_id: int,
        snap_df: pd.DataFrame,
        snap_data: SnapshotData,
        satellites: dict,
        previous_resp_sim: SparseCSC | None = None,
        compute_dynstate: bool = True,
    ) -> SnapshotResult:
        """Run processing and reduction for a single snapshot.

        Parameters
        ----------
        snap_id : int
            Snapshot identifier.
        snap_df : DataFrame
            Merger-tree data for this snapshot.
        snap_data : SnapshotData
            Particle data.
        satellites : dict
            Satellite membership map.
        previous_resp_sim : SparseCSC or None, optional
            Responsibilities from the previous snapshot.
        compute_dynstate : bool, default=True
            If ``True``, compute the dynamical-state classification.

        Returns
        -------
        result : SnapshotResult
            Processed snapshot data including assignment, properties,
            and dynamical state.
        """
        previous_resp = responsibilities_from_sim(previous_resp_sim, snap_data)
        newborn = detect_newborns(previous_resp, snap_data.positions.shape[0])

        ensemble, result = process_snapshot(
            snap_data, snap_df, newborn, previous_resp,
            self.assigner, self.processing_config,
        )

        if self.birth_tracker is not None:
            update_birth_tracker(self.birth_tracker, snap_id, snap_data, result)

        pop_ids = set(ensemble.sub_tree_ids[ensemble.populated_indices()])
        satellites = {k: (v & pop_ids) for k, v in satellites.items()
                      if k in pop_ids and (v & pop_ids)}

        if self.assembly_tracker is not None:
            update_assembly_tracker(
                self.assembly_tracker, snap_id, snap_data, result,
                satellites, self.birth_tracker,
            )

        galaxy_particles, galaxy_bound = build_reduction_input(
            snap_data, ensemble, self.assembly_tracker,
        )

        properties, dynstate = reduce_snapshot(
            snap_data, snap_df, result,
            galaxy_particles, galaxy_bound,
            self.reduction_config,
            compute_dynstate=compute_dynstate,
        )

        previous_resp_sim_out = responsibilities_to_sim(result.responsibilities, snap_data)

        return SnapshotResult(
            ensemble=ensemble,
            result=result,
            previous_resp_sim=previous_resp_sim_out,
            properties=properties,
            dynstate=dynstate,
            satellites=satellites,
        )