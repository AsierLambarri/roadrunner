import numpy as np
import pandas as pd

from roadrunner._mcf_types import ParticleAssigner, SnapshotData
from roadrunner.physics.boundness import compute_halo_bound_particles
from roadrunner.physics.halo_model import HaloModel
from roadrunner.physics.halo_ensemble import HaloEnsemble
from roadrunner.physics.timescales import compute_particle_dynamical_timescales
from roadrunner.physics.constants import DYN_TIME_FACTOR
from roadrunner.clustering.segmentation import HaloSegmenter
from roadrunner.postprocessing.properties import compute_galaxy_properties
from roadrunner.postprocessing.mixing import compute_riley_criterion


class SnapshotProcessor:
    def __init__(
        self,
        assigner: ParticleAssigner,
        halo_model: str,
        accretion_id: int,
        n_los: int = 11,
        search_factor: float = 2.0,
        min_particles: int = 10,
    ):
        self.assigner = assigner
        self.halo_model = halo_model
        self.accretion_id = accretion_id
        self.n_los = n_los
        self.search_factor = search_factor
        self.min_particles = min_particles

    def process(
        self,
        snap_df: pd.DataFrame,
        snapshot_data: SnapshotData,
        newborn_indices: np.ndarray,
        previous_resp=None,
    ):
        particle_coords = np.column_stack([
            snapshot_data.positions, snapshot_data.velocities,
        ])

        ensemble = HaloEnsemble([
            HaloModel.from_snapshot_row(row, model=self.halo_model, comoving=False)
            for _, row in snap_df.iterrows()
        ])
        compute_halo_bound_particles(
            ensemble, particle_coords, search_factor=self.search_factor,
        )

        csc_b, _ = ensemble.get_particles()
        pop_idx = ensemble.populated_indices()
        candidates = [csc_b.column_indices[i] for i in pop_idx]

        seg = HaloSegmenter(
            ensemble.positions[pop_idx],
            ensemble.virial_radii[pop_idx],
        )
        seg.overlap_groups().prune(candidates, self.min_particles, discard=False)

        groups = sorted(
            [pop_idx[g] for g in seg.pruned_groups],
            key=len, reverse=True,
        ) if seg.pruned_groups else []

        result = self.assigner.assign(
            ensemble, particle_coords, newborn_indices, groups,
            previous_resp=previous_resp,
        )

        result.particle_df = compute_particle_dynamical_timescales(
            result.particle_df, ensemble, groups,
            td_factor=DYN_TIME_FACTOR,
        )

        return ensemble, result
