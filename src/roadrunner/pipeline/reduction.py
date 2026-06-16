from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from roadrunner._mcf_types import AssignmentResult, SnapshotData
from roadrunner.postprocessing.mixing import compute_riley_criterion
from roadrunner.postprocessing.properties import compute_galaxy_properties


@dataclass(frozen=True)
class ReductionConfig:
    accretion_id: int
    halo_model: str = "kepler"
    n_los: int = 15
    use_gmm_centers: bool = True
    min_particles_structural: int = 30
    ssc_nmin: int = 30
    ssc_alpha: float = 0.9
    dynstate_snapshots: int | list[int] | str | None = field(default_factory=lambda: [-2, -1])


def reduce_snapshot(
    snap_data: SnapshotData,
    snap_df: pd.DataFrame,
    result: AssignmentResult,
    galaxy_particles: dict[int, np.ndarray],
    galaxy_bound: dict[int, np.ndarray],
    config: ReductionConfig,
    compute_dynstate: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    coords = np.column_stack([snap_data.positions, snap_data.velocities])

    galaxy_table = snap_df[["Sub_tree_id", "host_id", "mass", "distance_to_acc_id"]].copy()
    galaxy_table.set_index("Sub_tree_id", inplace=True)

    host_rows = snap_df[snap_df["Sub_tree_id"] == config.accretion_id]
    host_props = host_rows.iloc[0] if not host_rows.empty else snap_df.iloc[0]

    galaxy_centers = None
    if result.fitted_parameters:
        centers = {}
        for sid, params in result.fitted_parameters.items():
            mean = params.get("mean")
            if mean is not None:
                centers[int(sid)] = np.asarray(mean)
        if centers:
            galaxy_centers = centers

    properties = compute_galaxy_properties(
        accretion_id=config.accretion_id,
        particle_masses=snap_data.masses,
        particle_coords=coords,
        galaxy_particles=galaxy_particles,
        galaxy_table=galaxy_table,
        host_props=host_props,
        halo_model=config.halo_model,
        n_los=config.n_los,
        galaxy_centers=galaxy_centers,
        use_gmm_centers=config.use_gmm_centers,
        min_particles_structural=config.min_particles_structural,
        ssc_nmin=max(config.ssc_nmin, config.min_particles_structural),
        ssc_alpha=config.ssc_alpha,
    )

    dynstate = (
        compute_riley_criterion(
            main_id=config.accretion_id,
            particle_masses=snap_data.masses,
            particle_coords=coords,
            galaxy_allowed=galaxy_particles,
            galaxy_bound=galaxy_bound,
            redshift=snap_data.redshift,
        )
        if compute_dynstate else pd.DataFrame()
    )

    return properties, dynstate