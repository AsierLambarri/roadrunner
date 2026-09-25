"""Per-snapshot reduction: galaxy properties and dynamical state."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from roadrunner._mcf_types import AssignmentResult, SnapshotData
from roadrunner.postprocessing.mixing import compute_riley_criterion
from roadrunner.postprocessing.properties import compute_galaxy_properties
from roadrunner.randomness import current_seed


@dataclass(frozen=True)
class ReductionConfig:
    """Configuration for per-snapshot reduction.

    Parameters
    ----------
    accretion_id : int
        ``Sub_tree_id`` of the accretion host galaxy.
    halo_model : str, default='kepler'
        Potential model name.
    n_los : int, default=15
        Number of lines of sight for property estimation.
    use_gmm_centers : bool, default=True
        Use GMM-derived centres for property computation.
    min_particles_structural : int, default=30
        Minimum particles for structural property computation.
    ssc_nmin : int, default=30
        Minimum particles for shrink-sphere centering.
    ssc_alpha : float, default=0.9
        Shrink-sphere contraction factor.
    dynstate_snapshots : int, list, str, or None
        Which snapshots get dynamical-state classification.
    """
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
    """Compute galaxy properties and dynamical state for a snapshot.

    Parameters
    ----------
    snap_data : SnapshotData
        Particle data.
    snap_df : DataFrame
        Merger-tree data for this snapshot.
    result : AssignmentResult
        Assignment result from the assigner.
    galaxy_particles : dict of int → ndarray
        Allowed particle indices per galaxy.
    galaxy_bound : dict of int → ndarray
        Bound particle indices per galaxy.
    config : ReductionConfig
        Reduction configuration.
    compute_dynstate : bool, default=True
        Whether to compute dynamical-state classification.

    Returns
    -------
    properties : DataFrame
        Galaxy properties (mass, radius, velocity dispersion, etc.).
    dynstate : DataFrame
        Dynamical-state classification (empty if not computed).
    """
    coords = np.column_stack([snap_data.position, snap_data.velocity])

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
        particle_masses=snap_data.mass,
        particle_coords=coords,
        galaxy_bound=galaxy_bound,
        galaxy_table=galaxy_table,
        host_props=host_props,
        halo_model=config.halo_model,
        n_los=config.n_los,
        galaxy_centers=galaxy_centers,
        use_gmm_centers=config.use_gmm_centers,
        min_particles_structural=config.min_particles_structural,
        ssc_nmin=max(config.ssc_nmin, config.min_particles_structural),
        ssc_alpha=config.ssc_alpha,
        seed=current_seed("los"),
    )

    dynstate = (
        compute_riley_criterion(
            main_id=config.accretion_id,
            particle_masses=snap_data.mass,
            particle_coords=coords,
            galaxy_allowed=galaxy_particles,
            galaxy_bound=galaxy_bound,
            redshift=snap_data.redshift,
        )
        if compute_dynstate else pd.DataFrame()
    )

    return properties, dynstate