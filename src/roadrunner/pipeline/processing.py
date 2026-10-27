"""Per-snapshot processing: boundness, segmentation, and assignment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from roadrunner._mcf_types import AssignmentResult, ParticleAssigner, SnapshotData
from roadrunner.randomness import current_seed
from roadrunner.clustering.segmentation import HaloSegmenter
from roadrunner.physics.boundness import compute_halo_bound_particles
from roadrunner.physics.halo_ensemble import HaloEnsemble


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
    stars : str, default='free'
        Stars' own potential in the boundness: "none", "pinned" or "free".
    """
    halo_model: str = "kepler"
    search_factor: float = 1.0
    min_particles: int = 10
    comoving: bool = True
    stars: str = "free"            # stars' own potential in the boundness: "none", "pinned" or "free"

    def __post_init__(self):
        if self.stars not in ("none", "pinned", "free"):
            raise ValueError(f"stars must be 'none', 'pinned' or 'free', got '{self.stars}'")


def bound_ensemble(snap_data, snap_df, config) -> tuple[HaloEnsemble, list]:
    """Halos of a snapshot with boundness, ownership resolved, and their overlap groups (largest first).

    The assigner's input: :func:`process_snapshot` calls it before ``assign``.

    Parameters
    ----------
    snap_data : SnapshotData
        Particle data for this snapshot.
    snap_df : DataFrame
        Merger-tree data for this snapshot.
    config : ProcessingConfig

    Returns
    -------
    ensemble : HaloEnsemble
        Halos with their bound sets after the ownership resolution.
    groups : list of list of int
        Overlap groups of populated halos (ensemble indices), largest first.
    """
    ensemble = HaloEnsemble(compute_halo_bound_particles(snap_df.to_dict("records"), snap_data, config))

    # Ownership compares the boundness b = -E / scale across halos: the depth in each halo's own well.
    csc_b, _ = ensemble.get_particles(scaled_energies=True)
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
    # downstream consumer (the assigner) see corrected data.
    removals = seg.resolve_ownership(candidates, values, ensemble.sub_tree_ids[pop_idx])
    for local_h, rows in removals.items():
        halo = ensemble[pop_idx[local_h]]
        idx, ener, tdyn = halo.get_bound()
        keep = ~np.isin(idx, rows)
        halo.set_bound(idx[keep], ener[keep], tdyn[keep])

    # A halo can lose every candidate to the ownership resolution above;
    # it is no longer populated, so it leaves its group (and a group left
    # without halos is dropped) instead of reaching the assigner empty.
    emptied = {int(pop_idx[h]) for h in removals
               if ensemble[pop_idx[h]].get_bound()[0].size == 0}
    groups = sorted(
        (g for g in ([i for i in pop_idx[grp] if int(i) not in emptied]
                     for grp in seg.pruned_groups) if g),
        key=len, reverse=True,
    )

    return ensemble, groups


def process_snapshot(
    snap_data: SnapshotData,
    snap_df: pd.DataFrame,
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
    previous_resp : SparseCSC or None
        Responsibilities from the previous snapshot, keyed by array
        index (not simulation particle ID).
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
    ensemble, groups = bound_ensemble(snap_data, snap_df, config)

    result = assigner.assign(
        ensemble, particle_coords, groups,
        previous_resp=previous_resp, seed=current_seed("fit"),
        particle_masses=snap_data.mass,
    )

    return ensemble, result