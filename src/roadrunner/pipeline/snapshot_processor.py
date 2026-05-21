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
from roadrunner.io.hdf5_catalogue import HDF5CatalogueWriter
from roadrunner.io.hdf5_particles import HDF5ParticleWriter
from roadrunner.io.hdf5_assignment import HDF5AssignmentWriter
from roadrunner.postprocessing.tracking.birth import BirthTracker
from roadrunner.postprocessing.tracking.assembly import AssemblyTracker


class SnapshotProcessor:
    def __init__(
        self,
        assigner: ParticleAssigner,
        halo_model: str,
        accretion_id: int,
        n_los: int = 11,
        search_factor: float = 2.0,
        min_particles: int = 10,
        birth_tracker: BirthTracker | None = None,
        assembly_tracker: AssemblyTracker | None = None,
        cat_writer: HDF5CatalogueWriter | None = None,
        part_writer: HDF5ParticleWriter | None = None,
        assign_writer: HDF5AssignmentWriter | None = None,
    ):
        self.assigner = assigner
        self.halo_model = halo_model
        self.accretion_id = accretion_id
        self.n_los = n_los
        self.search_factor = search_factor
        self.min_particles = min_particles
        self.birth_tracker = birth_tracker
        self.assembly_tracker = assembly_tracker
        self.cat_writer = cat_writer
        self.part_writer = part_writer
        self.assign_writer = assign_writer

    def process(
        self,
        snap_df: pd.DataFrame,
        particle_coords: np.ndarray,
        particle_masses: np.ndarray,
        newborn_indices: np.ndarray,
        previous_resp=None,
    ) -> tuple[list[HaloModel], HaloEnsemble, object]:
        halos = []
        for _, row in snap_df.iterrows():
            h = HaloModel.from_snapshot_row(
                row, model=self.halo_model, comoving=False,
            )
            halos.append(h)

        halos = compute_halo_bound_particles(
            halos, particle_coords, search_factor=self.search_factor,
        )
        ensemble = HaloEnsemble(halos)

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
            halos, particle_coords, newborn_indices, groups,
            previous_resp=previous_resp,
        )

        result.particle_df = compute_particle_dynamical_timescales(
            result.particle_df, ensemble, groups,
            td_factor=DYN_TIME_FACTOR,
        )

        return halos, ensemble, result

    def reduce(
        self,
        snap_df: pd.DataFrame,
        snapshot_data: SnapshotData,
        ensemble: HaloEnsemble,
        assignment_result,
        satellites_map: dict[int, set[int]],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        snap_id = int(snap_df["Snapshot"].iloc[0])
        time = snapshot_data.time
        redshift = snapshot_data.redshift
        particle_coords = np.column_stack([
            snapshot_data.positions, snapshot_data.velocities,
        ])
        masses = snapshot_data.masses

        # ── Build preprocessing for galaxy properties ─────────────
        df = assignment_result.particle_df

        galaxy_particles = {}
        for sid in df["Sub_tree_id"].unique():
            if sid == -1:
                continue
            mask = df["Sub_tree_id"] == sid
            galaxy_particles[int(sid)] = df.loc[mask, "array_index"].values

        galaxy_table = snap_df[["Sub_tree_id", "host_id", "mass",
                                "distance_to_acc_id"]].copy()
        galaxy_table.set_index("Sub_tree_id", inplace=True)

        host_row = snap_df[snap_df["Sub_tree_id"] == self.accretion_id]
        if not host_row.empty:
            host_props = host_row.iloc[0]
        else:
            host_props = snap_df.iloc[0]

        galaxy_centers = {}
        for sid, params in assignment_result.fitted_parameters.items():
            mean = params.get("mean")
            if mean is not None:
                galaxy_centers[int(sid)] = np.asarray(mean)

        # ── Galaxy properties ─────────────────────────────────────
        properties = compute_galaxy_properties(
            accretion_id=self.accretion_id,
            particle_masses=masses,
            particle_coords=particle_coords,
            galaxy_particles=galaxy_particles,
            galaxy_table=galaxy_table,
            host_props=host_props,
            halo_model=self.halo_model,
            n_los=self.n_los,
            galaxy_centers=galaxy_centers if galaxy_centers else None,
        )

        # ── Build preprocessing for Riley criterion ───────────────
        bound_csc, _ = ensemble.get_particles()
        galaxy_bound = {}
        bound_sid_to_idx = {
            sid: i for i, sid in enumerate(bound_csc.column_id)
        }
        for j, gid in enumerate(assignment_result.responsibilities.column_id):
            col = bound_sid_to_idx.get(gid)
            if col is not None:
                galaxy_bound[int(gid)] = bound_csc.column_indices[col]

        dynstate = compute_riley_criterion(
            main_id=self.accretion_id,
            particle_masses=masses,
            particle_coords=particle_coords,
            galaxy_allowed=galaxy_particles,
            galaxy_bound=galaxy_bound,
            redshift=redshift,
        )

        # ── Update trackers ───────────────────────────────────────
        if self.birth_tracker is not None:
            particle_ids = df["array_index"].values
            host_ids = df["Sub_tree_id"].values
            timescales = df.get("timescale", np.full(len(df), 0.1)).values
            self.birth_tracker.update(
                t_snap=time,
                snapshot_id=snap_id,
                particle_ids=particle_ids,
                host_ids=host_ids,
                timescales=timescales,
            )

        if self.assembly_tracker is not None:
            assignment_map = {
                int(sid): set(indices.tolist())
                for sid, indices in galaxy_particles.items()
            }
            birth_map = (
                self.birth_tracker.current_birth_map()
                if self.birth_tracker is not None
                else {}
            )
            self.assembly_tracker.update(
                snapshot_id=snap_id,
                assignment_map=assignment_map,
                birth_map=birth_map,
                satellites_map=satellites_map,
            )

        # ── I/O ───────────────────────────────────────────────────
        if self.cat_writer is not None:
            self.cat_writer.write_snapshot(
                snapshot_id=snap_id, time=time,
                properties_df=properties,
                dynstate_df=dynstate,
                satellites_map=satellites_map,
            )

        if self.part_writer is not None:
            self.part_writer.write_snapshot(
                snap_id, time, redshift, snapshot_data,
            )

        if self.assign_writer is not None:
            bound_csc, _ = ensemble.get_particles()
            self.assign_writer.write_snapshot(
                snap_id, time, assignment_result, bound_csc,
            )

        return properties, dynstate
