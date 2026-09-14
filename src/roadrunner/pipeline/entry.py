"""Convenience entry point tying pipeline components together."""

from __future__ import annotations

import os
import warnings

from roadrunner.readers.merger_tree import MergerTreeReaderCSV
from roadrunner.readers.snapshot import SnapshotReader
from roadrunner.readers.npz_reader import NPZSnapshotReader
from roadrunner.readers.particle_data_reader import ParticleDataSnapshotReader
from roadrunner.readers.equivalence import EquivalenceTable
from roadrunner.physics.merger_tree import MergerTreeHandlerCSV
from roadrunner.clustering.assignment.gmm import XGMMAssigner
from roadrunner.pipeline.config import RunConfig
from roadrunner.pipeline.processing import ProcessingConfig
from roadrunner.pipeline.reduction import ReductionConfig
from roadrunner.pipeline.snapshot_orchestrator import SnapshotOrchestrator
from roadrunner.pipeline.accretion_pipeline import AccretionPipeline
from roadrunner.io.hdf5_catalogue import HDF5CatalogueWriter
from roadrunner.io.hdf5_particles import HDF5ParticleWriter
from roadrunner.io.hdf5_assignment import HDF5AssignmentWriter
from roadrunner.io.logging import RunLogger
from roadrunner.postprocessing.tracking.birth import BirthTracker
from roadrunner.postprocessing.tracking.assembly import AssemblyTracker


def run_accretion_history(config: RunConfig | dict) -> None:
    """Run the full accretion history pipeline from a configuration.

    Parameters
    ----------
    config : RunConfig or dict
        Pipeline configuration.  If a dict is passed it is converted
        to a :class:`RunConfig` instance internally.
    """
    if isinstance(config, dict):
        config = RunConfig(**config)

    reader = MergerTreeReaderCSV(config.merger_tree_path)
    merger_handler = MergerTreeHandlerCSV(reader.dataframe)

    equiv_table = EquivalenceTable(
        config.equivalence_path, base_dir=config.particle_data_dir,
    )

    if config.reader_type == "yt":
        snapshot_reader = SnapshotReader(
            config.code, config.ptype, config.fields, config.unit_base,
            assign_fields=config.assign_fields,
        )
    elif config.reader_type == "npz":
        snapshot_reader = NPZSnapshotReader(
            equiv_table, base_dir=config.particle_data_dir,
            mock_sim=config.npz_mock_sim,
            assign_fields=config.assign_fields,
        )
    elif config.reader_type == "pdata":
        _required = {"index", "mass", "position", "velocity"}
        _extra_field_names = [k for k in config.fields if k not in _required]
        snapshot_reader = ParticleDataSnapshotReader(
            equiv_table, base_dir=config.particle_data_dir,
            assign_fields=config.assign_fields,
            extra_fields=_extra_field_names,
        )
    else:
        raise ValueError(f"Unknown reader_type: {config.reader_type}")

    if config.selection_snapshot is not None:
        snaps = merger_handler.snapshots
        sel = config.selection_snapshot
        if sel < 0:
            sel = snaps[sel]
        ref_path = equiv_table.snapshot_path(sel)
        selected = snapshot_reader.select_indices(
            ref_path,
            sphere=config.selection_sphere,
            bbox=config.selection_bbox,
        )
        snapshot_reader.set_particle_filter(selected)

        end = config.end_snapshot
        end_id = snaps[-1] if end is None else (snaps[end] if end < 0 else end)
        if sel < end_id:
            warnings.warn(
                f"Selection snapshot {sel} precedes the last processed "
                f"snapshot {end_id}; particles formed after it will not be "
                "tracked.")

    accretion_id = config.accretion_id
    if accretion_id is None:
        last_snap = merger_handler.snapshots[-1]
        host_df = merger_handler.select_accretion_host(
            last_snap, criterion="max_mass",
        )
        accretion_id = int(host_df["Sub_tree_id"].iloc[0])

    assigner = XGMMAssigner(
        cov_type=config.cov_type,
        max_iter=config.max_iter,
        tol=config.tol,
        min_particles=config.min_particles,
        reg_covar=config.reg_covar,
        prior_type="",
        verbose=1,
        method=config.assignment_method,
        use_bgmm_priors=config.use_bgmm_priors,
        dtype_math=config.dtype_math,
        n_svi_iters=config.svi_iters,
        batch_size=config.svi_batch_size,
    )

    processing_config = ProcessingConfig(
        halo_model=config.halo_model,
        search_factor=config.search_factor,
        min_particles=config.min_particles,
        comoving=config.comoving,
    )
    reduction_config = ReductionConfig(
        accretion_id=accretion_id,
        halo_model=config.halo_model,
        n_los=config.n_los,
        use_gmm_centers=config.use_gmm_centers,
        min_particles_structural=config.min_particles_structural,
        ssc_nmin=config.ssc_nmin,
        ssc_alpha=config.ssc_alpha,
        dynstate_snapshots=config.dynstate_snapshots,
    )

    birth_tracker = BirthTracker(
        factor=config.birth_window_factor,
        enforce_initial_hosts=config.enforce_initial_hosts,
    ) if config.birth_window_factor > 0 else None
    assembly_tracker = (
        AssemblyTracker() if birth_tracker is not None else None
    )

    orchestrator = SnapshotOrchestrator(
        processing_config=processing_config,
        reduction_config=reduction_config,
        assigner=assigner,
        birth_tracker=birth_tracker,
        assembly_tracker=assembly_tracker,
    )

    cat_w = HDF5CatalogueWriter(config.output_dir)
    part_w = (
        HDF5ParticleWriter(config.output_dir, float_atol=config.float_atol)
        if config.save_particles else None
    )
    assign_w = (
        HDF5AssignmentWriter(config.output_dir, float_atol=config.float_atol)
        if config.save_assignment else None
    )

    logger = RunLogger(os.path.join(config.output_dir, "run.log"))

    pipeline = AccretionPipeline(
        merger_handler=merger_handler,
        snapshot_reader=snapshot_reader,
        equiv_table=equiv_table,
        orchestrator=orchestrator,
        cat_writer=cat_w,
        part_writer=part_w,
        assign_writer=assign_w,
        logger=logger,
    )

    pipeline.run(
        config.output_dir,
        start_snapshot=config.start_snapshot,
        end_snapshot=config.end_snapshot,
        resume=config.resume,
    )