import os

from roadrunner.readers.merger_tree import MergerTreeReaderCSV
from roadrunner.readers.snapshot import SnapshotReader
from roadrunner.readers.equivalence import EquivalenceTable
from roadrunner.physics.merger_tree import MergerTreeHandlerCSV
from roadrunner.clustering.assignment.gmm import GMMAssigner
from roadrunner.pipeline.snapshot_processor import SnapshotProcessor
from roadrunner.pipeline.accretion_pipeline import AccretionPipeline
from roadrunner.pipeline.config import RunConfig
from roadrunner.io.hdf5_catalogue import HDF5CatalogueWriter
from roadrunner.io.hdf5_particles import HDF5ParticleWriter
from roadrunner.io.hdf5_assignment import HDF5AssignmentWriter
from roadrunner.io.logging import RunLogger
from roadrunner.postprocessing.tracking.birth import BirthTracker
from roadrunner.postprocessing.tracking.assembly import AssemblyTracker


def run_accretion_history(config: RunConfig | dict) -> None:
    if isinstance(config, dict):
        config = RunConfig(**config)

    # 1. Readers
    reader = MergerTreeReaderCSV(config.merger_tree_path)
    merger_handler = MergerTreeHandlerCSV(reader.dataframe)

    snapshot_reader = SnapshotReader(
        config.code, config.ptype, config.fields, config.unit_base,
    )
    equiv_table = EquivalenceTable(
        config.equivalence_path, base_dir=config.particle_data_dir,
    )

    # 2. Auto-detect accretion_id if not provided
    accretion_id = config.accretion_id
    if accretion_id is None:
        last_snap = merger_handler.snapshots[-1]
        host_df = merger_handler.select_accretion_host(
            last_snap, criterion="max_mass",
        )
        accretion_id = int(host_df["Sub_tree_id"].iloc[0])

    # 3. Assigner
    assigner = GMMAssigner(
        cov_type=config.cov_type,
        max_iter=config.max_iter,
        tol=config.tol,
        min_particles=config.min_particles,
        reg_covar=config.reg_covar,
        prior_type="",
        verbose=1,
    )

    # 4. Processor
    processor = SnapshotProcessor(
        assigner=assigner,
        halo_model=config.halo_model,
        accretion_id=accretion_id,
        n_los=config.n_los,
        search_factor=config.search_factor,
        min_particles=config.min_particles,
    )

    # 5. Trackers (optional)
    birth_tracker = BirthTracker(
        factor=config.birth_window_factor,
        enforce_initial_hosts=config.enforce_initial_hosts,
    ) if config.birth_window_factor > 0 else None
    assembly_tracker = (
        AssemblyTracker() if birth_tracker is not None else None
    )

    # 6. Writers
    cat_w = HDF5CatalogueWriter(config.output_dir)
    part_w = (
        HDF5ParticleWriter(config.output_dir, float_atol=config.float_atol)
        if config.save_particles else None
    )
    assign_w = (
        HDF5AssignmentWriter(config.output_dir, float_atol=config.float_atol)
        if config.save_assignment else None
    )

    # 7. Logger
    logger = RunLogger(os.path.join(config.output_dir, "run.log"))
    logger.write_header({
        "output_dir": config.output_dir,
        "halo_model": config.halo_model,
        "cov_type": config.cov_type,
    })

    # 8. Pipeline
    pipeline = AccretionPipeline(
        merger_handler=merger_handler,
        snapshot_reader=snapshot_reader,
        equiv_table=equiv_table,
        snapshot_processor=processor,
        cat_writer=cat_w,
        part_writer=part_w,
        assign_writer=assign_w,
        logger=logger,
        birth_tracker=birth_tracker,
        assembly_tracker=assembly_tracker,
    )

    # 9. Run
    pipeline.run(
        config.output_dir,
        start_snapshot=config.start_snapshot,
        end_snapshot=config.end_snapshot,
    )
