"""Convenience entry point tying pipeline components together."""

from __future__ import annotations

import dataclasses
import functools
import os
import warnings

import numpy as np

from roadrunner._defaults import precision
from roadrunner._exceptions import ConfigurationError
from roadrunner.cosmology import Cosmology, cosmology
from roadrunner.threads import threads
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


def _make_snapshot_reader(config, equiv_table):
    """The snapshot reader named by ``config.reader_type``."""
    if config.reader_type == "yt":
        return SnapshotReader(config.code, config.ptype, config.fields, config.unit_base,
                              assign_fields=config.assign_fields)
    if config.reader_type == "npz":
        return NPZSnapshotReader(equiv_table, base_dir=config.particle_data_dir,
                                 mock_sim=config.npz_mock_sim, assign_fields=config.assign_fields)
    _required = {"index", "mass", "position", "velocity"}
    return ParticleDataSnapshotReader(equiv_table, base_dir=config.particle_data_dir,
                                      assign_fields=config.assign_fields,
                                      extra_fields=[k for k in config.fields if k not in _required])


def _resolve_cosmology(config, reader, equiv_table):
    """The run's Cosmology: the last snapshot's parameters, completed and overridden by ``config.cosmology``.

    Parameters
    ----------
    config : RunConfig
    reader : SnapshotReader, NPZSnapshotReader, or ParticleDataSnapshotReader
    equiv_table : EquivalenceTable

    Returns
    -------
    cosmology : Cosmology

    Raises
    ------
    ConfigurationError
        If a :class:`Cosmology` parameter is in neither the snapshot nor
        ``config.cosmology``.
    """
    from_file = reader.read_cosmology(equiv_table.snapshot_path(equiv_table.max_snapshot))
    clash = {k: (from_file[k], v) for k, v in config.cosmology.items() if k in from_file and not np.isclose(from_file[k], v)}
    if clash:
        warnings.warn(f"config cosmology overrides the snapshot's: {clash} (snapshot, config)")
    params = from_file | config.cosmology
    missing = [f.name for f in dataclasses.fields(Cosmology) if f.name not in params]
    if missing:
        raise ConfigurationError(f"cosmology parameters {missing} are not in the snapshots: set them under `cosmology:` in the config")
    return Cosmology(**{f.name: params[f.name] for f in dataclasses.fields(Cosmology)})


def _with_run_scope(func):
    """Run ``func`` inside the cosmology, precision and thread scopes from its ``RunConfig``."""
    @functools.wraps(func)
    def wrapper(config, *args, **kwargs):
        cfg = config if isinstance(config, RunConfig) else RunConfig(**config)
        equiv_table = EquivalenceTable(cfg.equivalence_path, base_dir=cfg.particle_data_dir)
        run_cosmology = _resolve_cosmology(cfg, _make_snapshot_reader(cfg, equiv_table), equiv_table)
        with cosmology(run_cosmology), precision(data=cfg.data_precision, math=cfg.math_precision), threads(cfg.threads):
            return func(config, *args, **kwargs)
    return wrapper


@_with_run_scope
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

    snapshot_reader = _make_snapshot_reader(config, equiv_table)

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
        mass_weighting=config.mass_weighting,
        plausibility=config.plausibility,
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
        comoving=config.comoving,
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
        seed=config.seed,
    )