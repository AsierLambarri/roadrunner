import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from roadrunner.pipeline.accretion_pipeline import AccretionPipeline
from roadrunner.pipeline.snapshot_processor import SnapshotProcessor
from roadrunner.clustering.assignment.gmm import GMMAssigner
from roadrunner.readers.merger_tree import MergerTreeReaderCSV
from roadrunner.readers.snapshot import SnapshotReader as RawSnapshotReader
from roadrunner.readers.equivalence import EquivalenceTable
from roadrunner.physics.merger_tree import MergerTreeHandlerCSV
from roadrunner.io.hdf5_catalogue import HDF5CatalogueWriter
from roadrunner.io.hdf5_particles import HDF5ParticleWriter
from roadrunner.io.hdf5_assignment import HDF5AssignmentWriter
from roadrunner.io.hdf5_reader import HDF5CatalogueReader
from roadrunner.io.logging import RunLogger
from roadrunner._mcf_types import SnapshotData

DATA_DIR = "test_data/mock_snap_tight"


def _build_mock_pipeline(tmp_path, cov_type="full", n_duplicate=2):
    """Build an AccretionPipeline wired with mock data.

    Duplicates the merger tree across multiple snapshots so the loop
    has more than one snapshot to process.
    """
    tree = pd.read_csv(os.path.join(DATA_DIR, "merger_tree.csv"))
    frames = []
    for i in range(n_duplicate):
        dup = tree.copy()
        dup["Snapshot"] = i
        frames.append(dup)
    tree = pd.concat(frames, ignore_index=True)

    if "host_id" not in tree.columns:
        tree["host_id"] = -1
    if "distance_to_acc_id" not in tree.columns:
        tree["distance_to_acc_id"] = 0.0

    merger_handler = MergerTreeHandlerCSV(tree.copy())
    assigner = GMMAssigner(
        cov_type=cov_type, max_iter=3, tol=1e-2,
        min_particles=10, reg_covar=1e-6, prior_type="", verbose=0,
    )
    processor = SnapshotProcessor(
        assigner=assigner,
        halo_model="kepler",
        accretion_id=1,
        n_los=3,
    )
    cat_w = HDF5CatalogueWriter(str(tmp_path))
    part_w = HDF5ParticleWriter(str(tmp_path), float_atol=1e-4)
    assign_w = HDF5AssignmentWriter(str(tmp_path), float_atol=1e-4)
    logger = RunLogger(os.path.join(str(tmp_path), "run.log"))

    # Snapshot reader that returns mock data from npz
    particles = np.load(os.path.join(DATA_DIR, "particles.npz"))
    coords = particles["coords"]
    masses = particles["masses"]
    N = coords.shape[0]

    class MockSnapshotReader:
        def load(self, path):
            return SnapshotData(
                indices=np.arange(N, dtype=np.uint64),
                masses=masses,
                positions=coords[:, :3],
                velocities=coords[:, 3:6],
                redshift=0.0,
                time=13.8,
            )

    equiv = EquivalenceTable(
        pd.DataFrame({
            "snapshot": list(range(n_duplicate)),
            "snapname": ["mock"] * n_duplicate,
            "time": [13.8 - i * 0.5 for i in range(n_duplicate)],
            "redshift": [i * 0.1 for i in range(n_duplicate)],
        })
    )

    mock_reader = MockSnapshotReader()
    pipeline = AccretionPipeline(
        merger_handler=merger_handler,
        snapshot_reader=mock_reader,
        equiv_table=equiv,
        snapshot_processor=processor,
        cat_writer=cat_w,
        part_writer=part_w,
        assign_writer=assign_w,
        logger=logger,
    )
    return pipeline, merger_handler, coords, masses, mock_reader


class TestAccretionPipeline:
    def test_full_run_two_snapshots(self, tmp_path):
        pipeline, _, _, _, _ = _build_mock_pipeline(tmp_path, n_duplicate=2)
        pipeline.run(str(tmp_path))
        cat = HDF5CatalogueReader(
            os.path.join(str(tmp_path), "catalogue.hdf5")
        )
        hdr = cat.read_header()
        assert len(hdr["snapshots"]) > 0
        assert cat.read_last_snapshot() is not None

    def test_previous_resp_translated(self, tmp_path):
        pipeline, merger_handler, coords, _, _ = _build_mock_pipeline(tmp_path)
        snap_df = merger_handler.select_snapshots(0)
        particles = np.load(os.path.join(DATA_DIR, "particles.npz"))
        snap_data = SnapshotData(
            indices=np.arange(coords.shape[0], dtype=np.uint64),
            masses=particles["masses"],
            positions=coords[:, :3],
            velocities=coords[:, 3:6],
            redshift=0.0, time=13.8,
        )
        newborn = np.arange(coords.shape[0], dtype=np.uint64)
        ensemble, result = pipeline.processor.process(
            snap_df, snap_data, newborn,
        )
        csc_sim = pipeline._to_sim_space(result.responsibilities, snap_data)
        assert csc_sim is not None
        assert len(csc_sim) > 0
        csc_back = pipeline._from_sim_space(csc_sim, snap_data)
        assert csc_back is not None
        assert csc_back.row_id.size == result.responsibilities.row_id.size

    def test_checkpoint_restart(self, tmp_path):
        pipeline, merger_handler, coords, masses, mock_reader = (
            _build_mock_pipeline(tmp_path, n_duplicate=2)
        )

        call_count = [0]

        def failing_load(path):
            call_count[0] += 1
            if call_count[0] == 2:
                raise RuntimeError("Simulated crash on snapshot 1")
            return SnapshotData(
                indices=np.arange(coords.shape[0], dtype=np.uint64),
                masses=masses,
                positions=coords[:, :3],
                velocities=coords[:, 3:6],
                redshift=0.0, time=13.8,
            )

        pipeline.snapshot_reader.load = failing_load

        try:
            pipeline.run(str(tmp_path))
            assert False, "Should have raised"
        except RuntimeError as e:
            assert "Simulated crash" in str(e)

        resume_pipeline, _, _, _, _ = _build_mock_pipeline(
            tmp_path, n_duplicate=2,
        )
        resume_pipeline.run(str(tmp_path), resume=True)

        cat_path = os.path.join(str(tmp_path), "catalogue.hdf5")
        if os.path.exists(cat_path):
            cat = HDF5CatalogueReader(cat_path)
            last = cat.read_last_snapshot()
            assert last is not None

    def test_newborn_detection(self, tmp_path):
        pipeline, merger_handler, coords, masses, _ = _build_mock_pipeline(tmp_path)
        snap_df = merger_handler.select_snapshots(0)
        particles = np.load(os.path.join(DATA_DIR, "particles.npz"))
        snap_data = SnapshotData(
            indices=np.arange(coords.shape[0], dtype=np.uint64),
            masses=particles["masses"],
            positions=coords[:, :3],
            velocities=coords[:, 3:6],
            redshift=0.0, time=13.8,
        )
        newborn = np.arange(coords.shape[0], dtype=np.uint64)
        ensemble, result = pipeline.processor.process(
            snap_df, snap_data, newborn,
        )
        csc_sim = pipeline._to_sim_space(result.responsibilities, snap_data)

        # Simulate second snapshot with new particles
        N = coords.shape[0]
        # Only use first half of particles as existing — rest are newborn
        from roadrunner.clustering.sparse import SparseCSC
        half = N // 2
        indices_existing = np.arange(half, dtype=np.uint64)
        mock_csc = SparseCSC(
            [indices_existing],
            [np.ones(half, dtype=np.float32)],
            column_id=np.array([1], dtype=np.int64),
        )

        # Convert mock_csc to sim IDs using snap_data, then back
        csc_sim_mock = pipeline._to_sim_space(mock_csc, snap_data)
        new_snap_data = SnapshotData(
            indices=np.arange(N, dtype=np.uint64),
            masses=np.ones(N),
            positions=np.zeros((N, 3)),
            velocities=np.zeros((N, 3)),
            redshift=0.1, time=13.0,
        )
        csc_back = pipeline._from_sim_space(csc_sim_mock, new_snap_data)
        # Newborn should be the second half
        existing_set = set(csc_back.row_id) if csc_back is not None else set()
        newborn = np.array(
            [i for i in range(N) if i not in existing_set], dtype=np.uint64,
        )
        assert len(newborn) == N - half
