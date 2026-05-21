import os
import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")

from roadrunner.pipeline.config import RunConfig
from roadrunner.pipeline.entry import run_accretion_history
from roadrunner.pipeline import AccretionPipeline
from roadrunner.clustering.assignment.gmm import GMMAssigner
from roadrunner.pipeline.snapshot_processor import SnapshotProcessor
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


def _write_mock_equivalence(path, n_snapshots=2):
    """Write a mock equivalence CSV."""
    df = pd.DataFrame({
        "snapshot": list(range(n_snapshots)),
        "snapname": ["mock"] * n_snapshots,
        "time": [13.8 - i * 0.5 for i in range(n_snapshots)],
        "redshift": [i * 0.1 for i in range(n_snapshots)],
    })
    df.to_csv(path, index=False)
    return df


class TestRunConfig:
    def test_defaults(self):
        c = RunConfig()
        assert c.halo_model == "kepler"
        assert c.cov_type == "full"
        assert c.output_dir == "./output"

    def test_from_dict(self):
        d = {"halo_model": "nfw", "cov_type": "diagonal"}
        c = RunConfig(**d)
        assert c.halo_model == "nfw"
        assert c.cov_type == "diagonal"

    def test_from_dict_extra_keys_rejected(self):
        with pytest.raises(TypeError, match="unknown_key"):
            RunConfig(**{"halo_model": "kepler", "unknown_key": 42})


class TestRunAccretionHistory:
    def test_run_with_mock_data(self, tmp_path):
        # Build minimal config targeting the mock dataset
        equiv_path = os.path.join(str(tmp_path), "equivalence.csv")
        _write_mock_equivalence(equiv_path, n_snapshots=1)

        # Need a custom snapshot reader factory — the real SnapshotReader
        # requires yt snapshots. Use a mock approach: the entry function
        # creates a real reader, but we test the wiring by calling it on
        # a dataset where the snapshot reader can use mock_data instead.
        #
        # Instead, test the wiring path by directly calling the pipeline
        # components without yt.
        from roadrunner.io.serialization import save_checkpoint

        tree = pd.read_csv(os.path.join(DATA_DIR, "merger_tree.csv"))
        tree["host_id"] = -1
        tree["distance_to_acc_id"] = 0.0

        merger_handler = MergerTreeHandlerCSV(tree.copy())
        # merger_handler needs the first snapshot data
        class MockSnapReader:
            def load(self, path):
                p = np.load(os.path.join(DATA_DIR, "particles.npz"))
                return SnapshotData(
                    indices=np.arange(p["coords"].shape[0], dtype=np.uint64),
                    masses=p["masses"],
                    positions=p["coords"][:, :3],
                    velocities=p["coords"][:, 3:6],
                    redshift=0.0, time=13.8,
                )

        assigner = GMMAssigner(
            cov_type="full", max_iter=3, tol=1e-2,
            min_particles=10, reg_covar=1e-6, prior_type="", verbose=0,
        )
        processor = SnapshotProcessor(
            assigner=assigner, halo_model="kepler", accretion_id=1, n_los=3,
        )
        cat_w = HDF5CatalogueWriter(str(tmp_path))
        part_w = HDF5ParticleWriter(str(tmp_path), float_atol=1e-4)
        assign_w = HDF5AssignmentWriter(str(tmp_path), float_atol=1e-4)

        equiv = EquivalenceTable(pd.DataFrame({
            "snapshot": [0], "snapname": ["mock"],
            "time": [13.8], "redshift": [0.0],
        }))

        import builtins
        logger = RunLogger(os.path.join(str(tmp_path), "run.log"))
        logger.write_header({"output_dir": str(tmp_path), "halo_model": "kepler", "cov_type": "full"})

        pipeline = AccretionPipeline(
            merger_handler=merger_handler,
            snapshot_reader=MockSnapReader(),
            equiv_table=equiv,
            snapshot_processor=processor,
            cat_writer=cat_w,
            part_writer=part_w,
            assign_writer=assign_w,
            logger=logger,
        )
        pipeline.run(str(tmp_path))

        cat = HDF5CatalogueReader(
            os.path.join(str(tmp_path), "catalogue.hdf5")
        )
        hdr = cat.read_header()
        assert hdr["accretion_id"] == 1
        assert cat.read_last_snapshot() is not None
