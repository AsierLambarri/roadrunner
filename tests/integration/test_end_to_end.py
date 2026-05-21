import os
import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")

from roadrunner._mcf_types import SnapshotData
from roadrunner.clustering.assignment.gmm import GMMAssigner
from roadrunner.pipeline.snapshot_processor import SnapshotProcessor
from roadrunner.pipeline.accretion_pipeline import AccretionPipeline
from roadrunner.pipeline.config import RunConfig
from roadrunner.physics.merger_tree import MergerTreeHandlerCSV
from roadrunner.io.hdf5_catalogue import HDF5CatalogueWriter
from roadrunner.io.hdf5_particles import HDF5ParticleWriter
from roadrunner.io.hdf5_assignment import HDF5AssignmentWriter
from roadrunner.io.hdf5_reader import HDF5CatalogueReader
from roadrunner.io.logging import RunLogger
from roadrunner.readers.equivalence import EquivalenceTable

DATA_DIR = "test_data/mock_snap_tight"


def _build_mock_data(tmp_path, n_snapshots=2):
    """Duplicate mock data across snapshots, write CSVs."""
    tree = pd.read_csv(os.path.join(DATA_DIR, "merger_tree.csv"))
    frames = []
    for i in range(n_snapshots):
        dup = tree.copy()
        dup["Snapshot"] = i
        frames.append(dup)
    tree = pd.concat(frames, ignore_index=True)
    if "host_id" not in tree.columns:
        tree["host_id"] = -1
    if "distance_to_acc_id" not in tree.columns:
        tree["distance_to_acc_id"] = 0.0

    # Write merger tree CSV
    mt_path = os.path.join(str(tmp_path), "merger_tree.csv")
    tree.to_csv(mt_path, index=False)

    # Write equivalence CSV
    eq_path = os.path.join(str(tmp_path), "equivalence.csv")
    pd.DataFrame({
        "snapshot": list(range(n_snapshots)),
        "snapname": ["mock"] * n_snapshots,
        "time": [13.8 - i * 0.5 for i in range(n_snapshots)],
        "redshift": [i * 0.1 for i in range(n_snapshots)],
    }).to_csv(eq_path, index=False)

    return tree, mt_path, eq_path


def _make_mock_reader(data_dir=DATA_DIR):
    class MockSnapReader:
        def load(self, path):
            p = np.load(os.path.join(data_dir, "particles.npz"))
            return SnapshotData(
                indices=np.arange(p["coords"].shape[0], dtype=np.uint64),
                masses=p["masses"],
                positions=p["coords"][:, :3],
                velocities=p["coords"][:, 3:6],
                redshift=0.0, time=13.8,
            )
    return MockSnapReader()


class TestEndToEnd:
    def test_full_pipeline_cycle(self, tmp_path):
        tree, mt_path, eq_path = _build_mock_data(tmp_path, n_snapshots=2)

        merger_handler = MergerTreeHandlerCSV(tree.copy())
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
        logger = RunLogger(os.path.join(str(tmp_path), "run.log"))

        equiv = EquivalenceTable(pd.DataFrame({
            "snapshot": [0, 1],
            "snapname": ["mock", "mock"],
            "time": [13.8, 13.0],
            "redshift": [0.0, 0.1],
        }))

        from roadrunner.postprocessing.tracking.birth import BirthTracker
        from roadrunner.postprocessing.tracking.assembly import AssemblyTracker
        bt = BirthTracker(factor=5)
        at = AssemblyTracker(n_sat_history=2)

        pipeline = AccretionPipeline(
            merger_handler=merger_handler,
            snapshot_reader=_make_mock_reader(),
            equiv_table=equiv,
            snapshot_processor=processor,
            cat_writer=cat_w,
            part_writer=part_w,
            assign_writer=assign_w,
            logger=logger,
            birth_tracker=bt,
            assembly_tracker=at,
        )

        # Run
        pipeline.run(str(tmp_path))

        # ── Verify output files exist ──────────────────────────
        cat_path = os.path.join(str(tmp_path), "catalogue.hdf5")
        part_path = os.path.join(str(tmp_path), "particles.hdf5")
        assign_path = os.path.join(str(tmp_path), "assignment.hdf5")
        assert os.path.exists(cat_path), "catalogue.hdf5 missing"
        assert os.path.exists(part_path), "particles.hdf5 missing"
        assert os.path.exists(assign_path), "assignment.hdf5 missing"

        # ── Read and verify catalogue ──────────────────────────
        cat = HDF5CatalogueReader(cat_path)
        hdr = cat.read_header()
        assert hdr["accretion_id"] == 1
        assert len(hdr["snapshots"]) == 2
        assert cat.read_last_snapshot() == 1

        # Galaxy properties for snapshot 0
        props = cat.read_galaxy_properties(snapshot_id=0)
        assert len(props) > 0
        assert "rh" in props.columns
        assert "sigma" in props.columns
        assert props["rh"].notna().all()
        assert props["sigma"].notna().all()

        # Riley criterion for snapshot 0
        dyn = cat.read_riley_criterion(snapshot_id=0)
        assert len(dyn) > 0
        assert "dynstate" in dyn.columns

        # Births and assembly (populated by trackers)
        births = cat.read_births()
        assert len(births) > 0
        assert "birth_id" in births.columns
        assembly = cat.read_assembly()
        assert len(assembly) > 0
        assert "galaxy_id" in assembly.columns

        # ── Verify particles file via inspection ───────────────
        import h5py
        with h5py.File(part_path, "r") as hf:
            assert "snapshots/0" in hf
            assert "snapshots/1" in hf
            grp0 = hf["snapshots/0"]
            assert grp0["indices"].shape[0] > 0
            assert grp0["positions"].shape[1] == 3
            assert grp0["velocities"].shape[1] == 3
            assert "scaler" in grp0
            assert "mean" in grp0["scaler"]
            assert "scale" in grp0["scaler"]

        # ── Verify assignment file ─────────────────────────────
        with h5py.File(assign_path, "r") as hf:
            assert "snapshots/0" in hf
            assert "snapshots/1" in hf
            # Check per-galaxy structure
            gals = list(hf["snapshots/0"]["galaxies"].keys())
            assert len(gals) > 0
            g1 = hf["snapshots/0"]["galaxies"][gals[0]]
            assert "indices" in g1
            assert "log_resp" in g1
            assert "boundness" in g1
            assert "mean" in g1
            assert "weight" in g1
            assert "covariance" in g1
            # Hard assignment
            assert "hard_assignment" in hf["snapshots/0"]

        # ── Verify properties are consistent ───────────────────
        # All galaxy IDs in properties should match those in hard assignment
        with h5py.File(assign_path, "r") as hf:
            hard_assign = hf["snapshots/0/hard_assignment"][()]
        # Properties should have Sub_tree_id that exist in hard assignment
        assign_ids = set(hard_assign["Sub_tree_id"])
        assert set(props["Sub_tree_id"]).issubset(assign_ids)

    def test_trackers_output(self, tmp_path):
        """Test with birth+assembly trackers enabled."""
        tree, _, _ = _build_mock_data(tmp_path, n_snapshots=2)
        merger_handler = MergerTreeHandlerCSV(tree.copy())
        assigner = GMMAssigner(
            cov_type="full", max_iter=3, tol=1e-2,
            min_particles=10, reg_covar=1e-6, prior_type="", verbose=0,
        )
        processor = SnapshotProcessor(
            assigner=assigner, halo_model="kepler", accretion_id=1, n_los=3,
        )
        cat_w = HDF5CatalogueWriter(str(tmp_path))
        logger = RunLogger(os.path.join(str(tmp_path), "run.log"))

        from roadrunner.postprocessing.tracking.birth import BirthTracker
        from roadrunner.postprocessing.tracking.assembly import AssemblyTracker
        bt = BirthTracker(factor=5)
        at = AssemblyTracker(n_sat_history=2)

        equiv = EquivalenceTable(pd.DataFrame({
            "snapshot": [0, 1], "snapname": ["mock", "mock"],
            "time": [13.8, 13.0], "redshift": [0.0, 0.1],
        }))

        pipeline = AccretionPipeline(
            merger_handler=merger_handler,
            snapshot_reader=_make_mock_reader(),
            equiv_table=equiv,
            snapshot_processor=processor,
            cat_writer=cat_w,
            part_writer=None,
            assign_writer=None,
            logger=logger,
            birth_tracker=bt,
            assembly_tracker=at,
        )
        pipeline.run(str(tmp_path))

        cat = HDF5CatalogueReader(
            os.path.join(str(tmp_path), "catalogue.hdf5")
        )
        births = cat.read_births()
        assert len(births) > 0
        assembly = cat.read_assembly()
        assert len(assembly) > 0
