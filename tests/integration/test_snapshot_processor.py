import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from roadrunner.pipeline.snapshot_processor import SnapshotProcessor
from roadrunner.clustering.assignment.gmm import GMMAssigner
from roadrunner._mcf_types import SnapshotData


DATA_DIR = "test_data/mock_snap_tight"


def _load_mock():
    tree = pd.read_csv(os.path.join(DATA_DIR, "merger_tree.csv"))
    particles = np.load(os.path.join(DATA_DIR, "particles.npz"))
    coords = particles["coords"]
    masses = particles["masses"]
    snap_data = SnapshotData(
        indices=np.arange(coords.shape[0], dtype=np.uint64),
        masses=masses,
        positions=coords[:, :3],
        velocities=coords[:, 3:6],
        redshift=0.0,
        time=13.8,
    )
    # Add columns needed by reduce() if missing
    if "host_id" not in tree.columns:
        tree["host_id"] = -1
    if "distance_to_acc_id" not in tree.columns:
        tree["distance_to_acc_id"] = 0.0
    return tree, coords, masses, snap_data


class TestSnapshotProcessor:
    def test_process_returns_assignment(self):
        tree, coords, masses, snap_data = _load_mock()
        assigner = GMMAssigner(
            cov_type="full", max_iter=5, tol=1e-2,
            min_particles=10, reg_covar=1e-6, prior_type="", verbose=0,
        )
        sp = SnapshotProcessor(
            assigner=assigner,
            halo_model="kepler",
            accretion_id=1,
        )
        newborn = np.arange(coords.shape[0], dtype=np.uint64)
        halos, ensemble, result = sp.process(
            tree, coords, masses, newborn,
        )
        assert len(halos) == len(tree)
        assert result.particle_df is not None
        assert "Sub_tree_id" in result.particle_df.columns
        assert "timescale" in result.particle_df.columns
        assert (result.particle_df["Sub_tree_id"] != -1).all()

    def test_reduce_returns_properties_and_dynstate(self):
        tree, coords, masses, snap_data = _load_mock()
        assigner = GMMAssigner(
            cov_type="full", max_iter=5, tol=1e-2,
            min_particles=10, reg_covar=1e-6, prior_type="", verbose=0,
        )
        sp = SnapshotProcessor(
            assigner=assigner,
            halo_model="kepler",
            accretion_id=1,
            n_los=3,
        )
        newborn = np.arange(coords.shape[0], dtype=np.uint64)
        halos, ensemble, result = sp.process(tree, coords, masses, newborn)
        properties, dynstate = sp.reduce(
            tree, snap_data, ensemble, result, {},
        )
        assert len(properties) > 0
        assert "rh" in properties.columns
        assert "sigma" in properties.columns
        assert len(dynstate) > 0
        assert "dynstate" in dynstate.columns

    def test_process_twice_with_previous_resp(self):
        tree, coords, masses, snap_data = _load_mock()
        assigner = GMMAssigner(
            cov_type="full", max_iter=3, tol=1e-2,
            min_particles=10, reg_covar=1e-6, prior_type="", verbose=0,
        )
        sp = SnapshotProcessor(
            assigner=assigner,
            halo_model="kepler",
            accretion_id=1,
        )
        newborn = np.arange(coords.shape[0], dtype=np.uint64)

        # First snapshot
        halos, ensemble, r1 = sp.process(tree, coords, masses, newborn)

        # r1.responsibilities is already a SparseCSC — use directly
        # (No conversion needed; remap_rows would be used for cross-snapshot
        #  translation in the real pipeline.)
        prev_csc = r1.responsibilities

        # Second snapshot (same data — uses sim_id space directly as array_index)
        halos, ensemble, r2 = sp.process(
            tree, coords, masses, newborn, previous_resp=prev_csc,
        )
        assert len(r2.particle_df) == len(r1.particle_df)
        assert r2.statistics.get("unassigned", 0) == 0

    def test_reduce_with_trackers(self):
        tree, coords, masses, snap_data = _load_mock()
        assigner = GMMAssigner(
            cov_type="full", max_iter=3, tol=1e-2,
            min_particles=10, reg_covar=1e-6, prior_type="", verbose=0,
        )
        from roadrunner.postprocessing.tracking.birth import BirthTracker
        from roadrunner.postprocessing.tracking.assembly import AssemblyTracker

        bt = BirthTracker(factor=5)
        at = AssemblyTracker(n_sat_history=2)
        sp = SnapshotProcessor(
            assigner=assigner,
            halo_model="kepler",
            accretion_id=1,
            n_los=3,
            birth_tracker=bt,
            assembly_tracker=at,
        )
        newborn = np.arange(coords.shape[0], dtype=np.uint64)
        halos, ensemble, result = sp.process(tree, coords, masses, newborn)
        properties, dynstate = sp.reduce(
            tree, snap_data, ensemble, result, {},
        )
        # After reduce, trackers should have state
        assert len(bt._active) > 0 or len(bt._finalized) > 0
        # Assembly should have galaxies
        assert len(at.current()) > 0

    def test_full_cycle_with_io(self, tmp_path):
        tree, coords, masses, snap_data = _load_mock()
        assigner = GMMAssigner(
            cov_type="full", max_iter=3, tol=1e-2,
            min_particles=10, reg_covar=1e-6, prior_type="", verbose=0,
        )
        from roadrunner.io.hdf5_catalogue import HDF5CatalogueWriter
        from roadrunner.io.hdf5_particles import HDF5ParticleWriter
        from roadrunner.io.hdf5_assignment import HDF5AssignmentWriter
        from roadrunner.io.hdf5_reader import HDF5CatalogueReader

        cat_w = HDF5CatalogueWriter(str(tmp_path), mode="w-")
        part_w = HDF5ParticleWriter(str(tmp_path), mode="w-", float_atol=1e-4)
        assign_w = HDF5AssignmentWriter(str(tmp_path), mode="w-", float_atol=1e-4)

        sp = SnapshotProcessor(
            assigner=assigner,
            halo_model="kepler",
            accretion_id=1,
            n_los=3,
            cat_writer=cat_w,
            part_writer=part_w,
            assign_writer=assign_w,
        )

        # Write header
        cat_w.write_header(
            accretion_id=1,
            snapshots=[0],
            config_dict={"halo_model": "kepler"},
            merger_tree_df=tree,
            equivalence_df=pd.DataFrame(
                {"snapshot": [0], "time": [13.8], "redshift": [0.0],
                 "snapname": ["placeholder"]}
            ),
        )

        newborn = np.arange(coords.shape[0], dtype=np.uint64)
        halos, ensemble, result = sp.process(tree, coords, masses, newborn)
        properties, dynstate = sp.reduce(
            tree, snap_data, ensemble, result, {},
        )

        # Read back and verify
        reader = HDF5CatalogueReader(
            os.path.join(str(tmp_path), "catalogue.hdf5")
        )
        hdr = reader.read_header()
        assert hdr["accretion_id"] == 1
        assert "galaxy_properties" in hf_list(
            str(tmp_path), "catalogue.hdf5"
        ) or reader.read_galaxy_properties(snapshot_id=0) is not None


def hf_list(dir_path, fname):
    """List HDF5 groups/datasets for diagnostic purposes."""
    import h5py
    keys = []
    with h5py.File(os.path.join(dir_path, fname), "r") as hf:
        hf.visit(keys.append)
    return keys
