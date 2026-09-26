import os

import h5py
import numpy as np
import pandas as pd
import pytest

from roadrunner.io.hdf5_catalogue import HDF5CatalogueWriter
from roadrunner.io.hdf5_reader import HDF5CatalogueReader


class TestHDF5CatalogueWriter:
    @pytest.fixture
    def tmp_dir(self, tmp_path):
        return str(tmp_path / "catalogue_test")

    def test_write_and_read_header(self, tmp_dir):
        os.makedirs(tmp_dir, exist_ok=True)
        w = HDF5CatalogueWriter(tmp_dir)
        merger = pd.DataFrame({"Sub_tree_id": [1], "mass": [1e10]})
        equiv = pd.DataFrame({"snapshot": [0], "time": [13.8]})
        w.write_header(
            accretion_id=1,
            snapshots=[0, 1, 2],
            config_dict={"halo_model": "kepler"},
            merger_tree_df=merger,
            equivalence_df=equiv,
        )
        r = HDF5CatalogueReader(os.path.join(tmp_dir, "catalogue.hdf5"))
        hdr = r.read_header()
        assert hdr["accretion_id"] == 1
        assert hdr["snapshots"] == [0, 1, 2]
        assert hdr["config"]["halo_model"] == "kepler"
        assert hdr["last_snapshot"] == -1

    def test_write_snapshot_properties(self, tmp_dir):
        os.makedirs(tmp_dir, exist_ok=True)
        w = HDF5CatalogueWriter(tmp_dir)
        merger = pd.DataFrame({"Sub_tree_id": [1], "mass": [1e10]})
        equiv = pd.DataFrame({"snapshot": [0], "time": [13.8]})
        w.write_header(1, [0], {}, merger, equiv)
        props = pd.DataFrame({"Sub_tree_id": [1], "rh": [3.0]})
        dyn = pd.DataFrame({"Sub_tree_id": [1], "dynstate": [0]})
        w.write_snapshot(
            snapshot_id=0, time=13.8,
            properties_df=props, dynstate_df=dyn,
            satellites_map={1: {2, 3}},
        )
        r = HDF5CatalogueReader(os.path.join(tmp_dir, "catalogue.hdf5"))
        assert r.read_last_snapshot() == 0
        p = r.read_galaxy_properties(snapshot_id=0)
        assert len(p) == 1
        assert p["rh"].iloc[0] == 3.0
        d = r.read_riley_criterion(snapshot_id=0)
        assert d["dynstate"].iloc[0] == 0

    def test_finalize(self, tmp_dir):
        os.makedirs(tmp_dir, exist_ok=True)
        w = HDF5CatalogueWriter(tmp_dir)
        merger = pd.DataFrame({"Sub_tree_id": [1], "mass": [1e10]})
        equiv = pd.DataFrame({"snapshot": [0], "time": [13.8]})
        w.write_header(1, [0], {}, merger, equiv)
        births = pd.DataFrame({"particle_index": [1, 2], "birth_id": [10, 20]})
        assembly = pd.DataFrame({"particle_index": [1, 3], "galaxy_id": [10, 30]})
        w.write_finalize(births, assembly)
        r = HDF5CatalogueReader(os.path.join(tmp_dir, "catalogue.hdf5"))
        b = r.read_births()
        assert len(b) == 2
        a = r.read_assembly()
        assert len(a) == 2

    def test_empty_finalize(self, tmp_dir):
        os.makedirs(tmp_dir, exist_ok=True)
        w = HDF5CatalogueWriter(tmp_dir)
        merger = pd.DataFrame({"Sub_tree_id": [1], "mass": [1e10]})
        equiv = pd.DataFrame({"snapshot": [0], "time": [13.8]})
        w.write_header(1, [0], {}, merger, equiv)
        w.write_finalize(pd.DataFrame(), pd.DataFrame())
        r = HDF5CatalogueReader(os.path.join(tmp_dir, "catalogue.hdf5"))
        assert r.read_births().empty
        assert r.read_assembly().empty

    def test_write_snapshot_retryable(self, tmp_dir):
        """C03: a second write_snapshot() for the same snapshot_id (the
        resume-after-crash scenario) must not crash on h5py's "name
        already exists", and must fully replace the first attempt's
        content rather than leaving stale entries behind."""
        os.makedirs(tmp_dir, exist_ok=True)
        w = HDF5CatalogueWriter(tmp_dir)
        merger = pd.DataFrame({"Sub_tree_id": [1], "mass": [1e10]})
        equiv = pd.DataFrame({"snapshot": [0], "time": [13.8]})
        w.write_header(1, [0], {}, merger, equiv)

        # First (later "crashed") attempt.
        props1 = pd.DataFrame({"Sub_tree_id": [1], "rh": [3.0]})
        dyn1 = pd.DataFrame({"Sub_tree_id": [1], "dynstate": [0]})
        w.write_snapshot(
            snapshot_id=0, time=13.8,
            properties_df=props1, dynstate_df=dyn1,
            satellites_map={1: {2, 3}, 99: {100}},
        )

        # Retry for the SAME snapshot_id, different data (different
        # galaxy-ID set in satellites_map too).
        props2 = pd.DataFrame({"Sub_tree_id": [1], "rh": [5.0]})
        dyn2 = pd.DataFrame({"Sub_tree_id": [1], "dynstate": [1]})
        w.write_snapshot(
            snapshot_id=0, time=13.8,
            properties_df=props2, dynstate_df=dyn2,
            satellites_map={1: {7}},
        )

        r = HDF5CatalogueReader(os.path.join(tmp_dir, "catalogue.hdf5"))
        p = r.read_galaxy_properties(snapshot_id=0)
        assert len(p) == 1
        assert p["rh"].iloc[0] == 5.0
        d = r.read_riley_criterion(snapshot_id=0)
        assert d["dynstate"].iloc[0] == 1

        with h5py.File(os.path.join(tmp_dir, "catalogue.hdf5"), "r") as hf:
            sat_grp = hf["snapshots"]["0"]["satellite_relations"]
            assert set(sat_grp.keys()) == {"1"}  # galaxy 99 from the first attempt is gone
            np.testing.assert_array_equal(sat_grp["1"][:], [7])
