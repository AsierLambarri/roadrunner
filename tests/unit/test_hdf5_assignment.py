import os

import h5py
import numpy as np
import pandas as pd
import pytest

from roadrunner.io.hdf5_assignment import HDF5AssignmentWriter
from roadrunner._mcf_types import AssignmentResult
from roadrunner.clustering.sparse import SparseCSC


def _make_assignment_result(n_particles=100, n_galaxies=5):
    rng = np.random.default_rng(42)
    indices = np.arange(n_particles, dtype=np.uint64)
    subtree_ids = rng.integers(1, n_galaxies + 1, n_particles)
    df = pd.DataFrame({"array_index": indices, "Sub_tree_id": subtree_ids})

    col_idx_list = []
    col_val_list = []
    gid_list = []
    for gid in range(1, n_galaxies + 1):
        mask = subtree_ids == gid
        pid = indices[mask]
        vals = rng.uniform(-1, 0, len(pid)).astype(np.float32)
        col_idx_list.append(pid)
        col_val_list.append(vals)
        gid_list.append(gid)

    resp_csc = SparseCSC(
        col_idx_list, col_val_list,
        column_id=np.array(gid_list, dtype=np.int64),
    )

    fitted = {}
    for gid in range(1, n_galaxies + 1):
        fitted[gid] = {
            "mean": rng.uniform(-10, 10, 6).astype(np.float64),
            "weight": rng.uniform(0, 1),
            "covariance": np.abs(rng.uniform(0, 5, 6)).astype(np.float64),
        }

    result = AssignmentResult(
        particle_df=df,
        responsibilities=resp_csc,
        fitted_parameters=fitted,
        statistics={"groups": 1},
    )
    bound_csc = SparseCSC(
        col_idx_list, col_val_list,
        column_id=np.arange(1, n_galaxies + 1, dtype=np.int64),
    )
    return result, bound_csc


class TestHDF5AssignmentWriter:
    @pytest.fixture
    def tmp_dir(self, tmp_path):
        return str(tmp_path / "assignment_test")

    def test_write_snapshot_galaxy_groups(self, tmp_dir):
        os.makedirs(tmp_dir, exist_ok=True)
        w = HDF5AssignmentWriter(tmp_dir)
        result, csc = _make_assignment_result()
        w.write_snapshot(0, 13.0, result, csc)

        path = os.path.join(tmp_dir, "assignment", "snapshot0000.hdf5")
        with h5py.File(path, "r") as hf:
            galaxies = hf["galaxies"]
            for gid in range(1, 6):
                grp = galaxies[str(gid)]
                assert "indices" in grp
                assert "log_resp" in grp
                assert "boundness" in grp
                assert "mean" in grp
                assert "weight" in grp
                assert "covariance" in grp
                assert grp["indices"].shape == grp["log_resp"].shape
                assert grp["indices"].shape == grp["boundness"].shape
            assert "hard_assignment" in hf
            assert hf.attrs["time"] == 13.0

    def test_timescales(self, tmp_dir):
        os.makedirs(tmp_dir, exist_ok=True)
        w = HDF5AssignmentWriter(tmp_dir)
        result, csc = _make_assignment_result()
        w.write_snapshot(0, 13.0, result, csc)
        ts = np.array(
            list(zip(range(100), np.full(100, 0.5, dtype=np.float32))),
            dtype=[("particle_index", np.uint64), ("timescale", np.float32)],
        )
        w.write_timescales(ts)
        ts_path = os.path.join(tmp_dir, "assignment", "timescales.txt")
        assert os.path.exists(ts_path)
        loaded = np.loadtxt(ts_path, dtype=np.uint64)
        assert loaded.shape == (100, 2)

        # A later batch: particles 50-149, where 50-99 overlap the first
        # batch (given a *different* timescale, to prove the overlap is
        # ignored rather than rewritten) and 100-149 are genuinely new.
        ts2 = np.array(
            list(zip(range(50, 150), np.full(100, 9.0, dtype=np.float32))),
            dtype=[("particle_index", np.uint64), ("timescale", np.float32)],
        )
        w.write_timescales(ts2)
        loaded2 = np.loadtxt(ts_path, dtype=np.float64)
        assert loaded2.shape == (150, 2)  # only the 50 genuinely-new rows were appended
        by_id = dict(zip(loaded2[:, 0].astype(np.uint64), loaded2[:, 1]))
        assert by_id[0] == pytest.approx(0.5)  # untouched by the second, conflicting batch

    def test_empty_assignment(self, tmp_dir):
        os.makedirs(tmp_dir, exist_ok=True)
        w = HDF5AssignmentWriter(tmp_dir)
        df = pd.DataFrame({"array_index": pd.array([], dtype=np.uint64),
                           "Sub_tree_id": pd.array([], dtype=np.int64)})
        empty_csc = SparseCSC(
            [np.array([], dtype=np.int64)],
            [np.array([], dtype=np.float32)],
            column_id=np.array([], dtype=np.int64),
        )
        result = AssignmentResult(
            particle_df=df,
            responsibilities=empty_csc,
            fitted_parameters={},
            statistics={},
        )
        bound_csc = SparseCSC([np.array([], dtype=np.int64)],
                              [np.array([], dtype=np.float32)],
                              column_id=np.array([1], dtype=np.int64))
        w.write_snapshot(0, 13.0, result, bound_csc)
        path = os.path.join(tmp_dir, "assignment", "snapshot0000.hdf5")
        with h5py.File(path, "r") as hf:
            assert "hard_assignment" in hf
