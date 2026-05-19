import os

import numpy as np
import h5py
import pytest

from roadrunner.io.hdf5_particles import HDF5ParticleWriter
from roadrunner._mcf_types import SnapshotData
from roadrunner.physics.scaler import StandardScaler


class TestHDF5ParticleWriter:
    @pytest.fixture
    def tmp_dir(self, tmp_path):
        return str(tmp_path / "particles_test")

    def _make_snap_data(self, n=50):
        rng = np.random.default_rng(42)
        return SnapshotData(
            indices=np.arange(n, dtype=np.uint64),
            masses=rng.uniform(0.1, 10.0, n).astype(np.float64),
            positions=rng.uniform(-100, 100, (n, 3)).astype(np.float64),
            velocities=rng.uniform(-200, 200, (n, 3)).astype(np.float64),
            redshift=0.1,
            time=13.0,
        )

    def test_write_snapshot_creates_datasets(self, tmp_dir):
        os.makedirs(tmp_dir, exist_ok=True)
        w = HDF5ParticleWriter(tmp_dir, mode="w")
        snap = self._make_snap_data()
        w.write_snapshot(0, 13.0, 0.1, snap)
        path = os.path.join(tmp_dir, "particles.hdf5")
        with h5py.File(path, "r") as hf:
            grp = hf["/snapshots/0"]
            assert "positions" in grp
            assert "velocities" in grp
            assert "masses" in grp
            assert "indices" in grp
            assert "scaler" in grp
            assert grp["positions"].shape == (50, 3)
            assert grp["masses"].shape == (50,)
            assert grp.attrs["time"] == 13.0
            assert grp.attrs["redshift"] == 0.1

    def test_scaler_round_trip(self, tmp_dir):
        os.makedirs(tmp_dir, exist_ok=True)
        w = HDF5ParticleWriter(tmp_dir, mode="w")
        snap = self._make_snap_data()
        w.write_snapshot(0, 13.0, 0.1, snap)
        path = os.path.join(tmp_dir, "particles.hdf5")
        with h5py.File(path, "r") as hf:
            grp = hf["/snapshots/0"]
            mean = grp["scaler/mean"][:]
            scale = grp["scaler/scale"][:]
            scaled_pos = grp["positions"][:]
            scaled_vel = grp["velocities"][:]

        coords = np.column_stack([snap.positions, snap.velocities])
        scaler = StandardScaler()
        expected = scaler.fit_transform(coords)
        assert np.allclose(scaled_pos, expected[:, :3], atol=1e-4)
        assert np.allclose(scaled_vel, expected[:, 3:6], atol=1e-4)
        assert np.allclose(mean, scaler.mean_)
        assert np.allclose(scale, scaler.scale_)

    def test_multiple_snapshots(self, tmp_dir):
        os.makedirs(tmp_dir, exist_ok=True)
        w = HDF5ParticleWriter(tmp_dir, mode="w")
        w.write_snapshot(0, 13.0, 0.1, self._make_snap_data(30))
        w.write_snapshot(1, 12.0, 0.2, self._make_snap_data(40))
        path = os.path.join(tmp_dir, "particles.hdf5")
        with h5py.File(path, "r") as hf:
            assert hf["/snapshots/0/indices"].shape == (30,)
            assert hf["/snapshots/1/indices"].shape == (40,)
