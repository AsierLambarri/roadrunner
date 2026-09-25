"""Tests for the extra_fields system: SnapshotData, processing, HDF5 writer."""

import numpy as np
import pytest

from roadrunner._mcf_types import SnapshotData


class TestSnapshotDataRegistries:
    def test_fields_default(self):
        """Default assign_fields = [positions, velocities]."""
        data = SnapshotData(
            index=np.array([0, 1]),
            mass=np.array([1.0, 2.0]),
            position=np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
            velocity=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            redshift=0.5, time=1.0,
        )
        assert data.fields == ["index", "mass", "position", "velocity"]
        assert data._assign_fields == ["position", "velocity"]
        assert data.extra_fields == []

    def test_extra_fields_populated(self):
        """Metallicity passed as kwarg ends up in extra_fields."""
        data = SnapshotData(
            index=np.array([0, 1]),
            mass=np.array([1.0, 2.0]),
            position=np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
            velocity=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            redshift=0.5, time=1.0,
            metallicity=np.array([0.01, 0.02]),
        )
        assert "metallicity" in data.fields
        assert "metallicity" in data.extra_fields
        assert "metallicity" not in data._assign_fields
        np.testing.assert_array_equal(data.metallicity, [0.01, 0.02])

    def test_data_property_shape(self):
        """.data column-concatenates assign_fields."""
        data = SnapshotData(
            index=np.array([0, 1]),
            mass=np.array([1.0, 2.0]),
            position=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            velocity=np.array([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]),
            redshift=0.5, time=1.0,
        )
        assert data.data.shape == (2, 6)
        np.testing.assert_array_equal(data.data[:, :3], data.position)
        np.testing.assert_array_equal(data.data[:, 3:], data.velocity)

    def test_custom_assign_fields(self):
        """Overriding assign_fields changes what .data returns."""
        data = SnapshotData(
            index=np.array([0, 1]),
            mass=np.array([1.0, 2.0]),
            position=np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            velocity=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            redshift=0.5, time=1.0,
            assign_fields=["position"],
        )
        assert data.data.shape == (2, 3)
        # velocity is not in assign_fields and not internal → is extra
        assert data.extra_fields == ["velocity"]

    def test_assign_fields_listed_in_extra_excluded(self):
        """A field in both assign_fields and kwargs is not extra."""
        data = SnapshotData(
            index=np.array([0, 1]),
            mass=np.array([1.0, 2.0]),
            position=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            velocity=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            redshift=0.5, time=1.0,
            assign_fields=["position", "velocity", "metallicity"],
            metallicity=np.array([0.01, 0.02]),
        )
        assert "metallicity" not in data.extra_fields

class TestExtraFieldsHDF5Roundtrip:
    def test_write_snapshot_writes_extra(self, tmp_path):
        """HDF5ParticleWriter writes extra_fields datasets."""
        from roadrunner.io.hdf5_particles import HDF5ParticleWriter
        import h5py

        data = SnapshotData(
            index=np.array([0, 1], dtype=np.uint64),
            mass=np.array([1.0, 2.0]),
            position=np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
            velocity=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            redshift=0.5, time=1.0,
            metallicity=np.array([0.0, 0.5]),
        )

        writer = HDF5ParticleWriter(str(tmp_path))
        writer.write_snapshot(0, 1.0, 0.5, data)

        with h5py.File(writer._snap_path(0), "r") as hf:
            assert "metallicity" in hf["data"]
            np.testing.assert_allclose(hf["data/metallicity"][:], [0.0, 0.5], atol=1e-4)


class TestConfigAssignFields:
    def test_run_config_accepts_assign_fields(self):
        """RunConfig parses with assign_fields set."""
        from roadrunner.pipeline.config import RunConfig

        config = RunConfig(
            fields={
                "index": "pid",
                "mass": "m",
                "position": "pos",
                "velocity": "vel",
                "metallicity": "Z",
            },
            assign_fields=["positions", "velocities"],
        )
        assert config.assign_fields == ["positions", "velocities"]
