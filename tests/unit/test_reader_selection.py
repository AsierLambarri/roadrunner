"""Particle selection tests for NPZ and particle-data readers."""

import os
from unittest.mock import MagicMock

import h5py
import numpy as np
import pytest

from roadrunner.pipeline.config import RunConfig
from roadrunner.readers.equivalence import EquivalenceTable
from roadrunner.readers.npz_reader import NPZSnapshotReader
from roadrunner.readers.particle_data_reader import ParticleDataSnapshotReader

N = 10
IDS = np.arange(100, 100 + N, dtype=np.uint64)
POSITIONS = np.column_stack([
    np.arange(N, dtype=np.float64),
    np.zeros(N),
    np.zeros(N),
])
VELOCITIES = np.ones((N, 3), dtype=np.float64)
MASSES = np.linspace(1.0, 2.0, N)
METALLICITY = np.linspace(-1.0, 0.0, N)
REDSHIFT = 0.5
TIME = 5.0


def _equiv(snapname):
    return EquivalenceTable(([0], [snapname], [TIME], [REDSHIFT]))


def _write_npz(path):
    np.savez(
        path,
        indices=IDS,
        masses=MASSES,
        coords=np.column_stack([POSITIONS, VELOCITIES]),
        metallicity=METALLICITY,
    )


def _write_pdata(path, scale=2.0):
    mean = np.zeros(6)
    scale_arr = np.full(6, scale)
    with h5py.File(path, "w") as hf:
        d = hf.create_group("data")
        d.create_dataset("indices", data=IDS)
        d.create_dataset("masses", data=MASSES)
        d.create_dataset("positions", data=POSITIONS * scale_arr[:3])
        d.create_dataset("velocities", data=VELOCITIES * scale_arr[3:6])
        d.create_dataset("metallicity", data=METALLICITY)
        sc = d.create_group("scaler")
        sc.create_dataset("mean", data=mean)
        sc.create_dataset("scale", data=scale_arr)
        hdr = hf.create_group("header")
        hdr.attrs["redshift"] = REDSHIFT
        hdr.attrs["time"] = TIME


def _make_reader(tmp_path, reader_cls):
    if reader_cls is NPZSnapshotReader:
        path = os.path.join(str(tmp_path), "snap.npz")
        _write_npz(path)
        reader = NPZSnapshotReader(
            _equiv("snap.npz"), base_dir=str(tmp_path), mock_sim=True)
    else:
        path = os.path.join(str(tmp_path), "snap.hdf5")
        _write_pdata(path)
        reader = ParticleDataSnapshotReader(
            _equiv("snap.hdf5"), base_dir=str(tmp_path),
            extra_fields=["metallicity"])
    return reader, path


@pytest.fixture(params=[NPZSnapshotReader, ParticleDataSnapshotReader])
def reader_and_path(request, tmp_path):
    return _make_reader(tmp_path, request.param)


class TestSelectionAPI:
    def test_load_all(self, reader_and_path):
        reader, path = reader_and_path
        snap = reader.load(path)
        assert np.array_equal(snap.index, IDS)
        assert np.allclose(snap.position, POSITIONS)

    def test_override_filter(self, reader_and_path):
        reader, path = reader_and_path
        snap = reader.load(path, particle_indices=np.array([101, 105]))
        assert set(snap.index.tolist()) == {101, 105}

    def test_persistent_filter_and_erase(self, reader_and_path):
        reader, path = reader_and_path
        assert reader.particle_filter is None
        reader.set_particle_filter([100, 103])
        assert np.array_equal(reader.particle_filter, [100, 103])
        assert set(reader.load(path).index.tolist()) == {100, 103}
        reader.erase_particle_filter()
        assert reader.particle_filter is None
        assert len(reader.load(path).index) == N

    def test_extra_field_alignment(self, reader_and_path):
        reader, path = reader_and_path
        snap = reader.load(path, particle_indices=np.array([102, 107]))
        expected = METALLICITY[np.array([102, 107]) - 100]
        assert np.allclose(snap.metallicity, expected)

    def test_select_indices_sphere(self, reader_and_path):
        reader, path = reader_and_path
        ids = reader.select_indices(path, sphere=((2.0, 0.0, 0.0), 1.5))
        assert set(ids.tolist()) == {101, 102, 103}

    def test_select_indices_bbox(self, reader_and_path):
        reader, path = reader_and_path
        ids = reader.select_indices(
            path, bbox=((1.5, -1.0, -1.0), (3.5, 1.0, 1.0)))
        assert set(ids.tolist()) == {102, 103}

    def test_select_indices_ignores_persistent_filter(self, reader_and_path):
        reader, path = reader_and_path
        reader.set_particle_filter([100])
        ids = reader.select_indices(path, sphere=((2.0, 0.0, 0.0), 1.5))
        assert set(ids.tolist()) == {101, 102, 103}
        assert np.array_equal(reader.particle_filter, [100])

    def test_region_argument_validation(self, reader_and_path):
        reader, path = reader_and_path
        with pytest.raises(ValueError, match="exactly one"):
            reader.select_indices(path)
        with pytest.raises(ValueError, match="exactly one"):
            reader.select_indices(
                path, sphere=((0, 0, 0), 1.0), bbox=((0, 0, 0), (1, 1, 1)))

    def test_pdata_unscaling(self, tmp_path):
        reader, path = _make_reader(tmp_path, ParticleDataSnapshotReader)
        snap = reader.load(path)
        assert np.allclose(snap.position, POSITIONS)
        assert np.allclose(snap.velocity, VELOCITIES)


class TestEntrySelection:
    @staticmethod
    def _patch(monkeypatch, reader_mock):
        import roadrunner.pipeline.entry as entry

        mh = MagicMock()
        mh.snapshots = [0, 50, 100]
        eq = MagicMock()
        eq.snapshot_path = lambda sid: f"/p/{sid}"

        monkeypatch.setattr(entry, "MergerTreeReaderCSV", MagicMock())
        monkeypatch.setattr(entry, "MergerTreeHandlerCSV",
                            MagicMock(return_value=mh))
        monkeypatch.setattr(entry, "EquivalenceTable",
                            MagicMock(return_value=eq))
        monkeypatch.setattr(entry, "NPZSnapshotReader",
                            MagicMock(return_value=reader_mock))
        monkeypatch.setattr(entry, "SnapshotReader", MagicMock())
        monkeypatch.setattr(entry, "ParticleDataSnapshotReader", MagicMock())
        for name in (
            "XGMMAssigner", "ProcessingConfig", "ReductionConfig",
            "SnapshotOrchestrator", "HDF5CatalogueWriter",
            "HDF5ParticleWriter", "HDF5AssignmentWriter", "RunLogger",
            "BirthTracker", "AssemblyTracker", "AccretionPipeline",
        ):
            monkeypatch.setattr(entry, name, MagicMock())

    def test_non_yt_selection(self, monkeypatch):
        reader = MagicMock()
        reader.select_indices.return_value = np.array([11, 22])
        self._patch(monkeypatch, reader)
        import roadrunner.pipeline.entry as entry

        entry.run_accretion_history(RunConfig(
            accretion_id=1, reader_type="npz", selection_snapshot=100,
            selection_sphere=[[0.0, 0.0, 0.0], 1.0]))

        reader.select_indices.assert_called_once_with(
            "/p/100", sphere=[[0.0, 0.0, 0.0], 1.0], bbox=None)
        reader.set_particle_filter.assert_called_once()

    def test_selection_precedes_range_warns(self, monkeypatch):
        reader = MagicMock()
        reader.select_indices.return_value = np.array([11])
        self._patch(monkeypatch, reader)
        import roadrunner.pipeline.entry as entry

        with pytest.warns(UserWarning, match="precedes the last processed"):
            entry.run_accretion_history(RunConfig(
                accretion_id=1, reader_type="npz", selection_snapshot=50,
                end_snapshot=100,
                selection_bbox=[[-1, -1, -1], [1, 1, 1]]))
