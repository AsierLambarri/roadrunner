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
        assert np.allclose(snap.velocity, VELOCITIES)

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
        assert set(snap.index.tolist()) == {102, 107}

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


class TestMaskedReads:
    """H03: a filtered load reads only the selected rows (block-wise for
    HDF5, member by member for NPZ) and must equal a full load masked
    afterwards, field by field and in file order."""

    @pytest.mark.parametrize("ids", [
        IDS[[1, 3, 4]],                              # subset
        IDS[::-1],                                   # everything, shuffled
        IDS[[2, 2, 5]],                              # duplicates
        np.array([7, 8], dtype=np.uint64),           # absent only
        np.array([7, 104, 109], dtype=np.uint64),    # absent and present
        np.array([], dtype=np.uint64),               # empty selection
    ])
    def test_filtered_load_equals_masked_full_load(self, reader_and_path, ids):
        reader, path = reader_and_path
        full = reader.load(path)
        sub = reader.load(path, particle_indices=ids)
        keep = np.isin(full.index, ids)
        assert sub.fields == full.fields
        for f in full.fields:
            np.testing.assert_array_equal(getattr(sub, f), getattr(full, f)[keep], err_msg=f)
        assert (sub.redshift, sub.time) == (full.redshift, full.time)

    @pytest.mark.parametrize("chunks", [None, (4,)])
    def test_hdf5_block_edges(self, tmp_path, monkeypatch, chunks):
        # Blocks of 3 rows (rounded up to the chunk length when chunked)
        # split the 10 rows unevenly; skipped blocks must not shift rows.
        from roadrunner.readers import particle_data_reader as pdr
        monkeypatch.setattr(pdr, "READ_BLOCK_ROWS", 3)
        path = os.path.join(str(tmp_path), "blocks.hdf5")
        data = np.column_stack([np.arange(N, dtype=float), -np.arange(N, dtype=float)])
        with h5py.File(path, "w") as hf:
            hf.create_dataset("x", data=data, chunks=None if chunks is None else chunks + (2,))
        rng = np.random.default_rng(0)
        with h5py.File(path, "r") as hf:
            for mask in (np.zeros(N, bool), np.ones(N, bool), rng.random(N) < 0.3,
                         np.r_[np.zeros(9, bool), True]):
                np.testing.assert_array_equal(pdr._read_rows(hf["x"], mask), data[mask])

    def test_hdf5_select_indices_block_wise(self, tmp_path, monkeypatch):
        from roadrunner.readers import particle_data_reader as pdr
        monkeypatch.setattr(pdr, "READ_BLOCK_ROWS", 3)
        reader, path = _make_reader(tmp_path, ParticleDataSnapshotReader)
        ids = reader.select_indices(path, sphere=((4.5, 0.0, 0.0), 2.6))
        assert set(ids.tolist()) == {102, 103, 104, 105, 106, 107}


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
