import os

import numpy as np
import pytest

from roadrunner.io.serialization import save_checkpoint, load_checkpoint
from roadrunner._exceptions import RestartError


class TestSerialization:
    def test_save_and_load_dict(self, tmp_path):
        path = str(tmp_path / "checkpoint.zst")
        data = {"key": "value", "num": 42}
        save_checkpoint(path, data)
        loaded = load_checkpoint(path)
        assert loaded == data

    def test_with_numpy_arrays(self, tmp_path):
        path = str(tmp_path / "arr_checkpoint.zst")
        data = {"array": np.array([1.0, 2.0, 3.0]), "int_arr": np.arange(10)}
        save_checkpoint(path, data)
        loaded = load_checkpoint(path)
        assert np.allclose(loaded["array"], data["array"])
        assert np.array_equal(loaded["int_arr"], data["int_arr"])

    def test_nested_structures(self, tmp_path):
        path = str(tmp_path / "nested.zst")
        data = {"outer": {"inner": [1, 2, 3], "scores": {1: 0.5, 2: 0.3}}}
        save_checkpoint(path, data)
        loaded = load_checkpoint(path)
        assert loaded == data

    def test_compression_levels(self, tmp_path):
        data = {"x": list(range(1000))}
        sizes = []
        for level in [1, 3, 9]:
            p = str(tmp_path / f"level_{level}.zst")
            save_checkpoint(p, data, level=level)
            sizes.append(os.path.getsize(p))
        # Higher compression should give smaller or equal files
        assert sizes[-1] <= sizes[0]

    def test_corrupted_file_raises(self, tmp_path):
        path = str(tmp_path / "corrupt.zst")
        with open(path, "wb") as f:
            f.write(b"not a zstd file")
        with pytest.raises(RestartError, match="Corrupted"):
            load_checkpoint(path)

    def test_missing_file_raises(self, tmp_path):
        path = str(tmp_path / "nonexistent.zst")
        with pytest.raises(RestartError, match="not found"):
            load_checkpoint(path)

    def test_version_migration(self, tmp_path):
        path = str(tmp_path / "v1_checkpoint.zst")
        data = {"version": 0, "data": {"hello": "world"}}
        import pickle, zstandard as zstd
        pickled = pickle.dumps(data)
        with open(path, "wb") as f:
            f.write(zstd.ZstdCompressor(level=3).compress(pickled))
        loaded = load_checkpoint(path)
        assert loaded["hello"] == "world"
