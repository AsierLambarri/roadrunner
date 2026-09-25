import numpy as np
import pytest

from roadrunner._mcf_types import SnapshotData
from roadrunner.readers.snapshot import SnapshotReader


class TestMockData:
    def test_correct_shapes(self):
        n = 50
        data = SnapshotReader.mock_data(n)
        assert isinstance(data, SnapshotData)
        assert data.index.shape == (n,)
        assert data.mass.shape == (n,)
        assert data.position.shape == (n, 3)
        assert data.velocity.shape == (n, 3)
        assert data.metallicity is not None
        assert data.metallicity.shape == (n,)
        assert data.redshift == 0.0
        assert data.time == 13.8

    def test_reproducible_seed(self):
        a = SnapshotReader.mock_data(10, seed=42)
        b = SnapshotReader.mock_data(10, seed=42)
        assert np.array_equal(a.index, b.index)
        assert np.array_equal(a.mass, b.mass)

    def test_different_seed_different(self):
        a = SnapshotReader.mock_data(10, seed=42)
        b = SnapshotReader.mock_data(10, seed=99)
        assert not np.array_equal(a.mass, b.mass)

    def test_zero_particles(self):
        data = SnapshotReader.mock_data(0)
        assert len(data.index) == 0
        assert data.position.shape == (0, 3)
        assert data.metallicity is not None and len(data.metallicity) == 0

    def test_single_particle(self):
        data = SnapshotReader.mock_data(1)
        assert data.position.shape == (1, 3)


class TestConstruction:
    def test_valid_code(self):
        reader = SnapshotReader("RAMSES", "stars", {"index": "particle_index"})
        assert reader.code == "RAMSES"
        assert reader.ptype == "stars"

    def test_unit_base_stored(self):
        ub = {"length": (1.0, "kpc")}
        reader = SnapshotReader("GEAR", "stars", {}, unit_base=ub)
        assert reader.unit_base == ub


class TestOpenDispatch:
    def test_unsupported_code_raises(self):
        reader = SnapshotReader("FOO", "stars", {"index": "pid"})
        with pytest.raises(ValueError, match="Unsupported code: FOO"):
            reader._open("/fake/path")

    def test_code_case_insensitive(self):
        for code in ("art", "Art", "ART-I", "gear", "Gear", "auriga", "AREPO",
                     "ramses", "RAMSES", "vintergatan", "VINTERGATAN"):
            reader = SnapshotReader(code, "stars", {"index": "pid"})
            assert reader.code in (
                "ART", "ART-I", "GEAR", "AURIGA", "AREPO",
                "RAMSES", "VINTERGATAN",
            )
