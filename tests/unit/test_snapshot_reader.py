from types import SimpleNamespace

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


class TestReadCosmology:
    @staticmethod
    def _reader(monkeypatch, code, **ds):
        reader = SnapshotReader(code, "star", {"index": "particle_index"})
        base = dict(cosmological_simulation=1, hubble_constant=0.6727, omega_matter=0.3139,
                    omega_lambda=0.6861, parameters={})
        monkeypatch.setattr(reader, "_open", lambda path: SimpleNamespace(**(base | ds)))
        return reader

    def test_ramses_reads_omega_b(self, monkeypatch):
        reader = self._reader(monkeypatch, "RAMSES", parameters={"omega_b": 0.045})
        assert reader.read_cosmology("snap") == {"h": 0.6727, "omega_m": 0.3139, "omega_b": 0.045}

    def test_omega_b_key_per_code(self, monkeypatch):
        assert self._reader(monkeypatch, "AREPO", parameters={"OmegaBaryon": 0.048}).read_cosmology("snap")["omega_b"] == 0.048
        assert "omega_b" not in self._reader(monkeypatch, "GEAR", parameters={"omega_b": 0.045}).read_cosmology("snap")

    def test_zero_omega_b_left_to_config(self, monkeypatch):
        assert "omega_b" not in self._reader(monkeypatch, "AREPO", parameters={"OmegaBaryon": 0.0}).read_cosmology("snap")

    def test_not_cosmological(self, monkeypatch):
        assert self._reader(monkeypatch, "RAMSES", cosmological_simulation=0).read_cosmology("snap") == {}

    def test_non_flat_warns(self, monkeypatch):
        with pytest.warns(UserWarning, match="taken as flat"):
            self._reader(monkeypatch, "RAMSES", omega_lambda=0.6).read_cosmology("snap")
