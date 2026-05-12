import pandas as pd
import pytest

from roadrunner.readers.merger_tree import MergerTreeReader

TEST_TREE = "test_data/test_tree.csv"


class TestConstruction:
    def test_from_csv_path(self):
        reader = MergerTreeReader(TEST_TREE)
        assert isinstance(reader.dataframe, pd.DataFrame)
        assert not reader.dataframe.empty

    def test_from_dataframe(self):
        df = pd.read_csv(TEST_TREE)
        reader = MergerTreeReader(df)
        assert isinstance(reader.dataframe, pd.DataFrame)
        assert not reader.dataframe.empty

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError, match="data must be str"):
            MergerTreeReader(123)


class TestProperties:
    @pytest.fixture
    def reader(self):
        return MergerTreeReader(TEST_TREE)

    def test_snapshots(self, reader):
        snaps = reader.snapshots
        assert isinstance(snaps, list)
        assert all(isinstance(s, int) for s in snaps)
        assert snaps == sorted(snaps)

    def test_subtree_ids(self, reader):
        ids = reader.subtree_ids
        assert isinstance(ids, list)
        assert len(ids) > 0
        assert ids == sorted(ids)


class TestSelectSnapshots:
    @pytest.fixture
    def reader(self):
        return MergerTreeReader(TEST_TREE)

    def test_select_single_snapshot(self, reader):
        snaps = reader.snapshots
        assert len(snaps) >= 1
        result = reader.select_snapshots([snaps[0]])
        assert not result.empty
        assert (result["Snapshot"] == snaps[0]).all()

    def test_select_multiple_snapshots(self, reader):
        snaps = reader.snapshots
        result = reader.select_snapshots(snaps)
        assert len(result) == len(reader.dataframe)

    def test_select_empty_snapshots(self, reader):
        result = reader.select_snapshots([99999])
        assert result.empty


class TestSelectSubtrees:
    @pytest.fixture
    def reader(self):
        return MergerTreeReader(TEST_TREE)

    def test_select_single_subtree(self, reader):
        ids = reader.subtree_ids
        result = reader.select_subtrees([ids[0]])
        assert not result.empty
        assert (result["Sub_tree_id"] == ids[0]).all()

    def test_select_multiple_subtrees(self, reader):
        ids = reader.subtree_ids[:3]
        result = reader.select_subtrees(ids)
        assert len(result["Sub_tree_id"].unique()) == len(ids)

    def test_select_empty_subtrees(self, reader):
        result = reader.select_subtrees([-1])
        assert result.empty


class TestSelectAccretionHost:
    @pytest.fixture
    def reader(self):
        return MergerTreeReader(TEST_TREE)

    def test_select_max_mass_host(self, reader):
        snap = reader.snapshots[0]
        host = reader.select_accretion_host(snap, criterion="max_mass")
        assert not host.empty
        host_mass = host.iloc[0]["mass"]
        snap_df = reader.select_snapshots([snap])
        max_mass = snap_df["mass"].max()
        assert host_mass == max_mass

    def test_unknown_criterion_raises(self, reader):
        with pytest.raises(ValueError, match="unknown criterion"):
            reader.select_accretion_host(0, criterion="invalid")

    def test_empty_snapshot(self, reader):
        result = reader.select_accretion_host(99999)
        assert result.empty


class TestToNumeric:
    def test_returns_dataframe(self):
        reader = MergerTreeReader(TEST_TREE)
        result = reader.to_numeric()
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(reader.dataframe)
