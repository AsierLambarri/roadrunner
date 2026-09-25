import pandas as pd
import pytest

from roadrunner.readers.equivalence import EquivalenceTable


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "snapshot": [100, 175, 200],
        "snapname": ["snap_100", "snap_175", "snap_200"],
        "time": [1.0, 5.0, 10.0],
        "redshift": [3.0, 1.5, 0.0],
    })


@pytest.fixture
def equiv_csv(tmp_path, sample_df):
    path = tmp_path / "equiv.csv"
    sample_df.to_csv(path, index=False)
    return str(path)


class TestConstruction:
    def test_from_csv_path(self, equiv_csv):
        eq = EquivalenceTable(equiv_csv)
        assert eq.snapshots == [100, 175, 200]

    def test_from_dataframe(self, sample_df):
        eq = EquivalenceTable(sample_df)
        assert eq.snapshots == [100, 175, 200]

    def test_from_arrays(self):
        eq = EquivalenceTable((
            [100, 175, 200],
            ["snap_100", "snap_175", "snap_200"],
            [1.0, 5.0, 10.0],
            [3.0, 1.5, 0.0],
        ))
        assert eq.snapshots == [100, 175, 200]

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError, match="data must be"):
            EquivalenceTable(123)


class TestSnapshotPath:
    def test_with_base_dir(self, sample_df):
        eq = EquivalenceTable(sample_df, base_dir="/data/sims")
        assert eq.snapshot_path(100) == "/data/sims/snap_100"

    def test_without_base_dir(self, sample_df):
        eq = EquivalenceTable(sample_df)
        assert eq.snapshot_path(175) == "snap_175"

    def test_trailing_slash_base_dir(self, sample_df):
        eq = EquivalenceTable(sample_df, base_dir="/data/")
        assert eq.snapshot_path(200) == "/data/snap_200"

    def test_missing_snapshot_raises(self, sample_df):
        eq = EquivalenceTable(sample_df)
        with pytest.raises(KeyError):
            eq.snapshot_path(999)


class TestSnapshotTime:
    def test_returns_float(self, sample_df):
        eq = EquivalenceTable(sample_df)
        assert eq.snapshot_time(100) == 1.0

    def test_missing_snapshot_raises(self, sample_df):
        eq = EquivalenceTable(sample_df)
        with pytest.raises(KeyError):
            eq.snapshot_time(999)


class TestSnapshotRedshift:
    def test_returns_float(self, sample_df):
        eq = EquivalenceTable(sample_df)
        assert eq.snapshot_redshift(100) == 3.0

    def test_missing_snapshot_raises(self, sample_df):
        eq = EquivalenceTable(sample_df)
        with pytest.raises(KeyError):
            eq.snapshot_redshift(999)


class TestProperties:
    def test_snapshots_sorted(self, sample_df):
        df = sample_df.iloc[[2, 0, 1]]  # unsorted
        eq = EquivalenceTable(df)
        assert eq.snapshots == [100, 175, 200]

    def test_min_snapshot(self, sample_df):
        eq = EquivalenceTable(sample_df)
        assert eq.min_snapshot == 100
        assert eq.max_snapshot == 200

    def test_dataframe_property(self, sample_df):
        eq = EquivalenceTable(sample_df)
        df = eq.dataframe
        assert isinstance(df, pd.DataFrame)
        assert "snapshot" in df.columns
        assert "snapname" in df.columns
        assert "time" in df.columns
        assert "redshift" in df.columns
