import numpy as np
import pandas as pd
import pytest

from roadrunner.physics.merger_tree import (
    MergerTreeHandlerCSV,
    nfw_cmz_relation_duffy,
)

TEST_TREE = "test_data/test_tree.csv"


# ── Synthetic Data Fixtures ──────────────────────────────────

@pytest.fixture
def two_halo_df():
    return pd.DataFrame({
        "Sub_tree_id": [1, 2],
        "Snapshot": [0, 0],
        "Redshift": [0.0, 0.0],
        "Time": [1.0, 1.0],
        "mass": [1e12, 1e10],
        "virial_radius": [100.0, 20.0],
        "scale_radius": [np.nan, 5.0],
        "position_x": [0.0, 1.0],
        "position_y": [0.0, 0.0],
        "position_z": [0.0, 0.0],
        "velocity_x": [0.0, 0.0],
        "velocity_y": [0.0, 0.0],
        "velocity_z": [0.0, 0.0],
    })


@pytest.fixture
def scale_radii_df():
    return pd.DataFrame({
        "Sub_tree_id": [1, 2, 3],
        "Snapshot": [0, 0, 0],
        "Redshift": [0.0, 0.0, 0.0],
        "mass": [1e12, 1e11, 1e10],
        "virial_radius": [100.0, 50.0, 20.0],
        "scale_radius": [np.nan, 10.0, np.nan],
        "position_x": [0.0, 0.0, 0.0],
        "position_y": [0.0, 0.0, 0.0],
        "position_z": [0.0, 0.0, 0.0],
        "velocity_x": [0.0, 0.0, 0.0],
        "velocity_y": [0.0, 0.0, 0.0],
        "velocity_z": [0.0, 0.0, 0.0],
    })


@pytest.fixture
def three_halo_bound_df():
    return pd.DataFrame({
        "Sub_tree_id": [10, 20, 30],
        "Snapshot": [0, 0, 0],
        "Redshift": [0.0, 0.0, 0.0],
        "mass": [1e12, 5e10, 1e10],
        "virial_radius": [100.0, 50.0, 20.0],
        "scale_radius": [10.0, 5.0, 2.0],
        "position_x": [0.0, 100.0, 5.0],     # 30 is close to 10
        "position_y": [0.0, 0.0, 0.0],
        "position_z": [0.0, 0.0, 0.0],
        "velocity_x": [0.0, 0.0, 0.0],
        "velocity_y": [0.0, 0.0, 0.0],
        "velocity_z": [0.0, 0.0, 0.0],
    })


# ── Static helper tests ─────────────────────────────────────

class TestComputeRsRow:
    def test_nan_scale_radius(self):
        row = pd.Series({
            "mass": 1e12, "Redshift": 0.0,
            "virial_radius": 100.0, "scale_radius": np.nan,
        })
        result = MergerTreeHandlerCSV._compute_rs_row(row)
        expected = 100.0 / nfw_cmz_relation_duffy(1e12, 0.0)
        assert np.isclose(result, expected)

    def test_valid_scale_radius(self):
        row = pd.Series({
            "mass": 1e12, "Redshift": 0.0,
            "virial_radius": 100.0, "scale_radius": 15.0,
        })
        assert MergerTreeHandlerCSV._compute_rs_row(row) == 15.0


class TestSatellitesImpl:
    def test_two_halos_one_bound(self, two_halo_df):
        result = MergerTreeHandlerCSV._satellites_impl(two_halo_df, rvir_factor=2.0)
        assert 1 in result
        assert 2 in result[1]

    def test_isolated_halo(self, two_halo_df):
        result = MergerTreeHandlerCSV._satellites_impl(
            two_halo_df.iloc[[0]], rvir_factor=1.0
        )
        host_id = two_halo_df.iloc[0]["Sub_tree_id"]
        assert result.get(host_id, set()) == set()

    def test_empty_df(self):
        empty = pd.DataFrame(columns=[
            "Sub_tree_id", "Redshift", "position_x", "position_y", "position_z",
            "velocity_x", "velocity_y", "velocity_z", "mass", "virial_radius",
        ])
        assert MergerTreeHandlerCSV._satellites_impl(empty, 1.0) == {}


class TestMostBoundSatelliteImpl:
    def test_satellite_bound_to_one_host(self, two_halo_df):
        result = MergerTreeHandlerCSV._most_bound_satellite_impl(two_halo_df, rvir_factor=2.0)
        assert 2 in result
        assert result[2] == 1

    def test_empty_df(self):
        empty = pd.DataFrame(columns=[
            "Sub_tree_id", "Redshift", "position_x", "position_y", "position_z",
            "velocity_x", "velocity_y", "velocity_z", "mass", "virial_radius",
        ])
        assert MergerTreeHandlerCSV._most_bound_satellite_impl(empty, 1.0) == {}


class TestDistanceToHostImpl:
    def test_basic_distance(self):
        df = pd.DataFrame({
            "Sub_tree_id": [1, 2],
            "host_id": [np.nan, 1],
            "position_x": [0.0, 3.0],
            "position_y": [0.0, 4.0],
            "position_z": [0.0, 0.0],
        }).infer_objects()
        result = MergerTreeHandlerCSV._distance_to_host_impl(df, "host_id")
        assert np.isclose(
            result.loc[result["Sub_tree_id"] == 2, "distance_to_host_id"].values[0],
            5.0,
        )

    def test_no_host_nan(self):
        df = pd.DataFrame({
            "Sub_tree_id": [1],
            "host_id": [np.nan],
            "position_x": [0.0],
            "position_y": [0.0],
            "position_z": [0.0],
        }).infer_objects()
        result = MergerTreeHandlerCSV._distance_to_host_impl(df, "host_id")
        assert np.isnan(result["distance_to_host_id"].iloc[0])

    def test_empty_df(self):
        empty = pd.DataFrame(columns=["Sub_tree_id", "host_id", "position_x", "position_y", "position_z"])
        result = MergerTreeHandlerCSV._distance_to_host_impl(empty, "host_id")
        assert "distance_to_host_id" in result.columns


# ── State-modifying method tests ────────────────────────────

class TestComputeScaleRadii:
    def test_nan_filled(self, scale_radii_df):
        handler = MergerTreeHandlerCSV(scale_radii_df)
        handler.compute_scale_radii()
        for _, row in handler.dataframe.iterrows():
            if pd.isna(row["scale_radius"]):
                pytest.fail("NaN remained after compute_scale_radii")

    def test_nan_filled_correct_value(self, scale_radii_df):
        handler = MergerTreeHandlerCSV(scale_radii_df)
        handler.compute_scale_radii()
        row = handler.dataframe.iloc[0]
        expected = row["virial_radius"] / nfw_cmz_relation_duffy(row["mass"], row["Redshift"])
        assert np.isclose(row["scale_radius"], expected)

    def test_valid_unchanged(self, scale_radii_df):
        handler = MergerTreeHandlerCSV(scale_radii_df)
        original = handler.dataframe.iloc[1]["scale_radius"]
        handler.compute_scale_radii()
        assert handler.dataframe.iloc[1]["scale_radius"] == original


class TestComputeMostBoundSatellite:
    @pytest.fixture
    def handler(self, three_halo_bound_df):
        h = MergerTreeHandlerCSV(three_halo_bound_df)
        h.compute_most_bound_satellite(rvir_factor=2.0)
        return h

    def test_host_id_column_added(self, handler):
        assert "host_id" in handler.dataframe.columns

    def test_large_halo_no_host(self, handler):
        host_row = handler.dataframe[handler.dataframe["Sub_tree_id"] == 10]
        assert host_row["host_id"].values[0] == -1

    def test_close_small_halo_has_host(self, handler):
        close_row = handler.dataframe[handler.dataframe["Sub_tree_id"] == 30]
        assert close_row["host_id"].values[0] == 10


class TestComputeDistanceToHost:
    @pytest.fixture
    def handler(self, three_halo_bound_df):
        h = MergerTreeHandlerCSV(three_halo_bound_df)
        h.compute_most_bound_satellite(rvir_factor=2.0)
        h.compute_distance_to_host(column="host_id")
        return h

    def test_distance_column_added(self, handler):
        assert "distance_to_host_id" in handler.dataframe.columns

    def test_central_halo_nan(self, handler):
        host_row = handler.dataframe[handler.dataframe["Sub_tree_id"] == 10]
        assert np.isnan(host_row["distance_to_host_id"].values[0])

    def test_satellite_has_positive_distance(self, handler):
        close_row = handler.dataframe[handler.dataframe["Sub_tree_id"] == 30]
        dist = close_row["distance_to_host_id"].values[0]
        assert dist > 0


class TestComputeSatellites:
    def test_returns_dict(self, three_halo_bound_df):
        result = MergerTreeHandlerCSV.compute_satellites(three_halo_bound_df, rvir_factor=2.0)
        assert isinstance(result, dict)

    def test_host_has_satellite(self, three_halo_bound_df):
        result = MergerTreeHandlerCSV.compute_satellites(three_halo_bound_df, rvir_factor=2.0)
        assert 30 in result.get(10, set())

    def test_no_internal_state_change(self, three_halo_bound_df):
        original = three_halo_bound_df.copy()
        _ = MergerTreeHandlerCSV.compute_satellites(three_halo_bound_df, rvir_factor=2.0)
        assert three_halo_bound_df.equals(original)
