import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")

from roadrunner.pipeline.reduction import ReductionConfig, reduce_snapshot
from roadrunner._mcf_types import AssignmentResult, SnapshotData


_RNG = np.random.default_rng(42)


def _make_snap_data(n_particles=200, redshift=0.0, time=13.8):
    positions = _RNG.uniform(-50, 50, (n_particles, 3)).astype(np.float64)
    velocities = _RNG.uniform(-10, 10, (n_particles, 3)).astype(np.float64)
    masses = _RNG.uniform(1e4, 1e6, n_particles).astype(np.float64)
    return SnapshotData(
        index=np.arange(n_particles, dtype=np.uint64),
        mass=masses,
        position=positions,
        velocity=velocities,
        redshift=redshift,
        time=time,
    )


def _make_snap_df(accretion_id=100, satellite_ids=None):
    if satellite_ids is None:
        satellite_ids = [1, 2, 3]
    rows = []
    for sid in satellite_ids:
        rows.append({
            "Sub_tree_id": sid,
            "host_id": accretion_id,
            "position_x": _RNG.uniform(-10, 10),
            "position_y": _RNG.uniform(-10, 10),
            "position_z": _RNG.uniform(-10, 10),
            "velocity_x": _RNG.uniform(-5, 5),
            "velocity_y": _RNG.uniform(-5, 5),
            "velocity_z": _RNG.uniform(-5, 5),
            "mass": _RNG.uniform(1e8, 1e10),
            "virial_radius": 30.0,
            "scale_radius": 10.0,
            "Redshift": 0.0,
            "distance_to_acc_id": _RNG.uniform(5, 50),
        })
    rows.append({
        "Sub_tree_id": accretion_id,
        "host_id": -1,
        "position_x": 0.0,
        "position_y": 0.0,
        "position_z": 0.0,
        "velocity_x": 0.0,
        "velocity_y": 0.0,
        "velocity_z": 0.0,
        "mass": 1e11,
        "virial_radius": 100.0,
        "scale_radius": 30.0,
        "Redshift": 0.0,
        "distance_to_acc_id": 0.0,
    })
    return pd.DataFrame(rows)


def _make_result(fitted_parameters=None):
    if fitted_parameters is None:
        fitted_parameters = {}
    return AssignmentResult(
        particle_df=pd.DataFrame(),
        responsibilities=None,
        fitted_parameters=fitted_parameters,
        statistics={},
    )


def _make_galaxy_dicts(n_particles, satellite_ids):
    rng = np.random.default_rng(99)
    galaxy_particles = {}
    galaxy_bound = {}
    per_sat = n_particles // (len(satellite_ids) + 2)
    offset = 0
    for sid in satellite_ids:
        idx = np.arange(offset, offset + per_sat, dtype=np.int64)
        galaxy_particles[sid] = idx
        galaxy_bound[sid] = idx
        offset += per_sat
    return galaxy_particles, galaxy_bound


class TestReductionConfig:
    def test_defaults(self):
        cfg = ReductionConfig(accretion_id=100)
        assert cfg.accretion_id == 100
        assert cfg.halo_model == "kepler"
        assert cfg.n_los == 15

    def test_frozen(self):
        cfg = ReductionConfig(accretion_id=1)
        with pytest.raises(AttributeError):
            cfg.accretion_id = 2


class TestReduceSnapshot:
    def test_properties_columns(self):
        n = 200
        snap_data = _make_snap_data(n_particles=n)
        satellite_ids = [1, 2]
        snap_df = _make_snap_df(satellite_ids=satellite_ids)
        result = _make_result()
        galaxy_particles, galaxy_bound = _make_galaxy_dicts(n, satellite_ids)
        cfg = ReductionConfig(accretion_id=100)

        props, dyn = reduce_snapshot(
            snap_data, snap_df, result, galaxy_particles, galaxy_bound, cfg,
        )
        assert isinstance(props, pd.DataFrame)
        assert isinstance(dyn, pd.DataFrame)
        expected_cols = [
            "Sub_tree_id", "mb_host_id",
            "position_x", "position_y", "position_z",
            "velocity_x", "velocity_y", "velocity_z",
            "Mtot", "r20", "rh", "r80", "Rhp",
            "sigma", "sigma_los", "r_t",
        ]
        assert list(props.columns) == expected_cols
        expected_dyn_cols = ["Sub_tree_id", "mstar", "f_bound", "sigma50", "dynstate"]
        assert list(dyn.columns) == expected_dyn_cols

    def test_empty_galaxy_particles(self):
        n = 50
        snap_data = _make_snap_data(n_particles=n)
        snap_df = _make_snap_df(satellite_ids=[])
        result = _make_result()
        cfg = ReductionConfig(accretion_id=100)

        props, dyn = reduce_snapshot(
            snap_data, snap_df, result, {}, {}, cfg,
        )
        assert len(props) == 0
        assert len(dyn) == 0

    def test_with_fitted_centers(self):
        n = 200
        snap_data = _make_snap_data(n_particles=n)
        satellite_ids = [5]
        snap_df = _make_snap_df(satellite_ids=satellite_ids)
        fitted_parameters = {
            5: {"mean": np.array([1.0, 2.0, 3.0, 0.1, -0.2, 0.3])},
        }
        result = _make_result(fitted_parameters=fitted_parameters)
        galaxy_particles, galaxy_bound = _make_galaxy_dicts(n, satellite_ids)
        cfg = ReductionConfig(accretion_id=100)

        props, _ = reduce_snapshot(
            snap_data, snap_df, result, galaxy_particles, galaxy_bound, cfg,
        )
        assert len(props) > 0