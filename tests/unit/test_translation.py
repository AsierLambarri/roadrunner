import numpy as np
import pandas as pd
import pytest

from roadrunner._mcf_types import AssignmentResult, SnapshotData
from roadrunner.clustering.sparse import SparseCSC
from roadrunner.physics.halo_ensemble import HaloEnsemble
from roadrunner.physics.halo_model import HaloModel
from roadrunner.physics.potentials import KeplerPotential
from roadrunner.postprocessing.tracking.assembly import AssemblyTracker
from roadrunner.postprocessing.tracking.birth import BirthTracker
from roadrunner.pipeline.translation import (
    build_reduction_input,
    detect_newborns,
    responsibilities_from_sim,
    responsibilities_to_sim,
    update_assembly_tracker,
    update_birth_tracker,
)


def _make_snap_data(n=10, start_id=0):
    return SnapshotData(
        indices=np.arange(start_id, start_id + n, dtype=np.uint64),
        masses=np.ones(n, dtype=np.float64),
        positions=np.zeros((n, 3), dtype=np.float64),
        velocities=np.zeros((n, 3), dtype=np.float64),
        redshift=0.0,
        time=1.0,
    )


def _make_csc(row_ids, col_ids, values, n_rows=None):
    unique_rows = np.unique(np.concatenate(row_ids)) if row_ids else np.array([], dtype=np.int64)
    if n_rows is not None:
        unique_rows = np.arange(n_rows, dtype=np.int64)

    col_indices = [np.asarray(r, dtype=np.int64) for r in row_ids]
    col_values = [np.asarray(v, dtype=np.float32) for v in values]
    column_id = np.asarray(col_ids, dtype=np.int64)

    return SparseCSC(col_indices, col_values, column_id=column_id)


class TestResponsibilitiesToSim:
    def test_none_returns_none(self):
        assert responsibilities_to_sim(None, _make_snap_data()) is None

    def test_empty_csc_returns_none(self):
        csc = _make_csc(row_ids=[[0]], col_ids=[1], values=[[1.0]])
        assert responsibilities_to_sim(csc, _make_snap_data()) is not None
        assert responsibilities_to_sim(None, _make_snap_data()) is None

    def test_identity_mapping(self):
        snap = _make_snap_data(n=5, start_id=0)
        csc = _make_csc(
            row_ids=[[0, 2], [1, 3]],
            col_ids=[10, 20],
            values=[[0.5, 0.3], [0.7, 0.9]],
        )
        result = responsibilities_to_sim(csc, snap)
        assert result is not None
        assert len(result.column_indices) == 2
        np.testing.assert_array_equal(result.column_indices[0], [0, 2])
        np.testing.assert_allclose(result.column_values[0], [0.5, 0.3], atol=1e-6)

    def test_offset_mapping(self):
        snap = _make_snap_data(n=5, start_id=100)
        csc = _make_csc(
            row_ids=[[0, 1], [2]],
            col_ids=[10, 20],
            values=[[0.8, 0.2], [0.6]],
        )
        result = responsibilities_to_sim(csc, snap)
        assert result is not None
        np.testing.assert_array_equal(result.column_indices[0], [100, 101])
        np.testing.assert_array_equal(result.column_indices[1], [102])


class TestResponsibilitiesFromSim:
    def test_none_returns_none(self):
        assert responsibilities_from_sim(None, _make_snap_data()) is None

    def test_empty_csc_returns_none(self):
        csc = _make_csc(row_ids=[[0]], col_ids=[1], values=[[1.0]])
        assert responsibilities_from_sim(csc, _make_snap_data()) is not None
        assert responsibilities_from_sim(None, _make_snap_data()) is None

    def test_offset_mapping(self):
        snap = _make_snap_data(n=5, start_id=100)
        csc_sim = _make_csc(
            row_ids=[[100, 101], [102]],
            col_ids=[10, 20],
            values=[[0.8, 0.2], [0.6]],
        )
        result = responsibilities_from_sim(csc_sim, snap)
        assert result is not None
        np.testing.assert_array_equal(result.column_indices[0], [0, 1])
        np.testing.assert_array_equal(result.column_indices[1], [2])


class TestRoundTrip:
    def test_roundtrip_sim_and_back(self):
        snap = _make_snap_data(n=10, start_id=50)
        csc_arr = _make_csc(
            row_ids=[[0, 3, 5], [2, 7]],
            col_ids=[1, 2],
            values=[[0.4, 0.3, 0.1], [0.5, 0.8]],
        )
        csc_sim = responsibilities_to_sim(csc_arr, snap)
        assert csc_sim is not None

        csc_back = responsibilities_from_sim(csc_sim, snap)
        assert csc_back is not None

        np.testing.assert_array_equal(
            sorted(csc_back.column_indices[0].tolist()),
            sorted([0, 3, 5]),
        )
        np.testing.assert_array_equal(
            sorted(csc_back.column_indices[1].tolist()),
            sorted([2, 7]),
        )

    def test_roundtrip_preserves_values(self):
        snap = _make_snap_data(n=8, start_id=200)
        csc_arr = _make_csc(
            row_ids=[[1, 4], [0, 2]],
            col_ids=[10, 20],
            values=[[0.9, 0.1], [0.6, 0.4]],
        )
        csc_sim = responsibilities_to_sim(csc_arr, snap)
        csc_back = responsibilities_from_sim(csc_sim, snap)

        np.testing.assert_allclose(csc_back.column_values[0], [0.9, 0.1], atol=1e-6)
        np.testing.assert_allclose(csc_back.column_values[1], [0.6, 0.4], atol=1e-6)


class TestDetectNewborns:
    def test_none_previous_resp(self):
        newborn = detect_newborns(None, 5)
        np.testing.assert_array_equal(newborn, [0, 1, 2, 3, 4])

    def test_none_previous_resp_dtype(self):
        newborn = detect_newborns(None, 3)
        assert newborn.dtype == np.uint64

    def test_with_previous_resp(self):
        csc = _make_csc(
            row_ids=[[0, 2, 4]],
            col_ids=[1],
            values=[[0.5, 0.3, 0.2]],
        )
        newborn = detect_newborns(csc, 6)
        np.testing.assert_array_equal(sorted(newborn.tolist()), [1, 3, 5])

    def test_all_existing(self):
        csc = _make_csc(
            row_ids=[[0, 1, 2]],
            col_ids=[1],
            values=[[0.3, 0.3, 0.4]],
        )
        newborn = detect_newborns(csc, 3)
        assert len(newborn) == 0

    def test_empty_previous_resp(self):
        csc = SparseCSC(
            [np.array([], dtype=np.int64)],
            [np.array([], dtype=np.float32)],
            column_id=np.array([1], dtype=np.int64),
        )
        newborn = detect_newborns(csc, 4)
        np.testing.assert_array_equal(sorted(newborn.tolist()), [0, 1, 2, 3])


def _halo(sid=1, n_bound=0, xcen=(0.0, 0.0, 0.0)):
    inner = KeplerPotential(M=1e10, G=4.3e-6)
    h = HaloModel(
        inner, np.array(xcen, dtype=np.float64),
        np.zeros(3, dtype=np.float64), 50.0,
        sub_tree_id=sid, redshift=0.0,
    )
    if n_bound > 0:
        h.set_boundness(
            np.arange(n_bound, dtype=np.uint64),
            np.full(n_bound, 0.5, dtype=np.float32),
            np.full(n_bound, 0.1, dtype=np.float32),
        )
    return h


def _make_result_with_df(df=None, fitted_parameters=None):
    if df is None:
        df = pd.DataFrame()
    return AssignmentResult(
        particle_df=df,
        responsibilities=None,
        fitted_parameters=fitted_parameters or {},
        statistics={},
    )


class TestUpdateBirthTracker:
    def test_receives_sim_ids(self):
        snap = _make_snap_data(n=10, start_id=100)
        df = pd.DataFrame({
            "array_index": [0, 3, 5],
            "Sub_tree_id": [100, 103, 105],
            "timescale": [0.1, 0.2, 0.3],
        })
        result = _make_result_with_df(df)
        bt = BirthTracker(factor=5, enforce_initial_hosts=False)
        update_birth_tracker(bt, snap_id=0, snap_data=snap, result=result)
        assert bt._last_snapshot == 0
        active_keys = list(bt._active.keys())
        assert set(active_keys) == {100, 103, 105}

    def test_without_timescale_column(self):
        snap = _make_snap_data(n=10, start_id=200)
        df = pd.DataFrame({
            "array_index": [1, 2],
            "Sub_tree_id": [201, 202],
        })
        result = _make_result_with_df(df)
        bt = BirthTracker(factor=5, enforce_initial_hosts=False)
        update_birth_tracker(bt, snap_id=0, snap_data=snap, result=result)
        assert len(bt._active) == 2


class TestUpdateAssemblyTracker:
    def test_assembly_tracker_receives_sim_ids(self):
        snap = _make_snap_data(n=10, start_id=500)
        df = pd.DataFrame({
            "array_index": [0, 1, 2, 3],
            "Sub_tree_id": [500, 500, 501, 501],
        })
        result = _make_result_with_df(df)
        at = AssemblyTracker(n_sat_history=2)
        satellites = {500: set(), 501: set()}
        update_assembly_tracker(at, snap_id=0, snap_data=snap, result=result, satellites=satellites)
        current = at.current()
        assert 500 in current
        assert 501 in current

    def test_with_birth_tracker(self):
        snap = _make_snap_data(n=10, start_id=300)
        df = pd.DataFrame({
            "array_index": [0, 1],
            "Sub_tree_id": [300, 300],
            "timescale": [0.1, 0.1],
        })
        result = _make_result_with_df(df)
        bt = BirthTracker(factor=5, enforce_initial_hosts=False)
        update_birth_tracker(bt, snap_id=0, snap_data=snap, result=result)
        satellites = {300: set()}
        at = AssemblyTracker(n_sat_history=2)
        update_assembly_tracker(at, snap_id=0, snap_data=snap, result=result,
                                satellites=satellites, birth_tracker=bt)
        current = at.current()
        assert 300 in current


class TestBuildReductionInput:
    def test_basic_intersection(self):
        n = 20
        snap = _make_snap_data(n=n, start_id=0)
        h1 = _halo(sid=10, n_bound=5, xcen=(0.0, 0.0, 0.0))
        h2 = _halo(sid=20, n_bound=3, xcen=(50.0, 0.0, 0.0))
        ensemble = HaloEnsemble([h1, h2])
        at = AssemblyTracker(n_sat_history=2)
        at._infall_lists = {10: {0, 1, 2, 3, 4}, 20: {0, 1, 2}}

        galaxy_particles, galaxy_bound = build_reduction_input(snap, ensemble, at)

        assert 10 in galaxy_particles
        assert 20 in galaxy_particles
        np.testing.assert_array_equal(sorted(galaxy_particles[10].tolist()), [0, 1, 2, 3, 4])
        np.testing.assert_array_equal(sorted(galaxy_particles[20].tolist()), [0, 1, 2])
        np.testing.assert_array_equal(sorted(galaxy_bound[10].tolist()), [0, 1, 2, 3, 4])
        np.testing.assert_array_equal(sorted(galaxy_bound[20].tolist()), [0, 1, 2])

    def test_particles_not_in_bound(self):
        n = 20
        snap = _make_snap_data(n=n, start_id=0)
        h1 = _halo(sid=10, n_bound=3)
        ensemble = HaloEnsemble([h1])
        at = AssemblyTracker(n_sat_history=2)
        at._infall_lists = {10: {0, 1, 5, 6}}

        galaxy_particles, galaxy_bound = build_reduction_input(snap, ensemble, at)

        assert 10 in galaxy_particles
        np.testing.assert_array_equal(sorted(galaxy_particles[10].tolist()), [0, 1, 5, 6])
        np.testing.assert_array_equal(sorted(galaxy_bound[10].tolist()), [0, 1])

    def test_unknown_galaxy_skipped(self):
        n = 20
        snap = _make_snap_data(n=n, start_id=0)
        h1 = _halo(sid=10, n_bound=3)
        ensemble = HaloEnsemble([h1])
        at = AssemblyTracker(n_sat_history=2)
        at._infall_lists = {10: {0, 1, 2}, 99: {3}}

        galaxy_particles, galaxy_bound = build_reduction_input(snap, ensemble, at)

        assert 10 in galaxy_particles
        assert 99 not in galaxy_particles

    def test_no_allowed_particles_skipped(self):
        n = 20
        snap = _make_snap_data(n=n, start_id=0)
        h1 = _halo(sid=10, n_bound=3)
        ensemble = HaloEnsemble([h1])
        at = AssemblyTracker(n_sat_history=2)
        at._infall_lists = {10: {100, 200}}

        galaxy_particles, galaxy_bound = build_reduction_input(snap, ensemble, at)

        assert 10 not in galaxy_particles

    def test_none_tracker_returns_all_bound(self):
        n = 20
        snap = _make_snap_data(n=n, start_id=0)
        h1 = _halo(sid=10, n_bound=5)
        h2 = _halo(sid=20, n_bound=3)
        ensemble = HaloEnsemble([h1, h2])

        galaxy_particles, galaxy_bound = build_reduction_input(snap, ensemble, None)

        assert 10 in galaxy_particles
        assert 20 in galaxy_particles
        np.testing.assert_array_equal(sorted(galaxy_particles[10].tolist()), [0, 1, 2, 3, 4])
        np.testing.assert_array_equal(sorted(galaxy_particles[20].tolist()), [0, 1, 2])
        np.testing.assert_array_equal(sorted(galaxy_bound[10].tolist()), [0, 1, 2, 3, 4])
        np.testing.assert_array_equal(sorted(galaxy_bound[20].tolist()), [0, 1, 2])

    def test_none_tracker_empty_halo_skipped(self):
        n = 20
        snap = _make_snap_data(n=n, start_id=0)
        h1 = _halo(sid=10, n_bound=0)
        h2 = _halo(sid=20, n_bound=3)
        ensemble = HaloEnsemble([h1, h2])

        galaxy_particles, galaxy_bound = build_reduction_input(snap, ensemble, None)

        assert 10 not in galaxy_particles
        assert 20 in galaxy_particles