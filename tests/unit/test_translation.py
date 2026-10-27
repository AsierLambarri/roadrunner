import numpy as np
import pandas as pd
import pytest

from roadrunner._mcf_types import AssignmentResult, SnapshotData
from roadrunner._defaults import LOCAL_IDX, UNBOUND
from roadrunner.clustering.sparse import SparseCSC
from roadrunner.physics.halo_ensemble import HaloEnsemble
from roadrunner.physics.halo_model import HaloModel
from roadrunner.physics.potentials import KeplerPotential
from roadrunner.postprocessing.tracking.assembly import AssemblyTracker
from roadrunner.postprocessing.tracking.birth import BirthTracker
from roadrunner.postprocessing.tracking.records import BirthChanges
from roadrunner.physics.constants import DYN_TIME_FACTOR
from roadrunner.pipeline.translation import (
    build_reduction_input,
    responsibilities_from_sim,
    responsibilities_to_sim,
    update_assembly_tracker,
    update_birth_tracker,
)


def _make_snap_data(n=10, start_id=0):
    return SnapshotData(
        index=np.arange(start_id, start_id + n, dtype=np.uint64),
        mass=np.ones(n, dtype=np.float64),
        position=np.zeros((n, 3), dtype=np.float64),
        velocity=np.zeros((n, 3), dtype=np.float64),
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


def _halo(sid=1, n_bound=0, xcen=(0.0, 0.0, 0.0)):
    inner = KeplerPotential(M=1e10, G=4.3e-6)
    h = HaloModel(
        inner, np.array(xcen, dtype=np.float64),
        np.zeros(3, dtype=np.float64), 50.0,
        sub_tree_id=sid, redshift=0.0,
    )
    if n_bound > 0:
        h.set_bound(
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


def _bound_halo(sub_tree_id, rows, dynamical_times):
    """Make a halo whose bound particles are ``rows``, with these dynamical times."""
    halo = _halo(sid=sub_tree_id)
    halo.set_bound(
        np.asarray(rows, dtype=np.int64),
        np.full(len(rows), -0.5, dtype=np.float32),
        np.asarray(dynamical_times, dtype=np.float32),
    )
    return halo


def _responsibility_result(particle_df, columns):
    """Make an assignment result whose responsibilities are ``columns``.

    ``columns`` maps each galaxy to its ``(rows, values)``.
    """
    responsibilities = SparseCSC(
        [np.asarray(rows) for rows, _ in columns.values()],
        [np.asarray(values, dtype=np.float32) for _, values in columns.values()],
        column_id=np.array(list(columns)),
    )
    return AssignmentResult(
        particle_df=particle_df,
        responsibilities=responsibilities,
        fitted_parameters={},
        statistics={},
    )


def _born_hosts(changes):
    """The birth host each particle was first seen in, as a dict."""
    return dict(zip(changes.born_ids.tolist(), changes.born_hosts.tolist()))


class TestUpdateBirthTracker:
    def test_receives_sim_ids(self):
        # one galaxy each: the particles are decided at once, under their
        # simulation IDs
        snap = _make_snap_data(n=3, start_id=100)
        result = _responsibility_result(
            pd.DataFrame(), {100: ([0], [1.0]), 103: ([1], [1.0]), 105: ([2], [1.0])}
        )
        ensemble = HaloEnsemble([
            _bound_halo(100, [0], [0.1]),
            _bound_halo(103, [1], [0.2]),
            _bound_halo(105, [2], [0.3]),
        ])
        birth_tracker = BirthTracker()
        changes = update_birth_tracker(
            birth_tracker, snap_id=0, snap_data=snap, result=result, ensemble=ensemble
        )
        assert birth_tracker._last_snapshot == 0
        assert _born_hosts(changes) == {100: 100, 101: 103, 102: 105}

    def test_particles_without_a_responsibility_are_born_unbound(self):
        snap = _make_snap_data(n=4, start_id=200)
        result = _responsibility_result(pd.DataFrame(), {201: ([1], [1.0])})
        birth_tracker = BirthTracker()
        changes = update_birth_tracker(
            birth_tracker,
            snap_id=0,
            snap_data=snap,
            result=result,
            ensemble=HaloEnsemble([_bound_halo(201, [1], [0.1])]),
        )
        assert _born_hosts(changes) == {
            200: UNBOUND, 201: 201, 202: UNBOUND, 203: UNBOUND,
        }

    def _soft_case(self):
        """Make a snapshot, a result and an ensemble with one split particle."""
        # Particle 400 is split 0.7/0.3 between galaxies 500 and 501, 401 is
        # fully in 501, and 402 has no responsibility (unbound).
        snap = _make_snap_data(n=3, start_id=400)
        particle_df = pd.DataFrame({
            "array_index": [0, 1, 2], "Sub_tree_id": [500, 501, -1],
        })
        result = _responsibility_result(
            particle_df, {500: ([0], [0.7]), 501: ([0, 1], [0.3, 1.0])}
        )
        ensemble = HaloEnsemble([
            _bound_halo(500, [0], [0.1]),
            _bound_halo(501, [0, 1], [0.3, 0.3]),
        ])
        return snap, result, ensemble

    def test_soft_evidence_and_dynamical_times_from_responsibilities(self):
        snap, result, ensemble = self._soft_case()
        birth_tracker = BirthTracker()
        changes = update_birth_tracker(
            birth_tracker, snap_id=0, snap_data=snap, result=result, ensemble=ensemble
        )
        # 401 and 402 have a single possible host and are decided at once;
        # 400 opens a window over its two hosts, as long as the mean of its
        # dynamical times in them, weighted by the responsibilities.
        assert birth_tracker.open_ids().tolist() == [400]
        evidence_shares = {
            host: round(np.exp(log_evidence), 6)
            for host, log_evidence in birth_tracker.evidence(400).items()
        }
        assert evidence_shares == {500: 0.7, 501: 0.3}
        expected_window = (0.7 * 0.1 + 0.3 * 0.3) * DYN_TIME_FACTOR
        assert birth_tracker.window(400) == pytest.approx(expected_window, rel=1e-6)
        assert _born_hosts(changes) == {400: 500, 401: 501, 402: -1}

    def test_the_galaxies_alive_are_the_responsibility_columns(self):
        snap, result, ensemble = self._soft_case()
        birth_tracker = BirthTracker()
        update_birth_tracker(
            birth_tracker, snap_id=0, snap_data=snap, result=result, ensemble=ensemble
        )
        later_snap = _make_snap_data(n=3, start_id=400)
        later_snap.time = snap.time + 0.01
        # 501 is gone
        later_result = _responsibility_result(
            pd.DataFrame({"array_index": [0, 1, 2], "Sub_tree_id": [500, 500, -1]}),
            {500: ([0, 1], [1.0, 1.0])},
        )
        update_birth_tracker(
            birth_tracker,
            snap_id=1,
            snap_data=later_snap,
            result=later_result,
            ensemble=HaloEnsemble([_bound_halo(500, [0, 1], [0.1, 0.1])]),
        )
        # 400's window closed: a candidate merged away
        assert birth_tracker.open_ids().size == 0
        births = birth_tracker.finalized()
        decided = dict(zip(births.particle_index.tolist(), births.birth_id.tolist()))
        assert decided[400] == 500

    def test_settled_particles_are_not_passed_on(self):
        snap, result, ensemble = self._soft_case()
        birth_tracker = BirthTracker()
        update_birth_tracker(
            birth_tracker, snap_id=0, snap_data=snap, result=result, ensemble=ensemble
        )
        later_snap = _make_snap_data(n=3, start_id=400)
        later_snap.time = snap.time + 0.01
        seen = []
        original_update = birth_tracker.update

        def recording_update(memberships):
            """Record the memberships the tracker receives, then update as usual."""
            seen.append(memberships)
            return original_update(memberships)

        birth_tracker.update = recording_update
        update_birth_tracker(
            birth_tracker,
            snap_id=1,
            snap_data=later_snap,
            result=result,
            ensemble=ensemble,
        )
        # 401 and 402 are settled: only 400, still open, reaches the tracker
        # (with its two pairs)
        assert seen[0].particle_ids.tolist() == [400, 400]


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
        update_assembly_tracker(
            at,
            snap_id=0,
            snap_data=snap,
            result=result,
            satellites=satellites,
            changes=BirthChanges.nothing(),
        )
        current = at.current()
        assert 500 in current
        assert 501 in current

    def test_with_birth_tracker(self):
        snap = _make_snap_data(n=10, start_id=300)
        result = _responsibility_result(
            pd.DataFrame({"array_index": [0, 1], "Sub_tree_id": [300, 300]}),
            {300: ([0, 1], [1.0, 1.0])},
        )
        birth_tracker = BirthTracker()
        changes = update_birth_tracker(
            birth_tracker,
            snap_id=0,
            snap_data=snap,
            result=result,
            ensemble=HaloEnsemble([_bound_halo(300, [0, 1], [0.1, 0.1])]),
        )
        assembly_tracker = AssemblyTracker(n_sat_history=2)
        update_assembly_tracker(
            assembly_tracker,
            snap_id=0,
            snap_data=snap,
            result=result,
            satellites={300: set()},
            changes=changes,
        )
        assert assembly_tracker.current()[300] == {300, 301}


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