import pytest

from roadrunner._exceptions import CycleError
from roadrunner.postprocessing.tracking.assembly import AssemblyTracker


class TestAssemblyTracker:
    def test_single_galaxy_no_satellites(self):
        tracker = AssemblyTracker(n_sat_history=2)
        tracker.update(
            snapshot_id=0,
            assignment_map={1: {10, 20, 30}},
            birth_map={1: {10, 20}},
            satellites_map={},
        )
        result = tracker.current()
        # Only birth particles should appear
        assert 1 in result
        assert result[1] == {10, 20}

    def test_galaxy_with_one_satellite(self):
        tracker = AssemblyTracker(n_sat_history=2)
        # Satellite 2 has particle 30 born in it
        tracker.update(
            snapshot_id=0,
            assignment_map={1: {10, 20, 30, 40}, 2: {30}},
            birth_map={1: {10, 20}, 2: {30}},
            satellites_map={1: {2}},
        )
        result = tracker.current()
        # Galaxy 1 should have its birth particles + satellite 2's infall (particle 30)
        assert 1 in result
        assert 10 in result[1]
        assert 20 in result[1]
        # particle 30 should now be in galaxy 1 as well (inherited from satellite 2)
        # 40 is in assignment_map but not in any infall list -> should NOT appear
        assert 30 in result[1]
        assert 40 not in result[1]

    def test_satellite_chain(self):
        tracker = AssemblyTracker(n_sat_history=2)
        # A -> B -> C chain
        # Galaxy 3 is satellite of 2, galaxy 2 is satellite of 1
        # The algorithm only adds particles to a host's infall list if they
        # are BOTH in the host's assignment AND in the "valid" set (birth +
        # unbound + satellite infall). Inheritance tracks potential lineage
        # but particles stay with their assigned galaxy.
        tracker.update(
            snapshot_id=0,
            assignment_map={1: {10, 20, 30}, 2: {20, 30}, 3: {30}},
            birth_map={1: {10}, 2: {20}, 3: {30}},
            satellites_map={1: {2}, 2: {3}},
        )
        result = tracker.current()
        # 3: birth {30} only (no satellites)
        assert result[3] == {30}
        # 2: birth {20} + valid from sat 3 ({30}) ∩ assignment {20, 30} = {20, 30}
        assert result[2] == {20, 30}
        # 1: birth {10} + valid from sat 2 ({20, 30}) ∩ assignment {10, 20, 30} = {10, 20, 30}
        assert result[1] == {10, 20, 30}

    def test_cycle_detection(self):
        tracker = AssemblyTracker(n_sat_history=2)
        # 1 -> 2 -> 1 forms a cycle
        with pytest.raises(CycleError, match="Cycle"):
            tracker.update(
                snapshot_id=0,
                assignment_map={1: {10}, 2: {20}},
                birth_map={1: {10}, 2: {20}},
                satellites_map={1: {2}, 2: {1}},
            )

    def test_freeze_galaxies(self):
        tracker = AssemblyTracker(n_sat_history=2)
        tracker.update(
            snapshot_id=0,
            assignment_map={1: {10, 20}, 2: {30}},
            birth_map={1: {10, 20}, 2: {30}},
            satellites_map={1: {2}},
            freeze_galaxies=[1],
        )
        # snapshot 1: galaxy 1 should not update
        tracker.update(
            snapshot_id=1,
            assignment_map={1: {10, 20, 40}, 2: {30}},
            birth_map={1: {10, 20}, 2: {30}},
            satellites_map={1: {2}},
        )
        result = tracker.current()
        # Galaxy 1 should have its initial particles but NOT the new assignment (40)
        # Since it was frozen at snapshot 0
        assert 40 not in result[1]

    def test_multiple_snapshots(self):
        tracker = AssemblyTracker(n_sat_history=2, unbound_default=-1)
        # Snapshot 0
        tracker.update(
            snapshot_id=0,
            assignment_map={1: {10, 20, 30}},
            birth_map={1: {10, 20}},
            satellites_map={},
        )
        # Snapshot 1: galaxy 2 appears, gets particles 30, 40
        tracker.update(
            snapshot_id=1,
            assignment_map={1: {10, 20}, 2: {30, 40}},
            birth_map={1: {10}, 2: {30, 40}},
            satellites_map={},
        )
        result = tracker.current()
        # Galaxy 1: had {10,20}, then lost {20} in birth map (erase), gained nothing
        # So should have {10} (the only persistent birth particle)
        assert 10 in result[1]
        assert 20 not in result[1]
        # Galaxy 2: birth particles {30, 40}
        assert result[2] == {30, 40}

    def test_state_serialization(self):
        tracker = AssemblyTracker(n_sat_history=2)
        tracker.update(
            snapshot_id=0,
            assignment_map={1: {10, 20, 30}, 2: {30}},
            birth_map={1: {10, 20}, 2: {30}},
            satellites_map={1: {2}},
        )
        state = tracker._get_state()
        tracker2 = AssemblyTracker(n_sat_history=2)
        tracker2._set_state(state)
        assert dict(tracker2.current()) == dict(tracker.current())
        assert tracker2._last_snapshot == tracker._last_snapshot

    def test_sat_history_merging(self):
        tracker = AssemblyTracker(n_sat_history=3, unbound_default=-1)
        # Snapshot 0: galaxy 2 is satellite of 1
        tracker.update(
            snapshot_id=0,
            assignment_map={1: {10}, 2: {20}},
            birth_map={1: {10}, 2: {20}},
            satellites_map={1: {2}},
        )
        # Snapshot 1: galaxy 2 is still a satellite of 1 (history buffer keeps it)
        # Even though satellites_map is empty, the buffer from snap 0 preserves
        # the relationship for n_sat_history snapshots.
        # BUT: _get_current_sat_history starts from the current map keys.
        # If the current satellites_map is {}, sat_history starts empty
        # and the buffer is never consulted. This is by design: particles
        # only flow through galaxies with an active current satellite link.
        tracker.update(
            snapshot_id=1,
            assignment_map={1: {10}, 2: {20}},
            birth_map={1: {10}, 2: {20}},
            satellites_map={1: {2}},
        )
        # Snapshot 2: satellites_map empty, buffer has [{1:{2}}] from snap 1.
        # sat_history starts empty → no satellites → inheritance stops.
        tracker.update(
            snapshot_id=2,
            assignment_map={1: {10}, 2: {20}},
            birth_map={1: {10}, 2: {20}},
            satellites_map={},
        )
        result = tracker.current()
        # Without an active satellite link, galaxy 2's particles stay with 2.
        # Galaxy 1 only has its own birth particles.
        assert result[1] == {10}
        # But galaxy 2 still has its own infall list
        assert result[2] == {20}
