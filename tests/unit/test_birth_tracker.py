import numpy as np
import pandas as pd

from roadrunner.postprocessing.tracking.birth import BirthTracker


class TestBirthTracker:
    def test_single_particle_across_snapshots(self):
        tracker = BirthTracker(factor=5)
        # snapshot 0: particle 1 appears in galaxy 10 with timescale 2.0
        tracker.update(t_snap=0.0, snapshot_id=0,
                       particle_ids=np.array([1]),
                       host_ids=np.array([10]),
                       timescales=np.array([2.0]))
        # not yet finalized: t=0, deadline = 0 + 5*2 = 10
        assert len(tracker._active) == 1
        assert len(tracker._finalized) == 0

        # snapshot at t=5: still within window
        tracker.update(t_snap=5.0, snapshot_id=1,
                       particle_ids=np.array([1]),
                       host_ids=np.array([10]),
                       timescales=np.array([2.0]))
        assert len(tracker._active) == 1
        assert len(tracker._finalized) == 0

        # snapshot at t=12: past deadline -> finalized
        tracker.update(t_snap=12.0, snapshot_id=2,
                       particle_ids=np.array([1]),
                       host_ids=np.array([10]),
                       timescales=np.array([2.0]))
        assert len(tracker._active) == 0
        assert len(tracker._finalized) == 1
        assert tracker._finalized[1].birth_id == 10

    def test_competition_between_hosts(self):
        tracker = BirthTracker(factor=5, window="gaussian")
        # particle 1 first seen in galaxy 10
        tracker.update(t_snap=0.0, snapshot_id=0,
                       particle_ids=np.array([1]),
                       host_ids=np.array([10]),
                       timescales=np.array([2.0]))
        # next snapshot: galaxy 20 competes
        tracker.update(t_snap=1.0, snapshot_id=1,
                       particle_ids=np.array([1]),
                       host_ids=np.array([20]),
                       timescales=np.array([2.0]))
        # deadline = 0 + (5-0.001)*2 ≈ 10, particle should still be active
        assert len(tracker._active) == 1
        # galaxy 20 has a more recent score, but galaxy 10 has the head start
        info = tracker._active[1]
        # With gaussian window at dt=1, tau=2: exp(-0.3*(1/2)^2) = exp(-0.075) ≈ 0.928
        # So both scores should be > 0
        assert 10 in info.counts
        assert 20 in info.counts

    def test_enforce_initial_hosts(self):
        tracker = BirthTracker(factor=5, enforce_initial_hosts=True, window="gaussian")
        tracker.update(t_snap=0.0, snapshot_id=0,
                       particle_ids=np.array([1]),
                       host_ids=np.array([10]),
                       timescales=np.array([2.0]))
        # galaxy 20 is NOT in the initial hosts -> score should NOT increase
        tracker.update(t_snap=1.0, snapshot_id=1,
                       particle_ids=np.array([1]),
                       host_ids=np.array([20]),
                       timescales=np.array([2.0]))
        info = tracker._active[1]
        # Only galaxy 10 should have a score > 0
        assert 10 in info.counts
        assert info.counts.get(20, 0) == 0.0

    def test_current_birth_map(self):
        tracker = BirthTracker(factor=5, window="gaussian")
        tracker.update(t_snap=0.0, snapshot_id=0,
                       particle_ids=np.array([1, 2]),
                       host_ids=np.array([10, 20]),
                       timescales=np.array([2.0, 2.0]))
        # Finalize particle 1 by advancing past its deadline
        tracker.update(t_snap=12.0, snapshot_id=1,
                       particle_ids=np.array([1, 2]),
                       host_ids=np.array([10, 20]),
                       timescales=np.array([2.0, 2.0]))
        bmap = tracker.current_birth_map()
        assert 10 in bmap
        assert 20 in bmap
        # particle 1 should be in galaxy 10's birth map
        assert 1 in bmap[10]
        # particle 2 should still be active (or finalized in 20)
        assert (20 in bmap and 2 in bmap[20]) or (2 in bmap.get(20, set()))

    def test_finalize_all_particles(self):
        tracker = BirthTracker(factor=5, window="gaussian")
        tracker.update(t_snap=0.0, snapshot_id=0,
                       particle_ids=np.array([1, 2, 3]),
                       host_ids=np.array([10, 20, 10]),
                       timescales=np.array([2.0, 3.0, 4.0]))
        df = tracker.finalize()
        assert len(df) == 3
        assert set(df["particle_index"]) == {1, 2, 3}
        assert all(df["birth_id"].isin([10, 20]))

    def test_empty_update(self):
        tracker = BirthTracker(factor=5, window="gaussian")
        tracker.update(t_snap=0.0, snapshot_id=0,
                       particle_ids=np.array([], dtype=int),
                       host_ids=np.array([], dtype=int),
                       timescales=np.array([], dtype=float))
        assert len(tracker._active) == 0
        assert len(tracker._finalized) == 0

    def test_all_particles_finalized_immediately(self):
        tracker = BirthTracker(factor=5, window="gaussian")
        # tau = 0 -> deadline = t + 0 = instantly finalized
        tracker.update(t_snap=0.0, snapshot_id=0,
                       particle_ids=np.array([1, 2]),
                       host_ids=np.array([10, 20]),
                       timescales=np.array([0.0, 0.0]))
        df = tracker.finalize()
        assert len(df) == 2

    def test_multiple_particles_different_timescales(self):
        tracker = BirthTracker(factor=5, window="gaussian")
        tracker.update(t_snap=0.0, snapshot_id=0,
                       particle_ids=np.array([1, 2]),
                       host_ids=np.array([10, 20]),
                       timescales=np.array([1.0, 10.0]))
        # deadline for 1: 0 + 5*1 = 5 (factor=5, but stored as 4.999)
        # deadline for 2: 0 + 5*10 = 50
        # At t=6, only particle 1 should be finalized
        tracker.update(t_snap=6.0, snapshot_id=1,
                       particle_ids=np.array([1, 2]),
                       host_ids=np.array([10, 20]),
                       timescales=np.array([1.0, 10.0]))
        assert 1 in tracker._finalized
        assert 2 in tracker._active

    def test_serialization_round_trip(self):
        tracker = BirthTracker(factor=5, window="gaussian")
        tracker.update(t_snap=0.0, snapshot_id=0,
                       particle_ids=np.array([1, 2]),
                       host_ids=np.array([10, 20]),
                       timescales=np.array([2.0, 3.0]))
        state = tracker._get_state()
        tracker2 = BirthTracker(factor=5, window="gaussian")
        tracker2._set_state(state)
        assert len(tracker2._active) == len(tracker._active)
        assert tracker2._last_snapshot == tracker._last_snapshot
        assert tracker2.factor == tracker.factor
        # Finalize both
        df1 = tracker.finalize()
        df2 = tracker2.finalize()
        assert len(df1) == len(df2)
