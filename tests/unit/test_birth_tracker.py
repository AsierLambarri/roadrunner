import numpy as np

from roadrunner.postprocessing.tracking.birth import ActiveParticleInfo, BirthTracker, FinalizedInfo


def _update(tracker, t, snap, pids, hosts, taus, weights=None):
    tracker.update(t_snap=t, snapshot_id=snap, particle_ids=np.array(pids),
                   host_ids=np.array(hosts), timescales=np.array(taus, dtype=float),
                   weights=None if weights is None else np.array(weights))


def _active(tracker):
    return set(tracker._act["pid"].tolist())


def _finalized(tracker):
    fin = tracker._concat_finalized(tracker._fin_chunks)
    return dict(zip(fin["pid"].tolist(), fin["birth_id"].tolist()))


def _scores(tracker, pid):
    m = tracker._ev["pid"] == pid
    return dict(zip(tracker._ev["host"][m].tolist(), tracker._ev["score"][m].tolist()))


class TestBirthTracker:
    def test_window_then_finalize(self):
        tracker = BirthTracker(factor=5)
        _update(tracker, 0.0, 0, [1], [10], [2.0])     # deadline = 0 + 4.999*2 ≈ 10
        assert _active(tracker) == {1} and _finalized(tracker) == {}
        _update(tracker, 5.0, 1, [1], [10], [2.0])
        assert _active(tracker) == {1} and _finalized(tracker) == {}
        _update(tracker, 12.0, 2, [1], [10], [2.0])
        assert _active(tracker) == set() and _finalized(tracker) == {1: 10}

    def test_competition_between_hosts(self):
        tracker = BirthTracker(factor=5, window="gaussian")
        _update(tracker, 0.0, 0, [1], [10], [2.0])
        _update(tracker, 1.0, 1, [1], [20], [2.0])
        scores = _scores(tracker, 1)
        assert set(scores) == {10, 20}
        assert scores[10] == 1.0 and 0 < scores[20] < 1.0      # 10 has the head start
        assert tracker.current_birth_map()[10] == {1}

    def test_enforce_initial_hosts_restricts_the_competition(self):
        # Born with responsibilities 0.6 / 0.4 in galaxies 10 and 20: only they compete.
        tracker = BirthTracker(factor=5, enforce_initial_hosts=True)
        _update(tracker, 0.0, 0, [1, 1], [10, 20], [2.0, 2.0], [0.6, 0.4])
        for s, t in enumerate((1.0, 2.0, 3.0), start=1):
            _update(tracker, t, s, [1, 1], [20, 30], [2.0, 2.0], [0.9, 0.1])
        assert set(_scores(tracker, 1)) == {10, 20}             # 30 was not there at birth
        tracker.finalize()
        assert _finalized(tracker) == {1: 20}                   # more evidence for 20

    def test_single_initial_host_is_decided_at_once(self):
        # With enforce_initial_hosts, nothing can compete with a lone initial
        # host (or with being born unbound): no window is opened.
        tracker = BirthTracker(factor=5, enforce_initial_hosts=True)
        _update(tracker, 0.0, 0, [1, 2], [10, -1], [2.0, 2.0])
        assert _active(tracker) == set()
        assert _finalized(tracker) == {1: 10, 2: -1}
        _update(tracker, 1.0, 1, [1], [30], [2.0])              # settled: later hosts ignored
        assert _finalized(tracker) == {1: 10, 2: -1}

    def test_current_birth_map(self):
        tracker = BirthTracker(factor=5)
        _update(tracker, 0.0, 0, [1, 2], [10, 20], [2.0, 20.0])
        _update(tracker, 12.0, 1, [1, 2], [10, 20], [2.0, 20.0])
        bmap = tracker.current_birth_map()
        assert bmap[10] == {1}        # finalised
        assert bmap[20] == {2}        # still active, under its leader

    def test_finalize_all_particles(self):
        tracker = BirthTracker(factor=5)
        _update(tracker, 0.0, 0, [1, 2, 3], [10, 20, 10], [2.0, 3.0, 4.0])
        df = tracker.finalize()
        assert len(df) == 3
        assert dict(zip(df["particle_index"], df["birth_id"])) == {1: 10, 2: 20, 3: 10}
        assert df["particle_index"].dtype == np.uint64 and df["birth_id"].dtype == np.int64

    def test_empty_update(self):
        tracker = BirthTracker(factor=5)
        _update(tracker, 0.0, 0, np.array([], int), np.array([], int), np.array([]))
        assert _active(tracker) == set() and _finalized(tracker) == {}
        assert tracker.finalize().empty

    def test_zero_timescale_finalized_immediately(self):
        tracker = BirthTracker(factor=5)
        _update(tracker, 0.0, 0, [1, 2], [10, 20], [0.0, 0.0])
        assert _finalized(tracker) == {1: 10, 2: 20}

    def test_different_timescales(self):
        tracker = BirthTracker(factor=5)
        _update(tracker, 0.0, 0, [1, 2], [10, 20], [1.0, 10.0])
        _update(tracker, 6.0, 1, [1, 2], [10, 20], [1.0, 10.0])  # deadlines ≈ 5 and 50
        assert _finalized(tracker) == {1: 10} and _active(tracker) == {2}

    def test_duplicate_pairs_are_merged(self):
        tracker = BirthTracker(factor=5)
        _update(tracker, 0.0, 0, [1, 1, 1], [10, 10, 20], [2.0, 2.0, 2.0], [0.25, 0.25, 0.4])
        assert _scores(tracker, 1) == {10: 0.5, 20: 0.4000000059604645}

    def test_serialization_round_trip(self):
        def run(tracker, snaps):
            for s in snaps:
                _update(tracker, 0.5 * s, s, [1, 2, 3, 3], [10, 20, 10, 20], [2.0, 3.0, 1.0, 1.0],
                        [1.0, 1.0, 0.3 + 0.1 * s, 0.7 - 0.1 * s])
        ref = BirthTracker(factor=5)
        run(ref, range(6))
        tracker = BirthTracker(factor=5)
        run(tracker, range(3))
        restored = BirthTracker(factor=5)
        restored._set_state(tracker._get_state())
        assert restored._last_snapshot == 2 and restored.factor == tracker.factor
        run(restored, range(3, 6))
        assert restored.current_birth_map() == ref.current_birth_map()
        assert restored.finalize().equals(ref.finalize())

    def test_legacy_checkpoint_loads(self):
        # A dict-based checkpoint from before the array rewrite.
        state = {
            "active": {np.uint64(1): ActiveParticleInfo(np.float32(0.0), np.int32(0), np.float32(2.0),
                                                        {10: np.float32(1.0), 20: 0.5}, {10})},
            "finalized": {np.uint64(2): FinalizedInfo(30, np.float32(0.0), np.int32(0), np.float32(1.0))},
            "heap": [(np.float32(9.998), np.uint64(1))],
            "birth_map": {30: [2]},
            "factor": 4.999, "last_snapshot": 0, "enforce_initial_hosts": False,
        }
        tracker = BirthTracker(factor=5)
        tracker._set_state(state)
        assert tracker.current_birth_map() == {10: {1}, 30: {2}}
        assert _scores(tracker, 1) == {10: 1.0, 20: 0.5}
        assert dict(zip(*tracker.finalize().values.T.tolist())) == {1: 10, 2: 30}


class TestBirthTrackerRecovery:
    """C02: a `_cheap_snapshot()` reference must survive later updates
    untouched, and finalised birth sets must accumulate across snapshots."""

    def test_cheap_snapshot_unaffected_by_later_update(self):
        tracker = BirthTracker(factor=5)
        _update(tracker, 0.0, 0, [1], [10], [2.0])
        ref = tracker._cheap_snapshot()
        before = {k: v.copy() for k, v in ref["ev"].items()}
        _update(tracker, 1.0, 1, [1], [20], [2.0])            # adds evidence for 20
        for k, v in before.items():
            np.testing.assert_array_equal(ref["ev"][k], v)
        assert 20 not in ref["ev"]["host"].tolist()

    def test_birth_sets_accumulate(self):
        tracker = BirthTracker(factor=5)
        _update(tracker, 0.0, 0, [1], [10], [0.0])
        first = tracker._birth_map[10]
        assert first == {1}
        _update(tracker, 1.0, 1, [2], [10], [0.0])
        assert tracker._birth_map[10] == {1, 2}
        assert first == {1}                                     # replaced, not edited
