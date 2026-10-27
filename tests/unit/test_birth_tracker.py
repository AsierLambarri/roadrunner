import numpy as np
import pytest

from roadrunner._defaults import BIRTH_LATE_PRIOR
from roadrunner.physics.constants import MAX_DYN_TIMESCALE
from roadrunner.postprocessing.tracking.birth import BirthTracker
from roadrunner.postprocessing.tracking.records import BirthChanges, Memberships

NAN = np.nan


def _memberships(
    time, snapshot, particles, hosts, dynamical_times, weights=None, alive=None
):
    """Build the memberships of one output from parallel lists, one entry per pair.

    Parameters
    ----------
    time : float
        Cosmic time of the output.
    snapshot : int
        Index of the snapshot.
    particles, hosts, dynamical_times : list
        The particle, the host and the dynamical time of each pair.
    weights : list of float, optional
        The weight of each pair; 1 for every pair by default.
    alive : list of int, optional
        The galaxies alive; the hosts that occur by default.

    Returns
    -------
    memberships : Memberships
    """
    hosts = np.array(hosts)
    return Memberships(
        time,
        snapshot,
        np.array(particles, dtype=np.uint64),
        hosts,
        np.ones(len(particles)) if weights is None else np.array(weights, dtype=float),
        np.array(dynamical_times, dtype=float),
        np.unique(hosts) if alive is None else np.array(alive),
    )


def _update(
    tracker, time, snapshot, particles, hosts, dynamical_times, weights=None, alive=None
):
    """Update a tracker with one output given as parallel lists (see `_memberships`)."""
    return tracker.update(
        _memberships(time, snapshot, particles, hosts, dynamical_times, weights, alive)
    )


def _open(tracker):
    """The set of particles whose window is open."""
    return set(tracker.open_ids().tolist())


def _finalized(tracker):
    """The births decided so far, as ``{particle: birth host}``."""
    births = tracker.finalized()
    return dict(zip(births["particle_index"].tolist(), births["birth_id"].tolist()))


def _persistence(time_step, window, budget=5.0):
    """The persistence ``lam = exp(-dt / tau_c)`` of the assignment noise."""
    return np.exp(-time_step * budget / window)


def _as_dict(keys, values):
    """Pair two arrays up as a dict."""
    return dict(zip(keys.tolist(), values.tolist()))


class TestDecidedAtOnce:
    def test_single_candidate(self):
        tracker = BirthTracker()
        _update(tracker, 0.0, 0, [1, 2], [10, -1], [0.2, NAN])
        assert _open(tracker) == set() and _finalized(tracker) == {1: 10, 2: -1}
        assert (tracker.finalize().p_birth == 1.0).all()

    def test_confident_first_output_is_final(self):
        # budget 5: a share of 1 / (1 + e^-5) = 0.9933 or more cannot be
        # overturned by the window
        tracker = BirthTracker(budget=5)
        _update(
            tracker, 0.0, 0,
            particles=[1, 1, 2, 2],
            hosts=[10, 20, 10, 20],
            dynamical_times=[0.2, 0.2, 0.2, 0.2],
            weights=[0.9934, 0.0066, 0.9932, 0.0068],
        )
        assert _finalized(tracker) == {1: 10} and _open(tracker) == {2}
        # settled: later outputs are ignored
        _update(tracker, 0.05, 1, [1, 1], [10, 20], [0.2, 0.2], [0.0, 1.0])
        assert _finalized(tracker)[1] == 10

    def test_negative_threshold_decides_nothing_at_once(self):
        tracker = BirthTracker(min_initial_resp=-1)
        _update(tracker, 0.0, 0, [1], [10], [0.2])
        assert _open(tracker) == {1} and _finalized(tracker) == {}

    def test_settled_particles_are_the_finalised_ones(self):
        tracker = BirthTracker()
        _update(
            tracker, 0.0, 0,
            particles=[1, 2, 2],
            hosts=[10, 20, 30],
            dynamical_times=[0.2, 0.4, 0.4],
            weights=[1.0, 0.6, 0.4],
        )
        settled = tracker.settled(np.array([1, 2, 3], dtype=np.uint64))
        assert settled.tolist() == [True, False, False]


class TestWindow:
    def test_window_is_the_weighted_mean_of_the_dynamical_times(self):
        tracker = BirthTracker()
        _update(tracker, 0.0, 0, [1, 1], [10, 20], [0.2, 0.4], [0.6, 0.4])
        assert tracker.window(1) == pytest.approx(0.6 * 0.2 + 0.4 * 0.4)
        # the prior is the log of the first-output share
        assert tracker.evidence(1) == {
            10: pytest.approx(np.log(0.6)),
            20: pytest.approx(np.log(0.4)),
        }

    def test_missing_and_zero_times_are_left_out_of_the_mean(self):
        tracker = BirthTracker()
        _update(
            tracker, 0.0, 0,
            particles=[1, 1, 2, 2, 3, 3],
            hosts=[10, 20, 10, 20, 10, 20],
            dynamical_times=[0.3, NAN, 0.3, 0.0, NAN, 0.0],
            weights=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        )
        assert tracker.window(1) == pytest.approx(0.3)
        assert tracker.window(2) == pytest.approx(0.3)
        assert tracker.window(3) == MAX_DYN_TIMESCALE  # nothing to average

    def test_window_closes_after_the_lag(self):
        tracker = BirthTracker()
        # L = 0.2
        _update(tracker, 0.0, 0, [1, 1], [10, 20], [0.2, 0.2], [0.6, 0.4])
        _update(tracker, 0.1, 1, [1, 1], [10, 20], [0.2, 0.2], [0.5, 0.5])
        assert _open(tracker) == {1}
        # past t0 + L: closes without this output
        _update(tracker, 0.25, 2, [1, 1], [10, 20], [0.2, 0.2], [0.0, 1.0])
        assert _open(tracker) == set() and _finalized(tracker) == {1: 10}

    def test_an_output_exactly_at_the_end_of_the_window_still_counts(self):
        tracker = BirthTracker()
        _update(tracker, 0.0, 0, [1, 1], [10, 20], [0.2, 0.2], [0.6, 0.4])
        _update(tracker, 0.2, 1, [1, 1], [10, 20], [0.2, 0.2], [0.05, 0.95])
        assert _open(tracker) == {1}
        assert tracker.evidence(1)[20] > tracker.evidence(1)[10]

    def test_a_candidate_that_died_closes_the_window(self):
        tracker = BirthTracker()
        _update(
            tracker, 0.0, 0, [1, 1], [10, 20], [0.4, 0.4], [0.6, 0.4], alive=[10, 20]
        )
        _update(
            tracker, 0.05, 1, [1, 1], [10, 20], [0.4, 0.4], [0.0, 1.0], alive=[10]
        )
        # 20 merged away: that output is not used
        assert _finalized(tracker) == {1: 10}

    def test_particle_missing_from_a_feed_still_closes(self):
        tracker = BirthTracker()
        _update(
            tracker, 0.0, 0, [1, 1], [10, 20], [0.2, 0.2], [0.6, 0.4], alive=[10, 20]
        )
        _update(tracker, 0.5, 1, [2], [10], [0.2], alive=[10, 20])
        assert _finalized(tracker)[1] == 10 and 1 not in _open(tracker)


class TestEvidence:
    def test_persistent_membership_overturns_an_ambiguous_first_output(self):
        tracker = BirthTracker()
        _update(tracker, 0.0, 0, [1, 1], [10, 20], [0.4, 0.4], [0.55, 0.45])
        for snapshot, time in enumerate((0.1, 0.2, 0.3), start=1):
            _update(tracker, time, snapshot, [1, 1], [10, 20], [0.4, 0.4], [0.1, 0.9])
        assert _open(tracker) == {1}
        assert tracker.finalize().birth_id.tolist() == [20]

    def test_evidence_is_the_leaky_likelihood(self):
        tracker = BirthTracker(budget=5)
        # L = 0.4
        _update(tracker, 0.0, 0, [1, 1], [10, 20], [0.4, 0.4], [0.7, 0.3])
        _update(tracker, 0.1, 1, [1, 1], [10, 20], [0.4, 0.4], [0.2, 0.8])
        persistence = _persistence(0.1, 0.4)
        expected_10 = np.log(0.7) + np.log(persistence + (1 - persistence) * 0.2)
        expected_20 = np.log(0.3) + np.log(persistence + (1 - persistence) * 0.8)
        assert tracker.evidence(1) == {
            10: pytest.approx(expected_10),
            20: pytest.approx(expected_20),
        }

    def test_dt_is_measured_from_the_last_output_used(self):
        tracker = BirthTracker(budget=5)
        _update(tracker, 0.0, 0, [1, 1], [10, 20], [0.4, 0.4], [0.5, 0.5])
        _update(tracker, 0.1, 1, [1, 1], [10, 20], [0.4, 0.4], [0.5, 0.5])
        before = tracker.evidence(1)
        _update(tracker, 0.2, 2, [1, 1], [10, 20], [0.4, 0.4], [0.3, 0.7])
        persistence = _persistence(0.1, 0.4)
        gain = tracker.evidence(1)[20] - before[20]
        assert gain == pytest.approx(np.log(persistence + (1 - persistence) * 0.7))

    def test_outputs_closer_than_the_correlation_time_count_little(self):
        tracker = BirthTracker(budget=5)
        # tau_c = 0.08
        _update(tracker, 0.0, 0, [1, 1], [10, 20], [0.4, 0.4], [0.6, 0.4])
        _update(tracker, 0.001, 1, [1, 1], [10, 20], [0.4, 0.4], [0.0, 1.0])
        assert abs(tracker.evidence(1)[20] - np.log(0.4)) < 0.02
        assert abs(tracker.evidence(1)[10] - np.log(0.6)) < 0.1

    def test_a_candidate_missing_from_an_output_loses_the_look(self):
        tracker = BirthTracker(budget=5)
        _update(
            tracker, 0.0, 0, [1, 1], [10, 20], [0.4, 0.4], [0.5, 0.5], alive=[10, 20]
        )
        # no pair with 20: its share is zero
        _update(tracker, 0.1, 1, [1], [10], [0.4], [1.0], alive=[10, 20])
        persistence = _persistence(0.1, 0.4)
        assert tracker.evidence(1)[20] - np.log(0.5) == pytest.approx(
            np.log(persistence)
        )

    def test_an_unbound_output_adds_nothing(self):
        tracker = BirthTracker()
        _update(
            tracker, 0.0, 0, [1, 1], [10, 20], [0.4, 0.4], [0.6, 0.4], alive=[10, 20]
        )
        before = tracker.evidence(1)
        _update(tracker, 0.1, 1, [1], [-1], [NAN], alive=[10, 20])
        assert tracker.evidence(1) == before
        _update(
            tracker, 0.15, 2, [1, 1], [10, 20], [0.4, 0.4], [0.5, 0.5], alive=[10, 20]
        )
        # dt is still measured from the first output
        persistence = _persistence(0.15, 0.4)
        gain = tracker.evidence(1)[10] - before[10]
        assert gain == pytest.approx(np.log(persistence + (1 - persistence) * 0.5))

    def test_equal_evidence_goes_to_the_lowest_host(self):
        tracker = BirthTracker()
        _update(
            tracker, 0.0, 0,
            particles=[1, 1, 1],
            hosts=[30, 10, 20],
            dynamical_times=[0.4, 0.4, 0.4],
            weights=[1.0, 1.0, 1.0],
        )
        assert tracker.finalize().birth_id.tolist() == [10]

    def test_duplicate_pairs_are_merged(self):
        tracker = BirthTracker()
        _update(
            tracker, 0.0, 0,
            particles=[1, 1, 1],
            hosts=[10, 10, 20],
            dynamical_times=[0.2, 0.2, 0.2],
            weights=[0.25, 0.25, 0.4],
        )
        assert tracker.evidence(1) == {
            10: pytest.approx(np.log(0.5 / 0.9)),
            20: pytest.approx(np.log(0.4 / 0.9)),
        }


class TestCandidates:
    def test_only_hosts_above_the_threshold_compete(self):
        tracker = BirthTracker(min_initial_resp=0.1)
        _update(
            tracker, 0.0, 0,
            particles=[1, 1, 1],
            hosts=[10, 20, 30],
            dynamical_times=[0.4, 0.4, 0.4],
            weights=[0.5, 0.45, 0.05],
            alive=[10, 20, 30],
        )
        assert set(tracker.evidence(1)) == {10, 20}
        # 30 was not a candidate
        _update(
            tracker, 0.05, 1, [1, 1], [20, 30], [0.4, 0.4], [0.1, 0.9],
            alive=[10, 20, 30],
        )
        assert set(tracker.evidence(1)) == {10, 20}

    def test_the_largest_host_always_competes(self):
        tracker = BirthTracker(min_initial_resp=0.5)
        _update(
            tracker, 0.0, 0,
            particles=[1, 1, 1, 1],
            hosts=[10, 20, 30, 40],
            dynamical_times=[0.4] * 4,
            weights=[0.26, 0.25, 0.25, 0.24],
        )
        assert set(tracker.evidence(1)) == {10}

    def test_zero_threshold_keeps_every_bound_host(self):
        tracker = BirthTracker(min_initial_resp=0.0)
        _update(
            tracker, 0.0, 0,
            particles=[1, 1, 1],
            hosts=[10, 20, 30],
            dynamical_times=[0.4, 0.4, 0.4],
            weights=[0.5, 0.45, 1e-6],
        )
        assert set(tracker.evidence(1)) == {10, 20, 30}

    def test_negative_threshold_lets_later_hosts_join(self):
        tracker = BirthTracker(min_initial_resp=-1)
        _update(
            tracker, 0.0, 0, [1, 1], [10, 20], [0.4, 0.4], [0.5, 0.5],
            alive=[10, 20, 30],
        )
        _update(
            tracker, 0.05, 1, [1, 1], [10, 30], [0.4, 0.4], [0.5, 0.5],
            alive=[10, 20, 30],
        )
        log_evidence = tracker.evidence(1)
        assert set(log_evidence) == {10, 20, 30}
        persistence = _persistence(0.05, 0.4)
        late_look = np.log(persistence + (1 - persistence) * 0.5)
        assert log_evidence[30] == pytest.approx(np.log(BIRTH_LATE_PRIOR) + late_look)
        assert log_evidence[20] == pytest.approx(np.log(0.5) + np.log(persistence))


class TestChanges:
    def test_a_particle_is_born_in_its_leading_host(self):
        tracker = BirthTracker()
        changes = _update(
            tracker, 0.0, 0,
            particles=[1, 2, 2, 3],
            hosts=[10, 20, 30, -1],
            dynamical_times=[0.2, 0.4, 0.4, NAN],
            weights=[1.0, 0.6, 0.4, 1.0],
        )
        assert isinstance(changes, BirthChanges)
        # 2 is open, led by 20
        assert _as_dict(changes.born_ids, changes.born_hosts) == {1: 10, 2: 20, 3: -1}
        assert changes.moved_ids.size == 0
        assert changes.stats == {
            "new": 3, "at_once": 2, "opened": 1, "open": 1, "moved": 0, "closed": 0,
        }

    def test_a_new_leader_is_a_move(self):
        tracker = BirthTracker()
        _update(tracker, 0.0, 0, [1, 1], [10, 20], [0.4, 0.4], [0.55, 0.45])
        changes = _update(tracker, 0.1, 1, [1, 1], [10, 20], [0.4, 0.4], [0.05, 0.95])
        assert changes.born_ids.size == 0
        assert changes.moved_ids.tolist() == [1]
        assert changes.moved_from.tolist() == [10]
        assert changes.moved_to.tolist() == [20]
        assert changes.stats["moved"] == 1
        # the leader stays
        again = _update(tracker, 0.2, 2, [1, 1], [10, 20], [0.4, 0.4], [0.05, 0.95])
        assert again.moved_ids.size == 0

    def test_closing_a_window_changes_no_birth(self):
        tracker = BirthTracker()
        _update(tracker, 0.0, 0, [1, 1], [10, 20], [0.2, 0.2], [0.6, 0.4])
        changes = _update(tracker, 0.5, 1, [2], [10], [0.2], alive=[10, 20])
        assert _as_dict(changes.born_ids, changes.born_hosts) == {2: 10}
        assert changes.moved_ids.size == 0
        assert changes.stats["closed"] == 1 and changes.stats["open"] == 0
        assert _finalized(tracker)[1] == 10

    def test_the_changes_by_galaxy(self):
        changes = BirthChanges(
            born_ids=np.array([1, 2], np.uint64),
            born_hosts=np.array([10, 10]),
            moved_ids=np.array([3], np.uint64),
            moved_from=np.array([10]),
            moved_to=np.array([20]),
            stats={},
        )
        gained, lost = changes.by_galaxy()
        assert gained == {10: {1, 2}, 20: {3}} and lost == {10: {3}}

    def test_an_update_without_changes_has_empty_by_galaxy(self):
        tracker = BirthTracker()
        changes = _update(
            tracker, 0.0, 0, np.array([], int), np.array([], int), np.array([])
        )
        assert changes.by_galaxy() == ({}, {})

    def test_an_earlier_snapshot_changes_nothing(self):
        tracker = BirthTracker()
        _update(tracker, 1.0, 5, [1], [10], [0.2])
        changes = _update(tracker, 0.5, 4, [2], [10], [0.2])
        assert changes.born_ids.size == 0 and _finalized(tracker) == {1: 10}


class TestOutputs:
    def test_finalize_closes_everything(self):
        tracker = BirthTracker()
        _update(
            tracker, 0.0, 0,
            particles=[1, 2, 2, 3],
            hosts=[10, 20, 30, 10],
            dynamical_times=[0.2, 0.4, 0.4, 0.3],
            weights=[1.0, 0.7, 0.3, 1.0],
        )
        births = tracker.finalize()
        decided = dict(zip(births["particle_index"], births["birth_id"]))
        assert decided == {1: 10, 2: 20, 3: 10}
        assert births["particle_index"].dtype == np.uint64
        assert births["birth_id"].dtype == np.int64
        assert births["p_birth"].dtype == np.float32
        assert births["birth_time"].dtype == np.float32
        p_birth_of_2 = births["p_birth"][births["particle_index"] == 2].item()
        assert p_birth_of_2 == pytest.approx(0.7)
        assert _open(tracker) == set()

    def test_birth_time_is_the_first_appearance(self):
        tracker = BirthTracker()
        _update(
            tracker, 0.5, 0,
            particles=[1, 2, 2],
            hosts=[10, 20, 30],
            dynamical_times=[0.2, 0.4, 0.4],
            weights=[1.0, 0.6, 0.4],
        )
        _update(
            tracker, 0.6, 1,
            particles=[2, 2, 3],
            hosts=[20, 30, 10],
            dynamical_times=[0.4, 0.4, 0.3],
            weights=[0.6, 0.4, 1.0],
        )
        births = tracker.finalize().set_index("particle_index")
        assert births.birth_time.to_dict() == {
            1: pytest.approx(0.5),
            2: pytest.approx(0.5),
            3: pytest.approx(0.6),
        }

    def test_empty_update(self):
        tracker = BirthTracker()
        _update(tracker, 0.0, 0, np.array([], int), np.array([], int), np.array([]))
        assert _open(tracker) == set() and _finalized(tracker) == {}
        assert tracker.finalize().empty

    def test_the_state_is_read_only(self):
        tracker = BirthTracker()
        _update(
            tracker, 0.0, 0,
            particles=[1, 1, 2],
            hosts=[10, 20, 10],
            dynamical_times=[0.4, 0.4, 0.2],
            weights=[0.6, 0.4, 1.0],
        )
        with pytest.raises(ValueError):
            tracker._open_table["window"][0] = 1.0
        with pytest.raises(ValueError):
            tracker._evidence_table["log_evidence"][0] = 0.0
        with pytest.raises(ValueError):
            tracker._finalized_chunks[0]["birth_id"][0] = 1


class TestCheckpoint:
    @staticmethod
    def _run(tracker, snapshots):
        """Feed the same synthetic outputs, one per snapshot, to a tracker."""
        for snapshot in snapshots:
            _update(
                tracker, 0.05 * snapshot, snapshot,
                particles=[1, 1, 2, 3, 3],
                hosts=[10, 20, 20, 10, 20],
                dynamical_times=[0.4, 0.4, 0.3, 0.2, 0.2],
                weights=[
                    0.5 + 0.02 * snapshot, 0.5 - 0.02 * snapshot, 1.0, 0.6, 0.4,
                ],
                alive=[10, 20],
            )

    def test_serialization_round_trip(self):
        uninterrupted = BirthTracker()
        self._run(uninterrupted, range(8))
        tracker = BirthTracker()
        self._run(tracker, range(4))
        restored = BirthTracker()
        restored.restore(tracker.serialize(tracker.capture()))
        assert restored._last_snapshot == 3 and restored.budget == tracker.budget
        self._run(restored, range(4, 8))
        assert restored.finalize().equals(uninterrupted.finalize())

    def test_an_earlier_tracker_checkpoint_is_refused(self):
        tracker = BirthTracker()
        earlier = {
            "format": 3, "active": {}, "evidence": {}, "finalized": {},
            "last_snapshot": 0,
        }
        with pytest.raises(ValueError, match="earlier birth tracker"):
            tracker.restore(earlier)
        older = {"active": {}, "finalized": {}, "heap": [], "last_snapshot": 0}
        with pytest.raises(ValueError, match="earlier birth tracker"):
            tracker.restore(older)

    def test_capture_unaffected_by_later_update(self):
        # C02: a reference captured now must survive later updates untouched
        tracker = BirthTracker()
        _update(tracker, 0.0, 0, [1, 1], [10, 20], [0.4, 0.4], [0.6, 0.4])
        reference = tracker.capture()
        evidence_before = {
            name: column.copy() for name, column in reference["evidence_table"].items()
        }
        open_before = {
            name: column.copy() for name, column in reference["open_table"].items()
        }
        _update(
            tracker, 0.1, 1, [1, 1, 2], [10, 20, 10], [0.4, 0.4, 0.2], [0.1, 0.9, 1.0]
        )
        # closes particle 1
        _update(tracker, 5.0, 2, [1], [10], [0.4])
        for name, column in evidence_before.items():
            np.testing.assert_array_equal(reference["evidence_table"][name], column)
        for name, column in open_before.items():
            np.testing.assert_array_equal(reference["open_table"][name], column)


class _Reference:
    """The rule, one particle at a time and in plain Python.

    It is the oracle of the vectorised tracker: slow, and simple enough to
    read against the rule.

    Parameters
    ----------
    budget : float
        The evidence budget ``N``.
    min_initial_resp : float
        The first-output share a host must exceed to compete.
    """

    def __init__(self, budget, min_initial_resp):
        self.budget = budget
        self.min_initial_resp = min_initial_resp
        self.open = {}
        self.done = {}

    def update(self, time, particles, hosts, dynamical_times, weights, alive):
        """Take one output, given as parallel lists with one entry per pair."""
        alive = set(alive)
        for particle in sorted(set(particles)):
            entries = [i for i, p in enumerate(particles) if p == particle]
            entry_hosts = [hosts[i] for i in entries]
            entry_weights = np.array([weights[i] for i in entries])
            entry_times = np.array([dynamical_times[i] for i in entries])
            if particle in self.done:
                continue
            state = self.open.get(particle)
            if state is None:
                self._first_output(
                    time, particle, entry_hosts, entry_weights, entry_times
                )
            elif not self._closes(state, time, alive):
                share = np.array([
                    sum(
                        weight
                        for host, weight in zip(entry_hosts, entry_weights)
                        if host == candidate
                    )
                    for candidate in state["hosts"]
                ])
                if share.sum() > 0:
                    persistence = np.exp(
                        -(time - state["last_time"]) * self.budget / state["window"]
                    )
                    state["log_evidence"] = state["log_evidence"] + np.log(
                        persistence + (1 - persistence) * share / share.sum()
                    )
                    state["last_time"] = time
        for particle in list(self.open):
            if self._closes(self.open[particle], time, alive):
                self.close(particle)

    def _first_output(self, time, particle, hosts, weights, times):
        """Decide a particle at once, or open its window."""
        share = weights / weights.sum()
        leading = int(np.argmax(share))
        decided_share = 1 / (1 + np.exp(-self.budget))
        if len(hosts) == 1 or share.max() >= decided_share:
            self.done[particle] = (hosts[leading], share.max(), time)
            return
        candidates = [
            i
            for i in range(len(hosts))
            if share[i] > self.min_initial_resp or i == leading
        ]
        with_time = [i for i in candidates if np.isfinite(times[i]) and times[i] > 0]
        if with_time:
            window = sum(share[i] * times[i] for i in with_time) / sum(
                share[i] for i in with_time
            )
        else:
            window = MAX_DYN_TIMESCALE
        self.open[particle] = {
            "first_time": time,
            "last_time": time,
            "window": window,
            "hosts": [hosts[i] for i in candidates],
            "log_evidence": np.log(share[candidates]),
        }

    @staticmethod
    def _closes(state, time, alive):
        """Whether a window closes at this output: it ran out, or a host is gone."""
        expired = time - state["first_time"] > state["window"] + 1e-9
        return expired or not all(host in alive for host in state["hosts"])

    def close(self, particle):
        """Finalise a particle on its current evidence."""
        state = self.open.pop(particle)
        leader, probability = self.lead(state)
        self.done[particle] = (
            state["hosts"][leader], probability, state["first_time"]
        )

    @staticmethod
    def lead(state):
        """Find the host with the most evidence, and its probability.

        The lowest host ID wins a tie. Returns the index of the host among the
        candidates, and its softmax probability.
        """
        by_host_id = sorted(
            range(len(state["hosts"])), key=lambda i: state["hosts"][i]
        )
        relative = np.exp(state["log_evidence"] - state["log_evidence"].max())
        leader = next(i for i in by_host_id if relative[i] == relative.max())
        return leader, relative[leader] / relative.sum()

    def leaders(self):
        """Give every particle seen with its birth host now.

        This is the final host, or the leader of the particle's open window.
        """
        final = {particle: record[0] for particle, record in self.done.items()}
        leading = {
            particle: state["hosts"][self.lead(state)[0]]
            for particle, state in self.open.items()
        }
        return {**final, **leading}


def _random_feeds(rng, n_particles=120, n_snapshots=25, n_galaxies=7):
    """Draw random outputs: particles come and go, hosts die, times are noisy.

    Returns
    -------
    feeds : list of tuple
        ``(time, snapshot, particles, hosts, dynamical_times, weights, alive)``
        per snapshot, the lists with one entry per pair.
    """
    times = np.cumsum(rng.uniform(0.02, 0.1, n_snapshots))
    death = np.where(
        rng.random(n_galaxies) < 0.4,
        rng.integers(4, n_snapshots, n_galaxies),
        10_000,
    )
    first = rng.integers(0, n_snapshots - 1, n_particles)
    base_hosts = [
        rng.choice(n_galaxies, rng.integers(1, 5), replace=False)
        for _ in range(n_particles)
    ]
    base_times = [rng.uniform(0.01, 0.6, hosts.size) for hosts in base_hosts]
    for particle_times in base_times:
        particle_times[rng.random(particle_times.size) < 0.1] = NAN
    feeds = []
    for snapshot in range(n_snapshots):
        alive = np.array([g for g in range(n_galaxies) if death[g] > snapshot])
        particles, hosts, weights, dynamical_times = [], [], [], []
        for i in range(n_particles):
            if snapshot < first[i] or rng.random() < 0.1:
                continue
            if snapshot == first[i]:
                present = set(base_hosts[i])
            else:
                present = {g for g in base_hosts[i] if g in alive}
            stray = set()
            if snapshot > first[i] and rng.random() < 0.15:
                stray = {int(rng.integers(0, n_galaxies))}
            present = sorted(present | stray)
            if not present:
                continue
            shares = rng.dirichlet(np.ones(len(present)) * 0.7)
            for host, share in zip(present, shares):
                particles.append(1000 + i)
                hosts.append(host)
                weights.append(share)
                if host in base_hosts[i]:
                    dynamical_times.append(
                        base_times[i][list(base_hosts[i]).index(host)]
                    )
                else:
                    dynamical_times.append(NAN)
        feeds.append((
            times[snapshot],
            snapshot,
            particles,
            hosts,
            dynamical_times,
            weights,
            alive,
        ))
    return feeds


@pytest.mark.parametrize(
    "budget, min_initial_resp",
    [(5.0, 0.01), (2.0, 0.01), (7.0, 0.0), (5.0, 0.2), (3.0, 0.4)],
)
def test_matches_the_reference_on_random_feeds(budget, min_initial_resp):
    for seed in range(8):
        tracker = BirthTracker(budget=budget, min_initial_resp=min_initial_resp)
        reference = _Reference(budget, min_initial_resp)
        # the birth hosts as a consumer of the changes sees them
        leaders = {}
        feeds = _random_feeds(np.random.default_rng(seed))
        for time, snapshot, particles, hosts, dynamical_times, weights, alive in feeds:
            changes = tracker.update(
                Memberships(
                    time,
                    snapshot,
                    np.array(particles, np.uint64),
                    np.array(hosts),
                    np.array(weights),
                    np.array(dynamical_times),
                    alive,
                )
            )
            reference.update(
                time, particles, hosts, dynamical_times, weights, alive
            )
            # born once
            assert not set(changes.born_ids.tolist()) & set(leaders)
            leaders.update(_as_dict(changes.born_ids, changes.born_hosts))
            moves = zip(
                changes.moved_ids.tolist(),
                changes.moved_from.tolist(),
                changes.moved_to.tolist(),
            )
            for particle, old, new in moves:
                assert leaders[particle] == old and old != new
                leaders[particle] = new
            # the changes are the whole difference
            assert leaders == reference.leaders(), (seed, snapshot)
        births = tracker.finalize().set_index("particle_index")
        for particle in list(reference.open):
            reference.close(particle)
        assert set(births.index.tolist()) == set(reference.done)
        for particle, (host, probability, first_time) in reference.done.items():
            assert births.birth_id[particle] == host, (seed, particle)
            assert births.p_birth[particle] == pytest.approx(probability, abs=2e-6)
            assert births.birth_time[particle] == pytest.approx(first_time, abs=2e-6)
        final_hosts = {
            particle: record[0] for particle, record in reference.done.items()
        }
        assert leaders == final_hosts
