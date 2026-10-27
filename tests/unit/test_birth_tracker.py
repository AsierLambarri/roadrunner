import numpy as np
import pytest

from roadrunner._defaults import BIRTH_LATE_PRIOR
from roadrunner.physics.constants import MAX_DYN_TIMESCALE
from roadrunner.postprocessing.tracking.birth import BirthTracker
from roadrunner.postprocessing.tracking.records import BirthChanges, Memberships

NAN = np.nan


def _looks(t, snap, pids, hosts, taus, weights=None, alive=None):
    hosts = np.array(hosts)
    return Memberships(t, snap, np.array(pids, dtype=np.uint64), hosts,
                       np.ones(len(pids)) if weights is None else np.array(weights, dtype=float),
                       np.array(taus, dtype=float), np.unique(hosts) if alive is None else np.array(alive))


def _update(tracker, t, snap, pids, hosts, taus, weights=None, alive=None):
    return tracker.update(_looks(t, snap, pids, hosts, taus, weights, alive))


def _open(tracker):
    return set(tracker.open_ids().tolist())


def _finalized(tracker):
    df = tracker.finalized()
    return dict(zip(df["particle_index"].tolist(), df["birth_id"].tolist()))


def _lam(dt, lag, budget=5.0):
    return np.exp(-dt * budget / lag)


def _pairs(arr_a, arr_b):
    return dict(zip(arr_a.tolist(), arr_b.tolist()))


class TestDecidedAtOnce:
    def test_single_candidate(self):
        tracker = BirthTracker()
        _update(tracker, 0.0, 0, [1, 2], [10, -1], [0.2, NAN])
        assert _open(tracker) == set() and _finalized(tracker) == {1: 10, 2: -1}
        assert (tracker.finalize().p_birth == 1.0).all()

    def test_confident_first_output_is_final(self):
        # budget 5: a share of 1 / (1 + e^-5) = 0.9933 or more cannot be overturned by the window
        tracker = BirthTracker(budget=5)
        _update(tracker, 0.0, 0, [1, 1, 2, 2], [10, 20, 10, 20], [0.2, 0.2, 0.2, 0.2], [0.9934, 0.0066, 0.9932, 0.0068])
        assert _finalized(tracker) == {1: 10} and _open(tracker) == {2}
        _update(tracker, 0.05, 1, [1, 1], [10, 20], [0.2, 0.2], [0.0, 1.0])     # settled: later outputs are ignored
        assert _finalized(tracker)[1] == 10

    def test_negative_threshold_decides_nothing_at_once(self):
        tracker = BirthTracker(min_initial_resp=-1)
        _update(tracker, 0.0, 0, [1], [10], [0.2])
        assert _open(tracker) == {1} and _finalized(tracker) == {}

    def test_settled_particles_are_the_finalised_ones(self):
        tracker = BirthTracker()
        _update(tracker, 0.0, 0, [1, 2, 2], [10, 20, 30], [0.2, 0.4, 0.4], [1.0, 0.6, 0.4])
        assert tracker.settled(np.array([1, 2, 3], dtype=np.uint64)).tolist() == [True, False, False]


class TestWindow:
    def test_window_is_the_weighted_mean_of_the_dynamical_times(self):
        tracker = BirthTracker()
        _update(tracker, 0.0, 0, [1, 1], [10, 20], [0.2, 0.4], [0.6, 0.4])
        assert tracker.window(1) == pytest.approx(0.6 * 0.2 + 0.4 * 0.4)
        # the prior is the log of the first-output share
        assert tracker.evidence(1) == {10: pytest.approx(np.log(0.6)), 20: pytest.approx(np.log(0.4))}

    def test_missing_and_zero_times_are_left_out_of_the_mean(self):
        tracker = BirthTracker()
        _update(tracker, 0.0, 0, [1, 1, 2, 2, 3, 3], [10, 20, 10, 20, 10, 20], [0.3, NAN, 0.3, 0.0, NAN, 0.0],
                [0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        assert tracker.window(1) == pytest.approx(0.3) and tracker.window(2) == pytest.approx(0.3)
        assert tracker.window(3) == MAX_DYN_TIMESCALE                              # nothing to average

    def test_window_closes_after_the_lag(self):
        tracker = BirthTracker()
        _update(tracker, 0.0, 0, [1, 1], [10, 20], [0.2, 0.2], [0.6, 0.4])        # L = 0.2
        _update(tracker, 0.1, 1, [1, 1], [10, 20], [0.2, 0.2], [0.5, 0.5])
        assert _open(tracker) == {1}
        _update(tracker, 0.25, 2, [1, 1], [10, 20], [0.2, 0.2], [0.0, 1.0])      # past t0 + L: closes without this output
        assert _open(tracker) == set() and _finalized(tracker) == {1: 10}

    def test_an_output_exactly_at_the_end_of_the_window_still_counts(self):
        tracker = BirthTracker()
        _update(tracker, 0.0, 0, [1, 1], [10, 20], [0.2, 0.2], [0.6, 0.4])
        _update(tracker, 0.2, 1, [1, 1], [10, 20], [0.2, 0.2], [0.05, 0.95])
        assert _open(tracker) == {1}
        assert tracker.evidence(1)[20] > tracker.evidence(1)[10]

    def test_a_candidate_that_died_closes_the_window(self):
        tracker = BirthTracker()
        _update(tracker, 0.0, 0, [1, 1], [10, 20], [0.4, 0.4], [0.6, 0.4], alive=[10, 20])
        _update(tracker, 0.05, 1, [1, 1], [10, 20], [0.4, 0.4], [0.0, 1.0], alive=[10])
        assert _finalized(tracker) == {1: 10}                                   # 20 merged away: that output is not used

    def test_particle_missing_from_a_feed_still_closes(self):
        tracker = BirthTracker()
        _update(tracker, 0.0, 0, [1, 1], [10, 20], [0.2, 0.2], [0.6, 0.4], alive=[10, 20])
        _update(tracker, 0.5, 1, [2], [10], [0.2], alive=[10, 20])
        assert _finalized(tracker)[1] == 10 and 1 not in _open(tracker)


class TestEvidence:
    def test_persistent_membership_overturns_an_ambiguous_first_output(self):
        tracker = BirthTracker()
        _update(tracker, 0.0, 0, [1, 1], [10, 20], [0.4, 0.4], [0.55, 0.45])
        for s, t in enumerate((0.1, 0.2, 0.3), start=1):
            _update(tracker, t, s, [1, 1], [10, 20], [0.4, 0.4], [0.1, 0.9])
        assert _open(tracker) == {1}
        assert tracker.finalize().birth_id.tolist() == [20]

    def test_evidence_is_the_leaky_likelihood(self):
        tracker = BirthTracker(budget=5)
        _update(tracker, 0.0, 0, [1, 1], [10, 20], [0.4, 0.4], [0.7, 0.3])        # L = 0.4
        _update(tracker, 0.1, 1, [1, 1], [10, 20], [0.4, 0.4], [0.2, 0.8])
        lam = _lam(0.1, 0.4)
        assert tracker.evidence(1) == {10: pytest.approx(np.log(0.7) + np.log(lam + (1 - lam) * 0.2)),
                                       20: pytest.approx(np.log(0.3) + np.log(lam + (1 - lam) * 0.8))}

    def test_dt_is_measured_from_the_last_output_used(self):
        tracker = BirthTracker(budget=5)
        _update(tracker, 0.0, 0, [1, 1], [10, 20], [0.4, 0.4], [0.5, 0.5])
        _update(tracker, 0.1, 1, [1, 1], [10, 20], [0.4, 0.4], [0.5, 0.5])
        before = tracker.evidence(1)
        _update(tracker, 0.2, 2, [1, 1], [10, 20], [0.4, 0.4], [0.3, 0.7])
        lam = _lam(0.1, 0.4)
        assert tracker.evidence(1)[20] - before[20] == pytest.approx(np.log(lam + (1 - lam) * 0.7))

    def test_outputs_closer_than_the_correlation_time_count_little(self):
        tracker = BirthTracker(budget=5)
        _update(tracker, 0.0, 0, [1, 1], [10, 20], [0.4, 0.4], [0.6, 0.4])       # tau_c = 0.08
        _update(tracker, 0.001, 1, [1, 1], [10, 20], [0.4, 0.4], [0.0, 1.0])
        assert abs(tracker.evidence(1)[20] - np.log(0.4)) < 0.02
        assert abs(tracker.evidence(1)[10] - np.log(0.6)) < 0.1

    def test_a_candidate_missing_from_an_output_loses_the_look(self):
        tracker = BirthTracker(budget=5)
        _update(tracker, 0.0, 0, [1, 1], [10, 20], [0.4, 0.4], [0.5, 0.5], alive=[10, 20])
        _update(tracker, 0.1, 1, [1], [10], [0.4], [1.0], alive=[10, 20])        # no pair with 20: its share is zero
        lam = _lam(0.1, 0.4)
        assert tracker.evidence(1)[20] - np.log(0.5) == pytest.approx(np.log(lam))

    def test_an_unbound_output_adds_nothing(self):
        tracker = BirthTracker()
        _update(tracker, 0.0, 0, [1, 1], [10, 20], [0.4, 0.4], [0.6, 0.4], alive=[10, 20])
        before = tracker.evidence(1)
        _update(tracker, 0.1, 1, [1], [-1], [NAN], alive=[10, 20])
        assert tracker.evidence(1) == before
        _update(tracker, 0.15, 2, [1, 1], [10, 20], [0.4, 0.4], [0.5, 0.5], alive=[10, 20])
        lam = _lam(0.15, 0.4)                                                   # dt is still measured from the first output
        assert tracker.evidence(1)[10] - before[10] == pytest.approx(np.log(lam + (1 - lam) * 0.5))

    def test_equal_evidence_goes_to_the_lowest_host(self):
        tracker = BirthTracker()
        _update(tracker, 0.0, 0, [1, 1, 1], [30, 10, 20], [0.4, 0.4, 0.4], [1.0, 1.0, 1.0])
        assert tracker.finalize().birth_id.tolist() == [10]

    def test_duplicate_pairs_are_merged(self):
        tracker = BirthTracker()
        _update(tracker, 0.0, 0, [1, 1, 1], [10, 10, 20], [0.2, 0.2, 0.2], [0.25, 0.25, 0.4])
        assert tracker.evidence(1) == {10: pytest.approx(np.log(0.5 / 0.9)), 20: pytest.approx(np.log(0.4 / 0.9))}


class TestCandidates:
    def test_only_hosts_above_the_threshold_compete(self):
        tracker = BirthTracker(min_initial_resp=0.1)
        _update(tracker, 0.0, 0, [1, 1, 1], [10, 20, 30], [0.4, 0.4, 0.4], [0.5, 0.45, 0.05], alive=[10, 20, 30])
        assert set(tracker.evidence(1)) == {10, 20}
        _update(tracker, 0.05, 1, [1, 1], [20, 30], [0.4, 0.4], [0.1, 0.9], alive=[10, 20, 30])      # 30 was not a candidate
        assert set(tracker.evidence(1)) == {10, 20}

    def test_the_largest_host_always_competes(self):
        tracker = BirthTracker(min_initial_resp=0.5)
        _update(tracker, 0.0, 0, [1, 1, 1, 1], [10, 20, 30, 40], [0.4] * 4, [0.26, 0.25, 0.25, 0.24])
        assert set(tracker.evidence(1)) == {10}

    def test_zero_threshold_keeps_every_bound_host(self):
        tracker = BirthTracker(min_initial_resp=0.0)
        _update(tracker, 0.0, 0, [1, 1, 1], [10, 20, 30], [0.4, 0.4, 0.4], [0.5, 0.45, 1e-6])
        assert set(tracker.evidence(1)) == {10, 20, 30}

    def test_negative_threshold_lets_later_hosts_join(self):
        tracker = BirthTracker(min_initial_resp=-1)
        _update(tracker, 0.0, 0, [1, 1], [10, 20], [0.4, 0.4], [0.5, 0.5], alive=[10, 20, 30])
        _update(tracker, 0.05, 1, [1, 1], [10, 30], [0.4, 0.4], [0.5, 0.5], alive=[10, 20, 30])
        logp = tracker.evidence(1)
        assert set(logp) == {10, 20, 30}
        lam = _lam(0.05, 0.4)
        assert logp[30] == pytest.approx(np.log(BIRTH_LATE_PRIOR) + np.log(lam + (1 - lam) * 0.5))
        assert logp[20] == pytest.approx(np.log(0.5) + np.log(lam))


class TestChanges:
    def test_a_particle_is_born_in_its_leading_host(self):
        tracker = BirthTracker()
        changes = _update(tracker, 0.0, 0, [1, 2, 2, 3], [10, 20, 30, -1], [0.2, 0.4, 0.4, NAN], [1.0, 0.6, 0.4, 1.0])
        assert isinstance(changes, BirthChanges)
        assert _pairs(changes.born_pid, changes.born_host) == {1: 10, 2: 20, 3: -1}          # 2 is open, led by 20
        assert changes.moved_pid.size == 0
        assert changes.stats == {"new": 3, "at_once": 2, "opened": 1, "open": 1, "moved": 0, "closed": 0}

    def test_a_new_leader_is_a_move(self):
        tracker = BirthTracker()
        _update(tracker, 0.0, 0, [1, 1], [10, 20], [0.4, 0.4], [0.55, 0.45])
        changes = _update(tracker, 0.1, 1, [1, 1], [10, 20], [0.4, 0.4], [0.05, 0.95])
        assert changes.born_pid.size == 0
        assert changes.moved_pid.tolist() == [1] and changes.moved_from.tolist() == [10] and changes.moved_to.tolist() == [20]
        assert changes.stats["moved"] == 1
        again = _update(tracker, 0.2, 2, [1, 1], [10, 20], [0.4, 0.4], [0.05, 0.95])        # the leader stays
        assert again.moved_pid.size == 0

    def test_closing_a_window_changes_no_birth(self):
        tracker = BirthTracker()
        _update(tracker, 0.0, 0, [1, 1], [10, 20], [0.2, 0.2], [0.6, 0.4])
        changes = _update(tracker, 0.5, 1, [2], [10], [0.2], alive=[10, 20])
        assert _pairs(changes.born_pid, changes.born_host) == {2: 10}
        assert changes.moved_pid.size == 0 and changes.stats["closed"] == 1 and changes.stats["open"] == 0
        assert _finalized(tracker)[1] == 10

    def test_the_changes_by_galaxy(self):
        changes = BirthChanges(np.array([1, 2], np.uint64), np.array([10, 10]), np.array([3], np.uint64),
                               np.array([10]), np.array([20]), {})
        gained, lost = changes.by_galaxy()
        assert gained == {10: {1, 2}, 20: {3}} and lost == {10: {3}}

    def test_an_update_without_changes_has_empty_by_galaxy(self):
        tracker = BirthTracker()
        changes = _update(tracker, 0.0, 0, np.array([], int), np.array([], int), np.array([]))
        assert changes.by_galaxy() == ({}, {})

    def test_an_earlier_snapshot_changes_nothing(self):
        tracker = BirthTracker()
        _update(tracker, 1.0, 5, [1], [10], [0.2])
        changes = _update(tracker, 0.5, 4, [2], [10], [0.2])
        assert changes.born_pid.size == 0 and _finalized(tracker) == {1: 10}


class TestOutputs:
    def test_finalize_closes_everything(self):
        tracker = BirthTracker()
        _update(tracker, 0.0, 0, [1, 2, 2, 3], [10, 20, 30, 10], [0.2, 0.4, 0.4, 0.3], [1.0, 0.7, 0.3, 1.0])
        df = tracker.finalize()
        assert dict(zip(df["particle_index"], df["birth_id"])) == {1: 10, 2: 20, 3: 10}
        assert df["particle_index"].dtype == np.uint64 and df["birth_id"].dtype == np.int64
        assert df["p_birth"].dtype == np.float32 and df["birth_time"].dtype == np.float32
        assert df["p_birth"][df["particle_index"] == 2].item() == pytest.approx(0.7)
        assert _open(tracker) == set()

    def test_birth_time_is_the_first_appearance(self):
        tracker = BirthTracker()
        _update(tracker, 0.5, 0, [1, 2, 2], [10, 20, 30], [0.2, 0.4, 0.4], [1.0, 0.6, 0.4])
        _update(tracker, 0.6, 1, [2, 2, 3], [20, 30, 10], [0.4, 0.4, 0.3], [0.6, 0.4, 1.0])
        df = tracker.finalize().set_index("particle_index")
        assert df.birth_time.to_dict() == {1: pytest.approx(0.5), 2: pytest.approx(0.5), 3: pytest.approx(0.6)}

    def test_empty_update(self):
        tracker = BirthTracker()
        _update(tracker, 0.0, 0, np.array([], int), np.array([], int), np.array([]))
        assert _open(tracker) == set() and _finalized(tracker) == {}
        assert tracker.finalize().empty

    def test_the_state_is_read_only(self):
        tracker = BirthTracker()
        _update(tracker, 0.0, 0, [1, 1, 2], [10, 20, 10], [0.4, 0.4, 0.2], [0.6, 0.4, 1.0])
        with pytest.raises(ValueError):
            tracker._act["lag"][0] = 1.0
        with pytest.raises(ValueError):
            tracker._ev["logp"][0] = 0.0
        with pytest.raises(ValueError):
            tracker._fin_chunks[0]["birth_id"][0] = 1


class TestCheckpoint:
    @staticmethod
    def _run(tracker, snaps):
        for s in snaps:
            _update(tracker, 0.05 * s, s, [1, 1, 2, 3, 3], [10, 20, 20, 10, 20], [0.4, 0.4, 0.3, 0.2, 0.2],
                    [0.5 + 0.02 * s, 0.5 - 0.02 * s, 1.0, 0.6, 0.4], alive=[10, 20])

    def test_serialization_round_trip(self):
        ref = BirthTracker()
        self._run(ref, range(8))
        tracker = BirthTracker()
        self._run(tracker, range(4))
        restored = BirthTracker()
        restored.restore(tracker.serialize(tracker.capture()))
        assert restored._last_snapshot == 3 and restored.budget == tracker.budget
        self._run(restored, range(4, 8))
        assert restored.finalize().equals(ref.finalize())

    def test_an_earlier_tracker_checkpoint_is_refused(self):
        tracker = BirthTracker()
        with pytest.raises(ValueError, match="earlier birth tracker"):
            tracker.restore({"format": 3, "active": {}, "evidence": {}, "finalized": {}, "last_snapshot": 0})
        with pytest.raises(ValueError, match="earlier birth tracker"):
            tracker.restore({"active": {}, "finalized": {}, "heap": [], "last_snapshot": 0})

    def test_capture_unaffected_by_later_update(self):
        # C02: a reference captured now must survive later updates untouched
        tracker = BirthTracker()
        _update(tracker, 0.0, 0, [1, 1], [10, 20], [0.4, 0.4], [0.6, 0.4])
        ref = tracker.capture()
        before = {k: v.copy() for k, v in ref["ev"].items()}
        before_act = {k: v.copy() for k, v in ref["act"].items()}
        _update(tracker, 0.1, 1, [1, 1, 2], [10, 20, 10], [0.4, 0.4, 0.2], [0.1, 0.9, 1.0])
        _update(tracker, 5.0, 2, [1], [10], [0.4])                              # closes particle 1
        for k, v in before.items():
            np.testing.assert_array_equal(ref["ev"][k], v)
        for k, v in before_act.items():
            np.testing.assert_array_equal(ref["act"][k], v)


class _Reference:
    """The rule, one particle at a time and in plain Python: the oracle of the vectorised tracker."""

    def __init__(self, budget, x):
        self.budget, self.x, self.open, self.done = budget, x, {}, {}

    def update(self, t, pids, hosts, taus, ws, alive):
        alive = set(alive)
        for pid in sorted(set(pids)):
            sel = [i for i, p in enumerate(pids) if p == pid]
            h, w, tt = [hosts[i] for i in sel], np.array([ws[i] for i in sel]), np.array([taus[i] for i in sel])
            if pid in self.done:
                continue
            st = self.open.get(pid)
            if st is None:
                p = w / w.sum()
                if len(h) == 1 or p.max() >= 1 / (1 + np.exp(-self.budget)):
                    self.done[pid] = (h[int(np.argmax(p))], p.max(), t)
                    continue
                cand = [i for i in range(len(h)) if p[i] > self.x or i == int(np.argmax(p))]
                ok = [i for i in cand if np.isfinite(tt[i]) and tt[i] > 0]
                lag = sum(p[i] * tt[i] for i in ok) / sum(p[i] for i in ok) if ok else MAX_DYN_TIMESCALE
                self.open[pid] = {"t0": t, "t_prev": t, "lag": lag, "hosts": [h[i] for i in cand], "lp": np.log(p[cand])}
            elif not self._closes(st, t, alive):
                q = np.array([sum(w[j] for j in range(len(h)) if h[j] == c) for c in st["hosts"]])
                if q.sum() > 0:
                    lam = np.exp(-(t - st["t_prev"]) * self.budget / st["lag"])
                    st["lp"] = st["lp"] + np.log(lam + (1 - lam) * q / q.sum())
                    st["t_prev"] = t
        for pid in list(self.open):
            if self._closes(self.open[pid], t, alive):
                self.close(pid)

    @staticmethod
    def _closes(st, t, alive):
        return t - st["t0"] > st["lag"] + 1e-9 or not all(c in alive for c in st["hosts"])

    def close(self, pid):
        st = self.open.pop(pid)
        k, prob = self.lead(st)
        self.done[pid] = (st["hosts"][k], prob, st["t0"])

    @staticmethod
    def lead(st):
        order = sorted(range(len(st["hosts"])), key=lambda i: st["hosts"][i])
        w = np.exp(st["lp"] - st["lp"].max())
        k = next(i for i in order if w[i] == w.max())
        return k, w[k] / w.sum()

    def leaders(self):
        """Every particle seen, with its birth host now: the final one, or its window's leader."""
        return {**{pid: v[0] for pid, v in self.done.items()},
                **{pid: st["hosts"][self.lead(st)[0]] for pid, st in self.open.items()}}


def _random_feeds(rng, n_particles=120, n_snap=25, n_gal=7):
    times = np.cumsum(rng.uniform(0.02, 0.1, n_snap))
    death = np.where(rng.random(n_gal) < 0.4, rng.integers(4, n_snap, n_gal), 10_000)
    first = rng.integers(0, n_snap - 1, n_particles)
    base = [rng.choice(n_gal, rng.integers(1, 5), replace=False) for _ in range(n_particles)]
    tdyn = [rng.uniform(0.01, 0.6, b.size) for b in base]
    for t in tdyn:
        t[rng.random(t.size) < 0.1] = NAN
    feeds = []
    for s in range(n_snap):
        alive = np.array([g for g in range(n_gal) if death[g] > s])
        P, H, W, T = [], [], [], []
        for i in range(n_particles):
            if s < first[i] or rng.random() < 0.1:
                continue
            hs = sorted(set(list(base[i]) if s == first[i] else [g for g in base[i] if g in alive]) | ({int(rng.integers(0, n_gal))} if s > first[i] and rng.random() < 0.15 else set()))
            if not hs:
                continue
            for g, wi in zip(hs, rng.dirichlet(np.ones(len(hs)) * 0.7)):
                P.append(1000 + i); H.append(g); W.append(wi)
                T.append(tdyn[i][list(base[i]).index(g)] if g in base[i] else NAN)
        feeds.append((times[s], s, P, H, T, W, alive))
    return feeds


@pytest.mark.parametrize("budget, x", [(5.0, 0.01), (2.0, 0.01), (7.0, 0.0), (5.0, 0.2), (3.0, 0.4)])
def test_matches_the_reference_on_random_feeds(budget, x):
    for seed in range(8):
        tracker, ref = BirthTracker(budget=budget, min_initial_resp=x), _Reference(budget, x)
        leaders = {}                                         # the birth hosts as a consumer of the changes sees them
        for t, s, P, H, T, W, alive in _random_feeds(np.random.default_rng(seed)):
            changes = tracker.update(Memberships(t, s, np.array(P, np.uint64), np.array(H), np.array(W), np.array(T), alive))
            ref.update(t, P, H, T, W, alive)
            assert not set(changes.born_pid.tolist()) & set(leaders)                       # born once
            leaders.update(_pairs(changes.born_pid, changes.born_host))
            for pid, old, new in zip(changes.moved_pid.tolist(), changes.moved_from.tolist(), changes.moved_to.tolist()):
                assert leaders[pid] == old and old != new
                leaders[pid] = new
            assert leaders == ref.leaders(), (seed, s)                                    # the changes are the whole difference
        out = tracker.finalize().set_index("particle_index")
        for pid in list(ref.open):
            ref.close(pid)
        assert set(out.index.tolist()) == set(ref.done)
        for pid, (host, prob, t0) in ref.done.items():
            assert out.birth_id[pid] == host, (seed, pid)
            assert out.p_birth[pid] == pytest.approx(prob, abs=2e-6)
            assert out.birth_time[pid] == pytest.approx(t0, abs=2e-6)
        assert leaders == {pid: v[0] for pid, v in ref.done.items()}
