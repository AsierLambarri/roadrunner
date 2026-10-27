import numpy as np
import pytest

from roadrunner._defaults import BIRTH_LATE_PRIOR
from roadrunner.physics.constants import MAX_DYN_TIMESCALE
from roadrunner.postprocessing.tracking.birth import BirthTracker

NAN = np.nan


def _update(tracker, t, snap, pids, hosts, taus, weights=None, alive=None):
    tracker.update(t_snap=t, snapshot_id=snap, particle_ids=np.array(pids), host_ids=np.array(hosts),
                   timescales=np.array(taus, dtype=float), weights=None if weights is None else np.array(weights),
                   alive=None if alive is None else np.array(alive))


def _open(tracker):
    return set(tracker._act["pid"].tolist())


def _finalized(tracker):
    fin = tracker._concat_finalized(tracker._fin_chunks)
    return dict(zip(fin["pid"].tolist(), fin["birth_id"].tolist()))


def _logp(tracker, pid):
    m = tracker._ev["pid"] == pid
    return dict(zip(tracker._ev["host"][m].tolist(), tracker._ev["logp"][m].tolist()))


def _lag(tracker, pid):
    return float(tracker._act["lag"][tracker._act["pid"] == pid][0])


def _lam(dt, lag, budget=5.0):
    return np.exp(-dt * budget / lag)


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


class TestWindow:
    def test_window_is_the_weighted_mean_of_the_dynamical_times(self):
        tracker = BirthTracker()
        _update(tracker, 0.0, 0, [1, 1], [10, 20], [0.2, 0.4], [0.6, 0.4])
        assert _lag(tracker, 1) == pytest.approx(0.6 * 0.2 + 0.4 * 0.4)
        # the prior is the log of the first-output share
        assert _logp(tracker, 1) == {10: pytest.approx(np.log(0.6)), 20: pytest.approx(np.log(0.4))}

    def test_missing_and_zero_times_are_left_out_of_the_mean(self):
        tracker = BirthTracker()
        _update(tracker, 0.0, 0, [1, 1, 2, 2, 3, 3], [10, 20, 10, 20, 10, 20], [0.3, NAN, 0.3, 0.0, NAN, 0.0],
                [0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        assert _lag(tracker, 1) == pytest.approx(0.3) and _lag(tracker, 2) == pytest.approx(0.3)
        assert _lag(tracker, 3) == MAX_DYN_TIMESCALE                              # nothing to average

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
        assert _logp(tracker, 1)[20] > _logp(tracker, 1)[10]

    def test_a_candidate_that_died_closes_the_window(self):
        tracker = BirthTracker()
        _update(tracker, 0.0, 0, [1, 1], [10, 20], [0.4, 0.4], [0.6, 0.4], alive=[10, 20])
        _update(tracker, 0.05, 1, [1, 1], [10, 20], [0.4, 0.4], [0.0, 1.0], alive=[10])
        assert _finalized(tracker) == {1: 10}                                   # 20 merged away: that output is not used

    def test_alive_defaults_to_the_hosts_of_the_feed(self):
        tracker = BirthTracker()
        _update(tracker, 0.0, 0, [1, 1], [10, 20], [0.4, 0.4], [0.6, 0.4])
        _update(tracker, 0.05, 1, [1, 2], [10, 10], [0.4, 0.4], [1.0, 1.0])      # nobody is paired with 20 now
        assert _finalized(tracker) == {1: 10, 2: 10}                             # 2 is new, with a single host

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
        assert tracker.current_birth_map()[20] == {1}                           # 20 leads while the window is open
        assert tracker.finalize().birth_id.tolist() == [20]

    def test_evidence_is_the_leaky_likelihood(self):
        tracker = BirthTracker(budget=5)
        _update(tracker, 0.0, 0, [1, 1], [10, 20], [0.4, 0.4], [0.7, 0.3])        # L = 0.4
        _update(tracker, 0.1, 1, [1, 1], [10, 20], [0.4, 0.4], [0.2, 0.8])
        lam = _lam(0.1, 0.4)
        assert _logp(tracker, 1) == {10: pytest.approx(np.log(0.7) + np.log(lam + (1 - lam) * 0.2)),
                                     20: pytest.approx(np.log(0.3) + np.log(lam + (1 - lam) * 0.8))}

    def test_dt_is_measured_from_the_last_output_used(self):
        tracker = BirthTracker(budget=5)
        _update(tracker, 0.0, 0, [1, 1], [10, 20], [0.4, 0.4], [0.5, 0.5])
        _update(tracker, 0.1, 1, [1, 1], [10, 20], [0.4, 0.4], [0.5, 0.5])
        before = _logp(tracker, 1)
        _update(tracker, 0.2, 2, [1, 1], [10, 20], [0.4, 0.4], [0.3, 0.7])
        lam = _lam(0.1, 0.4)
        assert _logp(tracker, 1)[20] - before[20] == pytest.approx(np.log(lam + (1 - lam) * 0.7))

    def test_outputs_closer_than_the_correlation_time_count_little(self):
        tracker = BirthTracker(budget=5)
        _update(tracker, 0.0, 0, [1, 1], [10, 20], [0.4, 0.4], [0.6, 0.4])       # tau_c = 0.08
        _update(tracker, 0.001, 1, [1, 1], [10, 20], [0.4, 0.4], [0.0, 1.0])
        assert abs(_logp(tracker, 1)[20] - np.log(0.4)) < 0.02
        assert abs(_logp(tracker, 1)[10] - np.log(0.6)) < 0.1

    def test_a_candidate_missing_from_an_output_loses_the_look(self):
        tracker = BirthTracker(budget=5)
        _update(tracker, 0.0, 0, [1, 1], [10, 20], [0.4, 0.4], [0.5, 0.5], alive=[10, 20])
        _update(tracker, 0.1, 1, [1], [10], [0.4], [1.0], alive=[10, 20])        # no pair with 20: its share is zero
        lam = _lam(0.1, 0.4)
        assert _logp(tracker, 1)[20] - np.log(0.5) == pytest.approx(np.log(lam))

    def test_an_unbound_output_adds_nothing(self):
        tracker = BirthTracker()
        _update(tracker, 0.0, 0, [1, 1], [10, 20], [0.4, 0.4], [0.6, 0.4], alive=[10, 20])
        before = _logp(tracker, 1)
        _update(tracker, 0.1, 1, [1], [-1], [NAN], alive=[10, 20])
        assert _logp(tracker, 1) == before
        _update(tracker, 0.15, 2, [1, 1], [10, 20], [0.4, 0.4], [0.5, 0.5], alive=[10, 20])
        lam = _lam(0.15, 0.4)                                                   # dt is still measured from the first output
        assert _logp(tracker, 1)[10] - before[10] == pytest.approx(np.log(lam + (1 - lam) * 0.5))

    def test_equal_evidence_goes_to_the_lowest_host(self):
        tracker = BirthTracker()
        _update(tracker, 0.0, 0, [1, 1, 1], [30, 10, 20], [0.4, 0.4, 0.4], [1.0, 1.0, 1.0])
        assert tracker.finalize().birth_id.tolist() == [10]

    def test_duplicate_pairs_are_merged(self):
        tracker = BirthTracker()
        _update(tracker, 0.0, 0, [1, 1, 1], [10, 10, 20], [0.2, 0.2, 0.2], [0.25, 0.25, 0.4])
        assert _logp(tracker, 1) == {10: pytest.approx(np.log(0.5 / 0.9)), 20: pytest.approx(np.log(0.4 / 0.9))}


class TestCandidates:
    def test_only_hosts_above_the_threshold_compete(self):
        tracker = BirthTracker(min_initial_resp=0.1)
        _update(tracker, 0.0, 0, [1, 1, 1], [10, 20, 30], [0.4, 0.4, 0.4], [0.5, 0.45, 0.05], alive=[10, 20, 30])
        assert set(_logp(tracker, 1)) == {10, 20}
        _update(tracker, 0.05, 1, [1, 1], [20, 30], [0.4, 0.4], [0.1, 0.9], alive=[10, 20, 30])      # 30 was not a candidate
        assert set(_logp(tracker, 1)) == {10, 20}

    def test_the_largest_host_always_competes(self):
        tracker = BirthTracker(min_initial_resp=0.5)
        _update(tracker, 0.0, 0, [1, 1, 1, 1], [10, 20, 30, 40], [0.4] * 4, [0.26, 0.25, 0.25, 0.24])
        assert set(_logp(tracker, 1)) == {10}

    def test_zero_threshold_keeps_every_bound_host(self):
        tracker = BirthTracker(min_initial_resp=0.0)
        _update(tracker, 0.0, 0, [1, 1, 1], [10, 20, 30], [0.4, 0.4, 0.4], [0.5, 0.45, 1e-6])
        assert set(_logp(tracker, 1)) == {10, 20, 30}

    def test_negative_threshold_lets_later_hosts_join(self):
        tracker = BirthTracker(min_initial_resp=-1)
        _update(tracker, 0.0, 0, [1, 1], [10, 20], [0.4, 0.4], [0.5, 0.5], alive=[10, 20, 30])
        _update(tracker, 0.05, 1, [1, 1], [10, 30], [0.4, 0.4], [0.5, 0.5], alive=[10, 20, 30])
        logp = _logp(tracker, 1)
        assert set(logp) == {10, 20, 30}
        lam = _lam(0.05, 0.4)
        assert logp[30] == pytest.approx(np.log(BIRTH_LATE_PRIOR) + np.log(lam + (1 - lam) * 0.5))
        assert logp[20] == pytest.approx(np.log(0.5) + np.log(lam))


class TestOutputs:
    def test_current_birth_map(self):
        tracker = BirthTracker()
        _update(tracker, 0.0, 0, [1, 2, 2], [10, 20, 30], [0.2, 0.4, 0.4], [1.0, 0.6, 0.4])
        bmap = tracker.current_birth_map()
        assert bmap == {10: {1}, 20: {2}}                                        # 1 finalised, 2 open under its leader

    def test_finalize_closes_everything(self):
        tracker = BirthTracker()
        _update(tracker, 0.0, 0, [1, 2, 2, 3], [10, 20, 30, 10], [0.2, 0.4, 0.4, 0.3], [1.0, 0.7, 0.3, 1.0])
        df = tracker.finalize()
        assert dict(zip(df["particle_index"], df["birth_id"])) == {1: 10, 2: 20, 3: 10}
        assert df["particle_index"].dtype == np.uint64 and df["birth_id"].dtype == np.int64
        assert df["p_birth"].dtype == np.float32
        assert df["p_birth"][df["particle_index"] == 2].item() == pytest.approx(0.7)
        assert _open(tracker) == set()

    def test_empty_update(self):
        tracker = BirthTracker()
        _update(tracker, 0.0, 0, np.array([], int), np.array([], int), np.array([]))
        assert _open(tracker) == set() and _finalized(tracker) == {}
        assert tracker.finalize().empty

    def test_an_earlier_snapshot_is_ignored(self):
        tracker = BirthTracker()
        _update(tracker, 1.0, 5, [1], [10], [0.2])
        _update(tracker, 0.5, 4, [2], [10], [0.2])
        assert _finalized(tracker) == {1: 10}


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
        restored._set_state(tracker._get_state())
        assert restored._last_snapshot == 3 and restored.budget == tracker.budget
        self._run(restored, range(4, 8))
        assert restored.current_birth_map() == ref.current_birth_map()
        assert restored.finalize().equals(ref.finalize())

    def test_an_earlier_tracker_checkpoint_is_refused(self):
        tracker = BirthTracker()
        with pytest.raises(ValueError, match="earlier birth tracker"):
            tracker._set_state({"format": 2, "active": {}, "evidence": {}, "finalized": {}, "last_snapshot": 0})
        with pytest.raises(ValueError, match="earlier birth tracker"):
            tracker._set_state({"active": {}, "finalized": {}, "heap": [], "last_snapshot": 0})

    def test_cheap_snapshot_unaffected_by_later_update(self):
        # C02: a reference captured now must survive later updates untouched
        tracker = BirthTracker()
        _update(tracker, 0.0, 0, [1, 1], [10, 20], [0.4, 0.4], [0.6, 0.4])
        ref = tracker._cheap_snapshot()
        before = {k: v.copy() for k, v in ref["ev"].items()}
        before_act = {k: v.copy() for k, v in ref["act"].items()}
        _update(tracker, 0.1, 1, [1, 1, 2], [10, 20, 10], [0.4, 0.4, 0.2], [0.1, 0.9, 1.0])
        _update(tracker, 5.0, 2, [1], [10], [0.4])                              # closes particle 1
        for k, v in before.items():
            np.testing.assert_array_equal(ref["ev"][k], v)
        for k, v in before_act.items():
            np.testing.assert_array_equal(ref["act"][k], v)

    def test_birth_sets_accumulate(self):
        tracker = BirthTracker()
        _update(tracker, 0.0, 0, [1], [10], [0.2])
        first = tracker._birth_map[10]
        assert first == {1}
        _update(tracker, 1.0, 1, [2], [10], [0.2])
        assert tracker._birth_map[10] == {1, 2}
        assert first == {1}                                                     # replaced, not edited


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
                    self.done[pid] = (h[int(np.argmax(p))], p.max())
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
        order = sorted(range(len(st["hosts"])), key=lambda i: st["hosts"][i])
        w = np.exp(st["lp"] - st["lp"].max())
        k = next(i for i in order if w[i] == w.max())
        self.done[pid] = (st["hosts"][k], w[k] / w.sum())


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
        for t, s, P, H, T, W, alive in _random_feeds(np.random.default_rng(seed)):
            tracker.update(t, s, np.array(P, np.uint64), np.array(H), np.array(T), np.array(W), alive=alive)
            ref.update(t, P, H, T, W, alive)
        out = tracker.finalize().set_index("particle_index")
        for pid in list(ref.open):
            ref.close(pid)
        assert set(out.index.tolist()) == set(ref.done)
        for pid, (host, prob) in ref.done.items():
            assert out.birth_id[pid] == host, (seed, pid)
            assert out.p_birth[pid] == pytest.approx(prob, abs=2e-6)
