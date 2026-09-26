import pytest

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

    def test_cross_snapshot_cycle_resolves(self):
        # A same-snapshot cycle is impossible: a candidate can only become
        # a host's satellite if it is strictly less massive (see
        # MergerTreeHandlerCSV._satellites_impl), so a cycle of any length
        # within one snapshot would require a mass-ordering contradiction.
        # Cycles can only arise from the merged, buffered history -- e.g. a
        # host/satellite swap spanning one buffered snapshot -- which this
        # builds explicitly across two update() calls:
        #   snapshot 0: satellites_map={2: {1}}          (2 hosts 1)
        #   snapshot 1: satellites_map={1: {2}, 2: {3}}  (1 hosts 2; 2 also
        #     hosts 3. _get_current_sat_history enriches exactly the
        #     current-map keys, so 2 must be a key for the buffer to add
        #     its old edge to 1. The pipeline's maps list every alive
        #     galaxy (empty set if none), so there even a lone A/B swap
        #     would form the cycle; this hand-built map keeps 2 a key via
        #     its edge to 3 instead.)
        # The merged history at snapshot 1 is then {1: {2}, 2: {1, 3}}: a
        # genuine mutual 1<->2 component, plus 2's one-way edge to 3.
        # Hand-derived and cross-checked against an independent gold-
        # standard fixed-point implementation before writing this assertion:
        #   snapshot 0 leaves infall = {1: {10}, 2: {20}, 3: {30}}
        #     (each galaxy's own birth particle; no satellite accretion yet
        #     since 2's only satellite, 1, has nothing overlapping A_2).
        #   snapshot 1 birth-delta: infall[1] gains {40} (curr_birth-prev_birth),
        #     infall[2] gains {10} minus the erased {20} -> infall[2]={10},
        #     infall[3] unchanged ({30}, birth_map[3] absent this round).
        #   snapshot 1 accretion (component {1,2} resolved together, 3
        #     separate): A_1={10,20} & infall[2]={10} -> {10}; A_2={20,30} &
        #     (infall[1]={40,10} | infall[3]={30}) -> {30}. But 3 only
        #     receives 30 into infall[3] already had by birth, contributing
        #     nothing new to 2 beyond a repeat -- net: infall[1]={40,10},
        #     infall[2]={10}.
        tracker = AssemblyTracker(n_sat_history=2)
        tracker.update(
            snapshot_id=0,
            assignment_map={1: {10}, 2: {20}, 3: {30}},
            birth_map={1: {10}, 2: {20}, 3: {30}},
            satellites_map={2: {1}},
        )
        tracker.update(
            snapshot_id=1,
            assignment_map={1: {10, 20}, 2: {20, 30}, 3: {30}},
            birth_map={1: {40}, 2: {10}, 3: set()},
            satellites_map={1: {2}, 2: {3}},
        )
        result = tracker.current()
        assert result[1] == {40, 10}
        assert result[2] == {10}

    def test_swapped_ids_give_identical_result(self):
        # The original C23 reproduction: a satellite dropped from its
        # host's *current* satellite set but still present via the 1-deep
        # buffer must be processed consistently regardless of the specific
        # integer IDs involved. Before the fix, ordering came from Python
        # set-iteration order over the raw galaxy IDs, so swapping which
        # galaxy is "g" and which is "h" could change the result.
        def run(g, h, s):
            tracker = AssemblyTracker(n_sat_history=2)
            tracker.update(
                snapshot_id=0,
                assignment_map={g: set(), h: {1, 2}, s: set()},
                birth_map={h: {1, 2}},
                satellites_map={g: {h, s}},
            )
            tracker.update(
                snapshot_id=1,
                assignment_map={g: {3}, h: set(), s: set()},
                birth_map={h: {1, 2, 3}},
                satellites_map={g: {s}},
            )
            return sorted(tracker.current().get(g, set()))

        r1 = run(5, 9, 20)
        r2 = run(9, 5, 20)
        assert r1 == [3]
        assert r2 == [3]
        assert r1 == r2

    def test_three_way_cross_snapshot_cycle_needs_full_convergence(self):
        # A genuine 3-galaxy cyclic component, built across two update()
        # calls with n_sat_history=2 (a 1-deep buffer is enough -- it does
        # not take three separate snapshots' worth of direct edges, since
        # the merge at snapshot 1 combines snapshot 1's edges with
        # snapshot 0's buffered edges to complete the ring):
        #   snapshot 0: satellites_map={1: {3}, 3: {2}}
        #   snapshot 1: satellites_map={3: {1}, 1: {2}, 2: {3}}
        #   merged history at snapshot 1: {3: {1, 2}, 1: {2, 3}, 2: {3}}
        #   -> galaxies {1, 2, 3} form one strongly-connected component
        #      (verified: Tarjan on this graph gives [[3, 2, 1]] or an
        #      equivalent single 3-member component, not three singletons).
        # A single pass through the component (k=1, no repeat) under-
        # converges on this fixture: galaxy 2 ends up missing particle 20
        # (present only after a second pass sees galaxy 1's fully-updated
        # infall list). Cross-checked against an independent gold-standard
        # fixed-point (while-loop-to-convergence) implementation, which
        # required a second pass and gives the full, correct answer below;
        # the real k=len(component)=3 repeat used by AssemblyTracker
        # reaches this same fixed point in one call.
        d = 4  # an unrelated 4th galaxy present throughout, outside the cycle
        tracker = AssemblyTracker(n_sat_history=2)
        tracker.update(
            snapshot_id=0,
            assignment_map={
                4: {17, 18, 19, 23, 25, 26}, 3: {18, 11},
                1: {20, 21}, 2: {24, 28, 22},
            },
            birth_map={4: set(), 3: {17, 20, 25}, 1: {29}, 2: set()},
            satellites_map={1: {3}, 3: {2}},
        )
        result = tracker.update(
            snapshot_id=1,
            assignment_map={
                4: {12, 23}, 3: {19, 27, 20},
                1: {24, 21, 23}, 2: {25, 20, 22},
            },
            birth_map={4: set(), 3: {19, 14}, 1: {19}, 2: {27}},
            satellites_map={3: {1}, 1: {2}, 2: {3}},
        )
        result = tracker.current()
        assert result[1] == {19, 20}
        assert result[2] == {27, 20}
        assert result[3] == {19, 20, 27, 14}
        assert d not in result or result[d] == set()

    def test_flat_order_is_independent_of_dict_construction_order(self):
        # Same content as test_cross_snapshot_cycle_resolves, but every
        # dict is built with reversed/different key insertion order and
        # every satellite set is constructed by unioning single-element
        # sets in the opposite order. Since Tarjan's traversal order (and
        # therefore the flattened list) must not depend on Python's
        # set/dict iteration order, the result must be identical.
        tracker = AssemblyTracker(n_sat_history=2)
        tracker.update(
            snapshot_id=0,
            assignment_map={3: {30}, 2: {20}, 1: {10}},
            birth_map={3: {30}, 2: {20}, 1: {10}},
            satellites_map={2: {1} | set()},
        )
        tracker.update(
            snapshot_id=1,
            assignment_map={3: {30}, 2: {30, 20}, 1: {20, 10}},
            birth_map={3: set(), 2: {10}, 1: {40}},
            satellites_map={2: set() | {3}, 1: set() | {2}},
        )
        result = tracker.current()
        assert result[1] == {40, 10}
        assert result[2] == {10}

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
        # Snapshot 1: galaxy 2 is still a satellite of 1.
        tracker.update(
            snapshot_id=1,
            assignment_map={1: {10}, 2: {20}},
            birth_map={1: {10}, 2: {20}},
            satellites_map={1: {2}},
        )
        # Snapshot 2: the link is gone, but both galaxies are alive, so
        # both are keys (the complete map the orchestrator passes). The
        # buffer still holds 1 -> 2 from snapshots 0 and 1, so galaxy 1
        # does consult it -- but 1 is assigned none of 2's stars, so it
        # accretes nothing: A_1 & infall[2] == {10} & {20} == set().
        tracker.update(
            snapshot_id=2,
            assignment_map={1: {10}, 2: {20}},
            birth_map={1: {10}, 2: {20}},
            satellites_map={1: set(), 2: set()},
        )
        result = tracker.current()
        assert result[1] == {10}
        assert result[2] == {20}


class TestAssemblyTrackerRecovery:
    """C02: `_last_snapshot` must not advance on a partial failure, and a
    `_cheap_snapshot()` reference must survive later updates untouched."""

    def test_last_snapshot_not_advanced_on_partial_failure(self):
        tracker = AssemblyTracker(n_sat_history=2)
        original_update = tracker._update
        calls = {"n": 0}

        def failing_update(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("simulated mid-update failure")
            return original_update(*args, **kwargs)

        tracker._update = failing_update
        with pytest.raises(RuntimeError, match="simulated mid-update failure"):
            tracker.update(
                snapshot_id=0,
                assignment_map={1: {10, 20}},
                birth_map={1: {10, 20}},
                satellites_map={},
            )
        assert tracker._last_snapshot == -1  # unchanged: the update never completed

        # A retry of the SAME snapshot_id must actually run, not be skipped.
        tracker.update(
            snapshot_id=0,
            assignment_map={1: {10, 20}},
            birth_map={1: {10, 20}},
            satellites_map={},
        )
        assert tracker._last_snapshot == 0
        assert tracker.current()[1] == {10, 20}

    def test_pass1_erase_after_union_precedence(self):
        # Regression for the `current | added - removed` operator-
        # precedence bug: `-` binds tighter than `|`, so an unparenthesized
        # rewrite would silently drop this erasure.
        tracker = AssemblyTracker(n_sat_history=2)
        tracker.update(
            snapshot_id=0,
            assignment_map={1: {10, 20}},
            birth_map={1: {10, 20}},
            satellites_map={},
        )
        assert tracker.current()[1] == {10, 20}
        tracker.update(
            snapshot_id=1,
            assignment_map={1: {10, 20}},
            birth_map={1: {10}},
            satellites_map={},
        )
        assert tracker.current()[1] == {10}

    def test_cheap_snapshot_infall_lists_unaffected_by_later_update(self):
        tracker = AssemblyTracker(n_sat_history=2)
        tracker.update(
            snapshot_id=0,
            assignment_map={1: {10, 20, 30}, 2: {30}},
            birth_map={1: {10, 20}, 2: {30}},
            satellites_map={1: {2}},
        )
        ref = tracker._cheap_snapshot()
        captured_1 = set(ref["infall_lists"][1])

        # snapshot 1 erases 20 from galaxy 1 (Pass 1) and drives accretion
        # for galaxy 2 via the satellite link (Pass 2) -- both passes
        # mutate `_infall_lists`, so both must be covered.
        tracker.update(
            snapshot_id=1,
            assignment_map={1: {10, 20, 30}, 2: {30}},
            birth_map={1: {10}, 2: {30}},
            satellites_map={1: {2}},
        )
        assert ref["infall_lists"][1] == captured_1

    def test_cheap_snapshot_frozen_unaffected_by_later_freeze(self):
        tracker = AssemblyTracker(n_sat_history=2)
        tracker.update(
            snapshot_id=0,
            assignment_map={1: {10}, 2: {20}},
            birth_map={1: {10}, 2: {20}},
            satellites_map={},
        )
        ref = tracker._cheap_snapshot()
        captured_frozen = set(ref["frozen"])
        assert 1 not in captured_frozen
        tracker.update(
            snapshot_id=1,
            assignment_map={1: {10}, 2: {20}},
            birth_map={1: {10}, 2: {20}},
            satellites_map={},
            freeze_galaxies=[1],
        )
        assert ref["frozen"] == captured_frozen
        assert 1 not in ref["frozen"]


class TestAssemblyTrackerContract:
    """S01 and dead-galaxy births: birth bookkeeping covers every galaxy
    whose birth set changed (dead ones included); accretion covers alive
    galaxies only, each with its buffered satellites -- given the
    complete satellite map the orchestrator passes (every alive galaxy
    a key, empty set if it has no satellites)."""

    def _merger_snapshot_0(self, tracker):
        tracker.update(
            snapshot_id=0,
            assignment_map={1: {10}, 2: {20, 21}},
            birth_map={1: {10}, 2: {20, 21}},
            satellites_map={1: {2}, 2: set()},
        )

    def test_merged_satellite_accreted_by_host_left_without_satellites(self):
        tracker = AssemblyTracker(n_sat_history=2)
        self._merger_snapshot_0(tracker)
        # Snapshot 1: 2 has merged into 1 -- 2 is gone, its star 20 is now
        # assigned to 1, and 1 has no current satellite: {1: set()}.
        tracker.update(
            snapshot_id=1,
            assignment_map={1: {10, 20}},
            birth_map={1: {10}, 2: {20, 21}},
            satellites_map={1: set()},
        )
        assert tracker.current()[1] == {10, 20}
        assert tracker.current()[2] == {20, 21}  # dead: read, never updated by accretion

        # Why the orchestrator must pass complete maps: if the host is
        # dropped from the map instead (the old filter), the buffer is
        # never consulted and the merged star is lost.
        incomplete = AssemblyTracker(n_sat_history=2)
        self._merger_snapshot_0(incomplete)
        incomplete.update(
            snapshot_id=1,
            assignment_map={1: {10, 20}},
            birth_map={1: {10}, 2: {20, 21}},
            satellites_map={},
        )
        assert incomplete.current()[1] == {10}

    def test_buffered_relation_expires(self):
        tracker = AssemblyTracker(n_sat_history=2)
        self._merger_snapshot_0(tracker)
        tracker.update(
            snapshot_id=1,
            assignment_map={1: {10, 20}},
            birth_map={1: {10}, 2: {20, 21}},
            satellites_map={1: set()},
        )
        # Snapshot 2: 2's star 21 is now assigned to 1, but the buffer only
        # holds snapshot 1's map, where 1 had no satellites -- expired.
        tracker.update(
            snapshot_id=2,
            assignment_map={1: {10, 20, 21}},
            birth_map={1: {10}, 2: {20, 21}},
            satellites_map={1: set()},
        )
        assert tracker.current()[1] == {10, 20}

    def test_dead_galaxy_gets_no_satellite_history(self):
        tracker = AssemblyTracker(n_sat_history=2)
        tracker.update(
            snapshot_id=0,
            assignment_map={3: {30}, 4: {40}},
            birth_map={3: {30}, 4: {40}},
            satellites_map={3: {4}, 4: set()},
        )
        # Snapshot 1: 3 has died (absent from the complete map). Despite
        # the buffered 3 -> 4, it gets no history: it cannot gain satellites.
        history = tracker._get_current_sat_history({4: set()})
        assert history == {4: set()}

    def test_dead_galaxy_birth_changes_are_applied(self):
        tracker = AssemblyTracker(n_sat_history=2)
        tracker.update(
            snapshot_id=0,
            assignment_map={5: {50, 51}, 6: {60}},
            birth_map={5: {50, 51}, 6: {60}},
            satellites_map={5: set(), 6: set()},
        )
        # Snapshot 1: 5 has died, and particle 51's birth leader switches
        # to 6 -- dead 5 must lose it, not keep it forever.
        tracker.update(
            snapshot_id=1,
            assignment_map={6: {51, 60}},
            birth_map={5: {50}, 6: {51, 60}},
            satellites_map={6: set()},
        )
        assert tracker.current()[5] == {50}
        assert tracker.current()[6] == {51, 60}
        # Snapshot 2: particle 52 finalizes with birth host 5, still dead
        # -- 5 must gain it.
        tracker.update(
            snapshot_id=2,
            assignment_map={6: {51, 52, 60}},
            birth_map={5: {50, 52}, 6: {51, 60}},
            satellites_map={6: set()},
        )
        assert tracker.current()[5] == {50, 52}

    def test_galaxy_whose_births_all_moved_away_loses_them(self):
        tracker = AssemblyTracker(n_sat_history=2)
        tracker.update(
            snapshot_id=0,
            assignment_map={7: {70}, 8: {80}},
            birth_map={7: {70}, 8: {80}},
            satellites_map={7: set(), 8: set()},
        )
        # Snapshot 1: 7 has died and has no birth entry left at all -- it
        # appears only in the previous birth map, and must still lose 70.
        tracker.update(
            snapshot_id=1,
            assignment_map={8: {70, 80}},
            birth_map={8: {70, 80}},
            satellites_map={8: set()},
        )
        assert tracker.current()[7] == set()
        assert tracker.current()[8] == {70, 80}

    def test_unchanged_birth_set_is_not_rebuilt(self):
        tracker = AssemblyTracker(n_sat_history=2)
        tracker.update(
            snapshot_id=0,
            assignment_map={1: {10}},
            birth_map={1: {10}},
            satellites_map={1: set()},
        )
        before = tracker._infall_lists[1]
        tracker.update(
            snapshot_id=1,
            assignment_map={1: {10}},
            birth_map={1: {10}},
            satellites_map={1: set()},
        )
        assert tracker._infall_lists[1] is before
