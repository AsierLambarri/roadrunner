#############################################################################
#
# package:   roadrunner.postprocessing.tracking
# file:      assembly.py
# brief:     Assembly and accretion history tracker.
#
# copyright: GPLv3
# author:    Asier Lambarri Martinez
# changes:   18 may 2026 - Created
#            18 may 2026 - Last edit
#
#############################################################################

from collections import defaultdict, deque

from tqdm import tqdm

from roadrunner._exceptions import CycleError


class AssemblyTracker:
    def __init__(self, n_sat_history=2, unbound_default=-1):
        self._infall_lists = defaultdict(set)
        self._previous_birth_map = defaultdict(set)
        self._unbound_default = unbound_default
        self._frozen = set([unbound_default])
        self._last_snapshot = -1
        self._sat_history_buffer = deque(maxlen=n_sat_history - 1)

    def _get_current_sat_history(self, satellites_map):
        sat_history = {g: set(sats) for g, sats in satellites_map.items()}
        for snap_map in self._sat_history_buffer:
            for g in sat_history:
                sat_history[g].update(snap_map.get(g, []))
        return sat_history

    def _order_galaxies_bottom_up(self, warm_galaxies, satellites_history):
        ordered = []
        visited = set()
        visiting = set()

        def dfs(g):
            if g in visited:
                return
            if g in visiting:
                raise CycleError(
                    f"Cycle detected in satellite graph at galaxy {g}"
                )
            visiting.add(g)
            for h in satellites_history.get(g, []):
                if h in warm_galaxies:
                    dfs(h)
            visiting.remove(g)
            visited.add(g)
            ordered.append(g)

        for g in warm_galaxies:
            if g not in visited:
                dfs(g)
        return ordered

    def update(self, snapshot_id, assignment_map, birth_map, satellites_map, freeze_galaxies=None):
        if snapshot_id > self._last_snapshot:
            self._last_snapshot = snapshot_id
            all_galaxies = (
                set(assignment_map.keys())
                | set(satellites_map.keys())
                | set(h for sats in satellites_map.values() for h in sats)
            )
            ordered_galaxies = self._order_galaxies_bottom_up(all_galaxies, satellites_map)

            self._update(
                snapshot_id,
                ordered_galaxies,
                assignment_map,
                birth_map,
                satellites_map,
                [] if freeze_galaxies is None else freeze_galaxies,
            )
            self._sat_history_buffer.append(satellites_map)

    def _update(self, snapshot_id, ordered_galaxies, assignment_map, birth_map, satellites_map, freeze_galaxies):
        for g in freeze_galaxies:
            self._frozen.add(g)

        satellites_history = self._get_current_sat_history(satellites_map)
        for g in tqdm(ordered_galaxies, desc="Updating each galaxy..."):
            prev_birth = self._previous_birth_map.get(g, set())
            curr_birth = birth_map.get(g, set())

            erase = prev_birth - curr_birth
            add = curr_birth - prev_birth

            self._infall_lists[g].update(add)
            self._infall_lists[g].difference_update(erase)

            if g in self._frozen:
                continue

            A_g = assignment_map.get(g, set())
            valid = curr_birth | self._infall_lists.get(self._unbound_default, set())
            for h in satellites_history.get(g, []):
                valid |= self._infall_lists.get(h, set())

            self._infall_lists[g].update(A_g & valid)

        self._previous_birth_map = dict(birth_map)

    def current(self):
        return self._infall_lists

    def _get_state(self):
        return {
            "infall_lists": {k: list(v) for k, v in self._infall_lists.items()},
            "previous_birth_map": {k: list(v) for k, v in self._previous_birth_map.items()},
            "frozen": set(self._frozen),
            "sat_history_buffer": list(self._sat_history_buffer),
            "last_snapshot": self._last_snapshot,
            "unbound_default": self._unbound_default,
        }

    def _set_state(self, state):
        self._infall_lists = defaultdict(set, {k: set(v) for k, v in state["infall_lists"].items()})
        self._previous_birth_map = defaultdict(set, {k: set(v) for k, v in state["previous_birth_map"].items()})
        self._frozen = state["frozen"]
        self._sat_history_buffer = deque(state["sat_history_buffer"], maxlen=self._sat_history_buffer.maxlen)
        self._last_snapshot = state["last_snapshot"]
        self._unbound_default = state["unbound_default"]
