"""Assembly and accretion history tracker.

Tracks which particles were accreted from which satellite, building
a map of each galaxy's infall-list across snapshots.
"""

from collections import defaultdict, deque

from tqdm import tqdm

from roadrunner._defaults import UNBOUND

_EMPTY = frozenset()


class AssemblyTracker:
    """Tracks the assembly history of each galaxy across snapshots.

    Maintains an infall list per galaxy: the set of particle IDs that
    have ever been accreted by that galaxy (including those inherited
    from merged satellites).

    Parameters
    ----------
    n_sat_history : int, default=2
        Number of past satellite maps to buffer for history queries.
    unbound_default : int, default=UNBOUND (-1)
        Galaxy ID used for unbound particles.
    """

    def __init__(self, n_sat_history=2, unbound_default=UNBOUND):
        self._infall_lists = defaultdict(set)
        self._previous_birth_map = defaultdict(set)
        self._unbound_default = unbound_default
        self._frozen = set([unbound_default])
        self._last_snapshot = -1
        self._sat_history_buffer = deque(maxlen=n_sat_history - 1)

    def _get_current_sat_history(self, satellites_map):
        """Combine current and buffered satellite maps into one history.

        Parameters
        ----------
        satellites_map : dict of {int: set of int}

        Returns
        -------
        sat_history : dict of {int: set of int}
        """
        sat_history = {g: set(sats) for g, sats in satellites_map.items()}
        for snap_map in self._sat_history_buffer:
            for g in sat_history:
                sat_history[g].update(snap_map.get(g, _EMPTY))
        return sat_history

    def _strongly_connected_components(self, nodes, graph):
        """Tarjan's SCC algorithm, restricted to ``nodes``.

        A same-snapshot satellite graph is always acyclic: a candidate
        can only become a host's satellite if it is strictly less
        massive (see ``MergerTreeHandlerCSV._satellites_impl``), so a
        cycle of any length within one snapshot would require a mass
        ordering contradiction. A genuine cycle can only arise from the
        merged, buffered history built by :meth:`_get_current_sat_history`
        -- e.g. a host/satellite swap spanning one buffered snapshot --
        since mass, position, and host/satellite relations genuinely
        change between snapshots.

        O(V+E), the same complexity as a plain DFS: this is literally
        one DFS pass with two extra O(1)-per-node bookkeeping values (a
        discovery index and a "lowlink") and a stack. For an acyclic
        graph every component is a singleton, in exactly the order a
        plain post-order DFS already gives -- this is a strict
        generalisation, not a different algorithm for the common case.

        Parameters
        ----------
        nodes : set of int
        graph : dict of {int: set of int}
            Edges outside ``nodes`` are ignored.

        Returns
        -------
        components : list of list of int
            Emitted in an order where a component ``graph`` points to
            (a satellite's) comes out before the component that points
            to it (its host) -- the "deepest satellite first" property
            needed so a host reads its satellites' fully-updated infall
            lists. A genuine cycle comes out as one multi-galaxy
            component instead of raising.
        """
        index_counter = [0]
        stack, on_stack, indices, lowlink, components = [], set(), {}, {}, []

        def strongconnect(v):
            """Tarjan's iterative-in-spirit (recursive) DFS core.

            Parameters
            ----------
            v : int
                Node to visit.
            """
            indices[v] = lowlink[v] = index_counter[0]
            index_counter[0] += 1
            stack.append(v)
            on_stack.add(v)
            for w in graph.get(v, _EMPTY):
                if w not in nodes:
                    continue
                if w not in indices:
                    strongconnect(w)
                    lowlink[v] = min(lowlink[v], lowlink[w])
                elif w in on_stack:
                    lowlink[v] = min(lowlink[v], indices[w])
            if lowlink[v] == indices[v]:
                component = []
                while True:
                    w = stack.pop()
                    on_stack.discard(w)
                    component.append(w)
                    if w == v:
                        break
                components.append(component)

        for v in nodes:
            if v not in indices:
                strongconnect(v)
        return components

    def _flatten_components(self, components):
        """Flatten Tarjan's satellite-first component order into one galaxy list.

        A singleton contributes itself once. A genuine multi-galaxy
        cycle contributes its members, sorted by galaxy ID for a
        deterministic tie-break, repeated ``len(component)`` times.

        A 2-galaxy cyclic component (the realistic case) is exact after
        a single pass in either order -- the tie-break there affects
        only determinism, not correctness. A 3+-galaxy cyclic component
        does not have that property in general, so the ``k``-repeat (a
        standard sufficient bound for a monotone fixed point over ``k``
        mutually-dependent variables: growth is bounded by the finite
        particle universe, so any pass that is not already the fixed
        point must have grown at least one member's set) is applied
        uniformly rather than special-casing size 2 vs. 3+. Verified
        against a while-loop-to-convergence reference across thousands
        of randomised multi-round scenarios (identical results) and
        found consistently faster in practice than that reference.

        Parameters
        ----------
        components : list of list of int
            Tarjan output, satellite-first order.

        Returns
        -------
        ordered : list of int
        """
        ordered = []
        for component in components:
            if len(component) == 1:
                ordered.append(component[0])
            else:
                members = sorted(component)
                ordered.extend(members * len(members))
        return ordered

    def update(self, snapshot_id, assignment_map, birth_map, satellites_map, freeze_galaxies=None):
        """Update the assembly tracker with data from a new snapshot.

        Parameters
        ----------
        snapshot_id : int
        assignment_map : dict of {int: set of int}
        birth_map : dict of {int: set of int}
        satellites_map : dict of {int: set of int}
        freeze_galaxies : list of int or None, optional
        """
        if snapshot_id > self._last_snapshot:
            self._last_snapshot = snapshot_id
            all_galaxies = (
                set(assignment_map.keys())
                | set(satellites_map.keys())
                | set(h for sats in satellites_map.values() for h in sats)
            )
            satellites_history = self._get_current_sat_history(satellites_map)

            self._update(
                all_galaxies,
                assignment_map,
                birth_map,
                satellites_history,
                [] if freeze_galaxies is None else freeze_galaxies,
            )
            self._sat_history_buffer.append(satellites_map)

    def _update(self, all_galaxies, assignment_map, birth_map, satellites_history, freeze_galaxies):
        """Core update logic: propagate infall lists across satellites.

        Two flat passes, not one combined loop: birth-map deltas have
        zero cross-galaxy dependency and must fully complete for every
        galaxy before satellite accretion starts, since accretion reads
        other galaxies' infall lists. Combining the two into a single
        per-galaxy step was tried and found to let a host, if visited
        before one of its satellites in the same pass, read that
        satellite's infall list before the satellite's own birth-delta
        "erase" had run this round -- picking up a particle that should
        already have been removed.

        Parameters
        ----------
        all_galaxies : set of int
        assignment_map : dict of {int: set of int}
        birth_map : dict of {int: set of int}
        satellites_history : dict of {int: set of int}
            The merged (current + buffered) satellite graph, already
            built by :meth:`_get_current_sat_history`.
        freeze_galaxies : list of int
        """
        for g in freeze_galaxies:
            self._frozen.add(g)

        # Pass 1: birth-map deltas. Order-independent -- apply for every
        # galaxy before any accretion reads an infall list.
        for g in all_galaxies:
            prev_birth = self._previous_birth_map.get(g, _EMPTY)
            curr_birth = birth_map.get(g, _EMPTY)
            self._infall_lists[g].update(curr_birth - prev_birth)
            self._infall_lists[g].difference_update(prev_birth - curr_birth)

        # Pass 2: satellite accretion, SCC-ordered and flattened (see
        # _strongly_connected_components / _flatten_components). The
        # original ``A_g & curr_birth`` term is dropped here: after pass 1
        # has run for every galaxy, ``curr_birth <= self._infall_lists[g]``
        # already holds, so intersecting with it again adds nothing
        # (verified empirically, not just asserted).
        components = self._strongly_connected_components(all_galaxies, satellites_history)
        for g in tqdm(self._flatten_components(components), desc="Updating each galaxy..."):
            if g in self._frozen:
                continue
            A_g = assignment_map.get(g, _EMPTY)
            # Distributive form of ``A_g & (unbound | sats...)``: intersect
            # per component so the full union is never materialised.
            accreted = A_g & self._infall_lists.get(self._unbound_default, _EMPTY)
            for h in satellites_history.get(g, _EMPTY):
                accreted |= A_g & self._infall_lists.get(h, _EMPTY)
            self._infall_lists[g].update(accreted)

        self._previous_birth_map = dict(birth_map)

    def current(self):
        """Return the current infall lists.

        Returns
        -------
        infall_lists : dict of {int: set of int}
        """
        return self._infall_lists

    def _get_state(self):
        """Serialise the tracker state for checkpointing.

        Returns
        -------
        state : dict
        """
        return {
            "infall_lists": {k: list(v) for k, v in self._infall_lists.items()},
            "previous_birth_map": {k: list(v) for k, v in self._previous_birth_map.items()},
            "frozen": set(self._frozen),
            "sat_history_buffer": list(self._sat_history_buffer),
            "last_snapshot": self._last_snapshot,
            "unbound_default": self._unbound_default,
        }

    def _set_state(self, state):
        """Restore tracker state from a checkpoint.

        Parameters
        ----------
        state : dict
        """
        self._infall_lists = defaultdict(set, {k: set(v) for k, v in state["infall_lists"].items()})
        self._previous_birth_map = defaultdict(set, {k: set(v) for k, v in state["previous_birth_map"].items()})
        self._frozen = state["frozen"]
        self._sat_history_buffer = deque(state["sat_history_buffer"], maxlen=self._sat_history_buffer.maxlen)
        self._last_snapshot = state["last_snapshot"]
        self._unbound_default = state["unbound_default"]
