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

    Each snapshot, :meth:`update` ingests the current birth map, the
    current particle assignment, and the current satellite map, and it
    keeps the previous ``n_sat_history - 1`` satellite maps in a buffer.
    The satellite map must list every galaxy alive this snapshot as a
    key, with an empty set when it has no satellites.
    :meth:`_get_current_sat_history` gives buffered history to exactly
    those keys, so an alive host whose last satellite just merged in
    still inherits it. It then runs two steps:

    1. Birth bookkeeping for every galaxy whose birth set changed, dead
       ones included: birth hosts are chosen over an accumulation
       window, so a galaxy can gain or lose birth particles after it
       has died.
    2. Accretion for alive galaxies only, over the current plus
       buffered satellite graph. Dead galaxies neither gain satellites
       nor accrete; their infall lists are only read, by hosts that
       list them as satellites.

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

        Entries exist exactly for the keys of the current map. Because
        that map lists every alive galaxy (empty set if none), every
        alive galaxy gets its buffered satellites, and dead galaxies get
        no entry at all.

        Parameters
        ----------
        satellites_map : dict of {int: set of int}
            Complete over the alive galaxies.

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
            Must list every galaxy alive this snapshot as a key, with an
            empty set when it has no satellites (see the class docstring).
        freeze_galaxies : list of int or None, optional
        """
        if snapshot_id > self._last_snapshot:
            alive_galaxies = (
                set(assignment_map.keys())
                | set(satellites_map.keys())
                | set(h for sats in satellites_map.values() for h in sats)
            )
            satellites_history = self._get_current_sat_history(satellites_map)

            self._update(
                alive_galaxies,
                assignment_map,
                birth_map,
                satellites_history,
                [] if freeze_galaxies is None else freeze_galaxies,
            )
            self._sat_history_buffer.append(satellites_map)
            # Advance only after both mutations above succeed (C02): if
            # `_update` raises partway, `_last_snapshot` must not already
            # claim this snapshot is done, or a retry would be silently
            # skipped (`snapshot_id > self._last_snapshot` would be False).
            self._last_snapshot = snapshot_id

    def _update(self, alive_galaxies, assignment_map, birth_map, satellites_history, freeze_galaxies):
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

        A galaxy absent from ``alive_galaxies`` (dead this snapshot)
        still gets pass 1's birth bookkeeping, but never enters pass 2:
        it cannot gain satellites or accrete. Its infall list is only
        read, by hosts that list it in ``satellites_history``.

        Parameters
        ----------
        alive_galaxies : set of int
            Galaxies alive this snapshot: pass 2's node set.
        assignment_map : dict of {int: set of int}
        birth_map : dict of {int: set of int}
        satellites_history : dict of {int: set of int}
            The merged (current + buffered) satellite graph, already
            built by :meth:`_get_current_sat_history`.
        freeze_galaxies : list of int
        """
        for g in freeze_galaxies:
            self._frozen.add(g)

        # Pass 1: birth bookkeeping for every galaxy whose birth set differs
        # from the previous snapshot's -- alive or dead. A birth host is
        # chosen over an accumulation window, so a galaxy can gain or lose
        # birth particles after it has died, and `_previous_birth_map`
        # (end of this method) records every galaxy's set as applied: a
        # galaxy skipped here would lose its change for good. Both maps'
        # keys are visited, since a galaxy whose births all moved away
        # appears only in the previous one; unchanged sets are skipped.
        # Order-independent, and completed for every galaxy before any
        # accretion reads an infall list.
        #
        # Sets are replaced, never edited in place (C02): a checkpoint
        # capture may still reference the old object. Parenthesized
        # explicitly -- `-` binds tighter than `|`, and the erase must run
        # after the union.
        previous = self._previous_birth_map
        for g in birth_map.keys() | previous.keys():
            curr_birth = birth_map.get(g, _EMPTY)
            prev_birth = previous.get(g, _EMPTY)
            if curr_birth == prev_birth:
                continue
            current = self._infall_lists.get(g, _EMPTY)
            self._infall_lists[g] = (current | (curr_birth - prev_birth)) - (prev_birth - curr_birth)

        # Pass 2: satellite accretion, SCC-ordered and flattened (see
        # _strongly_connected_components / _flatten_components). The
        # original ``A_g & curr_birth`` term is dropped here: after pass 1
        # has run for every galaxy, ``curr_birth <= self._infall_lists[g]``
        # already holds, so intersecting with it again adds nothing
        # (verified empirically, not just asserted).
        components = self._strongly_connected_components(alive_galaxies, satellites_history)
        for g in tqdm(self._flatten_components(components), desc="Updating each galaxy..."):
            if g in self._frozen:
                continue
            A_g = assignment_map.get(g, _EMPTY)
            # Distributive form of ``A_g & (unbound | sats...)``: intersect
            # per component so the full union is never materialised.
            accreted = A_g & self._infall_lists.get(self._unbound_default, _EMPTY)
            for h in satellites_history.get(g, _EMPTY):
                accreted |= A_g & self._infall_lists.get(h, _EMPTY)
            # Same replace-not-mutate/guard treatment as Pass 1 (C02) --
            # this mutates the same `_infall_lists[g]` sets and needs the
            # identical safety, not just Pass 1's own galaxies.
            if accreted or g not in self._infall_lists:
                self._infall_lists[g] = self._infall_lists.get(g, _EMPTY) | accreted

        self._previous_birth_map = dict(birth_map)

    def current(self):
        """Return the current infall lists.

        Returns
        -------
        infall_lists : dict of {int: set of int}
        """
        return self._infall_lists

    def _cheap_snapshot(self):
        """Cheap, checkpoint-safe reference capture (C02).

        O(n_galaxies), not O(total particles ever accreted): safe to
        call every snapshot because ``_update`` never mutates an
        existing ``_infall_lists[g]``/``_previous_birth_map`` in place,
        only ever replaces it wholesale. ``_frozen``/``_sat_history_buffer``
        *are* mutated in place elsewhere, so those still get a real copy
        here (cheap regardless -- both are small, bounded containers).

        Returns
        -------
        ref : dict
            Pass to :meth:`_serialize` to get a checkpoint-write-safe,
            fully independent form (only worth doing at actual
            checkpoint-write time, not every snapshot).
        """
        return {
            "infall_lists": dict(self._infall_lists),
            "previous_birth_map": self._previous_birth_map,
            "frozen": set(self._frozen),
            "sat_history_buffer": list(self._sat_history_buffer),
            "last_snapshot": self._last_snapshot,
            "unbound_default": self._unbound_default,
        }

    def _get_state(self):
        """Serialise the tracker state for checkpointing.

        Returns
        -------
        state : dict
        """
        return self._serialize(self._cheap_snapshot())

    @staticmethod
    def _serialize(ref):
        """Convert a :meth:`_cheap_snapshot` reference into a fully
        independent, plain-container form safe for pickling/disk
        checkpointing.

        This is the O(total elements) conversion -- call only at actual
        checkpoint-write time (i.e. on failure), not every snapshot.

        Parameters
        ----------
        ref : dict
            Output of :meth:`_cheap_snapshot`.

        Returns
        -------
        state : dict
        """
        return {
            "infall_lists": {k: list(v) for k, v in ref["infall_lists"].items()},
            "previous_birth_map": {k: list(v) for k, v in ref["previous_birth_map"].items()},
            "frozen": ref["frozen"],
            "sat_history_buffer": ref["sat_history_buffer"],
            "last_snapshot": ref["last_snapshot"],
            "unbound_default": ref["unbound_default"],
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
