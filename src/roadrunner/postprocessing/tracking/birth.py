"""Birth-tagging tracker for newly appearing star particles.

A particle is born in a galaxy it is bound to when it first appears. Its
birth host is decided from evidence accumulated over ``factor ×
timescale`` after that first appearance: each snapshot adds
``weight × window((t − t0) / τ)`` to every host the particle is assigned
to (the weight is its responsibility), and when the window closes the
host with the most evidence becomes its birth host.

With ``enforce_initial_hosts`` only the hosts the particle was bound to
at its first appearance can collect evidence: a particle that was not
bound to a galaxy when it was born very probably does not belong there.
A particle that first appears with a single possible host (or unbound)
then has nothing to compete over, and is decided at once.

The state is kept as NumPy arrays: active particles sorted by ID, one
evidence row per (active particle, host), and the finalised records.
Arrays are replaced, never edited in place, so a ``_cheap_snapshot()``
reference stays valid across later updates (C02).
"""

from typing import NamedTuple

import numpy as np
import pandas as pd

from roadrunner._defaults import (
    BIRTH_GAUSSIAN_WIDTH, GALAXY_ID, SIM_ID, SNAP_ID, math_dtype,
)


def _exp_window(x):
    """Exponential decay window function.

    Parameters
    ----------
    x : ndarray

    Returns
    -------
    w : ndarray
        ``exp(-x)``
    """
    return np.exp(-x)


def _cauchy_window(x):
    """Cauchy (Lorentzian) window function.

    Parameters
    ----------
    x : ndarray

    Returns
    -------
    w : ndarray
        ``1 / (1 + x²)``
    """
    return 1.0 / (1 + x ** 2)


def _gaussian_window(x):
    """Gaussian window function.

    Parameters
    ----------
    x : ndarray

    Returns
    -------
    w : ndarray
        ``exp(-width · x²)``
    """
    return np.exp(-BIRTH_GAUSSIAN_WIDTH * x ** 2)


_WINDOWS = {
    "gaussian": _gaussian_window,
    "cauchy": _cauchy_window,
    "exp": _exp_window,
}


class ActiveParticleInfo:
    """Legacy per-particle accumulation record.

    Checkpoints written before the array-based tracker pickle these
    objects; the class stays so they still load, and ``_set_state``
    converts them to arrays.
    """

    __slots__ = ("t0", "snap0", "tau", "counts", "initial_hosts")

    def __init__(self, t0, snap0, tau, counts, initial_hosts):
        self.t0 = t0
        self.snap0 = snap0
        self.tau = tau
        self.counts = counts
        self.initial_hosts = initial_hosts


class FinalizedInfo(NamedTuple):
    """Legacy per-particle finalised record (see :class:`ActiveParticleInfo`)."""

    birth_id: GALAXY_ID
    birth_time: np.float32
    birth_snap: SNAP_ID
    timescale: np.float32


def _member(sorted_ids, ids):
    """Boolean mask of which ``ids`` occur in the sorted array ``sorted_ids``."""
    if sorted_ids.size == 0:
        return np.zeros(ids.shape, dtype=bool)
    pos = np.searchsorted(sorted_ids, ids)
    pos[pos == sorted_ids.size] = 0
    return sorted_ids[pos] == ids


def _group_starts(sorted_keys):
    """Index of the first element of each run of equal values."""
    if sorted_keys.size == 0:
        return np.empty(0, dtype=np.intp)
    return np.flatnonzero(np.r_[True, sorted_keys[1:] != sorted_keys[:-1]])


def _empty_evidence():
    return {"pid": np.empty(0, SIM_ID), "host": np.empty(0, GALAXY_ID),
            "score": np.empty(0, np.float64), "initial": np.empty(0, bool),
            "rank": np.empty(0, np.int64)}


class BirthTracker:
    """Tracks the birth host of each star particle across snapshots.

    Parameters
    ----------
    factor : float, default=5
        The accumulation window lasts ``factor × timescale`` after the
        particle's first appearance.
    enforce_initial_hosts : bool, default=False
        If ``True``, only the hosts the particle was bound to at its first
        appearance collect evidence, and a particle with a single such
        host (or unbound) is decided immediately.
    window : str, default='gaussian'
        Window function: ``"gaussian"``, ``"cauchy"``, or ``"exp"``.
    """

    def __init__(self, factor=5, enforce_initial_hosts=False, window="gaussian"):
        if window not in _WINDOWS:
            raise ValueError(f"Unknown window: {window}. Choose from {list(_WINDOWS)}")
        self._window_fn = _WINDOWS[window]
        self.factor = factor - 0.001
        self._last_snapshot = -1
        self.enforce_initial_hosts = enforce_initial_hosts
        md = math_dtype()
        # Active particles, sorted by ID.
        self._act = {"pid": np.empty(0, SIM_ID), "t0": np.empty(0, md),
                     "snap0": np.empty(0, SNAP_ID), "tau": np.empty(0, md),
                     "deadline": np.empty(0, md)}
        # Evidence: one row per (active particle, host), sorted by (pid, host).
        # ``initial`` marks hosts seen at the first appearance; ``rank`` is the
        # insertion order, which breaks score ties (earliest host wins).
        self._ev = _empty_evidence()
        self._next_rank = 0
        # Finalised particles: record chunks in finalisation order, all IDs
        # sorted (for membership), and galaxy -> set of finalised IDs.
        self._fin_chunks = []
        self._fin_sorted = np.empty(0, SIM_ID)
        self._birth_map = {}

    # ── update ─────────────────────────────────────────────────────────
    def update(self, t_snap, snapshot_id, particle_ids, host_ids, timescales, weights=None):
        """Add one snapshot's evidence, then finalise expired windows.

        Parameters
        ----------
        t_snap : float
        snapshot_id : int
        particle_ids, host_ids : ndarray
            One entry per (particle, host) pair; a particle may appear with
            several hosts.
        timescales : ndarray
            Per pair (the particle's timescale; the maximum over its pairs
            is used when it first appears).
        weights : ndarray or None, optional
            Per pair evidence weight (e.g. the responsibility). Defaults to 1.
        """
        if snapshot_id <= self._last_snapshot:
            return
        md = math_dtype()
        pids = np.asarray(particle_ids).astype(SIM_ID, copy=False)
        hosts = np.asarray(host_ids).astype(GALAXY_ID, copy=False)
        taus = np.asarray(timescales, dtype=md)
        ws = (np.full(pids.shape, 1.0, dtype=md) if weights is None
              else np.asarray(weights, dtype=md))

        keep = ~_member(self._fin_sorted, pids)              # finalised particles are settled
        pids, hosts, taus, ws = pids[keep], hosts[keep], taus[keep], ws[keep]
        pids, hosts, taus, ws = self._merge_pairs(pids, hosts, taus, ws)

        active = _member(self._act["pid"], pids)
        self._accumulate(t_snap, pids[active], hosts[active], ws[active])
        new = ~active
        self._register(t_snap, snapshot_id, pids[new], hosts[new], taus[new], ws[new])
        self._finalize_particles(t_snap)
        self._last_snapshot = snapshot_id

    @staticmethod
    def _merge_pairs(pids, hosts, taus, ws):
        """Sort pairs by (particle, host) and merge duplicates (weights summed)."""
        order = np.lexsort((hosts, pids))
        pids, hosts, taus, ws = pids[order], hosts[order], taus[order], ws[order]
        if pids.size < 2:
            return pids, hosts, taus, ws
        dup = (pids[1:] == pids[:-1]) & (hosts[1:] == hosts[:-1])
        if not dup.any():
            return pids, hosts, taus, ws
        start = np.flatnonzero(np.r_[True, ~dup])
        return (pids[start], hosts[start], np.maximum.reduceat(taus, start),
                np.add.reduceat(ws, start).astype(ws.dtype, copy=False))

    def _evidence_rows(self, pids, hosts):
        """Row of each (particle, host) pair in the evidence table, or -1."""
        ev_pid, ev_host = self._ev["pid"], self._ev["host"]
        lo = np.searchsorted(ev_pid, pids, "left")
        width = np.searchsorted(ev_pid, pids, "right") - lo
        rows = np.full(pids.size, -1, dtype=np.int64)
        for k in range(int(width.max()) if width.size else 0):   # hosts per particle: a handful
            cand = lo + k
            ok = k < width
            match = ok & (ev_host[np.where(ok, cand, 0)] == hosts)
            rows[match] = cand[match]
        return rows

    def _accumulate(self, t_snap, pids, hosts, ws):
        """Add ``w · window((t − t0) / τ)`` to active particles' hosts."""
        if pids.size == 0:
            return
        md = math_dtype()
        a = np.searchsorted(self._act["pid"], pids)
        t0 = self._act["t0"][a].astype(md, copy=False)
        tau0 = np.maximum(self._act["tau"][a].astype(md, copy=False), md(1e-10))
        deltas = (ws * self._window_fn((t_snap - t0) / tau0)).astype(md, copy=False)

        rows = self._evidence_rows(pids, hosts)
        hit = rows >= 0
        ev = dict(self._ev)
        if hit.any():
            r, d = rows[hit], deltas[hit]
            score = ev["score"].copy()
            # Initial hosts accumulate in the math precision, hosts added later
            # in float64: the precision each score always had.
            score[r] = np.where(ev["initial"][r],
                                (score[r].astype(md) + d).astype(np.float64),
                                score[r] + d.astype(np.float64))
            ev["score"] = score
        miss = ~hit
        if miss.any() and not self.enforce_initial_hosts:
            n = int(miss.sum())
            new = {"pid": pids[miss], "host": hosts[miss],
                   "score": deltas[miss].astype(np.float64), "initial": np.zeros(n, bool),
                   "rank": self._next_rank + np.arange(n, dtype=np.int64)}
            self._next_rank += n
            ev = self._merge_evidence(ev, new)
        self._ev = ev

    @staticmethod
    def _merge_evidence(ev, new):
        """Evidence table with ``new`` rows added, sorted by (pid, host)."""
        merged = {k: np.concatenate([ev[k], new[k]]) for k in ev}
        order = np.lexsort((merged["host"], merged["pid"]))
        return {k: v[order] for k, v in merged.items()}

    def _register(self, t_snap, snapshot_id, pids, hosts, taus, ws):
        """First appearance: open a window, or decide at once if there is no competition."""
        if pids.size == 0:
            return
        md = math_dtype()
        start = _group_starts(pids)
        pid_u = pids[start]
        n_hosts = np.diff(np.r_[start, pids.size])
        tau = np.maximum.reduceat(taus, start)
        t0 = np.full(pid_u.size, t_snap, dtype=md)
        snap0 = np.full(pid_u.size, snapshot_id, dtype=SNAP_ID)

        decided = (n_hosts == 1) if self.enforce_initial_hosts else np.zeros(pid_u.size, bool)
        if decided.any():
            self._add_finalized(pid_u[decided], hosts[start[decided]],
                                t0[decided], snap0[decided], tau[decided])
        open_ = ~decided
        if not open_.any():
            return
        deadline = (t_snap + self.factor * tau[open_].astype(np.float64)).astype(md)
        act = {"pid": pid_u[open_], "t0": t0[open_], "snap0": snap0[open_],
               "tau": tau[open_], "deadline": deadline}
        merged = {k: np.concatenate([self._act[k], act[k]]) for k in act}
        order = np.argsort(merged["pid"], kind="stable")
        self._act = {k: v[order] for k, v in merged.items()}

        pair = np.repeat(open_, n_hosts)
        n = int(pair.sum())
        new = {"pid": pids[pair], "host": hosts[pair], "score": ws[pair].astype(np.float64),
               "initial": np.ones(n, bool),
               "rank": self._next_rank + np.arange(n, dtype=np.int64)}
        self._next_rank += n
        self._ev = self._merge_evidence(self._ev, new)

    def _leaders(self, pids):
        """Birth host of each (sorted, active) particle: most evidence, earliest host on ties."""
        m = _member(pids, self._ev["pid"])
        pid, host = self._ev["pid"][m], self._ev["host"][m]
        order = np.lexsort((self._ev["rank"][m], -self._ev["score"][m], pid))
        pid, host = pid[order], host[order]
        return host[_group_starts(pid)]

    def _finalize_particles(self, t_snap):
        """Finalise particles whose accumulation window has expired."""
        if self._act["pid"].size == 0:
            return
        due = self._act["deadline"].astype(np.float64) <= t_snap
        if not due.any():
            return
        pid_due = self._act["pid"][due]
        self._add_finalized(pid_due, self._leaders(pid_due), self._act["t0"][due],
                            self._act["snap0"][due], self._act["tau"][due])
        self._act = {k: v[~due] for k, v in self._act.items()}
        gone = _member(pid_due, self._ev["pid"])
        self._ev = {k: v[~gone] for k, v in self._ev.items()}

    def _add_finalized(self, pids, births, t0, snap0, tau):
        """Record finalised particles and add them to their galaxies' birth sets."""
        births = births.astype(GALAXY_ID, copy=False)
        self._fin_chunks.append({"pid": pids, "birth_id": births, "t0": t0,
                                 "snap0": snap0, "tau": tau})
        self._fin_sorted = np.sort(np.concatenate([self._fin_sorted, pids]), kind="stable")
        self._add_to_birth_sets(self._birth_map, pids, births)

    @staticmethod
    def _add_to_birth_sets(birth_map, pids, births):
        """``birth_map[g] |= pids born in g``, replacing (never editing) each set."""
        order = np.argsort(births, kind="stable")
        b_sorted, p_sorted = births[order], pids[order]
        starts = _group_starts(b_sorted)
        for b, grp in zip(b_sorted[starts].tolist(), np.split(p_sorted, starts[1:])):
            add = set(grp.tolist())
            old = birth_map.get(b)
            birth_map[b] = add if old is None else old | add

    # ── outputs ────────────────────────────────────────────────────────
    @staticmethod
    def _concat_finalized(chunks):
        """Finalised records as one dict of arrays."""
        md = math_dtype()
        if not chunks:
            return {"pid": np.empty(0, SIM_ID), "birth_id": np.empty(0, GALAXY_ID),
                    "t0": np.empty(0, md), "snap0": np.empty(0, SNAP_ID), "tau": np.empty(0, md)}
        return {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}

    def finalize(self):
        """Force-finalise all active particles and return the birth table.

        Returns
        -------
        birth_df : DataFrame
            Columns: ``particle_index``, ``birth_id``.
        """
        self._finalize_particles(np.inf)
        fin = self._concat_finalized(self._fin_chunks)
        return pd.DataFrame({"particle_index": fin["pid"].astype(SIM_ID, copy=False),
                             "birth_id": fin["birth_id"].astype(GALAXY_ID, copy=False)})

    def current_birth_map(self):
        """Return the current (incomplete) birth map.

        Finalised particles plus each active particle under its current
        leader. The finalised sets are shared, not copied: the tracker
        replaces them rather than editing them, and consumers only read.

        Returns
        -------
        birth_map : dict of {int: set of int}
        """
        birth_map = dict(self._birth_map)
        if self._act["pid"].size:
            self._add_to_birth_sets(birth_map, self._act["pid"], self._leaders(self._act["pid"]))
        return birth_map

    # ── checkpointing ──────────────────────────────────────────────────
    def _cheap_snapshot(self):
        """Cheap, checkpoint-safe reference capture (C02).

        Every array and birth set is replaced rather than edited by later
        updates, so references are enough; nothing is copied element-wise.

        Returns
        -------
        ref : dict
            Pass to :meth:`_serialize` for a plain, fully independent form.
        """
        return {
            "act": dict(self._act),
            "ev": dict(self._ev),
            "next_rank": self._next_rank,
            "fin_chunks": list(self._fin_chunks),
            "factor": self.factor,
            "last_snapshot": self._last_snapshot,
            "enforce_initial_hosts": self.enforce_initial_hosts,
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
        """Convert a :meth:`_cheap_snapshot` reference into its checkpoint form.

        Parameters
        ----------
        ref : dict
            Output of :meth:`_cheap_snapshot`.

        Returns
        -------
        state : dict
        """
        return {
            "format": 2,
            "active": dict(ref["act"]),
            "evidence": dict(ref["ev"]),
            "next_rank": ref["next_rank"],
            "finalized": BirthTracker._concat_finalized(ref["fin_chunks"]),
            "factor": ref["factor"],
            "last_snapshot": ref["last_snapshot"],
            "enforce_initial_hosts": ref["enforce_initial_hosts"],
        }

    def _set_state(self, state):
        """Restore tracker state from a checkpoint (array or legacy format).

        Parameters
        ----------
        state : dict
        """
        if state.get("format") == 2:
            self._act = dict(state["active"])
            self._ev = dict(state["evidence"])
            self._next_rank = state["next_rank"]
            fin = state["finalized"]
        else:
            fin = self._load_legacy(state)
        self._fin_chunks = [fin] if fin["pid"].size else []
        self._fin_sorted = np.sort(fin["pid"], kind="stable")
        self._birth_map = {}
        self._add_to_birth_sets(self._birth_map, fin["pid"], fin["birth_id"])
        self.factor = state["factor"]
        self._last_snapshot = state["last_snapshot"]
        self.enforce_initial_hosts = state["enforce_initial_hosts"]

    def _load_legacy(self, state):
        """Convert a dict-based checkpoint (active/finalized/heap) to arrays."""
        md = math_dtype()
        deadlines = {int(p): d for d, p in state.get("heap", [])}
        act = {k: [] for k in ("pid", "t0", "snap0", "tau", "deadline")}
        ev = {k: [] for k in ("pid", "host", "score", "initial", "rank")}
        rank = 0
        for p in sorted(state["active"], key=int):
            info = state["active"][p]
            if isinstance(info, dict):
                info = ActiveParticleInfo(info["t0"], info["snap0"], info["tau"],
                                          info["counts"], info["initial_hosts"])
            act["pid"].append(int(p)); act["t0"].append(info.t0)
            act["snap0"].append(info.snap0); act["tau"].append(info.tau)
            act["deadline"].append(deadlines.get(int(p), info.t0 + self.factor * info.tau))
            for host, score in info.counts.items():            # insertion order = rank order
                ev["pid"].append(int(p)); ev["host"].append(int(host))
                ev["score"].append(float(score)); ev["initial"].append(host in info.initial_hosts)
                ev["rank"].append(rank)
                rank += 1
        self._act = {"pid": np.asarray(act["pid"], SIM_ID), "t0": np.asarray(act["t0"], md),
                     "snap0": np.asarray(act["snap0"], SNAP_ID), "tau": np.asarray(act["tau"], md),
                     "deadline": np.asarray(act["deadline"], md)}
        ev = {"pid": np.asarray(ev["pid"], SIM_ID), "host": np.asarray(ev["host"], GALAXY_ID),
              "score": np.asarray(ev["score"], np.float64), "initial": np.asarray(ev["initial"], bool),
              "rank": np.asarray(ev["rank"], np.int64)}
        order = np.lexsort((ev["host"], ev["pid"]))
        self._ev = {k: v[order] for k, v in ev.items()}
        self._next_rank = rank

        recs = []
        for p, rec in state["finalized"].items():
            if isinstance(rec, dict):
                rec = FinalizedInfo(rec["birth_id"], rec["birth_time"], rec["birth_snap"], rec["timescale"])
            recs.append((int(p), rec.birth_id, rec.birth_time, rec.birth_snap, rec.timescale))
        cols = list(zip(*recs)) if recs else [[]] * 5
        return {"pid": np.asarray(cols[0], SIM_ID), "birth_id": np.asarray(cols[1], GALAXY_ID),
                "t0": np.asarray(cols[2], md), "snap0": np.asarray(cols[3], SNAP_ID),
                "tau": np.asarray(cols[4], md)}
