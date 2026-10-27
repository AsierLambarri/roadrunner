"""Birth-tagging tracker for newly appearing star particles.

A particle is born in a galaxy it is bound to when it first appears. The
soft membership of that first output is its prior: every candidate host
starts with the log of its share of the particle's responsibility. The
outputs that follow, within a short window, are further noisy looks at the
same membership, and each one multiplies in a leaky persistence
likelihood,

    ``ln [lam + (1 - lam) q_k]``,   ``lam = exp(-dt / tau_c)``,

``q_k`` the particle's membership of candidate ``k`` at that output
(normalised over the candidates) and ``dt`` the time since the previous
output used. ``tau_c`` is the decorrelation time of the assignment noise:
outputs closer than that share their noise and count little, outputs
further apart count fully. When the window closes the candidate with the
most evidence is the birth host, and its softmax probability is kept.

The window ``L`` is the responsibility-weighted arithmetic mean of the
candidates' own dynamical times (the ``timescales`` of the pairs), and
``tau_c = L / budget``. A look adds at most ``dt / tau_c`` nats, so the
window adds at most ``budget`` nats in all: a particle whose first-output
share is at least ``1 / (1 + exp(-budget))`` can never be overturned and is
decided at once, as is one with a single candidate. Evidence is added only
while every candidate is alive; after one has merged away the later
outputs cannot tell the candidates apart, and the window closes.

Only the hosts with a first-output share above ``min_initial_resp`` (and
always the largest) compete: a particle that was not bound to a galaxy
when it was born very probably does not belong there. A negative value
lets any host collect evidence: hosts first seen inside the window join
with a small prior, and nothing is decided at once.

The state is kept as NumPy arrays: open particles sorted by ID, one
evidence row per (open particle, host), and the finalised records. Arrays
are replaced, never edited in place, so a ``_cheap_snapshot()`` reference
stays valid across later updates (C02).
"""

import numpy as np
import pandas as pd

from roadrunner._defaults import BIRTH_LATE_PRIOR, GALAXY_ID, SIM_ID
from roadrunner.physics.constants import MAX_DYN_TIMESCALE

_TIME_TOLERANCE = 1e-9      # Gyr: an output exactly at the end of the window still counts
_FORMAT = 3                 # checkpoint format of the state


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


def _first_maximum(values, starts, sizes):
    """Index, within the whole array, of the first maximum of each run.

    Parameters
    ----------
    values : ndarray
    starts : ndarray of int
        First element of each run.
    sizes : ndarray of int
        Length of each run.

    Returns
    -------
    first : ndarray of int
        The runs are sorted by host, so the first maximum is the lowest host ID.
    """
    top = np.repeat(np.maximum.reduceat(values, starts), sizes)
    return np.minimum.reduceat(np.where(values == top, np.arange(values.size), values.size), starts)


def _empty_open():
    return {"pid": np.empty(0, SIM_ID), "t0": np.empty(0, np.float64),
            "t_prev": np.empty(0, np.float64), "lag": np.empty(0, np.float64)}


def _empty_evidence():
    return {"pid": np.empty(0, SIM_ID), "host": np.empty(0, GALAXY_ID), "logp": np.empty(0, np.float64)}


def _sorted_by(table, keys):
    """The table with its rows ordered by ``np.lexsort(keys)``."""
    order = np.lexsort(keys)
    return {k: v[order] for k, v in table.items()}


def _leaders(evidence):
    """Birth host and probability of each particle of an evidence table sorted by (particle, host).

    Parameters
    ----------
    evidence : dict of ndarray
        Evidence rows (``pid``, ``host``, ``logp``) of whole particles.

    Returns
    -------
    host : ndarray of GALAXY_ID
        The host with the most evidence, the lowest ID on ties.
    prob : ndarray of float64
        Its softmax probability.
    """
    starts = _group_starts(evidence["pid"])
    sizes = np.diff(np.r_[starts, evidence["pid"].size])
    lp = evidence["logp"]
    w = np.exp(lp - np.repeat(np.maximum.reduceat(lp, starts), sizes))
    prob = w / np.repeat(np.add.reduceat(w, starts), sizes)
    first = _first_maximum(w, starts, sizes)
    return evidence["host"][first], prob[first]


class BirthTracker:
    """Tracks the birth host of each star particle across snapshots.

    Parameters
    ----------
    budget : float, default=5
        ``N``: the window ``L`` lasts ``N`` correlation times, so the later
        outputs add at most ``N`` nats; a first-output share of at least
        ``1 / (1 + exp(-N))`` is final.
    min_initial_resp : float, default=0.01
        Only the hosts whose first-output share is above this compete
        (``0``: every host the particle is bound to). A negative value lets
        any host compete and decides nothing at once.
    """

    def __init__(self, budget=5.0, min_initial_resp=0.01):
        self.budget = float(budget)
        self.min_initial_resp = float(min_initial_resp)
        self._decide_share = 1.0 / (1.0 + np.exp(-self.budget))
        self._last_snapshot = -1
        # Open particles, sorted by ID: ``t0`` first appearance, ``t_prev`` the last output used, ``lag`` the window L.
        self._act = _empty_open()
        # Evidence: one row per (open particle, host), sorted by (pid, host): the log posterior up to a constant.
        self._ev = _empty_evidence()
        # Finalised particles: record chunks in finalisation order, all IDs
        # sorted (for membership), and galaxy -> set of finalised IDs.
        self._fin_chunks = []
        self._fin_sorted = np.empty(0, SIM_ID)
        self._birth_map = {}

    # ── update ─────────────────────────────────────────────────────────
    def update(self, t_snap, snapshot_id, particle_ids, host_ids, timescales, weights=None, alive=None):
        """Add one snapshot's evidence, then finalise closed windows.

        Parameters
        ----------
        t_snap : float
            Cosmic time of the snapshot, in the unit of ``timescales``.
        snapshot_id : int
        particle_ids, host_ids : ndarray
            One entry per (particle, host) pair; a particle may appear with
            several hosts.
        timescales : ndarray
            Per pair, the dynamical time of the particle in that host
            (``NaN`` where there is none); read when a particle first appears.
        weights : ndarray or None, optional
            Per pair evidence weight (the responsibility). Defaults to 1.
        alive : ndarray or None, optional
            The galaxies that exist at this output. Defaults to the hosts of
            the pairs, which takes a galaxy without a pair for dead.
        """
        if snapshot_id <= self._last_snapshot:
            return
        pids = np.asarray(particle_ids).astype(SIM_ID, copy=False)
        hosts = np.asarray(host_ids).astype(GALAXY_ID, copy=False)
        ts = np.asarray(timescales, dtype=np.float64)
        ws = np.ones(pids.shape) if weights is None else np.asarray(weights, dtype=np.float64)
        alive = np.unique(hosts) if alive is None else np.asarray(alive).astype(GALAXY_ID, copy=False)

        keep = ~_member(self._fin_sorted, pids)              # finalised particles are settled
        pids, hosts, ts, ws = self._merge_pairs(pids[keep], hosts[keep], ts[keep], ws[keep])

        looked = _member(self._act["pid"], pids)
        self._advance(t_snap, pids[looked], hosts[looked], ws[looked], alive)
        self._register(t_snap, pids[~looked], hosts[~looked], ts[~looked], ws[~looked])
        self._last_snapshot = snapshot_id

    @staticmethod
    def _merge_pairs(pids, hosts, ts, ws):
        """Sort pairs by (particle, host) and merge duplicates (weights summed)."""
        order = np.lexsort((hosts, pids))
        pids, hosts, ts, ws = pids[order], hosts[order], ts[order], ws[order]
        if pids.size < 2:
            return pids, hosts, ts, ws
        dup = (pids[1:] == pids[:-1]) & (hosts[1:] == hosts[:-1])
        if not dup.any():
            return pids, hosts, ts, ws
        start = np.flatnonzero(np.r_[True, ~dup])
        return pids[start], hosts[start], np.fmax.reduceat(ts, start), np.add.reduceat(ws, start)

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

    def _advance(self, t_snap, pids, hosts, ws, alive):
        """Close the windows that ran out or lost a host, and fuse this output into the others.

        Parameters
        ----------
        t_snap : float
        pids, hosts, ws : ndarray
            The pairs of the open particles, sorted by (particle, host).
        alive : ndarray
            Galaxies that exist at this output.
        """
        act = self._act
        n = act["pid"].size
        if n == 0:
            return
        starts = _group_starts(self._ev["pid"])
        lost = np.logical_or.reduceat(~np.isin(self._ev["host"], alive), starts)   # a host merged away
        closing = (t_snap - act["t0"] > act["lag"] + _TIME_TOLERANCE) | lost
        if self.min_initial_resp < 0:
            self._add_late_hosts(t_snap, pids, hosts, closing)
            starts = _group_starts(self._ev["pid"])
        ev = self._ev
        sizes = np.diff(np.r_[starts, ev["pid"].size])
        star = np.repeat(np.arange(n), sizes)                                      # open particle of each row

        rows = self._evidence_rows(pids, hosts)
        take = (rows >= 0) & ~closing[np.searchsorted(act["pid"], pids)]
        q = np.zeros(ev["pid"].size)
        q[rows[take]] = ws[take]
        total = np.add.reduceat(q, starts)
        looked = (total > 0) & ~closing                                           # an output with a candidate in it
        lam = np.exp(-np.where(looked, t_snap - act["t_prev"], 0.0) * self.budget / act["lag"])    # tau_c = L / budget
        share = q / np.where(total > 0, total, 1.0)[star]
        logp = np.where(looked[star], ev["logp"] + np.log(lam[star] + (1.0 - lam[star]) * share), ev["logp"])
        self._ev = {"pid": ev["pid"], "host": ev["host"], "logp": logp}
        self._act = {"pid": act["pid"], "t0": act["t0"], "lag": act["lag"],
                     "t_prev": np.where(looked, t_snap, act["t_prev"])}
        if closing.any():
            self._close(closing)

    def _add_late_hosts(self, t_snap, pids, hosts, closing):
        """Rows for the hosts an open particle is first seen in, with the late prior."""
        miss = (self._evidence_rows(pids, hosts) < 0) & ~closing[np.searchsorted(self._act["pid"], pids)]
        if not miss.any():
            return
        new = {"pid": pids[miss], "host": hosts[miss], "logp": np.full(int(miss.sum()), np.log(BIRTH_LATE_PRIOR))}
        merged = {k: np.concatenate([self._ev[k], new[k]]) for k in new}
        self._ev = _sorted_by(merged, (merged["host"], merged["pid"]))

    def _register(self, t_snap, pids, hosts, ts, ws):
        """First appearance: decide at once, or open a window over the candidate hosts.

        Parameters
        ----------
        t_snap : float
        pids, hosts, ts, ws : ndarray
            The pairs of the new particles, sorted by (particle, host).
        """
        if pids.size == 0:
            return
        starts = _group_starts(pids)
        sizes = np.diff(np.r_[starts, pids.size])
        share = ws / np.repeat(np.add.reduceat(ws, starts), sizes)
        first = _first_maximum(share, starts, sizes)
        top = share[first]
        late = self.min_initial_resp < 0
        decided = np.zeros(starts.size, bool) if late else (sizes == 1) | (top >= self._decide_share)
        if decided.any():
            self._add_finalized(pids[starts[decided]], hosts[first[decided]], top[decided])
        if decided.all():
            return
        # Candidates: the hosts above the threshold and always the largest. The window is the mean of their dynamical times.
        cand = (share > self.min_initial_resp) | (np.arange(pids.size) == np.repeat(first, sizes))
        valid = cand & np.isfinite(ts) & (ts > 0)
        weight = np.where(valid, share, 0.0)
        mass = np.add.reduceat(weight, starts)
        lag = np.where(mass > 0, np.add.reduceat(weight * np.where(valid, ts, 0.0), starts) / np.where(mass > 0, mass, 1.0),
                       MAX_DYN_TIMESCALE)
        opened = ~decided
        act = {"pid": pids[starts[opened]], "t0": np.full(int(opened.sum()), t_snap),
               "t_prev": np.full(int(opened.sum()), t_snap), "lag": lag[opened]}
        row = np.repeat(opened, sizes) & cand
        ev = {"pid": pids[row], "host": hosts[row], "logp": np.log(share[row])}
        merged = {k: np.concatenate([self._act[k], act[k]]) for k in act}
        self._act = _sorted_by(merged, (merged["pid"],))
        merged = {k: np.concatenate([self._ev[k], ev[k]]) for k in ev}
        self._ev = _sorted_by(merged, (merged["host"], merged["pid"]))

    def _close(self, closing):
        """Finalise the open particles flagged in ``closing`` with their current evidence."""
        act, ev = self._act, self._ev
        row = np.repeat(closing, np.diff(np.r_[_group_starts(ev["pid"]), ev["pid"].size]))
        host, prob = _leaders({k: v[row] for k, v in ev.items()})
        self._add_finalized(act["pid"][closing], host, prob)
        self._act = {k: v[~closing] for k, v in act.items()}
        self._ev = {k: v[~row] for k, v in ev.items()}

    def _add_finalized(self, pids, births, prob):
        """Record finalised particles and add them to their galaxies' birth sets."""
        births = births.astype(GALAXY_ID, copy=False)
        self._fin_chunks.append({"pid": pids, "birth_id": births, "p_birth": prob.astype(np.float32)})
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
        if not chunks:
            return {"pid": np.empty(0, SIM_ID), "birth_id": np.empty(0, GALAXY_ID), "p_birth": np.empty(0, np.float32)}
        return {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}

    def finalize(self):
        """Close every open particle with its current evidence and return the birth table.

        Returns
        -------
        birth_df : DataFrame
            Columns: ``particle_index``, ``birth_id``, ``p_birth`` (the
            probability of the birth host: 1 for a particle decided at once).
        """
        if self._act["pid"].size:
            self._close(np.ones(self._act["pid"].size, dtype=bool))
        fin = self._concat_finalized(self._fin_chunks)
        return pd.DataFrame({"particle_index": fin["pid"].astype(SIM_ID, copy=False),
                             "birth_id": fin["birth_id"].astype(GALAXY_ID, copy=False),
                             "p_birth": fin["p_birth"]})

    def current_birth_map(self):
        """Return the current (incomplete) birth map.

        Finalised particles plus each open particle under its current
        leader. The finalised sets are shared, not copied: the tracker
        replaces them rather than editing them, and consumers only read.

        Returns
        -------
        birth_map : dict of {int: set of int}
        """
        birth_map = dict(self._birth_map)
        if self._act["pid"].size:
            self._add_to_birth_sets(birth_map, self._act["pid"], _leaders(self._ev)[0])
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
            "fin_chunks": list(self._fin_chunks),
            "budget": self.budget,
            "min_initial_resp": self.min_initial_resp,
            "last_snapshot": self._last_snapshot,
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
            "format": _FORMAT,
            "active": dict(ref["act"]),
            "evidence": dict(ref["ev"]),
            "finalized": BirthTracker._concat_finalized(ref["fin_chunks"]),
            "budget": ref["budget"],
            "min_initial_resp": ref["min_initial_resp"],
            "last_snapshot": ref["last_snapshot"],
        }

    def _set_state(self, state):
        """Restore tracker state from a checkpoint.

        Parameters
        ----------
        state : dict

        Raises
        ------
        ValueError
            If the checkpoint was written by an earlier tracker (a window vote), whose
            open windows have no equivalent here: restart the run.
        """
        if state.get("format") != _FORMAT:
            raise ValueError("The checkpoint holds the state of an earlier birth tracker (a window vote), "
                             "which this tracker cannot continue: delete it and start a fresh run.")
        self.budget = state["budget"]
        self.min_initial_resp = state["min_initial_resp"]
        self._decide_share = 1.0 / (1.0 + np.exp(-self.budget))
        self._act = dict(state["active"])
        self._ev = dict(state["evidence"])
        fin = state["finalized"]
        self._fin_chunks = [fin] if fin["pid"].size else []
        self._fin_sorted = np.sort(fin["pid"], kind="stable")
        self._birth_map = {}
        self._add_to_birth_sets(self._birth_map, fin["pid"], fin["birth_id"])
        self._last_snapshot = state["last_snapshot"]
