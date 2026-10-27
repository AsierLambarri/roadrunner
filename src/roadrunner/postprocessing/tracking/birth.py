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
candidates' own dynamical times (the ``dynamical_times`` of the pairs), and
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

Each update answers with the
:class:`~roadrunner.postprocessing.tracking.records.BirthChanges` it caused:
the particles first seen, and the ones whose leading host changed. That is
all a consumer needs to follow the birth hosts, and it costs what changed,
not what exists.

The state is kept as NumPy arrays: open particles sorted by ID, one
evidence row per (open particle, host), and the finalised records. Arrays
are replaced, never edited in place (and are read-only), so a
``capture()`` reference stays valid across later updates (C02).
"""

import numpy as np
import pandas as pd

from roadrunner._defaults import BIRTH_LATE_PRIOR, GALAXY_ID, SIM_ID
from roadrunner.physics.constants import MAX_DYN_TIMESCALE
from roadrunner.postprocessing.tracking.records import BirthChanges

# Gyr: an output exactly at the end of the window still counts.
_TIME_TOLERANCE = 1e-9
# Checkpoint format of the tracker state.
_FORMAT = 5

# What an update reports when no leader changed: no IDs, and no hosts left or entered.
_NO_MOVES = (
    np.empty(0, SIM_ID),
    np.empty(0, GALAXY_ID),
    np.empty(0, GALAXY_ID),
)


def _isin_sorted(sorted_ids, ids):
    """Find which values occur in an array that is already sorted.

    Parameters
    ----------
    sorted_ids : ndarray of shape (n_sorted,)
        Sorted values, without duplicates.
    ids : ndarray
        Values to look for.

    Returns
    -------
    found : ndarray of bool, with the shape of ``ids``
        Whether each value of ``ids`` occurs in ``sorted_ids``.
    """
    if sorted_ids.size == 0:
        return np.zeros(ids.shape, dtype=bool)
    position = np.searchsorted(sorted_ids, ids)
    position[position == sorted_ids.size] = 0
    return sorted_ids[position] == ids


def _run_starts(sorted_keys):
    """Find where each run of equal values of a sorted array begins.

    Parameters
    ----------
    sorted_keys : ndarray of shape (n,)
        Values sorted so that equal ones are contiguous.

    Returns
    -------
    starts : ndarray of int, shape (n_runs,)
        Index of the first element of each run.
    """
    if sorted_keys.size == 0:
        return np.empty(0, dtype=np.intp)
    begins_a_run = np.r_[True, sorted_keys[1:] != sorted_keys[:-1]]
    return np.flatnonzero(begins_a_run)


def _run_sizes(starts, n_elements):
    """Find the length of each run, given where the runs begin.

    Parameters
    ----------
    starts : ndarray of int, shape (n_runs,)
        Index of the first element of each run, as from :func:`_run_starts`.
    n_elements : int
        Number of elements the runs partition.

    Returns
    -------
    sizes : ndarray of int, shape (n_runs,)
        Number of elements of each run.
    """
    return np.diff(np.r_[starts, n_elements])


def _run_argmax(values, starts, sizes):
    """Find the first maximum of each run of an array.

    Parameters
    ----------
    values : ndarray of shape (n,)
        Values, in runs.
    starts : ndarray of int, shape (n_runs,)
        First element of each run.
    sizes : ndarray of int, shape (n_runs,)
        Length of each run.

    Returns
    -------
    first : ndarray of int, shape (n_runs,)
        Index, within ``values`` as a whole, of the first maximum of each run.
        The runs are sorted by host, so it is the one of the lowest host ID.
    """
    run_maximum = np.repeat(np.maximum.reduceat(values, starts), sizes)
    is_maximum = values == run_maximum
    index_of_maximum = np.where(is_maximum, np.arange(values.size), values.size)
    return np.minimum.reduceat(index_of_maximum, starts)


def _freeze(table):
    """Make the arrays of a state table read-only.

    The tables are replaced, never edited, so a reference to one stays valid
    however far later updates go (C02).

    Parameters
    ----------
    table : dict of ndarray
        Columns of the table.

    Returns
    -------
    table : dict of ndarray
        The same dict, its arrays no longer writeable.
    """
    for column in table.values():
        column.flags.writeable = False
    return table


def _empty_open_table():
    """Make the table of open particles with no rows.

    Returns
    -------
    open_table : dict of ndarray
        Columns ``particle_id``, ``first_time``, ``last_time``, ``window`` and
        ``leader``; see the Notes of :class:`BirthTracker`.
    """
    return _freeze({
        "particle_id": np.empty(0, SIM_ID),
        "first_time": np.empty(0, np.float64),
        "last_time": np.empty(0, np.float64),
        "window": np.empty(0, np.float64),
        "leader": np.empty(0, GALAXY_ID),
    })


def _empty_evidence_table():
    """Make the table of evidence with no rows.

    Returns
    -------
    evidence_table : dict of ndarray
        Columns ``particle_id``, ``host_id`` and ``log_evidence``; see the
        Notes of :class:`BirthTracker`.
    """
    return _freeze({
        "particle_id": np.empty(0, SIM_ID),
        "host_id": np.empty(0, GALAXY_ID),
        "log_evidence": np.empty(0, np.float64),
    })


def _sort_table(table, keys):
    """Order the rows of a table.

    Parameters
    ----------
    table : dict of ndarray
        Columns of equal length.
    keys : tuple of ndarray
        Sort keys as for :func:`numpy.lexsort`: the last one is the primary key.

    Returns
    -------
    sorted_table : dict of ndarray
        A new table with the rows reordered.
    """
    order = np.lexsort(keys)
    return {name: column[order] for name, column in table.items()}


def _merge_sorted(sorted_ids, new_ids):
    """Insert values into a sorted array, keeping it sorted.

    This copies the array and does not sort it again.

    Parameters
    ----------
    sorted_ids : ndarray of shape (n,)
        Sorted values.
    new_ids : ndarray
        Values to add, in any order.

    Returns
    -------
    merged : ndarray of shape (n + len(new_ids),)
        The values of both, sorted.
    """
    new_ids = np.sort(new_ids)
    return np.insert(sorted_ids, np.searchsorted(sorted_ids, new_ids), new_ids)


def _leaders(evidence):
    """Find the birth host of each particle of an evidence table, and how likely it is.

    Parameters
    ----------
    evidence : dict of ndarray
        Evidence rows (``particle_id``, ``host_id``, ``log_evidence``) of whole
        particles, sorted by (particle, host).

    Returns
    -------
    host_ids : ndarray of GALAXY_ID, shape (n_particles,)
        The host with the most evidence of each particle, the lowest ID on ties.
    probabilities : ndarray of float64, shape (n_particles,)
        The softmax probability of that host among the particle's candidates.
    """
    starts = _run_starts(evidence["particle_id"])
    sizes = _run_sizes(starts, evidence["particle_id"].size)
    log_evidence = evidence["log_evidence"]
    run_maximum = np.repeat(np.maximum.reduceat(log_evidence, starts), sizes)
    relative = np.exp(log_evidence - run_maximum)
    probability = relative / np.repeat(np.add.reduceat(relative, starts), sizes)
    leading = _run_argmax(relative, starts, sizes)
    return evidence["host_id"][leading], probability[leading]


class BirthTracker:
    """Track the birth host of each star particle across snapshots.

    Parameters
    ----------
    budget : float, default=5.0
        ``N``: the window ``L`` lasts ``N`` correlation times, so the later
        outputs add at most ``N`` nats; a first-output share of at least
        ``1 / (1 + exp(-N))`` is final.
    min_initial_resp : float, default=0.01
        Only the hosts whose first-output share is above this compete
        (``0``: every host the particle is bound to). A negative value lets
        any host compete and decides nothing at once.

    Attributes
    ----------
    budget : float
        The evidence budget ``N``.
    min_initial_resp : float
        The first-output share a host must exceed to compete.

    Notes
    -----
    The state lives in three tables of equal-length arrays, each replaced and
    never edited:

    * Open particles, sorted by ``particle_id``: ``first_time`` (when the
      particle first appeared), ``last_time`` (the last output used), ``window``
      (its length ``L``) and ``leader`` (the host with the most evidence now).
    * Evidence, one row per (open particle, candidate host), sorted by
      (``particle_id``, ``host_id``): ``log_evidence``, the log posterior up to
      a constant.
    * Finalised particles, as chunks of records in the order they were
      finalised, with all their IDs kept sorted to answer membership queries.
    """

    def __init__(self, budget=5.0, min_initial_resp=0.01):
        self.budget = float(budget)
        self.min_initial_resp = float(min_initial_resp)
        self._decide_share = 1.0 / (1.0 + np.exp(-self.budget))
        self._last_snapshot = -1
        self._open_table = _empty_open_table()
        self._evidence_table = _empty_evidence_table()
        self._finalized_chunks = []
        self._finalized_ids = _freeze({"ids": np.empty(0, SIM_ID)})["ids"]

    # ── update ─────────────────────────────────────────────────────────
    def update(self, memberships):
        """Add one snapshot's evidence, finalise closed windows, report what changed.

        Parameters
        ----------
        memberships : Memberships
            The soft memberships of the output. A particle's dynamical times
            are read when it first appears; the galaxies alive are the ones
            whose windows may stay open.

        Returns
        -------
        changes : BirthChanges
            The particles first seen and the ones whose leading host changed.
            Empty if the snapshot was already used.
        """
        if memberships.snapshot_id <= self._last_snapshot:
            return BirthChanges.nothing()
        particle_ids = np.asarray(memberships.particle_ids).astype(SIM_ID, copy=False)
        host_ids = np.asarray(memberships.host_ids).astype(GALAXY_ID, copy=False)
        dynamical_times = np.asarray(memberships.dynamical_times, dtype=np.float64)
        weights = np.asarray(memberships.weights, dtype=np.float64)
        alive = np.asarray(memberships.alive).astype(GALAXY_ID, copy=False)

        unsettled = ~self.settled(particle_ids)
        particle_ids, host_ids, dynamical_times, weights = self._merge_duplicate_pairs(
            particle_ids[unsettled],
            host_ids[unsettled],
            dynamical_times[unsettled],
            weights[unsettled],
        )

        is_open = _isin_sorted(self._open_table["particle_id"], particle_ids)
        is_new = ~is_open
        moved, n_closed = self._fuse_output(
            memberships.snapshot_time,
            particle_ids[is_open],
            host_ids[is_open],
            weights[is_open],
            alive,
        )
        born, n_at_once = self._register_new_particles(
            memberships.snapshot_time,
            particle_ids[is_new],
            host_ids[is_new],
            dynamical_times[is_new],
            weights[is_new],
        )
        self._last_snapshot = memberships.snapshot_id

        n_born = born[0].size
        stats = {
            "new": n_born,
            "at_once": n_at_once,
            "opened": n_born - n_at_once,
            "open": self._open_table["particle_id"].size,
            "moved": moved[0].size,
            "closed": n_closed,
        }
        return BirthChanges(*born, *moved, stats)

    def settled(self, particle_ids):
        """Find which particles are finalised.

        The birth of a finalised particle is settled: later outputs of it are
        ignored.

        Parameters
        ----------
        particle_ids : ndarray of shape (n,)
            Simulation IDs of the particles to look up.

        Returns
        -------
        settled : ndarray of bool, shape (n,)
            Whether each particle is finalised.
        """
        particle_ids = np.asarray(particle_ids).astype(SIM_ID, copy=False)
        return _isin_sorted(self._finalized_ids, particle_ids)

    @staticmethod
    def _merge_duplicate_pairs(particle_ids, host_ids, dynamical_times, weights):
        """Sort (particle, host) pairs and merge the repeated ones.

        The weights of a repeated pair are summed and its dynamical time is the
        largest of its entries.

        Parameters
        ----------
        particle_ids, host_ids : ndarray of shape (n_pairs,)
            The particle and the host of each pair.
        dynamical_times, weights : ndarray of float, shape (n_pairs,)
            The dynamical time and the evidence weight of each pair.

        Returns
        -------
        particle_ids, host_ids, dynamical_times, weights : ndarray
            The pairs sorted by (particle, host), each one once.
        """
        order = np.lexsort((host_ids, particle_ids))
        particle_ids = particle_ids[order]
        host_ids = host_ids[order]
        dynamical_times = dynamical_times[order]
        weights = weights[order]
        if particle_ids.size < 2:
            return particle_ids, host_ids, dynamical_times, weights
        same_particle = particle_ids[1:] == particle_ids[:-1]
        same_host = host_ids[1:] == host_ids[:-1]
        is_repeat = same_particle & same_host
        if not is_repeat.any():
            return particle_ids, host_ids, dynamical_times, weights
        firsts = np.flatnonzero(np.r_[True, ~is_repeat])
        return (
            particle_ids[firsts],
            host_ids[firsts],
            np.fmax.reduceat(dynamical_times, firsts),
            np.add.reduceat(weights, firsts),
        )

    def _evidence_row_of_pairs(self, particle_ids, host_ids):
        """Find the evidence row of each (particle, host) pair.

        Parameters
        ----------
        particle_ids, host_ids : ndarray of shape (n_pairs,)
            The particle and the host of each pair.

        Returns
        -------
        rows : ndarray of int, shape (n_pairs,)
            Row of the pair in the evidence table, ``-1`` if it has none.
        """
        evidence_particles = self._evidence_table["particle_id"]
        evidence_hosts = self._evidence_table["host_id"]
        first_row = np.searchsorted(evidence_particles, particle_ids, "left")
        n_rows = np.searchsorted(evidence_particles, particle_ids, "right") - first_row
        rows = np.full(particle_ids.size, -1, dtype=np.int64)
        n_hosts_max = int(n_rows.max()) if n_rows.size else 0
        for offset in range(n_hosts_max):  # hosts per particle: a handful
            candidate_row = first_row + offset
            in_range = offset < n_rows
            same_host = evidence_hosts[np.where(in_range, candidate_row, 0)] == host_ids
            found = in_range & same_host
            rows[found] = candidate_row[found]
        return rows

    def _fuse_output(self, snapshot_time, particle_ids, host_ids, weights, alive):
        """Close the windows that ran out or lost a host, fuse the output into the rest.

        Parameters
        ----------
        snapshot_time : float
            Cosmic time of the output.
        particle_ids, host_ids, weights : ndarray of shape (n_pairs,)
            The pairs of the particles that are open, sorted by (particle, host).
        alive : ndarray of GALAXY_ID
            The galaxies that exist at this output.

        Returns
        -------
        moved : tuple of ndarray
            The particles whose leader changed, with the host they left and the
            one they entered.
        n_closed : int
            Number of windows closed.
        """
        open_table = self._open_table
        n_open = open_table["particle_id"].size
        if n_open == 0:
            return _NO_MOVES, 0

        evidence_starts = _run_starts(self._evidence_table["particle_id"])
        has_lost_host = np.logical_or.reduceat(
            ~np.isin(self._evidence_table["host_id"], alive), evidence_starts
        )
        has_expired = (
            snapshot_time - open_table["first_time"]
            > open_table["window"] + _TIME_TOLERANCE
        )
        closing = has_expired | has_lost_host
        if self.min_initial_resp < 0:
            self._admit_late_hosts(particle_ids, host_ids, closing)
            evidence_starts = _run_starts(self._evidence_table["particle_id"])
        evidence_table = self._evidence_table
        evidence_sizes = _run_sizes(evidence_starts, evidence_table["particle_id"].size)
        # The open particle each evidence row belongs to.
        owner = np.repeat(np.arange(n_open), evidence_sizes)

        # The weight each candidate has at this output: rows of the evidence table.
        pair_rows = self._evidence_row_of_pairs(particle_ids, host_ids)
        owner_of_pair = np.searchsorted(open_table["particle_id"], particle_ids)
        is_usable = (pair_rows >= 0) & ~closing[owner_of_pair]
        output_weight = np.zeros(evidence_table["particle_id"].size)
        output_weight[pair_rows[is_usable]] = weights[is_usable]
        weight_sum = np.add.reduceat(output_weight, evidence_starts)
        has_candidate = weight_sum > 0
        is_updated = has_candidate & ~closing  # this output has a candidate in it

        # Fuse the output in: tau_c = window / budget.
        elapsed = snapshot_time - open_table["last_time"]
        time_since_last = np.where(is_updated, elapsed, 0.0)
        persistence = np.exp(-time_since_last * self.budget / open_table["window"])
        share = output_weight / np.where(has_candidate, weight_sum, 1.0)[owner]
        likelihood = persistence[owner] + (1.0 - persistence[owner]) * share
        log_evidence = np.where(
            is_updated[owner],
            evidence_table["log_evidence"] + np.log(likelihood),
            evidence_table["log_evidence"],
        )
        self._evidence_table = _freeze({
            "particle_id": evidence_table["particle_id"],
            "host_id": evidence_table["host_id"],
            "log_evidence": log_evidence,
        })

        # The leader may have changed wherever the evidence did.
        updated = np.flatnonzero(is_updated)
        leader = open_table["leader"].copy()
        moved = _NO_MOVES
        if updated.size:
            updated_rows = {
                name: column[is_updated[owner]]
                for name, column in self._evidence_table.items()
            }
            new_leader, _ = _leaders(updated_rows)
            has_moved = new_leader != leader[updated]
            moved = (
                open_table["particle_id"][updated][has_moved],
                leader[updated][has_moved],
                new_leader[has_moved],
            )
            leader[updated] = new_leader
        self._open_table = _freeze({
            "particle_id": open_table["particle_id"],
            "first_time": open_table["first_time"],
            "last_time": np.where(is_updated, snapshot_time, open_table["last_time"]),
            "window": open_table["window"],
            "leader": leader,
        })

        if closing.any():
            self._close_windows(closing)
        return moved, int(closing.sum())

    def _admit_late_hosts(self, particle_ids, host_ids, closing):
        """Add evidence rows for the hosts an open particle is first seen in.

        The new rows start from the late prior ``BIRTH_LATE_PRIOR``.

        Parameters
        ----------
        particle_ids, host_ids : ndarray of shape (n_pairs,)
            The pairs of the particles that are open.
        closing : ndarray of bool, shape (n_open,)
            The open particles whose window closes at this output; they admit
            nothing.
        """
        owner_of_pair = np.searchsorted(self._open_table["particle_id"], particle_ids)
        is_missing = (
            (self._evidence_row_of_pairs(particle_ids, host_ids) < 0)
            & ~closing[owner_of_pair]
        )
        if not is_missing.any():
            return
        new_rows = {
            "particle_id": particle_ids[is_missing],
            "host_id": host_ids[is_missing],
            "log_evidence": np.full(int(is_missing.sum()), np.log(BIRTH_LATE_PRIOR)),
        }
        merged = {
            name: np.concatenate([self._evidence_table[name], new_rows[name]])
            for name in new_rows
        }
        self._evidence_table = _freeze(
            _sort_table(merged, (merged["host_id"], merged["particle_id"]))
        )

    def _register_new_particles(
        self, snapshot_time, particle_ids, host_ids, dynamical_times, weights
    ):
        """Decide the birth of the particles first seen, or open a window for them.

        Parameters
        ----------
        snapshot_time : float
            Cosmic time of the output.
        particle_ids, host_ids : ndarray of shape (n_pairs,)
            The pairs of the new particles, sorted by (particle, host).
        dynamical_times, weights : ndarray of float, shape (n_pairs,)
            The dynamical time and the evidence weight of each pair.

        Returns
        -------
        born : tuple of ndarray
            The new particles and their first (leading) host.
        n_at_once : int
            Number of particles whose birth was decided at once.
        """
        if particle_ids.size == 0:
            return _NO_MOVES[:2], 0
        starts = _run_starts(particle_ids)
        sizes = _run_sizes(starts, particle_ids.size)
        share = weights / np.repeat(np.add.reduceat(weights, starts), sizes)
        leading_pair = _run_argmax(share, starts, sizes)
        leading_share = share[leading_pair]
        accepts_late_hosts = self.min_initial_resp < 0
        if accepts_late_hosts:
            is_decided = np.zeros(starts.size, dtype=bool)
        else:
            is_decided = (sizes == 1) | (leading_share >= self._decide_share)
        n_at_once = int(is_decided.sum())
        if is_decided.any():
            self._record_finalized(
                particle_ids[starts[is_decided]],
                host_ids[leading_pair[is_decided]],
                leading_share[is_decided],
                np.full(n_at_once, snapshot_time),
            )
        born = (particle_ids[starts], host_ids[leading_pair])
        if is_decided.all():
            return born, n_at_once

        # Candidates are the hosts above the threshold, and always the largest;
        # the window is the mean of their dynamical times, weighted by the share.
        is_leading_pair = np.arange(particle_ids.size) == np.repeat(leading_pair, sizes)
        is_candidate = (share > self.min_initial_resp) | is_leading_pair
        has_time = is_candidate & np.isfinite(dynamical_times) & (dynamical_times > 0)
        time_weight = np.where(has_time, share, 0.0)
        weight_sum = np.add.reduceat(time_weight, starts)
        weighted_time = np.add.reduceat(
            time_weight * np.where(has_time, dynamical_times, 0.0), starts
        )
        has_any_time = weight_sum > 0
        window = np.where(
            has_any_time,
            weighted_time / np.where(has_any_time, weight_sum, 1.0),
            MAX_DYN_TIMESCALE,
        )

        is_opened = ~is_decided
        n_opened = int(is_opened.sum())
        new_open = {
            "particle_id": particle_ids[starts[is_opened]],
            "first_time": np.full(n_opened, snapshot_time),
            "last_time": np.full(n_opened, snapshot_time),
            "window": window[is_opened],
            "leader": host_ids[leading_pair[is_opened]],
        }
        opens_pair = np.repeat(is_opened, sizes) & is_candidate
        new_evidence = {
            "particle_id": particle_ids[opens_pair],
            "host_id": host_ids[opens_pair],
            "log_evidence": np.log(share[opens_pair]),
        }
        merged = {
            name: np.concatenate([self._open_table[name], new_open[name]])
            for name in new_open
        }
        self._open_table = _freeze(_sort_table(merged, (merged["particle_id"],)))
        merged = {
            name: np.concatenate([self._evidence_table[name], new_evidence[name]])
            for name in new_evidence
        }
        self._evidence_table = _freeze(
            _sort_table(merged, (merged["host_id"], merged["particle_id"]))
        )
        return born, n_at_once

    def _close_windows(self, closing):
        """Finalise the open particles flagged in ``closing``.

        Each takes the host with the most evidence it has now.

        Parameters
        ----------
        closing : ndarray of bool, shape (n_open,)
            Which of the open particles to finalise.
        """
        open_table, evidence_table = self._open_table, self._evidence_table
        evidence_particles = evidence_table["particle_id"]
        evidence_starts = _run_starts(evidence_particles)
        evidence_sizes = _run_sizes(evidence_starts, evidence_particles.size)
        is_closing_row = np.repeat(closing, evidence_sizes)
        host_ids, probabilities = _leaders({
            name: column[is_closing_row] for name, column in evidence_table.items()
        })
        self._record_finalized(
            open_table["particle_id"][closing],
            host_ids,
            probabilities,
            open_table["first_time"][closing],
        )
        self._open_table = _freeze({
            name: column[~closing] for name, column in open_table.items()
        })
        self._evidence_table = _freeze({
            name: column[~is_closing_row] for name, column in evidence_table.items()
        })

    def _record_finalized(self, particle_ids, birth_ids, probabilities, first_times):
        """Record particles whose birth is decided.

        Parameters
        ----------
        particle_ids : ndarray of shape (n,)
            Simulation IDs of the particles.
        birth_ids : ndarray of shape (n,)
            Their birth hosts.
        probabilities : ndarray of float, shape (n,)
            The probability of each birth host.
        first_times : ndarray of float, shape (n,)
            When each particle first appeared.
        """
        self._finalized_chunks.append(_freeze({
            "particle_id": particle_ids,
            "birth_id": birth_ids.astype(GALAXY_ID, copy=False),
            "p_birth": probabilities.astype(np.float32),
            "first_time": first_times.astype(np.float32),
        }))
        self._finalized_ids = _freeze(
            {"ids": _merge_sorted(self._finalized_ids, particle_ids)}
        )["ids"]

    # ── outputs ────────────────────────────────────────────────────────
    @staticmethod
    def _concatenate_finalized(chunks):
        """Join chunks of finalised records into one table.

        Parameters
        ----------
        chunks : list of dict of ndarray
            Chunks of records, in the order they were finalised.

        Returns
        -------
        finalized : dict of ndarray
            Columns ``particle_id``, ``birth_id``, ``p_birth`` and ``first_time``.
        """
        if not chunks:
            return {
                "particle_id": np.empty(0, SIM_ID),
                "birth_id": np.empty(0, GALAXY_ID),
                "p_birth": np.empty(0, np.float32),
                "first_time": np.empty(0, np.float32),
            }
        return {
            name: np.concatenate([chunk[name] for chunk in chunks])
            for name in chunks[0]
        }

    def finalize(self):
        """Close every open particle with its current evidence and return the births.

        Returns
        -------
        birth_df : DataFrame
            Columns: ``particle_index``, ``birth_id``, ``p_birth`` (the
            probability of the birth host: 1 for a particle decided at once)
            and ``birth_time`` (the cosmic time it first appeared).
        """
        n_open = self._open_table["particle_id"].size
        if n_open:
            self._close_windows(np.ones(n_open, dtype=bool))
        return self.finalized()

    def finalized(self):
        """Return the births of the particles finalised so far.

        Returns
        -------
        birth_df : DataFrame
            The columns of :meth:`finalize`.
        """
        finalized = self._concatenate_finalized(self._finalized_chunks)
        return pd.DataFrame({
            "particle_index": finalized["particle_id"].astype(SIM_ID, copy=False),
            "birth_id": finalized["birth_id"].astype(GALAXY_ID, copy=False),
            "p_birth": finalized["p_birth"],
            "birth_time": finalized["first_time"],
        })

    def open_ids(self):
        """Return the particles whose window is still open.

        Returns
        -------
        particle_ids : ndarray of SIM_ID
            Simulation IDs, sorted.
        """
        return self._open_table["particle_id"]

    def evidence(self, particle_id):
        """Return the evidence an open particle has collected for each candidate host.

        Parameters
        ----------
        particle_id : int
            Simulation ID of the particle.

        Returns
        -------
        log_evidence : dict of {int: float}
            ``{host ID: log posterior}`` of the particle's candidates, up to a
            constant; empty if the particle is not open.
        """
        is_particle = self._evidence_table["particle_id"] == particle_id
        hosts = self._evidence_table["host_id"][is_particle]
        log_evidence = self._evidence_table["log_evidence"][is_particle]
        return dict(zip(hosts.tolist(), log_evidence.tolist()))

    def window(self, particle_id):
        """Return the length of the window of an open particle.

        Parameters
        ----------
        particle_id : int
            Simulation ID of an open particle.

        Returns
        -------
        window : float
            The window ``L``, in the unit of the dynamical times.
        """
        is_particle = self._open_table["particle_id"] == particle_id
        return float(self._open_table["window"][is_particle][0])

    # ── checkpointing ──────────────────────────────────────────────────
    def capture(self):
        """Capture a checkpoint-safe reference to the state (C02).

        Every array is replaced rather than edited by later updates (and is
        read-only), so references are enough; nothing is copied element-wise.

        Returns
        -------
        reference : dict
            Pass to :meth:`serialize` for a plain, fully independent form.
        """
        return {
            "open_table": dict(self._open_table),
            "evidence_table": dict(self._evidence_table),
            "finalized_chunks": list(self._finalized_chunks),
            "budget": self.budget,
            "min_initial_resp": self.min_initial_resp,
            "last_snapshot": self._last_snapshot,
        }

    @staticmethod
    def serialize(reference):
        """Convert a :meth:`capture` reference into its checkpoint form.

        Parameters
        ----------
        reference : dict
            Output of :meth:`capture`.

        Returns
        -------
        state : dict
            The state in the form :meth:`restore` reads.
        """
        finalized = BirthTracker._concatenate_finalized(reference["finalized_chunks"])
        return {
            "format": _FORMAT,
            "open_particles": dict(reference["open_table"]),
            "evidence": dict(reference["evidence_table"]),
            "finalized": finalized,
            "budget": reference["budget"],
            "min_initial_resp": reference["min_initial_resp"],
            "last_snapshot": reference["last_snapshot"],
        }

    def restore(self, state):
        """Restore the tracker from a checkpoint.

        Parameters
        ----------
        state : dict
            Output of :meth:`serialize`.

        Raises
        ------
        ValueError
            If the checkpoint was written by an earlier tracker, whose state has
            no equivalent here: restart the run.
        """
        if state.get("format") != _FORMAT:
            raise ValueError(
                "The checkpoint holds the state of an earlier birth tracker, "
                "which this tracker cannot continue: delete it and start a fresh run."
            )
        self.budget = state["budget"]
        self.min_initial_resp = state["min_initial_resp"]
        self._decide_share = 1.0 / (1.0 + np.exp(-self.budget))
        self._open_table = _freeze(dict(state["open_particles"]))
        self._evidence_table = _freeze(dict(state["evidence"]))
        finalized = state["finalized"]
        has_records = finalized["particle_id"].size > 0
        self._finalized_chunks = [_freeze(dict(finalized))] if has_records else []
        self._finalized_ids = _freeze(
            {"ids": np.sort(finalized["particle_id"])}
        )["ids"]
        self._last_snapshot = state["last_snapshot"]
