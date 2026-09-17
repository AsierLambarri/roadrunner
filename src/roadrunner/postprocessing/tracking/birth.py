"""Birth-tagging tracker for newly appearing star particles.

Tracks which galaxy a particle first appears in and finalises the
association once ``factor × timescale`` snapshots have elapsed.
"""

import heapq
from collections import defaultdict
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
    """Per-particle accumulation record for unfinalised particles.

    Holds the five primary fields; ``__slots__`` fixes the attribute set,
    so instances store values in a compact array instead of a per-instance
    dict. The birth leader (``leader_host``/``leader_score``) is derived
    on demand as the argmax of ``counts`` rather than stored.
    """

    __slots__ = ("t0", "snap0", "tau", "counts", "initial_hosts")

    def __init__(self, t0, snap0, tau, counts, initial_hosts):
        self.t0 = t0
        self.snap0 = snap0
        self.tau = tau
        self.counts = counts
        self.initial_hosts = initial_hosts

    @property
    def leader_host(self):
        """Host with the highest accumulated window score."""
        return GALAXY_ID(max(self.counts.items(), key=lambda kv: kv[1])[0])

    @property
    def leader_score(self):
        """Highest accumulated window score (ambient math precision)."""
        return math_dtype()(max(self.counts.values()))


class FinalizedInfo(NamedTuple):
    """Per-particle finalized record.

    Tuple footprint (~100 B) instead of a per-particle dict (~350 B).
    The particle index is the ``_finalized`` dict key, not stored here.
    """

    birth_id: GALAXY_ID
    birth_time: np.float32
    birth_snap: SNAP_ID
    timescale: np.float32


class BirthTracker:
    """Tracks the birth host of each star particle across snapshots.

    A particle is considered "born" in the galaxy where it first appears.
    The tracker uses a window function to accumulate evidence over
    ``factor × timescale`` snapshots before finalising the assignment.

    Parameters
    ----------
    factor : float, default=5
        Multiplier on the particle's dynamical timescale to decide
        when to finalise.
    enforce_initial_hosts : bool, default=False
        If ``True``, only the hosts seen at the particle's first
        appearance are considered during the accumulation window.
    window : str, default='gaussian'
        Window function: ``"gaussian"``, ``"cauchy"``, or ``"exp"``.
    """

    def __init__(self, factor=5, enforce_initial_hosts=False, window="gaussian"):
        if window not in _WINDOWS:
            raise ValueError(f"Unknown window: {window}. Choose from {list(_WINDOWS)}")
        self._window_fn = _WINDOWS[window]
        self._heap = []
        self._active = {}
        self._finalized = {}
        self._birth_map = defaultdict(set)
        self.factor = factor - 0.001
        self._last_snapshot = -1
        self.enforce_initial_hosts = enforce_initial_hosts

    def _add_update_particles(self, t_snap, snapshot_id, particle_ids, host_ids, timescales, weights):
        """Update running host counts for active particles and register new ones.

        Parameters
        ----------
        t_snap : float
        snapshot_id : int
        particle_ids : ndarray
        host_ids : ndarray
        timescales : ndarray
        weights : ndarray
        """
        finalized_keys = self._finalized.keys()
        mask_not_finalized = ~np.isin(particle_ids, list(finalized_keys))
        if not np.any(mask_not_finalized):
            return

        pids = particle_ids[mask_not_finalized]
        hosts = host_ids[mask_not_finalized]
        taus = timescales[mask_not_finalized]
        ws = weights[mask_not_finalized]

        is_active = np.isin(pids, list(self._active.keys()))
        if np.any(is_active):
            p_active = pids[is_active]
            host_active = hosts[is_active]
            w_active = ws[is_active]

            md = math_dtype()
            t0 = np.array([self._active[p].t0 for p in p_active], dtype=md)
            tau0 = np.array([self._active[p].tau for p in p_active], dtype=md)
            tau0 = np.maximum(tau0, md(1e-10))

            deltas = (w_active * self._window_fn((t_snap - t0) / tau0)).astype(md, copy=False)
            for p, host, delta in zip(p_active, host_active, deltas):
                info = self._active[p]
                if self.enforce_initial_hosts and host not in info.initial_hosts:
                    continue
                info.counts[host] += delta

        not_active = ~is_active
        if np.any(not_active):
            p_new = pids[not_active]
            host_new = hosts[not_active]
            tau_new = taus[not_active]
            w_new = ws[not_active]

            order = np.argsort(p_new)
            p_sorted = p_new[order]
            host_sorted = host_new[order]
            tau_sorted = tau_new[order]
            w_sorted = w_new[order]

            unique_particles, start_idx = np.unique(p_sorted, return_index=True)
            end_idx = np.append(start_idx[1:], len(p_sorted))
            for p, start, end in zip(unique_particles, start_idx, end_idx):
                particle_hosts = host_sorted[start:end]
                particle_weights = w_sorted[start:end]
                particle_tau = tau_sorted[start:end].max()

                hosts_u, inverse_u = np.unique(particle_hosts, return_inverse=True)
                sums = np.zeros(len(hosts_u), dtype=math_dtype())
                np.add.at(sums, inverse_u, particle_weights)

                counts = defaultdict(float, zip(hosts_u, sums))
                initial_hosts = set(hosts_u)

                md = math_dtype()
                self._active[p] = ActiveParticleInfo(
                    t0=md(t_snap),
                    snap0=SNAP_ID(snapshot_id),
                    tau=md(particle_tau),
                    counts=counts,
                    initial_hosts=initial_hosts,
                )
                heapq.heappush(self._heap, (md(t_snap + self.factor * particle_tau), SIM_ID(p)))

    def _finalize_particles(self, t_snap):
        """Finalise particles whose accumulation window has expired.

        Pops particles from the heap where ``t0 + factor × tau ≤ t_snap``
        and moves them from active to finalised state.

        Parameters
        ----------
        t_snap : float
        """
        while self._heap and self._heap[0][0] <= t_snap:
            _, p_to_finalize = heapq.heappop(self._heap)
            if p_to_finalize not in self._active:
                continue

            info = self._active.pop(p_to_finalize)
            birth_id = info.leader_host

            self._finalized[p_to_finalize] = FinalizedInfo(
                birth_id=birth_id,
                birth_time=info.t0,
                birth_snap=info.snap0,
                timescale=info.tau,
            )

            self._birth_map[birth_id].add(p_to_finalize)

    def update(self, t_snap, snapshot_id, particle_ids, host_ids, timescales, weights=None):
        """Update the tracker with particles from a new snapshot.

        Parameters
        ----------
        t_snap : float
        snapshot_id : int
        particle_ids : ndarray
        host_ids : ndarray
        timescales : ndarray
        weights : ndarray or None, optional
            Defaults to uniform weights.
        """
        if snapshot_id > self._last_snapshot:
            if weights is None:
                weights = np.full(particle_ids.shape, 1.0, dtype=math_dtype())
            self._add_update_particles(t_snap, snapshot_id, particle_ids, host_ids, timescales, weights)
            self._finalize_particles(t_snap)
            self._last_snapshot = snapshot_id

    def finalize(self):
        """Force-finalise all active particles and return the birth table.

        Returns
        -------
        birth_df : DataFrame
            Columns: ``particle_index``, ``birth_id``.
        """
        self._finalize_particles(np.inf)
        records = [
            {"particle_index": p, "birth_id": self._finalized[p].birth_id}
            for particles in self._birth_map.values()
            for p in particles
        ]
        birth_df = pd.DataFrame.from_records(
            records, columns=["particle_index", "birth_id"],
        )
        return birth_df

    def current_birth_map(self):
        """Return the current (incomplete) birth map.

        Includes both finalised and active particles.

        Returns
        -------
        birth_map : dict of {int: set of int}
        """
        birth_map = {k: set(v) for k, v in self._birth_map.items()}
        for p, info in self._active.items():
            birth_map.setdefault(info.leader_host, set()).add(p)
        return birth_map

    def _get_state(self):
        """Serialise the tracker state for checkpointing.

        Returns
        -------
        state : dict
        """
        return {
            "active": dict(self._active),
            "finalized": dict(self._finalized),
            "heap": list(self._heap),
            "birth_map": {k: list(v) for k, v in self._birth_map.items()},
            "factor": self.factor,
            "last_snapshot": self._last_snapshot,
            "enforce_initial_hosts": self.enforce_initial_hosts,
        }

    def _set_state(self, state):
        """Restore tracker state from a checkpoint.

        Parameters
        ----------
        state : dict
        """
        # Normalise pre-B2 checkpoints whose active records are plain dicts.
        # Stored leader_host/leader_score (old format) are dropped: the
        # leader is now derived from counts.
        active = {}
        for p, info in state["active"].items():
            if isinstance(info, dict):
                info = ActiveParticleInfo(
                    info["t0"], info["snap0"], info["tau"],
                    defaultdict(float, info["counts"]),
                    info["initial_hosts"],
                )
            active[p] = info
        self._active = active
        # Normalise pre-B3 checkpoints whose finalized records are plain dicts.
        finalized = {}
        for p, rec in state["finalized"].items():
            if isinstance(rec, dict):
                rec = FinalizedInfo(
                    rec["birth_id"], rec["birth_time"],
                    rec["birth_snap"], rec["timescale"],
                )
            finalized[p] = rec
        self._finalized = finalized
        self._heap = state["heap"]
        heapq.heapify(self._heap)
        self._birth_map = defaultdict(
            set, {k: set(v) for k, v in state["birth_map"].items()}
        )
        self.factor = state["factor"]
        self._last_snapshot = state["last_snapshot"]
        self.enforce_initial_hosts = state["enforce_initial_hosts"]
