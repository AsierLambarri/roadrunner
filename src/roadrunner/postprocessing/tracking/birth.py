#############################################################################
#
# package:   roadrunner.postprocessing.tracking
# file:      birth.py
# brief:     Birth-tagging tracker for newborn particles.
#
# copyright: GPLv3
# author:    Asier Lambarri Martinez
# changes:   18 may 2026 - Created
#            18 may 2026 - Last edit
#
#############################################################################

import heapq
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from roadrunner._defaults import BIRTH_GAUSSIAN_WIDTH


def _exp_window(x):
    return np.exp(-x)


def _cauchy_window(x):
    return 1.0 / (1 + x ** 2)


def _gaussian_window(x):
    return np.exp(-BIRTH_GAUSSIAN_WIDTH * x ** 2)


_WINDOWS = {
    "gaussian": _gaussian_window,
    "cauchy": _cauchy_window,
    "exp": _exp_window,
}


class BirthTracker:
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

            t0 = np.array([self._active[p]["t0"] for p in p_active])
            tau0 = np.array([self._active[p]["tau"] for p in p_active])
            tau0 = np.maximum(tau0, 1e-10)

            deltas = (w_active * self._window_fn((t_snap - t0) / tau0)).astype(np.float64)
            for p, host, delta in zip(p_active, host_active, deltas):
                info = self._active[p]
                if self.enforce_initial_hosts and host not in info["initial_hosts"]:
                    continue
                info["counts"][host] += delta
                if info["counts"][host] > info["leader_score"]:
                    info["leader_host"] = np.uint64(host)
                    info["leader_score"] = info["counts"][host]

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
                sums = np.zeros(len(hosts_u), dtype=np.float64)
                np.add.at(sums, inverse_u, particle_weights)

                counts = Counter(dict(zip(hosts_u, sums)))
                initial_hosts = set(hosts_u)
                leader_host, leader_score = counts.most_common(1)[0]

                self._active[p] = {
                    "t0": np.float32(t_snap),
                    "snap0": np.uint32(snapshot_id),
                    "tau": np.float32(particle_tau),
                    "counts": counts,
                    "initial_hosts": initial_hosts,
                    "leader_host": np.uint64(leader_host),
                    "leader_score": np.float32(leader_score),
                }
                heapq.heappush(self._heap, (np.float32(t_snap + self.factor * particle_tau), np.uint64(p)))

    def _finalize_particles(self, t_snap):
        while self._heap and self._heap[0][0] <= t_snap:
            _, p_to_finalize = heapq.heappop(self._heap)
            if p_to_finalize not in self._active:
                continue

            info = self._active.pop(p_to_finalize)
            birth_id = info["leader_host"]

            self._finalized[p_to_finalize] = {
                "particle_index": p_to_finalize,
                "birth_time": info["t0"],
                "birth_snap": info["snap0"],
                "birth_id": birth_id,
                "timescale": info["tau"],
            }

            self._birth_map[birth_id].add(p_to_finalize)

    def update(self, t_snap, snapshot_id, particle_ids, host_ids, timescales, weights=None):
        if snapshot_id > self._last_snapshot:
            if weights is None:
                weights = np.full(particle_ids.shape, 1.0)
            self._add_update_particles(t_snap, snapshot_id, particle_ids, host_ids, timescales, weights)
            self._finalize_particles(t_snap)
            self._last_snapshot = snapshot_id

    def finalize(self):
        self._finalize_particles(np.inf)
        records = [
            {"particle_index": p, "birth_id": self._finalized[p]["birth_id"]}
            for particles in self._birth_map.values()
            for p in particles
        ]
        birth_df = pd.DataFrame.from_records(
            records, columns=["particle_index", "birth_id"],
        )
        return birth_df

    def current_birth_map(self):
        birth_map = {k: set(v) for k, v in self._birth_map.items()}
        for p, info in self._active.items():
            birth_map.setdefault(info["leader_host"], set()).add(p)
        return birth_map

    def _get_state(self):
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
        self._active = state["active"]
        self._finalized = state["finalized"]
        self._heap = state["heap"]
        heapq.heapify(self._heap)
        self._birth_map = defaultdict(set, {k: set(v) for k, v in state["birth_map"].items()})
        self.factor = state["factor"]
        self._last_snapshot = state["last_snapshot"]
        self.enforce_initial_hosts = state["enforce_initial_hosts"]
