"""Where each star lives: per-snapshot memberships and rightful owners,
reconstructed from roadrunner's bookkeeping (runs where the roadrunner
output lives).

    from replay.ownership import Ownership
    own = Ownership(rr_out)            # nothing is read yet
    m = own.at(87)                     # replays snapshots 0..87 on first use
    m.pid, m.owner                     # every star present at 87 (sorted by id) and its owner
    m.owner_of(ids)                    # owners of given particle ids (-2: not present)
    own.owners(ids, s)                 # owners of a few ids only (no full-snapshot view)
    m.memberships()                    # (pid, galaxy): every infall list holding the star
    own.in_list(ids, gids, 87)         # is star ids[i] in galaxy gids[i]'s list at 87?
    own.list_sums(ids, masses, 87)     # per galaxy: stars of its list among ids, their mass
    own.at(68)                         # earlier snapshots: instant (no new replay)
    own.at(146)                        # later ones: the replay continues from 87
    own.birth_of(ids), own.satellites(87), own.info

What roadrunner stores, and what is rebuilt
-------------------------------------------
roadrunner keeps, per galaxy, an infall list (AssemblyTracker): the stars
born in it (birth map, catalogue final/births) plus every star it accreted.
A galaxy g accretes star p at snapshot s when p is hard-assigned to g at s
(assignment/snapshotNNNN.hdf5 hard_assignment) and p is already in the list
of one of g's satellites (catalogue snapshots/N/satellite_relations, this or
the previous snapshot, restricted to galaxies with an assignment group, as
the pipeline does) or was born unbound (galaxy -1); a satellite is processed
before its host, so a star can climb several levels in one snapshot. Lists
only grow, and only their end state is saved (final/assembly: one
(particle, galaxy) row per membership), not when each membership began.

The replay walks the snapshots and lets only the memberships of
final/assembly be acquired, which dates each one:
  birth   (p, birth_id) at p's first appearance (final birth map);
  accrete (p, g) as above, repeated within the snapshot until nothing
          changes (satellite before host); the pass number orders the
          acquisitions of one snapshot.
memberships at s: every (p, g) acquired by s (a star sits in its birth
galaxy's list and in those of all hosts that took it in: not unique).
owner at s: the membership acquired last (latest snapshot, then pass): the
highest level that has legitimately claimed the star, i.e. where it should
be now. It is unique per star and snapshot (ties are counted in info: 0 in
v6). If that galaxy is no longer in the merger tree at s, ownership passes
down its descendant links to the galaxy it merged into. Owner -1: born
unbound and never accreted.

Birth window: roadrunner's provisional birth host can differ from the final
one for a few snapshots after a star's birth; here the final birth map owns
it from its first appearance. Check: v6 phase replayed to snapshot 178
acquires all but 85 of 3,854,032 rows of final/assembly (2e-5), every birth
row included (info['unacquired'] after own.at(last)).
"""
import os
import time

import numpy as np

MASK32 = (1 << 32) - 1


def _expand(off, p):
    """Pair indices of particles p (pairs sorted by particle): (idx, row)."""
    L = off[p + 1] - off[p]
    row = np.repeat(np.arange(p.size), L)
    idx = np.arange(L.sum()) - np.repeat(np.cumsum(L) - L, L) + np.repeat(off[p], L)
    return idx, row


class Snapshot:
    """Memberships and owners of the stars present at one snapshot."""

    def __init__(self, own, s, pid, owner, idx, row):
        self.snapshot = s
        self.pid = pid            # sorted particle ids present at s
        self.owner = owner        # their owners (after merger resolution)
        self._own, self._idx, self._row = own, idx, row

    def owner_of(self, ids):
        """Owners of particle ids (any order); -2 where not present at s."""
        ids = np.asarray(ids, dtype=np.int64)
        k = np.clip(np.searchsorted(self.pid, ids), 0, max(self.pid.size - 1, 0))
        return np.where(self.pid[k] == ids, self.owner[k], -2) if self.pid.size else np.full(ids.size, -2)

    def memberships(self):
        """(pid, galaxy): every infall list holding each present star at s."""
        o = self._own
        held = o._acq[self._idx] >= 0
        held &= o._acq[self._idx] <= self.snapshot
        return self.pid[self._row[held]], o._pg[self._idx[held]]


class Ownership:
    """Lazy per-snapshot reconstruction of roadrunner's memberships."""

    def __init__(self, rr_out, verbose=False):
        self.rr_out = rr_out
        self.verbose = verbose
        self._loaded = False
        self._done = -1               # last snapshot replayed
        self._cache = {}
        self.info = dict(ties=0, dead_resolved=0, unresolved_dead=0)
        self.progress = None          # optional wrapper of the replay loop, e.g. tqdm

    # -- catalogue tables (read once, on first use) --------------------------
    def _load(self):
        if self._loaded:
            return
        import h5py
        with h5py.File(os.path.join(self.rr_out, 'catalogue.hdf5'), 'r') as cat:
            self._tree = cat['header/merger_tree'][()]
            births = cat['final/births'][()]
            asm = cat['final/assembly'][()]
            self._last = int(cat['header/last_snapshot'][()])
        order = np.lexsort((asm['galaxy_id'], asm['particle_index']))
        self._pp = asm['particle_index'][order].astype(np.int64)
        self._pg = asm['galaxy_id'][order].astype(np.int64)
        npid = int(max(self._pp.max(), births['particle_index'].max())) + 1
        self._off = np.searchsorted(self._pp, np.arange(npid + 1))
        self._birth = np.full(npid, -2, np.int64)
        self._birth[births['particle_index'].astype(np.int64)] = births['birth_id']
        self._is_birth = self._pg == self._birth[self._pp]
        self._acq = np.full(self._pp.size, -1, np.int32)     # snapshot acquired
        self._rank = np.zeros(self._pp.size, np.int16)       # pass within it (birth 0)
        self._first = np.full(npid, -1, np.int32)            # first appearance
        self._assigned = np.full(npid, -9, np.int64)
        self._prev_keys = np.empty(0, np.int64)
        self._loaded = True

    @property
    def last_snapshot(self):
        self._load()
        return self._last

    def birth_of(self, ids):
        """Final birth galaxy of particle ids (-1 born unbound, -2 unknown)."""
        self._load()
        ids = np.asarray(ids, dtype=np.int64)
        ok = (ids >= 0) & (ids < self._birth.size)
        return np.where(ok, self._birth[np.where(ok, ids, 0)], -2)

    def satellites(self, s):
        """{host: array of satellites} of snapshot s, restricted to galaxies
        with an assignment group (the pipeline's alive set)."""
        alive = self._alive(s)
        out = {}
        for g, sats in self._sat_map(s).items():
            if g in alive:
                sats = sats[np.isin(sats, list(alive))]
                if sats.size:
                    out[g] = sats
        return out

    def _alive(self, s):
        import h5py
        with h5py.File(os.path.join(self.rr_out, 'assignment', f'snapshot{s:04d}.hdf5'), 'r') as fa:
            return set(int(k) for k in fa['galaxies'])

    def _sat_map(self, s):
        import h5py
        with h5py.File(os.path.join(self.rr_out, 'catalogue.hdf5'), 'r') as cat:
            grp = cat.get(f'snapshots/{s}/satellite_relations')
            return {} if grp is None else {int(k): grp[k][()].astype(np.int64) for k in grp}

    # -- replay ----------------------------------------------------------------
    def _advance(self, upto):
        import h5py
        t0 = time.time()
        pp, pg, off = self._pp, self._pg, self._off
        steps = range(self._done + 1, upto + 1)
        if self.progress is not None:
            steps = self.progress(steps, desc='ownership replay', unit='snap')
        for s in steps:
            with h5py.File(os.path.join(self.rr_out, 'assignment', f'snapshot{s:04d}.hdf5'), 'r') as fa:
                ha = fa['hard_assignment'][()]
                alive = set(int(k) for k in fa['galaxies'])
            pid = ha['particle_index'].astype(np.int64)
            self._assigned[:] = -9
            self._assigned[pid] = ha['Sub_tree_id']
            new = pid[self._first[pid] < 0]
            self._first[new] = s
            if new.size:                                   # birth memberships
                idx, _ = _expand(off, new)
                self._acq[idx[self._is_birth[idx]]] = s
            keys = []
            for g, sats in self._sat_map(s).items():
                if g in alive:
                    sats = sats[np.isin(sats, list(alive))]
                    keys.append((np.int64(g) << 32) | sats)
            keys = np.unique(np.concatenate(keys)) if keys else np.empty(0, np.int64)
            prev = self._prev_keys[np.isin(self._prev_keys >> 32, list(alive))]
            hist = np.union1d(keys, prev)                  # current + previous satellite relations
            cand = ((self._acq < 0) & (pg >= 0) & (self._first[pp] >= 0)
                    & (self._assigned[pp] == pg) & np.isin(pg, np.array(sorted(alive), np.int64)))
            n_pass = 0
            while True:                                    # accretion, satellite before host
                c = np.flatnonzero(cand & (self._acq < 0))
                if c.size == 0:
                    break
                idx, row = _expand(off, pp[c])
                h = pg[idx]
                key = (pg[c][row] << 32) | (h & MASK32)
                if hist.size:
                    j = np.clip(np.searchsorted(hist, key), 0, hist.size - 1)
                    in_hist = hist[j] == key
                else:
                    in_hist = np.zeros(key.size, bool)
                ok = (self._acq[idx] >= 0) & ((h == -1) | ((h >= 0) & in_hist))
                hit = np.bincount(row, weights=ok, minlength=c.size) > 0
                if not hit.any():
                    break
                n_pass += 1
                self._acq[c[hit]] = s
                self._rank[c[hit]] = n_pass
            self._prev_keys = keys
            self._done = s
            if self.verbose:
                print(f'  ownership replay: snapshot {s}, passes {n_pass}, '
                      f'acquired {int((self._acq >= 0).sum())}/{self._acq.size}', flush=True)
        self.info['replayed_to'] = self._done
        self.info['unacquired'] = int((self._acq < 0).sum())
        self.info['rows'] = int(self._acq.size)
        self.info['seconds'] = round(self.info.get('seconds', 0.0) + time.time() - t0, 1)

    def at(self, s):
        """Snapshot view: memberships and owners of the stars present at s."""
        s = int(s)
        if s in self._cache:
            return self._cache[s]
        self._ready(s)
        pid = np.flatnonzero((self._first >= 0) & (self._first <= s)).astype(np.int64)
        owner, idx, row = self._owners(pid, s)
        view = Snapshot(self, s, pid, owner, idx, row)
        self._cache[s] = view
        return view

    def owners(self, ids, s):
        """Owners at s of particle ids only (cheaper than at(s) for a few
        stars, e.g. one snapshot's newborns); -2 where not present at s."""
        s = int(s)
        self._ready(s)
        ids = np.asarray(ids, dtype=np.int64)
        ok = (ids >= 0) & (ids < self._first.size)
        ok[ok] = (self._first[ids[ok]] >= 0) & (self._first[ids[ok]] <= s)
        out = np.full(ids.size, -2, np.int64)
        out[ok] = self._owners(ids[ok], s)[0]
        return out

    def newborn_ids(self, s):
        """Particle ids first appearing at snapshot s (sorted)."""
        s = int(s)
        self._ready(s)
        return np.flatnonzero(self._first == s).astype(np.int64)

    # -- infall lists (memberships, not owners) ---------------------------------
    def _pair_keys(self):
        """(particle << 32) | (galaxy + 1) of every final/assembly row, ascending
        (the rows are sorted by particle, then galaxy). Computed once."""
        if getattr(self, '_pairs', None) is None:
            self._load()
            self._pairs = (self._pp << 32) | (self._pg + 1)
        return self._pairs

    def in_list(self, ids, gids, s):
        """Whether star ids[i] is in galaxy gids[i]'s infall list at snapshot s
        (that membership acquired by s)."""
        s = int(s)
        self._ready(s)
        keys = self._pair_keys()
        q = (np.asarray(ids, np.int64) << 32) | (np.asarray(gids, np.int64) + 1)
        if keys.size == 0 or q.size == 0:
            return np.zeros(q.size, bool)
        j = np.clip(np.searchsorted(keys, q), 0, keys.size - 1)
        a = self._acq[j]
        return (keys[j] == q) & (a >= 0) & (a <= s)

    def list_sums(self, ids, w, s):
        """Per galaxy: how many of the stars ids (present at s, weights w) are in
        its infall list at s, and their total weight. (gid ascending, n, w_sum);
        the 'born unbound' list (-1) is left out."""
        s = int(s)
        self._ready(s)
        ids = np.asarray(ids, np.int64)
        w = np.asarray(w, np.float64)
        ok = (ids >= 0) & (ids < self._off.size - 1)
        idx, row = _expand(self._off, ids[ok])
        a = self._acq[idx]
        g = self._pg[idx]
        held = (a >= 0) & (a <= s) & (g >= 0)
        gid, inv = np.unique(g[held], return_inverse=True)
        return (gid, np.bincount(inv, minlength=gid.size),
                np.bincount(inv, weights=w[ok][row[held]], minlength=gid.size))

    def _ready(self, s):
        self._load()
        if s > self._last or s < 0:
            raise ValueError(f'snapshot {s} outside 0..{self._last}')
        if s > self._done:
            self._advance(s)

    def _owners(self, pid, s):
        idx, row = _expand(self._off, pid)
        a = self._acq[idx]
        score = np.where((a >= 0) & (a <= s), a.astype(np.int64) * 64 + self._rank[idx], -1)
        o = np.lexsort((score, row))                       # per star ascending; last = owner
        last = np.flatnonzero(np.r_[row[o][1:] != row[o][:-1], True]) if o.size else np.empty(0, int)
        best = o[last]
        owner = np.where(score[best] >= 0, self._pg[idx[best]], -1)
        prev = o[np.maximum(last - 1, 0)]
        self.info['ties'] += int(np.sum((last > 0) & (row[prev] == row[best])
                                        & (score[prev] == score[best]) & (score[best] >= 0)))
        return self._resolve_dead(owner, s), idx, row

    def _resolve_dead(self, owner, s):
        """Owners absent from the merger tree at s pass to the galaxy they merged into."""
        tree = self._tree
        present = set(tree['Sub_tree_id'][tree['Snapshot'] == s].astype(np.int64).tolist())
        if not hasattr(self, '_uid_row'):
            uid = np.asarray(tree['uid'], dtype=np.float64)
            self._uid_row = {int(u): i for i, u in enumerate(uid) if np.isfinite(u)}
        out = owner.copy()
        for g in np.unique(owner):
            g = int(g)
            if g < 0 or g in present:
                continue
            rows = np.flatnonzero((tree['Sub_tree_id'] == g) & (tree['Snapshot'] < s))
            target = None
            if rows.size:
                r = rows[np.argmax(tree['Snapshot'][rows])]
                for _ in range(400):
                    d = float(tree['desc_uid'][r])
                    if not np.isfinite(d) or d < 0 or int(d) not in self._uid_row:
                        break
                    r = self._uid_row[int(d)]
                    if int(tree['Snapshot'][r]) >= s:
                        target = int(tree['Sub_tree_id'][r]) if int(tree['Snapshot'][r]) == s else None
                        break
                    if int(tree['Sub_tree_id'][r]) in present:
                        target = int(tree['Sub_tree_id'][r])
                        break
            n = int(np.sum(owner == g))
            if target is not None and target in present:
                out[owner == g] = target
                self.info['dead_resolved'] += n
            else:
                self.info['unresolved_dead'] += n
        return out
