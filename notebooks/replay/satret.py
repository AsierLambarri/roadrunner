"""Per-satellite f_retention over time (plot 3.2, plots/sim/s032_satellites.py),
computed in memory from the roadrunner output (runs where it lives).

For every galaxy g other than the host and every snapshot (stars within
r_keep host R_vir, replay.starmap; owners from replay.ownership), among the
stars g owns (where they should be) and binds:
  stolen  also bound to the host, and more bound to it (E_H < E_O): greedy
          most-bound assignment gives them to the host;
  f_ret(g, t) = 1 - M_stolen / M_bound (stars the host does not bind cannot
          be taken: they count as kept).
Also per snapshot: M_owned (all stars g owns, bound or not; the stellar
mass used to choose satellites) and N_owned.

sats_list: the same, with g's infall list (replay.ownership.in_list) in place
of the stars it owns: among the stars in g's list that it binds (roadrunner's
own stellar mass, the catalogue's Mtot; whole snapshot, no r_keep cut), stolen
= also bound to the host and E_H < E_g (its energy relative to g); owned, n:
the stars of its list present, bound or not. Star formation (born[g]): per
snapshot, the newborn mass (stars first appearing at that snapshot) whose
final birth galaxy (birth map) is g, in total and within the host R_vir,
and the part of the latter greedy gives to the host (bound to the host and
to its owner and E_H < E_O; m_in_taken_list: bound to the host and to g, in
whose list it is, and E_H < E_g).
"""
import os
import time

import numpy as np

FLAG_HOST, FLAG_OWNER_BOUND = 1, 8
_G = {}


def _snapshot(s):
    t0 = time.time()
    d = _G['sm'].at(s, cache=False)
    host = int(d['host'])
    g = np.asarray(d['owner_id'], dtype=np.int64)
    ok = (g >= 0) & (g != host)
    if not ok.any():
        return dict(snapshot=s, gid=np.empty(0, np.int64), owned=np.empty(0), n=np.empty(0),
                    bound=np.empty(0), stolen=np.empty(0), lists=d['lists'], born=_newborns(d, s, host),
                    seconds=time.time() - t0)
    born = _newborns(d, s, host)
    fl, m = d['flags'][ok], d['mass'][ok].astype(np.float64)
    gid, inv = np.unique(g[ok], return_inverse=True)
    bound = (fl & FLAG_OWNER_BOUND) > 0
    stolen = bound & ((fl & FLAG_HOST) > 0) & (d['E_H'][ok] < d['E_O'][ok])
    sums = lambda w: np.bincount(inv, weights=w, minlength=gid.size)
    return dict(snapshot=s, gid=gid, owned=sums(m), n=np.bincount(inv, minlength=gid.size).astype(float),
                bound=sums(m * bound), stolen=sums(m * stolen), lists=d['lists'], born=born,
                seconds=round(time.time() - t0, 1))


def _newborns(d, s, host):
    """{birth galaxy: (newborn mass, newborn mass within R_vir, of which given to the host:
    owner view, list view)}."""
    own = _G['sm'].ownership
    ids = d['ids']
    nb = ids < own._first.size
    nb[nb] = own._first[ids[nb]] == s
    if not nb.any():
        return {}
    b = own.birth_of(ids[nb])
    keep = (b >= 0) & (b != host)
    if not keep.any():
        return {}
    idx = np.flatnonzero(nb)[keep]
    m = d['mass'][idx].astype(np.float64)
    inside = np.linalg.norm(d['pos'][idx], axis=1) <= float(d['rvir_kpc'])
    fl = d['flags'][idx]
    taken = ((fl & FLAG_HOST) > 0) & ((fl & FLAG_OWNER_BOUND) > 0) & (d['E_H'][idx] < d['E_O'][idx])

    # the list view: bound to its birth galaxy (in whose list it is) and E_H < E_birth
    birth_row = np.full(ids.size, -9, np.int64)
    birth_row[idx] = b[keep]
    mem = d['mem']
    pair = birth_row[mem['row']] == mem['gid']
    r_, E_ = mem['row'][pair], mem['E'][pair]
    taken_l = np.zeros(ids.size, bool)
    taken_l[r_] = ((d['flags'][r_] & FLAG_HOST) > 0) & (d['E_H'][r_] < E_)
    taken_l = taken_l[idx]

    gid, inv = np.unique(b[keep], return_inverse=True)
    sums = lambda w: np.bincount(inv, weights=w, minlength=gid.size)
    tot, tin, tin_taken = sums(m), sums(m * inside), sums(m * (inside & taken))
    tin_taken_l = sums(m * (inside & taken_l))
    return {int(g): (float(tot[k]), float(tin[k]), float(tin_taken[k]), float(tin_taken_l[k]))
            for k, g in enumerate(gid)}


def collect(rr_out, particle_dir=None, first=1, last=177, workers=6, verbose=True, sm=None, progress=None):
    """sats {gid: dict(snap, owned, n, bound, stolen, fret)} (owner view), sats_list (the
    same, infall-list view), born, plus snapshot times and the tree.
    sm: an existing StarMaps of the run to reuse (its ownership replay);
    progress: optional wrapper of the snapshot loop, e.g. tqdm."""
    import h5py
    from .starmap import StarMaps
    T0 = time.time()
    sm = sm or StarMaps(rr_out, particle_dir)
    sm._catalogue()
    sm.ownership._ready(last)                       # one replay, shared by the forked workers
    with h5py.File(os.path.join(rr_out, 'catalogue.hdf5'), 'r') as f:
        tree = f['header/merger_tree'][()]
    host = sm._host
    h = tree[tree['Sub_tree_id'] == host]
    t_of = {int(r['Snapshot']): float(r['Time']) for r in h}
    snaps = [s for s in range(first, last + 1) if s in t_of]
    _G.update(sm=sm)
    res = []
    wrap = progress or (lambda it, **kw: it)
    if workers > 1:
        import multiprocessing as mp
        with mp.get_context('fork').Pool(workers) as pool:
            for r in wrap(pool.imap(_snapshot, snaps), total=len(snaps), desc='satellites', unit='snap'):
                res.append(r)
                if verbose:
                    print(f"  snapshot {r['snapshot']}: {r['gid'].size} galaxies with owned stars "
                          f"({r['seconds']} s)", flush=True)
    else:
        res = [_snapshot(s) for s in wrap(snaps, desc='satellites', unit='snap')]
    born = {}
    for r in res:
        for g, (mt, mi, mx, mxl) in r['born'].items():
            e = born.setdefault(g, dict(snap=[], m=[], m_in=[], m_in_taken=[], m_in_taken_list=[]))
            e['snap'].append(r['snapshot'])
            e['m'].append(mt)
            e['m_in'].append(mi)
            e['m_in_taken'].append(mx)
            e['m_in_taken_list'].append(mxl)
    for e in born.values():
        for key in list(e):
            e[key] = np.asarray(e[key])
        e['time'] = np.array([t_of[s] for s in e['snap']])
    sats = {}
    for r in res:
        for k, g in enumerate(r['gid'].tolist()):
            e = sats.setdefault(g, dict(snap=[], owned=[], n=[], bound=[], stolen=[]))
            e['snap'].append(r['snapshot'])
            for key in ('owned', 'n', 'bound', 'stolen'):
                e[key].append(float(r[key][k]))
    sats_list = {}
    for r in res:
        L = r['lists']
        for k, g in enumerate(L['gid'].tolist()):
            e = sats_list.setdefault(g, dict(snap=[], owned=[], n=[], bound=[], stolen=[]))
            e['snap'].append(r['snapshot'])
            for key in ('owned', 'n', 'bound', 'stolen'):
                e[key].append(float(L[key][k]))
    for e in list(sats.values()) + list(sats_list.values()):
        for key in list(e):
            e[key] = np.asarray(e[key])
        with np.errstate(invalid='ignore', divide='ignore'):
            e['fret'] = np.where(e['bound'] > 0, 1.0 - e['stolen'] / e['bound'], np.nan)
        e['time'] = np.array([t_of[s] for s in e['snap']])
    return dict(sats=sats, sats_list=sats_list, born=born, snap_list=np.array(snaps),
                snap_time=np.array([t_of[s] for s in snaps]), tree=tree, host=host,
                seconds=round(time.time() - T0, 1))
