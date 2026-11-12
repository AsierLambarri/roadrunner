"""Newborn stars of a roadrunner run: energy label vs the pipeline's birth
host, computed in memory (nothing is written). Used by
plots/sim/s03_time_radius.py; runs where the roadrunner output lives.

A star is newborn at snapshot s if its particle id is absent from snapshot
s-1 (its first appearance, as roadrunner's birth tracker counts it; snapshot
0 has no previous one). For every newborn, at s, from the stored energies
(assignment/snapshotNNNN.hdf5, E = -boundness * energy_scale where
boundness_valid; recomputed from particle_data, with a warning, where the
scale was stored as inf: replay.energies):
  E_H       energy relative to the central host (NaN if not bound/stored)
  E_S       lowest stored energy relative to any other halo (NaN if none)
  best_id   that halo's Sub_tree_id (-1 if none)
  flags     bit 0: bound to the host; bit 1: bound to another halo
  n_bound   number of halos (host included) that bind it; uncontested = 1
  owner_id  where it should be at s (replay.ownership: the final birth map
            plus any accretion up the satellite hierarchy in the same
            snapshot; -1 none)
  E_O       energy relative to the owner (= E_H if the owner is the host;
            NaN if the owner does not bind it)
  birth_id  the pipeline's final birth host (catalogue final/births; -1 born
            unbound, -2 missing from the table)
  r_rvir    3D distance from the host centre over the host R_vir(s) (BN98,
            tree virial_radius, physical)
  h_all, h_new  (only with ngb > 0) distance to the ngb-th nearest star among
            all stars / the snapshot's newborns, over R_vir (smoothing lengths)
plus mass and the snapshot; per snapshot cosmic time, z and R_vir.
Retained (as f_retention in s01/s02, owner definition): not taken by the
host, i.e. unless the star is bound to the host and to an owner other than
the host and E_H < E_O. Uncontested newborns (bound to one halo), and those
the host owns, are retained by construction. retained_label: the earlier
definition (energy label, the halo with the lowest stored energy, equals
birth_id), kept for comparison.
"""
import json
import os
import time

import numpy as np

from . import energies as EN
from . import particles as PT

FLAG_HOST, FLAG_OTHER, FLAG_OWNER_HOST, FLAG_OWNER_BOUND = 1, 2, 4, 8
_G = {}     # per-worker globals (set by _init)


def _init(rr_out, pd_dir, host, tree, b_pid, b_id, ngb=0, owners=None):
    _G.update(rr_out=rr_out, pd_dir=pd_dir, host=host, tree=tree, b_pid=b_pid, b_id=b_id, ngb=ngb,
              owners=owners or {})


def _ids(s):
    import h5py
    with h5py.File(os.path.join(_G['pd_dir'], f'snapshot{s:04d}.hdf5'), 'r') as f:
        return f['data/indices'][()].astype(np.int64)


def energy_label(E_H, E_S, best, flags, host):
    """Halo whose stored energy is lowest: host if E_H < E_S, else best_id;
    a star bound to one halo only gets that halo; unbound -1."""
    bh = (flags & FLAG_HOST) > 0
    bo = (flags & FLAG_OTHER) > 0
    lab = np.full(flags.size, -1, dtype=np.int64)
    lab[bh & ~bo] = host
    lab[~bh & bo] = best[~bh & bo]
    both = bh & bo
    lab[both] = np.where(E_H[both] < E_S[both], host, best[both])
    return lab


def do_snapshot(s):
    import h5py
    t0 = time.time()
    host, tree = _G['host'], _G['tree']
    trs = tree[tree['Snapshot'] == s]
    hrow = trs[trs['Sub_tree_id'] == host]
    if hrow.size == 0:
        return dict(snapshot=s, skipped='no host row')
    hrow = hrow[0]
    z = float(hrow['Redshift'])
    a = 1.0 / (1.0 + z)
    rvir = float(hrow['virial_radius']) * a
    ids, Xs, Vs, mean, scale, masses, _ = PT.load_particles(os.path.join(_G['pd_dir'], f'snapshot{s:04d}.hdf5'))
    rows = np.flatnonzero(~np.isin(ids, _ids(s - 1)))           # newborn rows of particle_data
    n = rows.size
    slot = np.full(ids.size, -1, dtype=np.int64)
    slot[rows] = np.arange(n)
    pos = (Xs[rows].astype(np.float64) / scale[:3] + mean[:3] - PT.xyz(hrow)) * a
    r = np.linalg.norm(pos, axis=1) / rvir

    pid = ids[rows]
    owner = np.full(n, -1, np.int64)
    if s in _G['owners']:
        o_pid, o_gal = _G['owners'][s]
        k = np.clip(np.searchsorted(o_pid, pid), 0, max(o_pid.size - 1, 0))
        if o_pid.size:
            owner = np.where(o_pid[k] == pid, o_gal[k], -1)
    E_O = np.full(n, np.nan)
    E_H = np.full(n, np.nan)
    E_S = np.full(n, np.inf)
    best = np.full(n, -1, dtype=np.int64)
    nb = np.zeros(n, dtype=np.int64)
    # lost energies (energy_scale inf) are recomputed from this snapshot's particles, already in memory
    fallback = EN.Fallback(_G['rr_out'], _G['pd_dir'], _G['tree'], particles=lambda _s: (
        Xs.astype(np.float64) / scale[:3] + mean[:3], Vs.astype(np.float64) / scale[3:] + mean[3:], masses))
    with h5py.File(os.path.join(_G['rr_out'], 'assignment', f'snapshot{s:04d}.hdf5'), 'r') as fa:
        for k, (idx, E) in EN.stored_energies(fa, s, fallback).items():
            j = slot[idx]
            ok = j >= 0
            if not ok.any():
                continue
            E, j = E[ok], j[ok]
            nb[j] += 1                         # indices are unique within a group
            mine = owner[j] == k
            E_O[j[mine]] = E[mine]
            if k == host:
                E_H[j] = E
            else:
                better = E < E_S[j]
                E_S[j[better]] = E[better]
                best[j[better]] = k
    E_S[~np.isfinite(E_S)] = np.nan
    flags = (np.isfinite(E_H) & (E_H < 0)).astype(np.uint8) * FLAG_HOST
    flags |= np.isfinite(E_S).astype(np.uint8) * FLAG_OTHER
    owner_host = owner == host
    flags |= owner_host.astype(np.uint8) * FLAG_OWNER_HOST
    flags |= (~owner_host & np.isfinite(E_O)).astype(np.uint8) * FLAG_OWNER_BOUND
    E_O[owner_host] = E_H[owner_host]

    extra = {}
    if _G['ngb'] > 0:
        extra = dict(h_all=np.empty(0, np.float32), h_new=np.empty(0, np.float32))
    if _G['ngb'] > 0 and n > 0:
        from scipy.spatial import cKDTree
        k_all = min(_G['ngb'], ids.size)
        pos_all = (Xs.astype(np.float64) / scale[:3] + mean[:3] - PT.xyz(hrow)) * a
        extra['h_all'] = (cKDTree(pos_all).query(pos, k=k_all)[0][:, -1] / rvir).astype(np.float32)
        k_new = min(_G['ngb'], n)
        h_new = cKDTree(pos).query(pos, k=k_new)[0]
        extra['h_new'] = ((h_new[:, -1] if h_new.ndim == 2 else h_new) / rvir).astype(np.float32)
    k = np.clip(np.searchsorted(_G['b_pid'], pid), 0, _G['b_pid'].size - 1)
    birth = np.where(_G['b_pid'][k] == pid, _G['b_id'][k], -2)
    return dict(snapshot=s, time=float(hrow['Time']), z=z, rvir=rvir, r_rvir=r.astype(np.float32),
                mass=masses[rows].astype(np.float32), E_H=E_H.astype(np.float32), E_S=E_S.astype(np.float32),
                best_id=best.astype(np.int32), flags=flags,
                owner_id=owner.astype(np.int32), E_O=E_O.astype(np.float32), n_bound=np.minimum(nb, 255).astype(np.uint8), birth_id=birth.astype(np.int32),
                seconds=time.time() - t0, **extra)


def tracks(tree, host, ids_ranked):
    """(track_id, track_snap, track_r): galaxy distance over host R_vir per snapshot."""
    h = tree[tree['Sub_tree_id'] == host]
    hpos = {int(r['Snapshot']): (PT.xyz(r), float(r['virial_radius'])) for r in h}
    out_id, out_s, out_r = [], [], []
    for g in ids_ranked:
        for row in tree[tree['Sub_tree_id'] == g]:
            s = int(row['Snapshot'])
            if s in hpos:      # comoving distance / comoving R_vir
                out_id.append(g)
                out_s.append(s)
                out_r.append(np.linalg.norm(PT.xyz(row) - hpos[s][0]) / hpos[s][1])
    return np.array(out_id, np.int32), np.array(out_s, np.int16), np.array(out_r, np.float32)


def collect(rr_out, pd_dir=None, first=1, last=177, host=None, workers=1, n_tracks=40, verbose=True, ngb=0,
            own=None, progress=None):
    """All newborns of snapshots first..last as one dict of arrays (in
    memory), with 'label', 'retained', per-snapshot tables, the tracks of the
    n_tracks galaxies with the most newborn mass born in them inside the
    host R_vir, and 'checks' (bookkeeping counts). own: an existing
    Ownership of the run to reuse (its replay is shared); progress: optional
    wrapper of the snapshot loops, e.g. tqdm."""
    if first < 1:
        raise ValueError('first must be >= 1 (newborns need a previous snapshot)')
    import h5py
    with h5py.File(os.path.join(rr_out, 'catalogue.hdf5'), 'r') as f:
        tree = f['header/merger_tree'][()]
        births = f['final/births'][()]
        host = host if host is not None else int(f['header/accretion_id'][()])
    order = np.argsort(births['particle_index'])
    b_pid = births['particle_index'][order].astype(np.int64)
    b_id = births['birth_id'][order].astype(np.int64)
    pd_dir = pd_dir or os.path.join(rr_out, 'particle_data')
    snaps = list(range(first, last + 1))
    from .ownership import Ownership
    own = own or Ownership(rr_out)
    owners = {}
    wrap = progress or (lambda it, **kw: it)
    for s in wrap(snaps, desc='newborn owners', unit='snap'):    # owners of each snapshot's newborns
        nid = own.newborn_ids(s)
        owners[s] = (nid, own.owners(nid, s))
    init = (rr_out, pd_dir, host, tree, b_pid, b_id, ngb, owners)

    T0 = time.time()
    res = []
    if workers > 1:
        import multiprocessing as mp
        pool = mp.get_context('fork').Pool(workers, initializer=_init, initargs=init)
        it = pool.imap(do_snapshot, snaps)
    else:
        _init(*init)
        pool, it = None, map(do_snapshot, snaps)
    for r in wrap(it, total=len(snaps), desc='newborns', unit='snap'):
        res.append(r)
        if verbose:
            info = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()
                    if not isinstance(v, np.ndarray)}
            info['newborn'] = int(r['mass'].size) if 'mass' in r else 0
            print(json.dumps(info), flush=True)
    if pool is not None:
        pool.close()
        pool.join()
    res = [r for r in res if 'skipped' not in r]

    cat = lambda k: np.concatenate([r[k] for r in res])
    d = dict(snapshot=np.concatenate([np.full(r['mass'].size, r['snapshot'], np.int16) for r in res]),
             r_rvir=cat('r_rvir'), mass=cat('mass').astype(np.float64), E_H=cat('E_H').astype(np.float64),
             E_S=cat('E_S').astype(np.float64), best_id=cat('best_id').astype(np.int64), flags=cat('flags'),
             owner_id=cat('owner_id').astype(np.int64), E_O=cat('E_O').astype(np.float64),
             n_bound=cat('n_bound'), birth_id=cat('birth_id').astype(np.int64),
             snap_list=np.array([r['snapshot'] for r in res]),
             snap_time=np.array([r['time'] for r in res]), snap_z=np.array([r['z'] for r in res]),
             snap_rvir=np.array([r['rvir'] for r in res]), host=host)
    if ngb > 0:
        d['h_all'] = cat('h_all').astype(np.float64)
        d['h_new'] = cat('h_new').astype(np.float64)
    b = d['birth_id']
    d['label'] = energy_label(d['E_H'], d['E_S'], d['best_id'], d['flags'], host)
    d['retained_label'] = d['label'] == b
    fl = d['flags']
    contested = ((fl & FLAG_HOST) > 0) & ((fl & FLAG_OWNER_BOUND) > 0)     # host and owner (not host) bind it
    d['contested'] = contested
    d['retained'] = ~(contested & (d['E_H'] < d['E_O']))

    sat = (b >= 0) & (b != host) & (d['r_rvir'] <= 1)
    gid, inv = np.unique(b[sat], return_inverse=True)
    gm = np.bincount(inv, weights=d['mass'][sat])
    ranked = gid[np.argsort(gm)[::-1][:n_tracks]]
    d['track_id'], d['track_snap'], d['track_r'] = tracks(tree, host, ranked)

    fl, nbd, lab = d['flags'], d['n_bound'], d['label']
    uncont, contested = nbd == 1, nbd >= 2
    d['checks'] = dict(
        seconds=round(time.time() - T0, 1), snapshots=len(res), newborns=int(fl.size),
        births_table=int(b_pid.size), missing_from_births=int(np.sum(b == -2)),
        birth_unbound=int(np.sum(b == -1)), energy_unbound=int(np.sum(nbd == 0)),
        uncontested=int(uncont.sum()), uncontested_disagree=int(np.sum(uncont & (lab != b))),
        contested_host=int(np.sum(contested & ((fl & FLAG_HOST) > 0))),
        contested_host_disagree=int(np.sum(contested & ((fl & FLAG_HOST) > 0) & (lab != b))),
        contested_others_only=int(np.sum(contested & ((fl & 3) == FLAG_OTHER))),
        outside_rvir=int(np.sum(d['r_rvir'] > 1)),
        owner_is_birth=int(np.sum(d['owner_id'] == b)), owner_host_birth_other=int(np.sum((d['owner_id'] == host) & (b >= 0) & (b != host))),
        contested_owner=int(d['contested'].sum()), retained=int(d['retained'].sum()),
        sat_born_in_rvir=int(np.sum(satin := (b >= 0) & (b != host) & (d['r_rvir'] <= 1))),
        sat_born_owner_binds=int(np.sum(satin & ((fl & FLAG_OWNER_BOUND) > 0))),
        sat_born_owner_unbound_host_bound=int(np.sum(satin & ((fl & FLAG_OWNER_BOUND) == 0)
                                                     & ((fl & FLAG_OWNER_HOST) == 0) & ((fl & FLAG_HOST) > 0))),
        sat_born_label_third_halo=int(np.sum(satin & (d['label'] != b) & (d['label'] != host) & (d['label'] >= 0))),
        retained_label=int(d['retained_label'].sum()),
        retained_differs=int(np.sum(d['retained'] != d['retained_label'])), ownership=own.info)
    return d
