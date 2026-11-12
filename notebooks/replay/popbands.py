"""Radial profiles of the whole host-bound population, one per snapshot
(plot 3.1, plots/sim/s031_population.py), computed in memory from the
roadrunner output (runs where it lives).

Per snapshot, the stars of replay.starmap (E_H, owner energy E_O, flags,
host-centred positions) are split as in s01/s02 (replay/simmap.py, --es
owner): contested stars, bound to the host and to an owner other than the
host, carry q = 1{E_O <= E_H} (fret) and q = log10(E_H/E_O) (logx); the
host-bound rest (owner the host, or an owner that does not bind them) is
the host-only layer, whose value is exact (fret 1, logx 0).

Smoothing as in s01 --smooth: every host-bound star (both layers) is a 3D
Gaussian of width max(kernel_frac x distance to its ngb-th nearest star,
sigma_floor R_vir), the neighbours (and so the smoothing lengths) always
from all stars; each Gaussian is integrated exactly over spherical shells
about the host centre (closed-form radial CDF), over the shells within
+-5 sigma of the star. Per shell and snapshot, for each layer, the
kernel-weighted sums of the weights w and w x q, for w = m (mass-weighted)
and w = rho (density-weighted: rho = mass of the ngb nearest stars over the
sphere of radius the ngb-th distance), and the kernel-weighted star count.
The plot combines them jointly (all stars averaged together) or layered
(contested stars on their own over the host-only layer).
"""
import os
import time

import numpy as np

FLAG_HOST, FLAG_OWNER_BOUND = 1, 8
# contested columns: m, m q_fret, m q_logx, count, rho, rho q_fret, rho q_logx; host-only: m, count, rho
C_COLS = ('m', 'mf', 'ml', 'n', 'rho', 'rhof', 'rhol')
B_COLS = ('m', 'n', 'rho')
_G = {}


def shell_cdf(r, a, s):
    """P(|a e + s g| < r), g a 3D standard normal (fraction of a Gaussian of
    width s centred at distance a within radius r)."""
    from scipy.special import ndtr
    a = np.maximum(a, 1e-3 * s)
    u, v = (r - a) / s, (r + a) / s
    phi = lambda x: np.exp(-0.5 * x * x) * 0.3989422804014327
    return ndtr(u) + ndtr(v) - 1.0 - (s / a) * (phi(u) - phi(v))


def deposit_shells(a, sig, W, edges, nsig=5.0, budget=4_000_000):
    """sum_i W[i, k] x (fraction of star i's 3D Gaussian in each shell), shape
    (nshell, K); each star only over the shells within +-nsig sigma."""
    nr = edges.size - 1
    dr, r0 = edges[1] - edges[0], edges[0]
    out = np.zeros((nr, W.shape[1]))
    lo = np.clip(np.floor((a - nsig * sig - r0) / dr).astype(int), 0, nr)
    hi = np.clip(np.ceil((a + nsig * sig - r0) / dr).astype(int), 0, nr)
    width = hi - lo
    for u in np.unique(width[width > 0]):
        grp = np.flatnonzero(width == u)
        for ch in np.array_split(grp, max(1, grp.size * (u + 1) // budget + 1)):
            j = lo[ch, None] + np.arange(u + 1)
            dF = np.diff(shell_cdf(edges[j], a[ch, None], sig[ch, None]), axis=1)      # (c, u)
            shell = (lo[ch, None] + np.arange(u)).ravel()
            for k in range(W.shape[1]):
                out[:, k] += np.bincount(shell, weights=(dF * W[ch, k, None]).ravel(), minlength=nr)
    return out


def _profile(s):
    from scipy.spatial import cKDTree
    t0 = time.time()
    sm, edges, cfg = _G['sm'], _G['edges'], _G['cfg']
    d = sm.at(s, cache=False)
    rv = float(d['rvir_kpc'])
    pos = d['pos']
    r = np.linalg.norm(pos, axis=1) / rv
    fl, m = d['flags'], d['mass'].astype(np.float64)
    host_bound = (fl & FLAG_HOST) > 0
    contested = host_bound & ((fl & FLAG_OWNER_BOUND) > 0)
    # smoothing lengths and densities from all stars, for the host-bound stars that reach the shells
    hb = np.flatnonzero(host_bound & (r <= 1.3))
    tree = cKDTree(pos)
    k = min(cfg['ngb'], r.size)
    h = np.empty(hb.size)
    rho = np.empty(hb.size)
    for ch in np.array_split(np.arange(hb.size), max(1, hb.size // 300000)):
        dist, ind = tree.query(pos[hb[ch]], k=k)
        hk = dist[:, -1] / rv
        h[ch] = hk
        rho[ch] = m[ind].sum(axis=1) / (4.0 / 3.0 * np.pi * np.maximum(hk, 1e-6) ** 3)
    sig = np.maximum(cfg['kernel_frac'] * h, cfg['sigma_floor'])
    a, mm = r[hb], m[hb]
    is_c = contested[hb]
    ci, bi = np.flatnonzero(is_c), np.flatnonzero(~is_c)
    E_H, E_O = d['E_H'][hb[ci]].astype(np.float64), d['E_O'][hb[ci]].astype(np.float64)
    q_f = (E_O <= E_H).astype(np.float64)
    q_l = np.log10(E_H / E_O)
    one = np.ones(ci.size)
    Wc = np.stack([mm[ci], mm[ci] * q_f, mm[ci] * q_l, one, rho[ci], rho[ci] * q_f, rho[ci] * q_l], 1)
    Wb = np.stack([mm[bi], np.ones(bi.size), rho[bi]], 1)
    C = deposit_shells(a[ci], sig[ci], Wc, edges) if ci.size else np.zeros((edges.size - 1, len(C_COLS)))
    B = deposit_shells(a[bi], sig[bi], Wb, edges) if bi.size else np.zeros((edges.size - 1, len(B_COLS)))
    # newborn mass per birth galaxy within R_vir (for the orbit ranking)
    own = sm.ownership
    ids = d['ids']
    nbm = (ids < own._first.size)
    nbm[nbm] = own._first[ids[nbm]] == s
    nbm &= r <= 1
    born = {}
    if nbm.any():
        b = own.birth_of(ids[nbm])
        ok = (b >= 0) & (b != d['host'])
        g, inv = np.unique(b[ok], return_inverse=True)
        born = dict(zip(g.tolist(), np.bincount(inv, weights=m[nbm][ok]).tolist()))
    inside = contested & (r <= 1)
    return dict(snapshot=s, rvir=rv, born=born, z=float(d['redshift']), C=C, B=B,
                n_host_bound=int(np.sum(host_bound & (r <= 1))), n_contested=int(inside.sum()),
                contested_mass=float(m[inside].sum()),
                retained_mass=float(m[inside][d['E_O'][inside] <= d['E_H'][inside]].sum()),
                seconds=round(time.time() - t0, 1))


def tracks(tree, host, gals, dt=0.005):
    """Spline-interpolated r/R_vir(t) of galaxies gals: list of (t, r) arrays,
    one per run of consecutive snapshots (NaN-separated when concatenated)."""
    from scipy.interpolate import CubicSpline
    h = tree[tree['Sub_tree_id'] == host]
    hs = {int(r['Snapshot']): r for r in h}
    out = []
    for g in gals:
        rows = tree[tree['Sub_tree_id'] == g]
        rows = rows[np.argsort(rows['Snapshot'])]
        rows = rows[[int(s) in hs for s in rows['Snapshot']]]
        if rows.size < 2:
            continue
        sn = rows['Snapshot'].astype(int)
        cuts = np.flatnonzero(np.diff(sn) != 1) + 1
        T, R = [], []
        for seg in np.split(np.arange(rows.size), cuts):
            if seg.size < 2:
                continue
            rr = rows[seg]
            hh = np.array([hs[int(s)] for s in rr['Snapshot']])
            t = rr['Time'].astype(float)
            o = np.argsort(t)
            t, keep = t[o], np.r_[True, np.diff(t[o]) > 0]
            t = t[keep]
            if t.size < 2:
                continue
            dx = np.stack([(rr[c] - hh[c]).astype(float)[o][keep] for c in ('position_x', 'position_y', 'position_z')], 1)
            rv = hh['virial_radius'].astype(float)[o][keep]
            tt = np.arange(t[0], t[-1] + 1e-9, dt)
            pos = CubicSpline(t, dx, axis=0)(tt)
            T.append(np.r_[tt, np.nan])
            R.append(np.r_[np.linalg.norm(pos, axis=1) / CubicSpline(t, rv)(tt), np.nan])
        if T:
            out.append((g, np.concatenate(T), np.concatenate(R)))
    return out


def collect(rr_out, particle_dir=None, first=1, last=177, workers=6, nr=200, rmin=0.0, ngb=32,
            kernel_frac=0.5, sigma_floor=0.01, verbose=True, n_tracks=8, track_dt=0.005, sm=None,
            progress=None):
    """Shell sums for snapshots first..last (dict of arrays, snapshot-major).
    sm: an existing StarMaps of the run to reuse (its ownership replay);
    progress: optional wrapper of the snapshot loop, e.g. tqdm."""
    from .starmap import StarMaps
    T0 = time.time()
    sm = sm or StarMaps(rr_out, particle_dir)
    sm._catalogue()
    sm.ownership._ready(last)              # one replay, shared by the forked workers
    import h5py
    with h5py.File(os.path.join(rr_out, 'catalogue.hdf5'), 'r') as f:
        tree = f['header/merger_tree'][()]
    host = sm._host
    h = tree[tree['Sub_tree_id'] == host]
    t_of = {int(r['Snapshot']): float(r['Time']) for r in h}
    edges = np.linspace(rmin, 1.0, nr + 1)
    _G.update(sm=sm, edges=edges, cfg=dict(ngb=ngb, kernel_frac=kernel_frac, sigma_floor=sigma_floor))
    snaps = [s for s in range(first, last + 1) if s in t_of]
    res = []
    wrap = progress or (lambda it, **kw: it)
    if workers > 1:
        import multiprocessing as mp
        with mp.get_context('fork').Pool(workers) as pool:
            for r in wrap(pool.imap(_profile, snaps), total=len(snaps), desc='population shells', unit='snap'):
                res.append(r)
                if verbose:
                    print(f"  snapshot {r['snapshot']}: {r['n_contested']} contested of {r['n_host_bound']} "
                          f"host-bound within R_vir ({r['seconds']} s)", flush=True)
    else:
        for s in wrap(snaps, desc='population shells', unit='snap'):
            res.append(_profile(s))
    stack = lambda k: np.stack([r[k] for r in res])
    out = dict(snap_list=np.array(snaps), snap_time=np.array([t_of[s] for s in snaps]),
               snap_z=stack('z'), snap_rvir=stack('rvir'), edges=edges,
               C=stack('C'), B=stack('B'))                    # (snapshot, shell, column): C_COLS, B_COLS
    born = {}
    for r in res:
        for g, mass in r['born'].items():
            born[g] = born.get(g, 0.0) + mass
    ranked = [g for g, _ in sorted(born.items(), key=lambda kv: -kv[1])[:n_tracks]]
    out['tracks'] = tracks(tree, host, ranked, track_dt)
    out['track_mass'] = [born[g] for g in ranked]
    cm, rm_ = sum(r['contested_mass'] for r in res), sum(r['retained_mass'] for r in res)
    out['checks'] = dict(seconds=round(time.time() - T0, 1), snapshots=len(res),
                         host_bound_star_snapshots=int(sum(r['n_host_bound'] for r in res)),
                         contested_star_snapshots=int(sum(r['n_contested'] for r in res)),
                         contested_retained_mass_frac=round(rm_ / cm, 5) if cm > 0 else None,
                         tracks=[int(g) for g in ranked],
                         ownership=sm.ownership.info)
    return out
