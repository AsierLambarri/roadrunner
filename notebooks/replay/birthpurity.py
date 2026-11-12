"""Birth-set purity (Comp02) and completeness (Comp03): is each star, when it is born (or a lag later), where
its birth galaxy is? Computed in memory from the roadrunner output (runs where
it lives: jotun); replay.simdata.birth_purity memoises it on the Run.

For every star with a birth galaxy g (final birth map), at its evaluation
snapshot (its first snapshot, or with lag > 0 the first snapshot at or after
t_birth + lag, exact birth times from particle_data/birth_times.hdf5):

  every galaxy h with >= n_min member stars at that snapshot is fitted with a
  6D normal in (position, velocity) (host-disc frame, replay.starmap): the
  mass-weighted mean and covariance of its members, i.e. the stars of its
  infall list that it binds (replay.ownership.in_list; the stars first
  appearing at that snapshot are left out of the fits). The host gets two
  components, inside and outside host_split kpc, mixed by their mass.
  (fit='refit', the default.)

  fit='rr' instead takes roadrunner's own fitted normals, stored per galaxy
  in the assignment file (mean, covariance, count), transformed into the
  same host-centred, host-disc frame; one component per galaxy, the host
  included, host_split ignored. These are not an independent check: they
  are fitted over the galaxy's own candidate stars, newborns included, and
  are what roadrunner itself uses to decide responsibilities and births.

  ln L_h(x) = -1/2 [(x - mu_h)^T Sigma_h^-1 (x - mu_h) + ln det Sigma_h + 6 ln 2 pi]
  Lambda    = ln L_g(x) - max_{h != g} ln L_h(x)    (ratio of densities: units
              and rotations cancel)
  chi2_g    = (x - mu_g)^T Sigma_g^-1 (x - mu_g)    (host: its nearer component)

A star is contaminated when Lambda < -ln K and chi2_g is beyond g's level
contour (both applied later, by flags(), so changing them costs nothing).

Completeness, the other way round: the likeliest galaxy of each star whatever
its labels (top), by how much it beats the next one (lam_top = ln L_top - max
over h != top of ln L_h) and chi2_top. The star belongs to top when lam_top >
ln K and it lies within top's level contour (members()); a galaxy's birth set
is complete when every star that belongs to it carries its label.
Unscored: g has no fit at that snapshot, or the star is not in the star table
there (beyond r_keep), or its evaluation snapshot is past the last one.
"""
import os
import time

import numpy as np

from . import particles as PT

FLAG_HOST = 1
LN2PI6 = 6 * np.log(2 * np.pi)
FIT_DEFS = ('refit', 'rr')
_G = {}


def _finish(mu, C, ridge=1e-6):
    """(mean, inverse covariance, ln det), ridging C (keeps small or flat groups invertible)."""
    C = C + ridge * np.trace(C) / 6 * np.eye(6)
    sign, logdet = np.linalg.slogdet(C)
    return mu, np.linalg.inv(C), logdet


def _gauss(x, w, ridge=1e-6):
    """(mean, inverse covariance, ln det) of points x (n, 6), weights w."""
    mu = np.average(x, axis=0, weights=w)
    d = x - mu
    C = (d * w[:, None]).T @ d / w.sum()
    return _finish(mu, C, ridge)


def _fits(d, s, own, host, n_min, host_split, exclude):
    """{galaxy: [(ln weight, mean, inverse covariance, ln det), ...]} of snapshot s."""
    X = np.hstack([d['pos'], d['vel']]).astype(np.float64)
    m = d['mass'].astype(np.float64)
    fits = {}

    mem = d['mem']                                     # (star, galaxy) pairs: in its list and bound to it
    ok = ~exclude[mem['row']]
    rows, gids = mem['row'][ok], mem['gid'][ok]
    o = np.argsort(gids, kind='stable')
    rows, gids = rows[o], gids[o]
    starts = np.flatnonzero(np.r_[True, gids[1:] != gids[:-1]]) if gids.size else np.empty(0, int)
    ends = np.r_[starts[1:], gids.size]
    for a, b in zip(starts, ends):
        if b - a >= n_min:
            r = rows[a:b]
            fits[int(gids[a])] = [(0.0, *_gauss(X[r], m[r]))]

    hb = np.flatnonzero(((d['flags'] & FLAG_HOST) > 0) & ~exclude)
    hb = hb[own.in_list(d['ids'][hb], np.full(hb.size, host, np.int64), s)]
    if hb.size >= n_min:
        parts = [hb]
        if host_split:
            inner = np.linalg.norm(d['pos'][hb], axis=1) < host_split
            parts = [p for p in (hb[inner], hb[~inner]) if p.size >= n_min] or [hb]
        mt = sum(m[p].sum() for p in parts)
        fits[int(host)] = [(np.log(m[p].sum() / mt), *_gauss(X[p], m[p])) for p in parts]
    return fits


def _fits_rr(d, s, rr_out, tree, n_min):
    """{galaxy: [(ln weight, mean, inverse covariance, ln det)]} of snapshot s, from
    roadrunner's own fitted normals (assignment/snapshotNNNN.hdf5, mean/covariance in
    raw particle coordinates), transformed into the star table's frame (host-centred,
    host-disc, physical kpc / km/s) with the same A, c as replay.starmap._compute: A =
    blockdiag(a * basis, basis), c = concat(host position, host velocity). One
    component per galaxy (the host included, no split); galaxies with < n_min
    effective particles (count) are skipped."""
    import h5py
    hrow = tree[(tree['Snapshot'] == s) & (tree['Sub_tree_id'] == d['host'])][0]
    a = 1.0 / (1.0 + float(hrow['Redshift']))
    Bm = d['basis']
    A = np.zeros((6, 6))
    A[:3, :3], A[3:, 3:] = a * Bm, Bm
    c = np.concatenate([PT.xyz(hrow), PT.vxyz(hrow)])
    fits = {}
    with h5py.File(os.path.join(rr_out, 'assignment', f'snapshot{s:04d}.hdf5'), 'r') as fa:
        for key in fa['galaxies']:
            g = fa['galaxies'][key]
            if 'mean' not in g or 'covariance' not in g:
                continue
            if float(np.asarray(g['count'][()]).reshape(-1)[0]) < n_min:
                continue
            mu_raw = g['mean'][()].astype(np.float64)
            Sigma_raw = g['covariance'][()].astype(np.float64)
            mu = A @ (mu_raw - c)
            Sigma = A @ Sigma_raw @ A.T
            fits[int(key)] = [(0.0, *_finish(mu, Sigma))]
    return fits


def _lnL(Y, comps):
    """ln L of points Y (n, 6) under a mixture, and the smallest chi2 over its components."""
    terms, chi2 = [], []
    for lw, mu, inv, logdet in comps:
        d = Y - mu
        q = np.einsum('ij,jk,ik->i', d, inv, d)
        terms.append(lw - 0.5 * (q + logdet + LN2PI6))
        chi2.append(q)
    T = np.array(terms)
    top = T.max(axis=0)
    return top + np.log(np.exp(T - top).sum(axis=0)), np.min(np.array(chi2), axis=0)


def _snapshot(s):
    t0 = time.time()
    sm, own = _G['sm'], _G['sm'].ownership
    ev = _G['eval'].get(s)
    out = dict(snapshot=s, pid=np.empty(0, np.int64), lam=np.empty(0), chi2=np.empty(0),
               best=np.empty(0, np.int64), lam_greedy=np.empty(0), chi2_greedy=np.empty(0),
               best_greedy=np.empty(0, np.int64), top=np.empty(0, np.int64), lam_top=np.empty(0),
               chi2_top=np.empty(0), n_fits=0)
    if ev is None or ev.size == 0:
        return out
    d = sm.at(s, cache=False)
    ids = d['ids']
    if _G['fit'] == 'rr':
        tree, _host = sm._catalogue()
        fits = _fits_rr(d, s, sm.rr_out, tree, _G['n_min'])
    else:
        first = np.full(ids.size, -1)
        known = ids < own._first.size
        first[known] = own._first[ids[known]]
        fits = _fits(d, s, own, int(d['host']), _G['n_min'], _G['host_split'], first == s)

    o = np.argsort(ids)
    k = np.clip(np.searchsorted(ids[o], ev), 0, max(ids.size - 1, 0))
    found = ids[o][k] == ev
    rows, pid = o[k][found], ev[found]
    gal = np.array(sorted(fits), np.int64)
    g = own.birth_of(pid)                              # roadrunner's birth galaxy
    gp, gl = _G['greedy']                              # greedy's (pid sorted, label)
    j = np.clip(np.searchsorted(gp, pid), 0, max(gp.size - 1, 0))
    gr = np.where((gp.size > 0) & (gp[j] == pid), gl[j], -1) if gp.size else np.full(pid.size, -1)
    have = np.isin(g, gal) | np.isin(gr, gal)
    rows, pid, g, gr = rows[have], pid[have], g[have], gr[have]
    if pid.size == 0:
        return out

    Y = np.hstack([d['pos'][rows], d['vel'][rows]]).astype(np.float64)
    L = np.empty((gal.size, pid.size))
    C = np.empty((gal.size, pid.size))
    for j, h in enumerate(gal):
        L[j], C[j] = _lnL(Y, fits[int(h)])
    i = np.arange(pid.size)

    def score(label):
        # Lambda, chi2 and the likeliest other galaxy for this label (NaN / -1 where it has no fit)
        ok = np.isin(label, gal)
        col = np.searchsorted(gal, np.where(ok, label, gal[0]))
        lnL_g, chi2_g = L[col, i], C[col, i]
        Lm = L.copy()
        Lm[col, i] = -np.inf
        jb = np.argmax(Lm, axis=0)
        return (np.where(ok, lnL_g - Lm[jb, i], np.nan), np.where(ok, chi2_g, np.nan),
                np.where(ok, gal[jb], -1).astype(np.int64))

    lam, chi2, best = score(g)
    lam_gr, chi2_gr, best_gr = score(gr)

    # the likeliest galaxy whatever the labels, and by how much it beats the next one (completeness)
    j1 = np.argmax(L, axis=0)
    Lm = L.copy()
    Lm[j1, i] = -np.inf
    lam_top = L[j1, i] - Lm.max(axis=0)                 # +inf with a single fitted galaxy
    out.update(pid=pid.astype(np.int64), lam=lam, chi2=chi2, best=best, lam_greedy=lam_gr,
               chi2_greedy=chi2_gr, best_greedy=best_gr, top=gal[j1].astype(np.int64), lam_top=lam_top,
               chi2_top=C[j1, i], n_fits=len(fits), seconds=round(time.time() - t0, 1))
    return out


def evaluation_snapshots(own, pid, birth_time, snap_list, snap_time, lag_myr):
    """Snapshot at which each star is judged: its first snapshot (lag 0), or the first
    snapshot at or after max(that, t_birth + lag). -1: none (past the last snapshot)."""
    first = np.full(pid.size, -1)
    known = pid < own._first.size
    first[known] = own._first[pid[known]]
    if lag_myr <= 0:
        return np.where(np.isin(first, snap_list), first, -1)
    t_of = dict(zip(snap_list.tolist(), snap_time.tolist()))
    t_first = np.array([t_of.get(int(f), np.inf) for f in first])
    t_need = np.maximum(t_first, birth_time + lag_myr / 1e3)
    k = np.searchsorted(snap_time, t_need - 1e-9)
    return np.where(k < snap_list.size, snap_list[np.minimum(k, snap_list.size - 1)], -1)


def collect(R, lag_myr=0.0, n_min=20, host_split=15.0, fit='refit', greedy=None, progress=None):
    """Lambda, chi2_g and the best other galaxy of every star with a birth galaxy, judged at
    its first snapshot + lag. dict(pid, g, birth_time, m0, s_eval, lam, chi2, best) over the
    birth table (NaN / -1 where unscored), plus the settings. fit: 'refit' (default) or 'rr'
    (roadrunner's own normals; host_split then ignored), see the module docstring.
    greedy: (pid sorted, label) of greedy's birth galaxies (simdata.newborn_fates); the same
    test is then also run with those labels: g_greedy, lam_greedy, chi2_greedy, best_greedy
    (stars with a greedy birth galaxy but none in roadrunner's birth map are included).
    Also, whatever the labels: top (the likeliest galaxy), lam_top, chi2_top (completeness)."""
    if fit not in FIT_DEFS:
        raise ValueError(f"fit must be one of {FIT_DEFS}, not {fit!r}")
    from . import simdata as SD
    R.load()
    own = R.sm.ownership
    B = SD.birth_table(R)
    pid = B['particle_index'].astype(np.int64)
    g = own.birth_of(pid)
    gp, gl = greedy if greedy is not None else (np.empty(0, np.int64), np.empty(0, np.int64))
    gp, gl = np.asarray(gp, np.int64), np.asarray(gl, np.int64)
    j = np.clip(np.searchsorted(gp, pid), 0, max(gp.size - 1, 0))
    g_gr = np.where(gp[j] == pid, gl[j], -1) if gp.size else np.full(pid.size, -1, np.int64)
    born = (g >= 0) | (g_gr >= 0)
    pid, g, g_gr = pid[born], g[born], g_gr[born]
    bt, m0 = B['birth_time'][born], B['initial_mass'][born]

    h = R.tree[R.tree['Sub_tree_id'] == R.host]
    sn = h['Snapshot'].astype(int)
    o = np.argsort(sn)
    snap_list = sn[o]
    snap_time = h['Time'].astype(float)[o]
    keep = (snap_list >= R.first) & (snap_list <= R.last)
    snap_list, snap_time = snap_list[keep], snap_time[keep]

    s_eval = evaluation_snapshots(own, pid, bt, snap_list, snap_time, lag_myr)
    by_snap = {}
    oo = np.argsort(s_eval, kind='stable')
    se = s_eval[oo]
    for s in np.unique(se[se >= 0]):
        a, b = np.searchsorted(se, s), np.searchsorted(se, s, side='right')
        by_snap[int(s)] = np.sort(pid[oo[a:b]])

    _G.update(sm=R.sm, eval=by_snap, n_min=int(n_min), host_split=host_split, fit=fit, greedy=(gp, gl))
    snaps = sorted(by_snap)
    wrap = progress or (lambda it, **kw: it)
    if R.workers > 1:
        import multiprocessing as mp
        with mp.get_context('fork').Pool(R.workers) as pool:
            res = list(wrap(pool.imap(_snapshot, snaps), total=len(snaps), desc='birth purity', unit='snap'))
    else:
        res = [_snapshot(s) for s in wrap(snaps, desc='birth purity', unit='snap')]

    cols = {k: np.full(pid.size, np.nan) for k in ('lam', 'chi2', 'lam_greedy', 'chi2_greedy', 'lam_top', 'chi2_top')}
    cols.update({k: np.full(pid.size, -1, np.int64) for k in ('best', 'best_greedy', 'top')})
    order = np.argsort(pid)
    for r in res:
        if r['pid'].size == 0:
            continue
        k = order[np.searchsorted(pid[order], r['pid'])]
        for key in cols:
            cols[key][k] = r[key]
    return dict(pid=pid, g=g, g_greedy=g_gr, birth_time=bt, m0=m0, s_eval=s_eval, **cols,
                lag_myr=float(lag_myr), n_min=int(n_min), host_split=host_split, fit=fit)


def flags(P, K=100.0, level=0.999, label='rr'):
    """(scored, contaminated) per star of collect()'s output, for roadrunner's birth galaxy
    (label 'rr') or greedy's ('greedy'): contaminated = more likely in another galaxy by a
    factor > K and beyond that galaxy's `level` contour (chi2 with 6 dof)."""
    from scipy.stats import chi2 as chi2_dist
    sfx = {'rr': '', 'greedy': '_greedy'}[label]
    scored = np.isfinite(P['lam' + sfx])
    bad = scored & (P['lam' + sfx] < -np.log(K)) & (P['chi2' + sfx] > chi2_dist.ppf(level, 6))
    return scored, bad


def members(P, K=100.0, level=0.999):
    """(scored, member) per star of collect()'s output: member = it belongs to its likeliest
    galaxy P['top'], more likely there than in any other by a factor > K and within its `level`
    contour (chi2 with 6 dof); the mirror of flags()."""
    from scipy.stats import chi2 as chi2_dist
    scored = np.isfinite(P['lam_top'])
    member = scored & (P['lam_top'] > np.log(K)) & (P['chi2_top'] <= chi2_dist.ppf(level, 6))
    return scored, member


def _series(gal, w, w_good, gids, birth_time, times, mode):
    """Per galaxy in gids: 1 - (sum of w - w_good) / (sum of w) over its stars (gal == g) born by
    each time ('total') or since the previous time ('differential'); NaN where sum w = 0."""
    out = {}
    for g in set(int(x) for x in gids):
        k = (gal == g) & (w > 0)
        if not k.any():
            continue
        o = np.argsort(birth_time[k])
        bt = birth_time[k][o]
        cm, cg = np.r_[0.0, np.cumsum(w[k][o])], np.r_[0.0, np.cumsum(w_good[k][o])]
        j1 = np.searchsorted(bt, times, side='right')
        j0 = np.r_[0, j1[:-1]] if mode == 'differential' else np.zeros_like(j1)
        m_in, g_in = cm[j1] - cm[j0], cg[j1] - cg[j0]
        with np.errstate(invalid='ignore', divide='ignore'):
            out[g] = dict(time=times, fret=np.where(m_in > 0, g_in / m_in, np.nan), step=mode == 'differential')
    return out


def reverse_series(P, gids, times, mode='total'):
    """Of the stars greedy gives to each galaxy at birth, the share roadrunner also gives to it
    (initial masses), at `times` ('total' or 'differential', as purity_series)."""
    if mode not in ('total', 'differential'):
        raise ValueError(f"mode must be 'total' or 'differential', not {mode!r}")
    m = P['m0'].astype(np.float64)
    agree = (P['g'] == P['g_greedy']).astype(np.float64)
    return _series(P['g_greedy'], m, m * agree, gids, P['birth_time'], np.asarray(times, float), mode)


def purity_series(P, gids, times, K=100.0, level=0.999, mode='total', label='rr'):
    """Birth-set purity of each galaxy at `times` (Gyr, the snapshots), initial-mass weighted,
    of roadrunner's birth sets (label 'rr') or greedy's ('greedy'): mode 'total': 1 -
    M0(contaminated) / M0(scored) over its stars born by then (cumulative); 'differential': the
    same over the stars born since the previous time only (the new stars; NaN where none is
    scored). {g: dict(time, fret, step)} (the 'fret' key so the orbit drawing helpers colour by
    it; step: the value holds over (previous time, time], no interpolation); NaN before its
    first scored star."""
    if mode not in ('total', 'differential'):
        raise ValueError(f"mode must be 'total' or 'differential', not {mode!r}")
    scored, bad = flags(P, K, level, label)
    m = P['m0'].astype(np.float64)
    gal = P['g'] if label == 'rr' else P['g_greedy']
    return _series(gal, m * scored, m * (scored & ~bad), gids, P['birth_time'], np.asarray(times, float), mode)


def forward_series(P, gids, times, mode='total'):
    """Of the stars roadrunner gives to each galaxy at birth, the share greedy also gives to it
    (initial masses), at `times` ('total' or 'differential'): reverse_series read the other way."""
    if mode not in ('total', 'differential'):
        raise ValueError(f"mode must be 'total' or 'differential', not {mode!r}")
    m = P['m0'].astype(np.float64)
    agree = (P['g'] == P['g_greedy']).astype(np.float64)
    return _series(P['g'], m, m * agree, gids, P['birth_time'], np.asarray(times, float), mode)


def completeness_series(P, gids, times, K=100.0, level=0.999, mode='total', label='rr'):
    """Birth-set completeness of each galaxy at `times`, initial-mass weighted: of the stars that
    belong to it at birth (members()), the share roadrunner's (label 'rr') or greedy's ('greedy')
    birth galaxy gives to it; 'total' (born by then) or 'differential' (born since the previous
    time; NaN where none belongs). {g: dict(time, fret, step)} as purity_series."""
    if mode not in ('total', 'differential'):
        raise ValueError(f"mode must be 'total' or 'differential', not {mode!r}")
    _, member = members(P, K, level)
    m = P['m0'].astype(np.float64)
    lab = P['g'] if label == 'rr' else P['g_greedy']
    gal = np.where(member, P['top'], -1)
    return _series(gal, m, m * (lab == P['top']), gids, P['birth_time'], np.asarray(times, float), mode)
