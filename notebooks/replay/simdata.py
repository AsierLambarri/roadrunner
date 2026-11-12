"""Loading and processing for the figure notebook (notebooks/sim_figures.ipynb):
the data of the simulation figures s01, s02, s03, s031 and s032, computed in
memory from the roadrunner output (runs where it lives: jotun) and kept on
one Run object, so the notebook's plotting cells only redraw. Nothing is
written to disk but the PNGs that show(..., save=True) saves.

    R = Run(workers=16)                  # nothing read yet
    R.load()                             # merger tree, host, ownership replay (shared by all)
    R.starmaps([68, 77, 87, 146])        # s01/s02 star tables (any other snapshot: on demand)
    D03, D031, D032 = R.newborns(), R.population(), R.satellites()

Per-satellite f_ret follows the run's definition, R.fret (see Run.satellites):
'list' (among the stars in the satellite's infall list that it binds:
roadrunner's own stellar mass) or 'owner' (among the stars it owns and binds).

Each dataset is computed once per kernel (redo=True recomputes it). The
per-figure processing below (square_map, polar_map, newborn_bands,
shell_value, satellites, orbits) is memoised on the Run, in memory only,
and mirrors plots/sim/s01_*, s02_*, s03_* (bands), s031_*, s032_*.
"""
import argparse
import io
import time
import warnings
from pathlib import Path

import numpy as np

from . import simmap as SM
from .starmap import DEFAULT_PARTICLE_DIR, DEFAULT_RR_OUT, RUN as _DEFAULT_RUN
from .custom_cmaps import register as _register_custom_cmaps

_register_custom_cmaps()          # octarine_tuned, ... (replay/custom_cmaps.py)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs' / 'figures' / 'sim'
warnings.filterwarnings('ignore', message='.*use of fork.*may lead to deadlocks', category=DeprecationWarning)


def progress(it, **kw):
    """tqdm progress bar (a widget under Jupyter) over it, if tqdm is installed."""
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return it
    return tqdm(it, **kw)


def run_tag(rr_out):
    """File-name tag of a run: none for vanilla/phase, else e.g. 'rank'; a run outside the
    default one (starmap.RUN, v6) gets its directory name first, e.g. 'halo685x095_v7', so
    its figures never overwrite the default run's."""
    run = Path(rr_out.rstrip('/'))
    tag = '' if (run.parent.name, run.name) == ('vanilla', 'phase') else \
        (run.name if run.parent.name == 'vanilla' else f'{run.parent.name}_{run.name}')
    root = run.parent.parent
    if root != Path(_DEFAULT_RUN):
        tag = root.name + (f'_{tag}' if tag else '')
    return tag


FRET_DEFS = ('list', 'owner')


class Run:
    """One roadrunner run: shared catalogue and ownership replay, and every
    dataset computed from it (R.data, in memory). fret: the per-satellite
    f_ret definition, 'list' or 'owner' (Run.satellites); can be changed at
    any time (R.fret = ...)."""

    def __init__(self, rr_out=DEFAULT_RR_OUT, particle_dir=DEFAULT_PARTICLE_DIR,
                 first=1, last=177, workers=16, r_keep=2.0, fret='owner'):
        self.rr_out, self.particle_dir = rr_out, particle_dir
        self.first, self.last, self.workers = first, last, workers
        self.r_keep = r_keep          # star tables keep stars within r_keep host R_vir
        self.fret = fret              # per-satellite f_ret definition: 'list' or 'owner'
        self.tag = run_tag(rr_out)
        self.sm = None
        self.data = {}

    def fret_def(self, fret=None):
        """The f_ret definition in use: fret if given, else the run's (R.fret)."""
        how = fret if fret is not None else getattr(self, 'fret', 'owner')
        if how not in FRET_DEFS:
            raise ValueError(f"f_ret definition must be one of {FRET_DEFS}, not {how!r}")
        return how

    def load(self):
        """Merger tree, host and the ownership replay to the last snapshot."""
        if self.sm is None:
            from .starmap import StarMaps
            t0 = time.time()
            sm = StarMaps(self.rr_out, self.particle_dir, r_keep=getattr(self, 'r_keep', 2.0))
            sm._catalogue()
            sm.ownership.progress = progress
            sm.ownership._ready(self.last)
            self.sm = sm
            print(f'catalogue and ownership replay: {time.time() - t0:.0f} s')
        self.tree, self.host = self.sm._catalogue()
        self.sm.ownership._pair_keys()   # infall-list lookup table, built once before any worker forks
        return self

    def set_r_keep(self, r_keep):
        """Keep stars within r_keep host R_vir in every star table from now on
        (e.g. to follow orbits further out). The ownership replay is kept;
        everything computed from star tables is dropped, to be recomputed on
        next use (star tables, maps, newborns, population, satellites, zooms)."""
        r_keep = float(r_keep)
        if r_keep == getattr(self, 'r_keep', 2.0) and (self.sm is None or self.sm.r_keep == r_keep):
            print(f'r_keep already {r_keep:g} R_vir')
            return self
        self.r_keep = r_keep
        if self.sm is not None:
            self.sm.r_keep = r_keep
        stale = [k for k in self.data if not (isinstance(k, tuple) and k[0] == 'orbit')]
        for k in stale:
            del self.data[k]
        print(f'r_keep = {r_keep:g} R_vir; dropped {len(stale)} computed datasets (recomputed on next use)')
        return self

    def _get(self, key, fn, redo=False, say=True):
        if redo or key not in self.data:
            t0 = time.time()
            self.data[key] = fn()
            if say:
                print(f'{key}: {time.time() - t0:.0f} s')
        return self.data[key]

    def stem(self, stem):
        return stem + (f'_{self.tag}' if self.tag else '')

    # ---------------------------------------------------------------- datasets
    def star_table(self, s, redo=False):
        """Per-star table of snapshot s (replay.starmap)."""
        self.load()

        def make():
            return self.sm.at(s, cache=False)
        return self._get(('stars', s), make, redo)

    def starmaps(self, snapshots, redo=False):
        """Star tables of several snapshots, {snapshot: table}."""
        tables = {}
        for s in progress(snapshots, desc='star tables', unit='snap'):
            tables[s] = self.star_table(s, redo)
        return tables

    def newborns(self, redo=False, ngb=32, n_tracks=40):
        """s03: every newborn star of the run (replay.newborn, smoothing lengths for the bands)."""
        from . import newborn as NB
        self.load()

        def make():
            return NB.collect(self.rr_out, self.particle_dir, self.first, self.last, self.host, self.workers,
                              n_tracks=n_tracks, verbose=False, ngb=ngb,
                              own=self.sm.ownership, progress=progress)
        return self._get('newborns', make, redo)

    def population(self, redo=False, nr=200, rmin=0.0, ngb=32, kernel_frac=0.5, sigma_floor=0.01,
                   n_tracks=20):
        """s031: shell sums of all host-bound stars per snapshot (replay.popbands)."""
        from . import popbands as PB
        self.load()

        def make():
            return PB.collect(self.rr_out, self.particle_dir, self.first, self.last, self.workers,
                              nr=nr, rmin=rmin, ngb=ngb, kernel_frac=kernel_frac, sigma_floor=sigma_floor,
                              n_tracks=n_tracks, verbose=False, sm=self.sm, progress=progress)
        return self._get('population', make, redo)

    def satellites(self, redo=False, fret=None):
        """s032: per-satellite f_ret and newborns over time (replay.satret), with
        'sats' by the f_ret definition fret (default: the run's, R.fret):
          'list'   among the stars in the satellite's infall list (roadrunner's
                   AssemblyTracker, reconstructed by replay.ownership) that it
                   binds: the stars roadrunner counts as its stellar mass (the
                   catalogue's Mtot), whole snapshot;
          'owner'  among the stars it owns (the last galaxy to claim each star)
                   and binds, within the star tables' r_keep.
        f_ret = 1 - M(also host-bound, more bound to the host) / M(that set); NaN
        where the set is empty. Both are computed in one pass: switching is instant."""
        from . import satret
        self.load()
        how = self.fret_def(fret)

        def make():
            return satret.collect(self.rr_out, self.particle_dir, self.first, self.last, self.workers,
                                  verbose=False, sm=self.sm, progress=progress)
        D = self._get('satellites', make, redo)
        if 'sats_list' not in D:                 # computed before the list view existed
            D = self._get('satellites', make, True)
        return {**D, 'sats': D['sats_list'] if how == 'list' else D['sats']}


# ---------------------------------------------------------------------------- s01 / s02
def map_args(snapshot, quantity='fret', stars='all', smooth=True, **kw):
    """The s01/s02 option namespace (defaults of replay.simmap.add_common_args)."""
    p = argparse.ArgumentParser()
    SM.add_common_args(p, '')
    a = p.parse_args(['--snapshot', str(snapshot)])
    a.snapshot, a.quantity = snapshot, quantity
    a.only_contested, a.smooth = stars == 'contested', smooth
    for k, v in kw.items():
        if not hasattr(a, k):
            raise TypeError(f'unknown map option {k!r}')
        setattr(a, k, v)
    return a


def colour_args(args, **kw):
    """Copy of a map's option namespace with colour settings (cmap, zmin, zmax, vlim) changed."""
    return argparse.Namespace(**{**vars(args), **kw})


def square_map(R, snapshot, quantity='fret', stars='all', smooth=True, joint=False, view='face', box=1.1,
               npix=None, sigma_min=1e4, min_n=5, ngb=32, project='logmean'):
    """s01 image (mass-weighted mean; layered unless joint): dict(img, edges, P, d, args, stem)."""
    a = map_args(snapshot, quantity, stars, smooth, joint=joint, view=view, box=box, npix=npix,
                 sigma_min=sigma_min, min_n=min_n, ngb=ngb, project=project)

    def make():
        d = R.star_table(snapshot)
        P = SM.prepare(d, a)
        rv = float(d['rvir_kpc'])
        n = npix or (800 if smooth else 300)
        e = np.linspace(-box, box, n + 1)
        if smooth:
            deposit = SM.smoother(d, view, ngb).grid(box * rv, n)
            sums = [SM.layer_sums(P, lay, deposit) for lay in P['layers']]
            enough = [s['w'] >= sigma_min for s in sums]
        else:
            x, y = SM.view_xy(d, view)
            deposit = lambda f: np.histogram2d(x, y, bins=(e, e), weights=f)[0]
            sums = [SM.layer_sums(P, lay, deposit) for lay in P['layers']]
            enough = [deposit(lay.astype(np.float64)) >= min_n for lay in P['layers']]
        img = SM.combine(P, sums, enough).T                          # rows = y for imshow
        return dict(img=img, edges=e, P=P, d=d, args=a,
                    stem=R.stem(f's01_s{snapshot:03d}' + SM.stem_suffix(a)))
    return R._get(('s01', snapshot, tuple(sorted(vars(a).items()))), make, say=False)


def polar_map(R, snapshot, quantity='fret', stars='all', smooth=True, joint=False, view='face', box=1.1,
              npix=None, sigma_min=1e4, min_n=5, ngb=32, project='logmean', nr=24, nphi=72, rscale='linear',
              phi0=0.0):
    """s02 image on polar cells: dict(img (nr, nphi), r_edges, phi_edges, sqrt, P, d, args, stem)."""
    a = map_args(snapshot, quantity, stars, smooth, joint=joint, view=view, box=box, npix=npix,
                 sigma_min=sigma_min, min_n=min_n, ngb=ngb, project=project)

    def polar(x, y):
        return np.hypot(x, y), np.mod(np.arctan2(y, x) - np.radians(phi0), 2 * np.pi)

    def make():
        d = R.star_table(snapshot)
        P = SM.prepare(d, a)
        rv = float(d['rvir_kpc'])
        r_edges = (np.linspace(0.0, np.sqrt(box), nr + 1) ** 2 if rscale == 'sqrt'
                   else np.linspace(0.0, box, nr + 1))
        phi_edges = np.linspace(0.0, 2 * np.pi, nphi + 1)
        if smooth:
            n = npix or 800
            grid = SM.smoother(d, view, ngb).grid(box * rv, n)
            c = (np.arange(n) + 0.5) / n * 2 * box - box                     # pixel centres, R_vir
            X, Y = np.meshgrid(c, c, indexing='ij')                          # [x, y] like meshoid's output
            Rr, phi = polar(X.ravel(), Y.ravel())
            ir = np.searchsorted(r_edges, Rr, side='right') - 1
            ip = np.minimum(np.searchsorted(phi_edges, phi, side='right') - 1, nphi - 1)
            cell = np.where((ir >= 0) & (ir < nr), ir * nphi + ip, nr * nphi)
            ncell = nr * nphi + 1
            npix_cell = np.bincount(cell, minlength=ncell)[:-1].reshape(nr, nphi)

            def deposit(f):     # Cartesian surface densities summed into the polar cells
                summed = np.bincount(cell, weights=grid(f).ravel(), minlength=ncell)
                return summed[:-1].reshape(nr, nphi)
            sums = [SM.layer_sums(P, lay, deposit) for lay in P['layers']]
            with np.errstate(invalid='ignore', divide='ignore'):
                enough = [(npix_cell > 0) & (s['w'] / np.maximum(npix_cell, 1) >= sigma_min) for s in sums]
        else:
            x, y = SM.view_xy(d, view)
            Rr, phi = polar(x, y)
            deposit = lambda f: np.histogram2d(Rr, phi, bins=(r_edges, phi_edges), weights=f)[0]
            sums = [SM.layer_sums(P, lay, deposit) for lay in P['layers']]
            area = 0.5 * np.diff(r_edges ** 2)[:, None] * np.diff(phi_edges)[None, :] * rv ** 2   # kpc^2
            enough = [(deposit(lay.astype(np.float64)) >= min_n) & (s['w'] / area >= sigma_min)
                      for lay, s in zip(P['layers'], sums)]
        img = SM.combine(P, sums, enough)
        stem = f's02_s{snapshot:03d}' + SM.stem_suffix(a) + ('_sqrt' if rscale == 'sqrt' else '')
        return dict(img=img, r_edges=r_edges, phi_edges=phi_edges, sqrt=rscale == 'sqrt', P=P, d=d, args=a,
                    stem=R.stem(stem))
    key = ('s02', snapshot, tuple(sorted(vars(a).items())), nr, nphi, rscale, phi0)
    return R._get(key, make, say=False)


# ---------------------------------------------------------------------------- s03 (bands)
def _newborn_selection(d, born, only_contested):
    b, host = d['birth_id'], d['host']
    sel = (d['n_bound'] >= 1) & (b >= 0) & (d['r_rvir'] >= 0)
    if born == 'satellite':
        sel &= b != host
    elif born == 'host':
        sel &= b == host
    if only_contested:
        sel &= d['contested']
    return sel


def newborn_bands(R, born='all', only_contested=False, hsml_from='all', kernel_frac=0.5, sigma_floor=0.01,
                  nr_bands=200, min_n_band=3.0, rmin=0.0):
    """s03 --mode bands: per snapshot, newborns as 3D Gaussians integrated over
    spherical shells, each profile spread over the interval its newborns formed
    in. dict(img (time pixel, shell), n_k, te, re, m, ret (newborns within
    R_vir), stem (without the run tag))."""
    from .popbands import shell_cdf

    def make():
        d = R.newborns()
        sel = _newborn_selection(d, born, only_contested)
        h = d['h_all' if hsml_from == 'all' else 'h_new'][sel]
        sig = np.maximum(kernel_frac * np.nan_to_num(h, nan=0.0), max(sigma_floor, 1e-6))
        a, m, mr = d['r_rvir'][sel], d['mass'][sel], d['mass'][sel] * d['retained'][sel]
        re = np.linspace(rmin, 1.0, nr_bands + 1)
        snaps = d['snapshot'][sel]
        sl, ts = d['snap_list'], d['snap_time']
        col_of = {int(s): i for i, s in enumerate(sl)}
        W, WR, N = (np.zeros((sl.size, re.size - 1)) for _ in range(3))
        order = np.argsort(snaps, kind='stable')
        chunks = np.array_split(order, max(1, order.size // 20000))
        for chunk in progress(chunks, desc='newborn shells', unit='chunk', leave=False):
            dF = np.diff(shell_cdf(re[None, :], a[chunk, None], sig[chunk, None]), axis=1)
            cols = np.array([col_of[int(s)] for s in snaps[chunk]])
            for c in np.unique(cols):
                k = cols == c
                W[c] += m[chunk][k] @ dF[k]
                WR[c] += mr[chunk][k] @ dF[k]
                N[c] += dF[k].sum(axis=0)
        t_now = float(ts.max())
        t_prev = np.interp(sl - 1, sl, ts, left=float(ts[0]) - float(ts[1] - ts[0]))
        te = np.linspace(0.0, t_now, int(round(t_now / 0.01)) + 1)
        tc = 0.5 * (te[1:] + te[:-1])
        col = np.minimum(np.searchsorted(ts, tc, side='left'), sl.size - 1)
        valid = (tc <= ts[-1]) & (tc > t_prev[col])
        pick = lambda A: np.where(valid[:, None], A[col], 0.0)
        w, wr, n_k = pick(W), pick(WR), pick(N)
        enough = n_k >= min_n_band
        with np.errstate(invalid='ignore', divide='ignore'):
            img = np.where(enough & (w > 0), wr / w, np.nan)
        inside = a <= 1.0
        tag = {'all': 'all', 'satellite': 'sat', 'host': 'host'}[born]
        stem = ('s03_fret_' + tag + ('_c' if only_contested else '') + '_bands'
                + ('_hnew' if hsml_from == 'newborn' else '')
                + (f'_sf{sigma_floor:g}' if sigma_floor != 0.01 else ''))
        return dict(img=img, n_k=n_k, te=te, re=re, t_now=t_now, m=m[inside], ret=d['retained'][sel][inside],
                    born=born, only_contested=only_contested, stem=stem)
    key = ('s03', born, only_contested, hsml_from, kernel_frac, sigma_floor, nr_bands, min_n_band, rmin)
    return R._get(key, make, say=False)


# ---------------------------------------------------------------------------- s031
def fine_columns(t, t_now, step=0.01):
    """Fine time grid and, per pixel, the snapshot whose band covers it
    (bands centred on the snapshots, edges halfway to the neighbours)."""
    mid = 0.5 * (t[1:] + t[:-1])
    lo = np.r_[t[0] - (mid[0] - t[0]), mid]
    hi = np.r_[mid, min(t[-1] + (t[-1] - mid[-1]), t_now)]
    te = np.linspace(0.0, t_now, int(round(t_now / step)) + 1)
    tc = 0.5 * (te[1:] + te[:-1])
    col = np.clip(np.searchsorted(hi, tc, side='left'), 0, t.size - 1)
    valid = (tc >= lo[col]) & (tc <= hi[col])
    return te, col, valid


def shell_value(P, quantity='fret', stat='mean', smoothing='joint', min_n=3.0):
    """(value, kernel-weighted star count) per (snapshot, shell). stat: mean
    (mass-weighted) or dmean (density-weighted, rho alone); smoothing: joint
    (all stars together) or layered (contested over the host-only layer)."""
    from .popbands import B_COLS, C_COLS
    C, B = P['C'], P['B']
    col = lambda name: C[..., C_COLS.index(name)]
    bcol = lambda name: B[..., B_COLS.index(name)]
    w = 'm' if stat == 'mean' else 'rho'
    Wc, Qc = col(w), col(w + ('f' if quantity == 'fret' else 'l'))
    Wb, Nc, Nb = bcol(w), col('n'), bcol('n')
    base_val = 1.0 if quantity == 'fret' else 0.0
    with np.errstate(invalid='ignore', divide='ignore'):
        if smoothing == 'joint':
            N = Nc + Nb
            return np.where((N >= min_n) & (Wc + Wb > 0), (Qc + base_val * Wb) / (Wc + Wb), np.nan), N
        show_c = (Nc >= min_n) & (Wc > 0)
        show_b = ~show_c & (Nb >= min_n)
        return np.where(show_c, Qc / Wc, np.where(show_b, base_val, np.nan)), np.where(show_c, Nc, Nb)


def sinking_time(t, r, r_sink):
    """First time after which r never exceeds r_sink again (None if never)."""
    future_max = np.fmax.accumulate(np.where(np.isfinite(r), r, -np.inf)[::-1])[::-1]
    below = np.isfinite(t) & (future_max < r_sink) & np.isfinite(r)
    return float(t[below][0]) if below.any() else None


# ---------------------------------------------------------------------------- s032
def _tree_r(tree, host, g):
    """(cosmic time, r/R_vir) of galaxy g from the merger tree (comoving ratio)."""
    hs = {int(r['Snapshot']): r for r in tree[tree['Sub_tree_id'] == host]}
    out = []
    for r in tree[tree['Sub_tree_id'] == g]:
        s = int(r['Snapshot'])
        if s in hs:
            dx = np.array([r[c] - hs[s][c] for c in ('position_x', 'position_y', 'position_z')], float)
            out.append((float(r['Time']), np.linalg.norm(dx) / float(hs[s]['virial_radius'])))
    out.sort()
    return np.array([o[0] for o in out]), np.array([o[1] for o in out])


def candidates(R, min_npart=100, logm=None, infall=None):
    """s032 candidates, structured (g, m, t, n, ms, mdm, ratio, sf): satellites
    entering R_vir that own >= min_npart star particles at their peak; m peak
    owned M*, t infall time (first tree snapshot within R_vir), ms / mdm owned
    M* / tree M_vir at infall, ratio = log10(ms / mdm), sf = newborn mass born
    in it within the host R_vir (Msun). logm, infall: (lo, hi) cuts. Always from the
    owner view (whatever R.fret), so selections do not change with the f_ret definition."""
    D = R.satellites(fret='owner')

    def make():
        tree, rows = D['tree'], []
        for g, e in progress(D['sats'].items(), total=len(D['sats']), desc='s032 candidates', leave=False):
            if e['n'].max() < min_npart:
                continue
            tt, rr = _tree_r(tree, D['host'], g)
            k = np.flatnonzero(rr < 1.0)
            if k.size == 0:
                continue
            t_inf = float(tt[k[0]])
            row = tree[(tree['Sub_tree_id'] == g) & np.isclose(tree['Time'], t_inf)]
            mdm = float(row['mass'][0]) if row.size else np.nan
            j = int(np.argmin(np.abs(e['time'] - t_inf)))
            ms = float(e['owned'][j]) if e['owned'][j] > 0 else float(e['owned'].max())
            ratio = np.log10(ms / mdm) if mdm > 0 and ms > 0 else np.nan
            sf = float(D['born'].get(g, {}).get('m_in', np.zeros(1)).sum())
            rows.append((g, float(e['owned'].max()), t_inf, int(e['n'].max()), ms, mdm, ratio, sf))
        return np.array(rows, dtype=[('g', 'i8'), ('m', 'f8'), ('t', 'f8'), ('n', 'i8'), ('ms', 'f8'),
                                     ('mdm', 'f8'), ('ratio', 'f8'), ('sf', 'f8')])
    c = R._get(('s032 candidates', min_npart), make, say=False)
    if logm:
        c = c[(np.log10(c['m']) >= logm[0]) & (np.log10(c['m']) <= logm[1])]
    if infall:
        c = c[(c['t'] >= infall[0]) & (c['t'] <= infall[1])]
    return c


def candidate_table(c):
    """The candidates as a DataFrame, by infall time (logs of masses)."""
    import pandas as pd
    with np.errstate(divide='ignore', invalid='ignore'):
        df = pd.DataFrame({'id': c['g'], 'logM*_peak': np.log10(c['m']), 't_infall': c['t'], 'N_peak': c['n'],
                           'logM*_inf': np.log10(c['ms']), 'logMdm_inf': np.log10(c['mdm']),
                           'log M*/Mdm': c['ratio'], 'logMnew_in': np.log10(c['sf'])})
    return df.sort_values('t_infall').reset_index(drop=True).round(2)


def select(c, how='sf', nsat=10, ids=None):
    """(chosen rows, ids that are not candidates). how: sf (most newborn mass
    inside R_vir), ratio (uniform in log M*/M_dm at infall x infall time),
    grid / even (log M* quantile x infall quantile / even bins, most massive
    per cell), mass (most massive), infall (spread evenly in infall time)."""
    if ids:
        known = {int(x['g']): x for x in c}
        return [known[g] for g in ids if g in known], [g for g in ids if g not in known]
    if c.size == 0:
        return [], []
    if how == 'sf':
        c = c[c['sf'] > 0]
        return list(c[np.argsort(-c['sf'])][:nsat]), []
    if how == 'ratio':
        c = c[np.isfinite(c['ratio'])]
        nr = max(1, int(round(np.sqrt(nsat))))
        nt = int(np.ceil(nsat / nr))
        re_ = np.linspace(c['ratio'].min(), c['ratio'].max() + 1e-9, nr + 1)
        te = np.linspace(c['t'].min(), c['t'].max() + 1e-9, nt + 1)
        chosen = []
        for a in range(nr):
            for b in range(nt):
                inside = ((c['ratio'] >= re_[a]) & (c['ratio'] < re_[a + 1])
                          & (c['t'] >= te[b]) & (c['t'] < te[b + 1]))
                cell = c[inside]
                if cell.size:                                  # nearest the cell centre (cell-size units)
                    d2 = (((cell['ratio'] - 0.5 * (re_[a] + re_[a + 1])) / (re_[a + 1] - re_[a])) ** 2
                          + ((cell['t'] - 0.5 * (te[b] + te[b + 1])) / (te[b + 1] - te[b])) ** 2)
                    chosen.append(cell[np.argmin(d2)])
        chosen.sort(key=lambda x: x['t'])
        return chosen, []
    if how == 'mass':
        return list(c[np.argsort(-c['m'])][:nsat]), []
    if how == 'infall':
        chosen = []
        for t0 in np.linspace(c['t'].min(), c['t'].max(), nsat):
            near = c[np.argsort(np.abs(c['t'] - t0))[:3]]
            pick = near[np.argmax(near['m'])]
            if int(pick['g']) not in [int(x['g']) for x in chosen]:
                chosen.append(pick)
        return chosen, []
    nm = max(1, int(round(np.sqrt(nsat))))
    nt = int(np.ceil(nsat / nm))
    lm = np.log10(c['m'])
    me = np.quantile(lm, np.linspace(0, 1, nm + 1))
    te = (np.linspace(c['t'].min(), c['t'].max(), nt + 1) if how == 'even'
          else np.quantile(c['t'], np.linspace(0, 1, nt + 1)))
    mi = np.clip(np.searchsorted(me, lm, side='right') - 1, 0, nm - 1)
    ti = np.clip(np.searchsorted(te, c['t'], side='right') - 1, 0, nt - 1)
    chosen = [c[(mi == a) & (ti == b)][np.argmax(c[(mi == a) & (ti == b)]['m'])]
              for a in range(nm) for b in range(nt) if np.any((mi == a) & (ti == b))]
    chosen.sort(key=lambda x: -x['m'])
    return chosen[:nsat], []


def background_ids(R, which='candidates', min_npart=100, exclude=()):
    """Galaxies for the faint background orbits, minus `exclude`: 'candidates' (they enter the
    host R_vir and own >= min_npart star particles at their peak, as for s032) or 'all' (every
    galaxy other than the host that owns >= min_npart star particles at some snapshot). Owner
    view, whatever R.fret."""
    ex = {int(g) for g in exclude}
    if which == 'candidates':
        ids = [int(g) for g in candidates(R, min_npart)['g']]
    elif which == 'all':
        ids = [int(g) for g, e in R.satellites(fret='owner')['sats'].items() if e['n'].max() >= min_npart]
    else:
        raise ValueError(f"which must be 'candidates' or 'all', not {which!r}")
    return [g for g in ids if g not in ex]


def orbits(R, gids, dt=0.005):
    """{g: (t, r/R_vir)}: cubic-spline orbits from the merger tree (replay.popbands.tracks)."""
    from .popbands import tracks
    R.load()
    need = [g for g in gids if ('orbit', g, dt) not in R.data]
    for g, tt, rr in tracks(R.tree, R.host, need, dt):
        R.data[('orbit', g, dt)] = (tt, rr)
    return {g: R.data[('orbit', g, dt)] for g in gids if ('orbit', g, dt) in R.data}


def fret_along(e, tt):
    """A satellite's f_ret (per snapshot) interpolated linearly to times tt (NaN where undefined)."""
    f = np.full(np.size(tt), np.nan)
    if e is not None and np.isfinite(e['fret']).any():
        ok = np.isfinite(e['fret'])
        f = np.interp(tt, e['time'][ok], e['fret'][ok], left=np.nan, right=np.nan)
    return f


# ---------------------------------------------------------------------------- comp01
def sf_peaks(R, ids):
    """Per satellite (in the order given): the snapshot where the most newborn
    mass forms in it within the host R_vir. List of dict(g, snapshot, time, m_in)."""
    D = R.satellites()
    peaks = []
    for g in ids:
        b = D['born'].get(int(g))
        if b is None or b['m_in'].max() <= 0:
            continue
        k = int(np.argmax(b['m_in']))
        peaks.append(dict(g=int(g), snapshot=int(b['snap'][k]), time=float(b['time'][k]), m_in=float(b['m_in'][k])))
    return peaks


def sat_logx(R, g, s, window=1):
    """Stars of satellite g at snapshot s: those it owns (replay.ownership) and
    binds. log X = log10(E_H/E_O) is defined for the contested ones (also bound
    to the host). Newborns: stars that first appear within the last `window`
    snapshots (s - window, s] and were born in g (birth map).

    dict(logx, m: contested stars; logx_new, m_new: contested newborns;
    m_bound, m_contested, m_new_bound (Msun); fret, fret_new (mass kept / bound);
    centre: median position of the bound stars in the host-disc frame, R_vir;
    snapshot, time)."""

    def make():
        d = R.star_table(s)
        own = R.sm.ownership
        fl, m = d['flags'], d['mass'].astype(np.float64)
        E_H, E_O = d['E_H'].astype(np.float64), d['E_O'].astype(np.float64)

        bound = (d['owner_id'] == g) & ((fl & SM.FLAG_OWNER_BOUND) > 0)
        contested = bound & ((fl & SM.FLAG_HOST) > 0)
        taken = contested & (E_H < E_O)
        with np.errstate(invalid='ignore', divide='ignore'):
            logx = np.log10(E_H / E_O)

        ids = d['ids']
        first = np.full(ids.size, -1)
        known = ids < own._first.size
        first[known] = own._first[ids[known]]
        recent = (first > s - window) & (first <= s)
        born_here = np.zeros(ids.size, bool)
        born_here[recent] = own.birth_of(ids[recent]) == g
        new_bound, new_contested = bound & born_here, contested & born_here

        def kept(b, t):
            return 1.0 - m[t].sum() / m[b].sum() if m[b].sum() > 0 else np.nan

        rv = float(d['rvir_kpc'])
        centre = np.median(d['pos'][bound], axis=0) / rv if bound.any() else np.full(3, np.nan)
        D = R.satellites()
        time_ = float(np.interp(s, D['snap_list'], D['snap_time']))
        return dict(logx=logx[contested], m=m[contested], logx_new=logx[new_contested], m_new=m[new_contested],
                    m_bound=m[bound].sum(), m_contested=m[contested].sum(), m_new_bound=m[new_bound].sum(),
                    fret=kept(bound, taken), fret_new=kept(new_bound, taken & born_here),
                    centre=centre, snapshot=s, time=time_)
    return R._get(('sat_logx', int(g), int(s), int(window)), make, say=False)


def zoom_range(z):
    """(first, last) snapshot of a Comp01 zoom entry: dict(snaps=(s1, s2), ...) or dict(snapshot=s, ...)."""
    if 'snaps' in z:
        return int(z['snaps'][0]), int(z['snaps'][1])
    return int(z['snapshot']), int(z['snapshot'])


_ZOOM = {}


def _zoom_reduce(d, own, s, r_lo, r_hi, ids, how='owner'):
    """The satellites' stars of snapshot s within r_lo..r_hi host R_vir, optionally
    only those of the galaxies `ids`. how='owner': each star owned by a galaxy
    other than the host and bound to it (its energy E_O); 'list': one row per
    (star, galaxy) with the star in that galaxy's infall list and bound to it
    (its energy to that galaxy), so a star in two such lists counts for both."""
    fl = d['flags']
    r = np.linalg.norm(d['pos'], axis=1) / float(d['rvir_kpc'])
    if how == 'list':
        mem = d['mem']
        r_m = r[mem['row']]
        pop = (r_m >= r_lo) & (r_m <= r_hi)
        if ids is not None:
            pop &= np.isin(mem['gid'], np.asarray(ids, np.int64))
        idx, gal, E_g = mem['row'][pop], mem['gid'][pop], mem['E'][pop]
    else:
        pop = ((fl & SM.FLAG_OWNER_BOUND) > 0) & (r >= r_lo) & (r <= r_hi)
        if ids is not None:
            pop &= np.isin(d['owner_id'], np.asarray(ids, np.int64))
        idx = np.flatnonzero(pop)
        gal, E_g = d['owner_id'][idx], d['E_O'][idx]
    host_bound = (fl[idx] & SM.FLAG_HOST) > 0
    with np.errstate(invalid='ignore', divide='ignore'):
        logx = np.where(host_bound, np.log10(d['E_H'][idx] / E_g), -np.inf)
    pid = d['ids'][idx]
    first = np.full(pid.size, -1)
    known = pid < own._first.size
    first[known] = own._first[pid[known]]
    return dict(logx=logx.astype(np.float64), m=d['mass'][idx].astype(np.float64), new=first == s,
                owner=np.asarray(gal, np.int64), snap=np.full(idx.size, s, np.int16))


def _zoom_worker(args):
    s, r_lo, r_hi, ids, how = args
    sm = _ZOOM['sm']
    return _zoom_reduce(sm.at(s, cache=False), sm.ownership, s, r_lo, r_hi, ids, how)


def zoom_sample(R, snaps, r, sats=None):
    """The stars of a Comp01 zoom: the satellites' stars (by the run's f_ret
    definition, R.fret: those in a satellite's infall list and bound to it, or
    those it owns and binds) at the snapshots snaps[0]..snaps[1], within
    r[0]..r[1] host R_vir (sats: only these satellites). One row per star (and
    galaxy) and snapshot: logx = log10(E_H/E_g) (-inf where the host does not
    bind it: kept), m, new (first appearance at that snapshot), owner (the
    galaxy), snap. Memoised; snapshots read in parallel."""
    s1, s2 = int(snaps[0]), int(snaps[1])
    r_lo, r_hi = float(r[0]), float(r[1])
    ids = None if sats is None else tuple(sorted(int(x) for x in sats))
    how = R.fret_def()

    def make():
        R.load()
        snaps_ = list(range(s1, s2 + 1))
        ready = {s for s in snaps_ if ('stars', s) in R.data and R.data[('stars', s)].get('mem') is not None}
        parts = [_zoom_reduce(R.data[('stars', s)], R.sm.ownership, s, r_lo, r_hi, ids, how)
                 for s in snaps_ if s in ready]
        todo = [(s, r_lo, r_hi, ids, how) for s in snaps_ if s not in ready]
        if len(todo) > 1 and R.workers > 1:
            import multiprocessing as mp
            _ZOOM['sm'] = R.sm
            with mp.get_context('fork').Pool(min(R.workers, len(todo))) as pool:
                parts += list(progress(pool.imap(_zoom_worker, todo), total=len(todo),
                                       desc=f'zoom snapshots {s1}-{s2}', unit='snap'))
        else:
            _ZOOM['sm'] = R.sm
            parts += [_zoom_worker(a) for a in todo]
        return {k: np.concatenate([p_[k] for p_ in parts]) for k in ('logx', 'm', 'new', 'owner', 'snap')}
    return R._get(('zoom', s1, s2, r_lo, r_hi, ids, how), make)


def kde(x, w, grid, bw):
    """Weighted Gaussian KDE of x (bandwidth bw) evaluated on grid; integrates
    to 1 over the whole line (mass outside the grid is not drawn)."""
    if np.size(x) == 0 or np.sum(w) <= 0:
        return np.zeros(np.size(grid))
    from scipy.ndimage import gaussian_filter1d
    step = min(bw / 5.0, float(np.min(np.diff(grid))))
    edges = np.arange(min(grid[0], np.min(x)) - 5 * bw, max(grid[-1], np.max(x)) + 5 * bw + step, step)
    h = np.histogram(x, bins=edges, weights=w)[0]
    dens = gaussian_filter1d(h, bw / step, mode='constant') / (np.sum(w) * step)
    return np.interp(grid, 0.5 * (edges[1:] + edges[:-1]), dens)


# ---------------------------------------------------------------------------- star formation beads
def birth_table(R):
    """The run's birth table (particle_data/birth_times.hdf5, export_birth_times.py): exact
    birth time [Gyr] and initial mass [Msun] of every star particle, sorted by particle_index."""
    import h5py

    def make():
        with h5py.File(Path(R.particle_dir) / 'birth_times.hdf5', 'r') as f:
            return {k: f[k][()] for k in ('particle_index', 'initial_mass', 'birth_time')}
    return R._get('birth_table', make, say=False)


_FATE = {}


def _fate_worker(s):
    sm = _FATE['sm']
    d = sm.at(s, cache=False)
    own = sm.ownership
    ids = d['ids']
    first = np.full(ids.size, -1)
    known = ids < own._first.size
    first[known] = own._first[ids[known]]
    nb = first == s
    fl = d['flags'][nb]
    bound = (fl & SM.FLAG_OWNER_BOUND) > 0                        # its galaxy (not the host) binds it
    taken = bound & ((fl & SM.FLAG_HOST) > 0) & (d['E_H'][nb] < d['E_O'][nb])

    # list view: its birth galaxy (in whose list it is) binds it; taken if E_H < E_birth
    rows = np.flatnonzero(nb)
    birth_row = np.full(ids.size, -9, np.int64)
    birth_row[rows] = own.birth_of(ids[rows])
    mem = d['mem']
    pair = birth_row[mem['row']] == mem['gid']
    r_, E_ = mem['row'][pair], mem['E'][pair]
    bound_l = np.zeros(ids.size, bool)
    taken_l = np.zeros(ids.size, bool)
    bound_l[r_] = True
    taken_l[r_] = ((d['flags'][r_] & SM.FLAG_HOST) > 0) & (d['E_H'][r_] < E_)

    # greedy's birth galaxy: the one that binds it most (lowest energy; the host where its energies are stored)
    E_H, E_S = d['E_H'][nb], d['E_S'][nb]
    host_ok = np.isfinite(E_H) & (E_H < 0)
    other_ok = np.isfinite(E_S)
    host_wins = host_ok & (~other_ok | (E_H < E_S))
    greedy = np.where(host_wins, int(d['host']), np.where(other_ok, d['best_id'][nb], -1)).astype(np.int64)
    return dict(pid=ids[nb].astype(np.int64), bound=bound, taken=taken, bound_list=bound_l[nb], taken_list=taken_l[nb],
                greedy=greedy)


def newborn_fates(R):
    """Every star at its first snapshot (within the star tables' r_keep): whether the galaxy
    that owns it binds it (bound), and whether greedy most-bound assignment gives it to the
    host there (taken: also host-bound, E_H < E_O); bound_list, taken_list: the same for its
    birth galaxy, in whose infall list it is (E_H < E_birth); greedy: greedy's birth galaxy,
    the galaxy that binds it most there (the host only where its energies are stored; -1: none).
    dict(pid sorted, bound, taken, bound_list, taken_list, greedy)."""

    def make():
        R.load()
        _FATE['sm'] = R.sm
        snaps = list(range(max(R.first, 1), R.last + 1))
        if R.workers > 1:
            import multiprocessing as mp
            with mp.get_context('fork').Pool(R.workers) as pool:
                parts = list(progress(pool.imap(_fate_worker, snaps), total=len(snaps),
                                      desc='newborn fates', unit='snap'))
        else:
            parts = [_fate_worker(s) for s in progress(snaps, desc='newborn fates', unit='snap')]
        out = {k: np.concatenate([p_[k] for p_ in parts])
               for k in ('pid', 'bound', 'taken', 'bound_list', 'taken_list', 'greedy')}
        o = np.argsort(out['pid'], kind='stable')
        return {k: v[o] for k, v in out.items()}
    F = R._get('newborn_fates', make)
    if 'bound_list' not in F or 'greedy' not in F:   # computed before the list view / greedy labels existed
        F = R._get('newborn_fates', make, redo=True)
    return F


def _entry_time(t, r, r0=1.0):
    """First time the orbit r(t) crosses r = r0 inwards, interpolated linearly between the last
    sample outside and the first inside (inf if it never enters; t[0] if it starts inside)."""
    inside = np.flatnonzero(r < r0)
    if inside.size == 0:
        return np.inf
    k = inside[0]
    if k == 0:
        return t[0]
    return t[k - 1] + (r[k - 1] - r0) / (r[k - 1] - r[k]) * (t[k] - t[k - 1])


def _inside_runs(orbit, r_max):
    """(starts, ends) [Gyr] of the stretches of the spline orbit (t, r/R_vir) with r <= r_max
    (from the first to the last sample inside of each run of consecutive samples)."""
    tt, rr = orbit if orbit is not None else (np.array([np.nan]), np.array([np.nan]))
    ok = np.isfinite(tt) & np.isfinite(rr)
    tt, rr = tt[ok], rr[ok]
    inside = rr <= r_max
    if not inside.any():
        return np.empty(0), np.empty(0)
    edge = np.diff(np.r_[0, inside.astype(np.int8), 0])
    a, b = np.flatnonzero(edge == 1), np.flatnonzero(edge == -1) - 1
    return tt[a], tt[b]


def _births_since(orbit, bt, since, r_max=1.0):
    """Which of the birth times bt count, by `since` ('infall': from the satellite's first entry
    within r_max R_vir on, wherever it is afterwards; 'inside': only while within r_max R_vir;
    'all'), from its spline orbit (t, r/R_vir)."""
    if since == 'all':
        return np.ones(np.size(bt), bool)
    tt, rr = orbit if orbit is not None else (np.array([np.nan]), np.array([np.nan]))
    ok = np.isfinite(tt) & np.isfinite(rr)
    if ok.sum() < 2:
        return np.zeros(np.size(bt), bool)
    if since == 'inside':
        return _run_of(orbit, bt, r_max) >= 0
    return bt >= _entry_time(tt[ok], rr[ok], r_max)


def _run_of(orbit, bt, r_max):
    """Index of the stretch with r <= r_max (_inside_runs) each birth time falls in; -1: none."""
    starts, ends = _inside_runs(orbit, r_max)
    k = np.searchsorted(starts, bt, side='right') - 1
    ok = (k >= 0) & (bt <= ends[np.clip(k, 0, max(ends.size - 1, 0))]) if starts.size else np.zeros(np.size(bt), bool)
    return np.where(ok, k, -1)


def sf_beads(R, gids, dt=0.1, since='infall', r_max=1.0):
    """Star formation of satellites in bins of dt Gyr, from the birth table: the stars born in
    each (birth map), with their exact birth times and initial masses. since: which births
    count: 'infall' (from the satellite's first entry within r_max host R_vir on, wherever it is
    afterwards), 'inside' (only while it is within r_max R_vir), 'all' (True / False: 'inside' /
    'all', as before). Entry and position from its spline orbit. With 'inside' a bin is split
    where the orbit crosses r_max, so every bead lies within it. One bead per (bin, piece) with
    births: at their mass-weighted mean birth time. Outside R_vir the host energies are not
    stored, so newborns there count as kept (f_new = 1 by construction). Bound / taken by the
    run's f_ret definition (R.fret: its birth galaxy's list, or its owner).
    {g: dict(t: bead times [Gyr], sfr: born mass / dt [Msun/yr], f_new: 1 - taken / bound newborn
    initial mass (NaN where none is bound), n: stars)}."""
    if since is True or since is False:
        since = 'inside' if since else 'all'
    if since not in ('infall', 'inside', 'all'):
        raise ValueError(f"since must be 'infall', 'inside' or 'all', not {since!r}")
    gids = [int(g) for g in gids]
    how = R.fret_def()
    r_max = float(r_max)
    key = ('sf_beads', tuple(sorted(gids)), float(dt), since, how, r_max)

    def make():
        R.load()
        B = birth_table(R)
        F = newborn_fates(R)
        pid = B['particle_index'].astype(np.int64)
        birth = R.sm.ownership.birth_of(pid)
        k = np.clip(np.searchsorted(F['pid'], pid), 0, max(F['pid'].size - 1, 0))
        found = F['pid'].size > 0
        found = (F['pid'][k] == pid) if found else np.zeros(pid.size, bool)
        suffix = '_list' if how == 'list' else ''
        bound = np.where(found, F['bound' + suffix][k], False)
        taken = np.where(found, F['taken' + suffix][k], False)
        orb = orbits(R, gids)
        out = {}
        empty = dict(t=np.empty(0), sfr=np.empty(0), f_new=np.empty(0), n=np.empty(0, int))
        for g in gids:
            sel = birth == g
            bt, m0 = B['birth_time'][sel], B['initial_mass'][sel]
            bd, tk = bound[sel], taken[sel]
            piece = np.zeros(bt.size, np.int64)
            if since == 'inside':
                piece = _run_of(orb.get(g), bt, r_max)
                keep = piece >= 0
            else:
                keep = _births_since(orb.get(g), bt, since, r_max)
            bt, m0, bd, tk, piece = bt[keep], m0[keep], bd[keep], tk[keep], piece[keep]
            if bt.size == 0:
                out[g] = dict(empty)
                continue
            i_bin = np.floor(bt / dt).astype(np.int64)
            groups, inv = np.unique(i_bin * (int(piece.max()) + 2) + piece, return_inverse=True)
            sums = lambda w: np.bincount(inv, weights=w, minlength=groups.size)
            m_all, m_bound, m_taken = sums(m0), sums(m0 * bd), sums(m0 * tk)
            t_mean = sums(m0 * bt) / m_all
            with np.errstate(invalid='ignore', divide='ignore'):
                f_new = np.where(m_bound > 0, 1.0 - m_taken / m_bound, np.nan)
            o = np.argsort(t_mean)
            out[g] = dict(t=t_mean[o], sfr=(m_all / (dt * 1e9))[o], f_new=f_new[o],
                          n=np.bincount(inv, minlength=groups.size)[o])
        return out
    return R._get(key, make, say=False)


# ---------------------------------------------------------------------------- birth-set purity (Comp02)
def birth_purity(R, lag_myr=0.0, n_min=20, host_split=15.0, fit='refit'):
    """Every star with a birth galaxy, judged at its first snapshot + lag_myr: the likelihood
    ratio Lambda of its birth galaxy against the likeliest other one, chi2 to its birth galaxy,
    and that other galaxy (replay.birthpurity). fit: 'refit' (6D normals fitted to each galaxy's
    members, the default) or 'rr' (roadrunner's own fitted normals; not an independent check,
    see replay.birthpurity). ~3 min per new (lag, n_min, host_split, fit); then in memory."""
    from . import birthpurity as BP
    key = ('birth_purity', float(lag_myr), int(n_min), None if not host_split else float(host_split), fit)

    def make():
        F = newborn_fates(R)                     # greedy's birth galaxies, judged with the same test
        return BP.collect(R, lag_myr, n_min, host_split, fit=fit, greedy=(F['pid'], F['greedy']), progress=progress)
    P = R._get(key, make)
    if 'top' not in P:                           # computed before greedy's labels / the likeliest galaxy were stored
        P = R._get(key, make, redo=True)
    return P


def _host_times(R):
    h = R.tree[R.tree['Sub_tree_id'] == R.host]
    o = np.argsort(h['Snapshot'].astype(int))
    t = h['Time'].astype(float)[o]
    s = h['Snapshot'].astype(int)[o]
    return t[(s >= R.first) & (s <= R.last)]


def purity_series(R, P, gids, K=100.0, level=0.999, mode='total', label='rr'):
    """Birth-set purity of the galaxies gids at the snapshot times, 'total' (cumulative) or
    'differential' (the stars born since the previous snapshot), of roadrunner's birth sets
    (label 'rr') or greedy's ('greedy'): {g: dict(time, fret, step)} (replay.birthpurity)."""
    from . import birthpurity as BP
    return BP.purity_series(P, gids, _host_times(R), K, level, mode, label)


def compare_series(R, P, gids, compare='reverse', K=100.0, level=0.999, mode='total'):
    """Greedy against roadrunner, per galaxy at the snapshot times ('total' or 'differential'
    as purity_series): compare 'reverse' = of the stars greedy gives to g at birth, the share
    roadrunner also gives to g (initial masses); 'ratio' = purity of greedy's birth set /
    purity of roadrunner's, by the same likelihood test. {g: dict(time, fret, step)}."""
    from . import birthpurity as BP
    times = _host_times(R)
    if compare == 'reverse':
        return BP.reverse_series(P, gids, times, mode)
    if compare == 'ratio':
        a = BP.purity_series(P, gids, times, K, level, mode, 'greedy')
        b = BP.purity_series(P, gids, times, K, level, mode, 'rr')
        out = {}
        for g in set(a) & set(b):
            with np.errstate(invalid='ignore', divide='ignore'):
                out[g] = dict(time=times, fret=a[g]['fret'] / b[g]['fret'], step=a[g]['step'])
        return out
    raise ValueError(f"compare must be 'reverse' or 'ratio', not {compare!r}")


def completeness_series(R, P, gids, K=100.0, level=0.999, mode='total', label='rr'):
    """Birth-set completeness of the galaxies gids at the snapshot times, 'total' or
    'differential' as purity_series, of roadrunner's birth galaxies (label 'rr') or greedy's
    ('greedy'): {g: dict(time, fret, step)} (replay.birthpurity)."""
    from . import birthpurity as BP
    return BP.completeness_series(P, gids, _host_times(R), K, level, mode, label)


def compare_completeness(R, P, gids, compare='forward', K=100.0, level=0.999, mode='total'):
    """Greedy against roadrunner, completeness side: compare 'forward' = of the stars roadrunner
    gives to g at birth, the share greedy also gives to g (initial masses); 'ratio' =
    completeness of greedy's birth set / of roadrunner's, by the same likelihood test.
    {g: dict(time, fret, step)}."""
    from . import birthpurity as BP
    times = _host_times(R)
    if compare == 'forward':
        return BP.forward_series(P, gids, times, mode)
    if compare == 'ratio':
        a = BP.completeness_series(P, gids, times, K, level, mode, 'greedy')
        b = BP.completeness_series(P, gids, times, K, level, mode, 'rr')
        out = {}
        for g in set(a) & set(b):
            with np.errstate(invalid='ignore', divide='ignore'):
                out[g] = dict(time=times, fret=a[g]['fret'] / b[g]['fret'], step=a[g]['step'])
        return out
    raise ValueError(f"compare must be 'forward' or 'ratio', not {compare!r}")


def purity_beads(R, P, gids, dt=0.2, since='infall', K=100.0, level=0.999):
    """Per satellite and dt bin (as sf_beads, same births): SFR from initial masses and the
    purity of those births, 1 - M0(contaminated) / M0(scored) (NaN where none is scored).
    {g: dict(t, sfr, purity, n)}."""
    from . import birthpurity as BP
    scored, bad = BP.flags(P, K, level)
    edges = np.arange(0.0, float(_host_times(R).max()) + dt, dt)
    centres = 0.5 * (edges[:-1] + edges[1:])
    orb = orbits(R, [int(g) for g in gids])
    out = {}
    for g in gids:
        sel = P['g'] == int(g)
        bt, m0, sc, bd = P['birth_time'][sel], P['m0'][sel], scored[sel], bad[sel]
        keep = _births_since(orb.get(int(g)), bt, since)
        bt, m0, sc, bd = bt[keep], m0[keep], sc[keep], bd[keep]
        i = np.clip(np.searchsorted(edges, bt, side='right') - 1, 0, centres.size - 1)
        sums = lambda w: np.bincount(i, weights=w, minlength=centres.size)
        m_all, m_sc, m_bad = sums(m0), sums(m0 * sc), sums(m0 * bd)
        with np.errstate(invalid='ignore', divide='ignore'):
            pur = np.where(m_sc > 0, 1.0 - m_bad / m_sc, np.nan)
        out[int(g)] = dict(t=centres, sfr=m_all / (dt * 1e9), purity=pur, n=np.bincount(i, minlength=centres.size))
    return out


# ---------------------------------------------------------------------------- stellar masses
def mstar_series(R, source='catalogue'):
    """Stellar mass of every galaxy over time, at the snapshots where it is defined:
    {g: dict(time [Gyr], logm = log10 M*/Msun)}. source:
      'catalogue'  roadrunner's Mtot (catalogue snapshots/<s>/galaxy_properties): the current
                   mass of the star particles roadrunner holds bound to the galaxy;
      'owned'      the stars it owns (replay.ownership: birth map and infall lists), bound or
                   not, within the star tables' r_keep ('owner' f_ret definition), or the stars
                   of its infall list ('list');
      'bound'      those it binds (the f_ret denominator; 'list': roadrunner's M*, i.e. Mtot).
    owned / bound follow the run's f_ret definition, R.fret."""
    if source not in ('catalogue', 'owned', 'bound'):
        raise ValueError(f"source must be 'catalogue', 'owned' or 'bound', not {source!r}")

    def make():
        R.load()
        h = R.tree[R.tree['Sub_tree_id'] == R.host]
        t_of = dict(zip(h['Snapshot'].astype(int).tolist(), h['Time'].astype(float).tolist()))
        rows = {}
        if source == 'catalogue':
            import h5py
            with h5py.File(Path(R.rr_out) / 'catalogue.hdf5', 'r') as f:
                for key, grp in f['snapshots'].items():
                    s = int(key)
                    if s not in t_of or 'galaxy_properties' not in grp:
                        continue
                    gp = grp['galaxy_properties'][()]
                    for g, m in zip(gp['Sub_tree_id'].astype(np.int64).tolist(), gp['Mtot'].astype(float).tolist()):
                        if m > 0:
                            rows.setdefault(g, []).append((t_of[s], m))
        else:
            for g, e in R.satellites()['sats'].items():
                for t, m in zip(e['time'].tolist(), e[source].tolist()):
                    if m > 0:
                        rows.setdefault(int(g), []).append((t, m))
        out = {}
        for g, r in rows.items():
            r.sort()
            t, m = np.array(r).T
            out[g] = dict(time=t, logm=np.log10(m))
        return out
    key = ('mstar', source) if source == 'catalogue' else ('mstar', source, R.fret_def())
    return R._get(key, make, say=False)



def dm_series(R, source='mass'):
    """Dark-matter mass of every galaxy over time, from the merger tree (header/merger_tree):
    {g: dict(time [Gyr], logm = log10 M/Msun)} where it is > 0. source: the tree column,
    'mass' (virial mass), 'bound_mass' or 'peak_mass'."""
    if source not in ('mass', 'bound_mass', 'peak_mass'):
        raise ValueError(f"source must be 'mass', 'bound_mass' or 'peak_mass', not {source!r}")

    def make():
        R.load()
        t = R.tree
        keep = np.isfinite(t[source]) & (t[source] > 0)
        g, tm, m = t['Sub_tree_id'][keep].astype(np.int64), t['Time'][keep].astype(float), t[source][keep].astype(float)
        o = np.lexsort((tm, g))
        g, tm, m = g[o], tm[o], m[o]
        starts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
        ends = np.r_[starts[1:], g.size]
        return {int(g[a]): dict(time=tm[a:b], logm=np.log10(m[a:b])) for a, b in zip(starts, ends)}
    return R._get(('dm', source), make, say=False)

# ---------------------------------------------------------------------------- colormaps
def colormap(name):
    """A colormap by name, as matplotlib.colormaps[name], plus cut versions: 'base-T' (e.g.
    'cmyt.arbre-0.8') is base from 0 to T, its value T stretched to 1 (so a too-light end can
    be dropped). A cut map is registered under its name, so the name then works anywhere."""
    import re
    import matplotlib
    from matplotlib.colors import ListedColormap
    if name in matplotlib.colormaps:
        return matplotlib.colormaps[name]
    m = re.fullmatch(r'(.+)-(0?\.\d+|1(?:\.0*)?)', str(name))
    if m is None or m.group(1) not in matplotlib.colormaps or not 0 < float(m.group(2)) <= 1:
        raise KeyError(f"unknown colormap {name!r} (a registered name, or 'name-T' with 0 < T <= 1)")
    base, top = matplotlib.colormaps[m.group(1)], float(m.group(2))
    cm = ListedColormap(base(np.linspace(0.0, top, 256)), name=name)
    cm.set_bad(base.get_bad())
    cm.set_under(base(0.0))
    cm.set_over(base(top))
    matplotlib.colormaps.register(cm)
    return cm


def bead_colours(cmap, norm, own=None, vmin=None, vmax=None):
    """The beads' fill map and range: own None (or 'none' / 'None'), the panel's cmap and norm;
    else the map named own (colormap), on the panel's kind of scale (linear or log) from vmin
    to vmax (None: the panel's ends). (cmap, norm, separate): separate, the beads have a map of
    their own (so they need a colour key)."""
    if own is None or (isinstance(own, str) and own.lower() == 'none'):
        return cmap, norm, False
    vmin = norm.vmin if vmin is None else vmin
    vmax = norm.vmax if vmax is None else vmax
    return colormap(own), type(norm)(vmin, vmax, clip=norm.clip), True


# ---------------------------------------------------------------------------- output
SAVE_FORMATS = ('png',)   # what show(save=True) writes (the notebook's first cell sets these two)
SAVE_DPI = 300            # at this resolution (pdf: of its rasterized parts, e.g. maps, background orbits)


def show(fig, stem, save=False, dpi=300, display_dpi=150, out=OUT, formats=None, save_dpi=None):
    """Render fig to PNG exactly as the scripts save it (rc savefig settings of
    replay.plotstyle) and display that image; save=True also writes it to
    out/<stem>.<format> for each of formats (default SAVE_FORMATS, e.g. 'pdf',
    'png') at save_dpi (default SAVE_DPI). The figure is closed."""
    import matplotlib.pyplot as plt
    from IPython.display import Image, display
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi)
    png = buf.getvalue()
    width = int.from_bytes(png[16:20], 'big')                 # PNG IHDR
    display(Image(data=png, width=int(width * display_dpi / dpi)))
    if save:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        formats = SAVE_FORMATS if formats is None else formats
        formats = (formats,) if isinstance(formats, str) else tuple(formats)
        save_dpi = SAVE_DPI if save_dpi is None else save_dpi
        for fmt in formats:
            fig.savefig(out / f'{stem}.{fmt}', format=fmt, dpi=save_dpi)
            print(f'saved {out / stem}.{fmt} ({save_dpi} dpi)')
    plt.close(fig)
    return stem
