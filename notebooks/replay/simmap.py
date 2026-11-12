"""Shared machinery of the simulation-only maps (plots/sim/s01_*, s02_*).

Per star (replay/starmap.py, computed in memory from the roadrunner output):
E_H, the energy relative to the central host,
and E_S, the energy relative to the galaxy the star is compared with:
  --es owner (default): where the star should be (replay/ownership.py: its
            birth galaxy plus the hierarchical accretion in roadrunner's
            assembly lists). E_S := E_H when that owner is the host, or
            when the owner does not bind it (it can not hold the star, so
            the host taking it is no theft).
  --es best: the lowest (most bound) energy relative to any other halo,
            E_S := E_H when no other halo binds the star (the first maps).
Quantities:
  logratio  q = log10(E_H/E_S): > 0 the host binds the star more, < 0 another
            halo does, = 0 host-only stars (and exact ties);
  fret      q = 1 if log10(E_H/E_S) <= 0 (E_S <= E_H), else 0: its mass-weighted
            mean is f_retention; host-only stars count as retained (1).
Only stars bound to the host enter (--only-contested: also bound to the
galaxy they are compared with).

Layers: contested stars are averaged on their own and drawn over a base
layer of the host-only stars, whose value is exact (0 for logratio, 1 for
fret), so the base does not dilute them; joint=True averages everything
together.

Deposition is linear (sums of m, m q, m E_H, m E_S over a layer), either
binned (histogram) or smoothed with meshoid (kernels with smoothing lengths
from ALL stars, so a sparse layer does not inflate them). A pixel's value is
the ratio of sums; it is drawn if the layer has enough there: >= min_n stars
binned, >= sigma_min Msun/kpc^2 deposited surface density smoothed. Because
the sums are linear, they can be re-binned (e.g. Cartesian pixels into polar
cells) without changing their meaning.
"""
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[1]
FLAG_HOST, FLAG_OTHER, FLAG_OWNER_HOST, FLAG_OWNER_BOUND = 1, 2, 4, 8
VIEWS = {'face': (0, 1, 'x', 'y'), 'edge': (0, 2, 'x', 'z')}

# coolwarm's ends with a mid-grey centre: log10(E_H/E_S) = 0 (host-only stars,
# ties) stays visible on the white background instead of fading into it.
_cw = matplotlib.colormaps['coolwarm']
COOLGREY = LinearSegmentedColormap.from_list(
    'coolgrey', [_cw(0.0), _cw(0.18), _cw(0.33), '#9a9a9a', _cw(0.67), _cw(0.82), _cw(1.0)], N=256)
for _cm in (COOLGREY, COOLGREY.reversed()):
    if _cm.name not in matplotlib.colormaps:
        matplotlib.colormaps.register(_cm)


def _fret_maps():
    """f_retention candidates (plots/sim/cmap_compare.py --quantity fret); usable as --cmap."""
    from . import plotstyle as _S
    warm = [(0.0, '#9a9a9a'), (0.07, '#bca592'), (0.07 + 0.93 * 0.25, '#e2a98c'),
            (0.07 + 0.93 * 0.5, _cw(0.75)), (0.07 + 0.93 * 0.75, _cw(0.9)), (1.0, _cw(1.0))]
    return {
        # cgbreak_warm_v2's warm leg reversed: f = 1 the same grey, a fast fade to sand, red at 0
        'fret_warm': ('grey (1) -> sand -> red (0): log X warm leg', LinearSegmentedColormap.from_list(
            'fret_warm', [(1.0 - x, c) for x, c in warm[::-1]])),
        # the main figures' f_steal ramp, as f_ret = 1 - f_steal: light = kept, dark = stolen
        'fret_blue': ('light (1) -> dark blue (0): f_steal ramp of figs 1-6', _S.CMAP.reversed('fret_blue')),
    }


for _name, (_label, _cm) in _fret_maps().items():
    if _name not in matplotlib.colormaps:
        matplotlib.colormaps.register(_cm, name=_name)
FRET_CMAPS = {k: v[0] for k, v in _fret_maps().items()}
FRET_CMAPS.update({'coolwarm_r': 'current: coolwarm_r', 'cividis': 'cividis (uniform, CVD-safe)'})


def broken_cmap(name, neg, pos, zero='#9a9a9a', n=256):
    """log10(E_H/E_S) map with a colour break at 0: a negative leg (zmin ...
    just below 0), a positive leg (just above 0 ... zmax) starting on another
    hue, and exact 0 (host-only stars) in its own middle LUT entry, where
    TwoSlopeNorm(0, zmin, zmax) sends it."""
    from matplotlib.colors import ListedColormap, to_rgb
    lo = LinearSegmentedColormap.from_list('lo', neg)(np.linspace(0, 1, n))
    hi = LinearSegmentedColormap.from_list('hi', pos)(np.linspace(0, 1, n))
    return ListedColormap(np.vstack([lo, [[*to_rgb(zero), 1.0]], hi]), name=name)


# candidates with a break at 0 (plots/sim/cmap_compare.py); usable as --cmap
BROKEN = {  # name: (label, negative leg, positive leg[, colour at exactly 0])
    'cgbreak_A': ("steel blue | grey | sand", [_cw(0.0), _cw(0.15), '#8ea6c8'], ['#d2b48c', _cw(0.75), _cw(0.9), _cw(1.0)]),
    'cgbreak_B': ("ice | grey | amber", ['#1f4e9c', '#4a8fd0', '#9fd3ea'], ['#f7d27a', '#f08c3a', '#c8312a', '#6e0f1c']),
    'cgbreak_C': ("purple | grey | orange (PuOr)", ['#3f1d73', '#7b5bab', '#c3b3dc'], ['#fbc98b', '#f08a2e', '#b4490f', '#5e2306']),
    'cgbreak_D': ("teal | grey | rose", ['#0b4f5c', '#2f8f9d', '#9ed3d0'], ['#f4b6b0', '#e0675f', '#b0213a', '#5a0a1f']),
    # the log X default (user): blue leg towards near-white below 0 (the break), exact 0
    # the coolgrey grey, warm leg continuous from that grey through sand into coolwarm's reds
    'cgbreak': ("blue -> white | grey -> sand -> red",
                [_cw(0.0), _cw(0.2), '#a9c3e6', '#e9eff7'],
                ['#9a9a9a', '#bca592', '#e2a98c', _cw(0.75), _cw(0.9), _cw(1.0)]),
    # the log X default (user): as cgbreak, but exact 0 takes the warm leg's own starting
    # sand, so 0 and the warm leg are continuous; the only break is blue/white -> sand
    'cgbreak_warm': ("blue -> white | sand -> red",
                     [_cw(0.0), _cw(0.2), '#a9c3e6', '#e9eff7'],
                     ['#bca592', '#e2a98c', _cw(0.75), _cw(0.9), _cw(1.0)], '#bca592'),
    # v2 (user): exact 0 the coolgrey grey again, the warm leg fading fast but continuously
    # from that pure grey into cgbreak_warm's sand (by ~0.2 in log X), then as cgbreak_warm
    'cgbreak_warm_v2': ("blue -> white | grey -> fast sand -> red",
                        [_cw(0.0), _cw(0.2), '#a9c3e6', '#e9eff7'],
                        [(0.0, '#9a9a9a'), (0.07, '#bca592'), (0.07 + 0.93 * 0.25, '#e2a98c'),
                         (0.07 + 0.93 * 0.5, _cw(0.75)), (0.07 + 0.93 * 0.75, _cw(0.9)), (1.0, _cw(1.0))]),
}
for _name, (_label, _neg, _pos, *_zero) in BROKEN.items():
    if _name not in matplotlib.colormaps:
        matplotlib.colormaps.register(broken_cmap(_name, _neg, _pos, *_zero))


# --------------------------------------------------------------------------
# command line
# --------------------------------------------------------------------------

def add_common_args(p, npix_help):
    from .starmap import DEFAULT_RR_OUT, DEFAULT_PARTICLE_DIR
    p.add_argument('--snapshot', type=int, nargs='+', required=True,
                   help='one or more snapshots (one figure per snapshot and variant; the ownership '
                        'replay and each snapshot are computed once)')
    p.add_argument('--rr-out', default=DEFAULT_RR_OUT, help='roadrunner run (default: v6 phase on jotun)')
    p.add_argument('--particle-dir', default=DEFAULT_PARTICLE_DIR)
    p.add_argument('--quantity', choices=('logratio', 'fret'), nargs='+', default=['logratio'],
                   help='logratio: log10(E_H/E_S); fret: fraction with log10(E_H/E_S) <= 0 (both: one '
                        'figure each)')
    p.add_argument('--stars', choices=('all', 'contested'), nargs='+', default=None,
                   help="star sets drawn: all host-bound, contested (as --only-contested); several: "
                        "one figure each")
    p.add_argument('--render', choices=('binned', 'smooth'), nargs='+', default=None,
                   help='binned and/or smooth (as --smooth); several: one figure each')
    p.add_argument('--es', choices=('owner', 'best'), default='owner',
                   help='E_S: energy to the galaxy the star should be in (owner, default), or to '
                        'the most bound other halo (best)')
    p.add_argument('--only-contested', action='store_true',
                   help='keep only stars bound to both the host and the galaxy they are compared with')
    p.add_argument('--joint', action='store_true',
                   help='average host-only and contested stars together (default: contested stars '
                        'averaged on their own and drawn over the host-only base layer)')
    p.add_argument('--view', choices=tuple(VIEWS), default='face', help='host disc face-on or edge-on')
    p.add_argument('--box', type=float, default=1.1, help='half-width / outer radius, in R_vir (default 1.1)')
    p.add_argument('--npix', type=int, default=None, help=npix_help)
    p.add_argument('--smooth', action='store_true', help='kernel deposition with meshoid instead of binning')
    p.add_argument('--project', choices=('logmean', 'energyratio'), default='logmean',
                   help='logratio maps: mass-weighted mean of log10(E_H/E_S) (default), or '
                        'log10 of the ratio of mass-weighted mean energies')
    p.add_argument('--sigma-min', type=float, default=1e4,
                   help='--smooth: blank cells below this deposited surface density, Msun/kpc^2 (default 1e4)')
    p.add_argument('--ngb', type=int, default=32, help='--smooth: neighbours per smoothing length')
    p.add_argument('--min-n', type=int, default=5, help='binned: blank cells with fewer stars (default 5)')
    p.add_argument('--cmap', default=None,
                   help='any matplotlib colormap name, or coolgrey (coolwarm with a mid-grey centre, '
                        'so 0 stands out on white), cgbreak (blue fading to white below 0, grey at 0, warm '
                        'above: the sign reads at a glance), cgbreak_warm (as cgbreak, 0 in the warm-leg sand). cgbreak_warm_v2 (grey 0, fast continuous fade to sand). Default: cgbreak_warm_v2 for logx, magma for fret. '
                        'Used as given; add _r to reverse. Dark-centred alternatives: berlin, managua, vanimo')
    p.add_argument('--zmin', type=float, default=None,
                   help='logx colour range low end (< 0): the blue leg spans [zmin, 0]')
    p.add_argument('--zmax', type=float, default=None,
                   help='logx colour range high end (> 0): the red leg spans [0, zmax]; grey stays at 0')
    p.add_argument('--vlim', type=float, default=None,
                   help='symmetric colour range +-VLIM when --zmin/--zmax are not given '
                        '(default: 99th percentile of |cells|)')
    p.add_argument('--out', default='outputs/figures/sim')
    p.add_argument('--format', default='png')
    p.add_argument('--dpi', type=int, default=300)
    p.add_argument('--show', action='store_true', help='show the figure instead of saving it')


def variants(args):
    """One namespace per (snapshot, quantity, star set, rendering)."""
    import copy
    stars = args.stars or (['contested'] if args.only_contested else ['all'])
    render = args.render or (['smooth'] if args.smooth else ['binned'])
    for s in args.snapshot:
        for q in args.quantity:
            for st in stars:
                for r in render:
                    v = copy.copy(args)
                    v.snapshot, v.quantity = s, q
                    v.only_contested, v.smooth = st == 'contested', r == 'smooth'
                    yield v


def validate(args):
    if args.zmin is not None and not args.zmin < 0:
        raise SystemExit('--zmin must be < 0 (the colour centre is fixed at 0)')
    if args.zmax is not None and not args.zmax > 0:
        raise SystemExit('--zmax must be > 0 (the colour centre is fixed at 0)')
    if args.cmap is not None and args.cmap not in matplotlib.colormaps:
        raise SystemExit(f'unknown colormap {args.cmap!r} (any name in matplotlib.colormaps)')


def stem_suffix(args, median=False):
    """'_<logx|fret>[_best][_c][_sm][_edge][_med][_eratio][_joint]'."""
    return ('_' + ('fret' if args.quantity == 'fret' else 'logx') + ('_best' if args.es == 'best' else '')
            + ('_c' if args.only_contested else '') + ('_sm' if args.smooth else '')
            + ('_edge' if args.view == 'edge' else '')
            + ('_med' if median else '')
            + ('_eratio' if args.quantity == 'logratio' and args.project == 'energyratio' else '')
            + ('_joint' if args.joint and not args.only_contested else ''))


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

_STARMAPS = {}


def load(args):
    """Per-star table of args.snapshot, computed from the run (cached per
    run, so several snapshots share one ownership replay)."""
    from .starmap import StarMaps
    key = (args.rr_out, args.particle_dir)
    if key not in _STARMAPS:
        _STARMAPS[key] = StarMaps(args.rr_out, args.particle_dir, owners=getattr(args, 'es', 'owner') == 'owner')
    return _STARMAPS[key].at(args.snapshot)


def prepare(d, args):
    """Per-star selection, quantity and layers. bound_other: bound to the
    galaxy compared with the host (owner or most bound other halo)."""
    flags = d['flags']
    bound_host = (flags & FLAG_HOST) > 0
    E_H = d['E_H'].astype(np.float64)
    if getattr(args, 'es', 'best') == 'owner':
        bound_other = (flags & FLAG_OWNER_BOUND) > 0                   # owner is not the host and binds it
        E_S = np.where(bound_other, d['E_O'].astype(np.float64), E_H)  # owner host / unbound: E_S = E_H
    else:
        bound_other = (flags & FLAG_OTHER) > 0
        E_S = np.where(bound_other, d['E_S'].astype(np.float64), E_H)  # no other halo: E_S = E_H
    sel = bound_host.copy()
    if args.only_contested:
        sel &= bound_other
    if args.quantity == 'fret':
        q = (E_S <= E_H).astype(np.float64)                   # 1: log10(E_H/E_S) <= 0 (host-only too)
    else:
        with np.errstate(invalid='ignore', divide='ignore'):
            q = np.log10(E_H / E_S)
    layers = [sel] if args.joint else [sel & bound_other, sel & ~bound_other]
    return dict(es=getattr(args, 'es', 'best'), flags=flags, bound_host=bound_host, bound_other=bound_other, E_H=E_H, E_S=E_S,
                sel=sel, q=np.where(sel, q, 0.0), m=d['mass'].astype(np.float64), layers=layers,
                energy_ratio=args.quantity == 'logratio' and args.project == 'energyratio',
                base=1.0 if args.quantity == 'fret' else 0.0, joint=args.joint)


def layer_sums(P, lay, deposit):
    """Linear sums over one layer, each deposited with deposit(per-star weights)."""
    m = P['m']
    out = dict(w=deposit(np.where(lay, m, 0.0)), n=None)
    if P['energy_ratio']:
        out['nh'] = deposit(np.where(lay, m * P['E_H'], 0.0))
        out['ns'] = deposit(np.where(lay, m * P['E_S'], 0.0))
    else:
        out['nq'] = deposit(np.where(lay, m * P['q'], 0.0))
    return out


def value(P, s):
    with np.errstate(invalid='ignore', divide='ignore'):
        if P['energy_ratio']:
            return np.log10(s['nh'] / s['ns'])
        return s['nq'] / s['w']


def combine(P, sums, enough, first_values=None):
    """Layered image: the first layer on top; later layers fill only where it
    is absent, the host-only layer with its exact value."""
    img = np.full(sums[0]['w'].shape, np.nan)
    for n, (s, ok) in enumerate(zip(sums, enough)):
        if P['joint'] or n == 0:
            vals = first_values if (n == 0 and first_values is not None) else value(P, s)
        else:
            vals = np.full(s['w'].shape, P['base'])
        put = np.isnan(img) & ok
        img[put] = vals[put]
    return img


_SMOOTHERS = {}


def smoother(d, view, ngb):
    """Cached Smoother (smoothing lengths are the costly part)."""
    key = (int(d['snapshot']), float(d['rvir_kpc']), view, ngb)
    if key not in _SMOOTHERS:
        _SMOOTHERS[key] = Smoother(d, view, ngb)
    return _SMOOTHERS[key]


class Smoother:
    """meshoid deposition on a Cartesian grid of the chosen view; smoothing
    lengths from all stars."""

    def __init__(self, d, view, ngb):
        import meshoid
        i, j, _, _ = VIEWS[view]
        k = 3 - i - j                                          # line of sight
        self.M = meshoid.Meshoid(d['pos'].astype(np.float64)[:, [i, j, k]], d['mass'].astype(np.float64),
                                 des_ngb=ngb, verbose=False)

    def grid(self, half_width_kpc, npix):
        return lambda f: self.M.SurfaceDensity(f, size=2 * half_width_kpc, center=np.zeros(3), res=npix)


def view_xy(d, view):
    """Projected coordinates of every star in units of R_vir."""
    i, j, _, _ = VIEWS[view]
    rv = float(d['rvir_kpc'])
    return d['pos'][:, i] / rv, d['pos'][:, j] / rv


# --------------------------------------------------------------------------
# drawing
# --------------------------------------------------------------------------

def colours(args, img):
    if args.quantity == 'fret':
        return args.cmap or 'magma', TwoSlopeNorm(0.5, 0.0, 1.0), r'$f_{\rm retention}$'   # user: magma/inferno
    v = args.vlim or float(np.nanpercentile(np.abs(img), 99)) or 1.0
    zmin = -v if args.zmin is None else args.zmin
    zmax = v if args.zmax is None else args.zmax
    # two independent linear legs: [zmin, 0] -> blue half, [0, zmax] -> red half
    return args.cmap or 'cgbreak_warm_v2', TwoSlopeNorm(0.0, zmin, zmax), r'$\log_{10}(E_H/E_S)$'


def finish_colorbar(fig, im, args, clabel, S, **kw):
    cb = S.colorbar(fig, im, label=clabel, ends=False, **kw)
    if args.quantity == 'logratio':
        # colours follow the two-leg norm (grey at 0), but the bar's axis stays linear
        # in value: even tick spacing, 0 off-centre when the legs differ
        from matplotlib.ticker import MaxNLocator
        cb.ax.set_yscale('linear')
        cb.ax.yaxis.set_major_locator(MaxNLocator(nbins=7, steps=[1, 2, 2.5, 5, 10]))
        cb.ax.minorticks_off()
    other = 'owner' if getattr(args, 'es', 'best') == 'owner' else 'other halo'
    top, bottom = (other, 'host') if args.quantity == 'fret' else ('host', other)
    cb.ax.text(0.5, 1.012, top, transform=cb.ax.transAxes, ha='center', va='bottom', fontsize=6.5,
               color=S.INK2)
    cb.ax.text(0.5, -0.012, bottom, transform=cb.ax.transAxes, ha='center', va='top', fontsize=6.5,
               color=S.INK2)
    return cb


def footnote(fig, P, S):
    bh, bo, n = P['bound_host'], P['bound_other'], P['flags'].size
    frac = lambda mask: f'{100 * mask.sum() / n:.1f}%'
    contested = bh & bo
    if P['es'] == 'owner':
        oh = (P['flags'] & FLAG_OWNER_HOST) > 0
        note = (f'host-bound: owner host {frac(bh & oh)} | owner other, binds it {frac(contested)} '
                f'(owner wins {frac(contested & (P["E_S"] < P["E_H"]))}) | owner other, does not bind '
                f'{frac(bh & ~oh & ~bo)} | not host-bound {frac(~bh)}')
        fig.text(0.5, 0.005, note, ha='center', va='bottom', fontsize=5.6, color=S.INK2)
        return
    note = (f'host only {frac(bh & ~bo)} | contested {frac(contested)} '
            f'(other halo wins {frac(contested & (P["E_S"] < P["E_H"]))}) | other halo only '
            f'{frac(~bh & bo)} | unbound {frac(~bh & ~bo)}')
    fig.text(0.5, 0.005, note, ha='center', va='bottom', fontsize=5.6, color=S.INK2)


def title(ax, d, **kw):
    ax.set_title(f'snapshot {int(d["snapshot"])}, $z = {float(d["redshift"]):.2f}$', fontsize=7.5, **kw)
