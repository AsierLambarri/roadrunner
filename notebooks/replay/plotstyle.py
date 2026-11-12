"""Shared look of the notebooks' figures: fonts and ticks (journal-ready), ink colours and a slim
colourbar (from lvjeans' greedy.plotstyle, trimmed to what the notebooks use)."""
import sys

import matplotlib
# Batch drawing never needs a display: force Agg unless the script was asked
# to --show its figures (then matplotlib picks an interactive backend).
SHOW = '--show' in sys.argv[1:]
if not SHOW:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402

# --- ink ---------------------------------------------------------------------
INK = '#0b0b0b'
INK2 = '#52514e'      # secondary text, guides
MUTED = '#898781'     # reference lines, annotations
HAIR = '#d9d8d1'      # bands, hatching
SPINE = '#3a3a38'

# --- f_steal ramp (blue 100 -> 700): simmap's 'fret_blue' -----------------------
_BLUE = ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7', '#3987e5',
         '#2a78d6', '#256abf', '#1c5cab', '#184f95']
CMAP = LinearSegmentedColormap.from_list('fsteal_blue', _BLUE, N=256)
CMAP.set_bad('none')

_RC = {
    'font.family': 'serif',
    'font.serif': ['STIXGeneral', 'STIX Two Text', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 8,
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'legend.fontsize': 7,
    'legend.frameon': False,
    'legend.handlelength': 2.2,
    'legend.borderaxespad': 0.4,
    'legend.labelspacing': 0.3,
    'axes.linewidth': 0.6,
    'axes.edgecolor': SPINE,
    'axes.labelcolor': INK,
    'text.color': INK,
    'xtick.color': SPINE,
    'ytick.color': SPINE,
    'xtick.labelcolor': INK,
    'ytick.labelcolor': INK,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'xtick.major.size': 3.5,
    'ytick.major.size': 3.5,
    'xtick.minor.size': 1.8,
    'ytick.minor.size': 1.8,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.minor.width': 0.45,
    'ytick.minor.width': 0.45,
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
    'lines.linewidth': 1.2,
    'lines.solid_capstyle': 'round',
    'lines.dash_capstyle': 'round',
    'hatch.linewidth': 0.4,
    'hatch.color': '#b5b4ad',
    'axes.titlepad': 3,
    'figure.dpi': 100,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.03,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
}


def use():
    """Apply the shared rcParams (call once per figure, before drawing)."""
    plt.rcParams.update(_RC)


def ylabel(quantity):
    return r'$f_{\rm steal}$' if quantity == 'steal' else r'$f_{\rm lost}$'


def colorbar(fig, mappable, cax=None, ax=None, quantity='steal', label=None, ends=True, **kw):
    """Slim colorbar; for f_steal, ticks at quarters and the two ends named."""
    cb = fig.colorbar(mappable, cax=cax, ax=ax, **kw)
    cb.outline.set_linewidth(0.5)
    cb.outline.set_edgecolor(SPINE)
    cb.ax.tick_params(which='both', direction='in', length=2.5, width=0.5, right=True, left=False)
    cb.ax.minorticks_off()
    if label is None:
        label = ylabel(quantity)
    cb.set_label(label, labelpad=3)
    if ends and isinstance(mappable.norm, Normalize) and mappable.norm.vmin == 0 and mappable.norm.vmax == 1:
        cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
        cb.set_ticklabels(['0', '', '0.5', '', '1'])
        top = 'to host' if quantity == 'steal' else 'lost'
        cb.ax.text(0.5, 1.012, top, transform=cb.ax.transAxes, ha='center', va='bottom',
                   fontsize=6.5, color=INK2)
        cb.ax.text(0.5, -0.012, 'kept', transform=cb.ax.transAxes, ha='center', va='top',
                   fontsize=6.5, color=INK2)
    return cb
