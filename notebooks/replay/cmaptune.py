"""Re-tune a colormap's lightness, hue and colourfulness while keeping its sequence of colours
(notebooks/cmap_tuner.ipynb). Everything happens in CAM02-UCS (J' lightness, a' b' colour),
a perceptually uniform space, so equal steps there look equal.

    cm = retune('cmyt.octarine', window=(0.15, 0.99), J=(27.2, 81.3), hue=10, chroma=0.9)
    cm.out_of_gamut                       # fraction of the 256 colours clipped to sRGB

squeeze: a diverging map with a narrower (or wider) middle band, both ends kept:
    squeeze('cmr.iceburn', middle=0.5)    # iceburn with a thinner black centre

retune: the base map is sampled on `window`; its lightness is remapped onto J = (start, end)
(linear in the base's own lightness, bent by `gamma`: < 1 lifts the dark half, > 1 deepens
it); its colours are rotated by `hue` degrees around the lightness axis and their
colourfulness scaled by `chroma`; back to sRGB, clipping what falls outside the screen gamut.
"""
import numpy as np
from colorspacious import cspace_convert
from matplotlib.colors import ListedColormap

ARBRE_J = (27.2, 81.3)          # cmyt.arbre-0.8's lightness range, J'


def to_ucs(rgb):
    return cspace_convert(np.asarray(rgb)[..., :3], 'sRGB1', 'CAM02-UCS')


def from_ucs(ucs):
    return cspace_convert(ucs, 'CAM02-UCS', 'sRGB1')


def retune(base, window=(0.0, 1.0), J=None, gamma=1.0, hue=0.0, chroma=1.0, reverse=False, n=256, name=None):
    """A ListedColormap from `base` (a name or a colormap): see the module docstring.
    J=None keeps the base's lightness range. Attribute out_of_gamut: clipped fraction."""
    from . import simdata as SD
    cm = SD.colormap(base) if isinstance(base, str) else base
    # the base's own colours (a lookup table does not interpolate), interpolated smoothly in CAM02-UCS
    N = getattr(cm, 'N', 256)
    U_base = to_ucs(cm(np.arange(N) / (N - 1)))
    x = np.linspace(window[0], window[1], n)
    U = np.stack([np.interp(x, np.linspace(0, 1, N), U_base[:, k]) for k in range(3)], axis=1)
    j = U[:, 0]
    j0, j1 = (j[0], j[-1]) if J is None else J
    span = j[-1] - j[0]
    u = (j - j[0]) / span if abs(span) > 1e-9 else np.linspace(0, 1, n)
    u = np.sign(u) * np.abs(u) ** gamma
    U[:, 0] = j0 + (j1 - j0) * u
    c, s = np.cos(np.radians(hue)), np.sin(np.radians(hue))
    a, b = U[:, 1].copy(), U[:, 2].copy()
    U[:, 1], U[:, 2] = chroma * (c * a - s * b), chroma * (s * a + c * b)
    rgb = from_ucs(U)
    out = np.any((rgb < -1e-3) | (rgb > 1 + 1e-3), axis=1)
    rgb = np.clip(rgb, 0.0, 1.0)
    if reverse:
        rgb = rgb[::-1]
    spec = (f"retune({cm.name!r}, window=({window[0]:.3g}, {window[1]:.3g}), J=({j0:.3g}, {j1:.3g}), "
            f"gamma={gamma:.3g}, hue={hue:.3g}, chroma={chroma:.3g}{', reverse=True' if reverse else ''})")
    new = ListedColormap(rgb, name=name or spec)
    new.out_of_gamut = float(out.mean())
    new.spec = spec
    return new


def boost(base, factor=1.0, n=256, name=None):
    """A ListedColormap from `base` (a name or a colormap) with its colourfulness M' (CAM02-UCS) times
    `factor`, at fixed lightness J' and hue. A colour that would leave the screen's gamut is brought
    back to the gamut's edge along its own hue and lightness instead of clipped, so neither shifts.
    Attribute at_edge: the fraction brought back."""
    from . import simdata as SD
    cm = SD.colormap(base) if isinstance(base, str) else base
    U = to_ucs(cm(np.linspace(0, 1, n)))
    J, M, h = U[:, 0], np.hypot(U[:, 1], U[:, 2]), np.arctan2(U[:, 2], U[:, 1])

    def inside(m):
        rgb = from_ucs(np.stack([J, m * np.cos(h), m * np.sin(h)], axis=1))
        return np.all((rgb >= -1e-4) & (rgb <= 1 + 1e-4), axis=1)

    lo, hi = np.zeros(n), np.full(n, 150.0)          # the gamut's edge along each colour's hue and lightness
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        ok = inside(mid)
        lo, hi = np.where(ok, mid, lo), np.where(ok, hi, mid)
    target = factor * M
    m = np.minimum(target, lo)
    rgb = np.clip(from_ucs(np.stack([J, m * np.cos(h), m * np.sin(h)], axis=1)), 0.0, 1.0)
    new = ListedColormap(rgb, name=name or f'boost({cm.name!r}, {factor:.3g})')
    new.at_edge = float(np.mean(target > lo + 1e-6))
    return new


def squeeze(base, middle=1.0, n=256, name=None):
    """A diverging map (`base`, a name or a colormap) with its centre squeezed: value x is drawn
    with the base's colour at 0.5 + 0.5 sign(u) |u|^middle, u = 2x - 1, so middle < 1 leaves the
    centre faster (a narrower middle band, e.g. iceburn's black), > 1 widens it; both ends stay.
    Interpolated in CAM02-UCS like retune. Attributes out_of_gamut (0 here) and spec."""
    from . import simdata as SD
    cm = SD.colormap(base) if isinstance(base, str) else base
    N = getattr(cm, 'N', 256)
    U_base = to_ucs(cm(np.arange(N) / (N - 1)))
    u = np.linspace(-1.0, 1.0, n)
    x = 0.5 + 0.5 * np.sign(u) * np.abs(u) ** middle
    U = np.stack([np.interp(x, np.linspace(0, 1, N), U_base[:, k]) for k in range(3)], axis=1)
    rgb = np.clip(from_ucs(U), 0.0, 1.0)
    spec = f"squeeze({cm.name!r}, middle={middle:.3g})"
    new = ListedColormap(rgb, name=name or spec)
    new.out_of_gamut = 0.0
    new.spec = spec
    return new


def profile(cm, n=256):
    """dict(v, J, hue (CIELab degrees, NaN where nearly grey), C (CIELab chroma), steps (dE' per step))."""
    v = np.linspace(0, 1, n)
    rgb = cm(v)[:, :3]
    U = to_ucs(rgb)
    lab = cspace_convert(rgb, 'sRGB1', 'CIELab')
    C = np.hypot(lab[:, 1], lab[:, 2])
    hue = (np.degrees(np.arctan2(lab[:, 2], lab[:, 1])) + 360) % 360
    return dict(v=v, J=U[:, 0], hue=np.where(C > 12, hue, np.nan), C=C,
                steps=np.linalg.norm(np.diff(U, axis=0), axis=1))


def ab_at(cm, Jq, n=256):
    """The map's a' b' at lightnesses Jq (NaN outside its lightness range; for monotonic maps)."""
    U = to_ucs(cm(np.linspace(0, 1, n)))
    o = np.argsort(U[:, 0])
    out = np.full((np.size(Jq), 2), np.nan)
    ok = (Jq >= U[:, 0].min()) & (Jq <= U[:, 0].max())
    out[ok, 0] = np.interp(Jq[ok], U[o, 0], U[o, 1])
    out[ok, 1] = np.interp(Jq[ok], U[o, 0], U[o, 2])
    return out


def distance_at_equal_lightness(cm, ref, J_range=ARBRE_J, n=60):
    """Colour distance (a' b' plane, CAM02-UCS) between cm and ref at the same lightness,
    over J_range: (median, minimum); NaN where they share no lightness."""
    Jq = np.linspace(J_range[0], J_range[1], n)
    d = np.hypot(*(ab_at(cm, Jq) - ab_at(ref, Jq)).T)
    return (float(np.nanmedian(d)), float(np.nanmin(d))) if np.isfinite(d).any() else (np.nan, np.nan)
