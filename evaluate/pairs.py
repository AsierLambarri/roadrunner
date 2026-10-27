"""Goal 3's reading of a snapshot's stored pairs, shared by model.py (3.1-3.3, 3.5), model_alpha.py
(3.2b, 3.4) and bgmm.py (the replay).

A stored pair is one (particle, candidate galaxy) entry of the snapshot's assignment file: the
particle's row, the galaxy, log_resp, log_alpha and the energy (``Run.pairs``). Pairs are grouped by
galaxy (``groups``) and by particle (``by_row``); galaxies linked through shared particles form
components (``components``); a particle with two or more candidates gives one candidate pair (k, j)
per combination of them (``pair_entries``, ``contested_pairs``), where the mixture makes a decision.
"""
import itertools

import numpy as np
import pandas as pd
from scipy.linalg import solve_triangular
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

MIN_PARTICLES = 1e5                     # goal 3 reads snapshots with at least this many particles
MIN_PAIR = 30                           # a statistic per galaxy pair needs at least this many particles
UNRESOLVED_RATIO = 10                   # the pipeline's UNRESOLVED_GROUP_RATIO
MASS_BINS = np.array([0.0, 1e7, 1e8, 1e9, np.inf])
MASS_LABELS = ["<1e7", "1e7-1e8", "1e8-1e9", ">1e9"]


# --- grouping -------------------------------------------------------------------------------------

def groups(keys):
    """``(key, positions)`` of each distinct key."""
    u, inv, n = np.unique(keys, return_inverse=True, return_counts=True)
    return zip(u, np.split(np.argsort(inv.ravel(), kind="stable"), np.cumsum(n)[:-1]))


def by_row(rows):
    """The order sorting ``rows``, and the start and size of each particle's run of entries in it."""
    o = np.argsort(rows, kind="stable")
    if not rows.size:
        return o, np.array([], np.int64), np.array([], np.int64)
    start = np.flatnonzero(np.r_[True, rows[o][1:] != rows[o][:-1]])
    return o, start, np.diff(np.r_[start, rows.size])


def components(p):
    """Each stored pair's component (the galaxies linked through shared particles) and whether that
    component is resolved (see model.py's Scope): K >= 2 galaxies with fewer than 10 K particles are
    unresolved."""
    rows, gal = p["row"].astype(np.int64), p["galaxy"].astype(np.int64)
    if not rows.size:
        return np.zeros(0, np.int64), np.zeros(0, bool)
    g, gi = np.unique(gal, return_inverse=True)
    gi = gi.ravel()
    o, start, count = by_row(rows)
    first = np.repeat(gi[o][start], count)           # each entry's particle's first galaxy
    graph = coo_matrix((np.ones(rows.size), (first, gi[o])), shape=(g.size, g.size))
    _, comp = connected_components(graph, directed=False)
    particles = np.bincount(comp[gi[o][start]], minlength=comp.max() + 1)
    galaxies = np.bincount(comp, minlength=comp.max() + 1)
    ok = (galaxies == 1) | (particles >= UNRESOLVED_RATIO * galaxies)
    return comp[gi], ok[comp[gi]]


# --- the Gaussians at the pairs -------------------------------------------------------------------

def log_gauss(x, mean, cov):
    """log N(x | mean, cov) of the rows of ``x``; ``cov`` full (d, d), diagonal (d,) or spherical ()."""
    cov = np.diag(np.broadcast_to(cov, mean.shape)) if np.ndim(cov) < 2 else cov
    L = np.linalg.cholesky(cov)
    z = solve_triangular(L, (x - mean).T, lower=True)
    return -0.5 * (z * z).sum(0) - np.log(np.diag(L)).sum() - 0.5 * mean.size * np.log(2.0 * np.pi)


def galaxy_terms(p, X, gauss):
    """``log N_k(x_n) + log pi_k`` at every stored pair; NaN without a positive-definite Gaussian."""
    out = np.full(p["row"].size, np.nan)
    for k, e in groups(p["galaxy"]):
        if int(k) in gauss:
            mean, cov, w, _ = gauss[int(k)]
            try:
                out[e] = log_gauss(X[p["row"][e].astype(np.int64)], mean, cov) + np.log(w)
            except np.linalg.LinAlgError:
                pass
    return out


# --- candidate pairs ------------------------------------------------------------------------------

def pair_entries(rows, gal, use):
    """Entry indices ``(a, b)``, in the order ``gal[a] < gal[b]``, of every candidate pair of each
    particle with >= 2 ``use`` entries in ``rows``. 3.2b and 3.4 use the entries themselves, to gather
    their own quantities per entry."""
    idx = np.flatnonzero(use)
    o, start, count = by_row(rows[idx])
    idx = idx[o]
    a, b = [], []
    for c in range(2, count.max(initial=1) + 1):
        base = start[count == c]
        for i, j in itertools.combinations(range(c), 2):
            a.append(idx[base + i])
            b.append(idx[base + j])
    if not a:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    a, b = np.concatenate(a), np.concatenate(b)
    swap = gal[a] > gal[b]
    return np.where(swap, b, a), np.where(swap, a, b)


def contested_pairs(p, logG, keep, ids):
    """3.2: ``(k, j, D_G, D_alpha, Delta log alpha, particle ID, entry a, entry b)`` over every candidate
    pair (k < j, a and b their original entry indices) of particles with >= 2 usable entries, and
    (contested particles, of them with an entry left out). ``ids`` are the snapshot's particle IDs by
    row (``row_labels``), to join pairs across snapshots."""
    rows, gal = p["row"].astype(np.int64), p["galaxy"].astype(np.int64)
    use = keep & np.isfinite(p["log_resp"]) & np.isfinite(logG)
    n_all = np.bincount(rows)
    n_use = np.bincount(rows[use], minlength=n_all.size)
    counts = (int((n_all >= 2).sum()), int(((n_all >= 2) & (n_use < n_all)).sum()))

    a, b = pair_entries(rows, gal, use)
    dG = logG[a] - logG[b]
    dA = (p["log_resp"][a] - p["log_resp"][b]) - dG
    dLa = p["log_alpha"][a] - p["log_alpha"][b]
    return (gal[a], gal[b], dG, dA, dLa, ids[rows[a]], a, b), counts


def best(rows, g, x):
    """Per particle (entries in any order): its row, its argmax galaxy, and the normalised top value
    of ``exp(x)``."""
    o, start, count = by_row(rows)
    rows, g, x = rows[o], g[o], x[o]
    seg = np.repeat(np.arange(start.size), count)
    mx = np.maximum.reduceat(x, start)
    lse = mx + np.log(np.add.reduceat(np.exp(x - mx[seg]), start))
    top = np.lexsort((-x, seg))[start]
    return rows[top], g[top], np.exp(x[top] - lse)


def mass_bin(values):
    """Goal 3's stellar-mass bins of ``values``, for the tables by mass."""
    return pd.cut(values, MASS_BINS, labels=MASS_LABELS)
