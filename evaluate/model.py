"""Goal 3: is the probabilistic model sound? It measures and reports; it gives no verdict.

    python3 evaluate/model.py --path RUN [--particles DIR] [--workers N] [--snaps A:B] [--every N]
                              [--feedback] [--dependence] [--light-every N]

The E-step is ``log r_nk = log N_k(x_n) + log pi_k + log alpha_nk - log Z_n`` (mixture/base.py), with
``Z_n = sum_j pi_j N_j(x_n) alpha_nj`` the particle's normaliser. The stored covariance is the inverse
of the precision the Bayesian GMM's E-step uses, so the quadratic term of log N is exact; its other
terms differ from the E-step's by a constant per galaxy, C_k, which cancels in everything below.
Statistics are oriented "more bound = larger"; Spearman correlations only see their order.

Scope: every --every-th snapshot (default 10) with >= 1e5 particles and a snapshot >= 150 Myr later;
galaxies with a stored, positive-definite Gaussian; resolved groups only. Groups are approximated by
the galaxies linked through shared bound particles; a group of K >= 2 galaxies is unresolved with
fewer than 10 K particles (the pipeline's ratio): its responsibilities are then the prior alone and its
Gaussians sample estimates, so none of the readings below holds there. (The pipeline's groups, from
overlapping spheres, can be larger, so this can leave out a little of a resolved one.) 3.2b and 3.4
(flags) run in a pass of their own over every --light-every-th snapshot (default 3; 1: all of them).

3.1 overlap inside a galaxy. For every galaxy with sum r >= 50 over its pairs: the Spearman correlation
    of log N_k with each statistic, weighted by r (soft) and over its >= 50 hard members (hard):
      depth      -E, the order of b, of t = -ln w and of the rank of b (energy, phase, rank models);
      -u^2       u^2 = v^2 / (v^2 - 2E), v relative to the tree velocity, the energy's frame with
                 pinned or no stars (kinematic model; not measured with free stars);
      log alpha  when the run stored it.
    Reported per statistic: galaxy-snapshots, median rho (soft, hard), soft-mass-weighted median,
    share with rho > 0.8; medians by stellar mass (sum m r) and central / satellite.
3.2 overlap where decisions are made. For every pair (k, j) of candidates of a particle with >= 2:
    D_G = Delta(log N + log pi), D_alpha = Delta log r - D_G (= Delta log alpha + C_k - C_j). Per
    (k, j, s) with >= 30 particles: rho = Pearson(D_G, D_alpha), and alpha's share of the decision
    v = Cov(D_alpha, D_G + D_alpha) / Var(D_G + D_alpha) = (Var D_alpha + Cov) / Var(D_G + D_alpha)
    (the shares of the two terms add to 1). Reported: n-weighted medians, n-weighted share with
    rho > 0.5, share of contested particles with a pair left out (underflowed r, no Gaussian,
    unresolved group); with stored log alpha, the median Pearson(D_alpha, Delta log alpha) (1 when C_k
    is constant per galaxy). Autocorrelation: is D_G smooth in time while D_alpha jitters, or the other
    way round? The same pair's D_G, D_alpha at s and s' = later[s] (150 Myr), joined on the particle
    ID (Sub_tree_id pairs and IDs are both stable across snapshots); per (k, j, s) with >= 30 common
    stars, ac_G = Pearson(D_G(s), D_G(s')), ac_alpha = Pearson(D_alpha(s), D_alpha(s')) (Pearson is
    invariant to the per-snapshot constant C_k - C_j, so exact). Reported: n-weighted medians of each,
    n-weighted share with ac_alpha < ac_G (alpha the flickier term).
3.2b does alpha feed back into the fit, or only gate it (flag --feedback; needs stored log alpha)? Each
    resolved component is replayed from its own written fit theta_A: the prior rebuilt with priors.py
    from the fit at the previous snapshot (bgmm.py's ``state``, mirroring the pipeline's
    ``_build_prior_kwargs`` / ``_history``), theta_A read as a posterior (written mean, covariance,
    count). One E-step with alpha at theta_A is checked against the written log r of the contested
    entries (the reproduction residual: median |delta log r|; the tail as p99 |delta r|, since log
    differences between underflowed r ~ 1e-12 mean nothing); components whose median |delta log r|
    > 1e-2 are dropped (counted). Two chains are then iterated to convergence from theta_A (<= 300 EM
    steps): A with alpha in every E-step (the run as written), B with alpha's support only, alpha
    applied once at the end. Per pair (k, j) with >= 30 stars, the leak L = (g^A_k - g^A_j) -
    (g^B_k - g^B_j), how far alpha in EM moved the Gaussians' own log-odds, with every moment weighted by
    p (1 - p), p = sigmoid(D_A), so that only the stars in play count (deep in a Gaussian's tail the
    log-odds move by thousands of nats without changing anything): sd(L) in nats, its size;
    f = Var(L) / Var(D_A) (D_A: chain A's log-odds), the share of the decision that moves (large when
    D_A barely varies); rho_L = corr(L, Delta log alpha), positive when the Gaussians absorbed alpha
    (double counting); gamma = Cov(L, Delta log alpha) / Var(Delta log alpha), A's log-odds being
    D_G^B + (1 + gamma) Delta log alpha (alpha counted 1 + gamma times), kept where Var(Delta log alpha)
    >= 0.01 Var(D_A); sd(L) also after one EM step (the linear response). Per galaxy: symmetric
    KL(A, B), Delta log|Sigma|, the mean shift in A's sigma, Delta N, log(M_B / M_A) (M: the
    responsibility-weighted stellar mass) and KL(written, A) (how converged the pipeline's own fit
    was). Labels: the mass share of contested particles whose argmax differs between A and B (of the
    contested, the snapshot's total and the host's contested mass). Sharpening: the share of contested
    mass with r_max >= 0.999 and the mass-weighted mean entropy, A against B. Coverage: the replayed
    share of resolved contested mass, the share in no-history galaxies (no usable previous fit), the
    excluded shares, iterations (and how many hit the cap). Reported: these SCOREs (per-galaxy medians
    weighted by contested mass, over components with contested entries; pair medians by the stars in
    play), a table by stellar mass, the 10 pairs with the largest sd(L) x stars in play.
3.3 calibration. Contested (>= 2 candidates) old (present >= 1 Gyr before) stars at s, k = argmax r;
    outcome: labelled k at s', the first snapshot >= 150 Myr later. Left out: k without particles then,
    unassigned then, a satellite's star beyond its catalogue r_t then. Mass-weighted per bin of r_max:
    mean r, persistence p, gap = mean r - p. Reported: the table, overconfidence
    O = sum n max(gap, 0) / sum n, apparent underconfidence U (inflated: EM starts from the previous
    responsibilities, and wrong labels can persist), persistence at r_max >= 0.999. Lag curve: the same
    selection and argmax, the outcome read at lags 0.05, 0.15 (= 3.3 itself), 0.3, 0.6, 1.0 Gyr, pooled
    over the snapshots that have every lag (so each lag follows the same stars); reported as mean r (the
    prediction, unconditional on any outcome) and persistence per lag, separating fast reversible
    flicker (a drop then a plateau) from slow loss.
3.4 how much of alpha the Gaussians already contain, when the run stored log alpha (else D_alpha; the
    two differ by a constant per pair, which an intercept absorbs). For every pair (k, j, s) with
    >= 500 stars (MIN_DEP), resolved: Y = Delta log alpha; the Gaussian view F_G = (d2_k, d2_j), each
    star's Mahalanobis^2 to its two galaxies' written Gaussians; the physical view F_P = (r_k, |v - v_k|,
    r_j, |v - v_j|) relative to the tree halos (what alpha is built from). R2_G (out-of-sample, cross-fit
    HistGradientBoostingRegressor; bias-corrected below DEP_BOOT stars) is the redundancy already in the
    Gaussians; R2_phys is a capacity check of the regressor (expected ~1); the unique share
    u = Var(Y - fhat_G(F_G)) / Var(D), D = Delta log r = D_G + D_alpha, is alpha's push the Gaussians
    cannot express, as a share of the decision's own variance. With --feedback, also R2_G with chain B's
    Gaussians (3.2b): R2_G(A) - R2_G(B) is the redundancy feedback creates. Reported: pairs measured and
    only counted (< 500 stars); n-weighted medians of R2_G, R2_phys, u (and R2_G(B)), separately for
    pairs >= DEP_BOOT (raw) and 500..DEP_BOOT (corrected, with their median standard error); a table by
    the smaller galaxy's stellar mass.
3.5 tempering, when log alpha is stored and the run's own beta (header "tempering", 1 when absent) is
    > 0: the stored log alpha is beta_run log alpha; 3.3 with log r' = log r + (beta - beta_run) log alpha
    renormalised, for beta = 1, 0.75, 0.5, 0.25 other than the run's (the final E-step with the fitted
    Gaussians; EM would move them), and the share of the stars whose argmax changes. 3.3 itself is the
    run as it is (its own beta).

Writes evaluation/model.txt and .log; plots/run/: model_overlap.pdf (3.1 and 3.2, including its
autocorrelation, against time), model_calibration.pdf (reliability curves, O and U against time),
model_persistence.pdf (3.3's lag curve, one line per r_max bin), model_feedback.pdf (3.2b: sd(L),
rho_L and gamma, label-change share, dlogdet and the reproduction residual against time),
model_dependence.pdf (3.4: R2_G, R2_phys, u against time); plots/g<id>/model_overlap.pdf (its 3.1 rho
and its 3.2 rho and v with its partners, against time), model_feedback.pdf (its dN, dlogdet, and sd(L)
and rho_L with its 5 largest partners), model_dependence.pdf (R2_G and u with its 5 largest partners),
for the plotted galaxies.
"""
import itertools
import sys
from pathlib import Path

if __package__ in (None, ""):                   # run as a file: make the repository root importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import solve_triangular
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingRegressor

from evaluate import bgmm, cli, plots
from evaluate.load import ABSENT, OLD_GYR, UNASSIGNED, Run, pool, seen_before, wmedian
from evaluate.report import Report

MIN_PARTICLES, MIN_R, MIN_HARD, MIN_PAIR = 1e5, 50.0, 50, 30
REPRO_TOL = 1e-2                        # 3.2b: a component is dropped when its median |delta log r| exceeds this
SHARP = 0.999                           # 3.2b: r_max this high or above counts as "sharp"
MAX_ITER = 300                          # 3.2b: bgmm.replay's cap, and the "hit max_iter" (non-converged) count
LATER_GYR = 0.15                        # 3.3: the outcome is read at the first snapshot this much later
LAGS_GYR = (0.05, 0.15, 0.3, 0.6, 1.0)  # 3.3's lag curve (0.15 = LATER_GYR itself)
UNRESOLVED_RATIO = 10                   # the pipeline's UNRESOLVED_GROUP_RATIO
STRONG_GALAXY, STRONG_PAIR = 0.8, 0.5
EDGES = np.array([0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999, 1.0 + 1e-9])
BIN_LABELS = ["[0,0.5)", "[0.5,0.6)", "[0.6,0.7)", "[0.7,0.8)", "[0.8,0.9)", "[0.9,0.95)",
              "[0.95,0.99)", "[0.99,0.999)", ">=0.999"]
MASS_BINS = np.array([0.0, 1e7, 1e8, 1e9, np.inf])
BETAS = (1.0, 0.75, 0.5, 0.25)
MIN_DEP = 500                            # 3.4: minimum stars per (k, j) to fit the regressors
DEP_BOOT, DEP_B = 5000, 4                # 3.4: below DEP_BOOT stars, R^2 is bias-corrected by DEP_B copies
# DEP_B was 8; the bootstrap fits (HistGradientBoostingRegressor x 2 folds x DEP_B copies) are most of
# 3.4's cost, so DEP_B is halved to 4 -- the bias estimate and its standard error get noisier by sqrt(2).
DEP_CAP = 20_000                         # 3.4: a pair's stars are capped by a fixed-seed random sample
HGB_KW = dict(max_iter=300, learning_rate=0.1, max_leaf_nodes=15, early_stopping=True,  # 3.4's regressor
              validation_fraction=0.15, n_iter_no_change=10, random_state=0)
STATS = ("depth", "-u^2", "log alpha")
POS = ["position_x", "position_y", "position_z"]
VEL = ["velocity_x", "velocity_y", "velocity_z"]


# --- shared pieces --------------------------------------------------------------------------------

def log_gauss(x, mean, cov):
    """log N(x | mean, cov) of the rows of ``x``; ``cov`` full (d, d), diagonal (d,) or spherical ()."""
    cov = np.diag(np.broadcast_to(cov, mean.shape)) if np.ndim(cov) < 2 else cov
    L = np.linalg.cholesky(cov)
    z = solve_triangular(L, (x - mean).T, lower=True)
    return -0.5 * (z * z).sum(0) - np.log(np.diag(L)).sum() - 0.5 * mean.size * np.log(2.0 * np.pi)


def mahalanobis2(x, mean, cov):
    """``(x - mean)^T cov^-1 (x - mean)`` of the rows of ``x``; ``cov`` full (d, d), diagonal (d,) or
    spherical () (same Cholesky pattern as ``log_gauss``, without the normalising terms)."""
    cov = np.diag(np.broadcast_to(cov, mean.shape)) if np.ndim(cov) < 2 else cov
    L = np.linalg.cholesky(cov)
    z = solve_triangular(L, (x - mean).T, lower=True)
    return (z * z).sum(0)


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
    component is resolved (see Scope): K >= 2 galaxies with fewer than 10 K particles are unresolved."""
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


def pd_ok(cov):
    """Whether ``cov`` is positive-definite (a Cholesky succeeds): a stored Gaussian can fail this even
    with a finite count."""
    try:
        np.linalg.cholesky(cov)
        return True
    except np.linalg.LinAlgError:
        return False


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


def galaxy_d2(p, X, gauss):
    """3.4: Mahalanobis^2 of each stored pair to its own galaxy's Gaussian; NaN without one (same
    pattern as ``galaxy_terms``, without the weight and normalising terms)."""
    out = np.full(p["row"].size, np.nan)
    for k, e in groups(p["galaxy"]):
        if int(k) in gauss:
            mean, cov, _, _ = gauss[int(k)]
            try:
                out[e] = mahalanobis2(X[p["row"][e].astype(np.int64)], mean, cov)
            except np.linalg.LinAlgError:
                pass
    return out


def wspearman(a, b, w):
    """Spearman correlation weighted by ``w`` (the weighted Pearson of the ranks)."""
    ra, rb = rankdata(a), rankdata(b)
    da, db = ra - np.average(ra, weights=w), rb - np.average(rb, weights=w)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (np.average(da * db, weights=w)
                / np.sqrt(np.average(da * da, weights=w) * np.average(db * db, weights=w)))


# --- 3.1 ----------------------------------------------------------------------------------------

def within(p, logG, stats, lab, m, satellites, keep):
    """3.1: one row per galaxy, from its usable pairs (``keep``: in a resolved group)."""
    r = np.exp(p["log_resp"])
    rows = p["row"].astype(np.int64)
    out = []
    for k, e in groups(p["galaxy"]):
        e = e[keep[e] & np.isfinite(logG[e]) & (r[e] > 0)]
        if r[e].sum() < MIN_R:
            continue
        hard = lab[rows[e]] == k
        row = {"galaxy": int(k), "mass": float((m[rows[e]] * r[e]).sum()), "satellite": int(k) in satellites}
        for name, x in stats.items():
            ok = np.isfinite(x[e])
            h = ok & hard
            row[f"soft {name}"] = (wspearman(logG[e][ok], x[e][ok], r[e][ok])
                                   if r[e][ok].sum() >= MIN_R else np.nan)
            row[f"hard {name}"] = (wspearman(logG[e][h], x[e][h], np.ones(h.sum()))
                                   if h.sum() >= MIN_HARD else np.nan)
        out.append(row)
    return out


# --- 3.2 ----------------------------------------------------------------------------------------

def pair_entries(rows, gal, use):
    """Entry indices ``(a, b)``, canonical order ``gal[a] < gal[b]``, of every candidate pair of each
    particle with >= 2 ``use`` entries in ``rows``. Shared by ``contested_pairs`` and the feedback
    worker, which also needs the entries themselves to gather per-entry quantities (d2, physical
    features, replayed g) rather than just the combined ``(k, j)`` statistics ``contested_pairs``
    reduces to."""
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


def pair_table(k, j, dG, dA, dLa, s, t):
    """3.2 per (k, j) at ``s`` with >= 30 particles: n, rho, v (and the stored-alpha check)."""
    if k.size == 0:
        return pd.DataFrame()
    key, inv = np.unique(np.stack([k, j]), axis=1, return_inverse=True)
    inv = inv.ravel()

    def S(v):
        return np.bincount(inv, weights=v, minlength=key.shape[1])

    n = S(np.ones(k.size))

    def moments(x, y):
        """Per pair: variances of x and y, and their covariance (centred, two passes)."""
        dx, dy = x - (S(x) / n)[inv], y - (S(y) / n)[inv]
        return S(dx * dx) / n, S(dy * dy) / n, S(dx * dy) / n

    vG, vA, cov = moments(dG, dA)
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = cov / np.sqrt(vG * vA)
        v = (vA + cov) / (vG + vA + 2.0 * cov)
        t_ = pd.DataFrame({"snap": s, "t": t, "k": key[0], "j": key[1], "n": n, "rho": rho, "v": v})
        if np.isfinite(dLa).all():
            vS, vL, cS = moments(dA, dLa)
            t_["rho stored"] = cS / np.sqrt(vS * vL)
    return t_[t_["n"] >= MIN_PAIR]


def autocorr_table(k1, j1, pid1, dG1, dA1, k2, j2, pid2, dG2, dA2, s, t):
    """3.2's autocorrelation: the contested entries of ``s`` and ``s'`` (150 Myr later), joined on
    (k, j, particle ID); per (k, j) with >= 30 common stars, the Pearson autocorrelation of D_G and
    D_alpha (exact despite the per-snapshot constants, which Pearson is invariant to)."""
    if k1.size == 0 or k2.size == 0:
        return pd.DataFrame()
    a = pd.DataFrame({"k": k1, "j": j1, "pid": pid1, "dG1": dG1, "dA1": dA1})
    b = pd.DataFrame({"k": k2, "j": j2, "pid": pid2, "dG2": dG2, "dA2": dA2})
    m = a.merge(b, on=["k", "j", "pid"])
    if m.empty:
        return pd.DataFrame()
    key, inv = np.unique(np.stack([m["k"].to_numpy(), m["j"].to_numpy()]), axis=1, return_inverse=True)
    inv = inv.ravel()

    def S(v):
        return np.bincount(inv, weights=v, minlength=key.shape[1])

    n = S(np.ones(len(m)))

    def pearson(x, y):
        dx, dy = x - (S(x) / n)[inv], y - (S(y) / n)[inv]
        return (S(dx * dy) / n) / np.sqrt((S(dx * dx) / n) * (S(dy * dy) / n))

    with np.errstate(invalid="ignore", divide="ignore"):
        t_ = pd.DataFrame({"snap": s, "t": t, "k": key[0], "j": key[1], "n": n,
                           "ac_G": pearson(m["dG1"].to_numpy(), m["dG2"].to_numpy()),
                           "ac_alpha": pearson(m["dA1"].to_numpy(), m["dA2"].to_numpy())})
    return t_[t_["n"] >= MIN_PAIR]


# --- 3.3 and 3.4 ----------------------------------------------------------------------------------

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


def outcome(run, s2, ids, k):
    """3.3 at the later snapshot: (counted, persisted) for the stars ``ids`` predicted in galaxies ``k``."""
    ids2, lab2 = run.row_labels(s2)
    X2, _ = run.particles(s2)
    o = np.argsort(ids2)
    j = o[np.minimum(np.searchsorted(ids2, ids, sorter=o), ids2.size - 1)]
    found = ids2[j] == ids
    L2 = np.where(found, lab2[j], ABSENT)
    gal = run.galaxies(s2)
    rt = gal["r_t"].reindex(k).to_numpy(np.float64)
    sat = np.isin(k, np.array(list(run.hosts(s2)), np.int64)) & (rt > 0)
    with np.errstate(invalid="ignore"):
        far = np.linalg.norm(X2[j, :3] - gal[POS].reindex(k).to_numpy(np.float64), axis=1) > rt
    counted = found & np.isin(k, np.unique(lab2)) & (L2 != UNASSIGNED) & ~(sat & far)
    return counted, L2 == k


def selected(p, old_rows, keep):
    """3.3's selection: contested (>= 2 candidates) old stars in a resolved group, and (rows, galaxy,
    log_resp) there."""
    rows = p["row"].astype(np.int64)
    sel = keep & (np.bincount(rows)[rows] >= 2) & old_rows[rows]
    return sel, rows[sel], p["galaxy"].astype(np.int64)[sel], p["log_resp"][sel]


def calibration(run, s, s2, p, m, old_rows, keep, beta_run):
    """3.3 and 3.4 at ``s``: mass-weighted bin sums per beta (``old_rows``: which rows hold old stars)."""
    C = {}
    sel, rows, g, lr = selected(p, old_rows, keep)
    if not sel.any():
        return C
    ids = run.row_labels(s)[0]
    la = p["log_alpha"][sel]
    stored = np.isfinite(la).all() and beta_run > 0
    k1 = best(rows, g, lr)[1]
    for beta in ([beta_run] + [b for b in BETAS if b != beta_run]) if stored else [beta_run]:
        shift = (beta - beta_run) * la / beta_run if stored else 0.0
        rr, k, rmax = best(rows, g, lr + shift)                  # rows come in the same order every beta
        counted, persisted = outcome(run, s2, ids[rr], k)
        w = m[rr] * counted
        b = np.digitize(rmax, EDGES) - 1
        size = EDGES.size - 1
        C[beta] = np.stack([np.bincount(b, weights=w, minlength=size),
                            np.bincount(b, weights=counted.astype(float), minlength=size),
                            np.bincount(b, weights=w * rmax, minlength=size),
                            np.bincount(b, weights=w * persisted, minlength=size)])
        C[f"changed {beta}"] = np.array([(w * (k != k1)).sum(), w.sum()])
    return C


def persistence(run, s, lags, p, m, old_rows, keep):
    """3.3's lag curve at ``s``: the run's own beta, the same selection and argmax as 3.3 (not recomputed
    per lag). Mass-weighted bins of r_max (mean r, unconditional: the prediction) and, per available lag,
    of counted and persisted mass."""
    sel, rows, g, lr = selected(p, old_rows, keep)
    size = EDGES.size - 1
    if not sel.any():
        return np.zeros(size), np.zeros(size), {}
    ids = run.row_labels(s)[0]
    rr, k, rmax = best(rows, g, lr)
    b = np.digitize(rmax, EDGES) - 1
    mass = np.bincount(b, weights=m[rr], minlength=size)
    mr = np.bincount(b, weights=m[rr] * rmax, minlength=size)
    out = {}
    for lag, s2 in lags.items():
        counted, persisted = outcome(run, s2, ids[rr], k)
        w = m[rr] * counted
        out[lag] = np.stack([np.bincount(b, weights=w, minlength=size),
                             np.bincount(b, weights=w * persisted, minlength=size)])
    return mass, mr, out


def over_under(sums):
    """Overconfidence O and apparent underconfidence U of one beta's bin sums."""
    mass, _, mr, mp = sums
    ok = mass > 0
    if not ok.any():
        return np.nan, np.nan
    gap = mr[ok] / mass[ok] - mp[ok] / mass[ok]
    return (float((mass[ok] * np.maximum(gap, 0)).sum() / mass.sum()),
            float((mass[ok] * np.maximum(-gap, 0)).sum() / mass.sum()))


# --- 3.4: how much of alpha the Gaussians already contain -----------------------------------------
#
# Per contested pair (k, j), alpha's push on the decision is Y = Delta log alpha (stored, when the run
# kept it; else D_alpha from 3.2 -- the two differ by a constant per pair, which an intercept absorbs).
# The Gaussian view is what the GMM sees of a star: its Mahalanobis distances to each galaxy, F_G =
# (d2_k, d2_j). The physical view is what alpha is built from: distances and speeds relative to the two
# tree halos, F_P = (r_k, |v - v_k|, r_j, |v - v_j|). R2_G on F_G is the redundancy already in the
# Gaussians; R2_phys on F_P is a capacity check of the regressor (expected ~1); the unique share
# u = Var(Y - fhat_G(F_G)) / Var(D), D = Delta log r = D_G + D_alpha, is alpha's push the Gaussians
# cannot express, as a share of the decision's own variance; u = (1 - R2_G) Var(Y) / Var(D).
# Out-of-sample R^2 is conservative: the early-stopped regressor, trained on half the stars, reads
# smooth alpha-like functions low by 0.02 at 5000 stars and 0.01 at 20 000 (left as is), more below. So
# under DEP_BOOT stars R2_G is corrected by the procedure's own bias at that size and shape, measured on
# synthetic copies with a known answer (a parametric bootstrap, ``corrected_r2``): tested on two shapes,
# within 0.03 of the truth from 500 stars (MIN_DEP), with a standard error that matches the scatter.
# Below 500 stars the correction falls short on bumpy shapes (0.11 at 200): those pairs are only counted.
# R2_phys, a capacity check, stays raw (conservative).

def cross_fit_r2(F, Y, seed):
    """Out-of-sample R^2 of ``Y`` on the rows of ``F``, by 2-fold cross-fitting a
    ``HistGradientBoostingRegressor`` (fixed hyperparameters, ``HGB_KW``, deterministic): each half
    predicted by the model fitted on the other half. Returns (r2, the cross-fitted predictions)."""
    n = Y.size
    fold = np.random.default_rng(seed).integers(0, 2, n)
    yhat = np.empty(n)
    for f in (0, 1):
        tr, te = fold != f, fold == f
        if not te.any():
            continue
        if tr.any():
            reg = HistGradientBoostingRegressor(**HGB_KW)
            reg.fit(F[tr], Y[tr])
            yhat[te] = reg.predict(F[te])
        else:
            yhat[te] = Y.mean()
    ss_res, ss_tot = ((Y - yhat) ** 2).sum(), ((Y - Y.mean()) ** 2).sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        return (1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan), yhat


def corrected_r2(F, Y, seed):
    """``cross_fit_r2`` minus the procedure's own bias at this size and shape, and that bias's spread
    (the standard error): the bias is measured on DEP_B synthetic copies with a known answer,
    Y* = fhat(F) + residuals resampled from the cross-fit (fhat: the regressor fitted on all the stars,
    in-sample, while the resampled residuals are the out-of-sample cross-fit ones, so Var(Y*) sits
    slightly above Var(Y); the calibration stays internally consistent, since it is corrected against
    this same Y*), whose true R^2 is Var(fhat) / (Var(fhat) + Var(residuals))."""
    r2, yhat = cross_fit_r2(F, Y, seed)
    fhat = HistGradientBoostingRegressor(**HGB_KW).fit(F, Y).predict(F)
    res = Y - yhat
    rng = np.random.default_rng(seed + 1)
    bias = []
    for b in range(DEP_B):
        eps = rng.choice(res, res.size)
        truth = np.var(fhat) / (np.var(fhat) + np.var(eps))
        bias.append(cross_fit_r2(F, fhat + eps, seed + 2 + b)[0] - truth)
    return r2 - float(np.mean(bias)), float(np.std(bias))


def dependence_table(k, j, Y, d2k, d2j, phys, D, s, t, seed, corrected=True):
    """3.4 per (k, j) at ``s`` with >= 500 entries (``MIN_DEP``): n, R2_G (the Gaussian view ``d2k``,
    ``d2j`` alone; bias-corrected below DEP_BOOT stars, with its standard error ``r2_G se``), R2_phys (the
    physical view, ``phys``'s four columns; raw) and the unique share u = (1 - R2_G) Var(Y) / Var(``D``),
    mirroring ``pair_table``'s grouping. Each pair's stars are capped at 20 000 (``DEP_CAP``) by a
    fixed-seed random sample (unbiased; this costs the R^2 estimates ~1e-3). Pairs under MIN_DEP are
    only counted: (pairs, their stars). ``corrected=False`` (3.2b's R2_G(B), a secondary cross-check
    whose own bootstrap would double 3.4's cost): pairs under DEP_BOOT are left out (only counted) too,
    rather than bias-corrected -- only the raw, large pairs are computed."""
    if k.size == 0:
        return pd.DataFrame(), (0, 0)
    key, inv = np.unique(np.stack([k, j]), axis=1, return_inverse=True)
    inv = inv.ravel()
    rng = np.random.default_rng(seed)
    rows, few = [], [0, 0]
    for idx in range(key.shape[1]):
        e = np.flatnonzero(inv == idx)
        if e.size < MIN_DEP or (not corrected and e.size < DEP_BOOT):
            few[0] += 1
            few[1] += e.size
            continue
        if e.size > DEP_CAP:
            e = rng.choice(e, DEP_CAP, replace=False)
        y, d = Y[e], D[e]
        FG = np.stack([d2k[e], d2j[e]], axis=1)
        r2_G, se = (corrected_r2(FG, y, seed) if (corrected and e.size < DEP_BOOT)
                   else (cross_fit_r2(FG, y, seed)[0], np.nan))
        r2_phys, _ = cross_fit_r2(phys[e], y, seed)
        with np.errstate(invalid="ignore", divide="ignore"):
            u = float((1.0 - r2_G) * np.var(y) / np.var(d))
        rows.append({"snap": s, "t": t, "k": key[0, idx], "j": key[1, idx], "n": e.size,
                     "r2_G": r2_G, "r2_G se": se, "r2_phys": r2_phys, "u": u})
    return pd.DataFrame(rows), tuple(few)


# --- 3.2b: does alpha feed back into the fit? --------------------------------------------------------

def sym_kl(mA, SA, mB, SB):
    """Symmetric KL(A, B) + KL(B, A) between two full-covariance Gaussians, batched over the leading
    axis; the log-determinant terms cancel, leaving the trace and Mahalanobis terms only."""
    D = mA.shape[-1]
    SAi, SBi = np.linalg.inv(SA), np.linalg.inv(SB)
    diff = mA - mB
    tr = (np.trace(SBi @ SA, axis1=-2, axis2=-1) + np.trace(SAi @ SB, axis1=-2, axis2=-1))
    maha = np.einsum("...i,...ij,...j->...", diff, SAi + SBi, diff)
    return 0.5 * (tr - 2.0 * D + maha)


def entropy_by_row(rows, log_r):
    """Per-particle Shannon entropy ``-sum r log r`` of its own candidates' responsibilities (``rows``,
    ``log_r``: one entry per candidate); its row and the entropy, one row per particle."""
    o, start, count = by_row(rows)
    lr = log_r[o]
    contrib = np.where(lr > -np.inf, -np.exp(lr) * lr, 0.0)
    return rows[o][start], np.add.reduceat(contrib, start)


GAMMA_MIN_VAR = 0.01                    # 3.2b: gamma needs Var(Delta log alpha) >= this x Var(D_A)


def gamma_table(k, j, L, dLa, DA, s, t):
    """3.2b per (k, j) at ``s`` with >= 30 particles (``MIN_PAIR``), mirroring ``pair_table``'s moments,
    each star weighted by p (1 - p), p = sigmoid(D_A) (``D_A``: chain A's log-odds, g_r_A[a] - g_r_A[b]):
    the sensitivity of the pair's probability to its log-odds, so the stars in play count and those far
    from the boundary do not (deep in a Gaussian's tail their log-odds move by thousands of nats without
    changing anything). n_play = sum p (1 - p), the stars in play. sd(L) in nats, how far alpha in EM moved
    the Gaussians' log-odds (the feedback's size); f = Var(L) / Var(D_A), the share of the decision that
    moves; rho_L = corr(L, Delta log alpha) (positive: the Gaussians absorbed alpha); gamma =
    Cov(L, Delta log alpha) / Var(Delta log alpha), ill-conditioned when Var(Delta log alpha) is tiny, so
    kept only when it is >= 0.01 Var(D_A) (``GAMMA_MIN_VAR``), NaN otherwise."""
    if k.size == 0:
        return pd.DataFrame()
    key, inv = np.unique(np.stack([k, j]), axis=1, return_inverse=True)
    inv = inv.ravel()
    e = np.exp(-np.abs(DA))
    w = e / (1.0 + e) ** 2                                  # p (1 - p), overflow-free

    def S(v):
        return np.bincount(inv, weights=v, minlength=key.shape[1])

    n, n_play = S(np.ones(k.size)), S(w)

    def moments(x, y):
        dx, dy = x - (S(w * x) / n_play)[inv], y - (S(w * y) / n_play)[inv]
        return S(w * dx * dx) / n_play, S(w * dy * dy) / n_play, S(w * dx * dy) / n_play

    vL, vY, cov = moments(L, dLa)
    vDA = moments(DA, DA)[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        f = vL / vDA
        rho_L = cov / np.sqrt(vL * vY)
        gamma = np.where(vY >= GAMMA_MIN_VAR * vDA, cov / vY, np.nan)
        t_ = pd.DataFrame({"snap": s, "t": t, "k": key[0], "j": key[1], "n": n, "n_play": n_play,
                           "sd_L": np.sqrt(vL), "f": f, "rho_L": rho_L, "gamma": gamma})
    return t_[t_["n"] >= MIN_PAIR]


def feedback_component(run, s, p, X, m, galaxies, gauss, gauss_prev, tree_xyzv, tree_mass,
                        tree_mass_prev, dependence=False):
    """3.2b for one resolved component (``galaxies``: its ids, all with a stored, finite-count Gaussian
    at ``s``): the reproduction residual, the two chains A/B and the measurements against each other.
    ``tree_xyzv``: (n_galaxies, 6) position + velocity, aligned with ``galaxies``.

    Returns a dict of the per-galaxy rows, the converged and one-step gamma tables, the label/sharpening
    mass sums, the reproduction residuals and the iterations, or None when no entry of this component is
    usable (no stored pairs)."""
    e = np.isin(p["galaxy"].astype(np.int64), galaxies)
    rows, gal, la, lr_written = (p[k][e] for k in ("row", "galaxy", "log_alpha", "log_resp"))
    rows, gal = rows.astype(np.int64), gal.astype(np.int64)
    if not rows.size:
        return None

    contested = np.bincount(rows)[rows] >= 2
    contested_mass = float(m[np.unique(rows[contested])].sum())

    tree_xyzv_dict = dict(zip(galaxies.tolist(), tree_xyzv))
    prior, theta_A, no_history = bgmm.state(gal, galaxies, gauss, gauss_prev, tree_xyzv_dict, tree_mass,
                                             tree_mass_prev)
    delta, delta_r = bgmm.reproduce(X, rows, gal, galaxies, la, theta_A, lr_written)
    delta_c, delta_r_c = delta[contested], delta_r[contested]   # single-candidate entries are always
    ok_c = np.isfinite(delta_c)                                  # reproduced exactly and would swamp these

    post_A, first_A, it_A, rows_c, gal_c, g_r_A, g_A = bgmm.replay(X, rows, gal, galaxies, la, prior,
                                                                    theta_A, with_alpha=True, max_iter=MAX_ITER)
    post_B, first_B, it_B, _, _, g_r_B, g_B = bgmm.replay(X, rows, gal, galaxies, la, prior, theta_A,
                                                           with_alpha=False, max_iter=MAX_ITER)

    la_c = la[contested]
    g1_A, _ = bgmm.estep(X, rows_c, gal_c, galaxies, la_c, first_A, with_alpha=True)
    g1_B, _ = bgmm.estep(X, rows_c, gal_c, galaxies, la_c, first_B, with_alpha=True)

    t = run.time[s]
    gal_row = {"snap": s, "t": t, "galaxy": galaxies, "no_history": no_history, "mass": contested_mass,
               "kl": sym_kl(post_A.m, post_A.Sigma, post_B.m, post_B.Sigma),
               "kl_written": sym_kl(theta_A.m, theta_A.Sigma, post_A.m, post_A.Sigma),
               "dlogdet": (np.linalg.slogdet(post_B.Sigma)[1] - np.linalg.slogdet(post_A.Sigma)[1]),
               "shift": np.sqrt(np.einsum("ki,kij,kj->k", post_B.m - post_A.m, np.linalg.inv(post_A.Sigma),
                                           post_B.m - post_A.m)),
               "dN": post_B.N - post_A.N}

    fixed = ~contested
    idx = np.searchsorted(galaxies, gal)
    M_fixed = np.bincount(idx[fixed], weights=m[rows[fixed]], minlength=galaxies.size)
    idx_c = np.searchsorted(galaxies, gal_c)
    M_A = M_fixed + np.bincount(idx_c, weights=m[rows_c] * np.exp(g_r_A), minlength=galaxies.size)
    M_B = M_fixed + np.bincount(idx_c, weights=m[rows_c] * np.exp(g_r_B), minlength=galaxies.size)
    with np.errstate(invalid="ignore", divide="ignore"):
        gal_row["dlogM"] = np.log(M_B) - np.log(M_A)

    a, b = pair_entries(rows_c, gal_c, np.ones(rows_c.size, bool))
    dLa_pair = la_c[a] - la_c[b]
    DA = g_r_A[a] - g_r_A[b]
    gam = gamma_table(gal_c[a], gal_c[b], (g_A[a] - g_A[b]) - (g_B[a] - g_B[b]), dLa_pair, DA, s, t)
    gam1 = gamma_table(gal_c[a], gal_c[b], (g1_A[a] - g1_A[b]) - (g1_B[a] - g1_B[b]), dLa_pair, DA, s, t)

    dep_B = pd.DataFrame()
    if dependence and a.size:
        Xr = X[rows_c]
        Sigma_inv_B = np.linalg.inv(post_B.Sigma)
        diff = Xr - post_B.m[idx_c]
        d2_B = np.einsum("ni,nij,nj->n", diff, Sigma_inv_B[idx_c], diff)
        dG_B = g_B[a] - g_B[b]
        dA_B = (g_r_B[a] - g_r_B[b]) - dG_B
        pos_c, vel_c = tree_xyzv[idx_c, :3], tree_xyzv[idx_c, 3:]
        r_all = np.linalg.norm(Xr[:, :3] - pos_c, axis=1)
        v_all = np.linalg.norm(Xr[:, 3:] - vel_c, axis=1)
        phys = np.stack([r_all[a], v_all[a], r_all[b], v_all[b]], axis=1)
        dep_B, _ = dependence_table(gal_c[a], gal_c[b], dLa_pair, d2_B[a], d2_B[b], phys,
                                     dG_B + dA_B, s, t, seed=s, corrected=False)

    rr_A, k_A, rmax_A = best(rows_c, gal_c, g_r_A)
    _, k_B, rmax_B = best(rows_c, gal_c, g_r_B)
    _, h_A = entropy_by_row(rows_c, g_r_A)
    _, h_B = entropy_by_row(rows_c, g_r_B)
    w = m[rr_A]
    changed = k_A != k_B
    host_mask = np.isin(rr_A, np.unique(rows_c[np.isin(gal_c, [run.host])]))

    labels = {"w": w, "changed": changed, "host": host_mask, "sharp_A": rmax_A >= SHARP,
              "sharp_B": rmax_B >= SHARP, "entropy_A": h_A, "entropy_B": h_B}
    repro = {"median": wmedian(delta_c[ok_c], np.ones(ok_c.sum())) if ok_c.any() else np.nan,
             "p99_r": float(np.nanpercentile(delta_r_c, 99)) if ok_c.any() else np.nan, "n": int(ok_c.sum())}
    no_history_mass = float(w[np.isin(k_A, galaxies[no_history])].sum())
    return dict(gal_row=gal_row, gamma=gam, gamma1=gam1, dep_B=dep_B, labels=labels, repro=repro,
                iterations=(it_A, it_B), resolved_mass=contested_mass, no_history_mass=no_history_mass)


# --- the workers ----------------------------------------------------------------------------------

_W = {}                                 # a worker's run and settings


def _init(path, source, snaps, frame, beta_run, vel, later, lags):
    run = Run(path, particles=source)
    run.snaps = snaps
    _W.update(run=run, frame=frame, beta_run=beta_run, vel=vel, later=later, lags=lags)


def _snapshot(s):
    """Goal 3 at ``s`` in a worker: galaxy rows, pair table, contested counts, calibration sums, the 3.2
    autocorrelation table, the 3.3 lag-curve sums and the particle count of every label (None with too
    few particles)."""
    run = _W["run"]
    ids, lab = run.row_labels(s)
    if ids.size < MIN_PARTICLES:
        return None
    p = run.pairs(s)
    X, m = run.particles(s)
    _, keep = components(p)
    logG = galaxy_terms(p, X, run.gaussians(s))
    stats = {"depth": -p["energy"]}
    if _W["frame"]:
        v_tree = _W["vel"][s].reindex(p["galaxy"].astype(np.int64)).to_numpy(np.float64)
        v2 = ((X[p["row"].astype(np.int64), 3:] - v_tree) ** 2).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            stats["-u^2"] = -v2 / (v2 - 2.0 * p["energy"])
    if np.isfinite(p["log_alpha"]).any():
        stats["log alpha"] = p["log_alpha"]

    rows = [dict(r, snap=s, t=run.time[s]) for r in within(p, logG, stats, lab, m, set(run.hosts(s)), keep)]
    (k, j, dG, dA, dLa, pid, _, _), counts = contested_pairs(p, logG, keep, ids)

    s2 = _W["later"][s]                                      # 3.2's autocorrelation partner, 3.3's outcome
    p2 = run.pairs(s2)
    X2, m2 = run.particles(s2)
    logG2 = galaxy_terms(p2, X2, run.gaussians(s2))
    (k2, j2, dG2, dA2, _, pid2, _, _), _ = contested_pairs(p2, logG2, components(p2)[1], run.row_labels(s2)[0])
    ac = autocorr_table(k, j, pid, dG, dA, k2, j2, pid2, dG2, dA2, s, run.time[s])

    old = seen_before(run, s, ids, OLD_GYR)
    C = calibration(run, s, s2, p, m, old, keep, _W["beta_run"])
    lag = persistence(run, s, _W["lags"].get(s, {}), p, m, old, keep)

    g, n_lab = np.unique(lab[lab != UNASSIGNED], return_counts=True)
    return (rows, pair_table(k, j, dG, dA, dLa, s, run.time[s]), np.array(counts, float), C, ac, lag,
            dict(zip(g.tolist(), n_lab.tolist())))


_WL = {}                                # the light worker's run and settings


def _init_light(path, source, snaps, tree_state, feedback, dependence):
    run = Run(path, particles=source)
    run.snaps = snaps
    _WL.update(run=run, tree_state=tree_state, feedback=feedback, dependence=dependence)


def _light(s):
    """3.4 (``--dependence``) and 3.2b (``--feedback``, snapshots after the first with a stored log alpha)
    at ``s``, one of every --light-every-th snapshot, in the light worker: the dependence table, the
    feedback snapshot (None when skipped) and each galaxy's responsibility-weighted stellar mass (for the
    mass-binned tables)."""
    run = _WL["run"]
    ids, lab = run.row_labels(s)
    if ids.size < MIN_PARTICLES:
        return None
    p = run.pairs(s)
    X, m = run.particles(s)
    comp, keep = components(p)
    gauss = run.gaussians(s)
    logG = galaxy_terms(p, X, gauss)
    t = run.time[s]

    dep, dep_few = pd.DataFrame(), (0, 0)
    if _WL["dependence"]:
        (k, j, dG, dA, dLa, pid, a, b), _ = contested_pairs(p, logG, keep, ids)
        tree = _WL["tree_state"].get(s)
        if k.size and tree is not None:
            d2 = galaxy_d2(p, X, gauss)
            rows, gal = p["row"].astype(np.int64), p["galaxy"].astype(np.int64)
            pos_g = tree[POS].reindex(gal).to_numpy(np.float64)
            vel_g = tree[VEL].reindex(gal).to_numpy(np.float64)
            r_all = np.linalg.norm(X[rows, :3] - pos_g, axis=1)
            v_all = np.linalg.norm(X[rows, 3:] - vel_g, axis=1)
            Y = dLa if np.isfinite(dLa).all() else dA
            phys = np.stack([r_all[a], v_all[a], r_all[b], v_all[b]], axis=1)
            dep, dep_few = dependence_table(k, j, Y, d2[a], d2[b], phys, dG + dA, s, t, seed=s)

    fb = None
    if _WL["feedback"] and np.isfinite(p["log_alpha"]).any():
        s_idx = int(np.searchsorted(run.snaps, s))
        if s_idx > 0:
            s_prev = int(run.snaps[s_idx - 1])
            tree_s, tree_p = _WL["tree_state"].get(s), _WL["tree_state"].get(s_prev)
            if tree_s is not None and tree_p is not None:
                fb = _feedback_snapshot(run, s, p, X, m, comp, keep, gauss, run.gaussians(s_prev),
                                         tree_s, tree_p, _WL["dependence"])

    r_all, rows_p, gal_p = np.exp(p["log_resp"]), p["row"].astype(np.int64), p["galaxy"].astype(np.int64)
    gal_mass = {int(k): float((m[rows_p[e]] * r_all[e]).sum()) for k, e in groups(gal_p)}
    return dep, dep_few, fb, gal_mass


def _feedback_snapshot(run, s, p, X, m, comp, keep, gauss, gauss_prev, tree_s, tree_p, dependence):
    """3.2b at ``s``: every resolved component whose galaxies all have a stored, finite-count Gaussian is
    replayed (``feedback_component``) and aggregated, against the snapshot's resolved contested mass
    (``total_mass``). A component is only counted when it is unresolved, missing a Gaussian or tree row
    (``mass_missing_gaussian``), or has a contested-only reproduction residual above REPRO_TOL
    (``mass_repro_excluded``); each tracks the contested mass it accounts for."""
    rows_all, gal_all = p["row"].astype(np.int64), p["galaxy"].astype(np.int64)
    contested_all = np.bincount(rows_all)[rows_all] >= 2
    resolved_rows = np.unique(rows_all[keep & contested_all])
    total_mass = float(m[resolved_rows].sum())

    parts, left_out = [], 0
    mass_missing_gaussian, mass_repro_excluded = 0.0, 0.0
    for _, e in groups(comp):
        if not keep[e][0]:
            continue
        galaxies = np.unique(gal_all[e])
        comp_mass = float(m[np.unique(rows_all[e][contested_all[e]])].sum())
        gdict = {int(k): gauss[int(k)] for k in galaxies.tolist()
                 if int(k) in gauss and np.isfinite(gauss[int(k)][3]) and pd_ok(gauss[int(k)][1])}
        if len(gdict) != galaxies.size or not np.isin(galaxies, tree_s.index.to_numpy()).all():
            left_out += 1
            mass_missing_gaussian += comp_mass
            continue
        tree_xyzv = np.stack([np.r_[tree_s.loc[int(k), POS].to_numpy(np.float64),
                                     tree_s.loc[int(k), VEL].to_numpy(np.float64)]
                               for k in galaxies.tolist()])
        tree_mass = {int(k): float(tree_s.loc[int(k), "mass"]) for k in galaxies.tolist()
                     if int(k) in tree_s.index}
        tree_mass_prev = {int(k): float(tree_p.loc[int(k), "mass"]) for k in galaxies.tolist()
                          if int(k) in tree_p.index}
        gauss_prev_c = {int(k): gauss_prev[int(k)] for k in galaxies.tolist() if int(k) in gauss_prev}

        out = feedback_component(run, s, p, X, m, galaxies, gdict, gauss_prev_c, tree_xyzv,
                                  tree_mass, tree_mass_prev, dependence=dependence)
        if out is None:
            left_out += 1
            mass_missing_gaussian += comp_mass
            continue
        if out["repro"]["n"] and not np.isnan(out["repro"]["median"]) and out["repro"]["median"] > REPRO_TOL:
            left_out += 1
            mass_repro_excluded += comp_mass
            continue
        parts.append(out)

    return dict(total_mass=total_mass, left_out=left_out, parts=parts, snapshot_mass=float(m.sum()),
                mass_missing_gaussian=mass_missing_gaussian, mass_repro_excluded=mass_repro_excluded)


# --- the point ------------------------------------------------------------------------------------

def evaluate(run, out, every=10, workers=1, feedback=False, dependence=False, light_every=3):
    rep = Report("Goal 3, probabilistic model", run.name, verdict=False)
    sn = list(run.snaps)
    t = np.array([run.time[s] for s in sn])
    later, lags = {}, {}
    for i in range(0, len(sn), every):
        after = np.flatnonzero(t >= t[i] + LATER_GYR)
        if after.size:
            later[int(sn[i])] = int(sn[after[0]])
        lm = {}
        for lag in LAGS_GYR:
            aft = np.flatnonzero(t >= t[i] + lag)
            if aft.size:
                lm[lag] = int(sn[aft[0]])
        if lm:
            lags[int(sn[i])] = lm
    picks = sorted(later)
    frame = run.config.get("stars", "none") != "free"
    beta_run = 1.0 if run.config.get("tempering") is None else float(run.config["tempering"])
    vel = {int(s): g.set_index("Sub_tree_id")[VEL]
           for s, g in run.tree[run.tree["Snapshot"].isin(picks)].groupby("Snapshot")}

    G, P, AC, C, contested, measured, per_snap, peak = [], [], [], {}, np.zeros(2), [], [], {}
    LAG, lag_base, lag_n = {}, np.zeros((2, EDGES.size - 1)), 0
    parts = pool(_snapshot, picks, workers, _init,
                 (run.path, run.source, run.snaps, frame, beta_run, vel, later, lags), desc="3 snapshots")
    for s, part in zip(picks, parts):
        if part is None:
            continue
        measured.append(s)
        rows, table, counts, c, ac, (mass, mr, lag_sums), n_lab = part
        G += rows
        P.append(table)
        AC.append(ac)
        contested += counts
        for key, v in c.items():
            C[key] = C[key] + v if key in C else v
        if beta_run in c:
            per_snap.append((s, run.time[s], *over_under(c[beta_run])))
        for g, n in n_lab.items():
            peak[g] = max(peak.get(g, 0), n)
        if len(lag_sums) == len(LAGS_GYR):
            lag_n += 1
            lag_base += np.stack([mass, mr])
            for lag, arr in lag_sums.items():
                LAG[lag] = LAG[lag] + arr if lag in LAG else arr

    rep.score("snapshots measured", len(measured))
    rep.table("snapshots (outcome of 3.3 at the first snapshot >= 150 Myr later)",
              "  " + " ".join(f"{s}->{later[s]}" for s in measured))
    G = pd.DataFrame(G)
    P = pd.concat(P, ignore_index=True) if P else pd.DataFrame()
    AC = pd.concat(AC, ignore_index=True) if AC else pd.DataFrame()
    report_within(G, rep)
    report_pairs(P, AC, contested, rep)
    report_calibration(C, rep, beta_run)
    report_persistence(lag_base, LAG, lag_n, rep)

    per_snap = pd.DataFrame(per_snap, columns=["snap", "t", "O", "U"])
    plot_overlap(run, G, P, AC, plots.run_plot(out, "model_overlap"))
    plot_calibration(run, C, per_snap, beta_run, plots.run_plot(out, "model_calibration"))
    plot_persistence(run, lag_base, LAG, plots.run_plot(out, "model_persistence"))

    DEP, dep_few, FB_gal, FB_gamma, FB_gamma1, depB, FB_per_snap = (pd.DataFrame(), (0, 0), pd.DataFrame(),
        pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    MASS = {}
    if feedback or dependence:
        light_snaps = sorted(run.snaps.tolist())[::light_every]
        tree_state = {int(s): g.set_index("Sub_tree_id")[POS + VEL + ["mass"]]       # every snapshot: 3.2b
                     for s, g in run.tree[run.tree["Snapshot"].isin(run.snaps)].groupby("Snapshot")}  # reads s-1
        light = pool(_light, light_snaps, workers, _init_light,
                    (run.path, run.source, run.snaps, tree_state, feedback, dependence), desc="3 light snapshots")
        dep_parts, dep_few_sum, fb_list, times = [], [0, 0], [], []
        for s, part in zip(light_snaps, light):
            if part is None:
                fb_list.append(None)
                times.append(run.time[s])
                continue
            dep, few, fb, gal_mass = part
            dep_parts.append(dep)
            dep_few_sum[0] += few[0]
            dep_few_sum[1] += few[1]
            fb_list.append(fb)
            times.append(run.time[s])
            MASS.update({(s, k): v for k, v in gal_mass.items()})
        DEP = pd.concat(dep_parts, ignore_index=True) if dep_parts else pd.DataFrame()
        dep_few = tuple(dep_few_sum)
        if feedback:
            FB_gal, FB_gamma, FB_gamma1, depB, _, FB_per_snap = assemble_feedback(fb_list, light_snaps, times)

    if dependence:
        report_dependence(DEP, dep_few, MASS, rep, depB=depB if feedback else None)
        plot_dependence(run, DEP, depB if feedback else None, plots.run_plot(out, "model_dependence"))
    if feedback:
        report_feedback(FB_gal, FB_gamma, FB_gamma1, MASS, FB_per_snap, rep)
        plot_feedback(run, FB_per_snap, FB_gal, FB_gamma, plots.run_plot(out, "model_feedback"))

    chosen = plots.chosen(run.host, peak, rep.episodes())
    items = []
    for g in chosen:
        mine = G[G["galaxy"] == g].sort_values("t") if not G.empty else G
        pairs = P[(P["k"] == g) | (P["j"] == g)] if not P.empty else P
        if not mine.empty or not pairs.empty:
            items.append((plots.galaxy_plot(out, g, "model_overlap"), g, mine, pairs))
    plots.per_galaxy(_draw_galaxy, items, workers, desc="3 galaxy plots")

    if feedback:
        items = []
        for g in chosen:
            mine = FB_gal[FB_gal["galaxy"] == g].sort_values("t") if not FB_gal.empty else FB_gal
            pairs = FB_gamma[(FB_gamma["k"] == g) | (FB_gamma["j"] == g)] if not FB_gamma.empty else FB_gamma
            if not mine.empty or not pairs.empty:
                items.append((plots.galaxy_plot(out, g, "model_feedback"), g, mine, pairs))
        plots.per_galaxy(_draw_galaxy_feedback, items, workers, desc="3 feedback galaxy plots")
    if dependence:
        items = []
        for g in chosen:
            pairs = DEP[(DEP["k"] == g) | (DEP["j"] == g)] if not DEP.empty else DEP
            if not pairs.empty:
                items.append((plots.galaxy_plot(out, g, "model_dependence"), g, pairs))
        plots.per_galaxy(_draw_galaxy_dependence, items, workers, desc="3 dependence galaxy plots")

    rep.write(out / "model.txt")
    return rep


def report_within(G, rep):
    if G.empty:
        return
    for name in [x for x in STATS if f"soft {x}" in G]:
        soft, hard = G[f"soft {name}"], G[f"hard {name}"]
        ok = soft.notna()
        rep.score(f"3.1 {name}: galaxy-snapshots", int(ok.sum()))
        rep.score(f"3.1 {name}: median rho soft", float(soft.median()))
        rep.score(f"3.1 {name}: median rho hard", float(hard.median()))
        rep.score(f"3.1 {name}: mass-weighted median rho soft",
                  wmedian(soft[ok].to_numpy(), G.loc[ok, "mass"].to_numpy()))
        rep.score(f"3.1 {name}: share rho soft > {STRONG_GALAXY}", float((soft[ok] > STRONG_GALAXY).mean()))
    G = G.assign(bin=pd.cut(G["mass"], MASS_BINS, labels=["<1e7", "1e7-1e8", "1e8-1e9", ">1e9"]),
                 kind=np.where(G["satellite"], "satellite", "central"))
    cols = [f"soft {x}" for x in STATS if f"soft {x}" in G]
    by = G.groupby(["bin", "kind"], observed=True)
    tab = by[cols].median().join(by.size().rename("n"))
    rep.table("3.1 median rho soft by stellar mass and kind", tab.to_string(float_format=lambda v: f"{v:.3f}"))


def report_pairs(P, AC, contested, rep):
    rep.score("3.2 contested particles (summed over snapshots)", int(contested[0]))
    rep.score("3.2 share with a pair left out (underflowed r, no Gaussian, unresolved group)",
              float(contested[1] / max(contested[0], 1)))
    if not P.empty:
        P = P[P["rho"].notna()]
        n = P["n"].to_numpy()
        rep.score("3.2 galaxy pairs x snapshots (>= 30 particles)", len(P))
        rep.score("3.2 n-weighted median rho(D_G, D_alpha)", wmedian(P["rho"].to_numpy(), n))
        rep.score("3.2 n-weighted median alpha share v", wmedian(P["v"].to_numpy(), n))
        rep.score(f"3.2 n-weighted share rho > {STRONG_PAIR}",
                  float(n[P["rho"] > STRONG_PAIR].sum() / n.sum()))
        if "rho stored" in P:
            rep.score("3.2 n-weighted median rho(D_alpha, stored Delta log alpha)",
                      wmedian(P["rho stored"].to_numpy(), n))
        top = P.sort_values("n", ascending=False).head(10).drop(columns="t")
        rep.table("3.2 largest galaxy pairs", top.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if not AC.empty:
        n = AC["n"].to_numpy()
        rep.score("3.2 n-weighted median autocorrelation of D_G over 150 Myr",
                  wmedian(AC["ac_G"].to_numpy(), n))
        rep.score("3.2 n-weighted median autocorrelation of D_alpha over 150 Myr",
                  wmedian(AC["ac_alpha"].to_numpy(), n))
        rep.score("3.2 n-weighted share of pairs with ac_alpha < ac_G",
                  float(n[AC["ac_alpha"] < AC["ac_G"]].sum() / n.sum()))


def report_calibration(C, rep, beta_run):
    if beta_run not in C:
        return
    for beta in [beta_run] + [b for b in BETAS if b in C and b != beta_run]:
        mass, count, mr, mp = C[beta]
        with np.errstate(invalid="ignore", divide="ignore"):
            rbar, pers = mr / mass, mp / mass
        tag = "3.3" if beta == beta_run else f"3.5 beta {beta}"
        O, U = over_under(C[beta])
        rep.score(f"{tag}: overconfidence O", O)
        rep.score(f"{tag}: apparent underconfidence U", U)
        rep.score(f"{tag}: persistence at r_max >= 0.999", float(pers[-1]))
        if beta != beta_run:
            changed, total = C[f"changed {beta}"]
            rep.score(f"{tag}: share whose argmax changes", float(changed / total))
        tab = pd.DataFrame({"mass share": mass / mass.sum(), "stars": count.astype(int), "mean r": rbar,
                            "persistence": pers, "gap": rbar - pers}, index=BIN_LABELS)
        rep.table(f"{tag} reliability (contested old stars, at the first snapshot >= 150 Myr later)",
                  tab.to_string(float_format=lambda v: f"{v:.4f}"))


def report_persistence(base, LAG, n, rep):
    """3.3's lag curve: the table and SCOREs, pooled over the ``n`` snapshots that have every lag."""
    if n == 0:
        return
    mass, mr = base
    with np.errstate(invalid="ignore", divide="ignore"):
        cols = {"mean r": mr / mass}
        for lag in LAGS_GYR:
            counted, persisted = LAG[lag]
            pers = persisted / counted
            cols[f"persistence, lag {lag}"] = pers
            rep.score(f"3.3 lag curve: persistence at r_max >= 0.999, lag {lag} Gyr", float(pers[-1]))
    tab = pd.DataFrame(cols, index=BIN_LABELS)
    rep.table(f"3.3 persistence against lag (snapshots with every lag: {n})",
              tab.to_string(float_format=lambda v: f"{v:.4f}"))


def mass_bin(values):
    """Bin stellar masses into the categories used throughout goal 3 (see ``report_within``)."""
    return pd.cut(values, MASS_BINS, labels=["<1e7", "1e7-1e8", "1e8-1e9", ">1e9"])


def galaxy_mass(df, mass, gal_col="k"):
    """``df``'s own ``(snap, gal_col)`` read against ``mass`` ({(snap, galaxy): responsibility-weighted
    stellar mass}, from the light worker's ``gal_mass``); NaN where missing."""
    return np.array([mass.get((s, g), np.nan) for s, g in zip(df["snap"], df[gal_col])])


def assemble_feedback(fb_list, snaps, times):
    """Collect every snapshot's ``_feedback_snapshot`` output into run-wide tables: per-galaxy rows, the
    converged and one-step gamma tables, R2_G(B) (with --dependence too), the per-particle label/mass
    rows, and one row per snapshot of the scalar SCOREs (coverage, residuals, labels, sharpening)."""
    gal_rows, gammas, gamma1s, dep_Bs, label_rows, per_snap = [], [], [], [], [], []
    for s, t, fb in zip(snaps, times, fb_list):
        if fb is None:
            continue
        parts = fb["parts"]
        replayed = sum(out["resolved_mass"] for out in parts)
        no_hist = sum(out["no_history_mass"] for out in parts)
        med = [out["repro"]["median"] for out in parts if out["repro"]["n"]]
        p99 = [out["repro"]["p99_r"] for out in parts if out["repro"]["n"]]
        nrep = [out["repro"]["n"] for out in parts if out["repro"]["n"]]
        contested_parts = [out for out in parts if out["resolved_mass"] > 0]    # trivial (no contested
        its_mass = np.array([out["resolved_mass"] for out in contested_parts])  # entries) parts would
        its_A = np.array([out["iterations"][0] for out in contested_parts])     # swamp the medians below
        its_B = np.array([out["iterations"][1] for out in contested_parts])
        not_converged = sum(out["resolved_mass"] for out in contested_parts
                            if out["iterations"][0] >= MAX_ITER or out["iterations"][1] >= MAX_ITER)

        def cat(key, dtype=float):
            return (np.concatenate([out["labels"][key] for out in parts]) if parts
                    else np.zeros(0, dtype))

        w, changed, host = cat("w"), cat("changed", bool), cat("host", bool)
        sharp_A, sharp_B, h_A, h_B = cat("sharp_A", bool), cat("sharp_B", bool), cat("entropy_A"), cat("entropy_B")

        for out in parts:
            gal_rows.append(pd.DataFrame(out["gal_row"]))
            gammas.append(out["gamma"])
            gamma1s.append(out["gamma1"])
            dep_Bs.append(out["dep_B"])
        if w.size:
            label_rows.append(pd.DataFrame({"snap": s, "t": t, "w": w, "changed": changed, "host": host}))

        with np.errstate(invalid="ignore", divide="ignore"):
            per_snap.append({
                "snap": s, "t": t,
                "coverage": replayed / fb["total_mass"] if fb["total_mass"] > 0 else np.nan,
                "no_history_share": no_hist / fb["total_mass"] if fb["total_mass"] > 0 else np.nan,
                "repro_median": wmedian(np.array(med), np.array(nrep)) if nrep else np.nan,
                "repro_p99_r": wmedian(np.array(p99), np.array(nrep)) if nrep else np.nan,
                "left_out": fb["left_out"],
                "mass_missing_gaussian_share": (fb["mass_missing_gaussian"] / fb["total_mass"]
                                                if fb["total_mass"] > 0 else np.nan),
                "mass_repro_excluded_share": (fb["mass_repro_excluded"] / fb["total_mass"]
                                              if fb["total_mass"] > 0 else np.nan),
                "not_converged_share": not_converged / replayed if replayed > 0 else np.nan,
                "iterations_A": wmedian(its_A, its_mass) if its_mass.size else np.nan,
                "iterations_B": wmedian(its_B, its_mass) if its_mass.size else np.nan,
                "maxiter_A": int((its_A >= MAX_ITER).sum()),
                "maxiter_B": int((its_B >= MAX_ITER).sum()),
                "label_share_contested": float((w * changed).sum() / w.sum()) if w.sum() > 0 else np.nan,
                "label_share_total": (float((w * changed).sum() / fb["snapshot_mass"])
                                      if fb["snapshot_mass"] > 0 else np.nan),
                "label_share_host": (float((w[host] * changed[host]).sum() / w[host].sum())
                                     if host.any() and w[host].sum() > 0 else np.nan),
                "sharp_A": float((w * sharp_A).sum() / w.sum()) if w.sum() > 0 else np.nan,
                "sharp_B": float((w * sharp_B).sum() / w.sum()) if w.sum() > 0 else np.nan,
                "entropy_A": float((w * h_A).sum() / w.sum()) if w.sum() > 0 else np.nan,
                "entropy_B": float((w * h_B).sum() / w.sum()) if w.sum() > 0 else np.nan,
            })

    gal = pd.concat(gal_rows, ignore_index=True) if gal_rows else pd.DataFrame()
    gamma = pd.concat(gammas, ignore_index=True) if gammas else pd.DataFrame()
    gamma1 = pd.concat(gamma1s, ignore_index=True) if gamma1s else pd.DataFrame()
    depB = pd.concat(dep_Bs, ignore_index=True) if dep_Bs else pd.DataFrame()
    labels = pd.concat(label_rows, ignore_index=True) if label_rows else pd.DataFrame()
    return gal, gamma, gamma1, depB, labels, pd.DataFrame(per_snap)


def report_dependence(DEP, dep_few, mass, rep, depB=None):
    """3.4: pairs measured and only counted, n-weighted medians of R2_G, R2_phys, u (and R2_G(B) with
    --feedback), split by pair size (>= DEP_BOOT raw, 500..DEP_BOOT corrected), and a table by the
    smaller galaxy's stellar mass."""
    rep.score("3.4 pairs x snapshots measured (>= 500 stars)", len(DEP))
    rep.score("3.4 pairs left out (< 500 stars)", dep_few[0])
    rep.score("3.4 stars in pairs left out", dep_few[1])
    if DEP.empty:
        return
    n = DEP["n"].to_numpy()
    for name, col in (("R2_G", "r2_G"), ("R2_phys", "r2_phys"), ("u", "u")):
        rep.score(f"3.4 n-weighted median {name}", wmedian(DEP[col].to_numpy(), n))
    big, small = DEP["n"] >= DEP_BOOT, DEP["n"] < DEP_BOOT
    for tag, sel in (("n >= 5000", big), ("500 <= n < 5000", small)):
        if sel.any():
            rep.score(f"3.4 n-weighted median R2_G, {tag}", wmedian(DEP.loc[sel, "r2_G"].to_numpy(),
                                                                     n[sel.to_numpy()]))
            se_ok = DEP.loc[sel, "r2_G se"].notna()
            if se_ok.any():
                rep.score(f"3.4 n-weighted median R2_G standard error, {tag}",
                          wmedian(DEP.loc[sel, "r2_G se"][se_ok].to_numpy(), n[sel.to_numpy()][se_ok.to_numpy()]))
    if depB is not None and not depB.empty:
        rep.score("3.4 n-weighted median R2_G(B)", wmedian(depB["r2_G"].to_numpy(), depB["n"].to_numpy()))
    by = DEP.assign(bin=mass_bin(galaxy_mass(DEP, mass)))
    g = by.groupby("bin", observed=True)
    tab = g[["r2_G", "r2_phys", "u"]].median().join(g.size().rename("n"))
    rep.table("3.4 median R2_G, R2_phys, u by the smaller galaxy's stellar mass",
              tab.to_string(float_format=lambda v: f"{v:.3f}"))


def report_feedback(gal, gamma, gamma1, mass, per_snap, rep):
    """3.2b: coverage, residuals (contested entries only), excluded mass shares, iterations (mass-
    weighted medians; components with no contested entries are trivial and excluded), label-change
    shares, sharpening, mass-weighted median per-galaxy KL/dlogdet/shift and KL(written, A), n-weighted
    median sd(L)/f/rho_L/gamma weighted by the stars in play (n_play); a table by stellar mass and the 10
    pairs with the largest sd(L) x n_play."""
    if per_snap.empty:
        rep.score("3.2b snapshots replayed", 0)
        return
    rep.score("3.2b snapshots replayed", len(per_snap))
    rep.score("3.2b median coverage (resolved contested mass replayed)", float(per_snap["coverage"].median()))
    rep.score("3.2b median share in no-history galaxies", float(per_snap["no_history_share"].median()))
    rep.score("3.2b median share excluded: missing/invalid Gaussian or tree row",
              float(per_snap["mass_missing_gaussian_share"].median()))
    rep.score("3.2b median share excluded: reproduction residual above 1e-2",
              float(per_snap["mass_repro_excluded_share"].median()))
    rep.score("3.2b median reproduction residual |delta log r| (contested entries)",
              float(per_snap["repro_median"].median()))
    rep.score("3.2b median reproduction residual, p99 |delta r| (contested entries)",
              float(per_snap["repro_p99_r"].median()))
    rep.score("3.2b components left out (summed over snapshots)", int(per_snap["left_out"].sum()))
    rep.score("3.2b median mass-weighted iterations, chain A", float(per_snap["iterations_A"].median()))
    rep.score("3.2b median mass-weighted iterations, chain B", float(per_snap["iterations_B"].median()))
    rep.score("3.2b components that hit max_iter, chain A", int(per_snap["maxiter_A"].sum()))
    rep.score("3.2b components that hit max_iter, chain B", int(per_snap["maxiter_B"].sum()))
    rep.score("3.2b median replayed mass share in non-converged components",
              float(per_snap["not_converged_share"].median()))
    rep.score("3.2b median label-change share of contested mass", float(per_snap["label_share_contested"].median()))
    rep.score("3.2b median label-change share of the snapshot's stellar mass",
              float(per_snap["label_share_total"].median()))
    rep.score("3.2b median label-change share of the host's contested mass",
              float(per_snap["label_share_host"].median()))
    rep.score("3.2b median sharpened share (r_max >= 0.999), chain A", float(per_snap["sharp_A"].median()))
    rep.score("3.2b median sharpened share (r_max >= 0.999), chain B", float(per_snap["sharp_B"].median()))
    rep.score("3.2b median mass-weighted mean entropy, chain A", float(per_snap["entropy_A"].median()))
    rep.score("3.2b median mass-weighted mean entropy, chain B", float(per_snap["entropy_B"].median()))

    if not gal.empty:
        w = gal["mass"].to_numpy()
        contested_gal = gal[w > 0]
        wc = w[w > 0]
        if contested_gal.empty:
            rep.score("3.2b components with a contested entry", 0)
        else:
            rep.score("3.2b mass-weighted median per-galaxy symmetric KL(A, B)",
                      wmedian(contested_gal["kl"].to_numpy(), wc))
            rep.score("3.2b mass-weighted median per-galaxy Delta log|Sigma|",
                      wmedian(contested_gal["dlogdet"].to_numpy(), wc))
            rep.score("3.2b mass-weighted median per-galaxy mean shift (A's sigma)",
                      wmedian(contested_gal["shift"].to_numpy(), wc))
            rep.score("3.2b mass-weighted median per-galaxy KL(written, A)",
                      wmedian(contested_gal["kl_written"].to_numpy(), wc))
            by = contested_gal.assign(bin=mass_bin(galaxy_mass(contested_gal, mass, gal_col="galaxy")))
            tab = by.groupby("bin", observed=True).apply(
                lambda d: pd.Series({c: wmedian(d[c].to_numpy(), d["mass"].to_numpy())
                                     for c in ("kl", "dlogdet", "shift", "dlogM")}), include_groups=False)
            rep.table("3.2b mass-weighted median per-galaxy measurements by stellar mass",
                      tab.to_string(float_format=lambda v: f"{v:.4f}"))
    if not gamma.empty:
        n = gamma["n_play"].to_numpy()
        rep.score("3.2b stars in play (sum p(1-p)) per pair, median", float(gamma["n_play"].median()))
        rep.score("3.2b in-play-weighted median sd(L), nats", wmedian(gamma["sd_L"].to_numpy(), n))
        rep.score("3.2b in-play-weighted median f (Var(L) / Var(D_A))", wmedian(gamma["f"].to_numpy(), n))
        rep.score("3.2b in-play-weighted median rho_L (alignment with Delta log alpha)",
                  wmedian(gamma["rho_L"].to_numpy(), n))
        ok = gamma["gamma"].notna()
        if ok.any():
            rep.score("3.2b in-play-weighted median gamma (Var(Delta log alpha) >= 0.01 Var(D_A))",
                      wmedian(gamma.loc[ok, "gamma"].to_numpy(), n[ok.to_numpy()]))
        if not gamma1.empty:
            rep.score("3.2b in-play-weighted median sd(L), nats, one step",
                      wmedian(gamma1["sd_L"].to_numpy(), gamma1["n_play"].to_numpy()))
        top = gamma.assign(size=gamma["sd_L"] * n).sort_values("size", ascending=False).head(10)
        rep.table("3.2b largest sd(L) x n_play galaxy pairs",
                  top.drop(columns=["t", "size"]).to_string(index=False, float_format=lambda v: f"{v:.3f}"))


# --- plots ----------------------------------------------------------------------------------------

def plot_overlap(run, G, P, AC, pdf):
    """3.1 median rho per statistic, 3.2 n-weighted median rho and v, and 3.2's D_G / D_alpha
    autocorrelation over 150 Myr, against time."""
    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    if not G.empty:
        for i, name in enumerate(x for x in STATS if f"soft {x}" in G):
            med = G.groupby("t")[[f"soft {name}", f"hard {name}"]].median()
            a1.plot(med.index, med[f"soft {name}"], "o-", ms=3, color=f"C{i}", label=f"{name}, soft")
            a1.plot(med.index, med[f"hard {name}"], "--", lw=0.8, color=f"C{i}", label=f"{name}, hard")
    a1.set_ylabel("3.1 median Spearman rho(log N_k, statistic)")
    a1.legend(fontsize=7, ncol=2)
    if not P.empty:
        ok = P[P["rho"].notna()]
        by = ok.groupby("t")
        rho = by.apply(lambda d: wmedian(d["rho"].to_numpy(), d["n"].to_numpy()), include_groups=False)
        v = by.apply(lambda d: wmedian(d["v"].to_numpy(), d["n"].to_numpy()), include_groups=False)
        a2.plot(rho.index, rho, "o-", ms=3, label="rho(D_G, D_alpha)")
        a2.plot(v.index, v, "s-", ms=3, label="alpha share of the decision v")
    a2.axhline(0, color="0.6", lw=0.6)
    a2.set_ylabel("3.2 n-weighted median")
    a2.legend(fontsize=8)
    if not AC.empty:
        by = AC.groupby("t")
        acG = by.apply(lambda d: wmedian(d["ac_G"].to_numpy(), d["n"].to_numpy()), include_groups=False)
        acA = by.apply(lambda d: wmedian(d["ac_alpha"].to_numpy(), d["n"].to_numpy()), include_groups=False)
        a3.plot(acG.index, acG, "o-", ms=3, label="autocorr(D_G)")
        a3.plot(acA.index, acA, "s-", ms=3, label="autocorr(D_alpha)")
    a3.axhline(0, color="0.6", lw=0.6)
    a3.set_ylabel("3.2 n-weighted median autocorrelation, 150 Myr")
    a3.set_xlabel("t [Gyr]")
    a3.legend(fontsize=8)
    fig.suptitle(f"{run.name}: goal 3, overlap against time")
    plots.save(fig, pdf)


def plot_calibration(run, C, per_snap, beta_run, pdf):
    """The reliability curve per beta (pooled), and the run's O and U per snapshot against time."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    for i, beta in enumerate(sorted((b for b in C if not isinstance(b, str)), reverse=True)):
        mass, _, mr, mp = C[beta]
        ok = mass > 0
        a1.plot(mr[ok] / mass[ok], mp[ok] / mass[ok], "o-", ms=3, color=f"C{i}",
                label=f"beta {beta}" + (" (the run)" if beta == beta_run else ""))
    a1.plot([0, 1], [0, 1], ":", color="0.5")
    a1.set_xlabel("3.3 mean r_max")
    a1.set_ylabel("persistence >= 150 Myr later")
    a1.legend(fontsize=8)
    a2.plot(per_snap["t"], per_snap["O"], "o-", ms=3, label="overconfidence O")
    a2.plot(per_snap["t"], per_snap["U"], "s-", ms=3, label="apparent underconfidence U")
    a2.set_xlabel("t [Gyr]")
    a2.set_ylabel("3.3 per snapshot")
    a2.legend(fontsize=8)
    fig.suptitle(f"{run.name}: goal 3, calibration")
    plots.save(fig, pdf)


def plot_persistence(run, base, LAG, pdf):
    """3.3's lag curve: persistence against lag, one line per r_max bin (sequential colormap), its mean r
    (the prediction) drawn at lag 0."""
    mass, mr = base
    lagv = list(LAGS_GYR)
    with np.errstate(invalid="ignore", divide="ignore"):
        rbar = mr / mass
        pers = np.stack([LAG[lag][1] / LAG[lag][0] for lag in lagv])        # (lag, bin)
    cmap = plt.get_cmap("viridis")
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for i, label in enumerate(BIN_LABELS):
        if mass[i] <= 0:
            continue
        ax.plot([0.0] + lagv, np.r_[rbar[i], pers[:, i]], "o-", ms=3,
                color=cmap(i / max(len(BIN_LABELS) - 1, 1)), label=label)
    ax.set_xlabel("lag [Gyr]")
    ax.set_ylabel("3.3 persistence (at lag 0: mean r_max)")
    ax.legend(fontsize=7, title="r_max bin", loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.suptitle(f"{run.name}: goal 3, persistence against lag")
    plots.save(fig, pdf)


def plot_feedback(run, per_snap, gal, gamma, pdf):
    """3.2b against time: n-weighted median sd(L) (nats), rho_L and gamma (where defined), the
    label-change share, the mass-weighted median per-galaxy Delta log|Sigma|, and the reproduction
    residual (median |delta log r| and p99 |delta r|, contested entries)."""
    fig, (a0, a1, a2, a3, a4) = plt.subplots(5, 1, figsize=(10, 15), sharex=True)
    if not gamma.empty:
        by = gamma.groupby("t")

        def med(col):
            return by.apply(lambda d: wmedian(d[col].to_numpy(), d["n_play"].to_numpy()), include_groups=False)

        sd = med("sd_L")
        a0.plot(sd.index, sd, "o-", ms=3)
        rho_L = med("rho_L")
        a1.plot(rho_L.index, rho_L, "s-", ms=3, label="rho_L = corr(L, Delta log alpha)")
        ok = gamma[gamma["gamma"].notna()]
        if not ok.empty:
            g = ok.groupby("t").apply(lambda d: wmedian(d["gamma"].to_numpy(), d["n_play"].to_numpy()),
                                      include_groups=False)
            a1.plot(g.index, g, "^--", ms=3, color="0.4", label="gamma (where defined)")
        a1.legend(fontsize=8)
    a0.set_yscale("log")
    a0.set_ylabel("3.2b sd(L) [nats], median over pairs")
    a1.axhline(0, color="0.6", lw=0.6)
    a1.set_ylabel("3.2b median over pairs")
    if not per_snap.empty:
        a2.plot(per_snap["t"], per_snap["label_share_contested"], "o-", ms=3, label="of contested mass")
        a2.plot(per_snap["t"], per_snap["label_share_total"], "s-", ms=3, label="of total stellar mass")
    a2.set_ylabel("3.2b label-change share")
    a2.legend(fontsize=8)
    if not gal.empty:
        contested_gal = gal[gal["mass"] > 0]
        if not contested_gal.empty:
            d = contested_gal.groupby("t").apply(lambda d: wmedian(d["dlogdet"].to_numpy(), d["mass"].to_numpy()),
                                                 include_groups=False)
            a3.plot(d.index, d, "o-", ms=3)
    a3.axhline(0, color="0.6", lw=0.6)
    a3.set_ylabel("3.2b mass-weighted median Delta log|Sigma|")
    if not per_snap.empty:
        a4.plot(per_snap["t"], per_snap["repro_median"], "o-", ms=3, label="median |delta log r|")
        a4.plot(per_snap["t"], per_snap["repro_p99_r"], "s-", ms=3, label="p99 |delta r|")
    a4.set_yscale("log")
    a4.set_xlabel("t [Gyr]")
    a4.set_ylabel("3.2b reproduction residual")
    a4.legend(fontsize=8)
    fig.suptitle(f"{run.name}: goal 3, feedback (3.2b)")
    plots.save(fig, pdf)


def plot_dependence(run, DEP, depB, pdf):
    """3.4 against time: n-weighted median R2_G, R2_phys and u (and R2_G(B) with --feedback), large
    (>= DEP_BOOT) and bias-corrected small pairs distinguished."""
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    if not DEP.empty:
        big = DEP[DEP["n"] >= DEP_BOOT]
        small = DEP[DEP["n"] < DEP_BOOT]
        for sel, tag, style in ((big, "n >= 5000", "o-"), (small, "500 <= n < 5000", "o--")):
            if sel.empty:
                continue
            by = sel.groupby("t")
            for i, col in enumerate(("r2_G", "r2_phys")):
                v = by.apply(lambda d, c=col: wmedian(d[c].to_numpy(), d["n"].to_numpy()), include_groups=False)
                a1.plot(v.index, v, style, ms=3, color=f"C{i}", label=f"{col}, {tag}")
        if depB is not None and not depB.empty:
            by = depB.groupby("t")
            v = by.apply(lambda d: wmedian(d["r2_G"].to_numpy(), d["n"].to_numpy()), include_groups=False)
            a1.plot(v.index, v, "x:", ms=4, color="C2", label="r2_G(B)")
    a1.set_ylabel("3.4 n-weighted median R^2")
    a1.legend(fontsize=7, ncol=2)
    if not DEP.empty:
        by = DEP.groupby("t")
        u = by.apply(lambda d: wmedian(d["u"].to_numpy(), d["n"].to_numpy()), include_groups=False)
        a2.plot(u.index, u, "o-", ms=3, label="u")
    a2.set_xlabel("t [Gyr]")
    a2.set_ylabel("3.4 n-weighted median u")
    a2.legend(fontsize=8)
    fig.suptitle(f"{run.name}: goal 3, dependence (3.4)")
    plots.save(fig, pdf)


def _draw_galaxy(item):
    """A galaxy's 3.1 rho per statistic, and its 3.2 rho and v with its 5 largest partners, against time."""
    path, g, mine, pairs = item
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for i, name in enumerate(x for x in STATS if f"soft {x}" in mine):
        a1.plot(mine["t"], mine[f"soft {name}"], "o-", ms=3, color=f"C{i}", label=f"{name}, soft")
        a1.plot(mine["t"], mine[f"hard {name}"], "--", lw=0.8, color=f"C{i}", label=f"{name}, hard")
    a1.set_ylabel("3.1 Spearman rho")
    if not mine.empty:
        a1.legend(fontsize=7, ncol=2)
    if not pairs.empty:
        other = np.where(pairs["k"] == g, pairs["j"], pairs["k"])
        top = pairs.assign(other=other).groupby("other")["n"].sum().nlargest(5).index
        for i, o in enumerate(top):
            d = pairs[other == o].sort_values("t")
            a2.plot(d["t"], d["rho"], "o-", ms=3, color=f"C{i}", label=f"rho with {o}")
            a2.plot(d["t"], d["v"], "x--", ms=4, lw=0.8, color=f"C{i}", label=f"v with {o}")
        a2.legend(fontsize=6, ncol=2)
    a2.axhline(0, color="0.6", lw=0.6)
    a2.set_ylabel("3.2 per partner")
    a2.set_xlabel("t [Gyr]")
    fig.suptitle(f"galaxy {g}: goal 3")
    plots.save(fig, path)


def _draw_galaxy_feedback(item):
    """A galaxy's 3.2b dN and Delta log|Sigma|, and sd(L) and rho_L with its 5 largest partners, against
    time."""
    path, g, mine, gamma = item
    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    if not mine.empty:
        a1.plot(mine["t"], mine["dN"], "o-", ms=3, color="C0", label="dN")
        ax2 = a1.twinx()
        ax2.plot(mine["t"], mine["dlogdet"], "s--", ms=3, color="C1", label="Delta log|Sigma|")
        a1.set_ylabel("3.2b dN (B - A)", color="C0")
        ax2.set_ylabel("Delta log|Sigma|", color="C1")
    if not gamma.empty:
        other = np.where(gamma["k"] == g, gamma["j"], gamma["k"])
        top = gamma.assign(other=other).groupby("other")["n_play"].sum().nlargest(5).index
        for i, o in enumerate(top):
            d = gamma[other == o].sort_values("t")
            a2.plot(d["t"], d["sd_L"], "o-", ms=3, color=f"C{i}", label=f"with {o}")
            a3.plot(d["t"], d["rho_L"], "s--", ms=3, color=f"C{i}", label=f"with {o}")
        a2.legend(fontsize=7)
    a2.set_yscale("log")
    a2.set_ylabel("3.2b sd(L) per partner [nats]")
    a3.axhline(0, color="0.6", lw=0.6)
    a3.set_ylabel("3.2b rho_L per partner")
    a3.set_xlabel("t [Gyr]")
    fig.suptitle(f"galaxy {g}: goal 3, feedback (3.2b)")
    plots.save(fig, path)


def _draw_galaxy_dependence(item):
    """A galaxy's R2_G and u with its 5 largest partners, against time."""
    path, g, dep = item
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    if not dep.empty:
        other = np.where(dep["k"] == g, dep["j"], dep["k"])
        top = dep.assign(other=other).groupby("other")["n"].sum().nlargest(5).index
        for i, o in enumerate(top):
            d = dep[other == o].sort_values("t")
            a1.plot(d["t"], d["r2_G"], "o-", ms=3, color=f"C{i}", label=f"R2_G with {o}")
            a2.plot(d["t"], d["u"], "s--", ms=3, color=f"C{i}", label=f"u with {o}")
        a1.legend(fontsize=7)
        a2.legend(fontsize=7)
    a1.set_ylabel("3.4 R2_G per partner")
    a2.set_ylabel("3.4 u per partner")
    a2.set_xlabel("t [Gyr]")
    fig.suptitle(f"galaxy {g}: goal 3, dependence (3.4)")
    plots.save(fig, path)


def main():
    args = cli.arguments(__doc__, every=10, light_every=True, flags={
        "feedback": "3.2b: replay the fit without alpha in EM (needs the run's stored log_alpha)",
        "dependence": "3.4: how much of alpha the Gaussians already contain",
    })
    run, out = cli.start("model", args)
    evaluate(run, out, every=args.every, workers=args.workers, feedback=args.feedback, dependence=args.dependence,
             light_every=args.light_every)


if __name__ == "__main__":                      # spawned workers import this file, never run it
    main()
