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
(flags; model_alpha.py) run in a pass of their own over every --light-every-th snapshot (default 3; 1:
all of them).

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
    entries (the reproduction residual: median |delta log r|; the tail as each component's p99
    |delta r|, since log differences between underflowed r ~ 1e-12 mean nothing; both medians over
    components, weighted by their contested entries); components whose median |delta log r| > 1e-2 are
    dropped (counted). Two chains are then iterated to convergence from theta_A (<= 300 EM
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
    excluded shares, iterations (and how many did not converge within the cap). Reported: these SCOREs
    (per-galaxy medians weighted by the galaxy's own contested mass, the stars it competes for, over
    galaxies with some; pair medians by the stars in play), a table by stellar mass, the 10 pairs with
    the largest sd(L) x stars in play.
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
import sys
from pathlib import Path

if __package__ in (None, ""):                   # run as a file: make the repository root importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from evaluate import cli, model_alpha, plots
from evaluate.load import (ABSENT, OLD_GYR, POS, UNASSIGNED, VEL, Run, pool, seen_before, wmedian,
                           wmedian_in_time)
from evaluate.pairs import (MIN_PAIR, MIN_PARTICLES, best, components, contested_pairs, galaxy_terms, groups,
                            mass_bin)
from evaluate.report import Report

MIN_R, MIN_HARD = 50.0, 50              # 3.1: a galaxy's sum of r, and its hard members
LATER_GYR = 0.15                        # 3.3: the outcome is read at the first snapshot this much later
LAGS_GYR = (0.05, 0.15, 0.3, 0.6, 1.0)  # 3.3's lag curve (0.15 = LATER_GYR itself)
STRONG_GALAXY, STRONG_PAIR = 0.8, 0.5
EDGES = np.array([0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999, 1.0 + 1e-9])
BIN_LABELS = ["[0,0.5)", "[0.5,0.6)", "[0.6,0.7)", "[0.7,0.8)", "[0.8,0.9)", "[0.9,0.95)",
              "[0.95,0.99)", "[0.99,0.999)", ">=0.999"]
BETAS = (1.0, 0.75, 0.5, 0.25)
STATS = ("depth", "-u^2", "log alpha")


# --- 3.1 ------------------------------------------------------------------------------------------

def wspearman(a, b, w):
    """Spearman correlation weighted by ``w`` (the weighted Pearson of the ranks)."""
    ra, rb = rankdata(a), rankdata(b)
    da, db = ra - np.average(ra, weights=w), rb - np.average(rb, weights=w)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (np.average(da * db, weights=w)
                / np.sqrt(np.average(da * da, weights=w) * np.average(db * db, weights=w)))


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


# --- 3.2 ------------------------------------------------------------------------------------------

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


# --- 3.3 and 3.5 ----------------------------------------------------------------------------------

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
    """3.3 and 3.5 at ``s``: mass-weighted bin sums per beta (``old_rows``: which rows hold old stars)."""
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


# --- the worker -----------------------------------------------------------------------------------

_W = {}                                 # a worker's run and settings


def _init(path, source, snaps, frame, beta_run, vel, later, lags):
    """Open the run once per worker process, with what its snapshots share."""
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
    keep2, ids2 = components(p2)[1], run.row_labels(s2)[0]
    (k2, j2, dG2, dA2, _, pid2, _, _), _ = contested_pairs(p2, logG2, keep2, ids2)
    ac = autocorr_table(k, j, pid, dG, dA, k2, j2, pid2, dG2, dA2, s, run.time[s])

    old = seen_before(run, s, ids, OLD_GYR)
    C = calibration(run, s, s2, p, m, old, keep, _W["beta_run"])
    lag = persistence(run, s, _W["lags"].get(s, {}), p, m, old, keep)

    g, n_lab = np.unique(lab[lab != UNASSIGNED], return_counts=True)
    return (rows, pair_table(k, j, dG, dA, dLa, s, run.time[s]), np.array(counts, float), C, ac, lag,
            dict(zip(g.tolist(), n_lab.tolist())))


# --- the point ------------------------------------------------------------------------------------

def schedule(run, every):
    """The snapshots goal 3 reads: every ``every``-th one that has a snapshot >= LATER_GYR later, as
    ``{s: s'}`` (3.3's outcome and 3.2's autocorrelation partner), and each one's lag-curve partners,
    ``{s: {lag: s'}}``."""
    sn = list(run.snaps)
    t = np.array([run.time[s] for s in sn])
    later, lags = {}, {}
    for i in range(0, len(sn), every):
        after = np.flatnonzero(t >= t[i] + LATER_GYR)
        if after.size:
            later[int(sn[i])] = int(sn[after[0]])
        partners = {}
        for lag in LAGS_GYR:
            after_lag = np.flatnonzero(t >= t[i] + lag)
            if after_lag.size:
                partners[lag] = int(sn[after_lag[0]])
        if partners:
            lags[int(sn[i])] = partners
    return later, lags


def evaluate(run, out, every=10, workers=1, feedback=False, dependence=False, light_every=3):
    """Goal 3 on ``run`` (see the module docstring), written to ``out``; 3.2b and 3.4 when flagged."""
    rep = Report("Goal 3, probabilistic model", run.name, verdict=False)
    later, lags = schedule(run, every)
    picks = sorted(later)
    frame = run.config.get("stars", "none") != "free"
    beta_run = 1.0 if run.config.get("tempering") is None else float(run.config["tempering"])
    vel = {int(s): g.set_index("Sub_tree_id")[VEL]
           for s, g in run.tree[run.tree["Snapshot"].isin(picks)].groupby("Snapshot")}

    measured, galaxy_rows, pair_tables, autocorr_tables, per_snap = [], [], [], [], []
    contested, calib, peak = np.zeros(2), {}, {}
    lag_base, lag_sums, lag_snaps = np.zeros((2, EDGES.size - 1)), {}, 0
    parts = pool(_snapshot, picks, workers, _init,
                 (run.path, run.source, run.snaps, frame, beta_run, vel, later, lags), desc="3 snapshots")
    for s, part in zip(picks, parts):
        if part is None:
            continue
        measured.append(s)
        rows, table, counts, sums, autocorr, (mass, mr, lag_part), n_lab = part
        galaxy_rows += rows
        pair_tables.append(table)
        autocorr_tables.append(autocorr)
        contested += counts
        for key, v in sums.items():
            calib[key] = calib[key] + v if key in calib else v
        if beta_run in sums:
            per_snap.append((s, run.time[s], *over_under(sums[beta_run])))
        for g, n in n_lab.items():
            peak[g] = max(peak.get(g, 0), n)
        if len(lag_part) == len(LAGS_GYR):              # the lag curve pools the snapshots with every lag
            lag_snaps += 1
            lag_base += np.stack([mass, mr])
            for lag, arr in lag_part.items():
                lag_sums[lag] = lag_sums[lag] + arr if lag in lag_sums else arr

    rep.score("snapshots measured", len(measured))
    rep.table("snapshots (outcome of 3.3 at the first snapshot >= 150 Myr later)",
              "  " + " ".join(f"{s}->{later[s]}" for s in measured))
    galaxies = pd.DataFrame(galaxy_rows)
    pairs = pd.concat(pair_tables, ignore_index=True) if pair_tables else pd.DataFrame()
    autocorr = pd.concat(autocorr_tables, ignore_index=True) if autocorr_tables else pd.DataFrame()
    report_within(galaxies, rep)
    report_pairs(pairs, autocorr, contested, rep)
    report_calibration(calib, rep, beta_run)
    report_persistence(lag_base, lag_sums, lag_snaps, rep)

    per_snap = pd.DataFrame(per_snap, columns=["snap", "t", "O", "U"])
    plot_overlap(run, galaxies, pairs, autocorr, plots.run_plot(out, "model_overlap"))
    plot_calibration(run, calib, per_snap, beta_run, plots.run_plot(out, "model_calibration"))
    plot_persistence(run, lag_base, lag_sums, plots.run_plot(out, "model_persistence"))

    chosen = plots.chosen(run.host, peak, rep.episodes())
    if feedback or dependence:
        model_alpha.evaluate(run, out, rep, chosen, workers, feedback, dependence, light_every)
    items = []
    for g in chosen:
        mine = galaxies[galaxies["galaxy"] == g].sort_values("t") if not galaxies.empty else galaxies
        partners = pairs[(pairs["k"] == g) | (pairs["j"] == g)] if not pairs.empty else pairs
        if not mine.empty or not partners.empty:
            items.append((plots.galaxy_plot(out, g, "model_overlap"), g, mine, partners))
    plots.per_galaxy(_draw_galaxy, items, workers, desc="3 galaxy plots")

    rep.write(out / "model.txt")
    return rep


def report_within(G, rep):
    """3.1 SCOREs per statistic, and the table by stellar mass and kind (``G``: one row per galaxy and
    snapshot)."""
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
    G = G.assign(bin=mass_bin(G["mass"]), kind=np.where(G["satellite"], "satellite", "central"))
    cols = [f"soft {x}" for x in STATS if f"soft {x}" in G]
    by = G.groupby(["bin", "kind"], observed=True)
    tab = by[cols].median().join(by.size().rename("n"))
    rep.table("3.1 median rho soft by stellar mass and kind",
              tab.to_string(float_format=lambda v: f"{v:.3f}"))


def report_pairs(P, AC, contested, rep):
    """3.2 SCOREs and the largest pairs (``P``: per pair and snapshot; ``AC``: its autocorrelation;
    ``contested``: contested particles, and those with a pair left out)."""
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
    """3.3 (the run's own beta) and 3.5 (the other betas): SCOREs and reliability tables from the pooled
    bin sums ``C``."""
    if beta_run not in C:
        return
    for beta in [beta_run] + [b for b in BETAS if b in C and b != beta_run]:
        mass, count, mr, mp = C[beta]
        with np.errstate(invalid="ignore", divide="ignore"):
            rbar, pers = mr / mass, mp / mass
        tag = "3.3" if beta == beta_run else f"3.5 beta {beta}"
        over, under = over_under(C[beta])
        rep.score(f"{tag}: overconfidence O", over)
        rep.score(f"{tag}: apparent underconfidence U", under)
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
        rho, v = wmedian_in_time(ok, "rho", "n"), wmedian_in_time(ok, "v", "n")
        a2.plot(rho.index, rho, "o-", ms=3, label="rho(D_G, D_alpha)")
        a2.plot(v.index, v, "s-", ms=3, label="alpha share of the decision v")
    a2.axhline(0, color="0.6", lw=0.6)
    a2.set_ylabel("3.2 n-weighted median")
    a2.legend(fontsize=8)
    if not AC.empty:
        acG, acA = wmedian_in_time(AC, "ac_G", "n"), wmedian_in_time(AC, "ac_alpha", "n")
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


def main():
    args = cli.arguments(__doc__, every=10, light_every=True, flags={
        "feedback": "3.2b: replay the fit without alpha in EM (needs the run's stored log_alpha)",
        "dependence": "3.4: how much of alpha the Gaussians already contain",
    })
    run, out = cli.start("model", args)
    evaluate(run, out, every=args.every, workers=args.workers, feedback=args.feedback,
             dependence=args.dependence, light_every=args.light_every)


if __name__ == "__main__":                      # spawned workers import this file, never run it
    main()
