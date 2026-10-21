"""Goal 3: is the probabilistic model sound? It measures and reports; it gives no verdict.

    python3 evaluate/model.py --path RUN [--particles DIR] [--workers N] [--snaps A:B] [--every N]

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
overlapping spheres, can be larger, so this can leave out a little of a resolved one.)

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
    is constant per galaxy).
3.3 calibration. Contested (>= 2 candidates) old (present >= 1 Gyr before) stars at s, k = argmax r;
    outcome: labelled k at s', the first snapshot >= 150 Myr later. Left out: k without particles then,
    unassigned then, a satellite's star beyond its catalogue r_t then. Mass-weighted per bin of r_max:
    mean r, persistence p, gap = mean r - p. Reported: the table, overconfidence
    O = sum n max(gap, 0) / sum n, apparent underconfidence U (inflated: EM starts from the previous
    responsibilities, and wrong labels can persist), persistence at r_max >= 0.999.
3.4 tempering, when log alpha is stored and the run's own beta (header "tempering", 1 when absent) is
    > 0: the stored log alpha is beta_run log alpha; 3.3 with log r' = log r + (beta - beta_run) log alpha
    renormalised, for beta = 1, 0.75, 0.5, 0.25 other than the run's (the final E-step with the fitted
    Gaussians; EM would move them), and the share of the stars whose argmax changes. 3.3 itself is the
    run as it is (its own beta).

Writes evaluation/model.txt and .log; plots/run/: model_overlap.pdf (3.1 and 3.2 against time),
model_calibration.pdf (reliability curves, O and U against time); plots/g<id>/model_overlap.pdf (its
3.1 rho and its 3.2 rho and v with its partners, against time) for the plotted galaxies.
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

from evaluate import cli, plots
from evaluate.load import ABSENT, OLD_GYR, UNASSIGNED, Run, pool, seen_before
from evaluate.report import Report

MIN_PARTICLES, MIN_R, MIN_HARD, MIN_PAIR = 1e5, 50.0, 50, 30
LATER_GYR = 0.15                        # 3.3: the outcome is read at the first snapshot this much later
UNRESOLVED_RATIO = 10                   # the pipeline's UNRESOLVED_GROUP_RATIO
STRONG_GALAXY, STRONG_PAIR = 0.8, 0.5
EDGES = np.array([0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999, 1.0 + 1e-9])
BIN_LABELS = ["[0,0.5)", "[0.5,0.6)", "[0.6,0.7)", "[0.7,0.8)", "[0.8,0.9)", "[0.9,0.95)",
              "[0.95,0.99)", "[0.99,0.999)", ">=0.999"]
MASS_BINS = np.array([0.0, 1e7, 1e8, 1e9, np.inf])
BETAS = (1.0, 0.75, 0.5, 0.25)
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


def resolved(p):
    """Whether each stored pair belongs to a resolved group (see Scope): the galaxies linked through
    shared particles form a group; K >= 2 of them with fewer than 10 K particles are unresolved."""
    rows, gal = p["row"].astype(np.int64), p["galaxy"].astype(np.int64)
    if not rows.size:
        return np.zeros(0, bool)
    g, gi = np.unique(gal, return_inverse=True)
    gi = gi.ravel()
    o, start, count = by_row(rows)
    first = np.repeat(gi[o][start], count)           # each entry's particle's first galaxy
    graph = coo_matrix((np.ones(rows.size), (first, gi[o])), shape=(g.size, g.size))
    _, comp = connected_components(graph, directed=False)
    particles = np.bincount(comp[gi[o][start]], minlength=comp.max() + 1)
    galaxies = np.bincount(comp, minlength=comp.max() + 1)
    ok = (galaxies == 1) | (particles >= UNRESOLVED_RATIO * galaxies)
    return ok[comp[gi]]


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


def wspearman(a, b, w):
    """Spearman correlation weighted by ``w`` (the weighted Pearson of the ranks)."""
    ra, rb = rankdata(a), rankdata(b)
    da, db = ra - np.average(ra, weights=w), rb - np.average(rb, weights=w)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (np.average(da * db, weights=w)
                / np.sqrt(np.average(da * da, weights=w) * np.average(db * db, weights=w)))


def wmedian(x, w):
    """The ``w``-weighted median of ``x``."""
    if not x.size:
        return np.nan
    o = np.argsort(x)
    c = np.cumsum(w[o])
    return float(x[o][np.searchsorted(c, 0.5 * c[-1])])


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

def contested_pairs(p, logG, keep):
    """3.2: ``(k, j, D_G, D_alpha, Delta log alpha)`` over every candidate pair (k < j) of particles with
    >= 2 usable entries, and (contested particles, of them with an entry left out)."""
    rows = p["row"].astype(np.int64)
    use = keep & np.isfinite(p["log_resp"]) & np.isfinite(logG)
    n_all = np.bincount(rows)
    n_use = np.bincount(rows[use], minlength=n_all.size)
    counts = (int((n_all >= 2).sum()), int(((n_all >= 2) & (n_use < n_all)).sum()))

    o, start, count = by_row(rows[use])
    g, lr, lg, la = (x[use][o] for x in (p["galaxy"].astype(np.int64), p["log_resp"], logG, p["log_alpha"]))
    out = []
    for c in range(2, count.max(initial=1) + 1):
        base = start[count == c]
        for i, j in itertools.combinations(range(c), 2):
            a, b = base + i, base + j
            sign = np.where(g[a] < g[b], 1.0, -1.0)            # canonical order k < j
            dG = sign * (lg[a] - lg[b])
            out.append((np.minimum(g[a], g[b]), np.maximum(g[a], g[b]), dG,
                        sign * (lr[a] - lr[b]) - dG, sign * (la[a] - la[b])))
    if not out:
        return [np.array([])] * 5, counts
    return [np.concatenate(x) for x in zip(*out)], counts


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


def calibration(run, s, s2, p, m, old_rows, keep, beta_run):
    """3.3 and 3.4 at ``s``: mass-weighted bin sums per beta (``old_rows``: which rows hold old stars)."""
    C = {}
    rows = p["row"].astype(np.int64)
    ids = run.row_labels(s)[0]
    sel = keep & (np.bincount(rows)[rows] >= 2) & old_rows[rows]
    if not sel.any():
        return C
    g, lr, la = p["galaxy"].astype(np.int64)[sel], p["log_resp"][sel], p["log_alpha"][sel]
    stored = np.isfinite(la).all() and beta_run > 0
    k1 = best(rows[sel], g, lr)[1]
    for beta in ([beta_run] + [b for b in BETAS if b != beta_run]) if stored else [beta_run]:
        shift = (beta - beta_run) * la / beta_run if stored else 0.0
        rr, k, rmax = best(rows[sel], g, lr + shift)             # rows come in the same order every beta
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


def over_under(sums):
    """Overconfidence O and apparent underconfidence U of one beta's bin sums."""
    mass, _, mr, mp = sums
    ok = mass > 0
    if not ok.any():
        return np.nan, np.nan
    gap = mr[ok] / mass[ok] - mp[ok] / mass[ok]
    return (float((mass[ok] * np.maximum(gap, 0)).sum() / mass.sum()),
            float((mass[ok] * np.maximum(-gap, 0)).sum() / mass.sum()))


# --- the workers ----------------------------------------------------------------------------------

_W = {}                                 # a worker's run and settings


def _init(path, source, snaps, frame, beta_run, vel, later):
    run = Run(path, particles=source)
    run.snaps = snaps
    _W.update(run=run, frame=frame, beta_run=beta_run, vel=vel, later=later)


def _snapshot(s):
    """Goal 3 at ``s`` in a worker: galaxy rows, pair table, contested counts, calibration sums and the
    particle count of every label (None with too few particles)."""
    run = _W["run"]
    ids, lab = run.row_labels(s)
    if ids.size < MIN_PARTICLES:
        return None
    p = run.pairs(s)
    X, m = run.particles(s)
    keep = resolved(p)
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
    pairs, counts = contested_pairs(p, logG, keep)
    C = calibration(run, s, _W["later"][s], p, m, seen_before(run, s, ids, OLD_GYR), keep, _W["beta_run"])
    g, n_lab = np.unique(lab[lab != UNASSIGNED], return_counts=True)
    return (rows, pair_table(*pairs, s, run.time[s]), np.array(counts, float), C,
            dict(zip(g.tolist(), n_lab.tolist())))


# --- the point ------------------------------------------------------------------------------------

def evaluate(run, out, every=10, workers=1):
    rep = Report("Goal 3, probabilistic model", run.name, verdict=False)
    sn = list(run.snaps)
    t = np.array([run.time[s] for s in sn])
    later = {}
    for i in range(0, len(sn), every):
        after = np.flatnonzero(t >= t[i] + LATER_GYR)
        if after.size:
            later[int(sn[i])] = int(sn[after[0]])
    picks = sorted(later)
    frame = run.config.get("stars", "none") != "free"
    beta_run = 1.0 if run.config.get("tempering") is None else float(run.config["tempering"])
    vel = {int(s): g.set_index("Sub_tree_id")[VEL]
           for s, g in run.tree[run.tree["Snapshot"].isin(picks)].groupby("Snapshot")}

    G, P, C, contested, measured, per_snap, peak = [], [], {}, np.zeros(2), [], [], {}
    parts = pool(_snapshot, picks, workers, _init,
                 (run.path, run.source, run.snaps, frame, beta_run, vel, later), desc="3 snapshots")
    for s, part in zip(picks, parts):
        if part is None:
            continue
        measured.append(s)
        rows, table, counts, c, n_lab = part
        G += rows
        P.append(table)
        contested += counts
        for key, v in c.items():
            C[key] = C[key] + v if key in C else v
        if beta_run in c:
            per_snap.append((s, run.time[s], *over_under(c[beta_run])))
        for g, n in n_lab.items():
            peak[g] = max(peak.get(g, 0), n)

    rep.score("snapshots measured", len(measured))
    rep.table("snapshots (outcome of 3.3 at the first snapshot >= 150 Myr later)",
              "  " + " ".join(f"{s}->{later[s]}" for s in measured))
    G = pd.DataFrame(G)
    P = pd.concat(P, ignore_index=True) if P else pd.DataFrame()
    report_within(G, rep)
    report_pairs(P, contested, rep)
    report_calibration(C, rep, beta_run)

    per_snap = pd.DataFrame(per_snap, columns=["snap", "t", "O", "U"])
    plot_overlap(run, G, P, plots.run_plot(out, "model_overlap"))
    plot_calibration(run, C, per_snap, beta_run, plots.run_plot(out, "model_calibration"))
    items = []
    for g in plots.chosen(run.host, peak, rep.episodes()):
        mine = G[G["galaxy"] == g].sort_values("t") if not G.empty else G
        pairs = P[(P["k"] == g) | (P["j"] == g)] if not P.empty else P
        if not mine.empty or not pairs.empty:
            items.append((plots.galaxy_plot(out, g, "model_overlap"), g, mine, pairs))
    plots.per_galaxy(_draw_galaxy, items, workers, desc="3 galaxy plots")
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


def report_pairs(P, contested, rep):
    rep.score("3.2 contested particles (summed over snapshots)", int(contested[0]))
    rep.score("3.2 share with a pair left out (underflowed r, no Gaussian, unresolved group)",
              float(contested[1] / max(contested[0], 1)))
    if P.empty:
        return
    P = P[P["rho"].notna()]
    n = P["n"].to_numpy()
    rep.score("3.2 galaxy pairs x snapshots (>= 30 particles)", len(P))
    rep.score("3.2 n-weighted median rho(D_G, D_alpha)", wmedian(P["rho"].to_numpy(), n))
    rep.score("3.2 n-weighted median alpha share v", wmedian(P["v"].to_numpy(), n))
    rep.score(f"3.2 n-weighted share rho > {STRONG_PAIR}", float(n[P["rho"] > STRONG_PAIR].sum() / n.sum()))
    if "rho stored" in P:
        rep.score("3.2 n-weighted median rho(D_alpha, stored Delta log alpha)",
                  wmedian(P["rho stored"].to_numpy(), n))
    top = P.sort_values("n", ascending=False).head(10).drop(columns="t")
    rep.table("3.2 largest galaxy pairs", top.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


def report_calibration(C, rep, beta_run):
    if beta_run not in C:
        return
    for beta in [beta_run] + [b for b in BETAS if b in C and b != beta_run]:
        mass, count, mr, mp = C[beta]
        with np.errstate(invalid="ignore", divide="ignore"):
            rbar, pers = mr / mass, mp / mass
        tag = "3.3" if beta == beta_run else f"3.4 beta {beta}"
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


# --- plots ----------------------------------------------------------------------------------------

def plot_overlap(run, G, P, pdf):
    """3.1 median rho per statistic, and 3.2 n-weighted median rho and v, against time."""
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
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
    a2.set_xlabel("t [Gyr]")
    a2.legend(fontsize=8)
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
    args = cli.arguments(__doc__, every=True)
    run, out = cli.start("model", args)
    evaluate(run, out, every=args.every, workers=args.workers)


if __name__ == "__main__":                      # spawned workers import this file, never run it
    main()
