"""Goal 3: is the probabilistic model sound? It measures and reports; it gives no verdict.

The E-step is ``log r_nk = log N_k(x_n) + log pi_k + log alpha_nk - log Z_n`` (mixture/base.py). The
stored mean, covariance and weight are the Bayesian GMM's posterior means, so ``log N + log pi`` read
from them differs from the E-step's own terms by a constant per galaxy, C_k. Statistics are oriented
"more bound = larger"; Spearman correlations only see their order.

Scope: every --every-th snapshot (default 10) with >= 1e5 particles and a snapshot two steps later;
galaxies with a stored Gaussian.

3.1 overlap inside a galaxy. For every galaxy with sum r >= 50 over its pairs: the Spearman correlation
    of log N_k with each statistic, weighted by r (soft) and over its >= 50 hard members (hard):
      depth      -E, the order of b, of t = -ln w and of the rank of b (energy, phase, rank models);
      -u^2       u^2 = v^2 / (v^2 - 2E), v relative to the tree velocity, the energy's frame with
                 pinned or no stars (kinematic model; not measured with free stars);
      log alpha  when the run stored it.
    Reported per statistic: galaxies, median rho (soft, hard), soft-mass-weighted median, share with
    rho > 0.8; medians by stellar mass (sum m r) and central / satellite.
3.2 overlap where decisions are made. For every pair (k, j) of candidates of a particle with >= 2:
    D_G = Delta(log N + log pi), D_alpha = Delta log r - D_G (= Delta log alpha + C_k - C_j). Per
    (k, j, s) with >= 30 particles: rho = Pearson(D_G, D_alpha) and v = Var(D_alpha) / (Var(D_alpha) +
    Var(D_G)), alpha's share of the spread of the decision. Reported: n-weighted medians, n-weighted share
    with rho > 0.5, share of contested particles with a pair left out (underflowed r, no Gaussian); with
    stored log alpha, the median Pearson(D_alpha, Delta log alpha) (1 when C_k is constant per galaxy).
3.3 calibration. Contested (>= 2 candidates) old (first seen >= 1 Gyr before) stars at s, k = argmax r;
    outcome: labelled k two snapshots later. Left out: k without particles then, unassigned then, a
    satellite's star beyond its catalogue r_t then. Mass-weighted per bin of r_max: mean r, persistence p,
    gap = mean r - p. Reported: the table, overconfidence O = sum n max(gap, 0) / sum n, apparent
    underconfidence U (inflated: EM starts from the previous responsibilities, and wrong labels can
    persist), persistence at r_max >= 0.999.
3.4 tempering, when log alpha is stored and the run's own beta (header "tempering", 1 when absent) is
    > 0: the stored log alpha is beta_run log alpha; 3.3 with log r' = log r + (beta - beta_run) log alpha
    renormalised, for beta = 1, 0.75, 0.5, 0.25 other than the run's (the final E-step with the fitted
    Gaussians; EM would move them), and the share of the stars whose argmax changes. 3.3 itself is
    the run as it is (its own beta).
"""
import itertools

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import solve_triangular
from scipy.stats import rankdata

from evaluate.load import ABSENT, UNASSIGNED, first_seen
from evaluate.report import Report

MIN_PARTICLES, MIN_R, MIN_HARD, MIN_PAIR = 1e5, 50.0, 50, 30
AHEAD, OLD_GYR = 2, 1.0
STRONG_GALAXY, STRONG_PAIR = 0.8, 0.5
EDGES = np.array([0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999, 1.0 + 1e-9])
MASS_BINS = np.array([0.0, 1e7, 1e8, 1e9, np.inf])
BETAS = (1.0, 0.75, 0.5, 0.25)
STATS = ("depth", "-u^2", "log alpha")
POS, VEL = ["position_x", "position_y", "position_z"], ["velocity_x", "velocity_y", "velocity_z"]


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
    """Order sorting ``rows``, and the start and size of each particle's run of entries in it."""
    o = np.argsort(rows, kind="stable")
    start = np.flatnonzero(np.r_[True, rows[o][1:] != rows[o][:-1]]) if rows.size else np.array([], np.int64)
    return o, start, np.diff(np.r_[start, rows.size])


def galaxy_terms(p, X, gauss):
    """``log N_k(x_n) + log pi_k`` at every stored pair; NaN without a positive-definite Gaussian."""
    out = np.full(p["row"].size, np.nan)
    for k, e in groups(p["galaxy"]):
        if int(k) in gauss:
            mean, cov, w = gauss[int(k)]
            try:
                out[e] = log_gauss(X[p["row"][e].astype(np.int64)], mean, cov) + np.log(w)
            except np.linalg.LinAlgError:
                pass
    return out


def wspearman(a, b, w):
    """Spearman correlation weighted by ``w`` (weighted Pearson of the ranks)."""
    ra, rb = rankdata(a), rankdata(b)
    da, db = ra - np.average(ra, weights=w), rb - np.average(rb, weights=w)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.average(da * db, weights=w) / np.sqrt(np.average(da * da, weights=w) * np.average(db * db, weights=w))


def wmedian(x, w):
    o = np.argsort(x)
    c = np.cumsum(w[o])
    return float(x[o][np.searchsorted(c, 0.5 * c[-1])]) if x.size else np.nan


def within(p, logG, stats, lab, m, satellites):
    """3.1: one row per galaxy."""
    r = np.exp(p["log_resp"])
    rows = p["row"].astype(np.int64)
    out = []
    for k, e in groups(p["galaxy"]):
        e = e[np.isfinite(logG[e]) & (r[e] > 0)]
        if r[e].sum() < MIN_R:
            continue
        hard = lab[rows[e]] == k
        row = {"galaxy": int(k), "mass": float((m[rows[e]] * r[e]).sum()), "satellite": int(k) in satellites}
        for name, x in stats.items():
            ok = np.isfinite(x[e])
            row[f"soft {name}"] = wspearman(logG[e][ok], x[e][ok], r[e][ok]) if r[e][ok].sum() >= MIN_R else np.nan
            h = ok & hard
            row[f"hard {name}"] = wspearman(logG[e][h], x[e][h], np.ones(h.sum())) if h.sum() >= MIN_HARD else np.nan
        out.append(row)
    return out


def contested_pairs(p, logG):
    """3.2: ``(k, j, D_G, D_alpha, Delta log alpha)`` over every candidate pair (k < j) of particles with >= 2
    usable entries, and (contested particles, of them with an entry left out)."""
    rows = p["row"].astype(np.int64)
    use = np.isfinite(p["log_resp"]) & np.isfinite(logG)
    n_all, n_use = np.bincount(rows), np.bincount(rows[use], minlength=rows.max(initial=-1) + 1)
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


def pair_table(k, j, dG, dA, dLa, s):
    """3.2 per (k, j) at ``s`` with >= 30 particles: n, rho, v (and the stored-alpha check)."""
    if k.size == 0:
        return pd.DataFrame()
    key, inv = np.unique(np.stack([k, j]), axis=1, return_inverse=True)
    inv = inv.ravel()
    S = lambda v: np.bincount(inv, weights=v, minlength=key.shape[1])
    n = S(np.ones(k.size))

    def moments(x, y):
        dx, dy = x - (S(x) / n)[inv], y - (S(y) / n)[inv]
        vx, vy = S(dx * dx) / n, S(dy * dy) / n
        with np.errstate(invalid="ignore", divide="ignore"):
            return S(dx * dy) / n / np.sqrt(vx * vy), vx, vy

    rho, vG, vA = moments(dG, dA)
    with np.errstate(invalid="ignore", divide="ignore"):
        t = pd.DataFrame({"snap": s, "k": key[0], "j": key[1], "n": n, "rho": rho, "v": vA / (vA + vG)})
    if np.isfinite(dLa).all():
        t["rho stored"] = moments(dA, dLa)[0]
    return t[t["n"] >= MIN_PAIR]


def best(rows, g, x):
    """Per particle (entries in any order): its rows, argmax galaxy and normalised top value of ``exp(x)``."""
    o, start, count = by_row(rows)
    rows, g, x = rows[o], g[o], x[o]
    seg = np.repeat(np.arange(start.size), count)
    mx = np.maximum.reduceat(x, start)
    lse = mx + np.log(np.add.reduceat(np.exp(x - mx[seg]), start))
    top = np.lexsort((-x, seg))[start]
    return rows[top], g[top], np.exp(x[top] - lse)


def outcome(run, s2, ids, k):
    """3.3 at the later snapshot: (counted, persisted) for stars ``ids`` predicted in galaxies ``k``."""
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
        beyond = sat & (np.linalg.norm(X2[j, :3] - gal[POS].reindex(k).to_numpy(np.float64), axis=1) > rt)
    counted = found & np.isin(k, np.unique(lab2)) & (L2 != UNASSIGNED) & ~beyond
    return counted, L2 == k


def calibration(run, s, s2, p, m, t_first, C, beta_run):
    """3.3 and 3.4 at ``s``: adds mass-weighted bin sums per beta to ``C``."""
    rows = p["row"].astype(np.int64)
    ids = run.row_labels(s)[0]
    multi = np.bincount(rows)[rows] >= 2
    old = run.time[s] - t_first[ids[rows]] >= OLD_GYR
    sel = multi & old
    if not sel.any():
        return
    g, lr, la = p["galaxy"].astype(np.int64)[sel], p["log_resp"][sel], p["log_alpha"][sel]
    stored = np.isfinite(la).all() and beta_run > 0
    k1 = best(rows[sel], g, lr)[1]
    for beta in [beta_run] + [b for b in BETAS if b != beta_run] if stored else [beta_run]:
        shift = (beta - beta_run) * la / beta_run if stored else 0.0
        rr, k, rmax = best(rows[sel], g, lr + shift)                 # rows: same order every beta
        counted, persisted = outcome(run, s2, ids[rr], k)
        w = m[rr] * counted
        b = np.digitize(rmax, EDGES) - 1
        acc = C.setdefault(beta, np.zeros((4, EDGES.size - 1)))
        acc += np.stack([np.bincount(b, weights=w, minlength=EDGES.size - 1),
                         np.bincount(b, weights=counted.astype(float), minlength=EDGES.size - 1),
                         np.bincount(b, weights=w * rmax, minlength=EDGES.size - 1),
                         np.bincount(b, weights=w * persisted, minlength=EDGES.size - 1)])
        C.setdefault(f"changed {beta}", np.zeros(2))[:] += [(w * (k != k1)).sum(), w.sum()]


def evaluate(run, out, every=10):
    rep = Report("Goal 3, probabilistic model", run.name, verdict=False)
    sn = list(run.snaps)
    picks = [sn[i] for i in range(0, len(sn) - AHEAD, every)]
    first = first_seen(run, max(picks))
    time_of = np.array([run.time.get(s, np.nan) for s in range(int(max(sn)) + 1)])
    t_first = time_of[np.maximum(first, 0)]
    frame = run.config.get("stars", "none") != "free"
    beta_run = 1.0 if run.config.get("tempering") is None else float(run.config["tempering"])
    G, P, C, contested, measured = [], [], {}, np.zeros(2), []
    for s in picks:
        ids, lab = run.row_labels(s)
        if ids.size < MIN_PARTICLES:
            continue
        measured.append(int(s))
        s2 = sn[sn.index(s) + AHEAD]
        p = run.pairs(s)
        X, m = run.particles(s)
        logG = galaxy_terms(p, X, run.gaussians(s))
        stats = {"depth": -p["energy"]}
        if frame:
            v_tree = run.tree[run.tree["Snapshot"] == s].set_index("Sub_tree_id")[VEL].reindex(p["galaxy"].astype(np.int64))
            v2 = ((X[p["row"].astype(np.int64), 3:] - v_tree.to_numpy(np.float64)) ** 2).sum(axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                stats["-u^2"] = -v2 / (v2 - 2.0 * p["energy"])
        if np.isfinite(p["log_alpha"]).any():
            stats["log alpha"] = p["log_alpha"]
        G += [dict(r, snap=s) for r in within(p, logG, stats, lab, m, set(run.hosts(s)))]
        pairs, counts = contested_pairs(p, logG)
        P.append(pair_table(*pairs, s))
        contested += counts
        calibration(run, s, s2, p, m, t_first, C, beta_run)
        print(f"snapshot {s}: {len(G)} galaxy rows, {sum(len(t) for t in P)} pair rows", flush=True)

    rep.score("snapshots measured", len(measured))
    rep.table("snapshots", "  " + " ".join(map(str, measured)))
    G, P = pd.DataFrame(G), pd.concat(P, ignore_index=True) if P else pd.DataFrame()
    report_within(G, rep)
    report_pairs(P, contested, rep)
    report_calibration(C, rep, beta_run)
    plot(run, G, P, C, out / "model.png")
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
        rep.score(f"3.1 {name}: mass-weighted median rho soft", wmedian(soft[ok].to_numpy(), G.loc[ok, "mass"].to_numpy()))
        rep.score(f"3.1 {name}: share rho soft > {STRONG_GALAXY}", float((soft[ok] > STRONG_GALAXY).mean()))
    G = G.assign(bin=pd.cut(G["mass"], MASS_BINS, labels=["<1e7", "1e7-1e8", "1e8-1e9", ">1e9"]),
                 kind=np.where(G["satellite"], "satellite", "central"))
    cols = [f"soft {x}" for x in STATS if f"soft {x}" in G]
    tab = G.groupby(["bin", "kind"], observed=True)[cols].median().join(
        G.groupby(["bin", "kind"], observed=True).size().rename("n"))
    rep.table("3.1 median rho soft by stellar mass and kind", tab.to_string(float_format=lambda v: f"{v:.3f}"))


def report_pairs(P, contested, rep):
    rep.score("3.2 contested particles (summed over snapshots)", int(contested[0]))
    rep.score("3.2 share with a pair left out (underflowed r, no Gaussian)", float(contested[1] / max(contested[0], 1)))
    if P.empty:
        return
    P = P[P["rho"].notna()]
    n = P["n"].to_numpy()
    rep.score("3.2 galaxy pairs x snapshots (>= 30 particles)", len(P))
    rep.score("3.2 n-weighted median rho(D_G, D_alpha)", wmedian(P["rho"].to_numpy(), n))
    rep.score("3.2 n-weighted median alpha share v", wmedian(P["v"].to_numpy(), n))
    rep.score(f"3.2 n-weighted share rho > {STRONG_PAIR}", float(n[P["rho"] > STRONG_PAIR].sum() / n.sum()))
    if "rho stored" in P:
        rep.score("3.2 n-weighted median rho(D_alpha, stored Delta log alpha)", wmedian(P["rho stored"].to_numpy(), n))
    top = P.sort_values("n", ascending=False).head(10)
    rep.table("3.2 largest galaxy pairs", top.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


def report_calibration(C, rep, beta_run):
    if beta_run not in C:
        return
    labels = ["[0,0.5)", "[0.5,0.6)", "[0.6,0.7)", "[0.7,0.8)", "[0.8,0.9)", "[0.9,0.95)", "[0.95,0.99)",
              "[0.99,0.999)", ">=0.999"]
    for beta in [beta_run] + [b for b in BETAS if b in C and b != beta_run]:
        mass, count, mr, mp = C[beta]
        with np.errstate(invalid="ignore", divide="ignore"):
            rbar, pers = mr / mass, mp / mass
        gap = rbar - pers
        ok = mass > 0
        tag = "3.3" if beta == beta_run else f"3.4 beta {beta}"
        rep.score(f"{tag}: overconfidence O", float((mass[ok] * np.maximum(gap[ok], 0)).sum() / mass.sum()))
        rep.score(f"{tag}: apparent underconfidence U", float((mass[ok] * np.maximum(-gap[ok], 0)).sum() / mass.sum()))
        rep.score(f"{tag}: persistence at r_max >= 0.999", float(pers[-1]))
        if beta != beta_run:
            ch = C[f"changed {beta}"]
            rep.score(f"{tag}: share whose argmax changes", float(ch[0] / ch[1]))
        tab = pd.DataFrame({"mass share": mass / mass.sum(), "stars": count.astype(int), "mean r": rbar,
                            "persistence": pers, "gap": gap}, index=labels)
        rep.table(f"{tag} reliability (contested old stars, two snapshots later)",
                  tab.to_string(float_format=lambda v: f"{v:.4f}"))


def plot(run, G, P, C, png):
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    for i, name in enumerate([x for x in STATS if not G.empty and f"soft {x}" in G]):
        ax[0].semilogx(G["mass"], G[f"soft {name}"], ".", ms=3, color=f"C{i}", label=name)
    ax[0].set_xlabel("stellar mass (sum m r)")
    ax[0].set_ylabel("3.1 Spearman rho(log N_k, statistic), soft")
    ax[0].legend(fontsize=8)
    if not P.empty:
        ok = P["rho"].notna()
        ax[1].hist(P.loc[ok, "rho"], bins=40, range=(-1, 1), weights=P.loc[ok, "n"], color="C0")
    ax[1].set_xlabel("3.2 rho(D_G, D_alpha) per galaxy pair")
    ax[1].set_ylabel("contested particles")
    for i, beta in enumerate(sorted((b for b in C if not isinstance(b, str)), reverse=True)):
        mass, _, mr, mp = C[beta]
        ok = mass > 0
        ax[2].plot(mr[ok] / mass[ok], mp[ok] / mass[ok], "o-", ms=3, color=f"C{i}", label=f"beta {beta}")
    ax[2].plot([0, 1], [0, 1], ":", color="0.5")
    ax[2].set_xlabel("3.3 mean r_max")
    ax[2].set_ylabel("persistence two snapshots later")
    ax[2].legend(fontsize=8)
    fig.suptitle(f"{run.name}: goal 3")
    fig.tight_layout()
    fig.savefig(png, dpi=110)
    plt.close(fig)
