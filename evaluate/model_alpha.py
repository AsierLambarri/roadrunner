"""Goal 3's measurements behind flags, in a pass of their own over every --light-every-th snapshot:
3.2b (--feedback), does alpha feed back into the fit, and 3.4 (--dependence), how much of alpha the
Gaussians already contain. model.py's docstring defines both; model.py calls ``evaluate`` here.

3.4's regression, in short. Per contested pair (k, j), alpha's push on the decision is
Y = Delta log alpha (stored, when the run kept it; else D_alpha from 3.2: the two differ by a constant
per pair, which an intercept absorbs). The Gaussian view is what the mixture sees of a star, its
Mahalanobis distances to the two galaxies, F_G = (d2_k, d2_j); the physical view is what alpha is built
from, distances and speeds relative to the two tree halos, F_P = (r_k, |v - v_k|, r_j, |v - v_j|).
R2_G on F_G is the redundancy already in the Gaussians; R2_phys on F_P checks the regressor's capacity
(expected ~1); u = Var(Y - fhat_G(F_G)) / Var(D) = (1 - R2_G) Var(Y) / Var(D), D = Delta log r, is the
push the Gaussians cannot express, as a share of the decision.

Out-of-sample R^2 is conservative: the early-stopped regressor, trained on half the stars, reads smooth
alpha-like functions low by 0.02 at 5000 stars and 0.01 at 20 000 (left as is), and more below. Under
DEP_BOOT stars, R2_G is therefore corrected by the procedure's own bias at that size and shape,
measured on synthetic copies with a known answer (a parametric bootstrap, ``corrected_r2``). Tested on
two shapes, it is within 0.03 of the truth from 500 stars (MIN_DEP), with a standard error that matches
the scatter; below 500 stars the correction falls short on bumpy shapes (0.11 at 200), so those pairs
are only counted. R2_phys, a capacity check, stays raw.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import solve_triangular
from sklearn.ensemble import HistGradientBoostingRegressor

from evaluate import bgmm, plots
from evaluate.load import POS, VEL, Run, pool, wmedian, wmedian_in_time
from evaluate.pairs import (MIN_PAIR, MIN_PARTICLES, best, by_row, components, contested_pairs, galaxy_terms,
                            groups, mass_bin, pair_entries)

REPRO_TOL = 1e-2                        # 3.2b: a component is left out when its median |delta log r| is above
SHARP = 0.999                           # 3.2b: r_max this high or above counts as sharp
MAX_ITER = 300                          # 3.2b: EM steps per chain at most
GAMMA_MIN_VAR = 0.01                    # 3.2b: gamma needs Var(Delta log alpha) >= this x Var(D_A)
MIN_DEP = 500                           # 3.4: stars per (k, j) to fit the regressors
DEP_BOOT, DEP_B = 5000, 4               # 3.4: below DEP_BOOT stars, R^2 is corrected with DEP_B copies
DEP_CAP = 20_000                        # 3.4: a pair's stars are capped by a fixed-seed random sample
HGB_KW = dict(max_iter=300, learning_rate=0.1, max_leaf_nodes=15, early_stopping=True,  # 3.4's regressor
              validation_fraction=0.15, n_iter_no_change=10, random_state=0)


# --- 3.4 ------------------------------------------------------------------------------------------

def mahalanobis2(x, mean, cov):
    """``(x - mean)^T cov^-1 (x - mean)`` of the rows of ``x``; ``cov`` full (d, d), diagonal (d,) or
    spherical () (same Cholesky pattern as ``log_gauss``, without the normalising terms)."""
    cov = np.diag(np.broadcast_to(cov, mean.shape)) if np.ndim(cov) < 2 else cov
    L = np.linalg.cholesky(cov)
    z = solve_triangular(L, (x - mean).T, lower=True)
    return (z * z).sum(0)


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


# --- 3.2b -----------------------------------------------------------------------------------------

def pd_ok(cov):
    """Whether ``cov`` is positive-definite (a Cholesky succeeds): a stored Gaussian can fail this even
    with a finite count."""
    try:
        np.linalg.cholesky(cov)
        return True
    except np.linalg.LinAlgError:
        return False


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

    post_A, first_A, it_A, conv_A, rows_c, gal_c, g_r_A, g_A = bgmm.replay(
        X, rows, gal, galaxies, la, prior, theta_A, with_alpha=True, max_iter=MAX_ITER)
    post_B, first_B, it_B, conv_B, _, _, g_r_B, g_B = bgmm.replay(
        X, rows, gal, galaxies, la, prior, theta_A, with_alpha=False, max_iter=MAX_ITER)

    la_c = la[contested]
    g1_A = bgmm.estep(X, rows_c, gal_c, galaxies, la_c, first_A)[0]     # the Gaussian term: alpha-free
    g1_B = bgmm.estep(X, rows_c, gal_c, galaxies, la_c, first_B)[0]

    t = run.time[s]
    idx_c = np.searchsorted(galaxies, gal_c)
    own_mass = np.bincount(idx_c, weights=m[rows_c], minlength=galaxies.size)   # the stars each competes for
    gal_row = {"snap": s, "t": t, "galaxy": galaxies, "no_history": no_history, "mass": own_mass,
               "kl": sym_kl(post_A.m, post_A.Sigma, post_B.m, post_B.Sigma),
               "kl_written": sym_kl(theta_A.m, theta_A.Sigma, post_A.m, post_A.Sigma),
               "dlogdet": (np.linalg.slogdet(post_B.Sigma)[1] - np.linalg.slogdet(post_A.Sigma)[1]),
               "shift": np.sqrt(np.einsum("ki,kij,kj->k", post_B.m - post_A.m, np.linalg.inv(post_A.Sigma),
                                           post_B.m - post_A.m)),
               "dN": post_B.N - post_A.N}

    fixed = ~contested
    idx = np.searchsorted(galaxies, gal)
    M_fixed = np.bincount(idx[fixed], weights=m[rows[fixed]], minlength=galaxies.size)
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
                iterations=(it_A, it_B), converged=(conv_A, conv_B), resolved_mass=contested_mass,
                no_history_mass=no_history_mass)


# --- the worker -----------------------------------------------------------------------------------

_W = {}                                 # a worker's run and settings


def _init(path, source, snaps, tree_state, feedback, dependence):
    """Open the run once per worker process, with what its snapshots share."""
    run = Run(path, particles=source)
    run.snaps = snaps
    _W.update(run=run, tree_state=tree_state, feedback=feedback, dependence=dependence)


def _snapshot(s):
    """3.4 (``--dependence``) and 3.2b (``--feedback``, snapshots after the first with a stored log alpha)
    at ``s`` in a worker: the dependence table, the feedback snapshot (None when skipped) and each
    galaxy's responsibility-weighted stellar mass (for the tables by mass)."""
    run = _W["run"]
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
    if _W["dependence"]:
        (k, j, dG, dA, dLa, pid, a, b), _ = contested_pairs(p, logG, keep, ids)
        tree = _W["tree_state"].get(s)
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
    if _W["feedback"] and np.isfinite(p["log_alpha"]).any():
        s_idx = int(np.searchsorted(run.snaps, s))
        if s_idx > 0:
            s_prev = int(run.snaps[s_idx - 1])
            tree_s, tree_p = _W["tree_state"].get(s), _W["tree_state"].get(s_prev)
            if tree_s is not None and tree_p is not None:
                fb = _feedback_snapshot(run, s, p, X, m, comp, keep, gauss, run.gaussians(s_prev),
                                         tree_s, tree_p, _W["dependence"])

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


# --- the pass -------------------------------------------------------------------------------------

def evaluate(run, out, rep, chosen, workers, feedback, dependence, light_every):
    """3.2b (``feedback``) and 3.4 (``dependence``) over every ``light_every``-th snapshot of ``run``: their
    SCOREs and tables into ``rep``, their plots into ``out`` (the run's, and the ``chosen`` galaxies')."""
    snaps = sorted(run.snaps.tolist())[::light_every]
    tree_state = {int(s): g.set_index("Sub_tree_id")[POS + VEL + ["mass"]]       # every snapshot: 3.2b
                  for s, g in run.tree[run.tree["Snapshot"].isin(run.snaps)].groupby("Snapshot")}  # reads s-1
    init = (run.path, run.source, run.snaps, tree_state, feedback, dependence)
    parts = pool(_snapshot, snaps, workers, _init, init, desc="3 light snapshots")
    dep_tables, dep_few, fb_list, times, gal_mass = [], [0, 0], [], [], {}
    for s, part in zip(snaps, parts):
        times.append(run.time[s])
        if part is None:
            fb_list.append(None)
            continue
        dep, few, fb, masses = part
        dep_tables.append(dep)
        dep_few[0] += few[0]
        dep_few[1] += few[1]
        fb_list.append(fb)
        gal_mass.update({(s, k): v for k, v in masses.items()})
    dep = pd.concat(dep_tables, ignore_index=True) if dep_tables else pd.DataFrame()

    dep_B = None                                       # 3.4's R2_G with chain B's Gaussians, from 3.2b
    if feedback:
        fb_gal, fb_gamma, fb_gamma1, dep_B, fb_per_snap = assemble_feedback(fb_list, snaps, times)
    if dependence:
        report_dependence(dep, tuple(dep_few), gal_mass, rep, depB=dep_B)
        plot_dependence(run, dep, dep_B, plots.run_plot(out, "model_dependence"))
    if feedback:
        report_feedback(fb_gal, fb_gamma, fb_gamma1, gal_mass, fb_per_snap, rep)
        plot_feedback(run, fb_per_snap, fb_gal, fb_gamma, plots.run_plot(out, "model_feedback"))

    if feedback:
        items = []
        for g in chosen:
            mine = fb_gal[fb_gal["galaxy"] == g].sort_values("t") if not fb_gal.empty else fb_gal
            pairs = fb_gamma[(fb_gamma["k"] == g) | (fb_gamma["j"] == g)] if not fb_gamma.empty else fb_gamma
            if not mine.empty or not pairs.empty:
                items.append((plots.galaxy_plot(out, g, "model_feedback"), g, mine, pairs))
        plots.per_galaxy(_draw_galaxy_feedback, items, workers, desc="3 feedback galaxy plots")
    if dependence:
        items = []
        for g in chosen:
            pairs = dep[(dep["k"] == g) | (dep["j"] == g)] if not dep.empty else dep
            if not pairs.empty:
                items.append((plots.galaxy_plot(out, g, "model_dependence"), g, pairs))
        plots.per_galaxy(_draw_galaxy_dependence, items, workers, desc="3 dependence galaxy plots")


def galaxy_mass(df, mass, gal_col="k"):
    """``df``'s own ``(snap, gal_col)`` read against ``mass`` ({(snap, galaxy): responsibility-weighted
    stellar mass}, from the worker's ``gal_mass``); NaN where missing."""
    return np.array([mass.get((s, g), np.nan) for s, g in zip(df["snap"], df[gal_col])])


def assemble_feedback(fb_list, snaps, times):
    """Collect every snapshot's ``_feedback_snapshot`` output into run-wide tables: per-galaxy rows, the
    converged and one-step gamma tables, R2_G(B) (with --dependence too) and one row per snapshot of the
    scalar SCOREs (coverage, residuals, labels, sharpening)."""
    gal_rows, gammas, gamma1s, dep_Bs, per_snap = [], [], [], [], []
    for s, t, fb in zip(snaps, times, fb_list):
        if fb is None:
            continue
        parts = fb["parts"]
        replayed = sum(out["resolved_mass"] for out in parts)
        no_hist = sum(out["no_history_mass"] for out in parts)
        med = [out["repro"]["median"] for out in parts if out["repro"]["n"]]
        p99 = [out["repro"]["p99_r"] for out in parts if out["repro"]["n"]]
        nrep = [out["repro"]["n"] for out in parts if out["repro"]["n"]]
        # components without contested entries are trivial and would swamp the medians below
        contested_parts = [out for out in parts if out["resolved_mass"] > 0]
        its_mass = np.array([out["resolved_mass"] for out in contested_parts])
        its_A = np.array([out["iterations"][0] for out in contested_parts])
        its_B = np.array([out["iterations"][1] for out in contested_parts])
        conv = np.array([out["converged"] for out in contested_parts], bool).reshape(-1, 2)
        not_converged = float(its_mass[~conv.all(axis=1)].sum())
        w, changed, host = _joined(parts, "w"), _joined(parts, "changed", bool), _joined(parts, "host", bool)
        sharp_A, sharp_B = _joined(parts, "sharp_A", bool), _joined(parts, "sharp_B", bool)
        h_A, h_B = _joined(parts, "entropy_A"), _joined(parts, "entropy_B")

        for out in parts:
            gal_rows.append(pd.DataFrame(out["gal_row"]))
            gammas.append(out["gamma"])
            gamma1s.append(out["gamma1"])
            dep_Bs.append(out["dep_B"])

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
                "unconverged_A": int((~conv[:, 0]).sum()),
                "unconverged_B": int((~conv[:, 1]).sum()),
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
    return gal, gamma, gamma1, depB, pd.DataFrame(per_snap)


def _joined(parts, key, dtype=float):
    """The ``labels[key]`` arrays of a snapshot's replayed components, end to end."""
    return np.concatenate([out["labels"][key] for out in parts]) if parts else np.zeros(0, dtype)


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
            part, n_part = DEP[sel], n[sel.to_numpy()]
            rep.score(f"3.4 n-weighted median R2_G, {tag}", wmedian(part["r2_G"].to_numpy(), n_part))
            se_ok = part["r2_G se"].notna().to_numpy()
            if se_ok.any():
                rep.score(f"3.4 n-weighted median R2_G standard error, {tag}",
                          wmedian(part["r2_G se"].to_numpy()[se_ok], n_part[se_ok]))
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
    rep.score("3.2b median reproduction residual, components' p99 |delta r| (contested entries)",
              float(per_snap["repro_p99_r"].median()))
    rep.score("3.2b components left out (summed over snapshots)", int(per_snap["left_out"].sum()))
    rep.score("3.2b median mass-weighted iterations, chain A", float(per_snap["iterations_A"].median()))
    rep.score("3.2b median mass-weighted iterations, chain B", float(per_snap["iterations_B"].median()))
    rep.score("3.2b components not converged within max_iter, chain A", int(per_snap["unconverged_A"].sum()))
    rep.score("3.2b components not converged within max_iter, chain B", int(per_snap["unconverged_B"].sum()))
    rep.score("3.2b median replayed mass share in non-converged components",
              float(per_snap["not_converged_share"].median()))
    rep.score("3.2b median label-change share of contested mass",
              float(per_snap["label_share_contested"].median()))
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

def plot_feedback(run, per_snap, gal, gamma, pdf):
    """3.2b against time: n-weighted median sd(L) (nats), rho_L and gamma (where defined), the
    label-change share, the mass-weighted median per-galaxy Delta log|Sigma|, and the reproduction
    residual (median |delta log r| and p99 |delta r|, contested entries)."""
    fig, (a0, a1, a2, a3, a4) = plt.subplots(5, 1, figsize=(10, 15), sharex=True)
    if not gamma.empty:
        sd = wmedian_in_time(gamma, "sd_L", "n_play")
        a0.plot(sd.index, sd, "o-", ms=3)
        rho_L = wmedian_in_time(gamma, "rho_L", "n_play")
        a1.plot(rho_L.index, rho_L, "s-", ms=3, label="rho_L = corr(L, Delta log alpha)")
        ok = gamma[gamma["gamma"].notna()]
        if not ok.empty:
            g = wmedian_in_time(ok, "gamma", "n_play")
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
            d = wmedian_in_time(contested_gal, "dlogdet", "mass")
            a3.plot(d.index, d, "o-", ms=3)
    a3.axhline(0, color="0.6", lw=0.6)
    a3.set_ylabel("3.2b mass-weighted median Delta log|Sigma|")
    if not per_snap.empty:
        a4.plot(per_snap["t"], per_snap["repro_median"], "o-", ms=3, label="median |delta log r|")
        a4.plot(per_snap["t"], per_snap["repro_p99_r"], "s-", ms=3, label="components' p99 |delta r|")
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
            for i, col in enumerate(("r2_G", "r2_phys")):
                v = wmedian_in_time(sel, col, "n")
                a1.plot(v.index, v, style, ms=3, color=f"C{i}", label=f"{col}, {tag}")
        if depB is not None and not depB.empty:
            v = wmedian_in_time(depB, "r2_G", "n")
            a1.plot(v.index, v, "x:", ms=4, color="C2", label="r2_G(B)")
    a1.set_ylabel("3.4 n-weighted median R^2")
    a1.legend(fontsize=7, ncol=2)
    if not DEP.empty:
        u = wmedian_in_time(DEP, "u", "n")
        a2.plot(u.index, u, "o-", ms=3, label="u")
    a2.set_xlabel("t [Gyr]")
    a2.set_ylabel("3.4 n-weighted median u")
    a2.legend(fontsize=8)
    fig.suptitle(f"{run.name}: goal 3, dependence (3.4)")
    plots.save(fig, pdf)


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
