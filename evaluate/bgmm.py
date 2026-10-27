"""The Bayesian GMM's E- and M-step, replayed on a run's own stored pairs from its written fit, for
goal 3's 3.2b (model_alpha.py): did alpha feed back into the fit, or only gate it? Two chains start
from the written fit theta_A: A with alpha in every E-step (the run as written), B with alpha's support
only (its entries) and alpha applied once at the end. Pure numpy and scipy, no I/O: the caller reads the
stored arrays (``Run.pairs``, ``Run.gaussians``, the merger tree) and passes plain arrays.

It mirrors the pipeline's mixture (roadrunner/mixture: ``_e_step``, ``_estimate_log_gaussian_prob``,
``_estimate_log_weights``, ``_estimate_weights``, ``_estimate_means``, ``_estimate_wishart_full``) and
its priors (roadrunner/clustering/assignment: ``_build_prior_kwargs`` and ``_history`` in gmm.py, the
functions of priors.py):

E-step: log rho_nk = E[log pi_k] + log N(x | m_k, Sigma_k) - D/2 log nu_k
    + 1/2 (D log 2 + sum_{i<D} digamma((nu_k - i)/2)) - D/(2 beta_k) + log alpha_nk, and
    log r = log rho - logsumexp over the particle's own candidates. Sigma_k is the fitted covariance
    (W^-1 / nu), the Gaussian's covariance as the pipeline uses it. E[log pi_k] = digamma(a_k) -
    digamma(sum_j a_j); its second term is the same for every candidate of a particle (they share one
    fit group), so it cancels and is dropped.
M-step: a = a0 + N, beta = beta0 + N, m = (beta0 m0 + N xbar) / beta, nu = nu0 + N,
    W^-1 = W0inv + N S + N beta0 / beta (xbar - m0)(xbar - m0)^T and Sigma = W^-1 / nu, with N, xbar
    and S the responsibility-weighted count, mean and scatter (point weights are 1: mass_weighting is
    off in these runs). The pipeline adds reg_covar = 1e-6 to the diagonal of S in the group's
    standardised coordinates, where the variances are ~1; in natural units that is reg_covar times the
    component's own variance, which is what is added here (negligible either way).
Prior, per galaxy k with n_b stored pairs at s: its history is usable when it has a written fit at s-1
    with count >= D + 1; then nk_n1 = max(count_prev, 1), ref_vars = degrade_covariance(cov_prev, D) and
    mass_ratio = tree mass(s) / tree mass(s-1). Without it (tagged no_history: the pipeline used pre-fit
    estimates that were never written), nk_n1 = max(count_s, 1), ref_vars = diag(cov_s) and
    mass_ratio = 1. Then a0 = weight_concentration_prior(nk_n1, n_b, n_comp),
    beta0 = mean_precision_prior(nk_n1, n_b), nu0 = D + PRIOR_DOF_OFFSET,
    W0inv = covariance_prior(ref_vars, mass_ratio, nk_n1, ones(D), nu0, "full") (the pipeline's scaler
    cancels in natural units) and m0 = the tree's (x, v) at s. The tree mass is the merger tree's own
    ``mass`` column, the quantity the pipeline's HaloModel.tree_mass holds for each fit.
Starting state theta_A: the written mean, covariance and count N, with a = a0 + N, beta = beta0 + N
    and nu = nu0 + N.

``estep`` and ``mstep`` stand on their own (the unit test checks them against
WeightedBayesianGaussianMixture); ``replay`` runs one chain to convergence and ``reproduce`` checks one
E-step against the run's written log r.
"""
from collections import namedtuple

import numpy as np
from scipy.special import digamma

from evaluate.pairs import by_row
from roadrunner._defaults import PRIOR_DOF_OFFSET
from roadrunner.clustering.assignment.priors import (
    covariance_prior,
    degrade_covariance,
    mean_precision_prior,
    weight_concentration_prior,
)

Prior = namedtuple("Prior", "a0 beta0 m0 nu0 W0inv")
Posterior = namedtuple("Posterior", "m Sigma N a beta nu")


# --- the entries, grouped -------------------------------------------------------------------------

def _group_by_galaxy(gal, galaxies):
    """``idx`` (``gal`` as an index into ``galaxies``), the order sorting the entries by it, and the
    ``n_galaxies + 1`` bounds of each galaxy's run in that order (an empty galaxy has
    ``bounds[k] == bounds[k + 1]``). Sorted once, each galaxy's entries are a slice in ``estep`` and
    ``mstep`` instead of a mask built at every EM iteration."""
    idx = np.searchsorted(galaxies, gal)
    o = np.argsort(idx)
    bounds = np.searchsorted(idx[o], np.arange(galaxies.size + 1))
    return idx, o, bounds


def _row_grouping(rows):
    """``(o, start, seg)`` of ``rows`` for ``_logsumexp_by_row``: the order sorting them, each particle's
    first entry in it and every sorted entry's particle, computed once for repeated calls."""
    o, start, count = by_row(rows)
    return o, start, np.repeat(np.arange(start.size), count)


def _logsumexp_by_row(rows, x, row_grouping=None):
    """``logsumexp`` of ``x`` over each particle's own entries, broadcast back to the entries;
    ``row_grouping`` is ``_row_grouping(rows)`` when it was computed already."""
    if not rows.size:
        return np.array([])
    if row_grouping is None:
        o, start, count = by_row(rows)
        seg = np.repeat(np.arange(start.size), count)
    else:
        o, start, seg = row_grouping
    xs = x[o]
    mx = np.maximum.reduceat(xs, start)
    lse = mx + np.log(np.add.reduceat(np.exp(xs - mx[seg]), start))
    out = np.empty_like(x)
    out[o] = lse[seg]
    return out


# --- prior and starting state ---------------------------------------------------------------------

def state(gal, galaxies, gauss, gauss_prev, tree_xyzv, tree_mass, tree_mass_prev):
    """The prior and the starting posterior theta_A of every galaxy in ``galaxies`` (sorted ids, each with
    a Gaussian in ``gauss``), and which of them have no usable history.

    ``gal`` is the galaxy of every stored pair at s, contested or not (the prior's reference count n_b
    counts them all); ``gauss`` and ``gauss_prev`` are the written fits at s and s - 1 (``Run.gaussians``:
    mean, covariance, weight and count in natural units; the count may be NaN); ``tree_xyzv`` is each
    galaxy's tree position and velocity at s (the prior mean m0); ``tree_mass`` and ``tree_mass_prev``
    are its tree mass at s and s - 1. Returns a ``Prior`` and a ``Posterior`` stacked over ``galaxies``
    (nu0 is one number, repeated), and the boolean array ``no_history``.
    """
    n_gal = galaxies.size
    D = next(iter(gauss.values()))[0].size
    in_gal = np.isin(gal, galaxies)                   # a galaxy outside ``galaxies`` is not one of ours
    n_b = np.bincount(np.searchsorted(galaxies, gal[in_gal]), minlength=n_gal).astype(np.float64)

    m = np.empty((n_gal, D))
    Sigma = np.empty((n_gal, D, D))
    N = np.empty(n_gal)
    m0 = np.empty((n_gal, D))
    nk_n1 = np.empty(n_gal)
    ref_vars = np.empty((n_gal, D))
    mass_ratio = np.empty(n_gal)
    no_history = np.zeros(n_gal, dtype=bool)

    for i, sid in enumerate(galaxies.tolist()):
        mean_s, cov_s, _, count_s = gauss[sid]
        m[i], Sigma[i], N[i] = mean_s, cov_s, count_s
        m0[i] = tree_xyzv[sid]

        prev = gauss_prev.get(sid)
        usable = prev is not None and np.isfinite(prev[3]) and prev[3] >= D + 1
        if usable:
            _, cov_prev, _, count_prev = prev
            nk_n1[i] = max(count_prev, 1.0)
            ref_vars[i] = degrade_covariance(cov_prev, D)
            mp, mn = tree_mass_prev.get(sid, 0.0), tree_mass.get(sid, 0.0)
            mass_ratio[i] = mn / mp if (mp > 0 and mn > 0) else 1.0
        else:
            no_history[i] = True
            nk_n1[i] = max(count_s, 1.0)
            ref_vars[i] = np.diag(cov_s)
            mass_ratio[i] = 1.0

    nu0_scalar = D + PRIOR_DOF_OFFSET
    a0 = np.array([weight_concentration_prior(nk_n1[i], n_b[i], n_gal) for i in range(n_gal)])
    beta0 = np.array([mean_precision_prior(nk_n1[i], n_b[i]) for i in range(n_gal)])
    ones_d = np.ones(D)
    W0inv = np.stack([covariance_prior(ref_vars[i], mass_ratio[i], nk_n1[i], ones_d, nu0_scalar, "full")
                       for i in range(n_gal)])
    nu0 = np.full(n_gal, float(nu0_scalar))

    prior = Prior(a0=a0, beta0=beta0, m0=m0, nu0=nu0, W0inv=W0inv)
    theta_A = Posterior(m=m, Sigma=Sigma, N=N, a=a0 + N, beta=beta0 + N, nu=nu0 + N)
    return prior, theta_A, no_history


# --- the two half-steps ---------------------------------------------------------------------------

def estep(X, rows, gal, galaxies, log_alpha, post, with_alpha=True, grouping=None, row_grouping=None,
          Xr=None):
    """E-step over the given entries: ``rows`` and ``gal`` are each entry's particle row in ``X`` and its
    galaxy, ``post`` is aligned with the sorted ids ``galaxies``. Returns ``g``, the alpha-free part of
    log rho (so that D_G = g_k - g_j), and the log responsibilities, normalised over each particle's own
    entries, with ``log_alpha`` in them only ``with_alpha`` (chain B iterates without it).
    ``grouping`` (``_group_by_galaxy``), ``row_grouping`` (``_row_grouping``) and ``Xr`` (``X`` at the
    entries, in the order of ``grouping``) can be passed in when computed once for many calls, as
    ``replay`` does.
    """
    D = X.shape[1]
    idx, o, bounds = grouping if grouping is not None else _group_by_galaxy(gal, galaxies)
    if Xr is None:
        Xr = X[rows[o]]

    # every galaxy at once: one batched call each for the factor, the log-determinant and the precision
    L = np.linalg.cholesky(post.Sigma)                                    # (n_gal, D, D)
    log_det = 2.0 * np.log(np.diagonal(L, axis1=1, axis2=2)).sum(axis=1)  # (n_gal,)
    precision = np.linalg.inv(post.Sigma)                                 # (n_gal, D, D)
    ar = np.arange(D)
    extra = (-0.5 * D * np.log(post.nu)
             + 0.5 * (D * np.log(2.0) + digamma(0.5 * (post.nu[:, None] - ar)).sum(axis=1))
             - 0.5 * D / post.beta
             + digamma(post.a))                                          # (n_gal,)

    g = np.empty(rows.size)
    for k in range(galaxies.size):
        s0, s1 = bounds[k], bounds[k + 1]
        if s0 == s1:
            continue
        diff = Xr[s0:s1] - post.m[k]
        quad = np.einsum("ni,ni->n", diff @ precision[k], diff)
        log_gauss = -0.5 * (D * np.log(2.0 * np.pi) + log_det[k] + quad)
        g[o[s0:s1]] = log_gauss + extra[k]
    log_rho = g + log_alpha if with_alpha else g
    log_r = log_rho - _logsumexp_by_row(rows, log_rho, row_grouping=row_grouping)
    return g, log_r


def _sufficient_stats(X, rows, gal, galaxies, centre, r, grouping=None, Xr=None):
    """Responsibility-weighted ``(N, sum x, sum x x^T)`` per galaxy, with the moments centred on
    ``centre[k]`` (the tree position) to avoid cancellation at ~1e4 kpc: one matrix product per galaxy,
    over its slice of the entries sorted by galaxy (``grouping`` and ``Xr``: see ``estep``)."""
    n_gal, D = galaxies.size, X.shape[1]
    N, Sx, Sxx = np.zeros(n_gal), np.zeros((n_gal, D)), np.zeros((n_gal, D, D))
    if not rows.size:
        return N, Sx, Sxx

    _, o, bounds = grouping if grouping is not None else _group_by_galaxy(gal, galaxies)
    if Xr is None:
        Xr = X[rows[o]]
    for k in range(n_gal):
        s0, s1 = bounds[k], bounds[k + 1]
        if s0 == s1:
            continue
        xc = Xr[s0:s1] - centre[k]
        w = r[o[s0:s1]]
        wxc = w[:, None] * xc
        N[k] = w.sum()
        Sx[k] = wxc.sum(axis=0)
        Sxx[k] = xc.T @ wxc
    return N, Sx, Sxx


def mstep(X, rows, gal, galaxies, r, prior, reg_covar=1e-6, fixed=None, grouping=None, Xr=None):
    """M-step: the posterior from the responsibilities ``r`` (not logs) of the given entries, with
    ``prior`` aligned with ``galaxies`` (its m0 is also the centring constant). ``fixed`` adds the
    sufficient statistics of entries not given, computed once: a particle with a single candidate always
    has r = 1, so ``replay`` passes those that way and iterates over the contested entries only.
    ``reg_covar``: see the module docstring; ``grouping`` and ``Xr``: see ``estep``.
    """
    D = X.shape[1]
    N, Sx, Sxx = _sufficient_stats(X, rows, gal, galaxies, prior.m0, r, grouping=grouping, Xr=Xr)
    if fixed is not None:
        N0, Sx0, Sxx0 = fixed
        N, Sx, Sxx = N + N0, Sx + Sx0, Sxx + Sxx0

    nk = N + 10.0 * np.finfo(np.float64).eps
    mean_c = Sx / nk[:, None]
    xk = prior.m0 + mean_c
    sk = Sxx / nk[:, None, None] - mean_c[:, :, None] * mean_c[:, None, :]
    diag_s = np.diagonal(sk, axis1=1, axis2=2)
    sk = sk + reg_covar * np.eye(D)[None, :, :] * diag_s[:, :, None]

    beta = prior.beta0 + nk
    m = (prior.beta0[:, None] * prior.m0 + nk[:, None] * xk) / beta[:, None]
    nu = prior.nu0 + nk
    diff = xk - prior.m0
    Sigma = (prior.W0inv + nk[:, None, None] * sk
             + (nk * prior.beta0 / beta)[:, None, None] * diff[:, :, None] * diff[:, None, :])
    Sigma = Sigma / nu[:, None, None]
    a = prior.a0 + nk
    return Posterior(m=m, Sigma=Sigma, N=nk, a=a, beta=beta, nu=nu)


# --- the two-chain replay and the written-fit check -----------------------------------------------

def replay(X, rows, gal, galaxies, log_alpha, prior, theta_A, with_alpha=True,
           reg_covar=1e-6, tol=1e-5, max_iter=100):
    """One chain of EM from the written fit ``theta_A`` on a component's stored pairs, until no contested
    log r moves by ``tol`` between consecutive E-steps, or for ``max_iter`` steps. ``with_alpha``: chain
    A, alpha in every E-step; False: chain B, log alpha = 0 while iterating (every entry is in alpha's
    support). Particles with one candidate are folded into fixed sufficient statistics; only the
    contested entries are iterated.

    Returns the final posterior, the posterior after one step (the linear response), the steps taken,
    whether it converged, the contested entries' rows and galaxies, and, from one more E-step with alpha
    at the final fit, their log responsibilities and alpha-free Gaussian terms ``g``.
    """
    counts = np.bincount(rows, minlength=(rows.max() + 1 if rows.size else 0))
    contested = counts[rows] >= 2 if rows.size else np.zeros(0, dtype=bool)
    f_rows, f_gal = rows[~contested], gal[~contested]
    c_rows, c_gal, c_log_alpha = rows[contested], gal[contested], log_alpha[contested]

    fixed = _sufficient_stats(X, f_rows, f_gal, galaxies, prior.m0, np.ones(f_rows.size))
    # the contested entries never change: sort them, and gather their coordinates, once
    grouping = _group_by_galaxy(c_gal, galaxies)
    row_grouping = _row_grouping(c_rows)
    Xr = X[c_rows[grouping[1]]]

    post = theta_A
    first_post = None
    prev_log_r = None
    it = 0
    for it in range(1, max_iter + 1):
        g, log_r = estep(X, c_rows, c_gal, galaxies, c_log_alpha, post, with_alpha=with_alpha,
                          grouping=grouping, row_grouping=row_grouping, Xr=Xr)
        post = mstep(X, c_rows, c_gal, galaxies, np.exp(log_r), prior, reg_covar=reg_covar,
                     fixed=fixed, grouping=grouping, Xr=Xr)
        if first_post is None:
            first_post = post
        converged = log_r.size == 0 or (prev_log_r is not None
                                         and np.max(np.abs(log_r - prev_log_r)) < tol)
        prev_log_r = log_r
        if converged:
            break

    g_final, log_r_final = estep(X, c_rows, c_gal, galaxies, c_log_alpha, post, with_alpha=True,
                                  grouping=grouping, row_grouping=row_grouping, Xr=Xr)
    return post, first_post, it, converged, c_rows, c_gal, log_r_final, g_final


def reproduce(X, rows, gal, galaxies, log_alpha, theta_A, written_log_r):
    """One E-step with alpha at ``theta_A``, against the run's written log r of the same entries:
    ``|delta log r|`` and ``|delta r|`` per entry, NaN where the written value is not finite. A log
    difference between underflowed responsibilities (r ~ 1e-12) means nothing, so a tail is read in r."""
    _, log_r = estep(X, rows, gal, galaxies, log_alpha, theta_A, with_alpha=True)
    delta, delta_r = np.full(written_log_r.shape, np.nan), np.full(written_log_r.shape, np.nan)
    finite = np.isfinite(written_log_r)
    delta[finite] = np.abs(log_r[finite] - written_log_r[finite])
    delta_r[finite] = np.abs(np.exp(log_r[finite]) - np.exp(written_log_r[finite]))
    return delta, delta_r
