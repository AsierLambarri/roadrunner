"""Replay of the Bayesian GMM's E/M-step on a run's own stored pairs and written fit.

Pure numpy/scipy, no I/O: Goal 3's "3.2b" test (did alpha feed back into the fit, or only gate it?)
replays the BGMM's EM in two chains from the written fit theta_A -- A with alpha in every E-step (the
run as written), B with alpha's support only (its entries), alpha applied once at the end -- and
compares. Callers (``evaluate/model.py``) read the run's stored arrays (``evaluate/load.py``'s
``Run.pairs``, ``Run.gaussians``, the merger tree) and pass plain arrays here.

Mirrors, exactly:

- E-step: ``roadrunner/mixture/base.py:347`` (``_e_step``) and
  ``roadrunner/mixture/bayesian_gmm.py:794-833`` (``_estimate_log_gaussian_prob``,
  ``_estimate_log_weights``): ``log rho_nk = E[log pi_k] + log N(x | m_k, Sigma_k) - D/2 log nu_k
  + 1/2 (D log 2 + sum_{i=0}^{D-1} digamma((nu_k - i)/2)) - D/(2 beta_k) + log alpha_nk``, then
  ``log r = log rho - logsumexp`` over the particle's own candidates. ``Sigma_k`` is the stored/fitted
  ``covariances_`` (``= W^-1 / nu``), so ``log N`` uses it directly as the covariance, exactly as
  ``_estimate_log_gaussian_prob_pchol`` does. ``E[log pi_k] = digamma(a_k) - digamma(sum_j a_j)``; the
  second term is the same number for every candidate of a given particle (its candidates are all in one
  fit group, i.e. share one ``a`` vector), so it cancels in the softmax and is dropped here.
- M-step: ``roadrunner/mixture/bayesian_gmm.py:639-717`` (``_estimate_weights``, ``_estimate_means``,
  ``_estimate_wishart_full``), fed by ``roadrunner/mixture/weighted_gmm.py``'s
  ``_estimate_gaussian_parameters`` (full covariance): ``a = a0 + N``, ``beta = beta0 + N``,
  ``m = (beta0 m0 + N xbar) / beta``, ``nu = nu0 + N``,
  ``W^-1 = W0inv + N S + N beta0 / beta (xbar - m0)(xbar - m0)^T``, ``Sigma = W^-1 / nu``. Point weights
  are 1 (``mass_weighting`` is false in these runs), so ``N``, ``xbar``, ``S`` are the plain
  responsibility-weighted count, mean and scatter. The pipeline adds ``reg_covar = 1e-6`` to ``S``'s
  diagonal in the group's standardised coordinates; here, in natural units, that is approximated as
  ``reg_covar`` times the component's own (unregularised) per-dimension variance -- the standardised
  variance the pipeline regularises is ~1, so this is the natural-unit image of the same addition, and
  it is negligible either way.
- Prior: ``roadrunner/clustering/assignment/gmm.py:693-790`` (``_build_prior_kwargs``) and ``:790-810``
  (``_history``), via the pure helpers in ``roadrunner/clustering/assignment/priors.py``. Per galaxy k,
  with ``n_b`` its number of stored pairs at s: history is usable when k has a written fit at s-1 with
  ``count >= D + 1``; then ``nk_n1 = max(count_prev, 1)``, ``ref_vars = degrade_covariance(cov_prev, D)``
  (natural units), ``mass_ratio = tree_mass(s) / tree_mass(s-1)``. Without usable history,
  ``nk_n1 = max(count_s, 1)``, ``ref_vars = diag(cov_s)``, ``mass_ratio = 1`` (tagged ``no_history``:
  the pipeline used pre-fit estimates that were never written). Then
  ``a0 = weight_concentration_prior(nk_n1, n_b, n_comp)``, ``beta0 = mean_precision_prior(nk_n1, n_b)``,
  ``nu0 = D + PRIOR_DOF_OFFSET``,
  ``W0inv = covariance_prior(ref_vars, mass_ratio, nk_n1, ones(D), nu0, "full")`` (natural units: the
  pipeline's scaler ``scale_`` is passed as ``s = ones(D)``, since its square cancels going back to
  natural units), ``m0 = tree(x, v)`` at s. The *tree mass* the integrator must pass is the same
  quantity ``HaloModel.tree_mass`` writes with each fit (``roadrunner/physics/halo_model.py:48-109``:
  "the catalogue's centre, bulk velocity and mass"; ``self.tree_mass = sum`` of the dark-matter group's
  particle masses) -- i.e. the merger tree's own ``mass`` column, ``run.tree`` indexed by
  (``Snapshot``, ``Sub_tree_id``), at s and at s-1 for each galaxy.
- Starting state theta_A: ``m`` = written mean, ``Sigma`` = written covariance, ``N`` = written count,
  then ``a = a0 + N``, ``beta = beta0 + N``, ``nu = nu0 + N`` (the written fit read as a BGMM posterior).

Public functions: ``state`` (priors and theta_A from the written/previous fits and the tree),
``estep``, ``mstep`` (the two half-steps, usable standalone -- the unit test checks them against
``WeightedBayesianGaussianMixture`` directly), ``replay`` (the full two-chain EM from theta_A) and
``reproduce`` (one check E-step against the run's own written log r).
"""
from collections import namedtuple

import numpy as np
from scipy.special import digamma

from roadrunner._defaults import PRIOR_DOF_OFFSET
from roadrunner.clustering.assignment.priors import (
    covariance_prior,
    degrade_covariance,
    mean_precision_prior,
    weight_concentration_prior,
)

Prior = namedtuple("Prior", "a0 beta0 m0 nu0 W0inv")
Posterior = namedtuple("Posterior", "m Sigma N a beta nu")


# --- row grouping (particle -> its run of candidate entries) --------------------------------------

def _row_groups(rows):
    """The order sorting ``rows``, and the start and size of each particle's run of entries in it."""
    o = np.argsort(rows, kind="stable")
    if not rows.size:
        return o, np.array([], np.int64), np.array([], np.int64)
    sr = rows[o]
    start = np.flatnonzero(np.r_[True, sr[1:] != sr[:-1]])
    return o, start, np.diff(np.r_[start, rows.size])


def _group_by_galaxy(gal, galaxies):
    """``idx`` (``gal`` as an index into ``galaxies``), the order sorting entries by it, and the
    ``n_galaxies + 1`` boundaries of each galaxy's run in that order (``start, start[1:]``, so an
    empty galaxy has ``start[k] == start[k + 1]``). Sorting once and slicing is what lets ``estep`` and
    ``mstep`` avoid a fresh ``idx == k`` boolean mask, or a fresh sparse matrix, every EM iteration."""
    idx = np.searchsorted(galaxies, gal)
    o = np.argsort(idx)
    bounds = np.searchsorted(idx[o], np.arange(galaxies.size + 1))
    return idx, o, bounds


def _row_grouping(rows):
    """``(o, start, seg)`` for ``_logsumexp_by_row``: the sort order, each run's start, and the run
    index of every sorted entry, computed once for reuse across repeated calls on the same ``rows``."""
    o, start, count = _row_groups(rows)
    return o, start, np.repeat(np.arange(start.size), count)


def _logsumexp_by_row(rows, x, row_grouping=None):
    """``logsumexp`` of ``x`` within each particle's own run of entries, broadcast back to entry order.

    ``row_grouping``: a precomputed ``(o, start, seg)`` (``_row_groups(rows)`` plus the run index of
    every sorted entry), reused across ``estep`` calls in ``replay``'s loop (``rows`` never changes
    there) instead of being resorted every call."""
    if not rows.size:
        return np.array([])
    if row_grouping is None:
        o, start, count = _row_groups(rows)
        seg = np.repeat(np.arange(start.size), count)
    else:
        o, start, seg = row_grouping
    xs = x[o]
    mx = np.maximum.reduceat(xs, start)
    lse = mx + np.log(np.add.reduceat(np.exp(xs - mx[seg]), start))
    out = np.empty_like(x)
    out[o] = lse[seg]
    return out


# --- prior and starting state ----------------------------------------------------------------------

def state(gal, galaxies, gauss, gauss_prev, tree_xyzv, tree_mass, tree_mass_prev):
    """Per-galaxy prior and starting posterior (theta_A), for every id in ``galaxies``.

    Parameters
    ----------
    gal : ndarray of int64
        Galaxy id of every stored pair at s (all of them, not only the contested ones: this is what
        ``n_b``, the prior's reference count, is counted over).
    galaxies : ndarray of int64, sorted, unique
        The galaxies to build state for; every one must have an entry in ``gauss`` (callers restrict
        to galaxies with a stored, positive-definite Gaussian, as the rest of Goal 3 does).
    gauss, gauss_prev : dict of {galaxy: (mean, covariance, weight, count)}
        The written fit at s and at s - 1 (``Run.gaussians``; natural units, ``count`` may be NaN).
    tree_xyzv : dict of {galaxy: ndarray of shape (D,)}
        Tree position and velocity at s (the prior mean m0).
    tree_mass, tree_mass_prev : dict of {galaxy: float}
        The merger tree's ``mass`` at s and at s - 1 (``HaloModel.tree_mass``; see the module docstring).

    Returns
    -------
    prior : Prior
        ``a0, beta0, m0, nu0, W0inv``, each stacked over ``galaxies`` (``nu0`` is the same scalar
        everywhere, kept as an array for uniform broadcasting).
    theta_A : Posterior
        The written fit read as a posterior: ``m, Sigma, N`` from ``gauss``, ``a = a0 + N`` etc.
    no_history : ndarray of bool
        Which galaxies had no usable previous fit (pre-fit estimates were used as the reference instead).
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


# --- the two half-steps ------------------------------------------------------------------------------

def estep(X, rows, gal, galaxies, log_alpha, post, with_alpha=True, grouping=None, row_grouping=None,
          Xr=None):
    """E-step over the given entries, one Cholesky/inverse per galaxy (vectorised over its entries).

    Parameters
    ----------
    X : ndarray of shape (n_particles, D)
    rows : ndarray of int64, shape (n_entries,)
        Particle row of each entry.
    gal : ndarray of int64, shape (n_entries,)
        Galaxy of each entry.
    galaxies : ndarray of int64, sorted, unique
        Aligned with ``post``.
    log_alpha : ndarray of shape (n_entries,)
        Ignored when ``with_alpha`` is False (chain B: alpha's support only).
    post : Posterior
        Current ``(m, Sigma, a, beta, nu)`` (``N`` is unused here).
    with_alpha : bool, default True
    grouping : (idx, order, bounds) or None
        ``_group_by_galaxy(gal, galaxies)``, precomputed once outside a loop that calls ``estep``
        repeatedly over the same ``rows``/``gal`` (``replay`` does this): recomputed here if None.
    row_grouping : (o, start, seg) or None
        ``_row_grouping(rows)``, likewise precomputed once by ``replay``.
    Xr : ndarray of shape (n_entries, D) or None
        ``X[rows[o]]`` (``o`` from ``grouping``), i.e. the entries' coordinates already gathered and
        sorted into galaxy order: fancy-indexing ``X`` by ``rows`` is itself a sizeable cost (measured
        tens of ms at 1e5-1e6 entries) that ``replay`` pays once rather than every EM iteration.
        Recomputed here if None.

    Returns
    -------
    g : ndarray of shape (n_entries,)
        ``log N_k(x) + log pi_k`` term of log rho_nk (no alpha): the Gaussian log term used to form
        D_G elsewhere (e.g. ``model.py``'s 3.2).
    log_r : ndarray of shape (n_entries,)
        Normalised log responsibility, per particle over its own entries only.
    """
    D = X.shape[1]
    idx, o, bounds = grouping if grouping is not None else _group_by_galaxy(gal, galaxies)
    if Xr is None:
        Xr = X[rows[o]]

    # Batched over every galaxy at once: numpy's batched linalg (one LAPACK call doing all K
    # components) measured ~100x faster than looping K separate (D, D) calls, and
    # scipy.linalg.solve_triangular carries a large fixed per-call overhead for a matrix this small.
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
        quad = np.einsum("ni,ni->n", diff @ precision[k], diff)    # a 3-operand einsum measured ~6x slower
        log_gauss = -0.5 * (D * np.log(2.0 * np.pi) + log_det[k] + quad)
        g[o[s0:s1]] = log_gauss + extra[k]
    log_rho = g + log_alpha if with_alpha else g
    log_r = log_rho - _logsumexp_by_row(rows, log_rho, row_grouping=row_grouping)
    return g, log_r


def _sufficient_stats(X, rows, gal, galaxies, centre, r, grouping=None, Xr=None):
    """Responsibility-weighted ``(N, sum x, sum x x^T)`` per galaxy, raw moments centred on
    ``centre[k]`` (the tree position) to avoid cancellation at ~1e4 kpc scale.

    One ``xc.T @ (r xc)`` (BLAS ``gemm``) per galaxy, over its own slice of a single sort
    (``_group_by_galaxy``, or the ``grouping`` passed in), rather than a Python loop of ``bincount``
    calls per dimension (``D(D+1)/2 + D + 1`` of them), a sparse matrix rebuilt from scratch, or a
    dense (n, D, D) outer product reduced with ``np.add.reduceat`` (all three measured slower here):
    with 1e5-1e6 entries and tens of galaxies this keeps an EM iteration (``replay`` calls this, via
    ``mstep``, every iteration) at milliseconds rather than tenths of a second. ``Xr``: see ``estep``."""
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
    """M-step: new posterior from responsibilities ``r`` over the given entries.

    Efficiency: particles with a single candidate always have r = 1 and never change across EM
    iterations, so their sufficient statistics can be computed once and passed as ``fixed`` -- only
    the contested entries (``rows``, ``gal``, ``r`` here) need redoing every iteration (see ``replay``).

    Parameters
    ----------
    X : ndarray of shape (n_particles, D)
    rows, gal : ndarray of int64, shape (n_entries,)
    galaxies : ndarray of int64, sorted, unique
    r : ndarray of shape (n_entries,)
        Responsibilities (already exponentiated) of these entries.
    prior : Prior
        Aligned with ``galaxies``; ``prior.m0`` is also used as the centring constant (see
        ``_sufficient_stats``).
    reg_covar : float, default 1e-6
        See the module docstring's note on natural units.
    fixed : (N0, Sx0, Sxx0) or None
        Precomputed sufficient statistics (same centring) of entries not given here, added in.
    grouping, Xr : see ``estep``.

    Returns
    -------
    post : Posterior
        Aligned with ``galaxies``.
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


# --- the two-chain replay and the written-fit check --------------------------------------------------

def replay(X, rows, gal, galaxies, log_alpha, prior, theta_A, with_alpha=True,
           reg_covar=1e-6, tol=1e-5, max_iter=100):
    """Replay the BGMM EM from theta_A, on the group's own stored pairs, in one chain.

    Particles with a single candidate (r = 1 always) are folded into a fixed sufficient-statistics
    term computed once; only particles with >= 2 candidates ("contested") are iterated.

    Parameters
    ----------
    X : ndarray of shape (n_particles, D)
    rows, gal, log_alpha : ndarray, shape (n_entries,)
        Every stored pair of the group at this snapshot (contested and not).
    galaxies : ndarray of int64, sorted, unique
    prior : Prior
    theta_A : Posterior
        The written fit read as a posterior (``state``'s second return).
    with_alpha : bool, default True
        True: chain A, alpha in every E-step (the run as written). False: chain B, alpha's support
        only during the iteration (every given entry already is in the support, so this is simply
        ``log_alpha := 0``); one more E-step WITH alpha is then taken at the end, on the converged fit,
        to score the final responsibilities.
    reg_covar, tol, max_iter : see ``mstep``; convergence is ``max |Delta log r| < tol`` over the
        contested entries between consecutive E-steps.

    Returns
    -------
    post : Posterior
        The final (converged, or ``max_iter``-truncated) fit.
    first_post : Posterior
        The posterior after one E+M step from theta_A (the linear response).
    iterations : int
        Number of E+M steps taken.
    converged : bool
        Whether the loop met ``tol`` (False: stopped by ``max_iter``).
    rows_c, gal_c : ndarray of int64
        The contested entries' row and galaxy (what ``log_r`` and ``g`` below are indexed by).
    log_r, g : ndarray of shape (rows_c.size,)
        The final, WITH-alpha responsibilities and the alpha-free Gaussian log term (``D_G = g_k - g_j``
        per pair), from the one extra E-step taken after the loop.
    """
    counts = np.bincount(rows, minlength=(rows.max() + 1 if rows.size else 0))
    contested = counts[rows] >= 2 if rows.size else np.zeros(0, dtype=bool)
    f_rows, f_gal = rows[~contested], gal[~contested]
    c_rows, c_gal, c_log_alpha = rows[contested], gal[contested], log_alpha[contested]

    fixed = _sufficient_stats(X, f_rows, f_gal, galaxies, prior.m0, np.ones(f_rows.size))
    # (rows, gal) of the contested entries do not change across iterations: sort once, not every
    # E-step and M-step call (the EM loop below is the hot path this module exists to keep cheap), and
    # gather X at those entries (in that same sorted order) once rather than every call too -- fancy
    # indexing X by rows is itself the dominant cost otherwise (measured tens of ms at 1e5+ entries).
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
    """One E-step WITH alpha at theta_A, checked against the run's own written log r.

    Parameters
    ----------
    X, rows, gal, galaxies, log_alpha : as in ``estep``.
    theta_A : Posterior
        ``state``'s starting state.
    written_log_r : ndarray of shape (n_entries,)
        The run's stored ``log_resp`` for these entries (``-inf`` where it underflowed).

    Returns
    -------
    delta, delta_r : ndarray of shape (n_entries,)
        ``|replayed log r - written log r|`` and ``|replayed r - written r|``, NaN where
        ``written_log_r`` is not finite (the caller aggregates over the finite entries, per component).
        Log differences between underflowed responsibilities (r ~ 1e-12) are large and meaningless, so
        a tail is read in r.
    """
    _, log_r = estep(X, rows, gal, galaxies, log_alpha, theta_A, with_alpha=True)
    delta, delta_r = np.full(written_log_r.shape, np.nan), np.full(written_log_r.shape, np.nan)
    finite = np.isfinite(written_log_r)
    delta[finite] = np.abs(log_r[finite] - written_log_r[finite])
    delta_r[finite] = np.abs(np.exp(log_r[finite]) - np.exp(written_log_r[finite]))
    return delta, delta_r
