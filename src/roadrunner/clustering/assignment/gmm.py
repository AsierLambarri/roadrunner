#############################################################################
#
# package:   roadrunner.clustering.assignment
# file:      gmm.py
# brief:     Parameterized Gaussian mixture assigner (XGMM) with method dispatch.
#
# Provides the XGMMAssigner, which dispatches to a concrete mixture
# class based on a ``method`` string ("gmm", "bgmm", "svi-bgmm").
# The assigner handles per-snapshot fitting, temporal smoothing via
# previous responsibilities (particles bound to a halo they had no
# previous responsibility for are padded by a predictive E-step under
# the previous snapshot's model),
# and building BGMM prior kwargs from the previous snapshot's fitted
# parameters.
#
# copyright: GPLv3
# author:    Asier Lambarri Martinez
# changes:   14 May 2026 - Created
#            20 Jun 2026 - Last edit
#
#############################################################################

"""Parameterized Gaussian mixture assigner with multi-method dispatch.

The :class:`XGMMAssigner` is the central assigner in the roadrunner
pipeline.  It is initialised with a ``method`` string (``'gmm'``,
``'bgmm'``, or ``'svi-bgmm'``) and an optional ``**mixture_kwargs``
dict that is forwarded to the concrete mixture constructor.
"""

import warnings

import numpy as np
import pandas as pd
from numba import njit, prange
from scipy.stats import rankdata

from roadrunner._mcf_types import AssignmentResult
from roadrunner.clustering.assignment.statistics import GMMAssignerStatistics
from roadrunner.clustering.sparse import SparseCSC
from roadrunner.mixture._math import logsumexp, row_l1_normalize
from roadrunner.mixture.base import BaseMixture
from roadrunner.mixture.weighted_gmm import (
    WeightedGaussianMixture,
    _estimate_gaussian_parameters,
    _estimate_log_gaussian_prob,
)
from roadrunner.mixture.bayesian_gmm import WeightedBayesianGaussianMixture
from roadrunner.mixture.svi_bayesian_gmm import SVIBayesianGaussianMixture
from roadrunner._defaults import (
    GALAXY_ID,
    LOCAL_IDX,
    PRIOR_DOF_OFFSET,
    UNBOUND,
    UNRESOLVED_GROUP_RATIO,
    math_dtype,
    precision,
)
from roadrunner.physics.halo_ensemble import HaloEnsemble
from roadrunner.randomness import spawn_seeds

from roadrunner.clustering.assignment import priors as prior


_MIXTURE_CLASSES: dict[str, type[BaseMixture]] = {
    "gmm": WeightedGaussianMixture,
    "bgmm": WeightedBayesianGaussianMixture,
    "svi-bgmm": SVIBayesianGaussianMixture,
}


def _get_mixture_class(method: str) -> type[BaseMixture]:
    """Look up the mixture class for a given method name.

    Parameters
    ----------
    method : str
        One of the keys in ``_MIXTURE_CLASSES``.

    Returns
    -------
    mixture_class : type
        A ``BaseMixture`` subclass.
    """
    try:
        return _MIXTURE_CLASSES[method]
    except KeyError:
        raise ValueError(
            f"Unknown assignment method {method!r}. "
            f"Choose from: {list(_MIXTURE_CLASSES)}")


def _no_transform(values):
    """Identity transform.

    Parameters
    ----------
    values : ndarray
        Input array.

    Returns
    -------
    values : ndarray
        Same array, unchanged.
    """
    return values


def _rank_transform(values, func_rank=np.log1p):
    """Rank-based transform applied to boundness values before normalisation.

    Ranks replace the raw boundness energies, then ``func_rank``
    is applied (default: ``log1p``).  Empty arrays are returned
    unchanged.

    Parameters
    ----------
    values : ndarray
        Input boundness values.
    func_rank : callable, default=np.log1p
        Transformation applied to the ranks.

    Returns
    -------
    transformed : ndarray
        Rank-transformed values (ambient math precision).
    """
    if len(values) == 0:
        return values
    ranks = rankdata(values, method="ordinal")
    return func_rank(ranks).astype(math_dtype(), copy=False)


@njit(parallel=True, cache=True)
def _rows_to_pad_kernel(prev, latent):
    """Mark the rows whose starting responsibilities the E-step recomputes (parallel).

    A row is marked when its particle is bound now (``latent > 0``) to a
    component it had no previous responsibility for: newborn (every row
    of a group is bound to one of its halos), unbound or in another
    group at the previous snapshot, newly bound, or underflowed to 0.

    Parameters
    ----------
    prev : ndarray of shape (n_particles, n_components)
        Previous responsibilities, aligned to the group.
    latent : ndarray of shape (n_particles, n_components)
        Current latent prior (normalised boundness).

    Returns
    -------
    pad : ndarray of bool, shape (n_particles,)
        Rows to recompute.
    """
    n, k = prev.shape
    pad = np.zeros(n, dtype=np.bool_)
    for i in prange(n):
        flag = False
        j = 0
        while not flag and j < k:
            flag = latent[i, j] > 0 and prev[i, j] <= 0
            j += 1
        pad[i] = flag
    return pad


@njit(parallel=True, cache=True)
def _join_resp_kernel(resp, rows, log_resp_new):
    """Overwrite ``resp[rows]`` with ``exp(log_resp_new)`` (in place, parallel).

    Parameters
    ----------
    resp : ndarray of shape (n_particles, n_components)
        Starting responsibilities. Modified in place.
    rows : ndarray of int, shape (m,)
        Rows of ``resp`` to overwrite.
    log_resp_new : ndarray of shape (m, n_components)
        Normalised log-responsibilities of those rows.
    """
    m, k = log_resp_new.shape
    for i in prange(m):
        r = rows[i]
        for j in range(k):
            resp[r, j] = np.exp(log_resp_new[i, j])


@njit(parallel=True, cache=True)
def _drop_unbound_kernel(resp, latent):
    """Zero ``resp[n, k]`` wherever particle ``n`` is no longer bound to ``k`` (in place, parallel).

    13-16x faster than ``resp[latent <= 0] = 0`` (benchmarked at
    200k x 8 and 3M x 20).

    Parameters
    ----------
    resp : ndarray of shape (n_particles, n_components)
        Starting responsibilities. Modified in place.
    latent : ndarray of shape (n_particles, n_components)
        Current latent prior (normalised boundness).
    """
    n, k = resp.shape
    for i in prange(n):
        for j in range(k):
            if latent[i, j] <= 0:
                resp[i, j] = 0.0


def _e_step(X, log_alpha, nk, means, covariances, cov_type):
    """E-step of a Gaussian mixture from given parameters.

    The computation of ``WeightedGaussianMixture._e_step``, with the
    component weights from the counts ``nk``:
    ``log_resp[i, k] = log N(x_i | mu_k, Sigma_k) + log(nk_k / sum(nk))
    + log_alpha[i, k] - log_norm[i]``.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Data.
    log_alpha : ndarray of shape (n_samples, n_components)
        Log latent prior (-inf where a component is excluded).
    nk : ndarray of shape (n_components,)
        Component counts (> 0).
    means : ndarray of shape (n_components, n_features)
        Component means.
    covariances : ndarray
        Component covariances (``cov_type`` shape).
    cov_type : str
        Covariance type.

    Returns
    -------
    log_resp : ndarray of shape (n_samples, n_components)
        Log posterior responsibilities.
    log_norm : ndarray of shape (n_samples, 1)
        Log normalisation constants.
    """
    with np.errstate(divide="ignore"):
        log_gauss = _estimate_log_gaussian_prob(X, means, covariances, cov_type)
        log_weights = np.log(nk / nk.sum()).astype(X.dtype, copy=False)
        w_log_gauss = log_gauss + log_weights + log_alpha
        log_norm = logsumexp(w_log_gauss)
        log_resp = w_log_gauss - log_norm
    return log_resp, log_norm


class XGMMAssigner:
    """Parameterised Gaussian-mixture assigner with method dispatch.

    Dispatches to a concrete mixture class (GMM, BGMM, or SVI-BGMM)
    based on the ``method`` string.  Handles per-snapshot fitting,
    temporal smoothing via previous responsibilities, and construction
    of BGMM prior kwargs from the previous snapshot's fitted parameters.

    Parameters
    ----------
    cov_type : str, default='full'
        Covariance type.
    max_iter : int, default=10
        Maximum EM iterations (``gmm`` / ``bgmm``).
    tol : float, default=1e-2
        EM convergence tolerance.
    min_particles : int, default=10
        Minimum particles per component.
    reg_covar : float, default=1e-6
        Covariance regularisation.
    prior_type : str, default=''
        Prior type string (experimental).
    verbose : int, default=1
        Verbosity level.
    method : str, default='gmm'
        One of ``'gmm'``, ``'bgmm'``, or ``'svi-bgmm'``.
    use_bgmm_priors : bool, default=True
        Use fitted parameters from the previous snapshot as BGMM priors.
        Halos without them (first snapshot, newborns, or with this off)
        take their own pre-fit estimate as the prior reference.
    mass_weighting : bool, default=False
        Weight each particle by its current mass in the fits, instead
        of counting particles. Weights are normalised per fitted group
        to sum to the group's effective sample size
        ``(sum m)**2 / sum m**2`` (see :meth:`_group_weights`), so the
        mass unit drops out, the BGMM priors compete with the data's
        actual information, and equal masses reproduce the unweighted
        fit exactly. Fitted ``count`` values, and the bound counts of
        the BGMM temporal priors, are then in these weighted units.
        Requires ``particle_masses`` in :meth:`assign`. Not supported
        by ``'svi-bgmm'``.
    **mixture_kwargs
        Additional keyword arguments forwarded to the mixture
        constructor (e.g. ``n_svi_iters``, ``batch_size`` for SVI).
    """

    def __init__(self, cov_type="full", max_iter=10, tol=1e-2,
                 min_particles=10, reg_covar=1e-6, prior_type="",
                 verbose=1, method="gmm", use_bgmm_priors=True,
                 mass_weighting=False, **mixture_kwargs):
        self.cov_type = cov_type
        self.max_iter = max_iter
        self.tol = tol
        self.min_particles = min_particles
        self.reg_covar = reg_covar
        self.prior_type = prior_type
        self.verbose = verbose
        self.method = method
        self.mixture_class = _get_mixture_class(method)
        self.statistics = GMMAssignerStatistics()
        self.previous_parameters = None
        self.parameters: dict[int, dict] = {}
        self.use_bgmm_priors = use_bgmm_priors
        if mass_weighting and method == "svi-bgmm":
            raise ValueError("mass_weighting is not supported with method='svi-bgmm'")
        self.mass_weighting = mass_weighting
        self.mixture_kwargs = mixture_kwargs

    def assign(self, halos, particle_coords, groups, **kwargs) -> AssignmentResult:
        """Run the assigner on one snapshot's data.

        Parameters
        ----------
        halos : HaloEnsemble or list of HaloModel
            Halos with boundness already computed.
        particle_coords : ndarray of shape (n_particles, 6)
            6-D phase-space coordinates.
        groups : list of list of int
            Indices of halos belonging to each overlapping group.
        **kwargs
            Additional arguments (``snap_id``, ``previous_resp``, etc.).
            ``seed`` (int or None): root seed for this snapshot's fit
            stage; each group gets an independent child seed spawned
            from it, in the order ``groups`` is given (already sorted
            largest-first by the caller). ``None`` uses OS entropy.
            ``particle_masses`` (ndarray of shape (n_particles,)):
            current particle masses, required with ``mass_weighting``.

        Returns
        -------
        result : AssignmentResult
            Particle assignment, responsibilities, and fitted parameters.
        """
        self.previous_parameters = dict(self.parameters) or None
        self.parameters = {}

        self.ensemble = (
            HaloEnsemble(halos)
            if not isinstance(halos, HaloEnsemble)
            else halos
        )
        # Tree mass by Sub_tree_id: stored with the fitted parameters, it is
        # the next snapshot's reference for the covariance prior's growth.
        self._tree_mass = dict(zip(self.ensemble.sub_tree_ids.tolist(),
                                   self.ensemble.masses.tolist()))
        self.particle_coords = particle_coords
        self.previous_resp = kwargs.get("previous_resp", {})
        self.particle_masses = kwargs.get("particle_masses")
        if self.mass_weighting and self.particle_masses is None:
            raise ValueError("mass_weighting requires particle_masses")

        N = particle_coords.shape[0]
        self.particles_df = pd.DataFrame(
            {"array_index": np.arange(N, dtype=LOCAL_IDX), "Sub_tree_id": UNBOUND}
        ).set_index("array_index")
        self.resp_map: dict[int, tuple[np.ndarray, np.ndarray]] = {}

        ngal = np.concatenate(groups).size if groups else 0
        print(f"{ngal} in a total of {len(groups)} groups")

        seed = kwargs.get("seed")
        group_seeds = spawn_seeds(seed, len(groups)) if seed is not None else [None] * len(groups)

        for group, group_seed in zip(groups, group_seeds):
            self.parameters.update(self._process_group(group, seed=group_seed))

        self.particles_df.reset_index(inplace=True)
        csc_b, _ = self.ensemble.get_particles()
        self.statistics.compute(
            self.particles_df, self.resp_map, self.parameters,
            boundness_csc=csc_b,
        )

        gids = np.array(sorted(self.resp_map.keys()), dtype=GALAXY_ID)
        resp_csc = SparseCSC(
            [self.resp_map[g][0] for g in gids],
            [self.resp_map[g][1] for g in gids],
            column_id=gids,
        )
        return AssignmentResult(
            self.particles_df, resp_csc, self.parameters,
            {"groups": len(groups),
             "halos_in_groups": sum(len(g) for g in groups),
             "bound_particles": self.ensemble.nstars,
             **self.statistics.values,
            })

    def _process_group(self, group, seed=None) -> dict:
        """Process a single overlapping group of halos.

        Selects the appropriate fit path based on the number of
        components: ``_fit_single`` for 1, ``_fit_unresolved``
        for poorly-sampled groups, and ``_fit_resolved`` otherwise.

        Parameters
        ----------
        group : list of int
            Indices of halos in this group.
        seed : int or None, optional
            Seed for this group's fit (only used by ``_fit_resolved``,
            the only path that constructs a mixture instance).

        Returns
        -------
        params : dict of ``{Sub_tree_id: {mean, count, covariance, condition}}``
            Fitted parameters for each component in the group.
        """
        sub_ensemble = self.ensemble.select(group)
        csc_b, _ = sub_ensemble.get_particles()
        group_subtrees = sub_ensemble.sub_tree_ids
        gp_idx = np.unique(np.concatenate(csc_b.column_indices))

        if gp_idx.size == 0:
            warnings.warn(
                "WARNING: A non-empty group turns out to be empty of particles!"
            )
            return {}

        n_comp = len(group_subtrees)

        if n_comp == 1:
            params, post_prob, nonz = self._fit_single(gp_idx, group_subtrees)
        elif gp_idx.size < UNRESOLVED_GROUP_RATIO * n_comp:
            warnings.warn(
                f"WARNING: {n_comp} galaxies are unresolved inside a group!"
            )
            params, post_prob, nonz = self._fit_unresolved(gp_idx, group_subtrees, csc_b)
        else:
            params, post_prob, nonz = self._fit_resolved(
                gp_idx, group_subtrees, csc_b, seed=seed
            )

        for i in range(len(group)):
            mask = nonz[:, i] > 0
            if np.any(mask):
                self.resp_map[group_subtrees[i]] = (
                    gp_idx[mask], post_prob[mask, i],
                )

        assigned = group_subtrees[post_prob.argmax(axis=1)]
        self.particles_df.loc[gp_idx, "Sub_tree_id"] = assigned

        return params

    def _group_weights(self, gp_idx):
        """Per-particle fit weights of one group, in the ambient math precision.

        Ones when counting particles. With ``mass_weighting``,
        ``w_i = m_i * sum(m) / sum(m**2)``: proportional to mass and
        summing to the group's effective sample size
        ``ESS = sum(m)**2 / sum(m**2) <= N_g``, the number of equal-mass
        particles carrying the same information. Computed in float64
        via the mean-1 masses ``u``, so equal masses give exactly 1.

        Parameters
        ----------
        gp_idx : ndarray of int64
            Particle indices of the group.

        Returns
        -------
        weights : ndarray of shape (len(gp_idx),)
        """
        if not self.mass_weighting:
            return np.ones(gp_idx.size, dtype=math_dtype())
        u = np.asarray(self.particle_masses[gp_idx], dtype=np.float64)
        u /= u.mean()
        return (u / np.mean(u * u)).astype(math_dtype(), copy=False)

    def _fit_single(self, gp_idx, group_subtrees):
        """Fit a single-component group (only one halo).

        Parameters
        ----------
        gp_idx : ndarray of int64
            Particle indices belonging to this group.
        group_subtrees : ndarray of int64
            ``Sub_tree_id`` of the single halo.

        Returns
        -------
        params : dict
            Fitted parameters.
        post_prob : ndarray of shape (n_particles, 1)
            Posterior responsibilities.
        nonz : ndarray of shape (n_particles, 1)
            Non-zero mask.
        """
        post_prob = np.ones((gp_idx.size, 1), dtype=math_dtype())
        nonz = (post_prob > 0).astype(np.uint8)

        nk, means, covs = _estimate_gaussian_parameters(
            self.particle_coords[gp_idx], post_prob,
            self._group_weights(gp_idx),
            self.cov_type, reg_covar=self.reg_covar,
        )
        sid = int(group_subtrees[0])
        vals = np.linalg.eigvalsh(covs[0]) if covs[0].ndim == 2 else covs[0]
        params = {
            sid: {
                "mean": means[0],
                "count": float(nk[0]),
                "weight": 1.0,
                "covariance": covs[0],
                "covariance_condition": float(vals.max() / max(vals.min(), 1e-30)),
                "tree_mass": self._tree_mass.get(sid, 0.0),
            }
        }
        return params, post_prob, nonz

    def _fit_unresolved(self, gp_idx, group_subtrees, csc_b):
        """Fit an unresolved group where particles are too few for EM.

        Falls back to the boundness matrix itself as the responsibility
        matrix, then calls ``_estimate_gaussian_parameters`` to
        produce standard fitted parameters.

        Parameters
        ----------
        gp_idx : ndarray of int64
            Particle indices.
        group_subtrees : ndarray of int64
            ``Sub_tree_id`` for each component.
        csc_b : SparseCSC
            Boundness matrix for this group.

        Returns
        -------
        params : dict
            Fitted parameters.
        post_prob : ndarray of shape (n_particles, n_components)
            Boundness-derived responsibilities.
        nonz : ndarray of shape (n_particles, n_components)
            Non-zero mask.
        """
        post_prob = row_l1_normalize(
            csc_b.to_dense(col_func=_no_transform)
        ).astype(math_dtype(), copy=False)
        nonz = (post_prob > 0).astype(np.uint8)

        nk, means, covs = _estimate_gaussian_parameters(
            self.particle_coords[gp_idx], post_prob,
            self._group_weights(gp_idx),
            self.cov_type, reg_covar=self.reg_covar,
        )
        total_count = float(nk.sum())
        params = {}
        for i, sid in enumerate(group_subtrees):
            sid = int(sid)
            vals = np.linalg.eigvalsh(covs[i]) if covs[i].ndim == 2 else covs[i]
            params[sid] = {
                "mean": means[i],
                "count": float(nk[i]),
                "weight": float(nk[i]) / total_count if total_count > 0 else 0.0,
                "covariance": covs[i],
                "covariance_condition": float(vals.max() / max(vals.min(), 1e-30)),
                "tree_mass": self._tree_mass.get(sid, 0.0),
            }
        return params, post_prob, nonz

    def _fit_resolved(self, gp_idx, group_subtrees, csc_b, seed=None):
        """Fit a resolved group with a proper XGMM fit.

        Scales coordinates, estimates initial parameters, builds
        prior kwargs for BGMM if applicable, and fits the mixture
        using the configured method.

        Parameters
        ----------
        gp_idx : ndarray of int64
            Particle indices.
        group_subtrees : ndarray of int64
            ``Sub_tree_id`` for each component.
        csc_b : SparseCSC
            Boundness matrix for this group.
        seed : int or None, optional
            Passed through as ``random_state`` to the mixture
            constructor (accepts a plain seed int or an rng/state
            object). ``None`` leaves the mixture's own default.

        Returns
        -------
        params : dict
            Fitted parameters.
        post_prob : ndarray of shape (n_particles, n_components)
            Posterior responsibilities.
        nonz : ndarray of shape (n_particles, n_components)
            Non-zero mask.
        """
        from roadrunner.physics.scaler import StandardScaler
        n_comp = len(group_subtrees)
        run_kwargs = dict(
            max_iter=self.max_iter,
            reg_covar=self.reg_covar,
            tol=self.tol,
            verbose=self.verbose,
            **self.mixture_kwargs,
        )
        if seed is not None:
            run_kwargs["random_state"] = seed

        # Precision comes from the ambient scope (set once per run from the
        # user "single"/"double" knobs). Everything below is rebuilt per
        # attempt so a double retry recomputes in double, not just refits.
        # (The old string-keyed ladder never matched and so never fired;
        # this one does.)
        first = "single" if math_dtype() == np.float32 else "double"
        for math_name in dict.fromkeys([first, "double"]):
            try:
                with precision(math=math_name):
                    md = math_dtype()
                    scaler = StandardScaler()
                    coords = scaler.fit_transform(
                        self.particle_coords[gp_idx].astype(md, copy=False))
                    w = self._group_weights(gp_idx)
                    prior, nk, means, covs, cov_t = self._estimate_initial_params(
                        coords, csc_b, w, scaler)
                    init_kwargs = dict(
                        n_components=n_comp,
                        counts_init=np.asarray(nk, dtype=md),
                        means_init=np.asarray(means, dtype=md),
                        covariance_init=np.asarray(covs, dtype=md),
                        cov_type=cov_t,
                        init_params="kmeans++",
                    )
                    prior_kwargs = self._build_prior_kwargs(
                        group_subtrees, csc_b, scaler, n_comp, gp_idx, w,
                        nk, covs)
                    # Unweighted fits pass no weights: bitwise the
                    # unweighted path.
                    fit_kwargs = {"point_weights": w} if self.mass_weighting else {}
                    gmm = self.mixture_class(
                        **init_kwargs, **prior_kwargs, **run_kwargs
                    ).fit(coords, latent_prior=prior, **fit_kwargs)
                break
            except (np.linalg.LinAlgError, ValueError):
                if math_name == "double":
                    raise
                warnings.warn("Numerical issue — retrying with double.")

        nonz = (prior > 0).astype(np.uint8)

        log_prob = gmm.predict_log_proba(coords, latent_prior=prior)
        post_prob = np.exp(log_prob)

        scaled = {
            "means": {int(sid): gmm.means_[i]
                      for i, sid in enumerate(group_subtrees)},
            "weights": {int(sid): gmm.weights_[i]
                        for i, sid in enumerate(group_subtrees)},
            "covariances": {int(sid): gmm.covariances_[i]
                            for i, sid in enumerate(group_subtrees)},
            "cov_type": gmm.cov_type,
        }
        natural = self.get_parameters_natural(scaled, scaler)

        nk_after = (post_prob * w[:, None]).sum(axis=0)

        params = {}
        for i, sid in enumerate(group_subtrees):
            sid = int(sid)
            if sid in natural["means"]:
                cov_scaled = gmm.covariances_[i]
                vals = np.linalg.eigvalsh(cov_scaled) if cov_scaled.ndim == 2 else cov_scaled
                cond = float(vals.max() / max(vals.min(), 1e-30))
                params[sid] = {
                    "mean": natural["means"][sid],
                    "count": float(nk_after[i]),
                    "weight": natural["weights"][sid],
                    "covariance": natural["covariances"][sid],
                    "covariance_condition": cond,
                    "tree_mass": self._tree_mass.get(sid, 0.0),
                }

        return params, post_prob, nonz

    def _build_prior_kwargs(self, group_subtrees, csc_b, scaler, n_comp,
                            gp_idx, w, nk_init, covs_init):
        """Build explicit BGMM prior kwargs for every component of a group.

        Each component's prior is built from a reference count and
        covariance: its previous snapshot's fitted parameters when
        available (and ``use_bgmm_priors``), otherwise its own pre-fit
        estimate (first snapshot, newborn halos). The covariance prior
        scales the reference by the halo's tree-mass growth since the
        reference snapshot (virial scaling, ``mass_ratio ** (2/3)``), and
        by ``priors.covariance_scale(n)``. All three priors take the same
        reference count. Returns an empty dict when the method is not
        BGMM.

        Parameters
        ----------
        group_subtrees : ndarray of int64
            ``Sub_tree_id`` for each component.
        csc_b : SparseCSC
            Current boundness matrix.
        scaler : StandardScaler
            Fitted scaler for this group.
        n_comp : int
            Number of components.
        gp_idx : ndarray of int64
            Particle indices of the group (sorted).
        w : ndarray of shape (len(gp_idx),)
            Fit weights of the group's particles: each component's bound
            count is the total weight of its bound particles, in the
            same units as the fitted ``count``.
        nk_init : ndarray of shape (n_comp,)
            Pre-fit expected counts (the initial responsibilities,
            carrying the previous snapshot's, summed with weights ``w``):
            the reference count of halos without history.
        covs_init : ndarray
            Pre-fit covariances in scaled coordinates (``cov_type`` shape).

        Returns
        -------
        prior_kwargs : dict or empty dict
            BGMM constructor kwargs (``mean_prior``, ``covariance_prior``,
            ``weight_concentration_prior``, etc.).
        """
        if self.method != "bgmm":
            return {}
        history = self.previous_parameters if self.use_bgmm_priors else None

        md = math_dtype()
        n_f = self.particle_coords.shape[1]
        s = scaler.scale_
        dof = n_f + PRIOR_DOF_OFFSET

        mp = np.zeros((n_comp, n_f), dtype=md)
        cp = (
            np.zeros((n_comp, n_f, n_f), dtype=md) if self.cov_type == "full"
            else np.zeros((n_comp, n_f), dtype=md) if "diag" in self.cov_type
            else np.zeros(n_comp, dtype=md)
        )
        wp = np.full(n_comp, 1.0 / n_comp, dtype=md)
        pp = np.ones(n_comp, dtype=md)

        bound_counts = np.array(
            [w[np.searchsorted(gp_idx, c)].sum(dtype=np.float64)
             for c in csc_b.column_indices],
            dtype=md)

        pos6, ratios = self._group_tree_state(group_subtrees)
        mp[:] = (pos6 - scaler.mean_) * s

        for i, sid in enumerate(group_subtrees):
            sid_int = int(sid)
            n_b = max(bound_counts[i], 1.0)
            p = history.get(sid_int) if history else None
            if p is not None:
                nk_n1 = max(p.get("count", 1.0), 1.0)
                ref_vars = prior.degrade_covariance(p["covariance"], n_f)
                mass_ratio = ratios[i]
            else:
                # No history: the halo's own pre-fit estimate is the
                # reference (its scaled variances back in natural units).
                nk_n1 = max(float(nk_init[i]), 1.0)
                ref_vars = prior.degrade_covariance(np.asarray(covs_init[i]), n_f) / s**2
                mass_ratio = 1.0

            wp[i] = prior.weight_concentration_prior(nk_n1, n_b, n_comp)
            pp[i] = prior.mean_precision_prior(nk_n1, n_b)
            cp[i] = prior.covariance_prior(ref_vars, mass_ratio, nk_n1, s, dof, self.cov_type)

        return dict(
            mean_prior=np.asarray(mp, dtype=md),
            covariance_prior=np.asarray(cp, dtype=md),
            weight_concentration_prior=np.asarray(wp, dtype=md),
            mean_precision_prior=np.asarray(pp, dtype=md),
            degrees_of_freedom_prior=dof,
        )

    def _group_tree_state(self, group_subtrees):
        """Current tree phase-space position and tree-mass ratio of a group's halos.

        Every halo of a group is in the ensemble (groups are
        sub-ensembles).

        Parameters
        ----------
        group_subtrees : ndarray of int64
            ``Sub_tree_id`` of each component.

        Returns
        -------
        pos6 : ndarray of shape (n_comp, 6)
            Tree position and velocity (natural units).
        mass_ratio : ndarray of shape (n_comp,)
            Current tree mass over the tree mass stored with the previous
            snapshot's fit; 1 without a previous fit or stored tree mass
            (old checkpoints carry none).
        """
        ids = self.ensemble.sub_tree_ids
        order = np.argsort(ids)
        idx = order[np.searchsorted(ids, group_subtrees, sorter=order)]
        pos6 = np.hstack([self.ensemble.positions[idx], self.ensemble.velocities[idx]])
        history = self.previous_parameters or {}
        m_ref = np.array([history.get(int(sid), {}).get("tree_mass", 0.0)
                          for sid in group_subtrees], dtype=np.float64)
        m_now = self.ensemble.masses[idx]
        ok = (m_ref > 0) & (m_now > 0)
        return pos6, np.where(ok, m_now / np.where(ok, m_ref, 1.0), 1.0)

    def _predictive_params(self, coords, latent, csc_b, scaler, w):
        """Counts, means and covariances of the predictive E-step for a group.

        Means: each halo's current tree position and velocity. Halos with
        a previous fit: the previous covariance grown by
        ``priors.covariance_growth`` of the tree-mass ratio, and the
        previous count (clipped at 1). New halos: from the particles bound
        only to them within the group, count ``max(sum w, D + 1)`` and
        their covariance, or that of their ``D + 1`` most bound particles
        when fewer are exclusive.

        Parameters
        ----------
        coords : ndarray of shape (n_particles, D)
            Scaled coordinates of the group.
        latent : ndarray of shape (n_particles, n_components)
            Latent prior (normalised boundness).
        csc_b : SparseCSC
            Boundness matrix of the group.
        scaler : StandardScaler
            The group's scaler.
        w : ndarray of shape (n_particles,)
            Fit weights.

        Returns
        -------
        nk : ndarray of shape (n_components,)
            Component counts.
        means : ndarray of shape (n_components, D)
            Component means (scaled units).
        covs : ndarray
            Component covariances (``cov_type`` shape, scaled units).
        """
        md, d = coords.dtype, coords.shape[1]
        s = scaler.scale_.astype(np.float64)
        sids = csc_b.column_id
        history = self.previous_parameters or {}
        prev = [history.get(int(sid)) for sid in sids]
        old = np.array([p is not None for p in prev], dtype=bool)
        pos6, mass_ratio = self._group_tree_state(sids)

        full, diag = self.cov_type == "full", "diag" in self.cov_type
        covs = np.empty((len(sids),) + ((d, d) if full else (d,) if diag else ()), dtype=md)
        nk = np.empty(len(sids), dtype=md)
        means = ((pos6 - scaler.mean_) * s).astype(md)

        if old.any():
            # Previous fit, moved to the current tree.
            C = np.stack([np.asarray(p["covariance"], dtype=np.float64)
                          for p in prev if p is not None])
            to_scaled = np.outer(s, s) if full else s**2 if diag else 1.0 / np.mean(1.0 / s**2)
            growth = prior.covariance_growth(mass_ratio[old]).reshape((-1,) + (1,) * (C.ndim - 1))
            covs[old] = C * to_scaled * growth
            nk[old] = np.maximum([p["count"] for p in prev if p is not None], 1.0)

        new = np.flatnonzero(~old)
        if new.size:
            # New halos: their exclusive particles, or their D + 1 most bound.
            exclusive = (latent[:, new] > 0) & (np.count_nonzero(latent, axis=1) == 1)[:, None]
            own = exclusive.copy()
            for j in np.flatnonzero(exclusive.sum(axis=0) <= d):
                k = new[j]
                rows_k = np.searchsorted(csc_b.row_id, csc_b.column_indices[k])
                if rows_k.size > d + 1:
                    rows_k = rows_k[np.argpartition(csc_b.column_values[k], -(d + 1))[-(d + 1):]]
                own[:, j] = False
                own[rows_k, j] = True
            sel = np.flatnonzero(own.any(axis=1))
            _, _, cv = _estimate_gaussian_parameters(
                coords[sel], own[sel].astype(md), w[sel], self.cov_type, reg_covar=self.reg_covar)
            covs[new] = cv
            nk[new] = np.maximum(w @ exclusive, d + 1.0)
        return nk, means, covs

    def _initial_responsibilities(self, coords, csc_b, w, scaler):
        """Latent prior and starting responsibilities of a resolved group.

        Without previous responsibilities (first snapshot) the start is
        the latent prior. Otherwise each particle keeps its previous
        responsibilities for the components it is still bound to, except
        those bound to a component they had no previous responsibility
        for (newborn, previously unbound or in another group, newly
        bound), whose rows come from the predictive E-step ``log a_nk(t) + log pi_k(t-1) + log N(x_n |
        m_k(t), Sigma_k(t-1))`` (see ``_predictive_params``).

        Parameters
        ----------
        coords : ndarray of shape (n_particles, n_features)
            Scaled coordinates of the group.
        csc_b : SparseCSC
            Boundness matrix of the group.
        w : ndarray of shape (n_particles,)
            Fit weights.
        scaler : StandardScaler
            The group's scaler.

        Returns
        -------
        latent : ndarray of shape (n_particles, n_components)
            Latent prior (the starting responsibilities themselves under
            ``prior_type="temporal-log-lik"``).
        resp : ndarray of shape (n_particles, n_components)
            Starting responsibilities (rows sum to 1).
        """
        latent = row_l1_normalize(
            csc_b.to_dense(col_func=_rank_transform)
        ).astype(math_dtype(), copy=False)
        if not self.previous_resp:
            return latent, latent

        resp = csc_b.align(self.previous_resp, how="left")[1].to_dense()
        rows = np.flatnonzero(_rows_to_pad_kernel(resp, latent))
        if rows.size:
            with np.errstate(divide="ignore"):
                log_alpha = np.log(latent[rows])
            log_resp_new, _ = _e_step(
                coords[rows], log_alpha,
                *self._predictive_params(coords, latent, csc_b, scaler, w), self.cov_type)
            _join_resp_kernel(resp, rows, log_resp_new)
        _drop_unbound_kernel(resp, latent)
        resp = row_l1_normalize(resp)
        return (resp if self.prior_type.lower() == "temporal-log-lik" else latent), resp

    def _estimate_initial_params(self, coords, csc_b, w, scaler):
        """Estimate initial mixture parameters from boundness and previous responsibilities.

        The starting responsibilities come from
        ``_initial_responsibilities``; the initial parameters are their
        Gaussian sufficient statistics.

        Parameters
        ----------
        coords : ndarray of shape (n_particles, n_features)
            Scaled phase-space coordinates.
        csc_b : SparseCSC
            Boundness matrix.
        w : ndarray of shape (n_particles,)
            Fit weights of the particles.
        scaler : StandardScaler
            The group's scaler.

        Returns
        -------
        prior : ndarray of shape (n_particles, n_components)
            Latent prior.
        nk : ndarray of shape (n_components,)
            Effective counts.
        means : ndarray of shape (n_components, n_features)
            Initial means.
        covs : ndarray
            Initial covariances.
        cov_t : str
            Covariance type.
        """
        prior, resp = self._initial_responsibilities(coords, csc_b, w, scaler)
        nk, means_init, covs_init = _estimate_gaussian_parameters(
            coords, resp, w, self.cov_type, reg_covar=self.reg_covar,
        )
        return prior, nk, means_init, covs_init, self.cov_type

    @staticmethod
    def get_parameters_natural(stored, scaler):
        """Convert fitted parameters from scaled to natural coordinates.

        Parameters
        ----------
        stored : dict
            Scaled parameters with keys ``'means'``, ``'weights'``,
            ``'covariances'``, and ``'cov_type'``.
        scaler : StandardScaler
            The scaler that was used to transform the data.

        Returns
        -------
        natural : dict
            Parameters in natural coordinates with the same structure
            as ``stored``.

        Raises
        ------
        FloatingPointError
            If a covariance overflows to inf in the conversion.
        """
        inv_s = 1.0 / scaler.scale_
        means, weights, covariances = {}, {}, {}
        cov_t = stored.get("cov_type", "full")

        for sid in stored.get("means", {}):
            mean_s = stored["means"][sid]
            w = stored["weights"][sid]
            cov_s = stored["covariances"][sid]

            if cov_t == "spherical":
                scaling_matrix = np.mean(inv_s**2)
            elif cov_t == "diagonal":
                scaling_matrix = inv_s**2
            else:
                scaling_matrix = np.outer(inv_s, inv_s)

            means[sid] = mean_s * inv_s + scaler.mean_
            weights[sid] = float(w)
            cov = cov_s * scaling_matrix
            if np.isinf(cov).any():
                # An overflow here means the fitted covariance has run
                # away (I01): fail where it arises, not one snapshot later
                # when it becomes the next prior's reference.
                raise FloatingPointError(
                    f"Covariance of halo {sid} overflowed converting to "
                    f"natural units (largest scaled entry {np.abs(cov_s).max():.3g})")
            covariances[sid] = cov

        return {"means": means, "weights": weights, "covariances": covariances}
