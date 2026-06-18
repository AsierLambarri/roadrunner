import warnings

import numpy as np
import pandas as pd
from numba import njit, prange
from scipy.stats import rankdata

from roadrunner._mcf_types import AssignmentResult
from roadrunner.clustering.assignment.statistics import GMMAssignerStatistics
from roadrunner.clustering.sparse import SparseCSC, stitch_zero_rows
from roadrunner.mixture._math import row_l1_normalize
from roadrunner.mixture.base import BaseMixture
from roadrunner.mixture.weighted_gmm import (
    WeightedGaussianMixture,
    _estimate_gaussian_parameters,
)
from roadrunner.mixture.bayesian_gmm import WeightedBayesianGaussianMixture
from roadrunner._defaults import (
    PRIOR_DOF_OFFSET,
    UNRESOLVED_GROUP_RATIO,
)
from roadrunner.physics.halo_ensemble import HaloEnsemble

_PRECISION_LADDER = {
    "float32": "float64",
    "float64": "float128",
}
from roadrunner.clustering.assignment import priors as prior


_MIXTURE_CLASSES: dict[str, type[BaseMixture]] = {
    "gmm": WeightedGaussianMixture,
    "bgmm": WeightedBayesianGaussianMixture,
}


def _get_mixture_class(method: str) -> type[BaseMixture]:
    try:
        return _MIXTURE_CLASSES[method]
    except KeyError:
        raise ValueError(
            f"Unknown assignment method {method!r}. "
            f"Choose from: {list(_MIXTURE_CLASSES)}")


def _no_transform(values):
    return values


def _rank_transform(values, func_rank=np.log1p):
    if len(values) == 0:
        return values
    ranks = rankdata(values, method="ordinal")
    return func_rank(ranks).astype(np.float32, copy=False)


@njit(parallel=True, cache=True)
def _merge_resp_kernel(raw, bound, n_components, newborn_1d):
    n = raw.shape[0]
    inv_ncomp = 1.0 / n_components
    for n_idx in prange(n):
        is_newb = newborn_1d[n_idx]
        for k_idx in range(n_components):
            b = bound[n_idx, k_idx]
            p = raw[n_idx, k_idx]
            if b > 0:
                if p > 0:
                    raw[n_idx, k_idx] = p
                elif not is_newb:
                    raw[n_idx, k_idx] = inv_ncomp
                else:
                    raw[n_idx, k_idx] = 0.0
            else:
                raw[n_idx, k_idx] = 0.0


class XGMMAssigner:
    def __init__(self, cov_type="full", max_iter=10, tol=1e-2,
                 min_particles=10, reg_covar=1e-6, prior_type="",
                 verbose=1, method="gmm", use_bgmm_priors=True,
                 dtype_math="float64"):
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
        self.dtype_math = dtype_math

    def assign(self, halos, particle_coords, newborn_indices, groups,
               **kwargs) -> AssignmentResult:
        self.previous_parameters = dict(self.parameters) or None
        self.parameters = {}

        self.ensemble = (
            HaloEnsemble(halos)
            if not isinstance(halos, HaloEnsemble)
            else halos
        )
        self.particle_coords = particle_coords
        self.newborn_indices = newborn_indices
        self.previous_resp = kwargs.get("previous_resp", {})

        N = particle_coords.shape[0]
        self.particles_df = pd.DataFrame(
            {"array_index": np.arange(N, dtype=np.uint64), "Sub_tree_id": -1}
        ).set_index("array_index")
        self.resp_map: dict[int, tuple[np.ndarray, np.ndarray]] = {}

        ngal = np.concatenate(groups).size if groups else 0
        print(f"{ngal} in a total of {len(groups)} groups")

        for group in groups:
            self.parameters.update(self._process_group(group))

        self.particles_df.reset_index(inplace=True)
        csc_b, _ = self.ensemble.get_particles()
        self.statistics.compute(
            self.particles_df, self.resp_map, self.parameters,
            boundness_csc=csc_b,
        )

        gids = np.array(sorted(self.resp_map.keys()), dtype=np.int64)
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

    def _process_group(self, group) -> dict:
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
                gp_idx, group_subtrees, csc_b
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

    def _fit_single(self, gp_idx, group_subtrees):
        post_prob = np.ones((gp_idx.size, 1))
        nonz = (post_prob > 0).astype(np.uint8)

        nk, means, covs = _estimate_gaussian_parameters(
            self.particle_coords[gp_idx], post_prob,
            np.ones(gp_idx.size, dtype=np.float32),
            self.cov_type, reg_covar=self.reg_covar,
        )
        sid = int(group_subtrees[0])
        vals = np.linalg.eigvalsh(covs[0]) if covs[0].ndim == 2 else covs[0]
        params = {
            sid: {
                "mean": means[0],
                "count": float(nk[0]),
                "covariance": covs[0],
                "covariance_condition": float(vals.max() / max(vals.min(), 1e-30)),
            }
        }
        return params, post_prob, nonz

    def _fit_unresolved(self, gp_idx, group_subtrees, csc_b):
        post_prob = row_l1_normalize(
            csc_b.to_dense(col_func=_no_transform)
        ).astype(np.float32, copy=False)
        nonz = (post_prob > 0).astype(np.uint8)

        nk, means, covs = _estimate_gaussian_parameters(
            self.particle_coords[gp_idx], post_prob,
            np.ones(gp_idx.size, dtype=np.float32),
            self.cov_type, reg_covar=self.reg_covar,
        )
        params = {}
        for i, sid in enumerate(group_subtrees):
            sid = int(sid)
            vals = np.linalg.eigvalsh(covs[i]) if covs[i].ndim == 2 else covs[i]
            params[sid] = {
                "mean": means[i],
                "count": float(nk[i]),
                "covariance": covs[i],
                "covariance_condition": float(vals.max() / max(vals.min(), 1e-30)),
            }
        return params, post_prob, nonz

    def _fit_resolved(self, gp_idx, group_subtrees, csc_b):
        from roadrunner.physics.scaler import StandardScaler
        scaler = StandardScaler()
        coords = scaler.fit_transform(
            self.particle_coords[gp_idx].astype(np.float64, copy=False))

        prior, nk, means, covs, cov_t = self._estimate_initial_params(
            coords, csc_b)
        nonz = (prior > 0).astype(np.uint8)
        n_comp = len(group_subtrees)

        init_kwargs = dict(
            n_components=n_comp,
            counts_init=nk,
            means_init=means,
            covariance_init=covs,
            cov_type=cov_t,
            init_params="kmeans++",
        )
        run_kwargs = dict(
            tol=self.tol,
            verbose=self.verbose,
        )

        prior_kwargs = self._build_prior_kwargs(group_subtrees, csc_b, scaler, n_comp)

        kwargs = dict(cast_dtype=getattr(np, self.dtype_math),
                      **init_kwargs, **prior_kwargs, **run_kwargs)

        for _ in range(3):
            try:
                gmm = self.mixture_class(**kwargs).fit(coords, latent_prior=prior)
                break
            except (np.linalg.LinAlgError, ValueError):
                next_dtype = _PRECISION_LADDER.get(kwargs["cast_dtype"])
                if next_dtype is not None and hasattr(np, next_dtype):
                    kwargs["cast_dtype"] = getattr(np, next_dtype)
                    warnings.warn(f"Numerical issue — retrying with {next_dtype}.")
                else:
                    raise

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

        nk_after = post_prob.sum(axis=0)

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
                    "covariance": natural["covariances"][sid],
                    "covariance_condition": cond,
                }

        return params, post_prob, nonz

    def _build_prior_kwargs(self, group_subtrees, csc_b, scaler, n_comp):
        if (self.method != "bgmm" or not self.use_bgmm_priors
                or self.previous_parameters is None):
            return {}

        n_f = self.particle_coords.shape[1]
        s = scaler.scale_
        dof = n_f + PRIOR_DOF_OFFSET

        mp = np.zeros((n_comp, n_f))
        cp = (
            np.zeros((n_comp, n_f, n_f)) if self.cov_type == "full"
            else np.zeros((n_comp, n_f)) if "diag" in self.cov_type
            else np.zeros(n_comp)
        )
        wp = np.full(n_comp, 1.0 / n_comp)
        pp = np.ones(n_comp)

        bound_counts = np.array([len(c) for c in csc_b.column_indices],
                                dtype=np.float64)

        tid_to_idx = dict(zip(self.ensemble.sub_tree_ids,
                              range(len(self.ensemble.sub_tree_ids))))

        for i, sid in enumerate(group_subtrees):
            sid_int = int(sid)

            idx = tid_to_idx.get(sid_int)
            if idx is not None:
                pos6 = np.concatenate([
                    self.ensemble.positions[idx],
                    self.ensemble.velocities[idx],
                ])
                mp[i] = (pos6 - scaler.mean_) * s

            p = self.previous_parameters.get(sid_int)
            if p is None:
                continue

            nk_n1 = max(p.get("count", 1.0), 1.0)
            n_b = max(bound_counts[i], 1.0)

            wp[i] = prior.weight_concentration_prior(nk_n1, n_b, n_comp)
            pp[i] = prior.mean_precision_prior(nk_n1, n_b)
            cp[i] = prior.covariance_prior(
                prior.degrade_covariance(p["covariance"], n_f),
                nk_n1, n_b,
                s, dof, self.cov_type,
            )

        return dict(
            mean_prior=np.asarray(mp, dtype=np.float64),
            covariance_prior=np.asarray(cp, dtype=np.float64),
            weight_concentration_prior=np.asarray(wp, dtype=np.float64),
            mean_precision_prior=np.asarray(pp, dtype=np.float64),
            degrees_of_freedom_prior=dof,
        )

    def _estimate_initial_params(self, coords, csc_b):
        n_samples = coords.shape[0]
        n_components = len(csc_b.column_indices)

        prior = row_l1_normalize(
            csc_b.to_dense(col_func=_rank_transform)
        ).astype(np.float32, copy=False)

        if self.previous_resp:
            _, aligned_p = csc_b.align(self.previous_resp, how="left")
            prev_dense = aligned_p.to_dense()

            newborn_1d = np.isin(csc_b.row_id, self.newborn_indices)
            _merge_resp_kernel(prev_dense, prior, n_components, newborn_1d)

            resp = stitch_zero_rows(prior, prev_dense)
            resp = row_l1_normalize(resp)
            if self.prior_type.lower() == "temporal-log-lik":
                prior = resp
        else:
            resp = prior

        nk, means_init, covs_init = _estimate_gaussian_parameters(
            coords, resp, np.ones(n_samples, dtype=np.float32),
            self.cov_type, reg_covar=self.reg_covar,
        )

        return prior, nk, means_init, covs_init, self.cov_type

    @staticmethod
    def get_parameters_natural(stored, scaler):
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
            covariances[sid] = cov_s * scaling_matrix

        return {"means": means, "weights": weights, "covariances": covariances}
