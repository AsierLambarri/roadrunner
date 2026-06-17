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
from roadrunner._defaults import UNRESOLVED_GROUP_RATIO
from roadrunner.physics.halo_ensemble import HaloEnsemble


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
                 verbose=1, method="gmm"):
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

    def assign(self, halos, particle_coords, newborn_indices, groups,
               **kwargs) -> AssignmentResult:
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
        self.parameters: dict[int, dict] = {}

        ngal = np.concatenate(groups).size if groups else 0
        print(f"{ngal} in a total of {len(groups)} groups")

        for group in groups:
            self._process_group(group)

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

    def _process_group(self, group):
        sub_ensemble = self.ensemble.select(group)
        csc_b, _ = sub_ensemble.get_particles()
        group_subtrees = sub_ensemble.sub_tree_ids
        gp_idx = np.unique(np.concatenate(csc_b.column_indices))

        if gp_idx.size == 0:
            warnings.warn(
                "WARNING: A non-empty group turns out to be empty of particles!"
            )
            return

        n_comp = len(group_subtrees)

        if n_comp == 1:
            post_prob, nonz = self._fit_single(gp_idx)
        elif gp_idx.size < UNRESOLVED_GROUP_RATIO * n_comp:
            warnings.warn(
                f"WARNING: {n_comp} galaxies are unresolved inside a group!"
            )
            post_prob, nonz = self._fit_unresolved(gp_idx, csc_b)
        else:
            post_prob, nonz = self._fit_resolved(
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

    def _fit_single(self, gp_idx):
        post_prob = np.ones((gp_idx.size, 1))
        nonz = (post_prob > 0).astype(np.uint8)
        return post_prob, nonz

    def _fit_unresolved(self, gp_idx, csc_b):
        post_prob = row_l1_normalize(
            csc_b.to_dense(col_func=_no_transform)
        ).astype(np.float32, copy=False)
        nonz = (post_prob > 0).astype(np.uint8)
        return post_prob, nonz

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
            cast_dtype=np.float32,
            tol=self.tol,
            verbose=self.verbose,
        )

        try:
            gmm = self.mixture_class(
                **init_kwargs, **run_kwargs,
            ).fit(coords, latent_prior=prior)
        except np.linalg.LinAlgError:
            run_kwargs["cast_dtype"] = np.float64
            gmm = self.mixture_class(
                **init_kwargs, **run_kwargs,
            ).fit(coords, latent_prior=prior)
            warnings.warn("Precision increased to float64.")

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

        for i, sid in enumerate(group_subtrees):
            sid = int(sid)
            if sid in natural["means"]:
                cov_scaled = gmm.covariances_[i]
                vals = np.linalg.eigvalsh(cov_scaled) if cov_scaled.ndim == 2 else cov_scaled
                cond = float(vals.max() / max(vals.min(), 1e-30))
                self.parameters[sid] = {
                    "mean": natural["means"][sid],
                    "weight": natural["weights"][sid],
                    "covariance": natural["covariances"][sid],
                    "covariance_condition": cond,
                }

        return post_prob, nonz

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
