import warnings

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from tqdm import tqdm

from roadrunner._mcf_types import AssignmentResult
from roadrunner.clustering.sparse import SparseCSC, build_dense_from_csc, stitch_zero_rows
from roadrunner.mixture._math import row_l1_normalize
from roadrunner.mixture.weighted_gmm import (
    WeightedGaussianMixture,
    _estimate_gaussian_parameters,
)
from roadrunner.physics.halo_ensemble import HaloEnsemble


def _no_transform(values):
    return values


def _rank_transform(values, func_rank=np.log1p):
    if len(values) == 0:
        return values
    ranks = rankdata(values, method="ordinal")
    return func_rank(ranks).astype(np.float32, copy=False)


def _build_raw_resp_with_previous_and_bound(
    matrix_shape,
    group_subtrees,
    newborn_array_indexes,
    true_indices,
    csc_boundness,
    prev_resp_map,
):
    n_samples, n_components = matrix_shape
    default_value = 1.0 / n_components
    col_idx_boundness, col_val_boundness = csc_boundness
    new_col_idx, new_col_val = [], []

    for k, sub_id in tqdm(
        enumerate(group_subtrees),
        desc="Building resp matrix...",
        total=n_components,
    ):
        curr_pids = col_idx_boundness[k]
        if curr_pids.size == 0:
            new_col_idx.append(np.array([], dtype=np.float32))
            new_col_val.append(np.array([], dtype=np.float32))
            continue

        prev_pids, prev_vals = prev_resp_map.pop(
            sub_id,
            (np.array([], dtype=np.float32), np.array([], dtype=np.float32)),
        )
        newbound = np.setdiff1d(curr_pids, prev_pids, assume_unique=True)
        newly_bound = np.setdiff1d(newbound, newborn_array_indexes, assume_unique=True)
        new_values = np.full(curr_pids.shape, 0, dtype=np.float32)
        new_values[np.isin(curr_pids, newly_bound)] = default_value

        if prev_pids.size > 0:
            ordering = np.argsort(prev_pids)
            sorted_prev_idx = prev_pids[ordering]
            sorted_prev_val = prev_vals[ordering]
            pos = np.searchsorted(sorted_prev_idx, curr_pids)
            valid = pos < sorted_prev_idx.size
            valid[valid] &= sorted_prev_idx[pos[valid]] == curr_pids[valid]
            new_values[valid] = sorted_prev_val[pos[valid]]

        new_col_idx.append(curr_pids)
        new_col_val.append(new_values)

    return build_dense_from_csc(
        matrix_shape, true_indices, new_col_idx, new_col_val
    )


class GMMAssigner:
    def __init__(
        self,
        cov_type="full",
        max_iter=10,
        tol=5e-2,
        min_particles=10,
        reg_covar=1e-6,
        prior_type="",
        verbose=1,
    ):
        self.cov_type = cov_type
        self.max_iter = max_iter
        self.tol = tol
        self.min_particles = min_particles
        self.reg_covar = reg_covar
        self.prior_type = prior_type
        self.verbose = verbose

    def assign(
        self, halos, particle_coords, newborn_indices, groups, **kwargs
    ) -> AssignmentResult:
        self.ensemble = (
            HaloEnsemble(halos) if not isinstance(halos, HaloEnsemble) else halos
        )
        self.particle_coords = particle_coords
        self.newborn_indices = newborn_indices
        self.groups = groups
        self.previous_resp = kwargs.get("previous_resp", {})

        N = particle_coords.shape[0]
        self.particles_df = pd.DataFrame(
            {
                "array_index": np.arange(N, dtype=np.uint64),
                "Sub_tree_id": -1,
            }
        ).set_index("array_index")
        self.resp_map: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self.parameters: dict[int, dict] = {}

        ngal = np.concatenate(groups).size if groups else 0
        print(f"{ngal} in a total of {len(groups)} groups")

        for group in groups:
            self._process_group(group)

        self.particles_df.reset_index(inplace=True)

        stats = {
            "groups": len(groups),
            "halos_in_groups": sum(len(g) for g in groups),
            "bound_particles": self.ensemble.nstars,
        }

        return AssignmentResult(
            self.particles_df, self.resp_map, self.parameters, stats
        )

    def _process_group(self, group):
        sub = self.ensemble.select(group)
        csc_b, _ = sub.get_particles()
        group_subtrees = sub.sub_tree_ids
        group_candidates = csc_b.column_indices
        group_boundness = csc_b.column_values
        gp_idx = np.unique(np.concatenate(group_candidates))

        if gp_idx.size == 0:
            warnings.warn(
                "WARNING: A non-empty group turns out to be empty of particles!"
            )
            return

        n_comp = len(group_subtrees)

        if n_comp == 1:
            post_prob, nonz = self._fit_single(gp_idx)
        elif gp_idx.size < 10 * n_comp:
            warnings.warn(
                f"WARNING: {n_comp} galaxies are unresolved inside a group!"
            )
            post_prob, nonz = self._fit_unresolved(
                gp_idx, group_candidates, group_boundness
            )
        else:
            post_prob, nonz = self._fit_resolved(
                gp_idx, group_subtrees, group_candidates, group_boundness
            )

        for i in range(len(group)):
            mask = nonz[:, i] > 0
            if np.any(mask):
                self.resp_map[group_subtrees[i]] = (
                    gp_idx[mask],
                    post_prob[mask, i],
                )

        assigned = group_subtrees[post_prob.argmax(axis=1)]
        self.particles_df.loc[gp_idx, "Sub_tree_id"] = assigned

    def _fit_single(self, gp_idx):
        post_prob = np.ones((gp_idx.size, 1))
        nonz = (post_prob > 0).astype(np.uint8)
        return post_prob, nonz

    def _fit_unresolved(self, gp_idx, group_candidates, group_boundness):
        csc = SparseCSC(group_candidates, group_boundness)
        post_prob = row_l1_normalize(
            csc.to_dense(col_func=_no_transform)
        ).astype(np.float32, copy=False)
        nonz = (post_prob > 0).astype(np.uint8)
        return post_prob, nonz

    def _fit_resolved(
        self, gp_idx, group_subtrees, group_candidates, group_boundness
    ):
        coords = self.particle_coords[gp_idx]
        mean_offset = np.mean(coords, axis=0)
        coords -= mean_offset
        scalings = 10.0 / (coords.max(axis=0) - coords.min(axis=0))
        coords *= scalings

        csc_boundness = (group_candidates, group_boundness)
        prior, w_init, m_init, c_init, cov_t = self._estimate_initial_params(
            coords, group_subtrees, gp_idx, csc_boundness
        )
        nonz = (prior > 0).astype(np.uint8)
        n_comp = len(group_subtrees)

        try:
            gmm = WeightedGaussianMixture(
                n_components=n_comp,
                means_init=m_init,
                covariance_init=c_init,
                weights_init=w_init,
                cov_type=cov_t,
                init_params="kmeans++",
                cast_dtype=np.float32,
                tol=self.tol,
                verbose=self.verbose,
            ).fit(coords, latent_prior=prior)
        except np.linalg.LinAlgError:
            gmm = WeightedGaussianMixture(
                n_components=n_comp,
                means_init=m_init,
                covariance_init=c_init,
                weights_init=w_init,
                cov_type=cov_t,
                init_params="kmeans++",
                cast_dtype=np.float64,
                tol=self.tol,
                verbose=self.verbose,
            ).fit(coords, latent_prior=prior)
            warnings.warn("Precision increased to float64.")

        log_prob = gmm.predict_log_proba(coords, latent_prior=prior)
        post_prob = np.exp(log_prob)

        scaled = {
            "means": gmm.means_,
            "weights": gmm.weights_,
            "covariances": gmm.covariances_,
            "cov_type": gmm.cov_type,
        }
        natural = self.get_parameters_natural(scaled, mean_offset, scalings)

        for sid in group_subtrees:
            if sid in natural["means"]:
                self.parameters[sid] = {
                    "mean": natural["means"][sid],
                    "weight": natural["weights"][sid],
                    "covariance": natural["covariances"][sid],
                }

        return post_prob, nonz

    def _estimate_initial_params(
        self, coords, subtrees, true_indices, csc_boundness
    ):
        n_samples = coords.shape[0]
        n_components = len(csc_boundness[0])

        csc = SparseCSC(csc_boundness[0], csc_boundness[1])
        prior = row_l1_normalize(
            csc.to_dense(col_func=_rank_transform)
        ).astype(np.float32, copy=False)

        if self.previous_resp:
            raw = _build_raw_resp_with_previous_and_bound(
                (n_samples, n_components),
                subtrees,
                self.newborn_indices,
                true_indices,
                csc_boundness,
                dict(self.previous_resp),
            )
            resp = stitch_zero_rows(prior, raw)
            row_sums = resp.sum(axis=1, keepdims=True)
            zero_rows = np.where(row_sums.flatten() == 0)[0]
            if zero_rows.size > 0:
                warnings.warn(f"{zero_rows.size} rows have zero sum.")
            row_sums[row_sums == 0.0] = 1.0
            resp /= row_sums
            if self.prior_type.lower() == "temporal-log-lik":
                prior = resp
        else:
            resp = prior

        nk, means_init, covs_init = _estimate_gaussian_parameters(
            coords,
            resp,
            np.ones(n_samples, dtype=np.float32),
            self.cov_type,
            reg_covar=self.reg_covar,
        )
        weights_init = nk / nk.sum()

        return prior, weights_init, means_init, covs_init, self.cov_type

    @staticmethod
    def get_parameters_natural(stored, mean_offset, scalings):
        inv_s = 1.0 / scalings
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

            means[sid] = mean_s * inv_s + mean_offset
            weights[sid] = float(w)
            covariances[sid] = cov_s * scaling_matrix

        return {"means": means, "weights": weights, "covariances": covariances}
