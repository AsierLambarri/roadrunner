from collections import defaultdict

import numpy as np


class GMMAssignerStatistics:
    def __init__(self):
        self.unassigned = 0
        self.fragments = 0
        self.avg_conf = float("nan")
        self.avg_entropy = float("nan")
        self.avg_cond = float("nan")
        self.avg_retention = float("nan")

    def compute(self, particle_df, resp_map, fitted_parameters, boundness_csc=None):
        N = len(particle_df)
        self.unassigned = int((particle_df["Sub_tree_id"] == -1).sum())

        counts = particle_df["Sub_tree_id"].value_counts()
        self.fragments = int(
            (counts[counts.index != -1] < 10).sum()
        ) if len(counts) > 0 else 0

        if not resp_map:
            return self

        # ── Soft metrics (avg_conf, avg_entropy) ──────────────────
        per_particle_max = defaultdict(float)
        per_particle_resps = defaultdict(list)
        for _, (indices, vals) in resp_map.items():
            for pid, v in zip(indices, vals):
                if v > per_particle_max[pid]:
                    per_particle_max[pid] = v
                per_particle_resps[pid].append(v)

        self.avg_conf = float(np.mean(list(per_particle_max.values())))

        entropies = []
        for _, rvals in per_particle_resps.items():
            K = len(rvals)
            if K <= 1:
                entropies.append(0.0)
            else:
                arr = np.array(rvals, dtype=np.float64)
                arr /= arr.sum()
                H = -np.sum(arr * np.log(np.maximum(arr, 1e-30))) / np.log(K)
                entropies.append(H)
        self.avg_entropy = float(np.mean(entropies))

        # ── Condition number (from scaled covariances) ────────────
        conds = []
        for _, params in fitted_parameters.items():
            cond = params.get("covariance_condition")
            if cond is not None and np.isfinite(cond):
                conds.append(float(cond))
        self.avg_cond = float(np.mean(conds)) if conds else float("nan")

        # ── Avg retention (normalised bound-to-tagged overlap) ────
        if boundness_csc is not None:
            retentions = []
            sid_to_idx = {
                sid: i for i, sid in enumerate(boundness_csc.column_id)
            }
            for gid in resp_map:
                col = sid_to_idx.get(gid)
                if col is None:
                    continue
                bound_idx = boundness_csc.column_indices[col]
                n_bound = bound_idx.size
                if n_bound == 0:
                    continue

                tagged = particle_df.loc[
                    particle_df["Sub_tree_id"] == gid, "array_index"
                ].values
                n_tagged = len(tagged)
                if n_tagged == 0:
                    continue

                overlap = np.intersect1d(bound_idx, tagged)
                n_overlap = len(overlap)
                retention_obs = n_overlap / n_bound
                expected_frac = n_tagged / max(N, 1)
                retentions.append(retention_obs / max(expected_frac, 1e-30))

            self.avg_retention = float(np.mean(retentions)) if retentions else float("nan")

        return self

    @property
    def values(self):
        return {
            "unassigned": self.unassigned,
            "fragments": self.fragments,
            "avg_conf": self.avg_conf,
            "avg_entropy": self.avg_entropy,
            "avg_cond": self.avg_cond,
            "avg_retention": self.avg_retention,
        }
