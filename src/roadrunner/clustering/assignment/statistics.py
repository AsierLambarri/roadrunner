from collections import defaultdict

import numpy as np


class GMMAssignerStatistics:
    def __init__(self):
        self.unassigned = 0
        self.fragments = 0
        self.avg_conf = float("nan")
        self.avg_entropy = float("nan")
        self.avg_cond = float("nan")

    def compute(self, particle_df, resp_map, fitted_parameters):
        N = len(particle_df)
        self.unassigned = int((particle_df["Sub_tree_id"] == -1).sum())

        counts = particle_df["Sub_tree_id"].value_counts()
        self.fragments = int(
            (counts[counts.index != -1] < 10).sum()
        ) if len(counts) > 0 else 0

        if not resp_map:
            return self

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

        conds = []
        for _, params in fitted_parameters.items():
            cov = params.get("covariance")
            if cov is not None and np.size(cov) > 1:
                c = np.asarray(cov, dtype=np.float64)
                vals = np.linalg.eigvalsh(c) if c.ndim == 2 else c
                conds.append(float(vals.max() / max(vals.min(), 1e-30)))
        self.avg_cond = float(np.mean(conds)) if conds else float("nan")

        return self

    @property
    def values(self):
        return {
            "unassigned": self.unassigned,
            "fragments": self.fragments,
            "avg_conf": self.avg_conf,
            "avg_entropy": self.avg_entropy,
            "avg_cond": self.avg_cond,
        }
