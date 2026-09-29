#############################################################################
#
# package:   roadrunner.clustering.assignment
# file:      statistics.py
# brief:     Summary statistics tracker for the XGMM assigner.
#
# copyright: GPLv3
# author:    Asier Lambarri Martinez
# changes:   19 May 2026 - Created
#            16 Jun 2026 - Last edit
#
#############################################################################

"""Assignment statistics tracker for the XGMM assigner.

The :class:`GMMAssignerStatistics` class collects per-snapshot metrics
such as the number of unassigned particles, average confusion and
entropy of responsibilities, and the average condition number of
fitted covariances.
"""

import numpy as np

from roadrunner._defaults import FRAGMENT_THRESHOLD, UNBOUND

BAD_CONDITION = 1e6   # covariance condition number flagged as degenerate


class GMMAssignerStatistics:
    """Tracks per-snapshot assignment statistics (fragments, confusion, entropy, condition).

    Statistics are computed from the particle DataFrame, the
    responsibility map, and the fitted parameters after each
    :meth:`XGMMAssigner.assign` call.

    Attributes
    ----------
    unassigned : int
        Number of particles with ``Sub_tree_id == -1``.
    fragments : int
        Number of components with fewer than ``FRAGMENT_THRESHOLD`` particles.
    avg_conf : float
        Mean maximum responsibility over contested particles (those with
        responsibility for more than one component); NaN if none.
    avg_entropy : float
        Mean entropy, normalised by ``log K``, over contested particles.
    avg_cond : float
        Median over components of ``log10`` of the covariance condition
        number.
    bad_cond : int
        Components with condition number above ``BAD_CONDITION`` or a
        non-finite one.
    avg_retention : float
        Median over galaxies of the fraction of their bound candidates
        that the fit assigns to them.
    """

    def __init__(self):
        self.unassigned = 0
        self.fragments = 0
        self.avg_conf = float("nan")
        self.avg_entropy = float("nan")
        self.avg_cond = float("nan")
        self.bad_cond = 0
        self.avg_retention = float("nan")

    def compute(self, particle_df, resp_map, fitted_parameters, boundness_csc=None):
        """Compute all statistics from the assignment results.

        Parameters
        ----------
        particle_df : DataFrame
            Particle-level assignment data with ``Sub_tree_id`` and ``array_index`` columns.
        resp_map : dict of (indices, values)
            Responsibility map keyed by Sub_tree_id.
        fitted_parameters : dict
            Per-component fitted parameters with ``covariance_condition`` entries.
        boundness_csc : SparseCSC, optional
            Boundness matrix for computing retention statistics.

        Returns
        -------
        self : GMMAssignerStatistics
            The statistics object with updated attributes.
        """
        N = len(particle_df)
        self.unassigned = int((particle_df["Sub_tree_id"] == UNBOUND).sum())

        counts = particle_df["Sub_tree_id"].value_counts()
        self.fragments = int(
            (counts[counts.index != UNBOUND] < FRAGMENT_THRESHOLD).sum()
        ) if len(counts) > 0 else 0

        if not resp_map:
            return self

        # ── Soft metrics (avg_conf, avg_entropy) ──────────────────
        # Vectorized over all (particle, component) responsibility
        # entries at once via scatter-reductions, instead of a Python
        # loop per entry -- see audit finding P02.
        all_pids = np.concatenate([np.asarray(indices) for indices, _ in resp_map.values()])
        all_vals = np.concatenate([np.asarray(vals) for _, vals in resp_map.values()])

        per_particle_max = np.full(N, -np.inf)
        np.maximum.at(per_particle_max, all_pids, all_vals)

        counts = np.zeros(N, dtype=np.int64)
        np.add.at(counts, all_pids, 1)
        # Only contested particles (responsibility for more than one
        # component) tell how decisively the fit separates halos.
        touched = counts > 1
        self.avg_conf = float(np.mean(per_particle_max[touched])) if touched.any() else float("nan")
        sums = np.zeros(N, dtype=np.float64)
        np.add.at(sums, all_pids, all_vals)
        log_vals = np.log(np.maximum(all_vals, 1e-30))
        weighted_log_sums = np.zeros(N, dtype=np.float64)
        np.add.at(weighted_log_sums, all_pids, all_vals * log_vals)

        K = counts[touched]
        S = sums[touched]
        T = weighted_log_sums[touched]
        with np.errstate(divide="ignore", invalid="ignore"):
            H = -(T / S - np.log(S)) / np.log(K)
        self.avg_entropy = float(np.mean(H)) if H.size else float("nan")

        # ── Condition number (from scaled covariances) ────────────
        conds = np.array([float(p["covariance_condition"]) for p in fitted_parameters.values()
                          if p.get("covariance_condition") is not None])
        finite = conds[np.isfinite(conds)]
        self.avg_cond = float(np.median(np.log10(np.maximum(finite, 1.0)))) if finite.size else float("nan")
        self.bad_cond = int((~np.isfinite(conds)).sum() + (finite > BAD_CONDITION).sum())

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
                retentions.append(len(overlap) / n_bound)

            self.avg_retention = float(np.median(retentions)) if retentions else float("nan")

        return self

    @property
    def values(self):
        """Return all statistics as a flat dictionary.

        Returns
        -------
        stats : dict of str → float
            Keys: ``unassigned``, ``fragments``, ``avg_conf``,
            ``avg_entropy``, ``avg_cond``, ``bad_cond``, ``avg_retention``.
        """
        return {
            "unassigned": self.unassigned,
            "fragments": self.fragments,
            "avg_conf": self.avg_conf,
            "avg_entropy": self.avg_entropy,
            "avg_cond": self.avg_cond,
            "bad_cond": self.bad_cond,
            "avg_retention": self.avg_retention,
        }
