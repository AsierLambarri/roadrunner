#############################################################################
#
# package:   roadrunner.clustering
# file:      segmentation.py
# brief:     Galaxy group segmentation based on overlapping virial radii.
#
# copyright: GPLv3
# author:    Asier Lambarri Martinez
# changes:   13 May 2026 - Created
#
#############################################################################

"""Halo group segmentation based on overlapping virial radii."""

import numpy as np

from roadrunner._defaults import LOCAL_IDX

from roadrunner.clustering.union_find import overlapping_groups


def find_populated_haloes(candidates_list):
    """Separate halo indices into empty and populated sets.

    Parameters
    ----------
    candidates_list : list of ndarray
        Per-halo list of candidate particle indices.

    Returns
    -------
    empty : ndarray of int64
        Indices of halos with zero candidates.
    populated : ndarray of int64
        Indices of halos with at least one candidate.
    """
    empty = [i for i, c in enumerate(candidates_list) if len(c) == 0]
    populated = [i for i, c in enumerate(candidates_list) if len(c) > 0]
    return np.array(empty, dtype=LOCAL_IDX), np.array(populated, dtype=LOCAL_IDX)


class HaloSegmenter:
    """Segment halos into overlapping groups based on virial radii.

    Two halos are considered to overlap when the distance between
    their centres is smaller than the sum of their virial radii.

    Parameters
    ----------
    positions : ndarray of shape (n_halos, 3)
        Halo centre positions.
    virial_radii : ndarray of shape (n_halos,)
        Virial radii of each halo.
    """

    def __init__(self, positions, virial_radii):
        self.positions = np.asarray(positions)
        self.rvirs = np.asarray(virial_radii)
        self.groups = None
        self.pruned_groups = None
        self._partition = None

    def overlap_groups(self):
        """Compute overlapping groups using the union-find algorithm.

        Populates ``self.groups`` with a list of lists, where each
        inner list contains the indices of overlapping halos.

        Returns
        -------
        self : HaloSegmenter
            The segmenter with computed groups.
        """
        if len(self.positions) == 0:
            self.groups = []
            return self
        self.groups = overlapping_groups(self.positions, self.rvirs)
        return self

    def prune(self, candidates_list, min_particles=0, discard=False):
        """Prune groups based on a minimum particle threshold.

        Halos with fewer than ``min_particles`` candidates are either
        placed in their own singleton group (``discard=False``) or
        removed entirely (``discard=True``).

        Parameters
        ----------
        candidates_list : list of ndarray
            Per-halo candidate particle indices.
        min_particles : int, default=0
            Minimum number of candidates for a halo to remain in
            its overlapping group.
        discard : bool, default=False
            If ``True``, underpopulated halos are discarded instead
            of being placed in singleton groups.

        Returns
        -------
        self : HaloSegmenter
            The segmenter with pruned groups in ``self.pruned_groups``.
        """
        if self.groups is None:
            raise RuntimeError(
                "Groups have not been computed. Call `overlap_groups()` first."
            )
        pruned = []
        partition = []
        for group in self.groups:
            large = [h for h in group if len(candidates_list[h]) > min_particles]
            small = [h for h in group if len(candidates_list[h]) <= min_particles]
            partition.append((large, small))
            if large:
                pruned.append(large)
            if small and not discard:
                for h in small:
                    pruned.append([h])
        self.pruned_groups = pruned
        self._partition = partition
        return self

    def resolve_ownership(self, candidates_list, values_list, halo_ids):
        """Resolve which halo owns each contested particle, before any fit.

        Two rules are applied per original overlap group, using the
        same large/small split ``prune()`` already computed (classified
        once -- never recomputed from post-exclusion counts):

        - Among "small" halos (``<= min_particles``), a particle
          claimed by more than one goes to whichever has the higher
          boundness value for it (already normalized to that halo's
          own virial scale, so directly comparable across halos);
          exact ties go to the smaller ``Sub_tree_id``.
        - Any "large" halo in the same group loses every particle
          claimed by any small halo, unconditionally -- small halos
          always take precedence over the group they were split out
          of (C05).

        Must be called after `prune()`.

        Parameters
        ----------
        candidates_list : list of ndarray
            Per-halo (local index) particle-ID arrays -- the same
            list passed to `prune()`.
        values_list : list of ndarray
            Per-halo boundness values, parallel to ``candidates_list``.
        halo_ids : ndarray
            Per-halo (local index) identifier, used only to break
            exact boundness ties (smaller id wins).

        Returns
        -------
        removals : dict of {int: ndarray}
            Local halo index -> particle-ID rows that should be
            removed from that halo's own boundness data (apply via
            :meth:`HaloModel.set_boundness`). Halos with nothing to
            remove are absent from the dict.
        """
        if self._partition is None:
            raise RuntimeError("Call `prune()` before `resolve_ownership()`.")

        removals: dict = {}
        for large, small in self._partition:
            if not small:
                continue

            if len(small) == 1:
                # No contest possible among a single small halo.
                pass
            else:
                # Vectorized max-boundness-wins, smaller-id-breaks-ties:
                # flatten every small halo's own (pid, value, halo id,
                # local index) rows, then sort so that within each pid's
                # rows the winner sorts first (value descending, id
                # ascending), and read off the first row per pid group.
                pids = np.concatenate([candidates_list[h] for h in small])
                vals = np.concatenate([values_list[h] for h in small])
                hids = np.concatenate([
                    np.full(len(candidates_list[h]), halo_ids[h]) for h in small
                ])
                h_idx = np.concatenate([
                    np.full(len(candidates_list[h]), h) for h in small
                ])

                order = np.lexsort((hids, -vals, pids))
                pids_s, h_idx_s = pids[order], h_idx[order]

                unique_pids, first_pos = np.unique(pids_s, return_index=True)
                winner_h = h_idx_s[first_pos]
                group_of_row = np.searchsorted(unique_pids, pids_s)
                lose_mask = h_idx_s != winner_h[group_of_row]

                if np.any(lose_mask):
                    losing_pids, losing_h = pids_s[lose_mask], h_idx_s[lose_mask]
                    for h in np.unique(losing_h):
                        removals.setdefault(h, []).append(losing_pids[losing_h == h])

            if large:
                # One combined isin() across every large halo in the group,
                # rather than one small call per halo -- with many large
                # halos sharing a group, per-call numpy overhead otherwise
                # dominates over the actual comparison work.
                reserved = np.unique(np.concatenate([candidates_list[h] for h in small]))
                large_pids = np.concatenate([candidates_list[h] for h in large])
                large_h_idx = np.concatenate([
                    np.full(len(candidates_list[h]), h) for h in large
                ])
                mask = np.isin(large_pids, reserved)
                if np.any(mask):
                    losing_pids, losing_h = large_pids[mask], large_h_idx[mask]
                    for h in np.unique(losing_h):
                        removals.setdefault(h, []).append(losing_pids[losing_h == h])

        return {h: np.unique(np.concatenate(v)) for h, v in removals.items()}
