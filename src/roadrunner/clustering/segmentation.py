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
    return np.array(empty, dtype=np.int64), np.array(populated, dtype=np.int64)


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
        for group in self.groups:
            large = [h for h in group if len(candidates_list[h]) > min_particles]
            small = [h for h in group if len(candidates_list[h]) <= min_particles]
            if large:
                pruned.append(large)
            if small and not discard:
                for h in small:
                    pruned.append([h])
        self.pruned_groups = pruned
        return self
