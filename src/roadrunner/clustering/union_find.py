#############################################################################
#
# package:   roadrunner.clustering
# file:      union_find.py
# brief:     Union-Find (disjoint set) data structure for group merging.
#
# copyright: GPLv3
# author:    Asier Lambarri Martinez
# changes:   13 May 2026 - Created
#
#############################################################################

"""Union-Find (disjoint set) data structure.

Used by the halo segmenter to merge overlapping groups efficiently.
"""

from collections import defaultdict

import numpy as np
from scipy.spatial import KDTree


class UnionFind:
    """Union-Find (disjoint set) data structure with path compression.

    Parameters
    ----------
    n : int
        Number of elements.
    """

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        """Find the root of element ``x`` with path compression.

        Parameters
        ----------
        x : int
            Element index.

        Returns
        -------
        root : int
            Root element index.
        """
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        """Merge the sets containing ``x`` and ``y``.

        Uses union-by-rank to keep the tree shallow.

        Parameters
        ----------
        x : int
            First element.
        y : int
            Second element.
        """
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            self.parent[rx] = ry
        elif self.rank[ry] < self.rank[rx]:
            self.parent[ry] = rx
        else:
            self.parent[ry] = rx
            self.rank[rx] += 1

    def groups(self) -> list[list[int]]:
        """Return all connected component groups.

        Returns
        -------
        groups : list of list of int
            Each element is a list of indices belonging to one group.
        """
        groups_dict: dict[int, list[int]] = defaultdict(list)
        for i in range(len(self.parent)):
            root = self.find(i)
            groups_dict[root].append(i)
        return list(groups_dict.values())


def overlapping_groups(positions, radii, linking_length_func=None):
    """Group overlapping spheres by a linking-length criterion.

    Two spheres overlap if the centre-to-centre distance is less than
    the sum of their radii.  A custom ``linking_length_func`` can be
    provided for alternative criteria.

    Parameters
    ----------
    positions : ndarray of shape (n, 3)
    radii : ndarray of shape (n,)
    linking_length_func : callable or None, optional
        If ``None`` the default ``dist ≤ r_i + r_j`` criterion is used.

    Returns
    -------
    groups : list of list of int
        Each inner list contains the indices of one connected component.
    """
    n = len(positions)
    if n == 0:
        return []

    if linking_length_func is None:
        # KDTree candidate search + exact filter, instead of a dense
        # O(H^2) pairwise matrix (see audit finding P05). Querying with
        # r = radii + radii.max() cannot miss a true overlap (the real
        # criterion dist <= r_i + r_j <= r_i + max_r), so it's safe to
        # then re-check the exact criterion on candidates only.
        tree = KDTree(positions)
        max_r = radii.max()
        neighbor_lists = tree.query_ball_point(positions, r=radii + max_r, workers=-1)
        links = []
        for i, neighbors in enumerate(neighbor_lists):
            for j in neighbors:
                if j <= i:
                    continue
                d2 = np.sum((positions[i] - positions[j]) ** 2)
                if d2 <= (radii[i] + radii[j]) ** 2:
                    links.append((i, j))
    else:
        links = [
            (i, j)
            for i in range(n)
            for j in range(i + 1, n)
            if linking_length_func(positions[i], radii[i], positions[j], radii[j])
        ]

    uf = UnionFind(n)
    for i, j in links:
        uf.union(i, j)
    return uf.groups()
