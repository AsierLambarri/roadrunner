import numpy as np

from roadrunner.clustering.union_find import UnionFind, overlapping_groups


class TestUnionFind:
    def test_union_same_root_noop(self):
        uf = UnionFind(2)
        uf.union(0, 1)
        root = uf.find(0)
        uf.union(0, 1)
        assert uf.find(0) == root

    def test_groups_no_connections(self):
        uf = UnionFind(3)
        result = uf.groups()
        assert len(result) == 3
        assert all(len(g) == 1 for g in result)

    def test_groups_all_connected(self):
        uf = UnionFind(3)
        uf.union(0, 1)
        uf.union(1, 2)
        result = uf.groups()
        assert len(result) == 1
        assert sorted(result[0]) == [0, 1, 2]

    def test_zero_elements(self):
        uf = UnionFind(0)
        assert uf.groups() == []


class TestOverlappingGroups:
    def test_no_overlap(self):
        pos = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        r = np.array([1.0, 1.0])
        result = overlapping_groups(pos, r)
        assert len(result) == 2

    def test_transitive_chain(self):
        pos = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.0, 0.0, 0.0]])
        r = np.array([1.0, 1.0, 1.0])
        result = overlapping_groups(pos, r)
        assert len(result) == 1
        assert sorted(result[0]) == [0, 1, 2]

    def test_two_separate_groups(self):
        pos = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [11.0, 0.0, 0.0],
        ])
        r = np.array([1.0, 1.0, 1.0, 1.0])
        result = overlapping_groups(pos, r)
        assert len(result) == 2
        sizes = sorted(len(g) for g in result)
        assert sizes == [2, 2]

    def test_custom_criterion(self):
        def criterion(p1, r1, p2, r2):
            d = np.linalg.norm(p1 - p2)
            return d <= r1 + r2 + 0.5

        pos = np.array([[0.0, 0.0, 0.0], [2.4, 0.0, 0.0]])
        r = np.array([1.0, 1.0])
        result = overlapping_groups(pos, r, linking_length_func=criterion)
        assert len(result) == 1

    def test_all_singletons_with_custom(self):
        def never(p1, r1, p2, r2):
            return False

        pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        r = np.array([1.0, 1.0])
        result = overlapping_groups(pos, r, linking_length_func=never)
        assert len(result) == 2

    def test_empty_input(self):
        result = overlapping_groups(np.array([]), np.array([]))
        assert result == []

    def test_single_point(self):
        pos = np.array([[0.0, 0.0, 0.0]])
        r = np.array([1.0])
        result = overlapping_groups(pos, r)
        assert len(result) == 1
        assert result[0] == [0]
