import numpy as np

from roadrunner.clustering.segmentation import HaloSegmenter, find_populated_haloes


class TestFindPopulatedHaloes:
    def test_mixed(self):
        empty, populated = find_populated_haloes([[1, 2], [], [3]])
        assert list(empty) == [1]
        assert list(populated) == [0, 2]

    def test_all_empty(self):
        empty, populated = find_populated_haloes([[], []])
        assert list(empty) == [0, 1]
        assert list(populated) == []

    def test_all_populated(self):
        empty, populated = find_populated_haloes([[1], [2]])
        assert list(empty) == []
        assert list(populated) == [0, 1]

    def test_empty_input(self):
        empty, populated = find_populated_haloes([])
        assert list(empty) == []
        assert list(populated) == []


class TestHaloSegmenter:
    def test_overlap_groups_chain(self):
        pos = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.0, 0.0, 0.0]])
        r = np.array([1.0, 1.0, 1.0])
        seg = HaloSegmenter(pos, r).overlap_groups()
        assert seg is seg  # returns self
        assert sorted(seg.groups[0]) == [0, 1, 2]

    def test_overlap_groups_none(self):
        pos = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        r = np.array([1.0, 1.0])
        seg = HaloSegmenter(pos, r).overlap_groups()
        assert len(seg.groups) == 2

    def test_overlap_groups_empty(self):
        seg = HaloSegmenter([], []).overlap_groups()
        assert seg.groups == []

    def test_fluent_chain(self):
        pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
        r = np.array([1.0, 1.0, 1.0])
        candidates = [[1] * 100, [1] * 3, [1] * 50]
        seg = HaloSegmenter(pos, r).overlap_groups().prune(candidates, min_particles=10)
        assert seg is seg
        assert seg.pruned_groups is not None

    def test_prune_below_threshold(self):
        # Chain overlap: 0-1 and 1-2 overlap → one group [0,1,2]
        pos = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.0, 0.0, 0.0]])
        r = np.array([1.0, 1.0, 1.0])
        candidates = [[1] * 100, [1] * 3, [1] * 50]
        seg = HaloSegmenter(pos, r).overlap_groups().prune(candidates, min_particles=10)
        assert [0, 2] in seg.pruned_groups
        assert [1] in seg.pruned_groups

    def test_prune_discard_true(self):
        # Chain overlap: 0-1 and 1-2 overlap → one group [0,1,2]
        pos = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.0, 0.0, 0.0]])
        r = np.array([1.0, 1.0, 1.0])
        candidates = [[1] * 100, [1] * 3, [1] * 50]
        seg = HaloSegmenter(pos, r).overlap_groups().prune(
            candidates, min_particles=10, discard=True
        )
        assert [0, 2] in seg.pruned_groups
        assert [1] not in seg.pruned_groups

    def test_prune_before_overlap_raises(self):
        seg = HaloSegmenter(np.array([[0.0, 0.0, 0.0]]), np.array([1.0]))
        import pytest
        with pytest.raises(RuntimeError, match="Groups have not been computed"):
            seg.prune([[1]], min_particles=1)
