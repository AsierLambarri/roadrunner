import numpy as np

from roadrunner.clustering.segmentation import HaloSegmenter, find_populated_haloes


class TestFindPopulatedHaloes:
    def test_mixed(self):
        empty, populated = find_populated_haloes([[1, 2], [], [3]])
        assert list(empty) == [1]
        assert list(populated) == [0, 2]

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

    def test_overlap_groups_empty(self):
        seg = HaloSegmenter([], []).overlap_groups()
        assert seg.groups == []

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


class TestResolveOwnership:
    def _seg_pair(self, min_particles=5):
        # Two overlapping halos, single union-find group [0, 1].
        pos = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
        r = np.array([1.0, 1.0])
        return HaloSegmenter(pos, r).overlap_groups()

    def test_before_prune_raises(self):
        import pytest
        seg = self._seg_pair()
        with pytest.raises(RuntimeError, match="Call `prune\\(\\)`"):
            seg.resolve_ownership([np.array([1]), np.array([1])],
                                   [np.array([0.1]), np.array([0.1])],
                                   np.array([1, 2]))

    def test_large_always_wins_over_small_regardless_of_boundness(self):
        # Halo 0 "large" (12 candidates), halo 1 "small" (3 candidates),
        # sharing particle 5. Halo 0's boundness for particle 5 is much
        # higher than halo 1's -- large must still lose it.
        candidates = [np.arange(12, dtype=np.int64), np.array([5, 20, 21], dtype=np.int64)]
        values = [np.concatenate([np.full(11, 0.1), [99.0]]), np.array([0.2, 0.3, 0.4])]
        halo_ids = np.array([1, 2])
        seg = self._seg_pair(min_particles=5)
        seg.prune(candidates, min_particles=5, discard=False)
        removals = seg.resolve_ownership(candidates, values, halo_ids)
        assert 0 in removals
        np.testing.assert_array_equal(removals[0], [5])
        assert 1 not in removals

    def test_small_vs_small_resolved_by_boundness(self):
        candidates = [np.array([1, 2, 3], dtype=np.int64), np.array([3, 4, 5], dtype=np.int64)]
        values = [np.array([0.1, 0.2, 0.9]), np.array([0.5, 0.6, 0.1])]
        halo_ids = np.array([10, 20])
        seg = self._seg_pair(min_particles=5)
        seg.prune(candidates, min_particles=5, discard=False)
        removals = seg.resolve_ownership(candidates, values, halo_ids)
        # Particle 3: halo0 has 0.9, halo1 has 0.5 -> halo0 wins, halo1 loses it.
        assert 1 in removals
        np.testing.assert_array_equal(removals[1], [3])
        assert 0 not in removals

    def test_small_vs_small_exact_tie_smaller_id_wins(self):
        candidates = [np.array([7], dtype=np.int64), np.array([7], dtype=np.int64)]
        values = [np.array([0.5]), np.array([0.5])]
        halo_ids = np.array([9, 3])  # halo index 1 has the smaller Sub_tree_id (3)
        seg = self._seg_pair(min_particles=5)
        seg.prune(candidates, min_particles=5, discard=False)
        removals = seg.resolve_ownership(candidates, values, halo_ids)
        assert 0 in removals
        np.testing.assert_array_equal(removals[0], [7])
        assert 1 not in removals

    def test_no_contested_rows_gives_empty_removals(self):
        candidates = [np.array([1, 2, 3], dtype=np.int64), np.array([4, 5], dtype=np.int64)]
        values = [np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5])]
        halo_ids = np.array([1, 2])
        seg = self._seg_pair(min_particles=5)
        seg.prune(candidates, min_particles=5, discard=False)
        removals = seg.resolve_ownership(candidates, values, halo_ids)
        assert removals == {}
