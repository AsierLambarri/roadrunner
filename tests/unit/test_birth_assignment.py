import numpy as np

from roadrunner.physics.birth_assignment import (
    assign_birth_snapshots,
    build_birth_dict,
    sanitize_birth_dict,
)


class TestAssignBirthSnapshots:
    def test_particle_at_exact_snapshot_time(self):
        result = assign_birth_snapshots(
            np.array([2.0]),
            np.array([1.0, 2.0, 5.0]),
            np.array([10, 20, 50]),
        )
        assert result[0] == 50

    def test_multiple_particles(self):
        result = assign_birth_snapshots(
            np.array([0.5, 3.0, 10.0]),
            np.array([1.0, 2.0, 5.0]),
            np.array([10, 20, 50]),
        )
        assert list(result) == [10, 50, 50]

    def test_empty_input(self):
        result = assign_birth_snapshots(
            np.array([]),
            np.array([1.0, 2.0, 5.0]),
            np.array([10, 20, 50]),
        )
        assert len(result) == 0

    def test_empty_snapshots_returns_empty(self):
        result = assign_birth_snapshots(
            np.array([1.0, 2.0]),
            np.array([]),
            np.array([]),
        )
        assert len(result) == 0


class TestBuildBirthDict:
    def test_basic_grouping(self):
        result = build_birth_dict(
            np.array([100, 101, 102, 103]),
            np.array([0, 0, 1, 1]),
        )
        assert list(result[0]) == [100, 101]
        assert list(result[1]) == [102, 103]
        assert result[0].dtype == np.uint64

    def test_empty_input(self):
        result = build_birth_dict(np.array([]), np.array([]))
        assert result == {}


class TestSanitizeBirthDict:
    def test_no_duplicates_unchanged(self):
        d = {0: np.array([1, 2]), 1: np.array([3])}
        result = sanitize_birth_dict(d)
        assert list(result[0]) == [1, 2]

    def test_duplicates_removed(self):
        d = {0: np.array([1]), 1: np.array([1, 2])}
        result = sanitize_birth_dict(d)
        assert list(result[0]) == [1]
        assert list(result[1]) == [2]
