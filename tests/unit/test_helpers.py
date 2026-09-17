import numpy as np
import pytest

from roadrunner.helpers import (
    check_particle_uniqueness,
    remove_duplicates,
    select_float_dtype,
    select_uint_dtype,
)


class TestSelectUintDtype:
    def test_positive_returns_uint_dtype(self):
        result = select_uint_dtype(100)
        assert np.issubdtype(result, np.unsignedinteger)

    def test_small_value_uint16(self):
        assert select_uint_dtype(10) == np.uint16

    def test_medium_value_uint32(self):
        assert select_uint_dtype(100_000) == np.uint32

    def test_large_value_uint64(self):
        assert select_uint_dtype(10_000_000_000) == np.uint64

    def test_negative_value_raises(self):
        with pytest.raises(ValueError):
            select_uint_dtype(-100)

    def test_zero(self):
        assert select_uint_dtype(0) is not None


class TestSelectFloatDtype:
    def test_very_small_value_float16(self):
        result = select_float_dtype(1.0, abs_tol=1e-3)
        assert result == np.float16

    def test_small_value_float32(self):
        result = select_float_dtype(100.0, abs_tol=1e-4)
        assert result == np.float32

    def test_tight_tolerance_float64(self):
        result = select_float_dtype(1e10, abs_tol=1e-4)
        assert result == np.float64

    def test_negative_max_value(self):
        result = select_float_dtype(-1.0, abs_tol=0.1)
        assert np.issubdtype(result, np.floating)


class TestCheckParticleUniqueness:
    def test_unique_returns_true(self):
        data = {0: [1, 2, 3], 1: [4, 5]}
        assert check_particle_uniqueness(data) is True

    def test_duplicate_returns_false(self):
        data = {0: [1, 2], 1: [2, 3]}
        assert check_particle_uniqueness(data) is False

    def test_empty_dict(self):
        assert check_particle_uniqueness({}) is True

    def test_single_element(self):
        data = {0: [1]}
        assert check_particle_uniqueness(data) is True


class TestRemoveDuplicates:
    def test_no_duplicates_unchanged(self):
        data = {0: [1, 2], 1: [3, 4]}
        result = remove_duplicates(data)
        assert list(result[0]) == [1, 2]
        assert list(result[1]) == [3, 4]

    def test_duplicates_removed(self):
        data = {0: [1, 2], 1: [2, 3]}
        result = remove_duplicates(data)
        assert list(result[0]) == [1, 2]
        assert list(result[1]) == [3]

    def test_preserves_first_occurrence(self):
        data = {0: [1], 1: [1, 2]}
        result = remove_duplicates(data)
        assert list(result[0]) == [1]
        assert list(result[1]) == [2]

    def test_all_duplicates(self):
        data = {0: [1], 1: [1]}
        result = remove_duplicates(data)
        assert list(result[0]) == [1]
        assert list(result[1]) == []

    def test_empty_dict(self):
        assert remove_duplicates({}) == {}

    def test_returns_numpy_arrays(self):
        data = {0: [1, 2]}
        result = remove_duplicates(data)
        assert isinstance(result[0], np.ndarray)

    def test_empty_value_list(self):
        data = {0: []}
        result = remove_duplicates(data)
        assert isinstance(result[0], np.ndarray)
        assert len(result[0]) == 0
