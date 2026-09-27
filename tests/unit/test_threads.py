import numba
import numpy as np
import pytest
from threadpoolctl import threadpool_info

from roadrunner.clustering.union_find import overlapping_groups
from roadrunner.mixture._math import row_l1_normalize
from roadrunner.pipeline.config import RunConfig
from roadrunner.threads import _available_cpus, thread_budget, threads, tree_workers


def _blas_threads():
    return [lib["num_threads"] for lib in threadpool_info() if lib["user_api"] == "blas"]


class TestThreadScope:
    def test_none_is_a_no_op(self):
        before = numba.get_num_threads()
        with threads(None):
            assert numba.get_num_threads() == before
            assert thread_budget() is None
            assert tree_workers() == -1
        assert tree_workers() == -1

    def test_scope_sets_every_backend_and_restores_it(self):
        before, blas_before = numba.get_num_threads(), _blas_threads()
        with threads(2):
            assert numba.get_num_threads() == min(2, numba.config.NUMBA_NUM_THREADS)
            assert thread_budget() == 2
            assert tree_workers() == 2
            assert all(n <= 2 for n in _blas_threads())
        assert numba.get_num_threads() == before
        assert _blas_threads() == blas_before
        assert thread_budget() is None and tree_workers() == -1

    def test_restored_after_an_exception(self):
        before = numba.get_num_threads()
        with pytest.raises(RuntimeError):
            with threads(1):
                raise RuntimeError("boom")
        assert numba.get_num_threads() == before
        assert thread_budget() is None

    def test_nested_scopes(self):
        with threads(2):
            with threads(1):
                assert tree_workers() == 1 and numba.get_num_threads() == 1
            assert tree_workers() == 2

    def test_capped_at_available_cpus(self):
        with threads(10**6):
            assert thread_budget() == _available_cpus()

    def test_rejects_non_positive(self):
        with pytest.raises(ValueError, match="positive"):
            with threads(0):
                pass


class TestResultsDoNotDependOnTheBudget:
    def test_numba_kernel(self):
        x = np.random.default_rng(0).random((5000, 4))
        ref = row_l1_normalize(x.copy())
        with threads(1):
            np.testing.assert_array_equal(row_l1_normalize(x.copy()), ref)

    def test_kdtree_call_site(self):
        rng = np.random.default_rng(1)
        pos, radii = rng.uniform(0, 100, (300, 3)), rng.uniform(1, 6, 300)
        ref = overlapping_groups(pos, radii)
        with threads(1):
            assert overlapping_groups(pos, radii) == ref


class TestRunConfigThreads:
    def test_default_is_none(self):
        assert RunConfig().threads is None

    def test_coerced_to_int(self):
        assert RunConfig(threads="4").threads == 4

    def test_rejects_non_positive(self):
        with pytest.raises(ValueError, match="threads"):
            RunConfig(threads=0)
