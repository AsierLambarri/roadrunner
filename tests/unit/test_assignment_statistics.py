import numpy as np
import pandas as pd

from roadrunner._mcf_types import AssignmentStatistics
from roadrunner.clustering.assignment.statistics import GMMAssignerStatistics


class TestGMMAssignerStatistics:
    def test_protocol_conformance(self):
        assert isinstance(GMMAssignerStatistics(), AssignmentStatistics)

    def test_initial_state(self):
        s = GMMAssignerStatistics()
        assert s.unassigned == 0
        assert s.fragments == 0
        assert np.isnan(s.avg_conf)
        assert np.isnan(s.avg_entropy)
        assert np.isnan(s.avg_cond)

    def test_unassigned_and_fragments(self):
        df = pd.DataFrame({
            "array_index": np.arange(110),
            "Sub_tree_id": [1] * 100 + [-1] * 10,
        })
        s = GMMAssignerStatistics().compute(df, {}, {})
        assert s.unassigned == 10
        # Sub_tree_id 1 has 100 particles → not a fragment
        assert s.fragments == 0

    def test_fragments_detected(self):
        df = pd.DataFrame({
            "array_index": np.arange(15),
            "Sub_tree_id": [1] * 10 + [2] * 3 + [3] * 2,
        })
        s = GMMAssignerStatistics().compute(df, {}, {})
        assert s.fragments == 2  # ids 2 and 3 have < 10

    def test_avg_conf_and_entropy(self):
        df = pd.DataFrame({
            "array_index": np.arange(4),
            "Sub_tree_id": [1, 1, 2, 2],
        })
        resp_map = {
            1: (np.array([0, 1], dtype=np.uint64), np.array([0.9, 0.8], dtype=np.float32)),
            2: (np.array([2, 3], dtype=np.uint64), np.array([0.7, 0.6], dtype=np.float32)),
        }
        s = GMMAssignerStatistics().compute(df, resp_map, {})
        # avg_conf = mean([0.9, 0.8, 0.7, 0.6]) = 0.75
        assert np.isclose(s.avg_conf, 0.75)
        assert not np.isnan(s.avg_entropy)
        assert s.avg_entropy >= 0.0

    def test_avg_cond_from_full_cov(self):
        df = pd.DataFrame({
            "array_index": np.arange(2),
            "Sub_tree_id": [1, 1],
        })
        resp_map = {
            1: (np.array([0, 1], dtype=np.uint64), np.array([0.9, 0.8], dtype=np.float32)),
        }
        params = {1: {"covariance_condition": 10.0}, 2: {"covariance_condition": 15.0}}
        s = GMMAssignerStatistics().compute(df, resp_map, params)
        assert np.isclose(s.avg_cond, 12.5, atol=0.5)  # mean(10, 15) = 12.5

    def test_avg_retention(self):
        rng = np.random.default_rng(42)
        N = 100
        df = pd.DataFrame({
            "array_index": np.arange(N),
            "Sub_tree_id": np.where(rng.random(N) < 0.5, 1, 2),
        })
        resp_map = {1: (np.array([], dtype=np.uint64), np.array([], dtype=np.float32))}
        # boundness_csc: galaxy 1 has bound indices [0..49]
        from roadrunner.clustering.sparse import SparseCSC
        bound_csc = SparseCSC(
            [np.arange(50, dtype=np.uint64), np.arange(50, 100, dtype=np.uint64)],
            [np.ones(50, dtype=np.float32), np.ones(50, dtype=np.float32)],
            column_id=np.array([1, 2], dtype=np.int64),
        )
        s = GMMAssignerStatistics().compute(df, resp_map, {}, boundness_csc=bound_csc)
        assert not np.isnan(s.avg_retention)

    def test_values_property(self):
        s = GMMAssignerStatistics()
        df = pd.DataFrame({"array_index": np.arange(5), "Sub_tree_id": [1] * 5})
        s.compute(df, {}, {})
        v = s.values
        assert "unassigned" in v
        assert "fragments" in v
        assert "avg_conf" in v
        assert "avg_entropy" in v
        assert "avg_cond" in v

    def test_gmm_assigner_integration(self):
        from roadrunner.clustering.assignment.gmm import XGMMAssigner
        a = XGMMAssigner(cov_type="diagonal", max_iter=1)
        assert isinstance(a.statistics, GMMAssignerStatistics)
