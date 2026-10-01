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

    def test_contested_confidence_and_condition_summary(self):
        # Particles 0-1 are shared by components 1 and 2; particle 2 is only
        # in component 3. Confidence/entropy look at the contested ones only.
        df = pd.DataFrame({"array_index": np.arange(3), "Sub_tree_id": [1, 2, 3]})
        resp_map = {
            1: (np.array([0, 1], dtype=np.uint64), np.array([0.9, 0.3], dtype=np.float32)),
            2: (np.array([0, 1], dtype=np.uint64), np.array([0.1, 0.7], dtype=np.float32)),
            3: (np.array([2], dtype=np.uint64), np.array([1.0], dtype=np.float32)),
        }
        params = {1: {"covariance_condition": 10.0}, 2: {"covariance_condition": 1000.0},
                  3: {"covariance_condition": 1e12},
                  4: {"covariance_condition": 1e12, "rank_deficient": True}}   # not counted
        s = GMMAssignerStatistics().compute(df, resp_map, params)
        assert np.isclose(s.avg_conf, 0.8)                  # mean(0.9, 0.7)
        assert 0.0 < s.avg_entropy < 1.0
        assert np.isclose(s.avg_cond, 3.0)                  # median log10(10, 1e3, 1e12)
        assert s.bad_cond == 1

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
        assert 0.0 <= s.avg_retention <= 1.0

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
