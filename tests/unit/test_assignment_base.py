import numpy as np
import pandas as pd

from roadrunner._mcf_types import AssignmentResult
from roadrunner._mcf_types import ParticleAssigner


class MockAssigner:
    def assign(self, halos, particle_coords, newborn_indices, groups, **kwargs):
        return AssignmentResult(
            particle_df=pd.DataFrame({"array_index": [0], "Sub_tree_id": [1]}),
            responsibilities=([[]], [[]]),
            fitted_parameters={},
            statistics={},
        )


class WrongAssigner:
    pass


class TestProtocol:
    def test_mock_passes_isinstance(self):
        assert isinstance(MockAssigner(), ParticleAssigner)

    def test_wrong_fails_isinstance(self):
        assert not isinstance(WrongAssigner(), ParticleAssigner)

    def test_extra_kwargs_ignored(self):
        m = MockAssigner()
        result = m.assign(
            halos=[],
            particle_coords=np.zeros((0, 6)),
            newborn_indices=np.array([]),
            groups=[],
            unknown_param=42,
            random_other="ignored",
        )
        assert isinstance(result, AssignmentResult)

    def test_halos_received(self):
        class CheckHalos:
            def assign(self, halos, particle_coords, newborn_indices, groups, **kwargs):
                assert len(halos) == 2
                return AssignmentResult(
                    particle_df=pd.DataFrame({"array_index": [0], "Sub_tree_id": [1]}),
                    responsibilities=([[]], [[]]),
                    fitted_parameters={},
                    statistics={},
                )

        assert isinstance(CheckHalos(), ParticleAssigner)

    def test_groups_received(self):
        class CheckGroups:
            def assign(self, halos, particle_coords, newborn_indices, groups, **kwargs):
                assert groups == [[0, 1], [2]]
                return AssignmentResult(
                    particle_df=pd.DataFrame({"array_index": [0], "Sub_tree_id": [1]}),
                    responsibilities=([[]], [[]]),
                    fitted_parameters={},
                    statistics={},
                )

        assert isinstance(CheckGroups(), ParticleAssigner)


class TestAssignmentResult:
    def test_construction(self):
        df = pd.DataFrame({"array_index": [0, 1], "Sub_tree_id": [10, 20]})
        result = AssignmentResult(
            particle_df=df,
            responsibilities=([([0], [1])], [np.array([0.5, 0.5])]),
            fitted_parameters={"subtree_10": {"mean": [0.0, 0.0]}},
            statistics={"n_particles": 2},
        )
        assert result.particle_df is not None
        assert result.responsibilities is not None
        assert result.fitted_parameters is not None
        assert result.statistics is not None

    def test_fields_accessible(self):
        df = pd.DataFrame({"array_index": [0], "Sub_tree_id": [1]})
        result = AssignmentResult(
            particle_df=df,
            responsibilities=([], []),
            fitted_parameters={},
            statistics={},
        )
        assert "particle_df" in dir(result)
        assert "responsibilities" in dir(result)
        assert "fitted_parameters" in dir(result)
        assert "statistics" in dir(result)
