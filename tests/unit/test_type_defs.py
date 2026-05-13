import numpy as np

from roadrunner._mcf_types import (
    AssignmentResult,
    BoundnessResult,
    ParticleAssigner,
    PotentialModel,
    SnapshotData,
)


class TestSnapshotData:
    def test_construction_defaults(self):
        data = SnapshotData(
            indices=np.array([0, 1]),
            masses=np.array([1.0, 2.0]),
            positions=np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
            velocities=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            redshift=0.5,
            time=1.0,
        )
        assert data.metallicity is None
        assert len(data.indices) == 2

    def test_construction_with_metallicity(self):
        data = SnapshotData(
            indices=np.array([0, 1]),
            masses=np.array([1.0, 2.0]),
            positions=np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
            velocities=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            redshift=0.5,
            time=1.0,
            metallicity=np.array([0.01, 0.02]),
        )
        assert data.metallicity is not None

    def test_fields_accessible(self):
        data = SnapshotData(
            indices=np.array([0, 1]),
            masses=np.array([1.0, 2.0]),
            positions=np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
            velocities=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            redshift=0.5,
            time=1.0,
        )
        assert "indices" in dir(data)
        assert "masses" in dir(data)
        assert "positions" in dir(data)
        assert "velocities" in dir(data)
        assert "redshift" in dir(data)
        assert "time" in dir(data)


class TestBoundnessResult:
    def test_construction(self):
        result = BoundnessResult(
            candidate_indices=np.array([0, 1]),
            boundness_values=np.array([0.5, 0.8]),
            dynamical_times=[np.array([1.0]), np.array([2.0])],
        )
        assert len(result.candidate_indices) == 2

    def test_fields_accessible(self):
        result = BoundnessResult(
            candidate_indices=np.array([0, 1]),
            boundness_values=np.array([0.5, 0.8]),
            dynamical_times=[np.array([1.0])],
        )
        assert "candidate_indices" in dir(result)
        assert "boundness_values" in dir(result)
        assert "dynamical_times" in dir(result)


class TestAssignmentResult:
    def test_construction(self):
        import pandas as pd

        result = AssignmentResult(
            particle_df=pd.DataFrame({"id": [0, 1]}),
            responsibilities=([[]], [[]]),
            fitted_parameters={"subtree_1": {"mean": [0.0, 0.0]}},
            statistics={"n_particles": 2},
        )
        assert result.particle_df is not None

    def test_fields_accessible(self):
        import pandas as pd

        result = AssignmentResult(
            particle_df=pd.DataFrame({"id": [0]}),
            responsibilities=([[]], [[]]),
            fitted_parameters={},
            statistics={},
        )
        assert "particle_df" in dir(result)
        assert "responsibilities" in dir(result)
        assert "fitted_parameters" in dir(result)
        assert "statistics" in dir(result)


class MockAssigner:
    def assign(self, halos, particle_coords, newborn_indices, groups, **kwargs):
        import pandas as pd

        return AssignmentResult(
            particle_df=pd.DataFrame({"id": [0]}),
            responsibilities=([[]], [[]]),
            fitted_parameters={},
            statistics={},
        )


class TestParticleAssignerProtocol:
    def test_mock_assigner_passes_isinstance(self):
        assert isinstance(MockAssigner(), ParticleAssigner)

    def test_assign_returns_assignment_result(self):
        import numpy as np

        result = MockAssigner().assign(
            halos=[],
            particle_coords=np.zeros((0, 6)),
            newborn_indices=np.array([]),
            groups=[],
        )
        assert isinstance(result, AssignmentResult)


class MockPotentialModel:
    def potential(self, r):
        return -1.0 / r

    def dynamical_time(self, x):
        return x * 2.0

    def tidal_denominator(self, r):
        return 3.0 * r


class TestPotentialModelProtocol:
    def test_mock_potential_passes_isinstance(self):
        assert isinstance(MockPotentialModel(), PotentialModel)

    def test_methods_return_expected(self):
        model = MockPotentialModel()
        r = np.array([1.0, 2.0])
        assert np.all(model.potential(r) < 0)
        assert np.all(model.dynamical_time(r) > 0)
        assert np.all(model.tidal_denominator(r) > 0)
