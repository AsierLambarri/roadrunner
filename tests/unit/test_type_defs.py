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
            index=np.array([0, 1]),
            mass=np.array([1.0, 2.0]),
            position=np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
            velocity=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            redshift=0.5,
            time=1.0,
        )
        assert not hasattr(data, "metallicity")
        assert len(data.index) == 2


class TestBoundnessResult:
    def test_construction(self):
        result = BoundnessResult(
            candidate_indices=np.array([0, 1]),
            boundness_values=np.array([0.5, 0.8]),
            dynamical_times=[np.array([1.0]), np.array([2.0])],
        )
        assert len(result.candidate_indices) == 2
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
        assert result.responsibilities is not None
        assert result.fitted_parameters is not None
        assert result.statistics is not None


class MockAssigner:
    def assign(self, halos, particle_coords, newborn_indices, groups, **kwargs):
        import pandas as pd

        return AssignmentResult(
            particle_df=pd.DataFrame({"id": [0]}),
            responsibilities=([[]], [[]]),
            fitted_parameters={},
            statistics={},
        )


class WrongAssigner:
    pass


class TestParticleAssignerProtocol:
    def test_mock_assigner_passes_isinstance(self):
        assert isinstance(MockAssigner(), ParticleAssigner)

    def test_wrong_fails_isinstance(self):
        assert not isinstance(WrongAssigner(), ParticleAssigner)


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


class TestSnapshotDataArrayIndex:
    def test_sparse_indices(self):
        ids = np.array([1000, 2000, 3000, 4000], dtype=np.uint64)
        sd = SnapshotData(
            index=ids,
            mass=np.ones(4),
            position=np.zeros((4, 3)),
            velocity=np.zeros((4, 3)),
            redshift=0.0, time=13.8,
        )
        result = sd.array_index(np.array([2000, 999, 4000], dtype=np.uint64))
        assert result[0] == 1   # 2000 at position 1
        assert result[1] == -1  # 999 not present
        assert result[2] == 3   # 4000 at position 3

    def test_empty_query(self):
        sd = SnapshotData(
            index=np.arange(10, dtype=np.uint64),
            mass=np.ones(10),
            position=np.zeros((10, 3)),
            velocity=np.zeros((10, 3)),
            redshift=0.0, time=13.8,
        )
        result = sd.array_index(np.array([], dtype=np.uint64))
        assert result.size == 0

    def test_zero_particle_snapshot_all_missing(self):
        sd = SnapshotData(
            index=np.array([], dtype=np.uint64),
            mass=np.array([]),
            position=np.empty((0, 3)),
            velocity=np.empty((0, 3)),
            redshift=0.0, time=13.8,
        )
        result = sd.array_index(np.array([1, 2, 3], dtype=np.uint64))
        assert np.array_equal(result, [-1, -1, -1])
