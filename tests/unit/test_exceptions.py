from roadrunner._exceptions import (
    ConfigurationError,
    ConvergenceError,
    CycleError,
    NoBoundParticlesError,
    RestartError,
    RoadrunnerError,
    SnapshotLoadError,
)


class TestExceptionHierarchy:
    def test_all_subclasses_inherit_roadrunner_error(self):
        assert issubclass(RoadrunnerError, Exception)
        assert issubclass(ConfigurationError, RoadrunnerError)
        assert issubclass(SnapshotLoadError, RoadrunnerError)
        assert issubclass(NoBoundParticlesError, RoadrunnerError)
        assert issubclass(ConvergenceError, RoadrunnerError)
        assert issubclass(RestartError, RoadrunnerError)
        assert issubclass(CycleError, RoadrunnerError)
