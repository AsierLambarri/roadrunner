import pytest

from roadrunner._exceptions import (
    ConfigurationError,
    ConvergenceError,
    NoBoundParticlesError,
    RestartError,
    RoadrunnerError,
    SnapshotLoadError,
)


class TestExceptionHierarchy:
    def test_roadrunner_error_base(self):
        with pytest.raises(RoadrunnerError):
            raise RoadrunnerError("base error")

    def test_configuration_error_inherits(self):
        assert issubclass(ConfigurationError, RoadrunnerError)

    def test_snapshot_load_error_inherits(self):
        assert issubclass(SnapshotLoadError, RoadrunnerError)

    def test_no_bound_particles_error_inherits(self):
        assert issubclass(NoBoundParticlesError, RoadrunnerError)

    def test_convergence_error_inherits(self):
        assert issubclass(ConvergenceError, RoadrunnerError)

    def test_restart_error_inherits(self):
        assert issubclass(RestartError, RoadrunnerError)


class TestExceptionMessages:
    def test_configuration_error_message(self):
        with pytest.raises(ConfigurationError, match="invalid config"):
            raise ConfigurationError("invalid config")

    def test_snapshot_load_error_message(self):
        with pytest.raises(SnapshotLoadError, match="failed to load"):
            raise SnapshotLoadError("failed to load")

    def test_no_bound_particles_message(self):
        with pytest.raises(NoBoundParticlesError, match="no bound"):
            raise NoBoundParticlesError("no bound particles")

    def test_convergence_error_message(self):
        with pytest.raises(ConvergenceError, match="did not converge"):
            raise ConvergenceError("did not converge")

    def test_restart_error_message(self):
        with pytest.raises(RestartError, match="corrupted"):
            raise RestartError("corrupted state")


class TestIsInstance:
    def test_subclass_isinstance_base(self):
        assert isinstance(ConfigurationError("x"), RoadrunnerError)
        assert isinstance(SnapshotLoadError("x"), RoadrunnerError)
        assert isinstance(NoBoundParticlesError("x"), RoadrunnerError)
        assert isinstance(ConvergenceError("x"), RoadrunnerError)
        assert isinstance(RestartError("x"), RoadrunnerError)
